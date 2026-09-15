#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>

#include "common/cuda_check.hpp"
#include "kernels/packq_gemm.hpp"

namespace dgpp {
namespace {
constexpr int kM = 32, kN = 64, kK = 64, kThreads = 128;
constexpr int kActStride = kK + 8;

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src),
               "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::);
}
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group 0;" ::);
}
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const uint16_t* src) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(src));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(address));
}

template <int Bits>
__device__ __forceinline__ uint32_t code_pair(const uint8_t* row, int col) {
  // Offset codes, low nibble/byte first. Every decoded integer is exactly
  // representable in BF16, including int8's -128 and +127 endpoints.
  const unsigned packed = Bits == 4 ? row[col / 2] : *reinterpret_cast<const uint16_t*>(row + col);
  constexpr unsigned mask = (1u << Bits) - 1;
  constexpr int bias = 1 << (Bits - 1);
  const float lo = static_cast<float>(static_cast<int>(packed & mask) - bias);
  const float hi = static_cast<float>(static_cast<int>((packed >> Bits) & mask) - bias);
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

template <int Bits, typename OutT, bool Grouped>
__global__ __launch_bounds__(kThreads) void packq_gemm_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, const uint8_t* __restrict__ weights,
    const uint16_t* __restrict__ scales, const MoeSegment* __restrict__ segs,
    const MoeExpertView* __restrict__ views, int which, const int32_t* __restrict__ act_rows,
    OutT* __restrict__ out, size_t out_stride, int m, int n, int k, int n_tiles, bool vector_act) {
  constexpr int kCodeStride = kK * Bits / 8 + 16;
  __shared__ __align__(16) uint16_t a[2][kM][kActStride];
  __shared__ __align__(16) uint8_t b[2][kN][kCodeStride];
  __shared__ float scale[2][kN];
  int row0 = 0;
  if constexpr (Grouped) {
    const MoeSegment seg = segs[blockIdx.y];
    row0 = seg.row0;
    m = seg.rows;
    const MoeExpertView view = views[seg.expert * 3 + which];
    weights = view.payload;
    scales = view.packed_scales;
  }
  const int m0 = (blockIdx.x / n_tiles) * kM;
  if (m0 >= m) return;
  const int n0 = (blockIdx.x % n_tiles) * kN;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int r = lane / 4, cc = (lane % 4) * 2;
  const int groups = k / kK;
  auto issue = [&](int group, int slot) {
    for (int idx = tid; idx < kM * (kK / 8); idx += kThreads) {
      const int row = idx / (kK / 8), col = (idx % (kK / 8)) * 8;
      const bool valid = m0 + row < m;
      const int gathered = row0 + m0 + row;
      const int source_row = valid ? (act_rows ? act_rows[gathered] : gathered) : 0;
      const uint16_t* src = act + static_cast<size_t>(source_row) * act_stride + group * kK + col;
      if (vector_act) {
        copy16(&a[slot][row][col], src, valid);
      } else {
        // Row maps and odd activation strides are valid; only the aligned
        // case uses 16-byte asynchronous copies.
#pragma unroll
        for (int j = 0; j < 8; ++j) a[slot][row][col + j] = valid ? src[j] : 0;
      }
    }
    constexpr int chunks = kK * Bits / 128;
    for (int idx = tid; idx < kN * chunks; idx += kThreads) {
      const int row = idx / chunks, col = (idx % chunks) * 16;
      const bool valid = n0 + row < n;
      const uint8_t* src = valid ? weights + static_cast<size_t>(n0 + row) * (k * Bits / 8) +
                                       group * (kK * Bits / 8) + col
                                 : weights;
      copy16(&b[slot][row][col], src, valid);
    }
    if (tid < kN)
      scale[slot][tid] =
          n0 + tid < n
              ? __uint_as_float(
                    static_cast<unsigned>(scales[static_cast<size_t>(n0 + tid) * groups + group])
                    << 16)
              : 0.f;
  };

  float acc[2][2][4] = {};
  issue(0, 0);
  commit();
  for (int group = 0; group < groups; ++group) {
    const int slot = group % 2;
    wait();
    __syncthreads();
    if (group + 1 < groups) issue(group + 1, slot ^ 1);
    commit();
    float partial[2][2][4] = {};
#pragma unroll
    for (int kk = 0; kk < kK; kk += 16) {
      uint32_t bf[2][2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint8_t* row = b[slot][warp * 16 + j * 8 + r];
        bf[j][0] = code_pair<Bits>(row, kk + cc);
        bf[j][1] = code_pair<Bits>(row, kk + cc + 8);
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t af[4];
        load_a(af, &a[slot][i * 16 + lane % 16][kk + (lane / 16) * 8]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(partial[i][j][0]), "+f"(partial[i][j][1]), "+f"(partial[i][j][2]),
                "+f"(partial[i][j][3])
              : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bf[j][0]), "r"(bf[j][1]));
      }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = warp * 16 + j * 8 + cc;
#pragma unroll
        for (int v = 0; v < 4; ++v)
          acc[i][j][v] = __fmaf_rn(scale[slot][col + (v % 2)], partial[i][j][v], acc[i][j][v]);
      }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = m0 + i * 16 + r + (v / 2) * 8;
        const int col = n0 + warp * 16 + j * 8 + cc + (v % 2);
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row0 + row) * out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = acc[i][j][v];
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}

void check_shape(const uint16_t* act, size_t act_stride, const void* out, size_t out_stride, int n,
                 int k, int bits) {
  if (!act || !out) throw std::invalid_argument("packq_gemm: null activation/output pointer");
  if (k <= 0 || k % 64 || (bits != 4 && bits != 8))
    throw std::invalid_argument("packq_gemm: K must be a positive multiple of 64 and bits 4 or 8");
  if (act_stride < static_cast<size_t>(k) || out_stride < static_cast<size_t>(n))
    throw std::invalid_argument("packq_gemm: row stride is smaller than the matrix width");
}

template <typename OutT, bool Grouped>
void launch(const uint16_t* act, size_t act_stride, const uint8_t* weights, const uint16_t* scales,
            const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
            const int32_t* act_rows, OutT* out, size_t out_stride, int m, int n, int k, int bits,
            cudaStream_t stream) {
  check_shape(act, act_stride, out, out_stride, n, k, bits);
  const int n_tiles = (n + kN - 1) / kN;
  const dim3 grid(n_tiles * ((m + kM - 1) / kM), n_segs);
  const bool vector_act = reinterpret_cast<uintptr_t>(act) % 16 == 0 && act_stride % 8 == 0;
#define DGPP_PACKQ_LAUNCH(Bits)                                                             \
  packq_gemm_kernel<Bits, OutT, Grouped>                                                    \
      <<<grid, kThreads, 0, stream>>>(act, act_stride, weights, scales, segs, views, which, \
                                      act_rows, out, out_stride, m, n, k, n_tiles, vector_act)
  if (bits == 4) {
    DGPP_PACKQ_LAUNCH(4);
  } else {
    DGPP_PACKQ_LAUNCH(8);
  }
#undef DGPP_PACKQ_LAUNCH
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void dense(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w, OutT* out, int m,
           int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;
  if (!w.packed || !w.scales || reinterpret_cast<uintptr_t>(w.packed) % 16 || w.rows < n ||
      w.cols != k)
    throw std::invalid_argument(
        "packq_gemm: invalid packed matrix pointers, alignment or N/K geometry");
  launch<OutT, false>(act, act_stride, reinterpret_cast<const uint8_t*>(w.packed), w.scales,
                      nullptr, 1, nullptr, 0, nullptr, out, n, m, n, k, w.bits, stream);
}
template <typename OutT>
void grouped(const uint16_t* act, size_t act_stride, const MoeSegment* segs, int n_segs,
             int max_rows, const MoeExpertView* views, int which, OutT* out, size_t out_stride,
             int n, int k, int bits, cudaStream_t stream, const int32_t* act_rows) {
  if (n_segs <= 0 || n <= 0) return;
  if (!segs || !views || max_rows <= 0 || which < 0 || which > 2 || n_segs > 65535)
    throw std::invalid_argument(
        "packq_gemm: invalid segments, max_rows, views or projection index");
  launch<OutT, true>(act, act_stride, nullptr, nullptr, segs, n_segs, views, which, act_rows, out,
                     out_stride, max_rows, n, k, bits, stream);
}
}  // namespace

void launch_packq_gemm_bf16(const uint16_t* a, size_t stride, const GlmPackedMatrix& w,
                            uint16_t* out, int m, int n, int k, cudaStream_t stream) {
  dense(a, stride, w, out, m, n, k, stream);
}
void launch_packq_gemm_f32(const uint16_t* a, size_t stride, const GlmPackedMatrix& w, float* out,
                           int m, int n, int k, cudaStream_t stream) {
  dense(a, stride, w, out, m, n, k, stream);
}
void launch_moe_grouped_mma_packq_bf16(const uint16_t* a, size_t stride, const MoeSegment* segs,
                                       int ns, int mr, const MoeExpertView* views, int which,
                                       uint16_t* out, size_t os, int n, int k, int bits,
                                       cudaStream_t stream, const int32_t* rows) {
  grouped(a, stride, segs, ns, mr, views, which, out, os, n, k, bits, stream, rows);
}
void launch_moe_grouped_mma_packq_f32(const uint16_t* a, size_t stride, const MoeSegment* segs,
                                      int ns, int mr, const MoeExpertView* views, int which,
                                      float* out, size_t os, int n, int k, int bits,
                                      cudaStream_t stream, const int32_t* rows) {
  grouped(a, stride, segs, ns, mr, views, which, out, os, n, k, bits, stream, rows);
}
}  // namespace dgpp
