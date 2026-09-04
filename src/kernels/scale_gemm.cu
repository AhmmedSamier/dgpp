#include "kernels/scale_gemm.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemv.cuh"

namespace dgpp {
namespace {

// Tile geometry. BN and BK both divide the 128-wide scale block, which is
// what lets every stage apply ONE scalar scale: the n-range [n0, n0+BN)
// (n0 a multiple of BN, BN | 128) lies inside scale row n0/128, and each
// k-slice [k0, k0+BK) (BK | 128) lies inside scale column k0/128.
constexpr int BM = 16;
constexpr int BN = 64;
// Rows at or below which the chunked GEMV beats the tile kernel (measured
// 2026-09-04: the crossover sits near 128-256 rows for the MoE experts'
// n=512 slabs; the tile kernel's grid is n/64 x m/16 blocks).
constexpr int kGemvMaxM = 128;
constexpr int BK = 32;
constexpr int BK_PAD = BK + 8;  // u16 pad breaks the worst bank conflicts
constexpr int kBlockThreads = (BN / 8) * 32;  // 8 warps, one n8 group each

template <typename OutT>
__global__ void scale_gemm_kernel(const uint16_t* __restrict__ act,
                                  size_t act_stride,
                                  const uint8_t* __restrict__ w,
                                  const float* __restrict__ scales,
                                  OutT* __restrict__ out, int m, int n,
                                  int k) {
  const int n0 = blockIdx.x * BN;
  const int m0 = blockIdx.y * BM;
  const int scale_cols = (k + 127) / 128;
  const int scale_row = n0 / 128;  // whole n-tile in one scale row (BN | 128)

  __shared__ uint16_t sA[BM][BK_PAD];
  __shared__ uint16_t sB[BN][BK_PAD];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // m16n8k16 fragment coordinates (PTX register layouts): A rows lane/4
  // (+8), k columns (lane%4)*2 (+1, +8); B n = lane/4 over the same k
  // columns; C rows lane/4 (+8), n columns (lane%4)*2 (+1).
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  const int bnr = warp * 8 + r;  // this thread's B row (n) within the tile

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int k0 = 0; k0 < k; k0 += BK) {
    // Single scalar scale per stage (see geometry note above).
    const float s = scales[(size_t)scale_row * scale_cols + (k0 / 128)];

    // Weight tile: decode + scale + one BF16 round — the exact operation
    // of the dequant bridge, so weight-tile bits cannot diverge from it.
    for (int idx = threadIdx.x; idx < BN * BK; idx += kBlockThreads) {
      const int nn = idx / BK, kk = idx % BK;
      const int gn = n0 + nn, gk = k0 + kk;
      sB[nn][kk] =
          (gn < n && gk < k)
              ? float_to_bf16_bits(
                    fp8_e4m3_bits_to_float(w[(size_t)gn * k + gk]) * s)
              : 0;
    }
    // Activation tile (zero-fill outside [M, K)).
    for (int idx = threadIdx.x; idx < BM * BK; idx += kBlockThreads) {
      const int mm = idx / BK, kk = idx % BK;
      const int gm = m0 + mm, gk = k0 + kk;
      sA[mm][kk] =
          (gm < m && gk < k) ? act[(size_t)gm * act_stride + gk] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      const uint32_t a0 = static_cast<uint32_t>(sA[r][kk + cc]) |
                          (static_cast<uint32_t>(sA[r][kk + cc + 1]) << 16);
      const uint32_t a1 =
          static_cast<uint32_t>(sA[r + 8][kk + cc]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 1]) << 16);
      const uint32_t a2 =
          static_cast<uint32_t>(sA[r][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r][kk + cc + 9]) << 16);
      const uint32_t a3 =
          static_cast<uint32_t>(sA[r + 8][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 9]) << 16);
      const uint32_t b0 =
          static_cast<uint32_t>(sB[bnr][kk + cc]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 1]) << 16);
      const uint32_t b1 =
          static_cast<uint32_t>(sB[bnr][kk + cc + 8]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 9]) << 16);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();  // tile reads done before the next stage overwrites
  }

  // Epilogue: C fragment (m16n8) — thread holds (r, cc), (r, cc+1),
  // (r+8, cc), (r+8, cc+1) within its warp's [16 x 8] output slice.
  const int out_col = n0 + warp * 8 + cc;
  auto store = [&](int row_off, int col_off, float v) {
    const int gm = m0 + row_off;
    const int gn = out_col + col_off;
    if (gm < m && gn < n)
      fp8_gemv::store_dot(out + (size_t)gm * n + gn, v);
  };
  store(r, 0, c0);
  store(r, 1, c1);
  store(r + 8, 0, c2);
  store(r + 8, 1, c3);
}

// The small-m path (m <= fp8_gemv::kMaxRows): the bandwidth GEMV core, one
// warp per weight row, activations staged in dynamic smem. Same dequant
// values as the tile kernel above, a different (deterministic) fp32
// accumulation order — see fp8_gemv.cuh.
template <int kRows, typename OutT>
__global__ void scale_gemv_kernel(const uint16_t* __restrict__ act,
                                  size_t act_stride,
                                  const uint8_t* __restrict__ w,
                                  const float* __restrict__ scales,
                                  OutT* __restrict__ out, int n, int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  fp8_gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  fp8_gemv::block_rows<kRows>(w, scales, sx, blockIdx.x * fp8_gemv::kWarps, n,
                              k, out, static_cast<size_t>(n));
}

template <int kRows, typename OutT>
void launch_scale_gemv(const uint16_t* act, size_t act_stride,
                       const uint8_t* w, const float* scales, OutT* out, int n,
                       int k, cudaStream_t stream) {
  const dim3 grid((n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps);
  scale_gemv_kernel<kRows, OutT>
      <<<grid, fp8_gemv::kThreads, fp8_gemv::smem_bytes(kRows, k), stream>>>(
          act, act_stride, w, scales, out, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <typename OutT>
void launch_scale_gemv_rows(const uint16_t* act, size_t act_stride,
                            const uint8_t* w, const float* scales, OutT* out,
                            int rows, int n, int k, cudaStream_t stream) {
  switch (rows) {
    case 1:
      launch_scale_gemv<1, OutT>(act, act_stride, w, scales, out, n, k,
                                 stream);
      return;
    case 2:
      launch_scale_gemv<2, OutT>(act, act_stride, w, scales, out, n, k,
                                 stream);
      return;
    case 3:
      launch_scale_gemv<3, OutT>(act, act_stride, w, scales, out, n, k,
                                 stream);
      return;
    case 4:
      launch_scale_gemv<4, OutT>(act, act_stride, w, scales, out, n, k,
                                 stream);
      return;
    default:
      throw std::invalid_argument("scale_gemv: rows outside [1,4]");
  }
}

// One dispatch for both output dtypes: the epilogue store is the ONLY
// difference between the bf16 and fp32 products (same tiles, same GEMV
// core, same accumulation order), so a value that rounds to bf16 in one
// is the unrounded fp32 of the other.
template <typename OutT>
void launch_scale_gemm(const uint16_t* act, size_t act_row_stride_elems,
                       const uint8_t* w_payload, const float* w_scales,
                       OutT* out, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;  // empty output by definition
  if (!act || !w_payload || !w_scales || !out)
    throw std::invalid_argument("scale_gemm: null pointer");
  if (k <= 0) {
    // Degenerate contraction: zero outputs (matches the fp64 oracle).
    DGPP_CUDA_OK(cudaMemsetAsync(
        out, 0, static_cast<size_t>(m) * n * sizeof(OutT), stream));
    return;
  }
  // Small-M calls take the row-independent bandwidth GEMV (the tile below
  // is latency-bound at small m — see fp8_gemv.cuh): one GEMV launch
  // carries at most four rows (fewer when K fills the 48-KiB smem budget),
  // and the rows are chunked through the same scalar-order core, so each
  // output row is bitwise invariant to m and to the path — the two kernels
  // agree bit for bit, which is what makes this threshold a pure
  // performance knob. It was 8 (the decode-row bound); the 2026-09-04
  // prefill profile found the tile kernel at 578 us for the MoE experts'
  // 9..30-row segments (grid 8 x 1..2 blocks on a 48-SM part) against 15 us
  // per four-row GEMV launch, so prefill-sized segments go this way too.
  // The tile kernel keeps the large-m shapes where its grid fills the GPU.
  if (m >= 1 && m <= kGemvMaxM && fp8_gemv::shape_ok(w_payload, /*rows=*/1, k)) {
    for (int row0 = 0; row0 < m;) {
      int rows = std::min(fp8_gemv::kMaxRows, m - row0);
      while (!fp8_gemv::shape_ok(w_payload, rows, k)) --rows;
      launch_scale_gemv_rows(
          act + static_cast<size_t>(row0) * act_row_stride_elems,
          act_row_stride_elems, w_payload, w_scales,
          out + static_cast<size_t>(row0) * n, rows, n, k, stream);
      row0 += rows;
    }
    return;
  }
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  scale_gemm_kernel<OutT><<<grid, kBlockThreads, 0, stream>>>(
      act, act_row_stride_elems, w_payload, w_scales, out, m, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream) {
  launch_scale_gemm<uint16_t>(act, act_row_stride_elems, w_payload, w_scales,
                              out, m, n, k, stream);
}

void launch_scale_gemm_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const uint8_t* w_payload, const float* w_scales,
                           float* out, int m, int n, int k,
                           cudaStream_t stream) {
  launch_scale_gemm<float>(act, act_row_stride_elems, w_payload, w_scales,
                           out, m, n, k, stream);
}

}  // namespace dgpp
