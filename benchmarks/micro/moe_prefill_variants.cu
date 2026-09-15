#include "moe_prefill_variants.hpp"

#include <algorithm>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/cuda_check.hpp"

namespace dgpp::bench {
// Short reduction dimensions spend a larger fraction of each block on
// setup. Compare 16-row and 64-row tiles and persistent block scheduling
// with the same FP8 decode and ascending-k16 MMA chain. These experiments
// are deliberately confined to the benchmark target.
namespace {
__device__ __forceinline__ void cp_async(void* smem, const void* gmem, int bytes, int valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  if (bytes == 16)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(address), "l"(gmem), "r"(valid));
  else
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(address), "l"(gmem), "r"(valid));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}
__device__ __forceinline__ uint32_t decode_pair_bf16(uint32_t two, float scale) {
  const __half2 h(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two), __NV_E4M3));
  const __nv_bfloat162 b = __floats2bfloat162_rn(__low2float(h) * scale, __high2float(h) * scale);
  return *reinterpret_cast<const uint32_t*>(&b);
}

template <typename OutT, int BM, int BN>
__global__ __launch_bounds__(2 * BN, BM == 16 ? 4 : 2) void moe_prefill_variant_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k, int jobs, int n_tiles) {
  constexpr int BK = 32;
  constexpr int kThreads = 2 * BN, kStages = 4;
  constexpr int A_PAD = BK + 8, kRawStride = 48;
  constexpr size_t kABytes = size_t(BM) * A_PAD * 2;
  constexpr size_t kCodeBytes = size_t(BN) * kRawStride;
  constexpr size_t kScaleBytes = size_t(BN) * sizeof(float);
  constexpr size_t kSlotBytes = kABytes + kCodeBytes + kScaleBytes;
  constexpr size_t kSmem = kStages * kSlotBytes;
  static_assert(BM * (BK / 8) <= kThreads && BN * (BK / 16) == kThreads);
  static_assert(kSlotBytes % 16 == 0);
  static_assert(kSmem == (BM == 16 ? 18432 : 47104));
  extern __shared__ __align__(16) uint8_t smem[];
  for (int job = static_cast<int>(blockIdx.x); job < jobs; job += gridDim.x) {
    const MoeSegment seg = segs[job / n_tiles];
    if (seg.rows == 0) continue;
    const int n_tile = job % n_tiles;
    const MoeExpertView v = views[seg.expert * 3 + which];
    const int n0 = n_tile * BN;
    const int rs = v.scale_shift_rows, cs = v.scale_shift_cols;
    const int scale_cols = (k + (1 << cs) - 1) >> cs;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32, lane = tid % 32;
    const int wn = warp;  // each warp owns 16 output columns
    const int r = lane / 4, cc = (lane % 4) * 2;
    // Copies: activation chunk (row tid/4, 8 elements (tid%4)*8); codes row
    // tid/2, 16-byte half tid%2; the row's scale by the half-0 thread.
    const int a_row = tid / 4, a_kq = (tid % 4) * 8;
    const int b_row = tid / 2, b_half = tid % 2;
    const int gn = n0 + b_row;
    const bool b_ok = gn < n;
    const uint8_t* b_codes = v.payload + static_cast<size_t>(b_ok ? gn : 0) * k;
    const float* b_scales = v.scales + (static_cast<size_t>(b_ok ? gn : 0) >> rs) * scale_cols;
    // A block walks only the tiles this expert actually owns. The caller
    // knows the prompt length, not the longest device-resident segment.
    for (int z0 = 0; z0 < seg.rows; z0 += BM) {
      const int m_rows = min(BM, seg.rows - z0);
      const bool a_ok = a_row < m_rows;
      const uint16_t* a_src = act;
      if (a_ok) {
        const int srow = seg.row0 + z0 + a_row;
        a_src = act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) * act_stride;
      }
      const int stages = k / BK;  // k % 32 == 0 (the launcher's contract)
      const int live_slabs = (m_rows + 15) / 16;

      auto slotA = [&](int slot) { return reinterpret_cast<uint16_t*>(smem + static_cast<size_t>(slot) * kSlotBytes); };
      auto slotCodes = [&](int slot) { return smem + static_cast<size_t>(slot) * kSlotBytes + kABytes; };
      auto slotScale = [&](int slot) {
        return reinterpret_cast<float*>(smem + static_cast<size_t>(slot) * kSlotBytes + kABytes + kCodeBytes);
      };
      auto issue = [&](int s, int slot) {
        const int k0 = s * BK;
        if (a_row < BM) {
          uint16_t* dst = slotA(slot) + static_cast<size_t>(a_row) * A_PAD + a_kq;
          const uint16_t* src = a_ok ? a_src + k0 + a_kq : act;
          if (act_vec != 0) {
            if (a_ok && (tid % 4) == 0)
              asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(src));
            cp_async(dst, src, 16, a_ok ? 16 : 0);
          } else {
            uint16_t e[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) e[h] = a_ok ? src[h] : static_cast<uint16_t>(0);
            *reinterpret_cast<uint4*>(dst) = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                                                        e[4] | (e[5] << 16), e[6] | (e[7] << 16));
          }
        }
        {
          // 16 codes = 16 bytes per half; k % 32 == 0 so a stage is in whole.
          cp_async(slotCodes(slot) + static_cast<size_t>(b_row) * kRawStride + b_half * 16,
                   b_ok ? b_codes + k0 + b_half * 16 : v.payload, 16, b_ok ? 16 : 0);
          // The row's next 128-byte line of codes (stages s+4 .. s+7) into L2.
          if (b_half == 0 && b_ok && (s % 4) == 0) {
            const int nk = k0 + 4 * BK;
            if (nk < k) asm volatile("prefetch.global.L2 [%0];" ::"l"(b_codes + nk));
          }
        }
        if (b_half == 0) cp_async(slotScale(slot) + b_row, b_scales + (k0 >> cs), 4, 4);
      };

      float acc[BM / 16][2][4];
#pragma unroll
      for (int i = 0; i < BM / 16; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
      for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages) issue(s, s);
        commit();
      }
      for (int s = 0; s < stages; ++s) {
        const int slot = s % kStages;
        wait<kStages - 2>();
        __syncthreads();
        if (s + kStages - 1 < stages) issue(s + kStages - 1, (s + kStages - 1) % kStages);
        commit();
        const uint16_t* a = slotA(slot);
        const uint8_t* codes = slotCodes(slot);
        const float* sc = slotScale(slot);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
          uint32_t bfrag[2][2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int brow = wn * 16 + j * 8 + r;
            const uint8_t* row = codes + static_cast<size_t>(brow) * kRawStride + kk;
            const float s_row = sc[brow];
            const uint32_t lo = *reinterpret_cast<const uint16_t*>(row + cc);      // k = cc, cc+1
            const uint32_t hi = *reinterpret_cast<const uint16_t*>(row + cc + 8);  // k = cc+8, cc+9
            bfrag[j][0] = decode_pair_bf16(lo, s_row);
            bfrag[j][1] = decode_pair_bf16(hi, s_row);
          }
#pragma unroll
          for (int i = 0; i < BM / 16; ++i) {
            if (i >= live_slabs) break;
            uint32_t af[4];
            ldmatrix_x4(af, a + static_cast<size_t>(i * 16 + (lane % 16)) * A_PAD + kk + (lane / 16) * 8);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              asm volatile(
                  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                  : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                  : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(bfrag[j][0]), "r"(bfrag[j][1]));
            }
          }
        }
      }
      wait<0>();

      // Preserve the reference's stores, including partial rows and columns.
#pragma unroll
      for (int i = 0; i < BM / 16; ++i) {
        const int row_lo = i * 16 + r;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int gcol = n0 + wn * 16 + j * 8 + cc;
          auto st = [&](int row_off, int col_off, float val) {
            const int mm = row_lo + row_off;
            if (mm < m_rows && gcol + col_off < n) {
              if constexpr (std::is_same_v<OutT, float>)
                out[static_cast<size_t>(seg.row0 + z0 + mm) * out_stride + gcol + col_off] = val;
              else
                out[static_cast<size_t>(seg.row0 + z0 + mm) * out_stride + gcol + col_off] =
                    __bfloat16_as_ushort(__float2bfloat16_rn(val));
            }
          };
          st(0, 0, acc[i][j][0]);
          st(0, 1, acc[i][j][1]);
          st(8, 0, acc[i][j][2]);
          st(8, 1, acc[i][j][3]);
        }
      }
      __syncthreads();  // the next m-tile may reuse every ring slot
    }
  }
}

}  // namespace

template <typename OutT>
void launch_variant(Variant variant, const uint16_t* act, size_t act_stride, const int32_t* act_rows,
    const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
    OutT* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  const int act_vec = (reinterpret_cast<uintptr_t>(act) % 16 == 0 && act_stride % 8 == 0) ? 1 : 0;
  static const int resident_blocks = [] {
    int device = 0, sms = 0;
    DGPP_CUDA_OK(cudaGetDevice(&device));
    DGPP_CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return 2 * sms;
  }();
  const int width = variant == Variant::kSmallTiles ? 64 : 128;
  const int n_tiles = (n + width - 1) / width;
  const int jobs = n_tiles * n_segs;
  const int blocks = variant == Variant::kPersistent ? std::min(jobs, resident_blocks) : jobs;
  if (variant == Variant::kSmallTiles)
    moe_prefill_variant_kernel<OutT, 16, 64><<<blocks, 128, 18432, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, jobs, n_tiles);
  else
    moe_prefill_variant_kernel<OutT, 64, 128><<<blocks, 256, 47104, stream>>>(
        act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n, k, jobs, n_tiles);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_variant_bf16(Variant variant, const uint16_t* act, size_t act_stride, const int32_t* act_rows,
    const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
    uint16_t* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  launch_variant(variant, act, act_stride, act_rows, segs, n_segs, views, which, out, out_stride, n, k, stream);
}
void launch_variant_f32(Variant variant, const uint16_t* act, size_t act_stride, const int32_t* act_rows,
    const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
    float* out, size_t out_stride, int n, int k, cudaStream_t stream) {
  launch_variant(variant, act, act_stride, act_rows, segs, n_segs, views, which, out, out_stride, n, k, stream);
}
}  // namespace dgpp::bench
