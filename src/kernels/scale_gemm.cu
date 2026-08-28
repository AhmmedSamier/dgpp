#include "kernels/scale_gemm.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

// Tile geometry. BN and BK both divide the 128-wide scale block, which is
// what lets every stage apply ONE scalar scale: the n-range [n0, n0+BN)
// (n0 a multiple of BN, BN | 128) lies inside scale row n0/128, and each
// k-slice [k0, k0+BK) (BK | 128) lies inside scale column k0/128.
constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int BK_PAD = BK + 8;  // u16 pad breaks the worst bank conflicts
constexpr int kBlockThreads = (BN / 8) * 32;  // 8 warps, one n8 group each

__global__ void scale_gemm_bf16_kernel(const uint16_t* __restrict__ act,
                                       size_t act_stride,
                                       const uint8_t* __restrict__ w,
                                       const float* __restrict__ scales,
                                       uint16_t* __restrict__ out, int m,
                                       int n, int k) {
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
    if (gm < m && gn < n) out[(size_t)gm * n + gn] = float_to_bf16_bits(v);
  };
  store(r, 0, c0);
  store(r, 1, c1);
  store(r + 8, 0, c2);
  store(r + 8, 1, c3);
}

}  // namespace

void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;  // empty output by definition
  if (!act || !w_payload || !w_scales || !out)
    throw std::invalid_argument("scale_gemm: null pointer");
  if (k <= 0) {
    // Degenerate contraction: zero outputs (matches the fp64 oracle).
    DGPP_CUDA_OK(cudaMemsetAsync(out, 0, static_cast<size_t>(m) * n * 2,
                                 stream));
    return;
  }
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  scale_gemm_bf16_kernel<<<grid, kBlockThreads, 0, stream>>>(
      act, act_row_stride_elems, w_payload, w_scales, out, m, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
