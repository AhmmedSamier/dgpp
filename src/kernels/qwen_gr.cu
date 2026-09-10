#include "kernels/qwen_gr.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/gemv_common.cuh"
#include "kernels/qwen_norm.cuh"

namespace dgpp {
namespace {

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}
__device__ __forceinline__ float sigmoid_f(float v) { return 1.0f / (1.0f + expf(-v)); }

__global__ void gate_act_kernel(uint16_t* __restrict__ t, int64_t n, float inv_hc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // t / hc is exact in bf16 (hc a power of two); silu in fp32, one rounding.
  const float v = bf16_bits_to_float(t[i]) * inv_hc;
  t[i] = float_to_bf16_bits(v * sigmoid_f(v));
}

__global__ void mix_finish_kernel(const uint16_t* __restrict__ logits,
                                  const uint16_t* __restrict__ rn,
                                  uint16_t* __restrict__ x, int hc, int hidden,
                                  float inv_hc) {
  const int64_t row = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= hidden) return;
  const int64_t base = row * static_cast<int64_t>(hc) * hidden + j;
  float acc = 0.0f;
  for (int i = 0; i < hc; ++i) {
    const int64_t at = base + static_cast<int64_t>(i) * hidden;
    const float g = round_bf16(sigmoid_f(bf16_bits_to_float(logits[at])));
    acc += round_bf16(g * bf16_bits_to_float(rn[at]));
  }
  x[row * hidden + j] = float_to_bf16_bits(acc * inv_hc);
}

// The inject dots (2026-09-09, the decode profile: the one-block form ran
// 29 us per site at T=1 — a 320-deep chain of dependent 2-byte loads on
// one SM, 96 sites per step): one block per row, warp i (i < hc) reduces
// the i-th dot over the hc*H-wide Rn row in the SAME fixed order (per-lane
// strided chain, xor tree), the loads issued 32 deep ahead of the chain so
// the warp is FMA-bound; gates[row*hc + i] = 2 * bf16(sigmoid(bf16(dot) / hc)).
constexpr int kCombineBlock = 256;
constexpr int kDotBatch = 32;

__global__ void combine_dots_kernel(const uint16_t* __restrict__ rn,
                                    const uint16_t* __restrict__ w_inject,
                                    float* __restrict__ gates, int hc, int width,
                                    float inv_hc) {
  const int64_t row = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp >= hc) return;
  const uint16_t* rn_row = rn + row * width;
  const uint16_t* w = w_inject + static_cast<int64_t>(warp) * width;
  float acc = 0.0f;
  int k = lane;
  for (; k + (kDotBatch - 1) * 32 < width; k += 32 * kDotBatch) {
    uint16_t wv[kDotBatch], rv[kDotBatch];
#pragma unroll
    for (int b = 0; b < kDotBatch; ++b) {
      wv[b] = w[k + b * 32];
      rv[b] = rn_row[k + b * 32];
    }
#pragma unroll
    for (int b = 0; b < kDotBatch; ++b)
      acc = fmaf(bf16_bits_to_float(wv[b]), bf16_bits_to_float(rv[b]), acc);
  }
  for (; k < width; k += 32)
    acc = fmaf(bf16_bits_to_float(w[k]), bf16_bits_to_float(rn_row[k]), acc);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
  if (lane == 0) {
    const float dot = round_bf16(acc);          // the linear's bf16 output
    gates[row * hc + warp] = 2.0f * round_bf16(sigmoid_f(dot * inv_hc));
  }
}

// The per-branch writes, one thread per element of the hc*H-wide row:
// R[row][i*H + j] += bf16(y[row][j] * gate_i).
__global__ void combine_apply_kernel(uint16_t* __restrict__ r_state,
                                     const float* __restrict__ gates,
                                     const uint16_t* __restrict__ y, int hc, int hidden,
                                     int width) {
  const int64_t row = blockIdx.y;
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (idx >= width) return;
  const int i = idx / hidden, j = idx - i * hidden;
  const float g = gates[row * hc + i];
  const float inj = round_bf16(bf16_bits_to_float(y[row * hidden + j]) * g);
  uint16_t* r = r_state + row * width + idx;
  *r = float_to_bf16_bits(bf16_bits_to_float(*r) + inj);
}

// ---- the site's GEMVs with their producers folded into the staging ----------
// (2026-09-09, the decode profile: the group norm was a 4-block, 7 us
// latency-bound launch before every down GEMV — 100 per step — and the
// gate activation a 1 us launch before every up GEMV.) The down GEMV's
// blocks each normalize the row while staging it (block_sum_squares in the
// norm kernel's own order per group, then the (1 + w) scale and the one
// rounding — the bytes the norm kernel would have written; block 0 writes
// them to Rn for the finish and the combine); the up GEMV's blocks apply
// gate_act_kernel's silu(t / hc) while staging t. The dots run
// bf16_gemv::row_dots on those staged values, so both outputs are bitwise
// the norm + GEMV and gate_act + GEMV chains.
constexpr int kGrGemvThreads = gemv::kThreads;  // 256: the norm kernel's block too

template <int kRows>
__global__ void __launch_bounds__(kGrGemvThreads)
    gr_norm_down_kernel(const uint16_t* __restrict__ r, size_t r_stride,
                        const uint16_t* __restrict__ norm_w, int hc, int hidden, float eps,
                        uint16_t* __restrict__ rn, const uint16_t* __restrict__ down_w,
                        uint16_t* __restrict__ t, int lowrank) {
  extern __shared__ __align__(16) uint16_t smem_u16[];
  const int W = hc * hidden;
  uint16_t* sx = smem_u16;  // [kRows, W] the normalized rows
  float* staged = reinterpret_cast<float*>(smem_u16 + static_cast<size_t>(kRows) * W);
  __shared__ float warp_sums[32];
#pragma unroll 1
  for (int row = 0; row < kRows; ++row) {
    for (int g = 0; g < hc; ++g) {
      const uint16_t* xr = r + static_cast<size_t>(row) * r_stride + static_cast<size_t>(g) * hidden;
      const uint16_t* wr = norm_w + static_cast<size_t>(g) * hidden;
      const float total = qwen_norm::block_sum_squares(xr, hidden, staged, warp_sums);
      const float rstd = rsqrtf(total / static_cast<float>(hidden) + eps);
      uint16_t* sx_g = sx + static_cast<size_t>(row) * W + static_cast<size_t>(g) * hidden;
      uint16_t* rn_g = rn ? rn + static_cast<size_t>(row) * W + static_cast<size_t>(g) * hidden : nullptr;
      for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float w1 = 1.0f + bf16_bits_to_float(wr[i]);
        const uint16_t v = float_to_bf16_bits(staged[i] * rstd * w1);
        sx_g[i] = v;
        if (rn_g && blockIdx.x == 0) rn_g[i] = v;
      }
      __syncthreads();  // staged / warp_sums reused by the next group
    }
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n_row = static_cast<int>(blockIdx.x) * gemv::kWarps + warp;
  if (n_row >= lowrank) return;
  float acc[kRows];
  bf16_gemv::row_dots<kRows>(down_w + static_cast<size_t>(n_row) * W, sx, W, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    t[static_cast<size_t>(row) * lowrank + n_row] = float_to_bf16_bits(acc[row]);
}

template <int kRows>
__global__ void __launch_bounds__(kGrGemvThreads)
    gr_act_up_kernel(const uint16_t* __restrict__ t, int lowrank, float inv_hc,
                     const uint16_t* __restrict__ up_w, uint16_t* __restrict__ logits, int W) {
  extern __shared__ __align__(16) uint16_t sx[];
  for (int i = threadIdx.x; i < kRows * lowrank; i += blockDim.x) {
    // gate_act_kernel, op for op: t / hc is exact in bf16, silu in fp32.
    const float v = bf16_bits_to_float(t[i]) * inv_hc;
    sx[i] = float_to_bf16_bits(v * sigmoid_f(v));
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n_row = static_cast<int>(blockIdx.x) * gemv::kWarps + warp;
  if (n_row >= W) return;
  float acc[kRows];
  bf16_gemv::row_dots<kRows>(up_w + static_cast<size_t>(n_row) * lowrank, sx, lowrank, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    logits[static_cast<size_t>(row) * W + n_row] = float_to_bf16_bits(acc[row]);
}

template <int kRows>
void launch_norm_down_rows(const uint16_t* r, size_t r_stride, const uint16_t* norm_w, int hc,
                           int hidden, float eps, uint16_t* rn, const uint16_t* down_w,
                           uint16_t* t, int lowrank, cudaStream_t stream) {
  const int W = hc * hidden;
  const size_t smem = gemv::smem_bytes(kRows, W) + static_cast<size_t>(hidden) * sizeof(float);
  // Beyond the 48 KB default (four 20 KB rows plus a group of floats at
  // the real shape): the opt-in tracks the largest launch per kRows.
  static size_t opted = 0;
  if (smem > opted && smem > gemv::kMaxSmemBytes) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(gr_norm_down_kernel<kRows>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    opted = smem;
  }
  const dim3 grid(static_cast<unsigned>((lowrank + gemv::kWarps - 1) / gemv::kWarps));
  gr_norm_down_kernel<kRows><<<grid, kGrGemvThreads, smem, stream>>>(
      r, r_stride, norm_w, hc, hidden, eps, rn, down_w, t, lowrank);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_act_up_rows(const uint16_t* t, int lowrank, int hc, const uint16_t* up_w,
                        uint16_t* logits, int W, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((W + gemv::kWarps - 1) / gemv::kWarps));
  gr_act_up_kernel<kRows><<<grid, kGrGemvThreads, gemv::smem_bytes(kRows, lowrank), stream>>>(
      t, lowrank, 1.0f / hc, up_w, logits, W);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool qwen_gr_fused_mix_accepts(int hc, int hidden, int lowrank) {
  const int W = hc * hidden;
  // The 99 KB per-block ceiling: four staged rows of W plus one group of
  // floats; W and lowrank in whole 16-byte chunks; hidden a multiple of 256
  // keeps every group's norm on the same strided lanes as the norm kernel.
  return hc >= 1 && hc <= 8 && hidden > 0 && hidden % 8 == 0 && lowrank > 0 && lowrank % 8 == 0 &&
         gemv::smem_bytes(4, W) + static_cast<size_t>(hidden) * 4 <= size_t{96} * 1024 &&
         gemv::smem_fits(4, lowrank);
}

void qwen_gr_norm_down_bf16(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden,
                            float eps, void* rn, const void* down_w, void* t, int lowrank,
                            int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!r || !norm_w || !down_w || !t) throw std::invalid_argument("qwen gr norm_down: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || r_stride < static_cast<size_t>(hc) * hidden ||
      !gemv::aligned16(r) || !gemv::aligned16(down_w) || (r_stride * 2) % 16 != 0)
    throw std::invalid_argument("qwen gr norm_down: shape outside the fused contract");
  const int W = hc * hidden;
  cudaGetLastError();
  for (int64_t row0 = 0; row0 < rows; row0 += gemv::kMaxRows) {
    const int n = static_cast<int>(rows - row0 < gemv::kMaxRows ? rows - row0 : gemv::kMaxRows);
    const uint16_t* rr = static_cast<const uint16_t*>(r) + static_cast<size_t>(row0) * r_stride;
    uint16_t* rnr = rn ? static_cast<uint16_t*>(rn) + static_cast<size_t>(row0) * W : nullptr;
    uint16_t* tr = static_cast<uint16_t*>(t) + static_cast<size_t>(row0) * lowrank;
    const uint16_t* nw = static_cast<const uint16_t*>(norm_w);
    const uint16_t* dw = static_cast<const uint16_t*>(down_w);
    switch (n) {
      case 1: launch_norm_down_rows<1>(rr, r_stride, nw, hc, hidden, eps, rnr, dw, tr, lowrank, stream); break;
      case 2: launch_norm_down_rows<2>(rr, r_stride, nw, hc, hidden, eps, rnr, dw, tr, lowrank, stream); break;
      case 3: launch_norm_down_rows<3>(rr, r_stride, nw, hc, hidden, eps, rnr, dw, tr, lowrank, stream); break;
      default: launch_norm_down_rows<4>(rr, r_stride, nw, hc, hidden, eps, rnr, dw, tr, lowrank, stream); break;
    }
  }
}

void qwen_gr_act_up_bf16(const void* t, int lowrank, int hc, const void* up_w, void* logits, int hidden,
                         int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!t || !up_w || !logits) throw std::invalid_argument("qwen gr act_up: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || !gemv::aligned16(t) || !gemv::aligned16(up_w))
    throw std::invalid_argument("qwen gr act_up: shape outside the fused contract");
  const int W = hc * hidden;
  cudaGetLastError();
  for (int64_t row0 = 0; row0 < rows; row0 += gemv::kMaxRows) {
    const int n = static_cast<int>(rows - row0 < gemv::kMaxRows ? rows - row0 : gemv::kMaxRows);
    const uint16_t* tr = static_cast<const uint16_t*>(t) + static_cast<size_t>(row0) * lowrank;
    uint16_t* lr = static_cast<uint16_t*>(logits) + static_cast<size_t>(row0) * W;
    const uint16_t* uw = static_cast<const uint16_t*>(up_w);
    switch (n) {
      case 1: launch_act_up_rows<1>(tr, lowrank, hc, uw, lr, W, stream); break;
      case 2: launch_act_up_rows<2>(tr, lowrank, hc, uw, lr, W, stream); break;
      case 3: launch_act_up_rows<3>(tr, lowrank, hc, uw, lr, W, stream); break;
      default: launch_act_up_rows<4>(tr, lowrank, hc, uw, lr, W, stream); break;
    }
  }
}

void qwen_gr_gate_act_bf16(void* t, int64_t rows, int r, int hc, cudaStream_t stream) {
  if (rows <= 0 || r <= 0 || hc <= 0) throw std::invalid_argument("qwen gr gate_act: empty problem");
  const int64_t n = rows * r;
  const int block = 256;
  const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
  cudaGetLastError();
  gate_act_kernel<<<grid, block, 0, stream>>>(static_cast<uint16_t*>(t), n, 1.0f / hc);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_gr_mix_finish_bf16(const void* logits, const void* rn, void* x, int64_t rows,
                             int hc, int hidden, cudaStream_t stream) {
  if (rows <= 0 || hc <= 0 || hidden <= 0) throw std::invalid_argument("qwen gr mix_finish: empty problem");
  const int block = 256;
  const dim3 grid(static_cast<unsigned>((hidden + block - 1) / block), static_cast<unsigned>(rows), 1);
  cudaGetLastError();
  mix_finish_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint16_t*>(logits), static_cast<const uint16_t*>(rn),
      static_cast<uint16_t*>(x), hc, hidden, 1.0f / hc);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_gr_combine_bf16(void* r_state, const void* rn, const void* w_inject,
                          const void* y, float* gates, int64_t rows, int hc, int hidden,
                          cudaStream_t stream) {
  if (rows <= 0 || hc <= 0 || hc > 8 || hidden <= 0)
    throw std::invalid_argument("qwen gr combine: empty problem or hc > 8");
  if (!gates) throw std::invalid_argument("qwen gr combine: gates scratch required");
  const int width = hc * hidden;
  cudaGetLastError();
  combine_dots_kernel<<<static_cast<unsigned>(rows), kCombineBlock, 0, stream>>>(
      static_cast<const uint16_t*>(rn), static_cast<const uint16_t*>(w_inject), gates, hc,
      width, 1.0f / hc);
  DGPP_CUDA_OK(cudaGetLastError());
  const dim3 grid(static_cast<unsigned>((width + kCombineBlock - 1) / kCombineBlock),
                  static_cast<unsigned>(rows), 1);
  combine_apply_kernel<<<grid, kCombineBlock, 0, stream>>>(
      static_cast<uint16_t*>(r_state), gates, static_cast<const uint16_t*>(y), hc, hidden,
      width);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
