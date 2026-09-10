#include "kernels/qwen_gr.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/fp8_gemv.cuh"
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

// The inject dots: one block per row, warp i (i < hc) reduces the i-th dot
// over the hc*H-wide Rn row (inject_dot_strided below); the fused mix
// (gr_norm_down_kernel) runs the same chain over its staged Rn as the row
// space's last hc rows, so its gates are bitwise this kernel's.
// gates[row*hc + i] = 2 * bf16(sigmoid(bf16(dot) / hc)).
constexpr int kCombineBlock = 256;

// One (row, branch) inject dot in the lane-strided order: lane l walks
// k = l, l + 32, ... with kDotBatch loads ahead of the chain, then the xor
// tree. `rn_row` may be global (this kernel) or the fused mix's staged copy
// (gr_norm_down_kernel) — the same values, the same FMA sequence, so the
// gate is bitwise either way. Kept in this order deliberately: on
// 2026-09-10 the 16-byte-chunk order (bf16_gemv::row_dots) was tried for
// these rows and flipped 2 of 4 greedy transcripts at near ties for no
// step time (T=1 is bandwidth-bound), so the decode's numerics stay put.
constexpr int kDotBatch = 32;
__device__ __forceinline__ float inject_dot_strided(const uint16_t* __restrict__ rn_row,
                                                    const uint16_t* __restrict__ w,
                                                    int width, int lane) {
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
  return acc;
}

__global__ void combine_dots_kernel(const uint16_t* __restrict__ rn,
                                    const uint16_t* __restrict__ w_inject,
                                    float* __restrict__ gates, int hc, int width,
                                    float inv_hc) {
  const int64_t row = blockIdx.x;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (warp >= hc) return;
  const float acc = inject_dot_strided(rn + row * width,
                                       w_inject + static_cast<size_t>(warp) * width,
                                       width, lane);
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

// The weight rows' form (2026-09-10, engine.dense_weights = "fp8"): BF16
// rows through bf16_gemv::row_dots, or block-FP8 rows (E4M3 payload, fp32
// 128 x 128 scales) through fp8_gemv::row_dots — the scale GEMM's own GEMV
// chain, so the FP8 outputs are bitwise the unfused fp8 chain's, as the
// BF16 ones are the bf16 chain's. `w` is the row space's base (bf16 or
// e4m3), `scales` the fp8 grid (null for bf16), k the row width.
template <int kRows, bool kFp8>
__device__ __forceinline__ void weight_row_dots(const void* __restrict__ w,
                                                const float* __restrict__ scales, int row,
                                                int k, const uint16_t* __restrict__ sx, int lane,
                                                float (&acc)[kRows]) {
  if constexpr (kFp8) {
    const int scale_cols = (k + 127) >> 7;
    fp8_gemv::row_dots<kRows>(static_cast<const uint8_t*>(w) + static_cast<size_t>(row) * k,
                              scales + static_cast<size_t>(row >> 7) * scale_cols, sx, k, lane, acc);
  } else {
    bf16_gemv::row_dots<kRows>(static_cast<const uint16_t*>(w) + static_cast<size_t>(row) * k, sx, k,
                               lane, acc);
  }
}

template <int kRows, bool kFp8>
__global__ void __launch_bounds__(kGrGemvThreads)
    gr_norm_down_kernel(const uint16_t* __restrict__ r, size_t r_stride,
                        const uint16_t* __restrict__ norm_w, int hc, int hidden, float eps,
                        uint16_t* __restrict__ rn, const void* __restrict__ down_w,
                        const float* __restrict__ down_s,
                        uint16_t* __restrict__ t, int lowrank,
                        const uint16_t* __restrict__ w_inject, float* __restrict__ gates,
                        float inv_hc) {
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
  // The combine's inject dots ride the mix (2026-09-10): a gate reads only
  // the normalized row, which every block has just staged, so the hc inject
  // rows become the row space's last rows — the same warp-per-row chain
  // (bf16_gemv::row_dots) as the down matrix's, over the staged Rn — and
  // what was a 21 us one-block kernel on the chain between the boundary
  // collective and the apply leaves the step.
  const int n_row = static_cast<int>(blockIdx.x) * gemv::kWarps + warp;
  const int rows_total = lowrank + (gates != nullptr ? hc : 0);
  if (n_row >= rows_total) return;
  if (n_row >= lowrank) {
    // An inject row: combine_dots_kernel's chain over the staged Rn.
    const int i = n_row - lowrank;
    const uint16_t* wi = w_inject + static_cast<size_t>(i) * W;
#pragma unroll 1
    for (int row = 0; row < kRows; ++row) {
      const float acc = inject_dot_strided(sx + static_cast<size_t>(row) * W, wi, W, lane);
      if (lane == 0) {
        const float dot = round_bf16(acc);  // the linear's bf16 output
        gates[static_cast<size_t>(row) * hc + i] =
            2.0f * round_bf16(sigmoid_f(dot * inv_hc));
      }
    }
    return;
  }
  float acc[kRows];
  weight_row_dots<kRows, kFp8>(down_w, down_s, n_row, W, sx, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    t[static_cast<size_t>(row) * lowrank + n_row] = float_to_bf16_bits(acc[row]);
}

// The batched rows' down GEMV with the inject rows appended (2026-09-10): for 2..4 rows the mix leaves the fused one-row kernel for
// the norm + GEMV chain, and the inject dots then ran as a side-stream
// kernel that returned about a third of its time (the chain is near the
// bandwidth wall; overlapped work competes for the same bytes). This
// kernel is the chain's down GEMV — the same staged rows
// (gemv::stage_activations) and row chain (bf16_gemv::row_dots) as
// bf16_gemv_kernel, so t is bitwise the chain's — with the hc inject rows
// as the row space's last rows, each the lane-strided chain over the
// staged Rn (combine_dots_kernel's order, so the gates are bitwise the
// standalone kernel's). No norm prologue: Rn is the norm kernel's output.
template <int kRows, bool kFp8>
__global__ void __launch_bounds__(kGrGemvThreads)
    gr_down_inject_kernel(const uint16_t* __restrict__ rn, int W,
                          const void* __restrict__ down_w, const float* __restrict__ down_s,
                          uint16_t* __restrict__ t,
                          int lowrank, const uint16_t* __restrict__ w_inject,
                          float* __restrict__ gates, int hc, float inv_hc) {
  extern __shared__ __align__(16) uint16_t sx_di[];
  gemv::stage_activations<kRows>(rn, static_cast<size_t>(W), W, sx_di);
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n_row = static_cast<int>(blockIdx.x) * gemv::kWarps + warp;
  if (n_row >= lowrank + hc) return;
  if (n_row >= lowrank) {
    const int i = n_row - lowrank;
    const uint16_t* wi = w_inject + static_cast<size_t>(i) * W;
#pragma unroll 1
    for (int row = 0; row < kRows; ++row) {
      const float acc = inject_dot_strided(sx_di + static_cast<size_t>(row) * W, wi, W, lane);
      if (lane == 0) {
        const float dot = round_bf16(acc);  // the linear's bf16 output
        gates[static_cast<size_t>(row) * hc + i] =
            2.0f * round_bf16(sigmoid_f(dot * inv_hc));
      }
    }
    return;
  }
  float acc[kRows];
  weight_row_dots<kRows, kFp8>(down_w, down_s, n_row, W, sx_di, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    t[static_cast<size_t>(row) * lowrank + n_row] = float_to_bf16_bits(acc[row]);
}

template <int kRows, bool kFp8>
void launch_down_inject_rows(const uint16_t* rn, int W, const void* down_w, const float* down_s,
                             uint16_t* t, int lowrank, const uint16_t* w_inject, float* gates,
                             int hc, cudaStream_t stream) {
  const size_t smem = gemv::smem_bytes(kRows, W);
  static size_t opted = 0;
  if (smem > opted && smem > gemv::kMaxSmemBytes) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(gr_down_inject_kernel<kRows, kFp8>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    opted = smem;
  }
  const dim3 grid(static_cast<unsigned>((lowrank + hc + gemv::kWarps - 1) / gemv::kWarps));
  gr_down_inject_kernel<kRows, kFp8><<<grid, kGrGemvThreads, smem, stream>>>(
      rn, W, down_w, down_s, t, lowrank, w_inject, gates, hc, 1.0f / static_cast<float>(hc));
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows, bool kFp8>
__global__ void __launch_bounds__(kGrGemvThreads)
    gr_act_up_kernel(const uint16_t* __restrict__ t, int lowrank, float inv_hc,
                     const void* __restrict__ up_w, const float* __restrict__ up_s,
                     uint16_t* __restrict__ logits, int W) {
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
  weight_row_dots<kRows, kFp8>(up_w, up_s, n_row, lowrank, sx, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int row = 0; row < kRows; ++row)
    logits[static_cast<size_t>(row) * W + n_row] = float_to_bf16_bits(acc[row]);
}

template <int kRows, bool kFp8>
void launch_norm_down_rows(const uint16_t* r, size_t r_stride, const uint16_t* norm_w, int hc,
                           int hidden, float eps, uint16_t* rn, const void* down_w,
                           const float* down_s, uint16_t* t, int lowrank,
                           const uint16_t* w_inject, float* gates, cudaStream_t stream) {
  const int W = hc * hidden;
  const size_t smem = gemv::smem_bytes(kRows, W) + static_cast<size_t>(hidden) * sizeof(float);
  // Beyond the 48 KB default (four 20 KB rows plus a group of floats at
  // the real shape): the opt-in tracks the largest launch per kRows.
  static size_t opted = 0;
  if (smem > opted && smem > gemv::kMaxSmemBytes) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(gr_norm_down_kernel<kRows, kFp8>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    opted = smem;
  }
  const bool inject = w_inject != nullptr && gates != nullptr;
  const int rows_total = lowrank + (inject ? hc : 0);
  const dim3 grid(static_cast<unsigned>((rows_total + gemv::kWarps - 1) / gemv::kWarps));
  gr_norm_down_kernel<kRows, kFp8><<<grid, kGrGemvThreads, smem, stream>>>(
      r, r_stride, norm_w, hc, hidden, eps, rn, down_w, down_s, t, lowrank,
      inject ? w_inject : nullptr, inject ? gates : nullptr,
      1.0f / static_cast<float>(hc));
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows, bool kFp8>
void launch_act_up_rows(const uint16_t* t, int lowrank, int hc, const void* up_w, const float* up_s,
                        uint16_t* logits, int W, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((W + gemv::kWarps - 1) / gemv::kWarps));
  gr_act_up_kernel<kRows, kFp8><<<grid, kGrGemvThreads, gemv::smem_bytes(kRows, lowrank), stream>>>(
      t, lowrank, 1.0f / hc, up_w, up_s, logits, W);
  DGPP_CUDA_OK(cudaGetLastError());
}

// The three public forms share their row loops: rows in chunks of
// gemv::kMaxRows (a row's chain never depends on how many rows share its
// launch), the kernel picked by the chunk's rows and the weights' form.
template <bool kFp8>
void norm_down_rows(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden, float eps,
                    void* rn, const void* down_w, const float* down_s, void* t, int lowrank,
                    int64_t rows, cudaStream_t stream, const void* w_inject, float* gates) {
  const int W = hc * hidden;
  cudaGetLastError();
  for (int64_t row0 = 0; row0 < rows; row0 += gemv::kMaxRows) {
    const int n = static_cast<int>(rows - row0 < gemv::kMaxRows ? rows - row0 : gemv::kMaxRows);
    const uint16_t* rr = static_cast<const uint16_t*>(r) + static_cast<size_t>(row0) * r_stride;
    uint16_t* rnr = rn ? static_cast<uint16_t*>(rn) + static_cast<size_t>(row0) * W : nullptr;
    uint16_t* tr = static_cast<uint16_t*>(t) + static_cast<size_t>(row0) * lowrank;
    const uint16_t* nw = static_cast<const uint16_t*>(norm_w);
    const uint16_t* wi = static_cast<const uint16_t*>(w_inject);
    float* gr = gates ? gates + static_cast<size_t>(row0) * hc : nullptr;
    switch (n) {
      case 1: launch_norm_down_rows<1, kFp8>(rr, r_stride, nw, hc, hidden, eps, rnr, down_w, down_s, tr, lowrank, wi, gr, stream); break;
      case 2: launch_norm_down_rows<2, kFp8>(rr, r_stride, nw, hc, hidden, eps, rnr, down_w, down_s, tr, lowrank, wi, gr, stream); break;
      case 3: launch_norm_down_rows<3, kFp8>(rr, r_stride, nw, hc, hidden, eps, rnr, down_w, down_s, tr, lowrank, wi, gr, stream); break;
      default: launch_norm_down_rows<4, kFp8>(rr, r_stride, nw, hc, hidden, eps, rnr, down_w, down_s, tr, lowrank, wi, gr, stream); break;
    }
  }
}

template <bool kFp8>
void down_inject_rows(const void* rn, const void* down_w, const float* down_s, void* t, int lowrank,
                      const void* w_inject, float* gates, int hc, int hidden, int64_t rows,
                      cudaStream_t stream) {
  const int W = hc * hidden;
  cudaGetLastError();
  for (int64_t row0 = 0; row0 < rows; row0 += gemv::kMaxRows) {
    const int n = static_cast<int>(rows - row0 < gemv::kMaxRows ? rows - row0 : gemv::kMaxRows);
    const uint16_t* rr = static_cast<const uint16_t*>(rn) + static_cast<size_t>(row0) * W;
    uint16_t* tr = static_cast<uint16_t*>(t) + static_cast<size_t>(row0) * lowrank;
    float* gr = gates + static_cast<size_t>(row0) * hc;
    const uint16_t* wi = static_cast<const uint16_t*>(w_inject);
    switch (n) {
      case 1: launch_down_inject_rows<1, kFp8>(rr, W, down_w, down_s, tr, lowrank, wi, gr, hc, stream); break;
      case 2: launch_down_inject_rows<2, kFp8>(rr, W, down_w, down_s, tr, lowrank, wi, gr, hc, stream); break;
      case 3: launch_down_inject_rows<3, kFp8>(rr, W, down_w, down_s, tr, lowrank, wi, gr, hc, stream); break;
      default: launch_down_inject_rows<4, kFp8>(rr, W, down_w, down_s, tr, lowrank, wi, gr, hc, stream); break;
    }
  }
}

template <bool kFp8>
void act_up_rows(const void* t, int lowrank, int hc, const void* up_w, const float* up_s, void* logits,
                 int hidden, int64_t rows, cudaStream_t stream) {
  const int W = hc * hidden;
  cudaGetLastError();
  for (int64_t row0 = 0; row0 < rows; row0 += gemv::kMaxRows) {
    const int n = static_cast<int>(rows - row0 < gemv::kMaxRows ? rows - row0 : gemv::kMaxRows);
    const uint16_t* tr = static_cast<const uint16_t*>(t) + static_cast<size_t>(row0) * lowrank;
    uint16_t* lr = static_cast<uint16_t*>(logits) + static_cast<size_t>(row0) * W;
    switch (n) {
      case 1: launch_act_up_rows<1, kFp8>(tr, lowrank, hc, up_w, up_s, lr, W, stream); break;
      case 2: launch_act_up_rows<2, kFp8>(tr, lowrank, hc, up_w, up_s, lr, W, stream); break;
      case 3: launch_act_up_rows<3, kFp8>(tr, lowrank, hc, up_w, up_s, lr, W, stream); break;
      default: launch_act_up_rows<4, kFp8>(tr, lowrank, hc, up_w, up_s, lr, W, stream); break;
    }
  }
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
                            int64_t rows, cudaStream_t stream, const void* w_inject,
                            float* gates) {
  if (rows <= 0) return;
  if (!r || !norm_w || !down_w || !t) throw std::invalid_argument("qwen gr norm_down: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || r_stride < static_cast<size_t>(hc) * hidden ||
      !gemv::aligned16(r) || !gemv::aligned16(down_w) || (r_stride * 2) % 16 != 0)
    throw std::invalid_argument("qwen gr norm_down: shape outside the fused contract");
  norm_down_rows<false>(r, r_stride, norm_w, hc, hidden, eps, rn, down_w, nullptr, t, lowrank, rows, stream,
                        w_inject, gates);
}

void qwen_gr_norm_down_fp8(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden,
                           float eps, void* rn, const uint8_t* down_p, const float* down_s, void* t,
                           int lowrank, int64_t rows, cudaStream_t stream, const void* w_inject,
                           float* gates) {
  if (rows <= 0) return;
  if (!r || !norm_w || !down_p || !down_s || !t) throw std::invalid_argument("qwen gr norm_down fp8: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || r_stride < static_cast<size_t>(hc) * hidden ||
      !gemv::aligned16(r) || !gemv::aligned16(down_p) || (r_stride * 2) % 16 != 0 || (hc * hidden) % 16 != 0)
    throw std::invalid_argument("qwen gr norm_down fp8: shape outside the fused contract");
  norm_down_rows<true>(r, r_stride, norm_w, hc, hidden, eps, rn, down_p, down_s, t, lowrank, rows, stream,
                       w_inject, gates);
}

void qwen_gr_down_inject_bf16(const void* rn, const void* down_w, void* t, int lowrank,
                              const void* w_inject, float* gates, int hc, int hidden,
                              int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!rn || !down_w || !t || !w_inject || !gates)
    throw std::invalid_argument("qwen gr down_inject: null pointer");
  const int W = hc * hidden;
  if (rows > 8 || hc < 1 || hc > 8 || W % 8 != 0 || lowrank <= 0 || !gemv::aligned16(rn) ||
      !gemv::aligned16(down_w) || !gemv::aligned16(w_inject) || (static_cast<size_t>(W) * 2) % 16 != 0 ||
      !gemv::smem_fits(1, W))
    throw std::invalid_argument("qwen gr down_inject: shape outside the contract");
  down_inject_rows<false>(rn, down_w, nullptr, t, lowrank, w_inject, gates, hc, hidden, rows, stream);
}

void qwen_gr_down_inject_fp8(const void* rn, const uint8_t* down_p, const float* down_s, void* t,
                             int lowrank, const void* w_inject, float* gates, int hc, int hidden,
                             int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!rn || !down_p || !down_s || !t || !w_inject || !gates)
    throw std::invalid_argument("qwen gr down_inject fp8: null pointer");
  const int W = hc * hidden;
  if (rows > 8 || hc < 1 || hc > 8 || W % 16 != 0 || lowrank <= 0 || !gemv::aligned16(rn) ||
      !gemv::aligned16(down_p) || !gemv::aligned16(w_inject) || !gemv::smem_fits(1, W))
    throw std::invalid_argument("qwen gr down_inject fp8: shape outside the contract");
  down_inject_rows<true>(rn, down_p, down_s, t, lowrank, w_inject, gates, hc, hidden, rows, stream);
}

void qwen_gr_act_up_bf16(const void* t, int lowrank, int hc, const void* up_w, void* logits, int hidden,
                         int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!t || !up_w || !logits) throw std::invalid_argument("qwen gr act_up: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || !gemv::aligned16(t) || !gemv::aligned16(up_w))
    throw std::invalid_argument("qwen gr act_up: shape outside the fused contract");
  act_up_rows<false>(t, lowrank, hc, up_w, nullptr, logits, hidden, rows, stream);
}

void qwen_gr_act_up_fp8(const void* t, int lowrank, int hc, const uint8_t* up_p, const float* up_s,
                        void* logits, int hidden, int64_t rows, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!t || !up_p || !up_s || !logits) throw std::invalid_argument("qwen gr act_up fp8: null pointer");
  if (rows > 8 || !qwen_gr_fused_mix_accepts(hc, hidden, lowrank) || !gemv::aligned16(t) ||
      !gemv::aligned16(up_p) || lowrank % 16 != 0)
    throw std::invalid_argument("qwen gr act_up fp8: shape outside the fused contract");
  act_up_rows<true>(t, lowrank, hc, up_p, up_s, logits, hidden, rows, stream);
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

void qwen_gr_combine_dots_bf16(const void* rn, const void* w_inject, float* gates,
                               int64_t rows, int hc, int hidden, cudaStream_t stream) {
  if (rows <= 0 || hc <= 0 || hc > 8 || hidden <= 0)
    throw std::invalid_argument("qwen gr combine: empty problem or hc > 8");
  if (!gates) throw std::invalid_argument("qwen gr combine: gates scratch required");
  const int width = hc * hidden;
  cudaGetLastError();
  combine_dots_kernel<<<static_cast<unsigned>(rows), kCombineBlock, 0, stream>>>(
      static_cast<const uint16_t*>(rn), static_cast<const uint16_t*>(w_inject), gates, hc,
      width, 1.0f / hc);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_gr_combine_apply_bf16(void* r_state, const float* gates, const void* y,
                                int64_t rows, int hc, int hidden, cudaStream_t stream) {
  if (rows <= 0 || hc <= 0 || hc > 8 || hidden <= 0)
    throw std::invalid_argument("qwen gr combine: empty problem or hc > 8");
  if (!gates) throw std::invalid_argument("qwen gr combine: gates scratch required");
  const int width = hc * hidden;
  cudaGetLastError();
  const dim3 grid(static_cast<unsigned>((width + kCombineBlock - 1) / kCombineBlock),
                  static_cast<unsigned>(rows), 1);
  combine_apply_kernel<<<grid, kCombineBlock, 0, stream>>>(
      static_cast<uint16_t*>(r_state), gates, static_cast<const uint16_t*>(y), hc, hidden,
      width);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_gr_combine_bf16(void* r_state, const void* rn, const void* w_inject,
                          const void* y, float* gates, int64_t rows, int hc, int hidden,
                          cudaStream_t stream) {
  qwen_gr_combine_dots_bf16(rn, w_inject, gates, rows, hc, hidden, stream);
  qwen_gr_combine_apply_bf16(r_state, gates, y, rows, hc, hidden, stream);
}

}  // namespace dgpp
