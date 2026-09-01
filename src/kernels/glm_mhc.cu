#include "kernels/glm_mhc_launch.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;
// n = 4 pinned by GlmMhcConfig::validate; static smem sized for exactly that.
constexpr int kN = 4;
constexpr int kCoeffs = (2 + kN) * kN;  // 24: pre[4] | post[4] | comb[16]

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The mHC site in two kernels (2026-09-01, the T=1 profile). The original
// one-block-per-token kernel streamed the 24 x (4*hidden) coefficient
// matrix (786KB at real dims) through ONE SM with one thread's worth of
// loads in flight — 212us per site, 90 sites a step, 19ms of a 190ms
// decode step for what is ~4us of reads at line rate. Split: the dots
// kernel gives each COEFFICIENT its own block, the finish kernel derives
// pre/post/comb and collapses. The numerics are the old kernel's bit for
// bit: every thread computes the same strided partials (d = tid + 256*m,
// FMA-sequential in m), thread 0 sums the 256 partials in the same
// sequential order, and each block re-derives the identical inv_rms from
// the identical sum-of-squares chain (deterministic, so 24 copies agree).
//
// dots kernel: grid (kCoeffs, tokens). Phase A: cooperative sum of squares
// (pass 1) then THIS block's coefficient dot against the NORMED flat vector
// (pass 2 — the reference normalizes every element before the linear, and
// we reproduce that rounding order). Phase B: fixed-order sequential
// reduction across the 256 partials -> logits[token][coeff].
__global__ void mhc_dots_kernel(const uint16_t* __restrict__ streams,
                                const uint16_t* __restrict__ fn,
                                float* __restrict__ logits, int tokens,
                                int hidden, float norm_eps) {
  __shared__ float partial[kThreads];
  __shared__ float inv_rms;

  const int coeff = blockIdx.x;
  const int token = blockIdx.y;
  if (coeff >= kCoeffs || token >= tokens) return;
  const int K = kN * hidden;
  const uint16_t* x = streams + static_cast<size_t>(token) * K;

  // Pass 1: sum of squares of the flattened streams (fixed stride order).
  float ssq = 0.f;
  for (int d = threadIdx.x; d < K; d += kThreads) {
    const float v = bf16_bits_to_float(x[d]);
    ssq = __fmaf_rn(v, v, ssq);
  }
  partial[threadIdx.x] = ssq;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int t = 0; t < kThreads; ++t)
      s += partial[t];  // deterministic sequential order
    inv_rms = rsqrtf(s / static_cast<float>(K) + norm_eps);
  }
  __syncthreads();
  const float r = inv_rms;

  // Pass 2: this coefficient's dot against flat_norm[d] = x[d] * r — the
  // reference norm output is fp32 (not rounded to bf16) and stays fp32
  // into the linear; x[d]*r in fp32 is exactly its per-element multiply.
  const uint16_t* fn_row = fn + static_cast<size_t>(coeff) * K;
  float dot = 0.f;
  for (int d = threadIdx.x; d < K; d += kThreads) {
    const float v = bf16_bits_to_float(x[d]) * r;
    dot = __fmaf_rn(v, bf16_bits_to_float(fn_row[d]), dot);
  }
  __syncthreads();  // partial[] reuse: pass-1 reads are done
  partial[threadIdx.x] = dot;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int t = 0; t < kThreads; ++t) s += partial[t];
    logits[static_cast<size_t>(token) * kCoeffs + coeff] = s;
  }
}

// finish kernel: grid (tokens). Thread 0 derives pre/post/comb from the 24
// logits (sigmoid/softmax/Sinkhorn in registers) and everyone folds pre
// into the collapse.
__global__ void mhc_finish_kernel(const uint16_t* __restrict__ streams,
                                  const float* __restrict__ logits_in,
                                  const float* __restrict__ base,
                                  const float* __restrict__ scale,
                                  uint16_t* __restrict__ collapsed,
                                  uint16_t* __restrict__ post_out,
                                  uint16_t* __restrict__ comb_out, int tokens,
                                  int hidden, float hc_eps,
                                  int sinkhorn_iters) {
  __shared__ float pre[kN];

  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int K = kN * hidden;
  const uint16_t* x = streams + static_cast<size_t>(token) * K;

  if (threadIdx.x == 0) {
    float logits[kCoeffs];
#pragma unroll
    for (int i = 0; i < kCoeffs; ++i)
      logits[i] = logits_in[static_cast<size_t>(token) * kCoeffs + i];
    // pre = sigmoid(pre_w * scale[0] + pre_b) + eps
    float pre_v[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i)
      pre_v[i] = sigmoidf_acc(logits[i] * scale[0] + base[i]) + hc_eps;
#pragma unroll
    for (int i = 0; i < kN; ++i) pre[i] = pre_v[i];
    // post = 2 * sigmoid(post_w * scale[1] + post_b)
    float post_v[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i)
      post_v[i] = 2.f * sigmoidf_acc(logits[kN + i] * scale[1] +
                                     base[kN + i]);
    // comb = softmax over each ROW of comb_logits * scale[2] + base, + eps.
    // The softmax max is taken AFTER the affine transform: scale[2] may be
    // negative, in which case it is not a plain shift and the raw-logit max
    // would not stabilize the exponentials.
    float c[kN][kN];
#pragma unroll
    for (int row = 0; row < kN; ++row) {
      float m = -INFINITY;
#pragma unroll
      for (int col = 0; col < kN; ++col) {
        const float v = logits[2 * kN + row * kN + col] * scale[2] +
                        base[2 * kN + row * kN + col];
        m = fmaxf(m, v);
      }
      float denom = 0.f;
#pragma unroll
      for (int col = 0; col < kN; ++col) {
        const float v = logits[2 * kN + row * kN + col] * scale[2] +
                        base[2 * kN + row * kN + col];
        c[row][col] = expf(v - m);
        denom += c[row][col];
      }
      const float inv = 1.0f / denom;
#pragma unroll
      for (int col = 0; col < kN; ++col) c[row][col] = c[row][col] * inv;
    }
#pragma unroll
    for (int row = 0; row < kN; ++row)
#pragma unroll
      for (int col = 0; col < kN; ++col) c[row][col] += hc_eps;
    // Sinkhorn: one column pass, then (iters-1) row+column passes.
    // column sum = sum over the FIRST index (torch dim=-2).
    {
      float colsum[kN];
#pragma unroll
      for (int col = 0; col < kN; ++col) colsum[col] = 0.f;
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col) colsum[col] += c[row][col];
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col)
          c[row][col] = c[row][col] / (colsum[col] + hc_eps);
    }
    for (int it = 1; it < sinkhorn_iters; ++it) {
      float rowsum[kN];
#pragma unroll
      for (int row = 0; row < kN; ++row) rowsum[row] = 0.f;
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col) rowsum[row] += c[row][col];
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col)
          c[row][col] = c[row][col] / (rowsum[row] + hc_eps);
      float colsum[kN];
#pragma unroll
      for (int col = 0; col < kN; ++col) colsum[col] = 0.f;
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col) colsum[col] += c[row][col];
#pragma unroll
      for (int row = 0; row < kN; ++row)
#pragma unroll
        for (int col = 0; col < kN; ++col)
          c[row][col] = c[row][col] / (colsum[col] + hc_eps);
    }

    const size_t t = static_cast<size_t>(token);
#pragma unroll
    for (int i = 0; i < kN; ++i)
      post_out[t * kN + i] = float_to_bf16_bits(post_v[i]);
#pragma unroll
    for (int row = 0; row < kN; ++row)
#pragma unroll
      for (int col = 0; col < kN; ++col)
        comb_out[t * kN * kN + row * kN + col] =
            float_to_bf16_bits(c[row][col]);
  }
  __syncthreads();  // pre[] published

  // Phase C: collapsed[d] = bf16(sum_j pre[j] * streams[j][d]), fp32.
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  for (int d = threadIdx.x; d < hidden; d += kThreads) {
    const float v =
        p0 * bf16_bits_to_float(x[d]) +
        p1 * bf16_bits_to_float(x[hidden + d]) +
        p2 * bf16_bits_to_float(x[2 * hidden + d]) +
        p3 * bf16_bits_to_float(x[3 * hidden + d]);
    collapsed[static_cast<size_t>(token) * hidden + d] =
        float_to_bf16_bits(v);
  }
}

__global__ void mhc_stream_update_kernel(const uint16_t* __restrict__ post,
                                         const uint16_t* __restrict__ comb,
                                         const uint16_t* __restrict__ sublayer,
                                         const uint16_t* __restrict__ streams_in,
                                         uint16_t* __restrict__ streams_out,
                                         int tokens, int hidden) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(tokens) * kN * hidden;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % hidden);
  const int i = static_cast<int>(idx / hidden) % kN;
  const int token = static_cast<int>(idx / (kN * hidden));
  const size_t t = static_cast<size_t>(token);

  const float pi = bf16_bits_to_float(post[t * kN + i]);
  const float h = bf16_bits_to_float(sublayer[t * hidden + d]);
  const uint16_t t1 = float_to_bf16_bits(pi * h);

  const uint16_t* res = streams_in + t * kN * hidden;
  float mix = 0.f;
#pragma unroll
  for (int j = 0; j < kN; ++j)
    mix = __fmaf_rn(bf16_bits_to_float(comb[t * kN * kN + j * kN + i]),
                    bf16_bits_to_float(res[j * hidden + d]), mix);
  const uint16_t t2 = float_to_bf16_bits(mix);

  streams_out[idx] =
      float_to_bf16_bits(bf16_bits_to_float(t1) + bf16_bits_to_float(t2));
}

__global__ void mhc_final_mean_kernel(const uint16_t* __restrict__ streams,
                                      uint16_t* __restrict__ out, int tokens,
                                      int hidden) {
  const size_t idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(tokens) * hidden;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % hidden);
  const int token = static_cast<int>(idx / hidden);
  const uint16_t* s = streams + static_cast<size_t>(token) * kN * hidden;
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < kN; ++j)
    sum += bf16_bits_to_float(s[j * hidden + d]);
  out[idx] = float_to_bf16_bits(sum * (1.0f / kN));
}

}  // namespace

void launch_mhc_compute(const uint16_t* streams, const GlmMhcWeights& w,
                        const GlmMhcConfig& cfg, uint16_t* collapsed,
                        uint16_t* post, uint16_t* comb, float* logits_scratch,
                        int tokens, cudaStream_t stream) {
  GlmMhcConfig::validate_config(cfg);
  if (tokens <= 0) return;
  if (!streams || !w.fn || !w.base || !w.scale || !collapsed || !post ||
      !comb || !logits_scratch)
    throw std::invalid_argument("mhc_compute: null pointer");
  if (tokens > 65535)
    throw std::invalid_argument("mhc_compute: grid dimension overflow");
  const dim3 dots_grid(kCoeffs, static_cast<unsigned>(tokens));
  mhc_dots_kernel<<<dots_grid, kThreads, 0, stream>>>(
      streams, w.fn, logits_scratch, tokens, cfg.hidden, cfg.norm_eps);
  DGPP_CUDA_OK(cudaGetLastError());
  mhc_finish_kernel<<<tokens, kThreads, 0, stream>>>(
      streams, logits_scratch, w.base, w.scale, collapsed, post, comb, tokens,
      cfg.hidden, cfg.hc_eps, cfg.sinkhorn_iters);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_mhc_stream_update(const uint16_t* post, const uint16_t* comb,
                              const uint16_t* sublayer_out,
                              const uint16_t* streams_in,
                              uint16_t* streams_out, const GlmMhcConfig& cfg,
                              int tokens, cudaStream_t stream) {
  GlmMhcConfig::validate_config(cfg);
  if (tokens <= 0) return;
  if (streams_in == streams_out)
    throw std::invalid_argument("mhc_stream_update: in/out must not alias");
  if (!post || !comb || !sublayer_out || !streams_in || !streams_out)
    throw std::invalid_argument("mhc_stream_update: null pointer");
  const size_t total = static_cast<size_t>(tokens) * 4 * cfg.hidden;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  mhc_stream_update_kernel<<<blocks, kThreads, 0, stream>>>(
      post, comb, sublayer_out, streams_in, streams_out, tokens, cfg.hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_mhc_final_mean(const uint16_t* streams, uint16_t* out,
                           const GlmMhcConfig& cfg, int tokens,
                           cudaStream_t stream) {
  GlmMhcConfig::validate_config(cfg);
  if (tokens <= 0) return;
  if (!streams || !out)
    throw std::invalid_argument("mhc_final_mean: null pointer");
  const size_t total = static_cast<size_t>(tokens) * cfg.hidden;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  mhc_final_mean_kernel<<<blocks, kThreads, 0, stream>>>(streams, out, tokens,
                                                         cfg.hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
