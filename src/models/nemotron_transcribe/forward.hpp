#pragma once
// Nemotron-3.5-Transcribe forward kernels (milestone 4): the CUDA kernels
// for the 52-slot hybrid walk. Pure BF16 activations, fp32 accumulation,
// matching src/models/nemotron_transcribe/reference.cpp bitwise. The Mamba-2
// conv/scan/gated-norm kernels live in src/kernels/mamba2.cu and are
// declared here for the walk's use.

#include <cstdint>
#include <cuda_runtime.h>

namespace dgpp {

// Embed lookup: out[t, :] = embed[tokens[t], :].  tokens is host [T].
void nemotron_embed(const uint16_t* embed, const int64_t* tokens, uint16_t* out,
                    int T, int H, cudaStream_t stream);

// RMSNorm: out[t] = w * x[t] * rsqrt(mean(x[t]^2) + eps), fp32 internal.
void nemotron_rmsnorm(const uint16_t* w, const uint16_t* x, uint16_t* out,
                      int T, int H, float eps, cudaStream_t stream);

// Residual add: out = a + b, elementwise over n bf16 values.
void nemotron_residual_add(uint16_t* out, const uint16_t* a, const uint16_t* b,
                           int n, cudaStream_t stream);

// GQA attention (position-free: no RoPE, no q/k norm).
//   q [T, q_heads*hd], k/v [T, kv_heads*hd], out [T, q_heads*hd]
//   scale = 1/sqrt(hd). Causal mask over the full [T, T] attention.
void nemotron_gqa_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                            uint16_t* out, int T, int q_heads, int kv_heads, int head_dim,
                            float scale, cudaStream_t stream);

// relu2 MoE: router sigmoid → top-k → stacked relu2 experts + shared expert.
//   x [T, H], gate [E, H], gate_bias [E] (F32),
//   gate_and_up [E, H, I], down [E, I, H],
//   shared_up [SI, H], shared_down [H, SI],
//   out [T, H]. topk = num_experts_per_tok, scale = routed_scaling_factor.
void nemotron_moe(const uint16_t* x, const uint16_t* gate, const float* gate_bias,
                  const uint16_t* gate_and_up, const uint16_t* down,
                  const uint16_t* shared_up, const uint16_t* shared_down,
                  uint16_t* out, int T, int H, int E, int I, int SI,
                  int topk, float scale, bool norm_topk, cudaStream_t stream);

// norm_f + lm_head: logits[t, :] = lm_head @ (norm_f(resid[t])), fp32.
//   resid [T, H] bf16, norm_w [H] bf16, lm_head [V, H] bf16,
//   logits [T, V] fp32.
void nemotron_head(const uint16_t* resid, const uint16_t* norm_w, const uint16_t* lm_head,
                   float* logits, int T, int H, int V, float eps, cudaStream_t stream);

// Mamba-2 wrappers (kernels/mamba2.cu).
void nemotron_mamba_conv_silu(const uint16_t* u, int64_t u_stride, const uint16_t* weight,
                              const uint16_t* bias, uint16_t* out, int64_t out_stride,
                              int seq, int channels, int kernel, cudaStream_t stream);
// Single-row conv step with persistent conv state (used for decode steps).
//   state: [channels][kernel-1] uint16 history, updated in place.
void mamba2_conv_silu_step(const void* u_row, void* state, const void* weight,
                           const void* bias, void* out_row, int channels, int kernel,
                           cudaStream_t stream);
void nemotron_mamba_scan(const uint16_t* x, int64_t x_stride, const uint16_t* B, int64_t B_stride,
                         const uint16_t* C, int64_t C_stride, const uint16_t* dt, int64_t dt_stride,
                         const float* A_log, const float* D, const float* dt_bias,
                         float* state, uint16_t* out, int64_t out_stride, int seq, int heads,
                         int head_dim, int groups, int state_size, float dt_min,
                         cudaStream_t stream);
void nemotron_mamba_gated_norm(const uint16_t* y, int64_t y_stride, const uint16_t* gate,
                               int64_t gate_stride, const uint16_t* weight, uint16_t* out,
                               int64_t out_stride, int seq, int d_inner, int group, float eps,
                               cudaStream_t stream);

// GEMM wrappers.
void nemotron_gemm_bf16(const uint16_t* A, const uint16_t* B, uint16_t* C, int M, int N, int K,
                        cudaStream_t stream);
void nemotron_gemm_f32(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K,
                       cudaStream_t stream);

// Elementwise silu.
void nemotron_silu(const uint16_t* in, uint16_t* out, int n, cudaStream_t stream);

}  // namespace dgpp
