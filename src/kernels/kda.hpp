#pragma once
// KDA (Kimi Delta Attention) forward kernels, DESIGN §7.1.
//
// The recurrence implemented here is pinned bit-for-bit (module fp32
// reduction order) to the vLLM GLM-5 reference kernels
// (flash_linear_attention/ops/fused_recurrent.py with IS_KDA/SIGMOID_BETA/
// COMPUTE_GATE/SAFE_GATE, and causal_conv1d with activation="silu"):
//
//   per head h, state S in R^{V x K} (fp32), per token t:
//     q, k   <- l2norm(q, k) with eps=1e-6 inside the sqrt
//     q      <- q * K^-1/2
//     g[k]   <- lower_bound / (1 + exp(-exp(A_log[h]) * (g_raw[h,k] + dt_bias[h,k])))
//     S[v,k] <- exp(g[k]) * S[v,k]                    (decay, K-major)
//     u[v]   <- v[v] - sum_k S[v,k] * k[k]             (delta error)
//     S[v,k] <- S[v,k] + sigmoid(beta_raw[h]) * u[v] * k[k]
//     o[v]   <- sum_k S[v,k] * q[k]                    (post-update read)
//
// One shared implementation serves chunked prefill and single-token decode:
// the only difference is the token count. State round-trips through memory
// in fp32, so carrying state across chunk boundaries is exact.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Causal depthwise short conv over the merged q|k|v channels with silu
// activation, matching the reference's runtime-merged causal_conv1d call.
//   src:            bf16 [tokens, channels], row stride src_row_stride elems
//   weight:         bf16 [channels, conv_width] (merged q|k|v conv weights)
//   conv_state:     bf16 [channels, state_width], in/out. Columns
//                   [0, conv_width-1) hold the committed history; the
//                   speculative reserve columns are left untouched (M8).
//   dst:            bf16 [tokens, channels] contiguous, silu applied.
// The state stores raw pre-activation inputs; after the call it holds the
// last conv_width-1 inputs (blended with the previous state when
// tokens < conv_width-1).
void kda_causal_conv_silu_bf16(const void* src, int64_t src_row_stride,
                               const void* weight, void* conv_state,
                               int state_width, void* dst, int tokens,
                               int channels, int conv_width,
                               cudaStream_t stream);

// Gated RMSNorm with sigmoid gate (reference o_norm: FusedRMSNormGated,
// activation="sigmoid"): y = rmsnorm(x) * w * sigmoid(gate), fp32 internal.
//   x/gate/y: bf16 [rows, dim] contiguous; weight: bf16 [dim].
void kda_gated_rmsnorm_sigmoid_bf16(const void* x, const void* gate,
                                    const void* weight, void* y, int64_t rows,
                                    int dim, float eps, cudaStream_t stream);

// The KDA recurrence above, sequential over tokens, FP32 state held in
// registers for the whole call. Handles any token count >= 1; decode is the
// tokens==1 case of the same code path.
//   qkv:      bf16 [tokens, H*K + H*K + H*V] contiguous (post-conv q|k|v)
//   g_raw:    bf16 [tokens, heads*k_dim] contiguous (f_b projection output)
//   beta_raw: bf16 [tokens, heads], row stride beta_row_stride elems
//   a_log:    fp32 [heads]; dt_bias: fp32 [heads, k_dim]
//   state:    fp32 [heads, v_dim, k_dim] in/out, k contiguous
//   out:      bf16 [tokens, heads, v_dim]
// Requires k_dim % 4 == 0 and k_dim in {32, 64, 128}.
void kda_recurrent_fwd(const void* qkv, const void* g_raw, const void* beta_raw,
                       int64_t beta_row_stride, const float* a_log,
                       const float* dt_bias, float* state, void* out,
                       int tokens, int heads, int k_dim, int v_dim,
                       float lower_bound, float scale, cudaStream_t stream);

}  // namespace dgpp
