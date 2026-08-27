#pragma once
// Host reference implementation of the KDA layer forward path.
//
// Two roles:
//  1. test oracle for the CUDA kernels (Acc=float mirrors device fp32 math;
//     Acc=double measures fp32 accumulation drift over long recurrences);
//  2. executable specification of the reference numerics — the Python dump
//     harness mirrors the same equations independently, so a disagreement
//     between the two references is itself a bug.
//
// All tensors are host memory. bf16 is carried as raw uint16 bits; boundary
// rounding points (GEMM outputs, conv/recurrent/norm outputs) match the
// device path exactly: fp32 accumulation inside, bf16 storage outside.
#include <cstddef>
#include <cstdint>

#include "models/kda_geometry.hpp"

namespace dgpp::kda_ref {

struct HostWeights {
  const uint16_t* in_proj = nullptr;  // [in_proj_cols, hidden] rows [q|k|v|b|f_a|g_a]
  const uint16_t* f_b = nullptr;      // [local_proj, head_dim]
  const uint16_t* g_b = nullptr;      // [local_proj, head_dim]
  const uint16_t* conv = nullptr;     // [conv_channels, conv_width]
  const float* a_log = nullptr;       // [local_heads]
  const float* dt_bias = nullptr;     // [local_proj]
  const uint16_t* o_norm = nullptr;   // [head_dim]
  const uint16_t* o_proj = nullptr;   // [hidden, local_proj]
};

// out[m,n] = bf16(sum_k act[m,k] * w[n,k]), accumulation in Acc.
// act rows may be strided (fused projection slices).
template <typename Acc>
void gemm_bf16(const uint16_t* act, int64_t act_row_stride, const uint16_t* w,
               uint16_t* out, int m, int n, int k);

// Causal depthwise conv + silu (see kda_causal_conv_silu_bf16 for layouts).
template <typename Acc>
void conv_silu(const uint16_t* src, int64_t src_row_stride, const uint16_t* w,
               uint16_t* state, int state_width, uint16_t* dst, int tokens,
               int channels, int conv_width);

// The KDA recurrence (see kda_recurrent_fwd). State is Acc-typed: float
// mirrors the device fp32 state slot, double is the high-precision oracle.
template <typename Acc>
void recurrent(const uint16_t* qkv, const uint16_t* g_raw,
               const uint16_t* beta_raw, int64_t beta_row_stride,
               const float* a_log, const float* dt_bias, Acc* state,
               uint16_t* out, int tokens, int heads, int k_dim, int v_dim,
               float lower_bound, float scale);

// Gated RMSNorm with sigmoid gate.
template <typename Acc>
void gated_rmsnorm(const uint16_t* x, const uint16_t* gate, const uint16_t* w,
                   uint16_t* y, int64_t rows, int dim, float eps);

// Full layer forward mirroring KdaLayer::enqueue. Updates state (Acc) and
// conv_state (bf16, [conv_channels, state_width]) in place; writes
// layer_out [tokens, hidden] and (optionally) core_out [tokens, local_proj].
template <typename Acc>
void layer_forward(const HostWeights& w, const KdaConfig& cfg,
                   const uint16_t* hidden_in, Acc* state, uint16_t* conv_state,
                   int conv_state_width, uint16_t* layer_out,
                   uint16_t* core_out, int tokens);

}  // namespace dgpp::kda_ref
