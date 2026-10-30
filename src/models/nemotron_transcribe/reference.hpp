#pragma once
// Host fp32 oracle for the Nemotron-H LM walk of Nemotron-3.5-Transcribe
// (docs/nemotron_transcribe_plan.md milestone 2): the 52-slot walk — embed,
// per-slot RMSNorm then the Mamba-2 / GQA-attention / MoE mixer, residual
// add — ending at norm_f and the head, computed one fp32 operation at a
// time with nothing fused (the repo builds -ffp-contract=off), so it can
// pin the CUDA kernels of milestone 3 and the device forward of milestone 4
// tensor for tensor.
//
// The semantics mirror the HF `nemotron_h` reference (transformers
// modeling_nemotron_h, the pure-torch fallback path — the same math the
// triton kernels compute): dt = clamp(softplus(dt_raw + dt_bias), min =
// time_step_min) and A = -exp(A_log); the recurrence per head is
// state *= exp(dt*A); state += dt * B (outer) x; out = state . C + D * x
// with the D skip on the pre-dt x; the gated norm silu-gates BEFORE the
// group variance (norm_before_gate = false), groups of d_inner / n_groups;
// the router is sigmoid on fp32-dequantized logits, top-k chosen on
// scores + e_score_correction_bias (group-limited through n_group /
// topk_group, a no-op at the release's 1/1), weights from the unbiased
// scores, renormalized (+1e-20) and scaled by routed_scaling_factor; the
// experts are non-gated relu2 MLPs in the checkpoint's stacked bmm layout
// (gate_and_up [E, H, I], down [E, I, H]); attention is GQA and POSITION-
// FREE — the Nemotron-H paper (arXiv:2504.03624 §2.1): "We do not use any
// position embeddings" for the hybrid (the rope_theta keys in the config
// are vestigial; the pure-Transformer Nemotron-T variant used RoPE). Softmax
// in fp32. The rope_neox helpers here exist for the perception encoder,
// which does use half-rotary RoPE (milestone 5).
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/nemotron_transcribe/config.hpp"

namespace dgpp {
namespace nemotron_ref {

// One oracle weight: fp32 values, or the checkpoint's bf16 bits (dequantized
// at use — the fp32 the HF reference computes on). Exactly one vector is
// populated.
struct Tensor {
  std::vector<float> f32;
  std::vector<uint16_t> bf16;
  size_t numel() const { return f32.empty() ? bf16.size() : f32.size(); }
  float at(size_t i) const {
    return f32.empty() ? bf16_bits_to_float(bf16[i]) : f32[i];
  }
  const Tensor& operator=(std::vector<float> v) { f32 = std::move(v); bf16.clear(); return *this; }
  const Tensor& operator=(std::vector<uint16_t> v) { bf16 = std::move(v); f32.clear(); return *this; }
};

using Weights = std::unordered_map<std::string, Tensor>;

float softplus(float x);
float silu(float x);

// y[i] = dot(x, W row i) for W [n, k] row-major.
void gemv(float* y, const Tensor& W, const float* x, size_t n, size_t k);

// RMSNorm over the row: y[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps).
void rmsnorm(const float* x, const Tensor& w, float eps, size_t n, float* y);

// The Mamba-2 gated norm: t = y * silu(gate), each group of `group` lanes
// normalized by its own rms, then scaled by w.
void rmsnorm_gated(const float* y, const float* gate, const Tensor& w, size_t n,
                   size_t group, float eps, float* out);

// Depthwise causal conv over `channels` ([seq, channels] row-major):
// out[t][c] = silu(bias[c] + sum_k weight[c][k] * u[t - (K-1) + k][c]),
// zero-padded on the left. weight is the checkpoint's [c, 1, K] flattening.
void causal_conv_silu(const float* u, size_t seq, size_t channels,
                      const Tensor& weight, const Tensor* bias, size_t kernel,
                      float* out);

// The selective scan, sequential form (single-chunk-exact against the
// reference chunk scan): x [seq, heads*head_dim], B/C [seq, groups*state],
// dt [seq, heads] RAW (bias + softplus + dt clamp applied here), A_log/D/
// dt_bias [heads] F32. out is pre-gate, pre-norm.
struct MambaGeom {
  size_t heads = 0, head_dim = 0, groups = 0, state = 0;
  float dt_min = 0.001f;
};
void mamba2_walk(const MambaGeom& g, const float* x, const float* B, const float* C,
                 const float* dt, const float* A_log, const float* D,
                 const float* dt_bias, size_t seq, float* out);

// The router: fills descending-by-index-sorted top-k. n_group > 1 limits
// the pool to the topk_group best groups by the sum of their top-2 biased
// scores (the release's 1/1 is the no-op path).
struct RouterResult {
  std::vector<size_t> index;
  std::vector<float> weight;
};
RouterResult router_select(const float* x, const Tensor& gate /*[E,H]*/,
                           const float* bias /*[E]*/, size_t E, size_t H,
                           size_t topk, size_t n_group, size_t topk_group,
                           bool norm_topk, float scale);

// out[H] = down . relu2(up . x): the non-gated expert MLP (up [I, H],
// down [H, I]; the stacked routed form takes the checkpoint layouts
// gate_and_up [E, H, I], down [E, I, H] and accumulates the top-k).
void relu2(float* v, size_t n);
void relu2_mlp(const float* x, const Tensor& up, const Tensor& down, size_t H,
               size_t I, float* out);
void experts_sum(const float* x, const RouterResult& sel, const Tensor& gate_up,
                 const Tensor& down, size_t E, size_t H, size_t I, float* out);

// NeoX RoPE, in place on row-major [seq, heads*head_dim] q (all heads) and
// [seq, kv_heads*head_dim] k rows, positions pos0..pos0+seq-1.
void rope_neox(float* q, size_t seq, size_t heads, size_t head_dim, size_t pos0,
               double theta);
void rope_neox_k(float* k, size_t seq, size_t kv_heads, size_t head_dim, size_t pos0,
                 double theta);

// Causal GQA attention over full rows: q [seq, heads*hd], k/v [seq, kv*hd],
// out [seq, heads*hd] (pre o_proj); scale is the head's 1/sqrt(hd).
void gqa_attention(const float* q, const float* k, const float* v, size_t seq,
                   size_t heads, size_t kv_heads, size_t head_dim, float scale,
                   float* out);

// One mixer's output, given the slot-normed input (the per-slot geometry
// from the config; weights are named as the checkpoint names them, minus
// the layer prefix — in_proj/conv1d/A_log/... ; the mamba tensors are read
// through the named ones). Provided slot-wise so the device kernels can be
// pinned one mixer at a time.
void mixer_mamba(const NemotronTranscribeLmConfig& lm, const Weights& w,
                 const std::string& prefix, const float* x, size_t seq, float* out);
void mixer_attention(const NemotronTranscribeLmConfig& lm, const Weights& w,
                     const std::string& prefix, const float* x, size_t seq, float* out);
void mixer_moe(const NemotronTranscribeLmConfig& lm, const Weights& w,
               const std::string& prefix, const float* x, size_t seq, float* out);

// The full base-LM walk (no MTP): embed ids, the config's slot list,
// norm_f, head. With last_logits_only only the final position's logits are
// computed (the head is 131 072 rows). When `record`, per-slot mixer inputs
// (post slot-norm) and outputs are captured for oracle-vs-engine diffs.
struct WalkResult {
  std::vector<float> logits;  // [rows, vocab]; rows = seq or 1
  std::vector<std::vector<float>> normed;   // per slot [seq, H]
  std::vector<std::vector<float>> mixer_out;  // per slot [seq, H]
};
WalkResult walk(const NemotronTranscribeLmConfig& lm, const Weights& w,
                const uint32_t* ids, size_t seq, bool record, bool last_logits_only);

}  // namespace nemotron_ref
}  // namespace dgpp
