#pragma once
// KDA (Kimi Delta Attention) geometry contract, DESIGN §7.1.
//
// Shapes and byte formulas below are the single source of truth for the state
// arena, snapshot sizing, and admission accounting. The M2 exit criterion
// "geometry and bytes agree exactly with DESIGN.md §7.1" is tested against
// the constants spelled out in that section — change either side only with
// the other.
//
// Reference layout decisions (pinned to the vLLM GLM-5 kernels):
//   * recurrent state: [local_heads, head_v_dim, head_k_dim] FP32,
//     K contiguous — the fused_recurrent KDA kernel indexes
//     h0[i_hv*V*K + o_v[:,None]*K + o_k[None,:]].
//   * conv state: [channels, state_width] BF16, width contiguous
//     ("DS" conv layout: dim first, state_len last).
//   * conv channels = 3 * local_heads * head_dim (merged q|k|v), matching the
//     runtime-merged causal_conv1d weight cat([q, k, v], dim=0).
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dgpp {

// Checkpoint-level KDA configuration. Defaults are the GLM-5.3-Flash values
// (hidden 4096, 64 heads, head_dim 128, conv width 4). lower_bound comes from
// config `linear_attn_config.gate_lower_bound` — the config class default is
// -5.0; M4's config-driven assembly must read the trained value rather than
// trusting this default.
struct KdaConfig {
  int hidden = 4096;
  int heads = 64;          // global, head-sharded across TP
  int head_dim = 128;      // K == V == head_dim for this checkpoint
  int conv_width = 4;      // causal depthwise conv kernel size
  float lower_bound = -5.0f;  // bounded sigmoid gate floor (safe gate)
  int num_kda_layers = 34;    // of the 45 text layers
  int spec_width = 3;         // MTP draft reserve, in token positions
  int tp_size = 1;

  static void validate_config(const KdaConfig& c) {
    auto fail = [](const char* what) {
      throw std::invalid_argument(std::string("KdaConfig: ") + what);
    };
    if (c.hidden <= 0) fail("hidden must be positive");
    if (c.heads <= 0) fail("heads must be positive");
    if (c.head_dim <= 0) fail("head_dim must be positive");
    if (c.conv_width < 2) fail("conv_width must be >= 2");
    if (c.num_kda_layers <= 0) fail("num_kda_layers must be positive");
    if (c.spec_width < 0) fail("spec_width must be non-negative");
    if (c.tp_size <= 0) fail("tp_size must be positive");
    if (c.heads % c.tp_size != 0) fail("heads must divide by tp_size");
    // Fused-layout constraint: g_a's column offset is head_dim elements
    // (2*head_dim bytes) and must stay 16-byte aligned for the strided
    // f_b/g_b GEMMs; the recurrent kernel additionally needs head_dim % 4.
    if (c.head_dim % 8 != 0)
      fail("head_dim must be a multiple of 8 (fused layout alignment)");
    if (!(c.lower_bound < 0.0f)) fail("lower_bound must be negative");
  }
};

// Derived per-rank geometry. All fields are exact; no padding is added
// anywhere, so byte totals reproduce DESIGN §7.1 to the last byte.
struct KdaGeometry {
  int local_heads = 0;      // heads / tp_size
  int local_proj = 0;       // local_heads * head_dim
  int conv_channels = 0;    // 3 * local_proj (merged q|k|v)
  int conv_hist = 0;        // conv_width - 1 (committed conv state width)
  int conv_state_width = 0; // conv_hist + spec_width (allocated slot width)
  int in_proj_cols = 0;     // 2*head_dim + 3*local_proj + local_heads
  // Fused in-projection row layout: [f_a | g_a | q | k | v | b]. Leading
  // with the replicated f_a/g_a keeps the strided f_b/g_b GEMM inputs on
  // 16-byte-aligned column offsets; q/k/v/b trail (scalar-load consumers).
  static constexpr int kNumReplicatedProjs = 2;  // f_a, g_a

  size_t recurrent_elems = 0;  // local_heads * head_dim * head_dim
  size_t recurrent_bytes = 0;  // per layer/request (FP32)
  size_t conv_committed_bytes = 0;  // per layer/request (BF16, width conv_hist)
  size_t conv_slot_bytes = 0;       // per layer/request (BF16, full width)
  size_t slot_bytes = 0;            // per request, all layers
  size_t pool_bytes = 0;            // num_slots * slot_bytes

  static KdaGeometry from_config(const KdaConfig& c, int num_slots = 1) {
    KdaConfig::validate_config(c);
    KdaGeometry g;
    g.local_heads = c.heads / c.tp_size;
    g.local_proj = g.local_heads * c.head_dim;
    g.conv_channels = 3 * g.local_proj;
    g.conv_hist = c.conv_width - 1;
    g.conv_state_width = g.conv_hist + c.spec_width;
    g.in_proj_cols = 2 * c.head_dim + 3 * g.local_proj + g.local_heads;
    g.recurrent_elems =
        static_cast<size_t>(g.local_heads) * c.head_dim * c.head_dim;
    g.recurrent_bytes = g.recurrent_elems * 4;
    g.conv_committed_bytes =
        static_cast<size_t>(g.conv_channels) * g.conv_hist * 2;
    g.conv_slot_bytes =
        static_cast<size_t>(g.conv_channels) * g.conv_state_width * 2;
    g.slot_bytes = c.num_kda_layers * (g.recurrent_bytes + g.conv_slot_bytes);
    g.pool_bytes = static_cast<size_t>(num_slots) * g.slot_bytes;
    return g;
  }
};

}  // namespace dgpp
