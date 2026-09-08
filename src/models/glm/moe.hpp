#pragma once
// MoE routing and expert geometry for the GLM text model (M4 deliverables 1
// and 5), CUDA-free so config, oracle, and kernels share the types.
//
// Semantics pinned to the transformers Glm5NextTextTopkRouter /
// Glm5NextTextExperts / Glm5NextTextMLP references (DESIGN §7.4):
//
// Router (noaux_tc, sigmoid):
//   logits = fp32 GEMM of the post-LN hidden (bf16 values cast to f32)
//            against the bf16 gate rows cast to f32.
//   scores = sigmoid(logits)                          [selection uses]
//   biased = scores + e_score_correction_bias         [selection uses]
//   top-k over biased, descending; TIES -> lower expert id (engine rule;
//   torch CUDA topk ties are unspecified).
//   weights = scores gathered at the selected ids (UNCORRECTED values),
//   then norm_topk_prob: w /= (sum + 1e-20), then w *= routed_scaling_factor
//   (2.5). All fp32, division per element.
//
// Experts (swiglu with asymmetric clamps):
//   gate = clamp_max(x @ Wg^T, limit)   — NO lower clamp on the gate
//   up   = clamp(x @ Wu^T, -limit, limit)
//   act  = bf16(bf16(silu_f32(gate)) * up)             — two roundings
//   y    = act @ Wd^T                                  — bf16 out
//
// Accumulation (per token, ASCENDING expert id — the reference's index_add
// over experts arrives in ascending order):
//   out = 0; for e ascending: out = bf16(out + bf16(w_e * y_e));
//   out = bf16(out + shared(x))    — shared expert added last, weight 1.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "models/quant_matrix.hpp"

namespace dgpp {

struct GlmMoeConfig {
  int hidden = 4096;
  int inter = 2048;          // moe_intermediate_size (per expert)
  int n_experts = 288;
  int top_k = 8;
  int n_shared_experts = 1;  // parser pins 1 (un-indexed tensor names)
  float routed_scaling_factor = 2.5f;
  bool norm_topk_prob = true;
  float swiglu_limit = 10.0f;

  // Weight bytes of one routed expert (payload + block scales): the number
  // the traffic model charges per selected expert.
  int64_t expert_bytes() const {
    const int64_t sc = 4 * ((inter + 127) / 128) * ((hidden + 127) / 128);
    return 3LL * inter * hidden + 3 * sc;  // gate, up: [I,H]; down: [H,I]
  }
  // The same expert in NVFP4: half a byte per element, an e4m3 scale per
  // 16, one F32 global per matrix.
  int64_t expert_bytes_fp4() const {
    const int64_t elems = 3LL * inter * hidden;
    return elems / 2 + elems / 16 + 3 * 4;
  }

  static void validate_config(const GlmMoeConfig& c) {
    auto fail = [](const char* what) {
      throw std::invalid_argument(std::string("GlmMoeConfig: ") + what);
    };
    if (c.hidden <= 0 || c.inter <= 0) fail("hidden/inter must be positive");
    if (c.n_experts <= 0) fail("n_experts must be positive");
    if (c.top_k <= 0 || c.top_k > 16)
      fail("top_k must be in [1, 16] (kernel register selection)");
    if (c.top_k > c.n_experts) fail("top_k must not exceed n_experts");
    if (c.n_experts > 4096)
      fail("n_experts must be <= 4096 (router smem: 2*n_experts floats)");
    if (c.n_shared_experts != 1)
      fail("only the single shared expert is implemented");
    if (!(c.swiglu_limit > 0)) fail("swiglu_limit must be positive");
  }
};

// Device-pointer weight view (GlmLayerStream's GlmMoeResident wires this).
//
// TP partition (2026-09-02, expert slicing): the router is REPLICATED and
// scores all cfg.n_experts on every rank (selection must be rank-identical);
// EVERY rank holds EVERY expert, sliced on the intermediate dimension —
// gate/up rows and down columns [rank*I/world, (rank+1)*I/world), exactly as
// the shared expert and the dense MLPs are sliced — and produces a partial
// hidden sum for the FFN all-reduce. `experts` therefore always holds
// n_experts triples; the slice width is the matrices' own rows/cols (world=1:
// the full I). Slicing instead of assigning whole experts to ranks keeps the
// per-rank bytes identical for every routing (top_k * 3 slices, always),
// which is what removes the busiest-rank wait at the FFN boundary.
struct GlmMoeWeights {
  const uint16_t* router_gate = nullptr;  // bf16 [n_experts, hidden]
  const float* router_bias = nullptr;     // f32 [n_experts]
  GlmQuantMatrix shared[3];               // gate, up, down (compressed, FP8)
  // The routed experts in EXACTLY ONE of the two formats (the layer's
  // GlmExpertFormat): FP8 block-128 triples or NVFP4 triples, [n_experts *
  // 3] gate,up,down either way. The shared expert is FP8 under both.
  const GlmQuantMatrix* experts = nullptr;
  const GlmFp4Matrix* experts_fp4 = nullptr;
  bool nvfp4() const { return experts_fp4 != nullptr; }
};

// The decode path's device-side expert table entry (2026-09-01): the
// slot kernels read the ROUTE from device memory, so the weight views
// they indirect through must live there too. Dims stay kernel args (all
// routed experts share them; only the shared expert's inter differs).
// Layout-compatible with nothing — one type, one producer (the layer's
// per-binding upload), the slot kernels its consumers.
struct MoeExpertView {
  const uint8_t* payload = nullptr;  // fp8: e4m3 [n, k]; nvfp4: e2m1 pairs [n, k/2]
  const float* scales = nullptr;     // fp8: F32 128x128 block scales (nvfp4: null)
  const uint8_t* fp4_scales = nullptr;  // nvfp4: e4m3 [n, k/16] (fp8: null)
  const float* fp4_global = nullptr;    // nvfp4: the matrix's F32 global scale
  static MoeExpertView of(const GlmQuantMatrix& m) {
    return MoeExpertView{m.payload, m.scales, nullptr, nullptr};
  }
  static MoeExpertView of(const GlmFp4Matrix& m) {
    return MoeExpertView{m.payload, nullptr, m.scales, m.global_scale};
  }
};

}  // namespace dgpp
