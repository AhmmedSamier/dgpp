#pragma once
// Double-precision oracle for the MoE module (DESIGN §7.4): fp32 router
// math, the bf16 expert chain up to the activations, then the engine's
// fp32 accumulation semantics in double — unrounded down dots, one fma per
// expert (ascending), shared last, one bf16 rounding of the sum (see
// glm_moe_layer.hpp). The strict path reproduces every bf16 rounding point
// the engine has, isolating the fp32-vs-double and mma-order gaps into the
// parity budgets.
#include <cstdint>
#include <vector>

#include "models/glm/moe.hpp"

namespace dgpp {

struct GlmMoeHostWeights {
  std::vector<uint16_t> router_gate;  // [E, H]
  std::vector<float> router_bias;     // [E]
  // Compressed expert/shared weights, byte-identical to what the engine
  // consumes (payload + block scales). FP8: 3 mats per expert then the
  // shared triple: gate, up, down. NVFP4 (`nvfp4` set): the routed experts
  // live in the fp4 vectors below (expert e matrix m at index e*3+m —
  // packed nibbles [rows, cols/2], e4m3 scales [rows, cols/16], one global
  // per matrix) and `payloads`/`scales` hold only the shared triple.
  std::vector<uint8_t> payloads;   // 3 mats per expert then shared: gate,up,down
  std::vector<float> scales;       // matching scale grids
  bool nvfp4 = false;
  // The shared expert in NVFP4 too (GLM-4.7): the fp4 vectors then hold
  // (E + 1) * 3 matrices, the shared triple last, and the fp8 vectors
  // are empty.
  bool shared_nvfp4 = false;
  std::vector<uint8_t> fp4_payloads;
  std::vector<uint8_t> fp4_scales;
  std::vector<float> fp4_globals;  // [E * 3] (or [(E + 1) * 3] with shared_nvfp4)
  int fp4_matrices(int n_experts) const { return (n_experts + (shared_nvfp4 ? 1 : 0)) * 3; }
};

// Expert payload layout inside GlmMoeHostWeights: expert e's matrix m at
// index e*3+m, shared at index E*3+... see glm_moe_reference.cpp — helpers
// below return views so callers never index by hand.
struct GlmQuantMatrixHost {
  const uint8_t* payload = nullptr;
  const float* scales = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
};
struct GlmFp4MatrixHost {
  const uint8_t* payload = nullptr;  // [rows, cols/2]
  const uint8_t* scales = nullptr;   // [rows, cols/16]
  float global_scale = 1.0f;
  int64_t rows = 0;
  int64_t cols = 0;
};

struct GlmMoeRouterRef {
  std::vector<int32_t> ids;       // [tokens, K] ascending
  std::vector<float> weights;     // [tokens, K] normalized + scaled
  std::vector<double> biased;     // [tokens, E] (near-tie certification)
};

// Router oracle. Same tie rule as the engine: biased descending, ties to the
// lower expert id.
void glm_moe_ref_router(const uint16_t* hidden, const uint16_t* gate,
                        const float* bias, const GlmMoeConfig& cfg,
                        int tokens, GlmMoeRouterRef& out);

// Full expert-path oracle (strict: bf16 rounding points exactly where the
// engine rounds): router -> per-expert swiglu MLPs in ascending order ->
// shared expert -> the chain's single rounding. hidden [tokens, H] bf16
// in, out [tokens, H] bf16 bits out.
void glm_moe_ref_forward(const uint16_t* hidden,
                         const GlmMoeHostWeights& w, const GlmMoeConfig& cfg,
                         int tokens, std::vector<uint16_t>& out);

// Build the host-weight views (expert e matrix m, and the shared triple).
GlmQuantMatrixHost glm_moe_host_view(const GlmMoeHostWeights& w,
                                     const GlmMoeConfig& cfg, int index);
GlmQuantMatrixHost glm_moe_host_shared(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int m);
// The NVFP4 expert view (index in [0, E*3), or [0, (E+1)*3) with
// shared_nvfp4); requires w.nvfp4.
GlmFp4MatrixHost glm_moe_host_view_fp4(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int index);

}  // namespace dgpp
