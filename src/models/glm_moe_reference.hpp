#pragma once
// Double-precision oracle for the MoE module (DESIGN §7.4): fp32 router
// math, the bf16 expert chain up to the activations, then the engine's
// fp32 accumulation semantics in double — unrounded down dots, one fma per
// expert (ascending), shared last, ONE bf16 rounding of the sum (see
// glm_moe_layer.hpp). The strict path reproduces every bf16 rounding point
// the engine has, isolating the fp32-vs-double and mma-order gaps into the
// parity budgets.
#include <cstdint>
#include <vector>

#include "models/glm_moe.hpp"

namespace dgpp {

struct GlmMoeHostWeights {
  std::vector<uint16_t> router_gate;  // [E, H]
  std::vector<float> router_bias;     // [E]
  // Compressed expert/shared weights, byte-identical to what the engine
  // consumes (payload + block scales).
  std::vector<uint8_t> payloads;   // 3 mats per expert then shared: gate,up,down
  std::vector<float> scales;       // matching scale grids
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

}  // namespace dgpp
