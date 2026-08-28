#pragma once
// Launchers for the MoE kernels (DESIGN §7.4; semantics in
// models/glm_moe.hpp). All deterministic (fixed reduction/selection order)
// and CUDA-graph capturable: no scratch, no host reads.
#include <cuda_runtime.h>

#include "models/glm_moe.hpp"

namespace dgpp {

// Router: hidden bf16 [tokens, hidden] -> ids int32 [tokens, top_k] in
// ASCENDING expert order (the accumulation order the reference's index_add
// produces), weights f32 [tokens, top_k] normalized and scaled.
void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       const GlmMoeConfig& cfg, int tokens,
                       cudaStream_t stream);

// swiglu with asymmetric clamps: gate clamp_max only, up clamp both; two
// bf16 rounding points (silu result, then the product). n = rows*inter.
void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream);

// dst[r, :] = src[rows[r], :] for n rows of width hidden.
void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream);

// acc[rows[r], c] = bf16(acc + bf16(row_w[r] * y[r, c])) — the reference's
// per-expert contribution rounding and bf16 accumulation, one expert's
// segment at a time (host loops experts in ascending order).
void launch_moe_accum(uint16_t* acc, const uint16_t* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream);

}  // namespace dgpp
