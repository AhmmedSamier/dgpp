#pragma once
// GLM decoder-path elementwise kernels (M4 chunk 6, DESIGN §7.5): the
// two-rounding RMSNorm and the mHC stream initialization.
//
// The norm reproduces the transformers Glm5NextTextRMSNorm choreography
// bit-for-bit: fp32 interior, then
//     u  = bf16(x * rsqrt(mean(x^2) + eps))     (round 1)
//     y  = bf16(w * u)                          (round 2 — bf16 multiply)
// The generic rmsnorm_bf16 (kernels.hpp) rounds ONCE and serves the toy
// doll; the GLM path must not mix the two.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// y[rows, dim] = two-rounding RMSNorm of x[rows, dim] with weight[dim].
// bf16 io, fp32 accumulation, deterministic (fixed reduction order).
void glm_rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                      int dim, float eps, cudaStream_t stream);

// Initializes the mHC residual streams: streams[t, s, h] = embed[tokens[t],
// h] for all s < hc_mult (the reference's expand of the embedding). hc_mult
// is pinned to 4 by the config parser (kernel smem layouts assume it).
// streams: bf16 [num_tokens, 4, hidden]; embed_table: bf16 [vocab, hidden].
void glm_embed_bcast_streams(const void* embed_table, const int64_t* tokens,
                             void* streams, int num_tokens, int hidden,
                             cudaStream_t stream);

}  // namespace dgpp
