#pragma once
// Launchers for the mHC module kernels (DESIGN §7.3; semantics documented in
// models/glm_mhc.hpp). All kernels are deterministic (fixed reduction order)
// and CUDA-graph capturable: no scratch, no host reads, no dynamic smem.
#include <cuda_runtime.h>

#include "models/glm_mhc.hpp"

namespace dgpp {

// Computes the mHC mapping for `tokens` rows:
//   streams   bf16 [tokens, n, D]      (the residual streams, pre-collapse)
//   collapsed bf16 [tokens, D] out     (weighted sum over streams, the
//                                       sublayer input)
//   post      bf16 [tokens, n] out     (block-output placement weights)
//   comb      bf16 [tokens, n, n] out  (stream mixer, ~doubly stochastic)
// pre is internal (consumed by the collapse) and not exported.
void launch_mhc_compute(const uint16_t* streams, const GlmMhcWeights& w,
                        const GlmMhcConfig& cfg, uint16_t* collapsed,
                        uint16_t* post, uint16_t* comb, int tokens,
                        cudaStream_t stream);

// Stream update after the sublayer: for every token,
//   streams_out[i] = bf16(bf16(post[i] * sublayer_out)
//                         + bf16(sum_j comb[j,i] * streams_in[j]))
// with post/comb already bf16 (as produced by launch_mhc_compute) — the
// reference's dtype choreography, including both intermediate roundings.
// streams_in and streams_out must not alias.
void launch_mhc_stream_update(const uint16_t* post, const uint16_t* comb,
                              const uint16_t* sublayer_out,
                              const uint16_t* streams_in,
                              uint16_t* streams_out, const GlmMhcConfig& cfg,
                              int tokens, cudaStream_t stream);

// Final head: out[d] = bf16(mean over streams), fp32 accumulate.
void launch_mhc_final_mean(const uint16_t* streams, uint16_t* out,
                           const GlmMhcConfig& cfg, int tokens,
                           cudaStream_t stream);

}  // namespace dgpp
