#pragma once
// Launchers for the mHC module kernels (DESIGN §7.3; semantics documented in
// models/glm_mhc.hpp). All kernels are deterministic (fixed reduction order)
// and CUDA-graph capturable: caller-owned scratch only, no host reads, no
// dynamic smem.
#include <cuda_runtime.h>

#include "models/glm/mhc.hpp"

namespace dgpp {

// Computes the mHC mapping for `tokens` rows:
//   streams   bf16 [tokens, n, D]      (the residual streams, pre-collapse)
//   collapsed bf16 [tokens, D] out     (weighted sum over streams, the
//                                       sublayer input)
//   post      bf16 [tokens, n] out     (block-output placement weights)
//   comb      bf16 [tokens, n, n] out  (stream mixer, ~doubly stochastic)
// pre is internal (consumed by the collapse) and not exported.
// logits_scratch: f32 [tokens, cfg.coeff_rows()] device scratch the caller
// owns — the hand-off between the per-coefficient dots kernel and the
// finish kernel (two launches; see glm_mhc.cu for why).
// Tests: switch the prefill-sized (>= 16 tokens, vector path) dots between
// the token-tiled form (default; bitwise the per-coefficient form) and the
// per-coefficient form.
void mhc_set_tiled_form(bool on);

void launch_mhc_compute(const uint16_t* streams, const GlmMhcWeights& w,
                        const GlmMhcConfig& cfg, uint16_t* collapsed,
                        uint16_t* post, uint16_t* comb, float* logits_scratch,
                        int tokens, cudaStream_t stream);

// The same, plus the sublayer's two-rounding RMSNorm of the collapsed row
// (glm_norm.hpp's semantics) in the finish kernel's tail:
//   normed bf16 [tokens, D] out = rmsnorm(collapsed, ln, ln_eps)
// One launch fewer per site than launch_mhc_compute + glm_rmsnorm_bf16;
// ln and normed are both null (plain compute) or both set.
// finish_counters (int32 [tokens] device scratch, zero at rest; the caller
// zeroes it once at allocation) fuses the finish phase into the dots
// launch: the last dots block of a token runs it. Null keeps the two-
// launch form. Returns true when comb was DEFERRED (defer_comb with the
// fused per-coefficient form): the caller must then launch_mhc_comb
// before anything reads comb; the tiled prefill form never defers.
bool launch_mhc_compute_normed(const uint16_t* streams, const GlmMhcWeights& w,
                               const GlmMhcConfig& cfg, uint16_t* collapsed,
                               uint16_t* post, uint16_t* comb,
                               float* logits_scratch, const uint16_t* ln,
                               uint16_t* normed, float ln_eps, int tokens,
                               cudaStream_t stream,
                               int* finish_counters = nullptr,
                               bool defer_comb = false);

// The deferred comb (2026-09-08; fused finish only): with defer_comb the
// finish above writes collapsed/post/normed and leaves comb to this
// launch, which reads the dots' logits_scratch (one warp per token, the
// in-block arithmetic exactly — bitwise). Only the stream update reads
// comb, so decode runs it on a side stream forked after the finish and
// joined before the update, off the sublayer's critical path.
void launch_mhc_comb(const float* logits_scratch, const GlmMhcWeights& w,
                     const GlmMhcConfig& cfg, uint16_t* comb, int tokens,
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
