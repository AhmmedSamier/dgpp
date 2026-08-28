#pragma once
// Double-precision oracle for the mHC module (DESIGN §7.3), mirroring the
// structure of kda_reference/dsa_reference: the engine kernels' ground
// truth. It reproduces the reference implementation's dtype choreography
// (bf16 rounding points) but accumulates in double where the reference
// computes in fp32; parity budgets absorb that gap.
#include <cstdint>
#include <vector>

#include "models/glm_mhc.hpp"

namespace dgpp {

// Host-owned weights for the oracle (bf16 bits for fn, exact f32 for
// base/scale — the checkpoint storage).
struct GlmMhcWeightsHost {
  std::vector<uint16_t> fn;   // [(2+n)*n, n*D]
  std::vector<float> base;    // [(2+n)*n]
  std::vector<float> scale;   // [3]
};

struct GlmMhcRefResult {
  std::vector<double> pre;      // [tokens, n]
  std::vector<double> post;     // [tokens, n]
  std::vector<double> comb;     // [tokens, n, n]
  std::vector<uint16_t> collapsed;  // [tokens, D] bf16 bits
};

// The mHC mapping: norm + 24-logit projection + pre/post/comb derivation +
// collapse. pre is exported so tests can inspect the collapse weights.
void glm_mhc_ref_compute(const uint16_t* streams,
                         const GlmMhcWeightsHost& w,
                         const GlmMhcConfig& cfg, int tokens,
                         GlmMhcRefResult& out);

// The stream update, taking post/comb as BF16 BITS (the choreography rounds
// them before the products — the same inputs the kernel receives).
// streams_out[i] = bf16(bf16(post[i]*h) + bf16(sum_j comb[j,i]*streams_in[j]))
void glm_mhc_ref_stream_update(const uint16_t* post_bf16,
                               const uint16_t* comb_bf16,
                               const uint16_t* sublayer_out,
                               const uint16_t* streams_in,
                               const GlmMhcConfig& cfg, int tokens,
                               uint16_t* streams_out);

// Final head: out[d] = bf16(mean over streams).
void glm_mhc_ref_final_mean(const uint16_t* streams, const GlmMhcConfig& cfg,
                            int tokens, uint16_t* out);

}  // namespace dgpp
