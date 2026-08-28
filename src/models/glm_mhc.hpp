#pragma once
// mHC (manifold-constrained hyper-connections) module geometry and weights,
// CUDA-free so the host reference and the model-assembly library share the
// types (same split as kda_geometry/dsa_geometry vs the kernels).
//
// Semantics pinned to the transformers Glm5NextTextHyperConnection reference
// (revision matching config transformers_version 5.16.0), DESIGN §7.3:
//
//   streams        bf16 [tokens, n, D]; all n streams start as the embedding;
//                  the model output is the unweighted mean over streams.
//   flat           fp32 [tokens, n*D] = unweighted RMSNorm of the flattened
//                  streams (eps = rms_norm_eps), computed per token.
//   logits         fp32 [tokens, (2+n)*n] = flat @ fn^T  (fn bf16, cast f32).
//   pre            = sigmoid(logits[:, 0:n]  * scale[0] + base[0:n]) + hc_eps
//                  — stream-collapse weights feeding the sublayer.
//   post           = 2 * sigmoid(logits[:, n:2n] * scale[1] + base[n:2n])
//                  — per-stream block-output placement, range [0, 2].
//   comb           = softmax(logits[:, 2n:].view(n,n) * scale[2]
//                            + base[2n:].view(n,n), dim=-1) + hc_eps,
//                  then Sinkhorn-Knopp: one column normalization followed by
//                  (sinkhorn_iters - 1) row+column passes (divisions each add
//                  hc_eps to the denominator).
//   collapsed      bf16 [tokens, D] = sum_j pre[j] * streams[j] (fp32
//                  accumulate, one bf16 round).
//
// The sublayer output h is folded back by the caller (DESIGN §7.3):
//   streams'[i] = bf16(bf16(post[i] * h) + bf16(sum_j comb[j,i] * streams[j]))
// with post/comb rounded to bf16 BEFORE the products — the reference's dtype
// choreography, reproduced bit-for-bit by the kernel and the oracle.
//
// scale has 3 entries — one per OUTPUT (pre/post/comb), not per head; "mHC"
// refers to the manifold-constrained mixing, and this checkpoint trains
// n = hc_mult = 4 streams.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dgpp {

struct GlmMhcConfig {
  int hc_mult = 4;        // n residual streams
  int hidden = 4096;      // D
  int sinkhorn_iters = 20;
  float hc_eps = 1e-6f;   // additive epsilon: pre, comb, Sinkhorn denominators
  float norm_eps = 1e-5f; // rms_norm_eps of the flattened-stream norm

  int coeff_rows() const { return (2 + hc_mult) * hc_mult; }  // (2+n)*n

  static void validate_config(const GlmMhcConfig& c) {
    auto fail = [](const char* what) {
      throw std::invalid_argument(std::string("GlmMhcConfig: ") + what);
    };
    if (c.hc_mult != 4)
      fail("hc_mult must be 4 (kernel smem layout is pinned to this "
           "checkpoint's stream count; widen when a real one differs)");
    if (c.hidden <= 0 || c.hidden % 8 != 0) fail("hidden must be 8-aligned");
    if (c.sinkhorn_iters < 1) fail("sinkhorn_iters must be >= 1");
    if (!(c.hc_eps > 0)) fail("hc_eps must be positive");
    if (!(c.norm_eps > 0)) fail("norm_eps must be positive");
  }
};

// Device-pointer weight view (what GlmLayerStream's GlmMhcResident provides).
struct GlmMhcWeights {
  const uint16_t* fn = nullptr;    // bf16 [(2+n)*n, n*D]
  const float* base = nullptr;     // f32 [(2+n)*n]
  const float* scale = nullptr;    // f32 [3] — one per output (pre/post/comb)
};

}  // namespace dgpp
