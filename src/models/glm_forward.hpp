#pragma once
// GlmDiagnosticModel: the M4 deliverable-1 assembly (DESIGN §7.5) — the
// full text-model forward over the streaming resident loader. One layer is
// resident at a time; the KDA/DSA/MoE layer objects are constructed once
// (shape-keyed scratch and GEMM plans) and REBOUND to each layer's resident
// weight views. Semantics per site, pinned to the transformers
// Glm5NextTextDecoderLayer:
//
//   streams (all 4 = embedding) -> mHC compute -> collapsed
//   ln1 = two-rounding RMSNorm(collapsed)
//   attn = KDA or DSA layer (their own parity-tested contracts)
//   streams' = mHC stream update (post, comb, attn_out, streams)
//   ... same with ffn_hc / ln2 / dense-MLP-or-MoE ...
//   final = two-rounding RMSNorm(mean over streams) -> lm head (bf16 GEMM)
//
// This is the diagnostic mode: correctness first, one full device sync per
// layer load (the loader's contract) and one per MoE enqueue (router
// round-trip). CUDA graphs and device-side expert grouping are M5+.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "kernels/gemm.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_state.hpp"
#include "models/glm_loader.hpp"
#include "models/glm_moe_layer.hpp"
#include "models/glm_trace.hpp"
#include "models/kda_layer.hpp"

namespace dgpp {

class GlmDiagnosticModel {
 public:
  struct Outputs {
    std::vector<uint16_t> final_hidden_bits;  // bf16 [tokens, hidden]
    std::vector<uint16_t> logits_bits;        // bf16 [tokens, vocab]
    // One entry per MoE layer, in layer order (ids ascending per token).
    std::vector<GlmRouteTraceLayer> routes;
  };

  // `max_tokens` bounds a forward's token count; `max_cache_tokens` bounds
  // the DSA cache (rounded up to a block). Both also size scratch.
  GlmDiagnosticModel(const GlmTextConfig& cfg,
                     const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens);
  ~GlmDiagnosticModel();
  GlmDiagnosticModel(const GlmDiagnosticModel&) = delete;
  GlmDiagnosticModel& operator=(const GlmDiagnosticModel&) = delete;

  // Runs layers 0..num_hidden_layers-1 for `token_ids` (single request,
  // fresh KDA/DSA state every call — no cross-call state survives). Returns
  // host copies of the final hidden state, logits, and routing decisions.
  Outputs forward(const std::vector<int64_t>& token_ids);

  const GlmTextConfig& config() const { return cfg_; }
  int max_tokens() const { return max_tokens_; }

  // Host-side top-k over bf16 logits: value-descending, lowest-id
  // tie-break. k <= 64. One entry per row.
  static std::vector<std::vector<std::pair<int32_t, float>>> topk(
      const std::vector<uint16_t>& logits_bits, int64_t rows, int vocab,
      int k);

 private:
  void enqueue_dense_mlp(const uint16_t* x, uint16_t* out,
                         const GlmQuantMatrix (&dense)[3], int tokens,
                         cudaStream_t stream);
  static GlmMoeWeights moe_weights(const GlmMoeResident& r);

  GlmTextConfig cfg_;
  KdaConfig kda_cfg_;
  DsaConfig dsa_cfg_;
  GlmMhcConfig mhc_cfg_;
  GlmMoeConfig moe_cfg_;
  KdaGeometry kda_geo_;
  int max_tokens_ = 0;

  GlmLayerStream loader_;
  GlmGlobalsResident globals_;
  CublasLtGemm gemm_;
  Arena arena_;
  DsaStatePool pool_;
  std::unique_ptr<KdaLayer> kda_;
  std::unique_ptr<DsaLayer> dsa_;
  std::unique_ptr<GlmMoeLayer> moe_;
  cudaStream_t stream_ = nullptr;

  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  uint8_t* dsa_scratch_ = nullptr;

  // Per-layer KDA state (contiguous across layers, zeroed per forward).
  float* kda_rec_ = nullptr;      // [num_kda_layers, local_heads, V, K]
  uint16_t* kda_conv_ = nullptr;  // [num_kda_layers, conv_channels, conv_hist]

  // Per-forward activations (managed; sized to max_tokens).
  int64_t* d_tokens_ = nullptr;
  uint16_t* streams_[2] = {nullptr, nullptr};  // [T, 4, hidden]
  uint16_t* post_ = nullptr;                   // [T, 4]
  uint16_t* comb_ = nullptr;                   // [T, 4, 4]
  uint16_t* collapsed_ = nullptr;              // [T, hidden] (also final mean)
  uint16_t* normed_ = nullptr;                 // [T, hidden] (also final out)
  uint16_t* sub_out_ = nullptr;                // [T, hidden]
  uint16_t* dense_g_ = nullptr;                // [T, dense_inter]
  uint16_t* dense_u_ = nullptr;
  uint16_t* dense_act_ = nullptr;
  uint16_t* logits_ = nullptr;                 // [T, vocab]
};

}  // namespace dgpp
