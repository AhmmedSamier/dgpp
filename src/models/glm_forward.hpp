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
#include "models/glm_tp.hpp"
#include "models/kda_layer.hpp"

namespace dgpp {

// M5 tensor-parallel block-boundary seam (DESIGN §5.1): folds a partial
// hidden activation [rows, hidden] bf16 into its replicated value, in
// place. Called after the attention output projection and after the
// FFN/MoE — the two row-parallel sites whose local sums are partial —
// always with the producing kernels already quiesced on the model stream.
// world=1 constructs the model without a reducer and the seam is skipped.
struct GlmBoundaryReducer {
  virtual ~GlmBoundaryReducer() = default;

  // Optional staging seam (the §6.3 evolution): hands the producing GEMM
  // a pinned, device-writable destination for the boundary partial so the
  // collective sends it straight from there (no device→slot staging copy).
  // Returns nullptr when the shape does not fit (rows*hidden above the
  // latency slot — prefill-sized boundaries stay on the device path) or
  // when the transport has no pre-stage support. The returned pointer is
  // consumed by the following reduce() call.
  virtual uint16_t* stage(int /*rows*/, int /*hidden*/) { return nullptr; }

  virtual void reduce(uint16_t* partial, int rows, int hidden) = 0;
};

class GlmDiagnosticModel {
 public:
  struct Outputs {
    std::vector<uint16_t> final_hidden_bits;  // bf16 [tokens, hidden]
    std::vector<uint16_t> logits_bits;        // bf16 [tokens, vocab]
    // One entry per MoE layer, in layer order (ids ascending per token).
    std::vector<GlmRouteTraceLayer> routes;
    // Aligned with routes (same order): each MoE layer's full biased router
    // score row [tokens, n_experts] fp32 — the near-tie certification
    // inputs (both sides' selections come from these scores).
    std::vector<std::vector<float>> route_biased;
  };

  // `max_tokens` bounds a forward's token count; `max_cache_tokens` bounds
  // the DSA cache (rounded up to a block). Both also size scratch.
  //
  // TP (M5): `tp_world` > 1 slices every layer's head/expert/inter shards
  // to this rank (`tp_rank`), runs the forward on the local geometry, and
  // folds the two block-boundary partials through `boundary` (required —
  // a TP model without a reducer would silently return partial sums).
  // The world=1 path (null reducer) is byte-identical to M4.
  //
  // RESIDENCY: `GlmResidency::Resident` materializes every layer once
  // (the production residency contract — storage is never touched during
  // inference after the first forward). Only worlds whose
  // GlmLayerStream::resident_bytes() fits can use it (real dims: world 4).
  GlmDiagnosticModel(const GlmTextConfig& cfg,
                     const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens,
                     GlmBoundaryReducer* boundary = nullptr, int tp_rank = 0,
                     int tp_world = 1,
                     GlmResidency residency = GlmResidency::Streaming);
  ~GlmDiagnosticModel();
  GlmDiagnosticModel(const GlmDiagnosticModel&) = delete;
  GlmDiagnosticModel& operator=(const GlmDiagnosticModel&) = delete;

  // Runs layers 0..num_hidden_layers-1 for `token_ids` (single request,
  // fresh KDA/DSA state every call — no cross-call state survives). Returns
  // host copies of the final hidden state, logits, and routing decisions.
  Outputs forward(const std::vector<int64_t>& token_ids);

  // Isolated parity runner (the curated suite's real-checkpoint mode):
  // every layer starts from the REFERENCE trajectory — layer_inputs[L]
  // replaces the stream state entering layer L (index 0 replaces the
  // embedding broadcast) — so module noise never compounds across layers.
  // capture receives num_layers+1 stream snapshots (initial + per-layer
  // outputs) as host copies. The head runs on the isolated final streams.
  // Routed-layer decisions are likewise isolated (comparable per layer).
  //
  // boundary_capture (optional) receives the two BLOCK-BOUNDARY FOLDS per
  // layer — the post-reduce attention output and FFN output, [tokens,
  // hidden] each, 2*num_layers vectors in layer order. At world=1 these
  // are the same buffers unfolded (the direct comparison targets). The
  // stream snapshots above pass through the mHC stream update, whose
  // mixing coefficients ATTENUATE boundary errors ~300x on the fixture —
  // the raw folds are where slicing bugs actually surface (the dense
  // scale-grid slice bug hid exactly there, 0.30 l2-wrong under a 0.001
  // stream-state reading).
  Outputs forward_isolated(
      const std::vector<int64_t>& token_ids,
      const std::vector<const uint16_t*>& layer_inputs,
      std::vector<std::vector<uint16_t>>& capture,
      std::vector<std::vector<uint16_t>>* boundary_capture = nullptr);

  const GlmTextConfig& config() const { return cfg_; }
  int max_tokens() const { return max_tokens_; }

  // Boot digest over the replicated weights (d4, §5.2 "boot checks hash
  // all replicated tensors"): computed at construction when tp_world > 1,
  // BEFORE the first forward — runners exchange it across ranks and a
  // mismatch pinpoints the layer. Empty at world=1 (no peers to convince).
  const GlmReplicatedDigest& boot_digest() const { return boot_digest_; }

  // Checkpoint source bytes this rank's loads have touched (the
  // per-rank byte-total reconcile input; see GlmLayerStream).
  uint64_t source_bytes_read() const { return loader_.source_bytes_read(); }

  // Host-side top-k over bf16 logits: value-descending, lowest-id
  // tie-break. k <= 64. One entry per row.
  static std::vector<std::vector<std::pair<int32_t, float>>> topk(
      const std::vector<uint16_t>& logits_bits, int64_t rows, int vocab,
      int k);

 private:
  // Produces the layer's weight views — full (world=1, byte-identical to
  // the M4 path) or this rank's slice (GlmTpViews). `dense_mlp` mirrors
  // cfg_.mlps[layer]; exactly one attention and one MLP view is set.
  GlmLayerBound bind_layer(const GlmLayerResident& r, bool dense_mlp);

  // Constructs every layer object with throwaway layer loads so run_stack
  // never allocates (allocations are implicit device syncs; a lazily
  // constructing rank deadlocks against a peer's spinning first
  // collective kernel — see the M5 loopback bring-up notes).
  void preconstruct_layers();

  void enqueue_dense_mlp(const uint16_t* x, uint16_t* out,
                          const GlmQuantMatrix* dense, int tokens,
                          cudaStream_t stream);
  static GlmMoeWeights moe_weights(const GlmMoeResident& r);
  Outputs run_stack(const std::vector<int64_t>& token_ids,
                    const uint16_t* const* layer_inputs,
                    std::vector<std::vector<uint16_t>>* capture,
                    std::vector<std::vector<uint16_t>>* boundary_capture);

  GlmTextConfig cfg_;
  KdaConfig kda_cfg_;
  DsaConfig dsa_cfg_;
  GlmMhcConfig mhc_cfg_;
  GlmMoeConfig moe_cfg_;
  KdaGeometry kda_geo_;
  int max_tokens_ = 0;
  GlmReplicatedDigest boot_digest_{};

  // Sharded at world>1 (M5 d4): the loader builds each resident layer
  // directly at this rank's geometry, and bind_layer is the identity
  // wiring (GlmTpViews::bind_sharded) — pinned bitwise against
  // full-load+bind by glm_tp_test's shard-parity test.
  GlmLayerStream loader_;
  GlmGlobalsResident globals_;
  CublasLtGemm gemm_;
  Arena arena_;
  DsaStatePool pool_;
  std::unique_ptr<KdaLayer> kda_;
  std::unique_ptr<DsaLayer> dsa_;
  std::unique_ptr<GlmMoeLayer> moe_;
  cudaStream_t stream_ = nullptr;

  // TP state: null at world=1 (the M4 path).
  GlmBoundaryReducer* boundary_ = nullptr;
  std::unique_ptr<GlmTpViews> tp_;

  // World=1 bind scratch (owned so the returned views outlive the call).
  GlmLayerBound full_{};
  GlmMoeWeights moe_full_{};

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
