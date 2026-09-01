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
    // bf16 [tokens, lm_vocab_count] — the rank's logits COLUMNS of the
    // full [tokens, vocab] matrix (Full head: the whole thing).
    std::vector<uint16_t> logits_bits;
    int lm_vocab_begin = 0;  // first vocab column of logits_bits
    int lm_vocab_count = 0;
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
  //
  // HEAD (M6 d3): `GlmHeadSharding::VocabSharded` loads only this rank's
  // lm-head rows; Outputs.logits_bits is then the rank's [tokens,
  // lm_vocab_count] slice (lm_vocab_begin/count report the bounds — the
  // sampling merge consumes exactly those). Default Full: byte-stable
  // with every M4/M5 parity gate.
  //
  // SESSIONS (M6 Stage 2b): `max_requests` sizes the concurrent-session
  // machinery — the DSA pool's per-request block tables/tail rings and the
  // per-request KDA state slots. Default 1 is byte-stable with every
  // Stage 2 parity gate. The shared DSA cache capacity (`max_cache_tokens`)
  // is the ADMISSION BUDGET across all concurrent requests, not a
  // per-request bound: the scheduler admits a request only when its full
  // reservation (prompt + max_steps, block-rounded) fits the free pool.
  GlmDiagnosticModel(const GlmTextConfig& cfg,
                     const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens,
                     GlmBoundaryReducer* boundary = nullptr, int tp_rank = 0,
                     int tp_world = 1,
                     GlmResidency residency = GlmResidency::Streaming,
                     GlmHeadSharding head = GlmHeadSharding::Full,
                     int max_requests = 1);
  ~GlmDiagnosticModel();
  GlmDiagnosticModel(const GlmDiagnosticModel&) = delete;
  GlmDiagnosticModel& operator=(const GlmDiagnosticModel&) = delete;

  // Runs layers 0..num_hidden_layers-1 for `token_ids` (single request,
  // fresh KDA/DSA state every call — no cross-call state survives). Returns
  // host copies of the final hidden state, logits, and routing decisions.
  Outputs forward(const std::vector<int64_t>& token_ids);

  // ---- stateful decode sessions (M6 Stage 2/2b, DESIGN §7/§9) ----------
  // One request's incremental decode. session_prefill OPENS request slot
  // `req` (zeroed KDA recurrent/conv state for that slot, released DSA
  // blocks, cold tail rings — the same starting state run_stack builds)
  // and processes the prompt in pool-aligned chunks (2048), keeping state
  // across chunks. session_step then processes exactly ONE token at the
  // slot's next position, updating the persistent state in place, and
  // returns that token's outputs. No cross-token scratch is re-derived:
  // the layer pipeline is elementwise in time (all temporal recurrence
  // lives in the KDA/DSA state), so a step equals the corresponding row
  // of a full re-forward up to (a) GEMM-batch ulps across different M,
  // and (b) the prefill-vs-decode kernel paths — the decode-session
  // parity gate CERTIFIES near ties, never assumes them away. A
  // single-chunk prefill (prompt <= 2048) runs the exact run_stack op
  // sequence and must match the re-forward bitwise — the gate pins that
  // tier too.
  //
  // CONCURRENT REQUESTS (Stage 2b): slots are independent — prefilling or
  // stepping one never touches another's state, so a request's transcript
  // is invariant to whatever else the scheduler interleaves (the property
  // the 2b gates pin and the fabric smoke proves). Steps are
  // TIME-MULTIPLEXED (one request per call): batching multiple requests'
  // rows into one step needs per-row state indexing in the KDA recurrence
  // (the DESIGN §9 MTP state-index surgery) and stays a later
  // optimization — the scheduler's policy is unchanged by it.
  //
  // Slot assignment and op ORDER are the caller's contract (the
  // scheduler): every rank must issue the same ops on the same slots in
  // the same order — session ops embed boundary folds and the consumer
  // picks over the sharded head, so identical rank order (DESIGN §11) is
  // what keeps the collectives aligned.
  //
  // HAZARD: forward()/forward_isolated() and the sessions share the
  // KDA/DSA state pools. A plain forward on the SAME model instance while
  // any session is open throws (it would clobber session state) — a
  // parity harness runs engine and reference on SEPARATE instances.
  Outputs session_prefill(int req, const std::vector<int64_t>& prompt_ids);
  Outputs session_step(int req, int64_t token_id);
  // Retires slot `req`: its DSA blocks return to the free pool (admission
  // meters see the capacity again) and the slot may be reopened by a
  // later prefill. No collective — safe between any two session ops.
  void session_close(int req);
  int64_t session_position(int req) const;

  // v1 single-request shims (slot 0) — the Stage 2 parity gates' shape.
  Outputs session_prefill(const std::vector<int64_t>& prompt_ids) {
    return session_prefill(0, prompt_ids);
  }
  Outputs session_step(int64_t token_id) { return session_step(0, token_id); }
  int64_t session_position() const { return session_position(0); }

  // ---- DSA admission meters (the scheduler's budget seam, Stage 2b) ----
  // A model with no DSA layers has no pool; its meters report an
  // unbounded budget so admission keys on the slot count alone.
  int max_session_requests() const { return max_requests_; }
  int64_t dsa_blocks_total() const;
  int64_t dsa_blocks_in_use() const;
  // Block count covering `tokens` tokens — the reserve arithmetic for
  // prompt + max_steps admissions.
  int64_t dsa_blocks_for_tokens(int64_t tokens) const;

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
  // tie-break. k <= 64. One entry per row. FULL-vocab logits only — a
  // sharded-head consumer must merge its slice with glm_sample's
  // helpers instead (this helper cannot see the other ranks' slices).
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
  // The session's row runner: processes `tokens` contiguous rows of OPEN
  // request slot `req` starting at absolute position `token_start`,
  // updating that slot's persistent KDA/DSA state in place (NO reset —
  // prefill resets once at open). `decode_row` selects the DSA path:
  // false = enqueue_prefill (pool-aligned chunks, tables grown
  // internally), true = enqueue_decode (arbitrary positions via the
  // caller-grown table + device metadata). Returns the LAST row's
  // logits/final_hidden; routes/route_biased cover every processed row
  // (audit inputs are per-token).
  Outputs session_run_rows(int req, const std::vector<int64_t>& ids,
                           int64_t token_start, bool decode_row);
  // Extends the route/route_biased outputs with one runner pass's entries,
  // merging same-layer chunks along the token axis (prefill chunks emit
  // per-chunk entries; the reference emits one per layer).
  void session_merge_routes(Outputs* out, Outputs&& chunk) const;
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
  // The logits slice this model computes (Full: [0, vocab)).
  int lm_vocab_begin_ = 0;
  int lm_vocab_count_ = 0;

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

  // Per-request, per-layer KDA state, slot-major:
  //   kda_rec_  [max_requests, num_kda_layers, local_heads, V, K]  fp32
  //   kda_conv_ [max_requests, num_kda_layers, conv_channels, conv_hist] bf16
  // Slot-major so opening a request is ONE memset pair (all its layers
  // contiguous); forward() (the fresh-state re-forward) uses slot 0.
  float* kda_rec_ = nullptr;
  uint16_t* kda_conv_ = nullptr;

  // Per-forward activations (managed; sized to max_tokens).
  int64_t* d_tokens_ = nullptr;
  // Decode-session state: per-slot positions (tokens processed; 0 = slot
  // closed) and the DSA decode-path metadata (enqueue_decode's
  // caller-owned device buffers — allocation happens at construction,
  // never mid-session; the synchronizing-call discipline applies between
  // collectives). One ROW per step call (time-multiplexed requests);
  // kDecodeRows=8 is DsaLayer's select-kernel bound and the batched-decode
  // ceiling the Stage 2b scheduler inherits.
  int max_requests_ = 1;
  std::vector<int64_t> session_pos_;  // [max_requests]; 0 = closed slot
  int32_t* d_req_ids_ = nullptr;      // device [kDecodeRows]
  int64_t* d_step_pos_ = nullptr;     // device [kDecodeRows]
  int32_t* d_req_spans_ = nullptr;    // device [kDecodeRows, 2]
  int32_t h_req_ids_[8] = {0};
  int64_t h_step_pos_[8] = {0};
  int32_t h_req_spans_[16] = {0};
  static constexpr int kDecodeRows = 8;  // DsaLayer's select-kernel bound
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
