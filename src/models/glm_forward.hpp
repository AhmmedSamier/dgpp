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
#include "kernels/glm_spec.hpp"
#include "kernels/l2_prefetch.hpp"
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
    // fp32 [tokens, lm_vocab_count] — the rank's logits COLUMNS of the
    // full [tokens, vocab] matrix (Full head: the whole thing). The head
    // GEMV's fp32 accumulators, unrounded (2026-09-03): the bf16 rounding
    // the reference applies here was the dominant noise on every pick and
    // every log-prob comparison — half a bf16 ulp at |logit| ~16 is 0.06
    // nat — and it bought nothing; the sampler consumes floats anyway.
    std::vector<float> logits;
    int lm_vocab_begin = 0;  // first vocab column of logits
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
  // lm-head rows; Outputs.logits is then the rank's [tokens,
  // lm_vocab_count] slice (lm_vocab_begin/count report the bounds — the
  // sampling merge consumes exactly those). Default Full: byte-stable
  // with every M4/M5 parity gate.
  //
  // MTP (DESIGN §9): `mtp` = true materializes the draft layer
  // (cfg.mtp_layer(), ~7.3 GiB/rank resident) and its DSA cache ordinal,
  // keeps a per-position hidden cache for the draft block's hnorm input,
  // and enables session_draft. Requires cfg.num_nextn_predict_layers == 1.
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
                     int max_requests = 1, bool mtp = false);
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

  // ---- speculative decode (DESIGN §9) -----------------------------------
  // session_verify runs T (1..kSpecRows) tokens at the slot's next T
  // positions in ONE decode call and returns EVERY row's logits
  // ([T, lm_vocab_count]) and final hidden ([T, hidden]). Row r's bits
  // equal what session_step would have produced for that token after
  // rows < r (the head GEMV's rows are independent, the KDA recurrence is
  // sequential, DSA rows attend causally including themselves, MoE and
  // mHC are per token) — so a verifier compares row r's argmax with the
  // draft it fed as row r+1. The state advances by T in place; the state
  // after every row < T-1 is snapshotted on the way, so session_rollback
  // can then retract the rows the verifier rejected:
  //   session_rollback(req, a) with 1 <= a <= T restores the KDA
  //   recurrent/conv state and the DSA tail rings to what they were after
  //   row a-1 and rewinds the position by T-a. Latent rows and index pools
  //   at the retracted positions are simply overwritten by the next call.
  // a == T is a no-op. The snapshot scratch serves ONE in-flight verify:
  // roll a request back before verifying another. kSpecRows = 4 is the
  // bf16 GEMV's row bound (gemv::kMaxRows): at T <= 4 every projection
  // takes the row-independent GEMV, so the rows' bits are the T=1 bits;
  // T >= 5 would fall to cuBLASLt and break the equality above.
  static constexpr int kSpecRows = 4;
  Outputs session_verify(int req, const std::vector<int64_t>& token_ids);
  void session_rollback(int req, int accepted);

  // The MTP draft block (constructed with mtp = true). The block's row at
  // main-stack position q takes [enorm(embed(tok_{q+1})) | hnorm(h_q)] and
  // predicts tok_{q+2}; h_q is the pre-final-norm mean of the mHC streams
  // the main stack left at position q (kept in a per-position cache by
  // session_prefill/session_verify). session_prefill also runs the block
  // over the prompt (rows 0..P-2) so its DSA cache covers it; the block's
  // row counter then trails the main stack's position by one.
  //
  // session_draft(req, tokens) runs the block over the rows the main stack
  // has advanced past since the last draft: tokens[j] is tok_{q_j+1} for
  // consecutive q_j starting at the block's row counter, and tokens.size()
  // must equal session_position - draft rows so far (after a verify that
  // accepted `a` rows: the a accepted rows' argmaxes). Returns the LAST
  // row's logits (this rank's vocab slice): the draft for the next verify.
  // Draft rows only ever carry accepted tokens, so the block's state
  // never rolls back.
  bool mtp_enabled() const { return mtp_; }
  Outputs session_draft(int req, const std::vector<int64_t>& tokens);
  // The draft's graph-era halves (a graph per row count, as for verify).
  void session_graph_capture_draft(int req, const std::vector<int64_t>& tokens);
  void session_graph_stage_draft(int req, const std::vector<int64_t>& tokens);
  Outputs session_graph_collect_draft(int req);
  // Retires slot `req`: its DSA blocks return to the free pool (admission
  // meters see the capacity again) and the slot may be reopened by a
  // later prefill. No collective — safe between any two session ops.
  void session_close(int req);
  int64_t session_position(int req) const;

  // ---- the graph era (DESIGN §6.2, the decode step) --------------------
  // The decode step is a fixed launch sequence, so it records ONCE into
  // a CUDA graph and replays per token. The bus-side collective nodes
  // are recorded through a graph-recorder boundary reducer the CALLER
  // installs for the capture (see glm_tp_bus.hpp) — the model only needs
  // to know that a capture walk may not sync and must source its
  // per-token inputs from stable pinned members.
  //
  //   stream()                    — the walk's stream (the capture and
  //                                 replay stream; kernels, memcpy nodes,
  //                                 and launches all target it).
  //   set_boundary(b)              — swaps the boundary reducer; returns
  //                                 the previous one (the capture dance:
  //                                 install the recorder, capture,
  //                                 restore).
  //   session_graph_capture_step() — ONE decode step enqueued in
  //                                 CAPTURE MODE between the caller's
  //                                 cudaStreamBeginCapture/EndCapture on
  //                                 stream(): every launch records, the
  //                                 per-token H2D uploads (metadata,
  //                                 token id) record as memcpy nodes
  //                                 off stable pinned members, and the
  //                                 eager path's synchronizations (the
  //                                 two fold drains and the final sync)
  //                                 are SKIPPED — a captured sync is an
  //                                 error. NOTHING EXECUTES: state,
  //                                 positions, and pinned staging are
  //                                 untouched; the first replay performs
  //                                 this step for real. The boundary
  //                                 reducer must be a recorder or the
  //                                 capture folds eagerly (and fails).
  //   session_graph_stage()       — the REPLAY path's host half: DSA
  //                                 admission plus the stable pinned
  //                                 member writes (request id, position,
  //                                 spans, token id). NO uploads — the
  //                                 graph's memcpy nodes re-read the
  //                                 pinned members at launch.
  //   session_graph_collect()     — the REPLAY path's result half, after
  //                                 the caller's launch + stream sync +
  //                                 graph_replay_finish: materializes the
  //                                 Outputs EXACTLY as the eager decode
  //                                 tail does (logits/final_hidden from
  //                                 the stable device buffers, route
  //                                 traces from the pinned staging the
  //                                 graph's D2H nodes filled) and
  //                                 advances the slot's position.
  //
  // The replay outputs are bitwise the eager session_step's (the
  // recorded kernels ARE the eager kernels; the bus gate pins its
  // eager-vs-graph fold equality).
  cudaStream_t stream() const { return stream_; }
  // The on-device pick's inputs (glm_tp_bus.hpp GlmDevicePicker): the head's
  // fp32 logits [rows, lm_vocab_count] and the decode call's token ids
  // [rows] as the last run/capture left them on the device. A verify's
  // rows are its fed tokens in order; a draft's head lands in row 0.
  const float* device_logits() const { return logits_; }
  const int64_t* device_tokens() const { return d_tokens_; }
  int lm_vocab_begin() const { return lm_vocab_begin_; }
  int lm_vocab_count() const { return lm_vocab_count_; }
  GlmBoundaryReducer* set_boundary(GlmBoundaryReducer* boundary) {
    GlmBoundaryReducer* prev = boundary_;
    boundary_ = boundary;
    return prev;
  }
  //   session_graph_prepare()     — BEFORE cudaStreamBeginCapture: uploads
  //                                 each MoE layer's expert-view table to
  //                                 its graph slot (a sync follows), so
  //                                 the recorded kernels read fixed device
  //                                 tables and the replay moves no table
  //                                 bytes. Resident stacks only.
  void session_graph_prepare();
  void session_graph_capture_step(int req, int64_t token_id);
  void session_graph_stage(int req, int64_t token_id);
  Outputs session_graph_collect(int req);
  // The T-row (speculative verify) graph: a separate graph per row count
  // — T is baked into every grid, GEMV m, and memcpy size. Stage the same
  // T you captured; collect materializes all T rows (session_verify's
  // Outputs shape) and advances the position by T, session_rollback then
  // retracts as in the eager path.
  void session_graph_capture_step(int req, const std::vector<int64_t>& ids);
  void session_graph_stage(int req, const std::vector<int64_t>& ids);

  // ---- the DEVICE-DRIVEN graph (DESIGN §9, the on-device step) ---------
  // The speculative step's control flow moves onto the device so a replay
  // carries it: the rows' positions come from the device-side session
  // position (glm_spec_positions off d_session_pos_, not the host's staged
  // h_step_pos_ upload), and a recorded COMMIT behind the caller's device
  // pick (kernels/glm_spec.hpp glm_spec_commit) reads the verdict on the
  // device, copies the post-row-(accepted-1) snapshots over the live KDA
  // recurrent/conv state and the DSA tail rings when accepted < T (the
  // bytes session_rollback copies, without the host, the sync, or the
  // unconditional-memcpy problem), and advances the device position by
  // `accepted`. The host keeps a MIRROR of the position (session_position)
  // for validation and the draft's invariant: every op that moves the
  // position on the host pushes it to the device (push_position), and
  // after a device-driven replay the host advances its mirror from the
  // pinned verdict with session_graph_settle — the device already moved.
  //
  //   session_reserve_blocks(req, tokens) — DSA admission for the whole
  //         run, BEFORE the capture (the per-step admission the staged path
  //         does in session_graph_stage would be a device copy inside a
  //         replay); throws when the pool cannot cover `tokens`.
  //   session_graph_capture_step(req, ids, device_positions = true) — the
  //         capture with the positions kernel in place of the position
  //         upload and no per-step admission.
  //   session_graph_capture_commit(req, device_verdict) — the commit,
  //         recorded right after the caller's recorded pick; decode_rows_
  //         (the captured T) bounds it.
  //   session_graph_stage(req, ids) — in this mode only validates and
  //         writes the token upload source (the graph reads positions from
  //         the device).
  //   session_graph_settle(req, accepted) — after the replay's sync and
  //         the bus finish: the host mirror advances by `accepted`. No
  //         collect, no rollback — the device did both.
  //   set_decode_tail_mirrors(false) — drops the tail's logits/hidden D2H
  //         nodes (a device-pick consumer reads neither); default on.
  //   session_graph_capture_draft(req, verify_verdict) — the draft block
  //         IN the graph (phase C), behind the commit: glm_spec_draft_rows
  //         turns the verify's verdict into the block's T rows (accepted
  //         rows real at the block's device position, the rest padding at
  //         position -1 — the DSA path skips them, nothing is written),
  //         the block runs its fixed T rows, its head runs on EVERY row,
  //         and the caller's second recorded pick reads the last accepted
  //         one (GlmDevicePicker::Inputs::row_select). The verify's `next`
  //         is parked on the device for the token feed.
  //   session_graph_capture_next_tokens(req, draft_verdict) — the graph's
  //         last node (phase D): d_tokens_ = [next, the draft's pick] for
  //         the next replay. Requires device_tokens at the capture (no
  //         token upload node) and session_graph_seed_tokens once before
  //         the first replay.
  //   session_graph_settle(req, accepted) — with the draft in the graph the
  //         block's row counter mirror advances by `accepted` too.
  void session_reserve_blocks(int req, int64_t tokens);
  void session_graph_capture_step(int req, const std::vector<int64_t>& ids,
                                  bool device_positions,
                                  bool device_tokens = false);
  void session_graph_capture_commit(int req,
                                    const GlmPickVerdict* device_verdict);
  void session_graph_capture_draft(int req,
                                   const GlmPickVerdict* verify_verdict);
  void session_graph_capture_next_tokens(int req,
                                         const GlmPickVerdict* draft_verdict);
  void session_graph_seed_tokens(int req, const std::vector<int64_t>& ids);
  void session_graph_settle(int req, int accepted);
  void set_decode_tail_mirrors(bool on) { decode_tail_mirrors_ = on; }
  // Materializes the replay's Outputs WITHOUT moving the position (the
  // device-driven flow's cross-check surface; needs the tail mirrors on).
  Outputs session_graph_outputs(int req);

  // Decode-step route traces (Outputs.routes / route_biased): the per-MoE-
  // layer ids, weights and biased scores the parity gates and the near-tie
  // audit read. They cost three D2H nodes per MoE layer per step (126 per
  // token here), which a serving loop that only wants logits should not
  // pay. Default on; the serving apps turn them off. Takes effect at the
  // next capture / eager step.
  void set_decode_route_traces(bool on) { decode_route_traces_ = on; }
  bool decode_route_traces() const { return decode_route_traces_; }

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
  // The last replayed decode step's first-node %globaltimer (see
  // launch_globaltimer_stamp); 0 before any decode step.
  uint64_t graph_start_globaltimer() const { return *h_graph_start_gt_; }

  // Boot digest over the replicated weights (d4, §5.2 "boot checks hash
  // all replicated tensors"): computed at construction when tp_world > 1,
  // BEFORE the first forward — runners exchange it across ranks and a
  // mismatch pinpoints the layer. Empty at world=1 (no peers to convince).
  const GlmReplicatedDigest& boot_digest() const { return boot_digest_; }
  // Startup phase durations (ms), for the record and the load-time work.
  double boot_digest_ms() const { return boot_digest_ms_; }
  double boot_globals_ms() const { return boot_globals_ms_; }

  // Checkpoint source bytes this rank's loads have touched (the
  // per-rank byte-total reconcile input; see GlmLayerStream).
  uint64_t source_bytes_read() const { return loader_.source_bytes_read(); }

  // Host-side top-k over fp32 logits: value-descending, lowest-id
  // tie-break. k <= 64. One entry per row. FULL-vocab logits only — a
  // sharded-head consumer must merge its slice with glm_sample's
  // helpers instead (this helper cannot see the other ranks' slices).
  static std::vector<std::vector<std::pair<int32_t, float>>> topk(
      const std::vector<float>& logits, int64_t rows, int vocab, int k);

 private:
  // Produces the layer's weight views — full (world=1, byte-identical to
  // the M4 path) or this rank's slice (GlmTpViews). `dense_mlp` mirrors
  // cfg_.mlps[layer]; exactly one attention and one MLP view is set.
  GlmLayerBound bind_layer(const GlmLayerResident& r, bool dense_mlp);

  // The block-boundary prefetch windows (decode rows): while the bus folds
  // one side of a layer, pull the OTHER side's first weights into L2.
  void prefetch_ffn_side(const GlmLayerBound& b, bool dense_mlp);
  void prefetch_attention_side(int layer);
  void prefetch_head();

  // The stack's ONLY layer-load path: load_layer plus the resident-mode
  // hand-off — the stack walks layers in order, so the last main layer's
  // materialization means every layer this model will ever read is on the
  // device, and the loader can drop its checkpoint mappings + page cache
  // right there (GlmLayerStream::release_sources: the memory-watermark
  // fix). Streaming mode and repeat passes are no-ops.
  const GlmLayerResident& stack_layer(int layer);

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
  //
  // capture_mode (the graph era): the launches record into the caller's
  // CUDA-graph capture; the fold drains, final sync, and output
  // materialization are skipped (a captured sync is an error) and the
  // token-id upload reads the stable pinned member instead of the
  // caller's vector (a memcpy node bakes its source address). NOTHING
  // executes — state and positions are untouched. The caller wraps the
  // call in BeginCapture/EndCapture and installs a recorder reducer.
  Outputs session_run_rows(int req, const std::vector<int64_t>& ids,
                           int64_t token_start, bool decode_row,
                           bool capture_mode = false);
  // The decode step's host-side half, shared by the eager step, the
  // capture step, and the replay stage: validation, DSA admission, the
  // pinned member writes (request id, position, spans, token id), and
  // (when `upload`) the metadata H2Ds — eager issues them, capture
  // records them as memcpy nodes, the replay stage skips them (its
  // graph re-uploads at launch).
  void session_decode_host_prep(int req, const std::vector<int64_t>& ids,
                                bool upload, bool device_positions = false);
  // Pushes the host position mirror of slot `req` to the device (the
  // device-driven graph's positions source); async on stream_.
  void push_position(int req);
  // The rollback segment table for slot `req` (glm_spec_commit's input):
  // the live KDA state slices and DSA tail rings with their snapshots.
  GlmSpecSegments spec_segments(int req);
  // ---- MTP (glm_mtp.cpp) ----
  // The draft block over T rows at the block's positions [first_pos,
  // first_pos + T): tokens from d_tokens_, hidden from the position
  // cache. decode_row selects the DSA decode path (positions from
  // d_step_pos_) and runs the head on the LAST row into logits_ row 0 +
  // the pinned mirror; prefill rows run the block only (no head).
  // head_rows: 1 = the last row only (the eager draft), T = every row (the
  // in-graph draft; the pick selects the last accepted row).
  void mtp_run_rows(int req, int64_t first_pos, int T, bool decode_row,
                    bool capture_mode, int head_rows = 1);
  void push_mtp_position(int req);
  // The block over a prompt's rows 0..P-2 in pool-aligned chunks.
  void mtp_prefill(int req, const std::vector<int64_t>& prompt_ids);
  // The draft's host half: validation, positions/tokens staging, uploads.
  void mtp_decode_host_prep(int req, const std::vector<int64_t>& tokens,
                            bool upload);
  Outputs mtp_decode_tail();
  uint16_t* mtp_hidden_cache(int req) const;
  int moe_graph_slots() const { return n_moe_layers_ + (mtp_ ? 1 : 0); }
  // The decode tail shared by the eager step and the graph-era collect:
  // materializes the route traces from the pinned per-MoE-layer staging
  // (the D2H copies must have joined — the eager final sync or the
  // replay's stream sync) and assigns EVERY row's logits/final_hidden
  // ([T, ...]) from the pinned mirrors the step's D2H copies filled. The route shape (one entry per MoE
  // layer, actual layer indices, [tokens, K]/[tokens, E] values) is the
  // eager path's exactly.
  Outputs session_decode_tail(int T);
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
  double boot_digest_ms_ = 0;
  double boot_globals_ms_ = 0;
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
  // L2 weight prefetch for decode rows (kernels/l2_prefetch.hpp): fed at
  // the block boundaries and by the attention layers; joined before the
  // step's tail so a capture closes cleanly.
  WeightPrefetcher prefetch_;

  // TP state: null at world=1 (the M4 path).
  GlmBoundaryReducer* boundary_ = nullptr;
  std::unique_ptr<GlmTpViews> tp_;

  bool decode_route_traces_ = true;

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
  int64_t* d_session_pos_ = nullptr;  // device [max_requests] — the
                                      // device-driven graph's position
  int64_t* h_session_pos_ = nullptr;  // pinned upload mirror
  bool graph_device_positions_ = false;  // the captured graph's mode
  bool graph_device_tokens_ = false;     // no token upload node
  bool graph_has_draft_ = false;         // the draft block is in the graph
  bool decode_tail_mirrors_ = true;
  int64_t* d_next_ = nullptr;  // device [max_requests]: the verify's next
                               // token, parked for the token feed
  int32_t* d_req_ids_ = nullptr;      // device [kDecodeRows]
  int64_t* d_step_pos_ = nullptr;     // device [kDecodeRows]
  int32_t* d_req_spans_ = nullptr;    // device [kDecodeRows, 2]
  // The H2D upload sources, PINNED (pageable async copies stream-sync
  // before initiating — a per-step pipeline drain; and CUDA-graph
  // memcpy nodes REQUIRE page-locked host sources). Stable addresses
  // for the graph's lifetime: the replay's memcpy nodes re-read these
  // after session_graph_stage refreshes their contents.
  int32_t* h_req_ids_ = nullptr;      // pinned [kDecodeRows]
  int64_t* h_step_pos_ = nullptr;     // pinned [kDecodeRows]
  int32_t* h_req_spans_ = nullptr;    // pinned [kDecodeRows, 2]
  int64_t* h_token_ = nullptr;       // pinned [kDecodeRows] — the step's
                                     // token id (the decode walk's H2D
                                     // source; a memcpy node's baked
                                     // address)
  static constexpr int kDecodeRows = 8;  // DsaLayer's select-kernel bound
  // DESIGN §7.1: pool-aligned prefill chunks. Must be a multiple of DSA's
  // kpool so continuation chunks stay pool-aligned; the FINAL chunk may end
  // mid-pool (the tail persists into decode).
  static constexpr int kPrefillChunkTokens = 2048;
  // The row count of the most recent decode call (staged or run): what
  // session_graph_collect materializes and session_rollback bounds.
  int decode_rows_ = 1;
  // Speculative post-row state snapshots (kernels/kda.hpp
  // KdaStateSnapshots, dsa.hpp tail_snapshots), one in-flight verify:
  //   spec_rec_  [kSpecRows-1][num_kda_layers][rec elems]  fp32
  //   spec_conv_ [kSpecRows-1][num_kda_layers][conv elems] bf16
  //   spec_tail_ [num_dsa_layers][kSpecRows][2, kpool, dim] bf16
  // Row-major over the layers so rolling every KDA layer back is ONE
  // memcpy from row a-1 onto the request's contiguous layer slots.
  float* spec_rec_ = nullptr;
  uint16_t* spec_conv_ = nullptr;
  uint16_t* spec_tail_ = nullptr;
  size_t spec_tail_ring_elems() const;
  // MTP draft layer (DESIGN §9). The draft's DSA cache is one more pool
  // ordinal (main_dsa_layers_) after the main stack's; its MoE table slot
  // is n_moe_layers_. mtp_hidden_ is the per-request, per-position cache
  // of the main stack's pre-final-norm hidden ([max_requests, max_tokens,
  // hidden] bf16) the draft block's hnorm reads; mtp_pos_ counts the
  // block's rows per request.
  bool mtp_ = false;
  int main_dsa_layers_ = 0;  // dsa_cfg_.num_dsa_layers minus the draft's
  std::vector<int64_t> mtp_pos_;
  int64_t* d_mtp_pos_ = nullptr;  // device [max_requests]: the block's row
                                  // counter for the in-graph draft
  int64_t* h_mtp_pos_ = nullptr;  // pinned upload mirror
  int draft_rows_ = 1;
  uint16_t* mtp_hidden_ = nullptr;  // [max_requests, max_tokens, H]
  uint16_t* mtp_cat_ = nullptr;     // [T, 2H] the eh_proj input
  uint16_t* mtp_x_ = nullptr;       // [T, H] the block's residual
  // Decode-path route traces (2026-09-01): per-MoE-layer pinned staging
  // filled by enqueue_decode's async D2H copies, materialized into
  // Outputs.routes after the step's final sync (the copies are
  // stream-ordered; the sync joins them). Allocated at construction.
  int n_moe_layers_ = 0;
  int32_t* moe_trace_ids_ = nullptr;    // [n_moe_layers_ * kDecodeRows * K]
  float* moe_trace_weights_ = nullptr;  // same shape
  float* moe_trace_biased_ = nullptr;   // [n_moe_layers_ * kDecodeRows * E]
  // The decode tail's PINNED mirrors of the last-row logits and final
  // hidden (2026-09-02). The host used to read those rows straight out of
  // the managed logits_/normed_ after the step's sync — a CPU access to a
  // page the GPU wrote, then a GPU write to a page the CPU touched, every
  // step: UVM fault servicing, and ~2% of steps paid it as a 9-10 ms
  // stall (a cudaLaunchKernel or the collect itself blocked behind the
  // fault) that every other rank then waited out at the pick. The rows now
  // ride D2H copies (graph memcpy nodes in the replay) into these, and the
  // managed pages stay GPU-resident.
  uint64_t* h_graph_start_gt_ = nullptr;  // pinned: %globaltimer at the step's first node
  float* h_tail_logits_ = nullptr;      // pinned [kDecodeRows * lm_vocab_count_]
  uint16_t* h_tail_hidden_ = nullptr;   // pinned [kDecodeRows * H]
  uint16_t* streams_[2] = {nullptr, nullptr};  // [T, 4, hidden]
  uint16_t* post_ = nullptr;                   // [T, 4]
  float* mhc_logits_ = nullptr;                // [T, 24] mHC dots scratch
  int* mhc_counters_ = nullptr;                // [T] fused-finish tickets
  uint16_t* comb_ = nullptr;                   // [T, 4, 4]
  uint16_t* collapsed_ = nullptr;              // [T, hidden] (also final mean)
  uint16_t* normed_ = nullptr;                 // [T, hidden] (also final out)
  uint16_t* sub_out_ = nullptr;                // [T, hidden]
  uint16_t* dense_g_ = nullptr;                // [T, dense_inter]
  uint16_t* dense_u_ = nullptr;
  uint16_t* dense_act_ = nullptr;
  float* logits_ = nullptr;                    // [T, vocab] fp32
};

}  // namespace dgpp
