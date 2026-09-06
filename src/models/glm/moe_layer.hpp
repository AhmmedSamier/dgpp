#pragma once
// Host-orchestrated MoE layer forward (M4 deliverable 1, diagnostic mode):
// router kernel -> host segmentation by expert -> per-expert gather +
// scale-aware GEMMs + swiglu + accumulation (ascending expert order) ->
// shared expert. One stream sync per enqueue (the router's ids/weights come
// back to the host for segmentation); correctness-first — the production
// grouped-expert kernel without host round-trips is M5+ work and is
// explicitly out of scope here.
//
// THE DECODE FAST PATH (2026-09-01, enqueue_decode): the production
// grouped-expert path for the decode shape — the router leaves its
// decision on the DEVICE (ids ascending per row, its kernel contract),
// the slot kernels read the route and the expert weight views from
// device memory, and one ordered accumulation reproduces the host path's
// exact chain (bitwise — glm_moe_test pins it). NO stream sync, NO host
// segmentation, NO per-segment H2D round trips; route traces ride async
// copies into caller-pinned staging (MoeTraceStaging) and materialize
// after the step's final sync. The prefill path keeps enqueue(): its sync
// amortizes over 2048-token chunks.
//
// NUMERICS (2026-09-02, expert slicing): every rank holds a slice of every
// expert's intermediate dim (see GlmMoeWeights), so an expert's down
// projection is a PARTIAL sum on each rank. The per-rank chain runs in
// fp32 — unrounded partial dots, fma per expert in ascending id order, the
// shared expert last — and rounds to bf16 exactly once, as the sum leaves
// for the FFN all-reduce (bf16 on the wire, folded in rank order there).
// The reference's per-expert bf16 roundings are deliberately not
// reproduced: fewer roundings, and a slice cannot reproduce a whole
// expert's rounding anyway. The oracle (glm_moe_reference) carries the
// same chain in double.
//
// Route-trace capture (M4 deliverable 5): the most recent routing decision
// is retained on the host (last_ids/last_weights) so callers can record it
// per layer into a trace file (models/glm_trace.hpp) for the corrected
// per-rank traffic model.
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

// Caller-owned PINNED staging for enqueue_decode's async route copies:
// the decode step's D2H copies land here mid-step (stream-ordered, no
// sync) and the caller reads them after its next stream sync — the
// session materializes Outputs.routes this way after the step's final
// sync. All three buffers may be null (the copies are skipped).
struct MoeTraceStaging {
  int32_t* ids = nullptr;     // [tokens * top_k]
  float* weights = nullptr;  // [tokens * top_k]
  float* biased = nullptr;    // [tokens * n_experts]
};

// The grouped expert path's kernel (2026-09-05). kGemv: the fp8 GEMV core,
// four rows per pass — the decode path's kernel, bitwise the decode slot
// path at the same routing. kMma: the bf16 tensor-core tile path
// (launch_moe_grouped_mma_*), 128-row m-tiles, bitwise the scale GEMM's
// tile kernel — the PREFILL's kernel, where segments run to hundreds of
// rows and the GEMV core re-read every expert once per four of them.
enum class MoeExpertKernel { kGemv, kMma };

class GlmMoeLayer {
 public:
  // weights: device-visible pointers (e.g. GlmLayerStream's GlmMoeResident
  // wired into GlmMoeWeights); they must outlive this layer.
  //
  // decode_slots: the decode-row bound enqueue_decode accepts (the
  // session engine's kDecodeRows). 0 disables the decode path (its
  // scratch is not allocated) — the M4 forward-only callers and the
  // pre-prefill unit tests use that shape.
  //
  // graph_table_slots: per-call STABLE pinned expert-table sources for
  // CUDA-graph capture (the session passes its MoE-layer ordinal as
  // table_slot). A captured graph replays the table upload NODE with
  // the source address baked — one shared pinned buffer would replay
  // the LAST capture-time refill for every layer (the wrong-weights
  // class). Per-slot buffers freeze each layer's table at capture time
  // (resident bindings are lifetime-stable, so the bytes never go
  // stale); the shared DEVICE destination stays safe because the
  // replay's memcpy nodes and kernels serialize on the stream. 0 keeps
  // the single shared pinned buffer (the eager path).
  GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
              int max_tokens, int decode_slots = 0,
              int graph_table_slots = 0);
  ~GlmMoeLayer();
  GlmMoeLayer(const GlmMoeLayer&) = delete;
  GlmMoeLayer& operator=(const GlmMoeLayer&) = delete;

  // out[tokens, hidden] = routed_sum + shared(hidden); out is zeroed
  // internally and accumulated in place. Synchronizes the stream once.
  // The prefill path without a host sync (2026-09-04): the router's
  // ids stay on the device, a segmentation kernel builds the rows, the
  // slot map and the segment table there, and the grouped chain runs on
  // them — nothing waits on the host. Routing traces ride async copies
  // into `trace` (pinned; materialize after the stream's next sync;
  // last_ids()/last_weights()/last_biased() are NOT updated). Bitwise the
  // host path's output (glm_moe_test pins it).
  // enqueue_prefill runs the grouped chain on the tensor-core kernel
  // (MoeExpertKernel::kMma); enqueue(), the host-orchestrated reference,
  // takes the kernel as an argument — kGemv by default (the decode pin),
  // kMma to serve as the prefill path's bitwise reference.
  void enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                       MoeTraceStaging* trace, cudaStream_t stream);
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream,
               MoeExpertKernel kernel = MoeExpertKernel::kGemv);

  // The decode fast path: same contract, no host round-trip. tokens
  // must fit decode_slots. Bitwise-equal to enqueue() at the same
  // routing (the unit gate's pin); traces land async in `trace` (null:
  // no trace copies at all).
  // table_slot >= 0 is capture mode: the kernels read graph slot's OWN
  // device table, uploaded beforehand by prepare_graph_table() — no
  // upload node is recorded, so a replay moves no table bytes at all.
  void enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                      MoeTraceStaging* trace, cudaStream_t stream,
                      int table_slot = -1);

  // Fills graph slot `table_slot`'s device expert-view table from the
  // CURRENT binding (an async H2D on `stream`; the caller syncs before
  // capturing). Must run OUTSIDE stream capture, once per slot, before
  // the capture that records the slot; the contents freeze from then on
  // (resident bindings). Rebinding to different weights afterwards
  // without re-preparing is the wrong-weights class — enqueue_decode
  // refuses an unprepared slot.
  void prepare_graph_table(int table_slot, cudaStream_t stream);

  // Host copies of the most recent enqueue's routing decision. last_biased()
  // holds every expert's biased score for the same enqueue
  // ([tokens, n_experts], fp32) — the near-tie certification inputs.
  const std::vector<int32_t>& last_ids() const { return h_ids_; }
  const std::vector<float>& last_weights() const { return h_weights_; }
  const std::vector<float>& last_biased() const { return h_biased_; }
  const GlmMoeConfig& config() const { return cfg_; }
  int max_tokens() const { return max_tokens_; }

  // The bytes the constructor allocates for this shape (2026-09-06, the
  // memory plan): device scratch, and the pinned host staging in
  // `*pinned_bytes` (optional). The formula mirrors the constructor line
  // for line so a plan can be checked before anything is allocated.
  static size_t scratch_bytes(const GlmMoeConfig& cfg, int max_tokens,
                              int decode_slots = 0, int graph_table_slots = 0,
                              size_t* pinned_bytes = nullptr);

  // Streaming-weight seam (M4 diagnostic forward): swap the device weight
  // views (router gate/bias, expert and shared matrices). Device scratch
  // and segmentation buffers are shape-keyed and unaffected.
  void rebind(const GlmMoeWeights& w) { w_ = w; }

 private:
  void check_expert_geometry() const;
  // The eager paths' expert-table upload: fills the ring's next pinned
  // entry with every routed expert's three views (and the shared expert's
  // three after them when with_shared) and copies it to d_dst on stream —
  // waiting first, only if the host is kViewRing uploads ahead of the
  // stream, for that entry's previous upload to have executed.
  void upload_expert_views(MoeExpertView* d_dst, bool with_shared,
                           cudaStream_t stream);
  // The grouped chain shared by enqueue() and enqueue_prefill(): gather,
  // gate/up over the routed segments and the shared segment, swiglu, the
  // fp32 down projection — on the chosen kernel.
  void grouped_expert_chain(MoeExpertKernel kernel, const uint16_t* hidden,
                            const MoeSegment* segs, int n_segs, int max_rows,
                            const MoeSegment* shared_seg, int tokens,
                            size_t rows_total, cudaStream_t stream);

  GlmMoeWeights w_;
  GlmMoeConfig cfg_;
  int max_tokens_;
  int decode_slots_ = 0;

  // device scratch (cudaMalloc; sized to max_tokens)
  int32_t* d_ids_ = nullptr;
  float* d_weights_ = nullptr;
  float* d_biased_ = nullptr;  // [max_tokens, n_experts] biased router scores
  float* d_scores_ = nullptr;  // [max_tokens, n_experts] sigmoid scores (router scratch)
  int32_t* d_rows_ = nullptr;
  float* d_row_w_ = nullptr;
  uint16_t* d_gather_ = nullptr;
  uint16_t* d_gate_ = nullptr;
  uint16_t* d_up_ = nullptr;
  uint16_t* d_act_ = nullptr;
  float* d_down_ = nullptr;  // [max_tokens, hidden] fp32 segment output
  float* d_acc_ = nullptr;   // [max_tokens, hidden] the fp32 chain

  // decode-slot scratch (cudaMalloc; sized to decode_slots*(top_k+1) rows —
  // the slot layout the kernels index: routed K + shared, per token)
  uint16_t* d_slot_act_ = nullptr;  // [slots, inter] (fused gate/up/swiglu)
  float* d_slot_down_ = nullptr;
  int32_t* d_slot_order_ = nullptr; // [slots] expert-sorted execution order
  int* d_router_counters_ = nullptr;  // [decode_slots] fused-select tickets    // [slots, hidden] fp32 partial dots
  // The device expert-view table, re-uploaded per enqueue_decode call.
  // NO CACHE, DELIBERATELY: the streaming loader refills ONE
  // GlmLayerResident per layer, so a binding-keyed cache collides across
  // layers (the first MoE layer's table served to every layer after it
  // — glm_tp_test's decode-parity gate caught exactly that, an
  // uncertifiable top-1 flip with infinite margin). One ~14KB async
  // upload per MoE layer per step buys lifetime correctness with zero
  // cleverness; the graph era bakes the tables in properly.
  MoeExpertView* d_expert_views_ = nullptr;  // [n_experts * 3]

  // host staging
  std::vector<int32_t> h_ids_;
  std::vector<float> h_weights_;
  std::vector<float> h_biased_;  // [tokens, n_experts] (certification)
  std::vector<int32_t> h_rows_;
  std::vector<float> h_row_w_;
  std::vector<int> h_counts_;
  // The prefill path's staging (2026-09-04): PINNED, so its copies are
  // truly asynchronous — a pageable async copy is a synchronous staged
  // copy, and the per-expert form of it cost 1.4 s of a 5 s 256-token
  // prefill. The router's ids/weights/biased come down into the pinned
  // mirrors (then into the vectors above, which callers keep references
  // to); the segmented rows/weights go up ONCE per layer from the pinned
  // arrays, each segment addressed by offset; the shared expert's identity
  // rows and unit weights live on the device from construction.
  // The grouped expert path (2026-09-04): the layer's rows — every routed
  // (token, slot) in ascending-expert segment order, then the tokens once
  // more for the shared expert — gathered at once; one grouped launch per
  // matrix over the routed segments and one over the shared segment; the
  // per-token ordered accumulation in one pass. Buffers are sized to
  // max_tokens * (top_k + 1) rows.
  int32_t* h_ids_pinned_ = nullptr;      // [max_tokens * top_k]
  float* h_weights_pinned_ = nullptr;    // [max_tokens * top_k]
  float* h_biased_pinned_ = nullptr;     // [max_tokens * n_experts]
  int32_t* h_seg_rows_ = nullptr;        // [rows_total], pinned
  int32_t* h_slot_row_ = nullptr;        // [max_tokens * top_k], pinned
  MoeSegment* h_segs_ = nullptr;         // [n_experts + 1], pinned
  int32_t* d_slot_row_ = nullptr;        // [max_tokens * top_k]
  MoeSegment* d_segs_ = nullptr;         // [n_experts + 1]
  MoeExpertView* d_views_prefill_ = nullptr;  // [(n_experts + 1) * 3]
  // The eager paths' expert-view upload source (the prefill paths' table
  // of n_experts + 1 rows and eager decode's of n_experts): a RING of
  // pinned tables, each guarded by the event its last upload recorded.
  // PINNED, not a plain vector: a pageable-source cudaMemcpyAsync performs
  // a stream sync before the copy initiates (driver contract), which
  // drains the whole step's pipeline once per MoE layer — the first fabric
  // run of the fused path paid 49ms/token for exactly that. Pinned sources
  // are true async DMA — and so a write-after-read hazard (found
  // 2026-09-05): ONE layer object is rebound for every MoE layer, and a
  // host that has run ahead of the stream (the session prefill has no
  // per-layer sync, and a world of one has no collective to wait on)
  // refilled the single table with the NEXT layer's pointers before the
  // previous layer's copy had executed, so that layer's expert chain ran
  // on the wrong experts' weights — a fresh model's first prefill and its
  // second disagreed by 0.19 rel_l2 (glm_tp_first_run_is_bitwise_the_
  // second_run pins it). Each fill takes the ring's next entry and waits
  // — only when the host is kViewRing uploads ahead — for that entry's
  // previous upload to have executed; the copy is enqueued right after a
  // layer's router and segmentation kernels, so the wait admits a host
  // several layers ahead and never drains the stream. Capture-mode decode
  // reads its per-slot graph tables instead and never touches the ring.
  static constexpr int kViewRing = 4;
  size_t view_table_entries_ = 0;         // (n_experts + 1) * 3
  MoeExpertView* h_view_ring_ = nullptr;  // [kViewRing][view_table_entries_]
  cudaEvent_t view_ring_event_[kViewRing] = {};
  bool view_ring_armed_[kViewRing] = {};
  int view_ring_next_ = 0;
  // Per-graph-slot tables (capture mode): [graph_table_slots_] rows of
  // [n_experts * 3] each. The pinned rows are the upload sources, the
  // device rows are what the recorded kernels read — each slot its own,
  // uploaded once by prepare_graph_table() (a replay copies nothing; the
  // former per-replay upload node cost 42 H2D nodes per token).
  MoeExpertView* h_expert_views_graph_ = nullptr;
  MoeExpertView* d_expert_views_graph_ = nullptr;
  std::vector<bool> graph_table_ready_;
  int graph_table_slots_ = 0;
};

}  // namespace dgpp
