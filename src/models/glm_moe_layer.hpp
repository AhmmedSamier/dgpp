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
// device memory, foreign experts early-exit, and one ordered
// accumulation reproduces the host path's exact bf16 chain (bitwise —
// glm_moe_test pins it). NO stream sync, NO host segmentation, NO
// per-segment H2D round trips; route traces ride async copies into
// caller-pinned staging (MoeTraceStaging) and materialize after the
// step's final sync. The prefill path keeps enqueue(): its sync
// amortizes over 2048-token chunks, and it warms the decode path's
// device expert tables (see ensure_device_views).
//
// Route-trace capture (M4 deliverable 5): the most recent routing decision
// is retained on the host (last_ids/last_weights) so callers can record it
// per layer into a trace file (models/glm_trace.hpp) for the corrected
// per-rank traffic model.
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "models/glm_moe.hpp"

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

class GlmMoeLayer {
 public:
  // weights: device-visible pointers (e.g. GlmLayerStream's GlmMoeResident
  // wired into GlmMoeWeights); they must outlive this layer.
  //
  // decode_slots: the decode-row bound enqueue_decode accepts (the
  // session engine's kDecodeRows). 0 disables the decode path (its
  // scratch is not allocated) — the M4 forward-only callers and the
  // pre-prefill unit tests use that shape.
  GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
              int max_tokens, int decode_slots = 0);
  ~GlmMoeLayer();
  GlmMoeLayer(const GlmMoeLayer&) = delete;
  GlmMoeLayer& operator=(const GlmMoeLayer&) = delete;

  // out[tokens, hidden] = routed_sum + shared(hidden); out is zeroed
  // internally and accumulated in place. Synchronizes the stream once.
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream);

  // The decode fast path: same contract, no host round-trip. tokens
  // must fit decode_slots. Bitwise-equal to enqueue() at the same
  // routing (the unit gate's pin); traces land async in `trace`.
  void enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                      MoeTraceStaging* trace, cudaStream_t stream);

  // Host copies of the most recent enqueue's routing decision. last_biased()
  // holds every expert's biased score for the same enqueue
  // ([tokens, n_experts], fp32) — the near-tie certification inputs.
  const std::vector<int32_t>& last_ids() const { return h_ids_; }
  const std::vector<float>& last_weights() const { return h_weights_; }
  const std::vector<float>& last_biased() const { return h_biased_; }
  const GlmMoeConfig& config() const { return cfg_; }
  int max_tokens() const { return max_tokens_; }

  // Streaming-weight seam (M4 diagnostic forward): swap the device weight
  // views (router gate/bias, expert and shared matrices). Device scratch
  // and segmentation buffers are shape-keyed and unaffected.
  void rebind(const GlmMoeWeights& w) { w_ = w; }

 private:
  void run_expert_segment(const uint16_t* x, const int32_t* rows_dev,
                          const float* row_w_dev, uint16_t* acc, int n_rows,
                          const GlmQuantMatrix& gate,
                          const GlmQuantMatrix& up,
                          const GlmQuantMatrix& down, cudaStream_t stream);

  GlmMoeWeights w_;
  GlmMoeConfig cfg_;
  int max_tokens_;
  int decode_slots_ = 0;

  // device scratch (managed; sized to max_tokens)
  int32_t* d_ids_ = nullptr;
  float* d_weights_ = nullptr;
  float* d_biased_ = nullptr;  // [max_tokens, n_experts] router scores
  int32_t* d_rows_ = nullptr;
  float* d_row_w_ = nullptr;
  uint16_t* d_gather_ = nullptr;
  uint16_t* d_gate_ = nullptr;
  uint16_t* d_up_ = nullptr;
  uint16_t* d_act_ = nullptr;
  uint16_t* d_down_ = nullptr;

  // decode-slot scratch (managed; sized to decode_slots*(top_k+1) rows —
  // the slot layout the kernels index: routed K + shared, per token)
  uint16_t* d_slot_gate_ = nullptr;  // [slots, inter]
  uint16_t* d_slot_up_ = nullptr;    // [slots, inter]
  uint16_t* d_slot_act_ = nullptr;   // [slots, inter]
  uint16_t* d_slot_down_ = nullptr;  // [slots, hidden] (contributions)
  // The device expert-view table, re-uploaded per enqueue_decode call.
  // NO CACHE, DELIBERATELY: the streaming loader refills ONE
  // GlmLayerResident per layer, so a binding-keyed cache collides across
  // layers (the first MoE layer's table served to every layer after it
  // — glm_tp_test's decode-parity gate caught exactly that, an
  // uncertifiable top-1 flip with infinite margin). One 3.5KB async
  // upload per MoE layer per step (~150us/step at real dims) buys
  // lifetime correctness with zero cleverness; the graph era bakes the
  // tables in properly.
  MoeExpertView* d_expert_views_ = nullptr;  // [n_experts * 3], max-sized

  // host staging
  std::vector<int32_t> h_ids_;
  std::vector<float> h_weights_;
  std::vector<float> h_biased_;  // [tokens, n_experts] (certification)
  std::vector<int32_t> h_rows_;
  std::vector<float> h_row_w_;
  std::vector<int> h_counts_;
  // The expert-view upload source. PINNED, not a plain vector: a
  // pageable-source cudaMemcpyAsync performs a stream sync before the
  // copy initiates (driver contract), which drains the whole step's
  // pipeline once per MoE layer — the first fabric run of the fused
  // path paid 49ms/token for exactly that. Pinned sources are true
  // async DMA.
  MoeExpertView* h_expert_views_pinned_ = nullptr;  // [n_experts * 3]
};

}  // namespace dgpp
