#pragma once
// Host-orchestrated MoE layer forward (M4 deliverable 1, diagnostic mode):
// router kernel -> host segmentation by expert -> per-expert gather +
// scale-aware GEMMs + swiglu + accumulation (ascending expert order) ->
// shared expert. One stream sync per enqueue (the router's ids/weights come
// back to the host for segmentation); correctness-first — the production
// grouped-expert kernel without host round-trips is M5+ work and is
// explicitly out of scope here.
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

class GlmMoeLayer {
 public:
  // weights: device-visible pointers (e.g. GlmLayerStream's GlmMoeResident
  // wired into GlmMoeWeights); they must outlive this layer.
  GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
              int max_tokens);
  ~GlmMoeLayer();
  GlmMoeLayer(const GlmMoeLayer&) = delete;
  GlmMoeLayer& operator=(const GlmMoeLayer&) = delete;

  // out[tokens, hidden] = routed_sum + shared(hidden); out is zeroed
  // internally and accumulated in place. Synchronizes the stream once.
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream);

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

  // host staging
  std::vector<int32_t> h_ids_;
  std::vector<float> h_weights_;
  std::vector<float> h_biased_;  // [tokens, n_experts] (certification)
  std::vector<int32_t> h_rows_;
  std::vector<float> h_row_w_;
  std::vector<int> h_counts_;
};

}  // namespace dgpp
