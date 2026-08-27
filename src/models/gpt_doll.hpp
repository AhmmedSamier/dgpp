#pragma once
#include <cstdint>
#include <vector>

#include "core/arena.hpp"
#include "core/graph.hpp"
#include "core/streams.hpp"
#include "kernels/gemm.hpp"

namespace dgpp {

// Synthetic transformer exercising every M1 runtime facility end to end:
// arena residency classes, stream/event orchestration, CUDA-graph keyed
// replay, cuBLASLt GEMM seam, fused elementwise kernels, GQA attention.
// Deviations from GLM-5.3 (documented, deliberate):
//   * no RoPE / no per-head qk-norm  (arrive with M2/M3 real ops)
//   * prefill attention is O(T^2) simple kernel; decode path is efficient
//   * weights are hash-seeded random, deterministic per seed
struct DollConfig {
  int vocab = 154880;
  int hidden = 4096;
  int layers = 8;
  int heads = 32;
  int kv_heads = 8;
  int head_dim = 128;
  int inter = 12288;
  float eps = 1e-5f;
  float swiglu_limit = 10.f;
  int max_seq = 4096;      // KV cache slots per layer
  int max_chunk = 2048;    // max prefill rows handled in one call
  uint64_t seed = 1234;
};

class DollModel {
 public:
  static constexpr int kMaxRolloutSteps = 4096;

  DollModel(Arena& arena, StreamPool& streams, GraphCache& graphs,
            IGemm& gemm, const DollConfig& cfg);
  ~DollModel();

  // Sizing helper so callers can init Arena before constructing weights.
  static size_t persistent_hot_bytes(const DollConfig& cfg,
                                     int rollout_steps_cap);

  void init_weights();

  // ---- generation API (single stream currently active) -----------------
  // Prefill consumes prompt tokens; caches fill [0..n); samples next token.
  // Throws when n > max_seq/max_chunk.
  void enqueue_prefill(const std::vector<int64_t>& tokens);
  int64_t sample_prefill_result();          // blocks until result ready

  // Decode one step: appends `token`, returns sampled continuation.
  void enqueue_decode_step(int64_t token);
  int64_t poll_decode_sample();             // waits for enqueue above

  // Captures a monolithic N-step self-feeding rollout graph (each step
  // consumes the argmax of the previous). Replayed by run_rollout();
  // transcript ids land in pinned buffer readable via rollout_ids().
  void capture_rollout(int steps);
  void run_rollout(int64_t seed_token);
  const std::vector<int64_t>& rollout_ids() const { return pinned_ids_; }

  // Naive (no-graph) single decode step used by bitwise self-tests.
  int64_t eager_decode_step(int64_t token);
  // Same without graph capture for prefill; returns nothing (use
  // enqueue-style sampling identical to graphed variant via logits buffer).
  int64_t eager_prefill_result(const std::vector<int64_t>& tokens);

  // Empties KV caches and position counter (self-test/reset path).
  void reset_kv();

  // Traffic model for roofline reporting at given cached sequence length.
  uint64_t decode_step_bytes(int seq_len) const;

  const DollConfig& config() const { return cfg_; }
  int cur_len() const { return cur_len_; }
  const float* debug_logits_f32() const { return logits32_; }
  // Diagnostic accessors (self-test probes only).
  const void* debug_embed() const { return embed_; }
  const void* debug_x_rows(size_t* stride_elems) const {
    *stride_elems = cfg_.hidden;
    return x_;
  }
  const void* debug_normed_row() const { return normed_; }
  const void* debug_act8_row() const { return act8_; }

 private:
  enum class Mode { Prefill, Decode };
  struct BodyCtx {
    Mode mode;
    int n_rows;        // query rows baked into this graph/eager invocation
    int logit_row_idx; // which activation row feeds lm_head
    int64_t* ids_out;  // argmax destination slot(s)
  };

  void record_body(cudaStream_t s, const BodyCtx& b);
  void launch_meta_copies(cudaStream_t s, int n_rows, int abs_first,
                          int64_t token);
  void finish_step_outputs(cudaStream_t s);
  void reset_generation_state();

  Arena& arena_;
  StreamPool& sp_;
  GraphCache& graphs_;
  IGemm& gemm_;
  DollConfig cfg_;

  // weights
  uint16_t* embed_ = nullptr;
  std::vector<uint16_t*> ln1_, ln2_;
  uint16_t* final_ln_ = nullptr;
  std::vector<uint16_t*> w_qkv_, w_o_, w13_, w2_;
  uint8_t* lm_head_fp8_ = nullptr;

  // kv caches
  std::vector<uint16_t*> kc_, vc_;

  // activations / scratch (fixed addresses; sized to max_chunk)
  uint16_t* x_ = nullptr;
  uint16_t* normed_ = nullptr;
  uint16_t* qkv_buf_ = nullptr;
  uint16_t* attn_out_ = nullptr;
  uint16_t* normed2_ = nullptr;
  uint16_t* gateup_ = nullptr;
  uint16_t* act_ = nullptr;
  uint16_t* ffn_out_ = nullptr;
  uint8_t* act8_ = nullptr;
  uint16_t* bf16_logits_ = nullptr;
  float* logits32_ = nullptr;
  int64_t* tok_dev_ = nullptr;
  int32_t* abs_first_dev_ = nullptr;
  int32_t* nq_dev_ = nullptr;
  int64_t* tok_out_dev_ = nullptr;
  int64_t* ids_hist_dev_ = nullptr;
  void* gemm_ws_ = nullptr;

  // host-pinned step interface (one slab: scalars + prompt staging + ids)
  int64_t* pin_tok_ = nullptr;
  int32_t* pin_pos_ = nullptr;
  int32_t* pin_nq_ = nullptr;
  int64_t* pin_tok_out_ = nullptr;
  int64_t* pin_prompt_ = nullptr;

  // bookkeeping
  int cur_len_ = 0;
  bool f32_lm_head_ok_ = false;
  GemmOut lm_head_out_ = GemmOut::F32;
  cudaEvent_t step_done_{};
  cudaEvent_t copy_done_{};
  int rollout_steps_ = 0;
  std::vector<int64_t> pinned_ids_;
};

}  // namespace dgpp