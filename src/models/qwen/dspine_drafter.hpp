#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "models/dflash2/drafter.hpp"
#include "models/qwen/dflash.hpp"

namespace dgpp {

// Eager DSpine execution hosted by the Qwen3.8 target. The target owns the
// paged planes and supplies its exact BF16 embedding/head and GEMM envelope.
class DSpineDrafter {
 public:
  DSpineDrafter() = default;
  DSpineDrafter(const DSpineDrafter&) = delete;
  DSpineDrafter& operator=(const DSpineDrafter&) = delete;
  ~DSpineDrafter();

  void init(const DFlash2Host& host, const DFlashConfig& cfg, const std::string& dir);
  void free();
  bool on() const { return on_; }
  const DFlashConfig& cfg() const { return cfg_; }
  const DFlashWeights& weights() const { return weights_; }
  int drafts() const { return cfg_.drafts(); }
  int query_rows() const { return cfg_.query_rows(); }
  int tap(int target_layer) const;
  float* acc() const { return acc_; }
  float* tap_gemm_out() const { return tap_out_; }

  void store_features(int rows, const int32_t* req_ids, const int64_t* positions);
  bool draft(int request, int64_t anchor, std::vector<int32_t>* out);

  static void plan_entries(const DFlashConfig& cfg, int max_tokens,
                           int64_t vocab_count,
                           std::vector<std::pair<std::string, size_t>>* out);

 private:
  void block_layers(int request);
  void select_chain();

  DFlash2Host host_{};
  DFlashConfig cfg_{};
  DFlashWeights weights_{};
  std::vector<int> tap_map_;
  bool on_ = false;
  int q_width_ = 0, kv_width_ = 0, inter_ = 0, heads_ = 0, kv_heads_ = 0;
  float* acc_ = nullptr;
  float* tap_out_ = nullptr;
  uint16_t *x_ = nullptr, *resid_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  uint16_t *attn_ = nullptr, *proj_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr;
  uint16_t *hidden_ = nullptr, *raw_ = nullptr, *normalized_ = nullptr, *codes_ = nullptr;
  uint16_t *packed_ = nullptr, *message_ = nullptr, *delta_ = nullptr, *scaled_ = nullptr;
  uint16_t *candidate_head_ = nullptr, *sender_codes_ = nullptr;
  uint16_t *tg_ = nullptr, *tsv_gate_ = nullptr, *tsv_value_ = nullptr, *tsv_ = nullptr;
  uint16_t* transposed_write_ = nullptr;
  float *rms_ = nullptr, *logits_ = nullptr, *prewrite_logits_ = nullptr;
  float* scaled_scores_ = nullptr;
  float* attn_partials_ = nullptr;
  void* topk_workspace_ = nullptr;
  size_t topk_workspace_bytes_ = 0;
  int32_t *ids_ = nullptr, *sender_ids_ = nullptr, *chain_index_ = nullptr, *drafts_ = nullptr;
  int64_t *positions_ = nullptr, *tokens_ = nullptr;
  int64_t* host_io_ = nullptr;
  float* inv_freq_ = nullptr;
};

}  // namespace dgpp
