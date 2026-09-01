#pragma once
// The scheduler-engine closure over GlmDiagnosticModel + the pick (M6
// Stage 1d's shape, extracted for Stage 4's serving app — ONE seam, two
// apps). The scheduler stays pure host code; at TP>1 the pick is the
// distributed greedy pick — a collective — so the adapter's call order
// IS the collective order (§11's identical-rank-order contract).
//
// The pick is a std::function closed over the caller's world: w1 uses
// the full-vocab argmax (make_w1_pick); the fabric uses bus_greedy_pick
// over the vocab-sharded head (glm_gen_check's fabric path).
//
// ALLOCATION DISCIPLINE (the burst-wedge lesson): hoist the pick's float
// row OUT of the closure — no per-token device-adjacent allocation on
// the decode path.
#include <cstdint>
#include <functional>
#include <vector>

#include "common/dtypes.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_scheduler.hpp"

namespace dgpp {

class GenEngineAdapter : public glm::SchedulerEngine {
 public:
  using Pick = std::function<int32_t(const GlmDiagnosticModel::Outputs&)>;
  GenEngineAdapter(GlmDiagnosticModel* model, int max_requests, Pick pick)
      : model_(model), slots_(max_requests), pick_(std::move(pick)) {}

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override {
    return model_->dsa_blocks_total();
  }
  int64_t pool_blocks_in_use() const override {
    return model_->dsa_blocks_in_use();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return model_->dsa_blocks_for_tokens(tokens);
  }
  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    return pick_(model_->session_prefill(req, prompt));
  }
  int32_t step(int req, int64_t prev_token) override {
    return pick_(model_->session_step(req, prev_token));
  }
  void close(int req) override { model_->session_close(req); }

 private:
  GlmDiagnosticModel* model_;
  int slots_;
  Pick pick_;
};

// The world-1 pick: full-vocab argmax over the bf16 logits row. The
// float conversion is hoisted into the returned closure so the decode
// path allocates nothing per token.
inline GenEngineAdapter::Pick make_w1_pick(int64_t vocab) {
  return [vocab, row = std::vector<float>()](
             const GlmDiagnosticModel::Outputs& out) mutable -> int32_t {
    row.resize(static_cast<size_t>(out.lm_vocab_count));
    for (int i = 0; i < out.lm_vocab_count; ++i)
      row[static_cast<size_t>(i)] =
          bf16_bits_to_float(out.logits_bits[static_cast<size_t>(i)]);
    const int32_t t =
        glm_sample::local_max(row.data(), static_cast<int>(out.lm_vocab_count),
                               /*vocab_begin=*/0)
            .id;
    if (t < 0 || t >= vocab)
      throw std::runtime_error("w1 pick out of range: " + std::to_string(t));
    return t;
  };
}

}  // namespace dgpp
