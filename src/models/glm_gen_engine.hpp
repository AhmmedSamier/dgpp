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
// SAMPLING (M6 6b): a second closure, `Sample`, is the stochastic pick —
// the request's spec, its counter RNG (advanced by the draw) and its
// context ids (prompt + generated, the penalties' count table). Per slot
// the adapter keeps exactly that state, armed by configure_sampling()
// before the prefill pick. temperature <= 0 is the greedy closure, at zero
// cost and with the exact op stream every gate pins; an adapter built
// without a Sample closure is greedy-only and says so.
//
// ALLOCATION DISCIPLINE (the burst-wedge lesson): hoist the pick's float
// row OUT of the closure — no per-token device-adjacent allocation on
// the decode path.
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/dtypes.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_scheduler.hpp"

namespace dgpp {

class GenEngineAdapter : public glm::SchedulerEngine {
 public:
  using Pick = std::function<int32_t(const GlmDiagnosticModel::Outputs&)>;
  using Sample = std::function<glm_sample::Result(
      const GlmDiagnosticModel::Outputs&, const glm_sample::Params&,
      glm_sample::Rng&, const std::vector<int32_t>& context)>;

  GenEngineAdapter(GlmDiagnosticModel* model, int max_requests, Pick pick,
                   Sample sample = nullptr)
      : model_(model),
        slots_(max_requests),
        pick_(std::move(pick)),
        sample_(std::move(sample)),
        pending_(static_cast<size_t>(max_requests), -1),
        state_(static_cast<size_t>(max_requests)) {}

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

  bool supports_sampling() const override {
    return static_cast<bool>(sample_);
  }
  bool supports_logprobs() const override {
    return static_cast<bool>(sample_);
  }
  void configure_sampling(int req, const glm_sample::Params& sampling,
                          uint64_t seed) override {
    glm_sample::validate_params(sampling);
    if (sampling.temperature > 0.0f && !sample_)
      throw std::logic_error(
          "generation engine: no sampler bound — this engine is greedy-only");
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.params = sampling;
    s.rng = glm_sample::Rng{seed, 0};
    s.context.clear();
    s.logprobs.clear();
  }
  void configure_logprobs(int req, int logprobs) override {
    if (logprobs >= 0 && !sample_)
      throw std::logic_error(
          "generation engine: no sampler bound — no logprobs either");
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.report_logprobs = logprobs >= 0;
    s.logprobs.clear();
  }
  std::vector<glm_sample::Result> take_logprobs(int req) override {
    SlotState& s = state_.at(static_cast<size_t>(req));
    std::vector<glm_sample::Result> out;
    out.swap(s.logprobs);
    return out;
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    SlotState& s = state_.at(static_cast<size_t>(req));
    s.context.clear();
    s.context.reserve(prompt.size() + 64);
    for (int64_t id : prompt) s.context.push_back(static_cast<int32_t>(id));
    const int32_t token = decide(s, model_->session_prefill(req, prompt));
    s.context.push_back(token);
    pending_.at(static_cast<size_t>(req)) = token;
    return token;
  }
  void reserve(int req, int64_t tokens) override {
    model_->session_reserve_blocks(req, tokens);
  }
  std::vector<int32_t> step(int req) override {
    int64_t& pending = pending_.at(static_cast<size_t>(req));
    if (pending < 0)
      throw std::logic_error("generation step on a slot without a pending "
                             "token");
    SlotState& s = state_.at(static_cast<size_t>(req));
    const int32_t next = decide(s, model_->session_step(req, pending));
    s.context.push_back(next);
    pending = next;
    return {next};
  }
  void close(int req) override {
    pending_.at(static_cast<size_t>(req)) = -1;
    // A reopened slot is greedy until the scheduler arms it again.
    state_.at(static_cast<size_t>(req)) = SlotState{};
    model_->session_close(req);
  }

  // The slot's RNG state (seed, counter) — the journal/audit view of how
  // many draws the request has consumed.
  const glm_sample::Rng& rng(int req) const {
    return state_.at(static_cast<size_t>(req)).rng;
  }

 private:
  struct SlotState {
    glm_sample::Params params = glm_sample::greedy_params();
    glm_sample::Rng rng;
    std::vector<int32_t> context;  // prompt + generated: the count table
    std::vector<glm_sample::Result> logprobs;  // the last op's, when asked
    bool report_logprobs = false;
  };

  // The plain greedy pick serves temperature 0 with neither logprobs nor
  // penalties (zero cost, the exact op stream every gate pins); anything
  // else goes through the sampler closure, which handles temperature 0 as
  // the greedy decision under the raw distribution.
  int32_t decide(SlotState& s, const GlmDiagnosticModel::Outputs& out) {
    const bool penalized = s.params.repetition_penalty != 1.0f ||
                           s.params.frequency_penalty != 0.0f ||
                           s.params.presence_penalty != 0.0f;
    if (s.params.temperature <= 0.0f && !s.report_logprobs && !penalized)
      return pick_(out);
    const glm_sample::Result r = sample_(out, s.params, s.rng, s.context);
    if (s.report_logprobs) s.logprobs.push_back(r);
    return r.token;
  }

  GlmDiagnosticModel* model_;
  int slots_;
  Pick pick_;
  Sample sample_;
  std::vector<int64_t> pending_;
  std::vector<SlotState> state_;
};

// The world-1 pick: full-vocab argmax over the fp32 logits row. The closure
// keeps the decode path allocation-free.
inline GenEngineAdapter::Pick make_w1_pick(int64_t vocab) {
  return [vocab](const GlmDiagnosticModel::Outputs& out) -> int32_t {
    const int32_t t =
        glm_sample::local_max(out.logits.data(),
                              static_cast<int>(out.lm_vocab_count),
                              /*vocab_begin=*/0)
            .id;
    if (t < 0 || t >= vocab)
      throw std::runtime_error("w1 pick out of range: " + std::to_string(t));
    return t;
  };
}

// The world-1 sampler: the complete distribution is on this host, so the
// decision is sample_full_logits at the one-slice layout — bitwise the
// sharded reference at world 1.
inline GenEngineAdapter::Sample make_w1_sample(int64_t vocab) {
  const std::vector<glm_sample::VocabSlice> layout =
      glm_sample::vocab_layout(static_cast<int>(vocab), 1);
  return [vocab, layout](const GlmDiagnosticModel::Outputs& out,
                         const glm_sample::Params& p, glm_sample::Rng& rng,
                         const std::vector<int32_t>& context)
             -> glm_sample::Result {
    if (out.lm_vocab_begin != 0 || out.lm_vocab_count != vocab)
      throw std::runtime_error("w1 sample: the head is not the full vocab");
    if (p.temperature <= 0.0f) {
      // The greedy decision with logprobs or penalties: penalties, the raw
      // normalizer, the canonical argmax under it.
      std::vector<float> v(out.logits.begin(), out.logits.begin() + vocab);
      glm_sample::apply_penalties(v.data(), static_cast<int>(vocab), 0, p,
                                  glm_sample::count_context(context));
      const double lse =
          glm_sample::sharded_scaled_logsumexp(v.data(), layout, 1.0f);
      const int n = std::max(1, p.logprobs);
      const std::vector<glm_sample::Candidate> top =
          glm_sample::local_topk(v.data(), static_cast<int>(vocab), 0, n);
      return glm_sample::greedy_from_prefix(top, lse, p.logprobs);
    }
    glm_sample::Result r = glm_sample::sample_full_logits(
        out.logits.data(), static_cast<int>(vocab), layout, p, rng, context);
    if (r.token < 0 || r.token >= vocab)
      throw std::runtime_error("w1 sample out of range: " +
                               std::to_string(r.token));
    return r;
  };
}

}  // namespace dgpp
