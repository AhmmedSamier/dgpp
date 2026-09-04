#pragma once
// Greedy speculative decode (DESIGN §9) over a GlmDiagnosticModel built
// with mtp = true.
//
// Per step the main stack verifies [next, draft] — the token it was going
// to consume anyway plus the MTP block's guess for the one after — in ONE
// T=2 call. Row 0's argmax is what a plain step would have produced for
// `next`; if it equals the draft, row 1 was a legitimate step too and its
// argmax is the next `next`: two tokens for one weight sweep. Otherwise
// row 1 is retracted (session_rollback) and row 0's argmax is the next
// `next`, exactly a plain step. Either way the block then drafts over the
// accepted rows. The transcript is BITWISE the plain greedy transcript:
// verify rows are the T=1 rows (GlmDiagnosticModel::session_verify), and
// the pick is the same canonical argmax — MTP changes the cost, never the
// output.
//
// judge_verify is the pure core; GreedySpeculator is the eager driver the
// tests run; the graph-era loop in glm_gen_check replays graphs around the
// same judge.
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/glm_forward.hpp"
#include "models/glm_sampler.hpp"

namespace dgpp {

struct SpecVerdict {
  int accepted = 0;                  // rows [0, accepted) of the verify stand
  std::vector<int32_t> committed;    // the tokens those rows consumed (final)
  int32_t next = -1;                 // the token the main stack consumes next
  std::vector<int64_t> draft_rows;   // the block's next inputs: rows'
                                     // argmaxes, i.e. committed[1..] + next
};

// fed: the verify's tokens [next, draft_1, ...]; winners: each row's
// argmax. Row r stands if every row before it stood and row r-1 predicted
// fed[r] (row 0 always stands: it consumed a token already decided).
inline SpecVerdict judge_verify(const std::vector<int64_t>& fed,
                                const std::vector<int32_t>& winners) {
  if (fed.empty() || winners.size() != fed.size())
    throw std::invalid_argument("judge_verify: fed/winners shape");
  SpecVerdict v;
  int a = 1;
  while (a < static_cast<int>(fed.size()) &&
         winners[static_cast<size_t>(a - 1)] == fed[static_cast<size_t>(a)])
    ++a;
  v.accepted = a;
  v.committed.assign(fed.begin(), fed.begin() + a);
  v.next = winners[static_cast<size_t>(a - 1)];
  v.draft_rows.assign(winners.begin(), winners.begin() + a);
  return v;
}

// Per-row local argmax over a rank's [rows, count] logits slice.
inline std::vector<glm_sample::Candidate> local_row_maxes(
    const GlmDiagnosticModel::Outputs& out, int rows) {
  std::vector<glm_sample::Candidate> locals;
  locals.reserve(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r)
    locals.push_back(glm_sample::local_max(
        out.logits.data() + static_cast<size_t>(r) * out.lm_vocab_count,
        out.lm_vocab_count, out.lm_vocab_begin));
  return locals;
}

// The eager driver. `pick_rows` turns every rank's per-row local maxes
// into the per-row global winners (bus_greedy_pick_rows on the fabric, the
// identity argmax at world 1).
class GreedySpeculator {
 public:
  using PickRows = std::function<std::vector<int32_t>(
      const std::vector<glm_sample::Candidate>&)>;

  GreedySpeculator(GlmDiagnosticModel& model, int req, PickRows pick_rows)
      : model_(model), req_(req), pick_rows_(std::move(pick_rows)) {
    if (!model_.mtp_enabled())
      throw std::invalid_argument("GreedySpeculator: the model has no MTP");
  }

  // After session_prefill: `first` is the pick off the prefill logits.
  // Drafts the first proposal (the block's row P-1).
  void start(int32_t first) {
    next_ = first;
    draft_ = draft_after({first});
  }
  // Test seam: the block still drafts its first row (its counter must
  // move), but the proposal fed to the first verify is `forced_draft` —
  // feeding the true next token makes the first step an accept-all.
  void start(int32_t first, int32_t forced_draft) {
    start(first);
    draft_ = forced_draft;
  }

  // One speculative step. Returns the tokens that became final this step
  // (1 or 2); next() is then the following token, already decided but not
  // yet consumed by the main stack.
  std::vector<int32_t> step() {
    const std::vector<int64_t> fed{next_, draft_};
    const GlmDiagnosticModel::Outputs out = model_.session_verify(req_, fed);
    const std::vector<int32_t> winners = pick_rows_(local_row_maxes(out, 2));
    const SpecVerdict v = judge_verify(fed, winners);
    if (v.accepted < 2) model_.session_rollback(req_, v.accepted);
    ++steps_;
    accepted_drafts_ += v.accepted - 1;
    next_ = v.next;
    draft_ = draft_after(v.draft_rows);
    return v.committed;
  }

  int32_t next() const { return next_; }
  // The block's current proposal for the token after next().
  int32_t draft() const { return draft_; }
  int steps() const { return steps_; }
  int accepted_drafts() const { return accepted_drafts_; }

 private:
  int32_t draft_after(const std::vector<int64_t>& rows) {
    const GlmDiagnosticModel::Outputs d = model_.session_draft(req_, rows);
    return pick_rows_(local_row_maxes(d, 1))[0];
  }

  GlmDiagnosticModel& model_;
  int req_ = 0;
  PickRows pick_rows_;
  int32_t next_ = -1;
  int32_t draft_ = -1;
  int steps_ = 0;
  int accepted_drafts_ = 0;
};

// ---------------------------------------------------------------------------
// Exact speculative SAMPLING with a deterministic draft (DESIGN §9/§10, M6
// 6b): the eager driver, and the oracle the on-device T=2 verdict is held
// to. Per step the main stack verifies [next, draft]; row 0's final
// distribution P0 (the plain sampler's, penalties over the context that
// includes `next`) accepts the draft with probability P0(draft) or yields
// the residual sample (glm_sample::spec_select_from_sorted, over the bus
// bus_spec_accept); when the draft stands, row 1 is sampled as any step's
// token would be (context + draft). Two draws per step. The draft itself
// stays the block's argmax — deterministic, so the accept test is the whole
// correction. The marginal distribution of every committed token is the
// plain sampler's, so a transcript is a legitimate sample; only the cost
// changes with the acceptance rate. The context (the penalties' count
// table) is prompt + committed + next, and the draft joins it only when it
// stands — the device count table follows the same rule.
// ---------------------------------------------------------------------------
class SampledSpeculator {
 public:
  using PickRows = GreedySpeculator::PickRows;
  using Row0 = std::function<glm_sample::SpecPrefixDecision(
      const GlmDiagnosticModel::Outputs& row0, int32_t draft,
      const glm_sample::Params& p, glm_sample::Rng& rng,
      const std::vector<int32_t>& context)>;
  using Row1 = std::function<glm_sample::Result(
      const GlmDiagnosticModel::Outputs& row1, const glm_sample::Params& p,
      glm_sample::Rng& rng, const std::vector<int32_t>& context)>;

  SampledSpeculator(GlmDiagnosticModel& model, int req, PickRows draft_pick,
                    Row0 row0, Row1 row1, const glm_sample::Params& params,
                    glm_sample::Rng rng, const std::vector<int64_t>& prompt)
      : model_(model),
        req_(req),
        draft_pick_(std::move(draft_pick)),
        row0_(std::move(row0)),
        row1_(std::move(row1)),
        params_(params),
        rng_(rng) {
    if (!model_.mtp_enabled())
      throw std::invalid_argument("SampledSpeculator: the model has no MTP");
    if (!(params_.temperature > 0.0f))
      throw std::invalid_argument("SampledSpeculator: temperature must be > 0 "
                                  "(the greedy driver is GreedySpeculator)");
    context_.assign(prompt.begin(), prompt.end());
  }

  // After session_prefill: `first` is the sampled pick off the prefill
  // logits (already drawn with this speculator's RNG by the caller, who
  // passes the advanced state in). Drafts the first proposal.
  void start(int32_t first) {
    next_ = first;
    context_.push_back(first);
    draft_ = draft_after({first});
  }

  // One speculative step: the tokens that became final (1 or 2); next() is
  // then the following token, decided but not yet consumed.
  std::vector<int32_t> step() {
    const std::vector<int64_t> fed{next_, draft_};
    const GlmDiagnosticModel::Outputs out = model_.session_verify(req_, fed);
    const glm_sample::SpecPrefixDecision d0 =
        row0_(row_view(out, 0), draft_, params_, rng_, context_);
    if (!d0.resolved)
      throw std::logic_error("SampledSpeculator: row 0 must resolve");
    std::vector<int32_t> winners;
    int32_t next_new = -1;
    if (d0.accepted) {
      context_.push_back(draft_);
      const glm_sample::Result r1 =
          row1_(row_view(out, 1), params_, rng_, context_);
      winners = {draft_, r1.token};
      next_new = r1.token;
    } else {
      model_.session_rollback(req_, 1);
      winners = {d0.result.token};
      next_new = d0.result.token;
    }
    const int accepted = static_cast<int>(winners.size());
    std::vector<int32_t> committed(fed.begin(), fed.begin() + accepted);
    ++steps_;
    accepted_drafts_ += accepted - 1;
    context_.push_back(next_new);
    next_ = next_new;
    draft_ = draft_after(std::vector<int64_t>(winners.begin(), winners.end()));
    return committed;
  }

  int32_t next() const { return next_; }
  int32_t draft() const { return draft_; }
  int steps() const { return steps_; }
  int accepted_drafts() const { return accepted_drafts_; }
  const glm_sample::Rng& rng() const { return rng_; }
  const std::vector<int32_t>& context() const { return context_; }

 private:
  static GlmDiagnosticModel::Outputs row_view(
      const GlmDiagnosticModel::Outputs& out, int row) {
    GlmDiagnosticModel::Outputs v;
    const size_t count = static_cast<size_t>(out.lm_vocab_count);
    v.logits.assign(out.logits.begin() + static_cast<long>(row * count),
                    out.logits.begin() + static_cast<long>((row + 1) * count));
    v.lm_vocab_begin = out.lm_vocab_begin;
    v.lm_vocab_count = out.lm_vocab_count;
    return v;
  }
  int32_t draft_after(const std::vector<int64_t>& rows) {
    const GlmDiagnosticModel::Outputs d = model_.session_draft(req_, rows);
    return draft_pick_(local_row_maxes(d, 1))[0];
  }

  GlmDiagnosticModel& model_;
  int req_ = 0;
  PickRows draft_pick_;
  Row0 row0_;
  Row1 row1_;
  glm_sample::Params params_;
  glm_sample::Rng rng_;
  std::vector<int32_t> context_;
  int32_t next_ = -1;
  int32_t draft_ = -1;
  int steps_ = 0;
  int accepted_drafts_ = 0;
};

}  // namespace dgpp
