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

}  // namespace dgpp
