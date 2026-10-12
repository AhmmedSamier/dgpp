#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dgpp {

// A continuation borrows its prompt and snapshot chain. The owning engine
// task keeps them alive; a session generation prevents reuse after close.
template <class Output, class Snapshot>
struct PrefillContinuation {
  uint64_t epoch = 0;
  int req = -1;
  const int64_t* ids = nullptr;
  int64_t start = 0, end = 0, next = 0, budget_tokens = 0;
  std::vector<int64_t> cuts;
  size_t cut_index = 0;
  bool span_start = true, suspended = false;
  bool relative_budget = false;
  bool final_output_only = false;
  Snapshot* snap = nullptr;
  Output output;

  bool closes_span(int64_t position) const {
    if (position == end) return true;
    for (auto* at = snap; at; at = at->next)
      if (!at->taken && at->position == position) return true;
    return false;
  }
  int64_t span_end() const { return cut_index < cuts.size() ? cuts[cut_index] : end; }
  int64_t next_end(int64_t budget) const {
    if (budget <= 0) return span_end();
    const auto limit = relative_budget ? next + budget : (next / budget + 1) * budget;
    return std::min(span_end(), limit);
  }
};

}  // namespace dgpp
