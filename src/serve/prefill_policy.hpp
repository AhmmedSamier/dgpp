#pragma once

#include <algorithm>
#include <optional>

#include "sched/scheduler.hpp"

namespace dgpp::serve {

inline constexpr int64_t kDefaultPrefillBudgetTokens = 256;
inline constexpr int64_t kDefaultPrefillIdleBudgetTokens = 4096;

// -1 is the serving default. Resolve it before publishing the warm record:
// every rank must run the same concrete budget, independent of HTTP timing.
inline sched::AdmissionPolicy resolve_prefill_policy(
    sched::AdmissionPolicy policy, const sched::SchedulerEngine& engine) {
  if (policy.prefill_budget_tokens != -1) return policy;
  const int64_t align = engine.prefill_chunk_alignment();
  const int64_t limit = engine.prefill_chunk_limit();
  policy.prefill_budget_tokens = 0;
  if (align > 0 && limit >= align) {
    // Keep decode opportunities close together while a reader is active.
    // These are shared starting values, adjusted only for execution geometry;
    // deployments can explicitly choose larger chunks to favor prefill rate.
    int64_t budget = std::min(limit, std::max(kDefaultPrefillBudgetTokens, align));
    if (policy.prefill_idle_budget_tokens > 0)
      budget = std::min<int64_t>(budget, policy.prefill_idle_budget_tokens);
    policy.prefill_budget_tokens = budget / align * align;
    // Grouped readers share a larger walk when no established decoder needs
    // the busy bound. Cap the automatic idle size too: an arriving request
    // or cancellation still has to wait for this chunk to finish.
    if (policy.prefill_idle_budget_tokens == 0 && engine.prefill_group_advance())
      policy.prefill_idle_budget_tokens = static_cast<int>(
          std::min(limit, std::max(kDefaultPrefillIdleBudgetTokens, align)) / align * align);
  }
  return policy;
}

// Configuration carries automatic/on/off until the engine exists. Resolve
// from execution support, never checkpoint architecture or drafter names.
// The warm record carries the resulting boolean to every rank.
inline sched::AdmissionPolicy resolve_prefill_policy(
    sched::AdmissionPolicy policy, const sched::SchedulerEngine& engine,
    std::optional<bool> early_admission) {
  policy = resolve_prefill_policy(policy, engine);
  policy.prefill_early_admission = early_admission.value_or(
      policy.prefill_budget_tokens > 0 && engine.supports_mixed_prefill());
  return policy;
}

}  // namespace dgpp::serve
