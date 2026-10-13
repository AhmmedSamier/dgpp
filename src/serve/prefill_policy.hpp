#pragma once

#include <algorithm>
#include <optional>

#include "sched/scheduler.hpp"

namespace dgpp::serve {

inline constexpr int64_t kDefaultPrefillBudgetTokens = sched::kInitialPrefillBudgetTokens;
inline constexpr int64_t kDefaultPrefillIdleBudgetTokens = 4096;

// -1 is the serving default. Resolve it before publishing the warm record:
// every rank must run the same concrete budget, independent of HTTP timing.
inline sched::AdmissionPolicy resolve_prefill_policy(
    sched::AdmissionPolicy policy, const sched::SchedulerEngine& engine) {
  if (policy.prefill_budget_tokens != -1) return policy;
  policy.prefill_budget_automatic = true;
  const int64_t align = engine.prefill_chunk_alignment();
  const int64_t limit = engine.prefill_chunk_limit();
  policy.prefill_budget_tokens = 0;
  if (align > 0 && limit >= align) {
    // Resolve the ceiling first so a configured cold-start value cannot
    // silently raise the automatic idle cap.
    if (policy.prefill_idle_budget_tokens == 0 && engine.prefill_group_advance())
      policy.prefill_idle_budget_tokens = static_cast<int>(
          std::min(limit, std::max(kDefaultPrefillIdleBudgetTokens, align)) / align * align);
    // Keep decode opportunities close together while a reader is active.
    // These are shared starting values, adjusted only for execution geometry;
    // deployments can explicitly choose larger chunks to favor prefill rate.
    int64_t budget = std::min(limit, std::max<int64_t>(policy.prefill_latency.initial_tokens, align));
    if (policy.prefill_idle_budget_tokens > 0)
      budget = std::min<int64_t>(budget, policy.prefill_idle_budget_tokens);
    policy.prefill_budget_tokens = budget / align * align;
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
