#pragma once
// The decode-step budget instrument (2026-09-01): env-gated
// (DGPP_STEP_TIMING=1) or app-flag-gated wall accumulators at the seams
// the 0.36s/token diagnosis cares about. Host-only, no device work; a
// disabled tick costs two branches, an enabled one a clock pair (~40ns)
// against millisecond-scale budgets.
//
// Nesting: kMoeSync ticks nest INSIDE kMoe (the host-orchestrated
// enqueue's router round-trip is part of its wall) — the report
// subtracts it; moe_launch = moe - moe_sync is the segmentation+launch
// tax on top of the sync.
//
// The accumulators are per-thread (the engine loop owns the ticks); the
// app that drives the steps prints the report from the same thread.
// reset() clears the budget — measurement apps call it between phases
// (e.g. after prefill, so the report is decode-steps only).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace dgpp::step_timing {

enum Slot {
  kStep = 0,     // session_step wall: the whole token
  kMoe,          // moe enqueue wall (either path), nested: kMoeSync
  kMoeSync,      // the host path's router round-trip (nested in kMoe)
  kFoldDrain,    // the pre-collective cudaStreamSynchronize per boundary
  kFold,         // the boundary reduce (collective submit->wake)
  kPick,         // the distributed greedy pick (gather+broadcast)
  kFinalSync,    // the step's final stream sync (logits to host)
  kCount
};

inline const char* slot_name(Slot s) {
  switch (s) {
    case kStep: return "step";
    case kMoe: return "moe";
    case kMoeSync: return "moe_sync";
    case kFoldDrain: return "fold_drain";
    case kFold: return "fold";
    case kPick: return "pick";
    case kFinalSync: return "final_sync";
    default: return "?";
  }
}

inline bool& enabled_storage() {
  // Env sets the default; the apps' --step-timing flag overrides it.
  static bool on = [] {
    const char* e = std::getenv("DGPP_STEP_TIMING");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
inline bool enabled() { return enabled_storage(); }
inline void set_enabled(bool on_flag) { enabled_storage() = on_flag; }

struct Acc {
  double us = 0.0;
  uint64_t n = 0;
};

inline Acc& acc(Slot s) {
  thread_local Acc a[kCount] = {};
  return a[s];
}

class Scope {
 public:
  explicit Scope(Slot s) : s_(s) {
    if (enabled()) t0_ = Clock::now();
  }
  ~Scope() {
    if (enabled()) {
      auto& a = acc(s_);
      a.us += std::chrono::duration<double, std::micro>(Clock::now() - t0_)
                  .count();
      ++a.n;
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  Slot s_;
  Clock::time_point t0_{};
};

inline void reset() {
  for (int i = 0; i < kCount; ++i) acc(static_cast<Slot>(i)) = Acc{};
}

// The budget: per-slot totals plus the derived splits the diagnosis
// reads off (moe_launch = moe - moe_sync; the step residual is whatever
// the named seams do not cover — attention/DSA/KDA kernel time and the
// host loop between launches).
inline void report(const char* who) {
  std::printf("[step-timing] %s\n", who);
  const uint64_t steps = acc(kStep).n;
  if (steps == 0) {
    std::printf("[step-timing] (no steps recorded)\n");
    return;
  }
  const double step = acc(kStep).us;
  const double moe = acc(kMoe).us;
  const double moe_sync = acc(kMoeSync).us;
  const double drain = acc(kFoldDrain).us;
  const double fold = acc(kFold).us;
  const double pick = acc(kPick).us;
  const double final_sync = acc(kFinalSync).us;
  const double per = 1.0 / static_cast<double>(steps);
  const auto pct = [&](double v) { return step > 0 ? 100.0 * v / step : 0.0; };
  std::printf(
      "[step-timing] steps=%llu  step=%.0fus/token\n"
      "[step-timing]   moe        %8.0fus/token (%4.0f%%)  enq n=%llu\n"
      "[step-timing]   moe_sync   %8.0fus/token (%4.0f%%)  sync n=%llu\n"
      "[step-timing]   moe_launch %8.0fus/token (%4.0f%%)\n"
      "[step-timing]   folds      %8.0fus/token (%4.0f%%)  n=%llu (drain "
      "%.0fus + coll %.0fus)\n"
      "[step-timing]   pick       %8.0fus/token (%4.0f%%)\n"
      "[step-timing]   final_sync %8.0fus/token (%4.0f%%)\n"
      "[step-timing]   residual   %8.0fus/token (kernels + host loop)\n",
      static_cast<unsigned long long>(steps), step * per, moe * per, pct(moe),
      static_cast<unsigned long long>(acc(kMoe).n), moe_sync * per,
      pct(moe_sync), static_cast<unsigned long long>(acc(kMoeSync).n),
      (moe - moe_sync) * per, pct(moe - moe_sync), (drain + fold) * per,
      pct(drain + fold), static_cast<unsigned long long>(acc(kFold).n),
      drain * per, fold * per, pick * per, pct(pick), final_sync * per,
      pct(final_sync),
      (step - moe - drain - fold - pick - final_sync) * per);
}

}  // namespace dgpp::step_timing
