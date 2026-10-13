#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace dgpp::sched {

inline constexpr int kDefaultDecodeGapTargetMs = 1000;
inline constexpr int kDefaultFirstTokenTargetMs = 3000;
inline constexpr int kMaxLatencyTargetMs = 3600000;
inline constexpr int64_t kInitialPrefillBudgetTokens = 256;

struct PrefillLatencyTuning {
  int headroom_pct = 15;  // reserve this fraction of the time target
  int growth_factor = 2;  // maximum upward probe relative to observed work
  int recovery_pct = 25;  // weight of a cheaper observation in the cost estimate
  int initial_tokens = kInitialPrefillBudgetTokens;
  bool valid() const {
    return headroom_pct >= 0 && headroom_pct <= 50 && growth_factor >= 2 && growth_factor <= 4 &&
           recovery_pct >= 1 && recovery_pct <= 100 && initial_tokens >= 1 && initial_tokens <= (1 << 30);
  }
  bool operator==(const PrefillLatencyTuning&) const = default;
};

// A deployment learns its own cost; no checkpoint or quantization lookup.
// Only the leader consults this feedback. Peers replay the selected token
// limit, never their own clocks. The caller supplies completed pass durations
// including decode, snapshots and other work that delays the next yield.
class PrefillLatencyController {
 public:
  void observe(int decoders, int64_t tokens, int64_t offered, double ms,
               const PrefillLatencyTuning& tuning = {}) {
    // Final tails and snapshot cuts are not representative full quanta.
    if (tokens <= 0 || tokens < offered / 2 || !(ms > 0) || !std::isfinite(ms)) return;
    auto& sample = samples_[bucket(decoders)];
    const double rate = ms / static_cast<double>(tokens);
    // React immediately to a slowdown, recover gradually as work gets cheaper.
    const double weight = tuning.recovery_pct / 100.0;
    sample.ms_per_token = sample.tokens == 0 || rate > sample.ms_per_token
        ? rate : (1 - weight) * sample.ms_per_token + weight * rate;
    sample.tokens = tokens;
  }

  int64_t limit(int decoders, double target_ms, int64_t initial,
                int64_t capacity, int64_t alignment, const PrefillLatencyTuning& tuning = {}) const {
    const Sample* sample = &samples_[bucket(decoders)];
    const int64_t cap = capacity / alignment * alignment;
    if (!(target_ms > 0)) return cap;
    if (!sample->tokens) {
      // Changing the number of decoders should not discard everything the
      // deployment already learned about its prefill cost. Borrow the most
      // conservative measured rate; the first observation at this load then
      // initializes its own estimate. No estimate is copied back to a peer.
      for (const auto& known : samples_)
        if (known.tokens && (!sample->tokens || known.ms_per_token > sample->ms_per_token)) sample = &known;
    }
    if (!sample->tokens) return std::min(cap, std::max(alignment, initial / alignment * alignment));
    // Leave room for cost variation and arrivals. Bound upward exploration;
    // the first unmeasured workload starts at the shared busy budget.
    const double predicted = std::min<double>(cap, (1 - tuning.headroom_pct / 100.0) * target_ms / sample->ms_per_token);
    const int64_t rows = std::min<int64_t>(static_cast<int64_t>(predicted), sample->tokens * tuning.growth_factor);
    const int64_t bound = std::min(cap, std::max(alignment, rows / alignment * alignment));
    // A small geometric set avoids oscillating among new GEMM shapes (and
    // repeatedly paying their first-use setup). Preserve an aligned capacity
    // that is not a power of two when the estimate can afford it.
    int64_t quantum = alignment;
    while (quantum <= bound / 2) quantum *= 2;
    return bound == cap ? cap : quantum;
  }

 private:
  static size_t bucket(int decoders) {
    return decoders <= 2 ? static_cast<size_t>(std::max(0, decoders))
                        : decoders <= 4 ? 3 : decoders <= 8 ? 4 : 5;
  }
  struct Sample { double ms_per_token = 0; int64_t tokens = 0; };
  std::array<Sample, 6> samples_{};
};

}  // namespace dgpp::sched
