// The periodic throughput line (2026-09-06): what an operator watching a
// rank's log sees of the world's aggregate behavior without a line per
// tick. Every pass on rank 0 and every tick on a peer hands the loop's
// scheduler meters to a ThroughputLog; at most once per interval (default
// 10 s, glm_serve --stats-interval-s) it writes ONE INFO line with the
// interval's deltas: prompts prefilled and prompt tokens per second, the
// prefill's share of the wall clock, decode steps and generated tokens per
// second, ms per step and tokens per request-step (MTP's acceptance in
// service clothing), the live and queued counts, the pool's occupancy and
// the prefix cache's hits — and on rank 0 the requests admitted, shed and
// cancelled. Since 2026-09-06 the line leads with decode — tokens per
// second, ms per token, ms per step, MTP's yield — the numbers an operator
// watches, in `|`-separated groups. An idle world writes one closing line (zeros) and then
// nothing until work returns, so a quiet log stays quiet.
//
// The per-tick lines (a line per generated token, the bus's per-window
// timeline, the prefix cache's per-decision line) moved to DEBUG the same
// day; DGPP_LOG_LEVEL=debug restores them, and the scripts that parse
// them (serve_pace.py, fabric_xrank.py) say so.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "sched/scheduler.hpp"

namespace dgpp::serve {

// The service-side counts rank 0 adds to its line; a peer passes none.
struct ServiceCounts {
  uint64_t requests_total = 0;
  uint64_t requests_shed = 0;
  uint64_t requests_cancelled = 0;
};

class ThroughputLog {
 public:
  using Clock = std::chrono::steady_clock;
  using Meters = dgpp::sched::Scheduler::Meters;

  // `interval_s` <= 0 disables the line entirely; `mtp` adds the MTP
  // group (tokens per request-step and the share of drafts accepted).
  ThroughputLog(double interval_s, int rank, bool mtp = false);

  bool enabled() const { return interval_s_ > 0.0; }

  // Call after every pass or tick, idle passes included: cheap when no
  // interval has elapsed (a clock read). Returns the line it logged, or
  // an empty string when it logged nothing.
  std::string observe(const Meters& m, const ServiceCounts* svc,
                      Clock::time_point now = Clock::now());

  // The line for one interval, from two snapshots `seconds` apart.
  static std::string format(int rank, double seconds, const Meters& prev,
                            const Meters& cur, const ServiceCounts* prev_svc,
                            const ServiceCounts* cur_svc, bool mtp = false);

 private:
  double interval_s_ = 0.0;
  int rank_ = 0;
  bool mtp_ = false;
  bool primed_ = false;
  bool was_busy_ = false;  // the previous interval had work or live requests
  Clock::time_point last_{};
  Meters prev_{};
  ServiceCounts prev_svc_{};
};

}  // namespace dgpp::serve
