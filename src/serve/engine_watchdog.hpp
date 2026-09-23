#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "common/prefill_progress.hpp"
#include "serve/emergency_exit.hpp"

namespace dgpp::serve {

// Rank 0's final serving backstop. Bus watchdogs cannot cover a model/driver
// wait outside a collective, or a blocked bus progress thread. This thread
// reads only atomics, including progress within an unchunked scheduler pass.
// On failure the journal closes with the process; peers follow its death path.
class EngineWatchdog {
 public:
  class Work {
   public:
    explicit Work(std::atomic<uint64_t>& epoch) : epoch_(epoch) {
      epoch_.fetch_add(1, std::memory_order_release);  // odd: work in progress
    }
    ~Work() { epoch_.fetch_add(1, std::memory_order_release); }  // even: idle
    Work(const Work&) = delete;
    Work& operator=(const Work&) = delete;

   private:
    std::atomic<uint64_t>& epoch_;
  };

  explicit EngineWatchdog(
      const PrefillMonitor& prefill,
      std::chrono::milliseconds timeout = std::chrono::seconds(120))
      : thread_([this, &prefill, timeout](std::stop_token stop) {
          uint64_t previous_work = 0;
          uint64_t previous_prefill = prefill.progress_epoch();
          auto deadline = std::chrono::steady_clock::now() + timeout;
          while (!stop.stop_requested()) {
            const auto now = std::chrono::steady_clock::now();
            const uint64_t work = epoch_.load(std::memory_order_acquire);
            const uint64_t progress = prefill.progress_epoch();
            if ((work & 1) == 0 || work != previous_work || progress != previous_prefill) {
              deadline = now + timeout;
            } else if (now >= deadline) {
              emergency_exit(
                  "ERROR serve: engine progress deadline exceeded; exiting with status 2 without engine teardown\n");
            }
            previous_work = work;
            previous_prefill = progress;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }) {}

  // One engine thread owns these non-overlapping scopes. Include the pass,
  // journal broadcast and metrics/log publication, not just sched.tick().
  [[nodiscard]] Work work() { return Work(epoch_); }

 private:
  static_assert(std::atomic<uint64_t>::is_always_lock_free);
  std::atomic<uint64_t> epoch_{0};
  std::jthread thread_;
};

}  // namespace dgpp::serve
