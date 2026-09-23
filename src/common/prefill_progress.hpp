#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace dgpp {

// The engine publishes completed chunk boundaries here. HTTP readers never
// inspect mutable model or scheduler state. A separate atomic epoch lets the
// serving watchdog observe forward progress without taking the metrics lock.
class PrefillMonitor {
 public:
  struct Request {
    std::string id;
    int slot = -1;
    int64_t total = 0;
    int64_t cached = 0;
    int64_t processed = 0;  // includes the attached prefix
    std::chrono::steady_clock::time_point started;
  };

  void begin(int slot, const std::string& id, int64_t total, int64_t cached = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(requests_, [slot](const Request& r) { return r.slot == slot; });
    requests_.push_back({id, slot, total, cached, cached, std::chrono::steady_clock::now()});
  }

  void update(int slot, int64_t position) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : requests_) {
      if (r.slot != slot) continue;
      const int64_t next = std::clamp(position, r.processed, r.total);
      if (next > r.processed) {
        r.processed = next;
        progress_epoch_.fetch_add(1, std::memory_order_release);
      }
    }
  }

  uint64_t progress_epoch() const {
    return progress_epoch_.load(std::memory_order_acquire);
  }

  void finish(int slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(requests_, [slot](const Request& r) { return r.slot == slot; });
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    requests_.clear();
  }

  std::vector<Request> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }

 private:
  std::atomic<uint64_t> progress_epoch_{0};
  mutable std::mutex mutex_;
  std::vector<Request> requests_;
};

// Shared ownership allows diagnostic models to outlive an engine adapter.
class PrefillReporting {
 public:
  void set_prefill_monitor(std::shared_ptr<PrefillMonitor> monitor) {
    prefill_monitor_ = std::move(monitor);
  }

 protected:
  void report_prefill_progress(int slot, int64_t position) {
    if (prefill_monitor_) prefill_monitor_->update(slot, position);
  }

 private:
  std::shared_ptr<PrefillMonitor> prefill_monitor_;
};

}  // namespace dgpp
