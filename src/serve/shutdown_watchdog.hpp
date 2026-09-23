#pragma once

#include <atomic>
#include <chrono>
#include <thread>

#include "serve/emergency_exit.hpp"

namespace dgpp::serve {

// Independent of the engine, HTTP loop, CUDA, and logging locks. The normal
// shutdown still drains the journal and answers clients; a stuck drain or
// destructor must not make SIGTERM require a subsequent SIGKILL.
class ShutdownWatchdog {
 public:
  explicit ShutdownWatchdog(
      const std::atomic<bool>& requested,
      std::chrono::milliseconds grace = std::chrono::seconds(30))
      : thread_([&requested, grace](std::stop_token stop) {
          while (!stop.stop_requested() && !requested.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          const auto deadline = std::chrono::steady_clock::now() + grace;
          while (!stop.stop_requested()) {
            if (std::chrono::steady_clock::now() >= deadline) {
              emergency_exit(
                  "ERROR serve: shutdown deadline exceeded; exiting with status 2 without engine teardown\n");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        }) {}

 private:
  std::jthread thread_;
};

}  // namespace dgpp::serve
