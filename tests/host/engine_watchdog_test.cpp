#include <chrono>
#include <csignal>
#include <stdexcept>
#include <sys/wait.h>

#include "common/test.hpp"
#include "serve/engine_watchdog.hpp"

namespace {
using namespace std::chrono_literals;

int wait_child(pid_t child) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  int status = 0;
  for (;;) {
    const auto result = ::waitpid(child, &status, WNOHANG);
    if (result == child) break;
    if (result < 0) throw std::runtime_error("waitpid");
    if (std::chrono::steady_clock::now() >= deadline) {
      ::kill(child, SIGKILL);
      ::waitpid(child, &status, 0);
      throw std::runtime_error("engine watchdog did not exit a stuck process");
    }
    std::this_thread::sleep_for(5ms);
  }
  if (!WIFEXITED(status)) throw std::runtime_error("child exited by signal");
  return WEXITSTATUS(status);
}

DGPP_TEST(engine_watchdog_exits_without_a_signal_even_with_a_full_log_pipe) {
  for (bool full_pipe : {false, true}) {
    int pipefd[2];
    if (::pipe2(pipefd, O_NONBLOCK) != 0) throw std::runtime_error("pipe");
    if (full_pipe) {
      char bytes[4096]{};
      while (::write(pipefd[1], bytes, sizeof(bytes)) > 0) {}
      ::fcntl(pipefd[1], F_SETFL, 0);
    }
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("fork");
    if (child == 0) {
      ::dup2(pipefd[1], STDERR_FILENO);
      ::close(pipefd[0]);
      ::close(pipefd[1]);
      dgpp::PrefillMonitor prefill;
      dgpp::serve::EngineWatchdog watchdog(prefill, 100ms);
      const auto work = watchdog.work();
      for (;;) ::pause();  // no stop request, no return from the engine call
    }
    ::close(pipefd[1]);
    const int result = wait_child(child);
    ::close(pipefd[0]);
    if (result != 2) throw std::runtime_error("expected engine failure status 2");
  }
}

DGPP_TEST(engine_watchdog_allows_idle_and_repeated_short_passes) {
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("fork");
  if (child == 0) {
    dgpp::PrefillMonitor prefill;
    {
      dgpp::serve::EngineWatchdog watchdog(prefill, 250ms);
      std::this_thread::sleep_for(400ms);
      for (int i = 0; i < 50; ++i) {
        const auto work = watchdog.work();
        std::this_thread::sleep_for(10ms);
      }
      std::this_thread::sleep_for(400ms);
    }
    std::_Exit(0);
  }
  if (wait_child(child) != 0) throw std::runtime_error("healthy engine timed out");
}

DGPP_TEST(engine_watchdog_allows_long_prefill_only_while_tokens_advance) {
  for (bool advance : {false, true}) {
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("fork");
    if (child == 0) {
      dgpp::PrefillMonitor prefill;
      prefill.begin(0, "long-prompt", 1000);
      {
        dgpp::serve::EngineWatchdog watchdog(prefill, 250ms);
        const auto work = watchdog.work();
        for (int i = 1; i <= 70; ++i) {
          // Publishing the same position is not forward progress.
          prefill.update(0, advance ? i : 0);
          std::this_thread::sleep_for(10ms);
        }
      }
      std::_Exit(0);
    }
    if (wait_child(child) != (advance ? 0 : 2))
      throw std::runtime_error("prefill progress did not govern the deadline");
  }
}

DGPP_TEST(engine_watchdog_expires_after_prefill_progress_stops) {
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("fork");
  if (child == 0) {
    dgpp::PrefillMonitor prefill;
    prefill.begin(0, "long-prompt", 1000);
    dgpp::serve::EngineWatchdog watchdog(prefill, 250ms);
    const auto work = watchdog.work();
    for (int i = 1; i <= 50; ++i) {
      prefill.update(0, i);
      std::this_thread::sleep_for(10ms);
    }
    for (;;) ::pause();
  }
  if (wait_child(child) != 2) throw std::runtime_error("stalled prefill survived");
}

DGPP_TEST(engine_watchdog_allows_collective_progress_inside_one_prefill_chunk) {
  for (bool advance : {false, true}) {
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("fork");
    if (child == 0) {
      dgpp::PrefillMonitor prefill;
      prefill.begin(0, "slow-chunk", 2048);
      std::atomic<uint64_t> collective_progress{0};
      {
        dgpp::serve::EngineWatchdog watchdog(prefill, 250ms, &collective_progress);
        const auto work = watchdog.work();
        for (int i = 0; i < 70; ++i) {
          if (advance) collective_progress.fetch_add(1, std::memory_order_release);
          std::this_thread::sleep_for(10ms);
        }
        // Finishing a layer's collective is liveness, not completed tokens.
        if (prefill.progress_epoch() != 0 || prefill.snapshot()[0].processed != 0)
          std::_Exit(1);
      }
      std::_Exit(0);
    }
    if (wait_child(child) != (advance ? 0 : 2))
      throw std::runtime_error("collective completion did not govern the deadline");
  }
}

DGPP_TEST(engine_watchdog_expires_when_collectives_stop_inside_a_chunk) {
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("fork");
  if (child == 0) {
    dgpp::PrefillMonitor prefill;
    prefill.begin(0, "stalled-chunk", 2048);
    std::atomic<uint64_t> collective_progress{0};
    dgpp::serve::EngineWatchdog watchdog(prefill, 250ms, &collective_progress);
    const auto work = watchdog.work();
    for (int i = 0; i < 50; ++i) {
      collective_progress.fetch_add(1, std::memory_order_release);
      std::this_thread::sleep_for(10ms);
    }
    for (;;) ::pause();
  }
  if (wait_child(child) != 2) throw std::runtime_error("stalled collective progress survived");
}

}  // namespace

int main() { return dgpp::test::run_all(); }
