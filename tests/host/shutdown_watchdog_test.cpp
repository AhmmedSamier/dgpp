#include <chrono>
#include <csignal>
#include <stdexcept>
#include <sys/wait.h>

#include "common/test.hpp"
#include "serve/shutdown_watchdog.hpp"

namespace {

int wait_child(pid_t child) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  int status = 0;
  while (::waitpid(child, &status, WNOHANG) == 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      ::kill(child, SIGKILL);
      ::waitpid(child, &status, 0);
      throw std::runtime_error("shutdown watchdog did not exit a stuck process");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!WIFEXITED(status)) throw std::runtime_error("child exited by signal");
  return WEXITSTATUS(status);
}

DGPP_TEST(shutdown_watchdog_bounds_stuck_shutdown_even_with_full_log_pipe) {
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
      std::atomic<bool> requested{false};
      dgpp::serve::ShutdownWatchdog watchdog(requested, std::chrono::milliseconds(40));
      requested = true;
      for (;;) ::pause();  // engine/HTTP threads never return to their loops
    }
    ::close(pipefd[1]);
    const int rc = wait_child(child);
    ::close(pipefd[0]);
    if (rc != 2) throw std::runtime_error("expected forced shutdown status 2");
  }
}

DGPP_TEST(shutdown_watchdog_does_not_time_out_serving_or_a_clean_stop) {
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("fork");
  if (child == 0) {
    std::atomic<bool> requested{false};
    {
      dgpp::serve::ShutdownWatchdog watchdog(requested, std::chrono::milliseconds(40));
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      requested = true;
    }
    std::_Exit(0);
  }
  if (wait_child(child) != 0) throw std::runtime_error("clean shutdown was forced");
}

}  // namespace

int main() { return dgpp::test::run_all(); }
