#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <stdexcept>
#include <string>

#include "common/process_memory.hpp"
#include "common/test.hpp"

namespace {

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// Lowering a hard limit cannot be undone without privilege. Isolate the
// experiment so neither the test runner nor later tests inherit the limit.
void check_limit_in_child(rlim_t hard_cap) {
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    try {
      rlimit original{};
      require(getrlimit(RLIMIT_MEMLOCK, &original) == 0, "getrlimit failed");
      const rlimit low{0, std::min(original.rlim_max, hard_cap)};
      require(setrlimit(RLIMIT_MEMLOCK, &low) == 0, "setting child limit failed");
      setenv("DGPP_MLOCK", "off", 1);
      const size_t locked = dgpp::process_memory_snapshot().locked;
      require(dgpp::memlock_status().find("RLIMIT_MEMLOCK soft 0 bytes") != std::string::npos,
              "diagnostic must report the actual inherited soft limit");
      std::string error;
      require(dgpp::raise_memlock_soft_limit(&error), error);
      rlimit after{};
      require(getrlimit(RLIMIT_MEMLOCK, &after) == 0, "reading raised limit failed");
      require(after.rlim_cur == low.rlim_max, "soft limit was not raised to hard limit");
      require(after.rlim_max == low.rlim_max, "hard limit changed");
      require(dgpp::process_memory_snapshot().locked == locked, "preparation locked memory");
      require(dgpp::raise_memlock_soft_limit(&error), "repeated preparation failed");
      const std::string status = dgpp::memlock_status();
      const std::string limit = std::to_string(low.rlim_max) + " bytes";
      require(status.find("soft " + limit + ", hard " + limit) != std::string::npos,
              "diagnostic limits disagree with getrlimit");
      require(status.find(", VmPin ") != std::string::npos &&
                  status.find("unavailable") == std::string::npos,
              "diagnostic must include the process's pinned-page count");
      _exit(0);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "memlock child: %s\n", e.what());
      _exit(1);
    }
  }
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "memlock child failed");
}

}  // namespace

DGPP_TEST(process_memory_raises_soft_limit_without_locking_or_raising_hard_limit) {
  check_limit_in_child(8 * 1024 * 1024);
}

DGPP_TEST(process_memory_respects_zero_hard_limit) {
  check_limit_in_child(0);
}
