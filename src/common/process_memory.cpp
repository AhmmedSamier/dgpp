#include "common/process_memory.hpp"

#include <sys/mman.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>

namespace dgpp {

namespace {

std::string limit_text(rlim_t v) {
  return v == RLIM_INFINITY ? "unlimited" : std::to_string(v) + " bytes";
}

// The soft limit is ours to raise up to the hard one without privilege;
// ssh-spawned processes on the lab fabric arrive with a finite soft limit
// (2026-09-02: two of four ranks) that the hard limit did not require.
void raise_memlock_soft_limit() {
  rlimit lim{};
  if (getrlimit(RLIMIT_MEMLOCK, &lim) != 0) return;
  if (lim.rlim_cur == lim.rlim_max) return;
  lim.rlim_cur = lim.rlim_max;
  setrlimit(RLIMIT_MEMLOCK, &lim);  // best effort; mlockall reports the truth
}

}  // namespace

bool lock_process_memory(std::string* error, size_t* locked_bytes) {
  raise_memlock_soft_limit();
  if (mlockall(MCL_CURRENT) != 0) {
    const int err = errno;
    rlimit lim{};
    getrlimit(RLIMIT_MEMLOCK, &lim);
    if (error)
      *error = std::string("mlockall(MCL_CURRENT) failed: ") +
               std::strerror(err) + " (RLIMIT_MEMLOCK soft " +
               limit_text(lim.rlim_cur) + ", hard " + limit_text(lim.rlim_max) +
               " — raise it with ulimit -l unlimited / LimitMEMLOCK=infinity)";
    return false;
  }
  if (locked_bytes) {
    *locked_bytes = 0;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      unsigned long long kb = 0;
      if (std::sscanf(line.c_str(), "VmLck: %llu kB", &kb) == 1) {
        *locked_bytes = static_cast<size_t>(kb) * 1024;
        break;
      }
    }
  }
  return true;
}

ProcessMemory process_memory_snapshot() {
  ProcessMemory m;
  std::ifstream in("/proc/self/status");
  std::string line;
  // Line by line: the file's first fields ("Name:", "State:") are not
  // numeric, and a token-wise read would stop there.
  while (std::getline(in, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    const uint64_t kib = std::strtoull(line.c_str() + colon + 1, nullptr, 10);
    if (key == "VmRSS") m.rss = static_cast<size_t>(kib) * 1024;
    else if (key == "RssAnon") m.anon = static_cast<size_t>(kib) * 1024;
    else if (key == "RssShmem") m.shmem = static_cast<size_t>(kib) * 1024;
    else if (key == "RssFile") m.file = static_cast<size_t>(kib) * 1024;
    else if (key == "VmLck") m.locked = static_cast<size_t>(kib) * 1024;
  }
  m.node_available = host_memory_available_bytes();
  return m;
}

size_t host_memory_available_bytes() {
  std::ifstream in("/proc/meminfo");
  std::string key;
  uint64_t kib = 0;
  std::string unit;
  while (in >> key >> kib >> unit)
    if (key == "MemAvailable:") return static_cast<size_t>(kib) * 1024;
  return 0;
}

}  // namespace dgpp
