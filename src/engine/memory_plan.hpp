#pragma once
// The memory plan (2026-09-06, extracted for the second family 2026-09-09):
// every byte a model's constructor (and its layer objects) will allocate
// for a shape, itemized, computed from the same formulas BEFORE anything is
// allocated — the serving app's pre-flight check and `--memory-plan`. The
// CUDA context, the bus and the prefix arena are the caller's to add.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "common/process_memory.hpp"
#include <vector>

namespace dgpp {

// The residual ledger (2026-09-13): one INFO line per boot phase with the
// process's resident split and the device's free memory, so the memory the
// plan does not itemize can be attributed to the phase that takes it
// (dgpp_serve and the family constructors call it; grep "memory ledger").
inline void log_memory_ledger(const std::string& phase) {
  const ProcessMemory m = process_memory_snapshot();
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) free_bytes = 0;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  DGPP_LOG_INFO("memory ledger: {} — rss {:.2f} GiB (anon {:.2f}, shmem {:.2f}, file {:.2f}; locked {:.2f}); "
                "device free {:.2f} GiB, node available {:.2f} GiB",
                phase, m.rss / kGiB, m.anon / kGiB, m.shmem / kGiB, m.file / kGiB, m.locked / kGiB,
                free_bytes / kGiB, m.node_available / kGiB);
}

struct MemoryPlan {
  struct Item {
    std::string name;
    size_t device = 0;
    size_t pinned = 0;
  };
  std::vector<Item> items;
  int64_t context_tokens = 0;  // the pool's token capacity (max_context())
  void add(std::string name, size_t device, size_t pinned = 0) {
    items.push_back(Item{std::move(name), device, pinned});
  }
  size_t device_bytes() const {
    size_t t = 0;
    for (const Item& i : items) t += i.device;
    return t;
  }
  size_t pinned_bytes() const {
    size_t t = 0;
    for (const Item& i : items) t += i.pinned;
    return t;
  }
  size_t total_bytes() const { return device_bytes() + pinned_bytes(); }
};

}  // namespace dgpp
