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
#include <vector>

namespace dgpp {

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
