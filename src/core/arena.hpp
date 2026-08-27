#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"

namespace dgpp {

// Residency classes for the staged runtime. DeviceCold exists for diagnostic
// and explicitly prefetched allocations; transparent migration is not allowed
// on a production hot path. Host-pinned memory is GPU-addressable on GB10.
enum class MemClass : int { DeviceHot, DeviceCold, HostPinned };

struct MemStats {
  size_t capacity = 0;
  size_t used = 0;
  size_t high_water = 0;
  uint64_t alloc_calls = 0;
  uint64_t slab_mallocs = 0;

  friend auto operator<=>(const MemStats&, const MemStats&) = default;
};

class ArenaSlab {
 public:
  ArenaSlab() = default;
  void init(MemClass cls, size_t bytes) {
    if (base_ || bytes_ != 0)
      throw std::logic_error("arena slab is already initialized");
    cls_ = cls;
    bytes_ = bytes;
    if (bytes == 0) return;
    if (cls == MemClass::HostPinned) {
      unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
      DGPP_CUDA_OK(cudaHostAlloc(&base_, bytes, flags));
    } else if (cls == MemClass::DeviceCold) {
      DGPP_CUDA_OK(cudaMallocManaged(&base_, bytes));
    } else {
      DGPP_CUDA_OK(cudaMalloc(&base_, bytes));
    }
    ++stats_.slab_mallocs;
    stats_.capacity = bytes;
  }

  ~ArenaSlab() noexcept { release_noexcept(); }

  ArenaSlab(const ArenaSlab&) = delete;
  ArenaSlab& operator=(const ArenaSlab&) = delete;

  void* alloc(size_t bytes, size_t align) {
    if (align == 0 || (align & (align - 1)) != 0)
      throw std::invalid_argument("arena alignment must be a power of two");
    if (!base_)
      throw std::runtime_error("arena slab is not initialized");
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_);
    if (base > std::numeric_limits<uintptr_t>::max() - cursor_ ||
        base + cursor_ > std::numeric_limits<uintptr_t>::max() - (align - 1))
      throw std::overflow_error("arena alignment overflow");
    const uintptr_t current = base + cursor_;
    const uintptr_t aligned = (current + (align - 1)) & ~(align - 1);
    const size_t start = static_cast<size_t>(aligned - base);
    if (start > bytes_ || bytes > bytes_ - start) {
      throw std::runtime_error(
          std::format("arena OOM class={} want={} have={}", static_cast<int>(cls_),
                      bytes, bytes_ - std::min(start, bytes_)));
    }
    cursor_ = start + bytes;
    stats_.used = cursor_;
    if (stats_.used > stats_.high_water) stats_.high_water = stats_.used;
    ++stats_.alloc_calls;
    return reinterpret_cast<void*>(base + start);
  }

  void reset_cursor() { cursor_ = 0; stats_.used = 0; }

  void release() {
    if (base_) {
      if (cls_ == MemClass::HostPinned)
        DGPP_CUDA_OK(cudaFreeHost(base_));
      else
        DGPP_CUDA_OK(cudaFree(base_));  // works for managed + device
    }
    base_ = nullptr;
    cursor_ = bytes_ = 0;
    stats_.capacity = 0;
    stats_.used = 0;
  }

  const MemStats& stats() const { return stats_; }
  MemClass mem_class() const { return cls_; }
  void* base() const { return base_; }
  size_t capacity() const { return bytes_; }
  size_t used() const { return cursor_; }
  size_t free_bytes() const { return bytes_ > cursor_ ? bytes_ - cursor_ : 0; }

 private:
  void release_noexcept() noexcept {
    if (base_) {
      if (cls_ == MemClass::HostPinned)
        (void)cudaFreeHost(base_);
      else
        (void)cudaFree(base_);
    }
    base_ = nullptr;
    cursor_ = bytes_ = 0;
    stats_.capacity = 0;
    stats_.used = 0;
  }

  MemClass cls_ = MemClass::DeviceHot;
  void* base_ = nullptr;
  size_t bytes_ = 0;
  size_t cursor_ = 0;
  MemStats stats_{};
};

// Two-region arena: persistent allocations grow monotonically for the process
// lifetime; scratch region is reset between engine steps so the steady-state
// loop performs zero allocator syscalls (replays land at identical addresses).
class Arena {
 public:
  Arena() = default;
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  struct Config {
    size_t persistent_hot = 0;
    size_t scratch_hot = 0;
    size_t persistent_cold = 0;
    size_t scratch_cold = 0;
    size_t host_pinned = 0;
  };

  void init(const Config& c) {
    if (initialized_) throw std::logic_error("arena is already initialized");
    initialized_ = true;
    auto init_if = [](ArenaSlab& s, MemClass cls, size_t bytes) {
      if (bytes > 0) s.init(cls, bytes);
    };
    init_if(persist_hot_, MemClass::DeviceHot, c.persistent_hot);
    init_if(scratch_hot_, MemClass::DeviceHot, c.scratch_hot);
    init_if(persist_cold_, MemClass::DeviceCold, c.persistent_cold);
    init_if(scratch_cold_, MemClass::DeviceCold, c.scratch_cold);
    init_if(host_pinned_, MemClass::HostPinned, c.host_pinned);
  }

  ~Arena() = default;

  void* alloc_persistent(MemClass cls, size_t bytes, size_t align = 256) {
    switch (cls) {
      case MemClass::DeviceHot: return persist_hot_.alloc(bytes, align);
      case MemClass::DeviceCold: return persist_cold_.alloc(bytes, align);
      case MemClass::HostPinned: return host_pinned_.alloc(bytes, align);
    }
    throw std::runtime_error("bad memclass");
  }

  void* alloc_scratch(MemClass cls, size_t bytes, size_t align = 256) {
    switch (cls) {
      case MemClass::DeviceHot: return scratch_hot_.alloc(bytes, align);
      case MemClass::DeviceCold: return scratch_cold_.alloc(bytes, align);
      case MemClass::HostPinned: throw std::runtime_error("host scratch unsupported");
    }
    throw std::runtime_error("bad memclass");
  }

  template <typename T>
  T* alloc_array_persistent(MemClass cls, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("persistent array byte size overflow");
    return static_cast<T*>(alloc_persistent(cls, sizeof(T) * count, alignof(T) < 256 ? 256 : alignof(T)));
  }

  template <typename T>
  T* alloc_array_scratch(MemClass cls, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("scratch array byte size overflow");
    return static_cast<T*>(alloc_scratch(cls, sizeof(T) * count, alignof(T) < 256 ? 256 : alignof(T)));
  }

  void reset_scratch() {
    scratch_hot_.reset_cursor();
    scratch_cold_.reset_cursor();
  }

  void release_all() {
    persist_hot_.release();
    scratch_hot_.release();
    persist_cold_.release();
    scratch_cold_.release();
    host_pinned_.release();
    initialized_ = false;
  }

  std::vector<std::pair<std::string, const MemStats*>> stats() const {
    return {
        {"persist_hot", &persist_hot_.stats()},
        {"scratch_hot", &scratch_hot_.stats()},
        {"persist_cold", &persist_cold_.stats()},
        {"scratch_cold", &scratch_cold_.stats()},
        {"host_pinned", &host_pinned_.stats()},
    };
  }

 private:
  ArenaSlab persist_hot_, scratch_hot_, persist_cold_, scratch_cold_, host_pinned_;
  bool initialized_ = false;
};

}  // namespace dgpp
