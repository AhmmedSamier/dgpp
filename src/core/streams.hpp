#pragma once
#include <cuda_runtime.h>

#include <array>
#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {

// Fixed-purpose stream pool: compute / copy / collective.
class StreamPool {
 public:
  enum Id : int { Compute = 0, Copy = 1, Collective = 2 };

  void init() {
    if (initialized_) throw std::logic_error("stream pool is already initialized");
    initialized_ = true;
    for (auto& s : streams_) DGPP_CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    for (auto& e : events_) DGPP_CUDA_OK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }

  ~StreamPool() {
    for (auto& s : streams_) if (s) cudaStreamDestroy(s);
    for (auto& e : events_) if (e) cudaEventDestroy(e);
  }

  StreamPool() = default;
  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;

  cudaStream_t stream(int id) const {
    if (!initialized_) throw std::logic_error("stream pool is not initialized");
    if (id < 0 || static_cast<size_t>(id) >= streams_.size())
      throw std::out_of_range("stream id");
    return streams_[static_cast<size_t>(id)];
  }
  cudaEvent_t event(size_t slot) const {
    if (!initialized_) throw std::logic_error("stream pool is not initialized");
    return events_[slot % events_.size()];
  }

  // Record event slot on stream `on`; make stream `waiter` wait on it.
  void fence(int on, size_t slot, int waiter) {
    DGPP_CUDA_OK(cudaEventRecord(event(slot), stream(on)));
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream(waiter), event(slot), 0));
  }

  void sync_all() {
    for (auto& s : streams_) DGPP_CUDA_OK(cudaStreamSynchronize(s));
  }

 private:
  std::array<cudaStream_t, 3> streams_{};
  static constexpr size_t kEventSlots = 64;
  std::array<cudaEvent_t, kEventSlots> events_{};
  bool initialized_ = false;
};

}  // namespace dgpp
