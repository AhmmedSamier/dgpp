#pragma once

#include <chrono>
#include <thread>
#include <cuda_runtime.h>

namespace dgpp {

// Unlike cudaStreamSynchronize, a kernel that never finishes cannot keep
// this wait alive indefinitely. A driver call that itself hangs still needs
// process supervision (the serving shutdown watchdog does not call CUDA).
inline cudaError_t wait_cuda_stream_until(
    cudaStream_t stream, std::chrono::steady_clock::time_point deadline) {
  const auto spin_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
  for (;;) {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status != cudaErrorNotReady) return status;
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return cudaErrorNotReady;
    if (now >= spin_until)
      std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

}  // namespace dgpp
