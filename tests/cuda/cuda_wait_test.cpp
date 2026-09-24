#include <atomic>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/cuda_wait.hpp"
#include "common/test.hpp"

DGPP_TEST(cuda_wait_deadline_covers_unfinished_stream_and_allows_completion) {
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::atomic<bool> release{false};
  DGPP_CUDA_OK(cudaLaunchHostFunc(stream, [](void* data) {
    auto& flag = *static_cast<std::atomic<bool>*>(data);
    while (!flag.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }, &release));
  const auto start = std::chrono::steady_clock::now();
  const auto status = dgpp::wait_cuda_stream_until(stream, start + std::chrono::milliseconds(30));
  release = true;
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const auto ready = dgpp::wait_cuda_stream_until(stream, std::chrono::steady_clock::now());
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  if (status != cudaErrorNotReady || ready != cudaSuccess)
    throw std::runtime_error("stream deadline/completion not reported correctly");
  if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2))
    throw std::runtime_error("stream wait exceeded its bounded budget");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
  return dgpp::test::run_all();
}
