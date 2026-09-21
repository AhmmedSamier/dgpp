#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#include "kernels/glm_spec.hpp"
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main() {
  constexpr size_t bytes = 1024;
  char *live, *draft, *chain;
  check(cudaMalloc(&live, 16 * bytes));
  check(cudaMalloc(&draft, 16 * bytes));
  check(cudaMalloc(&chain, 16 * bytes));
  check(cudaMemset(live, 1, 16 * bytes));
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  cudaEvent_t start, end;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  puts("round,slots,passes,us_per_replay");
  for (int round = 0; round < 10; ++round)
    for (int passes : {1, 3})
      for (int order = 0; order < 2; ++order) {
        int slots = ((round + order) % 2 == 0) ? 2 : 16;
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < slots; ++i)
          dgpp::glm_device_copy(draft + i * bytes, live + i * bytes, bytes, stream);
        if (passes == 3) {
          for (int i = 0; i < slots; ++i)
            dgpp::glm_device_copy(chain + i * bytes, live + i * bytes, bytes, stream);
          for (int i = 0; i < slots; ++i)
            dgpp::glm_device_copy(live + i * bytes, chain + i * bytes, bytes, stream);
        }
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, 0));
        for (int i = 0; i < 100; ++i) check(cudaGraphLaunch(exec, stream));
        check(cudaEventRecord(start, stream));
        for (int i = 0; i < 1000; ++i) check(cudaGraphLaunch(exec, stream));
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float ms;
        check(cudaEventElapsedTime(&ms, start, end));
        printf("%d,%d,%d,%.6f\n", round, slots, passes, ms);
        check(cudaGraphExecDestroy(exec));
        check(cudaGraphDestroy(graph));
      }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));
  check(cudaStreamDestroy(stream));
  check(cudaFree(live));
  check(cudaFree(draft));
  check(cudaFree(chain));
}
