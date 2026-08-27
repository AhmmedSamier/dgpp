// micro_mem_bw: characterize achievable DRAM bandwidth patterns on GB10.
// Patterns approximate engine load profiles: pure read (weight stream),
// write, d2d copy. Machine-parseable line: PATTERN <name> BEST <GB/s>
#include <cuda_runtime.h>
#include <getopt.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "common/log.hpp"

namespace {

constexpr int kDefaultMiB = 1024;
constexpr int kReps = 5;

__global__ void reduce_read_kernel(const float4* __restrict__ src, size_t n4,
                                   float* __restrict__ out) {
  float acc = 0.f;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1.234e30f) *out = acc;  // never taken; keeps reads alive
}

__global__ void fill_kernel(float4* __restrict__ dst, size_t n4, float val) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride)
    dst[i] = make_float4(val, val, val, val);
}

__global__ void copy_kernel(const float4* __restrict__ src,
                            float4* __restrict__ dst, size_t n4) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride)
    dst[i] = src[i];
}

// Times `launch(stream)` executed `iters` times after `warmups`; returns ms/iter.
template <typename F>
double time_avg_ms(F&& launch, cudaStream_t stream, int warmups, int iters) {
  cudaEvent_t beg{}, end{};
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  for (int i = 0; i < warmups; ++i) launch(stream);
  cudaEventRecord(beg, stream);
  for (int i = 0; i < iters; ++i) launch(stream);
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms_total = 0.f;
  cudaEventElapsedTime(&ms_total, beg, end);
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return static_cast<double>(ms_total) / iters;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  int mib = kDefaultMiB;
  int iters = 20;
  int opt{};
  while ((opt = getopt(argc, argv, "s:i:h")) != -1) {
    switch (opt) {
      case 's': mib = atoi(optarg); break;
      case 'i': iters = atoi(optarg); break;
      default:
        std::printf("usage: micro_mem_bw [-s size_MiB=%d] [-i iters=%d]\n",
                    kDefaultMiB, iters);
        return 2;
    }
  }

  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev <= 0) {
    DGPP_LOG_ERROR("no CUDA device");
    return 1;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  DGPP_LOG_INFO("device={} sm={}", prop.name, prop.multiProcessorCount);

  const size_t bytes = static_cast<size_t>(mib) << 20;
  const size_t n4 = bytes / sizeof(float4);
  float *a = nullptr, *b = nullptr;
  if (cudaMalloc(&a, bytes) != cudaSuccess ||
      cudaMalloc(&b, bytes) != cudaSuccess) {
    DGPP_LOG_ERROR("cudaMalloc {} MiB failed", mib);
    return 1;
  }
  cudaStream_t s{};
  cudaStreamCreate(&s);

  // Fill so reads touch real data (avoid zero-page/save-bandwidth artifacts).
  fill_kernel<<<prop.multiProcessorCount * 8, 256, 0, s>>>(
      reinterpret_cast<float4*>(a), n4, 1.0f);
  fill_kernel<<<prop.multiProcessorCount * 8, 256, 0, s>>>(
      reinterpret_cast<float4*>(b), n4, 2.0f);
  cudaStreamSynchronize(s);

  const int blocks = prop.multiProcessorCount * 12;
  const int threads = 256;

  const auto read_fn = [&](cudaStream_t st) {
    reduce_read_kernel<<<blocks, threads, 0, st>>>(reinterpret_cast<float4*>(a),
                                                   n4, b);
  };
  const auto write_fn = [&](cudaStream_t st) {
    fill_kernel<<<blocks, threads, 0, st>>>(reinterpret_cast<float4*>(a), n4,
                                            3.f);
  };
  const auto copy_fn = [&](cudaStream_t st) {
    copy_kernel<<<blocks, threads, 0, st>>>(reinterpret_cast<float4*>(a),
                                            reinterpret_cast<float4*>(b), n4);
  };

  struct Bench {
    const char* name;
    double traffic_bytes_per_iter;
    std::function<void(cudaStream_t)> fn;
  };
  std::vector<Bench> benches{
      {"read", static_cast<double>(bytes), read_fn},
      {"write", static_cast<double>(bytes), write_fn},
      {"copy(R+W)", 2.0 * static_cast<double>(bytes), copy_fn},
  };

  DGPP_LOG_INFO("working set: {} MiB", mib);
  for (auto& t : benches) {
    double best = 0.0;
    for (int rep = 0; rep < kReps; ++rep) {
      double ms = time_avg_ms(t.fn, s, /*warmups=*/2, iters);
      double secs = ms / 1000.0;
      double gbps =
          static_cast<double>(t.traffic_bytes_per_iter) / secs / 1e9;
      best = std::max(best, gbps);
    }
    DGPP_LOG_INFO("PATTERN {} BEST {:.1f} GB/s", t.name, best);
  }

  cudaFree(a);
  cudaFree(b);
  cudaStreamDestroy(s);
  DGPP_LOG_INFO("done");
  return 0;
}
