// micro_mem_bw: characterize achievable DRAM bandwidth patterns on GB10.
// Patterns approximate engine load profiles: pure read (weight stream),
// write, d2d copy. Machine-parseable line: PATTERN <name> BEST <GB/s>
#include <cuda_runtime.h>
#include <getopt.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
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
  // -m: cudaMallocManaged buffers, first-touched by the HOST (the resident
  // loader's pattern: memcpy from the shards, then GPU reads) — measures
  // the placement/translation path the model's weights actually take.
  // -a: managed with cudaMemAdvise preferred-location GPU + accessed-by,
  // then a device-side first touch. -p: cudaMallocHost (pinned).
  int mode = 0;
  int opt{};
  while ((opt = getopt(argc, argv, "s:i:mapf")) != -1) {
    switch (opt) {
      case 's': mib = atoi(optarg); break;
      case 'i': iters = atoi(optarg); break;
      case 'm': mode = 1; break;
      case 'a': mode = 2; break;
      case 'p': mode = 3; break;
      case 'f': mode = 4; break;
      default:
        std::printf("usage: micro_mem_bw [-s size_MiB=%d] [-i iters=%d] [-m|-a|-p]\n",
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
  const auto alloc = [&](float** p) -> bool {
    if (mode == 0) return cudaMalloc(p, bytes) == cudaSuccess;
    if (mode == 3) return cudaMallocHost(p, bytes) == cudaSuccess;
    if (cudaMallocManaged(p, bytes) != cudaSuccess) return false;
    if (mode == 1) {
      std::memset(*p, 0x3C, bytes);  // host first touch
    } else if (mode == 4) {
      std::memset(*p, 0x3C, bytes);  // host first touch, then an explicit
      cudaMemLocation gpu0{};        // prefetch (migration) to the GPU
      gpu0.type = cudaMemLocationTypeDevice;
      gpu0.id = 0;
      if (cudaMemPrefetchAsync(*p, bytes, gpu0, 0, nullptr) != cudaSuccess)
        DGPP_LOG_ERROR("prefetch failed: {}", cudaGetErrorString(cudaGetLastError()));
      cudaDeviceSynchronize();
    } else {
      cudaMemLocation gpu0{};
      gpu0.type = cudaMemLocationTypeDevice;
      gpu0.id = 0;
      cudaMemAdvise(*p, bytes, cudaMemAdviseSetPreferredLocation, gpu0);
      cudaMemAdvise(*p, bytes, cudaMemAdviseSetAccessedBy, gpu0);
    }
    return true;
  };
  if (!alloc(&a) || !alloc(&b)) {
    DGPP_LOG_ERROR("alloc {} MiB failed (mode {})", mib, mode);
    return 1;
  }
  DGPP_LOG_INFO("allocation mode {} ({})", mode,
                mode == 0 ? "cudaMalloc" : mode == 1 ? "managed, host first touch"
                : mode == 2 ? "managed, advised GPU"
                : mode == 4 ? "managed, host touch + prefetch to GPU" : "cudaMallocHost");
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
