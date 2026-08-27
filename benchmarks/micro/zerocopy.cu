// micro_zerocopy: GPU access to cudaHostAlloc (pinned) memory on GB10.
//
// GB10 is a unified-memory platform: CPU and GPU share one LPDDR5x pool, so
// "host pinned" buffers live in the same physical memory the GPU streams.
// This bench answers the CollectiveBus receive-path design questions:
//   1. GPU streaming-read bandwidth of pinned memory vs cudaMalloc memory.
//   2. Degradation when CPU writers flood other buffers in the same pool.
//   3. Staging-copy alternative cost (pinned <-> device copies) for compare.
//   4. CPU<->GPU flag visibility latency with correct fence discipline —
//      the round trip a completion notification actually pays.
//
// Machine-parseable lines: PATTERN <name> BEST <GB/s>, FLAG <name> <stats...>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "common/log.hpp"
#include "kernels/flag_protocol.cuh"

namespace {

using u64 = uint64_t;
using u32 = uint32_t;

constexpr size_t kBufBytes = 256u << 20;  // 256 MiB >> 24 MiB L2: DRAM-bound
constexpr int kWarmups = 3;
constexpr int kIters = 10;
// Watchdog budget for the persistent flag kernel: generous fixed cycle count
// (~8 s at GB10's ~2.4 GHz class clock). Only ever hit on a lost wakeup.
constexpr u64 kFlagDeadlineCycles = 20'000'000'000ull;
// CPU-side per-message timeout: if the kernel died or ordering is broken,
// we want a loud FAIL, not a hung bench.
constexpr double kMsgTimeoutSec = 5.0;

inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// Payload kernels. u4 = uint4 (128-bit) accesses; reads fold into an xor
// accumulator that is never used so loads cannot be elided.
// ---------------------------------------------------------------------------

__global__ void read_u4_kernel(const uint4* __restrict__ src, size_t n4,
                               uint4* __restrict__ sink) {
  uint4 acc = {0u, 0u, 0u, 0u};
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    const uint4 v = src[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if (acc.x == 0xDEADBEEFu) sink[blockIdx.x] = acc;  // never taken
}

__global__ void write_u4_kernel(uint4* __restrict__ dst, size_t n4, u32 val) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const uint4 v = make_uint4(val, val + 1u, val + 2u, val + 3u);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride)
    dst[i] = v;
}

// Persistent flag kernel comes from kernels/flag_protocol.cuh (shared with
// the regression test and, eventually, CollectiveBus itself — one protocol,
// one implementation).

// CPU writer thread: floods a buffer with independent uint4 stores.
struct WriterArgs {
  uint4* buf;
  size_t n4;
  const std::atomic<bool>* stop;
};

void writer_thread(WriterArgs a) {
  u32 seed = 0x9E3779B9u;
  size_t i = 0;
  while (!a.stop->load(std::memory_order_relaxed)) {
    seed = seed * 1664525u + 1013904223u;
    a.buf[i] = make_uint4(seed, seed ^ 1u, seed ^ 2u, seed ^ 3u);
    if (++i == a.n4) i = 0;
  }
}

// ---------------------------------------------------------------------------

double gb_per_s(size_t bytes, double ms) {
  return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

template <typename F>
double best_ms(F&& launch, cudaStream_t stream, int iters, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    for (int i = 0; i < kWarmups; ++i) launch(stream);
    cudaEventRecord(beg, stream);
    for (int i = 0; i < iters; ++i) launch(stream);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float total = 0.f;
    cudaEventElapsedTime(&total, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    best = std::min(best, static_cast<double>(total) / iters);
  }
  return best;
}

// Searches a small grid-size space: zero-copy latency hiding may prefer
// deeper grids than the device-buffer optimum.
double best_read_gb_s(const uint4* src, size_t n4, uint4* sink,
                      cudaStream_t stream) {
  const size_t bytes = n4 * sizeof(uint4);
  double best = 0.0;
  for (int blocks : {48, 96, 192, 384}) {
    double ms = best_ms(
        [&](cudaStream_t s) { read_u4_kernel<<<blocks, 256, 0, s>>>(src, n4, sink); },
        stream, kIters, 3);
    best = std::max(best, gb_per_s(bytes, ms));
  }
  return best;
}

// ---------------------------------------------------------------------------
// Flag visibility: CPU bumps start_seq, GPU observes + acks, CPU observes
// ack. Wall-clock round trip = what a completion notification costs.
// ---------------------------------------------------------------------------

bool run_flag_roundtrip(dgpp::StartSlot* start, dgpp::FlagAck* ack) {
  constexpr int kMsgs = 512;
  std::vector<double> us;
  us.reserve(kMsgs);
  start->seq = 0;
  ack->seq = 0;
  ack->cycles = 0;
  std::atomic_thread_fence(std::memory_order_release);
  for (int i = 1; i <= kMsgs; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    __atomic_store_n(&start->seq, static_cast<u32>(i), __ATOMIC_RELEASE);
    while (__atomic_load_n(&ack->seq, __ATOMIC_ACQUIRE) != static_cast<u32>(i)) {
      cpu_relax();
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count() > kMsgTimeoutSec) {
        printf("FLAG roundtrip TIMEOUT at seq=%d (kernel lost?)\n", i);
        return false;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  printf("FLAG roundtrip min %.2f us | p50 %.2f us | p99 %.2f us | max %.2f us\n",
         us.front(), us[us.size() / 2], us[us.size() * 99 / 100], us.back());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  DGPP_LOG_INFO("device={} sm={} l2={}MiB", prop.name, prop.multiProcessorCount,
           static_cast<int>(prop.l2CacheSize >> 20));

  const size_t n4 = kBufBytes / sizeof(uint4);
  uint4 *dev_buf = nullptr, *pin_buf = nullptr, *pin_alt = nullptr,
        *pin_alt2 = nullptr, *sink = nullptr;
  cudaMalloc(&dev_buf, kBufBytes);
  cudaMalloc(&sink, 4096 * sizeof(uint4));
  cudaHostAlloc(&pin_buf, kBufBytes, cudaHostAllocDefault);
  cudaHostAlloc(&pin_alt, kBufBytes, cudaHostAllocDefault);
  cudaHostAlloc(&pin_alt2, kBufBytes, cudaHostAllocDefault);
  if (!dev_buf || !pin_buf || !pin_alt || !pin_alt2 || !sink) {
    DGPP_LOG_ERROR("allocations failed");
    return 1;
  }
  cudaMemset(dev_buf, 0x5A, kBufBytes);
  cudaMemset(pin_buf, 0x5A, kBufBytes);
  cudaMemset(pin_alt, 0x5A, kBufBytes);
  cudaMemset(pin_alt2, 0x5A, kBufBytes);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // --- bandwidth patterns ---------------------------------------------------
  double gb = best_read_gb_s(dev_buf, n4, sink, stream);
  printf("PATTERN gpu_read_dev BEST %.1f GB/s\n", gb);

  gb = best_read_gb_s(pin_buf, n4, sink, stream);
  printf("PATTERN gpu_read_pin BEST %.1f GB/s\n", gb);

  {  // contended: writer hits a different pinned buffer (pool-level only)
    std::atomic<bool> stop{false};
    std::thread w(writer_thread, WriterArgs{pin_alt, n4, &stop});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gb = best_read_gb_s(pin_buf, n4, sink, stream);
    stop.store(true, std::memory_order_relaxed);
    w.join();
    printf("PATTERN gpu_read_pin_contended_other BEST %.1f GB/s\n", gb);
  }
  {  // two writers on distinct other buffers: multi-flow pool pressure
    std::atomic<bool> stop1{false}, stop2{false};
    std::thread w1(writer_thread, WriterArgs{pin_alt, n4, &stop1});
    std::thread w2(writer_thread, WriterArgs{pin_alt2, n4, &stop2});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gb = best_read_gb_s(pin_buf, n4, sink, stream);
    stop1.store(true, std::memory_order_relaxed);
    stop2.store(true, std::memory_order_relaxed);
    w1.join();
    w2.join();
    printf("PATTERN gpu_read_pin_contended_2other BEST %.1f GB/s\n", gb);
  }
  gb = best_ms(
      [&](cudaStream_t s) { write_u4_kernel<<<192, 256, 0, s>>>(pin_buf, n4, 7u); },
      stream, kIters, 3);
  printf("PATTERN gpu_write_pin BEST %.1f GB/s\n", gb_per_s(kBufBytes, gb));

  {  // staging-copy alternatives, both directions
    gb = best_ms(
        [&](cudaStream_t s) {
          cudaMemcpyAsync(dev_buf, pin_buf, kBufBytes, cudaMemcpyDefault, s);
        },
        stream, kIters, 3);
    printf("PATTERN copy_pin2dev BEST %.1f GB/s\n", gb_per_s(kBufBytes, gb));

    gb = best_ms(
        [&](cudaStream_t s) {
          cudaMemcpyAsync(pin_buf, dev_buf, kBufBytes, cudaMemcpyDefault, s);
        },
        stream, kIters, 3);
    printf("PATTERN copy_dev2pin BEST %.1f GB/s\n", gb_per_s(kBufBytes, gb));
  }

  {  // CPU producer ceiling into pinned memory
    std::vector<char> tmp(kBufBytes);
    memset(tmp.data(), 0x33, tmp.size());
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      memcpy(pin_buf, tmp.data(), kBufBytes);
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    printf("PATTERN cpu_memcpy_pin BEST %.1f GB/s\n", kBufBytes / best / 1e9);
  }

  // --- flag visibility --------------------------------------------------------
  dgpp::StartSlot* start = nullptr;
  dgpp::FlagAck* ack = nullptr;
  cudaHostAlloc(&start, sizeof(dgpp::StartSlot), cudaHostAllocDefault);
  cudaHostAlloc(&ack, sizeof(dgpp::FlagAck), cudaHostAllocDefault);
  if (!start || !ack) {
    DGPP_LOG_ERROR("flag slot alloc failed");
    return 1;
  }
  *start = {};
  *ack = {};
  dgpp::flag_heartbeat_kernel<<<1, 1, 0, stream>>>(&start->seq, ack,
                                                   kFlagDeadlineCycles);
  const bool flag_ok = run_flag_roundtrip(start, ack);
  __atomic_store_n(&start->seq, dgpp::kFlagStopSequence, __ATOMIC_RELEASE);
  const cudaError_t flag_sync = cudaStreamSynchronize(stream);
  if (flag_sync != cudaSuccess)
    DGPP_LOG_ERROR("flag stream sync failed: {}", cudaGetErrorString(flag_sync));
  if (cudaError_t err = cudaGetLastError(); err != cudaSuccess)
    DGPP_LOG_ERROR("flag phase error: {}", cudaGetErrorString(err));

  cudaFree(dev_buf);
  cudaFree(sink);
  cudaFreeHost(pin_buf);
  cudaFreeHost(pin_alt);
  cudaFreeHost(pin_alt2);
  cudaFreeHost(start);
  cudaFreeHost(ack);
  cudaStreamDestroy(stream);
  return flag_ok && flag_sync == cudaSuccess ? 0 : 1;
}
