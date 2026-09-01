// gemv_bw: the decode GEMV cores against the three memory placements the
// GB10 offers (cudaMalloc, pinned host, managed) at real shapes — separates
// "the kernel's access pattern" from "where the bytes live". Reports
// effective GB/s (weight bytes / kernel time).
#include <cuda_runtime.h>
#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {

enum class Place { Device, Pinned, Managed };

const char* place_name(Place p) {
  switch (p) {
    case Place::Device: return "cudaMalloc";
    case Place::Pinned: return "pinned";
    case Place::Managed: return "managed";
  }
  return "?";
}

void* alloc(Place p, size_t bytes) {
  void* ptr = nullptr;
  switch (p) {
    case Place::Device: DGPP_CUDA_OK(cudaMalloc(&ptr, bytes)); break;
    case Place::Pinned:
      DGPP_CUDA_OK(cudaHostAlloc(&ptr, bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable));
      break;
    case Place::Managed: DGPP_CUDA_OK(cudaMallocManaged(&ptr, bytes)); break;
  }
  return ptr;
}

void release(Place p, void* ptr) {
  if (p == Place::Pinned) cudaFreeHost(ptr);
  else cudaFree(ptr);
}

// Fill with a pseudo-random byte pattern from the host side for pinned/
// managed (the loader's path) and via cudaMemcpy for device memory.
void fill(Place p, void* dst, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < bytes; i += 8) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    std::memcpy(h.data() + i, &s, std::min<size_t>(8, bytes - i));
  }
  // bf16 payloads: keep values finite/small (clear the exponent's top bits).
  for (size_t i = 1; i < bytes; i += 2) h[i] = (h[i] & 0x80u) | 0x3Cu;
  if (p == Place::Device)
    DGPP_CUDA_OK(cudaMemcpy(dst, h.data(), bytes, cudaMemcpyHostToDevice));
  else
    std::memcpy(dst, h.data(), bytes);
}

template <typename F>
double time_ms(F&& launch, int iters) {
  cudaEvent_t beg{}, end{};
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  for (int i = 0; i < 3; ++i) launch();
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  cudaEventRecord(beg, nullptr);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(end, nullptr);
  cudaEventSynchronize(end);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, beg, end);
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return ms / iters;
}

struct Shape {
  const char* name;
  int n, k;
};

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  int iters = 10;
  int cold_gib = 0;  // -c GiB: stream DISJOINT windows of one big buffer
  int opt{};
  while ((opt = getopt(argc, argv, "i:c:")) != -1) {
    if (opt == 'i') iters = atoi(optarg);
    if (opt == 'c') cold_gib = atoi(optarg);
  }
  if (cold_gib > 0) {
    // The model's pattern: every kernel touches weights nobody touched
    // since the last token — no TLB or L2 reuse between launches. One
    // window per launch, windows disjoint across the buffer.
    const int n = 6416, k = 4096;  // the KDA in_proj shape
    const size_t wbytes = static_cast<size_t>(n) * k * 2;
    const size_t total = static_cast<size_t>(cold_gib) << 30;
    const int windows = static_cast<int>(total / wbytes);
    for (Place p : {Place::Device, Place::Pinned, Place::Managed}) {
      uint8_t* big = static_cast<uint8_t*>(alloc(p, total));
      // Touch every page once (host side for pinned/managed, memset for device).
      if (p == Place::Device) DGPP_CUDA_OK(cudaMemset(big, 0x3C, total));
      else std::memset(big, 0x3C, total);
      uint16_t* x = nullptr;
      uint16_t* out = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&x, static_cast<size_t>(k) * 2));
      DGPP_CUDA_OK(cudaMalloc(&out, static_cast<size_t>(n) * 2));
      DGPP_CUDA_OK(cudaMemset(x, 0x3C, static_cast<size_t>(k) * 2));
      int w = 0;
      const double ms = time_ms(
          [&] {
            const uint16_t* wp = reinterpret_cast<const uint16_t*>(
                big + static_cast<size_t>(w) * wbytes);
            w = (w + 1) % windows;
            dgpp::launch_bf16_gemv(x, k, wp, out, false, 1, n, k, nullptr);
          },
          std::min(iters, windows));
      DGPP_LOG_INFO("{:10s} COLD windows over {} GiB: bf16 gemv N6416xK4096  {:7.1f} us  {:6.1f} GB/s",
                    place_name(p), cold_gib, ms * 1e3, wbytes / (ms * 1e-3) / 1e9);
      cudaFree(x);
      cudaFree(out);
      release(p, big);
    }
    return 0;
  }

  const Shape shapes[] = {
      {"lm_head  N38720xK4096", 38720, 4096},
      {"kda in   N6416xK4096", 6416, 4096},
      {"dsa o    N4096xK4096", 4096, 4096},
      {"kda o    N4096xK2048", 4096, 2048},
  };
  for (Place p : {Place::Device, Place::Pinned, Place::Managed}) {
    for (const Shape& sh : shapes) {
      const size_t wbytes = static_cast<size_t>(sh.n) * sh.k * 2;
      uint16_t* w = static_cast<uint16_t*>(alloc(p, wbytes));
      fill(p, w, wbytes);
      uint16_t* x = nullptr;
      uint16_t* out = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&x, static_cast<size_t>(sh.k) * 2));
      DGPP_CUDA_OK(cudaMalloc(&out, static_cast<size_t>(sh.n) * 2));
      DGPP_CUDA_OK(cudaMemset(x, 0x3C, static_cast<size_t>(sh.k) * 2));
      const double ms = time_ms(
          [&] {
            dgpp::launch_bf16_gemv(x, sh.k, w, out, false, 1, sh.n, sh.k,
                                   nullptr);
          },
          iters);
      DGPP_LOG_INFO("{:10s} bf16 gemv {}  {:7.1f} us  {:6.1f} GB/s",
                    place_name(p), sh.name, ms * 1e3,
                    wbytes / (ms * 1e-3) / 1e9);
      cudaFree(x);
      cudaFree(out);
      release(p, w);
    }
    // The fp8 core at the expert shape (one expert, gate: N2048xK4096) x 8
    // experts stacked as rows (16384 rows) to reach a measurable size.
    {
      const int n = 2048 * 8, k = 4096;
      const size_t wbytes = static_cast<size_t>(n) * k;
      uint8_t* w = static_cast<uint8_t*>(alloc(p, wbytes));
      fill(p, w, wbytes);
      float* scales = nullptr;
      uint16_t* x = nullptr;
      uint16_t* out = nullptr;
      const size_t nscales = static_cast<size_t>(n / 128) * (k / 128);
      DGPP_CUDA_OK(cudaMalloc(&scales, nscales * 4));
      std::vector<float> hs(nscales, 1.0f);
      DGPP_CUDA_OK(cudaMemcpy(scales, hs.data(), nscales * 4, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMalloc(&x, static_cast<size_t>(k) * 2));
      DGPP_CUDA_OK(cudaMalloc(&out, static_cast<size_t>(n) * 2));
      DGPP_CUDA_OK(cudaMemset(x, 0x3C, static_cast<size_t>(k) * 2));
      const double ms = time_ms(
          [&] {
            dgpp::launch_scale_gemm_bf16(x, k, w, scales, out, 1, n, k,
                                         nullptr);
          },
          iters);
      DGPP_LOG_INFO("{:10s} fp8  gemv 8 experts N16384xK4096  {:7.1f} us  {:6.1f} GB/s",
                    place_name(p), ms * 1e3, wbytes / (ms * 1e-3) / 1e9);
      cudaFree(scales);
      cudaFree(x);
      cudaFree(out);
      release(p, w);
    }
  }
  return 0;
}
