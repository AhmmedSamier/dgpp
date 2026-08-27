// micro_gdr_probe: determine GPU Direct RDMA viability on this GB10 system.
// 1) reports whether peermem modules are currently loaded,
// 2) attempts ibverbs registration of cudaMalloc'd device memory,
// 3) profiles CUDA copies for diagnostic comparison. A failed direct-device
//    registration is an expected platform result on GB10 and is not itself a
//    benchmark failure; registered host memory remains GPU-readable on UMA.
#include <infiniband/verbs.h>

#include <cuda_runtime.h>
#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/log.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    DGPP_LOG_ERROR("{} failed: {}", operation, cudaGetErrorString(status));
    std::exit(1);
  }
}

bool module_loaded(const char* needle) {
  FILE* f = fopen("/proc/modules", "r");
  if (!f) return false;
  char line[512];
  bool found = false;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, needle)) {
      found = true;
      break;
    }
  }
  fclose(f);
  return found;
}

double bench_memcpy(size_t bytes, bool pinned, bool d2h, int iters) {
  void *host{}, *dev{};
  if (pinned)
    check_cuda(cudaHostAlloc(&host, bytes, cudaHostAllocDefault),
               "cudaHostAlloc");
  else
    host = malloc(bytes);
  if (!host) {
    DGPP_LOG_ERROR("host allocation failed bytes={}", bytes);
    std::exit(1);
  }
  check_cuda(cudaMalloc(&dev, bytes), "cudaMalloc copy buffer");
  cudaStream_t s{};
  check_cuda(cudaStreamCreate(&s), "cudaStreamCreate");
  // warmup
  for (int i = 0; i < 20; ++i) {
    if (d2h)
      check_cuda(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s),
                 "warmup D2H");
    else
      check_cuda(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s),
                 "warmup H2D");
  }
  check_cuda(cudaStreamSynchronize(s), "warmup sync");
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (d2h)
      check_cuda(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s),
                 "timed D2H");
    else
      check_cuda(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s),
                 "timed H2D");
  }
  check_cuda(cudaStreamSynchronize(s), "timed copy sync");
  auto t1 = std::chrono::steady_clock::now();
  double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  cudaStreamDestroy(s);
  cudaFree(dev);
  if (pinned) cudaFreeHost(host);
  else free(host);
  return us;
}

int try_gdr_registration() {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (!list || num == 0) {
    DGPP_LOG_ERROR("no verbs devices for GDR probe");
    if (list) ibv_free_device_list(list);
    return -1;
  }
  ibv_context* ctx = ibv_open_device(list[0]);
  ibv_free_device_list(list);
  if (!ctx) {
    DGPP_LOG_ERROR("could not open verbs device errno={}", errno);
    return -1;
  }
  ibv_pd* pd = ibv_alloc_pd(ctx);
  if (!pd) {
    DGPP_LOG_ERROR("could not allocate protection domain errno={}", errno);
    ibv_close_device(ctx);
    return -1;
  }

  void* devbuf = nullptr;
  size_t bytes = 4 << 20;
  cudaError_t ce = cudaMalloc(&devbuf, bytes);
  if (ce != cudaSuccess) {
    DGPP_LOG_ERROR("cudaMalloc failed: {}", cudaGetErrorString(ce));
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    return -1;
  }
  ibv_mr* mr = ibv_reg_mr(pd, devbuf, bytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ);
  const bool supported = mr != nullptr;
  if (mr) {
    DGPP_LOG_INFO("GDR-RESULT SUPPORTED cudaMalloc memory registered by ibverbs");
    ibv_dereg_mr(mr);
  } else {
    DGPP_LOG_WARN(
        "GDR-RESULT UNSUPPORTED cudaMalloc memory registration failed "
        "(errno={}); use registered host memory for direct GPU consumption "
        "on GB10 UMA (this does not imply a staging copy)",
        errno);
  }
  cudaFree(devbuf);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return supported ? 1 : 0;
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  bool peermem_nvidia = module_loaded("nvidia_peermem");
  bool peermem_legacy = module_loaded("nv_peer_mem");
  DGPP_LOG_INFO("loaded peermem modules: nvidia_peermem={} nv_peer_mem={}",
                peermem_nvidia, peermem_legacy);

  int rc = try_gdr_registration();

  constexpr int kIters = 500;
  for (size_t sz : {size_t(1024), size_t(65536)}) {
    double h2d_p = bench_memcpy(sz, true, false, kIters);
    double h2d_n = bench_memcpy(sz, false, false, kIters);
    double d2h_p = bench_memcpy(sz, true, true, kIters);
    DGPP_LOG_INFO(
        "COPY-PROFILE size={}B h2d_pinned={:.2f}us h2d_pageable={:.2f}us "
        "d2h_pinned={:.2f}us",
        sz, h2d_p, h2d_n, d2h_p);
  }
  return rc < 0 ? 1 : 0;
}
