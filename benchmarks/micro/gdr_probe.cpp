// micro_gdr_probe: determine GPU Direct RDMA viability on this GB10 system.
// 1) reports peermem module presence,
// 2) attempts ibverbs registration of cudaMalloc'd device memory,
// 3) benchmarks cudaMemcpyAsync staging latency (pinned vs pageable) so the
//    cost of the no-GDR fallback path is quantified.
#include <infiniband/verbs.h>

#include <cuda_runtime.h>
#include <getopt.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/log.hpp"

namespace {

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
  if (pinned) cudaHostAlloc(&host, bytes, cudaHostAllocDefault);
  else host = malloc(bytes);
  cudaMalloc(&dev, bytes);
  cudaStream_t s{};
  cudaStreamCreate(&s);
  // warmup
  for (int i = 0; i < 20; ++i) {
    if (d2h) cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s);
    else cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
  }
  cudaStreamSynchronize(s);
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (d2h) cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, s);
    else cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, s);
  }
  cudaStreamSynchronize(s);
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
    return 1;
  }
  ibv_context* ctx = ibv_open_device(list[0]);
  ibv_pd* pd = ibv_alloc_pd(ctx);

  void* devbuf = nullptr;
  size_t bytes = 4 << 20;
  cudaError_t ce = cudaMalloc(&devbuf, bytes);
  if (ce != cudaSuccess) {
    DGPP_LOG_ERROR("cudaMalloc failed: {}", cudaGetErrorString(ce));
    return 1;
  }
  ibv_mr* mr = ibv_reg_mr(pd, devbuf, bytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ);
  if (mr) {
    DGPP_LOG_INFO(
        "GDR-RESULT SUCCESS device memory registered with ibverbs (true "
        "GPUDirect likely usable)");
    ibv_dereg_mr(mr);
  } else {
    DGPP_LOG_WARN(
        "GDR-RESULT FAILED ibverbs could not register device memory "
        "(errno={}); falling back to pinned-bounce design",
        errno);
  }
  cudaFree(devbuf);
  ibv_dealloc_pd(pd);
  ibv_close_device(ctx);
  return mr ? 0 : 1;
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  bool peermem_nvidia = module_loaded("nvidia_peermem");
  bool peermem_legacy = module_loaded("nv_peer_mem");
  DGPP_LOG_INFO("peermem modules: nvidia_peermem={} nv_peer_mem={}",
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
  return rc;
}
