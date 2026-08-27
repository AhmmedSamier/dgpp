// dgppctl: environment inspection utility for the DGPP project.
// Subcommands are additive; "info" prints CUDA/device facts without heavy work.
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>
#include <string>

#include "common/log.hpp"

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    DGPP_LOG_ERROR("{} failed: {}", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

int cmd_info() {
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  DGPP_LOG_INFO("CUDA driver={} runtime={}.{}", driver / 1000,
                runtime / 1000 % 1000, runtime % 1000);

  int ndev = 0;
  check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
  for (int d = 0; d < ndev; ++d) {
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, d), "cudaGetDeviceProperties");
    int unified = 0, page_table = 0;
    check(cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, d),
          "unified attr");
    check(cudaDeviceGetAttribute(
              &page_table, cudaDevAttrPageableMemoryAccessUsesHostPageTables, d),
          "hmptable attr");
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);  // best-effort on UVM platforms
    DGPP_LOG_INFO(
        "dev[{}] name=\"{}\" cc={}.{} sm={} l2={:.1f}MiB unifiedAddr={} "
        "pageableTableAccess={}",
        d, p.name, p.major, p.minor, p.multiProcessorCount,
        static_cast<double>(p.l2CacheSize) / 1048576.0, unified, page_table);
    if (total_b > 0) {
      DGPP_LOG_INFO("dev[{}] mem total={:.1f}GiB free={:.1f}GiB", d,
                    static_cast<double>(total_b) / 1073741824.0,
                    static_cast<double>(free_b) / 1073741824.0);
    }
    int sm_clock = 0;
    check(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, d),
          "sm clock attr");
    DGPP_LOG_INFO("dev[{}] clock SM={:.1f}MHz", d,
                  static_cast<double>(sm_clock) / 1000.0);
  }
  return 0;
}

int cmd_alloc_probe(size_t mb) {
  void* p = nullptr;
  auto err = cudaMalloc(&p, mb << 20);
  if (err != cudaSuccess) {
    DGPP_LOG_ERROR("alloc {} MiB FAILED: {}", mb, cudaGetErrorString(err));
    return 1;
  }
  err = cudaMemset(p, 0xA5, mb << 20);
  check(err, "cudaMemset");
  check(cudaFree(p), "cudaFree");
  DGPP_LOG_INFO("alloc {} MiB OK", mb);
  return 0;
}

int cmd_stream_latency() {
  // Measures event record/synchronize round-trip cost as a coarse proxy for
  // launch/sync overheads. Refined kernels come with the executor.
  const int kIters = 5000;
  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start), "event create");
  check(cudaEventCreate(&stop), "event create");
  float ms = 0.f;
  for (int w = 0; w < 3; ++w) {  // warmup + best-of
    auto t0 = ::clock();
    for (int i = 0; i < kIters; ++i) {
      cudaEventRecord(start, 0);
      cudaEventRecord(stop, 0);
      check(cudaEventSynchronize(stop), "sync");
    }
    ms = 1000.f * (static_cast<float>(::clock()) - static_cast<float>(t0)) /
         CLOCKS_PER_SEC;
  }
  DGPP_LOG_INFO("record+sync pair: {:.3f} us/iter ({})", ms * 1000.f / kIters,
                kIters);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: dgppctl <command>\n"
                 "  info                 print CUDA/platform facts\n"
                 "  alloc-probe N_MB     allocate/free N MiB device buffer\n"
                 "  stream-latency       measure event/launch overhead\n");
    return 2;
  }
  std::string cmd = argv[1];
  if (cmd == "info") return cmd_info();
  if (cmd == "alloc-probe") {
    size_t mb = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 16;
    return cmd_alloc_probe(mb);
  }
  if (cmd == "stream-latency") return cmd_stream_latency();
  DGPP_LOG_ERROR("unknown command: {}", cmd);
  return 2;
}
