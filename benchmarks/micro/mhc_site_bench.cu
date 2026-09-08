// mhc_site_bench (2026-09-08): one mHC site as the decode graph runs it —
// launch_mhc_compute_normed with the fused finish (dots + finish in one
// launch) followed by launch_mhc_stream_update — timed per site at decode
// row counts. The per-kernel split comes from nsys over this binary.
//   mhc_site_bench [--tokens N] [--iters N] [--warmup N] [--no-update] [--unfused]
//                  [--sinkhorn N] [--no-norm] [--defer] [--sublayer-us N]
// --defer: the finish leaves comb to launch_mhc_comb on a side stream
// (forked after the finish, joined before the update); --sublayer-us N puts
// a spin kernel of N us between the finish and the update, as the real
// site's sublayer would, so the join has something to hide behind.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "models/glm/mhc.hpp"

using namespace dgpp;

namespace {
uint16_t bf16_of(float f) { return float_to_bf16_bits(f); }

__global__ void spin_kernel(long long cycles) {
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

template <typename T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T)));
  DGPP_CUDA_OK(cudaMemset(p, 0, n * sizeof(T)));
  return p;
}

template <typename T>
void upload(T* d, const std::vector<T>& h) {
  DGPP_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
}  // namespace

int main(int argc, char** argv) {
  int tokens = 1, iters = 400, warmup = 20;
  bool update = true;
  bool fused = true;
  int sinkhorn = 20;
  bool norm = true;
  bool defer = false;
  int sublayer_us = 0;
  int gemm = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-update")) update = false;
    else if (!std::strcmp(argv[i], "--unfused")) fused = false;  // dots + finish as two launches
    else if (!std::strcmp(argv[i], "--sinkhorn") && i + 1 < argc) sinkhorn = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-norm")) norm = false;
    else if (!std::strcmp(argv[i], "--defer")) defer = true;
    else if (!std::strcmp(argv[i], "--sublayer-us") && i + 1 < argc) sublayer_us = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gemm") && i + 1 < argc) gemm = std::atoi(argv[++i]);  // prefill dots: 1 tensor-core GEMM, 0 token-tiled
  }
  GlmMhcConfig cfg;  // the model's: n = 4, hidden = 4096, 20 Sinkhorn iterations
  cfg.sinkhorn_iters = sinkhorn;
  mhc_set_prefill_gemm(gemm != 0);
  const int H = cfg.hidden, N = cfg.hc_mult, C = cfg.coeff_rows();
  const size_t K = static_cast<size_t>(N) * H;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> h_streams(static_cast<size_t>(tokens) * K), h_fn(static_cast<size_t>(C) * K),
      h_ln(H), h_sub(static_cast<size_t>(tokens) * H);
  for (auto& v : h_streams) v = bf16_of(nd(rng));
  for (auto& v : h_fn) v = bf16_of(0.02f * nd(rng));
  for (auto& v : h_ln) v = bf16_of(1.f + 0.1f * nd(rng));
  for (auto& v : h_sub) v = bf16_of(nd(rng));
  std::vector<float> h_base(C, 0.f), h_scale(3, 1.f);

  uint16_t* streams = dev_alloc<uint16_t>(2 * tokens * K);  // [2][tokens][N][H]: cur | nxt
  uint16_t* fn = dev_alloc<uint16_t>(C * K);
  uint16_t* ln = dev_alloc<uint16_t>(H);
  uint16_t* sub = dev_alloc<uint16_t>(tokens * H);
  uint16_t* collapsed = dev_alloc<uint16_t>(tokens * H);
  uint16_t* normed = dev_alloc<uint16_t>(tokens * H);
  uint16_t* post = dev_alloc<uint16_t>(tokens * N);
  uint16_t* comb = dev_alloc<uint16_t>(tokens * N * N);
  float* logits = dev_alloc<float>(tokens * C);
  float* base = dev_alloc<float>(C);
  float* scale = dev_alloc<float>(3);
  int* counters = dev_alloc<int>(tokens);
  upload(streams, h_streams);
  upload(fn, h_fn);
  upload(ln, h_ln);
  upload(sub, h_sub);
  upload(base, h_base);
  upload(scale, h_scale);
  GlmMhcWeights w;
  w.fn = fn;
  w.base = base;
  w.scale = scale;

  cudaStream_t s, side;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
  cudaEvent_t fork, join;
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  int clock_khz = 0;
  DGPP_CUDA_OK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  const long long spin_cycles = static_cast<long long>(clock_khz) * sublayer_us / 1000;
  uint16_t* cur = streams;
  uint16_t* nxt = streams + tokens * K;
  auto site = [&]() {
    launch_mhc_compute_normed(cur, w, cfg, collapsed, post, comb, logits, norm ? ln : nullptr,
                              norm ? normed : nullptr, 1e-5f, tokens, s, fused ? counters : nullptr,
                              defer);
    if (defer) {
      DGPP_CUDA_OK(cudaEventRecord(fork, s));
      DGPP_CUDA_OK(cudaStreamWaitEvent(side, fork, 0));
      launch_mhc_comb(logits, w, cfg, comb, tokens, side);
      DGPP_CUDA_OK(cudaEventRecord(join, side));
    }
    if (sublayer_us > 0) spin_kernel<<<1, 32, 0, s>>>(spin_cycles);
    if (defer) DGPP_CUDA_OK(cudaStreamWaitEvent(s, join, 0));
    if (update) {
      launch_mhc_stream_update(post, comb, sub, cur, nxt, cfg, tokens, s);
      std::swap(cur, nxt);
    }
  };
  for (int i = 0; i < warmup; ++i) site();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  DGPP_CUDA_OK(cudaEventRecord(a, s));
  for (int i = 0; i < iters; ++i) site();
  DGPP_CUDA_OK(cudaEventRecord(b, s));
  DGPP_CUDA_OK(cudaEventSynchronize(b));
  float ms = 0.f;
  DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  std::printf("mhc_site_bench: tokens=%d hidden=%d n=%d coeffs=%d update=%d fused=%d sinkhorn=%d norm=%d defer=%d sublayer=%dus gemm=%d: %.2f us per site "
              "(stream-launched; x90 sites = %.2f ms)\n",
              tokens, H, N, C, update ? 1 : 0, fused ? 1 : 0, sinkhorn, norm ? 1 : 0, defer ? 1 : 0,
              sublayer_us, gemm, 1e3f * ms / iters, 90.f * ms / iters);
  return 0;
}
