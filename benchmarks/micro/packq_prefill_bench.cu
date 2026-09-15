// Compare packed prefill GEMV and tensor-core kernels on idle hardware.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"

namespace {
template <typename T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
};
double time_ms(const std::function<void()>& launch, int iters) {
  for (int i = 0; i < 3; ++i) launch();
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  DGPP_CUDA_OK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch();
  DGPP_CUDA_OK(cudaEventRecord(b));
  DGPP_CUDA_OK(cudaEventSynchronize(b));
  float elapsed;
  DGPP_CUDA_OK(cudaEventElapsedTime(&elapsed, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return elapsed / iters;
}
template <typename T>
double as_float(T x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else
    return dgpp::bf16_bits_to_float(x);
}

template <typename Out>
void run(int m, int n, int k, int bits, int experts, int top_k, int iters, bool hot) {
  using namespace dgpp;
  std::mt19937 rng(0x6144512);
  const size_t words = static_cast<size_t>(n) * k * bits / 32;
  const size_t factors = static_cast<size_t>(n) * k / 64;
  Buffer<uint32_t> packed(words * experts);
  Buffer<uint16_t> scales(factors * experts);
  for (size_t i = 0; i < words * experts; ++i) packed.p[i] = rng();
  for (size_t i = 0; i < factors * experts; ++i)
    scales.p[i] = float_to_bf16_bits(0.0001f + (rng() % 1000) * 0.000003f);
  Buffer<MoeExpertView> views(experts * 3);
  Buffer<MoeSegment> segs(experts);
  std::vector<int> counts(experts);
  for (int t = 0; t < m; ++t) {
    std::vector<int> chosen;
    while (static_cast<int>(chosen.size()) < top_k) {
      const int e = rng() % (hot ? top_k : experts);
      if (std::find(chosen.begin(), chosen.end(), e) != chosen.end()) continue;
      chosen.push_back(e);
      ++counts[e];
    }
  }
  int rows = 0, largest = 0, nonempty = 0;
  for (int e = 0; e < experts; ++e) {
    const GlmPackedMatrix w{packed.p + e * words, scales.p + e * factors, n, k, bits};
    views.p[e * 3] = MoeExpertView::of(w);
    segs.p[e] = {rows, counts[e], e};
    rows += counts[e];
    largest = std::max(largest, counts[e]);
    nonempty += counts[e] > 0;
  }
  Buffer<uint16_t> a(static_cast<size_t>(rows) * k);
  for (size_t i = 0; i < static_cast<size_t>(rows) * k; ++i)
    a.p[i] = float_to_bf16_bits((int(rng() % 2001) - 1000) * 0.001f);
  Buffer<Out> baseline(static_cast<size_t>(rows) * n), candidate(static_cast<size_t>(rows) * n);
  const GlmPackedMatrix dense{packed.p, scales.p, n, k, bits};
  auto launch = [&](bool mma) {
    Out* out = mma ? candidate.p : baseline.p;
    if (experts == 1) {
      if constexpr (std::is_same_v<Out, float>)
        (mma ? launch_packq_gemm_f32 : launch_packq_gemv_f32)(a.p, k, dense, out, m, n, k, nullptr);
      else
        (mma ? launch_packq_gemm_bf16 : launch_packq_gemv_bf16)(a.p, k, dense, out, m, n, k,
                                                                nullptr);
    } else if constexpr (std::is_same_v<Out, float>) {
      if (mma)
        launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, experts, m, views.p, 0, out, n, n, k, bits,
                                         nullptr);
      else
        launch_moe_grouped_gemv_packq_f32(a.p, k, segs.p, experts, m, 0, views.p, 0, out, n, n, k,
                                          bits, nullptr);
    } else {
      if (mma)
        launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, experts, m, views.p, 0, out, n, n, k,
                                          bits, nullptr);
      else
        launch_moe_grouped_gemv_packq_bf16(a.p, k, segs.p, experts, m, 0, views.p, 0, out, n, n, k,
                                           bits, nullptr);
    }
  };
  std::printf(
      "shape m=%d n=%d k=%d bits=%d experts=%d top_k=%d out=%s nonempty=%d max_rows=%d "
      "distribution=%s\n",
      m, n, k, bits, experts, top_k, std::is_same_v<Out, float> ? "f32" : "bf16", nonempty, largest,
      hot ? "hot" : "uniform");
  // Complete managed-memory placement before the measured event pairs.
  launch(false);
  launch(true);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int trial = 0; trial < 6; ++trial) {
    double ms[2];
    for (int mode : {trial % 2, 1 - trial % 2})
      ms[mode] = time_ms([&] { launch(mode != 0); }, iters);
    std::printf("trial=%d baseline_ms=%.6f candidate_ms=%.6f change_percent=%+.3f\n", trial, ms[0],
                ms[1], 100 * (ms[1] / ms[0] - 1));
    std::fflush(stdout);
  }
  double error = 0, norm = 0;
  for (size_t i = 0; i < static_cast<size_t>(rows) * n; ++i) {
    const double b = as_float(baseline.p[i]), c = as_float(candidate.p[i]);
    if (!std::isfinite(b) || !std::isfinite(c)) throw std::runtime_error("non-finite output");
    error += (b - c) * (b - c);
    norm += b * b;
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  std::printf("relative_l2=%.9g\n", relative);
  if (relative > (std::is_same_v<Out, float> ? 5e-6 : 1e-3))
    throw std::runtime_error("GEMM/GEMV numerical discrepancy");
}
}  // namespace

int main(int argc, char** argv) try {
  int m = 256, n = 512, k = 6144, bits = 4, experts = 256, top_k = 8, iters = 5;
  std::string distribution = "uniform", output = "bf16";
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--m")
      m = std::stoi(value);
    else if (key == "--n")
      n = std::stoi(value);
    else if (key == "--k")
      k = std::stoi(value);
    else if (key == "--bits")
      bits = std::stoi(value);
    else if (key == "--experts")
      experts = std::stoi(value);
    else if (key == "--top-k")
      top_k = std::stoi(value);
    else if (key == "--iters")
      iters = std::stoi(value);
    else if (key == "--distribution")
      distribution = value;
    else if (key == "--out")
      output = value;
    else
      throw std::invalid_argument("unknown option: " + key);
  }
  if (m <= 0 || n <= 0 || k <= 0 || k % 64 || (bits != 4 && bits != 8) || experts < top_k ||
      top_k <= 0 || iters <= 0 || (distribution != "uniform" && distribution != "hot") ||
      (output != "bf16" && output != "f32"))
    throw std::invalid_argument("invalid shape or options");
  if (output == "f32")
    run<float>(m, n, k, bits, experts, top_k, iters, distribution == "hot");
  else
    run<uint16_t>(m, n, k, bits, experts, top_k, iters, distribution == "hot");
} catch (const std::exception& e) {
  std::fprintf(stderr, "packq_prefill_bench: %s\n", e.what());
  return 1;
}
