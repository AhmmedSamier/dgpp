#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/scale_gemm.hpp"
int main() {
  constexpr int n = 124160, k = 2560, max_m = 16;
  uint16_t* a;
  uint8_t* w;
  float *s, *o;
  DGPP_CUDA_OK(cudaMallocManaged(&a, max_m * k * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, size_t(n) * k));
  DGPP_CUDA_OK(cudaMallocManaged(&s, size_t((n + 127) / 128) * (k / 128) * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&o, size_t(max_m) * n * 4));
  for (int i = 0; i < max_m * k; ++i)
    a[i] = dgpp::float_to_bf16_bits(float((i * 17) % 127 - 63) / 64);
  for (size_t i = 0; i < size_t(n) * k; ++i)
    w[i] = dgpp::float_to_fp8_e4m3_bits(float(int((i * 13 + i / 127) % 61) - 30) / 16);
  for (int i = 0; i < ((n + 127) / 128) * (k / 128); ++i) s[i] = 0.375f + float(i % 13) / 16;
  cudaEvent_t begin, end;
  DGPP_CUDA_OK(cudaEventCreate(&begin));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  auto run = [&](int m, int path) {
    dgpp::launch_scale_gemm_f32(a, k, w, s, o, m, n, k, nullptr, n, path ? 5 : 0);
  };
  std::printf("N=%d K=%d; synthetic weights; alternating order; 5 pairs x 20 products\n", n, k);
  for (int m : {1, 4, 5, 8, 12, 16}) {
    std::vector<float> reference(size_t(m) * n), got(reference.size());
    run(m, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(reference.data(), o, reference.size() * 4);
    run(m, 1);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(got.data(), o, got.size() * 4);
    double d2 = 0, r2 = 0, max_abs = 0, max_diff = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (!std::isfinite(got[i]) || !std::isfinite(reference[i])) return 2;
      double d = double(got[i]) - reference[i];
      d2 += d * d;
      r2 += double(reference[i]) * reference[i];
      max_abs = std::max(max_abs, std::abs(double(reference[i])));
      max_diff = std::max(max_diff, std::abs(d));
    }
    if (std::sqrt(d2 / r2) > 1e-5 || max_diff > 3e-5 * max_abs) return 3;
    // Sample every row across shard/block boundaries against FP64, without
    // making the host do a five-billion-product reference for every shape.
    std::vector<int> cols = {0, 1, 7, 8, 31, 32, 63, 64, 127, 128, n - 2, n - 1};
    for (int col = 4096; col < n; col += 4096) {
      cols.push_back(col - 1);
      cols.push_back(col);
    }
    double max_oracle_error = 0;
    for (int row = 0; row < m; ++row)
      for (int col : cols) {
        double ref = 0, mag = 0;
        for (int kk = 0; kk < k; ++kk) {
          float weight = dgpp::bf16_bits_to_float(
              dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(w[size_t(col) * k + kk]) *
                                       s[(col / 128) * (k / 128) + kk / 128]));
          double product = double(dgpp::bf16_bits_to_float(a[row * k + kk])) * weight;
          ref += product;
          mag += std::abs(product);
        }
        double error = std::abs(got[size_t(row) * n + col] - ref);
        if (error > 3e-5 * mag) return 4;
        max_oracle_error = std::max(max_oracle_error, error);
      }
    // Warm both paths after host reads of managed allocations.
    for (int i = 0; i < 4; ++i) {
      run(m, 0);
      run(m, 1);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> times[2];
    for (int pair = 0; pair < 5; ++pair)
      for (int turn = 0; turn < 2; ++turn) {
        int path = (pair + turn) % 2;
        DGPP_CUDA_OK(cudaEventRecord(begin));
        for (int i = 0; i < 20; ++i) run(m, path);
        DGPP_CUDA_OK(cudaEventRecord(end));
        DGPP_CUDA_OK(cudaEventSynchronize(end));
        float ms;
        DGPP_CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
        times[path].push_back(ms / 20);
      }
    for (auto& t : times) std::sort(t.begin(), t.end());
    std::printf("m=%d gemv_ms=%.6f mma_ms=%.6f ratio=%.4f rel_l2=%.9g fp64_max_abs=%.9g\n", m,
                times[0][2], times[1][2], times[0][2] / times[1][2], std::sqrt(d2 / r2),
                max_oracle_error);
    std::fflush(stdout);
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaFree(a);
  cudaFree(w);
  cudaFree(s);
  cudaFree(o);
}
