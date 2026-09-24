#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/mma_gemv.hpp"

int main() {
  constexpr int max_m = 8192, n = 124160, k = 2560;
  uint16_t* act = nullptr;
  uint8_t* weights = nullptr;
  float *scales = nullptr, *output = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, size_t(max_m) * k * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&weights, size_t(n) * k));
  DGPP_CUDA_OK(cudaMallocManaged(&scales, size_t(n / 128) * (k / 128) * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&output, size_t(max_m) * n * 4));
  for (size_t i = 0; i < size_t(max_m) * k; ++i)
    act[i] = dgpp::float_to_bf16_bits(float(int((i * 173 + 37) % 1009) - 504) / 257.0f);
  for (size_t i = 0; i < size_t(n) * k; ++i)
    weights[i] = static_cast<uint8_t>((i * 47 + 13) % 126 | ((i % 3) ? 128 : 0));
  for (size_t i = 0; i < size_t(n / 128) * (k / 128); ++i)
    scales[i] = float((i * 37) % 251 + 1) / 263.0f;
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  for (int m : {129, 4096, 8192}) {
    auto run = [&](int mode) {
      auto* a = act + size_t(m-1) * k;
      auto* out = output + size_t(m-1) * n;
      if (mode < 2)
        dgpp::launch_scale_gemm_f32(act, k, weights, scales, output, m, n, k, nullptr, n, 0, mode == 1);
      else if (mode == 2)
        dgpp::launch_scale_gemm_tile_f32(a, k, weights, scales, out, 1, n, k, nullptr);
      else if (mode == 3)
        dgpp::launch_scale_gemm_f32(a, k, weights, scales, out, 1, n, k, nullptr);
      else
        dgpp::launch_mma_gemv_fp8_f32(a, k, weights, scales, out, 1, n, k, n, 7, 7, nullptr);
    };
    run(0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> expected(n), actual(n);
    DGPP_CUDA_OK(cudaMemcpy(expected.data(), output + size_t(m-1) * n, size_t(n) * 4, cudaMemcpyDeviceToHost));
    const char* labels[] = {"full_dense", "last_dense_128", "last_tile_16", "last_gemv", "last_streaming_mma"};
    for (int mode = 0; mode < 5; ++mode) {
      run(mode);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemcpy(actual.data(), output + size_t(m-1) * n, size_t(n) * 4, cudaMemcpyDeviceToHost));
      size_t different = 0;
      for (int i = 0; i < n; ++i) different += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
      run(mode);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<float> ms;
      for (int trial = 0; trial < 5; ++trial) {
        DGPP_CUDA_OK(cudaEventRecord(start));
        run(mode);
        DGPP_CUDA_OK(cudaEventRecord(end));
        DGPP_CUDA_OK(cudaEventSynchronize(end));
        float elapsed;
        DGPP_CUDA_OK(cudaEventElapsedTime(&elapsed, start, end));
        ms.push_back(elapsed);
      }
      std::sort(ms.begin(), ms.end());
      std::printf("m=%d n=%d k=%d mode=%s differing_logits=%zu median_ms=%.3f\n", m, n, k,
                  labels[mode], different, ms[2]);
    }
    std::fflush(stdout);
  }
  DGPP_CUDA_OK(cudaEventDestroy(start));
  DGPP_CUDA_OK(cudaEventDestroy(end));
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(weights));
  DGPP_CUDA_OK(cudaFree(scales));
  DGPP_CUDA_OK(cudaFree(output));
}
