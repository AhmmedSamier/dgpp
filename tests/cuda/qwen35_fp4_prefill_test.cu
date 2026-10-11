// The fused FP4 prefill against the existing two GEMMs and Qwen SwiGLU.
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/fp4w_gemm.hpp"

// Link the existing Qwen epilogue without importing its host-only model and
// loader headers into nvcc. This is the same function used by dense_mlp.
namespace dgpp {
void qwen35_swiglu_bf16(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                       cudaStream_t stream);
}

namespace {
void check(int m, int n, int k, int pad, int offset, bool all_gate_bits = false) {
  const int stride = k + 8;
  const size_t nk = size_t(n) * k, mn = size_t(m) * n;
  const size_t out_stride = n + pad, out_size = offset + size_t(m) * out_stride + 19;
  constexpr uint16_t guard = 0xdead;
  std::vector<uint16_t> ha(size_t(m) * stride, 0), ref(mn), got(out_size);
  std::vector<uint8_t> pg(nk / 2), pu(nk / 2), sg(nk / 16), su(nk / 16);
  uint32_t rng = 7;
  auto random = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      ha[size_t(i) * stride + j] =
          __bfloat16_as_ushort(__float2bfloat16_rn((int(random() >> 16) - 32768) / 32768.f));
  for (auto* v : {&pg, &pu})
    for (auto& x : *v) x = random() >> 24;
  for (auto* v : {&sg, &su})
    for (auto& x : *v) {
      x = random() >> 24;
      if ((x & 0x7f) == 0x7f) x &= 0xf0;  // finite E4M3, including signed zeros
    }
  float globals[2] = {3.1f, -1.7f};
  if (all_gate_bits) {
    std::fill(ha.begin(), ha.end(), 0);
    std::fill(pg.begin(), pg.end(), 0);
    std::fill(pu.begin(), pu.end(), 0);
    std::fill(sg.begin(), sg.end(), 0x38);
    std::fill(su.begin(), su.end(), 0x38);
    globals[0] = globals[1] = 1.f;
    for (int i = 0; i < m; ++i) ha[size_t(i) * stride] = uint16_t(i);
    for (int j = 0; j < n; ++j) {
      pg[size_t(j) * k / 2] = 2;  // gate = activation's first element
      pu[size_t(j) * k / 2] = 1;  // up = one half of that element
    }
  }
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  uint16_t *act, *gate, *up, *out;
  uint8_t *gate_p, *up_p, *gate_s, *up_s;
  float* global;
  DGPP_CUDA_OK(cudaMalloc(&act, ha.size() * 2));
  DGPP_CUDA_OK(cudaMalloc(&gate, mn * 2));
  DGPP_CUDA_OK(cudaMalloc(&up, mn * 2));
  DGPP_CUDA_OK(cudaMalloc(&out, out_size * 2));
  DGPP_CUDA_OK(cudaMalloc(&gate_p, pg.size()));
  DGPP_CUDA_OK(cudaMalloc(&up_p, pu.size()));
  DGPP_CUDA_OK(cudaMalloc(&gate_s, sg.size()));
  DGPP_CUDA_OK(cudaMalloc(&up_s, su.size()));
  DGPP_CUDA_OK(cudaMalloc(&global, sizeof(globals)));
  DGPP_CUDA_OK(cudaMemcpy(act, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(gate_p, pg.data(), pg.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(up_p, pu.data(), pu.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(gate_s, sg.data(), sg.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(up_s, su.data(), su.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(global, globals, sizeof(globals), cudaMemcpyHostToDevice));
  std::fill(got.begin(), got.end(), guard);
  DGPP_CUDA_OK(cudaMemcpy(out, got.data(), got.size() * 2, cudaMemcpyHostToDevice));
  const dgpp::GlmFp4Matrix gw{gate_p, gate_s, global, n, k, 16};
  const dgpp::GlmFp4Matrix uw{up_p, up_s, global + 1, n, k, 16};
  dgpp::launch_fp4w_gemm_bf16(act, stride, gw, gate, m, n, k, stream);
  dgpp::launch_fp4w_gemm_bf16(act, stride, uw, up, m, n, k, stream);
  dgpp::qwen35_swiglu_bf16(gate, up, gate, mn, stream);
  dgpp::launch_fp4w_gate_up_swiglu_bf16(act, stride, gw, uw, out + offset, m, n, k, stream,
                                        out_stride);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  DGPP_CUDA_OK(cudaMemcpy(ref.data(), gate, mn * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
  size_t mismatches = 0, bad_guards = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const size_t local = i >= size_t(offset) ? i - offset : size_t(-1);
    if (local / out_stride < size_t(m) && local % out_stride < size_t(n))
      mismatches += got[i] != ref[(local / out_stride) * n + local % out_stride];
    else
      bad_guards += got[i] != guard;
  }
  for (void* p : {static_cast<void*>(act), static_cast<void*>(gate), static_cast<void*>(up),
                  static_cast<void*>(out), static_cast<void*>(gate_p), static_cast<void*>(up_p),
                  static_cast<void*>(gate_s), static_cast<void*>(up_s), static_cast<void*>(global)})
    DGPP_CUDA_OK(cudaFree(p));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::printf(
      "pair m=%d n=%d k=%d pad=%d offset=%d all_gate_bits=%d: %zu bit mismatches, %zu guard "
      "errors\n",
      m, n, k, pad, offset, all_gate_bits, mismatches, bad_guards);
  if (mismatches || bad_guards)
    throw std::runtime_error("paired FP4 prefill differs from existing calls");
}
}  // namespace

int main() {
  try {
    // Partial M/N tiles and every scalar/packed store path, first so a bounded
    // sanitizer run covers each layout without instrumenting the large GEMMs.
    for (int pad : {0, 1, 3})
      for (int offset : {0, 1}) check(257, 65, 128, pad, offset);
    check(257, 65, 192, 1, 1);  // odd K-step count exercises the final pipeline iteration
    check(65536, 3, 64, 2, 1, true);  // every BF16 gate input, including infinities/NaNs
    check(1024, 137, 256, 0, 0);
    check(3073, 137, 5120, 1, 1);
    // Actual TP1/2/4 gate/up geometry; distinct payloads, scales and globals.
    check(1025, 17408, 5120, 0, 0);
    check(3073, 8704, 5120, 0, 0);
    check(4096, 4352, 5120, 0, 0);
    std::puts("qwen35_fp4_prefill_test: all passed");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
