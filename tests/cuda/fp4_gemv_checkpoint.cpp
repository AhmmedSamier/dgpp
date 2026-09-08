// Real-checkpoint slice parity for the NVFP4 GEMV (docs/nvfp4_plan.md gate
// 2, the "real slices" half): mmaps one routed expert's triple out of the
// composed hybrid (`fp4_gemv_test --checkpoint-dir <snapshot>`), runs the
// launcher at m = 48 against the double oracle, and checks that single rows
// reproduce the batched rows bit for bit. Host-only TU (the safetensors
// reader's JSON parser does not mix with nvcc).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/fp4_gemv.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/safetensors.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Slice {
  int n = 0, k = 0;
  std::vector<uint8_t> payload, scales;
  float global = 1.0f;
};

std::vector<double> oracle(const Slice& s, const std::vector<uint16_t>& act, int m) {
  std::vector<double> out(static_cast<size_t>(m) * s.n, 0.0);
  std::vector<double> wrow(s.k);
  for (int nn = 0; nn < s.n; ++nn) {
    for (int kk = 0; kk < s.k; ++kk) {
      const uint8_t byte = s.payload[static_cast<size_t>(nn) * s.k / 2 + kk / 2];
      const uint8_t code = (kk & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0xF);
      const float sc = dgpp::fp8_e4m3_bits_to_float(
          s.scales[static_cast<size_t>(nn) * s.k / 16 + kk / 16]);
      wrow[kk] = static_cast<double>(dgpp::fp4_e2m1_bits_to_float(code)) *
                 static_cast<double>(sc);
    }
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = act.data() + static_cast<size_t>(mm) * s.k;
      for (int kk = 0; kk < s.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      acc /= static_cast<double>(s.global);
      out[static_cast<size_t>(mm) * s.n + nn] =
          bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  }
  return out;
}

std::vector<uint16_t> run(const Slice& s, const std::vector<uint16_t>& act, int m,
                          int mma = 0) {
  uint8_t* payload = nullptr;
  uint8_t* scales = nullptr;
  float* global = nullptr;
  uint16_t* d_act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&payload, s.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&scales, s.scales.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&global, 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_act, act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(m) * s.n * 2));
  std::memcpy(payload, s.payload.data(), s.payload.size());
  std::memcpy(scales, s.scales.data(), s.scales.size());
  *global = s.global;
  std::memcpy(d_act, act.data(), act.size() * 2);
  const dgpp::GlmFp4Matrix w{payload, scales, global, s.n, s.k};
  if (mma == 1) {
    dgpp::launch_dense_mma_fp4_bf16(d_act, s.k, w, out, m, s.n, s.k, nullptr);
  } else if (mma == 2) {
    // The pipelined grouped kernel over one segment (the prefill's path).
    dgpp::MoeSegment* seg = nullptr;
    dgpp::MoeExpertView* views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&seg, sizeof(dgpp::MoeSegment)));
    DGPP_CUDA_OK(cudaMallocManaged(&views, 3 * sizeof(dgpp::MoeExpertView)));
    *seg = dgpp::MoeSegment{0, m, 0};
    for (int i = 0; i < 3; ++i) views[i] = dgpp::MoeExpertView::of(w);
    dgpp::launch_moe_grouped_mma_fp4_bf16(d_act, s.k, seg, 1, m, /*rows_per_block=*/0,
                                          views, 0, out, s.n, s.n, s.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    cudaFree(seg);
    cudaFree(views);
  } else {
    dgpp::launch_fp4_gemv_bf16(d_act, s.k, w, out, m, s.n, s.k, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(m) * s.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(payload);
  cudaFree(scales);
  cudaFree(global);
  cudaFree(d_act);
  cudaFree(out);
  return got;
}

}  // namespace

int run_fp4_gemv_checkpoint_parity(const char* checkpoint_dir) {
  namespace fs = std::filesystem;
  std::vector<fs::path> paths;
  for (const auto& e : fs::directory_iterator(checkpoint_dir))
    if (e.path().extension() == ".safetensors") paths.push_back(e.path());
  std::sort(paths.begin(), paths.end());
  const std::string bases[2] = {
      "model.language_model.layers.5.mlp.experts.7.gate_proj.",
      "model.language_model.layers.20.mlp.experts.100.down_proj."};
  const char* suffix[3] = {"weight_packed", "weight_scale", "weight_global_scale"};
  for (const std::string& base : bases) {
    Slice s;
    bool have[3] = {false, false, false};
    std::vector<std::unique_ptr<dgpp::SafetensorsFile>> keep;
    for (const auto& p : paths) {
      auto f = dgpp::SafetensorsFile::open(p.string());
      bool hit = false;
      for (int i = 0; i < 3; ++i) {
        const dgpp::TensorInfo* t = f->find(base + suffix[i]);
        if (!t) continue;
        hit = have[i] = true;
        const uint8_t* d = static_cast<const uint8_t*>(t->data);
        if (i == 0) {
          s.n = static_cast<int>(t->shape[0]);
          s.k = static_cast<int>(t->shape[1] * 2);
          s.payload.assign(d, d + t->nbytes());
        } else if (i == 1) {
          s.scales.assign(d, d + t->nbytes());
        } else {
          std::memcpy(&s.global, d, 4);
        }
      }
      if (hit) keep.push_back(std::move(f));
      if (have[0] && have[1] && have[2]) break;
    }
    if (!(have[0] && have[1] && have[2])) {
      std::printf("fp4 checkpoint parity: %s* not found under %s\n", base.c_str(),
                  checkpoint_dir);
      return 2;
    }
    const int m = 48;
    Rng rng(0xC0FFEE);
    std::vector<uint16_t> act(static_cast<size_t>(m) * s.k);
    fill_act(rng, act);
    std::printf("fp4 checkpoint parity: %s [%d x %d] global %.6g, m=%d\n",
                base.c_str(), s.n, s.k, s.global, m);
    const std::vector<uint16_t> got = run(s, act, m);
    const auto want = oracle(s, act, m);
    const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-3);
    std::printf("[ OK ] real slice: l2_rel=%.3g mismatches=%ld/%zu\n", rep.l2_rel,
                rep.mismatches, rep.total);
    require_report(rep, 1e-3, 0, "fp4 real slice vs oracle");
    for (int r : {0, 17, 47}) {
      std::vector<uint16_t> row(act.begin() + static_cast<long>(r) * s.k,
                                act.begin() + static_cast<long>(r + 1) * s.k);
      const std::vector<uint16_t> got1 = run(s, row, 1);
      require(std::memcmp(got1.data(), got.data() + static_cast<size_t>(r) * s.n,
                          static_cast<size_t>(s.n) * 2) == 0,
              "real slice: row bits independent of m");
    }
    // The prefill's tensor-core form on the same slice (phase 3): within
    // the same budget of the oracle, and a row's bits independent of m.
    const std::vector<uint16_t> got_mma = run(s, act, m, /*mma=*/1);
    const auto rep_mma = compare_bf16_vs_oracle(got_mma.data(), want, 2.0, 1e-3);
    std::printf("[ OK ] real slice, tensor-core reference: l2_rel=%.3g mismatches=%ld/%zu\n",
                rep_mma.l2_rel, rep_mma.mismatches, rep_mma.total);
    require_report(rep_mma, 1e-3, 0, "fp4 real slice (mma) vs oracle");
    for (int r : {0, 17, 47}) {
      std::vector<uint16_t> row(act.begin() + static_cast<long>(r) * s.k,
                                act.begin() + static_cast<long>(r + 1) * s.k);
      const std::vector<uint16_t> got1 = run(s, row, 1, /*mma=*/1);
      require(std::memcmp(got1.data(), got_mma.data() + static_cast<size_t>(r) * s.n,
                          static_cast<size_t>(s.n) * 2) == 0,
              "real slice (mma): row bits independent of m");
    }
    // The pipelined grouped kernel: bitwise the reference on the real slice,
    // at m = 48 and at m = 1.
    const std::vector<uint16_t> got_pipe = run(s, act, m, /*mma=*/2);
    require(std::memcmp(got_pipe.data(), got_mma.data(), got_pipe.size() * 2) == 0,
            "real slice: pipelined grouped kernel bitwise the tile reference");
    for (int r : {0, 17, 47}) {
      std::vector<uint16_t> row(act.begin() + static_cast<long>(r) * s.k,
                                act.begin() + static_cast<long>(r + 1) * s.k);
      const std::vector<uint16_t> got1 = run(s, row, 1, /*mma=*/2);
      require(std::memcmp(got1.data(), got_mma.data() + static_cast<size_t>(r) * s.n,
                          static_cast<size_t>(s.n) * 2) == 0,
              "real slice (pipelined): row bits independent of m");
    }
    std::printf("[ OK ] real slice, pipelined grouped kernel bitwise the reference\n");
  }
  std::printf("[ OK ] fp4 real-checkpoint slice parity (2 tensors)\n");
  return 0;
}
