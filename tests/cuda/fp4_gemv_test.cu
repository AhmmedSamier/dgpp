// Parity tests for the NVFP4 GEMV core (docs/nvfp4_plan.md §5, gate 2):
// the single-matrix launcher against a double oracle across every row
// geometry the core has (32, 16, 8, 4, 2 and 1 lanes per row), the row-count
// invariance the batch family relies on, the f32 epilogue as the unrounded
// bf16, NaN scale propagation, the geometry contract, and — with
// --checkpoint-dir pointing at the composed hybrid — real expert slices.
//
// The oracle: w = e2m1(code) x e4m3(scale) exactly (no weight rounding —
// there is none in the engine), fp64 accumulation, the dot divided by the
// global scale, rounded to bf16 as the launcher's bf16 epilogue rounds.
// The kernel differs from it by fp32 accumulation order only, so the
// budget is the strict FP8 suite's: 2 bf16 ulps with a 1e-3 cancellation
// floor, zero mismatches.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/fp4_gemv.hpp"
#include "kernels/latent_format.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  std::vector<uint16_t> act;     // [m, k] bf16
  std::vector<uint8_t> payload;  // [n, k/2]
  std::vector<uint8_t> scales;   // [n, k/16] e4m3
  float global = 1.0f;
};

Problem make_problem(int m, int n, int k, uint64_t seed) {
  Problem p;
  p.m = m;
  p.n = n;
  p.k = k;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  fill_act(rng, p.act);
  p.payload.resize(static_cast<size_t>(n) * k / 2);
  for (auto& b : p.payload) b = static_cast<uint8_t>(rng.next() & 0xFF);
  p.scales.resize(static_cast<size_t>(n) * k / 16);
  for (auto& s : p.scales)
    s = dgpp::float_to_fp8_e4m3_bits(static_cast<float>(std::exp2(rng.unit() * 2.0)));
  p.global = static_cast<float>(std::exp2(rng.unit()));  // [0.5, 2)
  return p;
}

std::vector<double> oracle(const Problem& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n, 0.0);
  std::vector<double> wrow(p.k);
  for (int nn = 0; nn < p.n; ++nn) {
    for (int kk = 0; kk < p.k; ++kk) {
      const uint8_t byte = p.payload[static_cast<size_t>(nn) * p.k / 2 + kk / 2];
      const uint8_t code = (kk & 1) ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0xF);
      const float s = dgpp::fp8_e4m3_bits_to_float(
          p.scales[static_cast<size_t>(nn) * p.k / 16 + kk / 16]);
      wrow[kk] = static_cast<double>(dgpp::fp4_e2m1_bits_to_float(code)) *
                 static_cast<double>(s);
    }
    for (int mm = 0; mm < p.m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = p.act.data() + static_cast<size_t>(mm) * p.k;
      for (int kk = 0; kk < p.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      acc /= static_cast<double>(p.global);
      out[static_cast<size_t>(mm) * p.n + nn] =
          bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  }
  return out;
}

struct DevMatrix {
  uint8_t* payload = nullptr;
  uint8_t* scales = nullptr;
  float* global = nullptr;
  dgpp::GlmFp4Matrix view;
  explicit DevMatrix(const Problem& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&payload, p.payload.size()));
    DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size()));
    DGPP_CUDA_OK(cudaMallocManaged(&global, sizeof(float)));
    std::memcpy(payload, p.payload.data(), p.payload.size());
    std::memcpy(scales, p.scales.data(), p.scales.size());
    *global = p.global;
    view = dgpp::GlmFp4Matrix{payload, scales, global, p.n, p.k};
  }
  ~DevMatrix() {
    cudaFree(payload);
    cudaFree(scales);
    cudaFree(global);
  }
};

std::vector<uint16_t> run_bf16(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_fp4_gemv_bf16(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(act);
  cudaFree(out);
  return got;
}

std::vector<float> run_f32(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_fp4_gemv_f32(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  cudaFree(act);
  cudaFree(out);
  return got;
}

void check_oracle(const Problem& p, const std::vector<uint16_t>& got,
                  const char* label) {
  const auto want = oracle(p);
  const auto rep = compare_bf16_vs_oracle(got.data(), want, /*ulp_budget=*/2.0,
                                          /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s: l2_rel=%.3g mismatches=%ld/%zu\n", label, rep.l2_rel,
              rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, label);
}

Problem row_of(const Problem& p, int r) {
  Problem q;
  q.m = 1;
  q.n = p.n;
  q.k = p.k;
  q.act.assign(p.act.begin() + static_cast<long>(r) * p.k,
               p.act.begin() + static_cast<long>(r + 1) * p.k);
  q.payload = p.payload;
  q.scales = p.scales;
  q.global = p.global;
  return q;
}

}  // namespace

DGPP_TEST(fp4_gemv_matches_oracle_across_row_geometries) {
  // k picks the row geometry (four chunks per lane from k = 128 up): 4096
  // (32 lanes per row), 2048 (16), 1024 (8), 512 (4), 256 (2), 128 (1 lane,
  // 32 rows per step), 64 and 32 (one lane, 2 and 1 chunks); n leaves
  // partial blocks and dead lane groups.
  struct Shape {
    int m, n, k;
  };
  const Shape shapes[] = {{1, 520, 4096}, {3, 264, 1024}, {2, 100, 512},
                          {4, 40, 256},   {1, 33, 128},   {2, 17, 64},
                          {1, 9, 32},     {4, 512, 4096}, {2, 300, 2048}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(s.m, s.n, s.k, 0xF40 + i++);
    const std::string label = "fp4 gemv M" + std::to_string(s.m) + "xN" +
                              std::to_string(s.n) + "xK" + std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
}

DGPP_TEST(fp4_gemv_rows_are_independent_of_row_count) {
  // Each row's chain is the same sequence of FMAs whatever m: m=3 against
  // three m=1 runs, and the m=8 chunked launch against eight singles, for
  // both epilogues and for a short-row geometry too.
  for (int k : {4096, 512}) {
    const Problem p3 = make_problem(3, 296, k, 0xE3 + k);
    const std::vector<uint16_t> got3 = run_bf16(p3);
    for (int r = 0; r < 3; ++r) {
      const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
      require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                          static_cast<size_t>(p3.n) * 2) == 0,
              "fp4 row bits independent of m (3)");
    }
    const Problem p8 = make_problem(8, 136, k, 0xE8 + k);
    const std::vector<uint16_t> got8 = run_bf16(p8);
    const std::vector<float> got8f = run_f32(p8);
    for (int r = 0; r < 8; ++r) {
      const Problem p1 = row_of(p8, r);
      const std::vector<uint16_t> got1 = run_bf16(p1);
      const std::vector<float> got1f = run_f32(p1);
      require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * 2) == 0,
              "fp4 chunked bf16 row bits independent of m (8)");
      require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * 4) == 0,
              "fp4 chunked f32 row bits independent of m (8)");
    }
  }
  std::printf("[ OK ] fp4 gemv rows independent of m through m=8 chunks\n");
}

DGPP_TEST(fp4_gemv_f32_epilogue_is_the_unrounded_bf16) {
  const Problem p = make_problem(4, 200, 1024, 0xF32);
  const std::vector<uint16_t> b = run_bf16(p);
  const std::vector<float> f = run_f32(p);
  for (size_t i = 0; i < b.size(); ++i)
    require(dgpp::float_to_bf16_bits(f[i]) == b[i], "bf16(out_f32) == out_bf16");
  std::printf("[ OK ] fp4 gemv f32 epilogue rounds to the bf16 epilogue\n");
}

DGPP_TEST(fp4_gemv_propagates_nan_scales_exactly) {
  // Two poisoned scale codes: exactly their rows' outputs NaN.
  Problem p = make_problem(1, 300, 1024, 0xA5);
  p.scales[static_cast<size_t>(9) * (p.k / 16) + 3] = 0x7F;
  p.scales[static_cast<size_t>(250) * (p.k / 16) + 0] = 0xFF;
  const std::vector<uint16_t> got = run_bf16(p);
  for (int nn = 0; nn < p.n; ++nn) {
    const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
    if (nn == 9 || nn == 250)
      require(got_nan, "poisoned scale row is NaN");
    else
      require(!got_nan, "unpoisoned row stays finite");
  }
  std::printf("[ OK ] fp4 gemv nan propagation: rows 9 and 250 NaN\n");
}

DGPP_TEST(fp4_gemv_rejects_geometry_outside_the_contract) {
  auto rejects = [](int k) {
    Problem p = make_problem(1, 8, k, 0xBAD);
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(48), "k=48 rejected (not a power of two)");
  require(rejects(96), "k=96 rejected");
  require(rejects(1536), "k=1536 rejected");
  require(rejects(8192), "k=8192 rejected (above 4096)");
  require(!rejects(2048), "k=2048 accepted");
  std::printf("[ OK ] fp4 gemv geometry contract enforced\n");
}

// Real-checkpoint slice parity lives in fp4_gemv_checkpoint.cpp (host-only
// TU: the safetensors reader's JSON parser does not mix with nvcc).
int run_fp4_gemv_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_fp4_gemv_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
