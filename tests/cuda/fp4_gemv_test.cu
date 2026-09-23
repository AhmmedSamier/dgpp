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
  std::vector<uint8_t> scales;   // [n, k/16] e4m3, or [n, k/32] e8m0 under MXFP4
  float global = 1.0f;
  int group = 16;                // 16: NVFP4; 32: MXFP4 (e8m0 scales, no global)
};

// The value of one e8m0 scale byte (255 = NaN).
double e8m0_value(uint8_t code) {
  if (code == 255) return std::nan("");
  return std::ldexp(1.0, static_cast<int>(code) - 127);
}

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

// An MXFP4 problem: e8m0 scales spanning 2^(lo) .. 2^(hi) (exponent codes
// lo + 127 .. hi + 127), no global.
Problem make_problem_mx(int m, int n, int k, uint64_t seed, int lo = -10, int hi = 6) {
  Problem p;
  p.m = m;
  p.n = n;
  p.k = k;
  p.group = 32;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  fill_act(rng, p.act);
  p.payload.resize(static_cast<size_t>(n) * k / 2);
  for (auto& b : p.payload) b = static_cast<uint8_t>(rng.next() & 0xFF);
  p.scales.resize(static_cast<size_t>(n) * k / 32);
  for (auto& s : p.scales) {
    const int e = lo + static_cast<int>(rng.next() % static_cast<uint64_t>(hi - lo + 1));
    s = static_cast<uint8_t>(e + 127);
  }
  p.global = 1.0f;
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
      const double s = p.group == 32
                           ? e8m0_value(p.scales[static_cast<size_t>(nn) * p.k / 32 + kk / 32])
                           : static_cast<double>(dgpp::fp8_e4m3_bits_to_float(
                                 p.scales[static_cast<size_t>(nn) * p.k / 16 + kk / 16]));
      wrow[kk] = static_cast<double>(dgpp::fp4_e2m1_bits_to_float(code)) * s;
    }
    for (int mm = 0; mm < p.m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = p.act.data() + static_cast<size_t>(mm) * p.k;
      for (int kk = 0; kk < p.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      if (p.group == 16) acc /= static_cast<double>(p.global);
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
    view = dgpp::GlmFp4Matrix{payload, scales, p.group == 32 ? nullptr : global, p.n, p.k, p.group};
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
  q.group = p.group;
  return q;
}

}  // namespace

// ---- MXFP4 (2026-09-13, docs/deepseek_v41_flash_plan.md D2) ------------------

DGPP_TEST(fp4_gemv_mxfp4_matches_oracle_across_row_geometries) {
  // The MXFP4 compiled set: the power-of-two geometries and DeepSeek-V4.1's
  // widths — 5120 (32 lanes, five chunks in two passes), 576 / 1152 / 2304
  // (2 / 4 / 8 lanes x 9 chunks in three passes).
  struct Shape {
    int m, n, k;
  };
  const Shape shapes[] = {{1, 520, 1024}, {3, 264, 512},  {2, 100, 256}, {4, 40, 128},
                          {1, 33, 64},    {2, 17, 32},    {1, 390, 5120}, {3, 130, 5120},
                          {2, 520, 576},  {1, 77, 1152},  {4, 70, 2304},  {4, 300, 1024}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem_mx(s.m, s.n, s.k, 0x3F40 + i++);
    const std::string label = "mxfp4 gemv M" + std::to_string(s.m) + "xN" + std::to_string(s.n) +
                              "xK" + std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
}

DGPP_TEST(fp4_gemv_mxfp4_is_exact_beyond_the_f16_window) {
  // Scales from 2^-100 to 2^40: outside the range in which an f16 product
  // would be exact (the NVFP4 path's shortcut); the fp32 decode is exact
  // for every exponent, so the oracle budget holds unchanged.
  const Problem lo = make_problem_mx(2, 96, 1024, 0xE8, -100, -90);
  check_oracle(lo, run_bf16(lo), "mxfp4 scales 2^-100..2^-90");
  const Problem hi = make_problem_mx(2, 96, 1024, 0xE9, 30, 40);
  check_oracle(hi, run_bf16(hi), "mxfp4 scales 2^30..2^40");
  const Problem mixed = make_problem_mx(3, 64, 2304, 0xEA, -60, 20);
  check_oracle(mixed, run_bf16(mixed), "mxfp4 scales 2^-60..2^20");
}

DGPP_TEST(fp4_gemv_mxfp4_f16_window_boundaries_are_exact) {
  // The f16 fast path (fp4_gemv.cuh consume_chunk_mx, 2026-09-22) takes
  // exponent codes 104..140 and the exact path the rest: scales 2^-27 ..
  // 2^18 (codes 100..145) mix both, cross both boundaries and run
  // through the f16 subnormal scales (codes 104..112), every chunk exact
  // against the double oracle.
  for (int k : {1024, 4096, 512}) {
    const Problem p = make_problem_mx(2, 96, k, 0xEB + k, -27, 18);
    check_oracle(p, run_bf16(p), ("mxfp4 scales 2^-27..2^18 (k " + std::to_string(k) + ")").c_str());
  }
  const Problem sub = make_problem_mx(3, 64, 2304, 0xEC, -23, -15);
  check_oracle(sub, run_bf16(sub), "mxfp4 subnormal-f16 scales 2^-23..2^-15");
}

DGPP_TEST(fp4_gemv_mxfp4_rows_are_independent_of_row_count) {
  for (int k : {1024, 576, 5120, 2304}) {
    const Problem p8 = make_problem_mx(8, 136, k, 0x3E8 + k);
    const std::vector<uint16_t> got8 = run_bf16(p8);
    const std::vector<float> got8f = run_f32(p8);
    for (int r = 0; r < 8; ++r) {
      const Problem p1 = row_of(p8, r);
      const std::vector<uint16_t> got1 = run_bf16(p1);
      const std::vector<float> got1f = run_f32(p1);
      require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * 2) == 0,
              "mxfp4 chunked bf16 row bits independent of m (8)");
      require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * 4) == 0,
              "mxfp4 chunked f32 row bits independent of m (8)");
    }
    for (size_t i = 0; i < got8.size(); ++i)
      require(dgpp::float_to_bf16_bits(got8f[i]) == got8[i], "mxfp4 bf16(out_f32) == out_bf16");
  }
  std::printf("[ OK ] mxfp4 gemv rows independent of m through m=8 chunks\n");
}

DGPP_TEST(fp4_gemv_mxfp4_propagates_nan_scales_exactly) {
  Problem p = make_problem_mx(1, 300, 1024, 0x3A5);
  p.scales[static_cast<size_t>(9) * (p.k / 32) + 3] = 0xFF;
  p.scales[static_cast<size_t>(250) * (p.k / 32) + 0] = 0xFF;
  const std::vector<uint16_t> got = run_bf16(p);
  for (int nn = 0; nn < p.n; ++nn) {
    const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
    if (nn == 9 || nn == 250)
      require(got_nan, "poisoned e8m0 scale row is NaN");
    else
      require(!got_nan, "unpoisoned row stays finite");
  }
  std::printf("[ OK ] mxfp4 gemv nan propagation: rows 9 and 250 NaN\n");
}

DGPP_TEST(fp4_gemv_mxfp4_rejects_geometry_outside_its_set) {
  auto rejects = [](int k) {
    Problem p = make_problem_mx(1, 8, k, 0x3BAD);
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(48), "k=48 rejected (not a multiple of 32)");
  require(rejects(3072), "k=3072 rejected (not in the MXFP4 compiled set)");
  require(!rejects(576), "k=576 accepted (world-4 down)");
  require(!rejects(1152), "k=1152 accepted (world-2 down)");
  require(!rejects(2304), "k=2304 accepted (world-1 down)");
  // MiMo-V2.6-Flash's widths (2026-09-22): the hidden and the world-1 down.
  require(!rejects(4096), "k=4096 accepted (the MiMo gate/up)");
  require(!rejects(2048), "k=2048 accepted (the MiMo world-1 down)");
  // An NVFP4 view without its global is refused; an MXFP4 one needs none.
  Problem p = make_problem(1, 8, 1024, 0x3BAE);
  bool refused = false;
  try {
    DevMatrix w(p);
    dgpp::GlmFp4Matrix v = w.view;
    v.global_scale = nullptr;
    uint16_t* act = nullptr;
    uint16_t* out = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&out, 8 * 2));
    dgpp::launch_fp4_gemv_bf16(act, p.k, v, out, 1, 8, p.k, nullptr);
    cudaFree(act);
    cudaFree(out);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "an NVFP4 view without a global scale is refused");
  std::printf("[ OK ] mxfp4 gemv geometry contract enforced\n");
}

DGPP_TEST(fp4_gemv_matches_oracle_across_row_geometries) {
  // k picks the row geometry (four chunks per lane from k = 128 up): 4096
  // (32 lanes per row), 2048 (16), 1024 (8), 512 (4), 256 (2), 128 (1 lane,
  // 32 rows per step), 64 and 32 (one lane, 2 and 1 chunks); n leaves
  // partial blocks and dead lane groups.
  struct Shape {
    int m, n, k;
  };
  // The non-power-of-two widths (2026-09-09, GLM-4.7): 5120 (32 lanes,
  // five chunks per lane in two passes), 384 (4 lanes x 3 chunks), 768 (8 x
  // 3), 1536 (16 x 3), 3072 (32 x 3), 12288 (32 lanes x 12 chunks in three
  // passes).
  const Shape shapes[] = {{1, 520, 4096}, {3, 264, 1024}, {2, 100, 512},
                          {4, 40, 256},   {1, 33, 128},   {2, 17, 64},
                          {1, 9, 32},     {4, 512, 4096}, {2, 300, 2048},
                          {1, 390, 5120}, {3, 130, 5120}, {2, 520, 384},
                          {1, 77, 768},   {4, 70, 1536},  {2, 41, 3072},
                          {1, 20, 12288}};
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
  for (int k : {4096, 512, 5120, 384}) {
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
  require(rejects(48), "k=48 rejected (not a multiple of 32)");
  require(rejects(96), "k=96 rejected (not in the compiled set)");
  require(!rejects(1536), "k=1536 accepted (16 lanes x 3 chunks)");
  require(rejects(8192), "k=8192 rejected (not in the compiled set)");
  require(!rejects(2048), "k=2048 accepted");
  require(!rejects(5120), "k=5120 accepted");
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
