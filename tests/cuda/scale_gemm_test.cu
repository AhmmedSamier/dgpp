// Parity tests for the scale-aware GEMM (M4 deliverable 3: "block edges,
// saturation, NaN/Inf policy, and real checkpoint slices" — the real-slice
// half runs via --checkpoint-dir, mirroring the dump-parity runner
// pattern). Every synthetic case is checked against BOTH oracles: strict
// (bf16-rounded weights, fp64 accumulation — isolates the kernel) and
// semantic (true dequant — pins the DESIGN §4 scale contract).
#include <cstring>
#include <cstdio>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/scale_gemm.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  std::vector<uint16_t> act;      // [m, k]
  std::vector<uint8_t> payload;   // [n, k]
  std::vector<float> scales;      // [ceil(n/128), ceil(k/128)]
};

Problem make_problem(int m, int n, int k, uint64_t seed) {
  Problem p;
  p.m = m;
  p.n = n;
  p.k = k;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  p.payload.resize(static_cast<size_t>(n) * k);
  p.scales.resize(static_cast<size_t>((n + 127) / 128) * ((k + 127) / 128));
  fill_act(rng, p.act);
  fill_payload(rng, p.payload);
  fill_scales(rng, p.scales);
  return p;
}

// Runs the kernel into managed memory and returns the bf16 output.
std::vector<uint16_t> run_kernel(const Problem& p) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  dgpp::launch_scale_gemm_bf16(act, p.k, w, s, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  return got;
}

// Runs both oracles and asserts the budgets.
void check_both_oracles(const Problem& p, const std::vector<uint16_t>& got,
                        const char* label) {
  const auto strict = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/true);
  auto rep = compare_bf16_vs_oracle(got.data(), strict, /*ulp_budget=*/2.0,
                                    /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s strict: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, (std::string(label) + " strict").c_str());

  const auto semantic = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m,
                                    p.n, p.k, /*strict=*/false);
  rep = compare_bf16_vs_oracle(got.data(), semantic, /*ulp_budget=*/8.0,
                               /*floor_frac=*/2e-2);
  std::printf("[ OK ] %s semantic: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 8e-3, static_cast<long>(rep.total) / 20,
                 (std::string(label) + " semantic").c_str());
}

}  // namespace

DGPP_TEST(scale_gemm_full_blocks_match_both_oracles) {
  // Exact 128-block geometry with n/k tiles straddling block boundaries:
  // N=384 spans block rows 0-2; K=256 crosses the k=128 boundary mid-loop.
  const Problem p = make_problem(29, 384, 256, 0xA1);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "full-blocks M29xN384xK256");

  // Determinism: a second run must be bitwise identical.
  const std::vector<uint16_t> again = run_kernel(p);
  require(std::memcmp(got.data(), again.data(), got.size() * 2) == 0,
          "second run bitwise identical");
}

DGPP_TEST(scale_gemm_ragged_tails_match_both_oracles) {
  // Nothing aligned: N=K=1000 (7 full blocks + 104-wide tail on both axes),
  // M=17 (ragged m-tile), and the final BK stage only 8 k-values wide.
  const Problem p = make_problem(17, 1000, 1000, 0xB2);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "ragged M17xN1000xK1000");
}

DGPP_TEST(scale_gemm_decode_shape_single_row_real_qa_geometry) {
  // The decode-critical shape: M=1 at the real q_a geometry [1536, 4096].
  const Problem p = make_problem(1, 1536, 4096, 0xC3);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "decode M1xN1536xK4096");
}

DGPP_TEST(scale_gemm_saturation_extremes) {
  // All-max finite payloads (0x7E = 448) with scales that push products to
  // ~1e35 (bf16/fp32 representable), and a subnormal-minimum case.
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x7E);
    p.scales.assign(2 * 2, 1e30f);
    Rng rng(0xD4);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "saturation 448x1e30");
  }
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    // 0x01 = min subnormal (2^-9); scales 2^-20 keep products representable
    // but exercise the bottom of both exponent ranges.
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x01);
    p.scales.assign(2 * 2, 9.53674316e-07f);  // 2^-20
    Rng rng(0xD5);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "subnormal min");
  }
}

DGPP_TEST(scale_gemm_propagates_nan_exactly) {
  // e4m3fn 0x7F is NaN. Two poisoned weight entries must NaN exactly the
  // output columns they touch — every m, through the fp32 accumulator —
  // and nothing else (zero-filled tiles and finite neighbors stay finite).
  Problem p = make_problem(13, 512, 384, 0xE6);
  p.payload[static_cast<size_t>(5) * p.k + 7] = 0x7F;
  p.payload[static_cast<size_t>(130) * p.k + 300] = 0x7F;
  const std::vector<uint16_t> got = run_kernel(p);

  const auto oracle = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/false);
  // NaN pattern must match element-for-element.
  for (int mm = 0; mm < p.m; ++mm) {
    for (int nn = 0; nn < p.n; ++nn) {
      const size_t i = static_cast<size_t>(mm) * p.n + nn;
      const bool want_nan = std::isnan(oracle[i]);
      const bool got_nan = std::isnan(bf16_to_float(got[i]));
      if (nn == 5 || nn == 130) {
        require(want_nan && got_nan, "poisoned column is NaN");
      } else {
        require(!want_nan && !got_nan, "unpoisoned column stays finite");
      }
    }
  }
  // Finite columns still within the semantic budget.
  auto rep = compare_bf16_vs_oracle(got.data(), oracle, 8.0, 2e-2);
  require(rep.nan_pattern_errors == 0, "nan pattern");
  require(rep.mismatches <= static_cast<long>(rep.total) / 20, "nan test mismatches");
  std::printf("[ OK ] nan propagation: columns 5 and 130 NaN, others finite\n");
}

DGPP_TEST(scale_gemm_degenerate_k_zeroes_output) {
  Problem p = make_problem(4, 64, 0, 0xF7);
  p.payload.clear();
  p.scales.clear();
  p.act.clear();
  p.act.resize(4);  // act unused for k=0
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, 8));
  DGPP_CUDA_OK(cudaMallocManaged(&out, 4 * 64 * 2));
  std::memset(out, 0xFF, 4 * 64 * 2);  // garbage
  dgpp::launch_scale_gemm_bf16(act, 0, reinterpret_cast<const uint8_t*>(act),
                               reinterpret_cast<const float*>(act), out, 4,
                               64, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (size_t i = 0; i < 4 * 64; ++i)
    if (out[i] != 0) throw std::runtime_error("k=0 must zero outputs");
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(out));
}

// Real-checkpoint slice parity lives in scale_gemm_checkpoint.cpp (host-only
// TU: minijson does not mix with nvcc).
int run_scale_gemm_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_scale_gemm_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
