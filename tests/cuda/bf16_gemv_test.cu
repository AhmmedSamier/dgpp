// Parity tests for the bf16 decode GEMV (M6 Stage 2 round 3): the GEMM
// seam's m<=8 decode path (chunked above four rows), against an fp64 oracle
// over the same bf16
// operands. Exercises both outputs (bf16, f32), strided activation views
// (the KDA f_a/g_a K-column slices), ragged n, short and long k, and the
// row-independence property (row r's bits at m=1 == its bits in m=3/m=8).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemm.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  size_t act_stride;             // elements between activation rows
  std::vector<uint16_t> act;     // [m, act_stride] (k columns read)
  std::vector<uint16_t> weight;  // [n, k]
};

Problem make_problem(int m, int n, int k, size_t act_stride, uint64_t seed) {
  Problem p{m, n, k, act_stride, {}, {}};
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * act_stride);
  p.weight.resize(static_cast<size_t>(n) * k);
  fill_act(rng, p.act);
  fill_act(rng, p.weight);
  return p;
}

// fp64 dot over the bf16 operands: the strict oracle (fp64 accumulation).
std::vector<double> oracle(const Problem& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n);
  for (int r = 0; r < p.m; ++r)
    for (int c = 0; c < p.n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < p.k; ++kk)
        acc += static_cast<double>(bf16_to_float(p.act[r * p.act_stride + kk])) *
               static_cast<double>(bf16_to_float(p.weight[static_cast<size_t>(c) * p.k + kk]));
      out[static_cast<size_t>(r) * p.n + c] = acc;
    }
  return out;
}

struct Device {
  uint16_t* act = nullptr;
  uint16_t* w = nullptr;
  void* out = nullptr;
  explicit Device(const Problem& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&w, p.weight.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
    std::memcpy(act, p.act.data(), p.act.size() * 2);
    std::memcpy(w, p.weight.data(), p.weight.size() * 2);
  }
  ~Device() {
    cudaFree(act);
    cudaFree(w);
    cudaFree(out);
  }
};

// Runs through the SEAM (the production entry) and returns bf16 bits.
std::vector<uint16_t> run_bf16(dgpp::IGemm& gemm, const Problem& p,
                               const Device& d) {
  gemm.matmul(d.act, d.w, d.out, p.m, p.n, p.k, dgpp::DType::BF16,
              dgpp::GemmOut::BF16, p.act_stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), d.out, got.size() * 2);
  return got;
}

std::vector<float> run_f32(dgpp::IGemm& gemm, const Problem& p,
                           const Device& d) {
  gemm.matmul(d.act, d.w, d.out, p.m, p.n, p.k, dgpp::DType::BF16,
              dgpp::GemmOut::F32, p.act_stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), d.out, got.size() * 4);
  return got;
}

void check_bf16(const Problem& p, const std::vector<uint16_t>& got,
                const char* label) {
  const auto want = oracle(p);
  // 2 bf16 ULP against the fp64 oracle: the fp32 chain's error is far
  // inside one ULP; the budget covers double-rounding boundary cases.
  const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-2);
  if (rep.mismatches != 0)
    throw std::runtime_error(std::string(label) + ": bf16 out off the oracle");
  std::printf("[ OK ] %s bf16: l2_rel=%.3g mismatches=%ld/%ld\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
}

void check_f32(const Problem& p, const std::vector<float>& got,
               const char* label) {
  const auto want = oracle(p);
  double max_abs = 0.0;
  for (double v : want) max_abs = std::max(max_abs, std::abs(v));
  for (size_t i = 0; i < want.size(); ++i) {
    // fp32 chain over k products: k * 2^-24 relative to the magnitude scale.
    const double tol = 1e-6 * max_abs + 4e-6 * std::abs(want[i]) * std::sqrt(p.k);
    if (std::abs(got[i] - want[i]) > tol)
      throw std::runtime_error(std::string(label) + ": f32 out off the oracle");
  }
  std::printf("[ OK ] %s f32 within the fp32-chain budget\n", label);
}

}  // namespace

DGPP_TEST(bf16_gemv_real_shapes_match_oracle) {
  dgpp::CublasLtGemm gemm;
  // The KDA in_proj shape class (k = hidden), the o_proj class (k = local
  // proj), the DSA indexer wk (n = 128), the lm head slice at m=1.
  for (auto [m, n, k] : std::vector<std::tuple<int, int, int>>{
           {1, 1536, 4096}, {1, 4096, 1024}, {1, 128, 4096}, {1, 3000, 512}}) {
    const Problem p = make_problem(m, n, k, static_cast<size_t>(k), 0x5EED + n);
    Device d(p);
    const std::string label = "bf16_gemv M" + std::to_string(m) + "xN" +
                              std::to_string(n) + "xK" + std::to_string(k);
    check_bf16(p, run_bf16(gemm, p, d), label.c_str());
    // Determinism (graph replay premise).
    const auto again = run_bf16(gemm, p, d);
    const auto first = run_bf16(gemm, p, d);
    require(std::memcmp(again.data(), first.data(), first.size() * 2) == 0,
            "bf16 gemv deterministic");
  }
}

DGPP_TEST(bf16_gemv_f32_output_and_strided_activation_view) {
  dgpp::CublasLtGemm gemm;
  // GIVEN the DSA indexer-weights shape (F32 out, n = 32 heads) ...
  {
    const Problem p = make_problem(1, 32, 4096, 4096, 0xF32);
    Device d(p);
    check_f32(p, run_f32(gemm, p, d), "bf16_gemv f32 M1xN32xK4096");
  }
  // ... AND the KDA f_b view: k = head_dim columns read out of a fused
  // projection row of stride n_in (16B-aligned slice, wider stride).
  {
    const Problem p = make_problem(1, 1024, 128, 4352, 0xF33);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv strided M1xN1024xK128/4352");
  }
}

DGPP_TEST(bf16_gemv_ragged_n_short_k_and_multi_row) {
  dgpp::CublasLtGemm gemm;
  // GIVEN n not a multiple of the 8-row block and k = one chunk (8 elems):
  {
    const Problem p = make_problem(1, 1003, 8, 8, 0xA9);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv M1xN1003xK8");
  }
  // AND k past one batch (16 chunks per lane at 4096; 4224 = 16.5 chunks):
  {
    const Problem p = make_problem(1, 264, 4224, 4224, 0xAA);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv M1xN264xK4224");
  }
  // AND m=3: every row's bits equal its m=1 bits (row independence), m=4
  // matches the oracle.
  {
    const Problem p3 = make_problem(3, 520, 2048, 2048, 0xAB);
    Device d3(p3);
    const auto got3 = run_bf16(gemm, p3, d3);
    check_bf16(p3, got3, "bf16_gemv M3xN520xK2048");
    for (int r = 0; r < 3; ++r) {
      Problem p1{1, p3.n, p3.k, p3.act_stride, {}, p3.weight};
      p1.act.assign(p3.act.begin() + static_cast<long>(r) * p3.act_stride,
                    p3.act.begin() + static_cast<long>(r + 1) * p3.act_stride);
      Device d1(p1);
      const auto got1 = run_bf16(gemm, p1, d1);
      require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                          static_cast<size_t>(p3.n) * 2) == 0,
              "bf16 gemv row bits independent of m");
    }
    const Problem p4 = make_problem(4, 96, 1024, 1024, 0xAC);
    Device d4(p4);
    check_bf16(p4, run_bf16(gemm, p4, d4), "bf16_gemv M4xN96xK1024");
  }
  // AND the serving ceiling m=8 is lowered to deterministic GEMV chunks.
  // K=4096 means a single eight-row launch would exceed the default 48-KiB
  // smem ceiling, so this also pins output offsets between chunks. Both
  // epilogues must retain the scalar row bits.
  {
    const Problem p8 = make_problem(8, 136, 4096, 4352, 0xAD);
    Device d8(p8);
    const auto got8 = run_bf16(gemm, p8, d8);
    const auto got8f = run_f32(gemm, p8, d8);
    check_bf16(p8, got8, "bf16_gemv chunked M8xN136xK4096");
    check_f32(p8, got8f, "bf16_gemv chunked M8xN136xK4096");
    for (int r = 0; r < p8.m; ++r) {
      Problem p1{1, p8.n, p8.k, p8.act_stride, {}, p8.weight};
      p1.act.assign(p8.act.begin() + static_cast<long>(r) * p8.act_stride,
                    p8.act.begin() + static_cast<long>(r + 1) * p8.act_stride);
      Device d1(p1);
      const auto got1 = run_bf16(gemm, p1, d1);
      const auto got1f = run_f32(gemm, p1, d1);
      require(std::memcmp(got1.data(),
                          got8.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * sizeof(uint16_t)) == 0,
              "chunked bf16 row bits independent of m");
      require(std::memcmp(got1f.data(),
                          got8f.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * sizeof(float)) == 0,
              "chunked f32 row bits independent of m");
    }
  }
}

// The dual launch (the KDA layer's f_b/g_b pair) must be BITWISE the two
// single launches on both outputs, for every row count and both output
// types, including the strided-activation view and a ragged second n.
DGPP_TEST(bf16_gemv_dual_matches_two_single_launches_bitwise) {
  dgpp::CublasLtGemm gemm;
  for (int m = 1; m <= 4; ++m) {
    const Problem a = make_problem(m, 1024, 128, 4352, 0x0d0a + m);
    const Problem b = make_problem(m, 1000, 128, 4352, 0x0d0b + m);  // ragged n
    Device da(a), db(b);
    // GIVEN the two single-launch results (through the seam, bf16 and f32)
    const std::vector<uint16_t> want_a = run_bf16(gemm, a, da);
    const std::vector<uint16_t> want_b = run_bf16(gemm, b, db);
    const std::vector<float> want_a32 = run_f32(gemm, a, da);
    const std::vector<float> want_b32 = run_f32(gemm, b, db);
    // WHEN the same pair runs as one dual launch
    dgpp::Bf16GemvProblem p0, p1;
    p0.act = da.act; p0.act_row_stride = a.act_stride; p0.weight = da.w;
    p0.out = da.out; p0.n = a.n;
    p1.act = db.act; p1.act_row_stride = b.act_stride; p1.weight = db.w;
    p1.out = db.out; p1.n = b.n;
    dgpp::launch_bf16_gemv_dual(p0, p1, /*out_f32=*/false, m, a.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    // THEN both outputs are the single launches' bits
    if (std::memcmp(da.out, want_a.data(), want_a.size() * 2) != 0 ||
        std::memcmp(db.out, want_b.data(), want_b.size() * 2) != 0)
      throw std::runtime_error("dual bf16 GEMV differs from two launches (m=" +
                               std::to_string(m) + ")");
    dgpp::launch_bf16_gemv_dual(p0, p1, /*out_f32=*/true, m, a.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    if (std::memcmp(da.out, want_a32.data(), want_a32.size() * 4) != 0 ||
        std::memcmp(db.out, want_b32.data(), want_b32.size() * 4) != 0)
      throw std::runtime_error("dual f32 GEMV differs from two launches (m=" +
                               std::to_string(m) + ")");
  }
}

DGPP_TEST(bf16_gemv_contract_rejects_odd_k_and_falls_back) {
  // GIVEN k % 8 != 0: the seam must NOT take the GEMV (bf16_gemv_accepts
  // says so) — and the launcher refuses it outright.
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 1, 12),
          "k=12 outside the contract");
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 5, 64),
          "m=5 outside the contract");
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(8), 1, 64),
          "8B-aligned weight outside the contract");
  require(dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 4, 64),
          "m=4, k=64, 16B-aligned inside");
  bool threw = false;
  try {
    dgpp::launch_bf16_gemv(reinterpret_cast<const uint16_t*>(16), 12,
                           reinterpret_cast<const uint16_t*>(16),
                           reinterpret_cast<void*>(16), false, 1, 8, 12,
                           nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "launcher rejects k=12");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
