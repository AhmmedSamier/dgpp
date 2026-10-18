// The Dory nGPT normalizations (kernels/dory_norm.hpp) against the host
// reference transcribed from the vLLM port (models/dory/reference.cpp), over
// random rows of every width the model uses (256 per-head, 4096 the
// attention output, 7168 the MLP intermediate) and a few magnitude profiles
// (uniform, 1e3 outliers, 1e-3 floors, near-zero rows, a wide exponent
// range), with the learned-alpha residual at several scales.
//
// TWO COMPARISONS, because the two oracles answer different questions.
//
// 1. Against the fp64-exact oracle below — BITWISE, every value. The kernel
//    reduces in fp64 because every bf16 square is exact there and the partial
//    sums of a model-sized row stay exact too, so the sum is the exact one and
//    `float(sum) + eps` rounds once. An oracle that sums the same terms in
//    fp64 therefore has the same rstd bit for bit, whatever order either
//    uses, and pins the kernel's arithmetic — the two roundings of the slerp
//    combine included — exactly.
// 2. Against the transcription — the reference's own sequential fp32 fma
//    chain, whose sum differs from the exact one by ~sqrt(dim)·2^-24 relative
//    (2048 additions of rounding noise), so the kernel's outputs drift from
//    it by that much. The law is absolute and scaled to the case: the largest
//    difference must be under 2% of the case's largest magnitude, and at most
//    1% of the values may differ at all. The observed drift is ~0.03% of the
//    values (0.016% at dim 256, 0.035% at 2048/4096, none at dim 8). A ulp
//    distance is deliberately NOT the metric here: the residual's
//    `na + lr·(nb − na)` crosses zero, and near zero a bf16 ulp is vanishingly
//    small in absolute terms (a 38% relative difference at a value of ~1e-5 is
//    a 1e-8 absolute one). This still fails on any real semantic error, which
//    would move most of the values, not a thousandth of them.
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dory_norm.hpp"
#include "models/dory/reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}

std::vector<uint16_t> random_rows(std::mt19937& rng, int rows, int dim, int profile) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * dim);
  std::normal_distribution<float> n01(0.f, 1.f);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (int r = 0; r < rows; ++r)
    for (int d = 0; d < dim; ++d) {
      float x = n01(rng);
      switch (profile) {
        case 1: if (u01(rng) < 0.002f) x *= 1000.f; break;      // outliers
        case 2: x *= (u01(rng) < 0.5f ? 1e-3f : 1.f); break;    // a floor of tiny values
        case 3: x *= 1e-4f; break;                              // near-zero rows
        case 4: x = (u01(rng) < 0.001f) ? x * 3e4f : x * 1e-5f; break;  // wide exponent range
        default: break;
      }
      v[static_cast<size_t>(r) * dim + d] = dgpp::float_to_bf16_bits(x);
    }
  return v;
}

// Monotone key of a bf16 bit pattern, so "one ulp apart" is |key difference|
// <= 1 across the sign boundary too.
int32_t key(uint16_t bits) {
  return (bits & 0x8000u) ? -static_cast<int32_t>(bits & 0x7fffu) : static_cast<int32_t>(bits);
}

struct Diff {
  size_t exact_mismatch = 0;  // any difference
  size_t beyond_1ulp = 0;     // informational: a ulp distance is meaningless near zero
  double max_abs = 0.0;       // largest absolute difference
  double ref_scale = 0.0;     // largest |reference| in the case
};

Diff compare(const std::vector<uint16_t>& ref, const std::vector<uint16_t>& got) {
  Diff d;
  require(ref.size() == got.size(), "size");
  for (size_t i = 0; i < ref.size(); ++i) {
    const float x = dgpp::bf16_bits_to_float(ref[i]), y = dgpp::bf16_bits_to_float(got[i]);
    d.ref_scale = std::max(d.ref_scale, static_cast<double>(std::fabs(x)));
    if (ref[i] == got[i]) continue;
    ++d.exact_mismatch;
    if (std::abs(key(ref[i]) - key(got[i])) > 1) ++d.beyond_1ulp;
    d.max_abs = std::max(d.max_abs, static_cast<double>(std::fabs(x - y)));
  }
  return d;
}

// The transcription's law (file comment): the kernel must track the reference
// to well inside the case's own magnitude, with almost every value identical.
void check_transcription(const char* what, const Diff& d, size_t n) {
  require(d.max_abs <= 2e-2 * d.ref_scale + 1e-30,
          std::string(what) + ": difference beyond the reference's own reduction error");
  require(static_cast<double>(d.exact_mismatch) <= 1e-2 * static_cast<double>(n) + 4,
          std::string(what) + ": too many values differ for a reduction-order difference");
}

// The kernel's contract with the reference's interior difference removed: the
// sums run in fp64, so no order matters and a bitwise match is the right
// criterion. Same expressions, same roundings.
namespace exact_oracle {

float rstd(const uint16_t* x, int dim, float eps) {
  double ss = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = dgpp::bf16_bits_to_float(x[i]);
    ss = std::fma(v, v, ss);
  }
  return 1.0f / std::sqrt(static_cast<float>(ss) + eps);
}

void justnorm(const uint16_t* x, uint16_t* y, int rows, int dim, float eps) {
  for (int r = 0; r < rows; ++r) {
    const uint16_t* xr = x + static_cast<size_t>(r) * dim;
    uint16_t* yr = y + static_cast<size_t>(r) * dim;
    const float rd = rstd(xr, dim, eps);
    for (int i = 0; i < dim; ++i)
      yr[i] = dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(xr[i]) * rd);
  }
}

void slerp_alpha(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale, uint16_t* y,
                 int rows, int dim, float eps) {
  std::vector<uint16_t> na(static_cast<size_t>(dim)), nb(static_cast<size_t>(dim));
  for (int r = 0; r < rows; ++r) {
    const uint16_t* ar = a + static_cast<size_t>(r) * dim;
    const uint16_t* br = b + static_cast<size_t>(r) * dim;
    uint16_t* yr = y + static_cast<size_t>(r) * dim;
    justnorm(ar, na.data(), 1, dim, eps);
    justnorm(br, nb.data(), 1, dim, eps);
    for (int i = 0; i < dim; ++i) {
      const float x = dgpp::bf16_bits_to_float(na[static_cast<size_t>(i)]);
      const float lr = std::fabs(dgpp::bf16_bits_to_float(alpha[i]) * scale);
      const float d = dgpp::bf16_bits_to_float(nb[static_cast<size_t>(i)]) - x;
      yr[i] = dgpp::float_to_bf16_bits(x + lr * d);
    }
    const float rd = rstd(yr, dim, eps);
    for (int i = 0; i < dim; ++i)
      yr[i] = dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(yr[i]) * rd);
  }
}

}  // namespace exact_oracle

void justnorm_case(int rows, int dim, int profile, float eps, std::mt19937& rng) {
  const std::vector<uint16_t> x = random_rows(rng, rows, dim, profile);
  std::vector<uint16_t> expect(x.size()), exact(x.size());
  dgpp::dory_ref::justnorm(x.data(), expect.data(), rows, dim, eps);
  exact_oracle::justnorm(x.data(), exact.data(), rows, dim, eps);
  DevBuf d_x = up(x), d_y(x.size() * 2);
  dgpp::dory_justnorm_bf16(static_cast<const uint16_t*>(d_x.p), static_cast<uint16_t*>(d_y.p), rows, dim,
                           eps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, x.size());
  const Diff de = compare(exact, got), dr = compare(expect, got);
  std::printf("  [justnorm] dim %d profile %d: exact %zu, transcription %zu/%zu differ (%.4f%%), max abs %.2e "
              "of %.2e\n",
              dim, profile, de.exact_mismatch, dr.exact_mismatch, static_cast<size_t>(rows) * dim,
              100.0 * static_cast<double>(dr.exact_mismatch) / (static_cast<double>(rows) * dim), dr.max_abs,
              dr.ref_scale);
  require(de.exact_mismatch == 0, "justnorm: differs from the fp64-exact oracle");
  check_transcription("justnorm", dr, x.size());
}

void slerp_case(int rows, int dim, float scale, float alpha_spread, std::mt19937& rng) {
  const std::vector<uint16_t> a = random_rows(rng, rows, dim, 0);
  const std::vector<uint16_t> b = random_rows(rng, rows, dim, 1);
  std::vector<uint16_t> alpha(static_cast<size_t>(dim));
  std::normal_distribution<float> n01(0.f, alpha_spread);
  for (auto& e : alpha) e = dgpp::float_to_bf16_bits(n01(rng));
  std::vector<uint16_t> expect(a.size()), exact(a.size());
  dgpp::dory_ref::slerp_alpha(a.data(), b.data(), alpha.data(), scale, expect.data(), rows, dim,
                              dgpp::kDoryJustnormEps);
  exact_oracle::slerp_alpha(a.data(), b.data(), alpha.data(), scale, exact.data(), rows, dim,
                            dgpp::kDoryJustnormEps);
  DevBuf d_a = up(a), d_b = up(b), d_al = up(alpha), d_y(a.size() * 2);
  dgpp::dory_slerp_alpha_bf16(static_cast<const uint16_t*>(d_a.p), static_cast<const uint16_t*>(d_b.p),
                              static_cast<const uint16_t*>(d_al.p), scale, static_cast<uint16_t*>(d_y.p),
                              rows, dim, dgpp::kDoryJustnormEps, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
  const auto got = down<uint16_t>(d_y, a.size());
  const Diff de = compare(exact, got), dr = compare(expect, got);
  std::printf("  [slerp] dim %d scale %g spread %g: exact %zu, transcription %zu/%zu differ (%.4f%%), max abs "
              "%.2e of %.2e\n",
              dim, scale, alpha_spread, de.exact_mismatch, dr.exact_mismatch,
              static_cast<size_t>(rows) * dim,
              100.0 * static_cast<double>(dr.exact_mismatch) / (static_cast<double>(rows) * dim), dr.max_abs,
              dr.ref_scale);
  require(de.exact_mismatch == 0, "slerp_alpha: differs from the fp64-exact oracle");
  check_transcription("slerp_alpha", dr, a.size());
}

}  // namespace

DGPP_TEST(dory_norm_justnorm_matches_the_reference) {
  std::mt19937 rng(20261018);
  for (int dim : {8, 256, 512, 2048, 4096, 7168, 8192})
    for (int profile = 0; profile < 5; ++profile) justnorm_case(dim == 4096 ? 256 : 48, dim, profile, 1e-6f, rng);
  justnorm_case(37, 4096, 0, 1e-5f, rng);
}

DGPP_TEST(dory_norm_slerp_alpha_matches_the_reference) {
  std::mt19937 rng(20261019);
  // lr = |alpha · alpha_scale| with alpha_scale = ngpt_alpha_init /
  // initializer_range (0.05/0.0177 ~= 2.8249), the model's own scale; 0 and
  // 1e-3 are the lr -> 0 and lr -> 0+ limits, 1.0 the identity.
  for (int dim : {256, 4096, 7168})
    for (float scale : {0.0f, 1e-3f, 1.0f, 2.8249f})
      for (float spread : {0.3f, 3.0f}) slerp_case(dim == 4096 ? 256 : 48, dim, scale, spread, rng);
  slerp_case(31, 7168, 2.8249f, 0.3f, rng);
  // lr an order of magnitude past anything the model asks for: the combine's
  // amplification is real (it scales a one-ulp flip in na/nb by lr), and the
  // exact-oracle check still holds bitwise there.
  for (int dim : {256, 4096})
    for (float scale : {10.0f, 100.0f}) slerp_case(48, dim, scale, 3.0f, rng);
}

DGPP_TEST(dory_norm_rejects_bad_shapes) {
  DevBuf a(2 * 8192), b(2 * 8192);
  auto* ap = static_cast<uint16_t*>(a.p);
  auto* bp = static_cast<uint16_t*>(b.p);
  for (int dim : {12, 20, 8200}) {
    bool threw = false;
    try {
      dgpp::dory_justnorm_bf16(ap, bp, 1, dim, 1e-6f, nullptr);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "justnorm accepted dim " + std::to_string(dim));
    threw = false;
    try {
      dgpp::dory_slerp_alpha_bf16(ap, bp, bp, 1.0f, ap, 1, dim, 1e-6f, nullptr);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "slerp_alpha accepted dim " + std::to_string(dim));
  }
  bool threw = false;
  try {
    dgpp::dory_justnorm_bf16(nullptr, bp, 1, 256, 1e-6f, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "justnorm accepted a null input");
}

DGPP_TEST(dory_norm_timing) {
  const int rows = 1, dim = 4096;
  std::mt19937 rng(7);
  DevBuf x = up(random_rows(rng, rows, dim, 0)), y = up(random_rows(rng, rows, dim, 0));
  DevBuf w = up(random_rows(rng, 1, dim, 0)), o(static_cast<size_t>(rows) * dim * 2);
  cudaStream_t s = nullptr;
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  auto time = [&](auto fn) {
    for (int i = 0; i < 20; ++i) fn();
    float best = 1e30f;
    for (int i = 0; i < 200; ++i) {
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      fn();
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms);
    }
    return best * 1e3f;
  };
  const float one_row = time([&] {
    dgpp::dory_justnorm_bf16(static_cast<const uint16_t*>(x.p), static_cast<uint16_t*>(o.p), rows, dim,
                             1e-6f, s);
  });
  const float residual = time([&] {
    dgpp::dory_slerp_alpha_bf16(static_cast<const uint16_t*>(x.p), static_cast<const uint16_t*>(y.p),
                                static_cast<const uint16_t*>(w.p), 2.8249f, static_cast<uint16_t*>(o.p),
                                rows, dim, 1e-6f, s);
  });
  std::printf("  [timing] 1 x %d: justnorm %.1f us, slerp_alpha (fused residual) %.1f us\n", dim, one_row,
              residual);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
