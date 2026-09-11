#pragma once
// Host-side oracles and comparators for the scale-aware GEMM parity suite
// (M4 deliverable 3). Two oracles, each answering a different question:
//
//   strict    — weights BF16-rounded exactly as the kernel rounds them
//               (decode x scale in fp32, one BF16 round), fp64 accumulation.
//               Isolates the kernel itself: fragment layouts, tiling,
//               masked tails, accumulation order. A correct kernel is within
//               output-rounding of this oracle (~1 bf16 ulp).
//   semantic  — true dequantized weights (decode x scale in fp64, no BF16
//               rounding), fp64 accumulation. Measures the bf16-weight
//               POLICY against DESIGN §4's dequant contract; expected
//               deviation ~2^-8 relative per element from weight rounding.
//
// NaN policy: e4m3fn 0x7F decodes to NaN; it propagates through the fp32
// accumulator exactly as through the oracle (0 x NaN = NaN included). The
// comparator requires the NaN pattern to match element-for-element.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"

namespace scale_gemm_test {

// Deterministic xorshift64; independent per seed.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() {  // [-1, 1)
    // next()>>11 is a 53-bit unsigned value in [0, 2^53); map to [-1, 1).
    // (An earlier version divided by 2^52 without the shift of the range,
    // silently producing [0, 1) — all-positive test data. The router
    // parity's suspicious "max rel err 0" exposed it: sigmoid-saturated
    // scores made every weight exactly 2.5/top_k.)
    return static_cast<double>(next() >> 11) /
               static_cast<double>(1ull << 52) - 1.0;
  }
};

inline float bf16_to_float(uint16_t v) {
  return dgpp::bf16_bits_to_float(v);
}

// Step size of bf16 at |x| (7 explicit mantissa bits).
inline double bf16_ulp(double x) {
  if (x == 0.0) return std::ldexp(1.0, -133);  // min subnormal
  int e = 0;
  std::frexp(x, &e);  // |x| = f * 2^e, f in [0.5, 1)
  return std::ldexp(1.0, e - 8);
}

// Element fails when |diff| exceeds both the ulp budget at the oracle's
// magnitude AND the cancellation floor (a fraction of max |oracle| — sums
// that nearly cancel have huge relative error but tiny absolute error, and
// the floor is what keeps the check honest about that).
struct CompareReport {
  double l2_rel = 0;
  long mismatches = 0;
  size_t total = 0;
  long nan_pattern_errors = 0;
};

inline CompareReport compare_bf16_vs_oracle(const uint16_t* got,
                                            const std::vector<double>& oracle,
                                            double ulp_budget,
                                            double floor_frac) {
  CompareReport rep;
  rep.total = oracle.size();
  double max_abs = 0;
  for (double o : oracle)
    if (!std::isnan(o)) max_abs = std::max(max_abs, std::abs(o));
  const double floor_abs = floor_frac * max_abs;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < oracle.size(); ++i) {
    const double o = oracle[i];
    const double g = bf16_to_float(got[i]);
    if (std::isnan(o) || std::isnan(g)) {
      if (std::isnan(o) != std::isnan(g)) {
        ++rep.nan_pattern_errors;
        ++rep.mismatches;
      }
      continue;
    }
    const double d = std::abs(g - o);
    sum_d2 += d * d;
    sum_o2 += o * o;
    if (d > ulp_budget * bf16_ulp(o) && d > floor_abs) ++rep.mismatches;
  }
  rep.l2_rel = sum_o2 > 0 ? std::sqrt(sum_d2 / sum_o2) : 0.0;
  return rep;
}

// D[m, n] = sum_k act[m, k] * W[n, k]; W from payload x scales, either
// bf16-rounded (strict) or exact (semantic). act is bf16 values. The strict
// oracle also rounds its RESULT to bf16: both sides then speak bf16, and
// the comparison isolates accumulation-order differences instead of
// measuring output-rounding noise.
inline std::vector<double> gemm_oracle(
    const std::vector<uint16_t>& act, size_t act_stride,
    const std::vector<uint8_t>& payload, const std::vector<float>& scales,
    int m, int n, int k, bool strict) {
  const int scale_cols = (k + 127) / 128;
  std::vector<double> out(static_cast<size_t>(m) * n, 0.0);
  std::vector<double> wrow(k);
  for (int nn = 0; nn < n; ++nn) {
    const float* srow = scales.data() + static_cast<size_t>(nn / 128) *
                                            scale_cols;
    for (int kk = 0; kk < k; ++kk) {
      const float dec = dgpp::fp8_e4m3_bits_to_float(
          payload[static_cast<size_t>(nn) * k + kk]);
      if (strict) {
        // Reproduce the kernel's weight bits exactly: fp32 multiply, then
        // one bf16 round, evaluated back to double.
        const uint16_t bits =
            dgpp::float_to_bf16_bits(dec * srow[kk / 128]);
        wrow[kk] = bf16_to_float(bits);
      } else {
        wrow[kk] = static_cast<double>(dec) *
                   static_cast<double>(srow[kk / 128]);
      }
    }
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0;
      const uint16_t* arow = act.data() + static_cast<size_t>(mm) * act_stride;
      for (int kk = 0; kk < k; ++kk)
        acc += static_cast<double>(bf16_to_float(arow[kk])) * wrow[kk];
      if (strict)
        acc = bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
      out[static_cast<size_t>(mm) * n + nn] = acc;
    }
  }
  return out;
}

// Fillers producing adversarial-but-representable data.
inline void fill_act(Rng& rng, std::vector<uint16_t>& act) {
  for (auto& v : act)
    v = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit()));
}

inline void fill_payload(Rng& rng, std::vector<uint8_t>& payload) {
  for (auto& v : payload) {
    const float x = 3.0f * static_cast<float>(rng.unit());
    v = dgpp::float_to_fp8_e4m3_bits(x);
  }
}

inline void fill_scales(Rng& rng, std::vector<float>& scales) {
  for (auto& v : scales)
    v = static_cast<float>(std::exp2(rng.unit() * 2.0));  // [0.5, 2)
}

inline void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

inline void require_report(const CompareReport& rep, double l2_budget,
                           long mismatch_budget, const char* what) {
  require(rep.nan_pattern_errors == 0, what);
  require(rep.mismatches <= mismatch_budget, what);
  require(rep.l2_rel <= l2_budget, what);
}

}  // namespace scale_gemm_test
