#pragma once
// Deterministic exp/log, BITWISE identical on the host and the device.
//
// The sampler's semantics (models/glm_sampler.hpp) are defined by their
// arithmetic, and the device verdict kernel must reproduce the host oracle
// bit for bit — including libm's share of it. glibc's exp/log and CUDA's
// are different implementations with different last bits (and glibc's
// differ between versions, i.e. between fabric nodes), so the sampler uses
// these instead: Cody-Waite reduction plus fixed Taylor/atanh polynomials,
// every multiply-add an EXPLICIT fma (never a contraction the compiler may
// or may not perform), every other step a single IEEE round-to-nearest
// operation or an exact one (ldexp/frexp, doubling, bit casts). The host
// builds with -ffp-contract=off and the device side names fma() itself, so
// neither side can fuse or split anything here. Accuracy is a few ulps —
// more than a sampler needs; determinism is the contract, pinned by
// det_math_test (host == device bitwise over millions of inputs, and both
// within a handful of ulps of libm).
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#define DGPP_DET_HD __host__ __device__ __forceinline__
#else
#define DGPP_DET_HD inline
#endif

namespace dgpp::detmath {

DGPP_DET_HD double fma_d(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
  return fma(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}
DGPP_DET_HD double ldexp_d(double x, int e) {
#if defined(__CUDA_ARCH__)
  return ldexp(x, e);
#else
  return std::ldexp(x, e);
#endif
}
DGPP_DET_HD double frexp_d(double x, int* e) {
#if defined(__CUDA_ARCH__)
  return frexp(x, e);
#else
  return std::frexp(x, e);
#endif
}
DGPP_DET_HD uint64_t bits_of(double x) {
  uint64_t u = 0;
  memcpy(&u, &x, sizeof(u));
  return u;
}
DGPP_DET_HD double double_of(uint64_t u) {
  double x = 0.0;
  memcpy(&x, &u, sizeof(x));
  return x;
}
DGPP_DET_HD double pos_inf() { return double_of(0x7ff0000000000000ull); }
DGPP_DET_HD double quiet_nan() { return double_of(0x7ff8000000000000ull); }

// ln2 split as fdlibm's: the high part has 32 trailing zero bits, so
// k * ln2_hi is exact for every |k| below 2^21.
constexpr double kLn2Hi = 6.93147180369123816490e-01;
constexpr double kLn2Lo = 1.90821492927058770002e-10;
constexpr double kInvLn2 = 1.4426950408889634;
// 1.5 * 2^52: adding it rounds a double to the nearest integer value (ties
// to even), and subtracting it back is exact.
constexpr double kShift = 6755399441055744.0;

// e^x. Reduction x = k ln2 + r, |r| <= ln2/2; e^r by the degree-13 Taylor
// polynomial (its truncation error below 2^-56 relative on that interval),
// Horner with fused steps; the scale by 2^k exact.
DGPP_DET_HD double exp_d(double x) {
  if (x != x) return x;
  if (x > 709.782712893384) return pos_inf();
  if (x < -745.1332191019412) return 0.0;
  const double kd = fma_d(x, kInvLn2, kShift) - kShift;
  const int k = static_cast<int>(kd);
  const double r = fma_d(-kd, kLn2Lo, fma_d(-kd, kLn2Hi, x));
  double p = 1.6059043836821613e-10;                // 1/13!
  p = fma_d(p, r, 2.08767569878681e-09);            // 1/12!
  p = fma_d(p, r, 2.505210838544172e-08);           // 1/11!
  p = fma_d(p, r, 2.755731922398589e-07);           // 1/10!
  p = fma_d(p, r, 2.7557319223985893e-06);          // 1/9!
  p = fma_d(p, r, 2.48015873015873e-05);            // 1/8!
  p = fma_d(p, r, 0.0001984126984126984);           // 1/7!
  p = fma_d(p, r, 0.001388888888888889);            // 1/6!
  p = fma_d(p, r, 0.008333333333333333);            // 1/5!
  p = fma_d(p, r, 0.041666666666666664);            // 1/4!
  p = fma_d(p, r, 0.16666666666666666);             // 1/3!
  p = fma_d(p, r, 0.5);                             // 1/2!
  p = fma_d(p, r, 1.0);                             // 1/1!
  const double y = fma_d(p, r, 1.0);
  return ldexp_d(y, k);
}

// ln x. x = m 2^e with m in [sqrt(1/2), sqrt(2)); ln m = 2 atanh(s),
// s = (m-1)/(m+1), |s| < 0.1716, by the odd series through s^21 (truncation
// below 2^-57 relative); then e ln2 folded in with the split constant.
DGPP_DET_HD double log_d(double x) {
  if (x != x) return x;
  if (x < 0.0) return quiet_nan();
  if (x == 0.0) return -pos_inf();
  if (x == pos_inf()) return x;
  int e = 0;
  double m = frexp_d(x, &e);  // [0.5, 1)
  if (m < 0.70710678118654752440) {
    m = m + m;  // exact
    e -= 1;
  }
  const double f = m - 1.0;         // exact (Sterbenz)
  const double s = f / (2.0 + f);   // one IEEE division
  const double z = s * s;           // one rounding
  double p = 0.047619047619047616;  // 1/21
  p = fma_d(p, z, 0.05263157894736842);   // 1/19
  p = fma_d(p, z, 0.058823529411764705);  // 1/17
  p = fma_d(p, z, 0.06666666666666667);   // 1/15
  p = fma_d(p, z, 0.07692307692307693);   // 1/13
  p = fma_d(p, z, 0.09090909090909091);   // 1/11
  p = fma_d(p, z, 0.1111111111111111);    // 1/9
  p = fma_d(p, z, 0.14285714285714285);   // 1/7
  p = fma_d(p, z, 0.2);                   // 1/5
  p = fma_d(p, z, 0.3333333333333333);    // 1/3
  const double s2 = s + s;                // exact
  const double log_m = fma_d(s2, z * p, s2);
  const double ed = static_cast<double>(e);
  return fma_d(ed, kLn2Lo, fma_d(ed, kLn2Hi, log_m));
}

// The fp32 forms: the double result rounded once to float. The double
// rounding is deterministic on both sides (the only property needed) and
// at most one float ulp from correctly rounded.
DGPP_DET_HD float exp_f(float x) {
  return static_cast<float>(exp_d(static_cast<double>(x)));
}
DGPP_DET_HD float log_f(float x) {
  return static_cast<float>(log_d(static_cast<double>(x)));
}

}  // namespace dgpp::detmath
