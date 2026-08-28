#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"

namespace {
// Exhaustive decode table self-consistency: no legal encoding decodes to NaN
// except 0x7F/0xFF, and encode(decode(bits)) round-trips for magnitudes.
bool check_roundtrip(uint8_t bits) {
  const float v = dgpp::fp8_e4m3_bits_to_float(bits);
  if (bits == 0x7F || bits == 0xFF) return std::isnan(v);
  if (std::isnan(v)) return false;  // illegal NaN pattern
  if (v == 0.0f) return bits == 0x00 || bits == 0x80;
  // Encode of a decoded value must reproduce the same magnitude bucket:
  // |enc - v| <= half quantum of the decoded exponent zone.
  uint8_t re = dgpp::float_to_fp8_e4m3_bits(v);
  if (re != bits && !(bits & 0x7Fu)) return false;
  float rv = dgpp::fp8_e4m3_bits_to_float(re);
  double err = std::abs(static_cast<double>(rv) - v);
  double tol = std::abs(v) * (1.0 / 16.0);  // >1/2 ulp(3 mantissa bits)
  return err <= (tol + 1e-30);
}
}  // namespace

DGPP_TEST(fp8_e4m3_encode_decode_full_table) {
  for (int b = 0; b < 256; ++b) {
    if (!check_roundtrip(static_cast<uint8_t>(b))) {
      throw std::runtime_error("roundtrip failure at bits=" +
                               std::to_string(b));
    }
  }
}

DGPP_TEST(fp8_e4m3_never_mints_nan_for_finite_inputs) {
  // Dense sweep over representative magnitudes incl subnormal boundary and
  // saturation carry region, plus negatives. Error must stay within one
  // half-quantum of the landing bucket (quantum doubles across zones).
  auto quantum_of = [](float v) {
    double a = std::abs(static_cast<double>(v));
    if (a == 0) return 1.0 / 512.0;
    if (a < 0.015625) return 1.0 / 512.0;
    double p = std::pow(2.0, std::floor(std::log2(a)));
    return p / 8.0;
  };
  for (double x = 1e-10; x <= 500.0; x *= 1.0007) {
    for (int sgn = 0; sgn < 2; ++sgn) {
      float v = static_cast<float>(sgn ? -x : x);
      uint8_t bits = dgpp::float_to_fp8_e4m3_bits(v);
      if ((bits & 0x7F) == 0x7F)
        throw std::runtime_error("finite input encoded as NaN");
      float back = dgpp::fp8_e4m3_bits_to_float(bits);
      if (std::isnan(back)) throw std::runtime_error("decode NaN");
      if (std::abs(static_cast<double>(v)) > 448.0 + 1e-6) {
        // Saturation zone: strictly beyond max finite must clamp exactly.
        if (back != (sgn ? -448.0f : 448.0f))
          throw std::runtime_error("overflow did not saturate to max");
        continue;
      }
      double q = quantum_of(back);
      double err = std::abs(static_cast<double>(back) - static_cast<double>(v));
      if (err > q * 0.5000001 + 1e-12)
        throw std::runtime_error("error exceeds half quantum");
    }
  }
}

DGPP_TEST(fp8_e4m3_boundary_semantics) {
  auto enc = [](float f) { return dgpp::float_to_fp8_e4m3_bits(f); };
  auto dec = [](uint8_t b) { return dgpp::fp8_e4m3_bits_to_float(b); };

  uint32_t nan_u = 0x7FC00001u;
  float crafted_nan;
  std::memcpy(&crafted_nan, &nan_u, 4);

  if (enc(0.f) != 0x00) throw std::runtime_error("zero");
  if (enc(-0.f) != 0x80) throw std::runtime_error("neg zero");
  if (dec(0x38) != 1.0f) throw std::runtime_error("one");
  if (dec(0x40) != 2.0f) throw std::runtime_error("two");
  if (dec(0x7E) != 448.0f) throw std::runtime_error("max finite");
  if (enc(1000.f) != 0x7E) throw std::runtime_error("saturate up");
  if (enc(-1000.f) != 0xFE) throw std::runtime_error("saturate down");
  if (!std::isnan(dec(enc(crafted_nan))))
    throw std::runtime_error("nan passthru");
  if (dec(0x01) <= 0.f || dec(0x01) >= 0.0025f)
    throw std::runtime_error("min denormal");
}

DGPP_TEST(bf16_round_preserves_nan_across_hardware_payloads) {
  // The integer RNE trick (u += 0x7fff + lsb) assumes a finite exponent.
  // Hardware-produced NaNs — e.g. the all-ones payload FMUL emits — carry
  // far enough that the add overflows into the sign bit, which once
  // silently converted NaN to -0.0. Every NaN payload must map to the
  // canonical quiet NaN with sign preserved (found via the M4 scale-gemm
  // NaN-policy parity case; the raw mma propagates NaN, so the bug lived
  // entirely in this conversion).
  using dgpp::float_to_bf16_bits;
  const float nan_cases[] = {
      std::nanf(""),                       // canonical 0x7FC00000
      dgpp::bf16_bits_to_float(0x7FC0),    // bf16 NaN widened
      std::bit_cast<float>(0x7FFFFFFFu),   // all-ones payload (FMUL output)
      std::bit_cast<float>(0xFFC00001u),   // negative, odd payload
      std::bit_cast<float>(0x7F800001u),   // signaling NaN, tiny payload
      std::bit_cast<float>(0xFF800001u),   // negative signaling NaN
  };
  for (float v : nan_cases) {
    if (!std::isnan(v)) throw std::runtime_error("fixture bug: non-NaN case");
    const uint16_t bits = float_to_bf16_bits(v);
    const uint16_t want =
        (std::bit_cast<uint32_t>(v) >> 31) ? 0xFFC0 : 0x7FC0;
    if (bits != want) {
      std::printf("NaN 0x%08x -> 0x%04x (want 0x%04x)\n",
                  std::bit_cast<uint32_t>(v), bits, want);
      throw std::runtime_error("bf16 NaN not canonicalized");
    }
  }
  // Finite boundary behavior is unchanged: FLT_MAX rounds to bf16 inf,
  // and inf stays inf.
  const uint16_t flt_max =
      float_to_bf16_bits(std::bit_cast<float>(0x7F7FFFFFu));
  if (flt_max != 0x7F80) throw std::runtime_error("FLT_MAX must round to inf");
  const uint16_t inf_bits =
      float_to_bf16_bits(std::bit_cast<float>(0x7F800000u));
  if (inf_bits != 0x7F80) throw std::runtime_error("inf must stay inf");
  const uint16_t ninf_bits =
      float_to_bf16_bits(std::bit_cast<float>(0xFF800000u));
  if (ninf_bits != 0xFF80) throw std::runtime_error("-inf must stay -inf");
}
