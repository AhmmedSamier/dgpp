// Host-only tests for the NVFP4 numerics contract (docs/nvfp4_plan.md §3.1,
// gate 1): the exactness lemma the kernels rely on — an e2m1 code times an
// e4m3 scale is exact in bf16 (and fp32) for every pair — and the decode
// trick the core uses (bit placement into fp16 fields, corrected by 2^14
// folded into the scale) reproduces the codec's table for every code.
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/latent_format.hpp"

namespace {

void require(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

// IEEE fp16 bits -> float, host-side (subnormals included), for checking
// the bit-placement decode without a device.
float fp16_bits_to_float(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);  // man * 2^-10 * 2^-14
  } else if (exp == 31) {
    v = man ? std::nanf("") : INFINITY;
  } else {
    v = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f,
                   static_cast<int>(exp) - 15);
  }
  return sign ? -v : v;
}

}  // namespace

DGPP_TEST(fp4_times_e4m3_is_exact_in_bf16_for_every_pair) {
  // 16 codes x 254 finite scale codes: the product has <= 2 + 4 significant
  // bits, so bf16 (8) holds it exactly and fp32 trivially. The kernels keep
  // fp4 * s in fp32 unrounded; this is the lemma that says nothing was lost.
  long checked = 0;
  for (int code = 0; code < 16; ++code) {
    const double w = dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(code));
    for (int sc = 0; sc < 256; ++sc) {
      const float s = dgpp::fp8_e4m3_bits_to_float(static_cast<uint8_t>(sc));
      if (std::isnan(s)) continue;
      const double exact = w * static_cast<double>(s);
      const float f = static_cast<float>(w) * s;
      require(static_cast<double>(f) == exact, "fp4 x e4m3 exact in fp32");
      const uint16_t b = dgpp::float_to_bf16_bits(f);
      require(static_cast<double>(dgpp::bf16_bits_to_float(b)) == exact,
              "fp4 x e4m3 exact in bf16");
      ++checked;
    }
  }
  require(checked == 16 * 254, "every finite pair checked");
}

DGPP_TEST(fp4_bit_placement_decode_matches_the_codec_table) {
  // The core's decode: bits 11..9 <- the magnitude bits, bit 15 <- the
  // sign, read as fp16, times 2^14 (folded into the scale on the device).
  for (int code = 0; code < 16; ++code) {
    const uint32_t bits = ((code & 7u) << 9) | ((code & 8u) << 12);
    const float scaled = fp16_bits_to_float(static_cast<uint16_t>(bits));
    const float value = scaled * 16384.0f;
    const float want = dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(code));
    require(value == want, "bit-placement decode reproduces e2m1");
    // And the correction is exact against the scale: s * 2^14 is an exact
    // power-of-two scaling for every finite e4m3 s.
    for (int sc = 0; sc < 256; ++sc) {
      const float s = dgpp::fp8_e4m3_bits_to_float(static_cast<uint8_t>(sc));
      if (std::isnan(s)) continue;
      const float s16 = s * 16384.0f;
      require(std::ldexp(s, 14) == s16, "scale x 2^14 exact");
      require(scaled * s16 == want * s, "decode x corrected scale == fp4 x s");
    }
  }
}

DGPP_TEST(fp4_e2m1_table_and_nibble_order) {
  // The format's value table and packing convention, pinned once.
  const float want[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int m = 0; m < 8; ++m) {
    require(dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(m)) == want[m], "magnitude");
    require(dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(m | 8)) == -want[m], "sign");
  }
  // low nibble = even element: byte 0xF1 holds (0.5, -6).
  const uint8_t byte = 0xF1;
  require(dgpp::fp4_e2m1_bits_to_float(byte & 0xF) == 0.5f &&
              dgpp::fp4_e2m1_bits_to_float(byte >> 4) == -6.f,
          "nibble order");
}
