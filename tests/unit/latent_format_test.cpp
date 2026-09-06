// The latent cache's storage formats (2026-09-06): the e2m1 codec, the
// fp8/fp4 row quantizers and their dequantizers — the host reference the
// device append kernel and the attention tile loaders are pinned to.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/latent_format.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LatentFormat;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Deterministic pseudo-random bf16 rows in [-4, 4) with a few zeros and
// one large outlier per row (the block-scale case that matters).
std::vector<uint16_t> row(uint64_t seed, int n) {
  std::vector<uint16_t> r(static_cast<size_t>(n));
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (int i = 0; i < n; ++i) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    const uint32_t h = static_cast<uint32_t>((x * 0x2545F4914F6CDD1Dull) >> 32);
    float v = (static_cast<float>(h & 0xFFFF) / 65536.0f - 0.5f) * 8.0f;
    if ((h >> 16) % 17 == 0) v = 0.0f;
    if (i == static_cast<int>(seed % static_cast<uint64_t>(n))) v *= 20.0f;
    r[static_cast<size_t>(i)] = float_to_bf16_bits(v);
  }
  return r;
}

double rel_l2(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double d2 = 0, n2 = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = bf16_bits_to_float(a[i]), y = bf16_bits_to_float(b[i]);
    d2 += (x - y) * (x - y);
    n2 += x * x;
  }
  return std::sqrt(d2 / std::max(n2, 1e-30));
}

}  // namespace

DGPP_TEST(latent_format_names_and_row_bytes) {
  require(dgpp::latent_format_from_string("bf16") == LatentFormat::kBf16 &&
              dgpp::latent_format_from_string("fp8") == LatentFormat::kFp8 &&
              dgpp::latent_format_from_string("fp4") == LatentFormat::kFp4 &&
              !dgpp::latent_format_from_string("int8") &&
              !dgpp::latent_format_from_string(""),
          "the three names parse and nothing else does");
  require(std::string(dgpp::latent_format_name(LatentFormat::kFp4)) == "fp4",
          "the name round-trips");
  // The real geometry: 512-wide rows.
  require(dgpp::latent_row_bytes(LatentFormat::kBf16, 512) == 1024, "bf16 row");
  require(dgpp::latent_row_bytes(LatentFormat::kFp8, 512) == 512, "fp8 row");
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 512) == 288, "fp4 row: 256 codes + 32 scales");
  // Rows keep 16-byte alignment (the tile loaders' vector loads).
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 32) == 32, "fp4 row padded to 16 bytes");
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 256) == 144, "fp4 256-wide row");
  require(!dgpp::latent_format_has_row_scale(LatentFormat::kBf16) &&
              dgpp::latent_format_has_row_scale(LatentFormat::kFp8) &&
              dgpp::latent_format_has_row_scale(LatentFormat::kFp4),
          "only the quantized formats carry a row scale");
}

DGPP_TEST(latent_format_e2m1_codec_is_exact_and_rounds_to_even) {
  const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int code = 0; code < 16; ++code) {
    const float v = dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(code));
    const float want = (code & 8) ? -grid[code & 7] : grid[code & 7];
    require(v == want, "decode of code " + std::to_string(code));
    // Every grid value encodes back to itself (the zero's sign aside).
    const uint8_t re = dgpp::float_to_fp4_e2m1_bits(v);
    require((re & 7) == (code & 7) && ((code & 7) == 0 || (re & 8) == (code & 8)),
            "encode(decode) of code " + std::to_string(code));
  }
  // Midpoints round to the even code: .25->0, .75->1.0, 1.25->1.0,
  // 1.75->2.0, 2.5->2.0, 3.5->4.0, 5->4.0; beyond 6 saturates.
  const std::pair<float, int> ties[] = {{0.25f, 0}, {0.75f, 2}, {1.25f, 2}, {1.75f, 4},
                                        {2.5f, 4},  {3.5f, 6},  {5.0f, 6},  {7.0f, 7},
                                        {1e9f, 7},  {0.26f, 1}, {0.74f, 1}, {4.99f, 6}};
  for (const auto& [x, code] : ties) {
    require(dgpp::float_to_fp4_e2m1_bits(x) == code,
            "tie/saturation at " + std::to_string(x));
    require(dgpp::float_to_fp4_e2m1_bits(-x) == (code | 8), "negative of it");
  }
  require((dgpp::float_to_fp4_e2m1_bits(NAN) & 7) == 7, "NaN saturates (no NaN code)");
}

DGPP_TEST(latent_format_row_quantizers_bound_the_error_and_handle_zero_rows) {
  const int kv_lora = 512;
  for (uint64_t seed = 1; seed <= 24; ++seed) {
    const std::vector<uint16_t> src = row(seed, kv_lora);
    for (LatentFormat f : {LatentFormat::kBf16, LatentFormat::kFp8, LatentFormat::kFp4}) {
      std::vector<uint8_t> bytes(dgpp::latent_row_bytes(f, kv_lora), 0xAA);
      float scale = -1.0f;
      dgpp::latent_quantize_row_host(f, src.data(), kv_lora, bytes.data(), &scale);
      std::vector<uint16_t> back(static_cast<size_t>(kv_lora));
      dgpp::latent_dequantize_row_host(f, bytes.data(), scale, kv_lora, back.data());
      const double err = rel_l2(src, back);
      const double bound = f == LatentFormat::kBf16 ? 0.0 : f == LatentFormat::kFp8 ? 0.06 : 0.30;
      require(err <= bound, std::string("row ") + std::to_string(seed) + " in " +
                                dgpp::latent_format_name(f) + ": relative l2 " +
                                std::to_string(err) + " above " + std::to_string(bound));
      // The largest element survives within its own quantum: the row scale
      // puts the absmax at the top of the range.
      float amax = 0, back_max = 0;
      for (int i = 0; i < kv_lora; ++i) {
        amax = std::max(amax, std::fabs(bf16_bits_to_float(src[static_cast<size_t>(i)])));
        back_max = std::max(back_max, std::fabs(bf16_bits_to_float(back[static_cast<size_t>(i)])));
      }
      require(std::fabs(back_max - amax) <= amax * (f == LatentFormat::kFp4 ? 0.13 : 0.07) + 1e-6f,
              std::string("absmax preserved in ") + dgpp::latent_format_name(f));
      if (f == LatentFormat::kFp4) {
        // The fp4 row's padding bytes are written (never left as the
        // buffer's old contents).
        for (size_t k = dgpp::latent_fp4_scale_offset(kv_lora) + kv_lora / dgpp::kLatentFp4Block;
             k < bytes.size(); ++k)
          require(bytes[k] == 0, "fp4 padding zeroed");
      }
    }
  }
  // An all-zero row: scale 0, every code 0, exact zeros back.
  const std::vector<uint16_t> zeros(static_cast<size_t>(kv_lora), 0);
  for (LatentFormat f : {LatentFormat::kFp8, LatentFormat::kFp4}) {
    std::vector<uint8_t> bytes(dgpp::latent_row_bytes(f, kv_lora), 0xFF);
    float scale = -1.0f;
    dgpp::latent_quantize_row_host(f, zeros.data(), kv_lora, bytes.data(), &scale);
    require(scale == 0.0f, "zero row's scale");
    for (uint8_t b : bytes) require(b == 0, "zero row's codes");
    std::vector<uint16_t> back(static_cast<size_t>(kv_lora), 1);
    dgpp::latent_dequantize_row_host(f, bytes.data(), scale, kv_lora, back.data());
    for (uint16_t v : back) require(v == 0, "zero row decodes to zeros");
  }
}

DGPP_TEST(latent_format_fp8_row_matches_the_elementwise_recipe) {
  // The row codec is the per-element recipe applied with the row's scale:
  // scale = absmax / 448, code = e4m3(x * 448 / absmax).
  const int kv_lora = 64;
  const std::vector<uint16_t> src = row(7, kv_lora);
  float absmax = 0;
  for (uint16_t v : src) absmax = std::max(absmax, std::fabs(bf16_bits_to_float(v)));
  std::vector<uint8_t> bytes(dgpp::latent_row_bytes(LatentFormat::kFp8, kv_lora));
  float scale = 0;
  dgpp::latent_quantize_row_host(LatentFormat::kFp8, src.data(), kv_lora, bytes.data(), &scale);
  require(scale == absmax * (1.0f / 448.0f), "the row scale");
  for (int i = 0; i < kv_lora; ++i) {
    const uint8_t want =
        dgpp::float_to_fp8_e4m3_bits(bf16_bits_to_float(src[static_cast<size_t>(i)]) * (448.0f / absmax));
    require(bytes[static_cast<size_t>(i)] == want, "code " + std::to_string(i));
    require(dgpp::latent_fp8_decode_bf16(want, scale) ==
                float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(want) * scale),
            "decode " + std::to_string(i));
  }
}
