#pragma once
// The latent cache's storage formats (2026-09-06): the KV cache's dtype.
//
// The DSA latent cache holds one kv_lora-wide row per token per DSA layer
// (DESIGN §7.2). It is written once (the latent append after the kv_a
// RMSNorm) and read by every attention over that token, so its format is
// a storage decision the two attention kernels and the append kernel
// share. Three formats:
//
//   bf16  kv_lora x 2 bytes per row; the rows as the projection left them
//         (every parity gate's format, bitwise the historical cache).
//   fp8   kv_lora e4m3 codes per row plus ONE fp32 row scale
//         (scale = absmax / 448; code = e4m3(x * 448 / absmax)). Half the
//         bytes; ~2^-4 relative error per element.
//   fp4   e2m1 codes, two per byte, in blocks of 16 elements with one e4m3
//         block scale each (the NVFP4 recipe: block scale in units of a
//         per-row fp32 scale so the largest block reaches 448 and the
//         largest element of a block reaches 6), the row padded to 16
//         bytes. kv_lora x 9/16 bytes per row; ~2^-2 relative error per
//         element — a quality trade an operator makes deliberately.
//
// The codecs below are host/device functions with no libm beyond what
// dtypes.hpp already relies on: the device quantizes a row with the SAME
// arithmetic the host reference runs (the tests pin the codes bitwise),
// and the attention kernels dequantize a row with the same arithmetic the
// host oracle applies, so a quantized-cache attention equals the bf16
// kernel over the dequantized rows within the bf16 kernel's own tolerance.
// Every operation is a single fp32 multiply, divide or conversion (no
// contractable add), so -ffp-contract and --fmad cannot change a bit.
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "common/dtypes.hpp"

namespace dgpp {

enum class LatentFormat : int { kBf16 = 0, kFp8 = 1, kFp4 = 2 };

constexpr const char* latent_format_name(LatentFormat f) {
  switch (f) {
    case LatentFormat::kBf16: return "bf16";
    case LatentFormat::kFp8: return "fp8";
    case LatentFormat::kFp4: return "fp4";
  }
  return "?";
}

inline std::optional<LatentFormat> latent_format_from_string(std::string_view s) {
  if (s == "bf16") return LatentFormat::kBf16;
  if (s == "fp8" || s == "fp8_e4m3" || s == "e4m3") return LatentFormat::kFp8;
  if (s == "fp4" || s == "nvfp4" || s == "e2m1") return LatentFormat::kFp4;
  return std::nullopt;
}

constexpr int kLatentFp4Block = 16;  // elements per e4m3 block scale
constexpr float kLatentFp8Max = 448.0f;
constexpr float kLatentFp4Max = 6.0f;

// Bytes of one latent row (a token's kv_lora elements in this format).
constexpr size_t latent_row_bytes(LatentFormat f, int kv_lora) {
  switch (f) {
    case LatentFormat::kBf16: return static_cast<size_t>(kv_lora) * 2;
    case LatentFormat::kFp8: return static_cast<size_t>(kv_lora);
    case LatentFormat::kFp4: {
      const size_t raw = static_cast<size_t>(kv_lora) / 2 +
                         static_cast<size_t>(kv_lora) / kLatentFp4Block;
      return (raw + 15) / 16 * 16;
    }
  }
  return 0;
}
// Whether the format keeps a per-row fp32 scale beside the rows.
constexpr bool latent_format_has_row_scale(LatentFormat f) {
  return f != LatentFormat::kBf16;
}
// The row's block-scale bytes start here (fp4 only).
constexpr size_t latent_fp4_scale_offset(int kv_lora) {
  return static_cast<size_t>(kv_lora) / 2;
}

// ---- e2m1 ---------------------------------------------------------------
// Values {0, .5, 1, 1.5, 2, 3, 4, 6} with a sign bit (bit 3); the low
// nibble of a byte is the even element. Encode is round-to-nearest-even on
// that grid, saturating; NaN saturates too (no NaN code exists).
DGPP_HD inline float fp4_e2m1_bits_to_float(uint8_t v) {
  const uint32_t m = v & 7u;
  float mag;
  switch (m) {
    case 0: mag = 0.0f; break;
    case 1: mag = 0.5f; break;
    case 2: mag = 1.0f; break;
    case 3: mag = 1.5f; break;
    case 4: mag = 2.0f; break;
    case 5: mag = 3.0f; break;
    case 6: mag = 4.0f; break;
    default: mag = 6.0f; break;
  }
  return (v & 8u) ? -mag : mag;
}

DGPP_HD inline uint8_t float_to_fp4_e2m1_bits(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  const uint8_t sign = static_cast<uint8_t>((u >> 28) & 8u);
  const uint32_t absbits = u & 0x7FFFFFFFu;
  if (absbits > 0x7F800000u) return static_cast<uint8_t>(sign | 7u);  // NaN
  const float a = std::bit_cast<float>(absbits);
  uint8_t code;
  if (a <= 0.25f) code = 0;         // tie at .25 -> 0 (even)
  else if (a < 0.75f) code = 1;
  else if (a <= 1.25f) code = 2;    // ties at .75 and 1.25 -> 1.0 (even)
  else if (a < 1.75f) code = 3;
  else if (a <= 2.5f) code = 4;     // ties at 1.75 and 2.5 -> 2.0 (even)
  else if (a < 3.5f) code = 5;
  else if (a <= 5.0f) code = 6;     // ties at 3.5 and 5 -> 4.0 (even)
  else code = 7;
  return static_cast<uint8_t>(sign | code);
}

// ---- the fp8 row --------------------------------------------------------
// scale = absmax / 448 (0 for an all-zero row: every code is 0 and the
// decode is exact); inv is what the encoder multiplies by.
struct LatentFp8Scale {
  float scale;
  float inv;
};
DGPP_HD inline LatentFp8Scale latent_fp8_row_scale(float absmax) {
  LatentFp8Scale s;
  s.scale = absmax * (1.0f / kLatentFp8Max);
  s.inv = absmax > 0.0f ? kLatentFp8Max / absmax : 0.0f;
  return s;
}
DGPP_HD inline uint8_t latent_fp8_encode(float x, float inv) {
  return float_to_fp8_e4m3_bits(x * inv);
}
// The bf16 the attention kernels see for a stored code.
DGPP_HD inline uint16_t latent_fp8_decode_bf16(uint8_t code, float scale) {
  return float_to_bf16_bits(fp8_e4m3_bits_to_float(code) * scale);
}

// ---- the fp4 row --------------------------------------------------------
// row scale s_r = absmax / (6 * 448); a block's e4m3 scale code is
// e4m3(block_absmax * 448 / absmax), so its decoded value times s_r is the
// block's absolute scale S (the block's largest element lands near 6);
// codes are e2m1(x / S).
struct LatentFp4RowScale {
  float scale;  // s_r
  float inv;    // 448 / absmax (0 for an all-zero row)
};
DGPP_HD inline LatentFp4RowScale latent_fp4_row_scale(float absmax) {
  LatentFp4RowScale s;
  s.scale = absmax * (1.0f / (kLatentFp4Max * kLatentFp8Max));
  s.inv = absmax > 0.0f ? kLatentFp8Max / absmax : 0.0f;
  return s;
}
DGPP_HD inline uint8_t latent_fp4_block_scale_code(float block_absmax,
                                                   float row_inv) {
  return float_to_fp8_e4m3_bits(block_absmax * row_inv);
}
// The block's absolute scale S from its code and the row scale.
DGPP_HD inline float latent_fp4_block_scale(uint8_t code, float row_scale) {
  return fp8_e4m3_bits_to_float(code) * row_scale;
}
// What the encoder multiplies by: 1/S (0 for an all-zero block).
DGPP_HD inline float latent_fp4_block_inv(float block_scale) {
  return block_scale > 0.0f ? 1.0f / block_scale : 0.0f;
}
DGPP_HD inline uint8_t latent_fp4_encode(float x, float block_inv) {
  return float_to_fp4_e2m1_bits(x * block_inv);
}
DGPP_HD inline uint16_t latent_fp4_decode_bf16(uint8_t code, float block_scale) {
  return float_to_bf16_bits(fp4_e2m1_bits_to_float(code) * block_scale);
}

// ---- host reference codecs (the tests' oracle; also any host tooling) ----
// Quantizes one bf16 row into `out` (latent_row_bytes(f, kv_lora) bytes)
// and `*row_scale` (unused for bf16). The device append kernel reproduces
// these bytes bitwise.
inline void latent_quantize_row_host(LatentFormat f, const uint16_t* row,
                                     int kv_lora, uint8_t* out,
                                     float* row_scale) {
  if (f == LatentFormat::kBf16) {
    for (int i = 0; i < kv_lora; ++i) {
      out[2 * i] = static_cast<uint8_t>(row[i] & 0xFFu);
      out[2 * i + 1] = static_cast<uint8_t>(row[i] >> 8);
    }
    if (row_scale) *row_scale = 1.0f;
    return;
  }
  float absmax = 0.0f;
  for (int i = 0; i < kv_lora; ++i) {
    const float a = std::fabs(bf16_bits_to_float(row[i]));
    absmax = a > absmax ? a : absmax;
  }
  if (f == LatentFormat::kFp8) {
    const LatentFp8Scale s = latent_fp8_row_scale(absmax);
    for (int i = 0; i < kv_lora; ++i)
      out[i] = latent_fp8_encode(bf16_bits_to_float(row[i]), s.inv);
    if (row_scale) *row_scale = s.scale;
    return;
  }
  const LatentFp4RowScale s = latent_fp4_row_scale(absmax);
  const size_t bytes = latent_row_bytes(f, kv_lora);
  for (size_t i = 0; i < bytes; ++i) out[i] = 0;
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  for (int b = 0; b < kv_lora / kLatentFp4Block; ++b) {
    float bmax = 0.0f;
    for (int j = 0; j < kLatentFp4Block; ++j) {
      const float a = std::fabs(bf16_bits_to_float(row[b * kLatentFp4Block + j]));
      bmax = a > bmax ? a : bmax;
    }
    const uint8_t sc = latent_fp4_block_scale_code(bmax, s.inv);
    out[scale_off + static_cast<size_t>(b)] = sc;
    const float inv = latent_fp4_block_inv(latent_fp4_block_scale(sc, s.scale));
    for (int j = 0; j < kLatentFp4Block; j += 2) {
      const int e = b * kLatentFp4Block + j;
      const uint8_t lo = latent_fp4_encode(bf16_bits_to_float(row[e]), inv);
      const uint8_t hi = latent_fp4_encode(bf16_bits_to_float(row[e + 1]), inv);
      out[static_cast<size_t>(e) / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  if (row_scale) *row_scale = s.scale;
}

// Dequantizes one stored row into bf16 — exactly the values the attention
// kernels load.
inline void latent_dequantize_row_host(LatentFormat f, const uint8_t* in,
                                       float row_scale, int kv_lora,
                                       uint16_t* out) {
  if (f == LatentFormat::kBf16) {
    for (int i = 0; i < kv_lora; ++i)
      out[i] = static_cast<uint16_t>(in[2 * i] | (in[2 * i + 1] << 8));
    return;
  }
  if (f == LatentFormat::kFp8) {
    for (int i = 0; i < kv_lora; ++i)
      out[i] = latent_fp8_decode_bf16(in[i], row_scale);
    return;
  }
  const size_t scale_off = latent_fp4_scale_offset(kv_lora);
  for (int i = 0; i < kv_lora; ++i) {
    const float S = latent_fp4_block_scale(
        in[scale_off + static_cast<size_t>(i / kLatentFp4Block)], row_scale);
    const uint8_t byte = in[static_cast<size_t>(i) / 2];
    const uint8_t code = (i & 1) ? static_cast<uint8_t>(byte >> 4)
                                 : static_cast<uint8_t>(byte & 0xFu);
    out[i] = latent_fp4_decode_bf16(code, S);
  }
}

}  // namespace dgpp
