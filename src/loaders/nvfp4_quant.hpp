#pragma once
// The host NVFP4 encoder (2026-09-09, docs/glm47_plan.md D1): a BF16/fp32
// [N, K] matrix into the modelopt triple — e2m1 codes two per byte (low
// nibble = even element), one e4m3 block scale per 16 elements of a row,
// one fp32 per-tensor scale — with the NVIDIA recipe:
//
//   weight_scale_2 = amax(W) / (6 * 448)                       (fp32)
//   s_b            = e4m3( amax(block) / 6 / weight_scale_2 )  (saturating, RNE)
//   S_b            = float(s_b) * weight_scale_2               (the block's absolute scale)
//   code           = e2m1( w / S_b )                           (RNE on the e2m1 grid, saturating)
//
// so that the kernels' dequant e2m1(code) * e4m3(s_b) / (1 / weight_scale_2)
// reproduces w to the format's precision. The GLM-4.7 loader applies it to
// the draft layer's BF16 experts (every other NVFP4 tensor comes quantized
// from the checkpoint). Zero tensors and zero blocks encode as zeros with
// weight_scale_2 = 1 (a positive global keeps the kernels' division finite).
#include <cmath>
#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {

constexpr float kNvfp4CodeMax = 6.0f;
constexpr float kNvfp4ScaleMax = 448.0f;

// The per-tensor scale from the tensor's absolute maximum.
inline float nvfp4_tensor_scale(float amax) {
  if (!(amax > 0.0f) || !std::isfinite(amax)) return 1.0f;
  return amax / (kNvfp4CodeMax * kNvfp4ScaleMax);
}

// Encodes one row of `cols` fp32 values (cols % 16 == 0) under the tensor
// scale ws2: `payload` receives cols/2 bytes, `scales` cols/16 e4m3 codes.
inline void nvfp4_encode_row(const float* row, int64_t cols, float ws2, uint8_t* payload,
                             uint8_t* scales) {
  for (int64_t b = 0; b < cols / kLatentFp4Block; ++b) {
    float bmax = 0.0f;
    for (int j = 0; j < kLatentFp4Block; ++j) {
      const float a = std::fabs(row[b * kLatentFp4Block + j]);
      bmax = a > bmax ? a : bmax;
    }
    const uint8_t sc = float_to_fp8_e4m3_bits(bmax / kNvfp4CodeMax / ws2);
    scales[b] = sc;
    const float S = fp8_e4m3_bits_to_float(sc) * ws2;
    const float inv = S > 0.0f ? 1.0f / S : 0.0f;
    for (int j = 0; j < kLatentFp4Block; j += 2) {
      const int64_t e = b * kLatentFp4Block + j;
      const uint8_t lo = float_to_fp4_e2m1_bits(row[e] * inv);
      const uint8_t hi = float_to_fp4_e2m1_bits(row[e + 1] * inv);
      payload[e / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

// The absolute maximum of `n` bf16 values.
inline float nvfp4_amax_bf16(const uint16_t* w, int64_t n) {
  float amax = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float a = std::fabs(bf16_bits_to_float(w[i]));
    amax = a > amax ? a : amax;
  }
  return amax;
}

// Decodes element (r, c) of an encoded matrix back to the value the
// kernels compute (fp4 * e4m3 exact, then / global): the tests' oracle.
inline float nvfp4_decode(const uint8_t* payload, const uint8_t* scales, float global,
                          int64_t cols, int64_t r, int64_t c) {
  const uint8_t byte = payload[r * (cols / 2) + c / 2];
  const uint8_t code = (c & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
  const float s = fp8_e4m3_bits_to_float(scales[r * (cols / kLatentFp4Block) + c / kLatentFp4Block]);
  return fp4_e2m1_bits_to_float(code) * s / global;
}

}  // namespace dgpp
