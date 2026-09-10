#pragma once
// The host block-FP8 encoder (2026-09-10, engine.dense_weights = "fp8"): a
// BF16 [rows, cols] matrix (row stride in elements) into the form the fp8
// GEMV core and the scale-GEMM tile read — E4M3 codes [rows, cols] and one
// fp32 scale per 128 x 128 block:
//
//   scale_b = amax(block) / 448           (an all-zero block: 1, codes 0)
//   code    = e4m3( w / scale_b )         (round to nearest even, saturating)
//
// so the kernels' bf16(e4m3(code) * scale_b) is w to the format's precision
// (at most one part in sixteen per element at the grid's worst point). The
// recipe is the FP8 releases' (GLM-5.3-Flash's, this model's MTP experts):
// the same values quantized offline give the same codes — there is no
// calibration in block FP8, the block's maximum fixes its scale.
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp::fp8_quant {

constexpr int kBlock = 128;
constexpr float kE4M3Max = 448.0f;

inline int64_t scale_rows(int64_t rows) { return (rows + kBlock - 1) / kBlock; }
inline int64_t scale_cols(int64_t cols) { return (cols + kBlock - 1) / kBlock; }

// One block: rows [r0, r1) x cols [c0, c1) of src → payload (row-major
// [rows, cols], the block in place) and its scale.
inline void encode_block(const uint16_t* src, size_t src_stride, int64_t cols, int64_t r0, int64_t r1,
                         int64_t c0, int64_t c1, uint8_t* payload, float* scale) {
  float amax = 0.0f;
  for (int64_t r = r0; r < r1; ++r) {
    const uint16_t* row = src + static_cast<size_t>(r) * src_stride;
    for (int64_t c = c0; c < c1; ++c) amax = std::max(amax, std::fabs(bf16_bits_to_float(row[c])));
  }
  const float s = amax > 0.0f ? amax / kE4M3Max : 1.0f;
  *scale = s;
  for (int64_t r = r0; r < r1; ++r) {
    const uint16_t* row = src + static_cast<size_t>(r) * src_stride;
    uint8_t* out = payload + static_cast<size_t>(r) * cols;
    for (int64_t c = c0; c < c1; ++c) out[c] = float_to_fp8_e4m3_bits(bf16_bits_to_float(row[c]) / s);
  }
}

// The whole matrix, block rows spread over `threads` threads (0: the
// machine's, at most 16). scales: [scale_rows(rows), scale_cols(cols)].
inline void encode_block128(const uint16_t* src, size_t src_stride, int64_t rows, int64_t cols,
                            uint8_t* payload, float* scales, int threads = 0) {
  if (rows <= 0 || cols <= 0) return;
  const int64_t sr = scale_rows(rows), sc = scale_cols(cols);
  if (threads <= 0) threads = static_cast<int>(std::min<unsigned>(16u, std::max<unsigned>(1u, std::thread::hardware_concurrency())));
  threads = static_cast<int>(std::min<int64_t>(threads, sr));
  auto work = [&](int t) {
    for (int64_t br = t; br < sr; br += threads) {
      const int64_t r0 = br * kBlock, r1 = std::min(rows, r0 + kBlock);
      for (int64_t bc = 0; bc < sc; ++bc) {
        const int64_t c0 = bc * kBlock, c1 = std::min(cols, c0 + kBlock);
        encode_block(src, src_stride, cols, r0, r1, c0, c1, payload, scales + br * sc + bc);
      }
    }
  };
  if (threads <= 1) {
    work(0);
    return;
  }
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(work, t);
  for (auto& th : pool) th.join();
}

}  // namespace dgpp::fp8_quant
