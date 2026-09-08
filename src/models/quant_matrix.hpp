#pragma once
// Compressed resident view of one E4M3 matrix (DESIGN §4): payload + block
// scales, nothing else. Device-visible pointers. CUDA-free so the
// host-only model library and the kernel library share the type.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dgpp {

struct GlmQuantMatrix {
  const uint8_t* payload = nullptr;  // E4M3 [rows, cols]
  const float* scales = nullptr;     // F32 [ceil(rows/128), ceil(cols/128)]
  int64_t rows = 0;
  int64_t cols = 0;
  size_t scale_bytes() const {
    return static_cast<size_t>((rows + 127) / 128) *
           static_cast<size_t>((cols + 127) / 128) * sizeof(float);
  }
};

// Row-range view: payload rows are contiguous, so this is a pure pointer
// view — no copy. A 128-ALIGNED row_start re-anchors the scale grid exactly
// (local row r's true block is row_start/128 + r/128, which is what the
// local-frame consumer computes); a MISALIGNED start that crosses a block
// boundary would need two different scale values inside one local block —
// unrepresentable — and is rejected here, loudly. The real checkpoint's
// inter dims (12288 dense / 2048 MoE) are 128-multiples at every TP world;
// the loader-era fixture's 200 was this contract's counterexample and
// produced silently wrong scales (measured: layer folds 0.30 l2-wrong,
// invisible to the stream-state metric that attenuated it 300x — see the
// M5 record).
inline GlmQuantMatrix quant_rows_view(const GlmQuantMatrix& m,
                                      int64_t row_start, int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("quant_rows_view: range out of bounds");
  if (row_start % 128 != 0)
    throw std::invalid_argument(
        "quant_rows_view: row_start must be 128-aligned (quantized "
        "scale-grid slice contract; misaligned slices cannot re-anchor "
        "the block scales)");
  const int64_t sb = (m.cols + 127) / 128;  // scale blocks per scale row
  GlmQuantMatrix v;
  v.payload = m.payload + row_start * m.cols;
  v.scales = m.scales + (row_start / 128) * sb;
  v.rows = rows;
  v.cols = m.cols;
  return v;
}

// Compressed resident view of one NVFP4 matrix (docs/nvfp4_plan.md §3.1):
// the checkpoint's bytes, untouched — e2m1 codes two per byte (low nibble
// = even element), one e4m3 scale per 16 elements along K, one F32 global
// scale per tensor (a DEVICE address; the kernels divide the finished dot
// by it once). The dequantized value of element (n, k) is
//   e2m1(code) * (float(scales[n][k/16]) / *global_scale)
// and `e2m1(code) * float(scale)` is EXACT in bf16 (2 + 4 significant
// bits), which is what the kernels rely on.
constexpr int kFp4Group = 16;  // elements per e4m3 block scale

struct GlmFp4Matrix {
  const uint8_t* payload = nullptr;       // U8 [rows, cols/2]
  const uint8_t* scales = nullptr;        // e4m3 [rows, cols/16]
  const float* global_scale = nullptr;    // F32 [1], device
  int64_t rows = 0;
  int64_t cols = 0;                       // logical K (elements)

  int64_t payload_cols() const { return cols / 2; }
  int64_t scale_cols() const { return cols / kFp4Group; }
  size_t payload_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols / 2);
  }
  size_t scale_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols / kFp4Group);
  }
};

// K must be a multiple of 16 (one scale per block); a column slice must
// start on a block boundary (16, which is also even — a packed byte) and
// span whole blocks. Row slices are free: every row carries its own
// scales.
inline void fp4_check_cols(int64_t cols, const char* who) {
  if (cols <= 0 || cols % kFp4Group != 0)
    throw std::invalid_argument(std::string(who) +
                                ": NVFP4 K must be a positive multiple of 16");
}

inline GlmFp4Matrix fp4_rows_view(const GlmFp4Matrix& m, int64_t row_start,
                                  int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("fp4_rows_view: range out of bounds");
  GlmFp4Matrix v;
  v.payload = m.payload + row_start * m.payload_cols();
  v.scales = m.scales + row_start * m.scale_cols();
  v.global_scale = m.global_scale;
  v.rows = rows;
  v.cols = m.cols;
  return v;
}

}  // namespace dgpp
