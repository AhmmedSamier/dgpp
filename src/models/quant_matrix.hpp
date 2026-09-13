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
  const float* scales = nullptr;     // F32 [ceil(rows/sbr), ceil(cols/sbc)]
  int64_t rows = 0;
  int64_t cols = 0;
  // The resident scale grid's block size per axis (Q2, 2026-09-09,
  // docs/qwen38_flash_next_plan.md D2): the checkpoint's grid is 128x128;
  // a TP slice that starts mid-block is made exact by re-blocking the
  // sliced axis at gcd(128, slice) with the parent block's scale
  // replicated. 128 on both axes is the GLM contract, untouched.
  int scale_block_rows = 128;
  int scale_block_cols = 128;
  int64_t scale_rows() const { return (rows + scale_block_rows - 1) / scale_block_rows; }
  int64_t scale_cols() const { return (cols + scale_block_cols - 1) / scale_block_cols; }
  size_t scale_bytes() const {
    return static_cast<size_t>(scale_rows()) * static_cast<size_t>(scale_cols()) * sizeof(float);
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
  if (row_start % m.scale_block_rows != 0)
    throw std::invalid_argument(
        "quant_rows_view: row_start must be aligned to the scale block "
        "(quantized scale-grid slice contract; misaligned slices cannot "
        "re-anchor the block scales)");
  const int64_t sb = m.scale_cols();  // scale blocks per scale row
  GlmQuantMatrix v;
  v.payload = m.payload + row_start * m.cols;
  v.scales = m.scales + (row_start / m.scale_block_rows) * sb;
  v.rows = rows;
  v.cols = m.cols;
  v.scale_block_rows = m.scale_block_rows;
  v.scale_block_cols = m.scale_block_cols;
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

// Compressed resident view of one packed-int matrix (docs/glm53_plan.md
// §1.5, D2): the checkpoint's bytes, untouched — compressed-tensors
// `pack-quantized`, symmetric, group 64. Codes are 4- or 8-bit unsigned
// with an offset of 2^(bits-1) (nibble - 8, byte - 128), packed
// `32 / bits` per I32 word along K with the low nibble / byte the first
// element; one bf16 scale per 64 elements along K. The dequantized value
// of element (n, k) is
//   (code - 2^(bits-1)) * float(scales[n][k/64])
// exactly in fp32 (an 8-bit integer times an 8-bit-mantissa scale), and
// the kernels keep it exact: no weight is rounded to bf16 (the exact
// policy, packq_gemv.cuh).
constexpr int kPackedGroup = 64;  // elements per bf16 group scale

struct GlmPackedMatrix {
  const uint32_t* packed = nullptr;       // I32 [rows, cols*bits/32]
  const uint16_t* scales = nullptr;       // bf16 [rows, cols/64]
  int64_t rows = 0;
  int64_t cols = 0;                       // logical K (elements)
  int bits = 0;                           // 4 or 8

  int64_t packed_cols() const { return cols * bits / 32; }   // I32 words per row
  int64_t scale_cols() const { return cols / kPackedGroup; }
  size_t packed_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(bits) / 8;
  }
  size_t scale_bytes() const {
    return static_cast<size_t>(rows) * static_cast<size_t>(scale_cols()) * 2;
  }
};

// K must be a multiple of 64 (one scale per group, whole packed words); a
// column slice must start on a group boundary and span whole groups. Row
// slices are free: every row carries its own scales.
inline void packed_check_cols(int64_t cols, int bits, const char* who) {
  if (bits != 4 && bits != 8)
    throw std::invalid_argument(std::string(who) + ": the packed code width must be 4 or 8");
  if (cols <= 0 || cols % kPackedGroup != 0)
    throw std::invalid_argument(std::string(who) +
                                ": packed K must be a positive multiple of 64");
}

inline GlmPackedMatrix packed_rows_view(const GlmPackedMatrix& m, int64_t row_start,
                                        int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("packed_rows_view: range out of bounds");
  GlmPackedMatrix v;
  v.packed = m.packed + row_start * m.packed_cols();
  v.scales = m.scales + row_start * m.scale_cols();
  v.rows = rows;
  v.cols = m.cols;
  v.bits = m.bits;
  return v;
}

}  // namespace dgpp
