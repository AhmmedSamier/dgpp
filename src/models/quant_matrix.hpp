#pragma once
// Compressed resident view of one E4M3 matrix (DESIGN §4): payload + block
// scales, nothing else. Device-visible pointers. CUDA-free so the
// host-only model library and the kernel library share the type.
#include <cstdint>
#include <stdexcept>

namespace dgpp {

struct GlmQuantMatrix {
  const uint8_t* payload = nullptr;  // E4M3 [rows, cols]
  const float* scales = nullptr;     // F32 [ceil(rows/128), ceil(cols/128)]
  int64_t rows = 0;
  int64_t cols = 0;
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

}  // namespace dgpp
