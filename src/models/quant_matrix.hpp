#pragma once
// Compressed resident view of one E4M3 matrix (DESIGN §4): payload + block
// scales, nothing else. Device-visible pointers. CUDA-free so the
// host-only model library and the kernel library share the type.
#include <cstdint>

namespace dgpp {

struct GlmQuantMatrix {
  const uint8_t* payload = nullptr;  // E4M3 [rows, cols]
  const float* scales = nullptr;     // F32 [ceil(rows/128), ceil(cols/128)]
  int64_t rows = 0;
  int64_t cols = 0;
};

}  // namespace dgpp
