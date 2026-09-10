#include "models/qwen/norm_reference.hpp"

#include <cmath>

#include "common/dtypes.hpp"

namespace dgpp::qwen_ref {

void rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
             float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* xr = x + r * dim;
    uint16_t* yr = y + r * dim;
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) {
      const float v = bf16_bits_to_float(xr[i]);
      ss = std::fma(v, v, ss);
    }
    const float rstd = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
    for (int i = 0; i < dim; ++i)
      yr[i] = float_to_bf16_bits(bf16_bits_to_float(xr[i]) * rstd *
                                 (1.0f + bf16_bits_to_float(w[i])));
  }
}

void group_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows,
                   int groups, int group_dim, float eps) {
  for (int64_t r = 0; r < rows; ++r)
    for (int g = 0; g < groups; ++g) {
      const int64_t base = (r * groups + g) * group_dim;
      rmsnorm(x + base, w + static_cast<int64_t>(g) * group_dim, y + base, 1, group_dim, eps);
    }
}

}  // namespace dgpp::qwen_ref
