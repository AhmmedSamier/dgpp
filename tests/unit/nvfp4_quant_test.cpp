// The host NVFP4 encoder (2026-09-09, docs/glm47_plan.md D1): every code
// decodes within the format's error of its source, block scales never hit
// the NaN codes, exact e2m1 values round-trip bitwise, and zero tensors
// encode as zeros under a finite global.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/nvfp4_quant.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  float unit() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return static_cast<float>(s >> 11) / static_cast<float>(1ull << 52) - 1.0f;
  }
};

}  // namespace

DGPP_TEST(nvfp4_encoder_round_trips_within_the_format_error) {
  const int64_t rows = 7, cols = 96;
  std::vector<float> w(static_cast<size_t>(rows * cols));
  Rng rng;
  for (float& v : w) v = 0.05f * (rng.unit() + rng.unit() + rng.unit());
  w[5] = 0.9f;  // the tensor's amax, alone in its block
  float amax = 0.0f;
  for (float v : w) amax = std::max(amax, std::fabs(v));
  const float ws2 = dgpp::nvfp4_tensor_scale(amax);
  std::vector<uint8_t> payload(static_cast<size_t>(rows * cols / 2)), scales(static_cast<size_t>(rows * cols / 16));
  for (int64_t r = 0; r < rows; ++r)
    dgpp::nvfp4_encode_row(w.data() + r * cols, cols, ws2, payload.data() + r * cols / 2, scales.data() + r * cols / 16);
  const float global = 1.0f / ws2;
  for (uint8_t sc : scales) require((sc & 0x7F) != 0x7F, "no NaN scale codes");
  double worst = 0.0;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const float got = dgpp::nvfp4_decode(payload.data(), scales.data(), global, cols, r, c);
      const float want = w[static_cast<size_t>(r * cols + c)];
      // The format's bound: a block's scale S is its maximum / 6 up to the
      // e4m3 rounding (2^-4 relative), and the e2m1 grid's widest step is
      // 4 -> 6, so an element is off by at most one S: about 0.18 x the
      // block's maximum.
      float bmax = 0.0f;
      for (int j = 0; j < 16; ++j) bmax = std::max(bmax, std::fabs(w[static_cast<size_t>(r * cols + (c / 16) * 16 + j)]));
      const float tol = 0.2f * bmax + 1e-6f;
      worst = std::max(worst, static_cast<double>(std::fabs(got - want)) / (bmax + 1e-6f));
      require(std::fabs(got - want) <= tol, "element (" + std::to_string(r) + "," + std::to_string(c) + ") off by " +
                                                std::to_string(std::fabs(got - want)));
    }
  require(worst < 0.2, "worst relative error " + std::to_string(worst));
  // The amax element decodes to the top code under a scale of 448 x ws2.
  require(std::fabs(dgpp::nvfp4_decode(payload.data(), scales.data(), global, cols, 0, 5) - 0.9f) < 0.9f * 0.03f,
          "the amax element");
}

DGPP_TEST(nvfp4_encoder_exact_grid_and_zero_tensor) {
  // Values already on a block's e2m1 grid (scale 1 x ws2 = 1/(6*448) ... use
  // a block whose amax is 6 * s for an e4m3-exact s): every code exact.
  const int64_t cols = 16;
  std::vector<float> w(16);
  const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int i = 0; i < 16; ++i) w[static_cast<size_t>(i)] = (i % 2 ? -1.f : 1.f) * grid[i % 8] * 2.0f;
  const float ws2 = dgpp::nvfp4_tensor_scale(12.0f);  // amax 12 -> ws2 = 12 / 2688
  std::vector<uint8_t> payload(8), scales(1);
  dgpp::nvfp4_encode_row(w.data(), cols, ws2, payload.data(), scales.data());
  // The block scale code is e4m3(12 / 6 / ws2) = e4m3(448) exactly.
  require(scales[0] == dgpp::float_to_fp8_e4m3_bits(448.0f), "the block scale saturates at 448 exactly");
  for (int i = 0; i < 16; ++i) {
    const float got = dgpp::nvfp4_decode(payload.data(), scales.data(), 1.0f / ws2, cols, 0, i);
    require(std::fabs(got - w[static_cast<size_t>(i)]) <= 1e-5f * 12.0f, "grid value " + std::to_string(i));
  }
  // A zero tensor: finite global, zero codes and scales.
  std::vector<float> z(32, 0.0f);
  const float zs = dgpp::nvfp4_tensor_scale(0.0f);
  require(zs == 1.0f, "zero tensor scale");
  std::vector<uint8_t> zp(16), zsc(2);
  dgpp::nvfp4_encode_row(z.data(), 32, zs, zp.data(), zsc.data());
  for (uint8_t b : zp) require(b == 0, "zero codes");
  for (uint8_t b : zsc) require(b == 0, "zero scales");
}
