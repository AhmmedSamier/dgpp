#include "models/qwen/gr_reference.hpp"

#include <cmath>
#include <vector>

#include "common/dtypes.hpp"
#include "models/qwen/norm_reference.hpp"

namespace dgpp::qwen_ref {
namespace {
float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }
float sigmoid_f(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

void gemv_bf16(const uint16_t* act, const uint16_t* w, uint16_t* out, int64_t m, int n, int k) {
  for (int64_t row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int i = 0; i < k; ++i)
        acc = std::fma(bf16_bits_to_float(act[row * k + i]),
                       bf16_bits_to_float(w[static_cast<int64_t>(col) * k + i]), acc);
      out[row * n + col] = float_to_bf16_bits(acc);
    }
}

void gr_gate_act(uint16_t* t, int64_t rows, int r, int hc) {
  for (int64_t i = 0; i < rows * r; ++i) {
    const float v = bf16_bits_to_float(t[i]) / static_cast<float>(hc);
    t[i] = float_to_bf16_bits(v * sigmoid_f(v));
  }
}

void gr_mix_finish(const uint16_t* logits, const uint16_t* rn, uint16_t* x, int64_t rows, int hc,
                   int hidden) {
  for (int64_t row = 0; row < rows; ++row)
    for (int j = 0; j < hidden; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < hc; ++i) {
        const int64_t at = (row * hc + i) * hidden + j;
        const float g = rb(sigmoid_f(bf16_bits_to_float(logits[at])));
        acc += rb(g * bf16_bits_to_float(rn[at]));
      }
      x[row * hidden + j] = float_to_bf16_bits(acc / static_cast<float>(hc));
    }
}

void gr_combine(uint16_t* r_state, const uint16_t* rn, const uint16_t* w_inject,
                const uint16_t* y, int64_t rows, int hc, int hidden) {
  const int width = hc * hidden;
  std::vector<uint16_t> dots(static_cast<size_t>(hc));
  for (int64_t row = 0; row < rows; ++row) {
    gemv_bf16(rn + row * width, w_inject, dots.data(), 1, hc, width);
    for (int i = 0; i < hc; ++i) {
      const float s = 2.0f * rb(sigmoid_f(bf16_bits_to_float(dots[static_cast<size_t>(i)]) /
                                          static_cast<float>(hc)));
      for (int j = 0; j < hidden; ++j) {
        const int64_t at = (row * hc + i) * hidden + j;
        const float inj = rb(bf16_bits_to_float(y[row * hidden + j]) * s);
        r_state[at] = float_to_bf16_bits(bf16_bits_to_float(r_state[at]) + inj);
      }
    }
  }
}

void gr_mix(const uint16_t* r_state, const uint16_t* w_norm, const uint16_t* w_down,
            const uint16_t* w_up, uint16_t* rn, uint16_t* x, int64_t rows, int hc, int hidden,
            int rank, float eps) {
  const int width = hc * hidden;
  group_rmsnorm(r_state, w_norm, rn, rows, hc, hidden, eps);
  std::vector<uint16_t> t(static_cast<size_t>(rows) * rank), logits(static_cast<size_t>(rows) * width);
  gemv_bf16(rn, w_down, t.data(), rows, rank, width);
  gr_gate_act(t.data(), rows, rank, hc);
  gemv_bf16(t.data(), w_up, logits.data(), rows, width, rank);
  gr_mix_finish(logits.data(), rn, x, rows, hc, hidden);
}

}  // namespace dgpp::qwen_ref
