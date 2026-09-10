#include "models/qwen/ple_reference.hpp"

#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "models/qwen/gr_reference.hpp"
#include "models/qwen/norm_reference.hpp"

namespace dgpp::qwen_ref {
namespace {
float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }
float sigmoid_f(float v) { return 1.0f / (1.0f + std::exp(-v)); }
}  // namespace

void ple_hash_ids(const int32_t* tokens, int n, int32_t ctx_prev1, int32_t ctx_prev2, int32_t eos,
                  const int64_t* multipliers, const int64_t* head_vocab,
                  const int64_t* head_offset, int heads, int heads_per_ngram, int32_t* ids) {
  for (int t = 0; t < n; ++t) {
    const int64_t y0 = tokens[t];
    const int64_t prev1 = t >= 1 ? tokens[t - 1] : ctx_prev1;
    const int64_t prev2 = t >= 2 ? tokens[t - 2] : (t == 1 ? ctx_prev1 : ctx_prev2);
    const int64_t y1 = prev1;
    const int64_t y2 = prev1 == eos ? eos : prev2;
    const uint64_t m0 = static_cast<uint64_t>(y0) * static_cast<uint64_t>(multipliers[0]);
    const uint64_t m1 = static_cast<uint64_t>(y1) * static_cast<uint64_t>(multipliers[1]);
    const uint64_t m2 = static_cast<uint64_t>(y2) * static_cast<uint64_t>(multipliers[2]);
    for (int h = 0; h < heads; ++h) {
      uint64_t mix = m0 ^ m1;
      if (h >= heads_per_ngram) mix ^= m2;
      const int64_t mixed = static_cast<int64_t>(mix);
      int64_t r = mixed % head_vocab[h];
      if (r < 0) r += head_vocab[h];
      ids[static_cast<int64_t>(t) * heads + h] = static_cast<int32_t>(r + head_offset[h]);
    }
  }
}

void ple_gather(const uint8_t* table, int64_t row_begin, float scale, const int32_t* ids, int n,
                int heads, int head_begin, int heads_local, int head_dim, uint16_t* out) {
  for (int t = 0; t < n; ++t)
    for (int hl = 0; hl < heads_local; ++hl) {
      const int64_t row = ids[static_cast<int64_t>(t) * heads + head_begin + hl] - row_begin;
      if (row < 0) throw std::runtime_error("ple_gather: id below the slice");
      for (int d = 0; d < head_dim; ++d)
        out[(static_cast<int64_t>(t) * heads_local + hl) * head_dim + d] =
            float_to_bf16_bits(fp8_e4m3_bits_to_float(table[row * head_dim + d]) * scale);
    }
}

void ple_gate(const uint16_t* key_n, const uint16_t* query_n, const uint16_t* value,
              uint16_t* gated, int n, int hc, int hidden) {
  const float divisor = static_cast<float>(std::sqrt(static_cast<double>(hidden)));
  for (int t = 0; t < n; ++t)
    for (int i = 0; i < hc; ++i) {
      const int64_t base = (static_cast<int64_t>(t) * hc + i) * hidden;
      float acc = 0.f;
      for (int d = 0; d < hidden; ++d)
        acc += rb(bf16_bits_to_float(key_n[base + d]) * bf16_bits_to_float(query_n[base + d]));
      float g = rb(acc);
      g = rb(g / divisor);
      const float sign = g < 0.f ? -1.f : (g > 0.f ? 1.f : 0.f);
      g = rb(std::sqrt(std::max(std::fabs(g), 1e-6f))) * sign;
      const float s = rb(sigmoid_f(g));
      for (int d = 0; d < hidden; ++d)
        gated[base + d] =
            float_to_bf16_bits(s * bf16_bits_to_float(value[static_cast<int64_t>(t) * hidden + d]));
    }
}

void ple_conv(const uint16_t* un, const uint16_t* gv, uint16_t* state, const uint16_t* weight,
              const uint16_t* residual, uint16_t* out, int n, int channels, int width,
              int dilation) {
  const int S = (width - 1) * dilation;
  std::vector<float> hist(static_cast<size_t>(S));
  for (int c = 0; c < channels; ++c) {
    for (int j = 0; j < S; ++j) hist[j] = bf16_bits_to_float(state[static_cast<int64_t>(c) * S + j]);
    for (int t = 0; t < n; ++t) {
      const int64_t at = static_cast<int64_t>(t) * channels + c;
      const float u = bf16_bits_to_float(un[at]);
      float acc = 0.f;
      for (int k = 0; k < width - 1; ++k)
        acc = std::fma(bf16_bits_to_float(weight[static_cast<int64_t>(c) * width + k]),
                       hist[static_cast<size_t>(k * dilation)], acc);
      acc = std::fma(bf16_bits_to_float(weight[static_cast<int64_t>(c) * width + width - 1]), u, acc);
      const float cv = rb(acc);
      const float act = rb(cv * sigmoid_f(cv));
      const float ple = rb(bf16_bits_to_float(gv[at]) + act);
      const float res = residual ? bf16_bits_to_float(residual[at]) : 0.f;
      out[at] = float_to_bf16_bits(residual ? res + ple : ple);
      for (int j = 0; j + 1 < S; ++j) hist[j] = hist[j + 1];
      hist[S - 1] = u;
    }
    for (int j = 0; j < S; ++j) state[static_cast<int64_t>(c) * S + j] = float_to_bf16_bits(hist[j]);
  }
}

void ple_forward(const uint16_t* r, const uint16_t* e, const PleWeights& w, uint16_t* conv_state,
                 uint16_t* r_out, int n, int hc, int hidden, int embed_dim, int width,
                 int dilation, float eps) {
  const int64_t C = static_cast<int64_t>(hc) * hidden;
  std::vector<uint16_t> key(static_cast<size_t>(n) * C), key_n(key.size()), query_n(key.size()),
      value(static_cast<size_t>(n) * hidden), gated(key.size()), un(key.size());
  gemv_bf16(e, w.key_proj, key.data(), n, static_cast<int>(C), embed_dim);
  group_rmsnorm(key.data(), w.norm_key, key_n.data(), n, hc, hidden, eps);
  gemv_bf16(e, w.value_proj, value.data(), n, hidden, embed_dim);
  group_rmsnorm(r, w.norm_query, query_n.data(), n, hc, hidden, eps);
  ple_gate(key_n.data(), query_n.data(), value.data(), gated.data(), n, hc, hidden);
  group_rmsnorm(gated.data(), w.norm_conv, un.data(), n, hc, hidden, eps);
  ple_conv(un.data(), gated.data(), conv_state, w.conv, r, r_out, n, static_cast<int>(C), width,
           dilation);
}

}  // namespace dgpp::qwen_ref
