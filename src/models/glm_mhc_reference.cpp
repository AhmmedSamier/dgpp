#include "models/glm_mhc_reference.hpp"

#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {

namespace {

double bf16_to_d(uint16_t v) {
  return static_cast<double>(bf16_bits_to_float(v));
}

// Rounds a double to bf16 via float. The intermediate float cast rounds at
// 2^-24 before the bf16 round at 2^-8 — double rounding only matters within
// 2^-24 of a bf16 boundary, far below the parity budgets.
uint16_t d_to_bf16(double v) {
  return float_to_bf16_bits(static_cast<float>(v));
}

double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

}  // namespace

void glm_mhc_ref_compute(const uint16_t* streams, const GlmMhcWeightsHost& w,
                         const GlmMhcConfig& cfg, int tokens,
                         GlmMhcRefResult& out) {
  GlmMhcConfig::validate_config(cfg);
  const int n = cfg.hc_mult;
  const int D = cfg.hidden;
  const int K = n * D;
  const int coeffs = (2 + n) * n;
  if (static_cast<int>(w.fn.size()) != coeffs * K ||
      static_cast<int>(w.base.size()) != coeffs ||
      w.scale.size() != 3)
    throw std::invalid_argument("glm_mhc_ref_compute: weight geometry");

  out.pre.assign(static_cast<size_t>(tokens) * n, 0.0);
  out.post.assign(static_cast<size_t>(tokens) * n, 0.0);
  out.comb.assign(static_cast<size_t>(tokens) * n * n, 0.0);
  out.collapsed.assign(static_cast<size_t>(tokens) * D, 0);

  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = streams + static_cast<size_t>(t) * K;

    // Unweighted RMSNorm over the flattened streams, in double.
    double ssq = 0.0;
    for (int d = 0; d < K; ++d) ssq += bf16_to_d(x[d]) * bf16_to_d(x[d]);
    const double inv_rms =
        1.0 / std::sqrt(ssq / K + cfg.norm_eps);

    // 24 logits against the normalized flat vector.
    std::vector<double> logits(coeffs, 0.0);
    for (int i = 0; i < coeffs; ++i) {
      double dot = 0.0;
      const uint16_t* fn_row = w.fn.data() + static_cast<size_t>(i) * K;
      for (int d = 0; d < K; ++d)
        dot += bf16_to_d(x[d]) * inv_rms * bf16_to_d(fn_row[d]);
      logits[i] = dot;
    }

    // pre / post.
    for (int i = 0; i < n; ++i) {
      out.pre[static_cast<size_t>(t) * n + i] =
          sigmoid_d(logits[i] * w.scale[0] + w.base[i]) + cfg.hc_eps;
      out.post[static_cast<size_t>(t) * n + i] =
          2.0 * sigmoid_d(logits[n + i] * w.scale[1] + w.base[n + i]);
    }

    // comb: row softmax of the affine logits, +eps, then Sinkhorn
    // (one column pass, then iters-1 row+column passes).
    double c[16];  // n <= 4 pinned
    for (int row = 0; row < n; ++row) {
      double m = -INFINITY;
      for (int col = 0; col < n; ++col) {
        const double v =
            logits[2 * n + row * n + col] * w.scale[2] + w.base[2 * n + row * n + col];
        m = std::max(m, v);
      }
      double denom = 0.0;
      for (int col = 0; col < n; ++col) {
        const double v =
            logits[2 * n + row * n + col] * w.scale[2] + w.base[2 * n + row * n + col];
        c[row * n + col] = std::exp(v - m);
        denom += c[row * n + col];
      }
      for (int col = 0; col < n; ++col) c[row * n + col] /= denom;
    }
    for (int i = 0; i < n * n; ++i) c[i] += cfg.hc_eps;
    auto col_pass = [&] {
      for (int col = 0; col < n; ++col) {
        double s = 0.0;
        for (int row = 0; row < n; ++row) s += c[row * n + col];
        for (int row = 0; row < n; ++row)
          c[row * n + col] = c[row * n + col] / (s + cfg.hc_eps);
      }
    };
    auto row_pass = [&] {
      for (int row = 0; row < n; ++row) {
        double s = 0.0;
        for (int col = 0; col < n; ++col) s += c[row * n + col];
        for (int col = 0; col < n; ++col)
          c[row * n + col] = c[row * n + col] / (s + cfg.hc_eps);
      }
    };
    col_pass();
    for (int it = 1; it < cfg.sinkhorn_iters; ++it) {
      row_pass();
      col_pass();
    }
    for (int i = 0; i < n * n; ++i)
      out.comb[static_cast<size_t>(t) * n * n + i] = c[i];

    // Collapse: sum_j pre[j] * streams[j][d], one bf16 round.
    for (int d = 0; d < D; ++d) {
      double v = 0.0;
      for (int j = 0; j < n; ++j)
        v += out.pre[static_cast<size_t>(t) * n + j] *
             bf16_to_d(x[static_cast<size_t>(j) * D + d]);
      out.collapsed[static_cast<size_t>(t) * D + d] = d_to_bf16(v);
    }
  }
}

void glm_mhc_ref_stream_update(const uint16_t* post_bf16,
                               const uint16_t* comb_bf16,
                               const uint16_t* sublayer_out,
                               const uint16_t* streams_in,
                               const GlmMhcConfig& cfg, int tokens,
                               uint16_t* streams_out) {
  const int n = cfg.hc_mult;
  const int D = cfg.hidden;
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* res = streams_in + static_cast<size_t>(t) * n * D;
    uint16_t* dst = streams_out + static_cast<size_t>(t) * n * D;
    const uint16_t* h = sublayer_out + static_cast<size_t>(t) * D;
    for (int i = 0; i < n; ++i) {
      const double pi = bf16_to_d(post_bf16[static_cast<size_t>(t) * n + i]);
      for (int d = 0; d < D; ++d) {
        const uint16_t t1 = d_to_bf16(pi * bf16_to_d(h[d]));
        double mix = 0.0;
        for (int j = 0; j < n; ++j)
          mix += bf16_to_d(comb_bf16[static_cast<size_t>(t) * n * n +
                                     j * n + i]) *
                 bf16_to_d(res[static_cast<size_t>(j) * D + d]);
        const uint16_t t2 = d_to_bf16(mix);
        dst[static_cast<size_t>(i) * D + d] =
            d_to_bf16(bf16_to_d(t1) + bf16_to_d(t2));
      }
    }
  }
}

void glm_mhc_ref_final_mean(const uint16_t* streams, const GlmMhcConfig& cfg,
                            int tokens, uint16_t* out) {
  const int n = cfg.hc_mult;
  const int D = cfg.hidden;
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* s = streams + static_cast<size_t>(t) * n * D;
    uint16_t* o = out + static_cast<size_t>(t) * D;
    for (int d = 0; d < D; ++d) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j)
        sum += bf16_to_d(s[static_cast<size_t>(j) * D + d]);
      o[d] = d_to_bf16(sum / n);
    }
  }
}

}  // namespace dgpp
