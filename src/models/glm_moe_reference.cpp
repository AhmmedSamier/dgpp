#include "models/glm_moe_reference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {

namespace {

double bf16_to_d(uint16_t v) {
  return static_cast<double>(bf16_bits_to_float(v));
}
uint16_t d_to_bf16(double v) {
  return float_to_bf16_bits(static_cast<float>(v));
}
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Weights in the payload are stored as bf16-rounded dequants on the STRICT
// path — the engine's scale_gemm rounds decoded weights to bf16, and the
// oracle must match that policy to isolate only accumulation gaps.
struct StrictQuantRow {
  std::vector<double> w;  // [cols]
};

// D[rows_out, cols_out] = act[rows_out, cols_in] @ W^T with W [n, k]:
// strict GEMM — bf16-rounded weights, double accumulation, bf16-rounded
// output (the engine's output dtype).
void strict_gemm(const std::vector<uint16_t>& act, int64_t act_stride,
                 const std::vector<double>& weights_bf16, int m, int n,
                 int k, std::vector<uint16_t>& out) {
  out.assign(static_cast<size_t>(m) * n, 0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights_bf16.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow =
          act.data() + static_cast<size_t>(mm) * act_stride;
      for (int kk = 0; kk < k; ++kk)
        acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = d_to_bf16(acc);
    }
  }
}

// The down projection: the same strict GEMM with the accumulator handed
// back UNROUNDED (the engine's launch_scale_gemm_f32) — the accumulation
// chain owns the single rounding.
void strict_gemm_raw(const std::vector<uint16_t>& act, int64_t act_stride,
                     const std::vector<double>& weights_bf16, int m, int n,
                     int k, std::vector<double>& out) {
  out.assign(static_cast<size_t>(m) * n, 0.0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = weights_bf16.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow =
          act.data() + static_cast<size_t>(mm) * act_stride;
      for (int kk = 0; kk < k; ++kk)
        acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = acc;
    }
  }
}

// Decode payload x scales, rounded to bf16 (the engine's weight policy).
std::vector<double> dequant_bf16_weights(const GlmQuantMatrixHost& mat) {
  const int64_t scale_cols = (mat.cols + 127) / 128;
  std::vector<double> w(static_cast<size_t>(mat.rows) * mat.cols);
  for (int64_t r = 0; r < mat.rows; ++r)
    for (int64_t c = 0; c < mat.cols; ++c) {
      const float dec = fp8_e4m3_bits_to_float(
          mat.payload[static_cast<size_t>(r) * mat.cols + c]);
      const float s =
          mat.scales[static_cast<size_t>(r / 128) * scale_cols + c / 128];
      w[static_cast<size_t>(r) * mat.cols + c] =
          bf16_to_d(float_to_bf16_bits(dec * s));
    }
  return w;
}

// The expert swiglu chain on already-computed gate/up bf16 outputs.
void swiglu_oracle(const std::vector<uint16_t>& gate,
                   const std::vector<uint16_t>& up, int64_t n, float limit,
                   std::vector<uint16_t>& out) {
  out.assign(gate.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    double g = bf16_to_d(gate[i]);
    double u = bf16_to_d(up[i]);
    if (g > limit) g = limit;
    u = std::min(std::max(u, static_cast<double>(-limit)),
                 static_cast<double>(limit));
    const uint16_t t = d_to_bf16(g * sigmoid_d(g));  // rounding 1
    out[i] = d_to_bf16(bf16_to_d(t) * u);            // rounding 2
  }
}

}  // namespace

GlmQuantMatrixHost glm_moe_host_view(const GlmMoeHostWeights& w,
                                     const GlmMoeConfig& cfg, int index) {
  // Payload layout: [E*3 expert matrices][3 shared matrices], each expert
  // triple ordered gate, up, down; gate/up [I, H], down [H, I].
  const int64_t I = cfg.inter, H = cfg.hidden;
  GlmQuantMatrixHost m;
  const bool down = index % 3 == 2;
  m.rows = down ? H : I;
  m.cols = down ? I : H;
  const int64_t numel = m.rows * m.cols;
  const int64_t scale_numel =
      ((m.rows + 127) / 128) * ((m.cols + 127) / 128);
  const size_t p_off = static_cast<size_t>(index) * numel;
  const size_t s_off = static_cast<size_t>(index) * scale_numel;
  if (p_off + numel > w.payloads.size() || s_off + scale_numel > w.scales.size())
    throw std::invalid_argument("glm_moe_host_view: index out of range");
  m.payload = w.payloads.data() + p_off;
  m.scales = w.scales.data() + s_off;
  return m;
}

GlmQuantMatrixHost glm_moe_host_shared(const GlmMoeHostWeights& w,
                                       const GlmMoeConfig& cfg, int m) {
  return glm_moe_host_view(w, cfg, cfg.n_experts * 3 + m);
}

void glm_moe_ref_router(const uint16_t* hidden, const uint16_t* gate,
                        const float* bias, const GlmMoeConfig& cfg,
                        int tokens, GlmMoeRouterRef& out) {
  GlmMoeConfig::validate_config(cfg);
  const int E = cfg.n_experts, H = cfg.hidden, K = cfg.top_k;
  out.ids.assign(static_cast<size_t>(tokens) * K, 0);
  out.weights.assign(static_cast<size_t>(tokens) * K, 0.f);
  out.biased.assign(static_cast<size_t>(tokens) * E, 0.0);

  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * H;
    std::vector<double> scores(E), biased(E);
    for (int e = 0; e < E; ++e) {
      double dot = 0.0;
      const uint16_t* w = gate + static_cast<size_t>(e) * H;
      for (int k = 0; k < H; ++k) dot += bf16_to_d(x[k]) * bf16_to_d(w[k]);
      scores[e] = sigmoid_d(dot);
      biased[e] = scores[e] + bias[e];
      out.biased[static_cast<size_t>(t) * E + e] = biased[e];
    }
    std::vector<int> sel;
    std::vector<double> wsel;
    std::vector<char> taken(E, 0);
    for (int r = 0; r < K; ++r) {
      int best = -1;
      double bv = -INFINITY;
      for (int e = 0; e < E; ++e)
        if (!taken[e] && biased[e] > bv) {
          bv = biased[e];
          best = e;
        }
      taken[best] = 1;
      sel.push_back(best);
      wsel.push_back(scores[best]);
    }
    // ascending id (insertion sort, mirroring the kernel)
    for (size_t i = 1; i < sel.size(); ++i) {
      const int id = sel[i];
      const double w = wsel[i];
      int j = static_cast<int>(i) - 1;
      while (j >= 0 && sel[j] > id) {
        sel[j + 1] = sel[j];
        wsel[j + 1] = wsel[j];
        --j;
      }
      sel[j + 1] = id;
      wsel[j + 1] = w;
    }
    double denom = 0.0;
    for (int i = 0; i < K; ++i) denom += wsel[i];
    denom += 1e-20;
    for (int i = 0; i < K; ++i) {
      out.ids[static_cast<size_t>(t) * K + i] = sel[i];
      const double w =
          cfg.norm_topk_prob ? (wsel[i] / denom) * cfg.routed_scaling_factor
                             : wsel[i] * cfg.routed_scaling_factor;
      out.weights[static_cast<size_t>(t) * K + i] = static_cast<float>(w);
    }
  }
}

void glm_moe_ref_forward(const uint16_t* hidden, const GlmMoeHostWeights& w,
                         const GlmMoeConfig& cfg, int tokens,
                         std::vector<uint16_t>& out) {
  GlmMoeConfig::validate_config(cfg);
  const int E = cfg.n_experts, H = cfg.hidden, I = cfg.inter, K = cfg.top_k;

  GlmMoeRouterRef route;
  glm_moe_ref_router(hidden, w.router_gate.data(), w.router_bias.data(), cfg,
                     tokens, route);

  out.assign(static_cast<size_t>(tokens) * H, 0);

  // Pre-dequantize every expert's three matrices (bf16-rounded weights) —
  // small geometries only; the real-checkpoint path is the torch backend.
  auto weights_for = [&](int index) {
    return dequant_bf16_weights(glm_moe_host_view(w, cfg, index));
  };

  // Per token: the fp32-chain semantics in double — the down projection's
  // dots stay unrounded, each expert's contribution is one fma (weight x
  // dot + acc) in ASCENDING expert order, the shared expert last with
  // weight 1, and the sum rounds to bf16 exactly once. (The transformers
  // reference rounds per expert add; the engine deliberately does not —
  // see glm_moe_layer.hpp.)
  std::vector<uint16_t> gate_out, up_out, act;
  std::vector<double> down_out, acc(static_cast<size_t>(H));
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * H;
    std::vector<uint16_t> xrow(x, x + H);
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int i = 0; i < K; ++i) {
      const int e = route.ids[static_cast<size_t>(t) * K + i];
      const double we =
          static_cast<double>(route.weights[static_cast<size_t>(t) * K + i]);
      const auto wg = weights_for(e * 3 + 0);
      const auto wu = weights_for(e * 3 + 1);
      const auto wd = weights_for(e * 3 + 2);
      strict_gemm(xrow, H, wg, 1, I, H, gate_out);
      strict_gemm(xrow, H, wu, 1, I, H, up_out);
      swiglu_oracle(gate_out, up_out, I, cfg.swiglu_limit, act);
      strict_gemm_raw(act, I, wd, 1, H, I, down_out);
      for (int d = 0; d < H; ++d) acc[d] += we * down_out[d];
    }
    // Shared expert, weight 1, added last.
    {
      const auto wg = weights_for(E * 3 + 0);
      const auto wu = weights_for(E * 3 + 1);
      const auto wd = weights_for(E * 3 + 2);
      strict_gemm(xrow, H, wg, 1, I, H, gate_out);
      strict_gemm(xrow, H, wu, 1, I, H, up_out);
      swiglu_oracle(gate_out, up_out, I, cfg.swiglu_limit, act);
      strict_gemm_raw(act, I, wd, 1, H, I, down_out);
      for (int d = 0; d < H; ++d) acc[d] += down_out[d];
    }
    for (int d = 0; d < H; ++d)
      out[static_cast<size_t>(t) * H + d] = d_to_bf16(acc[d]);
  }
}

}  // namespace dgpp
