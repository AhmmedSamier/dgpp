#include "models/kda_reference.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp::kda_ref {

namespace {

template <typename Acc>
inline Acc to_acc(uint16_t bf16_bits) {
  return static_cast<Acc>(bf16_bits_to_float(bf16_bits));
}

template <typename Acc>
inline Acc sigmoid_acc(Acc x) {
  return Acc(1) / (Acc(1) + std::exp(-x));
}

}  // namespace

template <typename Acc>
void gemm_bf16(const uint16_t* act, int64_t act_row_stride, const uint16_t* w,
               uint16_t* out, int m, int n, int k) {
  for (int r = 0; r < m; ++r) {
    const uint16_t* arow = act + static_cast<int64_t>(r) * act_row_stride;
    for (int c = 0; c < n; ++c) {
      const uint16_t* wrow = w + static_cast<int64_t>(c) * k;
      Acc acc = 0;
      for (int i = 0; i < k; ++i) acc += to_acc<Acc>(arow[i]) * to_acc<Acc>(wrow[i]);
      out[static_cast<int64_t>(r) * n + c] =
          float_to_bf16_bits(static_cast<float>(acc));
    }
  }
}

template <typename Acc>
void conv_silu(const uint16_t* src, int64_t src_row_stride, const uint16_t* w,
               uint16_t* state, int state_width, uint16_t* dst, int tokens,
               int channels, int conv_width) {
  // Per-channel rolling history, same association order as the device kernel
  // so the fp32 instantiation is as close to bitwise as host libm allows.
  for (int c = 0; c < channels; ++c) {
    const uint16_t* wc = w + static_cast<int64_t>(c) * conv_width;
    uint16_t* sc = state + static_cast<int64_t>(c) * state_width;
    Acc hist[8];
    for (int j = 0; j < conv_width - 1; ++j) hist[j] = to_acc<Acc>(sc[j]);
    for (int t = 0; t < tokens; ++t) {
      const Acc x = to_acc<Acc>(src[static_cast<int64_t>(t) * src_row_stride + c]);
      Acc acc = to_acc<Acc>(wc[conv_width - 1]) * x;
      for (int j = 0; j < conv_width - 1; ++j)
        acc += to_acc<Acc>(wc[j]) * hist[j];
      const Acc y = acc * sigmoid_acc(acc);
      dst[static_cast<int64_t>(t) * channels + c] =
          float_to_bf16_bits(static_cast<float>(y));
      for (int j = 0; j < conv_width - 2; ++j) hist[j] = hist[j + 1];
      hist[conv_width - 2] = x;
    }
    for (int j = 0; j < conv_width - 1; ++j)
      sc[j] = float_to_bf16_bits(static_cast<float>(hist[j]));
  }
}

template <typename Acc>
void recurrent(const uint16_t* qkv, const uint16_t* g_raw,
               const uint16_t* beta_raw, int64_t beta_row_stride,
               const float* a_log, const float* dt_bias, Acc* state,
               uint16_t* out, int tokens, int heads, int k_dim, int v_dim,
               float lower_bound, float scale) {
  const int64_t qkv_stride = static_cast<int64_t>(2) * heads * k_dim +
                             static_cast<int64_t>(heads) * v_dim;
  // Scratch reused across (token, head) — the reference is a cold path but
  // not so cold that per-token allocations are acceptable under sanitizers.
  std::vector<Acc> kq(static_cast<size_t>(k_dim)),
      qq(static_cast<size_t>(k_dim));
  for (int h = 0; h < heads; ++h) {
    const Acc a = std::exp(static_cast<Acc>(a_log[h]));
    // state slice for head h: [v_dim, k_dim]
    Acc* s = state + static_cast<int64_t>(h) * v_dim * k_dim;
    for (int t = 0; t < tokens; ++t) {
      const uint16_t* qrow =
          qkv + static_cast<int64_t>(t) * qkv_stride + static_cast<int64_t>(h) * k_dim;
      const uint16_t* krow = qrow + static_cast<int64_t>(heads) * k_dim;
      const uint16_t* vrow =
          qkv + static_cast<int64_t>(t) * qkv_stride +
          static_cast<int64_t>(2) * heads * k_dim + static_cast<int64_t>(h) * v_dim;
      const uint16_t* grow =
          g_raw + (static_cast<int64_t>(t) * heads + h) * k_dim;
      const float* bias = dt_bias + static_cast<int64_t>(h) * k_dim;

      // l2norm with eps inside the sqrt, then q scaling. Normalized vectors
      // are hoisted per (token, head) — same association order as the
      // device kernel (which scales k/q once and fmas against them).
      Acc qs = 0, ks = 0;
      for (int i = 0; i < k_dim; ++i) {
        qs += to_acc<Acc>(qrow[i]) * to_acc<Acc>(qrow[i]);
        ks += to_acc<Acc>(krow[i]) * to_acc<Acc>(krow[i]);
      }
      const Acc qn = Acc(1) / std::sqrt(qs + Acc(1e-6));
      const Acc kn = Acc(1) / std::sqrt(ks + Acc(1e-6));
      for (int i = 0; i < k_dim; ++i) {
        kq[i] = to_acc<Acc>(krow[i]) * kn;
        qq[i] = to_acc<Acc>(qrow[i]) * qn * static_cast<Acc>(scale);
      }

      // Decay along K columns with the bounded safe gate.
      for (int i = 0; i < k_dim; ++i) {
        const Acc g = to_acc<Acc>(grow[i]) + static_cast<Acc>(bias[i]);
        const Acc gate = static_cast<Acc>(lower_bound) /
                         (Acc(1) + std::exp(-(a * g)));
        const Acc decay = std::exp(gate);
        for (int v = 0; v < v_dim; ++v)
          s[static_cast<int64_t>(v) * k_dim + i] *= decay;
      }

      // Delta error u = beta * (v - S k), rank-1 write, then output read.
      const Acc beta = sigmoid_acc(
          to_acc<Acc>(beta_raw[static_cast<int64_t>(t) * beta_row_stride + h]));
      for (int v = 0; v < v_dim; ++v) {
        Acc* srow = s + static_cast<int64_t>(v) * k_dim;
        Acc dot = 0;
        for (int i = 0; i < k_dim; ++i) dot += srow[i] * kq[i];
        const Acc u = (to_acc<Acc>(vrow[v]) - dot) * beta;
        Acc o = 0;
        for (int i = 0; i < k_dim; ++i) {
          srow[i] += u * kq[i];
          o += srow[i] * qq[i];
        }
        out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] =
            float_to_bf16_bits(static_cast<float>(o));
      }
    }
  }
}

template <typename Acc>
void gated_rmsnorm(const uint16_t* x, const uint16_t* gate, const uint16_t* w,
                   uint16_t* y, int64_t rows, int dim, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* xr = x + r * dim;
    const uint16_t* gr = gate + r * dim;
    uint16_t* yr = y + r * dim;
    Acc var = 0;
    for (int i = 0; i < dim; ++i) var += to_acc<Acc>(xr[i]) * to_acc<Acc>(xr[i]);
    const Acc rstd = Acc(1) / std::sqrt(var / dim + static_cast<Acc>(eps));
    for (int i = 0; i < dim; ++i)
      yr[i] = float_to_bf16_bits(static_cast<float>(
          to_acc<Acc>(xr[i]) * rstd * to_acc<Acc>(w[i]) *
          sigmoid_acc(to_acc<Acc>(gr[i]))));
  }
}

template <typename Acc>
void layer_forward(const HostWeights& w, const KdaConfig& cfg,
                   const uint16_t* hidden_in, Acc* state, uint16_t* conv_state,
                   int conv_state_width, uint16_t* layer_out,
                   uint16_t* core_out, int tokens) {
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  const int hid = cfg.hidden;
  const int lp = g.local_proj;
  const int n_in = g.in_proj_cols;
  // Must mirror KdaLayer's fused layout: [f_a | g_a | q | k | v | b].
  const int off_q = 2 * cfg.head_dim;
  const int off_k = off_q + lp;
  const int off_v = off_k + lp;
  const int off_b = off_v + lp;
  const float scale = static_cast<float>(
      std::pow(static_cast<double>(cfg.head_dim), -0.5));

  std::vector<uint16_t> proj(static_cast<size_t>(tokens) * n_in);
  std::vector<uint16_t> g1(static_cast<size_t>(tokens) * lp);
  std::vector<uint16_t> g2(static_cast<size_t>(tokens) * lp);
  std::vector<uint16_t> qkv_conv(static_cast<size_t>(tokens) * g.conv_channels);
  std::vector<uint16_t> core(static_cast<size_t>(tokens) * lp);
  std::vector<uint16_t> normed(static_cast<size_t>(tokens) * lp);

  gemm_bf16<Acc>(hidden_in, hid, w.in_proj, proj.data(), tokens, n_in, hid);
  gemm_bf16<Acc>(proj.data(), n_in, w.f_b, g1.data(), tokens, lp,
                 cfg.head_dim);
  gemm_bf16<Acc>(proj.data() + cfg.head_dim, n_in, w.g_b, g2.data(), tokens,
                 lp, cfg.head_dim);
  conv_silu<Acc>(proj.data() + off_q, n_in, w.conv, conv_state,
                 conv_state_width, qkv_conv.data(), tokens, g.conv_channels,
                 cfg.conv_width);
  recurrent<Acc>(qkv_conv.data(), g1.data(), proj.data() + off_b, n_in, w.a_log,
                 w.dt_bias, state, core.data(), tokens, g.local_heads,
                 cfg.head_dim, cfg.head_dim, cfg.lower_bound, scale);
  if (core_out)
    std::copy(core.begin(), core.end(), core_out);
  gated_rmsnorm<Acc>(core.data(), g2.data(), w.o_norm, normed.data(),
                     static_cast<int64_t>(tokens) * g.local_heads,
                     cfg.head_dim, 1e-5f);
  gemm_bf16<Acc>(normed.data(), lp, w.o_proj, layer_out, tokens, hid, lp);
}

// Explicit instantiations used by tests and the dump harness comparison.
template void gemm_bf16<float>(const uint16_t*, int64_t, const uint16_t*,
                               uint16_t*, int, int, int);
template void gemm_bf16<double>(const uint16_t*, int64_t, const uint16_t*,
                                uint16_t*, int, int, int);
template void conv_silu<float>(const uint16_t*, int64_t, const uint16_t*,
                               uint16_t*, int, uint16_t*, int, int, int);
template void conv_silu<double>(const uint16_t*, int64_t, const uint16_t*,
                                uint16_t*, int, uint16_t*, int, int, int);
template void recurrent<float>(const uint16_t*, const uint16_t*,
                               const uint16_t*, int64_t, const float*,
                               const float*, float*, uint16_t*, int, int, int,
                               int, float, float);
template void recurrent<double>(const uint16_t*, const uint16_t*,
                                const uint16_t*, int64_t, const float*,
                                const float*, double*, uint16_t*, int, int,
                                int, int, float, float);
template void gated_rmsnorm<float>(const uint16_t*, const uint16_t*,
                                   const uint16_t*, uint16_t*, int64_t, int,
                                   float);
template void gated_rmsnorm<double>(const uint16_t*, const uint16_t*,
                                    const uint16_t*, uint16_t*, int64_t, int,
                                    float);
template void layer_forward<float>(const HostWeights&, const KdaConfig&,
                                   const uint16_t*, float*, uint16_t*, int,
                                   uint16_t*, uint16_t*, int);
template void layer_forward<double>(const HostWeights&, const KdaConfig&,
                                    const uint16_t*, double*, uint16_t*, int,
                                    uint16_t*, uint16_t*, int);

}  // namespace dgpp::kda_ref
