#include "models/qwen/moe_reference.hpp"

#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {
namespace {

double bf16_to_d(uint16_t v) { return static_cast<double>(bf16_bits_to_float(v)); }
uint16_t d_to_bf16(double v) { return float_to_bf16_bits(static_cast<float>(v)); }
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// out[m, n] = bf16(sum_k act[m, k] * w[n, k]) in double (strict: the
// engine rounds each projection's output to bf16).
void strict_gemm(const uint16_t* act, int64_t act_stride,
                 const std::vector<double>& w, int m, int n, int k,
                 std::vector<uint16_t>& out) {
  out.assign(static_cast<size_t>(m) * n, 0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = w.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      const uint16_t* arow = act + static_cast<size_t>(mm) * act_stride;
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = d_to_bf16(acc);
    }
  }
}

// The down projection: handed back unrounded.
void raw_gemm(const uint16_t* act, int64_t act_stride,
              const std::vector<double>& w, int m, int n, int k,
              std::vector<double>& out) {
  out.assign(static_cast<size_t>(m) * n, 0.0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = w.data() + static_cast<size_t>(nn) * k;
    for (int mm = 0; mm < m; ++mm) {
      const uint16_t* arow = act + static_cast<size_t>(mm) * act_stride;
      double acc = 0.0;
      for (int kk = 0; kk < k; ++kk) acc += bf16_to_d(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * n + nn] = acc;
    }
  }
}

// silu(gate) * up with the engine's two roundings, no clamps.
void swiglu(const std::vector<uint16_t>& gate, const std::vector<uint16_t>& up,
            std::vector<uint16_t>& out) {
  out.assign(gate.size(), 0);
  for (size_t i = 0; i < gate.size(); ++i) {
    const double g = bf16_to_d(gate[i]);
    const double u = bf16_to_d(up[i]);
    const uint16_t t = d_to_bf16(g * sigmoid_d(g));
    out[i] = d_to_bf16(bf16_to_d(t) * u);
  }
}

// FP8 weights as the kernels see them: decode x scale in fp32, one bf16
// rounding (the strict rule of scale_gemm_test), on the matrix's own grid.
std::vector<double> dequant_fp8(const QwenMoeHostWeights& w, int index) {
  const int m = index % 3;
  const int64_t R = w.rows(m), C = w.cols(m);
  const int sbr = w.scale_block_rows(m), sbc = w.scale_block_cols(m);
  const int64_t sc = w.scale_cols(m);
  const uint8_t* payload = w.payloads.data() + w.payload_offset(index);
  const float* scales = w.scales.data() + w.scale_offset(index);
  std::vector<double> out(static_cast<size_t>(R * C));
  for (int64_t r = 0; r < R; ++r)
    for (int64_t c = 0; c < C; ++c) {
      const float dec = fp8_e4m3_bits_to_float(payload[r * C + c]);
      const float s = scales[(r / sbr) * sc + c / sbc];
      out[static_cast<size_t>(r * C + c)] = bf16_to_d(float_to_bf16_bits(dec * s));
    }
  return out;
}

std::vector<double> widen(const std::vector<uint16_t>& v) {
  std::vector<double> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = bf16_to_d(v[i]);
  return out;
}

}  // namespace

size_t QwenMoeHostWeights::payload_offset(int index) const {
  const size_t per_expert = payload_bytes(0) + payload_bytes(1) + payload_bytes(2);
  size_t off = static_cast<size_t>(index / 3) * per_expert;
  for (int m = 0; m < index % 3; ++m) off += payload_bytes(m);
  return off;
}

size_t QwenMoeHostWeights::scale_offset(int index) const {
  const size_t per_expert = scale_count(0) + scale_count(1) + scale_count(2);
  size_t off = static_cast<size_t>(index / 3) * per_expert;
  for (int m = 0; m < index % 3; ++m) off += scale_count(m);
  return off;
}

void QwenMoeHostWeights::allocate() {
  payloads.assign(payload_offset(n_experts * 3), 0);
  scales.assign(scale_offset(n_experts * 3), 0.f);
  router.assign(static_cast<size_t>(n_experts) * hidden, 0);
  shared_gate.assign(static_cast<size_t>(hidden), 0);
  shared_w[0].assign(static_cast<size_t>(shared_inter) * hidden, 0);
  shared_w[1].assign(static_cast<size_t>(shared_inter) * hidden, 0);
  shared_w[2].assign(static_cast<size_t>(hidden) * shared_inter, 0);
}

void qwen_moe_ref_forward(const uint16_t* hidden, const QwenMoeHostWeights& w,
                          const GlmMoeConfig& cfg, int tokens,
                          std::vector<uint16_t>& out, GlmMoeRouterRef* route) {
  GlmMoeConfig::validate_config(cfg);
  if (cfg.router_mode != MoeRouterMode::SoftmaxTopk || cfg.n_shared_experts != 0)
    throw std::invalid_argument("qwen_moe_ref_forward: the config must be the Qwen routed config");
  if (cfg.hidden != w.hidden || cfg.inter != w.inter || cfg.n_experts != w.n_experts)
    throw std::invalid_argument("qwen_moe_ref_forward: config/weights geometry mismatch");
  const int E = cfg.n_experts, H = cfg.hidden, I = cfg.inter, K = cfg.top_k;
  const int S = w.shared_inter;

  GlmMoeRouterRef local_route;
  GlmMoeRouterRef& r = route ? *route : local_route;
  glm_moe_ref_router(hidden, w.router.data(), nullptr, cfg, tokens, r);

  // Every expert's three matrices dequantized once (small geometries).
  std::vector<std::vector<double>> mats(static_cast<size_t>(E) * 3);
  for (int i = 0; i < E * 3; ++i) mats[static_cast<size_t>(i)] = dequant_fp8(w, i);
  const std::vector<double> sg = widen(w.shared_w[0]);
  const std::vector<double> su = widen(w.shared_w[1]);
  const std::vector<double> sd = widen(w.shared_w[2]);

  out.assign(static_cast<size_t>(tokens) * H, 0);
  std::vector<uint16_t> gate_out, up_out, act;
  std::vector<double> down_out, acc(static_cast<size_t>(H));
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * H;
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int i = 0; i < K; ++i) {
      const int e = r.ids[static_cast<size_t>(t) * K + i];
      const double we = static_cast<double>(r.weights[static_cast<size_t>(t) * K + i]);
      strict_gemm(x, H, mats[static_cast<size_t>(e) * 3 + 0], 1, I, H, gate_out);
      strict_gemm(x, H, mats[static_cast<size_t>(e) * 3 + 1], 1, I, H, up_out);
      swiglu(gate_out, up_out, act);
      raw_gemm(act.data(), I, mats[static_cast<size_t>(e) * 3 + 2], 1, H, I, down_out);
      for (int d = 0; d < H; ++d) acc[d] += we * down_out[d];
    }
    // The shared expert, weighed by sigma(x . g): the logit a bf16 Linear
    // output, the sigmoid a bf16 op — two roundings, as the reference.
    strict_gemm(x, H, sg, 1, S, H, gate_out);
    strict_gemm(x, H, su, 1, S, H, up_out);
    swiglu(gate_out, up_out, act);
    raw_gemm(act.data(), S, sd, 1, H, S, down_out);
    double logit = 0.0;
    for (int k = 0; k < H; ++k) logit += bf16_to_d(x[k]) * bf16_to_d(w.shared_gate[k]);
    logit = bf16_to_d(d_to_bf16(logit));
    const double ws = bf16_to_d(d_to_bf16(sigmoid_d(logit)));
    for (int d = 0; d < H; ++d) acc[d] += ws * down_out[d];
    for (int d = 0; d < H; ++d) out[static_cast<size_t>(t) * H + d] = d_to_bf16(acc[d]);
  }
}

}  // namespace dgpp
