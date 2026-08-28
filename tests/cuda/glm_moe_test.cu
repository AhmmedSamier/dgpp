// Parity tests for the MoE module (M4): router kernel vs the double oracle
// at REAL geometry (E=288, H=4096), the full expert path (router ->
// ascending-accumulation -> shared) on small geometry, swiglu clamp edge
// cases, accumulation-order exposure, and determinism. Tolerance design:
// the router's fp32 pipeline vs the oracle's double differs by ~1e-6
// relative, so id swaps are certified against the biased-score gap (the
// near-tie discipline the DSA selection audit established); the expert path
// adds the scale-gemm mma-order gap on top (same class as scale_gemm_test's
// strict oracle, budgeted there at 0 mismatches — here slightly loosened
// for the extra chained roundings).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "models/glm_moe.hpp"
#include "models/glm_moe_layer.hpp"
#include "models/glm_moe_reference.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::GlmMoeConfig;
using dgpp::GlmMoeLayer;
using dgpp::GlmMoeWeights;
using dgpp::GlmMoeHostWeights;
using dgpp::GlmMoeRouterRef;
using dgpp::GlmQuantMatrix;
using scale_gemm_test::require;

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu)
                         : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

// ---- router parity at real geometry -------------------------------------

void check_router(const GlmMoeConfig& cfg, int tokens, uint64_t seed) {
  Rng rng(seed);
  const int E = cfg.n_experts, H = cfg.hidden, K = cfg.top_k;

  std::vector<uint16_t> hidden(static_cast<size_t>(tokens) * H);
  fill_act(rng, hidden);
  std::vector<uint16_t> gate(static_cast<size_t>(E) * H);
  for (auto& v : gate) v = float_to_bf16_bits(0.05f * static_cast<float>(rng.unit()));
  std::vector<float> bias(E);
  for (auto& v : bias) v = 0.25f * static_cast<float>(rng.unit());

  uint16_t* d_hidden = nullptr;
  uint16_t* d_gate = nullptr;
  float* d_bias = nullptr;
  int32_t* d_ids = nullptr;
  float* d_w = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_hidden, hidden.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_gate, gate.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_bias, bias.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, static_cast<size_t>(tokens) * K * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, static_cast<size_t>(tokens) * K * 4));
  std::memcpy(d_hidden, hidden.data(), hidden.size() * 2);
  std::memcpy(d_gate, gate.data(), gate.size() * 2);
  std::memcpy(d_bias, bias.data(), bias.size() * 4);

  dgpp::launch_moe_router(d_hidden, d_gate, d_bias, d_ids, d_w, cfg, tokens,
                          nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  GlmMoeRouterRef ref;
  dgpp::glm_moe_ref_router(hidden.data(), gate.data(), bias.data(), cfg,
                           tokens, ref);

  std::vector<int32_t> got_ids(static_cast<size_t>(tokens) * K);
  std::vector<float> got_w(static_cast<size_t>(tokens) * K);
  std::memcpy(got_ids.data(), d_ids, got_ids.size() * 4);
  std::memcpy(got_w.data(), d_w, got_w.size() * 4);

  long certified = 0, hard = 0;
  double max_rel = 0;
  for (size_t i = 0; i < got_ids.size(); ++i) {
    if (got_ids[i] != ref.ids[i]) {
      // Near-tie certification: an id swap is acceptable only when the two
      // experts' biased scores differ by < 1e-5 (fp32-vs-double noise floor).
      const int t = static_cast<int>(i / K);
      const double a = ref.biased[static_cast<size_t>(t) * E + got_ids[i]];
      const double b = ref.biased[static_cast<size_t>(t) * E + ref.ids[i]];
      if (std::abs(a - b) < 1e-5)
        ++certified;
      else
        ++hard;
      continue;
    }
    const double rel = std::abs(static_cast<double>(got_w[i]) -
                                static_cast<double>(ref.weights[i])) /
                       std::max(1e-30,
                                std::abs(static_cast<double>(ref.weights[i])));
    max_rel = std::max(max_rel, rel);
  }
  DGPP_CUDA_OK(cudaFree(d_hidden));
  DGPP_CUDA_OK(cudaFree(d_gate));
  DGPP_CUDA_OK(cudaFree(d_bias));
  DGPP_CUDA_OK(cudaFree(d_ids));
  DGPP_CUDA_OK(cudaFree(d_w));

  require(hard == 0, "router id mismatch beyond near-tie certification");
  require(max_rel < 1e-5, "router weights within 1e-5 relative of oracle");
  std::printf("[ OK ] router E=%d H=%d K=%d tokens=%d: %ld certified "
              "near-tie swaps, max weight rel err %.2g\n",
              E, H, K, tokens, certified, max_rel);
}

// ---- full expert path on small geometry ----------------------------------

struct SmallCase {
  GlmMoeConfig cfg;
  int tokens = 0;
  std::vector<uint16_t> hidden;
  GlmMoeHostWeights host_w;
  GlmMoeWeights dev_w;
  std::vector<GlmQuantMatrix> expert_mats;
  GlmQuantMatrix shared_mats[3];
  // device backing
  uint16_t *d_hidden = nullptr, *d_gate_w = nullptr;
  float* d_bias = nullptr;
  std::vector<uint8_t*> d_payloads;
  std::vector<float*> d_scales;
  uint16_t* d_out = nullptr;

  void alloc() {
    const int E = cfg.n_experts;
    DGPP_CUDA_OK(cudaMallocManaged(&d_hidden, hidden.size() * 2));
    std::memcpy(d_hidden, hidden.data(), hidden.size() * 2);
    DGPP_CUDA_OK(cudaMallocManaged(&d_gate_w, host_w.router_gate.size() * 2));
    std::memcpy(d_gate_w, host_w.router_gate.data(),
                host_w.router_gate.size() * 2);
    DGPP_CUDA_OK(cudaMallocManaged(&d_bias, host_w.router_bias.size() * 4));
    std::memcpy(d_bias, host_w.router_bias.data(),
                host_w.router_bias.size() * 4);

    const int mats = (E + 1) * 3;
    expert_mats.resize(mats);
    for (int m = 0; m < mats; ++m) {
      auto view = dgpp::glm_moe_host_view(host_w, cfg, m);
      uint8_t* p = nullptr;
      float* s = nullptr;
      const size_t pn = static_cast<size_t>(view.rows) * view.cols;
      const size_t sn = ((view.rows + 127) / 128) * ((view.cols + 127) / 128);
      DGPP_CUDA_OK(cudaMallocManaged(&p, pn));
      DGPP_CUDA_OK(cudaMallocManaged(&s, sn * 4));
      std::memcpy(p, view.payload, pn);
      std::memcpy(s, view.scales, sn * 4);
      d_payloads.push_back(p);
      d_scales.push_back(s);
      expert_mats[m] = GlmQuantMatrix{p, s, view.rows, view.cols};
    }
    dev_w.router_gate = d_gate_w;
    dev_w.router_bias = d_bias;
    dev_w.experts = expert_mats.data();
    for (int m = 0; m < 3; ++m)
      dev_w.shared[m] = expert_mats[static_cast<size_t>(E) * 3 + m];
    DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) *
                                              cfg.hidden * 2));
  }

  void free_all() {
    cudaFree(d_hidden); cudaFree(d_gate_w); cudaFree(d_bias); cudaFree(d_out);
    for (auto* p : d_payloads) cudaFree(p);
    for (auto* s : d_scales) cudaFree(s);
  }
};

SmallCase make_small_case(int E, int H, int I, int K, int tokens,
                          uint64_t seed) {
  SmallCase c;
  c.cfg.hidden = H;
  c.cfg.inter = I;
  c.cfg.n_experts = E;
  c.cfg.top_k = K;
  c.cfg.routed_scaling_factor = 2.5f;
  c.cfg.norm_topk_prob = true;
  c.cfg.swiglu_limit = 10.0f;
  c.tokens = tokens;
  Rng rng(seed);

  c.hidden.resize(static_cast<size_t>(tokens) * H);
  fill_act(rng, c.hidden);
  c.host_w.router_gate.resize(static_cast<size_t>(E) * H);
  for (auto& v : c.host_w.router_gate)
    v = float_to_bf16_bits(0.05f * static_cast<float>(rng.unit()));
  c.host_w.router_bias.resize(E);
  for (auto& v : c.host_w.router_bias)
    v = 0.25f * static_cast<float>(rng.unit());

  // (E+1)*3 matrices: gate/up [I,H] + down [H,I] per expert, then shared.
  for (int m = 0; m < (E + 1) * 3; ++m) {
    const bool down = m % 3 == 2;
    const int64_t rows = down ? H : I, cols = down ? I : H;
    std::vector<uint8_t> payload(static_cast<size_t>(rows) * cols);
    fill_payload(rng, payload);
    const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
    std::vector<float> scales(sr * sc);
    fill_scales(rng, scales);
    c.host_w.payloads.insert(c.host_w.payloads.end(), payload.begin(),
                             payload.end());
    c.host_w.scales.insert(c.host_w.scales.end(), scales.begin(),
                           scales.end());
  }
  return c;
}

void run_small_case(SmallCase& c, const char* label) {
  std::vector<uint16_t> oracle;
  dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);

  GlmMoeLayer layer(c.dev_w, c.cfg, c.tokens);
  layer.enqueue(c.d_hidden, c.d_out, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  std::vector<uint16_t> got(static_cast<size_t>(c.tokens) * c.cfg.hidden);
  std::memcpy(got.data(), c.d_out, got.size() * 2);

  long hard = 0, soft = 0;
  double max_ulps = 0, sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const int u = bf16_ulps(got[i], oracle[i]);
    max_ulps = std::max(max_ulps, static_cast<double>(u));
    const double d = bf16_bits_to_float(got[i]) - bf16_bits_to_float(oracle[i]);
    sum_d2 += d * d;
    sum_o2 += std::pow(bf16_bits_to_float(oracle[i]), 2);
    if (u > 4) ++soft;
    if (u > 12) ++hard;
  }
  const double l2 = sum_o2 > 0 ? std::sqrt(sum_d2 / sum_o2) : 0;
  require(hard == 0, "expert path: hard ulp violations");
  require(static_cast<double>(soft) / got.size() < 0.02,
          "expert path: soft ulp budget");
  require(l2 < 4e-3, "expert path: l2 budget");

  // Determinism: a second enqueue must be bitwise identical.
  std::vector<uint16_t> again(got.size(), 0x7F7F);
  DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, got.size() * 2));
  layer.enqueue(c.d_hidden, c.d_out, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> second(got.size());
  std::memcpy(second.data(), c.d_out, second.size() * 2);
  require(std::memcmp(got.data(), second.data(), got.size() * 2) == 0,
          "expert path: second run bitwise identical");

  std::printf("[ OK ] %s: max %g ulps, %ld/%zu over soft, l2=%.2g\n", label,
              max_ulps, soft, got.size(), l2);
}

}  // namespace

DGPP_TEST(moe_router_matches_oracle_real_geometry) {
  GlmMoeConfig cfg;  // E=288, H=4096, K=8, scale 2.5, norm on, limit 10
  for (int tokens : {1, 3, 257}) check_router(cfg, tokens, 0xA11CE + tokens);
  // norm_topk_prob = false path (weights un-normalized, just scaled).
  GlmMoeConfig no_norm = cfg;
  no_norm.norm_topk_prob = false;
  check_router(no_norm, 17, 0xB0B);
}

DGPP_TEST(moe_router_ties_break_to_lower_expert_id) {
  // Zero hidden and zero gate => all logits 0, all scores 0.5; equal biases
  // => all biased equal: selection must be experts 0..7 with equal weights
  // (0.5 / 4.0 * 2.5 each).
  GlmMoeConfig cfg;
  const int tokens = 4, E = cfg.n_experts, K = cfg.top_k;
  std::vector<uint16_t> hidden(static_cast<size_t>(tokens) * cfg.hidden, 0);
  std::vector<uint16_t> gate(static_cast<size_t>(E) * cfg.hidden, 0);
  std::vector<float> bias(E, 0.f);

  uint16_t* d_h = nullptr;
  uint16_t* d_g = nullptr;
  float* d_b = nullptr;
  int32_t* d_ids = nullptr;
  float* d_w = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_h, hidden.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_g, gate.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_b, bias.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, static_cast<size_t>(tokens) * K * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, static_cast<size_t>(tokens) * K * 4));
  std::memcpy(d_h, hidden.data(), hidden.size() * 2);
  std::memcpy(d_g, gate.data(), gate.size() * 2);
  std::memcpy(d_b, bias.data(), bias.size() * 4);
  dgpp::launch_moe_router(d_h, d_g, d_b, d_ids, d_w, cfg, tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  const float want_w = 0.5f / (8 * 0.5f) * 2.5f;
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      require(d_ids[static_cast<size_t>(t) * K + i] == i,
              "tie selection must be experts 0..7 in ascending order");
      require(std::abs(d_w[static_cast<size_t>(t) * K + i] - want_w) < 1e-6,
              "tie weights are uniform");
    }
  DGPP_CUDA_OK(cudaFree(d_h));
  DGPP_CUDA_OK(cudaFree(d_g));
  DGPP_CUDA_OK(cudaFree(d_b));
  DGPP_CUDA_OK(cudaFree(d_ids));
  DGPP_CUDA_OK(cudaFree(d_w));
  std::printf("[ OK ] router ties: experts 0..7, uniform weights\n");
}

DGPP_TEST(moe_swiglu_clamp_edge_semantics) {
  // gate clamps ONLY the max; up clamps both; exactly-at-limit values pass.
  const float limit = 10.0f;
  const std::pair<float, float> cases[] = {
      {12.0f, 3.0f},    // gate above limit
      {-50.0f, 1.0f},   // gate below: NO clamp (sigmoid saturates anyway)
      {2.0f, 15.0f},    // up above limit
      {2.0f, -15.0f},   // up below limit
      {10.0f, 10.0f},   // exactly at limit
      {0.0f, 0.0f},     // zeros
      {-3.0f, -0.5f},   // ordinary negatives
  };
  const size_t n = std::size(cases);
  std::vector<uint16_t> g(n), u(n);
  for (size_t i = 0; i < n; ++i) {
    g[i] = float_to_bf16_bits(cases[i].first);
    u[i] = float_to_bf16_bits(cases[i].second);
  }
  uint16_t* d_g = nullptr;
  uint16_t* d_u = nullptr;
  uint16_t* d_o = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_g, n * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_u, n * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_o, n * 2));
  std::memcpy(d_g, g.data(), n * 2);
  std::memcpy(d_u, u.data(), n * 2);
  dgpp::launch_moe_swiglu_clamp(d_g, d_u, d_o, n, limit, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (size_t i = 0; i < n; ++i) {
    const double gc = std::min(
        static_cast<double>(bf16_bits_to_float(g[i])),
        static_cast<double>(limit));
    const double uc = std::min(
        std::max(static_cast<double>(bf16_bits_to_float(u[i])),
                 -static_cast<double>(limit)),
        static_cast<double>(limit));
    const double sig = 1.0 / (1.0 + std::exp(-gc));
    const uint16_t t1 =
        float_to_bf16_bits(static_cast<float>(gc * sig));
    const uint16_t want = float_to_bf16_bits(
        bf16_bits_to_float(t1) * static_cast<float>(uc));
    require(bf16_ulps(d_o[i], want) <= 1, "swiglu edge value");
  }
  DGPP_CUDA_OK(cudaFree(d_g));
  DGPP_CUDA_OK(cudaFree(d_u));
  DGPP_CUDA_OK(cudaFree(d_o));
  std::printf("[ OK ] swiglu clamps: asymmetric limits, boundary values\n");
}

DGPP_TEST(moe_expert_path_matches_oracle_small_geometry) {
  for (auto [E, H, I, K, M] :
       std::vector<std::tuple<int, int, int, int, int>>{
           {8, 512, 256, 2, 1}, {8, 512, 256, 2, 17}, {16, 1024, 512, 4, 5}}) {
    SmallCase c = make_small_case(E, H, I, K, M, 0xC0FFEE + E + M);
    c.alloc();
    run_small_case(c, ("expert path E=" + std::to_string(E) + " M=" +
                       std::to_string(M))
                          .c_str());
    c.free_all();
  }
}

DGPP_TEST(moe_accumulation_order_is_ascending_expert) {
  // Bias increasing with expert id: biased scores ascending in e, so the
  // top-3 SELECTION order is {7,6,5} but the kernel must emit and
  // accumulate {5,6,7}. Three chained bf16 adds are order-sensitive
  // (((a+b)+c) != ((a+c)+b) in general), so a wrong order exceeds the ulp
  // budget against the oracle (which accumulates ascending).
  SmallCase c = make_small_case(8, 512, 256, 3, 1, 0xD1CE);
  for (int e = 0; e < 8; ++e)
    c.host_w.router_bias[e] = 0.1f * static_cast<float>(e);
  c.alloc();
  std::memcpy(c.d_bias, c.host_w.router_bias.data(), 8 * 4);

  GlmMoeLayer layer(c.dev_w, c.cfg, 1);
  layer.enqueue(c.d_hidden, c.d_out, 1, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto& ids = layer.last_ids();
  require(ids.size() == 3, "three ids");
  require(ids[0] == 5 && ids[1] == 6 && ids[2] == 7,
          "ids ascending {5,6,7} (selection order was {7,6,5})");

  std::vector<uint16_t> oracle;
  dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, 1, oracle);
  std::vector<uint16_t> got(c.cfg.hidden);
  std::memcpy(got.data(), c.d_out, got.size() * 2);
  long hard = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (bf16_ulps(got[i], oracle[i]) > 12) ++hard;
  require(hard == 0, "ascending accumulation matches oracle");
  std::printf("[ OK ] accumulation order: ids {5,6,7} ascending, output "
              "matches oracle\n");
  c.free_all();
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
