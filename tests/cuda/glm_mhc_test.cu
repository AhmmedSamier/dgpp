// Parity tests for the mHC module kernels (M4): compute (norm + 24-logit
// projection + pre/post/comb + Sinkhorn + collapse), the stream update, and
// the final mean — each against the double-precision oracle
// (glm_mhc_reference), which mirrors the transformers reference's dtype
// choreography (bf16 rounding points) while accumulating in double.
//
// Budgets reflect the two error sources: the kernel's fp32 sigmoid/softmax/
// Sinkhorn pipeline vs the oracle's double (relative ~1e-6, so bf16 rounding
// may flip by one ulp near boundaries), and accumulated bf16 rounding in the
// update chain. See DESIGN §7.3 for the semantics.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "models/glm_mhc.hpp"
#include "models/glm_mhc_reference.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::GlmMhcConfig;
using dgpp::GlmMhcRefResult;
using dgpp::GlmMhcWeights;
using dgpp::GlmMhcWeightsHost;

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() {  // [-1, 1)
    return static_cast<double>(static_cast<int64_t>(next() >> 11)) /
           static_cast<double>(1ull << 52);
  }
};

GlmMhcConfig real_config() {
  GlmMhcConfig c;  // hc_mult 4, hidden 4096, iters 20, eps 1e-6/1e-5
  return c;
}

struct Case {
  GlmMhcConfig cfg;
  int tokens = 0;
  std::vector<uint16_t> streams;
  GlmMhcWeightsHost host_w;
  GlmMhcWeights dev_w;
  std::vector<uint16_t> sublayer_out;

  // Device buffers.
  uint16_t* d_streams = nullptr;
  uint16_t* d_fn = nullptr;
  float* d_base = nullptr;
  float* d_scale = nullptr;
  uint16_t* d_collapsed = nullptr;
  uint16_t* d_post = nullptr;
  uint16_t* d_comb = nullptr;
  uint16_t* d_sub = nullptr;
  uint16_t* d_streams_out = nullptr;
  uint16_t* d_mean = nullptr;

  void alloc() {
    const int n = cfg.hc_mult, D = cfg.hidden;
    DGPP_CUDA_OK(cudaMallocManaged(&d_streams, streams.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_fn, host_w.fn.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_base, host_w.base.size() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_scale, 3 * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_collapsed, static_cast<size_t>(tokens) * D * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_post, static_cast<size_t>(tokens) * n * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_comb, static_cast<size_t>(tokens) * n * n * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_sub, sublayer_out.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_streams_out, streams.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_mean, static_cast<size_t>(tokens) * D * 2));
    std::memcpy(d_streams, streams.data(), streams.size() * 2);
    std::memcpy(d_fn, host_w.fn.data(), host_w.fn.size() * 2);
    std::memcpy(d_base, host_w.base.data(), host_w.base.size() * 4);
    std::memcpy(d_scale, host_w.scale.data(), 3 * 4);
    std::memcpy(d_sub, sublayer_out.data(), sublayer_out.size() * 2);
    dev_w = GlmMhcWeights{d_fn, d_base, d_scale};
  }
  void free_all() {
    cudaFree(d_streams); cudaFree(d_fn); cudaFree(d_base); cudaFree(d_scale);
    cudaFree(d_collapsed); cudaFree(d_post); cudaFree(d_comb); cudaFree(d_sub);
    cudaFree(d_streams_out); cudaFree(d_mean);
  }
};

Case make_case(const GlmMhcConfig& cfg, int tokens, uint64_t seed,
               bool extreme_logits = false) {
  Case c;
  c.cfg = cfg;
  c.tokens = tokens;
  const int n = cfg.hc_mult, D = cfg.hidden;
  const int coeffs = cfg.coeff_rows();
  Rng rng(seed);

  c.streams.resize(static_cast<size_t>(tokens) * n * D);
  for (auto& v : c.streams)
    v = float_to_bf16_bits(0.25f * static_cast<float>(rng.unit()));
  c.sublayer_out.resize(static_cast<size_t>(tokens) * D);
  for (auto& v : c.sublayer_out)
    v = float_to_bf16_bits(0.5f * static_cast<float>(rng.unit()));

  c.host_w.fn.resize(static_cast<size_t>(coeffs) * n * D);
  for (auto& v : c.host_w.fn)
    v = float_to_bf16_bits(0.02f * static_cast<float>(rng.unit()));
  c.host_w.base.resize(coeffs);
  for (auto& v : c.host_w.base)
    v = extreme_logits ? (rng.unit() > 0 ? 40.f : -40.f)
                       : 0.1f * static_cast<float>(rng.unit());
  c.host_w.scale.resize(3);
  for (auto& v : c.host_w.scale)
    v = static_cast<float>(std::exp(rng.unit()));
  return c;
}

// bf16 ulp distance via biased sign-magnitude key.
int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu)
                         : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

struct UlpReport {
  long over = 0;       // beyond soft budget
  long hard = 0;       // beyond hard budget
  size_t total = 0;
  double max_ulps = 0;
};

UlpReport compare(const std::vector<uint16_t>& got,
                  const std::vector<uint16_t>& want, int soft, int hard) {
  UlpReport r;
  r.total = want.size();
  for (size_t i = 0; i < want.size(); ++i) {
    const int u = bf16_ulps(got[i], want[i]);
    r.max_ulps = std::max(r.max_ulps, static_cast<double>(u));
    if (u > soft) ++r.over;
    if (u > hard) ++r.hard;
  }
  return r;
}

void require_ulp(const UlpReport& r, double over_frac, const std::string& what) {
  if (r.hard != 0 ||
      static_cast<double>(r.over) / static_cast<double>(r.total) > over_frac)
    throw std::runtime_error(what + ": over=" +
                             std::to_string(r.over) + "/" +
                             std::to_string(r.total) +
                             " hard=" + std::to_string(r.hard) +
                             " max=" + std::to_string(r.max_ulps) + " ulps");
}

std::vector<uint16_t> read_back(const uint16_t* dev, size_t n) {
  std::vector<uint16_t> v(n);
  std::memcpy(v.data(), dev, n * 2);
  return v;
}

// Full pipeline on one case: compute + (isolated) update + mean.
void run_case(const Case& c, const char* label) {
  GlmMhcRefResult ref;
  dgpp::glm_mhc_ref_compute(c.d_streams, c.host_w, c.cfg, c.tokens, ref);

  dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed,
                           c.d_post, c.d_comb, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  // post/comb: oracle rounded to bf16 (the choreography point).
  std::vector<uint16_t> post_ref(c.tokens * c.cfg.hc_mult);
  std::vector<uint16_t> comb_ref(c.tokens * c.cfg.hc_mult *
                                 c.cfg.hc_mult);
  for (size_t i = 0; i < post_ref.size(); ++i)
    post_ref[i] = float_to_bf16_bits(static_cast<float>(ref.post[i]));
  for (size_t i = 0; i < comb_ref.size(); ++i)
    comb_ref[i] = float_to_bf16_bits(static_cast<float>(ref.comb[i]));
  require_ulp(compare(read_back(c.d_post, post_ref.size()), post_ref, 2, 4),
              0.005, std::string(label) + " post");
  require_ulp(compare(read_back(c.d_comb, comb_ref.size()), comb_ref, 2, 4),
              0.005, std::string(label) + " comb");
  require_ulp(compare(read_back(c.d_collapsed, ref.collapsed.size()),
                      ref.collapsed, 2, 4),
              0.005, std::string(label) + " collapsed");

  // Isolated update: both sides consume the SAME (oracle) bf16 post/comb.
  std::vector<uint16_t> streams_ref(c.streams.size());
  dgpp::glm_mhc_ref_stream_update(post_ref.data(), comb_ref.data(),
                                  c.sublayer_out.data(), c.streams.data(),
                                  c.cfg, c.tokens, streams_ref.data());
  std::memcpy(c.d_post, post_ref.data(), post_ref.size() * 2);
  std::memcpy(c.d_comb, comb_ref.data(), comb_ref.size() * 2);
  dgpp::launch_mhc_stream_update(c.d_post, c.d_comb, c.d_sub, c.d_streams,
                                 c.d_streams_out, c.cfg, c.tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require_ulp(compare(read_back(c.d_streams_out, streams_ref.size()),
                      streams_ref, 4, 8),
              0.01, std::string(label) + " stream update");

  // Final mean.
  std::vector<uint16_t> mean_ref(c.tokens * c.cfg.hidden);
  dgpp::glm_mhc_ref_final_mean(c.streams.data(), c.cfg, c.tokens,
                               mean_ref.data());
  dgpp::launch_mhc_final_mean(c.d_streams, c.d_mean, c.cfg, c.tokens,
                              nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require_ulp(compare(read_back(c.d_mean, mean_ref.size()), mean_ref, 2, 4),
              0.005, std::string(label) + " final mean");

  std::printf("[ OK ] %s: post/comb/collapsed/update/mean within budgets\n",
              label);
}

}  // namespace

DGPP_TEST(mhc_compute_update_mean_match_oracle_real_geometry) {
  for (int tokens : {1, 3, 17, 257, 2052}) {
    Case c = make_case(real_config(), tokens, 0xABCDEF + tokens);
    c.alloc();
    run_case(c, ("real-geometry tokens=" + std::to_string(tokens)).c_str());
    c.free_all();
  }
}

DGPP_TEST(mhc_small_hidden_and_extreme_logits) {
  {
    GlmMhcConfig cfg;
    cfg.hidden = 512;
    Case c = make_case(cfg, 17, 0x1234);
    c.alloc();
    run_case(c, "hidden=512 tokens=17");
    c.free_all();
  }
  {
    // Saturated sigmoids (|base| = 40): pre -> {eps, 1+eps}, post -> {0, 2},
    // softmax rows peaked — the Sinkhorn loop must stay finite and the
    // budgets must still hold.
    Case c = make_case(real_config(), 33, 0x5678, /*extreme_logits=*/true);
    c.alloc();
    run_case(c, "saturated logits");
    c.free_all();
  }
}

DGPP_TEST(mhc_zero_streams_survive_norm_of_zero) {
  // All-zero streams: sumsq = 0, inv_rms = rsqrt(eps) = large; flat = 0, so
  // logits = 0 and everything downstream is well-defined. A NaN or inf
  // anywhere fails the oracle comparison.
  Case c = make_case(real_config(), 5, 0x9ABC);
  std::fill(c.streams.begin(), c.streams.end(), 0);
  c.alloc();
  run_case(c, "zero streams");
  c.free_all();
}

DGPP_TEST(mhc_end_to_end_pipeline_is_deterministic) {
  // Real pipeline wiring: kernel's own post/comb feed its own update; two
  // full invocations must be bitwise identical (graph-capture premise).
  Case c = make_case(real_config(), 64, 0xDEF0);
  c.alloc();
  auto pipeline = [&] {
    dgpp::launch_mhc_compute(c.d_streams, c.dev_w, c.cfg, c.d_collapsed,
                             c.d_post, c.d_comb, c.tokens, nullptr);
    dgpp::launch_mhc_stream_update(c.d_post, c.d_comb, c.d_sub, c.d_streams,
                                   c.d_streams_out, c.cfg, c.tokens,
                                   nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    return read_back(c.d_streams_out, c.streams.size());
  };
  const auto a = pipeline();
  const auto b = pipeline();
  if (std::memcmp(a.data(), b.data(), a.size() * 2) != 0)
    throw std::runtime_error("pipeline is not bitwise deterministic");

  // And the end-to-end pipeline stays within budgets of the oracle pipeline
  // (oracle compute -> oracle-rounded post/comb -> oracle update).
  GlmMhcRefResult ref;
  dgpp::glm_mhc_ref_compute(c.d_streams, c.host_w, c.cfg, c.tokens, ref);
  std::vector<uint16_t> post_ref(c.tokens * c.cfg.hc_mult);
  std::vector<uint16_t> comb_ref(c.tokens * c.cfg.hc_mult * c.cfg.hc_mult);
  for (size_t i = 0; i < post_ref.size(); ++i)
    post_ref[i] = float_to_bf16_bits(static_cast<float>(ref.post[i]));
  for (size_t i = 0; i < comb_ref.size(); ++i)
    comb_ref[i] = float_to_bf16_bits(static_cast<float>(ref.comb[i]));
  std::vector<uint16_t> streams_ref(c.streams.size());
  dgpp::glm_mhc_ref_stream_update(post_ref.data(), comb_ref.data(),
                                  c.sublayer_out.data(), c.streams.data(),
                                  c.cfg, c.tokens, streams_ref.data());
  require_ulp(compare(a, streams_ref, 4, 8), 0.01, "end-to-end streams");
  std::printf("[ OK ] end-to-end pipeline deterministic + within budgets\n");
}

DGPP_TEST(mhc_config_validation_pins_supported_geometry) {
  bool threw = false;
  try {
    GlmMhcConfig bad;
    bad.hc_mult = 8;
    GlmMhcConfig::validate_config(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("hc_mult != 4 must be rejected");
  threw = false;
  try {
    GlmMhcConfig bad;
    bad.sinkhorn_iters = 0;
    GlmMhcConfig::validate_config(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("sinkhorn_iters < 1 must be rejected");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
