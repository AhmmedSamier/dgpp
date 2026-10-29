// The Llama world-1 forward (docs/zeta_plan.md, 2026-10-14).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/llama_fixture.hpp)
//   --smoke DIR                  LlamaModel over the fixture: finite outputs, bitwise repeat
//   --checkpoint-dir DIR         LlamaModel against the C++ host reference below:
//                                every layer's residual, the final hidden,
//                                the per-token top-k logits
//
// The ctest chain: llama_forward_fixture -> llama_forward_smoke, and
// llama_forward_fixture -> llama_forward_test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <unordered_map>

#include "common/dtypes.hpp"
#include "kernels/qsa.hpp"
#include "loaders/safetensors.hpp"
#include "models/llama/attn_reference.hpp"
#include "models/llama/config.hpp"
#include "models/llama/forward.hpp"
#include "models/llama/loader.hpp"
#include "llama_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LlamaLayerStream;
using dgpp::LlamaModel;
using dgpp::LlamaResidency;
using dgpp::LlamaTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<int64_t> smoke_tokens(const LlamaTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

// ---- the host reference ------------------------------------------------------
// Pure fp32 chains with the kernels' bf16 rounding points: RMSNorm as
// bf16(x * rstd) then bf16(w * u); GEMMs as sequential fp32 fma chains;
// the attention finish + softmax through llama_ref; SwiGLU with the
// kernel's two roundings. Compared under the ulp/l2 bands below, not
// bitwise (reduction orders differ by construction).

struct HostWeights {
  const LlamaTextConfig* cfg = nullptr;
  std::unordered_map<std::string, std::vector<uint8_t>> bytes;
  static HostWeights load(const LlamaTextConfig& cfg, const std::string& dir);
  const uint16_t* bf16(const std::string& name) const {
    auto it = bytes.find(name);
    if (it == bytes.end()) throw std::runtime_error("host weights: missing " + name);
    return reinterpret_cast<const uint16_t*>(it->second.data());
  }
};

}  // namespace

namespace {

std::unordered_map<std::string, std::vector<uint8_t>> read_shard(const std::string& dir) {
  std::unordered_map<std::string, std::vector<uint8_t>> out;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      std::vector<uint8_t> b(t.nbytes());
      std::memcpy(b.data(), t.data, b.size());
      out[t.name] = std::move(b);
    });
  }
  return out;
}

}  // namespace

HostWeights HostWeights::load(const LlamaTextConfig& cfg, const std::string& dir) {
  HostWeights w;
  w.cfg = &cfg;
  w.bytes = read_shard(dir);
  return w;
}

namespace {

float hdot(const uint16_t* a, const uint16_t* b, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i) acc = std::fma(bf16_bits_to_float(a[i]), bf16_bits_to_float(b[i]), acc);
  return acc;
}

void hrmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int rows, int h, float eps) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int i = 0; i < h; ++i) {
      const float v = bf16_bits_to_float(x[r * h + i]);
      ss += static_cast<double>(v) * v;
    }
    const float rstd = 1.0f / std::sqrt(static_cast<float>(ss / h) + eps);
    for (int i = 0; i < h; ++i) {
      const float u = rb(bf16_bits_to_float(x[r * h + i]) * rstd);
      y[r * h + i] = float_to_bf16_bits(rb(bf16_bits_to_float(w[i]) * u));
    }
  }
}

// y[r, n] = x[r, :] . W[n, :] (W row-major [n, k]).
void hgemm_f32(const uint16_t* x, const uint16_t* w, float* y, int rows, int n, int k) {
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < n; ++j) y[r * n + j] = hdot(x + r * k, w + j * k, k);
}

void hgemm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int rows, int n, int k) {
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < n; ++j) y[r * n + j] = float_to_bf16_bits(hdot(x + r * k, w + j * k, k));
}

struct HostOut {
  std::vector<std::vector<uint16_t>> layer_states;
  std::vector<uint16_t> final_hidden;
  std::vector<float> logits;
};

HostOut host_forward(const LlamaTextConfig& cfg, const HostWeights& w, const std::vector<int64_t>& tokens) {
  const int T = static_cast<int>(tokens.size()), H = cfg.hidden_size;
  const int I = cfg.intermediate_size, V = cfg.vocab_size;
  const int QH = cfg.num_attention_heads, KVH = cfg.num_key_value_heads, D = cfg.head_dim;
  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  std::vector<float> inv(cfg.head_dim / 2);
  dgpp::qsa_rope_inv_freq(cfg.rope_theta, cfg.head_dim, inv.data());

  std::vector<uint16_t> resid(T * H), x(T * H), attn(T * H), ffn(T * H);
  const uint16_t* embed = w.bf16("model.embed_tokens.weight");
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < H; ++i) resid[t * H + i] = embed[tokens[t] * H + i];

  // The K/V history per layer: finished rows [T, KVH * D].
  std::vector<std::vector<uint16_t>> khist(cfg.num_hidden_layers), vhist(cfg.num_hidden_layers);

  HostOut out;
  std::vector<float> qd(QH * D), kd(KVH * D), vd(KVH * D);
  std::vector<uint16_t> q(QH * D), k(KVH * D), v(KVH * D);
  std::vector<uint16_t> gate(T * I), up(T * I), act(T * I);
  std::vector<float> c;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    hrmsnorm(resid.data(), w.bf16(p + "input_layernorm.weight"), x.data(), T, H, cfg.rms_norm_eps);
    khist[l].resize(T * KVH * D);
    vhist[l].resize(T * KVH * D);
    for (int t = 0; t < T; ++t) {
      hgemm_f32(x.data() + t * H, w.bf16(p + "self_attn.q_proj.weight"), qd.data(), 1, QH * D, H);
      hgemm_f32(x.data() + t * H, w.bf16(p + "self_attn.k_proj.weight"), kd.data(), 1, KVH * D, H);
      hgemm_f32(x.data() + t * H, w.bf16(p + "self_attn.v_proj.weight"), vd.data(), 1, KVH * D, H);
      for (int h = 0; h < QH; ++h)
        dgpp::llama_ref::qkv_finish(qd.data() + h * D, cfg.rms_norm_eps, inv.data(), D, t,
                                    q.data() + h * D, D);
      for (int h = 0; h < KVH; ++h) {
        dgpp::llama_ref::qkv_finish(kd.data() + h * D, cfg.rms_norm_eps, inv.data(), D, t,
                                    k.data() + h * D, D);
        dgpp::llama_ref::qkv_finish(vd.data() + h * D, cfg.rms_norm_eps, nullptr, 0, t,
                                    v.data() + h * D, D);
      }
      std::memcpy(khist[l].data() + t * KVH * D, k.data(), KVH * D * 2);
      std::memcpy(vhist[l].data() + t * KVH * D, v.data(), KVH * D * 2);
      dgpp::llama_ref::attention(q.data(), khist[l].data(), vhist[l].data(), t + 1, QH, KVH, D, scale,
                                 c, /*tile=*/32, /*n_split=*/1);
      for (size_t i = 0; i < c.size(); ++i) attn[t * H + i] = float_to_bf16_bits(c[i]);
      // o_proj over the full Q width into the residual row's partial.
      std::vector<float> o(H, 0.f);
      const uint16_t* ow = w.bf16(p + "self_attn.o_proj.weight");
      for (int j = 0; j < H; ++j) o[j] = hdot(attn.data() + t * H, ow + j * (QH * D), QH * D);
      for (int j = 0; j < H; ++j) attn[t * H + j] = float_to_bf16_bits(o[j]);
    }
    for (int i = 0; i < T * H; ++i) resid[i] = float_to_bf16_bits(bf16_bits_to_float(resid[i]) + bf16_bits_to_float(attn[i]));
    hrmsnorm(resid.data(), w.bf16(p + "post_attention_layernorm.weight"), x.data(), T, H, cfg.rms_norm_eps);
    hgemm_bf16(x.data(), w.bf16(p + "mlp.gate_proj.weight"), gate.data(), T, I, H);
    hgemm_bf16(x.data(), w.bf16(p + "mlp.up_proj.weight"), up.data(), T, I, H);
    for (int i = 0; i < T * I; ++i) {
      const float g = bf16_bits_to_float(gate[i]), u = bf16_bits_to_float(up[i]);
      const float t = rb(g * (1.0f / (1.0f + std::exp(-g))));
      act[i] = float_to_bf16_bits(rb(t * u));
    }
    for (int t = 0; t < T; ++t) {
      std::vector<float> o(H, 0.f);
      const uint16_t* dw = w.bf16(p + "mlp.down_proj.weight");
      for (int j = 0; j < H; ++j) o[j] = hdot(act.data() + t * I, dw + j * I, I);
      for (int j = 0; j < H; ++j) ffn[t * H + j] = float_to_bf16_bits(o[j]);
    }
    for (int i = 0; i < T * H; ++i) resid[i] = float_to_bf16_bits(bf16_bits_to_float(resid[i]) + bf16_bits_to_float(ffn[i]));
    out.layer_states.push_back(resid);
  }
  out.final_hidden.resize(T * H);
  hrmsnorm(resid.data(), w.bf16("model.norm.weight"), out.final_hidden.data(), T, H, cfg.rms_norm_eps);
  out.logits.resize(T * V);
  hgemm_f32(out.final_hidden.data(), w.bf16("lm_head.weight"), out.logits.data(), T, V, H);
  return out;
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long soft = 0, hard = 0, total = 0;
};

// bf16 ulps with a cancellation floor at 2 % of the reference's RMS, the
// soft/hard counts and the relative l2 (the GLM-4.7 gate's bands).
Stats compare_bf16(const uint16_t* got, const uint16_t* want, size_t n, int soft, int hard) {
  Stats s;
  s.total = static_cast<long>(n);
  double rms = 0;
  for (size_t i = 0; i < n; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
  rms = std::sqrt(rms / std::max<size_t>(n, 1));
  const double floor_abs = 0.02 * rms;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= floor_abs) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    sum_d2 += (g - w) * (g - w);
    sum_o2 += w * w;
    if (u > soft) ++s.soft;
    if (u > hard) ++s.hard;
  }
  s.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return s;
}

int run_smoke(const std::string& dir) {
  const LlamaTextConfig cfg = LlamaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 48;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  LlamaModel model(cfg, dir, T, 256);
  const LlamaModel::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite state at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: h rms %.4g max %.4g\n", l, rms, mx);
  }
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const LlamaModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = LlamaModel::topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers, top-1 of the last row %d (%.4g); deterministic\n", T,
              cfg.num_hidden_layers, top.back()[0].first, top.back()[0].second);
  std::printf("[ OK ] llama_forward_smoke\n");
  return 0;
}

int run_host_parity(const std::string& dir) {
  const LlamaTextConfig cfg = LlamaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 48;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  LlamaModel model(cfg, dir, T, 256);
  const LlamaModel::Outputs out = model.forward(tokens, true);
  const LlamaModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");
  const HostWeights w = HostWeights::load(cfg, dir);
  const HostOut ref = host_forward(cfg, w, tokens);
  bool ok = true;
  const int H = cfg.hidden_size, V = cfg.vocab_size;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const Stats s = compare_bf16(out.layer_states[l].data(), ref.layer_states[l].data(),
                                 static_cast<size_t>(T) * H, 16, 128);
    std::printf("[ .. ] layer %d h: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", l, s.l2,
                s.max_ulps, s.soft, s.hard, s.total);
    if (s.l2 > 0.01 || static_cast<double>(s.hard) / s.total > 0.005) ok = false;
  }
  const Stats fs_ = compare_bf16(out.final_hidden_bits.data(), ref.final_hidden.data(),
                                 static_cast<size_t>(T) * H, 16, 128);
  std::printf("[ .. ] final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2, fs_.max_ulps,
              fs_.soft, fs_.hard, fs_.total);
  if (fs_.l2 > 0.01 || static_cast<double>(fs_.hard) / fs_.total > 0.005) ok = false;
  const auto got = LlamaModel::topk(out.logits, T, V, 4);
  std::vector<std::pair<int32_t, float>> ref_top;
  int top1_mismatch = 0;
  double max_rel = 0;
  for (int t = 0; t < T; ++t) {
    const float* row = ref.logits.data() + static_cast<size_t>(t) * V;
    int best = 0;
    for (int i = 1; i < V; ++i)
      if (row[i] > row[best]) best = i;
    if (got[t][0].first != best) ++top1_mismatch;
    max_rel = std::max(max_rel, std::fabs(got[t][0].second - row[got[t][0].first]) /
                                    (std::fabs(row[got[t][0].first]) + 1e-30));
  }
  std::printf("[ .. ] logits: top-1 mismatches %d of %d rows; matched logits within %.3g relative\n",
              top1_mismatch, T, max_rel);
  if (top1_mismatch != 0) ok = false;
  std::printf("[ %s ] llama_forward_host_parity\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// Dump logits and, when requested, per-layer residual states for a token
// sequence.  The raw state files are bf16 [tokens, hidden] rows and are a
// deliberately simple interchange format for comparison with a framework
// reference.
// Usage: --logits <checkpoint_dir> <t1,t2,...,tN> [--dump-states DIR]
int run_logits(const std::string& dir, const std::string& toks_str, const std::string& states_dir) {
  const LlamaTextConfig cfg = LlamaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int V = cfg.vocab_size;
  std::vector<int64_t> tokens;
  size_t start = 0;
  while (start < toks_str.size()) {
    size_t end = toks_str.find(',', start);
    if (end == std::string::npos) end = toks_str.size();
    tokens.push_back(std::stoll(toks_str.substr(start, end - start)));
    start = end + 1;
  }
  const int T = static_cast<int>(tokens.size());
  LlamaModel model(cfg, dir, T, 256, LlamaResidency::Streaming, nullptr, 0, 1, 1, false, 0, false);
  const LlamaModel::Outputs out = model.forward(tokens, !states_dir.empty());
  if (!states_dir.empty()) {
    fs::create_directories(states_dir);
    require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers),
            "logits: missing layer captures");
    for (size_t layer = 0; layer < out.layer_states.size(); ++layer) {
      const fs::path path = fs::path(states_dir) / ("layer_" + std::to_string(layer) + ".bf16");
      std::FILE* f = std::fopen(path.c_str(), "wb");
      require(f != nullptr, "logits: cannot open " + path.string());
      const auto& state = out.layer_states[layer];
      const size_t written = std::fwrite(state.data(), sizeof(uint16_t), state.size(), f);
      std::fclose(f);
      require(written == state.size(), "logits: cannot write " + path.string());
    }
    std::printf("wrote %zu layer states to %s\n", out.layer_states.size(), states_dir.c_str());
  }
  std::printf("position logits (top-5 id=value):\n");
  for (int t = 0; t < T; ++t) {
    const float* row = out.logits.data() + static_cast<size_t>(t) * V;
    std::vector<std::pair<float, int>> scores(V);
    for (int i = 0; i < V; ++i) scores[i] = {row[i], i};
    std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::printf("pos %2d: ", t);
    for (int k = 0; k < 5; ++k)
      std::printf("id=%d(%.4f) ", scores[k].second, scores[k].first);
    std::printf("\n");
  }
  return 0;
}

// engine.dense_weights = "fp8" (slice 7): the same forward under the fp8
// stack — deterministic (the scale GEMM's kernels are), finite, and within
// a tolerance of the bf16 run's logits (fp8 is lossy; the host parity's
// bitwise gates do not apply).
int run_fp8(const std::string& dir) {
  const LlamaTextConfig cfg = LlamaTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 48;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  LlamaModel bf16(cfg, dir, T, 256);
  const LlamaModel::Outputs base = bf16.forward(tokens, true);

  struct Fp8On {
    const bool saved;
    Fp8On() : saved(LlamaLayerStream::dense_weights_fp8()) { LlamaLayerStream::set_dense_weights_fp8(true); }
    ~Fp8On() { LlamaLayerStream::set_dense_weights_fp8(saved); }
  } on;
  LlamaModel model(cfg, dir, T, 256);
  const LlamaModel::Outputs out = model.forward(tokens, true);
  const LlamaModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "fp8 forward is not deterministic across calls");
  for (float v : out.logits) require(std::isfinite(v), "fp8: non-finite logit");

  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < out.logits.size(); ++i) {
    const double d = static_cast<double>(out.logits[i]) - static_cast<double>(base.logits[i]);
    sum_d2 += d * d;
    sum_o2 += static_cast<double>(base.logits[i]) * static_cast<double>(base.logits[i]);
  }
  const double l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  const auto got = LlamaModel::topk(out.logits, T, cfg.vocab_size, 4);
  const auto want = LlamaModel::topk(base.logits, T, cfg.vocab_size, 4);
  int top1_mismatch = 0;
  for (int t = 0; t < T; ++t) top1_mismatch += got[t][0].first != want[t][0].first;
  std::printf("[ .. ] fp8 logits: l2 %.3g against bf16, top-1 mismatches %d of %d rows; deterministic\n", l2,
              top1_mismatch, T);
  // fp8 is lossy: near-tie argmax rows may flip (the engine's near-tie
  // rule); the gate is the l2 band and every logit finite.
  require(l2 < 0.05 && top1_mismatch <= T / 8, "fp8 logits diverge from the bf16 forward");
  std::printf("[ OK ] llama_forward_fp8\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, smoke, checkpoint, fp8, logits_dir, logits_toks, dump_states;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--fp8" && i + 1 < argc) fp8 = argv[++i];
    else if (a == "--logits" && i + 2 < argc) { logits_dir = argv[++i]; logits_toks = argv[++i]; }
    else if (a == "--dump-states" && i + 1 < argc) dump_states = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!fixture.empty()) {
      llamafx::write_fixture(llamafx::tiny_config(), fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!checkpoint.empty()) return run_host_parity(checkpoint);
    if (!fp8.empty()) return run_fp8(fp8);
    if (!logits_dir.empty()) return run_logits(logits_dir, logits_toks, dump_states);
    std::fprintf(stderr,
                 "usage: --write-fixture DIR | --smoke DIR | --checkpoint-dir DIR | --fp8 DIR\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
