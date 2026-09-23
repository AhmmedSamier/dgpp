// The MiMo-V2.6-Flash world-1 forward (docs/mimo_v26_flash_plan.md G3, 2026-09-22).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/mimo_fixture.hpp)
//   --smoke DIR                  MimoModel over the fixture: finite outputs, bitwise repeat
//   --checkpoint-dir DIR
//   --dump-file FILE             MimoModel against tools/mimo_reference_dump.py's numpy
//                                double reference: every layer's residual, the final
//                                read, the per-token top-k logits, the routing decisions,
//                                the draft block's rows
//
// The ctest chain: mimo_forward_fixture -> mimo_forward_smoke, and
// mimo_forward_fixture -> mimo_forward_generate (python, end to end) ->
// mimo_forward_test_e2e (--relaxed, writes the engine's states) ->
// mimo_forward_generate_teacher (python, --teacher: every reference layer
// fed the engine's own input) -> mimo_forward_test (strict).
//
// Budgets: the GLM-4.7 gate's (16 / 128 bf16 ulps soft / hard with a 2 %
// RMS cancellation floor, l2 1 %, 0.5 % hard elements; top-1 exact outside
// certified near ties; routing flips under 5 % of the slots) on the
// teacher-forced dump, where each layer's error is one layer's worth of
// the fp32 order against the reference's doubles. The fp8 and MXFP4
// dequants are exact on both sides (bf16(e4m3 x scale) is one rounding
// the reference repeats bit for bit; e2m1 x 2^k is exact), so nothing is
// widened there. The end-to-end run is the relaxed yardstick (l2 3 %, 2 %
// hard): on the random fixture a routing near tie inside the accumulated
// noise flips for some prompts (the DeepSeek ladder's lesson), so its
// rows from the first flipped (layer, token) on are reported, not judged.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/mimo/config.hpp"
#include "models/mimo/forward.hpp"
#include "mimo_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::MimoModel;
using dgpp::MimoTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<int64_t> smoke_tokens(const MimoTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

// ---- the reference dump ------------------------------------------------------

struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0, moe_layers = 0;
  int64_t token_count = 0;

  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPMIMO", 8) == 0, "dump: bad magic");
    uint32_t version = 0, header_len = 0;
    std::memcpy(&version, d.bytes.data() + 8, 4);
    std::memcpy(&header_len, d.bytes.data() + 12, 4);
    require(version == 1, "dump: unsupported version");
    require(16 + static_cast<size_t>(header_len) <= d.bytes.size(), "dump: header exceeds file");
    const std::string_view header(reinterpret_cast<const char*>(d.bytes.data() + 16), header_len);
    dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(header);
    const dgpp::minijson::Value& root = parsed.root;
    const dgpp::minijson::Value& cfg = root.at("config");
    d.hidden = static_cast<int>(cfg.at("hidden").as_int());
    d.vocab = static_cast<int>(cfg.at("vocab").as_int());
    d.num_layers = static_cast<int>(cfg.at("num_layers").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.moe_layers = static_cast<int>(cfg.at("moe_layers").as_int());
    d.token_count = cfg.at("tokens").as_int();
    const size_t payload_base = 16 + header_len;
    for (const auto& m : root.at("tensors").members()) {
      Tensor t;
      t.dtype = std::string(m.value.at("dtype").as_string());
      for (const auto& dim : m.value.at("shape").items()) t.shape.push_back(dim.as_int());
      const uint64_t offset = m.value.at("offset").as_int();
      t.nbytes = static_cast<size_t>(m.value.at("nbytes").as_int());
      require(payload_base + offset + t.nbytes <= d.bytes.size(), "dump: tensor exceeds payload");
      t.data = d.bytes.data() + payload_base + offset;
      d.tensors[m.key] = t;
    }
    return d;
  }
  const Tensor& tensor(const std::string& name) const {
    auto it = tensors.find(name);
    require(it != tensors.end(), "dump: missing tensor " + name);
    return it->second;
  }
};

struct Stats {
  double l2 = 0, max_ulps = 0;
  long soft = 0, hard = 0, total = 0;
};

// bf16 ulps with a cancellation floor at 2 % of the reference's RMS, the
// soft/hard counts and the relative l2.
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
  const MimoTextConfig cfg = MimoTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 72;  // past the fixture's 32-token window
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  MimoModel model(cfg, dir, T, 256);
  const MimoModel::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite residual at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: h rms %.4g max %.4g\n", l, rms, mx);
  }
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const MimoModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = MimoModel::topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers, top-1 of the last row %d (%.4g); deterministic\n", T,
              cfg.num_hidden_layers, top.back()[0].first, top.back()[0].second);
  std::printf("[ OK ] mimo_forward_smoke\n");
  return 0;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path, bool relaxed) {
  const Dump dump = Dump::load(dump_path);
  const MimoTextConfig cfg = MimoTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab &&
              cfg.num_hidden_layers == dump.num_layers && cfg.num_moe_layers() == dump.moe_layers,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);

  MimoModel model(cfg, dir, T, std::max(256, T));
  const MimoModel::Outputs out = model.forward(tokens, true);
  const MimoModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");
  if (const char* states_path = std::getenv("DGPP_MIMO_DUMP_ENGINE_STATES")) {
    // The engine's layer residuals for an offline diagnosis of a budget
    // miss (tools/mimo_reference_dump.py's variants): int32 L, T, H then
    // bf16 rows, the final hidden after them.
    std::ofstream f(states_path, std::ios::binary);
    const int32_t hdr[3] = {cfg.num_hidden_layers, T, cfg.hidden_size};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : out.layer_states)
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()),
            static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
  }

  const int H = cfg.hidden_size, W = H;
  bool ok = true;
  // The first routing flip (relaxed mode): the flipped token's row diverges
  // at its layer's output, every row from it on at the layers after (the
  // attention carries it); the rows before stay under the strict budgets.
  int flip_layer = -1, flip_token = -1;
  if (relaxed && dump.tensors.count("route_ids")) {
    const Dump::Tensor& rt = dump.tensor("route_ids");
    const int K = static_cast<int>(rt.shape[2]);
    const int32_t* ref = reinterpret_cast<const int32_t*>(rt.data);
    int ordinal = 0;
    for (int l = 0; l < cfg.num_hidden_layers && flip_layer < 0; ++l) {
      if (!cfg.is_moe_layer(l)) continue;
      for (int t = 0; t < T && flip_layer < 0; ++t)
        for (int i = 0; i < K; ++i)
          if (out.route_ids[static_cast<size_t>(ordinal)][static_cast<size_t>(t) * K + i] !=
              ref[(static_cast<size_t>(ordinal) * T + t) * K + i]) {
            flip_layer = l;
            flip_token = t;
            break;
          }
      ++ordinal;
    }
    if (flip_layer >= 0)
      std::printf("[ .. ] relaxed: the first routing flip is at layer %d token %d — rows from it on are reported, not judged\n",
                  flip_layer, flip_token);
  }
  // The rows of layer l under judgement: [0, judged_rows(l)).
  auto judged_rows = [&](int l) {
    // The flip's own row at its layer, it and every row after it later.
    return flip_layer < 0 || l < flip_layer ? T : flip_token;
  };
  const double l2_budget = relaxed ? 0.03 : 0.01, hard_budget = relaxed ? 0.02 : 0.005;
  // ---- per-layer residual states ---------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers &&
              ls.shape[1] == T && ls.shape[2] == W,
          "dump: layer_states shape");
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * W;
    const int J = judged_rows(l);
    const Stats s = compare_bf16(out.layer_states[static_cast<size_t>(l)].data(), want,
                                 static_cast<size_t>(J) * W, 16, 128);
    std::printf("[ .. ] layer %d h: l2 %.3g max %g ulps, soft %ld hard %ld of %ld (%d of %d rows judged)\n", l, s.l2,
                s.max_ulps, s.soft, s.hard, s.total, J, T);
    // The worst rows (a routing flip shows as one row's divergence that
    // spreads to the rows after it through the attention).
    std::vector<std::pair<double, int>> rows;
    for (int t = 0; t < T; ++t) {
      const Stats r = compare_bf16(out.layer_states[static_cast<size_t>(l)].data() + static_cast<size_t>(t) * W,
                                   want + static_cast<size_t>(t) * W, static_cast<size_t>(W), 16, 128);
      rows.emplace_back(r.l2, t);
    }
    std::sort(rows.rbegin(), rows.rend());
    std::printf("[ .. ]   worst rows:");
    for (int i = 0; i < 4 && i < T; ++i) std::printf(" t%d %.3g", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first);
    std::printf("\n");
    if (s.hard > 0 && s.hard <= 8) {
      // Few hard elements: show them (cancellation in a residual sum, or a bug).
      const uint16_t* got = out.layer_states[static_cast<size_t>(l)].data();
      double rms = 0;
      for (size_t i = 0; i < static_cast<size_t>(T) * W; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
      rms = std::sqrt(rms / (static_cast<double>(T) * W));
      for (size_t i = 0, shown = 0; i < static_cast<size_t>(T) * W && shown < 8; ++i) {
        const float g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
        if (bf16_ulps(got[i], want[i]) > 128 && std::fabs(g - w) > 0.02 * rms) {
          std::printf("[ .. ]   hard element t%zu h%zu: got %g want %g\n", i / W, i % W, g, w);
          ++shown;
        }
      }
    }
    if (s.total > 0 && (s.l2 > l2_budget || static_cast<double>(s.hard) / s.total > hard_budget)) ok = false;
  }
  // ---- the final read ---------------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H,
          "dump: final_hidden shape");
  const int JF = judged_rows(cfg.num_hidden_layers);
  const Stats fs_ = compare_bf16(out.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(fh.data),
                                 static_cast<size_t>(JF) * H, 16, 128);
  std::printf("[ .. ] final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld (%d of %d rows judged)\n", fs_.l2,
              fs_.max_ulps, fs_.soft, fs_.hard, fs_.total, JF, T);
  if (fs_.total > 0 && (fs_.l2 > l2_budget || static_cast<double>(fs_.hard) / fs_.total > hard_budget)) ok = false;
  // ---- logits: top-1 (near ties certified), top-k overlap --------------------
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  require(tid.dtype == "I32" && tlog.dtype == "F32" && tid.shape[0] == T && tid.shape[1] == k, "dump: topk shape");
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = MimoModel::topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0;
  double max_rel = 0;
  for (int t = 0; t < JF; ++t) {
    const int32_t r1 = ref_ids[static_cast<size_t>(t) * k];
    const float v1 = ref_vals[static_cast<size_t>(t) * k], v2 = ref_vals[static_cast<size_t>(t) * k + 1];
    const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != r1) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    // The reference's own logit for our top-1 must be within 2 % of its top-1.
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const float* p = std::find(ref_ids + static_cast<size_t>(t) * k, ref_ids + static_cast<size_t>(t + 1) * k, id) ==
                               ref_ids + static_cast<size_t>(t + 1) * k
                           ? nullptr
                           : ref_vals + static_cast<size_t>(t) * k +
                                 (std::find(ref_ids + static_cast<size_t>(t) * k, ref_ids + static_cast<size_t>(t + 1) * k, id) -
                                  (ref_ids + static_cast<size_t>(t) * k));
      if (!p) { ++set_miss; continue; }
      max_rel = std::max(max_rel, std::fabs(*p - got[static_cast<size_t>(t)][i].second) / (std::fabs(*p) + 1e-30));
    }
  }
  std::printf("[ .. ] logits: top-1 hard mismatches %d, near-tie %d of %d judged rows; top-%d set misses %d of %d; "
              "matched logits within %.3g relative\n",
              top1_hard, top1_soft, JF, k, set_miss, JF * k, max_rel);
  if (top1_hard != 0 || set_miss > JF * k / 4) ok = false;
  // ---- the draft block: its rows over the prompt ---------------------------------
  if (dump.tensors.count("mtp_topk_ids") && cfg.mtp_layer() >= 0) {
    MimoModel mtp(cfg, dir, T, std::max(256, T), dgpp::MimoResidency::Streaming, nullptr, 0, 1, 1, /*mtp=*/true);
    const MimoModel::Outputs d = mtp.mtp_forward(tokens);
    const int R = T - 1;
    const int JR = std::min(R, JF);  // the draft row q reads h_q: judged as the rows before the flip
    const Dump::Tensor& mh = dump.tensor("mtp_final_hidden");
    require(mh.dtype == "BF16" && mh.shape.size() == 2 && mh.shape[0] == R && mh.shape[1] == H, "dump: mtp_final_hidden shape");
    const Stats ms = compare_bf16(d.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(mh.data),
                                  static_cast<size_t>(JR) * H, 16, 128);
    std::printf("[ .. ] draft final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld (%d of %d rows judged)\n", ms.l2,
                ms.max_ulps, ms.soft, ms.hard, ms.total, JR, R);
    if (ms.total > 0 && (ms.l2 > l2_budget || static_cast<double>(ms.hard) / ms.total > hard_budget)) ok = false;
    const Dump::Tensor& mid = dump.tensor("mtp_topk_ids");
    const Dump::Tensor& mlog = dump.tensor("mtp_topk_logits");
    require(mid.dtype == "I32" && mlog.dtype == "F32" && mid.shape[0] == R && mid.shape[1] == k, "dump: mtp topk shape");
    const int32_t* mref_ids = reinterpret_cast<const int32_t*>(mid.data);
    const float* mref_vals = reinterpret_cast<const float*>(mlog.data);
    const auto mgot = MimoModel::topk(d.logits, R, cfg.vocab_size, k);
    int mhard = 0, msoft = 0;
    for (int t = 0; t < JR; ++t) {
      const int32_t r1 = mref_ids[static_cast<size_t>(t) * k];
      const float v1 = mref_vals[static_cast<size_t>(t) * k], v2 = mref_vals[static_cast<size_t>(t) * k + 1];
      const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
      if (mgot[static_cast<size_t>(t)][0].first != r1) {
        if (margin < 0.02) ++msoft; else ++mhard;
      }
    }
    std::printf("[ .. ] draft logits: top-1 hard mismatches %d, near-tie %d of %d judged rows\n", mhard, msoft, JR);
    if (mhard != 0) ok = false;
  }
  // ---- routing ----------------------------------------------------------------
  if (dump.tensors.count("route_ids")) {
    const Dump::Tensor& rt = dump.tensor("route_ids");
    const int K = static_cast<int>(rt.shape[2]);
    require(rt.shape[0] == dump.moe_layers && out.route_ids.size() == static_cast<size_t>(dump.moe_layers),
            "dump: route_ids layers");
    const int32_t* ref = reinterpret_cast<const int32_t*>(rt.data);
    long diff = 0, total = 0;
    for (int l = 0; l < dump.moe_layers; ++l)
      for (int t = 0; t < T; ++t) {
        bool row_diff = false;
        for (int i = 0; i < K; ++i, ++total)
          if (out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i] !=
              ref[(static_cast<size_t>(l) * T + t) * K + i]) {
            ++diff;
            row_diff = true;
          }
        if (row_diff) {
          std::printf("[ .. ]   route flip: moe layer %d token %d got", l, t);
          for (int i = 0; i < K; ++i) std::printf(" %d", out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i]);
          std::printf(" ref");
          for (int i = 0; i < K; ++i) std::printf(" %d", ref[(static_cast<size_t>(l) * T + t) * K + i]);
          std::printf("\n");
        }
      }
    std::printf("[ .. ] routing: %ld of %ld slots differ from the reference\n", diff, total);
    if (diff > total / 20) ok = false;
  }
  std::printf("[ %s ] mimo_forward_dump_parity (%s)\n", ok ? "OK" : "FAIL", relaxed ? "end-to-end, relaxed" : "teacher-forced, strict");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, smoke, checkpoint, dump;
  bool relaxed = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else if (a == "--relaxed") relaxed = true;
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!fixture.empty()) {
      mimofx::write_fixture(mimofx::tiny_config(), fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump, relaxed);
    std::fprintf(stderr, "usage: --write-fixture DIR | --smoke DIR | --checkpoint-dir DIR --dump-file FILE [--relaxed]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
