// The DeepSeek-V4.1 forward against the python reference dump
// (tools/dsv41_reference_dump.py, the reference's wiring with the engine's
// rounding points) on the loader fixture: every layer's residual streams,
// the final read, the top-k logits, the routing decisions and the index
// sources' selections — flips certified as near ties by the reference's
// boundary margins, the flipped rows excluded from the residual budgets
// (a different selection is a legitimately different output).
//
//   dsv41_forward_test --write-fixture DIR | --smoke DIR | --checkpoint-dir DIR --dump-file FILE [--relaxed] [--bounded]
//
// --bounded: the engine's bounded prefill (plan §1.8) against a dump the
// reference generated with --prefill bounded: the layers from the last kv
// source on walk the last `sliding_window` rows only (the segment), so
// their captures, routes, selections and the head's rows are compared
// over the segment (the dump pads the rows before it).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "dsv41_fixture.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv41/model.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::Dsv41Model;
using dgpp::Dsv41TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7FFF) : static_cast<int>(v & 0x7FFF); };
  return std::abs(key(a) - key(b));
}

std::vector<int64_t> smoke_tokens(const Dsv41TextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x5A5A1234ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0, moe_layers = 0, index_layers = 0, max_selected = 0, streams = 0;
  int candidate_blocks = 0;
  int dspark_block = 0;
  int64_t token_count = 0;
  int segment_row0 = 0;  // the bounded reference's first walked decoder row (0: exact)
  std::string prefill = "exact";
  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPGDSD", 8) == 0, "dump: bad magic");
    uint32_t version = 0, header_len = 0;
    std::memcpy(&version, d.bytes.data() + 8, 4);
    std::memcpy(&header_len, d.bytes.data() + 12, 4);
    require(version == 1, "dump: unsupported version");
    const std::string_view header(reinterpret_cast<const char*>(d.bytes.data() + 16), header_len);
    dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(header);
    const dgpp::minijson::Value& root = parsed.root;
    const dgpp::minijson::Value& cfg = root.at("config");
    d.hidden = static_cast<int>(cfg.at("hidden").as_int());
    d.vocab = static_cast<int>(cfg.at("vocab").as_int());
    d.num_layers = static_cast<int>(cfg.at("num_layers").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.moe_layers = static_cast<int>(cfg.at("moe_layers").as_int());
    d.index_layers = static_cast<int>(cfg.at("index_layers").as_int());
    d.max_selected = static_cast<int>(cfg.at("max_selected").as_int());
    d.streams = static_cast<int>(cfg.at("streams").as_int());
    d.candidate_blocks = static_cast<int>(cfg.at("candidate_blocks").as_int());
    d.dspark_block = static_cast<int>(cfg.at("dspark_block").as_int());
    d.token_count = cfg.at("tokens").as_int();
    if (const auto* v = cfg.find("segment_row0")) d.segment_row0 = static_cast<int>(v->as_int());
    if (const auto* v = cfg.find("prefill")) d.prefill = std::string(v->as_string());
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
std::vector<std::vector<std::pair<int32_t, float>>> topk(const std::vector<float>& logits, int T, int V, int k) {
  std::vector<std::vector<std::pair<int32_t, float>>> out(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) {
    std::vector<int32_t> idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) idx[static_cast<size_t>(i)] = i;
    const float* row = logits.data() + static_cast<size_t>(t) * V;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
      return row[a] > row[b] || (row[a] == row[b] && a < b);
    });
    for (int i = 0; i < k; ++i) out[static_cast<size_t>(t)].emplace_back(idx[static_cast<size_t>(i)], row[idx[static_cast<size_t>(i)]]);
  }
  return out;
}

int run_smoke(const std::string& dir) {
  const Dsv41TextConfig cfg = Dsv41TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 72;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  Dsv41Model model(cfg, dir, T, 256);
  const Dsv41Model::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  require(out.dsa_selections.size() == cfg.index_source_layer_ids.size(), "smoke: selection captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite stream at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: streams rms %.4g max %.4g\n", l, rms, mx);
  }
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const Dsv41Model::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers (%zu index sources), top-1 of the last row %d (%.4g); deterministic\n",
              T, cfg.num_hidden_layers, cfg.index_source_layer_ids.size(), top.back()[0].first, top.back()[0].second);
  // The bounded prefill: the decoder over the last window rows; the rows
  // before them read as NaN, the segment's are finite and deterministic.
  model.set_prefill_bounded(true);
  const Dsv41Model::Outputs bounded = model.forward(tokens, true);
  const int seg0 = T - std::min(cfg.sliding_window, T);
  const int dec0 = cfg.decoder_first_layer();
  for (size_t l = 0; l < bounded.layer_states.size(); ++l) {
    const int rows = static_cast<int>(l) >= dec0 ? T - seg0 : T;
    require(bounded.layer_states[l].size() == static_cast<size_t>(rows) * cfg.hc_mult * cfg.hidden_size,
            "smoke: bounded layer " + std::to_string(l) + " rows");
    for (uint16_t v : bounded.layer_states[l]) require(std::isfinite(bf16_bits_to_float(v)), "smoke: bounded non-finite stream");
  }
  for (int t = 0; t < T; ++t)
    for (int v = 0; v < cfg.vocab_size; ++v) {
      const float x = bounded.logits[static_cast<size_t>(t) * cfg.vocab_size + v];
      require(t < seg0 ? std::isnan(x) : std::isfinite(x), "smoke: bounded logits rows");
    }
  const Dsv41Model::Outputs bounded_again = model.forward(tokens, false);
  require(std::equal(bounded.logits.begin() + static_cast<std::ptrdiff_t>(seg0) * cfg.vocab_size, bounded.logits.end(),
                     bounded_again.logits.begin() + static_cast<std::ptrdiff_t>(seg0) * cfg.vocab_size),
          "smoke: the bounded forward is not deterministic across calls");
  const auto btop = topk(bounded.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke bounded: the decoder from layer %d over rows %d..%d, top-1 of the last row %d (%.4g) vs exact %d\n",
              dec0, seg0, T - 1, btop.back()[0].first, btop.back()[0].second, top.back()[0].first);
  std::printf("[ OK ] dsv41_forward_smoke\n");
  return 0;
}

// The exact chain's engine states (DGPP_DSV41_DUMP_ENGINE_STATES: int32 L,
// T, W then bf16 rows) for the bounded end-to-end yardstick.
std::vector<std::vector<uint16_t>> load_engine_states(const std::string& path, int L, int T, int W) {
  std::ifstream f(path, std::ios::binary);
  require(f.good(), "cannot open the exact engine states " + path);
  int32_t hdr[3] = {0, 0, 0};
  f.read(reinterpret_cast<char*>(hdr), 12);
  require(hdr[0] == L && hdr[1] == T && hdr[2] == W, "the exact engine states' shape is not the dump's");
  std::vector<std::vector<uint16_t>> out(static_cast<size_t>(L), std::vector<uint16_t>(static_cast<size_t>(T) * W));
  for (auto& st : out) f.read(reinterpret_cast<char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
  require(f.good(), "the exact engine states are short");
  return out;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path, bool relaxed, bool bounded,
                    const std::string& exact_dump_path, const std::string& exact_states_path) {
  const Dump dump = Dump::load(dump_path);
  require(dump.prefill == (bounded ? "bounded" : "exact"), "the dump's prefill mode (" + dump.prefill + ") is not the test's");
  const Dsv41TextConfig cfg = Dsv41TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab && cfg.num_hidden_layers == dump.num_layers &&
              static_cast<int>(cfg.index_source_layer_ids.size()) == dump.index_layers && cfg.index_topk == dump.max_selected &&
              cfg.hc_mult == dump.streams,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);
  // The bounded walk: the layers from `dec0` on hold the segment's rows
  // (row t of the reference is engine row t - lo_layer(l)); the head's
  // rows before the segment read as NaN.
  const int seg0 = bounded ? T - std::min(cfg.sliding_window, T) : 0;
  const int dec0 = cfg.decoder_first_layer();
  require(dump.segment_row0 == seg0, "the dump's segment start disagrees with the config's window");
  const auto lo_layer = [&](int l) { return (bounded && l >= dec0) ? seg0 : 0; };
  const auto lo_source = [&](int li) { return lo_layer(cfg.index_source_layer_ids[static_cast<size_t>(li)]); };
  Dsv41Model model(cfg, dir, T, std::max(256, T));
  model.set_prefill_bounded(bounded);
  const Dsv41Model::Outputs out = model.forward(tokens, true);
  const std::vector<Dsv41Model::IndexLogits> index_logits = model.debug_index_logits();  // before the next walk clears them
  if (const char* states_path = std::getenv("DGPP_DSV41_DUMP_ENGINE_STATES")) {
    // The engine's layer streams for the reference's teacher-forced mode
    // (each python layer fed the engine's input): int32 L, T, W then bf16
    // (a bounded decoder layer's rows padded with NaN before the segment).
    std::ofstream f(states_path, std::ios::binary);
    const int32_t hdr[3] = {cfg.num_hidden_layers, T, cfg.hc_mult * cfg.hidden_size};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (size_t l = 0; l < out.layer_states.size(); ++l) {
      const auto& st = out.layer_states[l];
      const std::vector<uint16_t> pad(static_cast<size_t>(lo_layer(static_cast<int>(l))) * cfg.hc_mult * cfg.hidden_size, 0xFFFF);
      require(pad.size() + st.size() == static_cast<size_t>(T) * cfg.hc_mult * cfg.hidden_size, "engine layer state rows");
      f.write(reinterpret_cast<const char*>(pad.data()), static_cast<std::streamsize>(pad.size() * 2));
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    }
    std::printf("[ .. ] engine layer states written to %s\n", states_path);
  }
  const Dsv41Model::Outputs again = model.forward(tokens, false);
  const int H = cfg.hidden_size, W = cfg.hc_mult * H;
  require(std::equal(out.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(seg0) * H, out.final_hidden_bits.end(),
                     again.final_hidden_bits.begin() + static_cast<std::ptrdiff_t>(seg0) * H) &&
              std::equal(out.logits.begin() + static_cast<std::ptrdiff_t>(seg0) * cfg.vocab_size, out.logits.end(),
                         again.logits.begin() + static_cast<std::ptrdiff_t>(seg0) * cfg.vocab_size),
          "forward is not deterministic across calls");
  bool ok = true;
  // ---- the selections ----------------------------------------------------------------
  const int ms = dump.max_selected;
  const Dump::Tensor& dsel = dump.tensor("csa2_selections");
  const Dump::Tensor& dmarg = dump.tensor("csa2_margins");
  require(dsel.dtype == "I32" && dsel.shape.size() == 3 && dsel.shape[0] == dump.index_layers && dsel.shape[1] == T &&
              dsel.shape[2] == ms, "dump: csa2_selections shape");
  require(out.dsa_selections.size() == static_cast<size_t>(dump.index_layers), "captured selections");
  std::vector<bool> excluded(static_cast<size_t>(T), false);
  int flips = 0, uncertified = 0, first_flips = 0;
  double max_logit_dev = 0;
  {
    const int32_t* ref = reinterpret_cast<const int32_t*>(dsel.data);
    const float* marg = reinterpret_cast<const float*>(dmarg.data);
    const Dump::Tensor& dlog = dump.tensor("csa2_logits");
    require(dlog.dtype == "F32" && dlog.shape.size() == 3 && dlog.shape[0] == dump.index_layers && dlog.shape[1] == T,
            "dump: csa2_logits shape");
    const int64_t ref_stride = dlog.shape[2];
    const float* ref_logits = reinterpret_cast<const float*>(dlog.data);
    const auto& eng = index_logits;
    require(eng.size() == static_cast<size_t>(dump.index_layers), "captured index logits");
    const Dump::Tensor& dcand = dump.tensor("csa2_candidates");
    const Dump::Tensor& dbm = dump.tensor("csa2_block_margins");
    require(dcand.dtype == "I32" && dcand.shape.size() == 3 && dcand.shape[0] == dump.index_layers && dcand.shape[1] == T &&
                dcand.shape[2] == dump.candidate_blocks, "dump: csa2_candidates shape");
    const int32_t* ref_cand = reinterpret_cast<const int32_t*>(dcand.data);
    const float* ref_bmarg = reinterpret_cast<const float*>(dbm.data);
    // The engine's logits against the reference's per row: the relative
    // deviation over the visible entries (the fp4 code flips' scale), and
    // the certification of a flip: the reference's boundary gap within
    // twice that row's deviation (the flip is explained by the logits).
    std::vector<std::vector<double>> dev(static_cast<size_t>(dump.index_layers), std::vector<double>(static_cast<size_t>(T), 0.0));
    for (int l = 0; l < dump.index_layers; ++l) {
      const auto& e = eng[static_cast<size_t>(l)];
      const int lo_t = lo_source(l);
      for (int t = lo_t; t < T && t - lo_t < e.rows; ++t) {
        double lo = 1e300, hi = -1e300, d = 0;
        int n = 0;
        for (int64_t j = 0; j < std::min<int64_t>(ref_stride, e.stride); ++j) {
          const float r = ref_logits[(static_cast<size_t>(l) * T + t) * ref_stride + j];
          const float g = e.values[static_cast<size_t>(t - lo_t) * e.stride + j];
          if (std::isnan(r) || !std::isfinite(g)) continue;
          lo = std::min<double>(lo, r); hi = std::max<double>(hi, r);
          d = std::max<double>(d, std::fabs(g - r));
          ++n;
        }
        if (n > 0) dev[static_cast<size_t>(l)][static_cast<size_t>(t)] = d / std::max({hi - lo, std::fabs(hi), 1e-30});
        if (dev[static_cast<size_t>(l)][static_cast<size_t>(t)] > 1.0 && std::getenv("DGPP_DSV41_PARITY_VERBOSE")) {
          for (int64_t j = 0; j < std::min<int64_t>(ref_stride, e.stride); ++j) {
            const float r = ref_logits[(static_cast<size_t>(l) * T + t) * ref_stride + j];
            const float g = e.values[static_cast<size_t>(t - lo_t) * e.stride + j];
            if (std::isnan(r) || !std::isfinite(g)) continue;
            if (std::fabs(g - r) > 0.5 * (hi - lo))
              std::printf("[ .. ]     logit outlier: source %d row %d entry %lld: engine %g reference %g (range %g..%g)\n", l, t,
                          static_cast<long long>(j), g, r, lo, hi);
          }
        }
        max_logit_dev = std::max(max_logit_dev, dev[static_cast<size_t>(l)][static_cast<size_t>(t)]);
      }
    }
    for (int l = 0; l < dump.index_layers; ++l)
      for (int t = lo_source(l); t < T; ++t) {
        const int lo_t = lo_source(l);
        const int32_t* g = out.dsa_selections[static_cast<size_t>(l)].data() + static_cast<size_t>(t - lo_t) * ms;
        const int32_t* r = ref + (static_cast<size_t>(l) * T + t) * ms;
        if (std::memcmp(g, r, static_cast<size_t>(ms) * 4) == 0) continue;
        ++flips;
        if (l == 0) ++first_flips;
        const float m = marg[static_cast<size_t>(l) * T + t];
        std::set<int32_t> gs, rs;
        for (int i = 0; i < ms; ++i) {
          if (g[i] >= 0) gs.insert(g[i]);
          if (r[i] >= 0) rs.insert(r[i]);
        }
        std::string only_g, only_r;
        for (int32_t v : gs) if (!rs.count(v)) only_g += " " + std::to_string(v);
        for (int32_t v : rs) if (!gs.count(v)) only_r += " " + std::to_string(v);
        // Certified when the reference's boundary gap is within twice the
        // row's engine-vs-reference logit deviation (an fp4 code flip of
        // the query or a key moves single logits by percents of the range)
        // plus the fp32-vs-double noise, or a near tie outright. A row
        // whose candidate POOL differs (a candidate source or user) is
        // judged on the reference's block-level margin instead.
        const double dv = dev[static_cast<size_t>(l)][static_cast<size_t>(t)];
        double margin = m;
        const char* kind = "entry";
        if (!eng[static_cast<size_t>(l)].cand.empty()) {
          const int CB = dump.candidate_blocks;
          const int32_t* ec = eng[static_cast<size_t>(l)].cand.data() + static_cast<size_t>(t - lo_t) * CB;
          const int32_t* rc = ref_cand + (static_cast<size_t>(l) * T + t) * CB;
          const int en = eng[static_cast<size_t>(l)].cand_counts[static_cast<size_t>(t - lo_t)];
          bool same = true;
          for (int i = 0; i < CB; ++i)
            if ((i < en ? ec[i] : -1) != rc[i]) same = false;
          if (!same) {
            margin = ref_bmarg[static_cast<size_t>(l) * T + t];
            kind = "block";
            if (std::getenv("DGPP_DSV41_PARITY_VERBOSE")) {
              std::string ep, rp;
              for (int i = 0; i < CB; ++i) {
                ep += " " + std::to_string(i < en ? ec[i] : -1);
                rp += " " + std::to_string(rc[i]);
              }
              std::printf("[ .. ]     pools: engine (%d)%s, reference%s\n", en, ep.c_str(), rp.c_str());
            }
          }
        }
        const bool certified = margin >= 0 && (margin <= 2e-3 || margin <= 2.0 * dv + 2e-3);
        std::printf("[ .. ]   selection flip: index source %d token %d (reference %s margin %.3g, logit deviation %.3g%s): engine-only%s, reference-only%s\n",
                    l, t, kind, margin, dv, certified ? "" : ", UNCERTIFIED", only_g.c_str(), only_r.c_str());
        if (!certified) ++uncertified;
        excluded[static_cast<size_t>(t)] = true;
      }
    std::printf("[ .. ] selections: %d flips of %d rows (%d on the first source), %d uncertified; max logit deviation %.3g of the range\n",
                flips, dump.index_layers * T, first_flips, uncertified, max_logit_dev);
    if (uncertified > 0) ok = false;
  }
  // ---- the routing ---------------------------------------------------------------------
  {
    const Dump::Tensor& rt = dump.tensor("route_ids");
    const Dump::Tensor& rm = dump.tensor("route_margins");
    const int K = static_cast<int>(rt.shape[2]);
    require(rt.shape[0] == dump.num_layers && out.route_ids.size() == static_cast<size_t>(dump.num_layers), "dump: route_ids layers");
    const int32_t* ref = reinterpret_cast<const int32_t*>(rt.data);
    const float* marg = reinterpret_cast<const float*>(rm.data);
    long diff = 0, total = 0, route_uncertified = 0;
    for (int l = 0; l < dump.num_layers; ++l)
      for (int t = lo_layer(l); t < T; ++t) {
        if (excluded[static_cast<size_t>(t)]) continue;
        bool row_diff = false;
        for (int i = 0; i < K; ++i, ++total)
          if (out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t - lo_layer(l)) * K + i] !=
              ref[(static_cast<size_t>(l) * T + t) * K + i]) {
            ++diff;
            row_diff = true;
          }
        if (row_diff) {
          const float m = marg[static_cast<size_t>(l) * T + t];
          const bool certified = m <= 2e-2f;
          if (!certified) ++route_uncertified;
          std::printf("[ .. ]   route flip: layer %d token %d (reference margin %.3g%s)\n", l, t, m, certified ? "" : ", UNCERTIFIED");
          excluded[static_cast<size_t>(t)] = true;
        }
      }
    std::printf("[ .. ] routing: %ld of %ld slots differ from the reference, %ld uncertified\n", diff, total, route_uncertified);
    if (route_uncertified > 0 || (total > 0 && diff > total / 10)) ok = false;
  }
  // ---- the per-layer streams --------------------------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers && ls.shape[1] == T && ls.shape[2] == W,
          "dump: layer_states shape");
  // The bounded end-to-end yardstick: the hard-ulp fraction is a threshold
  // statistic of the row population, and the bounded gate's population is
  // the segment alone — the prompt's deepest rows, where the accumulated
  // cache drift is largest, with no early rows to dilute it (the exact
  // chain's same rows read above the 40-row budget too: 0.59 % at layer 7
  // on the fixture), and a segment row's window is floored at the segment
  // start, so its attention leans more on the fp4 entries (+10–25 % drift
  // on the first segment rows). So the exact chain's engine-vs-reference
  // hard fraction over the SAME kept rows, measured here from its own
  // states and dump, sets the bounded layers' budget: within twice it (and
  // never below the 40-row budget).
  std::vector<double> yardstick(static_cast<size_t>(cfg.num_hidden_layers), 0.0);
  const bool have_yardstick = bounded && relaxed && !exact_dump_path.empty() && !exact_states_path.empty();
  // The hard-ulp floor: 0.5 % of the elements; the relaxed end-to-end run
  // without a yardstick 2 % — its kept rows sit downstream of the certified
  // flips, whose cached entries move later rows' fp8/fp4 codes by whole
  // steps (a code step is hundreds of ulps: a hard mismatch by
  // construction; 0.7–1.7 % of the elements at layers 4–7 under the
  // tensor-core GEMM's fold order, under 0.5 % with the chunks' — a
  // different set of near-tie flips, the same kernels' accuracy: the
  // layer-local strict run holds 0 hard). The bounded run measures its
  // floor from the exact chain's own rows (the yardstick).
  const auto hard_budget = [&](int l) {
    return std::max(relaxed && !have_yardstick ? 0.02 : 0.005, 2.0 * yardstick[static_cast<size_t>(l)]);
  };
  if (have_yardstick) {
    const Dump exact = Dump::load(exact_dump_path);
    require(exact.prefill == "exact" && exact.token_count == T, "the yardstick dump is not the exact chain's on the same tokens");
    const Dump::Tensor& els = exact.tensor("layer_states");
    require(els.shape == ls.shape, "the yardstick dump's layer_states shape");
    const std::vector<std::vector<uint16_t>> states = load_engine_states(exact_states_path, cfg.num_hidden_layers, T, W);
    std::string line;
    for (int l = dec0; l < cfg.num_hidden_layers; ++l) {
      std::vector<uint16_t> g, w;
      const uint16_t* want = reinterpret_cast<const uint16_t*>(els.data) + static_cast<size_t>(l) * T * W;
      for (int t = seg0; t < T; ++t) {
        if (excluded[static_cast<size_t>(t)]) continue;
        g.insert(g.end(), states[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(t) * W,
                 states[static_cast<size_t>(l)].begin() + static_cast<std::ptrdiff_t>(t + 1) * W);
        w.insert(w.end(), want + static_cast<size_t>(t) * W, want + static_cast<size_t>(t + 1) * W);
      }
      const Stats es = compare_bf16(g.data(), w.data(), g.size(), 16, 128);
      yardstick[static_cast<size_t>(l)] = es.total > 0 ? static_cast<double>(es.hard) / es.total : 0.0;
      char buf[64];
      std::snprintf(buf, sizeof(buf), " L%d:%.3g%%", l, 100.0 * yardstick[static_cast<size_t>(l)]);
      line += buf;
    }
    std::printf("[ .. ] yardstick: the exact chain's hard-ulp fraction over the segment's kept rows%s (the bounded budget: twice it)\n",
                line.c_str());
  }
  // `got` holds the rows from `lo` on (row t at got + (t - lo) * width);
  // `want` every row.
  const auto layer_stats = [&](const uint16_t* got, const uint16_t* want, int width, Stats* kept, int lo, int hi = -1) {
    std::vector<uint16_t> g, w;
    for (int t = lo; t < (hi < 0 ? T : hi); ++t) {
      if (excluded[static_cast<size_t>(t)]) continue;
      g.insert(g.end(), got + static_cast<size_t>(t - lo) * width, got + static_cast<size_t>(t - lo + 1) * width);
      w.insert(w.end(), want + static_cast<size_t>(t) * width, want + static_cast<size_t>(t + 1) * width);
    }
    *kept = g.empty() ? Stats{} : compare_bf16(g.data(), w.data(), g.size(), 16, 128);
  };

  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const int lo = lo_layer(l);
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * W;
    const uint16_t* got = out.layer_states[static_cast<size_t>(l)].data();
    require(out.layer_states[static_cast<size_t>(l)].size() == static_cast<size_t>(T - lo) * W, "engine layer state rows");
    Stats s;
    layer_stats(got, want, W, &s, lo);
    std::vector<std::pair<double, int>> rows;
    for (int t = lo; t < T; ++t) {
      const Stats r = compare_bf16(got + static_cast<size_t>(t - lo) * W, want + static_cast<size_t>(t) * W, static_cast<size_t>(W), 16, 128);
      rows.emplace_back(r.l2, t);
    }
    std::sort(rows.rbegin(), rows.rend());
    if (s.hard > 0 && std::getenv("DGPP_DSV41_PARITY_VERBOSE")) {
      // The worst elements: (row, stream, dim, got, want).
      std::vector<std::pair<int, size_t>> worst;
      for (int t = lo; t < T; ++t) {
        if (excluded[static_cast<size_t>(t)]) continue;
        for (int i = 0; i < W; ++i) {
          const size_t at = static_cast<size_t>(t) * W + i;
          worst.emplace_back(bf16_ulps(got[at - static_cast<size_t>(lo) * W], want[at]), at);
        }
      }
      std::sort(worst.rbegin(), worst.rend());
      for (int i = 0; i < 6 && i < static_cast<int>(worst.size()); ++i) {
        const size_t at = worst[static_cast<size_t>(i)].second;
        std::printf("[ .. ]     outlier t%zu s%zu d%zu: got %g want %g (%d ulps)\n", at / W, (at % W) / H, at % H,
                    bf16_bits_to_float(got[at - static_cast<size_t>(lo) * W]), bf16_bits_to_float(want[at]),
                    worst[static_cast<size_t>(i)].first);
      }
    }
    std::printf("[ .. ] layer %d streams (kept rows%s): l2 %.3g max %g ulps, soft %ld hard %ld of %ld; worst rows:", l,
                lo > 0 ? " of the segment" : "", s.l2, s.max_ulps, s.soft, s.hard, s.total);
    for (int i = 0; i < 3 && i < static_cast<int>(rows.size()); ++i)
      std::printf(" t%d %.3g%s", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first,
                  excluded[static_cast<size_t>(rows[static_cast<size_t>(i)].second)] ? "(flip)" : "");
    std::printf("\n");
    // Relaxed (end to end): the kept rows ride the cascade of the certified
    // flips — a flipped row's cached entries perturb every later query, and
    // every row's own fp8/fp4 cache codes re-quantize sub-ulp kernel
    // differences into code steps layer by layer (the fixture amplifies
    // ~1.5x per layer; even the first row, with no flip anywhere, sits at
    // 0.047 by layer 7). Bounded at 0.03 — the realized values read 0.013
    // under the GEMV chunks and 0.021–0.023 under the tensor-core GEMM's
    // order, one bf16 rounding apart per element and a different set of
    // certified flips. The kernels' accuracy gate is the layer-local twin
    // (the strict run: every layer from the teacher's streams, l2 0.0046,
    // 0 hard), which does not move with the fold order.
    if (s.total > 0 && (s.l2 > (relaxed ? 0.03 : 0.006) || static_cast<double>(s.hard) / s.total > hard_budget(l))) ok = false;
  }
  // ---- the final read and the logits ---------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H, "dump: final_hidden shape");
  {
    Stats fs_;
    layer_stats(out.final_hidden_bits.data() + static_cast<size_t>(seg0) * H, reinterpret_cast<const uint16_t*>(fh.data), H, &fs_,
                seg0);
    std::printf("[ .. ] final hidden (kept rows): l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2, fs_.max_ulps, fs_.soft,
                fs_.hard, fs_.total);
    if (fs_.total > 0 && (fs_.l2 > (relaxed ? 0.035 : 0.008) ||
                          static_cast<double>(fs_.hard) / fs_.total > hard_budget(cfg.num_hidden_layers - 1)))
      ok = false;
  }
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0, kept_rows = 0;
  double max_rel = 0;
  for (int t = seg0; t < T; ++t) {
    if (excluded[static_cast<size_t>(t)]) continue;
    ++kept_rows;
    const int32_t r1 = ref_ids[static_cast<size_t>(t) * k];
    const float v1 = ref_vals[static_cast<size_t>(t) * k], v2 = ref_vals[static_cast<size_t>(t) * k + 1];
    const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != r1) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const int32_t* row = ref_ids + static_cast<size_t>(t) * k;
      const int32_t* f = std::find(row, row + k, id);
      if (f == row + k) { ++set_miss; continue; }
      const float rv = ref_vals[static_cast<size_t>(t) * k + (f - row)];
      max_rel = std::max(max_rel, std::fabs(rv - got[static_cast<size_t>(t)][i].second) / (std::fabs(rv) + 1e-30));
    }
  }
  std::printf("[ .. ] logits (kept rows): top-1 hard mismatches %d, near-tie %d of %d rows; top-%d set misses %d of %d; matched logits within %.3g relative\n",
              top1_hard, top1_soft, kept_rows, k, set_miss, kept_rows * k, max_rel);
  if (top1_hard != 0 || set_miss > kept_rows * k / 4) ok = false;
  // ---- the DSpark draft (plan D8) ------------------------------------------------
  // The reference's forward_spec after the prompt: the engine's session
  // prefill (the draft rings seeded through mtp_prefill_rows), the first
  // draft call over the last prompt row and the chain rows fed the
  // REFERENCE's picks (teacher-forced Markov chain), row by row against
  // the reference's biased logits; the picks equal or a certified near tie.
  if (dump.dspark_block > 0) {
    const int B = dump.dspark_block;
    require(B == cfg.dspark_block_size && cfg.num_nextn_predict_layers > 0, "dump: dspark block vs the config");
    const Dump::Tensor& dl = dump.tensor("dspark_logits");
    const Dump::Tensor& di = dump.tensor("dspark_ids");
    const Dump::Tensor& dmg = dump.tensor("dspark_margins");
    const Dump::Tensor& dc = dump.tensor("dspark_confidence");
    require(dl.dtype == "F32" && dl.shape.size() == 2 && dl.shape[0] == B && dl.shape[1] == cfg.vocab_size, "dump: dspark_logits");
    const float* ref_rows = reinterpret_cast<const float*>(dl.data);
    const int32_t* ref_ids_d = reinterpret_cast<const int32_t*>(di.data);
    const float* ref_marg = reinterpret_cast<const float*>(dmg.data);
    const float* ref_conf = reinterpret_cast<const float*>(dc.data);
    const int V = cfg.vocab_size;
    Dsv41Model dm(cfg, dir, T, std::max(256, T), dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, /*mtp=*/true,
                  /*decode_rows=*/1 + B);
    dm.set_prefill_bounded(bounded);
    (void)dm.session_prefill(0, tokens);
    std::vector<std::vector<float>> rows;
    setenv("DGPP_DSV41_CAPTURE_DRAFT", "1", 1);
    rows.push_back(dm.session_draft(0, {ref_ids_d[0]}).logits);
    unsetenv("DGPP_DSV41_CAPTURE_DRAFT");
    {
      // The block walk's sites per stage against the reference's (the localizer).
      const auto& sites = dm.debug_sites();
      const int S = cfg.num_nextn_predict_layers;
      require(static_cast<int>(sites.size()) == S, "draft site captures");
      const char* names[4] = {"x_attn", "attn_out", "x_ffn", "ffn_out"};
      for (int st = 0; st < S; ++st)
        for (int n = 0; n < 4; ++n) {
          const Dump::Tensor& dt = dump.tensor(std::string("dspark_") + names[n]);
          const uint16_t* ref = reinterpret_cast<const uint16_t*>(dt.data) + (static_cast<size_t>(st) * B) * H;
          const std::vector<uint16_t>& got = n == 0 ? sites[static_cast<size_t>(st)].x_attn
                                           : n == 1 ? sites[static_cast<size_t>(st)].attn_out
                                           : n == 2 ? sites[static_cast<size_t>(st)].x_ffn
                                                    : sites[static_cast<size_t>(st)].ffn_out;
          require(got.size() == static_cast<size_t>(B) * H, "draft site width");
          std::printf("[ .. ] dspark stage %d %s rows l2:", st, names[n]);
          for (int b = 0; b < B; ++b) {
            double d2 = 0, r2 = 0;
            for (int i = 0; i < H; ++i) {
              const double g = bf16_bits_to_float(got[static_cast<size_t>(b) * H + i]);
              const double r = bf16_bits_to_float(ref[static_cast<size_t>(b) * H + i]);
              d2 += (g - r) * (g - r);
              r2 += r * r;
            }
            std::printf(" %.3g", std::sqrt(d2) / std::max(std::sqrt(r2), 1e-30));
          }
          std::printf("\n");
        }
    }
    for (int i = 0; i < B - 1; ++i)
      rows.push_back(dm.session_draft_chain(0, ref_ids_d[i + 1], i, i == 0, i == B - 2).logits);
    std::vector<float> conf(static_cast<size_t>(B));
    DGPP_CUDA_OK(cudaMemcpy(conf.data(), dm.debug_confidence(), static_cast<size_t>(B) * 4, cudaMemcpyDeviceToHost));
    int pick_hard = 0, pick_soft = 0;
    double worst_l2 = 0, worst_conf = 0;
    for (int i = 0; i < B; ++i) {
      const float* r = ref_rows + static_cast<size_t>(i) * V;
      const std::vector<float>& g = rows[static_cast<size_t>(i)];
      require(g.size() == static_cast<size_t>(V), "draft row width");
      double d2 = 0, r2 = 0, dmax = 0, lo = 1e300, hi = -1e300;
      int gmax = 0;
      for (int v = 0; v < V; ++v) {
        d2 += double(g[static_cast<size_t>(v)] - r[v]) * (g[static_cast<size_t>(v)] - r[v]);
        r2 += double(r[v]) * r[v];
        dmax = std::max(dmax, std::fabs(double(g[static_cast<size_t>(v)]) - r[v]));
        lo = std::min<double>(lo, r[v]);
        hi = std::max<double>(hi, r[v]);
        if (g[static_cast<size_t>(v)] > g[static_cast<size_t>(gmax)]) gmax = v;
      }
      const double l2 = std::sqrt(d2) / std::max(std::sqrt(r2), 1e-30);
      const double dev = dmax / std::max({hi - lo, std::fabs(hi), 1e-30});
      const double margin = ref_marg[i] / std::max({hi - lo, std::fabs(hi), 1e-30});
      worst_l2 = std::max(worst_l2, l2);
      const bool same = gmax == ref_ids_d[i + 1];
      if (!same) {
        if (margin <= 2.0 * dev + 2e-3) ++pick_soft; else ++pick_hard;
      }
      const double cdiff = std::fabs(double(conf[static_cast<size_t>(i)]) - ref_conf[i]) / (1.0 + std::fabs(double(ref_conf[i])));
      worst_conf = std::max(worst_conf, cdiff);
      std::printf("[ .. ] dspark row %d: l2 %.3g, logit deviation %.3g of the range, pick %d vs %d%s (margin %.3g), confidence %.4f vs %.4f\n",
                  i, l2, dev, gmax, ref_ids_d[i + 1], same ? "" : (margin <= 2.0 * dev + 2e-3 ? " (near tie)" : " MISMATCH"),
                  margin, conf[static_cast<size_t>(i)], ref_conf[i]);
    }
    // End to end the rings carry the main path's drift (the reference's
    // target-layer states differ from the engine's by the layers' budget),
    // amplified by the attention; teacher-forced the rows sit at 0.5 %.
    std::printf("[ .. ] dspark: worst row l2 %.3g (budget %.3g), picks %d hard %d near-tie of %d, confidence within %.3g\n",
                worst_l2, relaxed ? 0.05 : 0.01, pick_hard, pick_soft, B, worst_conf);
    if (worst_l2 > (relaxed ? 0.05 : 0.01) || pick_hard != 0 || worst_conf > 0.05) ok = false;
    dm.session_close(0);
  }
  std::printf("[ %s ] dsv41_forward_dump_parity (%s budgets, %s prefill)\n", ok ? "OK" : "FAIL",
              relaxed ? "end-to-end" : "teacher-forced", bounded ? "bounded" : "exact");
  return ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
  std::string fixture, smoke, checkpoint, dump, exact_dump, exact_states;
  bool relaxed = false, bounded = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else if (a == "--relaxed") relaxed = true;
    else if (a == "--bounded") bounded = true;
    else if (a == "--exact-dump" && i + 1 < argc) exact_dump = argv[++i];
    else if (a == "--exact-states" && i + 1 < argc) exact_states = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!fixture.empty()) {
      const Dsv41TextConfig cfg = dsv41fx::tiny_config();
      dsv41fx::write_fixture(cfg, fixture);
      (void)dsv41fx::fixture_sidecar(cfg, fixture);
      std::printf("[ OK ] wrote the fixture and its sidecar to %s\n", fixture.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump, relaxed, bounded, exact_dump, exact_states);
    std::fprintf(stderr, "usage: --write-fixture DIR | --smoke DIR | --checkpoint-dir DIR --dump-file FILE [--relaxed] [--bounded "
                         "[--exact-dump FILE --exact-states FILE]]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
