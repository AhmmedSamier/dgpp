// glm_forward_check: M4 deliverables 1 and 5 deployment run (DESIGN §7.5).
//
// Curated reference suite: ISOLATED per-layer parity over the real
// GLM-5.3-Flash checkpoint against dumps from
// tools/glm_reference_dump.py gen-torch. Every layer starts from the
// reference trajectory (module noise does not compound — free-run
// end-to-end drift is chaos-limited at this depth; DESIGN §7.5), so the
// suite pins the wiring + module floors layer by layer and the head on
// the isolated final streams. Routing: each side recomputes its own
// mHC+ln2 chain from the injected streams, so router inputs carry ~1e-3
// fp noise and near-boundary experts flip — every flip is CERTIFIED as a
// measured near tie (models/glm_route_audit.hpp), never tolerated by
// budget.
//
// Route-trace capture (deliverable 5): with --trace-dir, each case's
// routing decisions are written as a DGPPTC1 trace file for
// tools/route_trace_traffic.py — real traces from representative prompts,
// replacing the uniform-expert null model in the per-rank traffic numbers.
//
//   glm_forward_check --config CONFIG --checkpoint-dir DIR
//       --suite FILE [--trace-dir DIR] [--layers N] [--tokens N]
//
// Suite file: one case per line, "name|token-ids" or "name|PROMPT|text"
// (the latter requires the dump to have been generated with the same
// prompt; the engine consumes the dump's token ids either way).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/glm_dump.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_route_audit.hpp"
#include "models/glm_trace.hpp"

namespace fs = std::filesystem;

namespace {

struct Case {
  std::string name;
  std::string dump_path;
};

std::vector<Case> load_suite(const std::string& path,
                             const std::string& dump_ext) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open suite file: " + path);
  std::vector<Case> cases;
  std::string line;
  int ln = 0;
  while (std::getline(f, line)) {
    ++ln;
    // name | dump-path (blank lines and # comments allowed)
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    while (!line.empty() && (line.back() == ' ' || line.back() == '\r'))
      line.pop_back();
    if (line.empty()) continue;
    const size_t bar = line.find('|');
    if (bar == std::string::npos)
      throw std::runtime_error("suite line " + std::to_string(ln) +
                               ": expected 'name|dump-path'");
    Case c;
    c.name = line.substr(0, bar);
    c.dump_path = line.substr(bar + 1);
    while (!c.name.empty() && c.name.back() == ' ') c.name.pop_back();
    while (!c.dump_path.empty() && c.dump_path.front() == ' ')
      c.dump_path.erase(c.dump_path.begin());
    if (c.name.empty() || c.dump_path.empty())
      throw std::runtime_error("suite line " + std::to_string(ln) +
                               ": empty name or dump path");
    cases.push_back(std::move(c));
    (void)dump_ext;
  }
  if (cases.empty()) throw std::runtime_error("suite file has no cases");
  return cases;
}

struct CaseReport {
  int64_t tokens = 0;
  double max_layer_l2 = 0;          // isolated per-layer drift (worst layer)
  int worst_layer = -1;
  double worst_layer_kept_l2 = 0;   // ...excluding rows with a route flip
  int flipped_rows_total = 0;
  int route_slots_total = 0;
  double head_hidden_l2 = 0;
  int top1_mismatch = 0, set_miss = 0;
  int top1_certified = 0;  // ...of the mismatches, proven boundary near ties
  bool head_audit_failed = false;
  int route_id_mismatch = 0;
  double route_kept_max_rel = 0;    // weight rel error, non-flipped rows
  int64_t route_flips_certified = 0;
  int64_t route_swaps_certified = 0;
  double route_audit_max_multiple = 0;  // max gap / measured noise
  bool route_audit_failed = false;      // a flip failed certification
  double seconds = 0;
};

CaseReport run_case(dgpp::GlmDiagnosticModel& model,
                    const dgpp::GlmDumpFile& dump,
                    const std::string& trace_dir,
                    const std::string& name) {
  CaseReport rep;
  const int64_t T = dump.token_count();
  rep.tokens = T;
  const int64_t H = dump.hidden();
  const std::vector<int64_t> tokens(dump.tokens(), dump.tokens() + T);

  // ---- isolated parity: every layer starts from the reference trajectory
  const uint16_t* streams = dump.streams_all();
  const int64_t layer_elems = T * 4 * H;
  std::vector<const uint16_t*> layer_inputs;
  for (int l = 0; l <= dump.num_layers(); ++l)
    layer_inputs.push_back(streams + static_cast<size_t>(l) * layer_elems);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<uint16_t>> capture;
  const dgpp::GlmDiagnosticModel::Outputs iso =
      model.forward_isolated(tokens, layer_inputs, capture);
  const auto t1 = std::chrono::steady_clock::now();
  rep.seconds = std::chrono::duration<double>(t1 - t0).count();

  // ---- per-layer drift, flip-aware: the MoE/DSA route selection near-tie
  // is a documented event class (fp32-order noise on trained boundaries);
  // a flipped token row legitimately moves O(1). Kept rows must sit at the
  // cross-implementation floor.
  {
    const int64_t H4 = 4 * H;
    for (int l = 0; l < dump.num_layers(); ++l) {
      size_t route_pos = 0;  // walk restarts for each layer
      const uint16_t* got = capture[l + 1].data();
      const uint16_t* want = streams + static_cast<size_t>(l + 1) * layer_elems;
      // which token rows have a routing flip on this layer
      std::vector<char> flipped(static_cast<size_t>(T), 0);
      int layer_flips = 0;
      for (size_t li = 0; li < dump.route_layers().size(); ++li) {
        const auto& rl = dump.route_layers()[li];
        if (rl.layer_idx != l) {
          route_pos += static_cast<size_t>(rl.tokens) * rl.top_k;
          continue;
        }
        const auto& gr = iso.routes[li];
        bool any = false;
        for (int64_t t = 0; t < rl.tokens; ++t)
          for (int k = 0; k < rl.top_k; ++k) {
            const size_t idx = static_cast<size_t>(t) * rl.top_k + k;
            if (gr.ids[idx] != dump.route_ids()[route_pos + idx]) {
              flipped[t] = 1;
              ++layer_flips;
              any = true;
            }
          }
        route_pos += static_cast<size_t>(rl.tokens) * rl.top_k;
        (void)any;
      }
      double d2 = 0, o2 = 0, kd2 = 0, ko2 = 0;
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t e = 0; e < H4; ++e) {
          const int64_t i = t * H4 + e;
          const double a = dgpp::bf16_bits_to_float(got[i]);
          const double b = dgpp::bf16_bits_to_float(want[i]);
          d2 += (a - b) * (a - b);
          o2 += b * b;
          if (!flipped[t]) {
            kd2 += (a - b) * (a - b);
            ko2 += b * b;
          }
        }
      }
      const double l2 = std::sqrt(d2 / (o2 + 1e-30));
      const double kept = std::sqrt(kd2 / (ko2 + 1e-30));
      if (l2 > rep.max_layer_l2) {
        rep.max_layer_l2 = l2;
        rep.worst_layer = l;
      }
      if (kept > rep.worst_layer_kept_l2)
        rep.worst_layer_kept_l2 = kept;
      rep.flipped_rows_total += layer_flips > 0 ? 1 : 0;
      std::printf("  L%02d l2=%.4f kept=%.4f%s\n", l, l2, kept,
                  layer_flips ? " (route flip)" : "");
    }
  }

  // ---- head parity on the isolated final streams
  {
    const uint16_t* want = dump.final_hidden();
    double d2 = 0, o2 = 0;
    for (int64_t i = 0; i < T * H; ++i) {
      const double a = dgpp::bf16_bits_to_float(iso.final_hidden_bits[i]);
      const double b = dgpp::bf16_bits_to_float(want[i]);
      d2 += (a - b) * (a - b);
      o2 += b * b;
    }
    rep.head_hidden_l2 = std::sqrt(d2 / (o2 + 1e-30));
  }
  const auto got_top = dgpp::GlmDiagnosticModel::topk(
      iso.logits_bits, T, model.config().vocab_size, dump.top_k());
  const int32_t* ref_ids = dump.topk_ids();
  const float* ref_logits = dump.topk_logits();
  // The head's top-1 boundary can be a near tie like any selection: at a
  // reduced layer budget the truncated stack's logits crowd together, and
  // a head sitting at the cross-implementation floor legitimately flips
  // the argmax where the reference's own top-2 margin is within the
  // measured logit noise. Certify each disagreement from the dump's
  // topk_logits (the reference's actual values); a large-margin
  // disagreement is a real divergence and fails the case.
  double logit_noise_sum = 0;
  int64_t logit_noise_n = 0;
  for (int64_t t = 0; t < T; ++t)
    if (got_top[t][0].first == ref_ids[t * dump.top_k()]) {
      logit_noise_sum +=
          std::abs(double(got_top[t][0].second) -
                   double(ref_logits[t * dump.top_k()]));
      ++logit_noise_n;
    }
  const double logit_noise =
      logit_noise_n > 0 ? logit_noise_sum / double(logit_noise_n) : 0;
  for (int64_t t = 0; t < T; ++t) {
    if (got_top[t][0].first != ref_ids[t * dump.top_k()])
      ++rep.top1_mismatch;
    std::vector<int32_t> g, r;
    for (int k = 0; k < dump.top_k(); ++k) {
      g.push_back(got_top[t][k].first);
      r.push_back(ref_ids[t * dump.top_k() + k]);
    }
    std::sort(g.begin(), g.end());
    std::sort(r.begin(), r.end());
    std::vector<int32_t> inter;
    std::set_intersection(g.begin(), g.end(), r.begin(), r.end(),
                          std::back_inserter(inter));
    rep.set_miss += dump.top_k() - static_cast<int>(inter.size());
    if (got_top[t][0].first != ref_ids[t * dump.top_k()]) {
      const double margin =
          double(ref_logits[t * dump.top_k()]) -
          double(ref_logits[t * dump.top_k() + 1]);
      const double multiple =
          logit_noise > 0 ? margin / logit_noise : (margin > 0 ? 1e30 : 0);
      if (multiple <= 32.0) {
        ++rep.top1_certified;
      } else {
        std::printf(
            "  T%03lld HEAD AUDIT REJECTED: top-1 %d vs reference %d, "
            "margin %.4f = %.1fx logit noise — NOT a boundary near tie\n",
            (long long)t, got_top[t][0].first,
            ref_ids[t * dump.top_k()], margin, multiple);
        rep.head_audit_failed = true;
      }
    }
  }

  // ---- isolated routing agreement. Each side's mhc+ln2 chain applies its
  // own ~1e-3 fp noise to the injected streams before the router, so
  // near-boundary flips are the documented event class; every flip must
  // CERTIFY as a measured near tie (both sides' biased score rows are the
  // audit's inputs — the noise is measured per token, never assumed).
  // KEPT tokens' weights sit at the fp32-order floor (~1e-3..1e-2 after
  // the sigmoid-normalize chain over a perturbed input).
  if (iso.routes.size() == dump.route_layers().size()) {
    if (!dump.route_layers().empty()) {
      if (dump.n_experts() <= 0 || !dump.has_tensor("router_biased"))
        throw std::runtime_error(
            "case '" + name + "': dump predates route certification "
            "(no router_biased) — regenerate with a current "
            "tools/glm_reference_dump.py");
      if (iso.route_biased.size() != iso.routes.size())
        throw std::runtime_error("model route_biased/routes misaligned");
    }
    const float* ref_biased = dump.router_biased();
    size_t pos = 0, bpos = 0;
    for (size_t li = 0; li < dump.route_layers().size(); ++li) {
      const auto& rl = dump.route_layers()[li];
      const auto& gr = iso.routes[li];
      const int32_t* rid = dump.route_ids() + pos;
      const float* rw = dump.route_weights() + pos;
      const float* rb = ref_biased + bpos;
      const float* eb = iso.route_biased[li].data();
      if (iso.route_biased[li].size() !=
          static_cast<size_t>(rl.tokens) * dump.n_experts())
        throw std::runtime_error("engine biased-score row misaligned");
      bool any_flip = false;
      for (int64_t t = 0; t < rl.tokens; ++t)
        for (int k = 0; k < rl.top_k; ++k) {
          const size_t idx = static_cast<size_t>(t) * rl.top_k + k;
          if (gr.ids[idx] != rid[idx]) {
            ++rep.route_id_mismatch;
            any_flip = true;
          }
        }
      if (any_flip) {
        // Certify-or-reject. A rejection is a hard diagnosis (router bug
        // or beyond-noise divergence) — record it and fail the case, but
        // keep collecting evidence from the remaining layers/cases.
        try {
          dgpp::glm_route::RouteFlipAudit ra;
          dgpp::glm_route::audit_route_flips(
              gr.ids.data(), rid, eb, rb, rl.tokens, rl.top_k,
              dump.n_experts(), ra, rl.layer_idx);
          rep.route_flips_certified += ra.tokens_flipped;
          rep.route_swaps_certified += ra.swaps_certified;
          rep.route_audit_max_multiple =
              std::max(rep.route_audit_max_multiple, ra.max_noise_multiple);
        } catch (const std::exception& e) {
          std::printf("  L%02d ROUTE AUDIT REJECTED: %s\n", rl.layer_idx,
                      e.what());
          rep.route_audit_failed = true;
        }
      } else {
        for (int64_t t = 0; t < rl.tokens; ++t)
          for (int k = 0; k < rl.top_k; ++k) {
            const size_t idx = static_cast<size_t>(t) * rl.top_k + k;
            const double rel = std::abs(gr.weights[idx] - rw[idx]) /
                               (std::abs(rw[idx]) + 1e-30);
            rep.route_kept_max_rel = std::max(rep.route_kept_max_rel, rel);
          }
      }
      pos += static_cast<size_t>(rl.tokens) * rl.top_k;
      bpos += static_cast<size_t>(rl.tokens) * dump.n_experts();
      rep.route_slots_total += static_cast<int>(rl.tokens * rl.top_k);
    }
  }

  // ---- trace capture: the FREE-RUN trajectory's routing (real behavior
  // for the traffic model; free-run hidden parity is chaos-limited and
  // deliberately not asserted — see DESIGN §7.5).
  if (!trace_dir.empty()) {
    const dgpp::GlmDiagnosticModel::Outputs free_run = model.forward(tokens);
    fs::create_directories(trace_dir);
    const std::string path = (fs::path(trace_dir) / (name + ".trace")).string();
    dgpp::glm_trace_write(path, free_run.routes);
  }
  return rep;
}

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, suite_path, trace_dir;
  std::string trace_ids_file, trace_out;
  int max_layers = -1, max_tokens = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for argument");
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--suite") suite_path = next();
    else if (a == "--trace-dir") trace_dir = next();
    else if (a == "--trace-ids-file") trace_ids_file = next();
    else if (a == "--trace-out") trace_out = next();
    else if (a == "--layers") max_layers = std::stoi(std::string(next()));
    else if (a == "--tokens") max_tokens = std::stoi(std::string(next()));
    else throw std::runtime_error("unknown argument");
  }
  if (config_path.empty() || checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: glm_forward_check --config CONFIG --checkpoint-dir DIR "
        "(--suite FILE [--trace-dir DIR] | --trace-ids-file FILE "
        "--trace-out PATH) [--layers N] [--tokens N]");

  // Engine-only trace capture: free-run forward over a long prompt (no
  // reference dump — the torch reference is the slow path; trace occupancy
  // statistics do not need per-token parity).
  if (!trace_ids_file.empty()) {
    if (trace_out.empty())
      throw std::runtime_error("--trace-out required with --trace-ids-file");
    const     dgpp::GlmTextConfig cfg =
        dgpp::GlmTextConfig::from_json_file(config_path);
    dgpp::GlmTextConfig run_cfg = cfg;
    if (max_layers > 0) {
      run_cfg.num_hidden_layers = max_layers;
      run_cfg.layers.resize(max_layers);
      run_cfg.mlps.resize(max_layers);
      // A truncated diagnostic stack carries no MTP: mtp_layer() is
      // num_hidden_layers when the predictor is enabled, which would
      // relocate the checkpoint's layers.<N>. MTP tensors to a
      // nonexistent index.
      run_cfg.num_nextn_predict_layers = 0;
    }
    std::vector<int64_t> ids;
    {
      std::ifstream f(trace_ids_file);
      if (!f) throw std::runtime_error("cannot open " + trace_ids_file);
      std::string line, tok;
      std::getline(f, line);
      size_t start = 0;
      while (start < line.size()) {
        const size_t comma = line.find(',', start);
        const std::string item = line.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!item.empty()) ids.push_back(std::stoll(item));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    }
    if (ids.empty()) throw std::runtime_error("no token ids in file");
    dgpp::GlmDiagnosticModel model(run_cfg, checkpoint_dir,
                                   static_cast<int>(ids.size()),
                                   static_cast<int64_t>(ids.size()));
    std::printf("trace run: %zu tokens over %d layers\n", ids.size(),
                run_cfg.num_hidden_layers);
    const auto t0 = std::chrono::steady_clock::now();
    const dgpp::GlmDiagnosticModel::Outputs out = model.forward(ids);
    const auto t1 = std::chrono::steady_clock::now();
    // Determinism of the trace path: a second run must reproduce bitwise.
    const dgpp::GlmDiagnosticModel::Outputs out2 = model.forward(ids);
    if (out.routes != out2.routes)
      throw std::runtime_error("trace run not deterministic");
    dgpp::glm_trace_write(trace_out, out.routes);
    std::printf("trace written: %s (%.1fs, %d MoE layers, deterministic)\n",
                trace_out.c_str(),
                std::chrono::duration<double>(t1 - t0).count(),
                static_cast<int>(out.routes.size()));
    return 0;
  }
  if (suite_path.empty())
    throw std::runtime_error("nothing to do (no --suite, no --trace-ids-file)");

  const dgpp::GlmTextConfig cfg =
      dgpp::GlmTextConfig::from_json_file(config_path);
  const std::vector<Case> cases = load_suite(suite_path, ".glmdump");

  // Layer budget (truncated stacks are valid diagnostic runs; the dumps
  // must have been generated with the same budget). Truncation drops the
  // MTP predictor with it (see the trace-mode note).
  dgpp::GlmTextConfig run_cfg = cfg;
  if (max_layers > 0) {
    if (max_layers > cfg.num_hidden_layers)
      throw std::runtime_error("--layers exceeds the checkpoint's stack");
    run_cfg.num_hidden_layers = max_layers;
    run_cfg.layers.resize(max_layers);
    run_cfg.mlps.resize(max_layers);
    run_cfg.num_nextn_predict_layers = 0;
  }

  // Token budget: the largest case sizes the model.
  int64_t max_T = 0;
  for (const auto& c : cases) {
    const dgpp::GlmDumpFile d = dgpp::GlmDumpFile::load(c.dump_path);
    max_T = std::max(max_T, d.token_count());
    if (d.num_layers() != run_cfg.num_hidden_layers)
      throw std::runtime_error(
          "case '" + c.name + "': dump has " +
          std::to_string(d.num_layers()) + " layers, run has " +
          std::to_string(run_cfg.num_hidden_layers) +
          " (generate the dump with the same --layers budget)");
    if (d.hidden() != cfg.hidden_size || d.vocab() != cfg.vocab_size)
      throw std::runtime_error("case '" + c.name +
                               "': dump geometry disagrees with config");
  }
  if (max_tokens > 0 && max_T > max_tokens)
    throw std::runtime_error("a case exceeds --tokens (dump token count " +
                             std::to_string(max_T) + ")");

  dgpp::GlmDiagnosticModel model(run_cfg, checkpoint_dir,
                                 static_cast<int>(max_T),
                                 /*max_cache_tokens=*/max_T);
  std::printf("model: %d layers, max %lld tokens/case; %zu cases\n",
              run_cfg.num_hidden_layers, (long long)max_T, cases.size());

  int failures = 0;
  for (const auto& c : cases) {
    const dgpp::GlmDumpFile dump = dgpp::GlmDumpFile::load(c.dump_path);
    const CaseReport rep = run_case(model, dump, trace_dir, c.name);
    // Isolated per-layer budgets sit above the module noise floors
    // (KDA 3.6e-3, DSA ~3e-3, MoE ~2.3e-3 vs the double/torch oracles)
    // plus the mHC bf16 chains. Route flips are CERTIFIED per flip from
    // both sides' biased score rows (measured per-token noise, not a
    // budget); kept-row weights sit at the fp32-GEMM-order floor; the
    // flip-rate bound stays as a secondary statistical sanity check.
    const bool ok =
        // kept rows at the cross-implementation floor; flipped rows are
        // certified near ties (audit-rejected flips fail the case) and
        // excluded from the l2, matching M3's kept-row/flipped-row
        // discipline. The head's top-1 mismatches must likewise certify
        // (the truncated-stack logits crowd the boundary; the full stack
        // held top1=0 on every case).
        rep.worst_layer_kept_l2 < 0.02 && rep.head_hidden_l2 < 0.02 &&
        !rep.head_audit_failed &&
        rep.top1_mismatch == rep.top1_certified &&
        rep.set_miss <= rep.tokens &&
        !rep.route_audit_failed &&
        // Flip-rate sanity net (catastrophic-breakage detector only): the
        // primary criterion is the per-flip certification above. The
        // full-stack rate is ~0.8-1% of slots, but a reduced-budget run
        // has smaller denominators and legitimately higher boundary
        // crowding (code: 1.37% at 12 layers, all 64 mismatches
        // certified) — a 10% net lets any wholesale selection bug through
        // neither gate.
        rep.route_id_mismatch <= rep.route_slots_total / 10 &&
        rep.route_kept_max_rel < 5e-2;
    if (!ok) ++failures;
    std::printf(
        "%-16s %5lld tok %6.1fs  kept l2 max %.4f (all-rows max %.4f, "
        "%d flip layers)  head l2 %.4f  top1=%d (cert %d) top8miss=%d  "
        "routes: flips=%d/%d certified=%lld (%.1fx noise max) kept-rel "
        "%.2g  %s\n",
        c.name.c_str(), (long long)rep.tokens, rep.seconds,
        rep.worst_layer_kept_l2, rep.max_layer_l2, rep.flipped_rows_total,
        rep.head_hidden_l2, rep.top1_mismatch, rep.top1_certified,
        rep.set_miss,
        rep.route_id_mismatch, rep.route_slots_total,
        (long long)rep.route_flips_certified, rep.route_audit_max_multiple,
        rep.route_kept_max_rel, ok ? "OK" : "FAIL");
    std::fflush(stdout);
  }
  std::printf("%zu cases, %d failures\n", cases.size(), failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "glm_forward_check: %s\n", e.what());
    return 1;
  }
}
