// M5 deliverable 3, forward integration: the TP fabric runner. Runs the
// block-boundary forward across real nodes over the CollectiveBus and
// writes this rank's observables for offline comparison:
//   {out}.final_hidden.bf16   [T, hidden]   — must be BITWISE identical on
//   {out}.logits.bf16        [T, vocab]      every rank (the canonical
//   {out}.routes.txt         per-MoE-layer   rank-order fold's guarantee)
//   ids/weights fp32
//   {out}.oracle.*           rank 0 only: the world=1 M4-path forward
//                            (the tolerance oracle, run before the world
//                            forms so it never shares the device with a
//                            spinning collective)
// Bus collective stats print at the end (count + submit->wait p50/p99 —
// the fabric timing record).
//
// The cross-rank BITWISE check is the assertion that matters on the
// fabric; oracle comparisons (tolerance + certified route flips) are the
// CI test's job (glm_tp_test) — here the files exist so the run records
// can show them.
//
// No construction barrier: ranks are separate processes, so a spinning
// peer kernel occupies the PEER's device and cannot block this rank's
// allocation-phase device syncs (the loopback test's barrier guards the
// one-process multi-rank shape only).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm_config.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_route_audit.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Same deterministic token/state streams as the CI test — the app's
// outputs are reproducible against it.
uint64_t tp_seed_next(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

std::vector<int64_t> make_tokens(int n, int vocab, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed | 1;
  for (auto& id : t) id = static_cast<int64_t>(tp_seed_next(s) % vocab);
  return t;
}

void write_out(const std::string& path, const void* data, size_t bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + path);
  if (std::fwrite(data, 1, bytes, f) != bytes)
    throw std::runtime_error("short write to " + path);
  std::fclose(f);
}

std::string to_hex(uint64_t v) {
  static const char* kDigits = "0123456789abcdef";
  std::string s(16, '0');
  for (int i = 15; i >= 0; --i) {
    s[static_cast<size_t>(i)] = kDigits[v & 0xf];
    v >>= 4;
  }
  return s;
}

BusOptions bus_options(int rank, int world, uint16_t port,
                       const std::string& peer, int rendezvous_timeout_ms) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : peer;
  o.rendezvous_timeout_ms = rendezvous_timeout_ms;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  // Real-mesh budgets, not loopback budgets: a cold peer's first boundary
  // waits behind seconds of cold NVMe weight streaming (the CI's 5 s was
  // calibrated for the in-process fixture world and fires on healthy
  // slow peers — found by the first fabric gate run, 2026-08-30). The lane
  // watchdog arms from each POST now, so these bounds measure genuine
  // in-flight stalls only.
  o.completion_timeout_ms = 120000;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

int run(const GlmTextConfig& cfg, const std::string& ckpt, int world,
        int rank, uint16_t port, const std::string& peer, int tokens,
        const std::string& out_prefix, int rendezvous_timeout_ms) {
  const std::vector<int64_t> ids = make_tokens(tokens, cfg.vocab_size,
                                               20260829);
  const int64_t cache = 128;

  // Rank 0 runs the world=1 oracle FIRST (separate model, no bus): its
  // files are the comparison targets, and the forward never shares the
  // device with an in-flight collective.
  GlmDiagnosticModel::Outputs oracle_out;  // rank 0: the inline verdict's
                                           // reference (kept in memory)
  if (rank == 0) {
    GlmDiagnosticModel oracle(cfg, ckpt, tokens, cache);
    oracle_out = oracle.forward(ids);
    write_out(out_prefix + ".oracle.final_hidden.bf16",
              oracle_out.final_hidden_bits.data(),
              oracle_out.final_hidden_bits.size() * 2);
    write_out(out_prefix + ".oracle.logits.bf16",
              oracle_out.logits_bits.data(),
              oracle_out.logits_bits.size() * 2);
    DGPP_LOG_INFO("oracle forward written ({} tokens)", tokens);
  }

  CollectiveBus bus(
      bus_options(rank, world, port, peer, rendezvous_timeout_ms));
  std::string err;
  if (!bus.start(&err)) {
    DGPP_LOG_ERROR("rank {}: bus start failed: {}", rank, err);
    return 1;
  }

  dgpp::GlmBusBoundaryReducer reducer(bus);
  GlmDiagnosticModel model(cfg, ckpt, tokens, cache, &reducer, rank, world);

  // Boot digest (M5 d4): every rank folds its replicated weights at
  // construction; this file is the run record's evidence. The offline
  // comparison asserts equality across ranks — a mismatch pinpoints the
  // diverging layer by index (the loopback test asserts the same thing
  // in CI).
  {
    const dgpp::GlmReplicatedDigest& d = model.boot_digest();
    std::string txt =
        "replicated tensors " + std::to_string(d.tensors) + " bytes " +
        std::to_string(d.bytes) + " globals 0x" +
        to_hex(d.globals) + "\n";
    for (size_t l = 0; l < d.layer.size(); ++l)
      txt += "layer " + std::to_string(l) + " 0x" +
             to_hex(d.layer[l]) + "\n";
    write_out(out_prefix + ".digest.txt", txt.data(), txt.size());
  }

  const auto t0 = std::chrono::steady_clock::now();
  const auto out = model.forward(ids);
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

  write_out(out_prefix + ".final_hidden.bf16", out.final_hidden_bits.data(),
            out.final_hidden_bits.size() * 2);
  write_out(out_prefix + ".logits.bf16", out.logits_bits.data(),
            out.logits_bits.size() * 2);
  {
    std::string routes;
    for (size_t l = 0; l < out.routes.size(); ++l) {
      const auto& r = out.routes[l];
      routes += "layer " + std::to_string(r.layer_idx) + " top_k " +
                std::to_string(r.top_k) + "\n";
      for (size_t t = 0; t < r.tokens; ++t) {
        for (uint32_t k = 0; k < r.top_k; ++k) {
          if (k) routes += " ";
          routes += std::to_string(r.ids[t * r.top_k + k]) + ":" +
                    std::to_string(r.weights[t * r.top_k + k]);
        }
        routes += "\n";
      }
    }
    write_out(out_prefix + ".routes.txt", routes.data(), routes.size());
  }

  const auto stats = bus.stats();
  // Both classes: latency one-shots and bulk flights record into separate
  // buckets (bulk p50/p99 is the deferred-ack flow-control bubble made
  // visible) — printing only the latency bucket reported "0 collectives"
  // on bulk-routed forwards.
  const auto summarize = [](const std::vector<double>& us) {
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double f) {
      return sorted.empty()
                 ? 0.0
                 : sorted[std::min(sorted.size() - 1,
                                   static_cast<size_t>(f * sorted.size()))];
    };
    return std::make_pair(sorted.size(), std::make_pair(pct(0.5), pct(0.99)));
  };
  const auto [lat_n, lat_tails] = summarize(stats.latency.latency_us);
  const auto [bulk_n, bulk_tails] = summarize(stats.bulk.latency_us);
  DGPP_LOG_INFO(
      "TP rank {} world {}: forward {:.1f}ms, lat {} (p50={:.1f}us "
      "p99={:.1f}us), bulk {} (p50={:.1f}us p99={:.1f}us), source {} MB",
      rank, world, ms, lat_n, lat_tails.first, lat_tails.second, bulk_n,
      bulk_tails.first, bulk_tails.second,
      model.source_bytes_read() >> 20);

  // THE INLINE VERDICT (rank 0): the parity tier's audit discipline —
  // the same code the loopback test asserts — run against the in-memory
  // oracle ON THE FABRIC. The 0.57 hunt's lesson: a gate failure must
  // self-diagnose on the night it happens, not become an offline
  // forensics project. Free-run l2 is REPORTED (the designated
  // compounding surface); top-1 near ties and every token's FIRST route
  // divergence are CERTIFIED — an uncertified divergence fails the run
  // with its diagnosis in this log.
  if (rank == 0) {
    try {
      const double rel = dgpp::glm_route::l2_rel(
          out.final_hidden_bits, oracle_out.final_hidden_bits);
      const double rms = dgpp::glm_route::l2_bf16(
          out.final_hidden_bits, oracle_out.final_hidden_bits);
      const dgpp::glm_route::Top1AuditSummary top1 =
          dgpp::glm_route::audit_top1_near_ties(
              out.logits_bits.data(), oracle_out.logits_bits.data(),
              cfg.vocab_size, ids.size());
      dgpp::glm_route::certify_top1_near_ties(top1, ids.size(), "fabric");
      DGPP_LOG_INFO(
          "fabric verdict: free-run final_hidden rel_l2={:.4f} "
          "(rms {:.4f}) — the reported compounding surface",
          rel, rms);
      dgpp::glm_route::audit_routes_cascade(
          out.routes, out.route_biased, oracle_out.routes,
          oracle_out.route_biased, cfg.moe_config().top_k,
          cfg.moe_config().n_experts, "fabric free-run");
      DGPP_LOG_INFO("fabric verdict: rank-0 inline verdict PASS");
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("fabric verdict FAILED: {}", e.what());
      bus.stop();
      return 1;
    }
  }
  bus.stop();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  std::string config_path, ckpt, peer, out_prefix = "glm_tp_rank";
  std::string model_id;
  int world = 2, rank = 0, tokens = 21, rendezvous_timeout_ms = 120000;
  uint16_t port = 29960;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--model") model_id = next();
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--tokens") tokens = std::stoi(next());
    else if (a == "--rendezvous-timeout-ms")
      rendezvous_timeout_ms = std::stoi(next());
    else if (a == "--out") out_prefix = next();
    else {
      std::fprintf(stderr,
                    "usage: glm_tp_check --model ORG/NAME | --checkpoint-dir "
                    "DIR [--world N --rank R --peer HOST --port N] "
                    "[--tokens N] [--out PREFIX]\n");
      return rank == 0 && a == "--help" ? 0 : 1;
    }
  }
  // --model resolves the canonical HF hub cache in $HOME — the deployment
  // location every node carries; --checkpoint-dir stays for fixtures and
  // staged dirs.
  if (!model_id.empty()) {
    if (!ckpt.empty()) {
      DGPP_LOG_ERROR("--model and --checkpoint-dir are mutually exclusive");
      return 1;
    }
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) {
      DGPP_LOG_ERROR("--model {}: {}", model_id, err);
      return 1;
    }
    ckpt = snapshot;
    if (config_path.empty()) config_path = snapshot + "/config.json";
    DGPP_LOG_INFO("model {} -> {}", model_id, snapshot);
  }
  if (config_path.empty() || ckpt.empty()) {
    std::fprintf(stderr,
                 "usage: glm_tp_check --model ORG/NAME | --checkpoint-dir DIR "
                 "[--world N --rank R --peer HOST --port N] [--tokens N] "
                 "[--out PREFIX]\n");
    return 1;
  }

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    DGPP_LOG_ERROR("no CUDA device visible");
    return 1;
  }
  const GlmTextConfig cfg =
      GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
  try {
    return run(cfg, ckpt, world, rank, port, peer, tokens, out_prefix,
               rendezvous_timeout_ms);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {}: {}", rank, e.what());
    return 1;
  }
}
