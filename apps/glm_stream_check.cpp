// glm_stream_check: deployment evidence for the streaming loader AND the
// resident serving mode (the production residency contract, DESIGN §3).
// Loads globals plus every layer of a real GLM-5.3 checkpoint, reconciling
// the bytes actually placed against the sizing formula (load_layer throws
// on drift, so surviving = reconciled) and reporting per-layer bytes and
// wall time. Correctness of the streamed bytes themselves is covered by
// glm_loader_test on a synthetic checkpoint (bitwise) and glm_shard_parity
// at real dims; this run proves the real thing fits the same contracts.
//
// --resident materializes every layer ONCE and keeps it (M6 d5, the
// production residency contract). The second pass re-loads every layer
// and proves the contract: identical views, ZERO storage reads
// (source_bytes_read cannot grow), per-layer wall time collapsed to a
// cache hit. The formula total is printed BEFORE the first allocation —
// only worlds whose resident_bytes() fits can run this (real dims:
// --world 4 is the deployment target, ~79 GiB/rank).
//
//   glm_stream_check --model ORG/NAME | (--config <dir>/config.json
//                                       --checkpoint-dir <dir>)
//       [--layer N]  (default: every layer + globals)
//       [--resident] [--rank R] [--world W]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "loaders/hf_cache.hpp"
#include "models/glm/config.hpp"
#include "models/glm/loader.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id;
  std::vector<int> only_layers;
  bool resident = false;
  int rank = 0, world = 1;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for argument");
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--layer") only_layers.push_back(std::stoi(std::string(next())));
    else if (a == "--resident") resident = true;
    else if (a == "--rank") rank = std::stoi(std::string(next()));
    else if (a == "--world") world = std::stoi(std::string(next()));
    else throw std::runtime_error("unknown argument");
  }
  // --model resolves the canonical HF hub cache in $HOME (the deployment
  // location); --checkpoint-dir stays for fixtures and staged dirs.
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty())
      throw std::runtime_error("--model and --checkpoint-dir are mutually "
                               "exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty())
      throw std::runtime_error("--model " + model_id + ": " + err);
    checkpoint_dir = snapshot;
    if (config_path.empty()) config_path = snapshot + "/config.json";
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (config_path.empty() || checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: glm_stream_check --model ORG/NAME | (--config CONFIG "
        "--checkpoint-dir DIR) [--layer N ...] [--resident] [--rank R] "
        "[--world W]");

  dgpp::GlmTextConfig cfg = dgpp::GlmTextConfig::from_json_file(config_path);
  std::printf("config: %d layers (%d KDA + %d DSA), vocab %d, mtp %s\n",
              cfg.num_hidden_layers, cfg.num_kda_layers(),
              cfg.num_dsa_layers(), cfg.vocab_size,
              cfg.mtp_layer() >= 0 ? "present" : "absent");

  // The formula total BEFORE any allocation: at real dims world 1 is
  // ~306 GiB and world 2 is ~155 GiB per rank — both impossible on a
  // 128 GB GB10; world 4 (~79 GiB) is the deployment target. Printing
  // the total first makes a doomed run legible instead of a mid-load OOM.
  if (resident)
    std::printf("resident formula (rank %d/world %d): %.2f GiB\n", rank, world,
                static_cast<double>(dgpp::GlmLayerStream::resident_bytes(
                    cfg, rank, world)) /
                    1073741824.0);

  dgpp::GlmLayerStream stream(
      cfg, checkpoint_dir, rank, world,
      resident ? dgpp::GlmResidency::Resident
               : dgpp::GlmResidency::Streaming);
  std::printf("binding validated; layer bump capacity %.2f GiB\n",
              static_cast<double>(stream.layer_capacity()) / 1073741824.0);

  const dgpp::GlmGlobalsResident& globals = stream.load_globals();
  std::printf("globals: %.3f GiB (embed+lm_head+final norm)\n",
              static_cast<double>(globals.bytes) / 1073741824.0);

  const int max_layer =
      cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  auto wanted = [&](int layer) {
    return only_layers.empty() ||
           std::find(only_layers.begin(), only_layers.end(), layer) !=
               only_layers.end();
  };
  double total_seconds = 0;
  size_t resident_total = 0;
  std::vector<const void*> first_pass_ln1;  // resident proof inputs
  for (int layer = 0; layer < max_layer; ++layer) {
    if (!wanted(layer)) continue;
    const auto t0 = std::chrono::steady_clock::now();
    const dgpp::GlmLayerResident& r = stream.load_layer(layer);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_seconds += ms / 1000.0;
    if (resident) {
      resident_total += r.bytes;
      first_pass_ln1.push_back(r.ln1);
    }
    const char* kind = r.kind == dgpp::GlmLayerKind::Kda ? "KDA" : "DSA";
    const char* mlp = layer < cfg.num_hidden_layers &&
                              cfg.mlps[layer] == dgpp::GlmMlpKind::Dense
                          ? "dense"
                          : "moe";
    std::printf("layer %2d [%s/%s%s]: %10zu B (%6.1f MiB) in %7.1f ms\n",
                layer, kind, mlp, layer == cfg.mtp_layer() ? "/mtp" : "",
                r.bytes, static_cast<double>(r.bytes) / 1048576.0, ms);
  }

  if (!resident) {
    std::printf("peak layer %zu B; streaming total %.2f s; formula reconciled "
                "on every load\n",
                stream.layer_capacity(), total_seconds);
    return 0;
  }

  // The residency contract's proof pass: every layer re-served from cache
  // — the same view addresses, ZERO storage reads (source_bytes_read is
  // frozen), wall time collapsed. Storage touched here means the contract
  // is broken and this says so instead of pretending.
  const uint64_t bytes_before = stream.source_bytes_read();
  const auto p0 = std::chrono::steady_clock::now();
  size_t n = 0;
  for (int layer = 0; layer < max_layer; ++layer) {
    if (!wanted(layer)) continue;
    const dgpp::GlmLayerResident& r = stream.load_layer(layer);
    if (r.ln1 != first_pass_ln1[n++])
      throw std::runtime_error("resident re-pass: view moved (layer " +
                               std::to_string(layer) + ")");
  }
  const double re_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - p0)
                           .count();
  const uint64_t bytes_after = stream.source_bytes_read();
  if (bytes_after != bytes_before)
    throw std::runtime_error("resident re-pass re-read " +
                             std::to_string(bytes_after - bytes_before) +
                             " storage bytes — the residency contract is "
                             "broken");
  std::printf("resident total %.2f GiB (layers %.2f GiB + globals %.3f GiB); "
              "materialized in %.2f s; re-pass %.2f ms across %zu layers, "
              "%llu storage bytes read — RESIDENCY HOLDS\n",
              static_cast<double>(resident_total + globals.bytes) /
                  1073741824.0,
              static_cast<double>(resident_total) / 1073741824.0,
              static_cast<double>(globals.bytes) / 1073741824.0,
              total_seconds, re_ms, first_pass_ln1.size(),
              static_cast<unsigned long long>(bytes_after - bytes_before));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "glm_stream_check: %s\n", e.what());
    return 1;
  }
}
