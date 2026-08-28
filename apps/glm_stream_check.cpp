// glm_stream_check: deployment evidence for the M4 streaming loader.
// Loads globals plus every layer of a real GLM-5.3 checkpoint one at a
// time, reconciling the bytes actually placed against the sizing formula
// (load_layer throws on drift, so surviving = reconciled) and reporting
// per-layer bytes and wall time. Correctness of the streamed bytes
// themselves is covered by glm_loader_test on a synthetic checkpoint; this
// run proves the real thing fits the same contract.
//
//   glm_stream_check --config <dir>/config.json --checkpoint-dir <dir>
//       [--layer N]  (default: every layer + globals)
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

#include "models/glm_config.hpp"
#include "models/glm_loader.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir;
  std::vector<int> only_layers;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for argument");
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--layer") only_layers.push_back(std::stoi(std::string(next())));
    else throw std::runtime_error("unknown argument");
  }
  if (config_path.empty() || checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: glm_stream_check --config CONFIG --checkpoint-dir DIR "
        "[--layer N ...]");

  dgpp::GlmTextConfig cfg = dgpp::GlmTextConfig::from_json_file(config_path);
  std::printf("config: %d layers (%d KDA + %d DSA), vocab %d, mtp %s\n",
              cfg.num_hidden_layers, cfg.num_kda_layers(),
              cfg.num_dsa_layers(), cfg.vocab_size,
              cfg.mtp_layer() >= 0 ? "present" : "absent");

  dgpp::GlmLayerStream stream(cfg, checkpoint_dir);
  std::printf("binding validated; layer bump capacity %.2f GiB\n",
              static_cast<double>(stream.layer_capacity()) / 1073741824.0);

  const dgpp::GlmGlobalsResident& globals = stream.load_globals();
  std::printf("globals: %.3f GiB (embed+lm_head+final norm)\n",
              static_cast<double>(globals.bytes) / 1073741824.0);

  const int max_layer =
      cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  double total_seconds = 0;
  size_t peak_layer_bytes = 0;
  for (int layer = 0; layer < max_layer; ++layer) {
    if (!only_layers.empty() &&
        std::find(only_layers.begin(), only_layers.end(), layer) ==
            only_layers.end())
      continue;
    const auto t0 = std::chrono::steady_clock::now();
    const dgpp::GlmLayerResident& r = stream.load_layer(layer);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_seconds += ms / 1000.0;
    if (r.bytes > peak_layer_bytes) peak_layer_bytes = r.bytes;
    const char* kind = r.kind == dgpp::GlmLayerKind::Kda ? "KDA" : "DSA";
    const char* mlp = layer < cfg.num_hidden_layers &&
                              cfg.mlps[layer] == dgpp::GlmMlpKind::Dense
                          ? "dense"
                          : "moe";
    std::printf("layer %2d [%s/%s%s]: %10zu B (%6.1f MiB) in %7.1f ms\n",
                layer, kind, mlp, layer == cfg.mtp_layer() ? "/mtp" : "",
                r.bytes, static_cast<double>(r.bytes) / 1048576.0, ms);
  }
  std::printf("peak layer %zu B; streaming total %.2f s; formula reconciled "
              "on every load\n",
              peak_layer_bytes, total_seconds);
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
