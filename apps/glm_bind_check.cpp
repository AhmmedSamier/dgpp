// glm_bind_check: offline validation of a real GLM-5.3 checkpoint against the
// M4 expected-tensor table (M4 exit criterion: every quantized matrix binds
// to a validated scale tensor). Reads config.json and every safetensors
// header; no payload bytes are touched. Exit 0 only when the binding is
// exact.
//
//   glm_bind_check --model ORG/NAME | (--config <dir>/config.json
//                                    --checkpoint-dir <dir>) [--max-errors N]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/glm_binding.hpp"
#include "models/glm_config.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id;
  size_t max_errors = 32;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc)
        throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--max-errors") max_errors = std::stoul(std::string(next()));
    else throw std::runtime_error(std::format("unknown argument {}", a));
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
      throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    if (config_path.empty())
      config_path = snapshot + "/config.json";
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (config_path.empty() || checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: glm_bind_check --model ORG/NAME | (--config <config.json> "
        "--checkpoint-dir <dir>) [--max-errors N]");

  // 1. Config: parse and cross-validate before trusting any tensor name.
  dgpp::GlmTextConfig cfg = dgpp::GlmTextConfig::from_json_file(config_path);
  std::printf("config: %d layers (%d KDA + %d DSA), vocab %d, mhc mult %d, "
               "mtp %s\n",
               cfg.num_hidden_layers, cfg.num_kda_layers(),
               cfg.num_dsa_layers(), cfg.vocab_size, cfg.hc_mult,
               cfg.mtp_layer() >= 0 ? "present" : "absent");

  // 2. Headers: iterate every shard (sorted for deterministic reports).
  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty())
    throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::GlmTensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(
          t.name, dgpp::GlmTensorDesc{t.dtype, t.shape});
      if (!inserted)
        throw std::runtime_error(
            std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(),
               total_tensors);

  // 3. Validate the text binding against header truth.
  dgpp::GlmBindReport rep =
      dgpp::glm_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, "
              "shape %zu) | vision %zu | unexpected %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch,
              rep.shape_mismatch, rep.vision, rep.unexpected);
  std::printf("quantized matrices: %zu | scales bound: %zu | scales bad: %zu\n",
               rep.quantized_matrices, rep.scales_bound, rep.scales_bad);
  for (const auto& err : rep.errors)
      std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors)
    std::printf("  ... error list capped at %zu\n", max_errors);

  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every quantized matrix has a validated scale\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "glm_bind_check: %s\n", e.what());
    return 2;
  }
}
