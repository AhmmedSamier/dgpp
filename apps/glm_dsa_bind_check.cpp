// glm_dsa_bind_check: offline validation of a full GLM-5.3 checkpoint
// (HawkBearPig/GLM-5.3-Int4-Int8Mix-*-g64) against the family's
// expected-tensor table (docs/glm53_plan.md G1). Reads config.json and
// every safetensors header; no payload bytes are touched. Exit 0 only
// when the binding is exact: every expected tensor present with its dtype
// and shape, nothing unexpected.
//
//   glm_dsa_bind_check --model ORG/NAME | (--config <dir>/config.json
//                                        --checkpoint-dir <dir>) [--max-errors N]
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
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id;
  size_t max_errors = 32;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--max-errors") max_errors = std::stoul(std::string(next()));
    else throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty())
      throw std::runtime_error("--model and --checkpoint-dir are mutually exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    if (config_path.empty()) config_path = snapshot + "/config.json";
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (config_path.empty() || checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: glm_dsa_bind_check --model ORG/NAME | (--config <config.json> "
        "--checkpoint-dir <dir>) [--max-errors N]");

  const dgpp::GlmDsaTextConfig cfg = dgpp::GlmDsaTextConfig::from_json_file(config_path);
  std::printf("config: %d layers (%d dense + %d MoE, %d own an indexer), vocab %d, mtp %s; "
              "packed layers [%d, %d): attention int%d, shared int%d, experts int%d, group %d\n",
              cfg.num_hidden_layers, cfg.first_k_dense_replace, cfg.num_moe_layers(),
              cfg.num_indexer_layers(), cfg.vocab_size, cfg.mtp_layer() >= 0 ? "present" : "absent",
              cfg.packed_layer_begin, cfg.packed_layer_end, cfg.attention_bits, cfg.shared_bits,
              cfg.expert_bits, cfg.packed_group_size);

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::GlmDsaTensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::GlmDsaTensorDesc{t.dtype, t.shape});
      if (!inserted)
        throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::GlmDsaBindReport rep = dgpp::glm_dsa_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch,
              rep.unexpected);
  std::printf("packed matrices: int4 %zu | int8 %zu | draft BF16 expert matrices: %zu\n",
              rep.packed_int4_matrices, rep.packed_int8_matrices, rep.bf16_expert_matrices);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every tensor of the table is present with its dtype and shape\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "glm_dsa_bind_check: %s\n", e.what());
    return 2;
  }
}
