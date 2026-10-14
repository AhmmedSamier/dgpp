// The Llama expected-tensor table (Zeta-2.1, docs/zeta_plan.md §1): the
// release shape (291 BF16 tensors), the TP geometry acceptance, and — when
// the checkpoint is in the hub cache — the full binding against every
// shard's header.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/llama/binding.hpp"
#include "models/llama/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::LlamaTextConfig zeta_config() {
  const char* kJson = R"({
    "architectures": ["LlamaForCausalLM"], "attention_bias": false,
    "bos_token_id": 0, "eos_token_id": 2, "head_dim": 128,
    "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 14336,
    "max_position_embeddings": 32768, "mlp_bias": false, "model_type": "llama",
    "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8,
    "pad_token_id": 1, "rms_norm_eps": 1e-06, "rope_scaling": null,
    "rope_theta": 500000.0, "tie_word_embeddings": false, "vocab_size": 155136
  })";
  const auto t = dgpp::minijson::parse(kJson);
  return dgpp::LlamaTextConfig::parse(t.root);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--zed-industries--zeta-2.1/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") &&
        fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(llama_binding_table_has_the_release_shape) {
  const dgpp::LlamaTextConfig cfg = zeta_config();
  // One layer: 2 norms + 4 attention + 3 MLP = 9 BF16 tensors.
  const auto layer = dgpp::llama_expected_layer_tensors(cfg, 0);
  require(layer.size() == 9, "layer " + std::to_string(layer.size()));
  require(layer[0].name == "model.layers.0.input_layernorm.weight", "prefix");
  const auto all = dgpp::llama_expected_text_tensors(cfg);
  require(all.size() == 3 + 32 * 9, "table size " + std::to_string(all.size()));
  size_t nbytes = 0;
  for (const auto& e : all) nbytes += e.nbytes();
  // 8,250,462,208 BF16 params = 16,500,924,416 bytes (the index metadata
  // total_size).
  require(nbytes == 16500924416ull, "table bytes " + std::to_string(nbytes));
}

DGPP_TEST(llama_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::LlamaTextConfig cfg = zeta_config();
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::llama_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::llama_tp_validate_geometry(cfg, 0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 is refused");
}

DGPP_TEST(llama_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  namespace fs = std::filesystem;
  const dgpp::LlamaTextConfig cfg =
      dgpp::LlamaTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::LlamaTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::LlamaTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::LlamaBindReport rep = dgpp::llama_validate_text_binding(cfg, present);
  require(rep.ok(), "binding mismatch, first error: " +
                        std::string(rep.errors.empty() ? "<none>" : rep.errors[0]));
  require(rep.matched == 291, "matched " + std::to_string(rep.matched));
}
