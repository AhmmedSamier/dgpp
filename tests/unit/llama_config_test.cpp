// The Llama config parser (Zeta-2.1, docs/zeta_plan.md §1): the release's
// values parse, the architecture registry dispatches on it, and the
// unsupported shapes are refused by name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/llama/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// zed-industries/zeta-2.1 @ 9a83b32, transcribed.
const char* kZeta = R"({
  "architectures": ["LlamaForCausalLM"], "attention_bias": false,
  "attention_dropout": 0.1, "bos_token_id": 0, "eos_token_id": 2,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
  "intermediate_size": 14336, "max_position_embeddings": 32768,
  "mlp_bias": false, "model_type": "llama", "num_attention_heads": 32,
  "num_hidden_layers": 32, "num_key_value_heads": 8, "pad_token_id": 1,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 500000.0,
  "tie_word_embeddings": false, "vocab_size": 155136
})";

std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "") {
  std::string s = kZeta;
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    require(at != std::string::npos, "patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

dgpp::LlamaTextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::LlamaTextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(llama_config_parses_zeta_2_1) {
  const dgpp::LlamaTextConfig c = parse(config_json());
  require(c.hidden_size == 4096 && c.vocab_size == 155136 && c.num_hidden_layers == 32, "shape");
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 8 && c.head_dim == 128, "attention");
  require(c.q_heads_per_kv() == 4, "gqa");
  require(!c.attention_bias && c.rope_theta == 500000.0, "rope");
  require(c.intermediate_size == 14336 && c.hidden_act == "silu", "mlp");
  require(c.rms_norm_eps == 1e-6f, "norm");
  require(c.max_position_embeddings == 32768 && c.context_limit() == 32768, "context");
  require(!c.tie_word_embeddings, "untied");
  require(c.eos_token_ids.size() == 1 && c.eos_token_ids[0] == 2, "eos");
  require(c.bos_token_id == 0 && c.pad_token_id == 1, "bos/pad");
}

DGPP_TEST(llama_config_refuses_unsupported_shapes) {
  auto has = [](const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; };
  require(has(refusal(config_json("\"model_type\": \"llama\"", "\"model_type\": \"mistral\"")), "model_type"),
          "model_type");
  require(has(refusal(config_json("\"head_dim\": 128", "\"head_dim\": 64")), "head_dim"), "head_dim");
  require(has(refusal(config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act"),
          "act");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")),
               "tie_word_embeddings"),
          "tied");
  require(has(refusal(config_json("\"attention_bias\": false", "\"attention_bias\": true")),
               "attention_bias"),
          "bias");
  require(has(refusal(config_json("\"mlp_bias\": false", "\"mlp_bias\": true")), "mlp_bias"),
          "mlp bias");
  require(has(refusal(config_json("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\"}")),
               "rope_scaling"),
          "rope scaling");
  require(has(refusal(config_json("\"intermediate_size\": 14336", "\"intermediate_size\": 14337")),
               "intermediate_size"),
          "intermediate");
}

DGPP_TEST(llama_architecture_registry_dispatches) {
  const auto g = dgpp::minijson::parse(R"({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})");
  require(dgpp::detect_architecture(g.root) == dgpp::ModelArchitecture::Llama, "llama");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Llama)) == "llama", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "llama"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Llama, "by model_type");
  const auto g5 = dgpp::minijson::parse(R"({"architectures": ["Glm5ForConditionalGeneration"]})");
  require(dgpp::detect_architecture(g5.root) == dgpp::ModelArchitecture::Glm5, "glm5 unchanged");
}

DGPP_TEST(llama_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return;
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--zed-industries--zeta-2.1/snapshots";
  if (!fs::is_directory(root)) return;
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::Llama, "arch");
    const dgpp::LlamaTextConfig c = dgpp::LlamaTextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 32 && c.num_attention_heads == 32 && c.vocab_size == 155136,
            "landed values");
    return;
  }
}
