// The full GLM-5.3 config parser: the release's values parse, the
// indexer schedule and the packed-int shape are derived from the file's
// own rules, the architecture registry dispatches on it, and the
// unsupported shapes are refused by name.
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "glm_dsa_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_dsa/config.hpp"

namespace {

using glm_dsa_test::config_json;
using glm_dsa_test::indexer_types_json;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::GlmDsaTextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::GlmDsaTextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; }

}  // namespace

DGPP_TEST(glm_dsa_config_parses_the_release) {
  const dgpp::GlmDsaTextConfig c = parse(config_json());
  require(c.hidden_size == 6144 && c.vocab_size == 154880 && c.num_hidden_layers == 78, "shape");
  require(c.first_k_dense_replace == 3 && c.num_moe_layers() == 75, "dense/moe split");
  require(!c.is_moe_layer(2) && c.is_moe_layer(3) && c.is_moe_layer(78), "is_moe_layer");
  require(c.num_attention_heads == 64 && c.q_lora_rank == 2048 && c.kv_lora_rank == 512, "mla ranks");
  require(c.qk_nope_head_dim == 192 && c.qk_rope_head_dim == 64 && c.v_head_dim == 256 &&
              c.qk_head_dim() == 256,
          "head dims");
  require(c.rope_theta == 8e6 && c.rope_interleave && !c.attention_bias, "rope");
  require(c.index_n_heads == 32 && c.index_head_dim == 128 && c.index_topk == 2048, "indexer");
  require(c.indexer_rope_interleave && c.index_share_for_mtp_iteration, "indexer flags");
  require(c.index_topk_freq == 4 && c.index_skip_topk_offset == 3, "schedule params");
  // The schedule: 0, 1, 2, then 6, 10, ..., 74 own an indexer; the draft always does.
  require(c.owns_indexer(0) && c.owns_indexer(1) && c.owns_indexer(2), "dense layers own indexers");
  require(!c.owns_indexer(3) && !c.owns_indexer(4) && !c.owns_indexer(5) && c.owns_indexer(6), "first shared run");
  require(c.owns_indexer(10) && !c.owns_indexer(11) && c.owns_indexer(74) && !c.owns_indexer(77), "cadence");
  require(c.owns_indexer(78) && !c.owns_indexer(79) && !c.owns_indexer(-1), "the draft owns one");
  require(c.num_indexer_layers() == 21, "21 main-stack indexers, got " + std::to_string(c.num_indexer_layers()));
  require(c.intermediate_size == 12288 && c.moe_intermediate_size == 2048, "mlp");
  require(c.n_routed_experts == 256 && c.num_experts_per_tok == 8 && c.n_shared_experts == 1, "moe");
  require(c.shared_expert_inter() == 2048, "shared inter");
  require(c.routed_scaling_factor == 2.5f && c.norm_topk_prob, "router");
  require(c.mtp_layer() == 78, "mtp layer");
  require(c.eos_token_ids.size() == 3 && c.eos_token_ids[0] == 154820 && c.pad_token_id == 154820, "tokens");
  // The packed shape derived from the rules: layers [3, 78), int8 attention
  // and shared expert, int4 routed experts, group 64.
  require(c.packed_group_size == 64, "group");
  require(c.packed_layer_begin == 3 && c.packed_layer_end == 78, "packed range");
  require(c.attention_bits == 8 && c.shared_bits == 8 && c.expert_bits == 4, "widths");
  require(c.attention_bits_of(2) == 0 && c.attention_bits_of(3) == 8 && c.attention_bits_of(77) == 8 &&
              c.attention_bits_of(78) == 0,
          "attention_bits_of");
  require(c.expert_bits_of(3) == 4 && c.expert_bits_of(78) == 0 && c.shared_bits_of(50) == 8, "expert bits");
  const dgpp::GlmMoeConfig m = c.moe_config(512);
  require(m.inter == 512 && m.n_experts == 256 && m.top_k == 8 && m.n_shared_experts == 1, "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SigmoidBias && m.swiglu_limit > 1e30f, "moe_config router");
}

DGPP_TEST(glm_dsa_config_derives_the_schedule_without_the_list) {
  // Without indexer_types the freq/offset rule stands alone; a pattern
  // string overrides it; the list must agree with whichever applies.
  const dgpp::GlmDsaTextConfig c = parse(config_json("\"indexer_types\": [" + indexer_types_json() + "],", ""));
  require(c.num_indexer_layers() == 21 && c.owns_indexer(6) && !c.owns_indexer(7), "derived schedule");
  std::string pattern;
  for (int i = 0; i < 78; ++i) pattern += (i % 2 == 0) ? 'F' : 'S';
  const dgpp::GlmDsaTextConfig p = parse(config_json("\"index_topk_pattern\": null", "\"index_topk_pattern\": \"" + pattern + "\"",
                                                     indexer_types_json(78, 2, 1)));
  require(p.num_indexer_layers() == 39 && p.owns_indexer(0) && !p.owns_indexer(1), "pattern schedule");
  require(has(refusal(config_json("\"index_topk_pattern\": null", "\"index_topk_pattern\": \"" + pattern + "\"")),
              "indexer_types"),
          "list vs pattern disagreement");
  require(has(refusal(config_json("", "", indexer_types_json(78, 4, 2))), "indexer_types"), "list vs schedule");
  require(has(refusal(config_json("\"index_skip_topk_offset\": 3", "\"index_skip_topk_offset\": 0",
                                  indexer_types_json(78, 4, 0))),
              "layer 0"),
          "layer 0 shared");
}

DGPP_TEST(glm_dsa_config_refuses_unsupported_shapes) {
  require(has(refusal(config_json("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 32")), "qk_rope_head_dim"), "rope width");
  require(has(refusal(config_json("\"rope_interleave\": true", "\"rope_interleave\": false")), "rope_interleave"), "interleave");
  require(has(refusal(config_json("\"indexer_rope_interleave\": true", "\"indexer_rope_interleave\": false")),
              "indexer_rope_interleave"), "indexer interleave");
  require(has(refusal(config_json("\"index_topk\": 2048", "\"index_topk\": 2000")), "index_topk"), "topk power of two");
  require(has(refusal(config_json("\"index_head_dim\": 128", "\"index_head_dim\": 96")), "index_head_dim"), "index dim");
  require(has(refusal(config_json("\"index_n_heads\": 32", "\"index_n_heads\": 32, \"index_kpool\": 4")), "index_kpool"), "kpool");
  require(has(refusal(config_json("\"num_key_value_heads\": 64", "\"num_key_value_heads\": 8")), "num_key_value_heads"), "kv heads");
  require(has(refusal(config_json("\"n_group\": 1", "\"n_group\": 2")), "n_group"), "n_group");
  require(has(refusal(config_json("\"n_shared_experts\": 1", "\"n_shared_experts\": 2")), "n_shared_experts"), "shared experts");
  require(has(refusal(config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")), "hidden_act"), "act");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")), "tie_word_embeddings"), "tie");
  require(has(refusal(config_json("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")), "rope_type"), "yarn");
  require(has(refusal(config_json("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 2")),
              "num_nextn_predict_layers"), "mtp depth");
  require(has(refusal(config_json("\"moe_router_dtype\": \"float32\"", "\"moe_router_dtype\": \"bfloat16\"")),
              "moe_router_dtype"), "router dtype");
  require(has(refusal(config_json("\"v_head_dim\": 256", "\"v_head_dim\": 256, \"swiglu_limit\": 10.0")), "swiglu_limit"), "clamp");
  require(has(refusal(config_json("\"v_head_dim\": 256", "\"v_head_dim\": 256, \"text_config\": {}")), "text_config"), "flash layout");
  require(has(refusal(config_json("\"model_type\": \"glm_moe_dsa\"", "\"model_type\": \"glm5_next\"")), "model_type"), "model type");
  // The quantization contract.
  require(has(refusal(config_json("\"quantization_config\": {", "\"quantization_config\": null, \"x\": {")),
              "quantization_config"), "missing quantization");
  require(has(refusal(config_json("\"group_size\": 64,\n                    \"num_bits\": 8",
                                  "\"group_size\": 128,\n                    \"num_bits\": 8")),
              "group_size"), "group 128");
  require(has(refusal(config_json("\"num_bits\": 4", "\"num_bits\": 3")), "num_bits"), "3 bits");
  require(has(refusal(config_json("\"transform_config\": {}", "\"transform_config\": {\"hadamard\": {}}")),
              "transform_config"), "transform");
  require(has(refusal(config_json("\"symmetric\": true, \"type\": \"int\",\n                    \"zp_dtype\": null}\n      },\n      \"group_1\"",
                                  "\"symmetric\": false, \"type\": \"int\",\n                    \"zp_dtype\": null}\n      },\n      \"group_1\"")),
              "symmetric"), "asymmetric");
  require(has(refusal(config_json("\"zp_dtype\": null}\n      }\n    },", "\"zp_dtype\": \"int8\"}\n      }\n    },")),
              "zp_dtype"), "zero points");
  require(has(refusal(config_json("\"kv_cache_scheme\": null", "\"kv_cache_scheme\": {\"num_bits\": 8}")),
              "kv_cache_scheme"), "kv scheme");
  require(has(refusal(config_json("\"type\": \"int\",\n                    \"zp_dtype\": null}\n      },\n      \"group_1\"",
                                  "\"type\": \"float\",\n                    \"zp_dtype\": null}\n      },\n      \"group_1\"")),
              "weights.type"), "float weights");
  // A target that reaches a dense layer's attention, or a class-name
  // target, describes a layout the loader does not implement.
  {
    std::string s = config_json("(?:[3-9]|[1-6][0-9]|7[0-7])\\\\.(?:self_attn", "(?:[2-9]|[1-6][0-9]|7[0-7])\\\\.(?:self_attn");
    const std::string ig = "\"re:model\\\\.layers\\\\.[0-2]\\\\..*\"";
    require(s.find(ig) != std::string::npos, "ignore anchor");
    s.replace(s.find(ig), ig.size(), "\"re:model\\\\.layers\\\\.[0-1]\\\\..*\"");
    require(has(refusal(s), "dense layer"), "packed dense attention");
  }
  require(has(refusal(config_json("\"targets\": [\"re:model\\\\.layers\\\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\\\.mlp\\\\.experts",
                                  "\"targets\": [\"Linear\", \"re:model\\\\.layers\\\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\\\.mlp\\\\.experts")),
              "Linear"), "class target");
  // Packed attention with BF16 experts (the expert target neutered) is not a
  // layout the loader implements either.
  require(has(refusal(config_json("mlp\\\\.experts\\\\.\\\\d+\\\\.(?:gate_proj|up_proj|down_proj)$",
                                  "mlp\\\\.experts\\\\.\\\\d+\\\\.(?:nothing)$")),
              "BF16 experts"), "packed attention, plain experts");
}

DGPP_TEST(glm_dsa_architecture_registry_dispatches) {
  const auto g = dgpp::minijson::parse(R"({"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa"})");
  require(dgpp::detect_architecture(g.root) == dgpp::ModelArchitecture::GlmMoeDsa, "glm_moe_dsa");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::GlmMoeDsa)) == "glm_moe_dsa", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "glm_moe_dsa"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::GlmMoeDsa, "by model_type");
  const auto g5 = dgpp::minijson::parse(R"({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"})");
  require(dgpp::detect_architecture(g5.root) == dgpp::ModelArchitecture::Glm5, "glm5 unchanged");
  const auto g5t = dgpp::minijson::parse(R"({"model_type": "glm5_next"})");
  require(dgpp::detect_architecture(g5t.root) == dgpp::ModelArchitecture::Glm5, "glm5 by model_type");
}

DGPP_TEST(glm_dsa_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return;
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--HawkBearPig--GLM-5.3-Int4-Int8Mix-RTN-g64/snapshots";
  if (!fs::is_directory(root)) return;
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::GlmMoeDsa, "arch");
    const dgpp::GlmDsaTextConfig c = dgpp::GlmDsaTextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 78 && c.mtp_layer() == 78 && c.n_routed_experts == 256, "landed values");
    require(c.num_indexer_layers() == 21 && c.packed_layer_begin == 3 && c.packed_layer_end == 78, "landed shape");
    require(c.attention_bits == 8 && c.expert_bits == 4 && c.shared_bits == 8, "landed widths");
    return;
  }
}
