#pragma once
// Synthetic mini-checkpoint writer for the full GLM-5.3 tests: enumerates
// the binding table for a small config and writes config.json + one
// safetensors shard, so fixture and table cannot disagree. Values are
// deterministic per tensor NAME (glm_rng's scheme), in magnitudes that
// keep the forward's nonlinearities informative; the packed triples carry
// random codes, release-like bf16 group scales and exact shape records;
// the draft layer's experts are BF16 as in the release.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_dsa/binding.hpp"
#include "models/glm_dsa/config.hpp"

namespace glmdsafx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::GlmDsaExpectedTensor;
using dgpp::GlmDsaTensorRole;
using dgpp::GlmDsaTextConfig;
using dgpp::GlmDsaWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 16 heads of nope 64 + rope 64 / v 64 on latents of
// 128 (4 local heads at world 4: the split attention kernel's head-group
// floor), a 32 x 128 indexer selecting 16 tokens, one dense layer then
// four MoE layers (indexers on 0, 2 and 4: freq 2, offset 1), a draft
// layer, 8 experts of 256 (a 64-wide slice at world 4), hidden 256, vocab
// 512; layers 1-4 packed (int8 attention and shared expert, int4 experts).
inline const char* tiny_config_json(int layers = 5, int dense = 1, bool mtp = true) {
  static std::string s;
  const std::string last = std::to_string(layers - 1);
  const std::string range = "[" + std::to_string(dense) + "-" + last + "]";
  s = std::string(R"json({
  "architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa",
  "attention_bias": false, "eos_token_id": [1, 2], "pad_token_id": 1,
  "first_k_dense_replace": )json") + std::to_string(dense) + R"json(,
  "hidden_act": "silu", "hidden_size": 256, "index_head_dim": 128, "index_n_heads": 32,
  "index_share_for_mtp_iteration": true, "index_skip_topk_offset": )json" + std::to_string(dense) + R"json(,
  "index_topk": 16, "index_topk_freq": 2, "indexer_rope_interleave": true,
  "intermediate_size": 256, "kv_lora_rank": 128, "max_position_embeddings": 4096,
  "moe_intermediate_size": 256, "moe_router_dtype": "float32", "n_group": 1,
  "n_routed_experts": 8, "n_shared_experts": 1, "norm_topk_prob": true,
  "num_attention_heads": 16, "num_experts_per_tok": 2,
  "num_hidden_layers": )json" + std::to_string(layers) + R"json(, "num_key_value_heads": 16,
  "num_nextn_predict_layers": )json" + (mtp ? "1" : "0") + R"json(,
  "q_lora_rank": 128, "qk_head_dim": 128, "qk_nope_head_dim": 64, "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-05, "rope_interleave": true,
  "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
  "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": false,
  "topk_group": 1, "topk_method": "noaux_tc", "v_head_dim": 64, "vocab_size": 512,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.)json" + range + R"json(\\.(?:self_attn\\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp\\.shared_experts\\.(?:gate_proj|up_proj|down_proj))$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 8, "observer": "memoryless_minmax", "strategy": "group",
                    "symmetric": true, "type": "int", "zp_dtype": null}
      },
      "group_1": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.)json" + range + R"json(\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 4, "observer": "memoryless_minmax", "strategy": "group",
                    "symmetric": true, "type": "int", "zp_dtype": null}
      }
    },
    "format": "pack-quantized",
    "ignore": ["lm_head", "re:.*embed_tokens.*", "re:model\\.layers\\.[0-)json" + std::to_string(dense - 1) + R"json(]\\..*",
               "re:.*self_attn\\.indexer\\..*", "re:.*norm.*", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}, "version": "0.18.0"
  }
})json";
  return s.c_str();
}

inline GlmDsaTextConfig tiny_config(int layers = 5, int dense = 1, bool mtp = true) {
  const std::string text = tiny_config_json(layers, dense, mtp);
  const auto t = dgpp::minijson::parse(text);
  return GlmDsaTextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

// The bytes of one tensor. `table` resolves a shape record's packed
// sibling (the record holds [N, K]).
inline std::vector<uint8_t> tensor_bytes(const GlmDsaTextConfig& cfg, const GlmDsaExpectedTensor& e,
                                         const std::vector<GlmDsaExpectedTensor>& table) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;   // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_router_bias = has(name, "e_score_correction_bias");
  if (e.role == GlmDsaTensorRole::IntShape) {
    const std::string sibling = name.substr(0, name.size() - std::strlen("_shape")) + "_packed";
    for (const auto& t : table)
      if (t.name == sibling) {
        const int64_t rec[2] = {t.shape[0], t.shape[1] * 32 / t.bits};
        std::memcpy(out.data(), rec, 16);
        return out;
      }
    throw std::runtime_error("fixture: packed sibling missing for " + name);
  }
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case GlmDsaTensorRole::IntPacked: {
        const uint32_t word = static_cast<uint32_t>(rng.next());
        std::memcpy(&out[i * 4], &word, 4);
        continue;
      }
      case GlmDsaTensorRole::IntScale:
        // Release-like group scales: ~4e-3 for int4 codes, ~3e-4 for int8
        // (the checkpoint's expert and attention magnitudes).
        v = (e.bits == 4 ? 0.004f : 0.0003f) * std::exp2(rng.unit() * 1.0f);
        break;
      case GlmDsaTensorRole::Bf16Expert:
        v = 0.05f * rng.normal3();
        break;
      case GlmDsaTensorRole::Plain:
      default:
        if (is_router_bias) v = 0.02f * rng.normal3();
        else if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
        else if (is_bias) v = 0.02f * rng.normal3();
        else if (e.cls == GlmDsaWeightClass::Embed || e.cls == GlmDsaWeightClass::LmHead) v = 0.3f * rng.normal3();
        else v = 0.05f * rng.normal3();  // projections, routers, indexers
        break;
    }
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

inline std::vector<uint8_t> fixture_bytes(const GlmDsaTextConfig& cfg, const GlmDsaExpectedTensor& e,
                                          const std::vector<GlmDsaExpectedTensor>& table) {
  return tensor_bytes(cfg, e, table);
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const GlmDsaTextConfig& cfg, const std::string& dir, const char* config_json) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  const auto table = dgpp::glm_dsa_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = fixture_bytes(cfg, e, table);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  const fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("glm_dsa fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const GlmDsaTextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace glmdsafx
