#pragma once
// Synthetic mini-checkpoint writer for the GLM-4.7 tests:
// enumerates the binding table for a small config and writes config.json
// + one safetensors shard, so fixture and table cannot disagree. Values
// are deterministic per tensor NAME (glm_rng's scheme), in magnitudes that
// keep the forward's nonlinearities informative; the NVFP4 triples carry
// random e2m1 nibbles, e4m3 scales without the NaN codes and positive
// per-tensor scales; the draft layer's experts are BF16 as in the release.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/glm4/binding.hpp"
#include "models/glm4/config.hpp"

namespace glm4fx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::Glm4ExpectedTensor;
using dgpp::Glm4TensorRole;
using dgpp::Glm4TextConfig;
using dgpp::Glm4WeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 128-wide heads at the smallest counts that keep worlds
// 1, 2 and 4 legal (4 query heads, 2 kv heads: the kv heads pair up at
// world 4), one dense layer then three MoE layers, a draft layer, 8 experts
// of 128 (a 32-wide slice at world 4), hidden 256, vocab 512.
inline const char* tiny_config_json(int layers = 4, int dense = 1, bool mtp = true) {
  static std::string s;
  std::string ignore = "\"lm_head\"";
  for (int l = 0; l < layers; ++l) ignore += ", \"model.layers." + std::to_string(l) + ".self_attn*\"";
  if (mtp) ignore += ", \"model.layers." + std::to_string(layers) + "*\"";
  s = std::string(R"json({
  "architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe",
  "attention_bias": true, "eos_token_id": [1, 2], "pad_token_id": 1,
  "first_k_dense_replace": )json") + std::to_string(dense) + R"json(,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 256,
  "intermediate_size": 256, "max_position_embeddings": 4096,
  "moe_intermediate_size": 128, "n_group": 1, "n_routed_experts": 8, "n_shared_experts": 1,
  "norm_topk_prob": true, "num_attention_heads": 4, "num_experts_per_tok": 2,
  "num_hidden_layers": )json" + std::to_string(layers) + R"json(, "num_key_value_heads": 2,
  "num_nextn_predict_layers": )json" + (mtp ? "1" : "0") + R"json(,
  "partial_rotary_factor": 0.5, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 1000000,
  "routed_scaling_factor": 2.5, "tie_word_embeddings": false, "topk_group": 1,
  "use_qk_norm": true, "vocab_size": 512,
  "quantization_config": {
    "config_groups": {"group_0": {
      "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": [)json" + ignore + R"json(],
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.41.0"},
    "quant_method": "modelopt"
  }
})json";
  return s.c_str();
}

inline Glm4TextConfig tiny_config(int layers = 4, int dense = 1, bool mtp = true) {
  const auto t = dgpp::minijson::parse(tiny_config_json(layers, dense, mtp));
  return Glm4TextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

inline std::vector<uint8_t> tensor_bytes(const Glm4TextConfig& cfg, const Glm4ExpectedTensor& e) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;   // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_router_bias = has(name, "e_score_correction_bias");
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case Glm4TensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // two random e2m1 codes
        continue;
      case Glm4TensorRole::Fp4Scale: {
        // e4m3 scales in [1, 64): no NaN codes, magnitudes the payload's
        // ±6 range spreads over (dequant ~ code x scale x global).
        const float s = 1.0f + 63.0f * (0.5f * (rng.unit() + 1.0f));
        out[i] = float_to_fp8_e4m3_bits(s);
        continue;
      }
      case Glm4TensorRole::Fp4Global:
        // weight_scale_2 ~ 1e-4..2e-4: with scales in [1, 64) and codes up
        // to 6 the dequantized weights sit at ~0.01 (max ~0.08), the
        // magnitude of the release's projections — a 20x larger global
        // made the residual explode (rms ~2000 by layer 3) and turned the
        // TP folds' bf16 rounding into hard element disagreements.
        v = 1.0e-4f + 1.0e-4f * (0.5f * (rng.unit() + 1.0f));
        break;
      case Glm4TensorRole::InputScale:
        v = 0.01f;
        break;
      case Glm4TensorRole::KvScale:
        v = 1.0f;
        break;
      case Glm4TensorRole::Bf16Expert:
        v = 0.05f * rng.normal3();
        break;
      case Glm4TensorRole::Duplicate:
      case Glm4TensorRole::Plain:
      default:
        if (is_router_bias) v = 0.02f * rng.normal3();
        else if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
        else if (is_bias) v = 0.02f * rng.normal3();
        else if (e.cls == Glm4WeightClass::Embed || e.cls == Glm4WeightClass::LmHead) v = 0.3f * rng.normal3();
        else v = 0.05f * rng.normal3();  // projections, routers
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

// The draft's embedding / head copies must be the globals' bytes.
inline std::vector<uint8_t> fixture_bytes(const Glm4TextConfig& cfg, const Glm4ExpectedTensor& e) {
  if (e.role == Glm4TensorRole::Duplicate) {
    const bool head = has(e.name, "shared_head.head");
    Glm4ExpectedTensor g = e;
    g.name = head ? "lm_head.weight" : "model.embed_tokens.weight";
    g.cls = head ? Glm4WeightClass::LmHead : Glm4WeightClass::Embed;
    g.role = Glm4TensorRole::Plain;
    return tensor_bytes(cfg, g);
  }
  return tensor_bytes(cfg, e);
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const Glm4TextConfig& cfg, const std::string& dir, const char* config_json) {
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
  const auto table = dgpp::glm4_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = fixture_bytes(cfg, e);
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
  std::printf("glm4 fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const Glm4TextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace glm4fx
