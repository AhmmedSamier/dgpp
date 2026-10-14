#pragma once
// Synthetic mini-checkpoint writer for the Llama tests: enumerates the
// binding table for a small config and writes config.json + one safetensors
// shard, so fixture and table cannot disagree. Values are deterministic per
// tensor NAME (glm_rng's scheme); norms sit near 1, the rest are small
// near-normals that keep a forward's nonlinearities informative.
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
#include "models/llama/binding.hpp"
#include "models/llama/config.hpp"

namespace llamafx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::LlamaExpectedTensor;
using dgpp::LlamaTextConfig;
using dgpp::LlamaWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: 128-wide heads at the smallest counts that keep worlds
// 1, 2 and 4 legal (4 query heads, 2 kv heads: the kv heads pair up at
// world 4), hidden 512 (heads x head_dim, as every Llama), MLP 256,
// vocab 512, 2 layers.
inline const char* tiny_config_json(int layers = 2) {
  static std::string s;
  s = std::string(R"json({
  "architectures": ["LlamaForCausalLM"], "attention_bias": false,
  "bos_token_id": 0, "eos_token_id": 2, "head_dim": 128,
  "hidden_act": "silu", "hidden_size": 512, "intermediate_size": 256,
  "max_position_embeddings": 4096, "mlp_bias": false, "model_type": "llama",
  "num_attention_heads": 4, "num_hidden_layers": )json") + std::to_string(layers) +
      R"json(, "num_key_value_heads": 2, "pad_token_id": 1,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 500000.0,
  "tie_word_embeddings": false, "vocab_size": 512
})json";
  return s.c_str();
}

inline LlamaTextConfig tiny_config(int layers = 2) {
  const auto t = dgpp::minijson::parse(tiny_config_json(layers));
  return LlamaTextConfig::parse(t.root);
}

inline std::vector<uint8_t> tensor_bytes(const LlamaTextConfig& cfg, const LlamaExpectedTensor& e) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = name.find("norm") != std::string::npos && e.shape.size() == 1;
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
    else if (e.cls == LlamaWeightClass::Embed || e.cls == LlamaWeightClass::LmHead) v = 0.3f * rng.normal3();
    else v = 0.05f * rng.normal3();  // projections
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

inline std::vector<uint8_t> fixture_bytes(const LlamaTextConfig& cfg, const LlamaExpectedTensor& e) {
  return tensor_bytes(cfg, e);
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const LlamaTextConfig& cfg, const std::string& dir, const char* config_json) {
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
  const auto table = dgpp::llama_expected_text_tensors(cfg);
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
  std::printf("llama fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const LlamaTextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace llamafx
