#pragma once
// Synthetic mini-checkpoint writer for the MiMo-V2.6-Flash tests:
// enumerates the binding table for a small config and writes config.json
// + one safetensors shard, so fixture and table cannot disagree. Values
// are deterministic per tensor NAME (glm_rng's scheme), in magnitudes that
// keep the forward's nonlinearities informative; the fp8 matrices carry
// random e4m3 codes (no NaN codes) under block scales that put the
// dequantized weights at the release's ~0.05 rms, the MXFP4 experts random
// e2m1 nibbles under e8m0 scales of 2^-7 .. 2^-5. A few tensors of the
// unserved classes (the vision encoder, the second draft layer) are written
// too, so the loader's ignore rule is exercised by every fixture gate.
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
#include "models/mimo/binding.hpp"
#include "models/mimo/config.hpp"

namespace mimofx {

namespace fs = std::filesystem;
using dgpp::float_to_bf16_bits;
using dgpp::MimoExpectedTensor;
using dgpp::MimoTensorRole;
using dgpp::MimoTextConfig;
using dgpp::MimoWeightClass;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: the kernels' 192 / 128 heads at the smallest counts
// that keep worlds 1, 2 and 4 legal (8 query heads, 4 global kv heads = 4
// fused-projection chunks, 8 sliding-window kv heads), a global dense layer
// then SWA, SWA, global MoE layers (hybrid_layer_pattern [0, 1, 1, 0]), a
// draft layer, 8 experts of 128 (a 32-wide slice at world 4), hidden 256,
// dense intermediate 512, vocab 512, window 32 (one attention tile: a
// 40-token prompt looks past it). The global chunk (2 x 192 + 192 + 128 =
// 704 rows) is not a 128 multiple, so the padded stacking is exercised at
// worlds 1 and 2; the SWA chunk (1024) is.
inline const char* tiny_config_json(int layers = 4, bool mtp = true) {
  static std::string s;
  std::string pattern, moe, ignore;
  for (int l = 0; l < layers; ++l) {
    // GA on the first and every fourth layer after; the rest SWA.
    pattern += (l ? ", " : "") + std::string(l % 4 == 0 || l == layers - 1 ? "0" : "1");
    moe += (l ? ", " : "") + std::string(l == 0 ? "0" : "1");
    ignore += (l ? ", " : "") + std::string("\"model.layers.") + std::to_string(l) + ".self_attn.o_proj\"";
  }
  ignore += ", \"model.decoder.self_attn.o_proj\"";
  s = std::string(R"json({
  "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2",
  "add_full_attention_sink_bias": false, "add_swa_attention_sink_bias": true,
  "attention_bias": false, "attention_chunk_size": 32, "attention_projection_layout": "fused_qkv",
  "attention_value_scale": 0.707, "moe_router_dtype": "bfloat16",
  "eos_token_id": [1, 2], "pad_token_id": 1,
  "head_dim": 192, "v_head_dim": 128, "hidden_act": "silu", "hidden_size": 256,
  "hybrid_block_size": null,
  "hybrid_layer_pattern": [)json") + pattern + R"json(],
  "intermediate_size": 512, "layernorm_epsilon": 1e-06, "max_position_embeddings": 4096,
  "moe_intermediate_size": 128,
  "moe_layer_freq": [)json" + moe + R"json(],
  "n_group": 1, "n_routed_experts": 8, "n_shared_experts": null, "norm_topk_prob": true,
  "num_attention_heads": 8, "num_experts_per_tok": 2,
  "num_hidden_layers": )json" + std::to_string(layers) + R"json(, "num_key_value_heads": 4,
  "num_nextn_predict_layers": )json" + (mtp ? "3" : "0") + R"json(,
  "partial_rotary_factor": 0.334,
  "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default"},
  "rope_theta": 10000000.0, "routed_scaling_factor": null, "scoring_func": "sigmoid",
  "sliding_window": 32, "sliding_window_size": 32,
  "swa_head_dim": 192, "swa_num_attention_heads": 8, "swa_num_key_value_heads": 8,
  "swa_rope_theta": 10000.0, "swa_v_head_dim": 128,
  "tie_word_embeddings": false, "topk_group": 1, "topk_method": "noaux_tc", "vocab_size": 512,
  "quantization_config": {
    "activation_scheme": "dynamic", "fmt": "e4m3",
    "ignored_layers": [)json" + ignore + R"json(],
    "mxfp4_block_size": 32, "quant_method": "fp8", "store_dtype": "mxfp4",
    "weight_block_size": [128, 128]
  }
})json";
  return s.c_str();
}

inline MimoTextConfig tiny_config(int layers = 4, bool mtp = true) {
  const auto t = dgpp::minijson::parse(tiny_config_json(layers, mtp));
  return MimoTextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

inline std::vector<uint8_t> tensor_bytes(const MimoTextConfig& cfg, const MimoExpectedTensor& e) {
  (void)cfg;
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;   // gains near 1
  const bool is_router_bias = has(name, "e_score_correction_bias");
  const bool is_sink = has(name, "attention_sink_bias");
  for (size_t i = 0; i < n; ++i) {
    float v;
    switch (e.role) {
      case MimoTensorRole::Fp8Payload: {
        // A random e4m3 code, never a NaN (0x7F / 0xFF).
        uint8_t b = static_cast<uint8_t>(rng.next() & 0xFFu);
        if ((b & 0x7Fu) == 0x7Fu) b = static_cast<uint8_t>(b & 0xF0u);
        out[i] = b;
        continue;
      }
      case MimoTensorRole::Fp8Scale:
        // Random e4m3 codes have an rms near 100; a block scale of 4e-4 ..
        // 6e-4 puts the dequantized weights at the ~0.05 rms of the
        // release's projections.
        v = 4.0e-4f + 2.0e-4f * (0.5f * (rng.unit() + 1.0f));
        break;
      case MimoTensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);  // two random e2m1 codes
        continue;
      case MimoTensorRole::Fp4Scale:
        // e8m0 codes 120 .. 122: scales 2^-7 .. 2^-5 (an e2m1 rms of ~2.9
        // lands the experts at ~0.05).
        out[i] = static_cast<uint8_t>(120 + (rng.next() % 3));
        continue;
      case MimoTensorRole::Plain:
      default:
        if (is_router_bias) v = 0.02f * rng.normal3();
        else if (is_sink) v = 0.5f * rng.normal3();
        else if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
        else if (e.cls == MimoWeightClass::Embed || e.cls == MimoWeightClass::LmHead) v = 0.3f * rng.normal3();
        else v = 0.05f * rng.normal3();  // o_proj, eh_proj, the router
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

// The unserved tensors the fixture carries beside the table: a vision
// block's norm, a speech embedding, and the second draft layer's enorm —
// each in the release's own naming, so the ignore rule is what admits them.
inline std::vector<MimoExpectedTensor> ignored_extras(const MimoTextConfig& cfg) {
  std::vector<MimoExpectedTensor> out;
  const int64_t H = cfg.hidden_size;
  auto add = [&](const std::string& name, std::vector<int64_t> shape) {
    MimoExpectedTensor e;
    e.name = name;
    e.dtype = dgpp::DType::BF16;
    e.shape = std::move(shape);
    e.cls = MimoWeightClass::LayerNorm;
    e.role = MimoTensorRole::Plain;
    out.push_back(std::move(e));
  };
  add("visual.blocks.0.norm1.weight", {64});
  add("speech_embeddings.0.weight", {16, H});
  add("audio_encoder.projection.0.weight", {H, 64});
  if (cfg.mtp_layer() >= 0) add("model.mtp.layers.1.enorm.weight", {H});
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for `cfg`.
inline void write_fixture(const MimoTextConfig& cfg, const std::string& dir, const char* config_json) {
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
  auto table = dgpp::mimo_expected_text_tensors(cfg);
  for (auto& e : ignored_extras(cfg)) table.push_back(std::move(e));
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = tensor_bytes(cfg, e);
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
  std::printf("mimo fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const MimoTextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace mimofx
