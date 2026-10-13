#pragma once
// The Qwen3-ASR language model's shape (docs/audio.md): a plain dense
// Qwen3 - GQA with per-head q/k RMSNorm after the projection, full-dim
// rotate_half RoPE, SwiGLU - parsed from the checkpoint's
// thinker_config.text_config (the same struct fills the GLM-4.7 layer
// objects' geometry: those kernels take any heads * 128 = projection
// width with no rotary truncation).
//
// The shapes are free where the kernels are parameterized (layers, hidden,
// heads, intermediate, vocab) and refused where they are not: 128-wide
// heads, full rotary, no attention bias, no quantization, no MoE, no MTP.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm4/config.hpp"  // Glm4TextConfig as the layer objects' geometry

namespace dgpp {
struct QwenAsrLmConfig {
  static constexpr int kHeadDim = 128;  // the attention kernels' head width
  int hidden_size = 2048;
  int vocab_size = 151936;
  int num_hidden_layers = 28;
  int num_attention_heads = 16;
  int num_key_value_heads = 8;
  int intermediate_size = 6144;
  int max_position_embeddings = 65536;
  float rms_norm_eps = 1e-6f;
  double rope_theta = 1000000.0;
  bool tie_word_embeddings = true;
  std::vector<int64_t> eos_token_ids;

  int q_proj_width() const { return num_attention_heads * kHeadDim; }
  int kv_proj_width() const { return num_key_value_heads * kHeadDim; }
  // The LM bytes the loader's table streams (BF16 verbatim, lm_head an
  // alias of the embedding; the audio tower is not part of this count).
  size_t weight_bytes() const {
    const size_t H = hidden_size, V = vocab_size;
    const size_t q = static_cast<size_t>(q_proj_width()) * H;
    const size_t kv = static_cast<size_t>(kv_proj_width()) * H;
    return sizeof(uint16_t) *
           (V * H + 2 * H +
            static_cast<size_t>(num_hidden_layers) *
                (q * 2 + kv * 4 + H * q + 2 * H +
                 2 * static_cast<size_t>(intermediate_size) * H * 2 + 2 * H * kHeadDim));
  }

  // The geometry the shared GLM-4.7 attention/pool machinery consumes.
  Glm4TextConfig glm_geometry() const {
    Glm4TextConfig g;
    g.hidden_size = hidden_size;
    g.vocab_size = vocab_size;
    g.num_hidden_layers = num_hidden_layers;
    g.rms_norm_eps = rms_norm_eps;
    g.tie_word_embeddings = tie_word_embeddings;
    g.hidden_act = "silu";
    g.max_position_embeddings = max_position_embeddings;
    g.first_k_dense_replace = num_hidden_layers;  // every layer is dense
    g.eos_token_ids = eos_token_ids;
    g.num_attention_heads = num_attention_heads;
    g.num_key_value_heads = num_key_value_heads;
    g.head_dim = kHeadDim;
    g.attention_bias = false;
    g.use_qk_norm = true;
    g.rotary_dim = kHeadDim;  // full-width, half-split pairs: plain Qwen3 rope
    g.rope_theta = rope_theta;
    g.intermediate_size = intermediate_size;
    g.moe_intermediate_size = 0;
    g.n_routed_experts = 0;
    g.n_shared_experts = 0;
    g.num_experts_per_tok = 1;
    g.num_nextn_predict_layers = 0;
    g.kv_scales_present = false;
    return g;
  }

  static QwenAsrLmConfig parse(const minijson::Value& root) {
    const minijson::Value* holder = root.find("thinker_config");
    if (!holder || !holder->is_object()) holder = &root;
    const auto* node = holder->find("text_config");
    if (!node || !node->is_object())
      throw std::invalid_argument("Qwen ASR: config carries no text_config object");
    const auto& v = *node;
    const auto* arch = v.find("model_type");
    if (arch && arch->is_string() && arch->as_string() != "qwen3")
      throw std::invalid_argument("Qwen ASR: text_config.model_type must be qwen3");
    QwenAsrLmConfig c;
    const auto num = [&](const char* key, long long want, bool strict) {
      const auto* f = v.find(key);
      if (!f || f->is_null()) {
        if (strict)
          throw std::invalid_argument(std::string("Qwen text_config.") + key + ": absent");
        return want;
      }
      if (!f->is_number() || !std::isfinite(f->as_double()) ||
          f->as_double() != std::floor(f->as_double()))
        throw std::invalid_argument(std::string("Qwen text_config.") + key + ": not an integer");
      const long long got = f->as_int();
      if (strict && got != want)
        throw std::invalid_argument(std::string("Qwen text_config.") + key + ": expected " +
                                    std::to_string(want));
      return got;
    };
    c.hidden_size = static_cast<int>(num("hidden_size", 2048, false));
    c.vocab_size = static_cast<int>(num("vocab_size", 151936, false));
    c.num_hidden_layers = static_cast<int>(num("num_hidden_layers", 28, false));
    c.num_attention_heads = static_cast<int>(num("num_attention_heads", 16, false));
    c.num_key_value_heads = static_cast<int>(num("num_key_value_heads", 8, false));
    c.intermediate_size = static_cast<int>(num("intermediate_size", 6144, false));
    c.max_position_embeddings = static_cast<int>(num("max_position_embeddings", 65536, false));
    if (const auto* f = v.find("head_dim"); f && f->is_number() && f->as_int() != kHeadDim)
      throw std::invalid_argument("Qwen text_config.head_dim: the kernels are 128-wide");
    if (const auto* f = v.find("rms_norm_eps"); f && f->is_number()) c.rms_norm_eps =
        static_cast<float>(f->as_double());
    if (const auto* f = v.find("rope_theta"); f && f->is_number()) c.rope_theta = f->as_double();
    if (const auto* f = v.find("tie_word_embeddings"); f && f->is_bool())
      c.tie_word_embeddings = f->as_bool();
    const auto* act = v.find("hidden_act");
    if (act && act->is_string() && act->as_string() != "silu")
      throw std::invalid_argument("Qwen text_config.hidden_act: silu only");
    if (const auto* rs = v.find("rope_scaling"); rs && rs->is_object()) {
      const auto* kind = rs->find("rope_type");
      if (kind && kind->is_string() && kind->as_string() != "default" &&
          kind->as_string() != "linear")
        throw std::invalid_argument("Qwen text_config.rope_scaling.rope_type: default only "
                                    "(YaRN is a launcher flag, not a family)");
    }
    for (const char* forbidden : {"attention_bias", "num_experts", "n_routed_experts",
                                  "num_experts_per_tok", "sliding_window",
                                  "intermediate_mlp_size"}) {
      if (const auto* f = v.find(forbidden); f && !f->is_null()) {
        const bool zero =
            (f->is_number() && f->as_int() == 0) || (f->is_bool() && !f->as_bool());
        if (!zero)
          throw std::invalid_argument(std::string("Qwen text_config.") + forbidden +
                                      ": this family is plain dense attention");
      }
    }
    if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 || c.vocab_size <= 0 ||
        c.intermediate_size <= 0)
      throw std::invalid_argument("Qwen ASR: text_config carries a non-positive shape");
    if (c.num_attention_heads % c.num_key_value_heads != 0 || c.num_key_value_heads <= 0 ||
        c.num_attention_heads <= 0)
      throw std::invalid_argument("Qwen ASR: the head counts must divide");
    if (const auto* ids = v.find("eos_token_id"); ids && ids->is_array())
      for (const auto& e : ids->items())
        if (e.is_number()) c.eos_token_ids.push_back(e.as_int());
    if (const auto* one = v.find("eos_token_id"); one && one->is_number())
      c.eos_token_ids.push_back(one->as_int());
    if (const auto* gen = root.find("eos_token_id"); gen && gen->is_number())
      c.eos_token_ids.push_back(gen->as_int());
    if (c.eos_token_ids.empty()) c.eos_token_ids.push_back(151645);  //  endoftext
    std::vector<int64_t> in_range;
    for (const auto id : c.eos_token_ids)
      if (id >= 0 && id < c.vocab_size) in_range.push_back(id);
    if (in_range.empty()) in_range.push_back(151645);
    c.eos_token_ids = in_range;
    return c;
  }

  static QwenAsrLmConfig from_json_file(const std::string& path);
};
}  // namespace dgpp
