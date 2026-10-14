#pragma once
// Zeta-2.1 (LlamaForCausalLM, model_type llama) configuration, parsed from
// the checkpoint's config.json root (2026-10-14, docs/zeta_plan.md §1).
// Same policy as the other families: every field the assembly consumes is
// parsed into a known-supported value or rejected with a message naming
// the field, at load time. The reference is transformers' modeling_llama.py.
//
// Dense BF16 scope: 32 GQA layers (full RoPE, no bias, no qk-norm), dense
// SwiGLU MLPs, untied embeddings/head. No MoE, no draft layer, no
// quantization_config, no rope_scaling.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

struct LlamaTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 4096;
  int vocab_size = 155136;
  int num_hidden_layers = 32;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 32768;
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1;
  int64_t pad_token_id = -1;

  // --- attention (GQA, no bias, no qk-norm, full RoPE) --------------------
  int num_attention_heads = 32;
  int num_key_value_heads = 8;
  int head_dim = 128;
  bool attention_bias = false;
  double rope_theta = 500000.0;

  // --- dense SwiGLU MLP ----------------------------------------------------
  int intermediate_size = 14336;

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static LlamaTextConfig parse(const minijson::Value& root);
  static LlamaTextConfig from_json_file(const std::string& path);

  int q_heads_per_kv() const { return num_attention_heads / num_key_value_heads; }
  int64_t context_limit() const { return static_cast<int64_t>(max_position_embeddings); }
};

}  // namespace dgpp
