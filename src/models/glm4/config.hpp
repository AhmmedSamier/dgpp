#pragma once
// GLM-4.7 (Glm4MoeForCausalLM) configuration, parsed from the checkpoint's
// config.json root (2026-09-09, docs/glm47_plan.md §1.1). The same policy
// as the GLM-5.3 and Qwen parsers: every field the assembly consumes is
// parsed into a known-supported value or rejected with a message naming
// the field, at load time. The reference is transformers'
// modeling_glm4_moe.py (4.57) and vLLM's glm4_moe_mtp.py for the draft.
//
// The quantization contract (nvidia/GLM-4.7-NVFP4, modelopt 0.41): every
// Linear except the attention projections and lm_head is NVFP4 — e2m1
// codes two per byte (`weight`, U8 [N, K/2]), e4m3 block scales per 16
// (`weight_scale`, [N, K/16]) and one fp32 per-tensor scale
// (`weight_scale_2`); `input_scale` (the W4A4 activation scale) rides
// along unused, as do the FP8 KV-cache scales (`k_scale`, `v_scale`). The
// draft layer (`model.layers.<num_hidden_layers>`) is BF16 throughout and
// is requantized to NVFP4 at load (plan D1).
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct Glm4TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 5120;
  int vocab_size = 151552;
  int num_hidden_layers = 92;
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 202752;
  int first_k_dense_replace = 3;   // layers [0, first_k_dense_replace) carry the dense MLP
  std::vector<int64_t> eos_token_ids;  // config.json's; serving resolves generation stop IDs separately
  int64_t pad_token_id = -1;

  // --- attention (GQA, biased projections, per-head q/k norm, partial RoPE)
  int num_attention_heads = 96;
  int num_key_value_heads = 8;
  int head_dim = 128;
  bool attention_bias = true;
  bool use_qk_norm = true;
  int rotary_dim = 64;      // head_dim * partial_rotary_factor; half-split pairs (i, i + 32)
  double rope_theta = 1e6;

  // --- MLP / MoE -----------------------------------------------------------
  int intermediate_size = 12288;      // the dense layers
  int moe_intermediate_size = 1536;   // per routed expert (and the shared expert, x n_shared_experts)
  int n_routed_experts = 160;
  int n_shared_experts = 1;
  int num_experts_per_tok = 8;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  int n_group = 1;
  int topk_group = 1;

  // --- MTP --------------------------------------------------------------------
  int num_nextn_predict_layers = 1;  // 0 or 1

  // --- weight formats -----------------------------------------------------
  int fp4_group_size = 16;           // the NVFP4 block (the kernels implement 16)
  bool kv_scales_present = true;     // k_proj.k_scale / v_proj.v_scale tensors exist (FP8 kv scheme)

  // Parses config.json's root object (Glm4MoeForCausalLM keeps its fields
  // at the root; quantization_config is a member of it). Throws
  // std::runtime_error naming the offending field on anything unsupported.
  static Glm4TextConfig parse(const minijson::Value& root);
  static Glm4TextConfig from_json_file(const std::string& path);

  // Layer index of the draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return num_nextn_predict_layers == 1 ? num_hidden_layers : -1; }
  // Whether layer `l` (a main layer or the draft) carries the MoE.
  bool is_moe_layer(int l) const { return l >= first_k_dense_replace; }
  int num_moe_layers() const { return num_hidden_layers - first_k_dense_replace; }
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int q_heads_per_kv() const { return num_attention_heads / num_key_value_heads; }
  // The routed chain's configuration (models/glm/moe.hpp): the GLM sigmoid
  // router, the shared expert in the chain, no swiglu clamps. `local_inter`
  // is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
