#pragma once
// The full GLM-5.3 config.json (HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64,
// transcribed 2026-09-12) as the unit tests build it: the per-layer lists
// are generated, the rest is the file's text. Shared by the config and
// binding tests.
#include <stdexcept>
#include <string>

namespace glm_dsa_test {

inline const char* kHead = R"({
  "architectures": ["GlmMoeDsaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
  "dtype": "bfloat16", "eos_token_id": [154820, 154827, 154829], "ep_size": 1,
  "first_k_dense_replace": 3, "head_dim": 192, "hidden_act": "silu", "hidden_size": 6144,
  "index_head_dim": 128, "index_n_heads": 32, "index_share_for_mtp_iteration": true,
  "index_skip_topk_offset": 3, "index_topk": 2048, "index_topk_freq": 4, "index_topk_pattern": null,
  "indexer_rope_interleave": true, "indexer_types": [INDEXER_TYPES], "initializer_range": 0.02,
  "intermediate_size": 12288, "kv_lora_rank": 512, "max_position_embeddings": 1048576,
  "mlp_layer_types": [MLP_TYPES], "model_type": "glm_moe_dsa", "moe_intermediate_size": 2048,
  "moe_layer_freq": 1, "moe_router_dtype": "float32", "n_group": 1, "n_routed_experts": 256,
  "n_shared_experts": 1, "norm_topk_prob": true, "num_attention_heads": 64,
  "num_experts_per_tok": 8, "num_hidden_layers": 78, "num_key_value_heads": 64,
  "num_nextn_predict_layers": 1, "pad_token_id": 154820, "pretraining_tp": 1, "q_lora_rank": 2048,
  "qk_head_dim": 256, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "rms_norm_eps": 1e-05,
  "rope_interleave": true, "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
  "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "tie_word_embeddings": false,
  "topk_group": 1, "topk_method": "noaux_tc", "transformers_version": "5.15.0", "use_cache": true,
  "v_head_dim": 256, "vocab_size": 154880,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.(?:self_attn\\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)|mlp\\.shared_experts\\.(?:gate_proj|up_proj|down_proj))$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 8, "observer": "memoryless_minmax", "observer_kwargs": {},
                    "scale_dtype": null, "strategy": "group", "symmetric": true, "type": "int",
                    "zp_dtype": null}
      },
      "group_1": {
        "format": "pack-quantized", "input_activations": null, "output_activations": null,
        "targets": ["re:model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": null, "block_structure": null, "dynamic": false, "group_size": 64,
                    "num_bits": 4, "observer": "memoryless_minmax", "observer_kwargs": {},
                    "scale_dtype": null, "strategy": "group", "symmetric": true, "type": "int",
                    "zp_dtype": null}
      }
    },
    "format": "pack-quantized", "global_compression_ratio": null,
    "ignore": ["lm_head", "re:.*embed_tokens.*", "re:model\\.layers\\.[0-2]\\..*",
               "re:.*self_attn\\.indexer\\..*", "re:.*norm.*", "re:.*mlp\\.gate$"],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}, "version": "0.18.0"
  }
})";

// transformers' schedule: full when max(i - offset + 1, 0) % freq == 0.
inline std::string indexer_types_json(int layers = 78, int freq = 4, int offset = 3) {
  std::string s;
  for (int i = 0; i < layers; ++i) {
    if (i) s += ", ";
    const int k = i - offset + 1 > 0 ? i - offset + 1 : 0;
    s += (k % freq == 0) ? "\"full\"" : "\"shared\"";
  }
  return s;
}

inline std::string mlp_types_json(int layers = 78, int dense = 3) {
  std::string s;
  for (int i = 0; i < layers; ++i) {
    if (i) s += ", ";
    s += i < dense ? "\"dense\"" : "\"sparse\"";
  }
  return s;
}

inline std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "",
                               const std::string& indexer_types = indexer_types_json()) {
  std::string s = kHead;
  s.replace(s.find("[INDEXER_TYPES]"), 15, "[" + indexer_types + "]");
  s.replace(s.find("[MLP_TYPES]"), 11, "[" + mlp_types_json() + "]");
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

}  // namespace glm_dsa_test
