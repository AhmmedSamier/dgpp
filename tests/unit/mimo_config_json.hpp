#pragma once
// The MiMo-V2.6-Flash config.json (XiaomiMiMo/MiMo-V2.6-Flash-RL,
// transcribed 2026-09-22; the vision / audio / processor sub-configs the
// parser never reads are omitted) as the unit tests build it. Shared by
// the config and binding tests; `config_json(from, to)` patches one anchor,
// `ignore_json(layers)` generates the quantization ignore list.
#include <stdexcept>
#include <string>

namespace mimo_test {

inline const char* kConfig = R"JSON({
 "add_full_attention_sink_bias": false,
 "add_swa_attention_sink_bias": true,
 "architectures": ["MiMoV2ForCausalLM"],
 "attention_bias": false,
 "attention_chunk_size": 128,
 "attention_dropout": 0.0,
 "attention_projection_layout": "fused_qkv",
 "attention_value_scale": 0.707,
 "audio_end_token_id": 151674,
 "audio_start_token_id": 151673,
 "audio_token_id": 151669,
 "bos_token_id": null,
 "dtype": "bfloat16",
 "eos_token_id": 151645,
 "head_dim": 192,
 "hidden_act": "silu",
 "hidden_size": 4096,
 "hybrid_block_size": null,
 "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
 "image_token_id": 151655,
 "initializer_range": 0.02,
 "intermediate_size": 16384,
 "layernorm_epsilon": 1e-06,
 "max_position_embeddings": 1048576,
 "model_type": "mimo_v2",
 "moe_intermediate_size": 2048,
 "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 "moe_router_dtype": "bfloat16",
 "n_group": 1,
 "n_routed_experts": 256,
 "n_shared_experts": null,
 "norm_topk_prob": true,
 "num_attention_heads": 64,
 "num_experts_per_tok": 8,
 "num_hidden_layers": 48,
 "num_key_value_heads": 4,
 "num_nextn_predict_layers": 3,
 "pad_token_id": 151643,
 "partial_rotary_factor": 0.334,
 "quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "mxfp4_block_size": 32,
  "quant_method": "fp8",
  "store_dtype": "mxfp4",
  "weight_block_size": [128, 128],
  "ignored_layers": [IGNORE]
 },
 "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default", "type": "default"},
 "rope_theta": 10000000.0,
 "routed_scaling_factor": null,
 "scoring_func": "sigmoid",
 "sliding_window": 128,
 "sliding_window_size": 128,
 "swa_head_dim": 192,
 "swa_num_attention_heads": 64,
 "swa_num_key_value_heads": 8,
 "swa_rope_theta": 10000.0,
 "swa_v_head_dim": 128,
 "tie_word_embeddings": false,
 "topk_group": 1,
 "topk_method": "noaux_tc",
 "transformers_version": "5.3.0",
 "use_cache": true,
 "v_head_dim": 128,
 "video_token_id": 151656,
 "vision_end_token_id": 151653,
 "vision_model_type": "mimovl",
 "vision_start_token_id": 151652,
 "vocab_size": 152576
})JSON";

inline std::string ignore_json(int layers = 48, bool decoder = true) {
  std::string s;
  for (int l = 0; l < layers; ++l) {
    if (l) s += ", ";
    s += "\"model.layers." + std::to_string(l) + ".self_attn.o_proj\"";
  }
  if (decoder) s += ", \"model.decoder.self_attn.o_proj\"";
  return s;
}

inline std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "",
                               const std::string& ignore = ignore_json()) {
  std::string s = kConfig;
  s.replace(s.find("[IGNORE]"), 8, "[" + ignore + "]");
  if (!patch_from.empty()) {
    const size_t at = s.find(patch_from);
    if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + patch_from);
    s.replace(at, patch_from.size(), patch_to);
  }
  return s;
}

}  // namespace mimo_test
