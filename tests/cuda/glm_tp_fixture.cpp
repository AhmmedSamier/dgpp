// g++-compiled fixture side of glm_tp_test (see glm_rng.hpp for why this
// is a separate TU: nvcc cannot compile minijson's vector-of-incomplete
// Member, so the .cu test never includes it).
#include <string>

#include "glm_fixture.hpp"

namespace {

// TP-divisible geometry; awkward inter dims kept deliberately (the scale
// grid slicing must handle partial 128 blocks). Values are identical to
// the M4 chain's fixture writer — same tensor names, same RNG.
const char* kTpJson = R"json({
  "hidden_size": 256, "vocab_size": 96, "num_hidden_layers": 6,
  "rms_norm_eps": 1e-5, "tie_word_embeddings": false,
  "hidden_act": "silu", "swiglu_limit": 7.5,
  "layer_types": ["linear_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention",
                  "deepseek_sparse_attention", "linear_attention"],
  "mlp_layer_types": ["dense", "dense", "sparse", "sparse", "sparse", "sparse"],
  "indexer_types": ["full", "full", "full", "full", "full", "full"],
  "first_k_dense_replace": 2,
  "linear_attn_config": {
    "num_heads": 4, "head_dim": 64, "short_conv_kernel_size": 4,
    "gate_lower_bound": -3.5,
    "kda_layers": [0, 1, 3, 5], "full_attn_layers": [2, 4]
  },
  "num_attention_heads": 8, "q_lora_rank": 64, "kv_lora_rank": 64,
  "qk_nope_head_dim": 32, "qk_rope_head_dim": 0, "v_head_dim": 32,
  "mla_use_nope": true,
  "index_n_heads": 32, "index_head_dim": 128, "index_kpool": 4,
  "index_topk": 32, "index_kpool_compress": true,
  "index_kpool_always_select_tail": true, "indexer_rope_interleave": true,
  "intermediate_size": 200, "moe_intermediate_size": 64,
  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
  "norm_topk_prob": true, "routed_scaling_factor": 1.5,
  "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
  "mhc": true, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
  "num_nextn_predict_layers": 1
})json";

}  // namespace

dgpp::GlmTextConfig glm_tp_test_config() {
  const auto parsed = dgpp::minijson::parse(kTpJson);
  return dgpp::GlmTextConfig::parse(parsed.root);
}

void glm_tp_write_fixture(const std::string& dir) {
  glmfx::write_fixture(glm_tp_test_config(), kTpJson, dir);
}
