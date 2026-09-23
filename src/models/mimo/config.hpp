#pragma once
// MiMo-V2.6-Flash (MiMoV2ForCausalLM, model_type mimo_v2) configuration,
// parsed from the checkpoint's config.json root (2026-09-22,
// docs/mimo_v26_flash_plan.md §1). The same policy as the GLM-4.7, Qwen
// and DeepSeek parsers: every field the assembly consumes is parsed into a
// known-supported value or rejected with a message naming the field, at
// load time. The reference is the release's own modeling_mimo_v2.py
// (transformers 5.3 remote code) and vLLM's mimo_v2.py / mimo_v2_mtp.py.
//
// The architecture (the text backbone; the vision and audio encoders in
// the checkpoint are not served): 48 layers interleaving sliding-window
// attention (SWA, window 128, a per-head attention sink bias) with global
// attention (GA) on `hybrid_layer_pattern` (1 = SWA); GQA heads of qk dim
// 192 / v dim 128 with the first int(192 x 0.334) = 64 dims rotated
// (rotate_half pairs (i, i + 32)); values scaled by 0.707 before the
// cache; a fused qkv_proj per layer pre-sharded in `num_key_value_heads`
// chunks; layer 0 a dense SwiGLU MLP, every other layer a 256-expert
// sigmoid-routed MoE (noaux_tc: the bias on the selection key, the picked
// scores normalized, top 8, no shared expert); three MTP layers each an
// SWA layer with a dense MLP (the first is the draft block; vLLM serves the
// first alone too).
//
// The quantization contract (the release as shipped): quant_method fp8,
// e4m3 on 128 x 128 blocks with `weight_scale_inv` for every dense Linear
// but the o_proj (BF16, on the ignore list) and the head / embedding /
// router (BF16); the routed experts stored MXFP4 (`store_dtype`): e2m1
// codes two per byte with one e8m0 scale per 32 along K (`weight_scale`).
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct MimoTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 4096;
  int vocab_size = 152576;
  int num_hidden_layers = 48;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  std::vector<int64_t> eos_token_ids;  // config.json's; generation_config.json overrides at serve
  int64_t pad_token_id = -1;
  std::vector<uint8_t> swa_layer;      // [num_hidden_layers]: 1 = sliding window, 0 = global
  std::vector<uint8_t> moe_layer;      // [num_hidden_layers]: 1 = routed MoE, 0 = dense MLP

  // --- attention -----------------------------------------------------------
  int num_attention_heads = 64;        // both kinds share the query head count
  int num_key_value_heads = 4;         // GA kv heads (and the qkv_proj's chunk count)
  int swa_num_key_value_heads = 8;     // SWA kv heads
  int head_dim = 192;                  // qk dim (both kinds)
  int v_head_dim = 128;                // v dim (both kinds)
  int rotary_dim = 64;                 // int(head_dim * partial_rotary_factor); pairs (i, i + rotary_dim/2)
  double rope_theta = 1e7;             // GA
  double swa_rope_theta = 1e4;         // SWA (and the MTP layers)
  int sliding_window = 128;            // a row attends [pos - W + 1, pos]
  bool attention_bias = false;
  bool swa_sink = true;                // attention_sink_bias on the SWA layers
  bool full_sink = false;              // ... and on the GA layers
  float attention_value_scale = 1.0f;  // v = bf16(v * scale) before the cache (1: none)

  // --- MLP / MoE -----------------------------------------------------------
  int intermediate_size = 16384;       // the dense layers (layer 0, the MTP layers)
  int moe_intermediate_size = 2048;    // per routed expert
  int n_routed_experts = 256;
  int num_experts_per_tok = 8;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 1.0f;
  int n_group = 1;
  int topk_group = 1;

  // --- MTP --------------------------------------------------------------------
  int num_nextn_predict_layers = 3;  // the checkpoint carries three heads
  int mtp_layers_loaded = 1;         // 0/1 by default; 3 with DGPP_MIMO_NATIVE_MTP=1

  // --- weight formats -----------------------------------------------------
  int fp8_block = 128;                 // weight_block_size (both axes)
  int mxfp4_block = 32;                // the experts' e8m0 group along K

  static MimoTextConfig parse(const minijson::Value& root);
  static MimoTextConfig from_json_file(const std::string& path);

  // Layer index of the draft layer in the family's layer space
  // (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return mtp_layers_loaded > 0 ? num_hidden_layers : -1; }
  bool is_mtp_layer(int l) const {
    return l >= num_hidden_layers && l < num_hidden_layers + mtp_layers_loaded;
  }
  // Whether layer `l` (a main layer or the draft) is a sliding-window
  // layer; the draft is SWA (vLLM's mimo_v2_mtp).
  bool is_swa_layer(int l) const { return is_mtp_layer(l) ? true : swa_layer[static_cast<size_t>(l)] != 0; }
  // Whether layer `l` carries the routed MoE (the draft is dense).
  bool is_moe_layer(int l) const { return is_mtp_layer(l) ? false : moe_layer[static_cast<size_t>(l)] != 0; }
  int num_moe_layers() const {
    int n = 0;
    for (uint8_t m : moe_layer) n += m != 0;
    return n;
  }
  int num_swa_layers() const {
    int n = 0;
    for (uint8_t s : swa_layer) n += s != 0;
    return n;
  }
  // The kv head count of layer `l`.
  int kv_heads_of(int l) const { return is_swa_layer(l) ? swa_num_key_value_heads : num_key_value_heads; }
  bool sink_of(int l) const { return is_swa_layer(l) ? swa_sink : full_sink; }
  double rope_theta_of(int l) const { return is_swa_layer(l) ? swa_rope_theta : rope_theta; }
  // The fused qkv_proj rows of a layer: chunk_count() chunks of
  // [Q (heads/chunks x head_dim) | K (kv/chunks x head_dim) | V (kv/chunks x
  // v_head_dim)] rows each; the fp8 scale grid is tiled per chunk
  // (ceil(chunk_rows / 128) scale rows per chunk).
  int qkv_chunks() const { return num_key_value_heads; }
  int64_t qkv_chunk_rows_of_kind(bool swa) const {
    const int64_t c = qkv_chunks();
    const int64_t kv = swa ? swa_num_key_value_heads : num_key_value_heads;
    return (num_attention_heads / c) * static_cast<int64_t>(head_dim) + (kv / c) * static_cast<int64_t>(head_dim) +
           (kv / c) * static_cast<int64_t>(v_head_dim);
  }
  int64_t qkv_chunk_rows(int l) const { return qkv_chunk_rows_of_kind(is_swa_layer(l)); }
  int64_t qkv_rows(int l) const { return qkv_chunk_rows(l) * qkv_chunks(); }
  int64_t qkv_scale_rows(int l) const { return ((qkv_chunk_rows(l) + fp8_block - 1) / fp8_block) * qkv_chunks(); }
  int64_t o_proj_cols() const { return static_cast<int64_t>(num_attention_heads) * v_head_dim; }
  // The routed chain's configuration (models/glm/moe.hpp): the sigmoid
  // router with its selection bias, no shared expert, no swiglu clamps.
  // `local_inter` is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
