#pragma once
// Full GLM-5.3 (GlmMoeDsaForCausalLM, model_type glm_moe_dsa) configuration,
// parsed from the checkpoint's config.json root (2026-09-12,
// docs/glm53_plan.md §1.1). The same policy as the other three parsers:
// every field the assembly consumes is parsed into a known-supported value
// or rejected with a message naming the field, at load time. The reference
// is transformers' modeling_glm_moe_dsa.py (main, 2026-09) and vLLM's
// deepseek_v2.py / deepseek_mtp.py for the draft layer.
//
// The quantization contract (HawkBearPig/GLM-5.3-Int4-Int8Mix-{RTN,AWQ}-g64,
// compressed-tensors 0.18 `pack-quantized`, plan §1.5): a contiguous range
// of main layers carries its attention projections (q_a, q_b, kv_a, kv_b,
// o_proj), its shared expert and its routed experts as packed-int triples —
// `weight_packed` I32 [N, K*bits/32] (unsigned codes offset 2^(bits-1), the
// low nibble/byte first), `weight_scale` BF16 [N, K/group] and
// `weight_shape` I64 [2] — symmetric, group 64 along K, no zero points.
// Everything else (the dense layers, the indexers, routers, norms, embedding,
// head and the whole draft layer) is BF16. Which module is packed, and at
// how many bits, is derived from the file's `config_groups` targets and
// `ignore` rules exactly as compressed-tensors applies them (a `re:` entry
// is a regex anchored at the start of the module name, anything else an
// exact name), then checked to be the one shape the loader implements.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct GlmDsaTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 6144;
  int vocab_size = 154880;
  int num_hidden_layers = 78;
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  int first_k_dense_replace = 3;   // layers [0, first_k_dense_replace) carry the dense MLP
  std::vector<int64_t> eos_token_ids;  // config.json's; serving resolves generation stop IDs separately
  int64_t pad_token_id = -1;

  // --- MLA attention (DeepSeek-V3 shape with interleaved decoupled RoPE)
  int num_attention_heads = 64;
  int q_lora_rank = 2048;
  int kv_lora_rank = 512;
  int qk_nope_head_dim = 192;
  int qk_rope_head_dim = 64;
  int v_head_dim = 256;
  bool attention_bias = false;
  double rope_theta = 8e6;
  bool rope_interleave = true;      // pairs (2i, 2i+1); the only form implemented

  // --- DSA indexer ----------------------------------------------------------
  int index_n_heads = 32;
  int index_head_dim = 128;
  int index_topk = 2048;            // selected tokens per query; kpool is 1 (per-token)
  bool indexer_rope_interleave = true;
  bool index_share_for_mtp_iteration = false;
  int index_topk_freq = 1;
  int index_skip_topk_offset = 2;
  // Per main layer: 1 = "full" (owns an indexer, selects), 0 = "shared"
  // (attends with the last full layer's selection). The draft layer always
  // owns one.
  std::vector<uint8_t> indexer_full;

  // --- MLP / MoE -----------------------------------------------------------
  int intermediate_size = 12288;      // the dense layers
  int moe_intermediate_size = 2048;   // per routed expert (and the shared expert, x n_shared_experts)
  int n_routed_experts = 256;
  int n_shared_experts = 1;
  int num_experts_per_tok = 8;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  int n_group = 1;
  int topk_group = 1;

  // --- MTP --------------------------------------------------------------------
  int num_nextn_predict_layers = 1;  // 0 or 1

  // --- packed-int weight formats (plan D2) -----------------------------------
  int packed_group_size = 64;   // elements per scale along K (the cores implement 64)
  int attention_bits = 8;       // q_a, q_b, kv_a, kv_b, o_proj of the packed layers
  int shared_bits = 8;          // the shared expert of the packed layers
  int expert_bits = 4;          // the routed experts of the packed layers
  int packed_layer_begin = 3;   // [begin, end): the main layers whose matrices are packed
  int packed_layer_end = 78;

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static GlmDsaTextConfig parse(const minijson::Value& root);
  static GlmDsaTextConfig from_json_file(const std::string& path);

  // Layer index of the draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return num_nextn_predict_layers == 1 ? num_hidden_layers : -1; }
  // Whether layer `l` (a main layer or the draft) carries the MoE.
  bool is_moe_layer(int l) const { return l >= first_k_dense_replace; }
  int num_moe_layers() const { return num_hidden_layers - first_k_dense_replace; }
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // Whether layer `l` (a main layer or the draft) runs its own indexer.
  bool owns_indexer(int l) const {
    if (l == mtp_layer()) return true;
    return l >= 0 && l < num_hidden_layers && indexer_full[static_cast<size_t>(l)] != 0;
  }
  // Main-stack layers that own an indexer (the index caches the pool holds).
  int num_indexer_layers() const {
    int n = 0;
    for (uint8_t f : indexer_full) n += f != 0;
    return n;
  }
  // Whether a main layer's attention, shared expert and routed experts are
  // packed-int triples (the draft is never packed in the file).
  bool packed_layer(int l) const { return l >= packed_layer_begin && l < packed_layer_end; }
  // The bits a module of layer `l` is stored at: 0 = BF16 `.weight`.
  int attention_bits_of(int l) const { return packed_layer(l) ? attention_bits : 0; }
  int shared_bits_of(int l) const { return packed_layer(l) ? shared_bits : 0; }
  int expert_bits_of(int l) const { return packed_layer(l) ? expert_bits : 0; }

  // The routed chain's configuration (models/glm/moe.hpp): the GLM sigmoid
  // router, the shared expert in the chain, no swiglu clamps. `local_inter`
  // is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
