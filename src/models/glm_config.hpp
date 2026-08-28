#pragma once
// GLM-5.3-Flash text-model configuration, parsed from the checkpoint's
// config.json (text_config object). This is the M4 config-driven assembly's
// single source of trained values: KdaConfig/DsaConfig defaults in their
// geometry headers must NOT be trusted when a checkpoint is loaded — the
// parser below fills them from the file and rejects anything the M4 assembly
// does not implement, at load time rather than silently at runtime.
//
// Validation policy: every field the assembly consumes is either parsed into
// a known-supported value or rejected with a message naming the field. A
// checkpoint that loads is one the engine can actually run; "parse-then-pray"
// on unrecognized enum strings is not a strategy (DESIGN §12).
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/dsa_geometry.hpp"
#include "models/glm_mhc.hpp"
#include "models/kda_geometry.hpp"

namespace dgpp {

enum class GlmLayerKind : int { Kda, Dsa };
enum class GlmMlpKind : int { Dense, Moe };

struct GlmTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 4096;
  int vocab_size = 154880;
  int num_hidden_layers = 45;
  float rms_norm_eps = 1e-5f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  float swiglu_limit = 10.0f;

  // Per-layer attention and MLP classes (sizes == num_hidden_layers).
  std::vector<GlmLayerKind> layers;
  std::vector<GlmMlpKind> mlps;
  int first_k_dense_replace = 3;

  // --- KDA (linear_attn_config) -------------------------------------------
  int kda_num_heads = 64;
  int kda_head_dim = 128;
  int kda_conv_width = 4;
  float kda_gate_lower_bound = -5.0f;

  // --- MLA / DSA ----------------------------------------------------------
  int num_attention_heads = 64;  // MLA heads on DSA layers
  int q_lora_rank = 1536;
  int kv_lora_rank = 512;
  int qk_nope_head_dim = 256;
  int qk_rope_head_dim = 0;  // engine implements the rope-free path only
  int v_head_dim = 256;
  bool mla_use_nope = true;

  // --- DSA indexer --------------------------------------------------------
  int index_n_heads = 32;
  int index_head_dim = 128;
  int index_kpool = 4;
  int index_topk = 2048;
  bool index_kpool_compress = true;
  bool index_kpool_always_select_tail = true;
  bool indexer_rope_interleave = true;

  // --- MLP / MoE ----------------------------------------------------------
  int intermediate_size = 12288;  // dense layers
  int moe_intermediate_size = 2048;
  int n_routed_experts = 288;
  int n_shared_experts = 1;
  int num_experts_per_tok = 8;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  int n_group = 1;
  int topk_group = 1;
  std::string moe_router_dtype = "float32";

  // --- mHC (multi-head hyper-connections) ----------------------------------
  bool mhc = true;
  int hc_mult = 4;
  int hc_sinkhorn_iters = 20;
  float hc_eps = 1e-6f;

  // --- MTP -----------------------------------------------------------------
  int num_nextn_predict_layers = 1;

  // Parses the text_config object of a GLM-5 checkpoint file. Throws
  // std::runtime_error naming the offending field on anything unsupported.
  static GlmTextConfig parse(const minijson::Value& text_config);

  // Reads config.json from disk and dispatches to parse().
  static GlmTextConfig from_json_file(const std::string& path);

  // Filled geometry configs for the layer modules (validated on fill).
  KdaConfig kda_config() const;
  DsaConfig dsa_config() const;
  GlmMhcConfig mhc_config() const;

  int num_kda_layers() const;
  int num_dsa_layers() const;
  // Layer index of the (single) MTP draft layer; -1 when absent.
  int mtp_layer() const {
    return num_nextn_predict_layers == 1 ? num_hidden_layers : -1;
  }
};

}  // namespace dgpp
