#pragma once
// GLM-5.3-Flash text-model configuration, parsed from the checkpoint's
// config.json (text_config object). This is the M4 config-driven assembly's
// single source of trained values: KdaConfig/DsaConfig defaults in their
// geometry headers must not be trusted when a checkpoint is loaded — the
// parser below fills them from the file and rejects anything the M4 assembly
// does not implement, at load time rather than silently at runtime.
//
// Validation policy: every field the assembly consumes is either parsed into
// a known-supported value or rejected with a message naming the field. A
// checkpoint that loads is one the engine can actually run; "parse-then-pray"
// on unrecognized enum strings is not a strategy (DESIGN §12).
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/vision_config.hpp"
#include "models/dsa_geometry.hpp"
#include "models/glm/mhc.hpp"
#include "models/glm/moe.hpp"
#include "models/kda_geometry.hpp"

namespace dgpp {

enum class GlmLayerKind : int { Kda, Dsa };
enum class GlmMlpKind : int { Dense, Moe };

// The routed experts' resident weight format (docs/nvfp4_plan.md §2, §4):
// what config.json's quantization_config declares for the main-stack MoE
// layers. Fp8Block128 is the FP8 release (e4m3 payload + F32 128x128
// block scales, `weight` + `weight_scale_inv`) and the default when the
// file says quant_method "fp8" or nothing. Nvfp4Group16 is the composed
// hybrid (quant_method "dgpp_mixed"): e2m1 pairs + e4m3 scales per 16 +
// one F32 global scale (`weight_packed` + `weight_scale` +
// `weight_global_scale`). Every OTHER tensor class — shared experts, dense
// MLPs, attention, the MTP layer — keeps the FP8 release's formats under
// both, which is why this is one enum and not a per-class table.
enum class GlmExpertFormat : int { Fp8Block128 = 0, Nvfp4Group16 = 1 };

constexpr const char* glm_expert_format_name(GlmExpertFormat f) {
  return f == GlmExpertFormat::Nvfp4Group16 ? "nvfp4-group16" : "fp8-block128";
}

// Sampling defaults shipped beside config.json in generation_config.json.
// Keep presence separate from the effective value: a missing sampling field
// is a configuration gap which callers must be able to name in their logs,
// while the effective getters deliberately choose greedy-safe values instead
// of silently inheriting the OpenAI wire defaults.
struct GlmGenerationDefaults {
  std::optional<bool> do_sample;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<float> min_p;
  std::optional<float> repetition_penalty;
  std::optional<std::vector<int64_t>> eos_token_ids;
  bool file_found = true;

  float effective_temperature() const {
    if (do_sample.has_value() && !*do_sample) return 0.0f;
    return temperature.value_or(0.0f);
  }
  float effective_top_p() const { return top_p.value_or(1.0f); }
  int effective_top_k() const { return top_k.value_or(0); }
  float effective_min_p() const { return min_p.value_or(0.0f); }
  float effective_repetition_penalty() const {
    return repetition_penalty.value_or(1.0f);
  }
  // Generation stop tokens may differ from the model's trained EOS (for
  // example, Qwen uses the latter to pad/reset PLE n-gram history).
  // Resolve the serving policy without overwriting model configuration.
  std::vector<int64_t> effective_eos_token_ids(
      const std::vector<int64_t>& model_eos_token_ids) const {
    return eos_token_ids.value_or(model_eos_token_ids);
  }

  // Names every absent sampler field whose neutral/greedy-safe fallback is
  // in effect. JSON null is treated as absent, matching generated HF config
  // files where an unset optional is serialized explicitly.
  std::vector<std::string> fallback_fields() const;

  // Parse the root object of generation_config.json. Present fields are
  // strict: wrong types, non-finite values, invalid ranges, and bad EOS ids
  // fail at model load rather than changing sampling semantics silently.
  static GlmGenerationDefaults parse(const minijson::Value& root,
                                     int vocab_size);
  static GlmGenerationDefaults from_json_file(const std::string& path,
                                              int vocab_size);

  // Load DIR/generation_config.json. A genuinely absent file is allowed and
  // returns greedy-safe defaults with file_found=false; malformed or
  // unreadable files still fail loudly. This function logs the exact missing
  // file/field fallback required by DESIGN §10.
  static GlmGenerationDefaults from_checkpoint_dir(const std::string& dir,
                                                   int vocab_size);
};

struct GlmTextConfig {
  std::optional<GlmVisionConfig> vision;
  // The prompt's image delimiters; an unset triple for a text-only export.
  ImageTokens image_tokens() const { return vision ? vision->tokens : ImageTokens{}; }
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

  // Generation terminators from config.json. generation_config.json is the
  // serving authority when it contains eos_token_id; this remains the
  // compatibility fallback for fixtures and older checkpoints.
  std::vector<int64_t> eos_token_ids;

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

  // --- weight formats -------------------------------------------------------
  // The main-stack routed experts' format (from quantization_config; see
  // GlmExpertFormat). The MTP layer's experts are FP8 under both.
  GlmExpertFormat routed_expert_format = GlmExpertFormat::Fp8Block128;
  GlmExpertFormat expert_format(int layer) const {
    return layer == mtp_layer() ? GlmExpertFormat::Fp8Block128
                                : routed_expert_format;
  }

  // Parses the text_config object of a GLM-5 checkpoint file. Throws
  // std::runtime_error naming the offending field on anything unsupported.
  // `quantization_config` (the root object's, nullable) selects the routed
  // experts' format; absent means the FP8 release.
  static GlmTextConfig parse(const minijson::Value& text_config,
                             const minijson::Value* quantization_config = nullptr);

  // The quantization_config -> GlmExpertFormat rule on its own (null =
  // FP8). Rejects unknown quant_method values and dgpp_mixed files whose
  // routed-expert group is not NVFP4 group 16 or whose MTP/DSA sources are
  // not the FP8 release (the engine has no BF16 expert path).
  static GlmExpertFormat parse_expert_format(
      const minijson::Value* quantization_config, const GlmTextConfig& text);

  // Reads config.json from disk and dispatches to parse().
  static GlmTextConfig from_json_file(const std::string& path);

  // Filled geometry configs for the layer modules (validated on fill).
  KdaConfig kda_config() const;
  DsaConfig dsa_config() const;
  GlmMhcConfig mhc_config() const;
  GlmMoeConfig moe_config() const;

  int num_kda_layers() const;
  int num_dsa_layers() const;
  // Layer index of the (single) MTP draft layer; -1 when absent.
  int mtp_layer() const {
    return num_nextn_predict_layers == 1 ? num_hidden_layers : -1;
  }
};

}  // namespace dgpp
