#include "models/glm4/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("GLM-4.7 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// The modelopt NVFP4 contract: quant_method modelopt, quant_algo NVFP4,
// the Linear group 4-bit float with group 16, and the ignore list naming
// exactly lm_head, every main layer's attention and the whole draft
// layer — the loader expects BF16 there and NVFP4 triples everywhere
// else, so an ignore entry it does not know is a tensor class it would
// bind wrongly.
void parse_quantization(const minijson::Value& root, Glm4TextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the NVFP4 release (modelopt); an "
           "unquantized Glm4Moe checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  const std::string method = optional_string(*qc, "quant_method", "");
  if (method != "modelopt")
    reject("quantization_config.quant_method", "only modelopt is implemented, got '" + method + "'");
  const std::string algo = optional_string(*qc, "quant_algo", "");
  if (algo != "NVFP4")
    reject("quantization_config.quant_algo", "only NVFP4 is implemented, got '" + algo + "'");
  const minijson::Value* groups = qc->find("config_groups");
  if (!groups || !groups->is_object() || groups->members().empty())
    reject("quantization_config.config_groups", "missing");
  int seen = 0;
  for (const auto& m : groups->members()) {
    const minijson::Value& g = m.value;
    if (!g.is_object()) reject("quantization_config.config_groups", "group is not an object");
    const minijson::Value* w = g.find("weights");
    if (!w || !w->is_object()) reject("quantization_config.config_groups.weights", "missing");
    if (require_int(*w, "num_bits") != 4)
      reject("quantization_config.config_groups.weights.num_bits", "must be 4");
    if (optional_string(*w, "type", "float") != "float")
      reject("quantization_config.config_groups.weights.type", "must be float (e2m1)");
    c.fp4_group_size = require_int(*w, "group_size");
    if (c.fp4_group_size != 16)
      reject("quantization_config.config_groups.weights.group_size", "the NVFP4 kernels implement 16");
    if (const minijson::Value* t = g.find("targets"); t && t->is_array())
      for (const auto& item : t->items())
        if (!item.is_string() || item.as_string() != "Linear")
          reject("quantization_config.config_groups.targets", "only the Linear target is implemented");
    ++seen;
  }
  if (seen != 1) reject("quantization_config.config_groups", "exactly one group is implemented");
  // The ignore list.
  std::set<std::string> ignore;
  if (const minijson::Value* ig = qc->find("ignore"); ig && ig->is_array())
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      ignore.insert(std::string(item.as_string()));
    }
  std::set<std::string> want;
  want.insert("lm_head");
  for (int l = 0; l < c.num_hidden_layers; ++l)
    want.insert("model.layers." + std::to_string(l) + ".self_attn*");
  if (c.mtp_layer() >= 0) want.insert("model.layers." + std::to_string(c.mtp_layer()) + "*");
  for (const std::string& w : want)
    if (!ignore.count(w))
      reject("quantization_config.ignore",
             "'" + w + "' is not ignored — the engine expects BF16 there (attention, the "
             "head, the draft layer)");
  for (const std::string& i : ignore)
    if (!want.count(i))
      reject("quantization_config.ignore",
             "unexpected ignored module '" + i + "' (a BF16 class the loader does not implement)");
  // The KV scheme: FP8 static (scale tensors present) or absent.
  c.kv_scales_present = false;
  if (const minijson::Value* kv = qc->find("kv_cache_scheme"); kv && !kv->is_null()) {
    if (!kv->is_object()) reject("quantization_config.kv_cache_scheme", "not an object");
    if (require_int(*kv, "num_bits") != 8 || optional_string(*kv, "type", "float") != "float")
      reject("quantization_config.kv_cache_scheme", "only the fp8 (e4m3) static scheme is implemented");
    c.kv_scales_present = true;
  }
}

}  // namespace

Glm4TextConfig Glm4TextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  Glm4TextConfig c;
  const std::string model_type = optional_string(root, "model_type", "glm4_moe");
  if (model_type != "glm4_moe") reject("model_type", "expected glm4_moe, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  c.first_k_dense_replace = optional_int(root, "first_k_dense_replace", 0);
  if (c.hidden_size <= 0 || c.hidden_size % 32 != 0)
    reject("hidden_size", "must be a positive multiple of 32 (the NVFP4 GEMV core's K)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (c.first_k_dense_replace < 0 || c.first_k_dense_replace > c.num_hidden_layers)
    reject("first_k_dense_replace", "outside [0, num_hidden_layers]");

  // --- tokens ---------------------------------------------------------------
  if (const minijson::Value* eos = root.find("eos_token_id"); eos && !eos->is_null()) {
    if (eos->is_array()) {
      for (const auto& item : eos->items()) {
        if (!item.is_number()) reject("eos_token_id", "non-numeric element");
        c.eos_token_ids.push_back(item.as_int());
      }
    } else if (eos->is_number()) {
      c.eos_token_ids.push_back(eos->as_int());
    } else {
      reject("eos_token_id", "not a number or array");
    }
    for (int64_t id : c.eos_token_ids)
      if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  }
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  c.pad_token_id = optional_int(root, "pad_token_id", -1);

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = optional_int(root, "head_dim", c.hidden_size / std::max(c.num_attention_heads, 1));
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.use_qk_norm = optional_bool(root, "use_qk_norm", false);
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 128) reject("head_dim", "the attention kernels implement 128-wide heads");
  {
    const double factor = optional_double(root, "partial_rotary_factor", 1.0);
    const double rd = c.head_dim * factor;
    if (!(rd > 0) || rd != std::floor(rd) || static_cast<int>(rd) % 2 != 0 || rd > c.head_dim)
      reject("partial_rotary_factor", "rotary dim must be a positive even integer within the head");
    c.rotary_dim = static_cast<int>(rd);
  }
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && rp->is_object()) {
    if (optional_string(*rp, "rope_type", "default") != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented");
    c.rope_theta = optional_double(*rp, "rope_theta", c.rope_theta);
  }
  if (!(c.rope_theta > 0)) reject("rope_theta", "must be positive");

  // --- MLP / MoE --------------------------------------------------------------
  c.intermediate_size = require_int(root, "intermediate_size");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.n_routed_experts = require_int(root, "n_routed_experts");
  c.n_shared_experts = optional_int(root, "n_shared_experts", 0);
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  c.routed_scaling_factor = static_cast<float>(optional_double(root, "routed_scaling_factor", 1.0));
  c.n_group = optional_int(root, "n_group", 1);
  c.topk_group = optional_int(root, "topk_group", 1);
  if (c.intermediate_size <= 0 || c.intermediate_size % 32 != 0)
    reject("intermediate_size", "must be a positive multiple of 32");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 32 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 32");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
    reject("n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 ||
      c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (c.n_shared_experts != 1)
    reject("n_shared_experts", "the MoE chain implements exactly one shared expert");
  if (c.n_group != 1 || c.topk_group != 1)
    reject("n_group", "group-limited routing (n_group/topk_group != 1) is not implemented");
  if (!(c.routed_scaling_factor > 0)) reject("routed_scaling_factor", "must be positive");
  if (const std::string sf = optional_string(root, "scoring_func", "sigmoid"); sf != "sigmoid")
    reject("scoring_func", "only sigmoid is implemented");
  if (const std::string tm = optional_string(root, "topk_method", "noaux_tc"); tm != "noaux_tc")
    reject("topk_method", "only noaux_tc is implemented");

  // --- MTP --------------------------------------------------------------------
  c.num_nextn_predict_layers = optional_int(root, "num_nextn_predict_layers", 0);
  if (c.num_nextn_predict_layers != 0 && c.num_nextn_predict_layers != 1)
    reject("num_nextn_predict_layers", "only the single draft layer is implemented");
  if (c.mtp_layer() >= 0 && !c.is_moe_layer(c.mtp_layer()))
    reject("num_nextn_predict_layers", "the draft layer carries the MoE (first_k_dense_replace)");

  parse_quantization(root, c);
  return c;
}

Glm4TextConfig Glm4TextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

GlmMoeConfig Glm4TextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // Glm4MoeMLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
