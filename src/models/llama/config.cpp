#include "models/llama/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Llama config.{}: {}", field, why));
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

}  // namespace

LlamaTextConfig LlamaTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  LlamaTextConfig c;
  const std::string model_type = optional_string(root, "model_type", "llama");
  if (model_type != "llama") reject("model_type", "expected llama, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(optional_double(root, "rms_norm_eps", 1e-6));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 32 != 0)
    reject("hidden_size", "must be a positive multiple of 32");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (!(c.rms_norm_eps > 0)) reject("rms_norm_eps", "must be positive");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");

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
  c.bos_token_id = optional_int(root, "bos_token_id", -1);
  c.pad_token_id = optional_int(root, "pad_token_id", -1);

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = optional_int(root, "head_dim", c.hidden_size / std::max(c.num_attention_heads, 1));
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 128) reject("head_dim", "the attention kernels implement 128-wide heads");
  if (c.attention_bias) reject("attention_bias", "biased attention is not implemented");
  if (optional_bool(root, "use_qk_norm", false))
    reject("use_qk_norm", "qk-norm is not implemented");
  if (optional_bool(root, "mlp_bias", false)) reject("mlp_bias", "biased MLP is not implemented");
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (!(c.rope_theta > 0)) reject("rope_theta", "must be positive");

  // --- MLP ------------------------------------------------------------------
  c.intermediate_size = require_int(root, "intermediate_size");
  if (c.intermediate_size <= 0 || c.intermediate_size % 32 != 0)
    reject("intermediate_size", "must be a positive multiple of 32");

  // --- scope guards: dense Llama only ---------------------------------------
  if (const minijson::Value* qc = root.find("quantization_config"); qc && !qc->is_null())
    reject("quantization_config", "not supported (the checkpoint's BF16 weights are served as published; block FP8 at load is selected via engine.dense_weights = \"fp8\")");
  if (optional_int(root, "num_nextn_predict_layers", 0) != 0)
    reject("num_nextn_predict_layers", "only the plain (no-draft) stack is implemented");
  if (optional_int(root, "num_key_value_groups", 0) != 0 &&
      optional_int(root, "num_key_value_groups", 1) != 1)
    reject("num_key_value_groups", "unsupported");
  return c;
}

LlamaTextConfig LlamaTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

}  // namespace dgpp
