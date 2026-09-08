#include "models/glm/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <set>

#include "common/log.hpp"

namespace dgpp {

namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(
      std::format("GLM text_config.{}: {}", field, why));
}

[[noreturn]] void reject_generation(std::string_view field,
                                    std::string_view why) {
  throw std::runtime_error(
      std::format("GLM generation_config.{}: {}", field, why));
}

std::optional<float> optional_generation_float(const minijson::Value& root,
                                               std::string_view field) {
  const minijson::Value* value = root.find(field);
  if (value == nullptr || value->is_null()) return std::nullopt;
  if (!value->is_number()) reject_generation(field, "not a number");
  const double parsed = value->as_double();
  if (!std::isfinite(parsed) ||
      parsed < -static_cast<double>(std::numeric_limits<float>::max()) ||
      parsed > static_cast<double>(std::numeric_limits<float>::max())) {
    reject_generation(field, "not a finite float");
  }
  return static_cast<float>(parsed);
}

std::optional<bool> optional_generation_bool(const minijson::Value& root,
                                             std::string_view field) {
  const minijson::Value* value = root.find(field);
  if (value == nullptr || value->is_null()) return std::nullopt;
  if (!value->is_bool()) reject_generation(field, "not a bool");
  return value->as_bool();
}

std::string read_json_file(const std::string& path, std::string_view kind) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open {} {}: {}", kind, path,
                                         std::strerror(errno)));
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len < 0) {
    std::fclose(f);
    throw std::runtime_error(std::format("cannot size {} {}", kind, path));
  }
  std::string text(static_cast<size_t>(len), '\0');
  if (len > 0 &&
      std::fread(text.data(), 1, static_cast<size_t>(len), f) !=
          static_cast<size_t>(len)) {
    std::fclose(f);
    throw std::runtime_error(std::format("short read on {} {}", kind, path));
  }
  std::fclose(f);
  return text;
}

int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}

float require_float(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<float>(f->as_double());
}

bool require_bool(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  // JSON booleans only; an int 0/1 here is a config we did not write.
  if (f->kind() != minijson::Value::Kind::Bool) reject(field, "not a bool");
  return f->as_bool();
}

std::string require_string(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}

template <typename T, typename Fn>
std::vector<T> require_string_array(const minijson::Value& v,
                                    std::string_view field, Fn&& map) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  if (!f->is_array()) reject(field, "not an array");
  std::vector<T> out;
  out.reserve(f->items().size());
  for (const auto& item : f->items()) {
    if (!item.is_string()) reject(field, "non-string element");
    T mapped = map(std::string(item.as_string()));
    if (mapped < T(0)) reject(field, "unrecognized value " +
                                          std::string(item.as_string()));
    out.push_back(mapped);
  }
  return out;
}

std::vector<int> require_int_array(const minijson::Value& v,
                                   std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f) reject(field, "missing");
  if (!f->is_array()) reject(field, "not an array");
  std::vector<int> out;
  out.reserve(f->items().size());
  for (const auto& item : f->items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(static_cast<int>(item.as_int()));
  }
  return out;
}

}  // namespace

std::vector<std::string> GlmGenerationDefaults::fallback_fields() const {
  std::vector<std::string> out;
  if (!temperature.has_value()) out.emplace_back("temperature");
  if (!top_p.has_value()) out.emplace_back("top_p");
  if (!top_k.has_value()) out.emplace_back("top_k");
  if (!min_p.has_value()) out.emplace_back("min_p");
  if (!repetition_penalty.has_value())
    out.emplace_back("repetition_penalty");
  return out;
}

GlmGenerationDefaults GlmGenerationDefaults::parse(
    const minijson::Value& root, int vocab_size) {
  if (!root.is_object()) reject_generation("root", "not an object");
  if (vocab_size <= 0) reject_generation("vocab_size", "must be positive");

  GlmGenerationDefaults out;
  out.do_sample = optional_generation_bool(root, "do_sample");
  out.temperature = optional_generation_float(root, "temperature");
  out.top_p = optional_generation_float(root, "top_p");
  out.min_p = optional_generation_float(root, "min_p");
  out.repetition_penalty =
      optional_generation_float(root, "repetition_penalty");

  if (const minijson::Value* value = root.find("top_k");
      value != nullptr && !value->is_null()) {
    if (value->kind() != minijson::Value::Kind::Int)
      reject_generation("top_k", "not an integer");
    const int64_t parsed = value->as_int();
    if (parsed < 0 || parsed > std::numeric_limits<int>::max())
      reject_generation("top_k", "must be in [0, INT_MAX]");
    out.top_k = static_cast<int>(parsed);
  }

  if (out.temperature.has_value() && *out.temperature < 0.0f)
    reject_generation("temperature", "must be >= 0");
  if (out.top_p.has_value() &&
      !(*out.top_p > 0.0f && *out.top_p <= 1.0f))
    reject_generation("top_p", "must be in (0, 1]");
  if (out.min_p.has_value() &&
      !(*out.min_p >= 0.0f && *out.min_p <= 1.0f))
    reject_generation("min_p", "must be in [0, 1]");
  if (out.repetition_penalty.has_value() &&
      !(*out.repetition_penalty > 0.0f))
    reject_generation("repetition_penalty", "must be > 0");

  if (const minijson::Value* eos = root.find("eos_token_id");
      eos != nullptr && !eos->is_null()) {
    std::vector<int64_t> ids;
    const auto append = [&](const minijson::Value& value) {
      if (value.kind() != minijson::Value::Kind::Int)
        reject_generation("eos_token_id", "contains a non-integer");
      const int64_t id = value.as_int();
      if (id < 0 || id >= vocab_size)
        reject_generation("eos_token_id", "id outside [0, vocab_size)");
      ids.push_back(id);
    };
    if (eos->is_array()) {
      for (const auto& item : eos->items()) append(item);
    } else {
      append(*eos);
    }
    out.eos_token_ids = std::move(ids);
  }
  return out;
}

GlmGenerationDefaults GlmGenerationDefaults::from_json_file(
    const std::string& path, int vocab_size) {
  const std::string text = read_json_file(path, "generation config");
  return parse(minijson::parse(text).root, vocab_size);
}

GlmGenerationDefaults GlmGenerationDefaults::from_checkpoint_dir(
    const std::string& dir, int vocab_size) {
  const std::filesystem::path path =
      std::filesystem::path(dir) / "generation_config.json";
  std::error_code ec;
  const bool exists = std::filesystem::exists(path, ec);
  if (ec) {
    throw std::runtime_error("cannot inspect generation config " +
                             path.string() + ": " + ec.message());
  }
  if (!exists) {
    GlmGenerationDefaults out;
    out.file_found = false;
    DGPP_LOG_WARN(
        "generation config {} is missing; using greedy-safe sampling "
        "defaults (temperature=0, top_p=1, top_k=0, min_p=0, "
        "repetition_penalty=1)",
        path.string());
    return out;
  }

  GlmGenerationDefaults out = from_json_file(path.string(), vocab_size);
  const std::vector<std::string> missing = out.fallback_fields();
  if (!missing.empty()) {
    std::string names;
    for (const std::string& name : missing) {
      if (!names.empty()) names += ", ";
      names += name;
    }
    DGPP_LOG_WARN(
        "generation config {} omits {}; using greedy-safe/neutral "
        "fallbacks for those fields",
        path.string(), names);
  }
  DGPP_LOG_INFO(
      "generation defaults: temperature={} top_p={} top_k={} min_p={} "
      "repetition_penalty={}{}",
      out.effective_temperature(), out.effective_top_p(),
      out.effective_top_k(), out.effective_min_p(),
      out.effective_repetition_penalty(),
      out.do_sample.has_value() && !*out.do_sample
          ? " (do_sample=false: greedy)"
          : "");
  return out;
}

namespace {
[[noreturn]] void reject_quant(std::string_view field, std::string_view why) {
  throw std::runtime_error(
      std::format("GLM quantization_config.{}: {}", field, why));
}
}  // namespace

GlmExpertFormat GlmTextConfig::parse_expert_format(
    const minijson::Value* qc, const GlmTextConfig& text) {
  if (!qc) return GlmExpertFormat::Fp8Block128;
  if (!qc->is_object()) reject_quant("", "not an object");
  const minijson::Value* method = qc->find("quant_method");
  if (!method) reject_quant("quant_method", "missing");
  if (!method->is_string()) reject_quant("quant_method", "not a string");
  const std::string m(method->as_string());
  if (m == "fp8") return GlmExpertFormat::Fp8Block128;
  if (m != "dgpp_mixed")
    reject_quant("quant_method",
                 "unsupported '" + m +
                     "' (the engine loads the FP8 release, quant_method "
                     "\"fp8\", and the composed NVFP4 hybrid, \"dgpp_mixed\")");
  const minijson::Value* re = qc->find("routed_experts");
  if (!re || !re->is_object()) reject_quant("routed_experts", "missing object");
  const std::string fmt = require_string(*re, "format");
  if (fmt != "nvfp4-pack-quantized")
    reject_quant("routed_experts.format",
                 "only nvfp4-pack-quantized is implemented, got " + fmt);
  if (require_int(*re, "group_size") != 16)
    reject_quant("routed_experts.group_size", "must be 16 (NVFP4)");
  if (const minijson::Value* nb = re->find("num_bits");
      nb && (!nb->is_number() || static_cast<int>(nb->as_int()) != 4))
    reject_quant("routed_experts.num_bits", "must be 4");
  // The layer list, when given, must be exactly the main stack's MoE
  // layers: a hybrid whose experts are NVFP4 elsewhere is not this format.
  if (const minijson::Value* layers = re->find("layers")) {
    if (!layers->is_array()) reject_quant("routed_experts.layers", "not an array");
    std::vector<int> want;
    for (int i = 0; i < text.num_hidden_layers; ++i)
      if (text.mlps[static_cast<size_t>(i)] == GlmMlpKind::Moe) want.push_back(i);
    std::vector<int> got;
    for (const auto& item : layers->items()) {
      if (!item.is_number()) reject_quant("routed_experts.layers", "non-numeric element");
      got.push_back(static_cast<int>(item.as_int()));
    }
    if (got != want)
      reject_quant("routed_experts.layers",
                   "must list exactly the main stack's MoE layers");
  }
  // The other classes must be the FP8 release's bytes: no BF16 expert or
  // attention path exists in the engine.
  for (const char* section : {"mtp_layer", "dsa_attention"}) {
    const minijson::Value* sec = qc->find(section);
    if (!sec) continue;
    if (!sec->is_object()) reject_quant(section, "not an object");
    if (const minijson::Value* src = sec->find("source");
        src && (!src->is_string() || src->as_string() != "base"))
      reject_quant(std::string(section) + ".source",
                   "must be \"base\" (the FP8 release); the BF16-repo variant "
                   "is not loadable");
  }
  return GlmExpertFormat::Nvfp4Group16;
}

GlmTextConfig GlmTextConfig::parse(const minijson::Value& tc,
                                   const minijson::Value* quantization_config) {
  if (!tc.is_object()) reject("text_config", "not an object");

  GlmTextConfig c;
  c.hidden_size = require_int(tc, "hidden_size");
  c.vocab_size = require_int(tc, "vocab_size");
  c.num_hidden_layers = require_int(tc, "num_hidden_layers");
  c.rms_norm_eps = require_float(tc, "rms_norm_eps");
  c.tie_word_embeddings = require_bool(tc, "tie_word_embeddings");
  c.hidden_act = require_string(tc, "hidden_act");
  c.swiglu_limit = require_float(tc, "swiglu_limit");

  if (c.hidden_act != "silu")
    reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings)
    reject("tie_word_embeddings",
           "tied embeddings are not implemented (lm_head must be separate)");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");

  // --- attention / MLP class vectors --------------------------------------
  c.layers = require_string_array<GlmLayerKind>(
      tc, "layer_types", [](const std::string& s) -> GlmLayerKind {
        if (s == "linear_attention") return GlmLayerKind::Kda;
        if (s == "deepseek_sparse_attention") return GlmLayerKind::Dsa;
        return static_cast<GlmLayerKind>(-1);
      });
  c.mlps = require_string_array<GlmMlpKind>(
      tc, "mlp_layer_types", [](const std::string& s) -> GlmMlpKind {
        if (s == "dense") return GlmMlpKind::Dense;
        if (s == "sparse") return GlmMlpKind::Moe;
        return static_cast<GlmMlpKind>(-1);
      });
  if (static_cast<int>(c.layers.size()) != c.num_hidden_layers)
    reject("layer_types", "length does not match num_hidden_layers");
  if (static_cast<int>(c.mlps.size()) != c.num_hidden_layers)
    reject("mlp_layer_types", "length does not match num_hidden_layers");

  // indexer_types must be all-"full": the sparse-indexer variant is not
  // implemented and we refuse to guess its geometry.
  require_string_array<int>(tc, "indexer_types",
                            [](const std::string& s) -> int {
                              return s == "full" ? 0 : -1;
                            });

  // The kda/full layer-id lists must partition [0, num_hidden_layers) and
  // agree with layer_types — two redundant encodings of the same fact is a
  // free consistency check on the checkpoint.
  const minijson::Value* la = tc.find("linear_attn_config");
  if (!la || !la->is_object()) reject("linear_attn_config", "missing");
  std::vector<int> kda_ids = require_int_array(*la, "kda_layers");
  std::vector<int> full_ids = require_int_array(*la, "full_attn_layers");
  std::set<int> seen;
  for (int id : kda_ids) {
    if (id < 0 || id >= c.num_hidden_layers)
      reject("kda_layers", "layer id out of range");
    if (!seen.insert(id).second) reject("kda_layers", "duplicate layer id");
  }
  for (int id : full_ids) {
    if (id < 0 || id >= c.num_hidden_layers)
      reject("full_attn_layers", "layer id out of range");
    if (!seen.insert(id).second)
      reject("full_attn_layers", "overlaps kda_layers");
  }
  if (static_cast<int>(seen.size()) != c.num_hidden_layers)
    reject("linear_attn_config",
           "kda_layers + full_attn_layers do not cover all layers");
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    bool listed_full = std::find(full_ids.begin(), full_ids.end(), i) !=
                       full_ids.end();
    bool listed_kda =
        std::find(kda_ids.begin(), kda_ids.end(), i) != kda_ids.end();
    if (listed_full != (c.layers[i] == GlmLayerKind::Dsa) ||
        listed_kda != (c.layers[i] == GlmLayerKind::Kda))
      reject("layer_types",
             "disagrees with linear_attn_config at layer " + std::to_string(i));
  }

  c.first_k_dense_replace = require_int(tc, "first_k_dense_replace");
  if (c.first_k_dense_replace < 0 || c.first_k_dense_replace > c.num_hidden_layers)
    reject("first_k_dense_replace", "out of range");

  // eos_token_id: OPTIONAL (the fixture carries none; HF emits int or
  // array). Ids outside [0, vocab_size) would crash the embed lookup, so
  // they are rejected here, at parse time.
  if (const minijson::Value* eos = tc.find("eos_token_id")) {
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
      if (id < 0 || id >= c.vocab_size)
        reject("eos_token_id",
               "id outside [0, vocab_size) — the embed lookup cannot serve it");
  }
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    GlmMlpKind want = i < c.first_k_dense_replace ? GlmMlpKind::Dense
                                                  : GlmMlpKind::Moe;
    if (c.mlps[i] != want)
      reject("mlp_layer_types",
             "dense layers must be exactly the first first_k_dense_replace "
             "(mismatch at layer " +
                 std::to_string(i) + ")");
  }

  // --- KDA ----------------------------------------------------------------
  c.kda_num_heads = require_int(*la, "num_heads");
  c.kda_head_dim = require_int(*la, "head_dim");
  c.kda_conv_width = require_int(*la, "short_conv_kernel_size");
  c.kda_gate_lower_bound = require_float(*la, "gate_lower_bound");

  // --- MLA / DSA -----------------------------------------------------------
  c.num_attention_heads = require_int(tc, "num_attention_heads");
  c.q_lora_rank = require_int(tc, "q_lora_rank");
  c.kv_lora_rank = require_int(tc, "kv_lora_rank");
  c.qk_nope_head_dim = require_int(tc, "qk_nope_head_dim");
  c.qk_rope_head_dim = require_int(tc, "qk_rope_head_dim");
  c.v_head_dim = require_int(tc, "v_head_dim");
  c.mla_use_nope = require_bool(tc, "mla_use_nope");
  if (c.qk_rope_head_dim != 0)
    reject("qk_rope_head_dim",
           "the engine implements the rope-free MLA path only (this checkpoint "
           "trains 0)");

  // --- indexer -------------------------------------------------------------
  c.index_n_heads = require_int(tc, "index_n_heads");
  c.index_head_dim = require_int(tc, "index_head_dim");
  c.index_kpool = require_int(tc, "index_kpool");
  c.index_topk = require_int(tc, "index_topk");
  c.index_kpool_compress = require_bool(tc, "index_kpool_compress");
  c.index_kpool_always_select_tail =
      require_bool(tc, "index_kpool_always_select_tail");
  c.indexer_rope_interleave = require_bool(tc, "indexer_rope_interleave");

  // --- MLP / MoE ------------------------------------------------------------
  c.intermediate_size = require_int(tc, "intermediate_size");
  c.moe_intermediate_size = require_int(tc, "moe_intermediate_size");
  c.n_routed_experts = require_int(tc, "n_routed_experts");
  c.n_shared_experts = require_int(tc, "n_shared_experts");
  c.num_experts_per_tok = require_int(tc, "num_experts_per_tok");
  c.scoring_func = require_string(tc, "scoring_func");
  c.topk_method = require_string(tc, "topk_method");
  c.norm_topk_prob = require_bool(tc, "norm_topk_prob");
  c.routed_scaling_factor = require_float(tc, "routed_scaling_factor");
  c.n_group = require_int(tc, "n_group");
  c.topk_group = require_int(tc, "topk_group");
  c.moe_router_dtype = require_string(tc, "moe_router_dtype");
  if (c.scoring_func != "sigmoid")
    reject("scoring_func", "only sigmoid is implemented, got " + c.scoring_func);
  if (c.topk_method != "noaux_tc")
    reject("topk_method", "only noaux_tc is implemented, got " + c.topk_method);
  if (c.moe_router_dtype != "float32")
    reject("moe_router_dtype",
           "router math must be float32, got " + c.moe_router_dtype);
  if (c.n_group != 1 || c.topk_group != 1)
    reject("n_group/topk_group", "grouped routing is not implemented");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, n_routed_experts]");
  if (c.n_shared_experts != 1)
    reject("n_shared_experts",
           "only the single shared expert (un-indexed tensor names) is "
           "implemented");

  // --- mHC -------------------------------------------------------------------
  c.mhc = require_bool(tc, "mhc");
  c.hc_mult = require_int(tc, "hc_mult");
  c.hc_sinkhorn_iters = require_int(tc, "hc_sinkhorn_iters");
  c.hc_eps = require_float(tc, "hc_eps");
  if (!c.mhc)
    reject("mhc", "the residual path is hyper-connection-shaped; a non-mHC "
                  "checkpoint is a different model");
  if (c.hc_mult <= 0) reject("hc_mult", "must be positive");
  if (c.hc_sinkhorn_iters < 1)
    reject("hc_sinkhorn_iters", "must be at least 1");

  // --- MTP --------------------------------------------------------------------
  c.num_nextn_predict_layers = require_int(tc, "num_nextn_predict_layers");
  if (c.num_nextn_predict_layers != 0 && c.num_nextn_predict_layers != 1)
    reject("num_nextn_predict_layers",
           "only the single inert draft layer is implemented");

  // Fill the module geometry configs to run their validators (the same
  // invariants M2/M3 pinned) before handing anything to a loader.
  (void)c.kda_config();
  (void)c.dsa_config();
  (void)c.mhc_config();
  (void)c.moe_config();
  c.routed_expert_format = parse_expert_format(quantization_config, c);
  return c;
}

GlmTextConfig GlmTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_json_file(path, "config");
  auto parsed = minijson::parse(json);
  const minijson::Value* tc = parsed.root.find("text_config");
  if (!tc)
    throw std::runtime_error(
        "config " + path + ": missing text_config object");
  return parse(*tc, parsed.root.find("quantization_config"));
}

int GlmTextConfig::num_kda_layers() const {
  int n = 0;
  for (auto k : layers) n += k == GlmLayerKind::Kda;
  return n;
}

int GlmTextConfig::num_dsa_layers() const {
  int n = 0;
  for (auto k : layers) n += k == GlmLayerKind::Dsa;
  return n;
}

KdaConfig GlmTextConfig::kda_config() const {
  KdaConfig k;
  k.hidden = hidden_size;
  k.heads = kda_num_heads;
  k.head_dim = kda_head_dim;
  k.conv_width = kda_conv_width;
  k.lower_bound = kda_gate_lower_bound;
  k.num_kda_layers = num_kda_layers();
  // spec_width (MTP draft reserve) is a runtime placement decision, not a
  // checkpoint fact; the loader overrides it from its placement parameters.
  k.spec_width = 3;
  k.tp_size = 1;
  KdaConfig::validate_config(k);
  return k;
}

DsaConfig GlmTextConfig::dsa_config() const {
  DsaConfig d;
  d.hidden = hidden_size;
  d.num_heads = num_attention_heads;
  d.q_lora_rank = q_lora_rank;
  d.kv_lora_rank = kv_lora_rank;
  d.qk_nope_head_dim = qk_nope_head_dim;
  d.qk_rope_head_dim = qk_rope_head_dim;
  d.v_head_dim = v_head_dim;
  d.index_n_heads = index_n_heads;
  d.index_head_dim = index_head_dim;
  d.index_topk = index_topk;
  d.index_kpool = index_kpool;
  d.always_select_tail = index_kpool_always_select_tail ? 1 : 0;
  d.num_dsa_layers = num_dsa_layers();
  d.block_tokens = 128;
  d.tp_size = 1;
  d.rms_norm_eps = rms_norm_eps;
  DsaConfig::validate_config(d);
  return d;
}

GlmMhcConfig GlmTextConfig::mhc_config() const {
  GlmMhcConfig m;
  m.hc_mult = hc_mult;
  m.hidden = hidden_size;
  m.sinkhorn_iters = hc_sinkhorn_iters;
  m.hc_eps = hc_eps;
  m.norm_eps = rms_norm_eps;
  GlmMhcConfig::validate_config(m);
  return m;
}

GlmMoeConfig GlmTextConfig::moe_config() const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = moe_intermediate_size;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = swiglu_limit;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
