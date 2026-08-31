#include "models/glm_config.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <format>
#include <set>

namespace dgpp {

namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(
      std::format("GLM text_config.{}: {}", field, why));
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

GlmTextConfig GlmTextConfig::parse(const minijson::Value& tc) {
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
  return c;
}

GlmTextConfig GlmTextConfig::from_json_file(const std::string& path) {
  std::string json = [&] {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
      throw std::runtime_error("cannot open config " + path + ": " +
                               std::strerror(errno));
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string s(static_cast<size_t>(len), '\0');
    if (len > 0 &&
        std::fread(s.data(), 1, static_cast<size_t>(len), f) !=
            static_cast<size_t>(len)) {
      std::fclose(f);
      throw std::runtime_error("short read on config " + path);
    }
    std::fclose(f);
    return s;
  }();
  auto parsed = minijson::parse(json);
  const minijson::Value* tc = parsed.root.find("text_config");
  if (!tc)
    throw std::runtime_error(
        "config " + path + ": missing text_config object");
  return parse(*tc);
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
