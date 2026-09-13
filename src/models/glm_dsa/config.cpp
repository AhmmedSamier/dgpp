#include "models/glm_dsa/config.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <regex>
#include <stdexcept>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("GLM-5.3 config.{}: {}", field, why));
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
// A field that must be absent or null (a feature the engine does not
// implement and must not silently ignore).
void require_absent(const minijson::Value& v, std::string_view field, std::string_view why) {
  const minijson::Value* f = v.find(field);
  if (f && !f->is_null()) reject(field, why);
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

// --- the indexer schedule ----------------------------------------------------

// transformers' rule: layer i is "full" when max(i - offset + 1, 0) % freq == 0.
std::vector<uint8_t> schedule_from_freq(int layers, int freq, int offset) {
  std::vector<uint8_t> full(static_cast<size_t>(layers), 0);
  for (int i = 0; i < layers; ++i)
    full[static_cast<size_t>(i)] = (std::max(i - offset + 1, 0) % freq) == 0;
  return full;
}

void parse_indexer_schedule(const minijson::Value& root, GlmDsaTextConfig& c) {
  const int L = c.num_hidden_layers;
  c.index_topk_freq = optional_int(root, "index_topk_freq", 1);
  c.index_skip_topk_offset = optional_int(root, "index_skip_topk_offset", 2);
  if (c.index_topk_freq < 1) reject("index_topk_freq", "must be >= 1");
  if (c.index_skip_topk_offset < 0) reject("index_skip_topk_offset", "must be >= 0");
  std::vector<uint8_t> from_freq = schedule_from_freq(L, c.index_topk_freq, c.index_skip_topk_offset);
  // An explicit pattern ("FSSF...") overrides the freq/offset schedule.
  bool pattern_given = false;
  std::vector<uint8_t> from_pattern;
  if (const minijson::Value* p = root.find("index_topk_pattern"); p && !p->is_null()) {
    if (!p->is_string()) reject("index_topk_pattern", "not a string");
    const std::string_view s = p->as_string();
    if (static_cast<int>(s.size()) != L) reject("index_topk_pattern", "length != num_hidden_layers");
    for (char ch : s) {
      if (ch != 'F' && ch != 'S') reject("index_topk_pattern", "characters must be F or S");
      from_pattern.push_back(ch == 'F');
    }
    pattern_given = true;
  }
  const std::vector<uint8_t>& derived = pattern_given ? from_pattern : from_freq;
  // The explicit list, when present, must agree with the derived schedule:
  // the two are one fact written twice and the loader trusts neither alone.
  if (const minijson::Value* it = root.find("indexer_types"); it && !it->is_null()) {
    if (!it->is_array()) reject("indexer_types", "not an array");
    if (static_cast<int>(it->items().size()) != L) reject("indexer_types", "length != num_hidden_layers");
    std::vector<uint8_t> listed;
    for (const auto& item : it->items()) {
      if (!item.is_string()) reject("indexer_types", "non-string entry");
      const std::string_view s = item.as_string();
      if (s == "full") listed.push_back(1);
      else if (s == "shared") listed.push_back(0);
      else reject("indexer_types", "entries must be 'full' or 'shared', got '" + std::string(s) + "'");
    }
    if (listed != derived)
      reject("indexer_types",
             pattern_given ? "disagrees with index_topk_pattern"
                           : "disagrees with the index_topk_freq / index_skip_topk_offset schedule");
    c.indexer_full = std::move(listed);
  } else {
    c.indexer_full = derived;
  }
  if (c.indexer_full.empty() || c.indexer_full[0] == 0)
    reject("indexer_types", "layer 0 must be 'full' (a shared layer needs a previous selection)");
}

// --- the quantization contract --------------------------------------------

// One compressed-tensors matching rule: a `re:` regex anchored at the start
// of the module name, or an exact module name.
struct MatchRule {
  bool is_regex = false;
  std::regex re;
  std::string literal;
  bool matches(const std::string& module) const {
    return is_regex ? std::regex_search(module, re) : module == literal;
  }
};

MatchRule make_rule(std::string_view field, const std::string& target) {
  MatchRule r;
  if (target.rfind("re:", 0) == 0) {
    r.is_regex = true;
    try {
      r.re = std::regex("^(?:" + target.substr(3) + ")", std::regex::ECMAScript);
    } catch (const std::regex_error& e) {
      reject(field, "unparseable regex '" + target + "': " + e.what());
    }
  } else {
    if (target == "Linear")
      reject(field, "class-name targets ('Linear') are not implemented; the file must name modules");
    r.literal = target;
  }
  return r;
}

struct QuantGroup {
  std::vector<MatchRule> targets;
  int bits = 0;
};

struct QuantRules {
  std::vector<QuantGroup> groups;
  std::vector<MatchRule> ignore;
  // The bits a module is stored at under these rules: 0 = not quantized.
  int bits_of(const std::string& module) const {
    for (const MatchRule& r : ignore)
      if (r.matches(module)) return 0;
    int bits = 0;
    for (const QuantGroup& g : groups)
      for (const MatchRule& r : g.targets)
        if (r.matches(module)) {
          if (bits != 0 && bits != g.bits)
            reject("quantization_config.config_groups", "module '" + module + "' matches two groups");
          bits = g.bits;
        }
    return bits;
  }
};

const char* kAttentionModules[] = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"};
const char* kMlpModules[] = {"gate_proj", "up_proj", "down_proj"};

QuantRules parse_quantization(const minijson::Value& root, GlmDsaTextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the compressed-tensors pack-quantized "
           "int4/int8 checkpoint; a BF16 GlmMoeDsa checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  if (const std::string m = optional_string(*qc, "quant_method", ""); m != "compressed-tensors")
    reject("quantization_config.quant_method", "only compressed-tensors is implemented, got '" + m + "'");
  if (const std::string f = optional_string(*qc, "format", ""); f != "pack-quantized")
    reject("quantization_config.format", "only pack-quantized is implemented, got '" + f + "'");
  if (const std::string s = optional_string(*qc, "quantization_status", "compressed"); s != "compressed")
    reject("quantization_config.quantization_status", "must be compressed, got '" + s + "'");
  require_absent(*qc, "kv_cache_scheme", "a KV cache scheme is not implemented (the engine picks its own latent format)");
  // A transform (a Hadamard rotation of the weights) would change what the
  // packed codes mean; an empty object is the only accepted value.
  if (const minijson::Value* t = qc->find("transform_config"); t && !t->is_null()) {
    if (!t->is_object() || !t->members().empty())
      reject("quantization_config.transform_config", "weight transforms are not implemented");
  }
  if (const minijson::Value* s = qc->find("sparsity_config"); s && !s->is_null()) {
    if (!s->is_object() || !s->members().empty())
      reject("quantization_config.sparsity_config", "sparsity is not implemented");
  }
  const minijson::Value* groups = qc->find("config_groups");
  if (!groups || !groups->is_object() || groups->members().empty())
    reject("quantization_config.config_groups", "missing");
  QuantRules rules;
  int group_size = 0;
  for (const auto& m : groups->members()) {
    const std::string gf = "quantization_config.config_groups." + m.key;
    const minijson::Value& g = m.value;
    if (!g.is_object()) reject(gf, "group is not an object");
    if (const std::string f = optional_string(g, "format", "pack-quantized"); f != "pack-quantized")
      reject(gf + ".format", "only pack-quantized is implemented, got '" + f + "'");
    require_absent(g, "input_activations", "activation quantization is not implemented (W4A16 / W8A16 only)");
    require_absent(g, "output_activations", "activation quantization is not implemented (W4A16 / W8A16 only)");
    const minijson::Value* w = g.find("weights");
    if (!w || !w->is_object()) reject(gf + ".weights", "missing");
    QuantGroup qg;
    qg.bits = require_int(*w, "num_bits");
    if (qg.bits != 4 && qg.bits != 8) reject(gf + ".weights.num_bits", "the packed-int cores implement 4 and 8");
    if (const std::string t = optional_string(*w, "type", "int"); t != "int")
      reject(gf + ".weights.type", "must be int, got '" + t + "'");
    if (!optional_bool(*w, "symmetric", true)) reject(gf + ".weights.symmetric", "asymmetric (zero-point) weights are not implemented");
    if (const std::string s = optional_string(*w, "strategy", "group"); s != "group")
      reject(gf + ".weights.strategy", "only the group strategy is implemented, got '" + s + "'");
    if (optional_bool(*w, "dynamic", false)) reject(gf + ".weights.dynamic", "dynamic weight quantization is not a checkpoint format");
    require_absent(*w, "zp_dtype", "zero points are not implemented (symmetric only)");
    require_absent(*w, "actorder", "activation ordering is not implemented");
    require_absent(*w, "block_structure", "block structures are not implemented");
    const int gs = require_int(*w, "group_size");
    if (gs != 64) reject(gf + ".weights.group_size", "the packed-int cores implement 64, got " + std::to_string(gs));
    if (group_size != 0 && gs != group_size) reject(gf + ".weights.group_size", "differs between groups");
    group_size = gs;
    const minijson::Value* t = g.find("targets");
    if (!t || !t->is_array() || t->items().empty()) reject(gf + ".targets", "missing");
    for (const auto& item : t->items()) {
      if (!item.is_string()) reject(gf + ".targets", "non-string entry");
      qg.targets.push_back(make_rule(gf + ".targets", std::string(item.as_string())));
    }
    rules.groups.push_back(std::move(qg));
  }
  c.packed_group_size = group_size;
  if (const minijson::Value* ig = qc->find("ignore"); ig && !ig->is_null()) {
    if (!ig->is_array()) reject("quantization_config.ignore", "not an array");
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      rules.ignore.push_back(make_rule("quantization_config.ignore", std::string(item.as_string())));
    }
  }
  return rules;
}

// Applies the file's rules to the module names the table will emit and
// checks they describe the one shape the loader implements: a contiguous
// range of MoE main layers whose attention (all five projections at one
// width), shared expert and routed experts are packed, every other module
// BF16.
void derive_packed_shape(const QuantRules& rules, GlmDsaTextConfig& c) {
  const int L = c.num_hidden_layers;
  const int E = c.n_routed_experts;
  auto layer_prefix = [](int l) { return "model.layers." + std::to_string(l) + "."; };
  auto expect_plain = [&](const std::string& module) {
    if (rules.bits_of(module) != 0)
      reject("quantization_config", "'" + module + "' is quantized, and the loader expects BF16 there");
  };
  // Globals and the draft are BF16.
  expect_plain("lm_head");
  expect_plain("model.embed_tokens");
  expect_plain("model.norm");
  const int sample_experts[] = {0, 1, E / 2, E - 1};
  int begin = -1, end = -1;
  for (int l = 0; l < L; ++l) {
    const std::string p = layer_prefix(l);
    int attn = -1;
    for (const char* m : kAttentionModules) {
      const int b = rules.bits_of(p + "self_attn." + m);
      if (attn == -1) attn = b;
      else if (attn != b)
        reject("quantization_config", "layer " + std::to_string(l) + ": the five attention projections must share one width");
    }
    expect_plain(p + "input_layernorm");
    expect_plain(p + "post_attention_layernorm");
    expect_plain(p + "self_attn.q_a_layernorm");
    expect_plain(p + "self_attn.kv_a_layernorm");
    if (c.owns_indexer(l))
      for (const char* m : {"wq_b", "wk", "weights_proj", "k_norm"}) expect_plain(p + "self_attn.indexer." + m);
    int shared = -1, routed = -1;
    if (c.is_moe_layer(l)) {
      expect_plain(p + "mlp.gate");
      for (const char* m : kMlpModules) {
        const int b = rules.bits_of(p + "mlp.shared_experts." + m);
        if (shared == -1) shared = b;
        else if (shared != b) reject("quantization_config", "layer " + std::to_string(l) + ": the shared expert's projections must share one width");
      }
      for (int e : sample_experts)
        for (const char* m : kMlpModules) {
          const int b = rules.bits_of(p + "mlp.experts." + std::to_string(e) + "." + m);
          if (routed == -1) routed = b;
          else if (routed != b) reject("quantization_config", "layer " + std::to_string(l) + ": the routed experts' projections must share one width");
        }
    } else {
      for (const char* m : kMlpModules) expect_plain(p + "mlp." + m);
      if (attn != 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": a packed attention on a dense layer is not implemented");
    }
    const bool packed = attn != 0;
    if (packed) {
      if (shared == 0 || routed == 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": packed attention but BF16 experts is not implemented");
      if (begin == -1) begin = l;
      else if (end != -1) reject("quantization_config", "the packed layers must be one contiguous range");
      if (c.attention_bits == 0 || begin == l) {
        c.attention_bits = attn;
        c.shared_bits = shared;
        c.expert_bits = routed;
      } else if (attn != c.attention_bits || shared != c.shared_bits || routed != c.expert_bits) {
        reject("quantization_config", "layer " + std::to_string(l) + ": widths differ from the first packed layer's");
      }
    } else {
      if (shared > 0 || routed > 0)
        reject("quantization_config", "layer " + std::to_string(l) + ": packed experts under BF16 attention is not implemented");
      if (begin != -1 && end == -1) end = l;
    }
  }
  if (begin == -1) reject("quantization_config", "no layer is packed — a BF16 expert path is not implemented");
  if (end == -1) end = L;
  c.packed_layer_begin = begin;
  c.packed_layer_end = end;
  if (c.mtp_layer() >= 0) {
    const std::string p = layer_prefix(c.mtp_layer());
    for (const char* m : kAttentionModules) expect_plain(p + "self_attn." + m);
    for (const char* m : kMlpModules) expect_plain(p + "mlp.shared_experts." + m);
    for (int e : sample_experts)
      for (const char* m : kMlpModules) expect_plain(p + "mlp.experts." + std::to_string(e) + "." + m);
    for (const char* m : {"enorm", "hnorm", "eh_proj", "shared_head.norm", "mlp.gate"}) expect_plain(p + m);
  }
}

}  // namespace

GlmDsaTextConfig GlmDsaTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  GlmDsaTextConfig c;
  const std::string model_type = optional_string(root, "model_type", "glm_moe_dsa");
  if (model_type != "glm_moe_dsa") reject("model_type", "expected glm_moe_dsa, got " + model_type);
  if (root.find("text_config"))
    reject("text_config", "a nested text_config is the GLM-5.3-Flash layout, not glm_moe_dsa's");

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  c.first_k_dense_replace = optional_int(root, "first_k_dense_replace", 0);
  if (c.hidden_size <= 0 || c.hidden_size % 64 != 0)
    reject("hidden_size", "must be a positive multiple of 64 (the packed-int cores' group)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (c.first_k_dense_replace < 0 || c.first_k_dense_replace > c.num_hidden_layers)
    reject("first_k_dense_replace", "outside [0, num_hidden_layers]");
  if (optional_int(root, "moe_layer_freq", 1) != 1) reject("moe_layer_freq", "only 1 is implemented");
  if (const minijson::Value* lt = root.find("layer_types"); lt && !lt->is_null()) {
    if (!lt->is_array() || static_cast<int>(lt->items().size()) != c.num_hidden_layers)
      reject("layer_types", "must list every layer");
    for (const auto& item : lt->items())
      if (!item.is_string() || item.as_string() != "deepseek_sparse_attention")
        reject("layer_types", "every layer is deepseek_sparse_attention in this family");
  }
  if (const minijson::Value* mt = root.find("mlp_layer_types"); mt && !mt->is_null()) {
    if (!mt->is_array() || static_cast<int>(mt->items().size()) != c.num_hidden_layers)
      reject("mlp_layer_types", "must list every layer");
    int i = 0;
    for (const auto& item : mt->items()) {
      const std::string_view s = item.is_string() ? item.as_string() : std::string_view{};
      const std::string_view want = i < c.first_k_dense_replace ? "dense" : "sparse";
      if (s != want)
        reject("mlp_layer_types", "layer " + std::to_string(i) + " must be '" + std::string(want) +
                                      "' (first_k_dense_replace)");
      ++i;
    }
  }

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

  // --- MLA attention ---------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  if (const int kv = optional_int(root, "num_key_value_heads", c.num_attention_heads); kv != c.num_attention_heads)
    reject("num_key_value_heads", "MLA has one latent per token: num_key_value_heads must equal num_attention_heads");
  c.q_lora_rank = require_int(root, "q_lora_rank");
  c.kv_lora_rank = require_int(root, "kv_lora_rank");
  c.qk_nope_head_dim = require_int(root, "qk_nope_head_dim");
  c.qk_rope_head_dim = require_int(root, "qk_rope_head_dim");
  c.v_head_dim = require_int(root, "v_head_dim");
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.rope_interleave = optional_bool(root, "rope_interleave", true);
  if (c.num_attention_heads <= 0) reject("num_attention_heads", "must be positive");
  if (c.q_lora_rank <= 0 || c.q_lora_rank % 64 != 0)
    reject("q_lora_rank", "must be a positive multiple of 64 (a packed q_b input)");
  if (c.kv_lora_rank <= 0 || c.kv_lora_rank % 64 != 0)
    reject("kv_lora_rank", "must be a positive multiple of 64 (a packed kv_b input)");
  if (c.qk_nope_head_dim <= 0) reject("qk_nope_head_dim", "must be positive");
  if (c.qk_rope_head_dim != 0 && c.qk_rope_head_dim != 64)
    reject("qk_rope_head_dim", "the MLA kernels implement a 0- or 64-wide rope key");
  if (c.v_head_dim <= 0) reject("v_head_dim", "must be positive");
  if (const int qk = optional_int(root, "qk_head_dim", c.qk_head_dim()); qk != c.qk_head_dim())
    reject("qk_head_dim", "must equal qk_nope_head_dim + qk_rope_head_dim");
  if (c.attention_bias) reject("attention_bias", "biased MLA projections are not implemented");
  if (!c.rope_interleave) reject("rope_interleave", "only the interleaved (DeepSeek) RoPE is implemented");
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && !rp->is_null()) {
    if (!rp->is_object()) reject("rope_parameters", "not an object");
    if (optional_string(*rp, "rope_type", "default") != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented");
    c.rope_theta = optional_double(*rp, "rope_theta", c.rope_theta);
  }
  if (!(c.rope_theta > 0)) reject("rope_theta", "must be positive");

  // --- DSA indexer ------------------------------------------------------------
  c.index_n_heads = require_int(root, "index_n_heads");
  c.index_head_dim = require_int(root, "index_head_dim");
  c.index_topk = require_int(root, "index_topk");
  c.indexer_rope_interleave = optional_bool(root, "indexer_rope_interleave", true);
  c.index_share_for_mtp_iteration = optional_bool(root, "index_share_for_mtp_iteration", false);
  if (c.index_n_heads <= 0) reject("index_n_heads", "must be positive");
  if (c.index_head_dim != 128) reject("index_head_dim", "the indexer implements 128 (Hadamard-128 is pinned)");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0)
    reject("index_topk", "must be a power of two (the select networks)");
  if (c.qk_rope_head_dim > c.index_head_dim) reject("qk_rope_head_dim", "exceeds index_head_dim");
  if (!c.indexer_rope_interleave) reject("indexer_rope_interleave", "only the interleaved indexer RoPE is implemented");
  require_absent(root, "index_kpool", "pooled indexing is the GLM-5.3-Flash layout; this family selects per token");
  require_absent(root, "index_kpool_compress", "pooled indexing is the GLM-5.3-Flash layout");
  require_absent(root, "index_kpool_always_select_tail", "pooled indexing is the GLM-5.3-Flash layout");
  parse_indexer_schedule(root, c);

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
  if (c.intermediate_size <= 0 || c.intermediate_size % 64 != 0)
    reject("intermediate_size", "must be a positive multiple of 64");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 64 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 64 (a packed down input)");
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
  if (const std::string rd = optional_string(root, "moe_router_dtype", "float32"); rd != "float32")
    reject("moe_router_dtype", "only float32 routing is implemented");
  require_absent(root, "swiglu_limit", "clamped swiglu is the GLM-5.3-Flash expert; this family's experts are unclamped");

  // --- MTP --------------------------------------------------------------------
  c.num_nextn_predict_layers = optional_int(root, "num_nextn_predict_layers", 0);
  if (c.num_nextn_predict_layers != 0 && c.num_nextn_predict_layers != 1)
    reject("num_nextn_predict_layers", "only the single draft layer is implemented");
  if (c.mtp_layer() >= 0 && !c.is_moe_layer(c.mtp_layer()))
    reject("num_nextn_predict_layers", "the draft layer carries the MoE (first_k_dense_replace)");

  const QuantRules rules = parse_quantization(root, c);
  derive_packed_shape(rules, c);
  if (c.packed_layer_begin < c.first_k_dense_replace)
    reject("quantization_config", "a packed dense layer is not implemented");
  return c;
}

GlmDsaTextConfig GlmDsaTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

GlmMoeConfig GlmDsaTextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // GlmMoeDsaMLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
