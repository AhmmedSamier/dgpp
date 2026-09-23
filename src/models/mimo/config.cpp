#include "models/mimo/config.hpp"

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
  throw std::runtime_error(std::format("MiMo-V2 config.{}: {}", field, why));
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
// A [num_hidden_layers] 0/1 list.
std::vector<uint8_t> require_pattern(const minijson::Value& v, std::string_view field, int layers) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<uint8_t> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    const int64_t x = item.as_int();
    if (x != 0 && x != 1) reject(field, "elements must be 0 or 1");
    out.push_back(static_cast<uint8_t>(x));
  }
  if (static_cast<int>(out.size()) != layers) reject(field, "length differs from num_hidden_layers");
  return out;
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

// The release's contract: quant_method fp8 (e4m3, dynamic activations —
// the engine runs bf16 activations against the dequantized weights, the
// pinned numerics of every fp8 family here), 128 x 128 weight blocks, the
// experts stored MXFP4 on 32-blocks, and the ignore list naming every
// layer's o_proj (BF16) — a module ignored that the loader does not expect
// in BF16 would be bound wrongly, so the list is checked both ways.
void parse_quantization(const minijson::Value& root, MimoTextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the fp8 + MXFP4 release as shipped; an unquantized "
           "MiMoV2 checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  const std::string method = optional_string(*qc, "quant_method", "");
  if (method != "fp8") reject("quantization_config.quant_method", "only fp8 is implemented, got '" + method + "'");
  if (const std::string fmt = optional_string(*qc, "fmt", "e4m3"); fmt != "e4m3")
    reject("quantization_config.fmt", "only e4m3 is implemented, got '" + fmt + "'");
  if (const std::string act = optional_string(*qc, "activation_scheme", "dynamic"); act != "dynamic")
    reject("quantization_config.activation_scheme", "only the dynamic scheme (no activation scales) is implemented");
  if (const minijson::Value* wb = qc->find("weight_block_size"); wb && !wb->is_null()) {
    if (!wb->is_array()) reject("quantization_config.weight_block_size", "not an array");
    std::vector<int> dims;
    for (const auto& item : wb->items()) {
      if (!item.is_number()) reject("quantization_config.weight_block_size", "non-numeric element");
      dims.push_back(static_cast<int>(item.as_int()));
    }
    if (dims.size() != 2 || dims[0] != 128 || dims[1] != 128)
      reject("quantization_config.weight_block_size", "the fp8 kernels implement 128 x 128 blocks");
  }
  c.fp8_block = 128;
  const std::string store = optional_string(*qc, "store_dtype", "");
  if (store != "mxfp4")
    reject("quantization_config.store_dtype", "only the MXFP4 expert store is implemented, got '" + store + "'");
  c.mxfp4_block = optional_int(*qc, "mxfp4_block_size", 32);
  if (c.mxfp4_block != 32) reject("quantization_config.mxfp4_block_size", "the MXFP4 kernels implement 32");
  // The ignore list: every main layer's o_proj (and the draft's, which the
  // release names oddly); the engine expects BF16 there and nowhere else.
  std::set<std::string> ignore;
  if (const minijson::Value* ig = qc->find("ignored_layers"); ig && ig->is_array())
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignored_layers", "non-string entry");
      ignore.insert(std::string(item.as_string()));
    }
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const std::string w = "model.layers." + std::to_string(l) + ".self_attn.o_proj";
    if (!ignore.count(w))
      reject("quantization_config.ignored_layers", "'" + w + "' is not ignored — the engine expects a BF16 o_proj");
    ignore.erase(w);
  }
  // The draft's o_proj entry: the release writes "model.decoder.self_attn.o_proj".
  for (const std::string& i : ignore)
    if (i.find("self_attn.o_proj") == std::string::npos)
      reject("quantization_config.ignored_layers",
             "unexpected ignored module '" + i + "' (a BF16 class the loader does not implement)");
}

}  // namespace

MimoTextConfig MimoTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  MimoTextConfig c;
  const std::string model_type = optional_string(root, "model_type", "mimo_v2");
  if (model_type != "mimo_v2") reject("model_type", "expected mimo_v2, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(optional_double(root, "layernorm_epsilon", optional_double(root, "rms_norm_eps", 1e-6)));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = optional_string(root, "hidden_act", "silu");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 128 != 0)
    reject("hidden_size", "must be a positive multiple of 128 (the fp8 scale block)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (!(c.rms_norm_eps > 0)) reject("layernorm_epsilon", "must be positive");
  c.swa_layer = require_pattern(root, "hybrid_layer_pattern", c.num_hidden_layers);
  c.moe_layer = require_pattern(root, "moe_layer_freq", c.num_hidden_layers);
  if (const minijson::Value* hb = root.find("hybrid_block_size"); hb && !hb->is_null())
    reject("hybrid_block_size", "only the explicit hybrid_layer_pattern is implemented");

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
  c.v_head_dim = optional_int(root, "v_head_dim", c.head_dim);
  c.attention_bias = optional_bool(root, "attention_bias", false);
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && rp->is_object()) {
    if (optional_string(*rp, "rope_type", optional_string(*rp, "type", "default")) != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented");
    c.rope_theta = optional_double(*rp, "rope_theta", c.rope_theta);
  }
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  c.swa_rope_theta = optional_double(root, "swa_rope_theta", c.rope_theta);
  if (optional_int(root, "swa_num_attention_heads", c.num_attention_heads) != c.num_attention_heads)
    reject("swa_num_attention_heads", "the SWA and GA layers must share the query head count");
  c.swa_num_key_value_heads = optional_int(root, "swa_num_key_value_heads", c.num_key_value_heads);
  if (optional_int(root, "swa_head_dim", c.head_dim) != c.head_dim)
    reject("swa_head_dim", "the SWA and GA layers must share the qk head dim");
  if (optional_int(root, "swa_v_head_dim", c.v_head_dim) != c.v_head_dim)
    reject("swa_v_head_dim", "the SWA and GA layers must share the v head dim");
  c.sliding_window = optional_int(root, "sliding_window", optional_int(root, "sliding_window_size", 0));
  if (const int alt = optional_int(root, "sliding_window_size", c.sliding_window); alt != c.sliding_window)
    reject("sliding_window_size", "differs from sliding_window");
  if (const int chunk = optional_int(root, "attention_chunk_size", c.sliding_window); chunk != c.sliding_window)
    reject("attention_chunk_size", "differs from sliding_window (chunked attention is not implemented)");
  c.swa_sink = optional_bool(root, "add_swa_attention_sink_bias", false);
  c.full_sink = optional_bool(root, "add_full_attention_sink_bias", false);
  c.attention_value_scale = static_cast<float>(optional_double(root, "attention_value_scale", 1.0));
  if (const std::string layout = optional_string(root, "attention_projection_layout", "fused_qkv"); layout != "fused_qkv")
    reject("attention_projection_layout", "only the fused qkv_proj layout is implemented, got " + layout);
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 || c.swa_num_key_value_heads <= 0)
    reject("num_attention_heads", "head counts must be positive");
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.num_attention_heads % c.swa_num_key_value_heads != 0)
    reject("swa_num_key_value_heads", "must divide num_attention_heads");
  if (c.swa_num_key_value_heads % c.num_key_value_heads != 0)
    reject("swa_num_key_value_heads", "must be a multiple of num_key_value_heads (the qkv_proj chunk count)");
  if (c.head_dim != 192 || c.v_head_dim != 128)
    reject("head_dim", "the attention kernels implement qk 192 / v 128 wide heads");
  if (c.attention_bias) reject("attention_bias", "biased projections are not implemented");
  {
    const double factor = optional_double(root, "partial_rotary_factor", 1.0);
    // The reference: int(head_dim * factor) — truncation.
    const int rd = static_cast<int>(static_cast<double>(c.head_dim) * factor);
    if (rd <= 0 || rd % 2 != 0 || rd > c.head_dim)
      reject("partial_rotary_factor", "rotary dim must be a positive even integer within the head");
    c.rotary_dim = rd;
  }
  if (c.rotary_dim != 64) reject("partial_rotary_factor", "the attention kernels implement a 64-wide rotary slice");
  if (!(c.rope_theta > 0) || !(c.swa_rope_theta > 0)) reject("rope_theta", "must be positive");
  if (c.num_swa_layers() > 0 && c.sliding_window <= 0) reject("sliding_window", "must be positive with SWA layers");
  if (c.sliding_window % 32 != 0) reject("sliding_window", "must be a multiple of the attention tile (32)");
  if (!(c.attention_value_scale > 0) || !std::isfinite(c.attention_value_scale))
    reject("attention_value_scale", "must be a positive finite number");

  // --- MLP / MoE --------------------------------------------------------------
  c.intermediate_size = require_int(root, "intermediate_size");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.n_routed_experts = require_int(root, "n_routed_experts");
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  c.routed_scaling_factor = static_cast<float>(optional_double(root, "routed_scaling_factor", 1.0));
  c.n_group = optional_int(root, "n_group", 1);
  c.topk_group = optional_int(root, "topk_group", 1);
  if (c.intermediate_size <= 0 || c.intermediate_size % 128 != 0)
    reject("intermediate_size", "must be a positive multiple of 128 (the fp8 scale block)");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 32 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 32 (the MXFP4 block)");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
    reject("n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 ||
      c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(n_routed_experts, 16)]");
  if (optional_int(root, "n_shared_experts", 0) != 0)
    reject("n_shared_experts", "the MiMo-V2 MoE has no shared expert (the chain implements none)");
  if (c.n_group != 1 || c.topk_group != 1)
    reject("n_group", "group-limited routing (n_group/topk_group != 1) is not implemented");
  if (!(c.routed_scaling_factor > 0)) reject("routed_scaling_factor", "must be positive");
  if (const std::string sf = optional_string(root, "scoring_func", "sigmoid"); sf != "sigmoid")
    reject("scoring_func", "only sigmoid is implemented");
  if (const std::string tm = optional_string(root, "topk_method", "noaux_tc"); tm != "noaux_tc")
    reject("topk_method", "only noaux_tc is implemented");
  if (const std::string rd = optional_string(root, "moe_router_dtype", "bfloat16"); rd != "bfloat16" && rd != "float32")
    reject("moe_router_dtype", "the router runs the bf16 gate in fp32");
  if (c.num_moe_layers() == 0) reject("moe_layer_freq", "no MoE layer");

  // --- MTP --------------------------------------------------------------------
  c.num_nextn_predict_layers = optional_int(root, "num_nextn_predict_layers", 0);
  if (c.num_nextn_predict_layers < 0) reject("num_nextn_predict_layers", "must be >= 0");
  c.mtp_layers_loaded = c.num_nextn_predict_layers > 0 ? 1 : 0;

  parse_quantization(root, c);
  return c;
}

MimoTextConfig MimoTextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

GlmMoeConfig MimoTextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // MiMoV2MLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
