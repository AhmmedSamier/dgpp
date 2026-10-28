#include "models/llama/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<LlamaExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         LlamaWeightClass cls, int layer) {
  out.push_back(LlamaExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              LlamaWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// The ModelOpt NVFP4 triple for one projection `name` (the MODULE path,
// e.g. model.layers.0.self_attn.q_proj): `name.weight` (e2m1 pairs, two
// per byte — [N, K/2] U8), `name.weight_scale` (one e4m3 per 16 elements —
// [N, K/16]) and `name.weight_scale_2` (the fp32 global, raw amax/(6*448)).
void add_fp4(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
             LlamaWeightClass cls, int layer) {
  add(out, name + ".weight", DType::U8, {rows, cols / 2}, cls, layer);
  add(out, name + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer);
  add(out, name + ".weight_scale_2", DType::F32, {}, cls, layer);  // ModelOpt exports the global as a scalar
}

}  // namespace

std::vector<LlamaExpectedTensor> llama_expected_layer_tensors(const LlamaTextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("llama_expected_layer_tensors: layer out of range");
  const std::string p = "model.layers." + std::to_string(layer) + ".";
  const int64_t H = cfg.hidden_size, d = cfg.head_dim, I = cfg.intermediate_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  TensorList out;
  add_bf16(out, p + "input_layernorm.weight", {H}, LlamaWeightClass::LayerNorm, layer);
  if (cfg.weights_nvfp4) {
    add_fp4(out, p + "self_attn.q_proj", qh * d, H, LlamaWeightClass::Attention, layer);
    add_fp4(out, p + "self_attn.k_proj", kvh * d, H, LlamaWeightClass::Attention, layer);
    add_fp4(out, p + "self_attn.v_proj", kvh * d, H, LlamaWeightClass::Attention, layer);
    add_fp4(out, p + "self_attn.o_proj", H, qh * d, LlamaWeightClass::Attention, layer);
    add_bf16(out, p + "post_attention_layernorm.weight", {H}, LlamaWeightClass::LayerNorm, layer);
    add_fp4(out, p + "mlp.gate_proj", I, H, LlamaWeightClass::DenseMlp, layer);
    add_fp4(out, p + "mlp.up_proj", I, H, LlamaWeightClass::DenseMlp, layer);
    add_fp4(out, p + "mlp.down_proj", H, I, LlamaWeightClass::DenseMlp, layer);
    return out;
  }
  add_bf16(out, p + "self_attn.q_proj.weight", {qh * d, H}, LlamaWeightClass::Attention, layer);
  add_bf16(out, p + "self_attn.k_proj.weight", {kvh * d, H}, LlamaWeightClass::Attention, layer);
  add_bf16(out, p + "self_attn.v_proj.weight", {kvh * d, H}, LlamaWeightClass::Attention, layer);
  add_bf16(out, p + "self_attn.o_proj.weight", {H, qh * d}, LlamaWeightClass::Attention, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {H}, LlamaWeightClass::LayerNorm, layer);
  add_bf16(out, p + "mlp.gate_proj.weight", {I, H}, LlamaWeightClass::DenseMlp, layer);
  add_bf16(out, p + "mlp.up_proj.weight", {I, H}, LlamaWeightClass::DenseMlp, layer);
  add_bf16(out, p + "mlp.down_proj.weight", {H, I}, LlamaWeightClass::DenseMlp, layer);
  return out;
}

std::vector<LlamaExpectedTensor> llama_expected_global_tensors(const LlamaTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, LlamaWeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, LlamaWeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, LlamaWeightClass::LmHead, -1);
  return out;
}

std::vector<LlamaExpectedTensor> llama_expected_text_tensors(const LlamaTextConfig& cfg) {
  TensorList out = llama_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = llama_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

LlamaBindReport llama_validate_text_binding(
    const LlamaTextConfig& cfg,
    const std::unordered_map<std::string, LlamaTensorDesc>& present, size_t max_errors) {
  LlamaBindReport rep;
  const auto expected = llama_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
  }
  for (const auto& [name, desc] : present) {
    (void)desc;
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void llama_tp_validate_geometry(const LlamaTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("llama tp geometry: " + what);
  };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 && world % cfg.num_key_value_heads != 0)
    fail("num_key_value_heads must divide world or be divided by it");
  // A rank's query heads must belong to its kv heads.
  const int q_per_kv = cfg.q_heads_per_kv();
  const int local_heads = cfg.num_attention_heads / world;
  const int head_begin = local_heads * rank;
  int kv_begin, local_kv;
  if (cfg.num_key_value_heads >= world) {
    local_kv = cfg.num_key_value_heads / world;
    kv_begin = local_kv * rank;
  } else {
    local_kv = 1;
    kv_begin = rank / (world / cfg.num_key_value_heads);
  }
  if (head_begin / q_per_kv != kv_begin || (head_begin + local_heads - 1) / q_per_kv != kv_begin + local_kv - 1)
    fail("the query heads of a rank straddle kv heads");
  if (cfg.intermediate_size % world != 0) fail("intermediate_size must divide by world");
}

}  // namespace dgpp
