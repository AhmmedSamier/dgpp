#include "models/glm4/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<Glm4ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Glm4WeightClass cls, int layer, int expert = -1,
         Glm4TensorRole role = Glm4TensorRole::Plain) {
  out.push_back(Glm4ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Glm4WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One NVFP4 [rows, cols] matrix: the packed codes, the per-16 e4m3 scales,
// the fp32 per-tensor scale, and the (unused) activation scale.
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Glm4WeightClass cls, int layer, int expert, int group) {
  if (cols % group != 0 || cols % 2 != 0)
    throw std::invalid_argument("glm4 binding: NVFP4 K must be a multiple of the group on " + base);
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, expert, Glm4TensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / group}, cls, layer, expert,
      Glm4TensorRole::Fp4Scale);
  add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert, Glm4TensorRole::Fp4Global);
  add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert, Glm4TensorRole::InputScale);
}

// A BF16 [rows, cols] expert matrix of the draft layer.
void add_bf16_expert(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
                     Glm4WeightClass cls, int layer, int expert) {
  add(out, base + ".weight", DType::BF16, {rows, cols}, cls, layer, expert, Glm4TensorRole::Bf16Expert);
}

void expect_attention(TensorList& out, const std::string& p, const Glm4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, d = cfg.head_dim;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const Glm4WeightClass c = Glm4WeightClass::Attention;
  add_bf16(out, p + "q_proj.weight", {qh * d, H}, c, layer);
  add_bf16(out, p + "k_proj.weight", {kvh * d, H}, c, layer);
  add_bf16(out, p + "v_proj.weight", {kvh * d, H}, c, layer);
  add_bf16(out, p + "o_proj.weight", {H, qh * d}, c, layer);
  if (cfg.attention_bias) {
    add_bf16(out, p + "q_proj.bias", {qh * d}, c, layer);
    add_bf16(out, p + "k_proj.bias", {kvh * d}, c, layer);
    add_bf16(out, p + "v_proj.bias", {kvh * d}, c, layer);
  }
  if (cfg.use_qk_norm) {
    add_bf16(out, p + "q_norm.weight", {d}, c, layer);
    add_bf16(out, p + "k_norm.weight", {d}, c, layer);
  }
  // The FP8 KV scales ride along on the main layers only (the draft layer
  // was excluded from quantization altogether).
  if (cfg.kv_scales_present && layer != cfg.mtp_layer()) {
    add(out, p + "k_proj.k_scale", DType::F32, {}, Glm4WeightClass::KvScale, layer, -1,
        Glm4TensorRole::KvScale);
    add(out, p + "v_proj.v_scale", DType::F32, {}, Glm4WeightClass::KvScale, layer, -1,
        Glm4TensorRole::KvScale);
  }
}

void expect_dense_mlp(TensorList& out, const std::string& p, const Glm4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const Glm4WeightClass c = Glm4WeightClass::DenseMlp;
  add_fp4(out, p + "gate_proj", I, H, c, layer, -1, cfg.fp4_group_size);
  add_fp4(out, p + "up_proj", I, H, c, layer, -1, cfg.fp4_group_size);
  add_fp4(out, p + "down_proj", H, I, c, layer, -1, cfg.fp4_group_size);
}

void expect_moe(TensorList& out, const std::string& p, const Glm4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_inter();
  const bool bf16 = layer == cfg.mtp_layer();
  add_bf16(out, p + "gate.weight", {cfg.n_routed_experts, H}, Glm4WeightClass::Router, layer);
  add(out, p + "gate.e_score_correction_bias", DType::F32, {cfg.n_routed_experts},
      Glm4WeightClass::Router, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    if (bf16) {
      add_bf16_expert(out, ep + "gate_proj", I, H, Glm4WeightClass::RoutedExpert, layer, e);
      add_bf16_expert(out, ep + "up_proj", I, H, Glm4WeightClass::RoutedExpert, layer, e);
      add_bf16_expert(out, ep + "down_proj", H, I, Glm4WeightClass::RoutedExpert, layer, e);
    } else {
      add_fp4(out, ep + "gate_proj", I, H, Glm4WeightClass::RoutedExpert, layer, e, cfg.fp4_group_size);
      add_fp4(out, ep + "up_proj", I, H, Glm4WeightClass::RoutedExpert, layer, e, cfg.fp4_group_size);
      add_fp4(out, ep + "down_proj", H, I, Glm4WeightClass::RoutedExpert, layer, e, cfg.fp4_group_size);
    }
  }
  const std::string sp = p + "shared_experts.";
  if (bf16) {
    add_bf16_expert(out, sp + "gate_proj", S, H, Glm4WeightClass::SharedExpert, layer, -1);
    add_bf16_expert(out, sp + "up_proj", S, H, Glm4WeightClass::SharedExpert, layer, -1);
    add_bf16_expert(out, sp + "down_proj", H, S, Glm4WeightClass::SharedExpert, layer, -1);
  } else {
    add_fp4(out, sp + "gate_proj", S, H, Glm4WeightClass::SharedExpert, layer, -1, cfg.fp4_group_size);
    add_fp4(out, sp + "up_proj", S, H, Glm4WeightClass::SharedExpert, layer, -1, cfg.fp4_group_size);
    add_fp4(out, sp + "down_proj", H, S, Glm4WeightClass::SharedExpert, layer, -1, cfg.fp4_group_size);
  }
}

int max_layer(const Glm4TextConfig& cfg) {
  return cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
}

}  // namespace

std::string glm4_layer_prefix(const Glm4TextConfig&, int layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

std::vector<Glm4ExpectedTensor> glm4_expected_layer_tensors(const Glm4TextConfig& cfg, int layer) {
  if (layer < 0 || layer >= max_layer(cfg))
    throw std::invalid_argument("glm4_expected_layer_tensors: layer out of range");
  const bool is_mtp = layer == cfg.mtp_layer();
  const std::string p = glm4_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  if (is_mtp) {
    // The draft head's own tensors (vLLM glm4_moe_mtp.py: enorm, hnorm,
    // eh_proj, shared_head.norm) and its copies of the embedding and the
    // head (byte-identical to the globals; verified, never loaded).
    add_bf16(out, p + "enorm.weight", {H}, Glm4WeightClass::MtpHead, layer);
    add_bf16(out, p + "hnorm.weight", {H}, Glm4WeightClass::MtpHead, layer);
    add_bf16(out, p + "eh_proj.weight", {H, 2 * H}, Glm4WeightClass::MtpHead, layer);
    add_bf16(out, p + "shared_head.norm.weight", {H}, Glm4WeightClass::MtpHead, layer);
    add(out, p + "embed_tokens.weight", DType::BF16, {cfg.vocab_size, H}, Glm4WeightClass::MtpHead,
        layer, -1, Glm4TensorRole::Duplicate);
    add(out, p + "shared_head.head.weight", DType::BF16, {cfg.vocab_size, H},
        Glm4WeightClass::MtpHead, layer, -1, Glm4TensorRole::Duplicate);
  }
  add_bf16(out, p + "input_layernorm.weight", {H}, Glm4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {H}, Glm4WeightClass::LayerNorm, layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  if (cfg.is_moe_layer(layer))
    expect_moe(out, p + "mlp.", cfg, layer);
  else
    expect_dense_mlp(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<Glm4ExpectedTensor> glm4_expected_global_tensors(const Glm4TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, Glm4WeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, Glm4WeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, Glm4WeightClass::LmHead, -1);
  return out;
}

std::vector<Glm4ExpectedTensor> glm4_expected_text_tensors(const Glm4TextConfig& cfg) {
  TensorList out = glm4_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer(cfg); ++l) {
    TensorList layer = glm4_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

Glm4BindReport glm4_validate_text_binding(
    const Glm4TextConfig& cfg, const std::unordered_map<std::string, Glm4TensorDesc>& present,
    size_t max_errors) {
  Glm4BindReport rep;
  const auto expected = glm4_expected_text_tensors(cfg);
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
    if (e.role == Glm4TensorRole::Fp4Payload) ++rep.fp4_matrices;
    if (e.role == Glm4TensorRole::Bf16Expert) ++rep.bf16_expert_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void glm4_tp_validate_geometry(const Glm4TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("glm4 tp geometry: " + what);
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
  for (const int inter : {cfg.intermediate_size, cfg.moe_intermediate_size, cfg.shared_expert_inter()}) {
    if (inter % world != 0) fail("an intermediate size must divide by world");
    if ((inter / world) % 32 != 0) fail("an intermediate slice must be a multiple of 32 (the NVFP4 GEMV core's K)");
  }
}

}  // namespace dgpp
