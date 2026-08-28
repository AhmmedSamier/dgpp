#include "models/glm_binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {

namespace {

// mHC coefficient geometry, from the transformers Glm5NextTextHyperConnection
// reference: fn is [(2+n)*n, n*hidden] — rows split [pre n | post n | comb
// n*n]; base is [(2+n)*n] with the same split; scale is [3], one per OUTPUT
// (pre, post, comb) — not per head (mHC's "multi-head" names the
// manifold-constrained mixing, and this checkpoint trains n = hc_mult = 4
// streams). Semantics: DESIGN §7.3.
constexpr int64_t kHcCoeffRows = 24;
constexpr int64_t kHcScaleOutputs = 3;

using TensorList = std::vector<GlmExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype,
         std::vector<int64_t> shape, GlmWeightClass cls, int layer,
         int expert = -1) {
  out.push_back(GlmExpectedTensor{std::move(name), dtype, std::move(shape),
                                  cls, layer, expert});
}

std::vector<int64_t> rows_cols(int64_t rows, int64_t cols) {
  return {rows, cols};
}

// F8 payload + its scale partner, as one call so they cannot drift.
void add_quantized(TensorList& out, const std::string& name, int64_t rows,
                   int64_t cols, GlmWeightClass cls, int layer,
                   int expert = -1) {
  add(out, name, DType::F8_E4M3, rows_cols(rows, cols), cls, layer, expert);
  add(out, name + "_scale_inv", DType::F32,
      glm_scale_shape(rows_cols(rows, cols)), cls, layer, expert);
}

void expect_mhc(TensorList& out, const std::string& p,
                const GlmTextConfig& cfg, int layer) {
  const int64_t fn_cols = static_cast<int64_t>(cfg.hc_mult) * cfg.hidden_size;
  for (std::string_view side : {"attn", "ffn"}) {
    add(out, p + "hc_" + std::string(side) + "_base", DType::F32,
        {kHcCoeffRows}, GlmWeightClass::Mhc, layer);
    add(out, p + "hc_" + std::string(side) + "_fn", DType::BF16,
        {kHcCoeffRows, fn_cols}, GlmWeightClass::Mhc, layer);
    add(out, p + "hc_" + std::string(side) + "_scale", DType::F32,
        {kHcScaleOutputs}, GlmWeightClass::Mhc, layer);
  }
}

void expect_kda(TensorList& out, const std::string& p,
                const GlmTextConfig& cfg, int layer) {
  const int64_t proj = static_cast<int64_t>(cfg.kda_num_heads) *
                       cfg.kda_head_dim;  // heads*head_dim
  const int64_t hidden = cfg.hidden_size;
  const int64_t head_dim = cfg.kda_head_dim;
  const int64_t heads = cfg.kda_num_heads;
  for (const char* w : {"q", "k", "v"})
    add(out, p + w + "_proj.weight", DType::BF16, rows_cols(proj, hidden),
        GlmWeightClass::Kda, layer);
  add(out, p + "f_a_proj.weight", DType::BF16, rows_cols(head_dim, hidden),
      GlmWeightClass::Kda, layer);
  add(out, p + "g_a_proj.weight", DType::BF16, rows_cols(head_dim, hidden),
      GlmWeightClass::Kda, layer);
  add(out, p + "f_b_proj.weight", DType::BF16, rows_cols(proj, head_dim),
      GlmWeightClass::Kda, layer);
  add(out, p + "g_b_proj.weight", DType::BF16, rows_cols(proj, head_dim),
      GlmWeightClass::Kda, layer);
  add(out, p + "b_proj.weight", DType::BF16, rows_cols(heads, hidden),
      GlmWeightClass::Kda, layer);
  add(out, p + "A_log", DType::F32, {heads}, GlmWeightClass::Kda, layer);
  add(out, p + "dt_bias", DType::F32, {proj}, GlmWeightClass::Kda, layer);
  for (const char* w : {"q", "k", "v"})
    add(out, p + w + "_conv1d.weight", DType::BF16,
        {proj, 1, cfg.kda_conv_width}, GlmWeightClass::Kda, layer);
  add(out, p + "o_norm.weight", DType::BF16, {head_dim},
      GlmWeightClass::Kda, layer);
  add(out, p + "o_proj.weight", DType::BF16, rows_cols(hidden, proj),
      GlmWeightClass::Kda, layer);
}

void expect_dsa(TensorList& out, const std::string& p,
                const GlmTextConfig& cfg, int layer) {
  const int64_t hidden = cfg.hidden_size;
  const int64_t heads = cfg.num_attention_heads;
  const int64_t q_lora = cfg.q_lora_rank;
  const int64_t kv_lora = cfg.kv_lora_rank;
  const int64_t nope = cfg.qk_nope_head_dim;
  const int64_t v_dim = cfg.v_head_dim;
  const int64_t idx_proj =
      static_cast<int64_t>(cfg.index_n_heads) * cfg.index_head_dim;

  // Attention core. kv_b stays BF16 in this checkpoint (modules_to_not_convert
  // for attn_mqa), which the table encodes as observed dtype.
  add_quantized(out, p + "q_a_proj.weight", q_lora, hidden,
                GlmWeightClass::Dsa, layer);
  add(out, p + "q_a_layernorm.weight", DType::BF16, {q_lora},
      GlmWeightClass::Dsa, layer);
  add_quantized(out, p + "q_b_proj.weight", heads * nope, q_lora,
                GlmWeightClass::Dsa, layer);
  add_quantized(out, p + "kv_a_proj_with_mqa.weight", kv_lora, hidden,
                GlmWeightClass::Dsa, layer);
  add(out, p + "kv_a_layernorm.weight", DType::BF16, {kv_lora},
      GlmWeightClass::Dsa, layer);
  add(out, p + "kv_b_proj.weight", DType::BF16,
      rows_cols(heads * (nope + v_dim), kv_lora), GlmWeightClass::Dsa, layer);
  add_quantized(out, p + "o_proj.weight", hidden, heads * v_dim,
                GlmWeightClass::Dsa, layer);

  // Indexer (all BF16; replicated across TP per DESIGN §5.2).
  const std::string ip = p + "indexer.";
  add(out, ip + "wq_b.weight", DType::BF16, rows_cols(idx_proj, q_lora),
      GlmWeightClass::DsaIndexer, layer);
  add(out, ip + "wk.weight", DType::BF16,
      rows_cols(cfg.index_head_dim, hidden), GlmWeightClass::DsaIndexer,
      layer);
  add(out, ip + "weights_proj.weight", DType::BF16,
      rows_cols(cfg.index_n_heads, hidden), GlmWeightClass::DsaIndexer, layer);
  add(out, ip + "k_norm.weight", DType::BF16, {cfg.index_head_dim},
      GlmWeightClass::DsaIndexer, layer);
  add(out, ip + "k_norm.bias", DType::BF16, {cfg.index_head_dim},
      GlmWeightClass::DsaIndexer, layer);
  if (cfg.index_kpool_compress) {
    add(out, ip + "index_kpool_compress_gate", DType::BF16,
        rows_cols(cfg.index_head_dim, hidden), GlmWeightClass::DsaIndexer,
        layer);
    add(out, ip + "index_kpool_compress_ape", DType::BF16,
        {cfg.index_kpool, cfg.index_head_dim}, GlmWeightClass::DsaIndexer,
        layer);
  }
}

void expect_dense_mlp(TensorList& out, const std::string& p,
                      const GlmTextConfig& cfg, int layer) {
  const int64_t hidden = cfg.hidden_size;
  const int64_t inter = cfg.intermediate_size;
  const std::string m = p + "mlp.";
  add_quantized(out, m + "gate_proj.weight", inter, hidden,
                GlmWeightClass::DenseMlp, layer);
  add_quantized(out, m + "up_proj.weight", inter, hidden,
                GlmWeightClass::DenseMlp, layer);
  add_quantized(out, m + "down_proj.weight", hidden, inter,
                GlmWeightClass::DenseMlp, layer);
}

void expect_moe(TensorList& out, const std::string& p,
                const GlmTextConfig& cfg, int layer) {
  const int64_t hidden = cfg.hidden_size;
  const int64_t inter = cfg.moe_intermediate_size;
  add(out, p + "mlp.gate.weight", DType::BF16,
      rows_cols(cfg.n_routed_experts, hidden), GlmWeightClass::Router, layer);
  add(out, p + "mlp.gate.e_score_correction_bias", DType::F32,
      {cfg.n_routed_experts}, GlmWeightClass::Router, layer);
  // Single shared expert, un-indexed name (the only observed layout; the
  // config parser rejects n_shared_experts != 1 rather than guessing the
  // indexed spelling).
  const std::string sp = p + "mlp.shared_experts.";
  add_quantized(out, sp + "gate_proj.weight", inter, hidden,
                GlmWeightClass::SharedExpert, layer);
  add_quantized(out, sp + "up_proj.weight", inter, hidden,
                GlmWeightClass::SharedExpert, layer);
  add_quantized(out, sp + "down_proj.weight", hidden, inter,
                GlmWeightClass::SharedExpert, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
    add_quantized(out, ep + "gate_proj.weight", inter, hidden,
                  GlmWeightClass::RoutedExpert, layer, e);
    add_quantized(out, ep + "up_proj.weight", inter, hidden,
                  GlmWeightClass::RoutedExpert, layer, e);
    add_quantized(out, ep + "down_proj.weight", hidden, inter,
                  GlmWeightClass::RoutedExpert, layer, e);
  }
}

}  // namespace

std::vector<GlmExpectedTensor> glm_expected_layer_tensors(
    const GlmTextConfig& cfg, int layer) {
  const bool is_mtp = layer == cfg.mtp_layer();
  const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  if (layer < 0 || layer >= max_layer)
    throw std::invalid_argument(
        "glm_expected_layer_tensors: layer index out of range");

  TensorList out;
  const std::string p =
      "model.language_model.layers." + std::to_string(layer) + ".";
  if (is_mtp) {
    // Draft layer: DSA + MoE layout without mHC, plus the draft head.
    add(out, p + "enorm.weight", DType::BF16, {cfg.hidden_size},
        GlmWeightClass::Mtp, layer);
    add(out, p + "hnorm.weight", DType::BF16, {cfg.hidden_size},
        GlmWeightClass::Mtp, layer);
    add(out, p + "eh_proj.weight", DType::BF16,
        rows_cols(cfg.hidden_size, 2 * cfg.hidden_size), GlmWeightClass::Mtp,
        layer);
    add(out, p + "shared_head.norm.weight", DType::BF16, {cfg.hidden_size},
        GlmWeightClass::Mtp, layer);
    add(out, p + "input_layernorm.weight", DType::BF16, {cfg.hidden_size},
        GlmWeightClass::LayerNorm, layer);
    expect_dsa(out, p + "self_attn.", cfg, layer);
    add(out, p + "post_attention_layernorm.weight", DType::BF16,
        {cfg.hidden_size}, GlmWeightClass::LayerNorm, layer);
    expect_moe(out, p, cfg, layer);
    return out;
  }
  expect_mhc(out, p, cfg, layer);
  add(out, p + "input_layernorm.weight", DType::BF16, {cfg.hidden_size},
      GlmWeightClass::LayerNorm, layer);
  if (cfg.layers[layer] == GlmLayerKind::Kda)
    expect_kda(out, p + "self_attn.", cfg, layer);
  else
    expect_dsa(out, p + "self_attn.", cfg, layer);
  add(out, p + "post_attention_layernorm.weight", DType::BF16,
      {cfg.hidden_size}, GlmWeightClass::LayerNorm, layer);
  if (cfg.mlps[layer] == GlmMlpKind::Dense)
    expect_dense_mlp(out, p, cfg, layer);
  else
    expect_moe(out, p, cfg, layer);
  return out;
}

std::vector<GlmExpectedTensor> glm_expected_text_tensors(
    const GlmTextConfig& cfg) {
  TensorList out;
  out.reserve(8192);  // real model: ~76k entries; reserve the arena, not hope

  const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  for (int i = 0; i < max_layer; ++i) {
    TensorList layer_entries = glm_expected_layer_tensors(cfg, i);
    out.insert(out.end(), layer_entries.begin(), layer_entries.end());
  }

  add(out, "model.language_model.embed_tokens.weight", DType::BF16,
      rows_cols(cfg.vocab_size, cfg.hidden_size), GlmWeightClass::Embed, -1);
  add(out, "lm_head.weight", DType::BF16,
      rows_cols(cfg.vocab_size, cfg.hidden_size), GlmWeightClass::LmHead, -1);
  add(out, "model.language_model.norm.weight", DType::BF16,
      {cfg.hidden_size}, GlmWeightClass::FinalNorm, -1);
  return out;
}

GlmBindReport glm_validate_text_binding(
    const GlmTextConfig& cfg,
    const std::unordered_map<std::string, GlmTensorDesc>& present,
    size_t max_errors) {
  GlmBindReport rep;
  const auto expected = glm_expected_text_tensors(cfg);
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

  // Consumed-present tracking so the orphan scan can skip validated entries
  // (an expected tensor and its scale partner are both accounted for).
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
      push_error(std::format("'{}' dtype {} != expected {}", e.name,
                             dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name,
                             shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;

    if (!e.quantized()) continue;
    ++rep.quantized_matrices;
    const std::string scale_name = e.name + "_scale_inv";
    auto sit = present.find(scale_name);
    if (sit == present.end()) {
      ++rep.scales_bad;
      push_error(std::format("quantized '{}' has no scale tensor '{}'",
                             e.name, scale_name));
      continue;
    }
    consumed.emplace(scale_name, 1);
    // Scale geometry is the 128x128 block grid of the payload: exact match
    // required — a wrong grid silently dequantizes the wrong blocks.
    const std::vector<int64_t>& want = glm_scale_shape(e.shape);
    if (sit->second.dtype != DType::F32 || sit->second.shape != want) {
      ++rep.scales_bad;
      push_error(std::format(
          "'{}' must be F32 {} for payload {} (got {} {})", scale_name,
          shape_str(want), shape_str(e.shape), dtype_name(sit->second.dtype),
          shape_str(sit->second.shape)));
      continue;
    }
    ++rep.scales_bound;
  }

  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (name.rfind("model.visual.", 0) == 0) {
      ++rep.vision;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}' ({}, {})", name,
                           dtype_name(desc.dtype),
                           shape_str(desc.shape)));
  }
  return rep;
}

std::vector<int64_t> glm_scale_shape(const std::vector<int64_t>& payload) {
  // Defensive: only meaningful for 2-D payloads; every quantized matrix in
  // this checkpoint is 2-D and the callers guarantee it.
  if (payload.size() != 2)
    throw std::invalid_argument(
        "glm_scale_shape: quantized payload must be 2-D");
  return {(payload[0] + 127) / 128, (payload[1] + 127) / 128};
}

}  // namespace dgpp
