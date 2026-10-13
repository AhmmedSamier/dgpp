#include "models/qwen_asr/binding.hpp"

#include <stdexcept>

namespace dgpp {
namespace {
QwenAsrExpectedTensor t(std::string name, DType dt, std::vector<int64_t> shape,
                        QwenAsrWeightClass cls, int layer,
                        QwenAsrTensorRole role = QwenAsrTensorRole::Plain) {
  QwenAsrExpectedTensor e;
  e.name = std::move(name);
  e.dtype = dt;
  e.shape = std::move(shape);
  e.cls = cls;
  e.layer = layer;
  e.role = role;
  return e;
}
}  // namespace

std::string qwen_asr_layer_prefix(const QwenAsrLmConfig&, int layer) {
  return "thinker.model.layers." + std::to_string(layer) + ".";
}

std::vector<QwenAsrExpectedTensor> qwen_asr_expected_layer_tensors(const QwenAsrLmConfig& c,
                                                                   int l) {
  const std::string p = qwen_asr_layer_prefix(c, l);
  const int H = c.hidden_size, I = c.intermediate_size, D = QwenAsrLmConfig::kHeadDim;
  const int Q = c.q_proj_width(), KV = c.kv_proj_width();
  std::vector<QwenAsrExpectedTensor> v = {
      t(p + "input_layernorm.weight", DType::BF16, {H}, QwenAsrWeightClass::LayerNorm, l),
      t(p + "post_attention_layernorm.weight", DType::BF16, {H}, QwenAsrWeightClass::LayerNorm, l),
      t(p + "self_attn.q_proj.weight", DType::BF16, {Q, H}, QwenAsrWeightClass::Attention, l),
      t(p + "self_attn.k_proj.weight", DType::BF16, {KV, H}, QwenAsrWeightClass::Attention, l),
      t(p + "self_attn.v_proj.weight", DType::BF16, {KV, H}, QwenAsrWeightClass::Attention, l),
      t(p + "self_attn.o_proj.weight", DType::BF16, {H, Q}, QwenAsrWeightClass::Attention, l),
      t(p + "self_attn.q_norm.weight", DType::BF16, {D}, QwenAsrWeightClass::Attention, l),
      t(p + "self_attn.k_norm.weight", DType::BF16, {D}, QwenAsrWeightClass::Attention, l),
      t(p + "mlp.gate_proj.weight", DType::BF16, {I, H}, QwenAsrWeightClass::DenseMlp, l),
      t(p + "mlp.up_proj.weight", DType::BF16, {I, H}, QwenAsrWeightClass::DenseMlp, l),
      t(p + "mlp.down_proj.weight", DType::BF16, {H, I}, QwenAsrWeightClass::DenseMlp, l),
  };
  return v;
}

std::vector<QwenAsrExpectedTensor> qwen_asr_expected_global_tensors(const QwenAsrLmConfig& c) {
  const int H = c.hidden_size, V = c.vocab_size;
  std::vector<QwenAsrExpectedTensor> v = {
      t("thinker.model.embed_tokens.weight", DType::BF16, {V, H}, QwenAsrWeightClass::Embed, -1),
      t("thinker.model.norm.weight", DType::BF16, {H}, QwenAsrWeightClass::FinalNorm, -1),
      t("thinker.lm_head.weight", DType::BF16, {V, H}, QwenAsrWeightClass::LmHead, -1,
        QwenAsrTensorRole::Duplicate),
  };
  // The tower is NOT in the table: like the qwen family's model.visual.*,
  // its tensors are counted, not validated (audio_config.hpp and the
  // tower loader own them).
  return v;
}

bool qwen_asr_is_replicated(QwenAsrWeightClass c) {
  switch (c) {
    case QwenAsrWeightClass::Embed:
    case QwenAsrWeightClass::LmHead:
    case QwenAsrWeightClass::FinalNorm:
    case QwenAsrWeightClass::LayerNorm:
    case QwenAsrWeightClass::Attention:
    case QwenAsrWeightClass::DenseMlp:
    case QwenAsrWeightClass::Tower: return true;
  }
  return true;
}

QwenAsrBindReport qwen_asr_validate_text_binding(
    const QwenAsrLmConfig& cfg, const std::unordered_map<std::string, QwenAsrTensorDesc>& present,
    size_t max_errors) {
  QwenAsrBindReport r;
  std::vector<QwenAsrExpectedTensor> all = qwen_asr_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    auto v = qwen_asr_expected_layer_tensors(cfg, l);
    all.insert(all.end(), v.begin(), v.end());
  }
  std::unordered_map<std::string, bool> seen;
  for (const auto& e : all) {
    ++r.expected;
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++r.missing;
      if (r.errors.size() < max_errors) r.errors.push_back("missing " + e.name);
      continue;
    }
    seen[e.name] = true;
    if (it->second.dtype != e.dtype) {
      ++r.dtype_mismatch;
      if (r.errors.size() < max_errors) r.errors.push_back("dtype " + e.name);
    } else if (it->second.shape != e.shape) {
      ++r.shape_mismatch;
      if (r.errors.size() < max_errors) r.errors.push_back("shape " + e.name);
    } else {
      ++r.matched;
    }
  }
  for (const auto& [name, desc] : present)
    if (!seen.count(name)) {
      if (name.starts_with("thinker.audio_tower.")) {
        ++r.tower;
        continue;  // counted, not validated (the tower loader owns them)
      }
      ++r.unexpected;
      if (r.errors.size() < max_errors) r.errors.push_back("unexpected " + name);
    }
  return r;
}
}  // namespace dgpp
