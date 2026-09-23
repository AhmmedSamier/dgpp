#include "models/mimo/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<MimoExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         MimoWeightClass cls, int layer, int expert = -1,
         MimoTensorRole role = MimoTensorRole::Plain) {
  out.push_back(MimoExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              MimoWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One fp8 [rows, cols] matrix on 128 x 128 blocks: the codes and the F32
// scale grid. `scale_rows` overrides ceil(rows / 128) for the fused qkv
// (its grid is tiled per chunk).
void add_fp8(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             MimoWeightClass cls, int layer, int64_t scale_rows = -1) {
  if (cols % 128 != 0)
    throw std::invalid_argument("mimo binding: fp8 K must be a multiple of 128 on " + base);
  if (scale_rows < 0) scale_rows = (rows + 127) / 128;
  add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, -1, MimoTensorRole::Fp8Payload);
  add(out, base + ".weight_scale_inv", DType::F32, {scale_rows, cols / 128}, cls, layer, -1,
      MimoTensorRole::Fp8Scale);
}

// One MXFP4 [rows, cols] matrix: the packed codes and the per-32 e8m0 scales.
void add_mxfp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
               MimoWeightClass cls, int layer, int expert) {
  if (cols % 32 != 0)
    throw std::invalid_argument("mimo binding: MXFP4 K must be a multiple of 32 on " + base);
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, expert, MimoTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::U8, {rows, cols / 32}, cls, layer, expert, MimoTensorRole::Fp4Scale);
}

void expect_attention(TensorList& out, const std::string& p, const MimoTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const MimoWeightClass c = MimoWeightClass::Attention;
  add_fp8(out, p + "qkv_proj", cfg.qkv_rows(layer), H, c, layer, cfg.qkv_scale_rows(layer));
  add_bf16(out, p + "o_proj.weight", {H, cfg.o_proj_cols()}, c, layer);
  if (cfg.sink_of(layer)) add_bf16(out, p + "attention_sink_bias", {cfg.num_attention_heads}, c, layer);
}

void expect_dense_mlp(TensorList& out, const std::string& p, const MimoTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const MimoWeightClass c = MimoWeightClass::DenseMlp;
  add_fp8(out, p + "gate_proj", I, H, c, layer);
  add_fp8(out, p + "up_proj", I, H, c, layer);
  add_fp8(out, p + "down_proj", H, I, c, layer);
}

void expect_moe(TensorList& out, const std::string& p, const MimoTextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  add_bf16(out, p + "gate.weight", {cfg.n_routed_experts, H}, MimoWeightClass::Router, layer);
  add(out, p + "gate.e_score_correction_bias", DType::F32, {cfg.n_routed_experts},
      MimoWeightClass::Router, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_mxfp4(out, ep + "gate_proj", I, H, MimoWeightClass::RoutedExpert, layer, e);
    add_mxfp4(out, ep + "up_proj", I, H, MimoWeightClass::RoutedExpert, layer, e);
    add_mxfp4(out, ep + "down_proj", H, I, MimoWeightClass::RoutedExpert, layer, e);
  }
}

int max_layer(const MimoTextConfig& cfg) {
  return cfg.num_hidden_layers + cfg.mtp_layers_loaded;
}

bool starts_with(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

}  // namespace

std::string mimo_layer_prefix(const MimoTextConfig& cfg, int layer) {
  if (cfg.is_mtp_layer(layer)) return "model.mtp.layers." + std::to_string(layer - cfg.num_hidden_layers) + ".";
  return "model.layers." + std::to_string(layer) + ".";
}

bool mimo_ignored_tensor(const MimoTextConfig& cfg, const std::string& name) {
  if (starts_with(name, "visual.") || starts_with(name, "audio_encoder.") ||
      starts_with(name, "speech_embeddings.") || starts_with(name, "audio_projector."))
    return true;
  if (starts_with(name, "model.mtp.layers.")) {
    // The draft layers past the first (and every draft layer when the
    // config declares none).
    const std::string rest = name.substr(std::string("model.mtp.layers.").size());
    const size_t dot = rest.find('.');
    if (dot == std::string::npos) return false;
    const std::string idx = rest.substr(0, dot);
    if (idx.empty() || idx.find_first_not_of("0123456789") != std::string::npos) return false;
    const int m = std::stoi(idx);
    return m >= cfg.mtp_layers_loaded;
  }
  return false;
}

std::vector<MimoExpectedTensor> mimo_expected_layer_tensors(const MimoTextConfig& cfg, int layer) {
  if (layer < 0 || layer >= max_layer(cfg))
    throw std::invalid_argument("mimo_expected_layer_tensors: layer out of range");
  const bool is_mtp = cfg.is_mtp_layer(layer);
  const std::string p = mimo_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  if (is_mtp) {
    // The draft head's own tensors (vLLM mimo_v2_mtp.py: enorm, hnorm,
    // eh_proj, final_layernorm); the draft shares the globals' embedding
    // and head (no copies in this release).
    add_bf16(out, p + "enorm.weight", {H}, MimoWeightClass::MtpHead, layer);
    add_bf16(out, p + "hnorm.weight", {H}, MimoWeightClass::MtpHead, layer);
    add_bf16(out, p + "eh_proj.weight", {H, 2 * H}, MimoWeightClass::MtpHead, layer);
    add_bf16(out, p + "final_layernorm.weight", {H}, MimoWeightClass::MtpHead, layer);
  }
  add_bf16(out, p + "input_layernorm.weight", {H}, MimoWeightClass::LayerNorm, layer);
  add_bf16(out, p + (is_mtp ? "pre_mlp_layernorm.weight" : "post_attention_layernorm.weight"), {H},
           MimoWeightClass::LayerNorm, layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  if (cfg.is_moe_layer(layer))
    expect_moe(out, p + "mlp.", cfg, layer);
  else
    expect_dense_mlp(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<MimoExpectedTensor> mimo_expected_global_tensors(const MimoTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, MimoWeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, MimoWeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, MimoWeightClass::LmHead, -1);
  return out;
}

std::vector<MimoExpectedTensor> mimo_expected_text_tensors(const MimoTextConfig& cfg) {
  TensorList out = mimo_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer(cfg); ++l) {
    TensorList layer = mimo_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

MimoBindReport mimo_validate_text_binding(
    const MimoTextConfig& cfg, const std::unordered_map<std::string, MimoTensorDesc>& present,
    size_t max_errors) {
  MimoBindReport rep;
  const auto expected = mimo_expected_text_tensors(cfg);
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
    if (e.role == MimoTensorRole::Fp8Payload) ++rep.fp8_matrices;
    if (e.role == MimoTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (mimo_ignored_tensor(cfg, name)) {
      ++rep.ignored;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void mimo_tp_validate_geometry(const MimoTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("mimo tp geometry: " + what);
  };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  // The fused qkv_proj is pre-sharded in num_key_value_heads chunks; a rank
  // holds whole chunks, so world divides the chunk count — and with it the
  // query heads and both kv head counts (each a multiple of the chunks).
  if (cfg.qkv_chunks() % world != 0)
    fail("world must divide num_key_value_heads (the qkv_proj chunk count: worlds 1, 2 and 4 for the release)");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 || cfg.swa_num_key_value_heads % world != 0)
    fail("the kv head counts must divide by world");
  for (const int inter : {cfg.intermediate_size, cfg.moe_intermediate_size}) {
    if (inter % world != 0) fail("an intermediate size must divide by world");
  }
  if ((cfg.intermediate_size / world) % 128 != 0) fail("the dense intermediate slice must be a multiple of 128 (the fp8 block)");
  if ((cfg.moe_intermediate_size / world) % 32 != 0) fail("the expert intermediate slice must be a multiple of 32 (the MXFP4 block)");
}

}  // namespace dgpp
