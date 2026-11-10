#include "models/qwen/dflash.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void reject(const std::string& field, const std::string& why) {
  throw std::runtime_error("DFlash v1 config " + field + ": " + why);
}

const minijson::Value& require(const minijson::Value& v, const char* key) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) return *f;
  reject(key, "missing");
  throw std::runtime_error("unreachable");
}

std::string str_at(const minijson::Value& v, const char* key, const std::string& dflt) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_string()) reject(key, "not a string");
    return std::string(f->as_string());
  }
  return dflt;
}

int int_at(const minijson::Value& v, const char* key, int dflt, bool required = false) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return static_cast<int>(f->as_int());
  }
  if (required) reject(key, "missing");
  return dflt;
}

int64_t i64_at(const minijson::Value& v, const char* key, int64_t dflt, bool required = false) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return f->as_int();
  }
  if (required) reject(key, "missing");
  return dflt;
}

double double_at(const minijson::Value& v, const char* key, double dflt) {
  if (const minijson::Value* f = v.find(key); f != nullptr && !f->is_null()) {
    if (!f->is_number()) reject(key, "not a number");
    return f->as_double();
  }
  return dflt;
}

}  // namespace

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

DFlashConfig DFlashConfig::from_json_file(const std::string& path) {
  const std::string text = read_file(path);
  minijson::Value root;
  try {
    root = minijson::parse(text).root;
  } catch (const std::exception& e) {
    throw std::runtime_error("DFlash v1 config " + path + ": " + e.what());
  }
  return parse(root);
}

DFlashConfig DFlashConfig::parse(const minijson::Value& root) {
  DFlashConfig c;
  const std::string arch =
      root.find("architectures") && root.at("architectures").is_array() &&
              !root.at("architectures").items().empty()
          ? std::string(root.at("architectures").items().front().as_string())
          : "";
  // v1 only: the DFlash2 draft (DFlash2DraftModel, selector + conv) parses
  // through DFlash2Config, never here.
  if (arch != "DFlashDraftModel") reject("architectures", "DFlashDraftModel required");
  if (str_at(root, "model_type", "") != "qwen3") reject("model_type", "qwen3 required");
  if (str_at(root, "dtype", "bfloat16") != "bfloat16") reject("dtype", "bfloat16 required");
  const minijson::Value& d = require(root, "dflash_config");
  c.dspine = d.find("selected_prefix_code_bridge") != nullptr;
  c.block_size = c.dspine ? int_at(d, "block_size", int_at(root, "block_size", 16, true))
                          : int_at(d, "block_size", 16, true);
  c.mask_token_id = int_at(d, "mask_token_id", -1, true);
  const minijson::Value& taps = require(d, "target_layer_ids");
  if (!taps.is_array() || taps.items().empty()) reject("dflash_config.target_layer_ids", "empty");
  for (const auto& t : taps.items()) {
    if (!t.is_number()) reject("dflash_config.target_layer_ids", "non-numeric element");
    c.target_layer_ids.push_back(static_cast<int>(t.as_int()));
  }
  if (!std::is_sorted(c.target_layer_ids.begin(), c.target_layer_ids.end()))
    reject("dflash_config.target_layer_ids", "must be in checkpoint feature order (ascending)");
  c.num_hidden_layers = int_at(root, "num_hidden_layers", 6, true);
  c.hidden_size = int_at(root, "hidden_size", 3072, true);
  c.vocab_size = int_at(root, "vocab_size", 248320, true);
  c.num_attention_heads = int_at(root, "num_attention_heads", 32, true);
  c.num_key_value_heads = int_at(root, "num_key_value_heads", 8, true);
  if (c.hidden_size <= 0 || c.vocab_size <= 0 || c.num_attention_heads <= 0 || c.num_key_value_heads <= 0)
    reject("dimensions", "hidden_size, vocab_size and head counts must be positive");
  c.head_dim = int_at(root, "head_dim", c.hidden_size / c.num_attention_heads, true);
  c.intermediate_size = int_at(root, "intermediate_size", 9216, true);
  c.rms_norm_eps = static_cast<float>(double_at(root, "rms_norm_eps", 1e-6));
  const minijson::Value& rp = require(root, "rope_parameters");
  if (str_at(rp, "rope_type", "default") != "default")
    reject("rope_parameters.rope_type", "only the plain default scaling is supported");
  c.rope_theta = double_at(rp, "rope_theta", 1e7);
  c.max_position_embeddings = i64_at(root, "max_position_embeddings", 262144);
  {
    const minijson::Value* sw = root.find("use_sliding_window");
    const bool sliding = sw && !sw->is_null() ? sw->as_bool(true) : true;
    c.sliding_window = sliding ? i64_at(root, "sliding_window", 4096, true) : INT64_MAX;
  }
  if (c.block_size < 2 || c.block_size > 16) reject("dflash_config.block_size", "2..16 supported");
  if (c.intermediate_size <= 0 || c.max_position_embeddings <= 0 || c.sliding_window <= 0)
    reject("dimensions", "intermediate_size, max_position_embeddings and sliding_window must be positive");
  if (c.num_hidden_layers < 1 || c.num_hidden_layers > 16)
    reject("num_hidden_layers", "1..16 supported");
  if (c.head_dim != 128) reject("head_dim", "only 128 is supported");
  if (c.num_attention_heads % c.num_key_value_heads ||
      c.num_attention_heads / c.num_key_value_heads > 8)
    reject("num_key_value_heads", "heads/kv_heads must be in [1, 8]");
  if (const auto* types = root.find("layer_types"); types && !types->is_null()) {
    if (!types->is_array() || types->items().size() != static_cast<size_t>(c.num_hidden_layers))
      reject("layer_types", "must name every draft layer");
    for (const auto& type : types->items()) {
      if (!type.is_string()) reject("layer_types", "expected attention type strings");
      if (type.as_string() == "full_attention") c.layer_windows.push_back(INT64_MAX);
      else if (type.as_string() == "sliding_attention") c.layer_windows.push_back(c.sliding_window);
      else reject("layer_types", "only full_attention and sliding_attention are supported");
    }
  }
  if (c.dspine) {
    const auto* causal = d.find("draft_block_causal");
    if (!causal || !causal->as_bool(false))
      reject("dflash_config.draft_block_causal", "DSpine requires causal draft blocks");
    if (root.find("is_causal") && !root.at("is_causal").as_bool(false))
      reject("is_causal", "contradicts dflash_config.draft_block_causal");
    const auto& bridge = require(d, "selected_prefix_code_bridge");
    c.code_size = int_at(bridge, "code_size", 0, true);
    c.code_rank = int_at(bridge, "rank", 0, true);
    if (!bridge.find("eps") || !bridge.at("eps").is_number() ||
        !std::isfinite(bridge.at("eps").as_double()) || bridge.at("eps").as_double() <= 0)
      reject("dflash_config.selected_prefix_code_bridge.eps", "must be positive");
    c.code_eps = static_cast<float>(bridge.at("eps").as_double());
    if (str_at(d, "selected_prefix_feature_write_mode", "") != "independent")
      reject("dflash_config.selected_prefix_feature_write_mode", "independent required");
    if (!d.find("selected_prefix_feature_writes") || !d.at("selected_prefix_feature_writes").as_bool(false))
      reject("dflash_config.selected_prefix_feature_writes", "true required");
    if (d.find("selected_prefix_terminal") && d.at("selected_prefix_terminal").as_bool(true))
      reject("dflash_config.selected_prefix_terminal", "terminal bridge is unsupported");
    if (d.find("selected_prefix_terminal_residual_rank") &&
        int_at(d, "selected_prefix_terminal_residual_rank", 0) != 0)
      reject("dflash_config.selected_prefix_terminal_residual_rank", "nonzero terminal residual is unsupported");
    const auto& active = require(d, "selected_prefix_feature_layers");
    if (!active.is_array() || active.items().empty())
      reject("dflash_config.selected_prefix_feature_layers", "empty");
    for (const auto& layer : active.items()) {
      if (!layer.is_number()) reject("dflash_config.selected_prefix_feature_layers", "non-numeric element");
      c.code_feature_layers.push_back(static_cast<int>(layer.as_int()));
    }
    if (!std::is_sorted(c.code_feature_layers.begin(), c.code_feature_layers.end()) ||
        std::adjacent_find(c.code_feature_layers.begin(), c.code_feature_layers.end()) != c.code_feature_layers.end())
      reject("dflash_config.selected_prefix_feature_layers", "must be strictly ascending");
    if (c.code_feature_layers.back() != c.num_hidden_layers)
      reject("dflash_config.selected_prefix_feature_layers", "the final draft layer must write the bridge");
    if (c.code_feature_layers.front() < 1 || c.code_feature_layers.back() > c.num_hidden_layers)
      reject("dflash_config.selected_prefix_feature_layers", "layer ids must be within the draft stack");
    const auto* refine = d.find("selected_prefix_last_write_refine");
    if (!refine) reject("dflash_config.selected_prefix_last_write_refine", "missing");
    if (str_at(*refine, "kind", "") != "exact_sender")
      reject("dflash_config.selected_prefix_last_write_refine.kind", "exact_sender required");
    const double refine_weight = double_at(*refine, "weight", 1.0);
    if (int_at(*refine, "version", 1) != 1 || !std::isfinite(refine_weight) || refine_weight < 0.0)
      reject("dflash_config.selected_prefix_last_write_refine", "version 1 and nonnegative weight required");
    c.code_refine_weight = static_cast<float>(refine_weight);
    c.code_head_topk = int_at(*refine, "head_topk", 0, true);
    const auto* second = root.find("code_bridge_second_pass");
    if (!second || str_at(*second, "mode", "") != "serial_table" ||
        int_at(*second, "first_exact", -1) != c.num_hidden_layers - 1 ||
        int_at(*second, "head_topk", -1) != c.code_head_topk)
      reject("code_bridge_second_pass", "matching serial_table final-layer refinement required");
    if (c.code_size <= 0 || c.code_size > 4096 || (c.code_size & (c.code_size - 1)) != 0 ||
        c.code_rank <= 0 || c.code_rank > 4096 ||
        c.code_rank % 128 != 0 || c.code_head_topk < 2 || c.code_head_topk > 64 ||
        (c.code_head_topk & (c.code_head_topk - 1)) != 0)
      reject("dflash_config.selected_prefix_code_bridge", "invalid code dimensions or top-k");
  }
  if (const minijson::Value* tw = root.find("tie_word_embeddings"))
    if (tw->as_bool(false))
      reject("tie_word_embeddings", "the draft shares the target's head; ties are not loaded");
  return c;
}

void DFlashConfig::validate_against(const Qwen35TextConfig& target) const {
  validate_against(target.hidden_size, target.vocab_size, target.num_hidden_layers,
                   target.context_limit());
}

void DFlashConfig::validate_against(int target_hidden, int target_vocab, int target_layers,
                                    int64_t target_context) const {
  if (hidden_size != target_hidden)
    reject("hidden_size", "the draft must share the target's width (fc and the shared head)");
  if (vocab_size != target_vocab)
    reject("vocab_size", "the draft must share the target's vocabulary (mask id, shared head)");
  if (mask_token_id < 0 || mask_token_id >= target_vocab)
    reject("dflash_config.mask_token_id", "outside the target vocabulary");
  if (block_size - 1 > 15)
    reject("dflash_config.block_size", "the engine verifies at most 15 drafts per step");
  for (size_t i = 0; i < target_layer_ids.size(); ++i) {
    if (target_layer_ids[i] < 0 || target_layer_ids[i] >= target_layers)
      reject("dflash_config.target_layer_ids",
             "layer " + std::to_string(target_layer_ids[i]) + " does not exist in the target");
    if (i && target_layer_ids[i] == target_layer_ids[i - 1])
      reject("dflash_config.target_layer_ids", "duplicate tap layer");
  }
  if (sliding_window <= block_size)
    reject("sliding_window", "the window must cover the draft block");
  if (max_position_embeddings > target_context)
    reject("max_position_embeddings", "the draft must not exceed the target's");
}

size_t dflash_weights_bytes(const DFlashConfig& cfg, int world) {
  if (!cfg.tp_divisible(world))
    throw std::invalid_argument("dflash_weights_bytes: the drafter does not split across this world");
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t QW = static_cast<size_t>(cfg.local_q_row(world));
  const size_t KW = static_cast<size_t>(cfg.local_kv_row(world));
  const size_t I = static_cast<size_t>(cfg.local_intermediate(world));
  const size_t per_layer = (QW + 2 * KW) * H + H * QW + 2 * I * H + H * I + 2 * H + 2 * cfg.head_dim;
  const size_t total =
      static_cast<size_t>(cfg.num_hidden_layers) * per_layer +
      static_cast<size_t>(cfg.target_layer_ids.size()) * H * H + 2 * H;
  size_t elems = total;
  if (cfg.dspine) {
    const size_t C = static_cast<size_t>(cfg.code_size);
    const size_t R = static_cast<size_t>(cfg.code_rank);
    const size_t V = static_cast<size_t>(cfg.vocab_size);
    elems += V * C + C * H + static_cast<size_t>(cfg.num_hidden_layers - 1) * C * H +
             static_cast<size_t>(cfg.num_hidden_layers) * H * R + 3 * R * C + C * R + R;
  }
  return elems * 2;
}

DFlashWeights load_dflash_weights(const DFlashConfig& cfg, const std::string& dir,
                                  cudaStream_t stream, int rank, int world) {
  if (!cfg.tp_divisible(world))
    throw std::invalid_argument("load_dflash_weights: the drafter does not split across this world");
  std::vector<std::string> shards;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") shards.push_back(e.path().string());
  if (shards.empty()) throw std::runtime_error("load_dflash_weights: no shards in " + dir);
  std::sort(shards.begin(), shards.end());
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  for (const auto& s : shards) files.push_back(SafetensorsFile::open(s));
  auto find = [&](const std::string& name) -> const TensorInfo* {
    for (const auto& f : files)
      if (const TensorInfo* t = f->find(name)) return t;
    return nullptr;
  };
  auto need = [&](const std::string& name) -> const TensorInfo* {
    if (const TensorInfo* t = find(name)) return t;
    throw std::runtime_error("load_dflash_weights: missing tensor '" + name + "'");
  };
  auto check = [&](const TensorInfo* t, DType dtype, std::vector<int64_t> shape) {
    if (t->dtype != dtype || t->shape != shape)
      throw std::runtime_error("load_dflash_weights: tensor '" + t->name + "' shape/dtype mismatch");
  };

  const int64_t H = cfg.hidden_size;
  const int64_t QW = cfg.local_q_row(world), KW = cfg.local_kv_row(world);
  const int64_t I = cfg.local_intermediate(world);
  const int64_t r = rank;
  const int64_t nTaps = static_cast<int64_t>(cfg.target_layer_ids.size());

  DFlashWeights w;
  DGPP_CUDA_OK(cudaMalloc(&w.arena, dflash_weights_bytes(cfg, world)));
  w.bytes = dflash_weights_bytes(cfg, world);
  uint16_t* base = static_cast<uint16_t*>(w.arena);
  size_t off = 0;
  auto bump = [&](size_t n) {
    uint16_t* p = base + off;
    off += n;
    return p;
  };
  auto copy_full = [&](const std::string& name, uint16_t* dst, std::vector<int64_t> shape) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, shape);
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, t->data, t->nbytes(), cudaMemcpyHostToDevice, stream));
  };
  auto copy_rows = [&](const std::string& name, uint16_t* dst, int64_t rows, int64_t cols,
                       int64_t row0, int64_t count) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, {rows, cols});
    const auto* src = static_cast<const uint16_t*>(t->data) + static_cast<size_t>(row0) * cols;
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, static_cast<size_t>(count) * cols * 2,
                                 cudaMemcpyHostToDevice, stream));
  };
  std::vector<uint16_t> colbuf;
  auto copy_cols = [&](const std::string& name, uint16_t* dst, int64_t rows, int64_t cols,
                       int64_t col0, int64_t count) {
    const TensorInfo* t = need(name);
    check(t, DType::BF16, {rows, cols});
    colbuf.resize(static_cast<size_t>(rows) * count);
    const auto* src = static_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < rows; ++i)
      std::memcpy(colbuf.data() + static_cast<size_t>(i) * count,
                  src + static_cast<size_t>(i) * cols + col0, static_cast<size_t>(count) * 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(dst, colbuf.data(), colbuf.size() * 2, cudaMemcpyHostToDevice,
                                 stream));
  };

  w.layers.resize(static_cast<size_t>(cfg.num_hidden_layers));
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    DFlashLayerWeights& x = w.layers[static_cast<size_t>(l)];
    x.input_norm = bump(H);
    copy_full(p + "input_layernorm.weight", const_cast<uint16_t*>(x.input_norm), {H});
    x.post_norm = bump(H);
    copy_full(p + "post_attention_layernorm.weight", const_cast<uint16_t*>(x.post_norm), {H});
    x.q = bump(static_cast<size_t>(QW) * H);
    copy_rows(p + "self_attn.q_proj.weight", const_cast<uint16_t*>(x.q), cfg.q_row(), H,
              r * QW, QW);
    x.k = bump(static_cast<size_t>(KW) * H);
    copy_rows(p + "self_attn.k_proj.weight", const_cast<uint16_t*>(x.k), cfg.kv_row(), H,
              r * KW, KW);
    x.v = bump(static_cast<size_t>(KW) * H);
    copy_rows(p + "self_attn.v_proj.weight", const_cast<uint16_t*>(x.v), cfg.kv_row(), H,
              r * KW, KW);
    x.o = bump(static_cast<size_t>(H) * QW);
    copy_cols(p + "self_attn.o_proj.weight", const_cast<uint16_t*>(x.o), H, cfg.q_row(),
              r * QW, QW);
    x.gate = bump(static_cast<size_t>(I) * H);
    copy_rows(p + "mlp.gate_proj.weight", const_cast<uint16_t*>(x.gate), cfg.intermediate_size,
              H, r * I, I);
    x.up = bump(static_cast<size_t>(I) * H);
    copy_rows(p + "mlp.up_proj.weight", const_cast<uint16_t*>(x.up), cfg.intermediate_size, H,
              r * I, I);
    x.down = bump(static_cast<size_t>(H) * I);
    copy_cols(p + "mlp.down_proj.weight", const_cast<uint16_t*>(x.down), H,
              cfg.intermediate_size, r * I, I);
    x.q_norm = bump(cfg.head_dim);
    copy_full(p + "self_attn.q_norm.weight", const_cast<uint16_t*>(x.q_norm), {cfg.head_dim});
    x.k_norm = bump(cfg.head_dim);
    copy_full(p + "self_attn.k_norm.weight", const_cast<uint16_t*>(x.k_norm), {cfg.head_dim});
  }
  w.fc = bump(static_cast<size_t>(nTaps) * H * H);
  {
    const TensorInfo* t = need("fc.weight");
    check(t, DType::BF16, {H, nTaps * H});
    colbuf.resize(static_cast<size_t>(nTaps) * H * H);
    const auto* src = static_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < H; ++i)
      for (int64_t s = 0; s < nTaps; ++s)
        std::memcpy(colbuf.data() + (static_cast<size_t>(s) * H + i) * H,
                    src + static_cast<size_t>(i) * nTaps * H + static_cast<size_t>(s) * H,
                    static_cast<size_t>(H) * 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(const_cast<uint16_t*>(w.fc), colbuf.data(), colbuf.size() * 2,
                                 cudaMemcpyHostToDevice, stream));
  }
  w.hidden_norm = bump(H);
  copy_full("hidden_norm.weight", const_cast<uint16_t*>(w.hidden_norm), {H});
  w.norm = bump(H);
  copy_full("norm.weight", const_cast<uint16_t*>(w.norm), {H});
  if (cfg.dspine) {
    const int64_t C = cfg.code_size, R = cfg.code_rank, L = cfg.num_hidden_layers;
    auto copy_bridge = [&](const std::string& name, const uint16_t*& dst,
                           std::vector<int64_t> shape) {
      const size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                       [](size_t a, int64_t b) { return a * static_cast<size_t>(b); });
      dst = bump(n);
      copy_full(name, const_cast<uint16_t*>(dst), std::move(shape));
    };
    copy_bridge("selected_prefix_bridge.codebook", w.codebook, {cfg.vocab_size, C});
    copy_bridge("selected_prefix_bridge.read_weight", w.read_weight, {C, H});
    copy_bridge("selected_prefix_bridge.early_read_weights", w.early_read_weights, {L - 1, C, H});
    copy_bridge("selected_prefix_bridge.feature_write_weights", w.feature_write_weights, {L, H, R});
    copy_bridge("selected_prefix_bridge.gate.weight", w.bridge_gate, {R, C});
    copy_bridge("selected_prefix_bridge.predecessor_gate.weight", w.predecessor_gate, {R, C});
    copy_bridge("selected_prefix_bridge.prev.weight", w.previous_value, {R, C});
    copy_bridge("selected_prefix_bridge.out.weight", w.bridge_output, {C, R});
    copy_bridge("selected_prefix_bridge.gate.bias", w.bridge_gate_bias, {R});
    const auto require_version = [&](const std::string& name) {
      const TensorInfo* t = need(name);
      check(t, DType::I64, {});
      int64_t value = 0;
      std::memcpy(&value, t->data, sizeof(value));
      if (value != 1) throw std::runtime_error("load_dflash_weights: unsupported DSpine version tensor " + name);
    };
    for (const char* name : {"draft_block_causal_version", "last_write_refine_version",
                             "selected_prefix_bridge.independent_write_version",
                             "selected_prefix_bridge.terminal_disabled_version"})
      require_version(name);
    const TensorInfo* terminal = find("selected_prefix_bridge.out.weight");
    const auto* terminal_values = static_cast<const uint16_t*>(terminal->data);
    if (std::any_of(terminal_values, terminal_values + terminal->numel(),
                    [](uint16_t x) { return x != 0; }))
      throw std::runtime_error("load_dflash_weights: terminal-disabled DSpine output weight is nonzero");
  }
  if (off * 2 != w.bytes)
    throw std::logic_error("load_dflash_weights: arena accounting mismatch");
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // host shard and column buffers die on return
  return w;
}

DFlashWeights::~DFlashWeights() {
  cudaFree(arena);
  arena = nullptr;
  bytes = 0;
}

}  // namespace dgpp
