#include "models/llama/loader.hpp"

#include <cstring>
#include <stdexcept>

#include "kernels/bf12_gemv.hpp"

namespace dgpp {
namespace {

// The replicated set (rank-invariant reads at world > 1): the layer norms,
// the embedding and the final norm. Everything else is a slice: the
// attention projections, every MLP matrix, the lm head under VocabSharded.
bool is_replicated(const LlamaExpectedTensor& e) {
  switch (e.cls) {
    case LlamaWeightClass::Embed:
    case LlamaWeightClass::FinalNorm:
    case LlamaWeightClass::LayerNorm:
      return true;
    case LlamaWeightClass::Attention:
    case LlamaWeightClass::DenseMlp:
    case LlamaWeightClass::LmHead:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct LlamaLoaderFamily::Builder : WeightBuilder<LlamaExpectedTensor> {
  const LlamaTextConfig& cfg;
  const LlamaLocalGeometry& geo;
  LlamaLayerResident& out;

  Builder(const LlamaTextConfig& cfg_, const LlamaLocalGeometry& geo_,
          const std::vector<LlamaExpectedTensor>& table_,
          const std::unordered_map<std::string, const LlamaExpectedTensor*>& by_name_,
          LayerBump& bump_, LlamaLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<LlamaExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                           geo_.rank, geo_.world, "llama loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const LlamaExpectedTensor& e) const override { return is_replicated(e); }

  void build_attention(const std::string& p) {
    const int64_t d = cfg.head_dim;
    LlamaAttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * d, qn = static_cast<int64_t>(geo.local_heads) * d;
    const int64_t k0 = static_cast<int64_t>(geo.kv_head_begin) * d, kn = static_cast<int64_t>(geo.local_kv_heads) * d;
    // Packable: the model's 12-bit companions replace the four projections
    // (WeightBuilder::grant_bf16 — aside under bf12-only residency).
    a.q_proj = load_bf16_rows(p + "q_proj.weight", q0, qn, /*packable=*/true);
    a.k_proj = load_bf16_rows(p + "k_proj.weight", k0, kn, /*packable=*/true);
    a.v_proj = load_bf16_rows(p + "v_proj.weight", k0, kn, /*packable=*/true);
    a.o_proj = load_bf16_cols(p + "o_proj.weight", q0, qn, /*packable=*/true);
  }

  void build_mlp(const std::string& p) {
    const int64_t I = geo.local_inter, r = rank;
    LlamaDenseMlpResident& m = out.mlp;
    m.local_inter = I;
    m.gate = load_bf16_rows(p + "gate_proj.weight", r * I, I, /*packable=*/true);
    m.up = load_bf16_rows(p + "up_proj.weight", r * I, I, /*packable=*/true);
    m.down = load_bf16_cols(p + "down_proj.weight", r * I, I, /*packable=*/true);
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers) fail("layer index out of range: " + std::to_string(layer));
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    out.layer = layer;
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    build_attention(p + "self_attn.");
    build_mlp(p + "mlp.");
  }
};

namespace {

std::pair<int, int> lm_head_slice(const LlamaTextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

// ---------------------------------------------------------------------------

LlamaLocalGeometry LlamaLocalGeometry::from_config(const LlamaTextConfig& cfg, int rank, int world,
                                                   LlamaHeadSharding head) {
  if (world > 1) llama_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("llama loader: rank/world out of range");
  LlamaLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  g.local_inter = cfg.intermediate_size / world;
  if (head == LlamaHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void LlamaLoaderFamily::validate_binding(const LlamaTextConfig& cfg, const PresentMap& present) {
  const LlamaBindReport rep = llama_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "llama loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

void LlamaLoaderFamily::check_sources(const Config&, const LoaderTensorMap&) {}

bool LlamaLoaderFamily::digest_included(const LlamaExpectedTensor& e) { return is_replicated(e); }

// The packed column slices (o_proj, down_proj) read their sources after the
// builder returns: the attention/MLP classes' sources are dropped after the
// packs.
bool LlamaLoaderFamily::discard_after_pack(const LlamaExpectedTensor& e) {
  return e.cls == LlamaWeightClass::Attention || e.cls == LlamaWeightClass::DenseMlp;
}

size_t LlamaLoaderFamily::globals_bytes(const LlamaTextConfig& cfg, int rank, int world,
                                        LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(LlamaLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  return b;
}

size_t LlamaLoaderFamily::globals_side_bytes(const LlamaTextConfig& cfg, int rank, int world,
                                             LoaderHeadSharding head) {
  const int count = LlamaLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void LlamaLoaderFamily::build_globals(const LlamaTextConfig& cfg, const LlamaLocalGeometry& geo,
                                      const LoaderTensorMap& tensors, LayerBump& bump,
                                      LlamaGlobalsResident& out, uint64_t& source_bytes,
                                      uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("llama loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    return dst;
  };
  out.embed = copy_global("model.embed_tokens.weight");
  out.final_norm = copy_global("model.norm.weight");
  const TensorInfo& t = lookup("lm_head.weight");
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  // The head is packable (the model's 12-bit companion replaces it).
  uint16_t* dst = static_cast<uint16_t*>(bf12_shape_ok(count, cfg.hidden_size)
                                             ? bump.alloc_side(static_cast<size_t>(count) * row_bytes)
                                             : bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<LlamaLoaderFamily>;

// ---------------------------------------------------------------------------

LlamaLayerStream::LlamaLayerStream(const LlamaTextConfig& cfg, const std::string& checkpoint_dir,
                                   int rank, int world, LlamaResidency residency,
                                   LlamaHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<LlamaLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                             resident_mtp) {
  open_resident_image();
}

void LlamaLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& LlamaLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
