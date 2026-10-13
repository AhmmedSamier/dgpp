#include "models/qwen_asr/loader.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp {
namespace {
std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}
}  // namespace

QwenAsrLocalGeometry QwenAsrLocalGeometry::from_config(const QwenAsrLmConfig& cfg, int rank,
                                                       int world, LoaderHeadSharding) {
  if (world != 1 || rank != 0)
    throw std::invalid_argument("qwen_asr loader: the family is world 1 (no TP path)");
  QwenAsrLocalGeometry g;
  g.lm_vocab_begin = 0;
  g.lm_vocab_count = cfg.vocab_size;
  return g;
}

struct QwenAsrLoaderFamily::Builder : WeightBuilder<QwenAsrExpectedTensor> {
  const QwenAsrLmConfig& cfg;
  QwenAsrLayerResident& out;

  Builder(const QwenAsrLmConfig& cfg_, const QwenAsrLocalGeometry& geo_,
          const std::vector<QwenAsrExpectedTensor>& table_,
          const std::unordered_map<std::string, const QwenAsrExpectedTensor*>& by_name_,
          LayerBump& bump_, QwenAsrLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenAsrExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_,
                                             copy_, geo_.rank, geo_.world, "qwen_asr loader"),
        cfg(cfg_), out(out_) {}

  bool replicated(const QwenAsrExpectedTensor& e) const override {
    return qwen_asr_is_replicated(e.cls);
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers)
      fail("layer index out of range: " + std::to_string(layer));
    const std::string p = qwen_asr_layer_prefix(cfg, layer);
    out.layer = layer;
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    out.attn.q_proj = load_bf16(p + "self_attn.q_proj.weight");
    out.attn.k_proj = load_bf16(p + "self_attn.k_proj.weight");
    out.attn.v_proj = load_bf16(p + "self_attn.v_proj.weight");
    out.attn.o_proj = load_bf16(p + "self_attn.o_proj.weight");
    out.attn.q_norm = load_bf16(p + "self_attn.q_norm.weight");
    out.attn.k_norm = load_bf16(p + "self_attn.k_norm.weight");
    out.attn.local_heads = cfg.num_attention_heads;
    out.attn.head_begin = 0;
    out.attn.local_kv_heads = cfg.num_key_value_heads;
    out.attn.kv_head_begin = 0;
    out.gate = load_bf16(p + "mlp.gate_proj.weight");
    out.up = load_bf16(p + "mlp.up_proj.weight");
    out.down = load_bf16(p + "mlp.down_proj.weight");
  }
};

void QwenAsrLoaderFamily::validate_binding(const Config& c, const PresentMap& present) {
  const auto r = qwen_asr_validate_text_binding(c, present);
  if (r.ok()) return;
  std::string msg = std::string(who()) + ": binding mismatch (" + std::to_string(r.missing) +
                    " missing, " + std::to_string(r.dtype_mismatch) + " dtype, " +
                    std::to_string(r.shape_mismatch) + " shape, " +
                    std::to_string(r.unexpected) + " unexpected)";
  for (const auto& e : r.errors) msg += "\n  " + e;
  throw std::runtime_error(msg);
}

void QwenAsrLoaderFamily::check_sources(const Config& c, const LoaderTensorMap& tensors) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error(std::string(who()) + ": tensor not in checkpoint: " + name);
    return *it->second;
  };
  // Tied weights: the duplicated lm_head must BE the embedding (the loader
  // serves one from the other; a distinct copy is not implemented).
  const TensorInfo& dup = lookup("thinker.lm_head.weight");
  const TensorInfo& orig = lookup("thinker.model.embed_tokens.weight");
  const size_t bytes = dup.nbytes();
  if (bytes != orig.nbytes())
    throw std::runtime_error("qwen_asr loader: thinker.lm_head.weight differs in size from the "
                             "embedding");
  const size_t window = std::min<size_t>(bytes, size_t{1} << 20);
  for (const size_t off : {size_t{0}, (bytes - window) / 2, bytes - window})
    if (std::memcmp(static_cast<const uint8_t*>(dup.data) + off,
                    static_cast<const uint8_t*>(orig.data) + off, window) != 0)
      throw std::runtime_error("qwen_asr loader: thinker.lm_head.weight is not a copy of the "
                               "embedding (tie_word_embeddings: the head aliases it)");
}

size_t QwenAsrLoaderFamily::globals_bytes(const Config& c, int rank, int world,
                                          LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  Geometry::from_config(c, rank, world, head);  // validates world
  return align_up_256(static_cast<size_t>(c.vocab_size) * H * 2) + align_up_256(H * 2);
}

void QwenAsrLoaderFamily::build_globals(const Config& c, const Geometry& geo,
                                        const LoaderTensorMap& tensors, LayerBump& bump,
                                        GlobalsResident& out, uint64_t& source_bytes,
                                        uint64_t& verbatim_bytes, LoaderHeadSharding) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error(std::string(who()) + ": global tensor missing: " + name);
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
  out.embed = copy_global("thinker.model.embed_tokens.weight");
  out.final_norm = copy_global("thinker.model.norm.weight");
  out.lm_head = out.embed;  // tied (check_sources compared the copy)
  out.lm_vocab_begin = geo.lm_vocab_begin;
  out.lm_vocab_count = c.vocab_size;
}

template class ResidentLayerStream<QwenAsrLoaderFamily>;

// ---------------------------------------------------------------------------

QwenAsrLayerStream::QwenAsrLayerStream(const QwenAsrLmConfig& cfg,
                                       const std::string& checkpoint_dir)
    : ResidentLayerStream<QwenAsrLoaderFamily>(cfg, checkpoint_dir, 0, 1, LoaderResidency::Resident,
                                               LoaderHeadSharding::Full, false) {
  open_resident_image();
}

void QwenAsrLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& QwenAsrLayerStream::resident_image_dir() { return resident_image_dir_storage(); }
}  // namespace dgpp
