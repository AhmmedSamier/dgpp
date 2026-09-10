#include "models/qwen/loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the gated
// residual sites and the final mixers, routers and shared gates, the whole
// indexer, the PLE's norms/conv/buffers and the table's scale, the head
// norms of GDN and QSA, the draft head's own tensors, and the embedding
// (a row gather; sharding it buys nothing in v1). Everything else is a
// slice: GDN/QSA projections, both expert classes, the PLE projections,
// the n-gram shards, the lm head under VocabSharded.
bool is_replicated(const QwenExpectedTensor& e) {
  switch (e.cls) {
    case QwenWeightClass::Embed:
    case QwenWeightClass::Mixer:
    case QwenWeightClass::Gr:
    case QwenWeightClass::Router:
    case QwenWeightClass::QsaIndexer:
    case QwenWeightClass::Ple:
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::PleTable:
      return e.role == QwenTensorRole::NgramScale;
    case QwenWeightClass::Gdn:
      return ends_with(e.name, "linear_attn.norm.weight");
    case QwenWeightClass::Qsa:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case QwenWeightClass::LmHead:
    case QwenWeightClass::SharedExpert:
    case QwenWeightClass::RoutedExpert:
      return false;
  }
  return false;
}

int gcd_int(int a, int b) { return std::gcd(a, b); }

std::pair<int, int> lm_head_slice(const QwenTextConfig& cfg, int rank, int world) {
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

// The per-class builders (loaders/weight_build.hpp's primitives).
struct QwenLoaderFamily::Builder : WeightBuilder<QwenExpectedTensor> {
  const QwenTextConfig& cfg;
  const QwenLocalGeometry& geo;
  QwenLayerResident& out;

  Builder(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
           const std::vector<QwenExpectedTensor>& table_,
           const std::unordered_map<std::string, const QwenExpectedTensor*>& by_name_,
           LayerBump& bump_, QwenLayerResident& out_,
           const std::unordered_map<std::string, const TensorInfo*>& tensors_,
           std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_,
                                          copy_, geo_.rank, geo_.world, "qwen loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const QwenExpectedTensor& e) const override { return is_replicated(e); }

  // A contiguous BF16 row range of a source matrix into a running
  // destination (the GDN's segmented q|k|v merge): rows [src_row, +rows)
  // of `name` land at `dst` + dst_row * width. Returns the source's row
  // width in elements.
  int64_t copy_rows_into(const std::string& name, int64_t src_row, int64_t rows,
                         uint16_t* dst, int64_t dst_row) {
    const QwenExpectedTensor& e = expected(name);
    const int64_t width = static_cast<int64_t>(e.numel()) / e.shape[0];
    check_range(name, src_row, rows, e.shape[0]);
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst) + static_cast<size_t>(dst_row) * width,
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(src_row) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
      // Not consumed: a later segment of the same tensor may follow.
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
    return width;
  }

  void build_gr(const std::string& p, QwenGrResident& g, bool combine) {
    g.hc_norm = load_bf16(p + "hc_norm.weight");
    g.down = load_bf16(p + "input_mix_weight_down.weight");
    g.up = load_bf16(p + "input_mix_weight_up.weight");
    g.inject = combine ? load_bf16(p + "block_inject_weight.weight") : nullptr;
  }

  void build_gdn(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // in_proj_qkv rows [q | k | v] at the local geometry: this rank's key
    // heads' q and k rows, its value heads' v rows. world=1: the whole
    // matrix, byte for byte.
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    uint16_t* qkv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(H) * 2));
    const std::string qkv_name = p + "in_proj_qkv.weight";
    copy_rows_into(qkv_name, r * lk * dk, lk * dk, qkv, 0);
    copy_rows_into(qkv_name, K + r * lk * dk, lk * dk, qkv, lk * dk);
    copy_rows_into(qkv_name, 2 * K + r * lv * dv, lv * dv, qkv, 2 * lk * dk);
    if (copy) consumed(source(qkv_name));
    g.in_proj_qkv = qkv;
    // The conv channels follow the same three segments ([C, 1, w] rows).
    const int64_t w = cfg.gdn_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(w) * 2));
    const std::string conv_name = p + "conv1d.weight";
    copy_rows_into(conv_name, r * lk * dk, lk * dk, conv, 0);
    copy_rows_into(conv_name, K + r * lk * dk, lk * dk, conv, lk * dk);
    copy_rows_into(conv_name, 2 * K + r * lv * dv, lv * dv, conv, 2 * lk * dk);
    if (copy) consumed(source(conv_name));
    g.conv = conv;
    g.in_proj_z = load_bf16_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    g.in_proj_a = load_bf16_rows(p + "in_proj_a.weight", r * lv, lv);
    g.in_proj_b = load_bf16_rows(p + "in_proj_b.weight", r * lv, lv);
    g.a_log = load_bf16_as_f32(p + "A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "norm.weight");
    g.out_proj = load_bf16_cols(p + "out_proj.weight", r * lv * dv, lv * dv);
  }

  void build_qsa(const std::string& p) {
    const int64_t d = cfg.head_dim;
    QwenQsaResident& a = out.qsa;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    a.q_proj = load_bf16_rows(p + "q_proj.weight", static_cast<int64_t>(geo.head_begin) * 2 * d,
                              static_cast<int64_t>(geo.local_heads) * 2 * d);
    a.k_proj = load_bf16_rows(p + "k_proj.weight", static_cast<int64_t>(geo.kv_head_begin) * d,
                              static_cast<int64_t>(geo.local_kv_heads) * d);
    a.v_proj = load_bf16_rows(p + "v_proj.weight", static_cast<int64_t>(geo.kv_head_begin) * d,
                              static_cast<int64_t>(geo.local_kv_heads) * d);
    a.o_proj = load_bf16_cols(p + "o_proj.weight", static_cast<int64_t>(geo.head_begin) * d,
                              static_cast<int64_t>(geo.local_heads) * d);
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.k_norm = load_bf16(p + "k_norm.weight");
    a.index_qk_proj = load_bf16(p + "indexer.index_qk_proj.weight");
    a.index_q_norm = load_bf16(p + "indexer.q_layernorm.weight");
    a.index_k_norm = load_bf16(p + "indexer.k_layernorm.weight");
  }

  void build_moe(const std::string& p) {
    QwenMoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    m.shared_gate = load_bf16(p + "shared_expert_gate.weight");
    const int64_t S = geo.local_shared_inter, I = geo.local_inter;
    const int64_t r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.scale_block = geo.scale_block;
    const std::string sp = p + "shared_expert.";
    m.shared[0] = load_bf16_rows(sp + "gate_proj.weight", r * S, S);
    m.shared[1] = load_bf16_rows(sp + "up_proj.weight", r * S, S);
    m.shared[2] = load_bf16_cols(sp + "down_proj.weight", r * S, S);
    const int E = cfg.num_experts;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      m.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", r * I, I, geo.scale_block);
    }
  }

  void build_ple(const std::string& p) {
    QwenPleResident& l = out.ple;
    const QwenNgramGeometry g = cfg.ngram_geometry();
    const int64_t hd = g.head_dim;
    l.hash_heads = geo.hash_heads;
    l.hash_head_begin = geo.hash_head_begin;
    l.row_begin = geo.table_row_begin;
    l.rows = geo.table_rows;
    const int64_t c0 = static_cast<int64_t>(geo.hash_head_begin) * hd;
    const int64_t cn = static_cast<int64_t>(geo.hash_heads) * hd;
    l.key_proj = load_bf16_cols(p + "key_proj.weight", c0, cn);
    l.value_proj = load_bf16_cols(p + "value_proj.weight", c0, cn);
    l.norm_key = load_bf16(p + "norm_key.weight");
    l.norm_query = load_bf16(p + "norm_query.weight");
    l.norm_conv = load_bf16(p + "norm_conv.weight");
    l.conv = load_bf16(p + "conv1d.weight");
    // The hash buffers ride along (replicated; the stream verified them
    // against the config at construction), and the table's scale.
    const std::string ep = p + "ple_embedding.";
    (void)load_raw(ep + "layer_multipliers");
    (void)load_raw(ep + "ngram_heads_offsets");
    (void)load_raw(ep + "ngram_heads_vocab_sizes");
    const std::string scale_name = ep + "ngram_embedding.weight_scale";
    (void)load_raw(scale_name);
    if (copy) {
      uint16_t bits;
      std::memcpy(&bits, tensors.at(scale_name)->data, 2);
      l.table_scale = bf16_bits_to_float(bits);
    }
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      fail("layer index out of range: " + std::to_string(layer));
    const bool is_mtp = layer == cfg.mtp_layer();
    const std::string p = qwen_layer_prefix(cfg, layer);
    out.layer = layer;
    out.kind = is_mtp ? QwenLayerKind::Qsa : cfg.layers[layer];
    out.has_ple = !is_mtp && layer == cfg.ple_layer();
    if (out.has_ple) build_ple(p + "ple.");
    build_gr(p + "attn_hyper_connection.", out.attn_gr, true);
    if (out.kind == QwenLayerKind::Gdn)
      build_gdn(p + "linear_attn.");
    else
      build_qsa(p + "self_attn.");
    build_gr(p + "mlp_hyper_connection.", out.mlp_gr, true);
    build_moe(p + "mlp.");
  }
};

// ---------------------------------------------------------------------------

QwenLocalGeometry QwenLocalGeometry::from_config(const QwenTextConfig& cfg, int rank, int world,
                                                 QwenHeadSharding head) {
  if (world > 1) qwen_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("qwen loader: rank/world out of range");
  QwenLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_key_heads = cfg.gdn_key_heads / world;
  g.local_value_heads = cfg.gdn_value_heads / world;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  // The rank's query heads must belong to its kv head(s).
  const int q_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
  if (g.head_begin / q_per_kv != g.kv_head_begin ||
      (g.head_begin + g.local_heads - 1) / q_per_kv != g.kv_head_begin + g.local_kv_heads - 1)
    throw std::invalid_argument("qwen loader: the query heads of a rank straddle kv heads");
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_intermediate_size / world;
  g.scale_block = gcd_int(128, static_cast<int>(g.local_inter));
  if (!cfg.ple_layer_ids.empty()) {
    const QwenNgramGeometry ng = cfg.ngram_geometry();
    g.hash_heads = ng.heads / world;
    g.hash_head_begin = g.hash_heads * rank;
    g.table_row_begin = ng.head_offset[static_cast<size_t>(g.hash_head_begin)];
    const int last = g.hash_head_begin + g.hash_heads;  // one past
    const int64_t end = last < ng.heads ? ng.head_offset[static_cast<size_t>(last)] : ng.total_rows;
    g.table_rows = end - g.table_row_begin;
  }
  if (head == QwenHeadSharding::VocabSharded) {
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

void QwenLoaderFamily::validate_binding(const QwenTextConfig& cfg, const PresentMap& present) {
  const QwenBindReport rep = qwen_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "qwen loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The checkpoint's stored hash buffers must equal the config's derivation
// (docs §1.7): a disagreement means a different hash, i.e. a different
// model, and is refused before a byte of weights moves.
void QwenLoaderFamily::check_sources(const QwenTextConfig& cfg_, const LoaderTensorMap& tensors_) {
  if (cfg_.ple_layer_ids.empty()) return;
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  auto read_i64 = [&](const std::string& name, size_t n) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.numel() != n) throw std::runtime_error("qwen loader: '" + name + "' has the wrong length");
    std::vector<int64_t> v(n);
    std::memcpy(v.data(), t.data, n * 8);
    return v;
  };
  const std::vector<int64_t> mult = read_i64(p + "layer_multipliers", static_cast<size_t>(cfg_.ngram_size));
  const std::vector<int64_t> sizes = read_i64(p + "ngram_heads_vocab_sizes", static_cast<size_t>(g.heads));
  const std::vector<int64_t> offs = read_i64(p + "ngram_heads_offsets", static_cast<size_t>(g.heads));
  if (mult != g.multipliers || sizes != g.head_vocab || offs != g.head_offset)
    throw std::runtime_error(
        "qwen loader: the checkpoint's n-gram hash buffers (layer_multipliers, "
        "ngram_heads_vocab_sizes, ngram_heads_offsets) disagree with the config's "
        "derivation — a different hash, refused");
}

bool QwenLoaderFamily::digest_included(const QwenExpectedTensor& e) { return is_replicated(e); }

// The packed column slices (a pack reads its source after the builder
// returns): the classes whose sources the one-pass load drops afterwards.
bool QwenLoaderFamily::discard_after_pack(const QwenExpectedTensor& e) {
  return e.cls == QwenWeightClass::Ple || e.cls == QwenWeightClass::Gdn ||
         e.cls == QwenWeightClass::Qsa || e.cls == QwenWeightClass::SharedExpert;
}

size_t QwenLoaderFamily::globals_bytes(const QwenTextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t W = static_cast<size_t>(cfg.hyper_width());
  const size_t r = static_cast<size_t>(cfg.hc_lowrank);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);                  // embed
  b += align_up_256(static_cast<size_t>(QwenLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  b += align_up_256(W * 2) + align_up_256(r * W * 2) + align_up_256(W * r * 2);   // mixer
  if (cfg.mtp_layer() >= 0) {
    b += 2 * align_up_256(H * H * 2);                          // fc_embedding, fc_hidden
    b += align_up_256(H * 2) + align_up_256(W * 2);            // pre_fc norms
    b += align_up_256(W * 2) + align_up_256(r * W * 2) + align_up_256(W * r * 2);  // mtp mixer
  }
  return b;
}

size_t QwenLoaderFamily::extra_resident_bytes(const QwenTextConfig& cfg, int rank, int world) {
  return QwenLayerStream::ngram_table_bytes(cfg, rank, world);
}

// The restored PLE layer's host-side scale: from the source when mapped.
void QwenLoaderFamily::after_restore(const QwenTextConfig& cfg, int layer,
                                     const LoaderTensorMap& tensors, QwenLayerResident& out) {
  if (!out.has_ple || tensors.empty()) return;
  const std::string name = qwen_layer_prefix(cfg, layer) + "ple.ple_embedding.ngram_embedding.weight_scale";
  uint16_t bits;
  std::memcpy(&bits, tensors.at(name)->data, 2);
  out.ple.table_scale = bf16_bits_to_float(bits);
}

void QwenLoaderFamily::build_globals(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
                                     const LoaderTensorMap& tensors_, LayerBump& bump,
                                     QwenGlobalsResident& globals_, uint64_t& source_bytes_,
                                     uint64_t& verbatim_bytes_, LoaderHeadSharding head_) {
  LayerBump* globals_bump_ = &bump;
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(t.nbytes()));
    std::memcpy(globals_bump_->host(dst), t.data, t.nbytes());
    source_bytes_ += t.nbytes();
    verbatim_bytes_ += t.nbytes();
    return dst;
  };
  auto copy_gr = [&](const std::string& p, QwenGrResident& g) {
    g.hc_norm = copy_global(p + "hc_norm.weight");
    g.down = copy_global(p + "input_mix_weight_down.weight");
    g.up = copy_global(p + "input_mix_weight_up.weight");
    g.inject = nullptr;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  {
    const TensorInfo& t = lookup("lm_head.weight");
    const size_t row_bytes = static_cast<size_t>(cfg_.hidden_size) * 2;
    const int begin = geo_.lm_vocab_begin, count = geo_.lm_vocab_count;
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(static_cast<size_t>(count) * row_bytes));
    std::memcpy(globals_bump_->host(dst),
                static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
                static_cast<size_t>(count) * row_bytes);
    source_bytes_ += static_cast<size_t>(count) * row_bytes;
    if (head_ == LoaderHeadSharding::Full) verbatim_bytes_ += static_cast<size_t>(count) * row_bytes;
    globals_.lm_head = dst;
    globals_.lm_vocab_begin = begin;
    globals_.lm_vocab_count = count;
  }
  copy_gr("model.language_model.hyper_connection_mixer.", globals_.mixer);
  if (cfg_.mtp_layer() >= 0) {
    globals_.mtp_fc_embedding = copy_global("mtp.fc_embedding.weight");
    globals_.mtp_fc_hidden = copy_global("mtp.fc_hidden.weight");
    globals_.mtp_pre_fc_norm_embedding = copy_global("mtp.pre_fc_norm_embedding.weight");
    globals_.mtp_pre_fc_norm_hidden = copy_global("mtp.pre_fc_norm_hidden.weight");
    copy_gr("mtp.hyper_connection_mixer.", globals_.mtp_mixer);
  }
}

template class ResidentLayerStream<QwenLoaderFamily>;

// ---------------------------------------------------------------------------

QwenLayerStream::QwenLayerStream(const QwenTextConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, QwenResidency residency,
                                 QwenHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<QwenLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  open_resident_image();
}

QwenLayerStream::~QwenLayerStream() {
  if (table_device_) cudaFree(table_device_);
}

size_t QwenLayerStream::ngram_table_bytes(const QwenTextConfig& cfg, int rank, int world) {
  if (cfg.ple_layer_ids.empty()) return 0;
  const QwenLocalGeometry g = QwenLocalGeometry::from_config(cfg, rank, world, QwenHeadSharding::Full);
  return static_cast<size_t>(g.table_rows) * static_cast<size_t>(cfg.ngram_geometry().head_dim);
}

void QwenLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& QwenLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

// ---- the n-gram table ----------------------------------------------------------

const QwenNgramTableResident& QwenLayerStream::load_ngram_table() {
  if (table_.rows_e4m3 || cfg_.ple_layer_ids.empty()) return table_;
  if (sources_released_)
    throw std::runtime_error("qwen loader: load_ngram_table after the checkpoint sources were released");
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const int64_t hd = g.head_dim;
  const int64_t row_begin = geo_.table_row_begin, rows = geo_.table_rows;
  const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(hd);
  DGPP_CUDA_OK(cudaMalloc(&table_device_, bytes));
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  const int64_t cap = qwen_ngram_shard_capacity(cfg_);
  const int64_t row_end = row_begin + rows;
  uint8_t* dst = static_cast<uint8_t*>(table_device_);
  size_t done = 0;
  for (int s = 0; s < cfg_.split_ngram_parts; ++s) {
    const int64_t s_begin = static_cast<int64_t>(s) * cap;
    const int64_t s_end = s_begin + qwen_ngram_shard_rows(cfg_, s);
    const int64_t lo = std::max(row_begin, s_begin), hi = std::min(row_end, s_end);
    if (lo >= hi) continue;
    const std::string name = p + "ngram_embedding.shard_" + std::to_string(s) + ".weight";
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.owner) t.owner->prefetch(t);
    const uint8_t* src = static_cast<const uint8_t*>(t.data) + static_cast<size_t>(lo - s_begin) * hd;
    size_t remaining = static_cast<size_t>(hi - lo) * static_cast<size_t>(hd);
    while (remaining > 0) {
      const size_t chunk = std::min(remaining, staging_bytes_);
      std::memcpy(staging_, src, chunk);
      DGPP_CUDA_OK(cudaMemcpyAsync(dst + done, staging_, chunk, cudaMemcpyHostToDevice, stream_));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      src += chunk;
      done += chunk;
      remaining -= chunk;
    }
    if (residency_ == QwenResidency::Resident && t.owner) t.owner->discard(t);
  }
  if (done != bytes)
    throw std::runtime_error("qwen loader: the n-gram shards did not cover this rank's rows");
  source_bytes_ += bytes;
  {
    const std::string name = p + "ngram_embedding.weight_scale";
    uint16_t bits;
    std::memcpy(&bits, tensors_.at(name)->data, 2);
    table_.scale = bf16_bits_to_float(bits);
  }
  table_.rows_e4m3 = dst;
  table_.row_begin = row_begin;
  table_.rows = rows;
  table_.head_dim = static_cast<int>(hd);
  table_.bytes = bytes;
  DGPP_LOG_INFO("qwen loader: rank {} n-gram table rows [{}, {}) resident ({:.2f} GiB, scale {})",
                rank_, row_begin, row_end, bytes / (1024.0 * 1024.0 * 1024.0), table_.scale);
  return table_;
}

// ---- sources and digests --------------------------------------------------------

}  // namespace dgpp
