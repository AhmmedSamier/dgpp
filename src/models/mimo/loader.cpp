#include "models/mimo/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

// The replicated set (rank-invariant reads at world > 1): the layer
// norms, the router and its bias, the draft head's own tensors, the
// embedding and the final norm. Everything else is a slice: the fused
// projection's chunks, o_proj's columns, the sink's heads, every fp8 MLP
// matrix, every MXFP4 expert, the lm head under VocabSharded.
bool is_replicated(const MimoExpectedTensor& e) {
  switch (e.cls) {
    case MimoWeightClass::Embed:
    case MimoWeightClass::FinalNorm:
    case MimoWeightClass::LayerNorm:
    case MimoWeightClass::Router:
    case MimoWeightClass::MtpHead:
      return true;
    case MimoWeightClass::Attention:
    case MimoWeightClass::LmHead:
    case MimoWeightClass::DenseMlp:
    case MimoWeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct MimoLoaderFamily::Builder : WeightBuilder<MimoExpectedTensor> {
  const MimoTextConfig& cfg;
  const MimoLocalGeometry& geo;
  MimoLayerResident& out;

  Builder(const MimoTextConfig& cfg_, const MimoLocalGeometry& geo_,
          const std::vector<MimoExpectedTensor>& table_,
          const std::unordered_map<std::string, const MimoExpectedTensor*>& by_name_,
          LayerBump& bump_, MimoLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<MimoExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                          geo_.rank, geo_.world, "mimo loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const MimoExpectedTensor& e) const override { return is_replicated(e); }

  // ---- MXFP4 pairs (e2m1 pairs + e8m0 per 32): `base`.weight / .weight_scale
  struct Fp4Source {
    const MimoExpectedTensor* w;
    const MimoExpectedTensor* s;
    int64_t N, K;
  };
  Fp4Source fp4_source(const std::string& base) {
    Fp4Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".weight_scale");
    if (f.w->role != MimoTensorRole::Fp4Payload || f.s->role != MimoTensorRole::Fp4Scale ||
        f.w->shape.size() != 2 || f.s->shape.size() != 2)
      fail("'" + base + "' is not an MXFP4 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1] * 2;
    fp4_check_cols(f.K, who.c_str(), kMxfp4Group);
    if (f.s->shape[0] != f.N || f.s->shape[1] != f.K / kMxfp4Group)
      fail("MXFP4 scale geometry mismatch on " + base);
    return f;
  }
  GlmFp4Matrix alloc_mxfp4(int64_t rows, int64_t cols) {
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_group = kMxfp4Group;
    q.global_scale = nullptr;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / 2)));
    q.scales = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / kMxfp4Group)));
    return q;
  }
  // Rows [row_start, +rows) of the [N, K] matrix `base`: payload and scale
  // rows are contiguous.
  GlmFp4Matrix load_mxfp4_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp4Source f = fp4_source(base);
    check_range(base, row_start, rows, f.N);
    const size_t pc = static_cast<size_t>(f.K / 2), sc = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(rows, f.K);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * pc);
    note_read(*f.s, static_cast<size_t>(rows) * sc);
    return q;
  }
  // Columns [col_start, +cols) of every row, packed: col_start on a
  // 32-block boundary.
  GlmFp4Matrix load_mxfp4_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const Fp4Source f = fp4_source(base);
    fp4_check_cols(cols, who.c_str(), kMxfp4Group);
    if (col_start % kMxfp4Group != 0)
      fail("MXFP4 column slice of '" + base + "' must start on a 32-element block boundary");
    check_range(base, col_start, cols, f.K);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(f.K / 2);
    const size_t sc = static_cast<size_t>(cols / kMxfp4Group), sc_full = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(f.N, cols);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(f.N) * sc);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kMxfp4Group, sc);
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * pc);
    note_read(*f.s, static_cast<size_t>(f.N) * sc);
    return q;
  }

  // ---- the fused qkv_proj: the rank's chunks stacked, padded per chunk ------
  // The source: [chunks_total * chunk_rows, hidden] fp8 with the scale grid
  // [chunks_total * sr, hidden / 128], sr = ceil(chunk_rows / 128), tiled
  // per chunk (vLLM's _shard_fp8_qkv_proj reads it so). The resident form:
  // chunk i of the rank's `chunks` at rows [i * stride, i * stride +
  // chunk_rows) with stride = sr * 128 (the last chunk unpadded), its
  // scale rows at [i * sr, (i + 1) * sr) — local row x's block is x / 128,
  // exactly the source chunk's, so no value is touched. The padding rows
  // are zero (a zero payload dequantizes to zero whatever its scale).
  GlmQuantMatrix load_qkv(const std::string& base, int layer, MimoAttnResident& a) {
    const MimoExpectedTensor& ep = expected(base + ".weight");
    const MimoExpectedTensor& es = expected(base + ".weight_scale_inv");
    const int64_t H = cfg.hidden_size;
    const int64_t chunk_rows = cfg.qkv_chunk_rows(layer);
    const int chunks_total = cfg.qkv_chunks();
    const int64_t sr = (chunk_rows + 127) / 128;
    const int64_t sb = H / 128;
    if (ep.shape[0] != chunk_rows * chunks_total || ep.shape[1] != H || es.shape[0] != sr * chunks_total ||
        es.shape[1] != sb)
      fail("fused qkv_proj geometry mismatch on " + base);
    const int chunks = geo.chunks;
    const int64_t stride = sr * 128;
    const int64_t rows = static_cast<int64_t>(chunks - 1) * stride + chunk_rows;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = H;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(H)));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(chunks) * static_cast<size_t>(sr) * sb * 4));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      if (ts.dtype != DType::F32) fail("'" + es.name + "' is not F32");
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      for (int i = 0; i < chunks; ++i) {
        const int64_t c = geo.chunk_begin + i;
        std::memcpy(hp + static_cast<size_t>(i) * stride * H,
                    static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(c) * chunk_rows * H,
                    static_cast<size_t>(chunk_rows) * H);
        if (i + 1 < chunks && stride > chunk_rows)
          std::memset(hp + (static_cast<size_t>(i) * stride + chunk_rows) * H, 0,
                      static_cast<size_t>(stride - chunk_rows) * H);
        std::memcpy(hs + static_cast<size_t>(i) * sr * sb,
                    static_cast<const float*>(ts.data) + static_cast<size_t>(c) * sr * sb,
                    static_cast<size_t>(sr) * sb * 4);
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(chunks) * chunk_rows * H);
    note_read(es, static_cast<size_t>(chunks) * sr * sb * 4);
    a.chunks = chunks;
    a.chunk_rows = chunk_rows;
    a.chunk_stride = stride;
    a.q_per_chunk = cfg.num_attention_heads / chunks_total;
    a.kv_per_chunk = cfg.kv_heads_of(layer) / chunks_total;
    return q;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p, int layer) {
    MimoAttnResident& a = out.attn;
    a.swa = cfg.is_swa_layer(layer);
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = a.swa ? geo.local_swa_kv_heads : geo.local_kv_heads;
    a.kv_head_begin = a.swa ? geo.swa_kv_head_begin : geo.kv_head_begin;
    a.qkv = load_qkv(p + "qkv_proj", layer, a);
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * cfg.v_head_dim;
    const int64_t on = static_cast<int64_t>(geo.local_heads) * cfg.v_head_dim;
    // Packable: the model's 12-bit companion replaces the o_proj
    // (WeightBuilder::grant_bf16 — aside under bf12-only residency).
    a.o_proj = load_bf16_cols(p + "o_proj.weight", o0, on, /*packable=*/true);
    a.sink = cfg.sink_of(layer) ? load_bf16_as_f32(p + "attention_sink_bias", geo.head_begin, geo.local_heads) : nullptr;
  }

  void build_dense(const std::string& p) {
    const int64_t I = geo.local_dense_inter, r = rank;
    MimoDenseMlpResident& m = out.dense;
    m.local_inter = I;
    m.gate = load_quant_rows(p + "gate_proj.weight", r * I, I);
    m.up = load_quant_rows(p + "up_proj.weight", r * I, I);
    m.down = load_quant_cols(p + "down_proj.weight", r * I, I);
  }

  void build_moe(const std::string& p) {
    MimoMoeResident& m = out.moe_w;
    m.router = load_bf16(p + "gate.weight");
    m.router_bias = load_f32(p + "gate.e_score_correction_bias");
    const int64_t I = geo.local_inter, r = rank;
    m.local_inter = I;
    const int E = cfg.n_routed_experts;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      t[0] = load_mxfp4_rows(ep + "gate_proj", r * I, I);
      t[1] = load_mxfp4_rows(ep + "up_proj", r * I, I);
      t[2] = load_mxfp4_cols(ep + "down_proj", r * I, I);
    }
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer) fail("layer index out of range: " + std::to_string(layer));
    const bool is_mtp = cfg.is_mtp_layer(layer);
    const std::string p = mimo_layer_prefix(cfg, layer);
    out.layer = layer;
    out.moe = cfg.is_moe_layer(layer);
    out.swa = cfg.is_swa_layer(layer);
    if (is_mtp) {
      out.enorm = load_bf16(p + "enorm.weight");
      out.hnorm = load_bf16(p + "hnorm.weight");
      out.eh_proj = load_bf16(p + "eh_proj.weight", /*packable=*/true);
      out.final_norm = load_bf16(p + "final_layernorm.weight");
    }
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + (is_mtp ? "pre_mlp_layernorm.weight" : "post_attention_layernorm.weight"));
    build_attention(p + "self_attn.", layer);
    if (out.moe)
      build_moe(p + "mlp.");
    else
      build_dense(p + "mlp.");
  }
};

namespace {

std::pair<int, int> lm_head_slice(const MimoTextConfig& cfg, int rank, int world) {
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

MimoLocalGeometry MimoLocalGeometry::from_config(const MimoTextConfig& cfg, int rank, int world,
                                                 MimoHeadSharding head) {
  if (world > 1) mimo_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("mimo loader: rank/world out of range");
  if (cfg.qkv_chunks() % world != 0)
    throw std::invalid_argument("mimo loader: world must divide the qkv_proj chunk count");
  MimoLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.chunks = cfg.qkv_chunks() / world;
  g.chunk_begin = g.chunks * rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_kv_heads = cfg.num_key_value_heads / world;
  g.kv_head_begin = g.local_kv_heads * rank;
  g.local_swa_kv_heads = cfg.swa_num_key_value_heads / world;
  g.swa_kv_head_begin = g.local_swa_kv_heads * rank;
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_dense_inter = cfg.intermediate_size / world;
  if (head == MimoHeadSharding::VocabSharded) {
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

void MimoLoaderFamily::validate_binding(const MimoTextConfig& cfg, const PresentMap& present) {
  const MimoBindReport rep = mimo_validate_text_binding(cfg, present);
  if (rep.ok()) {
    if (rep.ignored > 0)
      DGPP_LOG_INFO("mimo loader: {} checkpoint tensors ignored (the vision / audio encoders, the draft layers past the first)",
                    rep.ignored);
    return;
  }
  std::string msg = "mimo loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

bool MimoLoaderFamily::digest_included(const MimoExpectedTensor& e) {
  return is_replicated(e) && !e.unused();
}

// The packed column slice (o_proj) reads its source after the builder
// returns: the attention class's sources are dropped after the packs.
bool MimoLoaderFamily::discard_after_pack(const MimoExpectedTensor& e) {
  return e.cls == MimoWeightClass::Attention;
}

size_t MimoLoaderFamily::globals_bytes(const MimoTextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(MimoLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  return b;
}

size_t MimoLoaderFamily::globals_side_bytes(const MimoTextConfig& cfg, int rank, int world,
                                            LoaderHeadSharding head) {
  const int count = MimoLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void MimoLoaderFamily::build_globals(const MimoTextConfig& cfg, const MimoLocalGeometry& geo,
                                     const LoaderTensorMap& tensors, LayerBump& bump,
                                     MimoGlobalsResident& out, uint64_t& source_bytes,
                                     uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("mimo loader: global tensor missing: " + name);
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

template class ResidentLayerStream<MimoLoaderFamily>;

// ---------------------------------------------------------------------------

MimoLayerStream::MimoLayerStream(const MimoTextConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, MimoResidency residency,
                                 MimoHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<MimoLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  open_resident_image();
}

void MimoLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& MimoLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
