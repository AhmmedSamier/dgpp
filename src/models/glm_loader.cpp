#include "models/glm_loader.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "common/cuda_check.hpp"
#include "kernels/fp8_dequant.hpp"

namespace dgpp {

namespace {

constexpr size_t kAllocAlign = 256;

size_t align_up_256(size_t b) {
  return (b + kAllocAlign - 1) & ~(kAllocAlign - 1);
}

// One dequant job gathered during the copy phase, launched after all CPU
// writes complete (host access to managed memory during kernel execution is
// the gray zone this two-phase build exists to avoid).
struct DequantJob {
  const uint8_t* payload;
  const float* scales;
  uint16_t* out;
  int64_t rows;
  int64_t cols;
};

// One strided pack copy, executed AFTER the dequant jobs land (the DSA
// o_proj pack's source is a bridge buffer the kernel writes; the KDA
// o_proj pack's source is host mmap, but one phase for all packs keeps
// "packs happen after dequants" a single rule).
struct PackJob {
  const uint16_t* src;
  uint16_t* dst;
  size_t src_pitch;  // bytes
  size_t dst_pitch;
  size_t width;      // <= both pitches
  size_t rows;
};

void run_pack(const PackJob& j) {
  if (j.width == j.dst_pitch && j.width == j.src_pitch) {
    std::memcpy(j.dst, j.src, j.width * j.rows);  // degenerate: contiguous
    return;
  }
  for (size_t r = 0; r < j.rows; ++r)
    std::memcpy(j.dst + r * (j.dst_pitch / 2),
                j.src + r * (j.src_pitch / 2), j.width);
}

// FNV-1a (the replicated digest's per-tensor hash; checksum class, not
// adversarial — see GlmReplicatedDigest).
uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// The name after "model.language_model.layers.{N}." — the suffix the
// replicated/bridge classifiers match. Globals never reach here (their
// classes are unambiguous).
std::string_view layer_suffix(const std::string& name) {
  size_t dots = 0, i = 0;
  for (; i < name.size() && dots < 4; ++i)
    if (name[i] == '.') ++dots;
  return dots == 4 ? std::string_view(name).substr(i) : std::string_view(name);
}

// Replicated weights (DESIGN §5.2): every rank loads these byte-identically
// and the boot digest folds them. The scale partner of a replicated
// quantized tensor is replicated too — strip the "_scale_inv" suffix before
// matching. THIS LIST IS THE SINGLE SOURCE OF TRUTH for hash_replicated()
// and the build-time read-class assertions; the build functions must read
// exactly these tensors verbatim at world>1 (the assertions below enforce
// the agreement in the other direction).
bool is_replicated(const GlmExpectedTensor& e) {
  switch (e.cls) {
    case GlmWeightClass::Embed:
    case GlmWeightClass::LmHead:
    case GlmWeightClass::FinalNorm:
    case GlmWeightClass::LayerNorm:
    case GlmWeightClass::Mhc:
    case GlmWeightClass::Router:
    case GlmWeightClass::DsaIndexer:
      return true;
    case GlmWeightClass::Mtp:
      // v1: the draft head is replicated (the forward does not run the
      // MTP layer; sharding rules arrive with the M6 MTP spec).
      return true;
    case GlmWeightClass::Kda: {
      std::string_view s = layer_suffix(e.name);
      if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
        s.remove_suffix(10);
      return s == "self_attn.f_a_proj.weight" ||
             s == "self_attn.g_a_proj.weight" ||
             s == "self_attn.o_norm.weight";
    }
    case GlmWeightClass::Dsa: {
      std::string_view s = layer_suffix(e.name);
      if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
        s.remove_suffix(10);
      return s == "self_attn.q_a_proj.weight" ||
             s == "self_attn.kv_a_proj_with_mqa.weight" ||
             s == "self_attn.q_a_layernorm.weight" ||
             s == "self_attn.kv_a_layernorm.weight";
    }
    default:
      return false;  // DenseMlp, SharedExpert, RoutedExpert: sharded
  }
}

// DSA dequant-bridge tensors (the M3 bf16 seam): read in FULL by every rank
// (the quantized row/column slice contract forbids mid-block starts at this
// seam), then sliced/packed at the bf16 level — exactly what GlmTpViews
// does to the same bridge buffers. Full-read for the byte reconcile, but
// NOT replicated (their resident outputs are sharded), so the digest skips
// them. The scale-aware GEMM seam (M4 deliverable 3's successor) will turn
// these into true quant slices — and will then carry the same 128-alignment
// contract on their dims.
bool is_dsa_bridge(const GlmExpectedTensor& e) {
  if (e.cls != GlmWeightClass::Dsa) return false;
  std::string_view s = layer_suffix(e.name);
  if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
    s.remove_suffix(10);
  return s == "self_attn.q_b_proj.weight" || s == "self_attn.o_proj.weight";
}

}  // namespace

// Managed-memory bump for streamed layer weights (defined here, pimpl'd in
// the header). Counting mode walks the same grant sequence without touching
// memory — layer_bytes() and load_layer() run the SAME build code, so the
// sizing formula cannot drift.
struct GlmLayerBump {
  void* base = nullptr;
  size_t capacity = 0;
  size_t cursor = 0;
  bool counting = false;

  ~GlmLayerBump() {
    if (base) cudaFree(base);
  }
  GlmLayerBump() = default;
  GlmLayerBump(const GlmLayerBump&) = delete;
  GlmLayerBump& operator=(const GlmLayerBump&) = delete;

  void init(size_t cap) {
    DGPP_CUDA_OK(cudaMallocManaged(&base, cap));
    capacity = cap;
    cursor = 0;
  }

  // Every grant is 256-aligned, so a layer's total is exactly the sum of
  // per-tensor aligned sizes — no inter-allocation padding surprises.
  void* alloc(size_t bytes) {
    const size_t grant = align_up_256(bytes);
    if (cursor + grant > capacity)
      throw std::runtime_error("glm loader: layer bump OOM");
    void* p = counting ? nullptr : static_cast<char*>(base) + cursor;
    cursor += grant;
    return p;
  }

  void reset() { cursor = 0; }
};

namespace {

// Everything one layer build needs. `copy` false = counting pass: identical
// allocation sequence, no memcpys, no kernel launches (source lookups stay
// inside if(copy) — the counting pass has no tensors). Names and sizes come
// from the expected table (config-derived); mmap pointers only feed copies.
//
// SHARDING (M5 d4): rank/world produce the resident layer directly at this
// rank's local geometry. world=1 is the DEGENERATE RANK 0 — every slice
// covers the full extent, every grant sequence and byte is the M4 build's,
// so the world=1 path cannot drift. The local arithmetic mirrors
// GlmTpViews::bind exactly; glm_tp_test pins the two paths bitwise.
struct BuildCtx {
  const GlmTextConfig& cfg;
  const std::vector<GlmExpectedTensor>& table;
  const std::unordered_map<std::string, const GlmExpectedTensor*>& by_name;
  GlmLayerBump& bump;
  GlmLayerResident& out;  // pointer stores are harmless in counting mode
  const std::unordered_map<std::string, const TensorInfo*>& tensors;
  std::vector<DequantJob>& jobs;
  std::vector<PackJob>& packs;
  bool copy;
  int rank = 0;
  int world = 1;
  uint64_t source_bytes = 0;   // checkpoint bytes touched (copy mode)
  uint64_t verbatim_bytes = 0;  // the rank-invariant re-read subset

  // Local geometry at tp_size=world (world=1: the full geometry).
  KdaGeometry kgeo{};
  DsaGeometry dgeo{};
  int64_t dense_inter = 0;    // intermediate_size/world
  int64_t shared_inter = 0;   // moe_intermediate_size/world
  int64_t local_experts = 0;  // n_routed_experts/world
  int64_t dsa_kv_rows_head = 0;  // qk_nope + v per head

  bool sharded() const { return world > 1; }

  // Derives the local geometry from (cfg, rank, world). Called at the top
  // of build_layer — the single entry every build goes through. The
  // geometry validators run inside from_config, so a bad world fails
  // before any allocation.
  void init_geometry() {
    KdaConfig kc = cfg.kda_config();
    kc.tp_size = world;
    kgeo = KdaGeometry::from_config(kc);
    DsaConfig dc = cfg.dsa_config();
    dc.tp_size = world;
    dgeo = DsaGeometry::from_config(dc);
    dense_inter = cfg.intermediate_size / world;
    shared_inter = cfg.moe_config().inter / world;
    local_experts = cfg.moe_config().n_experts / world;
    dsa_kv_rows_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
  }

  const GlmExpectedTensor& expected(const std::string& name) const {
    auto it = by_name.find(name);
    if (it == by_name.end())
      throw std::runtime_error("glm loader: '" + name +
                               "' missing from expected table (builder bug)");
    return *it->second;
  }

  const TensorInfo& source(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("glm loader: tensor not in checkpoint: " + name);
    return *it->second;
  }

  // Byte accounting chokepoint: every checkpoint byte this build touches
  // flows through here, with its tensor. Replicated and DSA-bridge reads
  // form the rank-invariant re-read set (re-read by every rank at world>1
  // — the reconcile's constant term). Sharded-class tensors arrive as
  // slices or as whole-expert verbatim loads (the routed-expert partition
  // owns entire tensors); the drift guard that matters is in load_raw.
  void note_read(const GlmExpectedTensor& e, size_t bytes) {
    if (!copy) return;
    source_bytes += bytes;
    if (is_replicated(e) || is_dsa_bridge(e)) verbatim_bytes += bytes;
  }

  // ---- verbatim loads (replicated at every world; world=1: everything) --
  const void* load_raw(const std::string& name) {
    const GlmExpectedTensor& e = expected(name);
    if (sharded() && !is_replicated(e) && !is_dsa_bridge(e))
      throw std::runtime_error(
          "glm loader: TP read-class drift — '" + name +
          "' is sharded but was loaded verbatim (the slicing build and "
          "the replicated classifier disagree; fix one of them)");
    void* dst = bump.alloc(e.nbytes());
    if (copy) {
      std::memcpy(dst, source(name).data, e.nbytes());
      note_read(e, e.nbytes());
    }
    return dst;
  }
  uint16_t* load_bf16(const std::string& name) {
    return static_cast<uint16_t*>(const_cast<void*>(load_raw(name)));
  }
  float* load_f32(const std::string& name) {
    return static_cast<float*>(const_cast<void*>(load_raw(name)));
  }

  // ---- contiguous row/element slices (bf16/f32 — no scale grid) --------
  uint16_t* load_bf16_rows(const std::string& name, int64_t row_start,
                          int64_t rows) {
    const GlmExpectedTensor& e = expected(name);
    check_range(name, row_start, rows, e.shape[0]);
    const int64_t width = e.numel() / e.shape[0];  // row width, elements
    uint16_t* dst = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(width) *
                   2));
    if (copy) {
      const uint8_t* src = static_cast<const uint8_t*>(source(name).data);
      std::memcpy(dst, src + static_cast<size_t>(row_start) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
      note_read(e, static_cast<size_t>(rows) * width * 2);
    }
    return dst;
  }

  float* load_f32_range(const std::string& name, int64_t start, int64_t count) {
    const GlmExpectedTensor& e = expected(name);
    check_range(name, start, count, e.shape[0]);
    float* dst = static_cast<float*>(bump.alloc(static_cast<size_t>(count) * 4));
    if (copy) {
      const uint8_t* src = static_cast<const uint8_t*>(source(name).data);
      std::memcpy(dst, src + static_cast<size_t>(start) * 4,
                  static_cast<size_t>(count) * 4);
      note_read(e, static_cast<size_t>(count) * 4);
    }
    return dst;
  }

  // ---- quantized slices (E4M3 + 128x128 block scales) -----------------
  // Row slice: payload rows are contiguous and the scale grid re-anchors
  // exactly at the 128-ALIGNED start (the §5.2 contract). Degenerate
  // (0, full): the M4 load_quant, grant-for-grant.
  GlmQuantMatrix load_quant_rows(const std::string& name, int64_t row_start,
                                 int64_t rows) {
    const GlmExpectedTensor& e = expected(name);
    if (row_start % 128 != 0)
      throw std::invalid_argument(
          "glm loader: quantized row slice of '" + name +
          "' must start 128-aligned (scale-grid slice contract)");
    check_range(name, row_start, rows, e.shape[0]);
    const GlmExpectedTensor& es = expected(name + "_scale_inv");
    const int64_t cols = e.shape[1];
    const int64_t sb = (cols + 127) / 128;       // scale blocks per row
    const int64_t scale_rows = (rows + 127) / 128;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(scale_rows) * sb * 4));
    if (copy) {
      std::memcpy(const_cast<uint8_t*>(q.payload),
                  static_cast<const uint8_t*>(source(name).data) +
                      static_cast<size_t>(row_start) * cols,
                  static_cast<size_t>(rows) * cols);
      std::memcpy(const_cast<float*>(q.scales),
                  static_cast<const float*>(source(es.name).data) +
                      (row_start / 128) * sb,
                  static_cast<size_t>(scale_rows) * sb * 4);
      note_read(e,
               static_cast<size_t>(rows) * cols +
                   static_cast<size_t>(scale_rows) * sb * 4);
    }
    return q;
  }

  // Column slice -> PACKED payload + packed scale columns. Column starts
  // carry the same 128-alignment contract (the local scale grid re-anchors
  // at the pack origin). Degenerate (0, full_cols): the M4 load_quant,
  // grant-for-grant (the strided row loop collapses to one memcpy).
  GlmQuantMatrix load_quant_cols(const std::string& name, int64_t col_start,
                                 int64_t cols) {
    const GlmExpectedTensor& e = expected(name);
    if (col_start % 128 != 0)
      throw std::invalid_argument(
          "glm loader: quantized column slice of '" + name +
          "' must start 128-aligned (scale-grid slice contract)");
    check_range(name, col_start, cols, e.shape[1]);
    const GlmExpectedTensor& es = expected(name + "_scale_inv");
    const int64_t rows = e.shape[0];
    const int64_t full_cols = e.shape[1];
    const int64_t sb_full = (full_cols + 127) / 128;
    const int64_t sb_s = (cols + 127) / 128;
    const int64_t scale_rows = (rows + 127) / 128;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(scale_rows) * sb_s * 4));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(source(name).data);
      const float* ss = static_cast<const float*>(source(es.name).data);
      if (cols == full_cols && col_start == 0) {
        std::memcpy(const_cast<uint8_t*>(q.payload), sp,
                    static_cast<size_t>(rows) * cols);
        std::memcpy(const_cast<float*>(q.scales), ss,
                    static_cast<size_t>(scale_rows) * sb_s * 4);
      } else {
        for (int64_t r = 0; r < rows; ++r)
          std::memcpy(const_cast<uint8_t*>(q.payload) + r * cols,
                      sp + r * full_cols + col_start, cols);
        for (int64_t r = 0; r < scale_rows; ++r)
          std::memcpy(const_cast<float*>(q.scales) + r * sb_s,
                      ss + r * sb_full + col_start / 128, sb_s * 4);
      }
      note_read(e,
               static_cast<size_t>(rows) * cols +
                   static_cast<size_t>(scale_rows) * sb_s * 4);
    }
    return q;
  }

  void check_range(const std::string& name, int64_t start, int64_t count,
                   int64_t dim) const {
    if (start < 0 || count < 0 || count > dim || start > dim - count)
      throw std::runtime_error("glm loader: slice of '" + name +
                               "' out of bounds");
  }

  // Compressed residency: payload + scales, byte-for-byte.
  GlmQuantMatrix load_quant(const std::string& payload_name) {
    const GlmExpectedTensor& e = expected(payload_name);
    GlmQuantMatrix q;
    q.rows = e.shape[0];
    q.cols = e.shape[1];
    q.payload = static_cast<const uint8_t*>(load_raw(payload_name));
    q.scales = static_cast<const float*>(load_raw(payload_name + "_scale_inv"));
    return q;
  }

  // Transient bridge for the M3 bf16 seam: compressed copies land in the
  // bump, a dequant job writes the BF16 form alongside. The bridge reads
  // the FULL quantized source at every world (see is_dsa_bridge).
  uint16_t* load_dequant_bf16(const std::string& payload_name) {
    const GlmQuantMatrix q = load_quant(payload_name);
    uint16_t* out = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols) *
                   2));
    jobs.push_back(DequantJob{q.payload, q.scales, out, q.rows, q.cols});
    return out;
  }

  void build_mhc(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.mhc.attn_base = load_f32(p + "hc_attn_base");
    out.mhc.attn_fn = load_bf16(p + "hc_attn_fn");
    out.mhc.attn_scale = load_f32(p + "hc_attn_scale");
    out.mhc.ffn_base = load_f32(p + "hc_ffn_base");
    out.mhc.ffn_fn = load_bf16(p + "hc_ffn_fn");
    out.mhc.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_norms(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.ln1 = load_bf16(p + "input_layernorm.weight");
    out.ln2 = load_bf16(p + "post_attention_layernorm.weight");
  }

  void build_kda(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    // TP geometry: world=1 is the degenerate rank 0 — local == full, the
    // M4 merged layout, byte-for-byte. The local arithmetic mirrors
    // GlmTpViews::bind exactly.
    const int64_t head_dim = cfg.kda_head_dim;
    const int64_t hidden = cfg.hidden_size;
    const int64_t lp_s = kgeo.local_proj;        // this rank's q/k/v rows
    const int64_t lp_f = lp_s * world;           // full-heads rows
    const int64_t h_s = kgeo.local_heads;

    // Merged in_proj rows [f_a | g_a | q | k | v | b] at LOCAL geometry —
    // the M2 kernel contract; the checkpoint stores the six projections
    // separately. f_a/g_a are replicated (full rows); q/k/v and b carry
    // this rank's head rows.
    const int64_t in_rows = 2 * head_dim + 3 * lp_s + h_s;
    uint16_t* in_proj = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(in_rows) * static_cast<size_t>(hidden) *
                   2));
    struct Piece {
      const char* name;
      int64_t full_rows;  // source rows (geometry check)
      bool shard;         // head-sharded vs replicated
      int64_t src_off;    // this rank's first source row
      int64_t rows;       // this rank's rows
    };
    const Piece pieces[] = {
        {"f_a_proj.weight", head_dim, false, 0, head_dim},
        {"g_a_proj.weight", head_dim, false, 0, head_dim},
        {"q_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"k_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"v_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"b_proj.weight", h_s * world, true, rank * h_s, h_s},
    };
    int64_t dst_row = 0;
    for (const Piece& piece : pieces) {
      const GlmExpectedTensor& e = expected(p + piece.name);
      if (e.shape[0] != piece.full_rows || e.shape[1] != hidden)
        throw std::runtime_error("glm loader: KDA piece geometry mismatch");
      if (sharded() && piece.shard == is_replicated(e))
        throw std::runtime_error("glm loader: TP read-class drift on '" +
                                 p + piece.name + "'");
      if (copy) {
        std::memcpy(in_proj + static_cast<size_t>(dst_row) * hidden,
                    static_cast<const uint8_t*>(source(p + piece.name).data) +
                        static_cast<size_t>(piece.src_off) * hidden * 2,
                    static_cast<size_t>(piece.rows) * hidden * 2);
        note_read(e,
                  static_cast<size_t>(piece.rows) * hidden * 2);
      }
      dst_row += piece.rows;
    }
    if (dst_row != in_rows)
      throw std::runtime_error("glm loader: KDA in_proj row count mismatch");

    // Merged causal conv channels [q | k | v] at LOCAL geometry.
    const int64_t conv_w = cfg.kda_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(3 * lp_s) *
                   static_cast<size_t>(conv_w) * 2));
    static const char* convs[3] = {"q_conv1d.weight", "k_conv1d.weight",
                                   "v_conv1d.weight"};
    for (int i = 0; i < 3; ++i) {
      const GlmExpectedTensor& e = expected(p + convs[i]);
      if (e.shape[0] != lp_f || e.shape[2] != conv_w)
        throw std::runtime_error("glm loader: conv piece geometry mismatch");
      if (copy) {
        std::memcpy(conv + static_cast<size_t>(i) * lp_s * conv_w,
                    static_cast<const uint8_t*>(source(p + convs[i]).data) +
                        static_cast<size_t>(rank) * lp_s * conv_w * 2,
                    static_cast<size_t>(lp_s) * conv_w * 2);
        note_read(e, static_cast<size_t>(lp_s) * conv_w * 2);
      }
    }

    out.kda.in_proj = in_proj;
    out.kda.conv = conv;
    out.kda.f_b = load_bf16_rows(p + "f_b_proj.weight", rank * lp_s, lp_s);
    out.kda.g_b = load_bf16_rows(p + "g_b_proj.weight", rank * lp_s, lp_s);
    out.kda.a_log = load_f32_range(p + "A_log", rank * h_s, h_s);
    out.kda.dt_bias = load_f32_range(p + "dt_bias", rank * lp_s, lp_s);
    out.kda.o_norm = load_bf16(p + "o_norm.weight");

    // o_proj [hidden, lp_f] bf16: column slice -> packed [hidden, lp_s].
    // BF16 carries no scale grid, so any column start is representable;
    // the pack runs in the post-dequant phase for uniformity.
    {
      const std::string name = p + "o_proj.weight";
      const GlmExpectedTensor& e = expected(name);
      if (e.shape[0] != hidden || e.shape[1] != lp_f)
        throw std::runtime_error("glm loader: KDA o_proj geometry mismatch");
      uint16_t* packed = static_cast<uint16_t*>(
          bump.alloc(static_cast<size_t>(hidden) * lp_s * 2));
      if (copy) {
        packs.push_back(PackJob{
            static_cast<const uint16_t*>(source(name).data) + rank * lp_s,
            packed, static_cast<size_t>(lp_f) * 2,
            static_cast<size_t>(lp_s) * 2, static_cast<size_t>(lp_s) * 2,
            static_cast<size_t>(hidden)});
        note_read(e, static_cast<size_t>(hidden) * lp_s * 2);
      }
      out.kda.o_proj = packed;
    }
  }

  void build_dsa(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    const int64_t hidden = cfg.hidden_size;
    const int64_t q_lora = cfg.q_lora_rank;
    const int64_t lh = dgeo.local_heads;
    const int64_t local_v_rows = dgeo.local_v_rows;

    // Fused qkv_a [q_lora + kv_lora, hidden]: replicated latents — q_a rows
    // then kv_a rows, both dequantized (their scale grids differ — two
    // jobs, one buffer). Full at every world.
    const GlmExpectedTensor& qa = expected(p + "q_a_proj.weight");
    const GlmExpectedTensor& kva = expected(p + "kv_a_proj_with_mqa.weight");
    uint16_t* qkv_a = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(qa.shape[0] + kva.shape[0]) *
                   static_cast<size_t>(hidden) * 2));
    const GlmQuantMatrix qa_q = load_quant(p + "q_a_proj.weight");
    jobs.push_back(DequantJob{qa_q.payload, qa_q.scales, qkv_a, qa_q.rows,
                              qa_q.cols});
    const GlmQuantMatrix kva_q =
        load_quant(p + "kv_a_proj_with_mqa.weight");
    jobs.push_back(DequantJob{
        kva_q.payload, kva_q.scales,
        qkv_a + static_cast<size_t>(q_lora) * static_cast<size_t>(hidden),
        kva_q.rows, kva_q.cols});

    out.dsa.qkv_a = qkv_a;
    out.dsa.q_aln = load_bf16(p + "q_a_layernorm.weight");
    out.dsa.kv_aln = load_bf16(p + "kv_a_layernorm.weight");
    // q_b: the bridge dequantizes the FULL source (a head-row slice of the
    // quantized matrix starts mid-block at some worlds — unrepresentable);
    // the resident keeps this rank's rows of the bf16 bridge, exactly the
    // view bind slices from the same buffer.
    uint16_t* q_b = load_dequant_bf16(p + "q_b_proj.weight");
    out.dsa.q_b = sharded()
                      ? q_b + static_cast<size_t>(rank) *
                                  static_cast<size_t>(dgeo.local_q_rows) *
                                  q_lora
                      : q_b;
    // kv_b: BF16 in the checkpoint — this rank's head block is contiguous
    // source rows; a real slice read.
    out.dsa.kv_b = load_bf16_rows(
        p + "kv_b_proj.weight",
        static_cast<int64_t>(rank) * lh * dsa_kv_rows_head,
        lh * dsa_kv_rows_head);
    // o_proj: bridge dequant of the full source, then the packed column
    // slice [hidden, local_v_rows] in the post-dequant phase. world=1
    // keeps the bridge buffer itself (the M4 resident).
    uint16_t* o_bridge = load_dequant_bf16(p + "o_proj.weight");
    if (sharded()) {
      uint16_t* packed = static_cast<uint16_t*>(
          bump.alloc(static_cast<size_t>(hidden) * local_v_rows * 2));
      if (copy)
        packs.push_back(PackJob{o_bridge + rank * local_v_rows, packed,
                                static_cast<size_t>(local_v_rows) * world * 2,
                                static_cast<size_t>(local_v_rows) * 2,
                                static_cast<size_t>(local_v_rows) * 2,
                                static_cast<size_t>(hidden)});
      out.dsa.o_proj = packed;
    } else {
      out.dsa.o_proj = o_bridge;
    }
    const std::string ip = p + "indexer.";
    out.dsa.wq_b = load_bf16(ip + "wq_b.weight");
    out.dsa.wk = load_bf16(ip + "wk.weight");
    out.dsa.wp = load_bf16(ip + "weights_proj.weight");
    out.dsa.gate = load_bf16(ip + "index_kpool_compress_gate");
    out.dsa.k_norm_w = load_bf16(ip + "k_norm.weight");
    out.dsa.k_norm_b = load_bf16(ip + "k_norm.bias");
    // APE: checkpoint BF16 [kpool, index_head_dim]; the M3 kernel wants F32.
    // Read via memcpy: safetensors does not align individual tensors in
    // the file, so a typed uint16_t load from the mmap can be unaligned
    // (the fixture's byte layout lands one at an odd offset).
    {
      const std::string name = ip + "index_kpool_compress_ape";
      const GlmExpectedTensor& e = expected(name);
      const size_t n = e.numel();
      float* ape = static_cast<float*>(bump.alloc(n * 4));
      if (copy) {
        const uint8_t* src =
            static_cast<const uint8_t*>(source(name).data);
        for (size_t i = 0; i < n; ++i) {
          uint16_t bits;
          std::memcpy(&bits, src + i * 2, 2);
          ape[i] = bf16_bits_to_float(bits);
        }
        note_read(e, n * 2);
      }
      out.dsa.ape = ape;
    }
  }

  void build_dense_mlp(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    const int64_t I = dense_inter;
    out.dense[0] = load_quant_rows(p + "mlp.gate_proj.weight", rank * I, I);
    out.dense[1] = load_quant_rows(p + "mlp.up_proj.weight", rank * I, I);
    out.dense[2] = load_quant_cols(p + "mlp.down_proj.weight", rank * I, I);
  }

  void build_moe(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.moe.router_gate = load_bf16(p + "mlp.gate.weight");
    out.moe.router_bias = load_f32(p + "mlp.gate.e_score_correction_bias");
    const std::string sp = p + "mlp.shared_experts.";
    const int64_t M = shared_inter;
    out.moe.shared[0] = load_quant_rows(sp + "gate_proj.weight", rank * M, M);
    out.moe.shared[1] = load_quant_rows(sp + "up_proj.weight", rank * M, M);
    out.moe.shared[2] = load_quant_cols(sp + "down_proj.weight", rank * M, M);
    // Whole-expert partition: only this rank's contiguous expert ids load.
    const int64_t first = rank * local_experts;
    out.moe.experts.resize(static_cast<size_t>(local_experts) * 3);
    for (int64_t e = 0; e < local_experts; ++e) {
      const std::string ep = p + "mlp.experts." +
                             std::to_string(first + e) + ".";
      const int64_t full = cfg.moe_intermediate_size;
      out.moe.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", 0, full);
      out.moe.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", 0, full);
      out.moe.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", 0, full);
    }
    out.moe.expert_begin = sharded() ? static_cast<int>(first) : 0;
    // world=1 keeps the M4 "every expert" sentinel; the sharded forward
    // consumes the stamped range, the world=1 forward the sentinel.
    out.moe.expert_count = sharded() ? static_cast<int>(local_experts) : -1;
  }

  void build_mtp_head(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.enorm = load_bf16(p + "enorm.weight");
    out.hnorm = load_bf16(p + "hnorm.weight");
    out.eh_proj = load_bf16(p + "eh_proj.weight");
    out.shared_head_norm = load_bf16(p + "shared_head.norm.weight");
  }

  void build_layer(int layer) {
    init_geometry();
    const bool is_mtp = layer == cfg.mtp_layer();
    const int max_layer =
        cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      throw std::invalid_argument(
          "glm loader: layer index out of range: " + std::to_string(layer));

    out.layer = layer;
    out.kind = is_mtp || cfg.layers[layer] == GlmLayerKind::Dsa
                   ? GlmLayerKind::Dsa
                   : GlmLayerKind::Kda;
    if (!is_mtp) build_mhc(layer);
    build_norms(layer);
    if (out.kind == GlmLayerKind::Kda)
      build_kda(layer);
    else
      build_dsa(layer);
    const bool dense =
        !is_mtp && cfg.mlps[layer] == GlmMlpKind::Dense;
    if (dense)
      build_dense_mlp(layer);
    else
      build_moe(layer);
    if (is_mtp) build_mtp_head(layer);
  }
};

// Load-boundary synchronization: waits for every outstanding READER of the
// bump. With a registered reader stream that is exactly two streams (the
// model's compute stream + the loader's dequant stream, idle since the
// previous load's exit sync). Without one, the conservative M4 default:
// the whole device — standalone callers load with unknown readers, and no
// in-process peer exists to spin against.
void sync_load_boundary(cudaStream_t reader, cudaStream_t dequant) {
  if (reader) {
    DGPP_CUDA_OK(cudaStreamSynchronize(reader));
    DGPP_CUDA_OK(cudaStreamSynchronize(dequant));
  } else {
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }
}

// Runs one layer's build in counting mode; returns the exact byte total at
// the given rank's geometry.
size_t count_layer_bytes(const GlmTextConfig& cfg, int layer, int rank,
                         int world) {
  GlmLayerBump bump;
  bump.counting = true;
  bump.capacity = SIZE_MAX;
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  GlmLayerResident scratch;
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  std::unordered_map<std::string, const TensorInfo*> no_tensors;
  BuildCtx ctx{cfg,    table, by_name, bump,  scratch,
                no_tensors, jobs,  packs,   false, rank, world};
  ctx.build_layer(layer);
  return bump.cursor;
}

}  // namespace

size_t GlmLayerStream::layer_bytes(const GlmTextConfig& cfg, int layer,
                                   int rank, int world) {
  return count_layer_bytes(cfg, layer, rank, world);
}

size_t GlmLayerStream::globals_bytes(const GlmTextConfig& cfg) {
  const size_t vocab_bytes = align_up_256(static_cast<size_t>(cfg.vocab_size) *
                                          cfg.hidden_size * 2);
  const size_t norm_bytes = align_up_256(static_cast<size_t>(cfg.hidden_size) * 2);
  return 2 * vocab_bytes + norm_bytes;
}

size_t GlmLayerStream::resident_bytes(const GlmTextConfig& cfg, int rank,
                                      int world) {
  const int max_layer =
      cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  size_t total = globals_bytes(cfg);
  for (int l = 0; l < max_layer; ++l)
    total += layer_bytes(cfg, l, rank, world);
  return total;
}

GlmLayerStream::GlmLayerStream(const GlmTextConfig& cfg,
                               const std::string& checkpoint_dir, int rank,
                               int world, GlmResidency residency)
    : cfg_(cfg),
      rank_(rank),
      world_(world),
      residency_(residency),
      layer_bump_(std::make_unique<GlmLayerBump>()),
      globals_bump_(std::make_unique<GlmLayerBump>()) {
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument(
        "glm loader: rank/world out of range (rank " + std::to_string(rank) +
        ", world " + std::to_string(world) + ")");
  // TP geometry — BEFORE a single shard is opened, so a bad config fails
  // in milliseconds with its dim named, not after 328 GB of mmap.
  if (world > 1) glm_tp_validate_geometry(cfg, rank, world);

  // Open shards (sorted for determinism), index headers, validate the full
  // text binding before a single byte of payload moves.
  namespace fs = std::filesystem;
  std::vector<fs::path> shard_paths;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors")
      shard_paths.push_back(entry.path());
  if (shard_paths.empty())
    throw std::runtime_error("glm loader: no .safetensors shards in " +
                             checkpoint_dir);
  std::sort(shard_paths.begin(), shard_paths.end());

  std::unordered_map<std::string, GlmTensorDesc> present;
  for (const auto& path : shard_paths) {
    auto f = SafetensorsFile::open(path.string());
    f->for_each([&](const TensorInfo& t) {
      auto [it, inserted] =
          tensors_.emplace(t.name, &t);
      if (!inserted)
        throw std::runtime_error("glm loader: duplicate tensor '" + t.name +
                                 "' in " + path.string());
      present.emplace(t.name, GlmTensorDesc{t.dtype, t.shape});
    });
    shards_.push_back(std::move(f));
  }

  const GlmBindReport rep = glm_validate_text_binding(cfg_, present);
  if (!rep.ok()) {
    std::string msg = "glm loader: checkpoint binding failed: ";
    for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
      if (i) msg += "; ";
      msg += rep.errors[i];
    }
    throw std::runtime_error(msg);
  }

  // Size the bump at the largest layer AT THIS RANK'S GEOMETRY (headers
  // say which; the shared build code says how much) and take a dedicated
  // stream for the dequant launches. RESIDENT mode never allocates the
  // shared bump — each layer materializes into its own exact-formula bump
  // on first load — so the stream opens without a single layer allocation
  // and the per-layer stores are sized instead.
  size_t capacity = 0;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  for (int i = 0; i < max_layer; ++i)
    capacity = std::max(capacity, count_layer_bytes(cfg_, i, rank_, world_));
  capacity_ = capacity;
  if (residency_ == GlmResidency::Resident) {
    resident_bumps_.resize(static_cast<size_t>(max_layer));  // null bumps
    resident_layers_.assign(static_cast<size_t>(max_layer),
                            GlmLayerResident{});
  } else {
    layer_bump_->init(capacity);
  }
  globals_bump_->init(globals_bytes(cfg_));
  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
}

GlmLayerStream::~GlmLayerStream() {
  if (stream_) cudaStreamDestroy(stream_);
}

size_t GlmLayerStream::layer_capacity() const {
  return capacity_;
}

// The one layer build both residency modes share: the grant sequence,
// the byte accounting, the formula check, and the dequant/pack phases.
// Resident bytes are streaming bytes by construction — the parity
// driver pins this bitwise at every world.
void GlmLayerStream::build_layer_into(int layer, GlmLayerBump& bump,
                                      GlmLayerResident& out) {
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg_, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);

  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  BuildCtx ctx{cfg_,    table, by_name, bump,          out,
               tensors_, jobs,  packs,   true,         rank_,   world_};
  ctx.build_layer(layer);

  // Phase two: all CPU writes are done, launch the dequants and wait.
  for (const DequantJob& j : jobs)
    launch_fp8_dequant_blocks(j.payload, j.scales, j.out, j.rows, j.cols,
                              stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

  // Phase three: the packs — strided copies whose sources include bridge
  // buffers the dequants just wrote (the DSA o_proj pack; the KDA o_proj
  // pack's source is host mmap, but one phase for all packs keeps the
  // rule single).
  for (const PackJob& j : packs) run_pack(j);

  // The formula and the allocator share the build code; anything but
  // equality is a bug that must never pass silently. In resident mode the
  // bump was sized BY this formula, so allocation itself is a second,
  // independent enforcement of the same equality.
  const size_t used = bump.cursor;
  const size_t expected_bytes = layer_bytes(cfg_, layer, rank_, world_);
  if (used != expected_bytes)
    throw std::runtime_error(
        "glm loader: byte-formula drift on layer " + std::to_string(layer) +
        ": used " + std::to_string(used) + " != formula " +
        std::to_string(expected_bytes));
  out.bytes = used;
  source_bytes_ += ctx.source_bytes;
  verbatim_bytes_ += ctx.verbatim_bytes;
}

const GlmLayerResident& GlmLayerStream::load_layer(int layer) {
  // RESIDENT mode: a materialized layer is a pure cache hit — the stored
  // view, no storage reads (source_bytes_read cannot grow), and no sync
  // (the streaming contract's sync covers loader WRITES; a cache hit
  // writes nothing). First load materializes into the layer's own
  // exact-formula bump, which stays alive for the stream's lifetime.
  if (residency_ == GlmResidency::Resident) {
    if (layer < 0 || layer >= static_cast<int>(resident_layers_.size()))
      throw std::out_of_range(
          "glm loader: layer index out of range: " + std::to_string(layer));
    GlmLayerResident& slot =
        resident_layers_[static_cast<size_t>(layer)];
    if (slot.layer == layer) return slot;

    auto bump = std::make_unique<GlmLayerBump>();
    bump->init(layer_bytes(cfg_, layer, rank_, world_));
    sync_load_boundary(reader_, stream_);
    build_layer_into(layer, *bump, slot);
    resident_bumps_[static_cast<size_t>(layer)] = std::move(bump);
    return slot;
  }

  if (resident_.layer == layer) return resident_;

  // Phase one below writes weight bytes into the bump from the CPU — the
  // SAME managed region the previously loaded layer's kernels may still be
  // reading (this loader was first exercised mid-forward by the M4
  // diagnostic model; before that, callers always loaded with the device
  // idle). The boundary sync waits for exactly those readers — see
  // sync_load_boundary for why it must NOT be device-wide in a
  // one-process multi-rank world.
  sync_load_boundary(reader_, stream_);

  layer_bump_->reset();
  resident_ = GlmLayerResident{};
  build_layer_into(layer, *layer_bump_, resident_);
  return resident_;
}

GlmReplicatedDigest GlmLayerStream::hash_replicated() const {
  const auto lookup = [this](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: tensor not in checkpoint: " +
                               name);
    return *it->second;
  };
  GlmReplicatedDigest d;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  d.layer.assign(static_cast<size_t>(max_layer), 0);
  for (int l = 0; l < max_layer; ++l) {
    uint64_t sum = 0;
    for (const GlmExpectedTensor& e :
         glm_expected_layer_tensors(cfg_, l)) {
      if (!is_replicated(e)) continue;
      const TensorInfo& t = lookup(e.name);
      const size_t bytes = e.nbytes();
      if (t.numel() * dtype_size(t.dtype) != bytes)
        throw std::runtime_error(
            "glm loader: replicated tensor shape drift on '" + e.name +
            "' (expected table and checkpoint header disagree)");
      sum += fnv1a(t.data, bytes, fnv1a(e.name.data(), e.name.size(),
                                        1469598103934665603ull));
      d.bytes += bytes;
      ++d.tensors;
    }
    d.layer[static_cast<size_t>(l)] = sum;
  }
  static const char* kGlobals[] = {
      "model.language_model.embed_tokens.weight",
      "lm_head.weight",
      "model.language_model.norm.weight",
  };
  uint64_t g = 0;
  for (const char* name : kGlobals) {
    const TensorInfo& t = lookup(name);
    const size_t bytes = t.numel() * dtype_size(t.dtype);
    g += fnv1a(t.data, bytes,
               fnv1a(name, std::strlen(name), 1469598103934665603ull));
    d.bytes += bytes;
    ++d.tensors;
  }
  d.globals = g;
  return d;
}

const GlmGlobalsResident& GlmLayerStream::load_globals() {
  if (globals_.embed) return globals_;
  // Same entry-sync discipline as load_layer (phase-one CPU writes; the
  // boundary sync covers exactly the globals bump's readers).
  sync_load_boundary(reader_, stream_);
  globals_bump_->reset();
  globals_ = GlmGlobalsResident{};

  auto copy_global = [&](const std::string& name) -> uint16_t* {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: global tensor missing: " + name);
    const TensorInfo& t = *it->second;
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(t.nbytes()));
    std::memcpy(dst, t.data, t.nbytes());
    source_bytes_ += t.nbytes();
    verbatim_bytes_ += t.nbytes();  // globals are replicated in v1
    return dst;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  globals_.lm_head = copy_global("lm_head.weight");
  globals_.final_norm = copy_global("model.language_model.norm.weight");
  globals_.bytes = globals_bump_->cursor;

  const size_t expected_bytes = globals_bytes(cfg_);
  if (globals_.bytes != expected_bytes)
    throw std::runtime_error(
        "glm loader: globals byte-formula drift: used " +
        std::to_string(globals_.bytes) + " != formula " +
        std::to_string(expected_bytes));
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  return globals_;
}

void GlmLayerStream::release_layer() {
  // RESIDENT mode: the contract is that materialized layers stay resident
  // for the stream's lifetime — a no-op, so existing call sites (the
  // model's preconstruct pass) stay correct unchanged. The views remain
  // valid; a streaming release after resident loads would be a silent
  // use-after-free factory, which is exactly what this refuses to be.
  if (residency_ == GlmResidency::Resident) return;
  resident_ = GlmLayerResident{};
  layer_bump_->reset();
}

void GlmLayerStream::release_globals() {
  globals_ = GlmGlobalsResident{};
  globals_bump_->reset();
}

}  // namespace dgpp
