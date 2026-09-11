#include "models/glm/tp.hpp"

#include <algorithm>
#include <cstring>

#include "common/cuda_check.hpp"
#include "kernels/fp8_dequant.hpp"

namespace dgpp {

namespace {

size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

// One strided 2D block copy on the model stream (bytes, not elements).
void copy2d(const void* src, size_t src_pitch, void* dst, size_t dst_pitch,
            size_t width, size_t height, cudaStream_t s) {
  DGPP_CUDA_OK(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width,
                                 height, cudaMemcpyDeviceToDevice, s));
}

}  // namespace

GlmTpViews::GlmTpViews(const GlmTextConfig& cfg, int rank, int world,
                       cudaStream_t stream)
    : cfg_(cfg),
      kda_cfg_(cfg.kda_config()),
      dsa_cfg_(cfg.dsa_config()),
      moe_cfg_(cfg.moe_config()),
      rank_(rank),
      world_(world),
      stream_(stream) {
  if (!stream)
    throw std::invalid_argument("GlmTpViews: null stream");
  // Full TP geometry acceptance (shared with the sharded loader — the
  // two consumers must reject the same configs before any bytes move):
  // world/rank range, head divisibility, expert and inter divisibility,
  // and the 128-alignment of the inter quotients (the scale-grid slice
  // contract).
  glm_tp_validate_geometry(cfg, rank, world);
  kda_cfg_.tp_size = world;
  dsa_cfg_.tp_size = world;
  kda_geo_ = KdaGeometry::from_config(kda_cfg_);
  dsa_geo_ = DsaGeometry::from_config(dsa_cfg_);
  dense_inter_ = cfg.intermediate_size / world;
  moe_inter_ = moe_cfg_.inter / world;

  const SlabLayout lay = layout(cfg_, world_);
  slab_bytes_ = lay.total;
  DGPP_CUDA_OK(cudaMalloc(&slab_, slab_bytes_));
}

GlmTpViews::~GlmTpViews() {
  if (slab_) cudaFree(slab_);
  if (expert_pack_) cudaFree(expert_pack_);
  if (dsa_bridge_) cudaFree(dsa_bridge_);
}

GlmTpViews::SlabLayout GlmTpViews::layout(const GlmTextConfig& cfg,
                                          int world) {
  KdaConfig kc = cfg.kda_config();
  kc.tp_size = world;
  const KdaGeometry kg = KdaGeometry::from_config(kc);
  DsaConfig dc = cfg.dsa_config();
  dc.tp_size = world;
  const DsaGeometry dg = DsaGeometry::from_config(dc);
  const GlmMoeConfig mc = cfg.moe_config();
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size / world;      // dense inter slice
  const int64_t M = mc.inter / world;                   // shared inter slice

  SlabLayout l;
  auto add = [&](size_t bytes) {
    const size_t off = l.total;
    l.total += align256(bytes);
    return off;
  };
  l.off_kda_in_proj = add(size_t(kg.in_proj_cols) * H * 2);
  l.off_kda_conv =
      add(size_t(kg.conv_channels) * kc.conv_width * 2);
  l.off_kda_o_proj = add(size_t(H) * kg.local_proj * 2);
  l.off_dsa_o_proj = add(size_t(H) * dg.local_v_rows * 2);
  l.off_dense_down = add(size_t(H) * I);
  l.off_dense_scales =
      add(size_t((H + 127) / 128) * ((I + 127) / 128) * 4);
  l.off_shared_down = add(size_t(H) * M);
  l.off_shared_scales =
      add(size_t((H + 127) / 128) * ((M + 127) / 128) * 4);
  return l;
}

size_t GlmTpViews::slice_bytes(const GlmTextConfig& cfg, int world) {
  return layout(cfg, world).total;
}

// Per expert: the packed down payload [H, M] then its scale grid, each
// 256-aligned (the loader's grant alignment — the GEMV core wants 16).
size_t GlmTpViews::expert_pack_bytes(const GlmTextConfig& cfg, int world) {
  const GlmMoeConfig mc = cfg.moe_config();
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t M = static_cast<size_t>(mc.inter / world);
  const size_t per_expert =
      align256(H * M) + align256(((H + 127) / 128) * ((M + 127) / 128) * 4);
  return per_expert * static_cast<size_t>(mc.n_experts);
}

void GlmTpViews::ensure_expert_pack() {
  if (expert_pack_) return;
  const size_t bytes = expert_pack_bytes(cfg_, world_);
  DGPP_CUDA_OK(cudaMalloc(&expert_pack_, bytes));
  experts_.assign(static_cast<size_t>(moe_cfg_.n_experts) * 3,
                  GlmQuantMatrix{});
  experts_fp4_.assign(static_cast<size_t>(moe_cfg_.n_experts) * 3,
                      GlmFp4Matrix{});
}

// The NVFP4 column pack: nibble pairs and per-row block scales, both plain
// strided copies (a 16-block boundary is 8 packed bytes and 1 scale byte).
GlmFp4Matrix GlmTpViews::pack_fp4_cols(const GlmFp4Matrix& m, int64_t col_start,
                                       int64_t cols, uint8_t* payload,
                                       uint8_t* scales) {
  if (col_start < 0 || cols <= 0 || cols > m.cols || col_start > m.cols - cols)
    throw std::invalid_argument("GlmTpViews: NVFP4 column slice out of bounds");
  if (col_start % kFp4Group != 0 || cols % kFp4Group != 0)
    throw std::invalid_argument(
        "GlmTpViews: NVFP4 column slice must start and end on 16-element "
        "block boundaries");
  copy2d(m.payload + col_start / 2, size_t(m.payload_cols()), payload,
         size_t(cols / 2), size_t(cols / 2), size_t(m.rows), stream_);
  copy2d(m.scales + col_start / kFp4Group, size_t(m.scale_cols()), scales,
         size_t(cols / kFp4Group), size_t(cols / kFp4Group), size_t(m.rows),
         stream_);
  GlmFp4Matrix v;
  v.payload = payload;
  v.scales = scales;
  v.global_scale = m.global_scale;
  v.rows = m.rows;
  v.cols = cols;
  return v;
}

// Column slice -> PACKED payload + packed scale columns, on the model
// stream. Column starts carry the same 128-alignment contract as row
// slices: the local scale grid re-anchors at the pack origin, which is
// exact only when the origin sits on a scale-block boundary.
GlmQuantMatrix GlmTpViews::pack_quant_cols(const GlmQuantMatrix& m,
                                           int64_t col_start, int64_t cols,
                                           uint8_t* payload, float* scales) {
  if (col_start < 0 || cols <= 0 || cols > m.cols || col_start > m.cols - cols)
    throw std::invalid_argument("GlmTpViews: column slice out of bounds");
  if (col_start % 128 != 0)
    throw std::invalid_argument(
        "GlmTpViews: quantized column slice must start 128-aligned "
        "(scale-grid slice contract)");
  const int64_t sb_full = (m.cols + 127) / 128;
  const int64_t sb_s = (cols + 127) / 128;
  copy2d(m.payload + col_start, size_t(m.cols), payload, size_t(cols),
         size_t(cols), size_t(m.rows), stream_);
  copy2d(m.scales + col_start / 128, size_t(sb_full) * 4, scales,
         size_t(sb_s) * 4, size_t(sb_s) * 4, size_t((m.rows + 127) / 128),
         stream_);
  GlmQuantMatrix v;
  v.payload = payload;
  v.scales = scales;
  v.rows = m.rows;
  v.cols = cols;
  return v;
}

void GlmTpViews::pack2d_bf16(const void* src, size_t src_pitch_bytes,
                             uint16_t* dst, size_t dst_pitch_bytes,
                             size_t width_bytes, size_t rows) {
  copy2d(src, src_pitch_bytes, dst, dst_pitch_bytes, width_bytes, rows,
         stream_);
}

GlmLayerBound GlmTpViews::bind(const GlmLayerResident& r, bool dense_mlp) {
  const SlabLayout lay = layout(cfg_, world_);  // same arithmetic as alloc
  uint8_t* slab = static_cast<uint8_t*>(slab_);
  const int H = cfg_.hidden_size;
  bound_ = GlmLayerBound{};
  bound_.mhc = &r.mhc;
  bound_.ln1 = r.ln1;
  bound_.ln2 = r.ln2;

  if (r.kind == GlmLayerKind::Kda) {
    const int hd = kda_cfg_.head_dim;
    const int lp_f = kda_geo_.local_proj * world_;   // full-heads local_proj
    const int lp_s = kda_geo_.local_proj;           // this rank's slice
    const int h_s = kda_geo_.local_heads;
    const uint16_t* in = static_cast<const uint16_t*>(r.kda.in_proj);
    uint16_t* in_s =
        reinterpret_cast<uint16_t*>(slab + lay.off_kda_in_proj);
    // Fused layout [f_a|g_a|q|k|v|b]: the replicated prefix copies whole;
    // each head-sharded section contributes this rank's row range.
    const size_t pitch = size_t(H) * 2;
    copy2d(in, pitch, in_s, pitch, pitch, size_t(2 * hd), stream_);
    const int off_q = 2 * hd;
    for (int s = 0; s < 3; ++s)  // q, k, v sections
      copy2d(in + size_t(off_q + s * lp_f + rank_ * lp_s) * H, pitch,
             in_s + size_t(off_q + s * lp_s) * H, pitch, pitch, lp_s,
             stream_);
    copy2d(in + size_t(off_q + 3 * lp_f + rank_ * h_s) * H, pitch,
           in_s + size_t(off_q + 3 * lp_s) * H, pitch, pitch, h_s, stream_);
    kda_.in_proj = in_s;

    // conv [q|k|v] sections along channels; the same per-section range.
    const uint16_t* conv = static_cast<const uint16_t*>(r.kda.conv);
    uint16_t* conv_s =
        reinterpret_cast<uint16_t*>(slab + lay.off_kda_conv);
    const size_t cpitch = size_t(kda_cfg_.conv_width) * 2;
    for (int s = 0; s < 3; ++s)
      copy2d(conv + size_t(s * lp_f + rank_ * lp_s) * kda_cfg_.conv_width,
             cpitch, conv_s + size_t(s * lp_s) * kda_cfg_.conv_width,
             cpitch, cpitch, lp_s, stream_);
    kda_.conv = conv_s;

    kda_.f_b = static_cast<const uint16_t*>(r.kda.f_b) +
               size_t(rank_ * lp_s) * hd;
    kda_.g_b = static_cast<const uint16_t*>(r.kda.g_b) +
               size_t(rank_ * lp_s) * hd;
    kda_.a_log = r.kda.a_log + rank_ * h_s;
    kda_.dt_bias = r.kda.dt_bias + rank_ * lp_s;
    kda_.o_norm = r.kda.o_norm;

    // o_proj [H, lp_f]: column slice -> packed [H, lp_s].
    uint16_t* op_s =
        reinterpret_cast<uint16_t*>(slab + lay.off_kda_o_proj);
    pack2d_bf16(static_cast<const uint16_t*>(r.kda.o_proj) + rank_ * lp_s,
                size_t(lp_f) * 2, op_s, size_t(lp_s) * 2, size_t(lp_s) * 2,
                H);
    kda_.o_proj = op_s;
    bound_.kda = &kda_;
  } else {
    // DSA: latents (qkv_a, q_aln, kv_aln) and the indexer are replicated
    // full views; q_b/kv_b are contiguous head blocks; o_proj packs.
    dsa_ = r.dsa;  // replicated defaults, overwritten below
    const int lh = dsa_geo_.local_heads;
    const int kv_rows =
        dsa_cfg_.qk_nope_head_dim + dsa_cfg_.v_head_dim;  // per-head rows
    dsa_.kv_b = static_cast<const uint16_t*>(r.dsa.kv_b) +
                size_t(rank_ * lh) * kv_rows * dsa_cfg_.kv_lora_rank;
    // The same rule the loader applies at this world (is_dsa_bridge): the
    // FP8 pairs directly when the slices start on the 128-wide scale grid.
    const bool aligned = (dsa_geo_.local_q_rows % 128) == 0 &&
                         (dsa_geo_.local_v_rows % 128) == 0;
    if (r.dsa.quantized() && aligned) {
      // The FP8 form: q_b this rank's 128-aligned row range of
      // the full pair, o_proj its packed column slice — payload then scale
      // grid in the slab region the bf16 pack used (the fp8 pack is
      // smaller than the bf16 one).
      dsa_.q_b_q = quant_rows_view(r.dsa.q_b_q,
                                   int64_t(rank_) * dsa_geo_.local_q_rows,
                                   dsa_geo_.local_q_rows);
      uint8_t* op_payload = slab + lay.off_dsa_o_proj;
      float* op_scales = reinterpret_cast<float*>(
          op_payload + size_t(H) * dsa_geo_.local_v_rows);
      dsa_.o_proj_q = pack_quant_cols(r.dsa.o_proj_q,
                                      int64_t(rank_) * dsa_geo_.local_v_rows,
                                      dsa_geo_.local_v_rows, op_payload, op_scales);
    } else {
      const uint16_t* q_b_full = static_cast<const uint16_t*>(r.dsa.q_b);
      const uint16_t* o_full = static_cast<const uint16_t*>(r.dsa.o_proj);
      if (r.dsa.quantized()) {
        // The resident holds the pairs but this world's slices are
        // misaligned: dequantize the full pairs (the bridge's kernel and
        // rounding) and slice the bf16 exactly as the sharded loader's
        // bridge does at this world.
        const size_t ql = static_cast<size_t>(cfg_.q_lora_rank);
        const size_t kvl = static_cast<size_t>(cfg_.kv_lora_rank);
        const size_t qkv_bytes = (ql + kvl) * H * 2;
        const size_t qb_bytes = static_cast<size_t>(r.dsa.q_b_q.rows) * ql * 2;
        const size_t o_bytes = static_cast<size_t>(H) * r.dsa.o_proj_q.cols * 2;
        const size_t want = qkv_bytes + qb_bytes + o_bytes;
        if (dsa_bridge_bytes_ < want) {
          if (dsa_bridge_) cudaFree(dsa_bridge_);
          DGPP_CUDA_OK(cudaMalloc(&dsa_bridge_, want));
          dsa_bridge_bytes_ = want;
        }
        uint16_t* qkv = reinterpret_cast<uint16_t*>(dsa_bridge_);
        uint16_t* qb = reinterpret_cast<uint16_t*>(dsa_bridge_ + qkv_bytes);
        uint16_t* ob = reinterpret_cast<uint16_t*>(dsa_bridge_ + qkv_bytes + qb_bytes);
        launch_fp8_dequant_blocks(r.dsa.q_a_q.payload, r.dsa.q_a_q.scales, qkv,
                                  r.dsa.q_a_q.rows, r.dsa.q_a_q.cols, nullptr);
        launch_fp8_dequant_blocks(r.dsa.kv_a_q.payload, r.dsa.kv_a_q.scales,
                                  qkv + ql * H, r.dsa.kv_a_q.rows, r.dsa.kv_a_q.cols,
                                  nullptr);
        launch_fp8_dequant_blocks(r.dsa.q_b_q.payload, r.dsa.q_b_q.scales, qb,
                                  r.dsa.q_b_q.rows, r.dsa.q_b_q.cols, nullptr);
        launch_fp8_dequant_blocks(r.dsa.o_proj_q.payload, r.dsa.o_proj_q.scales, ob,
                                  r.dsa.o_proj_q.rows, r.dsa.o_proj_q.cols, nullptr);
        DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
        dsa_.qkv_a = qkv;
        dsa_.q_a_q = GlmQuantMatrix{};
        dsa_.kv_a_q = GlmQuantMatrix{};
        dsa_.q_b_q = GlmQuantMatrix{};
        dsa_.o_proj_q = GlmQuantMatrix{};
        q_b_full = qb;
        o_full = ob;
      }
      dsa_.q_b = q_b_full + size_t(rank_ * dsa_geo_.local_q_rows) * cfg_.q_lora_rank;
      uint16_t* op_s =
          reinterpret_cast<uint16_t*>(slab + lay.off_dsa_o_proj);
      // Full o_proj columns at world=1 = local_v_rows * world_.
      pack2d_bf16(o_full + rank_ * dsa_geo_.local_v_rows,
                  size_t(dsa_geo_.local_v_rows) * world_ * 2, op_s,
                  size_t(dsa_geo_.local_v_rows) * 2,
                  size_t(dsa_geo_.local_v_rows) * 2, H);
      dsa_.o_proj = op_s;
    }
    bound_.dsa = &dsa_;
  }

  if (dense_mlp) {
    const int64_t I = dense_inter_;
    dense_[0] = quant_rows_view(r.dense[0], rank_ * I, I);
    dense_[1] = quant_rows_view(r.dense[1], rank_ * I, I);
    // down [H, I_full]: column slice -> packed payload + scale columns in
    // the slab (the carve is non-const for the copies; the views store
    // const).
    dense_[2] = pack_quant_cols(
        r.dense[2], rank_ * I, I, slab + lay.off_dense_down,
        reinterpret_cast<float*>(slab + lay.off_dense_scales));
    bound_.dense = dense_;
  } else {
    const GlmMoeResident& m = r.moe;
    const int64_t M = moe_inter_;
    moe_.router_gate = m.router_gate;
    moe_.router_bias = m.router_bias;
    shared_[0] = quant_rows_view(m.shared[0], rank_ * M, M);
    shared_[1] = quant_rows_view(m.shared[1], rank_ * M, M);
    shared_[2] = pack_quant_cols(
        m.shared[2], rank_ * M, M, slab + lay.off_shared_down,
        reinterpret_cast<float*>(slab + lay.off_shared_scales));
    for (int i = 0; i < 3; ++i) moe_.shared[i] = shared_[i];
    // Every routed expert, sliced like the shared one: gate/up row views
    // into the full resident, down column-packed into the (lazily
    // allocated) expert pack region. The per-expert slot is the FP8 size
    // under both formats (an NVFP4 pack is smaller and fits the same slot).
    const int E = moe_cfg_.n_experts;
    const size_t want = static_cast<size_t>(E) * 3;
    if ((m.nvfp4() ? m.experts_fp4.size() : m.experts.size()) != want)
      throw std::invalid_argument(
          "GlmTpViews::bind: full resident layer does not carry every "
          "expert");
    ensure_expert_pack();
    const size_t H = static_cast<size_t>(cfg_.hidden_size);
    const size_t payload_bytes = align256(H * static_cast<size_t>(M));
    const size_t scale_bytes = align256(
        ((H + 127) / 128) * ((static_cast<size_t>(M) + 127) / 128) * 4);
    uint8_t* cursor = expert_pack_;
    if (m.nvfp4()) {
      const size_t fp4_payload = align256(H * static_cast<size_t>(M) / 2);
      for (int e = 0; e < E; ++e) {
        const size_t at = static_cast<size_t>(e) * 3;
        experts_fp4_[at + 0] = fp4_rows_view(m.experts_fp4[at + 0], rank_ * M, M);
        experts_fp4_[at + 1] = fp4_rows_view(m.experts_fp4[at + 1], rank_ * M, M);
        experts_fp4_[at + 2] = pack_fp4_cols(m.experts_fp4[at + 2], rank_ * M, M,
                                             cursor, cursor + fp4_payload);
        cursor += payload_bytes + scale_bytes;
      }
      moe_.experts = nullptr;
      moe_.experts_fp4 = experts_fp4_.data();
      bound_.moe = &moe_;
      return bound_;
    }
    for (int e = 0; e < E; ++e) {
      const size_t at = static_cast<size_t>(e) * 3;
      experts_[at + 0] = quant_rows_view(m.experts[at + 0], rank_ * M, M);
      experts_[at + 1] = quant_rows_view(m.experts[at + 1], rank_ * M, M);
      uint8_t* payload = cursor;
      float* scales = reinterpret_cast<float*>(cursor + payload_bytes);
      experts_[at + 2] =
          pack_quant_cols(m.experts[at + 2], rank_ * M, M, payload, scales);
      cursor += payload_bytes + scale_bytes;
    }
    moe_.experts = experts_.data();
    moe_.experts_fp4 = nullptr;
    bound_.moe = &moe_;
  }
  return bound_;
}

GlmLayerBound GlmTpViews::bind_sharded(const GlmLayerResident& r,
                                       bool dense_mlp) {
  // The resident IS this rank's geometry (the sharded loader's contract,
  // M5 d4): wire the pointers wholesale and stamp the expert partition
  // the loader recorded. No slab, no copies — the loader did the merges
  // and packs at load time, on the CPU, straight from the mmaps.
  bound_ = GlmLayerBound{};
  bound_.mhc = &r.mhc;
  bound_.ln1 = r.ln1;
  bound_.ln2 = r.ln2;
  if (r.kind == GlmLayerKind::Kda) {
    kda_ = r.kda;
    bound_.kda = &kda_;
  } else {
    dsa_ = r.dsa;
    bound_.dsa = &dsa_;
  }
  if (dense_mlp) {
    for (int i = 0; i < 3; ++i) dense_[i] = r.dense[i];
    bound_.dense = dense_;
  } else {
    // A full-resident layer here would silently execute every expert
    // UNSLICED as this rank's "partial" — the one footgun worth a guard at
    // the interface. The slice width is the geometry's signature.
    const size_t want = static_cast<size_t>(moe_cfg_.n_experts) * 3;
    const bool fp4 = r.moe.nvfp4();
    const int64_t slice = fp4 ? (r.moe.experts_fp4.empty() ? -1 : r.moe.experts_fp4[0].rows)
                              : (r.moe.experts.empty() ? -1 : r.moe.experts[0].rows);
    if ((fp4 ? r.moe.experts_fp4.size() : r.moe.experts.size()) != want ||
        slice != moe_inter_ || r.moe.shared[0].rows != moe_inter_)
      throw std::invalid_argument(
          "GlmTpViews::bind_sharded: resident MoE layer is not this world's "
          "slice geometry — was it loaded by a world=1 stream?");
    moe_.router_gate = r.moe.router_gate;
    moe_.router_bias = r.moe.router_bias;
    for (int i = 0; i < 3; ++i) moe_.shared[i] = r.moe.shared[i];
    moe_.experts = fp4 ? nullptr : r.moe.experts.data();
    moe_.experts_fp4 = fp4 ? r.moe.experts_fp4.data() : nullptr;
    bound_.moe = &moe_;
  }
  return bound_;
}

}  // namespace dgpp
