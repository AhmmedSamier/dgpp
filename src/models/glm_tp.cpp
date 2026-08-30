#include "models/glm_tp.hpp"

#include <algorithm>
#include <cstring>

#include "common/cuda_check.hpp"

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
  if (world <= 1)
    throw std::invalid_argument("GlmTpViews: world must be > 1");
  if (rank < 0 || rank >= world)
    throw std::invalid_argument("GlmTpViews: rank out of range");
  if (!stream)
    throw std::invalid_argument("GlmTpViews: null stream");
  const auto fail = [](const char* what) {
    throw std::invalid_argument(std::string("GlmTpViews: ") + what);
  };

  // Head divisibility is enforced by the geometry validators (they throw
  // with the exact dim); the inter/expert splits are ours to check.
  kda_cfg_.tp_size = world;
  dsa_cfg_.tp_size = world;
  if (moe_cfg_.n_experts % world != 0)
    fail("n_routed_experts must divide by world (whole-expert shards)");
  if (cfg.intermediate_size % world != 0)
    fail("intermediate_size must divide by world (dense MLP TP)");
  if (moe_cfg_.inter % world != 0)
    fail("moe_intermediate_size must divide by world (shared expert TP)");
  kda_geo_ = KdaGeometry::from_config(kda_cfg_);
  dsa_geo_ = DsaGeometry::from_config(dsa_cfg_);
  local_experts_ = moe_cfg_.n_experts / world;
  dense_inter_ = cfg.intermediate_size / world;

  const SlabLayout lay = layout(cfg_, world_);
  slab_bytes_ = lay.total;
  DGPP_CUDA_OK(cudaMalloc(&slab_, slab_bytes_));
}

GlmTpViews::~GlmTpViews() {
  if (slab_) cudaFree(slab_);
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
    dsa_.q_b = static_cast<const uint16_t*>(r.dsa.q_b) +
               size_t(rank_ * dsa_geo_.local_q_rows) * cfg_.q_lora_rank;
    const int kv_rows =
        dsa_cfg_.qk_nope_head_dim + dsa_cfg_.v_head_dim;  // per-head rows
    dsa_.kv_b = static_cast<const uint16_t*>(r.dsa.kv_b) +
                size_t(rank_ * lh) * kv_rows * dsa_cfg_.kv_lora_rank;
    uint16_t* op_s =
        reinterpret_cast<uint16_t*>(slab + lay.off_dsa_o_proj);
    // Full o_proj columns at world=1 = local_v_rows * world_.
    pack2d_bf16(static_cast<const uint16_t*>(r.dsa.o_proj) +
                    rank_ * dsa_geo_.local_v_rows,
                size_t(dsa_geo_.local_v_rows) * world_ * 2, op_s,
                size_t(dsa_geo_.local_v_rows) * 2,
                size_t(dsa_geo_.local_v_rows) * 2, H);
    dsa_.o_proj = op_s;
    bound_.dsa = &dsa_;
  }

  if (dense_mlp) {
    const int64_t I = dense_inter_;
    dense_[0] = quant_rows_view(r.dense[0], rank_ * I, I);
    dense_[1] = quant_rows_view(r.dense[1], rank_ * I, I);
    // down [H, I_full]: column slice -> packed payload + scale columns.
    // The slab carve is non-const for the copies; the views store const.
    const GlmQuantMatrix& dn = r.dense[2];
    const int64_t sb_full = (dn.cols + 127) / 128;
    const int64_t sb_s = (I + 127) / 128;
    uint8_t* down_payload = slab + lay.off_dense_down;
    float* down_scales =
        reinterpret_cast<float*>(slab + lay.off_dense_scales);
    copy2d(dn.payload + rank_ * I, size_t(dn.cols), down_payload,
           size_t(I), size_t(I), size_t(dn.rows), stream_);
    copy2d(dn.scales + (rank_ * I) / 128, size_t(sb_full) * 4, down_scales,
           size_t(sb_s) * 4, size_t(sb_s) * 4,
           size_t((dn.rows + 127) / 128), stream_);
    dense_[2].payload = down_payload;
    dense_[2].scales = down_scales;
    dense_[2].rows = dn.rows;
    dense_[2].cols = I;
    bound_.dense = dense_;
  } else {
    const GlmMoeResident& m = r.moe;
    const int64_t M = moe_cfg_.inter / world_;
    moe_.router_gate = m.router_gate;
    moe_.router_bias = m.router_bias;
    shared_[0] = quant_rows_view(m.shared[0], rank_ * M, M);
    shared_[1] = quant_rows_view(m.shared[1], rank_ * M, M);
    const GlmQuantMatrix& dn = m.shared[2];
    const int64_t sb_full = (dn.cols + 127) / 128;
    const int64_t sb_s = (M + 127) / 128;
    uint8_t* down_payload = slab + lay.off_shared_down;
    float* down_scales =
        reinterpret_cast<float*>(slab + lay.off_shared_scales);
    copy2d(dn.payload + rank_ * M, size_t(dn.cols), down_payload,
           size_t(M), size_t(M), size_t(dn.rows), stream_);
    copy2d(dn.scales + (rank_ * M) / 128, size_t(sb_full) * 4, down_scales,
           size_t(sb_s) * 4, size_t(sb_s) * 4,
           size_t((dn.rows + 127) / 128), stream_);
    shared_[2].payload = down_payload;
    shared_[2].scales = down_scales;
    shared_[2].rows = dn.rows;
    shared_[2].cols = M;
    for (int i = 0; i < 3; ++i) moe_.shared[i] = shared_[i];
    moe_.experts = m.experts.data() + size_t(rank_) * local_experts_ * 3;
    moe_.expert_begin = rank_ * static_cast<int>(local_experts_);
    moe_.expert_count = static_cast<int>(local_experts_);
    bound_.moe = &moe_;
  }
  return bound_;
}

}  // namespace dgpp
