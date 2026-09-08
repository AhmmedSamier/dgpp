#include "models/glm/tp_parity.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "models/dsa_geometry.hpp"
#include "models/glm/loader.hpp"
#include "models/glm/tp.hpp"
#include "models/kda_geometry.hpp"

namespace dgpp {

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Compares the two bind paths' outputs for one layer. Sizes come from the
// config at the TEST's local geometry — the slicing spec, spelled out.
struct BoundCmp {
  const GlmTextConfig& cfg;
  int world;
  int checked = 0;

  // Both sides are DEVICE addresses (the loader's bumps and the views'
  // slabs are cudaMalloc'd): fetch, then compare.
  void bytes(const std::string& what, const void* a, const void* b, size_t n) {
    std::vector<uint8_t> ha(n), hb(n);
    if (n) {
      DGPP_CUDA_OK(cudaMemcpy(ha.data(), a, n, cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(hb.data(), b, n, cudaMemcpyDeviceToHost));
    }
    if (ha != hb)
      throw std::runtime_error(what + " differs bitwise (full+bind vs "
                                   "sharded)");
    ++checked;
  }
  void ints(const std::string& what, int64_t a, int64_t b) {
    if (a != b)
      throw std::runtime_error(what + " differs (" + std::to_string(a) +
                               " vs " + std::to_string(b) + ")");
    ++checked;
  }
  void fp4(const std::string& what, const dgpp::GlmFp4Matrix& a,
           const dgpp::GlmFp4Matrix& b) {
    ints(what + ".rows", a.rows, b.rows);
    ints(what + ".cols", a.cols, b.cols);
    bytes(what, a.payload, b.payload, a.payload_bytes());
    bytes(what + ".scales", a.scales, b.scales, a.scale_bytes());
    bytes(what + ".global_scale", a.global_scale, b.global_scale, 4);
  }
  void quant(const std::string& what, const dgpp::GlmQuantMatrix& a,
             const dgpp::GlmQuantMatrix& b) {
    ints(what + ".rows", a.rows, b.rows);
    ints(what + ".cols", a.cols, b.cols);
    bytes(what, a.payload, b.payload,
          static_cast<size_t>(a.rows) * static_cast<size_t>(a.cols));
    bytes(what + ".scales", a.scales, b.scales,
          static_cast<size_t>((a.rows + 127) / 128) *
              static_cast<size_t>((a.cols + 127) / 128) * 4);
  }

  void run(int layer, bool dense_mlp, const dgpp::GlmLayerBound& a,
           const dgpp::GlmLayerBound& b) {
    const int64_t H = cfg.hidden_size;
    // Layer-tagged surface names: a bitwise mismatch names the layer and
    // the surface, not just "differs somewhere".
    const auto tag = [layer](const char* what) {
      return "shard parity (layer " + std::to_string(layer) + "): " + what;
    };
    bytes(tag("ln1"), a.ln1, b.ln1, H * 2);
    bytes(tag("ln2"), a.ln2, b.ln2, H * 2);
    // mHC (empty on the MTP layer — both sides null).
    const bool has_mhc = a.mhc && a.mhc->attn_base;
    if (has_mhc != (b.mhc && b.mhc->attn_base))
      throw std::runtime_error(tag("mhc presence differs"));
    if (has_mhc) {
      // kHcCoeffRows (24) / kHcScaleOutputs (3) — glm_binding's pinned
      // mHC coefficient row counts.
      const size_t base_b = 24 * 4;
      const size_t fn_b = static_cast<size_t>(cfg.hc_mult) * H * 24 * 2;
      const size_t scale_b = 3 * 4;
      bytes(tag("mhc.attn_base"), a.mhc->attn_base, b.mhc->attn_base, base_b);
      bytes(tag("mhc.attn_fn"), a.mhc->attn_fn, b.mhc->attn_fn, fn_b);
      bytes(tag("mhc.attn_scale"), a.mhc->attn_scale, b.mhc->attn_scale,
           scale_b);
      bytes(tag("mhc.ffn_base"), a.mhc->ffn_base, b.mhc->ffn_base, base_b);
      bytes(tag("mhc.ffn_fn"), a.mhc->ffn_fn, b.mhc->ffn_fn, fn_b);
      bytes(tag("mhc.ffn_scale"), a.mhc->ffn_scale, b.mhc->ffn_scale,
           scale_b);
    }
    if (a.kda) {
      dgpp::KdaConfig kc = cfg.kda_config();
      kc.tp_size = world;
      const dgpp::KdaGeometry kg = dgpp::KdaGeometry::from_config(kc);
      const int64_t hd = cfg.kda_head_dim;
      const int64_t lp_s = kg.local_proj;
      const int64_t h_s = kg.local_heads;
      bytes(tag("kda.in_proj"), a.kda->in_proj, b.kda->in_proj,
            static_cast<size_t>(kg.in_proj_cols) * H * 2);
      bytes(tag("kda.conv"), a.kda->conv, b.kda->conv,
            static_cast<size_t>(kg.conv_channels) * cfg.kda_conv_width * 2);
      bytes(tag("kda.f_b"), a.kda->f_b, b.kda->f_b, lp_s * hd * 2);
      bytes(tag("kda.g_b"), a.kda->g_b, b.kda->g_b, lp_s * hd * 2);
      bytes(tag("kda.a_log"), a.kda->a_log, b.kda->a_log, h_s * 4);
      bytes(tag("kda.dt_bias"), a.kda->dt_bias, b.kda->dt_bias, lp_s * 4);
      bytes(tag("kda.o_norm"), a.kda->o_norm, b.kda->o_norm, hd * 2);
      bytes(tag("kda.o_proj"), a.kda->o_proj, b.kda->o_proj, H * lp_s * 2);
    }
    if (a.dsa) {
      dgpp::DsaConfig dc = cfg.dsa_config();
      dc.tp_size = world;
      const dgpp::DsaGeometry dgeo = dgpp::DsaGeometry::from_config(dc);
      const int64_t ql = cfg.q_lora_rank;
      const int64_t kvl = cfg.kv_lora_rank;
      const int64_t lh = dgeo.local_heads;
      const int64_t kv_rows = lh * (cfg.qk_nope_head_dim + cfg.v_head_dim);
      const int64_t idx_proj = cfg.index_n_heads * cfg.index_head_dim;
      bytes(tag("dsa.qkv_a"), a.dsa->qkv_a, b.dsa->qkv_a, (ql + kvl) * H * 2);
      bytes(tag("dsa.q_aln"), a.dsa->q_aln, b.dsa->q_aln, ql * 2);
      bytes(tag("dsa.kv_aln"), a.dsa->kv_aln, b.dsa->kv_aln, kvl * 2);
      bytes(tag("dsa.q_b"), a.dsa->q_b, b.dsa->q_b,
            static_cast<size_t>(dgeo.local_q_rows) * ql * 2);
      bytes(tag("dsa.kv_b"), a.dsa->kv_b, b.dsa->kv_b, kv_rows * kvl * 2);
      bytes(tag("dsa.o_proj"), a.dsa->o_proj, b.dsa->o_proj,
            H * static_cast<size_t>(dgeo.local_v_rows) * 2);
      bytes(tag("dsa.wq_b"), a.dsa->wq_b, b.dsa->wq_b, idx_proj * ql * 2);
      bytes(tag("dsa.wk"), a.dsa->wk, b.dsa->wk,
            static_cast<size_t>(cfg.index_head_dim) * H * 2);
      bytes(tag("dsa.wp"), a.dsa->wp, b.dsa->wp,
            static_cast<size_t>(cfg.index_n_heads) * H * 2);
      bytes(tag("dsa.k_norm_w"), a.dsa->k_norm_w, b.dsa->k_norm_w,
            cfg.index_head_dim * 2);
      bytes(tag("dsa.k_norm_b"), a.dsa->k_norm_b, b.dsa->k_norm_b,
            cfg.index_head_dim * 2);
      if (a.dsa->gate) {
        bytes(tag("dsa.gate"), a.dsa->gate, b.dsa->gate,
              static_cast<size_t>(cfg.index_head_dim) * H * 2);
        bytes(tag("dsa.ape"), a.dsa->ape, b.dsa->ape,
              static_cast<size_t>(cfg.index_kpool) * cfg.index_head_dim * 4);
      }
    }
    if (dense_mlp) {
      for (int i = 0; i < 3; ++i)
        quant(tag("dense"), a.dense[i], b.dense[i]);
    } else {
      const int64_t E = cfg.moe_config().n_experts;
      const int64_t M = cfg.moe_config().inter / world;
      bytes(tag("moe.router_gate"), a.moe->router_gate, b.moe->router_gate,
            E * H * 2);
      bytes(tag("moe.router_bias"), a.moe->router_bias, b.moe->router_bias,
            E * 4);
      for (int i = 0; i < 3; ++i)
        quant(tag("moe.shared"), a.moe->shared[i], b.moe->shared[i]);
      // Every expert, sliced: the full+bind path views/packs the rank's
      // inter slice out of the full expert; the sharded path loaded exactly
      // that slice. Same width on both sides, by construction — assert it
      // once so a whole-expert resident cannot pass as "equal". Both sides
      // must agree on the format, too.
      require(a.moe->nvfp4() == b.moe->nvfp4(),
              tag("expert format differs between the bind paths"));
      if (a.moe->nvfp4()) {
        require(a.moe->experts_fp4[0].rows == M && b.moe->experts_fp4[0].rows == M,
                tag("expert slice width is not inter/world"));
        for (int64_t e = 0; e < E; ++e)
          for (int i = 0; i < 3; ++i)
            fp4(tag("moe.expert(nvfp4)"),
                a.moe->experts_fp4[static_cast<size_t>(e) * 3 + i],
                b.moe->experts_fp4[static_cast<size_t>(e) * 3 + i]);
      } else {
        require(a.moe->experts[0].rows == M && b.moe->experts[0].rows == M,
                tag("expert slice width is not inter/world"));
        for (int64_t e = 0; e < E; ++e)
          for (int i = 0; i < 3; ++i)
            quant(tag("moe.expert"),
                  a.moe->experts[static_cast<size_t>(e) * 3 + i],
                  b.moe->experts[static_cast<size_t>(e) * 3 + i]);
      }
    }
  }
};

}  // namespace

// Address fingerprint of one resident view — the representative pointer
// of every site (norms, mHC, one attention matrix per kind, the MLP
// payload/scale pair, the router and the first expert). A cache hit must
// return the SAME addresses for every one of these; a rebuild or a bump
// move changes them.
std::vector<const void*> view_fingerprint(const GlmLayerResident& r) {
  std::vector<const void*> f;
  f.push_back(r.ln1);
  f.push_back(r.ln2);
  f.push_back(r.mhc.attn_fn);
  if (r.kind == GlmLayerKind::Kda) {
    f.push_back(r.kda.in_proj);
    f.push_back(r.kda.o_proj);
  } else {
    f.push_back(r.dsa.wq_b);
    f.push_back(r.dsa.qkv_a);
  }
  f.push_back(r.dense[0].payload);
  f.push_back(r.dense[0].scales);
  f.push_back(r.moe.router_gate);
  if (!r.moe.experts.empty()) {
    f.push_back(r.moe.experts[0].payload);
    f.push_back(r.moe.experts[0].scales);
  }
  if (!r.moe.experts_fp4.empty()) {
    f.push_back(r.moe.experts_fp4[0].payload);
    f.push_back(r.moe.experts_fp4[0].scales);
    f.push_back(r.moe.expert_global_scales);
  }
  f.push_back(reinterpret_cast<const void*>(r.bytes));
  return f;
}

GlmShardParityReport glm_shard_parity_check(
    const GlmTextConfig& cfg, const std::string& checkpoint_dir, int world,
    const std::vector<int>* layers, bool resident) {
  cudaStream_t st;
  DGPP_CUDA_OK(cudaStreamCreate(&st));
  GlmShardParityReport report;
  report.world = world;

  try {
    // Fresh full (world=1) stream: its byte counters accumulate across
    // loads, so reusing one across worlds would double-count (the parity
    // loop reloads every layer per world).
    GlmLayerStream full(cfg, checkpoint_dir);
    const GlmReplicatedDigest ref_digest = full.hash_replicated();
    report.digest_bytes = ref_digest.bytes;
    report.digest_tensors = ref_digest.tensors;

    std::vector<std::unique_ptr<GlmLayerStream>> shards(
        static_cast<size_t>(world));
    std::vector<std::unique_ptr<GlmTpViews>> views(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
      shards[static_cast<size_t>(r)] = std::make_unique<GlmLayerStream>(
          cfg, checkpoint_dir, r, world,
          resident ? GlmResidency::Resident : GlmResidency::Streaming);
      views[static_cast<size_t>(r)] =
          std::make_unique<GlmTpViews>(cfg, r, world, st);
      // The boot digest must be rank-invariant AND identical to the
      // world=1 pass — same files, same replicated set, order-independent
      // fold. A mismatch here is the silent-corruption class: some rank
      // would compute different routers/latent projections.
      const GlmReplicatedDigest d =
          shards[static_cast<size_t>(r)]->hash_replicated();
      require(d.layer == ref_digest.layer && d.globals == ref_digest.globals &&
                  d.bytes == ref_digest.bytes &&
                  d.tensors == ref_digest.tensors,
              "shard parity: replicated digest differs (rank " +
                  std::to_string(r) + ")");
    }

    // The layer list: all layers (MTP last) or the caller's subset,
    // validated against the model's range.
    const int max_layer =
        cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    std::vector<int> all;
    if (!layers) {
      all.reserve(static_cast<size_t>(max_layer));
      for (int l = 0; l < max_layer; ++l) all.push_back(l);
      layers = &all;
    }
    for (int l : *layers) {
      if (l < 0 || l >= max_layer)
        throw std::invalid_argument(
            "shard parity: layer index out of range: " + std::to_string(l));
    }

    int surfaces = 0;
    // Resident mode: fingerprints of each rank's first-pass views, so the
    // post-loop proof pass can demand the SAME addresses back from cache.
    std::vector<std::vector<std::vector<const void*>>> fingerprints(
        static_cast<size_t>(world));
    for (int l : *layers) {
      const bool dense_mlp = l < cfg.num_hidden_layers &&
                             cfg.mlps[static_cast<size_t>(l)] ==
                                 GlmMlpKind::Dense;
      const GlmLayerResident& fr = full.load_layer(l);
      for (int r = 0; r < world; ++r) {
        const GlmLayerResident& lr = shards[static_cast<size_t>(r)]->load_layer(l);
        if (resident)
          fingerprints[static_cast<size_t>(r)].push_back(view_fingerprint(lr));
        const GlmLayerBound a = views[static_cast<size_t>(r)]->bind(fr, dense_mlp);
        const GlmLayerBound b =
            views[static_cast<size_t>(r)]->bind_sharded(lr, dense_mlp);
        // bind()'s slab packs are async on the stream; both sides must be
        // landed before the memcmps.
        DGPP_CUDA_OK(cudaStreamSynchronize(st));
        BoundCmp cmp{cfg, world};
        cmp.run(l, dense_mlp, a, b);
        surfaces += cmp.checked;
      }
      DGPP_LOG_INFO("shard parity world={} layer {} ({}): {} bound surfaces "
                    "bitwise-equal",
                    world, l,
                    dense_mlp ? "dense" : "moe", surfaces);
      ++report.layers_checked;
    }

    // ---- byte reconcile (the arithmetic pin) --------------------------
    full.load_globals();
    uint64_t sum = 0;
    for (int r = 0; r < world; ++r) {
      shards[static_cast<size_t>(r)]->load_globals();
      sum += shards[static_cast<size_t>(r)]->source_bytes_read();
      require(shards[static_cast<size_t>(r)]->verbatim_source_bytes() ==
                  shards[0]->verbatim_source_bytes(),
              "shard parity: verbatim re-read set differs across ranks");
      require(shards[static_cast<size_t>(r)]->source_bytes_read() <
                  full.source_bytes_read(),
              "shard parity: sharded rank reads as much as the full load");
    }
    const uint64_t expect = full.source_bytes_read() +
                           static_cast<uint64_t>(world - 1) *
                               shards[0]->verbatim_source_bytes();
    require(sum == expect,
            "shard parity: byte reconcile failed — sum " +
                std::to_string(sum) + " != total + (world-1)*verbatim " +
                std::to_string(expect));

    // ---- resident cache-hit proof (the residency contract) -----------
    // Every requested layer re-served from cache: SAME addresses, ZERO
    // storage reads. A rebuild, a bump move, or a stray re-read fails
    // here — this is the "storage is never touched again" clause, as an
    // assertion instead of a promise.
    if (resident) {
      int hits = 0;
      for (int r = 0; r < world; ++r) {
        const uint64_t before = shards[static_cast<size_t>(r)]->source_bytes_read();
        for (size_t i = 0; i < fingerprints[static_cast<size_t>(r)].size();
             ++i) {
          const int l = (*layers)[i];
          const GlmLayerResident& again =
              shards[static_cast<size_t>(r)]->load_layer(l);
          const std::vector<const void*> got = view_fingerprint(again);
          require(got == fingerprints[static_cast<size_t>(r)][i],
                  "shard parity: resident cache hit changed the view "
                  "(layer " +
                      std::to_string(l) + ", rank " + std::to_string(r) +
                      ") — the layer was rebuilt, not served");
          ++hits;
        }
        require(shards[static_cast<size_t>(r)]->source_bytes_read() == before,
                "shard parity: resident cache pass re-read storage (rank " +
                    std::to_string(r) + ")");
      }
      report.cache_hits = hits;
    }

    report.surfaces_checked = surfaces;
    report.full_source_bytes = full.source_bytes_read();
    report.shard_source_bytes = shards[0]->source_bytes_read();
    report.verbatim_bytes = shards[0]->verbatim_source_bytes();
    DGPP_CUDA_OK(cudaStreamDestroy(st));
    return report;
  } catch (...) {
    cudaStreamDestroy(st);  // no throw path may leak the stream
    throw;
  }
}

}  // namespace dgpp
