// DSA reference-dump parity runner (M3 deliverable). Host-only TU: the dump
// reader needs minijson, which nvcc's frontend refuses, so this file is
// compiled by the host compiler and linked into the dsa_test binary.
// Called from dsa_test.cu's main() when --dump-file is given.
//
// Declared tolerances: the pure backend computes in IEEE double with bf16
// boundary rounding and the exact pinned fp8 codec (cross-checked
// bit-exactly against the C++ encoder); our path is fp32 accumulation with
// cuBLASLt reduction orders. The layer output and cache contents agree
// within a couple of bf16 ulps; the top-k selection is compared with the
// selection-aware near-tie audit's fallback shape (single boundary swaps
// only) — GEMM order noise can flip a rank-(select_k) tie but never the
// selection structure.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/dsa_dump.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_state.hpp"
#include "tests/cuda/kda_test_helpers.hpp"

namespace {

using namespace dgpp::kda_test;

int check_bf16_tensor(const std::string& what,
                      const std::vector<uint16_t>& got,
                      const dgpp::DsaDumpFile::TensorView& tv, int ulps,
                      double l2_tol, double mismatch_frac) {
  std::vector<uint16_t> want(tv.numel());
  std::memcpy(want.data(), tv.data, tv.nbytes);
  const Stats st = compare_bf16(got, want, ulps);
  require_bf16(what, st, l2_tol, mismatch_frac);
  std::printf("[ OK ] %-28s max_rel=%.3g l2_rel=%.3g mismatches=%ld/%ld\n",
              what.c_str(), st.max_rel, st.l2_rel, st.mismatches, st.n);
  return 0;
}

}  // namespace

int run_dsa_dump_parity(const std::string& path) {
  try {
    dgpp::DsaDumpFile dump = dgpp::DsaDumpFile::load(path);
    const dgpp::DsaConfig cfg = dump.single_layer_config();
    const dgpp::DsaGeometry g = dgpp::DsaGeometry::from_config(cfg);
    const int tokens =
        static_cast<int>(dump.config_json().at("tokens").as_int());

    // Weights: upload the dump's bytes to device memory (the dump is an
    // mmap'd host file — passing file pointers to GEMMs fails with a
    // confusing NOT_SUPPORTED at execution time).
    dgpp::DsaLayerWeights w;
    std::vector<DevBuf> dev;
    auto upload = [&](const char* name, const void** dst) {
      const auto& tv = dump.tensor(name);
      dev.emplace_back(tv.nbytes);
      DGPP_CUDA_OK(cudaMemcpy(dev.back().p, tv.data, tv.nbytes,
                              cudaMemcpyHostToDevice));
      *dst = dev.back().p;
    };
    upload("qkv_a", &w.qkv_a);
    upload("q_aln", &w.q_aln);
    upload("kv_aln", &w.kv_aln);
    upload("q_b", &w.q_b);
    upload("wq_b", &w.wq_b);
    upload("wk", &w.wk);
    upload("gate", &w.gate);
    upload("wp", &w.wp);
    upload("k_norm_w", &w.k_norm_w);
    upload("k_norm_b", &w.k_norm_b);
    upload("kv_b", &w.kv_b);
    upload("o_proj", &w.o_proj);
    {
      const auto& tv = dump.tensor("ape");
      if (tv.dtype != dgpp::DType::F32)
        throw std::runtime_error("ape: expected F32");
      dev.emplace_back(tv.nbytes);
      DGPP_CUDA_OK(cudaMemcpy(dev.back().p, tv.data, tv.nbytes,
                              cudaMemcpyHostToDevice));
      w.ape = static_cast<const float*>(dev.back().p);
    }

    cudaStream_t s = test_stream();
    dgpp::CublasLtGemm gemm;
    DevBuf ws(64ull << 20);

    const int64_t capacity =
        (tokens + cfg.block_tokens - 1) / cfg.block_tokens * cfg.block_tokens;
    dgpp::Arena arena;
    dgpp::Arena::Config ac;
    ac.persistent_hot =
        dgpp::DsaStatePool::cache_bytes(cfg, 1, capacity) +
        dgpp::DsaLayer::scratch_bytes(cfg, tokens, tokens);
    arena.init(ac);
    dgpp::DsaStatePool pool;
    pool.init(arena, cfg, 1, capacity);
    dgpp::DsaLayer layer(gemm, w, cfg, tokens, tokens,
                         arena.alloc_persistent(dgpp::MemClass::DeviceHot,
                                                dgpp::DsaLayer::scratch_bytes(
                                                    cfg, tokens, tokens),
                                                256),
                         dgpp::DsaLayer::scratch_bytes(cfg, tokens, tokens),
                         ws.p, ws.bytes);
    if (!layer.prepare(tokens)) throw std::runtime_error("gemm plans");

    const auto& in_tv = dump.tensor("hidden_in");
    std::vector<uint16_t> hidden_in(
        static_cast<const uint16_t*>(in_tv.data),
        static_cast<const uint16_t*>(in_tv.data) + in_tv.numel());
    DevBuf din(hidden_in.size() * 2), dout(int64_t(tokens) * cfg.hidden * 2);
    din.upload(hidden_in.data(), hidden_in.size() * 2);

    layer.enqueue_prefill(din.p, pool, 0, 0, 0, tokens, dout.p, s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));

    std::vector<uint16_t> got_out(int64_t(tokens) * cfg.hidden);
    dout.download(got_out.data(), got_out.size() * 2);

    // Layer output.
    const std::string tag = " [" + dump.backend() + "]";
    // 8-ulp budget: the pure oracle computes in IEEE double with exact
    // pinned codecs; our path is fp32 GEMMs — the same budget the
    // layer-vs-host-reference tests use for this comparison class.
    check_bf16_tensor("dump layer_out" + tag, got_out,
                      dump.tensor("layer_out"), 8, 5e-3, 5e-2);

    // Cache contents through the block table (the pool is fresh, so the
    // table maps logical->physical for exactly the blocks prefill wrote).
    const std::vector<int32_t> bt = [&] {
      std::vector<int32_t> row(size_t(pool.total_blocks()));
      DGPP_CUDA_OK(cudaMemcpyAsync(
          row.data(),
          pool.block_tables(),
          row.size() * 4, cudaMemcpyDeviceToHost, s));
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      return row;
    }();

    // Latent rows (gather to logical order).
    {
      std::vector<uint16_t> phys(size_t(tokens) * cfg.kv_lora_rank);
      DGPP_CUDA_OK(cudaMemcpyAsync(phys.data(), pool.latent(0),
                                   phys.size() * 2, cudaMemcpyDeviceToHost, s));
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      std::vector<uint16_t> logical(phys.size());
      for (int64_t t = 0; t < tokens; ++t) {
        const int64_t p = int64_t(bt[size_t(t / cfg.block_tokens)]) *
                              cfg.block_tokens +
                          (t % cfg.block_tokens);
        std::memcpy(&logical[size_t(t) * cfg.kv_lora_rank],
                    &phys[size_t(p) * cfg.kv_lora_rank],
                    cfg.kv_lora_rank * 2);
      }
      check_bf16_tensor("dump latent" + tag, logical, dump.tensor("latent"),
                        2, 2e-2, 0.02);
    }

    // Index cache (fp8 codes + fp32 scales) — dequantized-value comparison:
    // GEMM noise near a power-of-two boundary can shift a row's scale by 2x
    // with the VALUES still within a ulp.
    {
      const auto& kref = dump.tensor("index_k");
      const auto& sref = dump.tensor("index_scale");
      const int64_t pools = tokens / cfg.index_kpool;
      const int dim = cfg.index_head_dim;
      std::vector<uint8_t> got_k(size_t(pools) * dim);
      std::vector<float> got_s(size_t(pools), 0.0f);
      std::vector<uint8_t> ref_k(size_t(pools) * dim);
      std::vector<float> ref_s(size_t(pools), 0.0f);
      for (int64_t j = 0; j < pools; ++j) {
        const int64_t slot =
            int64_t(bt[size_t(j / g.pools_per_block)]) * g.pools_per_block +
            (j % g.pools_per_block);
        DGPP_CUDA_OK(cudaMemcpyAsync(&got_k[size_t(j) * dim],
                                     static_cast<const uint8_t*>(
                                         pool.index_k(0)) +
                                         size_t(slot) * dim,
                                     dim, cudaMemcpyDeviceToHost, s));
        DGPP_CUDA_OK(cudaMemcpyAsync(&got_s[size_t(j)],
                                     pool.index_scale(0) + slot, 4,
                                     cudaMemcpyDeviceToHost, s));
      }
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      std::memcpy(ref_k.data(), kref.data, kref.nbytes);
      std::memcpy(ref_s.data(), sref.data, sref.nbytes);
      long bad = 0;
      double l2 = 0, ref2 = 0;
      for (int64_t i = 0; i < pools * dim; ++i) {
        const double a = dgpp::fp8_e4m3_bits_to_float(
                             got_k[size_t(i)]) * double(got_s[size_t(i / dim)]);
        const double b = dgpp::fp8_e4m3_bits_to_float(
                             ref_k[size_t(i)]) * double(ref_s[size_t(i / dim)]);
        const double dd = std::abs(a - b);
        if (dd > std::abs(b) * 0.141 + 1e-9) ++bad;  // ~1.1 e4m3 ulp
        l2 += dd * dd;
        ref2 += b * b;
      }
      if (bad > long(pools) * dim / 20)
        throw std::runtime_error("dump index_k: " + std::to_string(bad) +
                                 " elements beyond one ulp");
      if (l2 / std::max(ref2, 1e-30) > 4e-3)
        throw std::runtime_error("dump index_k: l2 drift");
      std::printf("[ OK ] %-28s l2_rel=%.3g bad=%ld/%lld\n",
                  ("dump index_k" + tag).c_str(), std::sqrt(l2 / std::max(ref2, 1e-30)),
                  bad, (long long)(pools * dim));
    }

    // Top-k: structural comparison (single boundary swaps allowed; the
    // dense-prefix rows are often exact).
    {
      const auto& tv = dump.tensor("topk");
      std::vector<int32_t> ref_topk(tv.numel());
      std::memcpy(ref_topk.data(), tv.data, tv.nbytes);
      std::vector<int32_t> got_topk(size_t(tokens) * g.max_selected);
      DGPP_CUDA_OK(cudaMemcpyAsync(got_topk.data(), layer.debug_topk(),
                                   got_topk.size() * 4,
                                   cudaMemcpyDeviceToHost, s));
      DGPP_CUDA_OK(cudaStreamSynchronize(s));
      int64_t flipped = 0, sparse = 0;
      for (int t = 0; t < tokens; ++t) {
        const int32_t* gt = got_topk.data() + int64_t(t) * g.max_selected;
        const int32_t* rt = ref_topk.data() + int64_t(t) * g.max_selected;
        std::vector<int32_t> gp, rp;
        for (int c = 0; c < g.max_selected; ++c) {
          if (gt[c] >= 0) gp.push_back(gt[c]);
          if (rt[c] >= 0) rp.push_back(rt[c]);
        }
        if (g.visible_pools(t) > g.select_k) ++sparse;
        if (gp == rp) continue;
        ++flipped;
        std::vector<int32_t> og, orr;
        std::sort(gp.begin(), gp.end());
        std::sort(rp.begin(), rp.end());
        std::set_difference(gp.begin(), gp.end(), rp.begin(), rp.end(),
                            std::back_inserter(og));
        std::set_difference(rp.begin(), rp.end(), gp.begin(), gp.end(),
                            std::back_inserter(orr));
        const bool one_swap = og.size() == size_t(cfg.index_kpool) &&
                              orr.size() == size_t(cfg.index_kpool);
        if (!one_swap)
          throw std::runtime_error("dump topk row " + std::to_string(t) +
                                   ": multi-pool divergence");
      }
      if (flipped > sparse / 4 + 1)
        throw std::runtime_error("dump topk: " + std::to_string(flipped) +
                                 "/" + std::to_string(sparse) +
                                 " sparse rows flipped");
      std::printf("[ OK ] %-28s flipped=%lld/%lld sparse rows\n",
                  ("dump topk" + tag).c_str(), (long long)flipped,
                  (long long)sparse);
    }
    return 0;
  } catch (const std::exception& e) {
    std::printf("[FAIL] dsa dump parity: %s\n", e.what());
    return 1;
  }
}
