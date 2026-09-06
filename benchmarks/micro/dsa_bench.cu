// DSA M3 benchmark: decode-path latency (single-stream goal, DESIGN §3),
// CUDA-graph decode replay, prefill chunk throughput, and the fused select
// kernel's index-cache streaming bandwidth at long context.
//
// Reports effective GB/s against the 230 GB/s planning floor from
// docs/measurements.md. Diagnostic probe, not a production component; run
// on an otherwise idle node.
//
// Usage: dsa_bench [--iters N] [--warmup N] [--ctx N]
//   --ctx: decode context length in tokens (default 65536; must be a
//          multiple of block_tokens).
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "core/arena.hpp"
#include "core/graph.hpp"
#include "kernels/dsa.hpp"
#include "kernels/gemm.hpp"
#include "models/dsa_geometry.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_state.hpp"


namespace {

using dgpp::DsaConfig;
using dgpp::DsaGeometry;
using dgpp::DsaLayer;
using dgpp::DsaLayerWeights;
using dgpp::DsaStatePool;

uint32_t hash_u32_pair(uint64_t seed, uint64_t idx) {
  uint64_t x = seed * 0x9E3779B97F4A7C15ULL ^ idx;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x ^ (x >> 32));
}

std::vector<uint16_t> random_bf16_bits(uint64_t seed, int64_t n, int lo_exp,
                                       int hi_exp) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  const uint32_t span = static_cast<uint32_t>(hi_exp - lo_exp);
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t h = hash_u32_pair(seed, i);
    const uint32_t e = static_cast<uint32_t>(lo_exp) + (h >> 8) % span;
    v[static_cast<size_t>(i)] =
        static_cast<uint16_t>((h & 0x8000u) | ((e + 127u) << 7) | (h & 0x7Fu));
  }
  return v;
}

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(size_t n) {
    if (n == 0) n = 16;
    DGPP_CUDA_OK(cudaMalloc(&p, n));
    bytes = n;
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
    o.p = nullptr;
    o.bytes = 0;
  }
  DevBuf& operator=(DevBuf&&) = delete;  // vector storage never reassigns
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  void upload(const void* host, size_t n) {
    DGPP_CUDA_OK(cudaMemcpy(p, host, n, cudaMemcpyHostToDevice));
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
};

double event_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

struct Timing {
  double mean_ms;
  double min_ms;
  double max_ms;
};

Timing time_loop(int iters, const std::function<void()>& fn, cudaStream_t s) {
  cudaEvent_t start, stop;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&stop));
  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    DGPP_CUDA_OK(cudaEventRecord(start, s));
    fn();
    DGPP_CUDA_OK(cudaEventRecord(stop, s));
    DGPP_CUDA_OK(cudaEventSynchronize(stop));
    samples.push_back(event_ms(start, stop));
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  double sum = 0, mn = samples[0], mx = samples[0];
  for (double v : samples) {
    sum += v;
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  return {sum / samples.size(), mn, mx};
}

// Random weight set + device uploads for one DSA layer (real geometry).
struct BenchWeights {
  DsaLayerWeights views;
  std::vector<DevBuf> dev;

  BenchWeights(const DsaConfig& cfg, uint64_t seed) {
    const DsaGeometry g = DsaGeometry::from_config(cfg);
    const int heads = cfg.index_n_heads;
    const int dim = cfg.index_head_dim;
    const auto up = [&](const std::vector<uint16_t>& v) -> const void* {
      dev.emplace_back(v.size() * 2);
      dev.back().upload(v.data(), v.size() * 2);
      return dev.back().p;
    };
    views.wq_b = up(random_bf16_bits(seed ^ 0xA1,
                                    int64_t(heads) * dim * cfg.q_lora_rank, -3, 0));
    views.wk = up(random_bf16_bits(seed ^ 0xB2, int64_t(dim) * cfg.hidden, -3, 0));
    views.wp = up(random_bf16_bits(seed ^ 0xC3, int64_t(heads) * cfg.hidden, -3, 0));
    views.gate = up(random_bf16_bits(seed ^ 0xD4, int64_t(dim) * cfg.hidden, -3, 0));
    views.k_norm_w = up(random_bf16_bits(seed ^ 0xE5, dim, -2, 0));
    views.k_norm_b = up(random_bf16_bits(seed ^ 0xF6, dim, -2, 0));
    views.qkv_a = up(random_bf16_bits(
        seed ^ 0x179,
        int64_t(cfg.q_lora_rank + cfg.kv_lora_rank) * cfg.hidden, -6, -3));
    views.q_aln = up(random_bf16_bits(seed ^ 0x28A, cfg.q_lora_rank, -2, 0));
    views.kv_aln = up(random_bf16_bits(seed ^ 0x39B, cfg.kv_lora_rank, -2, 0));
    views.q_b = up(random_bf16_bits(
        seed ^ 0x4AC, int64_t(g.local_q_rows) * cfg.q_lora_rank, -3, 0));
    views.kv_b = up(random_bf16_bits(
        seed ^ 0x5BD,
        int64_t(g.local_heads) * (cfg.qk_nope_head_dim + cfg.v_head_dim) *
            cfg.kv_lora_rank,
        -3, 0));
    views.o_proj = up(random_bf16_bits(
        seed ^ 0x6CE, int64_t(cfg.hidden) * g.local_v_rows, -6, -3));
    // fp32 APE [kpool, dim].
    std::vector<float> ape(size_t(cfg.index_kpool) * dim);
    for (size_t i = 0; i < ape.size(); ++i)
      ape[i] = (double(hash_u32_pair(seed ^ 0x7DF, i)) / 0xFFFFFFFFu - 0.5) * 0.2f;
    dev.emplace_back(ape.size() * 4);
    dev.back().upload(ape.data(), ape.size() * 4);
    views.ape = dev.back().as<float>();
  }
};

// ---------------------------------------------------------------------------
// Full-layer decode: eager vs CUDA-graph replay at context length ctx.
// ---------------------------------------------------------------------------

void bench_decode(int warmup, int iters, const DsaConfig& cfg, int64_t ctx,
                  bool use_graph, const char* label) {
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));

  BenchWeights bw(cfg, 31);
  const int max_tokens = 2048;
  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot =
      DsaStatePool::cache_bytes(cfg, 1, ctx) +
      DsaLayer::scratch_bytes(cfg, max_tokens, ctx);
  arena.init(ac);
  dgpp::CublasLtGemm gemm;
  DevBuf ws(64ull << 20);
  DsaStatePool pool;
  pool.init(arena, cfg, 1, ctx);
  const size_t sb = DsaLayer::scratch_bytes(cfg, max_tokens, ctx);
  DsaLayer layer(gemm, bw.views, cfg, max_tokens, ctx,
                 arena.alloc_persistent(dgpp::MemClass::DeviceHot, sb, 256),
                 sb, ws.p, ws.bytes);
  if (!layer.prepare(max_tokens) || !layer.prepare(1))
    throw std::runtime_error("gemm plans unavailable");

  // No prefill: decode timing only cares that the select STREAMS the full
  // visible index cache and attention gathers latent rows — zero content
  // moves the same bytes as real content (and prefilling 65k tokens just to
  // warm a bandwidth bench would measure setup, not decode).
  DevBuf din(size_t(ctx + 1) * cfg.hidden * 2), dout(cfg.hidden * 2);
  din.upload(random_bf16_bits(32, int64_t(ctx + 1) * cfg.hidden, -2, 0).data(),
             size_t(ctx + 1) * cfg.hidden * 2);
  if (!pool.ensure_request_blocks(0, ctx, s))
    throw std::runtime_error("pool exhaustion in bench setup");
  DGPP_CUDA_OK(cudaMemsetAsync(pool.index_k(0), 0,
                               size_t(ctx / cfg.block_tokens) *
                                   g.pools_per_block * cfg.index_head_dim, s));
  DGPP_CUDA_OK(cudaMemsetAsync(
      pool.index_scale(0), 0,
      size_t(ctx / cfg.block_tokens) * g.pools_per_block * 4, s));
  DGPP_CUDA_OK(cudaMemsetAsync(pool.latent(0), 0, size_t(ctx) *
                                                       g.latent_bytes_per_token,
                               s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  DevBuf dreq(4), dpos(8), dspans(8);
  int32_t req = 0;
  int64_t pos = ctx - 1;
  int32_t spans[2] = {0, 1};
  dreq.upload(&req, 4);
  dpos.upload(&pos, 8);
  dspans.upload(spans, 8);

  dgpp::GraphCache graphs;
  auto step = [&] {
    layer.enqueue_decode(
        static_cast<const uint16_t*>(din.p) + size_t(ctx) * cfg.hidden, pool, 0,
                         dreq.as<int32_t>(), dpos.as<int64_t>(),
                         dspans.as<int32_t>(), 1, 1, dout.p, s);
  };
  auto step_graph = [&] { graphs.launch(1, s); };
  for (int i = 0; i < warmup; ++i) step();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));

  Timing t;
  if (use_graph) {
    graphs.replay_or_capture(1, "dsa-decode-bench", s, [&](cudaStream_t cap) {
      layer.enqueue_decode(
        static_cast<const uint16_t*>(din.p) + size_t(ctx) * cfg.hidden, pool, 0,
                           dreq.as<int32_t>(), dpos.as<int64_t>(),
                           dspans.as<int32_t>(), 1, 1, dout.p, cap);
    });
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    for (int i = 0; i < warmup; ++i) step_graph();
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    t = time_loop(iters, step_graph, s);
  } else {
    t = time_loop(iters, step, s);
  }
  // Effective decode traffic per step (what a decode MUST move):
  //   weights: qkv_a + q_b + wq_b + wk + gate + wp + kv_b + o_proj (bf16)
  //   caches:  index K+scales streamed by select (visible pools) +
  //            selected latent rows (topk+tail) x2 (attn read) ... latent
  //            write of the new row is negligible.
  const double weight_bytes =
      (double(cfg.q_lora_rank + cfg.kv_lora_rank) * cfg.hidden +
       double(g.local_q_rows) * cfg.q_lora_rank +
       double(cfg.index_n_heads) * cfg.index_head_dim * cfg.q_lora_rank +
       double(cfg.index_head_dim) * cfg.hidden * 2 +
       double(cfg.index_n_heads) * cfg.hidden +
       double(g.local_heads) * (cfg.qk_nope_head_dim + cfg.v_head_dim) *
           cfg.kv_lora_rank +
       double(cfg.hidden) * g.local_v_rows) *
      2;
  const int64_t visible_pools = (ctx + 1) / cfg.index_kpool;
  const double index_bytes =
      double(visible_pools) * (cfg.index_head_dim + 4);
  const double latent_bytes = double(g.max_selected) * cfg.kv_lora_rank * 2;
  const double total = weight_bytes + index_bytes + latent_bytes;
  std::printf(
      "%-30s mean=%8.3f ms min=%8.3f max=%8.3f  eff=%6.1f GB/s  "
      "(x11 layers ~= %5.2f ms/token)\n",
      label, t.mean_ms, t.min_ms, t.max_ms, total / (t.mean_ms * 1e-3) / 1e9,
      t.mean_ms * cfg.num_dsa_layers);
  std::printf("%-30s   weights=%.0f MiB index=%.0f MiB latent=%.0f MiB\n", "",
              weight_bytes / (1024 * 1024), index_bytes / (1024 * 1024),
              latent_bytes / (1024 * 1024));
  DGPP_CUDA_OK(cudaStreamDestroy(s));
}

// ---------------------------------------------------------------------------
// Prefill chunk throughput at real geometry.
// ---------------------------------------------------------------------------

void bench_prefill(int warmup, int iters, const DsaConfig& cfg, int tokens,
                   const char* label) {
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  BenchWeights bw(cfg, 41);
  const int64_t capacity =
      (tokens + cfg.block_tokens - 1) / cfg.block_tokens * cfg.block_tokens;
  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot =
      DsaStatePool::cache_bytes(cfg, 1, capacity) +
      DsaLayer::scratch_bytes(cfg, tokens, capacity);
  arena.init(ac);
  dgpp::CublasLtGemm gemm;
  DevBuf ws(64ull << 20);
  DsaStatePool pool;
  pool.init(arena, cfg, 1, capacity);
  const size_t sb = DsaLayer::scratch_bytes(cfg, tokens, capacity);
  DsaLayer layer(gemm, bw.views, cfg, tokens, capacity,
                 arena.alloc_persistent(dgpp::MemClass::DeviceHot, sb, 256),
                 sb, ws.p, ws.bytes);
  if (!layer.prepare(tokens)) throw std::runtime_error("gemm plans unavailable");

  DevBuf din(size_t(tokens) * cfg.hidden * 2), dout(size_t(tokens) *
                                                     cfg.hidden * 2);
  din.upload(random_bf16_bits(42, int64_t(tokens) * cfg.hidden, -2, 0).data(),
             size_t(tokens) * cfg.hidden * 2);

  auto step = [&] { layer.enqueue_prefill(din.p, pool, 0, 0, 0, tokens, dout.p, s); };
  for (int i = 0; i < warmup; ++i) step();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const Timing t = time_loop(iters, step, s);
  std::printf("%-30s mean=%8.3f ms min=%8.3f max=%8.3f  %8.0f tok/s  "
              "(x11 layers ~= %5.1f ms/chunk)\n",
              label, t.mean_ms, t.min_ms, t.max_ms,
              tokens / (t.mean_ms * 1e-3), t.mean_ms * cfg.num_dsa_layers);
  DGPP_CUDA_OK(cudaStreamDestroy(s));
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 50, warmup = 5, tp = 1;
  int64_t ctx = 65536;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc)
      ctx = std::atoll(argv[++i]);
    else if (std::strcmp(argv[i], "--tp") == 0 && i + 1 < argc)
      tp = std::atoi(argv[++i]);
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::printf("no CUDA device; nothing to do\n");
    return 2;
  }
  if (ctx % DsaConfig{}.block_tokens != 0 || ctx <= 0) {
    std::printf("--ctx must be a positive multiple of block_tokens (128)\n");
    return 1;
  }
  std::printf(
      "DSA M3 benchmark (iters=%d warmup=%d ctx=%lld); planning floor 230 "
      "GB/s\n\n",
      iters, warmup, (long long)ctx);

  DsaConfig cfg{};  // real GLM-5.3-Flash geometry; --tp N for a rank's slice
  cfg.tp_size = tp;
  const bool decode_only =
      std::any_of(argv + 1, argv + argc, [](const char* a) {
        return std::strcmp(a, "--decode-only") == 0;
      });
  std::printf("== full DSA layer decode, real geometry, TP=%d ==\n", tp);
  bench_decode(warmup, iters, cfg, ctx, false, "decode eager");
  if (!decode_only) {
    bench_decode(warmup, iters, cfg, ctx, true, "decode graph replay");

    std::printf("\n== full DSA layer prefill, real geometry, TP=1 ==\n");
    bench_prefill(2, 5, cfg, 2048, "prefill chunk (T=2048)");
    bench_prefill(1, 3, cfg, 8192, "prefill chunk (T=8192)");
  }
  return 0;
}
