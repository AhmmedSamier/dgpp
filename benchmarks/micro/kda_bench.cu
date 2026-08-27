// KDA M2 benchmark: batch-size-1 state-update traffic (PLAN M2 exit
// criterion "state update traffic is profiled at batch size 1") plus the
// layer-wide decode and prefill-chunk timings the M4+ roofline will need.
//
// Reports effective bytes/s against the 230 GB/s weight-stream planning
// floor from docs/measurements.md. This is a diagnostic probe, not a
// production runtime component; run it on an otherwise idle node.
//
// Usage: kda_bench [--iters N] [--warmup N]
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
#include "kernels/gemm.hpp"
#include "kernels/kda.hpp"
#include "models/kda_geometry.hpp"
#include "models/kda_layer.hpp"
#include "models/kda_state.hpp"

namespace {

using dgpp::KdaConfig;
using dgpp::KdaGeometry;
using dgpp::KdaLayer;
using dgpp::KdaStatePool;

uint32_t hash_u32_pair(uint64_t seed, uint64_t idx) {
  uint64_t x = seed * 0x9E3779B97F4A7C15ULL ^ idx;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<uint32_t>(x ^ (x >> 32));
}

// bf16 weights straight from hash bits (values in ±[2^lo, 2^hi)): the bench
// needs distinct, sane-scale weights fast, not statistical quality.
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

// ---------------------------------------------------------------------------
// State-traffic benchmark: the recurrent kernel alone, decode (T=1), across
// all 34 KDA layers' state slots.
// ---------------------------------------------------------------------------

void bench_state_traffic(int warmup, int iters, int heads, const char* label) {
  KdaConfig cfg;
  cfg.heads = heads;
  cfg.tp_size = 1;  // `heads` already encodes the rank's share
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));

  // Per-layer state blocks (fp32 recurrent + bf16 conv, full slot width),
  // 256-byte aligned so vectorized state access stays aligned.
  const size_t layer_state =
      (g.recurrent_bytes + g.conv_slot_bytes + 255) / 256 * 256;
  DevBuf states(static_cast<size_t>(cfg.num_kda_layers) * layer_state);
  DGPP_CUDA_OK(cudaMemsetAsync(states.p, 0, states.bytes, s));

  const int64_t qkv_elems = 2 * heads * cfg.head_dim + heads * cfg.head_dim;
  DevBuf qkv(qkv_elems * 2), g1(int64_t(heads) * cfg.head_dim * 2),
      beta(heads * 2), alog(heads * 4), dtb(int64_t(heads) * cfg.head_dim * 4),
      out(int64_t(heads) * cfg.head_dim * 2);
  qkv.upload(random_bf16_bits(1, qkv_elems, 0, 1).data(), qkv_elems * 2);
  g1.upload(random_bf16_bits(2, int64_t(heads) * cfg.head_dim, 0, 1).data(),
            int64_t(heads) * cfg.head_dim * 2);
  beta.upload(random_bf16_bits(3, heads, 0, 1).data(), heads * 2);
  std::vector<float> zeros_f(std::max<size_t>(dtb.bytes / 4, 1), 0.f);
  alog.upload(zeros_f.data(), heads * 4);
  dtb.upload(zeros_f.data(), dtb.bytes);
  const float scale =
      static_cast<float>(std::pow(static_cast<double>(cfg.head_dim), -0.5));

  float* state_base = states.as<float>();
  auto step = [&] {
    for (int layer = 0; layer < cfg.num_kda_layers; ++layer) {
      float* layer_state_ptr = reinterpret_cast<float*>(
          reinterpret_cast<uint8_t*>(state_base) +
          static_cast<size_t>(layer) * layer_state);
      dgpp::kda_recurrent_fwd(
          qkv.p, g1.p, beta.p, heads, alog.as<float>(), dtb.as<float>(),
          layer_state_ptr, out.p, 1, heads, cfg.head_dim, cfg.head_dim,
          cfg.lower_bound, scale, s);
    }
  };
  for (int i = 0; i < warmup; ++i) step();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const Timing t = time_loop(iters, step, s);

  // Traffic per step: state read + write (fp32 recurrent dominates; the
  // conv slot is written by the layer path, not this kernel, but the pool
  // accounts it — report the recurrent bytes the kernel actually moves).
  const double bytes =
      static_cast<double>(cfg.num_kda_layers) * g.local_heads *
      cfg.head_dim * cfg.head_dim * 4 * 2;
  const double gbps = bytes / (t.mean_ms * 1e-3) / 1e9;
  std::printf(
      "%-28s mean=%8.3f ms min=%8.3f max=%8.3f  state=%6.1f MiB  eff=%7.1f "
      "GB/s\n",
      label, t.mean_ms, t.min_ms, t.max_ms, bytes / (1024.0 * 1024.0), gbps);
  DGPP_CUDA_OK(cudaStreamDestroy(s));
}

// ---------------------------------------------------------------------------
// Full-layer decode / prefill benchmark (TP=1, real geometry, 1 layer).
// ---------------------------------------------------------------------------

void bench_layer(int warmup, int iters, const KdaConfig& cfg, int tokens,
                 const char* label) {
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));

  const int64_t n_in_proj = int64_t(g.in_proj_cols) * cfg.hidden;
  std::vector<uint16_t> in_proj = random_bf16_bits(11, n_in_proj, -6, -3);
  std::vector<uint16_t> f_b =
      random_bf16_bits(12, int64_t(g.local_proj) * cfg.head_dim, -3, 0);
  std::vector<uint16_t> g_b =
      random_bf16_bits(13, int64_t(g.local_proj) * cfg.head_dim, -3, 0);
  std::vector<uint16_t> conv =
      random_bf16_bits(14, int64_t(g.conv_channels) * cfg.conv_width, -3, 0);
  std::vector<uint16_t> o_norm = random_bf16_bits(15, cfg.head_dim, -2, 0);
  std::vector<uint16_t> o_proj =
      random_bf16_bits(16, int64_t(cfg.hidden) * g.local_proj, -6, -3);
  std::vector<float> a_log(g.local_heads, 0.f), dt_bias(g.local_proj, 0.f);
  std::vector<uint16_t> hidden_in =
      random_bf16_bits(17, int64_t(tokens) * cfg.hidden, -2, 0);

  DevBuf d_in_proj(in_proj.size() * 2), d_f_b(f_b.size() * 2),
      d_g_b(g_b.size() * 2), d_conv(conv.size() * 2), d_o_norm(o_norm.size() * 2),
      d_o_proj(o_proj.size() * 2), d_a_log(a_log.size() * 4),
      d_dt_bias(dt_bias.size() * 4), d_in(hidden_in.size() * 2),
      d_out(int64_t(tokens) * cfg.hidden * 2);
  d_in_proj.upload(in_proj.data(), in_proj.size() * 2);
  d_f_b.upload(f_b.data(), f_b.size() * 2);
  d_g_b.upload(g_b.data(), g_b.size() * 2);
  d_conv.upload(conv.data(), conv.size() * 2);
  d_o_norm.upload(o_norm.data(), o_norm.size() * 2);
  d_o_proj.upload(o_proj.data(), o_proj.size() * 2);
  d_a_log.upload(a_log.data(), a_log.size() * 4);
  d_dt_bias.upload(dt_bias.data(), dt_bias.size() * 4);
  d_in.upload(hidden_in.data(), hidden_in.size() * 2);

  dgpp::Arena arena;
  dgpp::Arena::Config ac;
  ac.persistent_hot = KdaLayer::persistent_hot_bytes(cfg, tokens);
  arena.init(ac);
  dgpp::CublasLtGemm gemm;
  DevBuf ws(64ull << 20);
  dgpp::KdaLayerWeights w;
  w.in_proj = d_in_proj.p;
  w.f_b = d_f_b.p;
  w.g_b = d_g_b.p;
  w.conv = d_conv.p;
  w.a_log = d_a_log.as<float>();
  w.dt_bias = d_dt_bias.as<float>();
  w.o_norm = d_o_norm.p;
  w.o_proj = d_o_proj.p;
  KdaLayer layer(arena, gemm, w, cfg, tokens, ws.p, ws.bytes);
  if (!layer.prepare(tokens))
    throw std::runtime_error("gemm plans unavailable");

  dgpp::Arena pool_arena;
  dgpp::Arena::Config pc;
  pc.persistent_hot = g.slot_bytes;
  pool_arena.init(pc);
  KdaStatePool pool;
  pool.init(pool_arena, cfg, 1);
  pool.zero_all(s);

  auto step = [&] {
    layer.enqueue(d_in.p, pool.recurrent(0, 0), pool.conv(0, 0),
                  g.conv_state_width, d_out.p, tokens, s);
  };
  for (int i = 0; i < warmup; ++i) step();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const Timing t = time_loop(iters, step, s);
  std::printf("%-28s mean=%8.3f ms min=%8.3f max=%8.3f  (x34 layers ~= %6.1f ms/%s)\n",
              label, t.mean_ms, t.min_ms, t.max_ms, t.mean_ms * 34,
              tokens == 1 ? "token" : "chunk");
  DGPP_CUDA_OK(cudaStreamDestroy(s));
}

}  // namespace

int main(int argc, char** argv) {
  int iters = 50, warmup = 5;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    std::printf("no CUDA device; nothing to do\n");
    return 2;
  }
  std::printf("KDA M2 benchmark (iters=%d warmup=%d); planning floor 230 GB/s\n\n",
              iters, warmup);

  // State traffic: the recurrent kernel's unavoidable fp32 state read+write,
  // the floor any decode optimization must beat.
  std::printf("== batch-1 decode state traffic (34 layers, recurrent kernel only) ==\n");
  bench_state_traffic(warmup, iters, 64, "TP=1 (64 heads)");
  bench_state_traffic(warmup, iters, 16, "TP=4 rank (16 heads)");

  // Full layer (projections + conv + recurrence + norm): one layer's cost,
  // extrapolated to the 34 KDA layers of the real model.
  std::printf("\n== full KDA layer, real geometry, TP=1 ==\n");
  bench_layer(warmup, iters, KdaConfig{}, 1, "decode (T=1)");
  bench_layer(2, 5, KdaConfig{}, 2048, "prefill chunk (T=2048)");
  return 0;
}
