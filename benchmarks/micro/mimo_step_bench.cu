// mimo_step_bench: one MiMo-V2.6-Flash decode layer's kernel chain at the
// per-rank production geometry (world 4: 16 query heads, 1 global / 2
// sliding-window kv heads, the experts sliced to 512, the fused projection
// chunk of 3392 / 3712 rows), replayed on random weights in device memory
// with every launch timed on its own and the whole layer timed as one
// CUDA graph — the per-kernel microseconds against each kernel's byte
// floor at the line rate, which is where a decode step's time goes once the
// folds are subtracted (docs/mimo_v26_flash_plan.md §7, 2026-09-22).
//
// Usage: mimo_step_bench [--rows N] [--ctx N] [--world W] [--iters N] [--warmup N]
//                        [--kind ga|swa|both] [--splits N] [--rate GBPS] [--cold] [--attn-chain]
//   --rows: the decode rows of the launch (1: T=1; 2: an MTP pass; 8: the
//   four-slot batch); --ctx: the request's position (the K/V rows the
//   attention reads); --world: the TP world the slices are cut for.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/mimo_attn.hpp"
#include "kernels/qsa.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_layer.hpp"

namespace {

uint32_t hash32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return uint32_t(x);
}
uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); }
float unit(uint64_t seed) { return float(hash32(seed) & 0xFFFF) / 65536.f; }

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DevBuf(size_t n) : bytes(n) { DGPP_CUDA_OK(cudaMalloc(&p, n ? n : 16)); }
  ~DevBuf() { cudaFree(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  template <typename T> T* as() { return static_cast<T*>(p); }
  template <typename T> const T* as() const { return static_cast<const T*>(p); }
};

// Random bf16 [n] at scale `s`.
DevBuf* bf16_buf(size_t n, float s, uint64_t seed, std::vector<DevBuf*>& keep) {
  std::vector<uint16_t> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = bf16_of((unit(seed + i) - 0.5f) * 2.f * s);
  DevBuf* b = new DevBuf(n * 2);
  DGPP_CUDA_OK(cudaMemcpy(b->p, h.data(), n * 2, cudaMemcpyHostToDevice));
  keep.push_back(b);
  return b;
}
// A random fp8 [rows, cols] matrix on 128 x 128 blocks (no NaN codes).
dgpp::GlmQuantMatrix fp8_matrix(int64_t rows, int64_t cols, uint64_t seed, std::vector<DevBuf*>& keep) {
  const size_t pn = size_t(rows) * cols, sn = size_t((rows + 127) / 128) * ((cols + 127) / 128);
  std::vector<uint8_t> payload(pn);
  for (size_t i = 0; i < pn; ++i) {
    uint8_t v = uint8_t(hash32(seed * 1315423911ull + i) & 0xFF);
    if ((v & 0x7Fu) == 0x7Fu) v ^= 1u;
    payload[i] = v;
  }
  std::vector<float> scales(sn);
  for (size_t i = 0; i < sn; ++i) scales[i] = 4e-4f + unit(seed + 77 + i) * 2e-4f;
  DevBuf* dp = new DevBuf(pn);
  DevBuf* ds = new DevBuf(sn * 4);
  DGPP_CUDA_OK(cudaMemcpy(dp->p, payload.data(), pn, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(ds->p, scales.data(), sn * 4, cudaMemcpyHostToDevice));
  keep.push_back(dp); keep.push_back(ds);
  return dgpp::GlmQuantMatrix{dp->as<uint8_t>(), ds->as<float>(), rows, cols};
}
// A random MXFP4 [rows, cols] matrix (e2m1 pairs, e8m0 scales 2^-7 .. 2^-5).
dgpp::GlmFp4Matrix mxfp4_matrix(int64_t rows, int64_t cols, uint64_t seed, std::vector<DevBuf*>& keep) {
  const size_t pn = size_t(rows) * cols / 2, sn = size_t(rows) * cols / 32;
  std::vector<uint8_t> payload(pn), scales(sn);
  for (size_t i = 0; i < pn; ++i) payload[i] = uint8_t(hash32(seed * 2654435761ull + i) & 0xFF);
  for (size_t i = 0; i < sn; ++i) scales[i] = uint8_t(120 + (hash32(seed * 7919 + i) % 3));
  DevBuf* dp = new DevBuf(pn);
  DevBuf* ds = new DevBuf(sn);
  DGPP_CUDA_OK(cudaMemcpy(dp->p, payload.data(), pn, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(ds->p, scales.data(), sn, cudaMemcpyHostToDevice));
  keep.push_back(dp); keep.push_back(ds);
  dgpp::GlmFp4Matrix m;
  m.payload = dp->as<uint8_t>();
  m.scales = ds->as<uint8_t>();
  m.global_scale = nullptr;
  m.rows = rows;
  m.cols = cols;
  m.scale_group = dgpp::kMxfp4Group;
  return m;
}

struct Timed {
  std::string name;
  std::function<void(cudaStream_t)> fn;
  double bytes;  // the launch's DRAM floor bytes
  float us = 0;
};

// --cold: an L2 flush (a read sweep of a 96 MB buffer, four times the 24
// MB L2) on the stream before every timed launch, so the launch reads its
// weights from DRAM the way the step's cold kernels do (the MoE experts,
// whose routing the prefetcher cannot know). Without it a repeated launch
// of the same kernel finds part of its weights in L2 — the 2026-09-22 nsys
// profile of the fabric step read the MXFP4 slot kernels at 182-202 GB/s
// where the warm loop here said 242. The sweep only READS (the store below
// never fires): a read-modify-write sweep leaves the L2 full of dirty
// lines whose write-backs then halve the timed launch's read bandwidth,
// which no step kernel sees.
__global__ void l2_flush_kernel(uint4* __restrict__ buf, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint4 v = buf[i];
  if (v.x == 0x7fffffffu && v.y == 0x7ffffffeu) buf[i].z = v.w + 1;
}

struct L2Flush {
  uint4* buf = nullptr;
  size_t n = 0;
  explicit L2Flush(bool enabled) {
    if (!enabled) return;
    n = (96u << 20) / sizeof(uint4);
    DGPP_CUDA_OK(cudaMalloc(&buf, n * sizeof(uint4)));
    DGPP_CUDA_OK(cudaMemset(buf, 0, n * sizeof(uint4)));
  }
  void operator()(cudaStream_t s) const {
    if (buf == nullptr) return;
    l2_flush_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(buf, n);
    DGPP_CUDA_OK(cudaGetLastError());
  }
};

// Every launch on its own: warm-ups, then the min over `iters` of the
// event-timed launch (the per-kernel floor; warm L2 unless --cold: the
// step's own L2 prefetcher warms the projections the same way, the experts
// it cannot).
void time_each(std::vector<Timed>& ks, cudaStream_t s, int warmup, int iters, const L2Flush& flush) {
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  for (Timed& k : ks) {
    for (int i = 0; i < warmup; ++i) k.fn(s);
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    float best = 1e30f;
    for (int i = 0; i < iters; ++i) {
      flush(s);
      DGPP_CUDA_OK(cudaEventRecord(e0, s));
      k.fn(s);
      DGPP_CUDA_OK(cudaEventRecord(e1, s));
      DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0.f;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      best = std::min(best, ms);
    }
    k.us = best * 1e3f;
  }
  DGPP_CUDA_OK(cudaEventDestroy(e0));
  DGPP_CUDA_OK(cudaEventDestroy(e1));
}

// The chain as one CUDA graph: the layer's own launch overhead structure.
float time_graph(std::vector<Timed>& ks, cudaStream_t s, int warmup, int iters, const L2Flush& flush) {
  cudaGraph_t g = nullptr;
  cudaGraphExec_t x = nullptr;
  DGPP_CUDA_OK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (Timed& k : ks) k.fn(s);
  DGPP_CUDA_OK(cudaStreamEndCapture(s, &g));
  DGPP_CUDA_OK(cudaGraphInstantiate(&x, g, 0));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  for (int i = 0; i < warmup; ++i) DGPP_CUDA_OK(cudaGraphLaunch(x, s));
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  float best = 1e30f;
  for (int i = 0; i < iters; ++i) {
    flush(s);
    DGPP_CUDA_OK(cudaEventRecord(e0, s));
    DGPP_CUDA_OK(cudaGraphLaunch(x, s));
    DGPP_CUDA_OK(cudaEventRecord(e1, s));
    DGPP_CUDA_OK(cudaEventSynchronize(e1));
    float ms = 0.f;
    DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, ms);
  }
  size_t nodes = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(g, nullptr, &nodes));
  std::printf("  graph of the layer: %zu nodes\n", nodes);
  DGPP_CUDA_OK(cudaGraphExecDestroy(x));
  DGPP_CUDA_OK(cudaGraphDestroy(g));
  return best * 1e3f;
}

}  // namespace

int main(int argc, char** argv) {
  int rows = 1, ctx = 300, world = 4, iters = 50, warmup = 10, splits = 0;
  bool cold = false, fused_attn = true;
  double rate = 248.0;  // GB/s: the measured cold streaming rate of this box
  std::string kind = "both";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--world") && i + 1 < argc) world = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--splits") && i + 1 < argc) splits = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--rate") && i + 1 < argc) rate = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--kind") && i + 1 < argc) kind = argv[++i];
    else if (!std::strcmp(argv[i], "--cold")) cold = true;
    else if (!std::strcmp(argv[i], "--attn-chain")) fused_attn = false;
  }
  const L2Flush flush(cold);
  // The release's geometry.
  const int H = 4096, E = 256, TOPK = 8, DK = dgpp::kMimoQkDim, DV = dgpp::kMimoVDim;
  const int heads = 64, ga_kv = 4, swa_kv = 8, chunks_total = 4, window = 128;
  const int I = 2048 / world;
  const int lh = heads / world, chunks = chunks_total / world;
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  std::vector<DevBuf*> keep;
  dgpp::set_bf16_residency(dgpp::Bf16Residency::Bf12);
  dgpp::CublasLtGemm gemm;
  gemm.set_decode_rows(std::min(rows, dgpp::dense_gemv_rows()));
  gemm.set_bf12_wide(true);
  dgpp::Bf12Companions bf12;
  const size_t ws_bytes = size_t(64) << 20;
  DevBuf gemm_ws(ws_bytes);
  const int block_tokens = 64;
  const int64_t slots = ((int64_t(ctx) + rows + block_tokens - 1) / block_tokens + 1) * block_tokens;
  const int blocks_per_request = int(slots / block_tokens);
  std::vector<int32_t> table(static_cast<size_t>(blocks_per_request));
  for (int i = 0; i < blocks_per_request; ++i) table[size_t(i)] = i;
  DevBuf dtable(table.size() * 4);
  DGPP_CUDA_OK(cudaMemcpy(dtable.p, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
  std::vector<int32_t> req(size_t(rows), 0);
  std::vector<int64_t> pos(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) pos[size_t(r)] = ctx + r;
  DevBuf dreq(req.size() * 4), dpos(pos.size() * 8);
  DGPP_CUDA_OK(cudaMemcpy(dreq.p, req.data(), req.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dpos.p, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice));
  std::vector<float> inv(32u);
  dgpp::qsa_rope_inv_freq(1e7, 64, inv.data());
  DevBuf dinv(32 * 4);
  DGPP_CUDA_OK(cudaMemcpy(dinv.p, inv.data(), 128, cudaMemcpyHostToDevice));
  // Activations.
  DevBuf* resid = bf16_buf(size_t(rows) * H, 1.0f, 1, keep);
  DevBuf x(size_t(rows) * H * 2), y(size_t(rows) * H * 2), acc(size_t(rows) * H * 4);
  DevBuf* in_norm = bf16_buf(H, 0.1f, 2, keep);
  DevBuf* post_norm = bf16_buf(H, 0.1f, 3, keep);
  // The MoE (every kind but layer 0 runs it): the router and the sliced
  // MXFP4 experts.
  dgpp::GlmMoeConfig mcfg;
  mcfg.hidden = H;
  mcfg.inter = I;
  mcfg.n_experts = E;
  mcfg.top_k = TOPK;
  mcfg.n_shared_experts = 0;
  mcfg.routed_scaling_factor = 1.0f;
  mcfg.norm_topk_prob = true;
  mcfg.swiglu_limit = 1e30f;
  mcfg.router_mode = dgpp::MoeRouterMode::SigmoidBias;
  std::vector<dgpp::GlmFp4Matrix> experts(size_t(E) * 3);
  for (int e = 0; e < E; ++e) {
    experts[size_t(e) * 3 + 0] = mxfp4_matrix(I, H, 1000 + e * 3, keep);
    experts[size_t(e) * 3 + 1] = mxfp4_matrix(I, H, 1001 + e * 3, keep);
    experts[size_t(e) * 3 + 2] = mxfp4_matrix(H, I, 1002 + e * 3, keep);
  }
  DevBuf* router = bf16_buf(size_t(E) * H, 0.05f, 4, keep);
  std::vector<float> bias(static_cast<size_t>(E));
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = (unit(5 + i) - 0.5f) * 0.05f;
  DevBuf dbias(bias.size() * 4);
  DGPP_CUDA_OK(cudaMemcpy(dbias.p, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
  dgpp::GlmMoeWeights mw;
  mw.router_gate = router->as<uint16_t>();
  mw.router_bias = dbias.as<float>();
  mw.experts_fp4 = experts.data();
  dgpp::GlmMoeLayer moe(mw, mcfg, rows, rows, 1);
  moe.prepare_graph_table(0, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  const double expert_bytes = double(TOPK) * 3.0 * (double(H) * I / 2 + double(H) * I / 32);

  std::printf("mimo_step_bench: world %d, rows %d, ctx %d, per rank: %d query heads, experts sliced to %d, line rate %.0f GB/s\n",
              world, rows, ctx, lh, I, rate);
  for (const std::string k : {"ga", "swa"}) {
    if (kind != "both" && kind != k) continue;
    const bool swa = k == "swa";
    const int lkv = (swa ? swa_kv : ga_kv) / world;
    const int kv_per_chunk = (swa ? swa_kv : ga_kv) / chunks_total;
    const int64_t chunk_rows = int64_t(heads / chunks_total) * DK + int64_t(kv_per_chunk) * (DK + DV);
    const int64_t stride = ((chunk_rows + 127) / 128) * 128;
    const int64_t n_qkv = int64_t(chunks - 1) * stride + chunk_rows;
    const int64_t qkv_cols = int64_t(chunks) * stride;
    dgpp::GlmQuantMatrix qkv = fp8_matrix(n_qkv, H, swa ? 21 : 20, keep);
    DevBuf qkv_out(size_t(rows) * qkv_cols * 4);
    DevBuf q(size_t(rows) * lh * DK * 2), o(size_t(rows) * lh * DV * 2);
    DevBuf kc(size_t(slots) * lkv * DK * 2), vc(size_t(slots) * lkv * DV * 2);
    // The cache's history: random rows the finish will not overwrite.
    {
      std::vector<uint16_t> hk(size_t(slots) * lkv * DK), hv(size_t(slots) * lkv * DV);
      for (size_t i = 0; i < hk.size(); ++i) hk[i] = bf16_of((unit(31 + i) - 0.5f) * 2.f);
      for (size_t i = 0; i < hv.size(); ++i) hv[i] = bf16_of((unit(32 + i) - 0.5f) * 2.f);
      DGPP_CUDA_OK(cudaMemcpy(kc.p, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(vc.p, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    }
    std::vector<float> sink_h(static_cast<size_t>(lh));
    for (size_t i = 0; i < sink_h.size(); ++i) sink_h[i] = (unit(41 + i) - 0.5f) * 4.f;
    DevBuf dsink(sink_h.size() * 4);
    DGPP_CUDA_OK(cudaMemcpy(dsink.p, sink_h.data(), sink_h.size() * 4, cudaMemcpyHostToDevice));
    const int n_split = splits > 0 ? splits : (swa ? std::min(32, window / 32) : 32);
    const size_t part = size_t(rows) * n_split * lh;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4);
    DevBuf* o_proj = bf16_buf(size_t(H) * lh * DV, 0.05f, swa ? 51 : 50, keep);
    if (!bf12.pack(o_proj->as<uint16_t>(), H, int64_t(lh) * DV, gemm, s)) std::printf("  (o_proj kept bf16)\n");
    DGPP_CUDA_OK(cudaStreamSynchronize(s));
    dgpp::MimoQkvLayout layout;
    layout.chunks = chunks;
    layout.chunk_stride = stride;
    layout.q_per_chunk = heads / chunks_total;
    layout.kv_per_chunk = kv_per_chunk;
    const int visible = swa ? std::min(ctx + 1, window) : ctx + 1;
    const double kv_bytes = double(visible) * lkv * (DK + DV) * 2;
    const float scale = 1.0f / std::sqrt(float(DK));
    std::vector<Timed> ks;
    ks.push_back({"add + input rmsnorm (fused)", [&](cudaStream_t st) {
      dgpp::add_rmsnorm_bf16(resid->as<uint16_t>(), y.as<uint16_t>(), in_norm->as<uint16_t>(), x.as<uint16_t>(), rows, H, 1e-6f, st); },
      double(rows) * H * 4});
    ks.push_back({"qkv fp8 mma (" + std::to_string(n_qkv) + " x 4096)", [&](cudaStream_t st) {
      dgpp::launch_scale_gemm_grid_f32(x.as<uint16_t>(), size_t(H), qkv.payload, qkv.scales, qkv_out.as<float>(), rows,
                                       int(n_qkv), H, st, size_t(qkv_cols), 7, 7, true); },
      double(n_qkv) * H + qkv.scale_bytes()});
    DevBuf counters(size_t(rows) * lkv * 4);
    DGPP_CUDA_OK(cudaMemset(counters.p, 0, counters.bytes));
    dgpp::MimoAttnFusedArgs fa;
    fa.qkv = qkv_out.as<float>();
    fa.qkv_stride = qkv_cols;
    fa.layout = layout;
    fa.inv_freq = dinv.as<float>();
    fa.value_scale = 0.707f;
    fa.req_ids = dreq.as<int32_t>();
    fa.pos = dpos.as<int64_t>();
    fa.rows = rows;
    fa.n_split = n_split;
    fa.local_heads = lh;
    fa.kv_heads = lkv;
    fa.block_tables = dtable.as<int32_t>();
    fa.blocks_per_request = blocks_per_request;
    fa.block_tokens = block_tokens;
    fa.window = swa ? window : 0;
    fa.scale = scale;
    fa.sink = swa ? dsink.as<float>() : nullptr;
    fa.k_cache = kc.as<uint16_t>();
    fa.v_cache = vc.as<uint16_t>();
    fa.m_ws = m_ws.as<float>();
    fa.l_ws = l_ws.as<float>();
    fa.c_ws = c_ws.as<float>();
    fa.counters = counters.as<int>();
    fa.out = o.as<uint16_t>();
    if (fused_attn) {
      ks.push_back({"attention fused (finish + " + std::to_string(n_split) + " splits + combine, " +
                    std::to_string(visible) + " rows)", [&](cudaStream_t st) { dgpp::mimo_attn_fused(fa, st); },
                    double(rows) * kv_bytes});
    } else {
    ks.push_back({"qkv finish (rope, v scale, append)", [&](cudaStream_t st) {
      dgpp::mimo_qkv_finish(qkv_out.as<float>(), qkv_cols, layout, dinv.as<float>(), 0.707f, dreq.as<int32_t>(),
                            dpos.as<int64_t>(), rows, lh, lkv, dtable.as<int32_t>(), blocks_per_request, block_tokens,
                            q.as<uint16_t>(), int64_t(lh) * DK, kc.as<uint16_t>(), vc.as<uint16_t>(), st); },
      double(rows) * (lh * DK + lkv * (DK + DV)) * 6});
    ks.push_back({"attention partial (" + std::to_string(n_split) + " splits, " + std::to_string(visible) + " rows)",
      [&](cudaStream_t st) {
      dgpp::mimo_attn_partial(q.as<uint16_t>(), int64_t(lh) * DK, kc.as<uint16_t>(), vc.as<uint16_t>(), dreq.as<int32_t>(),
                              dpos.as<int64_t>(), rows, n_split, lh, lkv, block_tokens, dtable.as<int32_t>(),
                              blocks_per_request, swa ? window : 0, scale, swa ? dsink.as<float>() : nullptr,
                              m_ws.as<float>(), l_ws.as<float>(), c_ws.as<float>(), st); },
      double(rows) * kv_bytes});
    ks.push_back({"attention combine", [&](cudaStream_t st) {
      dgpp::mimo_attn_combine(m_ws.as<float>(), l_ws.as<float>(), c_ws.as<float>(), rows, n_split, lh, o.as<uint16_t>(), st); },
      double(part) * (DV + 2) * 4});
    }
    ks.push_back({"o_proj bf12 gemv (4096 x " + std::to_string(lh * DV) + ")", [&](cudaStream_t st) {
      gemm.matmul(o.as<uint16_t>(), o_proj->as<uint16_t>(), y.as<uint16_t>(), rows, H, lh * DV, dgpp::DType::BF16,
                  dgpp::GemmOut::BF16, size_t(lh) * DV, gemm_ws.p, ws_bytes, st); },
      double(H) * lh * DV * 1.5});
    ks.push_back({"add + post rmsnorm (fused)", [&](cudaStream_t st) {
      dgpp::add_rmsnorm_bf16(resid->as<uint16_t>(), y.as<uint16_t>(), post_norm->as<uint16_t>(), x.as<uint16_t>(), rows, H, 1e-6f, st); },
      double(rows) * H * 4});
    ks.push_back({"moe decode f32 (router + 8 x 3 MXFP4 slots)", [&](cudaStream_t st) {
      moe.enqueue_decode_f32(x.as<uint16_t>(), acc.as<float>(), rows, nullptr, st, 0); },
      double(E) * H * 2 + double(rows) * expert_bytes});
    ks.push_back({"moe round", [&](cudaStream_t st) {
      dgpp::launch_moe_round_bf16(y.as<uint16_t>(), acc.as<float>(), int64_t(rows) * H, st); },
      double(rows) * H * 6});
    time_each(ks, s, warmup, iters, flush);
    std::printf("== %s layer (kv heads %d, hpk %d)%s\n", swa ? "sliding-window" : "global", lkv, lh / lkv,
                cold ? " [cold L2]" : "");
    double sum_us = 0, sum_floor = 0;
    for (const Timed& t : ks) {
      const double floor_us = t.bytes / rate / 1e3;
      sum_us += t.us;
      sum_floor += floor_us;
      std::printf("  %-46s %8.1f us   floor %7.1f us  (%5.1f MB, %5.0f GB/s)\n", t.name.c_str(), t.us, floor_us, t.bytes / 1e6,
                  t.us > 0 ? t.bytes / (t.us * 1e-6) / 1e9 : 0.0);
    }
    const float graph_us = time_graph(ks, s, warmup, iters, flush);
    std::printf("  %-46s %8.1f us   floor %7.1f us   (graph replay %.1f us)\n", "layer, launches summed", sum_us, sum_floor, graph_us);
    const int layers = swa ? 39 : 9;
    std::printf("  x %d such layers: %.2f ms summed, %.2f ms as graphs, %.2f ms floor\n", layers, sum_us * layers / 1e3,
                graph_us * layers / 1e3, sum_floor * layers / 1e3);
  }
  for (DevBuf* b : keep) delete b;
  return 0;
}
