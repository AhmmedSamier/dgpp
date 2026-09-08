// moe_slot_bench: the decode MoE chain (router, slot order, gate_up+swiglu,
// down, accumulate) at the per-rank production geometry — every expert
// sliced on its intermediate dim across TP=4 (inter 2048 -> 512 per rank),
// 288 experts, top-8, one shared expert — for the MTP shape (two rows) and
// the batched one (eight). Random fp8 weights in device memory (the
// resident placement). Prints microseconds per enqueue_decode; the step
// pays 42 of them (2026-09-06: gate_up 255 us + down 168 us per layer on
// rank 0's trace, the down at 224 GB/s against gate_up's 295).
//
// Usage: moe_slot_bench [--rows N] [--iters N] [--warmup N] [--inter N] [--format fp8|fp4]
//   --format fp4: the routed experts as NVFP4 triples (e2m1 pairs, e4m3
//   scales per 16, one global per matrix — docs/nvfp4_plan.md), the shared
//   expert FP8 as in the composed checkpoint; bytes accounted per format.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_layer.hpp"

namespace {
uint32_t hash32(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  return uint32_t(x);
}
uint16_t bf16_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); }

struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DevBuf(size_t n) : bytes(n) { DGPP_CUDA_OK(cudaMalloc(&p, n ? n : 16)); }
  ~DevBuf() { cudaFree(p); }
  template <typename T> T* as() { return static_cast<T*>(p); }
};
}  // namespace

int main(int argc, char** argv) {
  int rows = 2, iters = 100, warmup = 10, inter = 512;
  bool fp4 = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--inter") && i + 1 < argc) inter = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--format") && i + 1 < argc) fp4 = !std::strcmp(argv[++i], "fp4");
  }
  dgpp::GlmMoeConfig cfg;
  cfg.inter = inter;  // the per-rank slice
  const int E = cfg.n_experts, H = cfg.hidden, I = cfg.inter;
  // (E+1)*3 matrices: gate/up [I,H] + down [H,I] per expert, then shared.
  const int mats = (E + 1) * 3;
  std::vector<dgpp::GlmQuantMatrix> qm(static_cast<size_t>(mats));
  std::vector<dgpp::GlmFp4Matrix> fm(static_cast<size_t>(E) * 3);
  std::vector<DevBuf*> keep;
  size_t total_bytes = 0;
  DevBuf* dglob = nullptr;
  if (fp4) {
    dglob = new DevBuf(size_t(E) * 3 * 4);
    keep.push_back(dglob);
    std::vector<float> g(size_t(E) * 3);
    for (size_t i = 0; i < g.size(); ++i) g[i] = 0.5f + float(hash32(11 + i) & 0xFFFF) / 65536.f * 1.5f;
    DGPP_CUDA_OK(cudaMemcpy(dglob->p, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
  }
  for (int m = 0; m < mats; ++m) {
    const bool down = m % 3 == 2;
    const int64_t r = down ? H : I, c = down ? I : H;
    if (fp4 && m < E * 3) {
      const size_t pn = size_t(r) * c / 2, sn = size_t(r) * c / 16;
      std::vector<uint8_t> payload(pn), scales(sn);
      for (size_t i = 0; i < pn; ++i) payload[i] = uint8_t(hash32(uint64_t(m) * 2654435761ull + i) & 0xFF);
      for (size_t i = 0; i < sn; ++i) {
        // e4m3 in a sane range, never the NaN codes.
        uint8_t v = uint8_t(0x20 + (hash32(99 + uint64_t(m) * 7919 + i) & 0x1F));
        scales[i] = v;
      }
      DevBuf* dp = new DevBuf(pn);
      DevBuf* ds = new DevBuf(sn);
      DGPP_CUDA_OK(cudaMemcpy(dp->p, payload.data(), pn, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(ds->p, scales.data(), sn, cudaMemcpyHostToDevice));
      keep.push_back(dp); keep.push_back(ds);
      fm[size_t(m)] = dgpp::GlmFp4Matrix{dp->as<uint8_t>(), ds->as<uint8_t>(), dglob->as<float>() + m, r, c};
      total_bytes += pn + sn + 4;
      continue;
    }
    const size_t pn = size_t(r) * c, sn = size_t((r + 127) / 128) * ((c + 127) / 128);
    std::vector<uint8_t> payload(static_cast<size_t>(pn));
    for (size_t i = 0; i < pn; ++i) {
      uint8_t v = uint8_t(hash32(uint64_t(m) * 1315423911ull + i) & 0xFF);
      if ((v & 0x7Fu) == 0x7Fu) v ^= 1u;  // no NaN codes
      payload[i] = v;
    }
    std::vector<float> scales(static_cast<size_t>(sn));
    for (size_t i = 0; i < sn; ++i) scales[i] = 0.002f + float(hash32(77 + i) & 0xFFFF) / 65536.f * 0.004f;
    DevBuf* dp = new DevBuf(pn);
    DevBuf* ds = new DevBuf(sn * 4);
    DGPP_CUDA_OK(cudaMemcpy(dp->p, payload.data(), pn, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(ds->p, scales.data(), sn * 4, cudaMemcpyHostToDevice));
    keep.push_back(dp); keep.push_back(ds);
    qm[size_t(m)] = dgpp::GlmQuantMatrix{dp->as<uint8_t>(), ds->as<float>(), r, c};
    total_bytes += pn + sn * 4;
  }
  std::vector<uint16_t> gate(size_t(E) * H);
  for (size_t i = 0; i < gate.size(); ++i) gate[i] = bf16_of((float(hash32(3 + i) & 0xFFFF) / 65536.f - 0.5f) * 0.1f);
  std::vector<float> bias(static_cast<size_t>(E));
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = (float(hash32(5 + i) & 0xFFFF) / 65536.f - 0.5f) * 0.5f;
  DevBuf dgate(gate.size() * 2), dbias(bias.size() * 4);
  DGPP_CUDA_OK(cudaMemcpy(dgate.p, gate.data(), gate.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dbias.p, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice));
  dgpp::GlmMoeWeights w;
  w.router_gate = dgate.as<uint16_t>();
  w.router_bias = dbias.as<float>();
  w.experts = fp4 ? nullptr : qm.data();
  w.experts_fp4 = fp4 ? fm.data() : nullptr;
  for (int m = 0; m < 3; ++m) w.shared[m] = qm[size_t(E) * 3 + m];

  dgpp::GlmMoeLayer layer(w, cfg, rows, /*decode_slots=*/rows);
  std::vector<uint16_t> hidden(size_t(rows) * H);
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = bf16_of((float(hash32(9 + i) & 0xFFFF) / 65536.f - 0.5f) * 2.f);
  DevBuf dh(hidden.size() * 2), dout(hidden.size() * 2);
  DGPP_CUDA_OK(cudaMemcpy(dh.p, hidden.data(), hidden.size() * 2, cudaMemcpyHostToDevice));
  cudaStream_t s;
  DGPP_CUDA_OK(cudaStreamCreate(&s));
  auto launch = [&] { layer.enqueue_decode(dh.as<uint16_t>(), dout.as<uint16_t>(), rows, nullptr, s); };
  for (int i = 0; i < warmup; ++i) launch();
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  float best = 1e30f, total = 0.f;
  for (int i = 0; i < iters; ++i) {
    DGPP_CUDA_OK(cudaEventRecord(e0, s));
    launch();
    DGPP_CUDA_OK(cudaEventRecord(e1, s));
    DGPP_CUDA_OK(cudaEventSynchronize(e1));
    float ms = 0.f;
    DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
    total += ms;
    if (ms < best) best = ms;
  }
  // Bytes the chain must move for `rows` rows: top_k routed + 1 shared
  // expert per row (no sharing assumed), 3 matrices each.
  const double per_expert_fp8 = 3.0 * double(H) * I + 3.0 * 4 * ((I + 127) / 128) * ((H + 127) / 128);
  const double per_expert_fp4 = 3.0 * double(H) * I / 2 + 3.0 * double(H) * I / 16 + 12;
  const double bytes = double(rows) * (cfg.top_k * (fp4 ? per_expert_fp4 : per_expert_fp8) + per_expert_fp8);
  std::printf("moe_slot_bench [%s]: rows=%d E=%d H=%d I(per rank)=%d top_k=%d weights %.1f GiB resident\n",
              fp4 ? "nvfp4 routed + fp8 shared" : "fp8", rows, E, H, I, cfg.top_k,
              total_bytes / (1024.0 * 1024 * 1024));
  std::printf("enqueue_decode: mean %.1f us, min %.1f us; %.1f GB/s at %.0f MB per call (no expert sharing); x42 layers = %.2f ms\n",
              total / iters * 1e3f, best * 1e3f, bytes / (best * 1e-3) / 1e9, bytes / 1e6,
              total / iters * 42);
  for (DevBuf* b : keep) delete b;
  return 0;
}
