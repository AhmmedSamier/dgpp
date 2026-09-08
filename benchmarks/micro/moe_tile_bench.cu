// The prefill tile kernels at the production expert shape (2026-09-08,
// docs/nvfp4_plan.md phase 5): the grouped tensor-core launch for the
// routed experts of one layer on one rank — gate (bf16 out) and down (f32
// out) — with every expert holding tokens*top_k/experts rows, as a
// 2,048-token prefill does (~57 rows), timed per launch for the fp8 tile
// kernel, the fp4 reference tile kernel and the pipelined fp4 kernel.
//
//   moe_tile_bench [--tokens 2048] [--top-k 8] [--experts 288] [--inter 512]
//                  [--hidden 4096] [--iters 20]
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"

using namespace dgpp;

namespace {
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  float unit() { return static_cast<float>(next() >> 40) / static_cast<float>(1ull << 24); }
};

template <typename T>
T* dev(size_t n) { T* p = nullptr; DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); return p; }

double time_ms(cudaStream_t stream, int iters, const std::function<void()>& f) {
  for (int i = 0; i < 3; ++i) f();
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  DGPP_CUDA_OK(cudaEventRecord(a, stream));
  for (int i = 0; i < iters; ++i) f();
  DGPP_CUDA_OK(cudaEventRecord(b, stream));
  DGPP_CUDA_OK(cudaEventSynchronize(b));
  float ms = 0.f;
  DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return static_cast<double>(ms) / iters;
}
}  // namespace

int main(int argc, char** argv) {
  int tokens = 2048, top_k = 8, E = 288, I = 512, H = 4096, iters = 20, ragged = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    const int v = std::atoi(argv[i + 1]);
    if (a == "--tokens") tokens = v;
    else if (a == "--ragged") ragged = v;
    else if (a == "--top-k") top_k = v;
    else if (a == "--experts") E = v;
    else if (a == "--inter") I = v;
    else if (a == "--hidden") H = v;
    else if (a == "--iters") iters = v;
  }
  const int rows_per_expert = (tokens * top_k + E - 1) / E;
  Rng rng(0x7E11E);
  // Segments: uniform (expert e owns R rows) or ragged (--ragged 1: the
  // rows drawn with an exponential-ish spread around R, as real routing
  // is: a few experts hold hundreds of rows, many hold a handful).
  std::vector<MoeSegment> h_segs(E);
  std::vector<int> lens(E, rows_per_expert);
  if (ragged) {
    double sum = 0;
    std::vector<double> w(E);
    for (int e = 0; e < E; ++e) { w[e] = -std::log(1e-3 + rng.unit()); sum += w[e]; }
    int total = 0;
    for (int e = 0; e < E; ++e) { lens[e] = std::max(1, static_cast<int>(w[e] / sum * tokens * top_k)); total += lens[e]; }
    lens[0] += std::max(0, tokens * top_k - total);
  }
  int max_rows = 0;
  size_t rows_total = 0;
  for (int e = 0; e < E; ++e) {
    h_segs[e] = MoeSegment{static_cast<int>(rows_total), lens[e], e};
    rows_total += lens[e];
    max_rows = std::max(max_rows, lens[e]);
  }
  std::printf("tile bench: tokens %d top_k %d experts %d inter %d hidden %d -> %s, mean %d rows/expert, "
              "max %d, %zu rows\n", tokens, top_k, E, I, H, ragged ? "ragged" : "uniform",
              rows_per_expert, max_rows, rows_total);
  MoeSegment* d_segs = dev<MoeSegment>(E);
  DGPP_CUDA_OK(cudaMemcpy(d_segs, h_segs.data(), E * sizeof(MoeSegment), cudaMemcpyHostToDevice));

  // Activations: hidden rows [rows_total, H] bf16 and the down's act [rows_total, I].
  std::vector<uint16_t> h_act(rows_total * H);
  for (auto& v : h_act) v = float_to_bf16_bits(rng.unit() - 0.5f);
  uint16_t* d_hidden = dev<uint16_t>(rows_total * H);
  DGPP_CUDA_OK(cudaMemcpy(d_hidden, h_act.data(), h_act.size() * 2, cudaMemcpyHostToDevice));
  std::vector<uint16_t> h_act2(rows_total * I);
  for (auto& v : h_act2) v = float_to_bf16_bits(rng.unit() - 0.5f);
  uint16_t* d_act = dev<uint16_t>(rows_total * I);
  DGPP_CUDA_OK(cudaMemcpy(d_act, h_act2.data(), h_act2.size() * 2, cudaMemcpyHostToDevice));

  // fp4 experts: gate/up [I, H], down [H, I]: payload, scales, global per matrix.
  const size_t codes_gu = static_cast<size_t>(I) * H, codes_dn = static_cast<size_t>(H) * I;
  const size_t fp4_bytes = (codes_gu / 2) * 2 + codes_dn / 2;      // per expert, payload
  const size_t fp4_scale_bytes = (codes_gu / 16) * 2 + codes_dn / 16;
  uint8_t* d_fp4 = dev<uint8_t>(static_cast<size_t>(E) * fp4_bytes);
  uint8_t* d_fp4s = dev<uint8_t>(static_cast<size_t>(E) * fp4_scale_bytes);
  float* d_glob = dev<float>(static_cast<size_t>(E) * 3);
  {
    std::vector<uint8_t> h(static_cast<size_t>(E) * fp4_bytes);
    for (auto& b : h) b = static_cast<uint8_t>(rng.next() & 0xFF);
    DGPP_CUDA_OK(cudaMemcpy(d_fp4, h.data(), h.size(), cudaMemcpyHostToDevice));
    std::vector<uint8_t> hs(static_cast<size_t>(E) * fp4_scale_bytes);
    for (auto& b : hs) b = float_to_fp8_e4m3_bits(0.02f + 0.05f * rng.unit());
    DGPP_CUDA_OK(cudaMemcpy(d_fp4s, hs.data(), hs.size(), cudaMemcpyHostToDevice));
    std::vector<float> hg(static_cast<size_t>(E) * 3, 1.0f);
    for (auto& g : hg) g = 0.5f + rng.unit();
    DGPP_CUDA_OK(cudaMemcpy(d_glob, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));
  }
  // fp8 experts: payload only + 128-block scales.
  const size_t fp8_bytes = codes_gu * 2 + codes_dn;
  uint8_t* d_fp8 = dev<uint8_t>(static_cast<size_t>(E) * fp8_bytes);
  const size_t sc_gu = static_cast<size_t>((I + 127) / 128) * ((H + 127) / 128);
  const size_t sc_dn = static_cast<size_t>((H + 127) / 128) * ((I + 127) / 128);
  float* d_fp8s = dev<float>(static_cast<size_t>(E) * (2 * sc_gu + sc_dn));
  {
    std::vector<uint8_t> h(static_cast<size_t>(E) * fp8_bytes);
    for (auto& b : h) b = static_cast<uint8_t>(rng.next() & 0x7F);
    DGPP_CUDA_OK(cudaMemcpy(d_fp8, h.data(), h.size(), cudaMemcpyHostToDevice));
    std::vector<float> hs(static_cast<size_t>(E) * (2 * sc_gu + sc_dn));
    for (auto& v : hs) v = 0.01f + 0.02f * rng.unit();
    DGPP_CUDA_OK(cudaMemcpy(d_fp8s, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
  }
  std::vector<MoeExpertView> h_v4(static_cast<size_t>(E) * 3), h_v8(static_cast<size_t>(E) * 3);
  for (int e = 0; e < E; ++e) {
    const uint8_t* p = d_fp4 + static_cast<size_t>(e) * fp4_bytes;
    const uint8_t* sc = d_fp4s + static_cast<size_t>(e) * fp4_scale_bytes;
    h_v4[e * 3 + 0] = MoeExpertView{p, nullptr, sc, d_glob + e * 3 + 0};
    h_v4[e * 3 + 1] = MoeExpertView{p + codes_gu / 2, nullptr, sc + codes_gu / 16, d_glob + e * 3 + 1};
    h_v4[e * 3 + 2] = MoeExpertView{p + codes_gu, nullptr, sc + codes_gu / 8, d_glob + e * 3 + 2};
    const uint8_t* q = d_fp8 + static_cast<size_t>(e) * fp8_bytes;
    const float* qs = d_fp8s + static_cast<size_t>(e) * (2 * sc_gu + sc_dn);
    h_v8[e * 3 + 0] = MoeExpertView{q, qs, nullptr, nullptr};
    h_v8[e * 3 + 1] = MoeExpertView{q + codes_gu, qs + sc_gu, nullptr, nullptr};
    h_v8[e * 3 + 2] = MoeExpertView{q + 2 * codes_gu, qs + 2 * sc_gu, nullptr, nullptr};
  }
  MoeExpertView* d_v4 = dev<MoeExpertView>(h_v4.size());
  MoeExpertView* d_v8 = dev<MoeExpertView>(h_v8.size());
  DGPP_CUDA_OK(cudaMemcpy(d_v4, h_v4.data(), h_v4.size() * sizeof(MoeExpertView), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_v8, h_v8.data(), h_v8.size() * sizeof(MoeExpertView), cudaMemcpyHostToDevice));

  uint16_t* d_gate = dev<uint16_t>(rows_total * I);
  uint16_t* d_gate2 = dev<uint16_t>(rows_total * I);
  float* d_down = dev<float>(rows_total * H);
  float* d_down2 = dev<float>(rows_total * H);
  cudaStream_t stream = nullptr;

  const double gb_gu4 = static_cast<double>(E) * codes_gu * 9 / 16 / 1e9;
  const double gb_dn4 = static_cast<double>(E) * codes_dn * 9 / 16 / 1e9;
  const double gb_gu8 = static_cast<double>(E) * codes_gu / 1e9;
  const double gb_dn8 = static_cast<double>(E) * codes_dn / 1e9;
  const double flop_gu = 2.0 * rows_total * I * H / 1e12;
  auto report = [&](const char* name, double ms, double gb) {
    std::printf("  %-34s %8.3f ms/launch  %7.1f GB/s weights  %6.1f TFLOP/s\n", name, ms,
                gb / ms * 1e3, flop_gu / ms * 1e3);
  };
  std::printf("gate [%zu x %d] x [%d x %d]^T:\n", rows_total, H, I, H);
  report("fp8 tile kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v8, 0, d_gate, I, I, H, stream);
  }), gb_gu8);
  report("fp8 tile kernel, z split 128", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_bf16(d_hidden, H, d_segs, E, max_rows, 128, d_v8, 0, d_gate, I, I, H, stream);
  }), gb_gu8);
  report("fp4 reference tile kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate, I, I, H, stream);
  }), gb_gu4);
  report("fp4 reference, 64-row tiles", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 1);
  }), gb_gu4);
  report("fp4 reference, BK 32, 4 blocks/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 2);
  }), gb_gu4);
  report("fp4 reference, BK 32, 2 blocks/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 3);
  }), gb_gu4);
  report("fp4 reference, BN 128, BK 32, 3/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 5);
  }), gb_gu4);
  report("fp4 reference, BN 128, BK 64, 2/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 6);
  }), gb_gu4);
  report("fp4 reference, BN 128, BK 32, 2/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 7);
  }), gb_gu4);
  report("fp4 reference, BK 64, 4 blocks/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 4);
  }), gb_gu4);
  report("  decomposition: no weight loads", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 8);
  }), gb_gu4);
  report("  decomposition: no activation loads", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 9);
  }), gb_gu4);
  report("  decomposition: neither (MMA floor)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 10);
  }), gb_gu4);
  report("  decomposition: no MMA", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 11);
  }), gb_gu4);
  report("fp4 three-stage cp.async kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 12);
  }), gb_gu4);
  {
    std::vector<uint16_t> a(rows_total * I), b(rows_total * I);
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_gate2, b.size() * 2, cudaMemcpyDeviceToHost));
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate, I, I, H, stream, nullptr, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_gate, a.size() * 2, cudaMemcpyDeviceToHost));
    std::printf("  gate: three-stage %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 2) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  report("fp4 ldmatrix kernel (variant 13)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 13);
  }), gb_gu4);
  {
    std::vector<uint16_t> a(rows_total * I), b(rows_total * I);
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_gate, a.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_gate2, b.size() * 2, cudaMemcpyDeviceToHost));
    std::printf("  gate: ldmatrix %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 2) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  report("fp4 two-stage 64x128x32 (retired, v14)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream, nullptr, 14);
  }), gb_gu4);
  report("fp4 production launcher", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_bf16(d_hidden, H, d_segs, E, max_rows, 0, d_v4, 0, d_gate2, I, I, H, stream);
  }), gb_gu4);
  {
    std::vector<uint16_t> a(rows_total * I), b(rows_total * I);
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_gate, a.size() * 2, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_gate2, b.size() * 2, cudaMemcpyDeviceToHost));
    std::printf("  gate: production launcher %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 2) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  std::printf("down [%zu x %d] x [%d x %d]^T (f32 out):\n", rows_total, I, H, I);
  report("fp8 tile kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_f32(d_act, I, d_segs, E, max_rows, 0, d_v8, 2, d_down, H, H, I, stream);
  }), gb_dn8);
  report("fp8 tile kernel, z split 128", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_f32(d_act, I, d_segs, E, max_rows, 128, d_v8, 2, d_down, H, H, I, stream);
  }), gb_dn8);
  report("fp4 reference tile kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down, H, H, I, stream);
  }), gb_dn4);
  report("fp4 reference, BK 32, 4 blocks/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 2);
  }), gb_dn4);
  report("fp4 reference, BK 64, 4 blocks/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 4);
  }), gb_dn4);
  report("  decomposition: no weight loads", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 8);
  }), gb_dn4);
  report("  decomposition: no activation loads", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 9);
  }), gb_dn4);
  report("  decomposition: neither (MMA floor)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 10);
  }), gb_dn4);
  report("  decomposition: no MMA", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 11);
  }), gb_dn4);
  report("fp4 three-stage cp.async kernel", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 12);
  }), gb_dn4);
  {
    std::vector<float> a(rows_total * H), b(rows_total * H);
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_down2, b.size() * 4, cudaMemcpyDeviceToHost));
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down, H, H, I, stream, nullptr, 0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_down, a.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("  down: three-stage %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 4) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  report("fp4 reference, BN 128, BK 64, 2/SM", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 6);
  }), gb_dn4);
  report("fp4 ldmatrix kernel (variant 13)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 13);
  }), gb_dn4);
  {
    std::vector<float> a(rows_total * H), b(rows_total * H);
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_down, a.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_down2, b.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("  down: ldmatrix %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 4) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  report("fp4 two-stage 64x128x32 (retired, v14)", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_ref_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream, nullptr, 14);
  }), gb_dn4);
  report("fp4 production launcher", time_ms(stream, iters, [&] {
    launch_moe_grouped_mma_fp4_f32(d_act, I, d_segs, E, max_rows, 0, d_v4, 2, d_down2, H, H, I, stream);
  }), gb_dn4);
  {
    std::vector<float> a(rows_total * H), b(rows_total * H);
    DGPP_CUDA_OK(cudaMemcpy(a.data(), d_down, a.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), d_down2, b.size() * 4, cudaMemcpyDeviceToHost));
    std::printf("  down: production launcher %s the reference\n",
                std::memcmp(a.data(), b.data(), a.size() * 4) == 0 ? "BITWISE" : "DIFFERS FROM");
  }
  return 0;
}
