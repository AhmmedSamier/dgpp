// The chunked GDN prefill (src/kernels/gdn_chunk.cu) against the sequential
// recurrence (gdn_recurrent_fwd): output and final state relative l2, and speed.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/gdn_chunk.hpp"
#include "kernels/kda.hpp"

using namespace dgpp;

static uint16_t bf16(float f) {
  const __nv_bfloat16 b = __float2bfloat16_rn(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
static float fbf(uint16_t u) {
  uint32_t x = static_cast<uint32_t>(u) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
template <typename T>
static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d, h.size() * sizeof(T)));
  DGPP_CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

static int run(int tokens, int heads, int kv_ratio, bool time_it) {
  const int K = 128, V = 128, hk = heads / kv_ratio;
  std::mt19937 rng(7 + tokens);
  std::normal_distribution<float> nd(0.f, 1.f);
  const size_t qs = static_cast<size_t>(2 * hk * K + heads * V);
  std::vector<uint16_t> qkv(static_cast<size_t>(tokens) * qs), a(static_cast<size_t>(tokens) * heads),
      b(static_cast<size_t>(tokens) * heads);
  for (auto& x : qkv) x = bf16(nd(rng));
  for (auto& x : a) x = bf16(nd(rng));
  for (auto& x : b) x = bf16(nd(rng));
  std::vector<float> alog(heads), dtb(heads), s0(static_cast<size_t>(heads) * V * K);
  std::uniform_real_distribution<float> u(0.f, 1.f);
  for (int h = 0; h < heads; ++h) {
    alog[h] = std::log(1.f + 15.f * u(rng));
    dtb[h] = nd(rng) * 0.5f;
  }
  for (auto& x : s0) x = nd(rng) * 0.1f;
  const float scale = 1.f / std::sqrt(static_cast<float>(K));
  auto* dq = dev(qkv);
  auto* da = dev(a);
  auto* db = dev(b);
  auto* dal = dev(alog);
  auto* ddt = dev(dtb);
  auto* st_ref = dev(s0);
  auto* st_chk = dev(s0);
  uint16_t *o_ref = nullptr, *o_chk = nullptr;
  const size_t on = static_cast<size_t>(tokens) * heads * V;
  DGPP_CUDA_OK(cudaMalloc(&o_ref, on * 2));
  DGPP_CUDA_OK(cudaMalloc(&o_chk, on * 2));
  gdn_recurrent_fwd(dq, da, heads, db, heads, dal, ddt, st_ref, o_ref, tokens, heads, kv_ratio, K, V, scale,
                    nullptr);
  gdn_chunked_fwd(dq, da, heads, db, heads, dal, ddt, st_chk, o_chk, tokens, heads, kv_ratio, K, V, scale, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> hr(on), hc(on);
  std::vector<float> sr(s0.size()), sc(s0.size());
  DGPP_CUDA_OK(cudaMemcpy(hr.data(), o_ref, on * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(hc.data(), o_chk, on * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(sr.data(), st_ref, sr.size() * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(sc.data(), st_chk, sc.size() * 4, cudaMemcpyDeviceToHost));
  double on2 = 0, od2 = 0, sn2 = 0, sd2 = 0;
  for (size_t i = 0; i < on; ++i) {
    const double r = fbf(hr[i]), c = fbf(hc[i]);
    on2 += r * r;
    od2 += (r - c) * (r - c);
  }
  for (size_t i = 0; i < sr.size(); ++i) {
    sn2 += double(sr[i]) * sr[i];
    sd2 += (double(sr[i]) - sc[i]) * (double(sr[i]) - sc[i]);
  }
  const double eo = std::sqrt(od2 / on2), es = std::sqrt(sd2 / sn2);
  std::printf("[ .. ] tokens %d heads %d kv_ratio %d: out rel l2 %.4g, state rel l2 %.4g\n", tokens, heads, kv_ratio,
              eo, es);
  int fails = 0;
  if (!(eo < 2e-2) || !(es < 2e-2)) {
    std::printf("[FAIL] chunked GDN differs from the recurrence beyond bf16 tolerance\n");
    ++fails;
  }
  if (time_it) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto t = [&](auto f) {
      f();
      cudaEventRecord(e0);
      for (int i = 0; i < 10; ++i) f();
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      return ms / 10;
    };
    const float tr = t([&] {
      gdn_recurrent_fwd(dq, da, heads, db, heads, dal, ddt, st_ref, o_ref, tokens, heads, kv_ratio, K, V, scale,
                        nullptr);
    });
    const float tc = t([&] {
      gdn_chunked_fwd(dq, da, heads, db, heads, dal, ddt, st_chk, o_chk, tokens, heads, kv_ratio, K, V, scale,
                      nullptr);
    });
    std::printf("[ .. ] %d tokens x %d heads: recurrent %.3f ms, chunked %.3f ms (%.2fx)\n", tokens, heads, tr, tc,
                tr / tc);
  }
  return fails;
}

int main() {
  int fails = 0;
  fails += run(64, 3, 3, false);
  fails += run(300, 6, 3, false);   // a ragged tail chunk
  fails += run(1000, 6, 3, false);
  fails += run(8192, 24, 3, true);  // one rank's heads at a prefill chunk
  std::printf(fails == 0 ? "[ OK ] gdn_chunk_test\n" : "[FAIL] gdn_chunk_test\n");
  return fails == 0 ? 0 : 1;
}
