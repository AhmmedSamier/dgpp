// The GLM-4.7 attention kernels against the host reference on
// small geometry: the qkv finish (bias, the two-rounding head norm, the
// interleaved RoPE, the paged K/V append) within two bf16 ulps of the
// reference and bitwise across a second run; the split-KV paged GQA
// attention (one and several splits, positions across block boundaries,
// padding rows) within the DSA/QSA split tolerance class of the reference
// over the visible rows; decode rows of many requests bitwise the same rows
// run one request at a time.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/qsa.hpp"
#include "models/glm4/attn_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int D = dgpp::kGlm4HeadDim;

struct Geo {
  int local_heads = 24, kv_heads = 2, rotary = 64;
  int block_tokens = 32, blocks_per_request = 12, max_requests = 3;
  double theta = 1e6;
  float eps = 1e-5f;
  int slots() const { return block_tokens * blocks_per_request; }
  int width() const { return kv_heads * D; }
};

template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
template <class T>
const T* ptr(const DevBuf& b) { return static_cast<const T*>(b.p); }
template <class T>
T* mptr(DevBuf& b) { return static_cast<T*>(b.p); }

// Every request owns a distinct physical block range: request q's block b
// is physical q * blocks_per_request + b (the pool's tables in miniature).
std::vector<int32_t> tables(const Geo& g) {
  std::vector<int32_t> t(static_cast<size_t>(g.max_requests) * g.blocks_per_request);
  for (int q = 0; q < g.max_requests; ++q)
    for (int b = 0; b < g.blocks_per_request; ++b)
      t[static_cast<size_t>(q) * g.blocks_per_request + b] = q * g.blocks_per_request + b;
  return t;
}
int64_t phys(const Geo& g, int req, int64_t pos) {
  return static_cast<int64_t>(req) * g.blocks_per_request * g.block_tokens + pos;
}

std::vector<float> inv_freq_host(const Geo& g) {
  std::vector<float> f(static_cast<size_t>(g.rotary / 2));
  dgpp::qsa_rope_inv_freq(g.theta, g.rotary, f.data());
  return f;
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

}  // namespace

DGPP_TEST(glm4_qkv_finish_matches_the_reference) {
  Geo g;
  cudaStream_t st = test_stream();
  const int rows = 9;
  // Rows of three requests at scattered positions (one padding row).
  const std::vector<int32_t> req_ids = {0, 0, 1, 2, 2, 1, 0, 2, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 383, -1, 5};
  const int qw = g.local_heads * D, kw = g.kv_heads * D;
  const std::vector<float> qd = random_f32_uniform(11, static_cast<int64_t>(rows) * qw, 2.0f);
  const std::vector<float> kd = random_f32_uniform(12, static_cast<int64_t>(rows) * kw, 2.0f);
  const std::vector<float> vd = random_f32_uniform(13, static_cast<int64_t>(rows) * kw, 2.0f);
  const std::vector<uint16_t> qb = random_bf16_normal(14, qw, 0.1f), kb = random_bf16_normal(15, kw, 0.1f),
                              vb = random_bf16_normal(16, kw, 0.1f);
  std::vector<uint16_t> qn(D), kn(D);
  for (int d = 0; d < D; ++d) {
    qn[static_cast<size_t>(d)] = fast_bf16_rne(0.8f + 0.4f * uniform_01(17, d, 1.0f));
    kn[static_cast<size_t>(d)] = fast_bf16_rne(0.8f + 0.4f * uniform_01(18, d, 1.0f));
  }
  const std::vector<float> inv = inv_freq_host(g);
  const std::vector<int32_t> tbl = tables(g);
  DevBuf dqd = up(qd), dkd = up(kd), dvd = up(vd), dqb = up(qb), dkb = up(kb), dvb = up(vb), dqn = up(qn),
         dkn = up(kn), dinv = up(inv), dreq = up(req_ids), dpos = up(pos), dtbl = up(tbl);
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  DevBuf kc(cache_bytes), vc(cache_bytes), qo(static_cast<size_t>(rows) * qw * 2);
  DGPP_CUDA_OK(cudaMemset(kc.p, 0x7F, cache_bytes));
  DGPP_CUDA_OK(cudaMemset(vc.p, 0x7F, cache_bytes));
  DGPP_CUDA_OK(cudaMemset(qo.p, 0x7F, static_cast<size_t>(rows) * qw * 2));
  auto run = [&] {
    dgpp::glm4_qkv_finish(ptr<float>(dqd), qw, ptr<float>(dkd), kw, ptr<float>(dvd), kw, ptr<uint16_t>(dqb),
                          ptr<uint16_t>(dkb), ptr<uint16_t>(dvb), ptr<uint16_t>(dqn), ptr<uint16_t>(dkn), g.eps,
                          ptr<float>(dinv), g.rotary, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, g.local_heads,
                          g.kv_heads, ptr<int32_t>(dtbl), g.blocks_per_request, g.block_tokens, mptr<uint16_t>(qo),
                          qw, mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
  };
  run();
  const std::vector<uint16_t> got_q = down<uint16_t>(qo, static_cast<size_t>(rows) * qw);
  const std::vector<uint16_t> got_k = down<uint16_t>(kc, cache_bytes / 2), got_v = down<uint16_t>(vc, cache_bytes / 2);
  int max_ulps = 0;
  long checked = 0;
  std::vector<uint16_t> ref(D);
  for (int r = 0; r < rows; ++r) {
    const int64_t p = pos[static_cast<size_t>(r)];
    for (int h = 0; h < g.local_heads; ++h) {
      const uint16_t* got = got_q.data() + static_cast<size_t>(r) * qw + h * D;
      if (p < 0) {
        for (int d = 0; d < D; ++d) require(got[d] == 0x7F7F, "a padding row writes no q");
        continue;
      }
      dgpp::glm4_ref::qkv_finish(qd.data() + static_cast<size_t>(r) * qw + h * D, qb.data() + h * D, qn.data(), g.eps,
                                 inv.data(), g.rotary, p, ref.data(), D);
      for (int d = 0; d < D; ++d) {
        max_ulps = std::max(max_ulps, bf16_ulps(got[d], ref[static_cast<size_t>(d)]));
        ++checked;
      }
    }
    if (p < 0) continue;
    const int64_t slot = phys(g, req_ids[static_cast<size_t>(r)], p);
    for (int h = 0; h < g.kv_heads; ++h) {
      dgpp::glm4_ref::qkv_finish(kd.data() + static_cast<size_t>(r) * kw + h * D, kb.data() + h * D, kn.data(), g.eps,
                                 inv.data(), g.rotary, p, ref.data(), D);
      const uint16_t* gk = got_k.data() + slot * g.width() + h * D;
      for (int d = 0; d < D; ++d) max_ulps = std::max(max_ulps, bf16_ulps(gk[d], ref[static_cast<size_t>(d)]));
      dgpp::glm4_ref::qkv_finish(vd.data() + static_cast<size_t>(r) * kw + h * D, vb.data() + h * D, nullptr, g.eps,
                                 nullptr, 0, p, ref.data(), D);
      const uint16_t* gv = got_v.data() + slot * g.width() + h * D;
      for (int d = 0; d < D; ++d) require(gv[d] == ref[static_cast<size_t>(d)], "v: bf16(dot + bias) bitwise");
      checked += 2 * D;
    }
  }
  std::printf("[ .. ] qkv finish: %ld elements, max %d bf16 ulps\n", checked, max_ulps);
  require(max_ulps <= 2, "qkv finish within two bf16 ulps of the reference");
  // Untouched slots keep their poison; a second run is bitwise the first.
  for (size_t i = 0; i < got_k.size(); ++i) {
    const int64_t slot = static_cast<int64_t>(i) / g.width();
    bool written = false;
    for (int r = 0; r < rows; ++r)
      if (pos[static_cast<size_t>(r)] >= 0 && phys(g, req_ids[static_cast<size_t>(r)], pos[static_cast<size_t>(r)]) == slot)
        written = true;
    if (!written) require(got_k[i] == 0x7F7F && got_v[i] == 0x7F7F, "untouched cache slots keep their poison");
  }
  run();
  require(down<uint16_t>(qo, got_q.size()) == got_q && down<uint16_t>(kc, got_k.size()) == got_k,
          "second run bitwise");
}

DGPP_TEST(glm4_attention_matches_the_reference_dense_and_split) {
  Geo g;
  cudaStream_t st = test_stream();
  // A request of 300 tokens: K/V rows into the cache (request 1's blocks).
  const int seq = 300, req = 1;
  const std::vector<uint16_t> k = random_bf16_normal(31, static_cast<int64_t>(seq) * g.width(), 1.0f);
  const std::vector<uint16_t> v = random_bf16_normal(32, static_cast<int64_t>(seq) * g.width(), 1.0f);
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0x7F7F), vh(cache_bytes / 2, 0x7F7F);
  for (int t = 0; t < seq; ++t) {
    std::copy(k.begin() + static_cast<size_t>(t) * g.width(), k.begin() + static_cast<size_t>(t + 1) * g.width(),
              kh.begin() + phys(g, req, t) * g.width());
    std::copy(v.begin() + static_cast<size_t>(t) * g.width(), v.begin() + static_cast<size_t>(t + 1) * g.width(),
              vh.begin() + phys(g, req, t) * g.width());
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  // Query rows at positions across tile and block boundaries, one padding row.
  const std::vector<int64_t> pos = {0, 1, 31, 32, 33, 63, 64, 100, 255, 256, 299, -1};
  const int rows = static_cast<int>(pos.size());
  const std::vector<int32_t> req_ids(static_cast<size_t>(rows), req);
  const int qw = g.local_heads * D;
  const std::vector<uint16_t> q = random_bf16_normal(33, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (int n_split : {1, 3, 16}) {
    // The reference per row over the visible K/V rows, modelling the
    // kernel's tile-and-split chain (docs/glm47_plan.md D4: the online
    // softmax rounds each tile's probabilities against the running max).
    std::vector<uint16_t> ref(static_cast<size_t>(rows) * qw, 0);
    for (int r = 0; r < rows; ++r) {
      const int64_t p = pos[static_cast<size_t>(r)];
      if (p < 0) continue;
      std::vector<float> c;
      dgpp::glm4_ref::attention(q.data() + static_cast<size_t>(r) * qw, k.data(), v.data(), static_cast<int>(p + 1),
                                g.local_heads, g.kv_heads, D, scale, c, dgpp::kGlm4AttnTile, n_split);
      for (int i = 0; i < qw; ++i) ref[static_cast<size_t>(r) * qw + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
    }
    const size_t part = static_cast<size_t>(rows) * n_split * g.local_heads;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), out(ref.size() * 2);
    DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 2));
    dgpp::glm4_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                            ptr<int64_t>(dpos), rows, n_split, g.local_heads, g.kv_heads, g.block_tokens,
                            ptr<int32_t>(dtbl), g.blocks_per_request, scale, mptr<float>(m_ws), mptr<float>(l_ws),
                            mptr<float>(c_ws), st);
    dgpp::glm4_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, g.local_heads,
                            mptr<uint16_t>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> got = down<uint16_t>(out, ref.size());
    std::vector<float> gf(got.size()), wf(ref.size());
    double rms = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      gf[i] = dgpp::bf16_bits_to_float(got[i]);
      wf[i] = dgpp::bf16_bits_to_float(ref[i]);
      rms += static_cast<double>(wf[i]) * wf[i];
    }
    rms = std::sqrt(rms / static_cast<double>(ref.size()));
    // The reference reproduces the chain's structure; what remains is the
    // fp32 order of the score sums and of l, and the device's expf: two
    // bf16 ulps with a 0.5 % of RMS absolute floor for the cancelled
    // elements, under 1 % of the elements outside it.
    const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                    0.005 * rms);
    std::printf("[ .. ] attention n_split=%d: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n", n_split,
                s.max_abs, s.l2_rel, s.mismatches, s.n, rms);
    require_bf16("attention n_split=" + std::to_string(n_split), s, 2e-3, 0.01);
    // The padding row is zeros.
    for (int i = 0; i < qw; ++i) require(got[static_cast<size_t>(rows - 1) * qw + i] == 0, "padding row is zero");
  }
}

DGPP_TEST(glm4_attention_batched_rows_are_bitwise_single_request_rows) {
  Geo g;
  cudaStream_t st = test_stream();
  // Three requests with different lengths, K/V rows in each one's blocks.
  const int lens[3] = {70, 5, 200};
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * g.width() * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0), vh(cache_bytes / 2, 0);
  for (int q = 0; q < 3; ++q) {
    const std::vector<uint16_t> k = random_bf16_normal(40 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(50 + q, static_cast<int64_t>(lens[q]) * g.width(), 1.0f);
    for (int t = 0; t < lens[q]; ++t) {
      std::copy(k.begin() + static_cast<size_t>(t) * g.width(), k.begin() + static_cast<size_t>(t + 1) * g.width(),
                kh.begin() + phys(g, q, t) * g.width());
      std::copy(v.begin() + static_cast<size_t>(t) * g.width(), v.begin() + static_cast<size_t>(t + 1) * g.width(),
                vh.begin() + phys(g, q, t) * g.width());
    }
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  const int qw = g.local_heads * D;
  // The batch: two rows of request 2, one of 0, one padding, one of 1.
  const std::vector<int32_t> req_ids = {2, 2, 0, 1, 1};
  const std::vector<int64_t> pos = {198, 199, 69, -1, 4};
  const int rows = 5;
  const std::vector<uint16_t> q = random_bf16_normal(60, static_cast<int64_t>(rows) * qw, 1.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  auto run = [&](const std::vector<int32_t>& ids, const std::vector<int64_t>& ps, const std::vector<uint16_t>& qq) {
    const int n = static_cast<int>(ps.size());
    DevBuf dq = up(qq), dpos = up(ps), dreq = up(ids);
    const int n_split = 4;
    const size_t part = static_cast<size_t>(n) * n_split * g.local_heads;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), out(static_cast<size_t>(n) * qw * 2);
    dgpp::glm4_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                            ptr<int64_t>(dpos), n, n_split, g.local_heads, g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl),
                            g.blocks_per_request, scale, mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st);
    dgpp::glm4_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), n, n_split, g.local_heads,
                            mptr<uint16_t>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    return down<uint16_t>(out, static_cast<size_t>(n) * qw);
  };
  const std::vector<uint16_t> batched = run(req_ids, pos, q);
  for (int r = 0; r < rows; ++r) {
    const std::vector<uint16_t> qr(q.begin() + static_cast<size_t>(r) * qw, q.begin() + static_cast<size_t>(r + 1) * qw);
    const std::vector<uint16_t> single = run({req_ids[static_cast<size_t>(r)]}, {pos[static_cast<size_t>(r)]}, qr);
    require(std::equal(single.begin(), single.end(), batched.begin() + static_cast<size_t>(r) * qw),
            "row " + std::to_string(r) + " of the batch bitwise its single-request run");
  }
  std::printf("[ OK ] batched decode rows bitwise their single-request runs\n");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
