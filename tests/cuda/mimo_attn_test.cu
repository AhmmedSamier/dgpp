// The MiMo-V2.6-Flash attention kernels against the host reference on
// small geometry: the qkv finish over the fused projection's chunked
// fp32 output (the padded chunk stride, both rope tables, the value
// scale's second rounding, the paged K/V append) within two bf16 ulps of
// the reference for q/k (the device's cosf/sinf against the host's) and
// bitwise for v, padding rows and columns untouched, bitwise across a
// second run; the split-KV paged GQA attention with its window and sink
// (global and sliding windows, one to 32 splits — empty splits included —
// positions across tile and block boundaries, padding rows, one to 16
// query heads per kv head) within the DSA/QSA split tolerance class of the
// reference modelling the kernel's chain, and within a looser budget of
// the plain softmax; decode rows of several requests bitwise the same rows
// run one request at a time.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/mimo_attn.hpp"
#include "kernels/qsa.hpp"
#include "models/mimo/attn_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int DK = dgpp::kMimoQkDim;
constexpr int DV = dgpp::kMimoVDim;
constexpr int R = dgpp::kMimoRotaryDim;

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

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<float> inv_freq_host(double theta) {
  std::vector<float> f(static_cast<size_t>(R / 2));
  dgpp::qsa_rope_inv_freq(theta, R, f.data());
  return f;
}

// The paged cache in miniature: `max_requests` requests of
// `blocks_per_request` blocks each, the physical blocks a fixed
// permutation of the request-major order (so a slot's address is not its
// position).
struct Pool {
  int block_tokens = 64, blocks_per_request = 8, max_requests = 2;
  std::vector<int32_t> table;
  Pool() {
    const int n = max_requests * blocks_per_request;
    table.resize(static_cast<size_t>(n));
    std::iota(table.begin(), table.end(), 0);
    // A deterministic shuffle.
    uint64_t s = 0x1234567ull;
    for (int i = n - 1; i > 0; --i) {
      s ^= s << 13; s ^= s >> 7; s ^= s << 17;
      std::swap(table[static_cast<size_t>(i)], table[static_cast<size_t>(s % static_cast<uint64_t>(i + 1))]);
    }
  }
  int slots() const { return block_tokens * blocks_per_request * max_requests; }
  int64_t phys(int req, int64_t pos) const {
    const int32_t blk = table[static_cast<size_t>(req) * blocks_per_request + pos / block_tokens];
    return static_cast<int64_t>(blk) * block_tokens + pos % block_tokens;
  }
};

}  // namespace

DGPP_TEST(mimo_qkv_finish_matches_the_reference) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int rows = 9;
  // Rows of two requests at scattered positions (one padding row).
  const std::vector<int32_t> req_ids = {0, 0, 1, 1, 0, 1, 0, 1, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 383, -1, 5};
  DevBuf dreq = up(req_ids), dpos = up(pos), dtbl = up(pool.table);
  const std::vector<float> inv_ga = inv_freq_host(1e7), inv_swa = inv_freq_host(1e4);
  DevBuf dinv_ga = up(inv_ga), dinv_swa = up(inv_swa);
  long checked = 0;
  int max_ulps = 0;
  for (const int kvpc : {1, 2}) {
    for (const float vscale : {0.707f, 1.0f}) {
      for (const bool swa : {false, true}) {
        dgpp::MimoQkvLayout layout;
        layout.chunks = 2;
        layout.q_per_chunk = 2;
        layout.kv_per_chunk = kvpc;
        const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * DK + kvpc * (DK + DV);
        layout.chunk_stride = ((chunk_rows + 127) / 128) * 128;
        const int lh = layout.chunks * layout.q_per_chunk, lkv = layout.chunks * kvpc;
        const int64_t qkv_stride = layout.chunks * layout.chunk_stride;
        // The fused output: random values in the chunks' rows, NaN in the
        // padding columns (never read).
        std::vector<float> qkv = random_f32_uniform(11 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
        for (int r = 0; r < rows; ++r)
          for (int c = 0; c < layout.chunks; ++c)
            for (int64_t i = chunk_rows; i < layout.chunk_stride; ++i)
              qkv[static_cast<size_t>(r) * qkv_stride + c * layout.chunk_stride + i] = std::nanf("");
        DevBuf dqkv = up(qkv);
        const int qw = lh * DK, kw = lkv * DK, vw = lkv * DV;
        const size_t kbytes = static_cast<size_t>(pool.slots()) * kw * 2, vbytes = static_cast<size_t>(pool.slots()) * vw * 2;
        DevBuf kc(kbytes), vc(vbytes), qo(static_cast<size_t>(rows) * qw * 2);
        DGPP_CUDA_OK(cudaMemset(kc.p, 0x7F, kbytes));
        DGPP_CUDA_OK(cudaMemset(vc.p, 0x7F, vbytes));
        DGPP_CUDA_OK(cudaMemset(qo.p, 0x7F, static_cast<size_t>(rows) * qw * 2));
        auto run = [&] {
          dgpp::mimo_qkv_finish(ptr<float>(dqkv), qkv_stride, layout, swa ? ptr<float>(dinv_swa) : ptr<float>(dinv_ga),
                                vscale, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, lh, lkv, ptr<int32_t>(dtbl),
                                pool.blocks_per_request, pool.block_tokens, mptr<uint16_t>(qo), qw, mptr<uint16_t>(kc),
                                mptr<uint16_t>(vc), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
        };
        run();
        const std::vector<uint16_t> got_q = down<uint16_t>(qo, static_cast<size_t>(rows) * qw);
        const std::vector<uint16_t> got_k = down<uint16_t>(kc, kbytes / 2), got_v = down<uint16_t>(vc, vbytes / 2);
        const std::vector<float>& inv = swa ? inv_swa : inv_ga;
        std::vector<uint16_t> ref(static_cast<size_t>(DK));
        for (int r = 0; r < rows; ++r) {
          const int64_t p = pos[static_cast<size_t>(r)];
          const float* src_row = qkv.data() + static_cast<size_t>(r) * qkv_stride;
          for (int h = 0; h < lh; ++h) {
            const uint16_t* got = got_q.data() + static_cast<size_t>(r) * qw + h * DK;
            if (p < 0) {
              for (int d = 0; d < DK; ++d) require(got[d] == 0x7F7F, "a padding row writes no q");
              continue;
            }
            const int c = h / layout.q_per_chunk;
            const float* src = src_row + c * layout.chunk_stride + static_cast<int64_t>(h % layout.q_per_chunk) * DK;
            dgpp::mimo_ref::qk_finish(src, inv.data(), p, ref.data());
            for (int d = 0; d < DK; ++d) {
              max_ulps = std::max(max_ulps, bf16_ulps(got[d], ref[static_cast<size_t>(d)]));
              ++checked;
            }
          }
          if (p < 0) continue;
          const int64_t slot = pool.phys(req_ids[static_cast<size_t>(r)], p);
          for (int j = 0; j < lkv; ++j) {
            const int c = j / kvpc;
            const float* ksrc = src_row + c * layout.chunk_stride + static_cast<int64_t>(layout.q_per_chunk) * DK +
                                static_cast<int64_t>(j % kvpc) * DK;
            dgpp::mimo_ref::qk_finish(ksrc, inv.data(), p, ref.data());
            const uint16_t* gk = got_k.data() + slot * kw + j * DK;
            for (int d = 0; d < DK; ++d) max_ulps = std::max(max_ulps, bf16_ulps(gk[d], ref[static_cast<size_t>(d)]));
            const float* vsrc = ksrc - static_cast<int64_t>(j % kvpc) * DK + static_cast<int64_t>(kvpc) * DK +
                                static_cast<int64_t>(j % kvpc) * DV;
            dgpp::mimo_ref::v_finish(vsrc, vscale, ref.data());
            const uint16_t* gv = got_v.data() + slot * vw + j * DV;
            for (int d = 0; d < DV; ++d) require(gv[d] == ref[static_cast<size_t>(d)], "v: bf16(bf16(dot) x scale) bitwise");
            checked += DK + DV;
          }
        }
        // Untouched slots keep their poison; a second run is bitwise the first.
        std::vector<bool> written(static_cast<size_t>(pool.slots()), false);
        for (int r = 0; r < rows; ++r)
          if (pos[static_cast<size_t>(r)] >= 0)
            written[static_cast<size_t>(pool.phys(req_ids[static_cast<size_t>(r)], pos[static_cast<size_t>(r)]))] = true;
        for (int s = 0; s < pool.slots(); ++s) {
          if (written[static_cast<size_t>(s)]) continue;
          for (int i = 0; i < kw; ++i) require(got_k[static_cast<size_t>(s) * kw + i] == 0x7F7F, "untouched K slots keep their poison");
          for (int i = 0; i < vw; ++i) require(got_v[static_cast<size_t>(s) * vw + i] == 0x7F7F, "untouched V slots keep their poison");
        }
        run();
        require(down<uint16_t>(qo, got_q.size()) == got_q && down<uint16_t>(kc, got_k.size()) == got_k &&
                    down<uint16_t>(vc, got_v.size()) == got_v,
                "second run bitwise");
      }
    }
  }
  std::printf("[ .. ] qkv finish: %ld elements, max %d bf16 ulps\n", checked, max_ulps);
  require(max_ulps <= 2, "qkv finish within two bf16 ulps of the reference");
}

namespace {

struct Seq {
  int req = 1, len = 300;
  std::vector<uint16_t> k, v;  // [len, kv * DK] / [len, kv * DV]
};

// The request's rows into its blocks; the other request's slots poisoned.
void fill_cache(const Pool& pool, const Seq& seq, int kv, std::vector<uint16_t>& kh, std::vector<uint16_t>& vh) {
  const int kw = kv * DK, vw = kv * DV;
  kh.assign(static_cast<size_t>(pool.slots()) * kw, 0x7F7F);
  vh.assign(static_cast<size_t>(pool.slots()) * vw, 0x7F7F);
  for (int t = 0; t < seq.len; ++t) {
    const int64_t s = pool.phys(seq.req, t);
    std::copy(seq.k.begin() + static_cast<size_t>(t) * kw, seq.k.begin() + static_cast<size_t>(t + 1) * kw,
              kh.begin() + s * kw);
    std::copy(seq.v.begin() + static_cast<size_t>(t) * vw, seq.v.begin() + static_cast<size_t>(t + 1) * vw,
              vh.begin() + s * vw);
  }
}

}  // namespace

DGPP_TEST(mimo_attention_matches_the_reference_windows_sinks_and_splits) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 2;
  Seq seq;
  seq.k = random_bf16_normal(31, static_cast<int64_t>(seq.len) * kv * DK, 1.0f);
  seq.v = random_bf16_normal(32, static_cast<int64_t>(seq.len) * kv * DV, 1.0f);
  std::vector<uint16_t> kh, vh;
  fill_cache(pool, seq, kv, kh, vh);
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(pool.table);
  // Query rows at positions across tile and block boundaries, one padding row.
  const std::vector<int64_t> pos = {0, 5, 31, 32, 63, 64, 200, 299, -1};
  const int rows = static_cast<int>(pos.size());
  const std::vector<int32_t> req_ids(static_cast<size_t>(rows), seq.req);
  DevBuf dpos = up(pos), dreq = up(req_ids);
  const float scale = ref_scale(DK);
  for (const int hpk : {1, 2, 16}) {
    const int lh = hpk * kv, qw = lh * DK, ow = lh * DV;
    const std::vector<uint16_t> q = random_bf16_normal(33 + hpk, static_cast<int64_t>(rows) * qw, 1.0f);
    const std::vector<float> sink = random_f32_uniform(77, lh, 2.0f);
    DevBuf dq = up(q), dsink = up(sink);
    for (const int window : {0, 32, 128}) {
      for (const bool with_sink : {false, true}) {
        for (const int n_split : {1, 4, 32}) {
          const std::string tag = "hpk " + std::to_string(hpk) + " window " + std::to_string(window) + " sink " +
                                  std::to_string(with_sink) + " n_split " + std::to_string(n_split);
          // The reference per row over the visible K/V rows, modelling the
          // kernel's tile-and-split chain, and the plain softmax.
          std::vector<uint16_t> ref(static_cast<size_t>(rows) * ow, 0), plain(static_cast<size_t>(rows) * ow, 0);
          for (int r = 0; r < rows; ++r) {
            const int64_t p = pos[static_cast<size_t>(r)];
            if (p < 0) continue;
            const int64_t lo = window > 0 ? std::max<int64_t>(0, p - window + 1) : 0;
            const int n = static_cast<int>(p - lo + 1);
            const uint16_t* krows = seq.k.data() + static_cast<size_t>(lo) * kv * DK;
            const uint16_t* vrows = seq.v.data() + static_cast<size_t>(lo) * kv * DV;
            std::vector<float> c;
            dgpp::mimo_ref::attention(q.data() + static_cast<size_t>(r) * qw, krows, vrows, n, lh, kv, scale,
                                      with_sink ? sink.data() : nullptr, c, dgpp::kMimoAttnTile, n_split);
            for (int i = 0; i < ow; ++i) ref[static_cast<size_t>(r) * ow + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
            dgpp::mimo_ref::attention(q.data() + static_cast<size_t>(r) * qw, krows, vrows, n, lh, kv, scale,
                                      with_sink ? sink.data() : nullptr, c, 0, 1);
            for (int i = 0; i < ow; ++i) plain[static_cast<size_t>(r) * ow + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
          }
          const size_t part = static_cast<size_t>(rows) * n_split * lh;
          DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4), out(ref.size() * 2);
          DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 2));
          dgpp::mimo_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                                  ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                  pool.blocks_per_request, window, scale, with_sink ? ptr<float>(dsink) : nullptr,
                                  mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st);
          dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, lh,
                                  mptr<uint16_t>(out), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
          const std::vector<uint16_t> got = down<uint16_t>(out, ref.size());
          std::vector<float> gf(got.size()), wf(ref.size()), pf(plain.size());
          double rms = 0;
          for (size_t i = 0; i < got.size(); ++i) {
            gf[i] = dgpp::bf16_bits_to_float(got[i]);
            wf[i] = dgpp::bf16_bits_to_float(ref[i]);
            pf[i] = dgpp::bf16_bits_to_float(plain[i]);
            rms += static_cast<double>(wf[i]) * wf[i];
          }
          rms = std::sqrt(rms / static_cast<double>(ref.size()));
          // The reference reproduces the chain's structure; what remains is
          // the fp32 order of the score sums and of l, and the device's
          // expf: two bf16 ulps with a 0.5 % of RMS absolute floor for the
          // cancelled elements, under 1 % of the elements outside it.
          const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()), 2 * std::pow(2.0, -7.0),
                                          0.005 * rms);
          // The plain softmax: the tile chain's rounding of each tile's
          // probabilities against a running max instead of the final one —
          // a few bf16 ulps.
          const Stats sp = compare_abs_rel(gf.data(), pf.data(), static_cast<long>(got.size()), 6 * std::pow(2.0, -7.0),
                                           0.01 * rms);
          std::printf("[ .. ] attention %s: chain max_abs %.3g l2_rel %.3g mismatches %ld/%ld; plain l2_rel %.3g "
                      "mismatches %ld (rms %.3g)\n",
                      tag.c_str(), s.max_abs, s.l2_rel, s.mismatches, s.n, sp.l2_rel, sp.mismatches, rms);
          require_bf16("attention chain " + tag, s, 2e-3, 0.01);
          require_bf16("attention plain " + tag, sp, 1e-2, 0.02);
          // The padding row is zeros.
          for (int i = 0; i < ow; ++i) require(got[static_cast<size_t>(rows - 1) * ow + i] == 0, "padding row is zero");
        }
      }
    }
  }
}

DGPP_TEST(mimo_attention_sink_and_window_change_the_output) {
  // A sanity pin on the semantics themselves: with a window the row at
  // position 200 ignores the rows before 73; with a sink the outputs
  // shrink (the sink's mass takes from the value sum). Both against the
  // kernel's own global / sinkless run.
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 2, hpk = 2, lh = hpk * kv, qw = lh * DK, ow = lh * DV;
  Seq seq;
  seq.k = random_bf16_normal(41, static_cast<int64_t>(seq.len) * kv * DK, 1.0f);
  seq.v = random_bf16_normal(42, static_cast<int64_t>(seq.len) * kv * DV, 1.0f);
  std::vector<uint16_t> kh, vh;
  fill_cache(pool, seq, kv, kh, vh);
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(pool.table);
  const std::vector<int64_t> pos = {200};
  const std::vector<int32_t> req_ids = {seq.req};
  const std::vector<uint16_t> q = random_bf16_normal(43, qw, 1.0f);
  std::vector<float> sink(static_cast<size_t>(lh), 8.0f);  // a large sink: most of the mass
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids), dsink = up(sink);
  auto run = [&](int window, bool with_sink) {
    const int n_split = 4;
    const size_t part = static_cast<size_t>(n_split) * lh;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4), out(static_cast<size_t>(ow) * 2);
    dgpp::mimo_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                            ptr<int64_t>(dpos), 1, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                            pool.blocks_per_request, window, ref_scale(DK), with_sink ? ptr<float>(dsink) : nullptr,
                            mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st);
    dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), 1, n_split, lh, mptr<uint16_t>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    return down<uint16_t>(out, static_cast<size_t>(ow));
  };
  const std::vector<uint16_t> global = run(0, false), windowed = run(128, false), sunk = run(0, true);
  require(global != windowed, "the window changes a row that sees past it");
  double n_global = 0, n_sunk = 0;
  for (int i = 0; i < ow; ++i) {
    n_global += std::pow(dgpp::bf16_bits_to_float(global[static_cast<size_t>(i)]), 2);
    n_sunk += std::pow(dgpp::bf16_bits_to_float(sunk[static_cast<size_t>(i)]), 2);
  }
  require(n_sunk < 0.5 * n_global, "a large sink takes most of the softmax mass from the values");
  // Perturbing a row outside the window leaves the windowed output alone.
  std::vector<uint16_t> kh2 = kh;
  for (int i = 0; i < kv * DK; ++i) kh2[static_cast<size_t>(pool.phys(seq.req, 10)) * kv * DK + i] ^= 0x4000;
  kc = up(kh2);
  require(run(128, false) == windowed, "a row outside the window is invisible");
  require(run(0, false) != global, "... and visible to the global row");
  std::printf("[ OK ] the window and the sink act on the softmax as the reference says\n");
}

DGPP_TEST(mimo_attention_batched_rows_are_bitwise_single_request_rows) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 2, hpk = 8, lh = hpk * kv, qw = lh * DK, ow = lh * DV;
  // Two requests with different lengths, K/V rows in each one's blocks.
  const int lens[2] = {70, 200};
  const int kw = kv * DK, vw = kv * DV;
  std::vector<uint16_t> kh(static_cast<size_t>(pool.slots()) * kw, 0), vh(static_cast<size_t>(pool.slots()) * vw, 0);
  for (int r = 0; r < 2; ++r) {
    const std::vector<uint16_t> k = random_bf16_normal(50 + r, static_cast<int64_t>(lens[r]) * kw, 1.0f);
    const std::vector<uint16_t> v = random_bf16_normal(60 + r, static_cast<int64_t>(lens[r]) * vw, 1.0f);
    for (int t = 0; t < lens[r]; ++t) {
      const int64_t s = pool.phys(r, t);
      std::copy(k.begin() + static_cast<size_t>(t) * kw, k.begin() + static_cast<size_t>(t + 1) * kw, kh.begin() + s * kw);
      std::copy(v.begin() + static_cast<size_t>(t) * vw, v.begin() + static_cast<size_t>(t + 1) * vw, vh.begin() + s * vw);
    }
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(pool.table);
  const std::vector<float> sink = random_f32_uniform(78, lh, 2.0f);
  DevBuf dsink = up(sink);
  // The batch: two rows of request 1, one of 0, one padding, one of 1.
  const std::vector<int32_t> req_ids = {1, 1, 0, 1, 1};
  const std::vector<int64_t> pos = {198, 199, 69, -1, 4};
  const int rows = 5;
  const std::vector<uint16_t> q = random_bf16_normal(70, static_cast<int64_t>(rows) * qw, 1.0f);
  for (const int window : {0, 128}) {
    auto run = [&](const std::vector<int32_t>& ids, const std::vector<int64_t>& ps, const std::vector<uint16_t>& qq) {
      const int n = static_cast<int>(ps.size());
      DevBuf dq = up(qq), dpos = up(ps), dreq = up(ids);
      const int n_split = 4;
      const size_t part = static_cast<size_t>(n) * n_split * lh;
      DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4), out(static_cast<size_t>(n) * ow * 2);
      dgpp::mimo_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                              ptr<int64_t>(dpos), n, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                              pool.blocks_per_request, window, ref_scale(DK), ptr<float>(dsink), mptr<float>(m_ws),
                              mptr<float>(l_ws), mptr<float>(c_ws), st);
      dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), n, n_split, lh, mptr<uint16_t>(out), st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      return down<uint16_t>(out, static_cast<size_t>(n) * ow);
    };
    const std::vector<uint16_t> batched = run(req_ids, pos, q);
    for (int r = 0; r < rows; ++r) {
      const std::vector<uint16_t> qr(q.begin() + static_cast<size_t>(r) * qw, q.begin() + static_cast<size_t>(r + 1) * qw);
      const std::vector<uint16_t> single = run({req_ids[static_cast<size_t>(r)]}, {pos[static_cast<size_t>(r)]}, qr);
      require(std::equal(single.begin(), single.end(), batched.begin() + static_cast<size_t>(r) * ow),
              "row " + std::to_string(r) + " of the batch bitwise its single-request run (window " + std::to_string(window) + ")");
    }
  }
  std::printf("[ OK ] batched decode rows bitwise their single-request runs\n");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}

// ---- the fp8 cache (engine.kv_dtype fp8) ----------------------------------------
namespace {

// The host codec of kernels/latent_format.hpp over one head row of bf16
// values: the codes and the scale the finish writes.
void host_fp8_row(const uint16_t* row, int dim, uint8_t* codes, float* scale) {
  float amax = 0.f;
  for (int d = 0; d < dim; ++d) amax = std::max(amax, std::fabs(dgpp::bf16_bits_to_float(row[d])));
  const dgpp::LatentFp8Scale s = dgpp::latent_fp8_row_scale(amax);
  for (int d = 0; d < dim; ++d) codes[d] = dgpp::latent_fp8_encode(dgpp::bf16_bits_to_float(row[d]), s.inv);
  *scale = s.scale;
}

}  // namespace

DGPP_TEST(mimo_fp8_cache_finish_codes_are_the_host_codec_of_the_bf16_finish) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int rows = 9;
  const std::vector<int32_t> req_ids = {0, 0, 1, 1, 0, 1, 0, 1, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 383, -1, 5};
  DevBuf dreq = up(req_ids), dpos = up(pos), dtbl = up(pool.table);
  const std::vector<float> inv_swa = inv_freq_host(1e4);
  DevBuf dinv = up(inv_swa);
  dgpp::MimoQkvLayout layout;
  layout.chunks = 2;
  layout.q_per_chunk = 2;
  layout.kv_per_chunk = 2;
  const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * DK + layout.kv_per_chunk * (DK + DV);
  layout.chunk_stride = ((chunk_rows + 127) / 128) * 128;
  const int lh = layout.chunks * layout.q_per_chunk, lkv = layout.chunks * layout.kv_per_chunk;
  const int64_t qkv_stride = layout.chunks * layout.chunk_stride;
  const std::vector<float> qkv = random_f32_uniform(19, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
  DevBuf dqkv = up(qkv);
  const int qw = lh * DK, kw = lkv * DK, vw = lkv * DV;
  const size_t slots = static_cast<size_t>(pool.slots());
  // The bf16 finish: the values the fp8 form quantizes.
  DevBuf kc16(slots * kw * 2), vc16(slots * vw * 2), qo(static_cast<size_t>(rows) * qw * 2);
  dgpp::mimo_qkv_finish(ptr<float>(dqkv), qkv_stride, layout, ptr<float>(dinv), 0.707f, ptr<int32_t>(dreq),
                        ptr<int64_t>(dpos), rows, lh, lkv, ptr<int32_t>(dtbl), pool.blocks_per_request,
                        pool.block_tokens, mptr<uint16_t>(qo), qw, mptr<uint16_t>(kc16), mptr<uint16_t>(vc16), st);
  // The fp8 finish: codes and scales, the untouched slots poisoned.
  DevBuf kc8(slots * kw), vc8(slots * vw), ks(slots * lkv * 4), vs(slots * lkv * 4), qo8(static_cast<size_t>(rows) * qw * 2);
  DGPP_CUDA_OK(cudaMemset(kc8.p, 0x7F, slots * kw));
  DGPP_CUDA_OK(cudaMemset(vc8.p, 0x7F, slots * vw));
  DGPP_CUDA_OK(cudaMemset(ks.p, 0x7F, slots * lkv * 4));
  DGPP_CUDA_OK(cudaMemset(vs.p, 0x7F, slots * lkv * 4));
  dgpp::mimo_qkv_finish(ptr<float>(dqkv), qkv_stride, layout, ptr<float>(dinv), 0.707f, ptr<int32_t>(dreq),
                        ptr<int64_t>(dpos), rows, lh, lkv, ptr<int32_t>(dtbl), pool.blocks_per_request,
                        pool.block_tokens, mptr<uint16_t>(qo8), qw, mptr<uint16_t>(kc8), mptr<uint16_t>(vc8), st,
                        mptr<float>(ks), mptr<float>(vs));
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> k16 = down<uint16_t>(kc16, slots * kw), v16 = down<uint16_t>(vc16, slots * vw);
  const std::vector<uint8_t> k8 = down<uint8_t>(kc8, slots * kw), v8 = down<uint8_t>(vc8, slots * vw);
  const std::vector<float> ks_h = down<float>(ks, slots * lkv), vs_h = down<float>(vs, slots * lkv);
  require(down<uint16_t>(qo8, static_cast<size_t>(rows) * qw) == down<uint16_t>(qo, static_cast<size_t>(rows) * qw),
          "the q rows are the bf16 finish's");
  std::vector<uint8_t> codes(static_cast<size_t>(DK));
  long checked = 0;
  for (int r = 0; r < rows; ++r) {
    const int64_t p = pos[static_cast<size_t>(r)];
    if (p < 0) continue;
    const size_t slot = static_cast<size_t>(pool.phys(req_ids[static_cast<size_t>(r)], p));
    for (int j = 0; j < lkv; ++j) {
      float scale = 0.f;
      host_fp8_row(k16.data() + slot * kw + j * DK, DK, codes.data(), &scale);
      for (int d = 0; d < DK; ++d) require(k8[slot * kw + j * DK + d] == codes[static_cast<size_t>(d)], "K codes bitwise");
      require(ks_h[slot * lkv + j] == scale, "K scale bitwise");
      host_fp8_row(v16.data() + slot * vw + j * DV, DV, codes.data(), &scale);
      for (int d = 0; d < DV; ++d) require(v8[slot * vw + j * DV + d] == codes[static_cast<size_t>(d)], "V codes bitwise");
      require(vs_h[slot * lkv + j] == scale, "V scale bitwise");
      checked += DK + DV;
    }
  }
  std::printf("[ .. ] fp8 finish: %ld codes bitwise the host codec, the scales bitwise\n", checked);
}

DGPP_TEST(mimo_fp8_cache_attention_is_the_bf16_kernel_over_the_dequantized_rows) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const int kv = 2;
  Seq seq;
  seq.k = random_bf16_normal(41, static_cast<int64_t>(seq.len) * kv * DK, 1.0f);
  seq.v = random_bf16_normal(42, static_cast<int64_t>(seq.len) * kv * DV, 1.0f);
  // Quantize every head row on the host; the dequantized rows are what the
  // bf16 kernel sees when it runs over them.
  const int kw = kv * DK, vw = kv * DV;
  const size_t slots = static_cast<size_t>(pool.slots());
  std::vector<uint8_t> k8(slots * kw, 0x7F), v8(slots * vw, 0x7F);
  std::vector<float> ks(slots * kv, 0.f), vs(slots * kv, 0.f);
  std::vector<uint16_t> kdq(slots * kw, 0x7F7F), vdq(slots * vw, 0x7F7F);
  std::vector<uint8_t> codes(static_cast<size_t>(DK));
  for (int t = 0; t < seq.len; ++t) {
    const size_t s = static_cast<size_t>(pool.phys(seq.req, t));
    for (int j = 0; j < kv; ++j) {
      float scale = 0.f;
      host_fp8_row(seq.k.data() + static_cast<size_t>(t) * kw + j * DK, DK, codes.data(), &scale);
      for (int d = 0; d < DK; ++d) {
        k8[s * kw + j * DK + d] = codes[static_cast<size_t>(d)];
        kdq[s * kw + j * DK + d] = dgpp::latent_fp8_decode_bf16(codes[static_cast<size_t>(d)], scale);
      }
      ks[s * kv + j] = scale;
      host_fp8_row(seq.v.data() + static_cast<size_t>(t) * vw + j * DV, DV, codes.data(), &scale);
      for (int d = 0; d < DV; ++d) {
        v8[s * vw + j * DV + d] = codes[static_cast<size_t>(d)];
        vdq[s * vw + j * DV + d] = dgpp::latent_fp8_decode_bf16(codes[static_cast<size_t>(d)], scale);
      }
      vs[s * kv + j] = scale;
    }
  }
  DevBuf dk8 = up(k8), dv8 = up(v8), dks = up(ks), dvs = up(vs), dkdq = up(kdq), dvdq = up(vdq), dtbl = up(pool.table);
  const std::vector<int64_t> pos = {0, 5, 31, 32, 63, 64, 200, 299, -1};
  const int rows = static_cast<int>(pos.size());
  const std::vector<int32_t> req_ids(static_cast<size_t>(rows), seq.req);
  DevBuf dpos = up(pos), dreq = up(req_ids);
  const float scale = ref_scale(DK);
  long checked = 0;
  for (const int hpk : {1, 16}) {
    const int lh = hpk * kv, qw = lh * DK, ow = lh * DV;
    const std::vector<uint16_t> q = random_bf16_normal(43 + hpk, static_cast<int64_t>(rows) * qw, 1.0f);
    const std::vector<float> sink = random_f32_uniform(78, lh, 2.0f);
    DevBuf dq = up(q), dsink = up(sink);
    for (const int window : {0, 128}) {
      for (const int n_split : {1, 32}) {
        const size_t part = static_cast<size_t>(rows) * n_split * lh;
        DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4), out8(static_cast<size_t>(rows) * ow * 2),
            out16(static_cast<size_t>(rows) * ow * 2);
        dgpp::mimo_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(dk8), ptr<uint16_t>(dv8), ptr<int32_t>(dreq),
                                ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                pool.blocks_per_request, window, scale, ptr<float>(dsink), mptr<float>(m_ws),
                                mptr<float>(l_ws), mptr<float>(c_ws), st, ptr<float>(dks), ptr<float>(dvs));
        dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, lh,
                                mptr<uint16_t>(out8), st);
        dgpp::mimo_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(dkdq), ptr<uint16_t>(dvdq), ptr<int32_t>(dreq),
                                ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                pool.blocks_per_request, window, scale, ptr<float>(dsink), mptr<float>(m_ws),
                                mptr<float>(l_ws), mptr<float>(c_ws), st);
        dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split, lh,
                                mptr<uint16_t>(out16), st);
        DGPP_CUDA_OK(cudaStreamSynchronize(st));
        const std::vector<uint16_t> a = down<uint16_t>(out8, static_cast<size_t>(rows) * ow);
        const std::vector<uint16_t> b = down<uint16_t>(out16, static_cast<size_t>(rows) * ow);
        require(a == b, "fp8 attention bitwise the bf16 kernel over the dequantized rows (hpk " + std::to_string(hpk) +
                            " window " + std::to_string(window) + " n_split " + std::to_string(n_split) + ")");
        checked += static_cast<long>(a.size());
      }
    }
  }
  std::printf("[ .. ] fp8 attention: %ld outputs bitwise the bf16 kernel over the dequantized cache\n", checked);
}

// ---- the fused decode attention (plan §7.1) ---------------------------------------
// The one-launch kernel against the three-kernel chain over the same
// batch: bf16 and fp8 caches, global and sliding-window layers, one and
// two kv heads per chunk, a batch whose requests contribute several rows
// at consecutive positions (the rows attend each other's just-finished
// K/V), a padding row, many empty splits. The output rows, the cache rows
// the batch appends and (fp8) their scale planes must be bitwise.
DGPP_TEST(mimo_fused_decode_attention_is_the_three_kernel_chain) {
  cudaStream_t st = test_stream();
  const Pool pool;
  const std::vector<float> inv_ga = inv_freq_host(1e7), inv_swa = inv_freq_host(1e4);
  DevBuf dinv_ga = up(inv_ga), dinv_swa = up(inv_swa), dtbl = up(pool.table);
  // The batch: request 1 at 198..200 (three rows), request 0 at 69 and 70,
  // a padding row, request 1 at 4 (its own earlier context only).
  const std::vector<int32_t> req_ids = {1, 1, 1, 0, 0, 1, 0};
  const std::vector<int64_t> pos = {198, 199, 200, 69, 70, -1, 4};
  const int rows = static_cast<int>(pos.size());
  const int64_t first_batch_pos[2] = {69, 198};  // positions below hold prior K/V
  DevBuf dreq = up(req_ids), dpos = up(pos);
  long checked = 0;
  for (const int kvpc : {1, 2}) {
    dgpp::MimoQkvLayout layout;
    layout.chunks = 2;
    layout.q_per_chunk = 8;
    layout.kv_per_chunk = kvpc;
    const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * DK + kvpc * (DK + DV);
    layout.chunk_stride = ((chunk_rows + 127) / 128) * 128;
    const int lh = layout.chunks * layout.q_per_chunk, kv = layout.chunks * kvpc;
    const int64_t qkv_stride = layout.chunks * layout.chunk_stride;
    const int kw = kv * DK, vw = kv * DV, ow = lh * DV, qw = lh * DK;
    const std::vector<float> qkv = random_f32_uniform(400 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
    const std::vector<float> sink = random_f32_uniform(410 + kvpc, lh, 2.0f);
    DevBuf dqkv = up(qkv), dsink = up(sink);
    for (const bool fp8 : {false, true}) {
      // The prior context: bf16 rows written through the codec when fp8.
      const size_t slots = static_cast<size_t>(pool.slots());
      std::vector<uint16_t> kh(slots * kw, 0), vh(slots * vw, 0);
      std::vector<uint8_t> k8(slots * kw, 0), v8(slots * vw, 0);
      std::vector<float> ks(slots * kv, 0.f), vs(slots * kv, 0.f);
      std::vector<uint8_t> codes(static_cast<size_t>(DK));
      for (int r = 0; r < 2; ++r) {
        const std::vector<uint16_t> k = random_bf16_normal(420 + r, first_batch_pos[r] * kw, 1.0f);
        const std::vector<uint16_t> v = random_bf16_normal(430 + r, first_batch_pos[r] * vw, 1.0f);
        for (int64_t t = 0; t < first_batch_pos[r]; ++t) {
          const size_t s = static_cast<size_t>(pool.phys(r, t));
          std::copy(k.begin() + t * kw, k.begin() + (t + 1) * kw, kh.begin() + s * kw);
          std::copy(v.begin() + t * vw, v.begin() + (t + 1) * vw, vh.begin() + s * vw);
          for (int j = 0; j < kv; ++j) {
            host_fp8_row(kh.data() + s * kw + j * DK, DK, codes.data(), &ks[s * kv + j]);
            std::copy(codes.begin(), codes.begin() + DK, k8.begin() + s * kw + j * DK);
            host_fp8_row(vh.data() + s * vw + j * DV, DV, codes.data(), &vs[s * kv + j]);
            std::copy(codes.begin(), codes.begin() + DV, v8.begin() + s * vw + j * DV);
          }
        }
      }
      for (const bool swa : {false, true}) {
        const int window = swa ? 128 : 0;
        const int n_split = swa ? 4 : 32;
        const float* inv = swa ? ptr<float>(dinv_swa) : ptr<float>(dinv_ga);
        const size_t part = static_cast<size_t>(rows) * n_split * lh;
        struct Run {
          DevBuf kc, vc, ksc, vsc, m_ws, l_ws, c_ws, out;
        };
        auto make = [&]() {
          Run x;
          x.kc = fp8 ? up(k8) : up(kh);
          x.vc = fp8 ? up(v8) : up(vh);
          x.ksc = up(ks);
          x.vsc = up(vs);
          x.m_ws = DevBuf(part * 4);
          x.l_ws = DevBuf(part * 4);
          x.c_ws = DevBuf(part * DV * 4);
          x.out = DevBuf(static_cast<size_t>(rows) * ow * 2);
          DGPP_CUDA_OK(cudaMemset(x.out.p, 0x7F, x.out.bytes));
          return x;
        };
        Run chain = make(), fused = make();
        // The chain.
        {
          DevBuf q(static_cast<size_t>(rows) * qw * 2);
          dgpp::mimo_qkv_finish(ptr<float>(dqkv), qkv_stride, layout, inv, 0.707f, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                                rows, lh, kv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens,
                                mptr<uint16_t>(q), qw, mptr<uint16_t>(chain.kc), mptr<uint16_t>(chain.vc), st,
                                fp8 ? mptr<float>(chain.ksc) : nullptr, fp8 ? mptr<float>(chain.vsc) : nullptr);
          dgpp::mimo_attn_partial(ptr<uint16_t>(q), qw, ptr<uint16_t>(chain.kc), ptr<uint16_t>(chain.vc),
                                  ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, n_split, lh, kv, pool.block_tokens,
                                  ptr<int32_t>(dtbl), pool.blocks_per_request, window, ref_scale(DK), ptr<float>(dsink),
                                  mptr<float>(chain.m_ws), mptr<float>(chain.l_ws), mptr<float>(chain.c_ws), st,
                                  fp8 ? ptr<float>(chain.ksc) : nullptr, fp8 ? ptr<float>(chain.vsc) : nullptr);
          dgpp::mimo_attn_combine(ptr<float>(chain.m_ws), ptr<float>(chain.l_ws), ptr<float>(chain.c_ws), rows, n_split,
                                  lh, mptr<uint16_t>(chain.out), st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
        }
        // The fused kernel, twice on the same counters (they must come back
        // to zero).
        {
          DevBuf counters(static_cast<size_t>(rows) * kv * 4);
          DGPP_CUDA_OK(cudaMemset(counters.p, 0, counters.bytes));
          dgpp::MimoAttnFusedArgs a;
          a.qkv = ptr<float>(dqkv);
          a.qkv_stride = qkv_stride;
          a.layout = layout;
          a.inv_freq = inv;
          a.value_scale = 0.707f;
          a.req_ids = ptr<int32_t>(dreq);
          a.pos = ptr<int64_t>(dpos);
          a.rows = rows;
          a.n_split = n_split;
          a.local_heads = lh;
          a.kv_heads = kv;
          a.block_tables = ptr<int32_t>(dtbl);
          a.blocks_per_request = pool.blocks_per_request;
          a.block_tokens = pool.block_tokens;
          a.window = window;
          a.scale = ref_scale(DK);
          a.sink = ptr<float>(dsink);
          a.k_cache = mptr<uint16_t>(fused.kc);
          a.v_cache = mptr<uint16_t>(fused.vc);
          a.k_scale = fp8 ? mptr<float>(fused.ksc) : nullptr;
          a.v_scale = fp8 ? mptr<float>(fused.vsc) : nullptr;
          a.m_ws = mptr<float>(fused.m_ws);
          a.l_ws = mptr<float>(fused.l_ws);
          a.c_ws = mptr<float>(fused.c_ws);
          a.counters = static_cast<int*>(counters.p);
          a.out = mptr<uint16_t>(fused.out);
          dgpp::mimo_attn_fused(a, st);
          dgpp::mimo_attn_fused(a, st);
          DGPP_CUDA_OK(cudaStreamSynchronize(st));
          const auto cnt = down<int32_t>(counters, static_cast<size_t>(rows) * kv);
          for (int c : cnt) require(c == 0, "the fused kernel leaves its counters zero");
        }
        const std::string tag = std::string(fp8 ? "fp8" : "bf16") + (swa ? " swa" : " global") + " kvpc " +
                                std::to_string(kvpc);
        const auto co = down<uint16_t>(chain.out, static_cast<size_t>(rows) * ow);
        const auto fo = down<uint16_t>(fused.out, static_cast<size_t>(rows) * ow);
        require(co == fo, "fused output bitwise the chain's (" + tag + ")");
        const size_t kbytes = fp8 ? slots * kw : slots * kw * 2, vbytes = fp8 ? slots * vw : slots * vw * 2;
        require(down<uint8_t>(chain.kc, kbytes) == down<uint8_t>(fused.kc, kbytes), "K cache bitwise (" + tag + ")");
        require(down<uint8_t>(chain.vc, vbytes) == down<uint8_t>(fused.vc, vbytes), "V cache bitwise (" + tag + ")");
        if (fp8) {
          require(down<float>(chain.ksc, slots * kv) == down<float>(fused.ksc, slots * kv), "K scales bitwise (" + tag + ")");
          require(down<float>(chain.vsc, slots * kv) == down<float>(fused.vsc, slots * kv), "V scales bitwise (" + tag + ")");
        }
        checked += static_cast<long>(co.size());
      }
    }
  }
  std::printf("[ .. ] fused decode attention: %ld outputs and the appended cache rows bitwise the chain\n", checked);
}

// ---- the query-tiled prefill attention (plan §7.2) ------------------------------
// The tensor-core kernel against the split-KV chain over the same prefill
// rows: bf16 and fp8 caches, global and sliding-window layers, sinks, one
// and two kv heads per chunk, a group prefill whose spans meet inside a
// query tile, a padding row. The chains' fp32 summation orders differ, so
// the pin is a tolerance: at most 2 bf16 ulps on any output, almost all
// exact.
DGPP_TEST(mimo_tiled_prefill_attention_matches_the_split_kv_chain) {
  cudaStream_t st = test_stream();
  const Pool pool;  // 2 requests x 8 blocks x 64 tokens
  const std::vector<float> inv_ga = inv_freq_host(1e7), inv_swa = inv_freq_host(1e4);
  DevBuf dinv_ga = up(inv_ga), dinv_swa = up(inv_swa), dtbl = up(pool.table);
  // Rows: request 0's positions 100..189 (90 rows), then request 1's 0..45
  // (46 rows), then a padding row, then request 1's 46..60 — 152 rows, so
  // the tiles of 4 (global) and 8 (sliding) rows straddle the span joins.
  std::vector<int32_t> req_ids;
  std::vector<int64_t> pos;
  for (int64_t p = 100; p < 190; ++p) { req_ids.push_back(0); pos.push_back(p); }
  for (int64_t p = 0; p < 46; ++p) { req_ids.push_back(1); pos.push_back(p); }
  req_ids.push_back(1); pos.push_back(-1);
  for (int64_t p = 46; p < 61; ++p) { req_ids.push_back(1); pos.push_back(p); }
  const int rows = static_cast<int>(pos.size());
  DevBuf dreq = up(req_ids), dpos = up(pos);
  long checked = 0;
  int worst = 0;
  for (const int kvpc : {1, 2}) {
    dgpp::MimoQkvLayout layout;
    layout.chunks = 2;
    layout.q_per_chunk = 8;
    layout.kv_per_chunk = kvpc;
    const int64_t chunk_rows = static_cast<int64_t>(layout.q_per_chunk) * DK + kvpc * (DK + DV);
    layout.chunk_stride = ((chunk_rows + 127) / 128) * 128;
    const int lh = layout.chunks * layout.q_per_chunk, kv = layout.chunks * kvpc;
    const int64_t qkv_stride = layout.chunks * layout.chunk_stride;
    const int kw = kv * DK, vw = kv * DV, ow = lh * DV, qw = lh * DK;
    const std::vector<float> qkv = random_f32_uniform(500 + kvpc, static_cast<int64_t>(rows) * qkv_stride, 2.0f);
    const std::vector<float> sink = random_f32_uniform(510 + kvpc, lh, 2.0f);
    DevBuf dqkv = up(qkv), dsink = up(sink);
    for (const bool fp8 : {false, true}) {
      // Request 0's context below position 100 (through the codec when fp8).
      const size_t slots = static_cast<size_t>(pool.slots());
      std::vector<uint16_t> kh(slots * kw, 0), vh(slots * vw, 0);
      std::vector<uint8_t> k8(slots * kw, 0), v8(slots * vw, 0);
      std::vector<float> ks(slots * kv, 0.f), vs(slots * kv, 0.f);
      std::vector<uint8_t> codes(static_cast<size_t>(DK));
      const std::vector<uint16_t> k = random_bf16_normal(520, 100 * kw, 1.0f);
      const std::vector<uint16_t> v = random_bf16_normal(530, 100 * vw, 1.0f);
      for (int64_t tk = 0; tk < 100; ++tk) {
        const size_t s = static_cast<size_t>(pool.phys(0, tk));
        std::copy(k.begin() + tk * kw, k.begin() + (tk + 1) * kw, kh.begin() + s * kw);
        std::copy(v.begin() + tk * vw, v.begin() + (tk + 1) * vw, vh.begin() + s * vw);
        for (int j = 0; j < kv; ++j) {
          host_fp8_row(kh.data() + s * kw + j * DK, DK, codes.data(), &ks[s * kv + j]);
          std::copy(codes.begin(), codes.begin() + DK, k8.begin() + s * kw + j * DK);
          host_fp8_row(vh.data() + s * vw + j * DV, DV, codes.data(), &vs[s * kv + j]);
          std::copy(codes.begin(), codes.begin() + DV, v8.begin() + s * vw + j * DV);
        }
      }
      for (const bool swa : {false, true}) {
        const int window = swa ? 128 : 0;
        const float* inv = swa ? ptr<float>(dinv_swa) : ptr<float>(dinv_ga);
        DevBuf kc = fp8 ? up(k8) : up(kh), vc = fp8 ? up(v8) : up(vh), ksc = up(ks), vsc = up(vs);
        DevBuf q(static_cast<size_t>(rows) * qw * 2);
        dgpp::mimo_qkv_finish(ptr<float>(dqkv), qkv_stride, layout, inv, 0.707f, ptr<int32_t>(dreq), ptr<int64_t>(dpos),
                              rows, lh, kv, ptr<int32_t>(dtbl), pool.blocks_per_request, pool.block_tokens,
                              mptr<uint16_t>(q), qw, mptr<uint16_t>(kc), mptr<uint16_t>(vc), st,
                              fp8 ? mptr<float>(ksc) : nullptr, fp8 ? mptr<float>(vsc) : nullptr);
        // The chain (one split per row).
        const size_t part = static_cast<size_t>(rows) * lh;
        DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * DV * 4), chain_out(static_cast<size_t>(rows) * ow * 2);
        dgpp::mimo_attn_partial(ptr<uint16_t>(q), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc), ptr<int32_t>(dreq),
                                ptr<int64_t>(dpos), rows, 1, lh, kv, pool.block_tokens, ptr<int32_t>(dtbl),
                                pool.blocks_per_request, window, ref_scale(DK), ptr<float>(dsink), mptr<float>(m_ws),
                                mptr<float>(l_ws), mptr<float>(c_ws), st, fp8 ? ptr<float>(ksc) : nullptr,
                                fp8 ? ptr<float>(vsc) : nullptr);
        dgpp::mimo_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, 1, lh,
                                mptr<uint16_t>(chain_out), st);
        // The tiled kernel.
        DevBuf tiled_out(static_cast<size_t>(rows) * ow * 2);
        DGPP_CUDA_OK(cudaMemset(tiled_out.p, 0x7F, tiled_out.bytes));
        dgpp::MimoAttnPrefillArgs a;
        a.q = ptr<uint16_t>(q);
        a.q_stride = qw;
        a.k_cache = ptr<uint16_t>(kc);
        a.v_cache = ptr<uint16_t>(vc);
        a.k_scale = fp8 ? ptr<float>(ksc) : nullptr;
        a.v_scale = fp8 ? ptr<float>(vsc) : nullptr;
        a.req_ids = ptr<int32_t>(dreq);
        a.pos = ptr<int64_t>(dpos);
        a.rows = rows;
        a.local_heads = lh;
        a.kv_heads = kv;
        a.block_tokens = pool.block_tokens;
        a.block_tables = ptr<int32_t>(dtbl);
        a.blocks_per_request = pool.blocks_per_request;
        a.window = window;
        a.scale = ref_scale(DK);
        a.sink = ptr<float>(dsink);
        a.out = mptr<uint16_t>(tiled_out);
        dgpp::mimo_attn_prefill(a, st);
        DGPP_CUDA_OK(cudaStreamSynchronize(st));
        const auto co = down<uint16_t>(chain_out, static_cast<size_t>(rows) * ow);
        const auto to = down<uint16_t>(tiled_out, static_cast<size_t>(rows) * ow);
        const std::string tag = std::string(fp8 ? "fp8" : "bf16") + (swa ? " swa" : " global") + " kvpc " +
                                std::to_string(kvpc);
        std::vector<float> cf(co.size()), tf(co.size());
        double rms = 0;
        for (size_t i = 0; i < co.size(); ++i) {
          cf[i] = dgpp::bf16_bits_to_float(co[i]);
          tf[i] = dgpp::bf16_bits_to_float(to[i]);
          rms += static_cast<double>(cf[i]) * cf[i];
        }
        rms = std::sqrt(rms / static_cast<double>(co.size()));
        const Stats s = compare_abs_rel(tf.data(), cf.data(), static_cast<long>(co.size()), 2 * std::pow(2.0, -7.0),
                                        0.005 * rms);
        std::printf("[ .. ] tiled prefill %s: l2_rel %.3g max_abs %.3g mismatches %ld/%ld (rms %.3g)\n", tag.c_str(),
                    s.l2_rel, s.max_abs, s.mismatches, s.n, rms);
        require_bf16("tiled prefill attention " + tag, s, 2e-3, 0.01);
        worst = std::max(worst, static_cast<int>(s.mismatches));
        // The padding row (row 136) is zeros.
        for (int i = 0; i < ow; ++i)
          require(to[static_cast<size_t>(136) * ow + i] == 0, "the tiled kernel's padding row is zero (" + tag + ")");
        checked += static_cast<long>(co.size());
      }
    }
  }
  std::printf("[ .. ] tiled prefill attention: %ld outputs vs the split-KV chain, at most %d past two ulps per case\n",
              checked, worst);
}
