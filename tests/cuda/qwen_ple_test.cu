// The PLE layer's kernels (Q3, 2026-09-09) against the host reference:
// the hash ids bitwise (and inside their heads' ranges, EOS resets
// included), the table gather bitwise at world 1 and on a TP=2 rank's
// head slice, the gate within two bf16 ulps (the dot's reduction order),
// the dilated conv bitwise with its state — one call against two chunks
// with the state carried — and the whole layer end to end at world 1
// through the GEMM seam, at a GEMV-shaped and a GEMM-shaped token count.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/gemm.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qwen_ple.hpp"
#include "models/qwen/ple_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Small geometry with the real hash multipliers and vocabulary; sixteen
// small primes for the heads' vocabularies.
struct Geo {
  int hidden = 256, hc = 4, heads = 16, heads_per_ngram = 8, head_dim = 16;
  int width = 4, dilation = 3;
  int32_t eos = 248044;
  int32_t vocab = 248320;
  float eps = 1e-6f;
  std::vector<int64_t> mult{23703573157769LL, 20109073645365LL, 8052911324071LL};
  std::vector<int64_t> head_vocab{101, 103, 107, 109, 113, 127, 131, 137,
                                  139, 149, 151, 157, 163, 167, 173, 179};
  std::vector<int64_t> head_offset;
  int64_t total_rows = 0, padded_rows = 0;
  float scale = 0.000199317f;
  int channels() const { return hc * hidden; }
  int embed() const { return heads * head_dim; }
  int state_len() const { return (width - 1) * dilation; }
  Geo() {
    head_offset.resize(head_vocab.size());
    for (size_t h = 0; h < head_vocab.size(); ++h) {
      head_offset[h] = total_rows;
      total_rows += head_vocab[h];
    }
    padded_rows = (total_rows + 127) / 128 * 128;
  }
};

std::vector<int32_t> make_tokens(const Geo& g, int n, uint64_t seed) {
  std::vector<int32_t> t(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    t[static_cast<size_t>(i)] = static_cast<int32_t>(hash_u64(seed * 1000003ull + static_cast<uint64_t>(i)) % g.vocab);
  // EOS runs: single, double, and at the start of the sequence.
  if (n > 0) t[0] = g.eos;
  if (n > 6) t[5] = t[6] = g.eos;
  if (n > 20) t[20] = g.eos;
  return t;
}

std::vector<uint8_t> make_table(const Geo& g, uint64_t seed) {
  std::vector<uint8_t> t(static_cast<size_t>(g.padded_rows) * g.head_dim);
  for (size_t i = 0; i < t.size(); ++i)
    t[i] = dgpp::float_to_fp8_e4m3_bits(64.0f * normal_f(seed, static_cast<int64_t>(i)));
  return t;
}

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

std::vector<int32_t> host_ids(const Geo& g, const std::vector<int32_t>& tokens) {
  std::vector<int32_t> ids(tokens.size() * g.heads);
  dgpp::qwen_ref::ple_hash_ids(tokens.data(), static_cast<int>(tokens.size()), g.eos, g.eos, g.eos,
                               g.mult.data(), g.head_vocab.data(), g.head_offset.data(), g.heads,
                               g.heads_per_ngram, ids.data());
  return ids;
}

}  // namespace

DGPP_TEST(qwen_ple_hash_ids_match_the_reference_and_stay_in_range) {
  Geo g;
  const int n = 37;
  const std::vector<int32_t> tokens = make_tokens(g, n, 1);
  const std::vector<int32_t> ref = host_ids(g, tokens);
  DevBuf dt = up(tokens), dm = up(g.mult), dv = up(g.head_vocab), doff = up(g.head_offset);
  DevBuf dids(ref.size() * 4);
  cudaStream_t st = test_stream();
  dgpp::qwen_ple_hash_ids(static_cast<const int32_t*>(dt.p), n, g.eos, g.eos, g.eos,
                          static_cast<const int64_t*>(dm.p), static_cast<const int64_t*>(dv.p),
                          static_cast<const int64_t*>(doff.p), g.heads, g.heads_per_ngram,
                          static_cast<int32_t*>(dids.p), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<int32_t> got = down<int32_t>(dids, ref.size());
  require_bitwise("hash ids", got.data(), ref.data(), ref.size() * 4);
  for (int t = 0; t < n; ++t)
    for (int h = 0; h < g.heads; ++h) {
      const int64_t id = got[static_cast<size_t>(t) * g.heads + h];
      require(id >= g.head_offset[h] && id < g.head_offset[h] + g.head_vocab[h],
              "id outside its head's range");
    }
  // The EOS rule, spelled out: after the double EOS at 5, 6 token 7's
  // trigram context is (x7, EOS, EOS) — the same ids as a sequence start
  // with x7; token 8's is (x8, x7, EOS).
  std::vector<int32_t> solo{tokens[7]};
  const std::vector<int32_t> solo_ids = host_ids(g, solo);
  for (int h = 0; h < g.heads; ++h)
    require(got[7 * g.heads + h] == solo_ids[h], "EOS reset: token 7 is not a sequence start");
  std::vector<int32_t> pair{tokens[7], tokens[8]};
  const std::vector<int32_t> pair_ids = host_ids(g, pair);
  for (int h = 0; h < g.heads; ++h)
    require(got[8 * g.heads + h] == pair_ids[g.heads + h], "EOS reset: token 8's context");
  // A different mix for the trigram heads than the bigram heads (the
  // third id matters) whenever x[t-2] is not EOS.
  bool differs = false;
  for (int h = 8; h < 16 && !differs; ++h)
    differs = got[10 * g.heads + h] - g.head_offset[h] != got[10 * g.heads + h - 8] - g.head_offset[h - 8];
  require(differs, "trigram heads equal the bigram heads");
}

DGPP_TEST(qwen_ple_gather_is_bitwise_at_world_1_and_on_a_head_slice) {
  Geo g;
  const int n = 37;
  const std::vector<int32_t> tokens = make_tokens(g, n, 2);
  const std::vector<int32_t> ids = host_ids(g, tokens);
  const std::vector<uint8_t> table = make_table(g, 3);
  std::vector<uint16_t> ref(static_cast<size_t>(n) * g.embed());
  dgpp::qwen_ref::ple_gather(table.data(), 0, g.scale, ids.data(), n, g.heads, 0, g.heads, g.head_dim,
                             ref.data());
  DevBuf dtab = up(table), dids = up(ids);
  DevBuf dout(ref.size() * 2);
  cudaStream_t st = test_stream();
  dgpp::qwen_ple_gather_bf16(static_cast<const uint8_t*>(dtab.p), 0, g.padded_rows, g.scale,
                             static_cast<const int32_t*>(dids.p), n, g.heads, 0, g.heads, g.head_dim,
                             static_cast<uint16_t*>(dout.p), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dout, ref.size());
  require_bitwise("gather world 1", got.data(), ref.data(), ref.size() * 2);
  // Rank 1 of 2: heads 8..15, the table rows from head 8's offset.
  const int hb = 8, hl = 8;
  const int64_t row_begin = g.head_offset[static_cast<size_t>(hb)];
  const int64_t rows = g.padded_rows - row_begin;
  std::vector<uint16_t> ref_slice(static_cast<size_t>(n) * hl * g.head_dim);
  dgpp::qwen_ref::ple_gather(table.data() + row_begin * g.head_dim, row_begin, g.scale, ids.data(), n,
                             g.heads, hb, hl, g.head_dim, ref_slice.data());
  DevBuf dslice(ref_slice.size() * 2);
  dgpp::qwen_ple_gather_bf16(static_cast<const uint8_t*>(dtab.p) + row_begin * g.head_dim, row_begin,
                             rows, g.scale, static_cast<const int32_t*>(dids.p), n, g.heads, hb, hl,
                             g.head_dim, static_cast<uint16_t*>(dslice.p), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got_slice = down<uint16_t>(dslice, ref_slice.size());
  require_bitwise("gather head slice", got_slice.data(), ref_slice.data(), ref_slice.size() * 2);
  // The slice is the right half of the world-1 embedding, row by row.
  for (int t = 0; t < n; ++t)
    for (int j = 0; j < hl * g.head_dim; ++j)
      require(got_slice[static_cast<size_t>(t) * hl * g.head_dim + j] ==
                  got[static_cast<size_t>(t) * g.embed() + hb * g.head_dim + j],
              "head slice is not the embedding's column range");
}

DGPP_TEST(qwen_ple_gate_matches_the_reference_within_two_ulps) {
  Geo g;
  const int n = 37;
  const int64_t C = g.channels();
  const std::vector<uint16_t> kn = random_bf16_normal(11, n * C, 1.0f);
  const std::vector<uint16_t> qn = random_bf16_normal(12, n * C, 1.0f);
  const std::vector<uint16_t> v = random_bf16_normal(13, static_cast<int64_t>(n) * g.hidden, 0.5f);
  std::vector<uint16_t> ref(static_cast<size_t>(n) * C);
  dgpp::qwen_ref::ple_gate(kn.data(), qn.data(), v.data(), ref.data(), n, g.hc, g.hidden);
  DevBuf dk = up(kn), dq = up(qn), dv = up(v), dout(ref.size() * 2);
  cudaStream_t st = test_stream();
  dgpp::qwen_ple_gate_bf16(static_cast<const uint16_t*>(dk.p), static_cast<const uint16_t*>(dq.p),
                           static_cast<const uint16_t*>(dv.p), static_cast<uint16_t*>(dout.p), n,
                           g.hc, g.hidden, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dout, ref.size());
  const Stats s = compare_bf16(got, ref, 2);
  std::printf("[ .. ] gate: max_rel %.3g l2_rel %.3g mismatches %ld/%ld\n", s.max_rel, s.l2_rel,
              s.mismatches, s.n);
  require_bf16("gate", s, 2e-3, 0.01);
}

DGPP_TEST(qwen_ple_conv_is_bitwise_and_chunk_invariant) {
  Geo g;
  const int n = 37, n1 = 13;
  const int64_t C = g.channels();
  const int S = g.state_len();
  const std::vector<uint16_t> un = random_bf16_normal(21, n * C, 1.0f);
  const std::vector<uint16_t> gv = random_bf16_normal(22, n * C, 0.5f);
  const std::vector<uint16_t> state0 = random_bf16_normal(23, C * S, 1.0f);
  const std::vector<uint16_t> w = random_bf16_normal(24, C * g.width, 0.3f);
  const std::vector<uint16_t> r = random_bf16_normal(25, n * C, 1.0f);
  std::vector<uint16_t> ref(static_cast<size_t>(n) * C), ref_state = state0;
  dgpp::qwen_ref::ple_conv(un.data(), gv.data(), ref_state.data(), w.data(), r.data(), ref.data(), n,
                           static_cast<int>(C), g.width, g.dilation);
  DevBuf dun = up(un), dgv = up(gv), dstate = up(state0), dw = up(w), dr = up(r), dout(ref.size() * 2);
  cudaStream_t st = test_stream();
  auto conv = [&](int t0, int count) {
    dgpp::qwen_ple_conv_bf16(static_cast<const uint16_t*>(dun.p) + t0 * C,
                             static_cast<const uint16_t*>(dgv.p) + t0 * C,
                             static_cast<uint16_t*>(dstate.p), static_cast<const uint16_t*>(dw.p),
                             static_cast<const uint16_t*>(dr.p) + t0 * C,
                             static_cast<uint16_t*>(dout.p) + t0 * C, count, static_cast<int>(C),
                             g.width, g.dilation, st);
  };
  conv(0, n);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dout, ref.size());
  const std::vector<uint16_t> got_state = down<uint16_t>(dstate, ref_state.size());
  require_bitwise("conv output", got.data(), ref.data(), ref.size() * 2);
  require_bitwise("conv state", got_state.data(), ref_state.data(), ref_state.size() * 2);
  // Two chunks with the state carried: the same bytes.
  dstate.upload(state0.data(), state0.size() * 2);
  DGPP_CUDA_OK(cudaMemset(dout.p, 0, ref.size() * 2));
  conv(0, n1);
  conv(n1, n - n1);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got2 = down<uint16_t>(dout, ref.size());
  const std::vector<uint16_t> state2 = down<uint16_t>(dstate, ref_state.size());
  require_bitwise("chunked conv output", got2.data(), ref.data(), ref.size() * 2);
  require_bitwise("chunked conv state", state2.data(), ref_state.data(), ref_state.size() * 2);
  // In place on the residual.
  dstate.upload(state0.data(), state0.size() * 2);
  DevBuf dr2 = up(r);
  dgpp::qwen_ple_conv_bf16(static_cast<const uint16_t*>(dun.p), static_cast<const uint16_t*>(dgv.p),
                           static_cast<uint16_t*>(dstate.p), static_cast<const uint16_t*>(dw.p),
                           static_cast<const uint16_t*>(dr2.p), static_cast<uint16_t*>(dr2.p), n,
                           static_cast<int>(C), g.width, g.dilation, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got3 = down<uint16_t>(dr2, ref.size());
  require_bitwise("in-place conv output", got3.data(), ref.data(), ref.size() * 2);
}

namespace {

void end_to_end(int n, uint64_t seed) {
  Geo g;
  const int64_t C = g.channels();
  const int E = g.embed();
  const std::vector<int32_t> tokens = make_tokens(g, n, seed);
  const std::vector<int32_t> ids = host_ids(g, tokens);
  const std::vector<uint8_t> table = make_table(g, seed + 1);
  const std::vector<uint16_t> key_proj = random_bf16_normal(seed + 2, C * E, 0.06f);
  const std::vector<uint16_t> value_proj = random_bf16_normal(seed + 3, static_cast<int64_t>(g.hidden) * E, 0.06f);
  const std::vector<uint16_t> w_key = random_bf16_uniform(seed + 4, C, 0.3f);
  const std::vector<uint16_t> w_query = random_bf16_uniform(seed + 5, C, 0.3f);
  const std::vector<uint16_t> w_conv = random_bf16_uniform(seed + 6, C, 0.3f);
  const std::vector<uint16_t> conv_w = random_bf16_normal(seed + 7, C * g.width, 0.3f);
  const std::vector<uint16_t> state0 = random_bf16_normal(seed + 8, C * g.state_len(), 0.5f);
  const std::vector<uint16_t> r = random_bf16_normal(seed + 9, n * C, 1.0f);

  // The oracle.
  std::vector<uint16_t> e(static_cast<size_t>(n) * E);
  dgpp::qwen_ref::ple_gather(table.data(), 0, g.scale, ids.data(), n, g.heads, 0, g.heads, g.head_dim,
                             e.data());
  dgpp::qwen_ref::PleWeights pw;
  pw.key_proj = key_proj.data();
  pw.value_proj = value_proj.data();
  pw.norm_key = w_key.data();
  pw.norm_query = w_query.data();
  pw.norm_conv = w_conv.data();
  pw.conv = conv_w.data();
  std::vector<uint16_t> ref(static_cast<size_t>(n) * C), ref_state = state0;
  dgpp::qwen_ref::ple_forward(r.data(), e.data(), pw, ref_state.data(), ref.data(), n, g.hc, g.hidden,
                              E, g.width, g.dilation, g.eps);

  // The device chain.
  DevBuf dtab = up(table), dids = up(ids), dkp = up(key_proj), dvp = up(value_proj), dwk = up(w_key),
         dwq = up(w_query), dwc = up(w_conv), dcw = up(conv_w), dstate = up(state0), dr = up(r);
  DevBuf de(static_cast<size_t>(n) * E * 2), dkey(static_cast<size_t>(n) * C * 2),
      dkn(static_cast<size_t>(n) * C * 2), dval(static_cast<size_t>(n) * g.hidden * 2),
      dqn(static_cast<size_t>(n) * C * 2), dgv(static_cast<size_t>(n) * C * 2),
      dun(static_cast<size_t>(n) * C * 2);
  dgpp::CublasLtGemm gemm;
  const size_t ws_bytes = 8u << 20;
  DevBuf ws(ws_bytes);
  cudaStream_t st = test_stream();
  dgpp::qwen_ple_gather_bf16(static_cast<const uint8_t*>(dtab.p), 0, g.padded_rows, g.scale,
                             static_cast<const int32_t*>(dids.p), n, g.heads, 0, g.heads, g.head_dim,
                             static_cast<uint16_t*>(de.p), st);
  gemm.matmul(de.p, dkp.p, dkey.p, n, static_cast<int>(C), E, dgpp::DType::BF16, dgpp::GemmOut::BF16,
              static_cast<size_t>(E), ws.p, ws_bytes, st);
  dgpp::qwen_group_rmsnorm_bf16(dkey.p, dwk.p, dkn.p, n, g.hc, g.hidden, g.eps, st);
  gemm.matmul(de.p, dvp.p, dval.p, n, g.hidden, E, dgpp::DType::BF16, dgpp::GemmOut::BF16,
              static_cast<size_t>(E), ws.p, ws_bytes, st);
  dgpp::qwen_group_rmsnorm_bf16(dr.p, dwq.p, dqn.p, n, g.hc, g.hidden, g.eps, st);
  dgpp::qwen_ple_gate_bf16(static_cast<const uint16_t*>(dkn.p), static_cast<const uint16_t*>(dqn.p),
                           static_cast<const uint16_t*>(dval.p), static_cast<uint16_t*>(dgv.p), n,
                           g.hc, g.hidden, st);
  dgpp::qwen_group_rmsnorm_bf16(dgv.p, dwc.p, dun.p, n, g.hc, g.hidden, g.eps, st);
  dgpp::qwen_ple_conv_bf16(static_cast<const uint16_t*>(dun.p), static_cast<const uint16_t*>(dgv.p),
                           static_cast<uint16_t*>(dstate.p), static_cast<const uint16_t*>(dcw.p),
                           static_cast<const uint16_t*>(dr.p), static_cast<uint16_t*>(dr.p), n,
                           static_cast<int>(C), g.width, g.dilation, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dr, ref.size());
  const std::vector<uint16_t> got_state = down<uint16_t>(dstate, ref_state.size());
  const Stats s = compare_bf16(got, ref, 2);
  const Stats ss = compare_bf16(got_state, ref_state, 2);
  std::printf("[ .. ] ple end to end n=%d: out max_rel %.3g l2_rel %.3g mismatches %ld/%ld; state l2_rel %.3g\n",
              n, s.max_rel, s.l2_rel, s.mismatches, s.n, ss.l2_rel);
  require_bf16("ple out", s, 2e-3, 0.01);
  require_bf16("ple state", ss, 2e-3, 0.01);
}

}  // namespace

DGPP_TEST(qwen_ple_layer_end_to_end_at_decode_shape) { end_to_end(3, 40); }

DGPP_TEST(qwen_ple_layer_end_to_end_at_prefill_shape) { end_to_end(37, 50); }

int main() { return dgpp::test::run_all(); }

// The staged gather (2026-09-10, the mmap'ed table): rows the host gathered
// into a pinned buffer, converted on the device, are bitwise the device
// gather's own from the same table and ids.
DGPP_TEST(qwen_ple_staged_gather_is_bitwise_the_table_gather) {
  constexpr int rows = 4096, head_dim = 160, heads = 16, n = 37;
  std::vector<uint8_t> table(static_cast<size_t>(rows) * head_dim);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (auto& b : table) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; b = static_cast<uint8_t>(x & 0x7F); }
  std::vector<int32_t> ids(static_cast<size_t>(n) * heads);
  for (auto& id : ids) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; id = static_cast<int32_t>(x % rows); }
  const float scale = 0.0378f;
  // The host gather: row (t, hl) = table[ids[t, hl]].
  uint8_t* staged = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&staged), static_cast<size_t>(n) * heads * head_dim, cudaHostAllocDefault));
  for (int t = 0; t < n; ++t)
    for (int h = 0; h < heads; ++h)
      std::memcpy(staged + (static_cast<size_t>(t) * heads + h) * head_dim,
                  table.data() + static_cast<size_t>(ids[static_cast<size_t>(t) * heads + h]) * head_dim, head_dim);
  uint8_t* d_table = nullptr; int32_t* d_ids = nullptr; uint16_t* d_out = nullptr; uint16_t* d_out2 = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d_table, table.size()));
  DGPP_CUDA_OK(cudaMalloc(&d_ids, ids.size() * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_out, static_cast<size_t>(n) * heads * head_dim * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_out2, static_cast<size_t>(n) * heads * head_dim * 2));
  DGPP_CUDA_OK(cudaMemcpy(d_table, table.data(), table.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  dgpp::qwen_ple_gather_bf16(d_table, 0, rows, scale, d_ids, n, heads, 0, heads, head_dim, d_out, nullptr);
  dgpp::qwen_ple_gather_staged_bf16(staged, scale, n, heads, head_dim, d_out2, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> a(static_cast<size_t>(n) * heads * head_dim), b(a.size());
  DGPP_CUDA_OK(cudaMemcpy(a.data(), d_out, a.size() * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(b.data(), d_out2, b.size() * 2, cudaMemcpyDeviceToHost));
  require(std::memcmp(a.data(), b.data(), a.size() * 2) == 0, "the staged gather differs from the table gather");
  cudaFree(d_table); cudaFree(d_ids); cudaFree(d_out); cudaFree(d_out2); cudaFreeHost(staged);
  std::printf("[ OK ] the staged gather is bitwise the table gather over %d rows x %d heads\n", n, heads);
}
