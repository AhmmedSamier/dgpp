// DSA/MLA kernel tests (DESIGN §7.2, PLAN M3).
//
// Comparison philosophy carried over from M2:
//   * device-vs-device sequencing comparisons (multi-token vs single-token
//     decode, repeat runs) are bitwise;
//   * device-vs-host oracle comparisons are tolerance-based where libm or
//     FMA contraction can differ (compress, attention);
//   * the top-k selection is exact by construction: the host oracle mirrors
//     the kernel's contraction-proof logit arithmetic (separate mul/add in
//     the documented order plus the warp butterfly reduction), so pool
//     positions and expanded token rows are compared BITWISE over the fuzz
//     corpus — the M3 exit criterion.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/dsa.hpp"
#include "models/dsa_geometry.hpp"
#include "models/dsa_reference.hpp"

#include "kda_test_helpers.hpp"  // DevBuf, random_bf16_bits, compare helpers

namespace {

namespace dsa_ref = dgpp::dsa_ref;
using namespace dgpp;
using dgpp::bf16_bits_to_float;
using dgpp::DsaConfig;
using dgpp::DsaGeometry;
using dgpp::dsa_ref::HostState;
using dgpp::dsa_ref::HostWeights;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;
using dgpp::kda_test::compare_bf16;
using dgpp::kda_test::compare_abs_rel;
using dgpp::kda_test::DevBuf;
using dgpp::kda_test::random_bf16_bits;
using dgpp::kda_test::require_bf16;
using dgpp::kda_test::require_bitwise;
using dgpp::kda_test::require_rel;

uint32_t hash32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return uint32_t(x);
}

// Sane-magnitude random fp32 (exponent clamped well away from inf/nan).
float random_f32(uint64_t seed, int64_t i) {
  const uint32_t h = hash32(seed * 2 + 1 + uint64_t(i) * 2654435761u);
  const uint32_t bits = (h & 0x80000000u) | (uint32_t(112 + (h >> 27) % 12)
                                              << 23) |
                        (h & 0x7FFFFFu);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Mirror of the kernels' warp butterfly sum (parallel-step semantics).
float butterfly_sum_32(const float* in) {
  float a[32], b[32];
  std::memcpy(a, in, sizeof(a));
  for (int off = 16; off > 0; off >>= 1) {
    for (int i = 0; i < 32; ++i) b[i] = a[i] + a[i ^ off];
    std::memcpy(a, b, sizeof(a));
  }
  return a[0];
}

// Mirror of PrefillKeyFn: logit = butterfly_h((w[h]*ks)*dot[h]).
float prefill_logit_mirror(const float* dot_row_ptrs, const float* w,
                           float ks) {
  float contrib[32];
  for (int h = 0; h < 32; ++h)
    contrib[h] = (w[h] * ks) * dot_row_ptrs[h];
  return butterfly_sum_32(contrib);
}

// Host simulation of the tail ring + pool writes for the state machinery
// tests (mirrors dsa_ref::layer_forward's cache updates without the GEMM
// path).
struct RingSim {
  std::vector<uint16_t> tail;    // [2, kpool, dim]
  std::vector<uint8_t> index_k;  // [max_pools, dim]
  std::vector<float> index_scale;
  int kpool, dim;

  void seed(const std::vector<uint16_t>& k, const std::vector<uint16_t>& gate,
            int64_t token_start, int64_t tokens) {
    const int64_t end = token_start + tokens;
    for (int64_t pos = std::max<int64_t>(0, end - kpool); pos < end; ++pos) {
      const int slot = int(pos % kpool);
      std::memcpy(&tail[size_t(slot) * dim], &k[size_t(pos - token_start) * dim],
                  dim * 2);
      std::memcpy(&tail[size_t(kpool + slot) * dim],
                  &gate[size_t(pos - token_start) * dim], dim * 2);
    }
  }

  // One decode token at position pos with raw k/gate rows.
  void decode_token(const uint16_t* krow, const uint16_t* gamerow,
                    const float* ape, int64_t pos, int64_t pool_slot) {
    const int slot = int(pos % kpool);
    if (slot == kpool - 1) {
      const int64_t pool_start = pos - (kpool - 1);
      // Assemble the pool's k/gate from the ring with the current override.
      std::vector<uint16_t> pk(size_t(kpool) * dim), pg(size_t(kpool) * dim);
      for (int s = 0; s < kpool; ++s) {
        const int ring = int((pool_start + s) % kpool);
        const bool cur = (s == kpool - 1);
        std::memcpy(&pk[size_t(s) * dim],
                    cur ? krow : &tail[size_t(ring) * dim], dim * 2);
        std::memcpy(&pg[size_t(s) * dim],
                    cur ? gamerow : &tail[size_t(kpool + ring) * dim], dim * 2);
      }
      dsa_ref::compress_pool<float>(pk.data(), pg.data(), ape, kpool, dim,
                                    &index_k[size_t(pool_slot) * dim],
                                    &index_scale[size_t(pool_slot)]);
    }
    std::memcpy(&tail[size_t(slot) * dim], krow, dim * 2);
    std::memcpy(&tail[size_t(kpool + slot) * dim], gamerow, dim * 2);
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// fwht + fp8 quant: the pinned boundary sequence is portable, so this is
// bitwise against the host reference.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_fwht_quant_matches_host_reference) {
  const int rows = 257;
  auto q = random_bf16_bits(11, rows * 128, -3, 1);
  std::vector<uint8_t> want8(size_t(rows) * 128);
  std::vector<float> want_scale(size_t(rows), 0.0f);
  dsa_ref::fwht128_quant_fp8<float>(q.data(), rows, 128, want8.data(),
                                    want_scale.data());

  DevBuf dq(size_t(rows) * 128 * 2), d8(size_t(rows) * 128),
      ds(size_t(rows) * 4);
  dq.upload(q.data(), q.size() * 2);
  dsa_fwht_quant_rows(dq.p, rows, d8.p, static_cast<float*>(ds.p), 0);

  std::vector<uint8_t> got8(size_t(rows) * 128);
  std::vector<float> got_scale(size_t(rows), 0.0f);
  d8.download(got8.data(), got8.size());
  ds.download(got_scale.data(), got_scale.size() * 4);
  require_bitwise("fwht fp8 bits", got8.data(), want8.data(), got8.size());
  require_bitwise("fwht scales", got_scale.data(), want_scale.data(),
                  want_scale.size() * 4);
}

// ---------------------------------------------------------------------------
// Pool compression. Two exactly-comparable modes make the real correctness
// risks (slot mapping, pool assembly, softmax indexing) bitwise-testable
// despite expf/FMA ulp divergence between host and device:
//   * hard-max gate: the dominant slot's score leads by 100, so the other
//     slots' softmax weights underflow to subnormals (~1e-43) and the
//     compressed pool is bitwise fwht(k_dominant) on both sides;
//   * random gate: tolerance smoke test (fp8 scale-bin flips from expf ulps
//     legitimately reach ~1.5 quanta).
// Exercises a non-identity block table.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_kpool_compress_matches_host_reference) {
  const int kpool = 4, dim = 128, n_pools = 96;
  const int tokens = n_pools * kpool;
  auto k = random_bf16_bits(21, tokens * dim, -2, 1);
  auto gate_random = random_bf16_bits(22, tokens * dim, -2, 1);
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(23, i) * 0.1f;

  // Hard-max gate: dominant slot varies per pool so the assembly is probed.
  auto gate = gate_random;
  for (int j = 0; j < n_pools; ++j) {
    const int dom = j % kpool;
    for (int s = 0; s < kpool; ++s)
      for (int d = 0; d < dim; ++d)
        gate[size_t(j * kpool + s) * dim + d] =
            float_to_bf16_bits((s == dom) ? 50.0f : -50.0f);
  }

  // Block table: 32 pools per block, shuffled physical assignment.
  const int ppb = 32, n_blocks = n_pools / ppb;
  std::vector<int32_t> bt(n_blocks);
  for (int b = 0; b < n_blocks; ++b) bt[b] = (b * 7 + 3) % n_blocks;
  const int total_slots = n_blocks * ppb;

  auto run_device = [&](const std::vector<uint16_t>& g,
                        std::vector<uint8_t>& out_k,
                        std::vector<float>& out_s) {
    DevBuf dk(k.size() * 2), dgate(g.size() * 2), dape(ape.size() * 4),
        dbt(bt.size() * 4), dki(size_t(total_slots) * dim),
        dks(size_t(total_slots) * 4);
    dk.upload(k.data(), k.size() * 2);
    dgate.upload(g.data(), g.size() * 2);
    dape.upload(ape.data(), ape.size() * 4);
    dbt.upload(bt.data(), bt.size() * 4);
    dsa_kpool_compress_write(dk.p, dim, dgate.p, dim,
                             static_cast<const float*>(dape.p),
                             static_cast<const int32_t*>(dbt.p), ppb,
                             /*first_pool=*/0, n_pools, dki.p,
                             static_cast<float*>(dks.p), kpool, dim, 0);
    out_k.assign(size_t(total_slots) * dim, 0);
    out_s.assign(size_t(total_slots), 0.0f);
    dki.download(out_k.data(), out_k.size());
    dks.download(out_s.data(), out_s.size() * 4);
  };

  std::vector<uint8_t> got_hard, got_rand;
  std::vector<float> got_s_hard, got_s_rand;
  run_device(gate, got_hard, got_s_hard);
  run_device(gate_random, got_rand, got_s_rand);

  for (int j = 0; j < n_pools; ++j) {
    std::vector<uint8_t> w8(dim), w8r(dim);
    float ws = 0, wsr = 0;
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate[size_t(j * kpool) * dim], ape.data(),
                                  kpool, dim, w8.data(), &ws);
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate_random[size_t(j * kpool) * dim],
                                  ape.data(), kpool, dim, w8r.data(), &wsr);
    const int64_t slot = int64_t(bt[j / ppb]) * ppb + (j % ppb);
    // Hard-max: bitwise.
    if (std::memcmp(&got_hard[size_t(slot) * dim], w8.data(), dim) != 0 ||
        got_s_hard[size_t(slot)] != ws)
      throw std::runtime_error("compress hard-max bitwise mismatch pool=" +
                               std::to_string(j));
    // Random gate: tolerance smoke test. fp8 scale ties to the row absmax, so
    // each side's quantization error is bounded by absmax/16; a shared budget
    // of absmax/8 covers a full scale-bin flip and stays far below wrong-pool
    // garbage (the hard-max mode above carries the assembly correctness).
    float rowmax = 0.0f;
    for (int d = 0; d < dim; ++d) {
      const float gv =
          fp8_e4m3_bits_to_float(got_rand[size_t(slot) * dim + d]) *
          got_s_rand[size_t(slot)];
      const float wv = fp8_e4m3_bits_to_float(w8r[d]) * wsr;
      rowmax = std::max(rowmax, std::max(std::fabs(gv), std::fabs(wv)));
    }
    for (int d = 0; d < dim; ++d) {
      const float gv =
          fp8_e4m3_bits_to_float(got_rand[size_t(slot) * dim + d]) *
          got_s_rand[size_t(slot)];
      const float wv = fp8_e4m3_bits_to_float(w8r[d]) * wsr;
      if (std::fabs(gv - wv) > rowmax * 0.135f + 1e-6f)
        throw std::runtime_error("compress mismatch pool=" + std::to_string(j) +
                                 " dim=" + std::to_string(d) + " gv=" +
                                 std::to_string(gv) + " wv=" +
                                 std::to_string(wv));
    }
  }
}

// ---------------------------------------------------------------------------
// Tail seed: the reference's ahead-check rule over a multi-request batch.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_tail_seed_matches_reference_rule) {
  const int kpool = 4, dim = 128;
  // Two requests: 6 tokens and 3 tokens in one flattened batch.
  const int t0 = 6, t1 = 3, tokens = t0 + t1;
  auto k = random_bf16_bits(31, tokens * dim, -2, 1);
  auto gate = random_bf16_bits(32, tokens * dim, -2, 1);
  std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 0, 1, 1, 1};
  std::vector<int64_t> pos = {0, 1, 2, 3, 4, 5, 0, 1, 2};

  const size_t tail_bytes = 2ull * 2 * kpool * dim * 2;  // 2 requests
  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dri(req_ids.size() * 4),
      dpos(pos.size() * 8), dtail(tail_bytes);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  std::vector<uint16_t> zero(tail_bytes / 2, 0);
  dtail.upload(zero.data(), tail_bytes);
  dsa_kpool_tail_seed(dk.p, dim, dgate.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), tokens, dtail.p,
                      kpool, dim, 0);

  std::vector<uint16_t> got(tail_bytes / 2);
  dtail.download(got.data(), tail_bytes);
  // Expected: request 0's last 4 tokens (pos 2..5, ring slots 2,3,0,1) and
  // request 1's 3 tokens (pos 0..2, slots 0,1,2).
  auto at = [&](int req, int half, int slot) {
    return &got[(size_t(req) * 2 * kpool + half * kpool + slot) * dim];
  };
  auto row = [&](int t) { return &k[size_t(t) * dim]; };
  auto grow = [&](int t) { return &gate[size_t(t) * dim]; };
  require_bitwise("r0 k slot2", at(0, 0, 2), row(2), dim * 2);
  require_bitwise("r0 k slot3", at(0, 0, 3), row(3), dim * 2);
  require_bitwise("r0 k slot0", at(0, 0, 0), row(4), dim * 2);
  require_bitwise("r0 k slot1", at(0, 0, 1), row(5), dim * 2);
  require_bitwise("r0 g slot0", at(0, 1, 0), grow(4), dim * 2);
  require_bitwise("r1 k slot1", at(1, 0, 1), row(7), dim * 2);
  require_bitwise("r1 k slot2", at(1, 0, 2), row(8), dim * 2);
  // Ring slots never touched stay zero.
  require_bitwise("r1 slot3 untouched", at(1, 0, 3), zero.data(), dim * 2);
}

// ---------------------------------------------------------------------------
// Decode ring continuation: chunked prefill (ending mid-pool) then
// single-token decode steps; the pool written across the boundary must
// match the host simulation, and multi-token decode must equal
// single-token decode bitwise (the ordered-loop invariant).
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_decode_update_ring_continuation_matches_host) {
  const int kpool = 4, dim = 128;
  const int64_t prefill_tokens = 10;  // pools 0,1 complete; 2-token tail
  const int64_t decode_tokens = 16;   // completes pools 2..5 and starts 6
  const int max_pools = 8;
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(41, i) * 0.1f;

  auto gen = [&](uint64_t seed, int64_t n) {
    return random_bf16_bits(seed, n * dim, -2, 1);
  };
  auto pk_rows = gen(51, prefill_tokens);
  auto dg_rows = gen(52, decode_tokens);

  // Hard-max gates with the dominant slot = each pool's completing token
  // (pos % kpool == 3): the compressed pool is bitwise fwht(k[4j+3]) on
  // host and device, so any ring bookkeeping error — stale stash, wrong
  // ring slot, missing is_current override — fails exactly.
  auto hardmax_gate = [&](std::vector<uint16_t>& g, int64_t token_start) {
    for (int64_t t = 0; t < int64_t(g.size()) / dim; ++t) {
      const bool dom = ((token_start + t) % kpool) == kpool - 1;
      for (int d = 0; d < dim; ++d)
        g[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
    }
  };
  auto pg_rows = gen(53, prefill_tokens);
  auto ddg_rows = gen(54, decode_tokens);
  hardmax_gate(pg_rows, 0);
  hardmax_gate(ddg_rows, prefill_tokens);

  // ---- host simulation ----
  RingSim sim{{/*tail=*/std::vector<uint16_t>(2ull * kpool * dim, 0)},
              {/*index_k=*/std::vector<uint8_t>(size_t(max_pools) * dim, 0)},
              {/*index_scale=*/std::vector<float>(max_pools, 0)}, kpool, dim};
  for (int j = 0; j < prefill_tokens / kpool; ++j)
    dsa_ref::compress_pool<float>(&pk_rows[size_t(j * kpool) * dim],
                                  &pg_rows[size_t(j * kpool) * dim],
                                  ape.data(), kpool, dim,
                                  &sim.index_k[size_t(j) * dim],
                                  &sim.index_scale[j]);
  sim.seed(pk_rows, pg_rows, 0, prefill_tokens);
  for (int64_t t = 0; t < decode_tokens; ++t) {
    const int64_t pos = prefill_tokens + t;
    sim.decode_token(&dg_rows[size_t(t) * dim], &ddg_rows[size_t(t) * dim],
                     ape.data(), pos, pos / kpool);
  }

  // ---- device: prefill compress + seed, then decode one token per call ----
  DevBuf dkp(pk_rows.size() * 2), dgp(pg_rows.size() * 2), dape(ape.size() * 4),
      dbt(4), dki(size_t(max_pools) * dim), dks(max_pools * 4),
      dtail(2ull * kpool * dim * 2), dpos(8), dspans(8);
  // Zero-init every cache buffer: unwritten pools stay defined so the
  // downloads are initcheck-clean and comparisons cannot read garbage.
  dki.upload(std::vector<uint8_t>(size_t(max_pools) * dim, 0).data(),
             size_t(max_pools) * dim);
  dks.upload(std::vector<float>(max_pools, 0.0f).data(), max_pools * 4);
  dkp.upload(pk_rows.data(), pk_rows.size() * 2);
  dgp.upload(pg_rows.data(), pg_rows.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  int32_t bt0 = 0;
  dbt.upload(&bt0, 4);
  std::vector<uint16_t> zt(2ull * kpool * dim, 0);
  dtail.upload(zt.data(), zt.size() * 2);

  dsa_kpool_compress_write(dkp.p, dim, dgp.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), max_pools, 0,
                           int(prefill_tokens / kpool), dki.p,
                           static_cast<float*>(dks.p), kpool, dim, 0);
  // Per-token request ids and positions for the whole prefill batch (the
  // kernel indexes req_ids[i]/pos[i] for every token, not just the tail).
  std::vector<int32_t> seed_req(prefill_tokens, 0);
  std::vector<int64_t> seed_pos(prefill_tokens);
  for (int64_t t = 0; t < prefill_tokens; ++t) seed_pos[size_t(t)] = t;
  DevBuf dri(seed_req.size() * 4), dseed_pos(seed_pos.size() * 8);
  dri.upload(seed_req.data(), seed_req.size() * 4);
  dseed_pos.upload(seed_pos.data(), seed_pos.size() * 8);
  dsa_kpool_tail_seed(dkp.p, dim, dgp.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dseed_pos.p),
                      prefill_tokens, dtail.p, kpool, dim, 0);

  DevBuf dkd(dg_rows.size() * 2), dgd(ddg_rows.size() * 2);
  dkd.upload(dg_rows.data(), dg_rows.size() * 2);
  dgd.upload(ddg_rows.data(), ddg_rows.size() * 2);
  std::vector<int32_t> spans = {0, 1};
  dspans.upload(spans.data(), 8);
  for (int64_t t = 0; t < decode_tokens; ++t) {
    const int64_t pos = prefill_tokens + t;
    dpos.upload(&pos, 8);
    dsa_kpool_decode_update(
        static_cast<const uint16_t*>(dkd.p) + size_t(t) * dim, dim,
        static_cast<const uint16_t*>(dgd.p) + size_t(t) * dim, dim,
        static_cast<const float*>(dape.p),
        static_cast<const int64_t*>(dpos.p),
        static_cast<const int32_t*>(dspans.p), 1,
        static_cast<const int32_t*>(dbt.p), 1, dtail.p, dki.p,
        static_cast<float*>(dks.p), max_pools, kpool, dim, 0);
  }

  std::vector<uint16_t> got_tail(2ull * kpool * dim);
  dtail.download(got_tail.data(), got_tail.size() * 2);
  std::vector<uint8_t> got_k(size_t(max_pools) * dim);
  std::vector<float> got_s(max_pools, 0.0f);
  dki.download(got_k.data(), got_k.size());
  dks.download(got_s.data(), got_s.size() * 4);
  require_bitwise("tail ring", got_tail.data(), sim.tail.data(),
                  sim.tail.size() * 2);
  // Hard-max gates: pool contents are bitwise comparable.
  const int64_t pools_written =
      std::min<int64_t>(max_pools, (prefill_tokens + decode_tokens) / kpool);
  for (int64_t j = 0; j < pools_written; ++j) {
    if (std::memcmp(&got_k[size_t(j) * dim], &sim.index_k[size_t(j) * dim],
                    dim) != 0 ||
        got_s[j] != sim.index_scale[j])
      throw std::runtime_error("decode pool bitwise mismatch j=" +
                               std::to_string(j));
  }

  // ---- multi-token decode == single-token decode (bitwise) ----
  DevBuf dki2(size_t(max_pools) * dim), dks2(max_pools * 4),
      dtail2(2ull * kpool * dim * 2);
  dki2.upload(std::vector<uint8_t>(size_t(max_pools) * dim, 0).data(),
              size_t(max_pools) * dim);
  dks2.upload(std::vector<float>(max_pools, 0.0f).data(), max_pools * 4);
  dtail2.upload(std::vector<uint16_t>(2ull * kpool * dim, 0).data(),
                2ull * kpool * dim * 2);
  dsa_kpool_compress_write(dkp.p, dim, dgp.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), max_pools, 0,
                           int(prefill_tokens / kpool), dki2.p,
                           static_cast<float*>(dks2.p), kpool, dim, 0);
  dsa_kpool_tail_seed(dkp.p, dim, dgp.p, dim,
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dseed_pos.p),
                      prefill_tokens, dtail2.p, kpool, dim, 0);
  std::vector<int64_t> pos_all(decode_tokens);
  for (int64_t t = 0; t < decode_tokens; ++t) pos_all[t] = prefill_tokens + t;
  DevBuf dpos_all(pos_all.size() * 8);
  dpos_all.upload(pos_all.data(), pos_all.size() * 8);
  std::vector<int32_t> spans_all = {0, int(decode_tokens)};
  dspans.upload(spans_all.data(), 8);
  dsa_kpool_decode_update(dkd.p, dim, dgd.p, dim,
                          static_cast<const float*>(dape.p),
                          static_cast<const int64_t*>(dpos_all.p),
                          static_cast<const int32_t*>(dspans.p), 1,
                          static_cast<const int32_t*>(dbt.p), 1, dtail2.p,
                          dki2.p, static_cast<float*>(dks2.p), max_pools,
                          kpool, dim, 0);
  std::vector<uint8_t> got_k2(size_t(max_pools) * dim);
  std::vector<float> got_s2(max_pools);
  std::vector<uint16_t> got_tail2(2ull * kpool * dim);
  dki2.download(got_k2.data(), got_k2.size());
  dks2.download(got_s2.data(), got_s2.size() * 4);
  dtail2.download(got_tail2.data(), got_tail2.size() * 2);
  require_bitwise("multi-token pools", got_k2.data(), got_k.data(),
                  got_k.size());
  require_bitwise("multi-token scales", got_s2.data(), got_s.data(),
                  got_s.size() * 4);
  require_bitwise("multi-token tail", got_tail2.data(), got_tail.data(),
                  got_tail.size() * 2);
}

// ---------------------------------------------------------------------------
// Prefill select: the >=1000-case bitwise fuzz around pool boundaries
// (M3 exit criterion), using synthetic dots so the host mirror and the
// kernel see identical logits.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_select_prefill_bitwise_fuzz) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int select_k = g.select_k, kpool = cfg.index_kpool;
  const int max_selected = g.max_selected;
  const int heads = cfg.index_n_heads;

  // Case corpus: lengths 0..7, every residue around 4k boundaries up to
  // 2052, and hashed randoms — >= 1000 distinct (pos, pool-count) cases.
  std::vector<std::pair<int64_t, int64_t>> cases;  // (pos, n_pools available)
  for (int p = 0; p <= 7; ++p) cases.push_back({p, (p / 4) + 3});
  for (int base = 1; base <= 515; ++base)
    for (int r = -1; r <= 2; ++r) {
      const int64_t pos = int64_t(base) * 4 + r;
      if (pos >= 0) cases.push_back({pos, pos / 4 + 1});
    }
  for (int i = 0; cases.size() < 1100; ++i) {
    const int64_t pos = hash32(9000 + i) % 9000;
    cases.push_back({pos, pos / 4 + 1 + hash32(7000 + i) % 4});
  }

  int checked = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const int64_t pos = cases[ci].first;
    const int64_t n_pools = cases[ci].second;
    const int64_t visible = (pos + 1) / kpool;
    const int64_t avail = std::max<int64_t>(n_pools, visible);
    if (avail <= 0) {
      // No pools at all: only the tail token list.
    }
    // Random dots [heads, avail], weights [heads], scales [avail].
    std::vector<float> dots(size_t(heads) * avail);
    std::vector<float> w(size_t(heads), 0.0f);
    std::vector<float> ks(size_t(avail), 0.0f);
    for (auto& v : dots) v = random_f32(100 + ci, &v - dots.data());
    for (auto& v : w) v = random_f32(200 + ci, &v - w.data());
    for (auto& v : ks) v = random_f32(300 + ci, &v - ks.data());
    std::vector<int64_t> posv = {pos};

    DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
        dpos(8), dtopk(size_t(max_selected) * 4), dcnt(4);
    dd.upload(dots.data(), dots.size() * 4);
    dw.upload(w.data(), w.size() * 4);
    dks.upload(ks.data(), ks.size() * 4);
    dpos.upload(posv.data(), 8);
    dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                       static_cast<const float*>(dw.p),
                       static_cast<const float*>(dks.p),
                       static_cast<const int64_t*>(dpos.p), 1, avail, heads,
                       select_k, kpool, max_selected,
                       static_cast<int32_t*>(dtopk.p),
                       static_cast<int32_t*>(dcnt.p), 0);

    // Host mirror: identical logit arithmetic, then the pinned selection.
    std::vector<float> logits(size_t(visible), 0.0f);
    for (int64_t j = 0; j < visible; ++j) {
      float dot_h[32];
      for (int h = 0; h < heads; ++h) dot_h[h] = dots[size_t(h) * avail + j];
      logits[size_t(j)] = prefill_logit_mirror(dot_h, w.data(), ks[size_t(j)]);
    }
    std::vector<int32_t> pool_ids(select_k, 0);
    const int n_sel =
        dsa_ref::select_pools(logits.data(), visible, select_k, pool_ids.data());
    std::vector<int32_t> want(size_t(max_selected), -1);
    const int n_tok =
        dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos, kpool,
                                    max_selected, want.data());

    std::vector<int32_t> got(size_t(max_selected), -12345);
    int32_t got_cnt = -1;
    dtopk.download(got.data(), got.size() * 4);
    dcnt.download(&got_cnt, 4);
    if (got_cnt != n_tok)
      throw std::runtime_error("count mismatch case " + std::to_string(ci) +
                               " pos=" + std::to_string(pos) + " got=" +
                               std::to_string(got_cnt) + " want=" +
                               std::to_string(n_tok));
    if (std::memcmp(got.data(), want.data(), max_selected * 4) != 0) {
      std::string detail;
      for (int i = 0; i < max_selected; ++i)
        if (got[size_t(i)] != want[size_t(i)])
          detail += " [" + std::to_string(i) + "] got=" +
                    std::to_string(got[size_t(i)]) + " want=" +
                    std::to_string(want[size_t(i)]);
      throw std::runtime_error("token mismatch case " + std::to_string(ci) +
                               " pos=" + std::to_string(pos) + detail);
    }
    ++checked;
  }
  if (checked < 1000)
    throw std::runtime_error("fuzz corpus too small: " + std::to_string(checked));
}

// ---------------------------------------------------------------------------
// Fused decode select: same bitwise oracle, logits computed inline from the
// fp8 cache; also repeat-run determinism.
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_select_decode_fused_bitwise) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int select_k = g.select_k, kpool = cfg.index_kpool;
  const int max_selected = g.max_selected;
  const int heads = cfg.index_n_heads, dim = cfg.index_head_dim;
  const int n_pools = 6000;  // > select_k: real sparse selection
  const int64_t pos = 4 * n_pools + 2;  // visible = n_pools, tail 2

  // q_fp8 with sane magnitudes and matching scales; random fp8 cache.
  std::vector<uint16_t> qb = random_bf16_bits(61, heads * dim, -2, 1);
  std::vector<uint8_t> q8(size_t(heads) * dim);
  std::vector<float> q_scale(size_t(heads), 0.0f);
  dsa_ref::fwht128_quant_fp8<float>(qb.data(), heads, dim, q8.data(),
                                    q_scale.data());
  std::vector<float> w_raw(heads, 0.0f);
  for (auto& v : w_raw) v = random_f32(62, &v - w_raw.data()) * 0.01f;
  std::vector<float> w(heads, 0.0f);
  const float logit_scale =
      float(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
  for (int h = 0; h < heads; ++h)
    w[h] = (w_raw[h] * q_scale[h]) * logit_scale;

  std::vector<uint8_t> k8(size_t(n_pools) * dim);
  std::vector<float> ks(n_pools, 0.0f);
  // Random fp8 bytes, remapping the two NaN encodings (0x7F/0xFF) to max
  // finite: real cache rows are always finite (the saturating encoder never
  // mints NaN), and the select spec assumes finite logits.
  for (auto& v : k8) {
    v = uint8_t(hash32(63 + (&v - k8.data())) & 0xFF);
    if ((v & 0x7Fu) == 0x7Fu) v ^= 0x01u;
  }
  for (auto& v : ks) v = std::fabs(random_f32(64, &v - ks.data())) * 0.05f;

  DevBuf dq8(q8.size()), dks_cache(ks.size() * 4), dw(w.size() * 4),
      dki(k8.size()), dpos(8), dri(4), dbt(4), dtopk(size_t(max_selected) * 4),
      dcnt(4), dpart(size_t(48) * 1 * select_k * 8), dctr(4);
  dq8.upload(q8.data(), q8.size());
  dw.upload(w.data(), w.size() * 4);
  dki.upload(k8.data(), k8.size());
  dks_cache.upload(ks.data(), ks.size() * 4);
  std::vector<int64_t> posv = {pos};
  std::vector<int32_t> req0 = {0}, bt0 = {0};
  dpos.upload(posv.data(), 8);
  dri.upload(req0.data(), 4);
  dbt.upload(bt0.data(), 4);
  int32_t zero32 = 0;
  dctr.upload(&zero32, 4);

  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_cache.p), n_pools, heads,
                    dim, select_k, kpool, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p),
                    static_cast<uint64_t*>(dpart.p),
                    static_cast<int32_t*>(dctr.p), 48, 0);

  // Host mirror: identical inline dot arithmetic.
  std::vector<float> logits(n_pools, 0.0f);
  for (int j = 0; j < n_pools; ++j) {
    float contrib[32];
    for (int h = 0; h < 32; ++h) {
      float partial = 0.0f;
      for (int d = 0; d < 128; ++d)
        partial = partial +
                  fp8_e4m3_bits_to_float(q8[size_t(h) * 128 + d]) *
                      fp8_e4m3_bits_to_float(k8[size_t(j) * 128 + d]);
      contrib[h] = (w[h] * ks[j]) * partial;
    }
    logits[size_t(j)] = butterfly_sum_32(contrib);
  }
  std::vector<int32_t> pool_ids(select_k, 0);
  const int n_sel =
      dsa_ref::select_pools(logits.data(), n_pools, select_k, pool_ids.data());
  std::vector<int32_t> want(size_t(max_selected), -1);
  dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos, kpool, max_selected,
                              want.data());

  std::vector<int32_t> got(size_t(max_selected), -12345);
  int32_t got_cnt = -1;
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(&got_cnt, 4);
  if (got_cnt != int(n_sel * kpool + (pos + 1) % kpool))
    throw std::runtime_error("decode select count");
  if (std::memcmp(got.data(), want.data(), max_selected * 4) != 0) {
    // Diagnostics: which pools are selected by one side but not the other,
    // and their host-mirror logits (a boundary flip shows ulp-close logits;
    // a kernel bug shows a clearly-better pool missing).
    std::vector<int32_t> gset, wset;
    for (int i = 0; i < max_selected; ++i) {
      if (got[size_t(i)] >= 0) gset.push_back(got[size_t(i)] / kpool);
      if (want[size_t(i)] >= 0) wset.push_back(want[size_t(i)] / kpool);
    }
    std::sort(gset.begin(), gset.end());
    std::sort(wset.begin(), wset.end());
    std::string detail;
    for (int p : gset)
      if (!std::binary_search(wset.begin(), wset.end(), p))
        detail += " device-only pool " + std::to_string(p) + " logit " +
                  std::to_string(logits[size_t(p)]) + ";";
    for (int p : wset)
      if (!std::binary_search(gset.begin(), gset.end(), p))
        detail += " host-only pool " + std::to_string(p) + " logit " +
                  std::to_string(logits[size_t(p)]) + ";";
    std::vector<float> sorted = logits;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    detail += " 512th logit=" + std::to_string(sorted[511]) + " 513th=" +
              std::to_string(sorted[512]);
    throw std::runtime_error("decode select tokens mismatch: " + detail);
  }

  // Determinism: a second run must be identical.
  std::vector<int32_t> got2(size_t(max_selected), -12345);
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_cache.p), n_pools, heads,
                    dim, select_k, kpool, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p),
                    static_cast<uint64_t*>(dpart.p),
                    static_cast<int32_t*>(dctr.p), 48, 0);
  dtopk.download(got2.data(), got2.size() * 4);
  require_bitwise("decode select repeat", got2.data(), got.data(),
                  got.size() * 4);
}

// ---------------------------------------------------------------------------
// Absorbed attention chain vs the host oracle (tolerance: online-softmax
// rescaling vs one-shot).
// ---------------------------------------------------------------------------
DGPP_TEST(dsa_absorbed_attention_matches_host_reference) {
  const DsaConfig cfg{};
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int rows = 3, local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  const int max_selected = g.max_selected;

  auto q = random_bf16_bits(71, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(72, int64_t(local_heads) * (nope + v) * kv_lora,
                                -2, 1);
  const int cnt = 37;
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts(rows, cnt);
  std::vector<int32_t> req_ids(rows, 0);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < cnt; ++i)
      tokens[size_t(r) * max_selected + i] =
          int32_t((r * 131 + i * 17) % 200);

  // Latent cache: 2 blocks of block_tokens rows.
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  auto latent =
      random_bf16_bits(73, int64_t(total_tokens) * kv_lora, -2, 1);
  std::vector<int32_t> bt = {1, 0};  // reversed on purpose

  const int n_split = 4;
  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8),
      dqt(size_t(rows) * local_heads * kv_lora * 2),
      dm(size_t(rows) * n_split * local_heads * 4),
      dl(size_t(rows) * n_split * local_heads * 4),
      dc(size_t(rows) * n_split * local_heads * kv_lora * 4),
      dc_out(size_t(rows) * local_heads * kv_lora * 4),
      dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dlat.upload(latent.data(), latent.size() * 2);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);

  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
  const float scale = 1.0f / std::sqrt(float(nope));
  dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                   static_cast<const int32_t*>(dtopk.p), max_selected,
                   static_cast<const int32_t*>(dcnt.p), rows, n_split,
                   local_heads, kv_lora, block_tokens,
                   static_cast<const int32_t*>(dbt.p), n_blocks, scale,
                   static_cast<float*>(dm.p), static_cast<float*>(dl.p),
                   static_cast<float*>(dc.p), 0);
  dsa_attn_combine(static_cast<const float*>(dm.p),
                   static_cast<const float*>(dl.p),
                   static_cast<const float*>(dc.p), rows, n_split, local_heads,
                   kv_lora, static_cast<float*>(dc_out.p), 0);
  dsa_vout_gemm(dc_out.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora,
                0);

  // Host oracle: build the physical latent view via the block table.
  std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int32_t blk = bt[size_t(tok / block_tokens)];
    const int64_t phys = int64_t(blk) * block_tokens + (tok % block_tokens);
    std::memcpy(&phys_latent[size_t(tok) * kv_lora],
                &latent[size_t(phys) * kv_lora], kv_lora * 2);
  }
  std::vector<uint16_t> want(size_t(rows) * local_heads * v);
  for (int r = 0; r < rows; ++r)
    dsa_ref::absorbed_attn<float>(&q[size_t(r) * local_heads * nope],
                                  phys_latent.data(), kv_lora,
                                  &tokens[size_t(r) * max_selected], cnt,
                                  kv_b.data(), local_heads, nope, v, kv_lora,
                                  scale, &want[size_t(r) * local_heads * v]);
  std::vector<uint16_t> got(size_t(rows) * local_heads * v);
  dout.download(got.data(), got.size() * 2);
  require_bf16("absorbed attention vs host f32", compare_bf16(got, want, 8),
               0.01, 0.006);
}

// ---------------------------------------------------------------------------
// Decode-select scenarios shared by the MTP-row, grid-invariance,
// long-context, and graph-replay tests. The host expectation mirrors the
// kernel's contraction-proof arithmetic exactly (see the bitwise test above).
// ---------------------------------------------------------------------------

struct DecodeSelScenario {
  int rows = 0;
  int64_t n_pools = 0;  // complete pools resident in the cache
  std::vector<uint8_t> q8, k8;
  std::vector<float> w, ks;
  std::vector<int64_t> pos;
  std::vector<int32_t> req_ids;
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  int heads = 32, dim = 128;

  void init(uint64_t seed, const std::vector<int64_t>& positions) {
    pos = positions;
    rows = int(pos.size());
    n_pools = 8;
    for (int64_t p : pos) n_pools = std::max(n_pools, (p + 1) / 4 + 8);

    std::vector<uint16_t> qb =
        random_bf16_bits(seed, int64_t(rows) * heads * dim, -2, 1);
    q8.assign(size_t(rows) * heads * dim, 0);
    w.assign(size_t(rows) * heads, 0.0f);
    std::vector<float> q_scale(size_t(rows) * heads, 0.0f);
    dsa_ref::fwht128_quant_fp8<float>(qb.data(), rows * heads, dim,
                                      q8.data(), q_scale.data());
    const float logit_scale =
        float(std::pow(128.0, -0.5) * std::pow(32.0, -0.5));
    for (int64_t i = 0; i < int64_t(rows) * heads; ++i) {
      const float w_raw = random_f32(seed + 1, i) * 0.01f;
      w[size_t(i)] = (w_raw * q_scale[size_t(i)]) * logit_scale;
    }
    k8.assign(size_t(n_pools) * dim, 0);
    for (auto& v : k8) {
      v = uint8_t(hash32(seed + 2 + (&v - k8.data())) & 0xFF);
      if ((v & 0x7Fu) == 0x7Fu) v ^= 0x01u;  // no NaN encodings
    }
    ks.assign(size_t(n_pools), 0.0f);
    for (auto& v : ks) v = std::fabs(random_f32(seed + 3, &v - ks.data())) * 0.05f;
    req_ids.assign(size_t(rows), 0);
  }

  // Runs the fused decode select on the default stream (or a captured
  // graph) and returns topk rows + counts.
  void run(int grid_blocks, std::vector<int32_t>& topk,
           std::vector<int32_t>& counts, cudaStream_t stream = 0,
           uint64_t* partial_ws = nullptr, int32_t* counter_ws = nullptr) {
    const int max_selected = g.max_selected;
    DevBuf dq8(q8.size()), dw(w.size() * 4), dki(k8.size()),
        dks_c(ks.size() * 4), dpos(pos.size() * 8), dri(req_ids.size() * 4),
        dbt(4), dtopk(size_t(max_selected) * rows * 4), dcnt(rows * 4),
        dpart(size_t(grid_blocks) * rows * g.select_k * 8), dctr(4);
    dq8.upload(q8.data(), q8.size());
    dw.upload(w.data(), w.size() * 4);
    dki.upload(k8.data(), k8.size());
    dks_c.upload(ks.data(), ks.size() * 4);
    dpos.upload(pos.data(), pos.size() * 8);
    dri.upload(req_ids.data(), req_ids.size() * 4);
    int32_t bt0 = 0, ctr0 = 0;
    dbt.upload(&bt0, 4);
    dctr.upload(&ctr0, 4);
    dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                      static_cast<const int32_t*>(dri.p),
                      static_cast<const int64_t*>(dpos.p), rows,
                      static_cast<const int32_t*>(dbt.p), 1, dki.p,
                      static_cast<const float*>(dks_c.p), int(n_pools), heads,
                      dim, g.select_k, 4, max_selected,
                      static_cast<int32_t*>(dtopk.p),
                      static_cast<int32_t*>(dcnt.p),
                      static_cast<uint64_t*>(dpart.p),
                      static_cast<int32_t*>(dctr.p), grid_blocks, stream);
    topk.assign(size_t(max_selected) * rows, -12345);
    counts.assign(rows, -1);
    cudaDeviceSynchronize();
    dtopk.download(topk.data(), topk.size() * 4);
    dcnt.download(counts.data(), counts.size() * 4);
  }

  // Host mirror of the kernel's logits + the pinned selection + expansion.
  void expect(std::vector<int32_t>& topk, std::vector<int32_t>& counts) const {
    const int max_selected = g.max_selected;
    topk.assign(size_t(max_selected) * rows, -1);
    counts.assign(rows, 0);
    std::vector<float> logits(size_t(n_pools), 0.0f);
    std::vector<int32_t> pool_ids(g.select_k, 0);
    for (int r = 0; r < rows; ++r) {
      const int64_t visible = (pos[size_t(r)] + 1) / 4;
      for (int64_t j = 0; j < visible; ++j) {
        float contrib[32];
        for (int h = 0; h < 32; ++h) {
          float partial = 0.0f;
          for (int d = 0; d < 128; ++d)
            partial =
                partial +
                fp8_e4m3_bits_to_float(
                    q8[size_t(r) * heads * dim + size_t(h) * 128 + d]) *
                    fp8_e4m3_bits_to_float(k8[size_t(j) * dim + d]);
          contrib[h] = (w[size_t(r) * heads + h] * ks[size_t(j)]) * partial;
        }
        logits[size_t(j)] = butterfly_sum_32(contrib);
      }
      const int n_sel = dsa_ref::select_pools(logits.data(), visible,
                                              g.select_k, pool_ids.data());
      counts[size_t(r)] =
          dsa_ref::expand_append_tail(pool_ids.data(), n_sel, pos[size_t(r)],
                                      4, max_selected,
                                      &topk[size_t(r) * max_selected]);
    }
  }
};

// MTP-shaped multi-row decode: 4 rows at consecutive positions (differing
// visible counts), plus short-context rows where most blocks get empty
// stripes, plus the pos=0 edge.
DGPP_TEST(dsa_select_decode_mtp_rows_and_short_context) {
  DecodeSelScenario sc;
  sc.init(81, {40001, 40002, 40003, 40004});
  std::vector<int32_t> got, counts, want, want_counts;
  sc.run(48, got, counts);
  sc.expect(want, want_counts);
  for (int r = 0; r < sc.rows; ++r) {
    if (counts[size_t(r)] != want_counts[size_t(r)])
      throw std::runtime_error("mtp rows count r=" + std::to_string(r));
    if (std::memcmp(&got[size_t(r) * sc.g.max_selected],
                    &want[size_t(r) * sc.g.max_selected],
                    sc.g.max_selected * 4) != 0)
      throw std::runtime_error("mtp rows tokens r=" + std::to_string(r));
  }

  // Short context: visible 9 pools over a 48-block grid — 39 empty stripes
  // and an all-MAX partial merge.
  DecodeSelScenario sh;
  sh.init(82, {37});
  std::vector<int32_t> got2, counts2, want2, want_counts2;
  sh.run(48, got2, counts2);
  sh.expect(want2, want_counts2);
  if (counts2[0] != want_counts2[0] || std::memcmp(got2.data(), want2.data(),
                                                   sc.g.max_selected * 4) != 0)
    throw std::runtime_error("short-context decode select");

  // pos = 0: no pools, single tail token.
  DecodeSelScenario z;
  z.init(83, {0});
  std::vector<int32_t> got3, counts3, want3, want_counts3;
  z.run(48, got3, counts3);
  z.expect(want3, want_counts3);
  if (counts3[0] != 1 || got3[0] != 0 || got3[1] != -1)
    throw std::runtime_error("pos=0 decode select");
}

// The selection must not depend on the grid: 7 blocks vs 48 blocks produce
// identical rows (uneven stripes at grid=7 exercise the stripe bounds).
DGPP_TEST(dsa_select_decode_grid_invariance) {
  DecodeSelScenario sc;
  sc.init(84, {40003});
  std::vector<int32_t> a, ca, b, cb;
  sc.run(48, a, ca);
  sc.run(7, b, cb);
  if (ca != cb || a != b)
    throw std::runtime_error("decode select depends on grid size");
}

// Long-context decode: 100k visible pools — stripes exceed the 2048-pool
// selection tile, so blocks run multiple tiles before the merge (the shape
// the single-stream design targets).
DGPP_TEST(dsa_select_decode_long_context_multitile) {
  DecodeSelScenario sc;
  sc.init(85, {400002});  // visible = 100000 pools
  if (sc.n_pools < 100000)
    throw std::runtime_error("scenario too small");
  std::vector<int32_t> got, counts, want, want_counts;
  sc.run(48, got, counts);
  sc.expect(want, want_counts);
  if (counts[0] != want_counts[0] ||
      std::memcmp(got.data(), want.data(), sc.g.max_selected * 4) != 0)
    throw std::runtime_error("long-context decode select mismatch");
}

// Multi-row prefill select: block-per-row with a chunk-shaped position mix
// (sparse rows at ~3k positions, dense rows at 0..7 in the same batch).
DGPP_TEST(dsa_select_prefill_multi_row) {
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  const int heads = 32;
  std::vector<int64_t> positions;
  for (int p = 0; p <= 7; ++p) positions.push_back(p);
  for (int p = 3000; p < 3056; ++p) positions.push_back(p);
  const int rows = int(positions.size());
  const int64_t avail = 800;
  std::vector<float> dots(size_t(rows) * heads * avail, 0.0f);
  std::vector<float> w(size_t(rows) * heads, 0.0f);
  std::vector<float> ks(size_t(avail), 0.0f);
  for (auto& v : dots) v = random_f32(90, &v - dots.data());
  for (auto& v : w) v = random_f32(91, &v - w.data());
  for (auto& v : ks) v = random_f32(92, &v - ks.data());

  DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
      dpos(positions.size() * 8),
      dtopk(size_t(g.max_selected) * rows * 4), dcnt(rows * 4);
  dd.upload(dots.data(), dots.size() * 4);
  dw.upload(w.data(), w.size() * 4);
  dks.upload(ks.data(), ks.size() * 4);
  dpos.upload(positions.data(), positions.size() * 8);
  dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                     static_cast<const float*>(dw.p),
                     static_cast<const float*>(dks.p),
                     static_cast<const int64_t*>(dpos.p), rows, avail, heads,
                     g.select_k, 4, g.max_selected,
                     static_cast<int32_t*>(dtopk.p),
                     static_cast<int32_t*>(dcnt.p), 0);
  std::vector<int32_t> got(size_t(g.max_selected) * rows, -12345);
  std::vector<int32_t> counts(rows, -1);
  cudaDeviceSynchronize();
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(counts.data(), counts.size() * 4);

  std::vector<float> logits(size_t(avail), 0.0f);
  std::vector<int32_t> pool_ids(g.select_k, 0);
  for (int r = 0; r < rows; ++r) {
    const int64_t visible = (positions[size_t(r)] + 1) / 4;
    for (int64_t j = 0; j < visible; ++j) {
      float dot_h[32];
      for (int h = 0; h < heads; ++h)
        dot_h[h] = dots[size_t(r) * heads * avail + size_t(h) * avail + j];
      logits[size_t(j)] =
          prefill_logit_mirror(dot_h, &w[size_t(r) * heads], ks[size_t(j)]);
    }
    const int n_sel = dsa_ref::select_pools(logits.data(), visible,
                                            g.select_k, pool_ids.data());
    std::vector<int32_t> want(size_t(g.max_selected), -1);
    const int n_tok = dsa_ref::expand_append_tail(
        pool_ids.data(), n_sel, positions[size_t(r)], 4, g.max_selected,
        want.data());
    if (counts[size_t(r)] != n_tok ||
        std::memcmp(&got[size_t(r) * g.max_selected], want.data(),
                    g.max_selected * 4) != 0)
      throw std::runtime_error("multi-row prefill r=" + std::to_string(r));
  }
}

// Constructed exact ties, including a tie group that straddles the 512th
// position: the composite key must resolve every tie to the lower pool
// index on both the host oracle and the device kernels.
DGPP_TEST(dsa_select_exact_ties_break_to_lower_pool) {
  const DsaGeometry g = DsaGeometry::from_config(DsaConfig{});
  const int heads = 32;
  const int64_t avail = 700;
  const int rows = 1;
  // Pools 0..99: distinct high logits. Pools 100..699: one shared value
  // (below the top-100) — the 512th selection boundary cuts inside the tie
  // group, so the winners must be exactly 100..511.
  std::vector<float> dots(size_t(rows) * heads * avail, 0.0f);
  std::vector<float> w(size_t(heads), 1.0f / 32.0f);
  std::vector<float> ks(size_t(avail), 1.0f);
  for (int64_t j = 0; j < avail; ++j) {
    const float base = (j < 100) ? 10.0f - float(j) * 0.01f : -1.0f;
    for (int h = 0; h < heads; ++h)
      dots[size_t(h) * avail + j] = base;
  }
  std::vector<int64_t> positions = {4 * avail - 1};  // visible = avail

  DevBuf dd(dots.size() * 4), dw(w.size() * 4), dks(ks.size() * 4),
      dpos(8), dtopk(size_t(g.max_selected) * 4), dcnt(4);
  dd.upload(dots.data(), dots.size() * 4);
  dw.upload(w.data(), w.size() * 4);
  dks.upload(ks.data(), ks.size() * 4);
  dpos.upload(positions.data(), 8);
  dsa_select_prefill(static_cast<const float*>(dd.p), avail,
                     static_cast<const float*>(dw.p),
                     static_cast<const float*>(dks.p),
                     static_cast<const int64_t*>(dpos.p), rows, avail, heads,
                     g.select_k, 4, g.max_selected,
                     static_cast<int32_t*>(dtopk.p),
                     static_cast<int32_t*>(dcnt.p), 0);
  std::vector<int32_t> got(size_t(g.max_selected), -12345);
  int32_t got_cnt = -1;
  cudaDeviceSynchronize();
  dtopk.download(got.data(), got.size() * 4);
  dcnt.download(&got_cnt, 4);
  if (got_cnt != 2048)  // pos = 4*avail-1 is pool-aligned: no tail tokens
    throw std::runtime_error("ties count: " + std::to_string(got_cnt));
  for (int i = 0; i < 512; ++i) {
    const int32_t want_tok = (i < 100) ? i : i;  // pools 0..99 then 100..511
    if (got[size_t(i)] != want_tok)
      throw std::runtime_error("tie resolution at rank " + std::to_string(i) +
                               ": got " + std::to_string(got[size_t(i)]));
  }
  // Host oracle agrees.
  std::vector<float> logits(size_t(avail), 0.0f);
  for (int64_t j = 0; j < avail; ++j) {
    float dot_h[32];
    for (int h = 0; h < heads; ++h) dot_h[h] = dots[size_t(h) * avail + j];
    logits[size_t(j)] = prefill_logit_mirror(dot_h, w.data(), ks[size_t(j)]);
  }
  std::vector<int32_t> pool_ids(g.select_k, 0);
  const int n_sel =
      dsa_ref::select_pools(logits.data(), avail, g.select_k, pool_ids.data());
  for (int i = 0; i < n_sel; ++i)
    if (pool_ids[size_t(i)] != i)
      throw std::runtime_error("host tie resolution at " + std::to_string(i));
}

// latent_append + gather_index_pools: the co-located block-table round trip
// (token blocks and pool blocks are the same table's two views).
DGPP_TEST(dsa_latent_append_and_gather_roundtrip) {
  const int block_tokens = 128, kv_lora = 512, ppb = 32;
  const int n_blocks = 3, total_tokens = n_blocks * block_tokens;
  auto latent = random_bf16_bits(95, int64_t(total_tokens) * kv_lora, -2, 1);
  auto k8src = std::vector<uint8_t>(size_t(total_tokens / 4) * 128);
  for (auto& v : k8src) v = uint8_t(hash32(96 + (&v - k8src.data())) & 0x7Eu);
  std::vector<float> ks_src(size_t(total_tokens / 4), 0.0f);
  for (auto& v : ks_src) v = std::fabs(random_f32(97, &v - ks_src.data()));

  std::vector<int32_t> bt = {2, 0, 1};  // shuffled physical assignment
  const int ppb_lat = block_tokens / 4;
  std::vector<uint8_t> phys_k(k8src.size());
  std::vector<float> phys_s(ks_src.size(), 0.0f);
  for (int64_t j = 0; j < total_tokens / 4; ++j) {
    const int64_t slot = int64_t(bt[size_t(j / ppb_lat)]) * ppb_lat + (j % ppb_lat);
    std::memcpy(&phys_k[size_t(slot) * 128], &k8src[size_t(j) * 128], 128);
    phys_s[size_t(slot)] = ks_src[size_t(j)];
  }
  // Separate source and cache buffers: the append is never in-place in the
  // layer (the source is the fresh projection output), and in-place
  // aliasing is a scheduling-dependent data race (physical writes clobber
  // unread source rows through the shuffled block table).
  DevBuf dsrc(latent.size() * 2), dlat(latent.size() * 2),
      dk8(k8src.size()), dks(ks_src.size() * 4),
      dbt(bt.size() * 4), dri(size_t(total_tokens) * 4),
      dgk(k8src.size()), dgs(ks_src.size() * 4);
  dsrc.upload(latent.data(), latent.size() * 2);
  dlat.upload(std::vector<uint16_t>(latent.size(), 0).data(),
              latent.size() * 2);
  dk8.upload(phys_k.data(), phys_k.size());
  dks.upload(phys_s.data(), phys_s.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);
  // Per-token request ids sized to the token count (the kernel indexes
  // req_ids[i] for every token).
  std::vector<int32_t> req_ids_all(total_tokens, 0);
  dri.upload(req_ids_all.data(), req_ids_all.size() * 4);

  // Append all 384 latent rows through the block table.
  std::vector<int64_t> positions(total_tokens);
  for (int t = 0; t < total_tokens; ++t) positions[size_t(t)] = t;
  DevBuf dpos_all(positions.size() * 8);
  dpos_all.upload(positions.data(), positions.size() * 8);
  dsa_latent_append(dsrc.p, static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos_all.p), total_tokens,
                    static_cast<const int32_t*>(dbt.p), n_blocks, block_tokens,
                    dlat.p, kv_lora, 0);
  std::vector<uint16_t> got_lat(latent.size());
  cudaDeviceSynchronize();
  dlat.download(got_lat.data(), got_lat.size() * 2);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys =
        int64_t(bt[size_t(tok / block_tokens)]) * block_tokens +
        (tok % block_tokens);
    require_bitwise("latent append", &got_lat[size_t(phys) * kv_lora],
                    &latent[size_t(tok) * kv_lora], kv_lora * 2);
  }

  // Gather pools back into logical order and compare.
  dsa_gather_index_pools(static_cast<const int32_t*>(dbt.p), ppb, dk8.p,
                         static_cast<const float*>(dks.p), total_tokens / 4,
                         dgk.p, static_cast<float*>(dgs.p), 128, 0);
  std::vector<uint8_t> got_k(k8src.size());
  std::vector<float> got_s(ks_src.size(), 0.0f);
  cudaDeviceSynchronize();
  dgk.download(got_k.data(), got_k.size());
  dgs.download(got_s.data(), got_s.size() * 4);
  require_bitwise("gather k", got_k.data(), k8src.data(), k8src.size());
  require_bitwise("gather scales", got_s.data(), ks_src.data(),
                  ks_src.size() * 4);
}

// The elementwise paths: k LayerNorm (strided input), the fused q/kv
// RMSNorm, and the weights fold. rsqrtf on device vs 1/sqrtf on host can
// differ by ulps, so the norms compare within a couple of bf16 ulps; the
// fold is pure fp32 and compares bitwise.
DGPP_TEST(dsa_elementwise_norms_and_fold) {
  const int rows = 33, dim = 128, q_dim = 1536, kv_dim = 512;
  auto kraw = random_bf16_bits(101, int64_t(rows) * 256, -2, 1);  // strided
  auto kw = random_bf16_bits(102, dim, -1, 1);
  auto kb = random_bf16_bits(103, dim, -1, 1);
  DevBuf dk(kraw.size() * 2), dkw(kw.size() * 2), dkb(kb.size() * 2),
      dkout(size_t(rows) * dim * 2);
  dk.upload(kraw.data(), kraw.size() * 2);
  dkw.upload(kw.data(), kw.size() * 2);
  dkb.upload(kb.data(), kb.size() * 2);
  dsa_k_layernorm(dk.p, 256, dkw.p, dkb.p, dkout.p, rows, dim, 1e-6f, 0);
  std::vector<uint16_t> got_k(size_t(rows) * dim);
  cudaDeviceSynchronize();
  dkout.download(got_k.data(), got_k.size() * 2);
  for (int r = 0; r < rows; ++r) {
    const uint16_t* row = &kraw[size_t(r) * 256];
    double mean = 0;
    for (int d = 0; d < dim; ++d) mean += bf16_bits_to_float(row[d]);
    mean /= dim;
    double var = 0;
    for (int d = 0; d < dim; ++d) {
      const double t = bf16_bits_to_float(row[d]) - mean;
      var += t * t;
    }
    var /= dim;
    const double inv = 1.0 / std::sqrt(var + 1e-6);
    for (int d = 0; d < dim; ++d) {
      const double y = (bf16_bits_to_float(row[d]) - mean) * inv *
                           bf16_bits_to_float(kw[size_t(d)]) +
                       bf16_bits_to_float(kb[size_t(d)]);
      const double got = bf16_bits_to_float(got_k[size_t(r) * dim + d]);
      if (std::fabs(got - y) > std::max(1e-3, std::fabs(y) * 0.02))
        throw std::runtime_error("k layernorm r=" + std::to_string(r) + " d=" +
                                 std::to_string(d));
    }
  }

  auto qkv = random_bf16_bits(104, int64_t(rows) * (q_dim + kv_dim), -2, 1);
  auto qw = random_bf16_bits(105, q_dim, -1, 1);
  auto kvw = random_bf16_bits(106, kv_dim, -1, 1);
  DevBuf dqkv(qkv.size() * 2), dqw(qw.size() * 2), dkvw(kvw.size() * 2),
      dqc(size_t(rows) * q_dim * 2), dkvc(size_t(rows) * kv_dim * 2);
  dqkv.upload(qkv.data(), qkv.size() * 2);
  dqw.upload(qw.data(), qw.size() * 2);
  dkvw.upload(kvw.data(), kvw.size() * 2);
  dsa_fused_qkv_rmsnorm(dqkv.p, dqc.p, dkvc.p, q_dim, kv_dim, rows, dqw.p,
                        dkvw.p, 1e-5f, 0);
  std::vector<uint16_t> got_qc(size_t(rows) * q_dim),
      got_kvc(size_t(rows) * kv_dim);
  cudaDeviceSynchronize();
  dqc.download(got_qc.data(), got_qc.size() * 2);
  dkvc.download(got_kvc.data(), got_kvc.size() * 2);
  auto rms_expect = [&](const uint16_t* row, int ddim, const uint16_t* wgt) {
    std::vector<double> exp(ddim);
    double ss = 0;
    for (int d = 0; d < ddim; ++d)
      ss += double(bf16_bits_to_float(row[d])) * bf16_bits_to_float(row[d]);
    const double inv = 1.0 / std::sqrt(ss / ddim + 1e-5);
    for (int d = 0; d < ddim; ++d)
      exp[size_t(d)] = bf16_bits_to_float(row[d]) * inv *
                       bf16_bits_to_float(wgt[size_t(d)]);
    return exp;
  };
  for (int r = 0; r < rows; ++r) {
    auto eq = rms_expect(&qkv[size_t(r) * (q_dim + kv_dim)], q_dim, qw.data());
    auto ev = rms_expect(&qkv[size_t(r) * (q_dim + kv_dim) + q_dim], kv_dim,
                         kvw.data());
    for (int d = 0; d < q_dim; ++d) {
      const double got = bf16_bits_to_float(got_qc[size_t(r) * q_dim + d]);
      if (std::fabs(got - eq[size_t(d)]) >
          std::max(1e-3, std::fabs(eq[size_t(d)]) * 0.02))
        throw std::runtime_error("q rmsnorm");
    }
    for (int d = 0; d < kv_dim; ++d) {
      const double got = bf16_bits_to_float(got_kvc[size_t(r) * kv_dim + d]);
      if (std::fabs(got - ev[size_t(d)]) >
          std::max(1e-3, std::fabs(ev[size_t(d)]) * 0.02))
        throw std::runtime_error("kv rmsnorm");
    }
  }

  // Fold: bitwise fp32.
  const int64_t n = rows * 32;
  std::vector<float> a(size_t(n), 0.0f), b(size_t(n), 0.0f), out(size_t(n), 0.0f);
  for (int64_t i = 0; i < n; ++i) {
    a[size_t(i)] = random_f32(107, i);
    b[size_t(i)] = random_f32(108, i);
  }
  DevBuf da(n * 4), db(n * 4), dout(n * 4);
  da.upload(a.data(), n * 4);
  db.upload(b.data(), n * 4);
  const float scale = 0.015625f;
  dsa_fold_weights(static_cast<const float*>(da.p),
                   static_cast<const float*>(db.p),
                   static_cast<float*>(dout.p), n, scale, 0);
  std::vector<float> got_f(size_t(n), 0.0f);
  cudaDeviceSynchronize();
  dout.download(got_f.data(), n * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float want_f = (a[size_t(i)] * b[size_t(i)]) * scale;
    if (got_f[size_t(i)] != want_f)
      throw std::runtime_error("fold bitwise");
  }
}

// Multi-request decode update: two requests in one batch with a padding row
// (pos = -1, skipped) and a multi-block pool space; hard-max gates keep the
// pool contents bitwise.
DGPP_TEST(dsa_decode_update_multi_request_and_padding) {
  const int kpool = 4, dim = 128;
  const int ppb = 8;  // small blocks: 5 blocks for ~37 pools
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(110, i) * 0.1f;
  // Tokens: req0 at pos 50..55 (6 tokens, one padding at index 2), req1 at
  // pos 30..32 (3 tokens). Pools 12 (pos 51), 13 (pos 55), 7 (pos 31)
  // complete within the batch.
  const int tokens = 9;
  std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 0, 1, 1, 1};
  std::vector<int64_t> pos = {50, 51, -1, 52, 53, 54, 30, 31, 32};
  // NOTE: pos 51 completes pool 12 (48..51) — but its earlier members
  // (48..50) are NOT in this batch: they live in the seeded ring.
  auto k = random_bf16_bits(111, int64_t(tokens) * dim, -2, 1);
  auto gate = std::vector<uint16_t>(size_t(tokens) * dim);
  for (int t = 0; t < tokens; ++t) {
    const bool dom = (pos[size_t(t)] % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      gate[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }
  // Seed rings with the tokens just before the batch: req0 pos 47..49,
  // req1 pos 26..29 (their last kpool tokens before the batch).
  const int seed_tokens = 7;
  std::vector<int32_t> seed_req = {0, 0, 0, 1, 1, 1, 1};
  std::vector<int64_t> seed_pos = {47, 48, 49, 26, 27, 28, 29};
  auto seed_k = random_bf16_bits(112, int64_t(seed_tokens) * dim, -2, 1);
  auto seed_gate = std::vector<uint16_t>(size_t(seed_tokens) * dim);
  for (int t = 0; t < seed_tokens; ++t) {
    const bool dom = (seed_pos[size_t(t)] % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      seed_gate[size_t(t) * dim + d] =
          float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }

  const int max_pools = 40, n_blocks = (max_pools + ppb - 1) / ppb;
  const int total_slots = n_blocks * ppb;
  std::vector<int32_t> bt(n_blocks);
  for (int b = 0; b < n_blocks; ++b) bt[size_t(b)] = (b * 3 + 1) % n_blocks;
  // Both requests' rows of the block table ([max_requests, n_blocks]).
  std::vector<int32_t> bt2(2 * n_blocks);
  for (int b = 0; b < n_blocks; ++b) {
    bt2[size_t(b)] = bt[size_t(b)];
    bt2[size_t(n_blocks + b)] = bt[size_t(b)];
  }

  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dape(ape.size() * 4),
      dpos(pos.size() * 8), dspans(2 * 2 * 4), dbt(bt2.size() * 4),
      dki(size_t(total_slots) * dim), dks(total_slots * 4),
      dtail(2ull * 2 * kpool * dim * 2), dsk(seed_k.size() * 2),
      dsg(seed_gate.size() * 2), dsr(seed_req.size() * 4),
      dsp(seed_pos.size() * 8);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  dpos.upload(pos.data(), pos.size() * 8);
  dbt.upload(bt2.data(), bt2.size() * 4);
  dki.upload(std::vector<uint8_t>(size_t(total_slots) * dim, 0).data(),
             size_t(total_slots) * dim);
  dks.upload(std::vector<float>(total_slots, 0.0f).data(), total_slots * 4);
  dtail.upload(std::vector<uint16_t>(2ull * 2 * kpool * dim, 0).data(),
               2ull * 2 * kpool * dim * 2);
  dsk.upload(seed_k.data(), seed_k.size() * 2);
  dsg.upload(seed_gate.data(), seed_gate.size() * 2);
  dsr.upload(seed_req.data(), seed_req.size() * 4);
  dsp.upload(seed_pos.data(), seed_pos.size() * 8);

  dsa_kpool_tail_seed(dsk.p, dim, dsg.p, dim,
                      static_cast<const int32_t*>(dsr.p),
                      static_cast<const int64_t*>(dsp.p), seed_tokens,
                      dtail.p, kpool, dim, 0);
  // spans: req0 = tokens [0, 6), req1 = tokens [6, 9).
  std::vector<int32_t> spans = {0, 6, 6, 3};
  dspans.upload(spans.data(), spans.size() * 4);
  dsa_kpool_decode_update(dk.p, dim, dgate.p, dim,
                          static_cast<const float*>(dape.p),
                          static_cast<const int64_t*>(dpos.p),
                          static_cast<const int32_t*>(dspans.p), 2,
                          static_cast<const int32_t*>(dbt.p), n_blocks,
                          dtail.p, dki.p, static_cast<float*>(dks.p), ppb,
                          kpool, dim, 0);

  // Host expectation: simulate ring + completions.
  std::vector<uint16_t> tail(2ull * 2 * kpool * dim, 0);
  auto stash = [&](int req, int64_t p, const uint16_t* krow,
                   const uint16_t* grow) {
    const int slot = int(p % kpool);
    std::memcpy(&tail[size_t((req * 2 * kpool + slot)) * dim], krow, dim * 2);
    std::memcpy(&tail[size_t((req * 2 * kpool + kpool + slot)) * dim], grow,
                dim * 2);
  };
  for (int t = 0; t < seed_tokens; ++t)
    stash(seed_req[size_t(t)], seed_pos[size_t(t)],
          &seed_k[size_t(t) * dim], &seed_gate[size_t(t) * dim]);
  std::vector<uint8_t> want_k(size_t(total_slots) * dim, 0);
  std::vector<float> want_s(total_slots, 0.0f);
  auto complete = [&](int req, int64_t p, const uint16_t* krow,
                      const uint16_t* grow) {
    const int64_t pool = p / kpool;
    const int64_t slot =
        int64_t(bt[size_t(pool / ppb)]) * ppb + (pool % ppb);
    std::vector<uint16_t> pk(size_t(kpool) * dim), pg(size_t(kpool) * dim);
    for (int s = 0; s < kpool; ++s) {
      const int ring = int((p - (kpool - 1) + s) % kpool);
      const bool cur = (s == kpool - 1);
      std::memcpy(&pk[size_t(s) * dim],
                  cur ? krow : &tail[size_t((req * 2 * kpool + ring)) * dim],
                  dim * 2);
      std::memcpy(&pg[size_t(s) * dim],
                  cur ? grow
                      : &tail[size_t((req * 2 * kpool + kpool + ring)) * dim],
                  dim * 2);
    }
    dsa_ref::compress_pool<float>(pk.data(), pg.data(), ape.data(), kpool,
                                  dim, &want_k[size_t(slot) * dim],
                                  &want_s[size_t(slot)]);
  };
  for (int t = 0; t < tokens; ++t) {
    const int64_t p = pos[size_t(t)];
    if (p < 0) continue;
    const int req = req_ids[size_t(t)];
    if (p % kpool == kpool - 1)
      complete(req, p, &k[size_t(t) * dim], &gate[size_t(t) * dim]);
    stash(req, p, &k[size_t(t) * dim], &gate[size_t(t) * dim]);
  }

  std::vector<uint16_t> got_tail(2ull * 2 * kpool * dim);
  std::vector<uint8_t> got_kk(size_t(total_slots) * dim);
  std::vector<float> got_ss(total_slots, 0.0f);
  cudaDeviceSynchronize();
  dtail.download(got_tail.data(), got_tail.size() * 2);
  dki.download(got_kk.data(), got_kk.size());
  dks.download(got_ss.data(), got_ss.size() * 4);
  require_bitwise("multi-request tail", got_tail.data(), tail.data(),
                  tail.size() * 2);
  require_bitwise("multi-request pools", got_kk.data(), want_k.data(),
                  want_k.size());
  require_bitwise("multi-request scales", got_ss.data(), want_s.data(),
                  want_s.size() * 4);
}

// Attention at TP=1 (64 local heads -> 4 head-groups per split) with an
// empty padding row (counts == 0 -> zero output) and n_split == 1.
DGPP_TEST(dsa_attention_tp1_headgroups_and_empty_row) {
  DsaConfig cfg{};
  cfg.tp_size = 1;
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int rows = 2, local_heads = g.local_heads, nope = cfg.qk_nope_head_dim;
  const int v = cfg.v_head_dim, kv_lora = cfg.kv_lora_rank;
  const int block_tokens = cfg.block_tokens;
  const int max_selected = g.max_selected;

  auto q = random_bf16_bits(120, int64_t(rows) * local_heads * nope, -2, 1);
  auto kv_b = random_bf16_bits(
      121, int64_t(local_heads) * (nope + v) * kv_lora, -2, 1);
  const int cnt = 45;
  std::vector<int32_t> tokens(size_t(rows) * max_selected, -1);
  std::vector<int32_t> counts = {cnt, 0};  // row 1 is padding
  std::vector<int32_t> req_ids = {0, 0};
  for (int i = 0; i < cnt; ++i)
    tokens[i] = int32_t((i * 37) % 200);
  const int n_blocks = 2, total_tokens = n_blocks * block_tokens;
  auto latent = random_bf16_bits(122, int64_t(total_tokens) * kv_lora, -2, 1);
  std::vector<int32_t> bt = {1, 0};

  DevBuf dq(q.size() * 2), dkb(kv_b.size() * 2), dlat(latent.size() * 2),
      dtopk(tokens.size() * 4), dcnt(rows * 4), dri(rows * 4), dbt(8),
      dqt(size_t(rows) * local_heads * kv_lora * 2),
      dm(size_t(rows) * 2 * local_heads * 4),
      dl(size_t(rows) * 2 * local_heads * 4),
      dc(size_t(rows) * 2 * local_heads * kv_lora * 4),
      dc_out(size_t(rows) * local_heads * kv_lora * 4),
      dout(size_t(rows) * local_heads * v * 2);
  dq.upload(q.data(), q.size() * 2);
  dkb.upload(kv_b.data(), kv_b.size() * 2);
  dlat.upload(latent.data(), latent.size() * 2);
  dtopk.upload(tokens.data(), tokens.size() * 4);
  dcnt.upload(counts.data(), counts.size() * 4);
  dri.upload(req_ids.data(), req_ids.size() * 4);
  dbt.upload(bt.data(), bt.size() * 4);

  dsa_absorb_q(dq.p, dkb.p, dqt.p, rows, local_heads, nope, v, kv_lora, 0);
  const float scale = 1.0f / std::sqrt(float(nope));
  dsa_attn_partial(dqt.p, dlat.p, static_cast<const int32_t*>(dri.p),
                   static_cast<const int32_t*>(dtopk.p), max_selected,
                   static_cast<const int32_t*>(dcnt.p), rows, 2, local_heads,
                   kv_lora, block_tokens, static_cast<const int32_t*>(dbt.p),
                   n_blocks, scale, static_cast<float*>(dm.p),
                   static_cast<float*>(dl.p), static_cast<float*>(dc.p), 0);
  dsa_attn_combine(static_cast<const float*>(dm.p),
                   static_cast<const float*>(dl.p),
                   static_cast<const float*>(dc.p), rows, 2, local_heads,
                   kv_lora, static_cast<float*>(dc_out.p), 0);
  dsa_vout_gemm(dc_out.p, dkb.p, dout.p, rows, local_heads, nope, v, kv_lora,
                0);

  std::vector<uint16_t> phys_latent(size_t(total_tokens) * kv_lora);
  for (int64_t tok = 0; tok < total_tokens; ++tok) {
    const int64_t phys =
        int64_t(bt[size_t(tok / block_tokens)]) * block_tokens +
        (tok % block_tokens);
    std::memcpy(&phys_latent[size_t(tok) * kv_lora],
                &latent[size_t(phys) * kv_lora], kv_lora * 2);
  }
  std::vector<uint16_t> got(size_t(rows) * local_heads * v);
  cudaDeviceSynchronize();
  dout.download(got.data(), got.size() * 2);
  // Row 0: vs the host oracle.
  std::vector<uint16_t> want(size_t(local_heads) * v);
  dsa_ref::absorbed_attn<float>(&q[0], phys_latent.data(), kv_lora,
                                tokens.data(), cnt, kv_b.data(), local_heads,
                                nope, v, kv_lora, scale, want.data());
  auto st = compare_bf16(
      std::vector<uint16_t>(got.begin(), got.begin() + want.size()), want, 8);
  require_bf16("tp1 attention row0", st, 0.01, 0.006);
  // Row 1: empty selection -> exactly zero output.
  for (size_t i = want.size(); i < got.size(); ++i)
    if (got[i] != 0)
      throw std::runtime_error("empty row output must be zero at " +
                               std::to_string(i));
}

// Graph capture/replay of the fused decode select: the counter self-reset
// and the fixed-grid shape must replay correctly with a changed position
// (different visible count) without re-capture.
DGPP_TEST(dsa_select_decode_graph_replay) {
  DecodeSelScenario sc;
  sc.init(130, {40001});
  const int max_selected = sc.g.max_selected;
  const int grid = 48;

  DevBuf dq8(sc.q8.size()), dw(sc.w.size() * 4), dki(sc.k8.size()),
      dks_c(sc.ks.size() * 4), dpos(8), dri(4), dbt(4),
      dtopk(size_t(max_selected) * 4), dcnt(4),
      dpart(size_t(grid) * sc.g.select_k * 8), dctr(4);
  dq8.upload(sc.q8.data(), sc.q8.size());
  dw.upload(sc.w.data(), sc.w.size() * 4);
  dki.upload(sc.k8.data(), sc.k8.size());
  dks_c.upload(sc.ks.data(), sc.ks.size() * 4);
  std::vector<int32_t> req0 = {0}, bt0 = {0};
  dri.upload(req0.data(), 4);
  dbt.upload(bt0.data(), 4);
  int32_t ctr0 = 0;
  dctr.upload(&ctr0, 4);
  int64_t pos_a = 40001;
  dpos.upload(&pos_a, 8);

  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  // Warm up outside capture (plan/attribute setup), then capture.
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_c.p), int(sc.n_pools), 32,
                    128, sc.g.select_k, 4, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p),
                    static_cast<uint64_t*>(dpart.p),
                    static_cast<int32_t*>(dctr.p), grid, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));

  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  dsa_select_decode(dq8.p, static_cast<const float*>(dw.p),
                    static_cast<const int32_t*>(dri.p),
                    static_cast<const int64_t*>(dpos.p), 1,
                    static_cast<const int32_t*>(dbt.p), 1, dki.p,
                    static_cast<const float*>(dks_c.p), int(sc.n_pools), 32,
                    128, sc.g.select_k, 4, max_selected,
                    static_cast<int32_t*>(dtopk.p),
                    static_cast<int32_t*>(dcnt.p),
                    static_cast<uint64_t*>(dpart.p),
                    static_cast<int32_t*>(dctr.p), grid, stream);
  cudaGraph_t graph;
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Replay A: same position.
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_a(size_t(max_selected), -12345);
  int32_t cnt_a = -1;
  dtopk.download(got_a.data(), got_a.size() * 4);
  dcnt.download(&cnt_a, 4);

  // Replay B: changed position -> different visible count; must match an
  // eager run at the same position.
  int64_t pos_b = 20005;  // visible 5001 vs 10000
  dpos.upload(&pos_b, 8);
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_b(size_t(max_selected), -12345);
  int32_t cnt_b = -1;
  dtopk.download(got_b.data(), got_b.size() * 4);
  dcnt.download(&cnt_b, 4);

  DecodeSelScenario sb;
  sb.init(130, {pos_b});
  sb.q8 = sc.q8;
  sb.w = sc.w;
  sb.k8 = sc.k8;
  sb.ks = sc.ks;
  std::vector<int32_t> want_b, want_cnt_b;
  sb.expect(want_b, want_cnt_b);
  if (cnt_b != want_cnt_b[0] ||
      std::memcmp(got_b.data(), want_b.data(), max_selected * 4) != 0)
    throw std::runtime_error("graph replay with changed pos mismatch");

  // Replay C: back to position A — bitwise with replay A.
  dpos.upload(&pos_a, 8);
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<int32_t> got_c(size_t(max_selected), -12345);
  dtopk.download(got_c.data(), got_c.size() * 4);
  require_bitwise("graph replay determinism", got_c.data(), got_a.data(),
                  max_selected * 4);

  DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
  DGPP_CUDA_OK(cudaGraphDestroy(graph));
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
}

// kpool=2 smoke: the config allows it; compress + decode ring must behave.
DGPP_TEST(dsa_kpool2_compress_and_ring_smoke) {
  const int kpool = 2, dim = 128, n_pools = 12;
  const int tokens = n_pools * kpool;
  auto k = random_bf16_bits(140, tokens * dim, -2, 1);
  std::vector<float> ape(size_t(kpool) * dim);
  for (int i = 0; i < kpool * dim; ++i) ape[i] = random_f32(141, i) * 0.1f;
  auto gate = std::vector<uint16_t>(size_t(tokens) * dim);
  for (int t = 0; t < tokens; ++t) {
    const bool dom = (t % kpool) == kpool - 1;
    for (int d = 0; d < dim; ++d)
      gate[size_t(t) * dim + d] = float_to_bf16_bits(dom ? 50.0f : -50.0f);
  }
  DevBuf dk(k.size() * 2), dgate(gate.size() * 2), dape(ape.size() * 4),
      dbt(4), dki(k.size()), dks(size_t(n_pools) * 4);
  dk.upload(k.data(), k.size() * 2);
  dgate.upload(gate.data(), gate.size() * 2);
  dape.upload(ape.data(), ape.size() * 4);
  int32_t bt0 = 0;
  dbt.upload(&bt0, 4);
  dki.upload(std::vector<uint8_t>(k.size(), 0).data(), k.size());
  dsa_kpool_compress_write(dk.p, dim, dgate.p, dim,
                           static_cast<const float*>(dape.p),
                           static_cast<const int32_t*>(dbt.p), n_pools, 0,
                           n_pools, dki.p, static_cast<float*>(dks.p), kpool,
                           dim, 0);
  std::vector<uint8_t> got(k.size());
  std::vector<float> got_s(size_t(n_pools), 0.0f);
  cudaDeviceSynchronize();
  dki.download(got.data(), got.size());
  dks.download(got_s.data(), got_s.size() * 4);
  for (int j = 0; j < n_pools; ++j) {
    std::vector<uint8_t> w8(dim);
    float ws = 0;
    dsa_ref::compress_pool<float>(&k[size_t(j * kpool) * dim],
                                  &gate[size_t(j * kpool) * dim], ape.data(),
                                  kpool, dim, w8.data(), &ws);
    if (std::memcmp(&got[size_t(j) * dim], w8.data(), dim) != 0 ||
        got_s[size_t(j)] != ws)
      throw std::runtime_error("kpool2 compress pool " + std::to_string(j));
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int, char**) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
