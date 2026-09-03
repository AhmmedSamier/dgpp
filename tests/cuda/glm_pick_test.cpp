// The on-device greedy pick (kernels/glm_pick.hpp) and the step's device
// commit (kernels/glm_spec.hpp) against their host oracles: glm_pick_local's top-2 vs glm_sample::local_max plus the gen
// log's runner-up scan; glm_pick_verdict's merge vs glm_sample::merge_greedy
// and its judge vs judge_verify, over a SIMULATED world (each rank's table
// produced by the kernel on its slice, the fold emulated as an exact host
// sum — which is what the bus's SUM over disjoint slots is); the digest
// group's agreement/mismatch detection; the wire table's layout; and the
// fixed-batch MTP control/cache helpers.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_pick.hpp"
#include "kernels/glm_spec.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_speculative.hpp"

namespace {

using dgpp::GlmPickLocal;
using dgpp::GlmPickVerdict;
using dgpp::kPickIdDigits;
using dgpp::kPickLogitDigits;
using dgpp::kPickSlotsPerRank;
using dgpp::glm_sample::Candidate;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
};

// Logits with deliberate structure: values quantized to a handful of
// levels (ties everywhere — the canonical tie-break is what's under test),
// a sprinkling of -inf, and a few distinct peaks.
std::vector<float> make_logits(Rng& rng, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const uint64_t r = rng.next();
    if ((r & 63) == 0) {
      v[i] = -INFINITY;
    } else {
      v[i] = static_cast<float>((r >> 8) % 17) * 0.5f - 4.0f;
    }
  }
  return v;
}

float host_runner_up(const float* slice, int n, int vocab_begin,
                     int32_t best_id) {
  float second = -INFINITY;
  for (int i = 0; i < n; ++i)
    if (vocab_begin + i != best_id && slice[i] > second) second = slice[i];
  return second;
}

uint64_t decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

template <typename T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
  return p;
}

// One rank's local pick on `logits` [rows, count]; returns its table
// (host copy) and locals.
struct LocalRun {
  std::vector<uint16_t> table;
  std::vector<GlmPickLocal> locals;
};

LocalRun run_local(const std::vector<float>& logits, int rows, int count,
                   int vocab_begin, int rank, int world, uint64_t carry) {
  const size_t table_elems = dgpp::glm_pick_table_elems(rows, world);
  float* d_logits = device_alloc<float>(logits.size());
  uint16_t* d_table = device_alloc<uint16_t>(table_elems);
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  GlmPickLocal* d_locals = device_alloc<GlmPickLocal>(static_cast<size_t>(rows));
  DGPP_CUDA_OK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_carry, &carry, 8, cudaMemcpyHostToDevice));
  // Poison the table: the kernel must zero every slot it does not write.
  DGPP_CUDA_OK(cudaMemset(d_table, 0xff, table_elems * 2));
  dgpp::glm_pick_local(d_logits, rows, count, vocab_begin, rank, world,
                       d_carry, d_table, d_locals, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  LocalRun out;
  out.table.resize(table_elems);
  out.locals.resize(static_cast<size_t>(rows));
  DGPP_CUDA_OK(cudaMemcpy(out.table.data(), d_table, table_elems * 2,
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.locals.data(), d_locals,
                          sizeof(GlmPickLocal) * rows, cudaMemcpyDeviceToHost));
  cudaFree(d_logits);
  cudaFree(d_table);
  cudaFree(d_carry);
  cudaFree(d_locals);
  return out;
}

GlmPickVerdict run_verdict(const std::vector<uint16_t>& table, int rows,
                           int world, int rank,
                           const std::vector<int64_t>& fed,
                           uint64_t* carry_out) {
  uint16_t* d_table = device_alloc<uint16_t>(table.size());
  int64_t* d_fed = device_alloc<int64_t>(fed.size());
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  GlmPickVerdict* d_verdict = device_alloc<GlmPickVerdict>(1);
  DGPP_CUDA_OK(cudaMemcpy(d_table, table.data(), table.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                          cudaMemcpyHostToDevice));
  dgpp::glm_pick_verdict(d_table, rows, world, rank, d_fed, d_verdict,
                         /*device_verdict=*/nullptr, d_carry, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  GlmPickVerdict v;
  DGPP_CUDA_OK(cudaMemcpy(&v, d_verdict, sizeof(v), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(carry_out, d_carry, 8, cudaMemcpyDeviceToHost));
  cudaFree(d_table);
  cudaFree(d_fed);
  cudaFree(d_carry);
  cudaFree(d_verdict);
  return v;
}

std::vector<GlmPickVerdict> run_verdict_batch(
    const std::vector<uint16_t>& table, int rows, int world, int rank,
    const std::vector<int64_t>& fed, const std::vector<int64_t>& positions,
    int requests, int rows_per_request, uint64_t* carry_out) {
  uint16_t* d_table = device_alloc<uint16_t>(table.size());
  int64_t* d_fed = device_alloc<int64_t>(fed.size());
  int64_t* d_positions = device_alloc<int64_t>(positions.size());
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  GlmPickVerdict* d_verdict =
      device_alloc<GlmPickVerdict>(static_cast<size_t>(requests));
  DGPP_CUDA_OK(cudaMemcpy(d_table, table.data(), table.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_fed, fed.data(), fed.size() * 8,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_positions, positions.data(), positions.size() * 8,
                          cudaMemcpyHostToDevice));
  dgpp::glm_pick_verdict_batched(
      d_table, rows, world, rank, d_fed, d_positions, requests,
      rows_per_request, rows_per_request, d_verdict,
      /*device_verdicts=*/nullptr, d_carry, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<GlmPickVerdict> out(static_cast<size_t>(requests));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), d_verdict,
                          out.size() * sizeof(GlmPickVerdict),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(carry_out, d_carry, 8, cudaMemcpyDeviceToHost));
  cudaFree(d_table);
  cudaFree(d_fed);
  cudaFree(d_positions);
  cudaFree(d_carry);
  cudaFree(d_verdict);
  return out;
}

}  // namespace

DGPP_TEST(pick_local_top2_matches_host_local_max_and_runner_up) {
  Rng rng(0x5151);
  struct Shape {
    int rows, count, vocab_begin, rank, world;
  };
  const Shape shapes[] = {{1, 1, 0, 0, 1},        {1, 37, 0, 0, 1},
                          {2, 38720, 38720, 1, 4}, {4, 1000, 3000, 3, 4},
                          {3, 2049, 0, 0, 2},      {1, 154880, 0, 0, 1}};
  for (const Shape& s : shapes) {
    const std::vector<float> logits =
        make_logits(rng, static_cast<size_t>(s.rows) * s.count);
    const uint64_t carry = 0x2a5a5a5a5a5a5ull;  // 54 bits
    const LocalRun got =
        run_local(logits, s.rows, s.count, s.vocab_begin, s.rank, s.world, carry);
    for (int r = 0; r < s.rows; ++r) {
      const float* slice = logits.data() + static_cast<size_t>(r) * s.count;
      const Candidate want =
          dgpp::glm_sample::local_max(slice, s.count, s.vocab_begin);
      const GlmPickLocal& l = got.locals[static_cast<size_t>(r)];
      require(l.best_id == want.id,
              "row " + std::to_string(r) + ": best id " +
                  std::to_string(l.best_id) + " != " + std::to_string(want.id));
      require(std::memcmp(&l.best_logit, &want.logit, 4) == 0,
              "row " + std::to_string(r) + ": best logit bits differ");
      const float second = host_runner_up(slice, s.count, s.vocab_begin, want.id);
      require(std::memcmp(&l.second_logit, &second, 4) == 0,
              "row " + std::to_string(r) + ": runner-up " +
                  std::to_string(l.second_logit) + " != " +
                  std::to_string(second));
      // The wire slots: (logit bits, id) at (row, rank); zero elsewhere.
      for (int k = 0; k < s.world; ++k) {
        const uint16_t* slot =
            got.table.data() +
            (static_cast<size_t>(r) * s.world + k) * kPickSlotsPerRank;
        if (k != s.rank) {
          for (int d = 0; d < kPickSlotsPerRank; ++d)
            require(slot[d] == 0, "foreign slot not zeroed");
          continue;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &want.logit, 4);
        require(decode_digits(slot, kPickLogitDigits) == bits,
                "logit digits do not round-trip");
        require(decode_digits(slot + kPickLogitDigits, kPickIdDigits) ==
                    static_cast<uint64_t>(want.id),
                "id digits do not round-trip");
      }
    }
    // The digest group: this rank's carry, zero for the others.
    const uint16_t* digests =
        got.table.data() +
        static_cast<size_t>(s.rows) * s.world * kPickSlotsPerRank;
    for (int k = 0; k < s.world; ++k) {
      const uint64_t d =
          decode_digits(digests + k * kPickSlotsPerRank, kPickSlotsPerRank);
      require(d == (k == s.rank ? carry : 0), "digest group slot wrong");
    }
  }
}

DGPP_TEST(pick_verdict_over_simulated_world_matches_merge_greedy_and_judge) {
  Rng rng(0x7e57);
  constexpr int kWorld = 4;
  constexpr int kCount = 96;  // the fixture vocab / world
  for (int rows = 1; rows <= dgpp::kPickMaxRows; ++rows) {
    for (int trial = 0; trial < 6; ++trial) {
      // The full [rows, vocab] logits, sliced per rank.
      const std::vector<float> full =
          make_logits(rng, static_cast<size_t>(rows) * kWorld * kCount);
      std::vector<std::vector<Candidate>> per_row_locals(
          static_cast<size_t>(rows));
      std::vector<uint16_t> folded(dgpp::glm_pick_table_elems(rows, kWorld), 0);
      const uint64_t carry = 0x123456789abcdull;
      for (int k = 0; k < kWorld; ++k) {
        std::vector<float> slice(static_cast<size_t>(rows) * kCount);
        for (int r = 0; r < rows; ++r)
          std::memcpy(slice.data() + static_cast<size_t>(r) * kCount,
                      full.data() +
                          (static_cast<size_t>(r) * kWorld + k) * kCount,
                      kCount * 4);
        const LocalRun local =
            run_local(slice, rows, kCount, k * kCount, k, kWorld, carry);
        for (int r = 0; r < rows; ++r)
          per_row_locals[static_cast<size_t>(r)].push_back(
              dgpp::glm_sample::local_max(
                  slice.data() + static_cast<size_t>(r) * kCount, kCount,
                  k * kCount));
        // The fold: an exact SUM over disjoint slots (bf16 small ints).
        for (size_t i = 0; i < folded.size(); ++i) {
          require(folded[i] == 0 || local.table[i] == 0,
                  "two ranks wrote the same wire slot");
          folded[i] = static_cast<uint16_t>(folded[i] + local.table[i]);
        }
      }
      std::vector<int32_t> want_winners;
      for (int r = 0; r < rows; ++r)
        want_winners.push_back(dgpp::glm_sample::merge_greedy(
            per_row_locals[static_cast<size_t>(r)]));
      // Fed tokens: alternate trials feed the true next tokens (accept
      // all) and random ones (reject somewhere).
      std::vector<int64_t> fed(static_cast<size_t>(rows));
      fed[0] = static_cast<int64_t>(rng.next() % (kWorld * kCount));
      for (int r = 1; r < rows; ++r)
        fed[static_cast<size_t>(r)] =
            (trial % 2 == 0 || (rng.next() & 1))
                ? want_winners[static_cast<size_t>(r - 1)]
                : static_cast<int64_t>(rng.next() % (kWorld * kCount));
      const dgpp::SpecVerdict want = dgpp::judge_verify(fed, want_winners);

      for (int rank = 0; rank < kWorld; ++rank) {
        uint64_t carry_out = 0;
        const GlmPickVerdict got =
            run_verdict(folded, rows, kWorld, rank, fed, &carry_out);
        require(got.rows == rows, "verdict rows");
        for (int r = 0; r < rows; ++r)
          require(got.winners[r] == want_winners[static_cast<size_t>(r)],
                  "rows " + std::to_string(rows) + " row " +
                      std::to_string(r) + ": winner " +
                      std::to_string(got.winners[r]) + " != merge_greedy's " +
                      std::to_string(want_winners[static_cast<size_t>(r)]));
        require(got.accepted == want.accepted,
                "accepted " + std::to_string(got.accepted) + " != " +
                    std::to_string(want.accepted));
        require(got.next == want.next, "next differs from judge_verify");
        require(got.digest_mismatch == 0, "identical carries flagged");
        require(got.digest ==
                    dgpp::glm_pick_digest(rows, got.accepted, got.winners),
                "digest != host digest");
        require(carry_out == got.digest, "carry not updated to the digest");
        for (int k = 0; k < kWorld; ++k)
          require(got.peer_digests[k] == carry, "peer digest decoded wrong");
      }
    }
  }
}

DGPP_TEST(pick_verdict_flags_the_rank_whose_carried_digest_differs) {
  Rng rng(0xd16e);
  constexpr int kWorld = 4;
  constexpr int rows = 2;
  constexpr int kCount = 96;
  std::vector<uint16_t> folded(dgpp::glm_pick_table_elems(rows, kWorld), 0);
  for (int k = 0; k < kWorld; ++k) {
    const std::vector<float> slice =
        make_logits(rng, static_cast<size_t>(rows) * kCount);
    // Rank 2 carries a different digest: it computed a different verdict
    // last step (its table was corrupt).
    const uint64_t carry = k == 2 ? 0xbadull : 0x600dull;
    const LocalRun local =
        run_local(slice, rows, kCount, k * kCount, k, kWorld, carry);
    for (size_t i = 0; i < folded.size(); ++i)
      folded[i] = static_cast<uint16_t>(folded[i] + local.table[i]);
  }
  const std::vector<int64_t> fed{1, 2};
  uint64_t carry_out = 0;
  const GlmPickVerdict from_rank0 =
      run_verdict(folded, rows, kWorld, 0, fed, &carry_out);
  require(from_rank0.digest_mismatch == (1u << 2),
          "rank 0 must see exactly rank 2 disagreeing, mask " +
              std::to_string(from_rank0.digest_mismatch));
  const GlmPickVerdict from_rank2 =
      run_verdict(folded, rows, kWorld, 2, fed, &carry_out);
  require(from_rank2.digest_mismatch == 0b1011u,
          "rank 2 must see every peer disagreeing with it");
  require(from_rank0.peer_digests[2] == 0xbadull &&
              from_rank0.peer_digests[1] == 0x600dull,
          "peer digests must decode to what each rank carried");
}

DGPP_TEST(pick_batch_judges_each_request_and_skips_padding) {
  constexpr int requests = 3;
  constexpr int per = 2;
  constexpr int rows = requests * per;
  constexpr int count = 32;
  std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
  const int32_t winners[rows] = {3, 5, 7, 9, 11, 13};
  for (int r = 0; r < rows; ++r)
    logits[static_cast<size_t>(r) * count + winners[r]] = 10.0f + r;
  const LocalRun local =
      run_local(logits, rows, count, /*vocab_begin=*/0, /*rank=*/0,
                /*world=*/1, /*carry=*/0x1234);
  std::vector<int64_t> fed = {17, winners[0], 19, winners[2], 23, 24};
  const std::vector<int64_t> positions = {100, 101, -1, -1, 300, 301};
  uint64_t carry = 0;
  const std::vector<GlmPickVerdict> got = run_verdict_batch(
      local.table, rows, /*world=*/1, /*rank=*/0, fed, positions, requests,
      per, &carry);
  require(got[0].rows == 2 && got[0].accepted == 2 &&
              got[0].next == winners[1],
          "request 0 must accept both rows");
  require(got[1].rows == 0 && got[1].accepted == 0 && got[1].next == -1,
          "request 1 must be an inactive padding verdict");
  require(got[2].rows == 2 && got[2].accepted == 1 &&
              got[2].next == winners[4],
          "request 2 must reject its draft row");
  for (const GlmPickVerdict& v : got)
    require(v.digest == carry && v.digest_mismatch == 0,
            "every request must carry the physical pass digest");
  require((carry >> dgpp::kPickDigestBits) == 0,
          "batched digest must fit the wire's carried digit group");
}

DGPP_TEST(pick_batch_draft_selects_last_accepted_row_per_request) {
  constexpr int requests = 3;
  constexpr int stride = 2;
  constexpr int count = 24;
  std::vector<float> logits(static_cast<size_t>(requests * stride) * count,
                            -20.0f);
  const int32_t winners[requests * stride] = {2, 3, 5, 7, 11, 13};
  for (int r = 0; r < requests * stride; ++r)
    logits[static_cast<size_t>(r) * count + winners[r]] = 20.0f;
  float* d_logits = device_alloc<float>(logits.size());
  uint16_t* d_table = device_alloc<uint16_t>(
      dgpp::glm_pick_table_elems(requests, /*world=*/1));
  uint64_t* d_carry = device_alloc<uint64_t>(1);
  GlmPickLocal* d_locals = device_alloc<GlmPickLocal>(requests);
  GlmPickVerdict select[requests];
  select[0].accepted = 1;
  select[1].accepted = 0;  // inactive: harmless first padding row
  select[2].accepted = 2;
  GlmPickVerdict* d_select = device_alloc<GlmPickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_logits, logits.data(), logits.size() * 4,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(d_carry, 0, sizeof(uint64_t)));
  DGPP_CUDA_OK(cudaMemcpy(d_select, select, sizeof(select),
                          cudaMemcpyHostToDevice));
  dgpp::glm_pick_local_batched(
      d_logits, requests, count, /*vocab_begin=*/0, /*rank=*/0, /*world=*/1,
      d_carry, d_table, d_locals, nullptr, d_select, requests, stride);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  GlmPickLocal got[requests];
  DGPP_CUDA_OK(cudaMemcpy(got, d_locals, sizeof(got), cudaMemcpyDeviceToHost));
  require(got[0].best_id == winners[0], "request 0 selected wrong draft row");
  require(got[1].best_id == winners[2], "inactive request must select row 0");
  require(got[2].best_id == winners[5], "request 2 selected wrong draft row");
  cudaFree(d_logits);
  cudaFree(d_table);
  cudaFree(d_carry);
  cudaFree(d_locals);
  cudaFree(d_select);
}

// The commit: with rows = 3 and a verdict accepting a rows, every segment's
// live state must become snapshot row a-1 when a < 3 and stay untouched
// when a == 3; the position advances by a either way; the positions kernel
// derives the rows' positions from the device position.
DGPP_TEST(spec_commit_copies_snapshot_row_when_rejected_and_advances_position) {
  constexpr int rows = 3;
  constexpr size_t kBytesA = 4096, kBytesB = 64;  // two families, uneven
  std::vector<uint8_t> live_a(kBytesA, 0xaa), live_b(kBytesB, 0xbb);
  // Snapshot rows: row r of family A is filled with 0x10 + r, B with 0x20 + r.
  std::vector<uint8_t> snaps_a(kBytesA * (rows - 1)), snaps_b(kBytesB * (rows - 1));
  for (int r = 0; r < rows - 1; ++r) {
    std::fill_n(snaps_a.data() + r * kBytesA, kBytesA, static_cast<uint8_t>(0x10 + r));
    std::fill_n(snaps_b.data() + r * kBytesB, kBytesB, static_cast<uint8_t>(0x20 + r));
  }
  uint8_t* d_live_a = device_alloc<uint8_t>(kBytesA);
  uint8_t* d_live_b = device_alloc<uint8_t>(kBytesB);
  uint8_t* d_snaps_a = device_alloc<uint8_t>(snaps_a.size());
  uint8_t* d_snaps_b = device_alloc<uint8_t>(snaps_b.size());
  GlmPickVerdict* d_verdict = device_alloc<GlmPickVerdict>(1);
  int64_t* d_pos = device_alloc<int64_t>(1);
  int64_t* d_step_pos = device_alloc<int64_t>(rows);
  DGPP_CUDA_OK(cudaMemcpy(d_snaps_a, snaps_a.data(), snaps_a.size(),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_snaps_b, snaps_b.data(), snaps_b.size(),
                          cudaMemcpyHostToDevice));
  dgpp::GlmSpecSegments segs;
  segs.count = 2;
  segs.seg[0] = dgpp::GlmSpecSegment{d_live_a, d_snaps_a, kBytesA, kBytesA};
  segs.seg[1] = dgpp::GlmSpecSegment{d_live_b, d_snaps_b, kBytesB, kBytesB};

  for (int accepted = 1; accepted <= rows; ++accepted) {
    DGPP_CUDA_OK(cudaMemcpy(d_live_a, live_a.data(), kBytesA, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_live_b, live_b.data(), kBytesB, cudaMemcpyHostToDevice));
    GlmPickVerdict v;
    v.rows = rows;
    v.accepted = accepted;
    DGPP_CUDA_OK(cudaMemcpy(d_verdict, &v, sizeof(v), cudaMemcpyHostToDevice));
    const int64_t pos0 = 1000;
    DGPP_CUDA_OK(cudaMemcpy(d_pos, &pos0, 8, cudaMemcpyHostToDevice));
    dgpp::glm_spec_commit(d_verdict, rows, segs, d_pos, nullptr);
    dgpp::glm_spec_positions(d_pos, rows, d_step_pos, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> got_a(kBytesA), got_b(kBytesB);
    int64_t pos = 0;
    std::vector<int64_t> step_pos(rows);
    DGPP_CUDA_OK(cudaMemcpy(got_a.data(), d_live_a, kBytesA, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(got_b.data(), d_live_b, kBytesB, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(&pos, d_pos, 8, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(step_pos.data(), d_step_pos, 8 * rows, cudaMemcpyDeviceToHost));
    require(pos == pos0 + accepted, "position must advance by accepted");
    for (int r = 0; r < rows; ++r)
      require(step_pos[static_cast<size_t>(r)] == pos + r,
              "step positions must follow the device position");
    const uint8_t want_a = accepted == rows ? 0xaa : static_cast<uint8_t>(0x10 + accepted - 1);
    const uint8_t want_b = accepted == rows ? 0xbb : static_cast<uint8_t>(0x20 + accepted - 1);
    for (size_t i = 0; i < kBytesA; ++i)
      require(got_a[i] == want_a, "family A: accepted " + std::to_string(accepted) +
                                       " byte " + std::to_string(i));
    for (size_t i = 0; i < kBytesB; ++i)
      require(got_b[i] == want_b, "family B: accepted " + std::to_string(accepted));
  }
  // A model with no stateful KDA/DSA family still uses the commit kernel to
  // advance its device position. A rejection must not index the empty
  // segment table.
  GlmPickVerdict position_only;
  position_only.rows = rows;
  position_only.accepted = 1;
  DGPP_CUDA_OK(cudaMemcpy(d_verdict, &position_only, sizeof(position_only),
                          cudaMemcpyHostToDevice));
  const int64_t position_only_start = 77;
  DGPP_CUDA_OK(cudaMemcpy(d_pos, &position_only_start, sizeof(int64_t),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_commit(d_verdict, rows, dgpp::GlmSpecSegments{}, d_pos,
                        nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t position_only_got = 0;
  DGPP_CUDA_OK(cudaMemcpy(&position_only_got, d_pos, sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
  require(position_only_got == position_only_start + 1,
          "position-only commit did not advance exactly once");
  cudaFree(d_live_a);
  cudaFree(d_live_b);
  cudaFree(d_snaps_a);
  cudaFree(d_snaps_b);
  cudaFree(d_verdict);
  cudaFree(d_pos);
  cudaFree(d_step_pos);
}

DGPP_TEST(spec_batch_positions_draft_rows_and_token_feeds_are_slot_local) {
  constexpr int requests = 3;
  constexpr int per = 2;
  constexpr int rows = requests * per;
  const int64_t session_pos[requests] = {100, 0, 300};
  const int32_t req_ids[rows] = {0, 0, 1, 1, 2, 2};
  int64_t* d_session = device_alloc<int64_t>(requests);
  int32_t* d_req = device_alloc<int32_t>(rows);
  int64_t* d_pos = device_alloc<int64_t>(rows);
  int64_t* d_tokens = device_alloc<int64_t>(rows);
  int64_t* d_next = device_alloc<int64_t>(requests);
  int64_t* d_block = device_alloc<int64_t>(requests);
  GlmPickVerdict verify[requests];
  verify[0].rows = 2;
  verify[0].accepted = 2;
  verify[0].next = 31;
  verify[0].winners[0] = 29;
  verify[0].winners[1] = 31;
  verify[1].rows = 0;
  verify[1].accepted = 0;
  verify[2].rows = 2;
  verify[2].accepted = 1;
  verify[2].next = 41;
  verify[2].winners[0] = 41;
  verify[2].winners[1] = 43;
  GlmPickVerdict* d_verify = device_alloc<GlmPickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_session, session_pos, sizeof(session_pos),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_req, req_ids, sizeof(req_ids),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_verify, verify, sizeof(verify),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_positions_batched(d_session, d_req, rows, per, d_pos,
                                   nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t pos[rows];
  DGPP_CUDA_OK(cudaMemcpy(pos, d_pos, sizeof(pos), cudaMemcpyDeviceToHost));
  const int64_t want_verify_pos[rows] = {100, 101, -1, -1, 300, 301};
  require(std::equal(std::begin(pos), std::end(pos),
                     std::begin(want_verify_pos)),
          "batched verify positions differ");

  const int64_t block_pos[requests] = {90, 190, 290};
  DGPP_CUDA_OK(cudaMemcpy(d_block, block_pos, sizeof(block_pos),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_draft_rows_batched(d_verify, requests, per, d_block, d_pos,
                                    d_tokens, d_next, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int64_t draft_pos[rows], draft_tokens[rows], next[requests], block[requests];
  DGPP_CUDA_OK(cudaMemcpy(draft_pos, d_pos, sizeof(draft_pos),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(draft_tokens, d_tokens, sizeof(draft_tokens),
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(next, d_next, sizeof(next), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(block, d_block, sizeof(block), cudaMemcpyDeviceToHost));
  const int64_t want_draft_pos[rows] = {90, 91, -1, -1, 290, -1};
  const int64_t want_draft_tokens[rows] = {29, 31, 0, 0, 41, 41};
  require(std::equal(std::begin(draft_pos), std::end(draft_pos),
                     std::begin(want_draft_pos)),
          "batched draft positions differ");
  require(std::equal(std::begin(draft_tokens), std::end(draft_tokens),
                     std::begin(want_draft_tokens)),
          "batched draft tokens differ");
  require(block[0] == 92 && block[1] == 190 && block[2] == 291 &&
              next[0] == 31 && next[2] == 41,
          "batched draft counters/next tokens are not slot-local");

  GlmPickVerdict draft[requests];
  draft[0].rows = 1;
  draft[0].accepted = 1;
  draft[0].next = 37;
  draft[1].rows = 0;
  draft[1].accepted = 0;
  draft[2].rows = 1;
  draft[2].accepted = 1;
  draft[2].next = 47;
  GlmPickVerdict* d_draft = device_alloc<GlmPickVerdict>(requests);
  DGPP_CUDA_OK(cudaMemcpy(d_draft, draft, sizeof(draft),
                          cudaMemcpyHostToDevice));
  dgpp::glm_spec_next_tokens_batched(d_next, d_draft, requests, per,
                                     d_tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(draft_tokens, d_tokens, sizeof(draft_tokens),
                          cudaMemcpyDeviceToHost));
  const int64_t want_next_tokens[rows] = {31, 37, 0, 0, 41, 47};
  require(std::equal(std::begin(draft_tokens), std::end(draft_tokens),
                     std::begin(want_next_tokens)),
          "batched next token feed differs");

  cudaFree(d_session);
  cudaFree(d_req);
  cudaFree(d_pos);
  cudaFree(d_tokens);
  cudaFree(d_next);
  cudaFree(d_block);
  cudaFree(d_verify);
  cudaFree(d_draft);
}

DGPP_TEST(mtp_batch_hidden_cache_input_and_scatter_are_request_indexed) {
  constexpr int requests = 3;
  constexpr int rows = 6;
  constexpr int hidden = 8;
  constexpr int vocab = 4;
  constexpr int cache_positions = 4;
  constexpr int64_t cache_stride = cache_positions * hidden;
  const int64_t tokens[rows] = {0, 1, 2, 3, 1, 0};
  const int32_t req_ids[rows] = {2, 2, 1, 1, 0, 0};
  const int64_t positions[rows] = {0, 1, -1, -1, 2, 3};

  std::vector<uint16_t> embed(vocab * hidden);
  std::vector<uint16_t> cache(requests * cache_stride);
  std::vector<uint16_t> enorm(hidden), hnorm(hidden);
  for (int v = 0; v < vocab; ++v)
    for (int h = 0; h < hidden; ++h)
      embed[static_cast<size_t>(v) * hidden + h] =
          dgpp::float_to_bf16_bits((v + 1) * 0.25f + (h + 1) * 0.03125f);
  for (int q = 0; q < requests; ++q)
    for (int p = 0; p < cache_positions; ++p)
      for (int h = 0; h < hidden; ++h)
        cache[static_cast<size_t>(q) * cache_stride + p * hidden + h] =
            dgpp::float_to_bf16_bits((q + 1) * 2.0f + p * 0.25f +
                                     (h + 1) * 0.015625f);
  for (int h = 0; h < hidden; ++h) {
    enorm[h] = dgpp::float_to_bf16_bits(0.5f + h * 0.0625f);
    hnorm[h] = dgpp::float_to_bf16_bits(1.0f + h * 0.03125f);
  }

  uint16_t* d_embed = device_alloc<uint16_t>(embed.size());
  uint16_t* d_cache = device_alloc<uint16_t>(cache.size());
  uint16_t* d_enorm = device_alloc<uint16_t>(enorm.size());
  uint16_t* d_hnorm = device_alloc<uint16_t>(hnorm.size());
  int64_t* d_tokens = device_alloc<int64_t>(rows);
  int32_t* d_req = device_alloc<int32_t>(rows);
  int64_t* d_pos = device_alloc<int64_t>(rows);
  uint16_t* d_batch = device_alloc<uint16_t>(rows * 2 * hidden);
  uint16_t* d_scalar = device_alloc<uint16_t>(rows * 2 * hidden);
  DGPP_CUDA_OK(cudaMemcpy(d_embed, embed.data(), embed.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_cache, cache.data(), cache.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_enorm, enorm.data(), enorm.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_hnorm, hnorm.data(), hnorm.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_tokens, tokens, sizeof(tokens),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_req, req_ids, sizeof(req_ids),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_pos, positions, sizeof(positions),
                          cudaMemcpyHostToDevice));

  dgpp::glm_mtp_input_bf16_batched(
      d_embed, d_tokens, d_cache, cache_stride, d_req, d_pos, d_enorm,
      d_hnorm, d_batch, rows, hidden, 1e-5f, nullptr);
  for (int r = 0; r < rows; ++r) {
    const uint16_t* request_cache =
        d_cache + static_cast<int64_t>(req_ids[r]) * cache_stride;
    dgpp::glm_mtp_input_bf16(
        d_embed, d_tokens + r, request_cache, d_pos + r, /*first_pos=*/0,
        d_enorm, d_hnorm, d_scalar + r * 2 * hidden, /*rows=*/1, hidden,
        1e-5f, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> batch(rows * 2 * hidden);
  std::vector<uint16_t> scalar(rows * 2 * hidden);
  DGPP_CUDA_OK(cudaMemcpy(batch.data(), d_batch, batch.size() * 2,
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(scalar.data(), d_scalar, scalar.size() * 2,
                          cudaMemcpyDeviceToHost));
  require(batch == scalar,
          "batched MTP input differs from request-based scalar rows");
  for (int r : {2, 3})
    require(std::all_of(batch.begin() + r * 2 * hidden,
                        batch.begin() + (r + 1) * 2 * hidden,
                        [](uint16_t v) { return v == 0; }),
            "negative-position MTP input row was not zero padded");

  std::vector<uint16_t> scatter_rows(rows * hidden);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < hidden; ++h)
      scatter_rows[static_cast<size_t>(r) * hidden + h] =
          static_cast<uint16_t>(0x100 + r * hidden + h);
  std::vector<uint16_t> scatter_want(requests * cache_stride, 0x5a5a);
  for (int r = 0; r < rows; ++r) {
    if (positions[r] < 0) continue;
    std::copy_n(scatter_rows.begin() + r * hidden, hidden,
                scatter_want.begin() +
                    static_cast<int64_t>(req_ids[r]) * cache_stride +
                    positions[r] * hidden);
  }
  uint16_t* d_rows = device_alloc<uint16_t>(scatter_rows.size());
  uint16_t* d_scatter = device_alloc<uint16_t>(scatter_want.size());
  DGPP_CUDA_OK(cudaMemcpy(d_rows, scatter_rows.data(), scatter_rows.size() * 2,
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(d_scatter, 0x5a, scatter_want.size() * 2));
  dgpp::glm_rows_scatter_bf16_batched(
      d_rows, d_req, d_pos, d_scatter, cache_stride, rows, hidden, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> scatter_got(scatter_want.size());
  DGPP_CUDA_OK(cudaMemcpy(scatter_got.data(), d_scatter,
                          scatter_got.size() * 2, cudaMemcpyDeviceToHost));
  require(scatter_got == scatter_want,
          "batched hidden scatter crossed a request slot or wrote padding");

  cudaFree(d_embed);
  cudaFree(d_cache);
  cudaFree(d_enorm);
  cudaFree(d_hnorm);
  cudaFree(d_tokens);
  cudaFree(d_req);
  cudaFree(d_pos);
  cudaFree(d_batch);
  cudaFree(d_scalar);
  cudaFree(d_rows);
  cudaFree(d_scatter);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
