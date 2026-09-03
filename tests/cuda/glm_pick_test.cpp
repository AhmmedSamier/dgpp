// The on-device greedy pick (kernels/glm_pick.hpp) against its host
// oracles: glm_pick_local's top-2 vs glm_sample::local_max plus the gen
// log's runner-up scan; glm_pick_verdict's merge vs glm_sample::merge_greedy
// and its judge vs judge_verify, over a SIMULATED world (each rank's table
// produced by the kernel on its slice, the fold emulated as an exact host
// sum — which is what the bus's SUM over disjoint slots is); the digest
// group's agreement/mismatch detection; the wire table's layout.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/glm_pick.hpp"
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
  dgpp::glm_pick_verdict(d_table, rows, world, rank, d_fed, d_verdict, d_carry,
                         nullptr);
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

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
