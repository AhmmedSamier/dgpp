// Sampler tests (host-only): the three distributed fast paths must be
// BITWISE-EQUAL to the centralized oracle (the M6 d3 exit criterion —
// no tolerance is permitted, the design's canonical candidate order
// makes the paths converge on identical inputs to the shared selection
// core). These pin: the merge is the exact global top-k under ties, the
// RNG is counter-reproducible, penalties follow the HF warper formulas,
// and the documented filter edge cases hold.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/test.hpp"
#include "models/glm_sampler.hpp"

namespace {

using dgpp::glm_sample::apply_penalties;
using dgpp::glm_sample::Candidate;
using dgpp::glm_sample::count_context;
using dgpp::glm_sample::merge_greedy;
using dgpp::glm_sample::merge_topk;
using dgpp::glm_sample::local_max;
using dgpp::glm_sample::local_topk;
using dgpp::glm_sample::Params;
using dgpp::glm_sample::Rng;
using dgpp::glm_sample::Result;
using dgpp::glm_sample::sample_reference;
using dgpp::glm_sample::select_from_sorted;
using dgpp::glm_sample::sort_slice;
using dgpp::glm_sample::uniform01;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Bitwise float compare (uint32 pattern): the parity contract is exact,
// so == on floats is too loose an instrument for a -0.0/+0.0 or NaN mix.
bool same_bits(float a, float b) {
  uint32_t ua, ub;
  std::memcpy(&ua, &a, 4);
  std::memcpy(&ub, &b, 4);
  return ua == ub;
}

// A deterministic fixture RNG (NOT the sampler's): builds test vectors.
struct FixtureRng {
  uint64_t s = 0x1234567;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() {  // [0, 1): 53-bit draw scaled by 2^-53
    return static_cast<float>(next() >> 11) * (1.0f / 9007199254740992.0f);
  }
};

std::vector<float> make_logits(int n, FixtureRng& fx, float spread = 8.0f) {
  std::vector<float> v(n);
  for (float& x : v) x = (fx.unit() - 0.5f) * spread;
  return v;
}

// Splits a vector into `world` contiguous slices (uneven on purpose —
// vocab/world is rarely integral).
std::vector<std::pair<int, int>> slices_for(int n, int world) {
  std::vector<std::pair<int, int>> out;
  int begin = 0;
  for (int r = 0; r < world; ++r) {
    const int len = (n - begin + (world - r - 1)) / (world - r);
    out.push_back({begin, len});
    begin += len;
  }
  return out;
}

// The distributed top-k pipeline over simulated shards: per-slice
// penalties, per-slice exact local top-k, merge, shared selection.
Result distributed_topk(const std::vector<float>& logits, int world,
                        const Params& p, Rng& rng,
                        const std::vector<int32_t>& context) {
  const std::vector<Candidate> no_candidates;  // unused sentinel
  (void)no_candidates;
  const auto counts = count_context(context);
  const int k_eff = p.top_k;
  std::vector<std::vector<Candidate>> locals;
  for (const auto& [begin, len] : slices_for(static_cast<int>(logits.size()), world)) {
    std::vector<float> slice(logits.begin() + begin, logits.begin() + begin + len);
    apply_penalties(slice.data(), len, begin, p, counts);
    locals.push_back(local_topk(slice.data(), len, begin, k_eff));
  }
  return select_from_sorted(merge_topk(std::move(locals), k_eff), p, rng);
}

// The distributed greedy pipeline over simulated shards.
int32_t distributed_greedy(const std::vector<float>& logits, int world,
                           const Params& p,
                           const std::vector<int32_t>& context) {
  const auto counts = count_context(context);
  std::vector<Candidate> maxima;
  for (const auto& [begin, len] : slices_for(static_cast<int>(logits.size()), world)) {
    std::vector<float> slice(logits.begin() + begin, logits.begin() + begin + len);
    apply_penalties(slice.data(), len, begin, p, counts);
    maxima.push_back(local_max(slice.data(), len, begin));
  }
  return merge_greedy(maxima);
}

bool results_identical(const Result& a, const Result& b) {
  if (a.token != b.token) return false;
  if (!same_bits(a.logprob, b.logprob)) return false;
  if (a.top_logprobs.size() != b.top_logprobs.size()) return false;
  for (size_t i = 0; i < a.top_logprobs.size(); ++i) {
    if (a.top_logprobs[i].first != b.top_logprobs[i].first) return false;
    if (!same_bits(a.top_logprobs[i].second, b.top_logprobs[i].second)) {
      return false;
    }
  }
  return true;
}

}  // namespace

DGPP_TEST(rng_is_counter_reproducible_and_seed_sensitive) {
  Rng a{42, 0}, b{42, 0}, c{42, 1}, d{43, 0};
  for (int i = 0; i < 100; ++i) {
    const double da = uniform01(a), db = uniform01(b);
    require(da == db, "same seed+counter must reproduce the draw");
    require(da != uniform01(c), "counter must change the draw");
    require(da != uniform01(d), "seed must change the draw");
    ++a.counter; ++b.counter; ++c.counter; ++d.counter;
  }
  Rng r{42, 0};
  for (int i = 0; i < 1000; ++i) {
    const double u = uniform01(r);
    require(u >= 0.0 && u < 1.0, "uniform01 out of [0,1)");
    ++r.counter;
  }
}

DGPP_TEST(greedy_takes_argmax_lowest_id_on_ties) {
  const std::vector<float> v{0.5f, 3.25f, 3.25f, -1.0f, 3.25f, 0.0f};
  const std::vector<Candidate> sorted = sort_slice(v.data(), v.size(), 0);
  Params p;  // temperature 1: not the greedy branch — use the greedy one:
  p.temperature = 0.0f;
  p.logprobs = 3;
  Rng rng{1, 0};
  const uint64_t counter_before = rng.counter;
  const Result r = select_from_sorted(sorted, p, rng);
  require(r.token == 1, "tie must break to the LOWEST id");
  require(rng.counter == counter_before, "greedy must not consume RNG");
  require(r.top_logprobs.size() == 3, "logprobs count");
  require(r.top_logprobs[0].first == 1, "top logprob is the argmax");
}

DGPP_TEST(distributed_greedy_matches_oracle_with_cross_shard_ties) {
  FixtureRng fx;
  for (int world : {1, 2, 3, 5, 7}) {
    // Duplicate the max value across MANY positions so ties land across
    // shard boundaries for every split.
    std::vector<float> v = make_logits(29, fx);
    for (int i : {0, 3, 8, 14, 22, 28}) v[i] = 9.5f;
    Params p;
    p.temperature = 0.0f;
    Rng oracle_rng{1, 0};
    const int32_t oracle =
        select_from_sorted(sort_slice(v.data(), v.size(), 0), p, oracle_rng)
            .token;
    const int32_t dist = distributed_greedy(v, world, p, {});
    require(oracle == 0, "fixture sanity: lowest tied id wins");
    require(dist == oracle, "distributed greedy != oracle at world " + std::to_string(world));
  }
}

DGPP_TEST(distributed_topk_matches_oracle_bitwise_across_params) {
  FixtureRng fx;
  int cases = 0;
  for (int world : {1, 2, 3, 5}) {
    for (int n : {1, 13, 29}) {
      const std::vector<float> v = make_logits(n, fx);
      // context for penalties, deliberately including ids that fall in
      // different shards
      const std::vector<int32_t> context{0, n / 2, n - 1, n / 2, 0};
      for (float temperature : {1.0f, 0.7f}) {
        for (int k : {1, 3, 40}) {
          for (float top_p : {1.0f, 0.5f, 0.9f}) {
            for (float min_p : {0.0f, 0.05f}) {
              for (int seed : {0, 1, 2}) {
                for (int use_penalties : {0, 1}) {
                  Params p;
                  p.temperature = temperature;
                  p.top_k = k;
                  p.top_p = top_p;
                  p.min_p = min_p;
                  p.logprobs = 5;
                  if (use_penalties) {
                    p.repetition_penalty = 1.3f;
                    p.frequency_penalty = 0.2f;
                    p.presence_penalty = 0.1f;
                  }
                  Rng ra{static_cast<uint64_t>(seed), 0};
                  Rng rb{static_cast<uint64_t>(seed), 0};
                  const Result oracle = sample_reference(v.data(), n, p, ra, context);
                  const Result dist = distributed_topk(v, world, p, rb, context);
                  require(results_identical(oracle, dist),
                          "top-k path drifted from the oracle: world=" +
                              std::to_string(world) + " n=" +
                              std::to_string(n) + " seed=" +
                              std::to_string(seed));
                  require(ra.counter == rb.counter,
                          "RNG counters must advance identically");
                  ++cases;
                }
              }
            }
          }
        }
      }
    }
  }
  require(cases > 2000, "expected a broad sweep, got " + std::to_string(cases));
}

DGPP_TEST(topk_merge_is_exact_under_ties_spanning_shards) {
  const std::vector<float> v{5.0f, 1.0f, 5.0f, 5.0f, 0.5f, 2.0f, 5.0f};
  // Global top-3 under (value desc, id asc) = {0, 2, 3}.
  std::vector<std::vector<Candidate>> locals;
  locals.push_back(local_topk(v.data(), 3, 0, 3));  // ids 0,1,2
  locals.push_back(local_topk(v.data() + 3, 4, 3, 3));  // ids 3,4,5,6
  const std::vector<Candidate> merged = merge_topk(std::move(locals), 3);
  require(merged.size() == 3, "merge keeps k");
  require(merged[0].id == 0 && merged[0].logit == 5.0f, "tie order 0");
  require(merged[1].id == 2 && merged[1].logit == 5.0f, "tie order 1");
  require(merged[2].id == 3 && merged[2].logit == 5.0f, "tie order 2");
}

DGPP_TEST(penalties_follow_hf_warper_formulas) {
  // logit 4.0f at id 0: positive -> rep divides; id 1 negative -> rep
  // multiplies; frequency is additive per count; presence additive once.
  std::vector<float> v{4.0f, -2.0f, 1.0f};
  std::vector<int32_t> context{0, 0, 0, 1, 2};
  Params p;
  p.repetition_penalty = 2.0f;
  p.frequency_penalty = 0.25f;
  p.presence_penalty = 0.5f;
  apply_penalties(v.data(), 3, 0, p, count_context(context));
  // id 0: 4/2 - 0.25*3 - 0.5 = 2 - 0.75 - 0.5 = 0.75
  require(same_bits(v[0], 0.75f), "id 0 penalty math");
  // id 1: -2*2 - 0.25*1 - 0.5 = -4.75
  require(same_bits(v[1], -4.75f), "id 1 penalty math");
  // id 2: 1/2 - 0.25*1 - 0.5 = -0.25
  require(same_bits(v[2], -0.25f), "id 2 penalty math");
}

DGPP_TEST(apply_penalties_touches_only_its_slice) {
  std::vector<float> v{1.0f, 1.0f, 1.0f, 1.0f};
  const auto counts = count_context({1});  // global id 1
  Params p;
  p.presence_penalty = 1.0f;
  // Slice [2, 4): global id 1 is NOT in it — nothing changes.
  apply_penalties(v.data() + 2, 2, 2, p, counts);
  require(same_bits(v[2], 1.0f) && same_bits(v[3], 1.0f),
          "out-of-slice ids must be ignored");
  // Slice [0, 2): global id 1 IS in it — only v[1] changes.
  apply_penalties(v.data(), 2, 0, p, counts);
  require(same_bits(v[0], 1.0f), "id 0 untouched");
  require(same_bits(v[1], 0.0f), "id 1 penalized");
}

DGPP_TEST(min_p_drops_tail_never_sampled) {
  // Three well-separated levels; min_p 0.5 must make the tail level
  // unsampleable regardless of draw.
  std::vector<float> v{4.0f, 3.9f, 3.5f, -6.0f, -6.5f, -7.0f};
  Params p;
  p.temperature = 1.0f;
  p.top_k = 6;
  p.min_p = 0.5f;
  for (int seed = 0; seed < 200; ++seed) {
    Rng rng{static_cast<uint64_t>(seed), 0};
    const Result r = select_from_sorted(sort_slice(v.data(), v.size(), 0), p, rng);
    require(r.token < 3, "min_p must make the tail unsampleable");
  }
}

DGPP_TEST(top_p_keeps_the_crossing_token) {
  // Crafted so only two tokens can survive top-p; assert the third is
  // never chosen and never appears in the logprobs list.
  std::vector<float> v{5.0f, 4.0f, 3.0f, 2.0f};
  Params p;
  p.temperature = 1.0f;
  p.top_k = 4;
  p.top_p = 0.9f;
  p.logprobs = 4;
  for (int seed = 0; seed < 200; ++seed) {
    Rng rng{static_cast<uint64_t>(seed), 0};
    const Result r = select_from_sorted(sort_slice(v.data(), v.size(), 0), p, rng);
    require(r.token < 3, "top-p tail must be unsampleable");
    for (const auto& lp : r.top_logprobs) {
      require(lp.first < 3, "top-p must drop the tail from logprobs");
    }
  }
}

DGPP_TEST(temperature_zero_equals_greedy) {
  FixtureRng fx;
  const std::vector<float> v = make_logits(31, fx);
  Params sampled;
  sampled.temperature = 0.0f;  // sampling spec with T=0 -> greedy branch
  Params greedy;
  greedy.temperature = 0.0f;
  Rng ra{7, 0}, rb{7, 0};
  const Result a = sample_reference(v.data(), v.size(), sampled, ra, {});
  const Result b = select_from_sorted(sort_slice(v.data(), v.size(), 0), greedy, rb);
  require(results_identical(a, b), "T=0 sample_reference must be greedy");
}

DGPP_TEST(empty_candidates_rejected) {
  Params p;
  Rng rng{1, 0};
  bool threw = false;
  try {
    select_from_sorted({}, p, rng);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "empty candidate list must throw");
}
