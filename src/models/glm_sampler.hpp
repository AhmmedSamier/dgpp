#pragma once
// Exact sampling over the (possibly vocab-sharded) lm head — M6 d3,
// DESIGN §10. Three execution paths, ONE selection semantics:
//
//   greedy  : reduce the per-rank (value, token_id) maxima
//   top_k   : merge each rank's exact local top-k, filter/sample at rank 0
//   gather  : pull the full FP32 vocab to rank 0, sample centrally
//
// The distributed paths are BITWISE-EQUAL to the centralized oracle by
// construction, not by tolerance: every path funnels into
// select_from_sorted() over a candidate list in the canonical order
// (logit descending, then token id ascending — a total order), and the
// merge provably produces the same set in the same order as sorting the
// full vocabulary (a token in the global top-k is, per slice, inside that
// slice's local top-k; the merged list is the k-way prefix of the same
// total order). The oracle "match" gate therefore asserts exact float
// equality, and any arithmetic that could drift (softmax denominators,
// top-p cumulative sums) is defined to run in that same listed order in
// the one shared implementation.
//
// Numerics (the contract, in HF warper order):
//   1. penalties on raw logits: repetition (sign-based divide/multiply),
//      then frequency (additive per count), then presence (additive once)
//   2. temperature scaling (T <= 0 -> greedy; sampling requires T > 0)
//   3. top-k truncation to the first k candidates (k >= vocab: no-op)
//   4. min-p: p_i = e_i/den_k >= min_p * p_max (den over the post-top-k
//      survivors; p_max = p_0, which every later filter keeps)
//   5. top-p: keep the smallest prefix with cumulative p >= top_p
//      (the crossing token stays in — the HF warper's rule)
//   6. final probabilities over the surviving set; one uniform draw
//      selects (fp64 walk over fp32 probabilities)
//
// The RNG is COUNTER-based, not stream-based: one draw per sampled token
// from (seed, counter). The counter is the resumption token DESIGN §10
// broadcasts with the chosen id — any rank can reconstruct any request's
// draw sequence from the seed plus the step count, which is what makes
// speculative verification (M8) and rank-consistency checks cheap.
// Greedy draws NOTHING: the counter advances only on stochastic draws.
//
// Host-side by design: the logits buffers are managed memory (the
// loader contract already grants CPU-readable resident buffers), so a
// rank samples its vocab slice without a staging copy.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dgpp::glm_sample {

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

// splitmix64 finalizer: a PractRand-clean scramble, so a linear counter
// yields a well-mixed stream (this is the same construction splitmix64
// itself uses over sequential state).
inline uint64_t splitmix64_step(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// The sampling RNG state. `counter` advances exactly once per stochastic
// draw; `seed` is the request's fixed-seed identity.
struct Rng {
  uint64_t seed = 0;
  uint64_t counter = 0;
};

// One uniform in [0, 1): the top 53 bits of the draw, as an fp64.
inline double uniform01(const Rng& rng) {
  const uint64_t draw = splitmix64_step(splitmix64_step(rng.counter) ^ rng.seed);
  return static_cast<double>(draw >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Request spec
// ---------------------------------------------------------------------------

struct Params {
  float temperature = 1.0f;        // <= 0 -> greedy
  int top_k = 0;                    // 0 = disabled (the gather path)
  float top_p = 1.0f;               // >= 1 = disabled
  float min_p = 0.0f;               // <= 0 = disabled
  float repetition_penalty = 1.0f;  // 1 = disabled (multiplicative)
  float frequency_penalty = 0.0f;   // additive per context count
  float presence_penalty = 0.0f;    // additive once for seen ids
  int logprobs = 0;                 // report top-N (id, logprob) pairs
};

struct Candidate {
  int32_t id = -1;
  float logit = 0.0f;
};

struct Result {
  int32_t token = -1;
  float logprob = 0.0f;  // log-softmax of the chosen token, final dist
  std::vector<std::pair<int32_t, float>> top_logprobs;  // if logprobs > 0
};

// ---------------------------------------------------------------------------
// Penalties (warper stage 1)
// ---------------------------------------------------------------------------

// Token counts over the request context (prompt + generated so far).
// Same on every rank — the scheduler's rank-consistency guarantee.
inline std::unordered_map<int32_t, int32_t> count_context(
    const std::vector<int32_t>& ids) {
  std::unordered_map<int32_t, int32_t> counts;
  for (int32_t id : ids) ++counts[id];
  return counts;
}

// Applies the three penalties to one rank's vocab slice, in HF warper
// order, on the RAW (unscaled) logits. Ids outside
// [slice_begin, slice_begin + n) are ignored: a token lives on exactly
// one rank, so the union of slices applies each penalty exactly once.
inline void apply_penalties(float* logits, int n, int slice_begin,
                            const Params& p,
                            const std::unordered_map<int32_t, int32_t>& counts) {
  for (int i = 0; i < n; ++i) {
    const auto it = counts.find(slice_begin + i);
    if (it == counts.end()) continue;
    const int32_t count = it->second;
    float v = logits[i];
    if (p.repetition_penalty != 1.0f) {
      // HF semantics: divide positive logits, multiply negative ones.
      v = v > 0.0f ? v / p.repetition_penalty : v * p.repetition_penalty;
    }
    v -= p.frequency_penalty * static_cast<float>(count);
    v -= p.presence_penalty * 1.0f;
    logits[i] = v;
  }
}

// ---------------------------------------------------------------------------
// Canonical order
// ---------------------------------------------------------------------------

// The total order shared by every path (and the parity proof's anchor).
inline bool candidate_before(const Candidate& a, const Candidate& b) {
  if (a.logit != b.logit) return a.logit > b.logit;
  return a.id < b.id;
}

// Sorts a full slice into the canonical order (the oracle's candidate
// list; also the shard test's reference).
inline std::vector<Candidate> sort_slice(const float* logits, int n,
                                        int slice_begin) {
  std::vector<Candidate> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) out.push_back({slice_begin + i, logits[i]});
  std::sort(out.begin(), out.end(), candidate_before);
  return out;
}

// ---------------------------------------------------------------------------
// Selection core (shared by every path — the bitwise-parity anchor)
// ---------------------------------------------------------------------------

// `sorted` is the (already canonical-order) candidate list covering the
// global top-k-or-more set with penalties applied and is NOT modified.
// Selects per the documented pipeline; returns the chosen token with
// its logprob (and the top-N survivors' logprobs when requested).
// Throws on an empty list.
inline Result select_from_sorted(const std::vector<Candidate>& sorted,
                                const Params& p, Rng& rng) {
  if (sorted.empty()) {
    throw std::runtime_error("glm_sample: empty candidate list");
  }

  // Greedy: argmax under the canonical order; reported probabilities are
  // the RAW distribution's (what a greedy consumer expects logprobs to
  // mean), not a renormalized singleton.
  if (p.temperature <= 0.0f) {
    Result r;
    r.token = sorted[0].id;
    float mx = sorted[0].logit;
    double acc = 0.0;
    for (const Candidate& c : sorted) acc += std::exp(c.logit - mx);
    const float lse = mx + static_cast<float>(std::log(acc));
    r.logprob = sorted[0].logit - lse;
    const int n = std::min<int>(p.logprobs, static_cast<int>(sorted.size()));
    for (int i = 0; i < n; ++i) {
      r.top_logprobs.emplace_back(sorted[i].id, sorted[i].logit - lse);
    }
    return r;
  }

  // Temperature scale (fp32 division — IEEE-deterministic everywhere).
  std::vector<float> scaled;
  scaled.reserve(sorted.size());
  for (const Candidate& c : sorted) scaled.push_back(c.logit / p.temperature);

  // top-k truncation (k >= size: no-op).
  size_t kept = sorted.size();
  if (p.top_k > 0) kept = std::min<size_t>(kept, static_cast<size_t>(p.top_k));

  // min-p over the post-top-k distribution. exps are computed once —
  // the max is scaled[0] and every filter keeps index 0, so no recompute
  // can drift. Denominators always sum survivors in listed order.
  std::vector<float> exps(kept);
  for (size_t i = 0; i < kept; ++i) {
    exps[i] = std::exp(scaled[i] - scaled[0]);
  }
  float den = 0.0f;
  for (size_t i = 0; i < kept; ++i) den += exps[i];
  if (p.min_p > 0.0f) {
    const float threshold = p.min_p * (exps[0] / den);
    std::vector<float> survivors;
    for (size_t i = 0; i < kept; ++i) {
      if (exps[i] / den >= threshold) survivors.push_back(exps[i]);
    }
    // index 0 always survives (p_max is itself), so the set is non-empty;
    // the explicit compaction keeps the survivor order = listed order.
    exps = std::move(survivors);
  }

  // top-p: smallest prefix with cumulative probability >= top_p, the
  // crossing token included.
  den = 0.0f;
  for (float e : exps) den += e;
  size_t final_count = exps.size();
  if (p.top_p < 1.0f) {
    float cum = 0.0f;
    size_t cut = exps.size();  // default: keep all (sum may fall short)
    for (size_t i = 0; i < exps.size(); ++i) {
      cum += exps[i] / den;
      if (cum >= p.top_p) {
        cut = i + 1;
        break;
      }
    }
    final_count = cut;
  }

  // Final distribution over the surviving set.
  float final_den = 0.0f;
  for (size_t i = 0; i < final_count; ++i) final_den += exps[i];
  const float lse = scaled[0] + std::log(final_den);

  // One draw. fp64 walk over the fp32 probabilities: the walk order is
  // the listed order, so every rank that computes this computes it
  // identically.
  const double r = uniform01(rng);
  ++rng.counter;
  double cum = 0.0;
  size_t chosen = final_count - 1;
  for (size_t i = 0; i < final_count; ++i) {
    cum += exps[i] / final_den;
    if (cum > r) {
      chosen = i;
      break;
    }
  }

  Result out;
  out.token = sorted[chosen].id;
  out.logprob = scaled[chosen] - lse;
  const int n = std::min<int>(p.logprobs, static_cast<int>(final_count));
  for (int i = 0; i < n; ++i) {
    out.top_logprobs.emplace_back(sorted[i].id, scaled[i] - lse);
  }
  return out;
}

// ---------------------------------------------------------------------------
// The centralized oracle (full vocab on one host)
// ---------------------------------------------------------------------------

inline Result sample_reference(const float* logits, int vocab,
                               const Params& p, Rng& rng,
                               const std::vector<int32_t>& context_ids) {
  std::vector<float> v(logits, logits + vocab);
  apply_penalties(v.data(), vocab, 0, p, count_context(context_ids));
  const std::vector<Candidate> sorted = sort_slice(v.data(), vocab, 0);
  return select_from_sorted(sorted, p, rng);
}

// ---------------------------------------------------------------------------
// Distributed fast paths (per-rank host code over vocab slices)
// ---------------------------------------------------------------------------

// Path 1 — greedy. Each rank contributes its slice maximum.
inline Candidate local_max(const float* logits, int n, int slice_begin) {
  // Ascending scan with strict '>' keeps the LOWEST id on ties — the
  // canonical order's tie-break, for free.
  Candidate best{slice_begin, logits[0]};
  for (int i = 1; i < n; ++i) {
    if (logits[i] > best.logit) best = {slice_begin + i, logits[i]};
  }
  return best;
}

inline int32_t merge_greedy(const std::vector<Candidate>& local_maxima) {
  Candidate best = local_maxima[0];
  for (const Candidate& c : local_maxima) {
    if (candidate_before(c, best)) best = c;
  }
  return best.id;
}

// Path 2 — finite top-k. Each rank contributes its exact local top-k
// (canonical order); the merge is a sort of the union under the same
// total order, truncated to k — provably the global top-k in the global
// canonical order. k <= 0 is invalid here (that is the gather path).
inline std::vector<Candidate> local_topk(const float* logits, int n,
                                        int slice_begin, int k) {
  std::vector<Candidate> all;
  all.reserve(n);
  for (int i = 0; i < n; ++i) all.push_back({slice_begin + i, logits[i]});
  // partial_sort: O(n log k) — this runs per decode step on a ~38k-entry
  // vocab slice, so the full O(n log n) sort would show up in the
  // step-time budget.
  const size_t k_eff = std::min<size_t>(n, static_cast<size_t>(k));
  std::partial_sort(all.begin(), all.begin() + k_eff, all.end(),
                    candidate_before);
  all.resize(k_eff);
  return all;
}

inline std::vector<Candidate> merge_topk(
    std::vector<std::vector<Candidate>> local_topks, int k) {
  std::vector<Candidate> merged;
  for (auto& l : local_topks) {
    merged.insert(merged.end(), l.begin(), l.end());
  }
  std::sort(merged.begin(), merged.end(), candidate_before);
  merged.resize(std::min<size_t>(merged.size(), static_cast<size_t>(k)));
  return merged;
}

// Exact full-distribution normalizer for a host-visible logit slice. The
// teacher-forced sampling profiler uses one value per vocab shard and folds
// those with logaddexp; this is intentionally fp64 measurement arithmetic,
// separate from the fp32 device verdict that the production sampler will use.
inline double slice_logsumexp(const float* logits, int n) {
  if (logits == nullptr || n <= 0)
    throw std::invalid_argument("glm_sample: empty logit slice");
  double top = -INFINITY;
  for (int i = 0; i < n; ++i)
    top = std::max(top, static_cast<double>(logits[i]));
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    sum += std::exp(static_cast<double>(logits[i]) - top);
  return top + std::log(sum);
}

// Probability mass covered by each requested prefix of an already
// canonical global candidate list, under the FULL vocabulary normalizer.
// This is the sizing instrument for the device pick table: at top_p=0.95 a
// prefix whose mass is below 0.95 cannot resolve nucleus sampling locally
// and must take the exact full-logit fallback.
inline std::vector<double> topk_probability_masses(
    const std::vector<Candidate>& sorted, double global_logsumexp,
    const std::vector<int>& ks) {
  if (sorted.empty())
    throw std::invalid_argument("glm_sample: empty top-k candidate list");
  if (!std::isfinite(global_logsumexp))
    throw std::invalid_argument("glm_sample: non-finite global log-sum-exp");
  int previous = 0;
  for (int k : ks) {
    if (k <= previous)
      throw std::invalid_argument(
          "glm_sample: top-k mass prefixes must be positive and increasing");
    previous = k;
  }

  std::vector<double> out;
  out.reserve(ks.size());
  double mass = 0.0;
  size_t next_k = 0;
  for (size_t i = 0; i < sorted.size() && next_k < ks.size(); ++i) {
    mass += std::exp(static_cast<double>(sorted[i].logit) - global_logsumexp);
    while (next_k < ks.size() && i + 1 >= static_cast<size_t>(ks[next_k])) {
      out.push_back(mass);
      ++next_k;
    }
  }
  // A diagnostic over a tiny synthetic vocabulary may request a prefix
  // wider than the vocabulary. Its mass is simply the complete available
  // candidate set, matching top-k's k>=vocab no-op semantics.
  while (next_k < ks.size()) {
    out.push_back(mass);
    ++next_k;
  }
  return out;
}

}  // namespace dgpp::glm_sample
