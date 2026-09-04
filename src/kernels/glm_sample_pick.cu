#include "kernels/glm_sample_pick.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/det_math.hpp"
#include "kernels/topk_select.cuh"

// Compiled with --fmad=false (CMakeLists): no multiply-add here may be
// contracted, so every expression rounds exactly as the host's does under
// -ffp-contract=off. The deterministic exp/log name fma() themselves.

namespace dgpp {

namespace {

constexpr int kLocalThreads = 1024;
constexpr int kVerdictThreads = 256;
constexpr int kKeyIdxBits = 21;  // vocab ids < 2^21
constexpr uint64_t kKeyIdxMask = (1ull << kKeyIdxBits) - 1;
constexpr uint64_t kKeyMax = ~0ull;
constexpr int32_t kNoId = -1;

static_assert(kSampleLseChunk == 256, "the host's glm_sample::kLseChunk");
static_assert(kSelectTile >= 2 * kSampleMaxCandidates,
              "select_topk_stream needs a tile of at least 2k keys");

__device__ inline void encode_digits(uint16_t* slots, uint64_t value,
                                     int digits) {
  for (int d = 0; d < digits; ++d)
    slots[d] = static_cast<uint16_t>((value >> (6 * d)) & 63);
}
__device__ inline uint64_t decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

// (~sortable(logit) << 21) | id: the smallest key is the canonical first
// candidate (glm_sample::candidate_before as an integer order).
__device__ inline uint64_t composite_key(float logit, int32_t id) {
  return (static_cast<uint64_t>(~sortable_f32_dev(logit)) << kKeyIdxBits) |
         static_cast<uint64_t>(id);
}
__device__ inline float key_logit(uint64_t key) {
  const uint32_t s = ~static_cast<uint32_t>(key >> kKeyIdxBits);
  const uint32_t u = (s & 0x80000000u) ? (s & 0x7fffffffu) : ~s;
  return __uint_as_float(u);
}
__device__ inline int32_t key_id(uint64_t key) {
  return static_cast<int32_t>(key & kKeyIdxMask);
}

__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
// glm_sample::uniform01: the top 53 bits of the draw as an fp64 in [0, 1).
__device__ inline double uniform01(uint64_t seed, uint64_t counter) {
  const uint64_t draw = splitmix64(splitmix64(counter) ^ seed);
  return static_cast<double>(draw >> 11) * (1.0 / 9007199254740992.0);
}
__host__ __device__ inline uint64_t verdict_digest(int rows, int accepted,
                                                   const int32_t* winners) {
  uint64_t h = splitmix64(static_cast<uint64_t>(rows));
  h = splitmix64(h ^ static_cast<uint64_t>(accepted));
  for (int r = 0; r < rows; ++r)
    h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(winners[r])));
  return h & ((1ull << kPickDigestBits) - 1);
}

// glm_sample::apply_penalties for one id with count `count`.
__device__ inline float penalize(float v, int32_t count,
                                 const GlmSampleSpec& s) {
  if (s.repetition_penalty != 1.0f)
    v = v > 0.0f ? __fdiv_rn(v, s.repetition_penalty)
                 : __fmul_rn(v, s.repetition_penalty);
  v = __fmaf_rn(-s.frequency_penalty, static_cast<float>(count), v);
  v = __fsub_rn(v, s.presence_penalty);
  return v;
}

// Block-wide min over 64-bit keys (one per thread).
__device__ inline uint64_t block_min_key(uint64_t key, uint64_t* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const uint32_t hi = __shfl_xor_sync(~0u, static_cast<uint32_t>(key >> 32), off);
    const uint32_t lo = __shfl_xor_sync(~0u, static_cast<uint32_t>(key), off);
    const uint64_t other = (static_cast<uint64_t>(hi) << 32) | lo;
    key = other < key ? other : key;
  }
  if (lane == 0) scratch[warp] = key;
  __syncthreads();
  key = threadIdx.x < nwarps ? scratch[threadIdx.x] : kKeyMax;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const uint32_t hi = __shfl_xor_sync(~0u, static_cast<uint32_t>(key >> 32), off);
      const uint32_t lo = __shfl_xor_sync(~0u, static_cast<uint32_t>(key), off);
      const uint64_t other = (static_cast<uint64_t>(hi) << 32) | lo;
      key = other < key ? other : key;
    }
    if (lane == 0) scratch[0] = key;
  }
  __syncthreads();
  const uint64_t out = scratch[0];
  __syncthreads();
  return out;
}

__device__ inline float block_max_f(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = threadIdx.x < nwarps ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();
  return out;
}

struct SliceKeyFn {
  static constexpr bool kWarpCooperative = false;
  const float* slice;
  int vocab_begin;
  __device__ uint64_t operator()(int64_t i) const {
    return composite_key(slice[i], vocab_begin + static_cast<int32_t>(i));
  }
};

// ---------------------------------------------------------------------------
// Kernel 1
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kLocalThreads) sample_local_kernel(
    float* __restrict__ logits, int rows, int vocab_count, int vocab_begin,
    int vocab_size, int rank, int world, int candidates,
    const GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ fed, const int64_t* __restrict__ positions,
    int position_stride, int32_t* __restrict__ counts,
    const uint64_t* __restrict__ carry_digest, uint16_t* __restrict__ table,
    GlmPickLocal* __restrict__ locals) {
  __shared__ uint32_t best_hi[kSampleMaxCandidates];
  __shared__ uint32_t best_lo[kSampleMaxCandidates];
  __shared__ uint32_t tile_hi[kSelectTile];
  __shared__ uint32_t tile_lo[kSelectTile];
  __shared__ double partials[kLocalThreads];
  __shared__ float fred[32];
  __shared__ uint64_t kred[32];

  // One block per REQUEST, its rows in order: a T=2 request's second row
  // penalizes with the draft already counted, which the first row's pass
  // must precede.
  const int q = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const size_t row_slots = static_cast<size_t>(world) * group;
  const bool active =
      positions == nullptr || positions[q * position_stride] >= 0;
  GlmSampleSpec spec = specs[q];
  const bool penalized = spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  // The full path: stochastic rows, and greedy rows that report logprobs or
  // carry penalties (their normalizer and scaled logits are the raw
  // distribution's: temperature 1).
  const bool sampled =
      active && (spec.temperature > 0.0f || spec.logprobs >= 0 || penalized);
  if (spec.temperature <= 0.0f) spec.temperature = 1.0f;
  int32_t* my_counts = counts + static_cast<size_t>(q) * vocab_size;

  for (int t = 0; t < rows_per_request; ++t) {
    const int row = q * rows_per_request + t;
    uint16_t* row_base = table + static_cast<size_t>(row) * row_slots;
    uint16_t* mine = row_base + static_cast<size_t>(rank) * group;
    float* slice = logits + static_cast<size_t>(row) * vocab_count;

    // Zero this row's candidate region (every rank's group: the fold needs
    // zeros in the foreign slots); block 0 also owns the digest group and
    // the even-count pad, once.
    for (size_t i = tid; i < row_slots; i += kLocalThreads) row_base[i] = 0;
    if (q == 0 && t == 0) {
      const size_t total = glm_sample_table_elems(rows, world, candidates);
      const size_t begin = static_cast<size_t>(rows) * row_slots;
      for (size_t i = begin + tid; i < total; i += kLocalThreads) table[i] = 0;
    }
    __syncthreads();

    if (!active) {
      if (tid == 0) {
        locals[row].best_id = kNoId;
        locals[row].best_logit = -INFINITY;
        locals[row].second_logit = -INFINITY;
      }
      continue;
    }

    if (!sampled) {
      // The greedy row: the canonical argmax (the smallest composite key),
      // written as the one candidate; every other slot carries the empty id.
      uint64_t best = kKeyMax;
      for (int i = tid; i < vocab_count; i += kLocalThreads) {
        const uint64_t key = composite_key(slice[i], vocab_begin + i);
        best = key < best ? key : best;
      }
      best = block_min_key(best, kred);
      for (int j = tid; j < candidates; j += kLocalThreads) {
        uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
        if (j == 0) {
          encode_digits(slot, static_cast<uint64_t>(__float_as_uint(key_logit(best))),
                        kPickLogitDigits);
          encode_digits(slot + kPickLogitDigits,
                        static_cast<uint64_t>(key_id(best)), kPickIdDigits);
        } else {
          encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
        }
      }
      if (tid == 0) {
        locals[row].best_id = key_id(best);
        locals[row].best_logit = key_logit(best);
        locals[row].second_logit = -INFINITY;
      }
      __syncthreads();
      continue;
    }

    // ---- the sampled row ------------------------------------------------
    // 1. The row's fed token joins the request's context, then the
    //    penalties apply in place (glm_sample::apply_penalties).
    if (tid == 0) {
      const int64_t token = fed[row];
      if (token >= 0 && token < vocab_size) my_counts[token] += 1;
    }
    __syncthreads();
    for (int i = tid; i < vocab_count; i += kLocalThreads) {
      const int32_t c = my_counts[vocab_begin + i];
      if (c != 0) slice[i] = penalize(slice[i], c, spec);
    }
    __syncthreads();

    // 2. The slice's temperature-scaled log-sum-exp in the host's chunked
    //    order: the max, then per-chunk sequential fp64 sums, then the
    //    chunk partials folded in order by one thread.
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_count; i += kLocalThreads)
      local_max = fmaxf(local_max, __fdiv_rn(slice[i], spec.temperature));
    const float top = block_max_f(local_max, fred);
    const int nchunks = (vocab_count + kSampleLseChunk - 1) / kSampleLseChunk;
    for (int c = tid; c < nchunks; c += kLocalThreads) {
      const int c0 = c * kSampleLseChunk;
      const int c1 = min(vocab_count, c0 + kSampleLseChunk);
      double partial = 0.0;
      for (int i = c0; i < c1; ++i)
        partial += detmath::exp_d(
            static_cast<double>(__fdiv_rn(slice[i], spec.temperature)) -
            static_cast<double>(top));
      partials[c] = partial;
    }
    __syncthreads();
    if (tid == 0) {
      double sum = 0.0;
      for (int c = 0; c < nchunks; ++c) sum += partials[c];
      const double lse = static_cast<double>(top) + detmath::log_d(sum);
      encode_digits(mine + static_cast<size_t>(candidates) * kPickSlotsPerRank,
                    detmath::bits_of(lse), kSampleLseDigits);
    }

    // 3. The exact local top-k in canonical order. The streaming select's
    //    bitonic merge needs a power-of-two width: select the next power of
    //    two at or above k (at most kSampleMaxCandidates, half a tile) and
    //    publish the first k — the same set and order, k being a prefix of
    //    the sorted selection.
    const int k = min(candidates, vocab_count);
    int k_sel = 1;
    while (k_sel < k) k_sel <<= 1;
    for (int i = tid; i < kSampleMaxCandidates; i += kLocalThreads) {
      best_hi[i] = 0xFFFFFFFFu;
      best_lo[i] = 0xFFFFFFFFu;
    }
    __syncthreads();
    SliceKeyFn fn{slice, vocab_begin};
    select_topk_stream(fn, 0, vocab_count, best_hi, best_lo, tile_hi, tile_lo,
                       k_sel);
    for (int j = tid; j < candidates; j += kLocalThreads) {
      uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
      const uint64_t key =
          j < k ? (static_cast<uint64_t>(best_hi[j]) << 32) | best_lo[j] : kKeyMax;
      if (key == kKeyMax) {
        encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
      } else {
        encode_digits(slot, static_cast<uint64_t>(__float_as_uint(key_logit(key))),
                      kPickLogitDigits);
        encode_digits(slot + kPickLogitDigits,
                      static_cast<uint64_t>(key_id(key)), kPickIdDigits);
      }
    }
    if (tid == 0) {
      const uint64_t b0 = (static_cast<uint64_t>(best_hi[0]) << 32) | best_lo[0];
      locals[row].best_id = key_id(b0);
      locals[row].best_logit = key_logit(b0);
      if (k > 1) {
        const uint64_t b1 = (static_cast<uint64_t>(best_hi[1]) << 32) | best_lo[1];
        locals[row].second_logit = b1 == kKeyMax ? -INFINITY : key_logit(b1);
      } else {
        locals[row].second_logit = -INFINITY;
      }
    }
    __syncthreads();
  }
  if (q == 0 && tid == 0)
    encode_digits(table + static_cast<size_t>(rows) * row_slots +
                      static_cast<size_t>(rank) * kPickSlotsPerRank,
                  *carry_digest, kPickSlotsPerRank);
}

// ---------------------------------------------------------------------------
// Kernel 2a: the decision, one block per request (thread 0 decides; the
// block decodes the table cooperatively).
// ---------------------------------------------------------------------------
struct Decision {
  bool resolved;
  bool accepted;  // the speculative accept test (T=2 row 0)
  int32_t token;
  float logprob;
  double covered;
};

// The reported top-N (the host's Result::top_logprobs): the first
// min(N, n) candidates of the decided list under `lse`.
__device__ inline void report_top(const float* logit, const int32_t* id, int n,
                                  float temperature, float lse, int N,
                                  int32_t* out_ids, float* out_lps,
                                  int32_t* out_count) {
  const int count = min(min(N, n), kSampleMaxTopLogprobs);
  for (int i = 0; i < count; ++i) {
    out_ids[i] = id[i];
    out_lps[i] = __fsub_rn(__fdiv_rn(logit[i], temperature), lse);
  }
  *out_count = count;
}

// glm_sample::selector_state's stages 2-5 over prefix [0, n) of the merged
// candidates; exps[] receives the survivors' exps. Returns final_count,
// final_den, lse through the out-params.
__device__ inline void selector_state(const float* logit, int n,
                                      float temperature, int top_k,
                                      float min_p, float top_p, float* exps,
                                      int* final_count, float* final_den,
                                      float* lse) {
  const float scaled0 = __fdiv_rn(logit[0], temperature);
  int kept = n;
  if (top_k > 0) kept = min(kept, top_k);
  for (int i = 0; i < kept; ++i)
    exps[i] = detmath::exp_f(__fsub_rn(__fdiv_rn(logit[i], temperature), scaled0));
  float den = 0.0f;
  for (int i = 0; i < kept; ++i) den = __fadd_rn(den, exps[i]);
  int survivors = kept;
  if (min_p > 0.0f) {
    const float threshold = __fmul_rn(min_p, __fdiv_rn(exps[0], den));
    int w = 0;
    for (int i = 0; i < kept; ++i)
      if (__fdiv_rn(exps[i], den) >= threshold) exps[w++] = exps[i];
    survivors = w;
  }
  den = 0.0f;
  for (int i = 0; i < survivors; ++i) den = __fadd_rn(den, exps[i]);
  int fc = survivors;
  if (top_p < 1.0f) {
    float cum = 0.0f;
    int cut = survivors;
    for (int i = 0; i < survivors; ++i) {
      cum = __fadd_rn(cum, __fdiv_rn(exps[i], den));
      if (cum >= top_p) {
        cut = i + 1;
        break;
      }
    }
    fc = cut;
  }
  float fd = 0.0f;
  for (int i = 0; i < fc; ++i) fd = __fadd_rn(fd, exps[i]);
  *final_count = fc;
  *final_den = fd;
  *lse = __fadd_rn(scaled0, detmath::log_f(fd));
}

// glm_sample::select_from_sorted's draw over the state. `top` (optional)
// receives the final set's top-N report.
struct TopReport {
  int N;
  int32_t* ids;
  float* lps;
  int32_t* count;
};

__device__ inline void selector(const float* logit, const int32_t* id, int n,
                                float temperature, int top_k, float min_p,
                                float top_p, float* exps, uint64_t seed,
                                uint64_t* counter, int32_t* token,
                                float* logprob, const TopReport* top = nullptr) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, n, temperature, top_k, min_p, top_p, exps,
                 &final_count, &final_den, &lse);
  if (top != nullptr)
    report_top(logit, id, final_count, temperature, lse, top->N, top->ids,
               top->lps, top->count);
  const double r = uniform01(seed, *counter);
  *counter += 1;
  double cum = 0.0;
  int chosen = final_count - 1;
  for (int i = 0; i < final_count; ++i) {
    cum += static_cast<double>(__fdiv_rn(exps[i], final_den));
    if (cum > r) {
      chosen = i;
      break;
    }
  }
  *token = id[chosen];
  *logprob = __fsub_rn(__fdiv_rn(logit[chosen], temperature), lse);
}

// glm_sample::spec_select_from_sorted: accept the draft with its exact
// probability under the final set, else the residual walk.
__device__ inline bool spec_select(const float* logit, const int32_t* id,
                                   int n, float temperature, int top_k,
                                   float min_p, float top_p, int32_t draft,
                                   float* exps, uint64_t seed,
                                   uint64_t* counter, int32_t* token,
                                   float* logprob, const TopReport* top = nullptr) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, n, temperature, top_k, min_p, top_p, exps,
                 &final_count, &final_den, &lse);
  if (top != nullptr)
    report_top(logit, id, final_count, temperature, lse, top->N, top->ids,
               top->lps, top->count);
  int j = final_count;
  for (int i = 0; i < final_count; ++i)
    if (id[i] == draft) {
      j = i;
      break;
    }
  const float p_draft = j < final_count ? __fdiv_rn(exps[j], final_den) : 0.0f;
  const double u1 = uniform01(seed, *counter);
  *counter += 1;
  if (static_cast<double>(p_draft) > u1) {
    *token = draft;
    *logprob = __fsub_rn(__fdiv_rn(logit[j], temperature), lse);
    return true;
  }
  const double u2 = uniform01(seed, *counter);
  *counter += 1;
  const float res_den =
      j < final_count ? __fsub_rn(final_den, exps[j]) : final_den;
  double cum = 0.0;
  int chosen = final_count;
  int last = final_count;
  for (int i = 0; i < final_count; ++i) {
    if (i == j) continue;
    last = i;
    cum += static_cast<double>(__fdiv_rn(exps[i], res_den));
    if (cum > u2) {
      chosen = i;
      break;
    }
  }
  if (chosen == final_count) chosen = last;
  *token = id[chosen];
  *logprob = __fsub_rn(__fdiv_rn(logit[chosen], temperature), lse);
  return false;
}

// glm_sample::resolve_support over the merged prefix [0, held).
struct Support {
  int kind;  // 0 fallback, 1 materialized, 2 pure
  int n;
  int top_k;
  float min_p;
  float top_p;
  double covered;
};

__device__ inline Support resolve_support(const float* logit, int held,
                                          int vocab_size, double Z,
                                          const GlmSampleSpec& s) {
  Support out{0, 0, s.top_k, s.min_p, s.top_p, 0.0};
  const bool complete = held == vocab_size;
  const float T = s.temperature;
  for (int i = 0; i < held; ++i)
    out.covered += detmath::exp_d(static_cast<double>(__fdiv_rn(logit[i], T)) - Z);
  if (s.top_k > 0) {
    const int required = min(s.top_k, vocab_size);
    if (held < required) return out;
    out.kind = 1;
    out.n = required;
    return out;
  }
  if (s.min_p > 0.0f) {
    const float threshold =
        __fadd_rn(__fdiv_rn(logit[0], T), detmath::log_f(s.min_p));
    int survivors = held;
    bool bounded = complete;
    for (int i = 0; i < held; ++i) {
      if (__fdiv_rn(logit[i], T) < threshold) {
        survivors = i;
        bounded = true;
        break;
      }
    }
    if (!bounded) return out;
    out.kind = 1;
    out.n = survivors;
    out.min_p = 0.0f;
    return out;
  }
  if (s.top_p < 1.0f) {
    const double top_p = static_cast<double>(s.top_p);
    double cumulative = 0.0;
    int nucleus = 0;
    for (int i = 0; i < held; ++i) {
      cumulative += detmath::exp_d(static_cast<double>(__fdiv_rn(logit[i], T)) - Z);
      if (cumulative >= top_p) {
        nucleus = i + 1;
        break;
      }
    }
    if (nucleus == 0) {
      if (!complete) return out;
      nucleus = held;
    }
    out.kind = 1;
    out.n = nucleus;
    out.top_p = 1.0f;
    return out;
  }
  out.kind = 2;
  out.n = held;
  return out;
}

// glm_sample::sample_from_prefix.
__device__ inline Decision decide_prefix(const float* logit, const int32_t* id,
                                         int held, int vocab_size, double Z,
                                         const GlmSampleSpec& s, float* exps,
                                         uint64_t* counter,
                                         const TopReport* top = nullptr) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup = resolve_support(logit, held, vocab_size, Z, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    selector(logit, id, sup.n, T, sup.top_k, sup.min_p, sup.top_p, exps,
             s.seed, counter, &d.token, &d.logprob, top);
    d.resolved = true;
    return d;
  }
  // Pure temperature sampling: the fp64 walk over the fold masses.
  const bool complete = held == vocab_size;
  if (top != nullptr && !complete && top->N > held) return d;  // the host's rule
  const double draw = uniform01(s.seed, *counter);
  double cumulative = 0.0;
  int chosen = held;
  for (int i = 0; i < held; ++i) {
    cumulative += detmath::exp_d(static_cast<double>(__fdiv_rn(logit[i], T)) - Z);
    if (cumulative > draw) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) return d;
    chosen = held - 1;
  }
  *counter += 1;
  d.token = id[chosen];
  d.logprob = __fsub_rn(__fdiv_rn(logit[chosen], T), static_cast<float>(Z));
  if (top != nullptr)
    report_top(logit, id, held, T, static_cast<float>(Z), top->N, top->ids,
               top->lps, top->count);
  d.resolved = true;
  return d;
}

// glm_sample::spec_accept_from_prefix.
__device__ inline Decision spec_decide_prefix(const float* logit,
                                              const int32_t* id, int held,
                                              int vocab_size, double Z,
                                              int32_t draft,
                                              const GlmSampleSpec& s,
                                              float* exps, uint64_t* counter,
                                              const TopReport* top = nullptr) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup = resolve_support(logit, held, vocab_size, Z, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    d.accepted = spec_select(logit, id, sup.n, T, sup.top_k, sup.min_p,
                             sup.top_p, draft, exps, s.seed, counter,
                             &d.token, &d.logprob, top);
    d.resolved = true;
    return d;
  }
  const bool complete = held == vocab_size;
  int j = held;
  for (int i = 0; i < held; ++i)
    if (id[i] == draft) {
      j = i;
      break;
    }
  if (j == held && !complete) return d;
  const double p_draft =
      j < held ? detmath::exp_d(static_cast<double>(__fdiv_rn(logit[j], T)) - Z)
               : 0.0;
  const uint64_t entry = *counter;
  const double u1 = uniform01(s.seed, *counter);
  *counter += 1;
  const float lse = static_cast<float>(Z);
  if (p_draft > u1) {
    d.resolved = true;
    d.accepted = true;
    d.token = draft;
    d.logprob = __fsub_rn(__fdiv_rn(logit[j], T), lse);
    return d;
  }
  const double u2 = uniform01(s.seed, *counter);
  *counter += 1;
  const double threshold = u2 * (1.0 - p_draft);
  double cumulative = 0.0;
  int chosen = held;
  int last = held;
  for (int i = 0; i < held; ++i) {
    if (i == j) continue;
    last = i;
    cumulative += detmath::exp_d(static_cast<double>(__fdiv_rn(logit[i], T)) - Z);
    if (cumulative > threshold) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) {
      *counter = entry;
      return d;
    }
    chosen = last;
  }
  d.resolved = true;
  d.token = id[chosen];
  d.logprob = __fsub_rn(__fdiv_rn(logit[chosen], T), lse);
  return d;
}

// Decodes one row's groups into shared memory (cooperative) and returns,
// through thread 0's merge, the canonical prefix.
__device__ inline void decode_row(const uint16_t* row_base, int world,
                                  int candidates, int vocab_size,
                                  float* c_logit, int32_t* c_id,
                                  double* lses) {
  const size_t group = glm_sample_rank_group_slots(candidates);
  for (int i = threadIdx.x; i < world * candidates; i += kVerdictThreads) {
    const int r = i / candidates;
    const int j = i % candidates;
    const uint16_t* slot = row_base + static_cast<size_t>(r) * group +
                           static_cast<size_t>(j) * kPickSlotsPerRank;
    const uint32_t id = static_cast<uint32_t>(
        decode_digits(slot + kPickLogitDigits, kPickIdDigits));
    if (id >= static_cast<uint32_t>(vocab_size)) {
      c_id[i] = kNoId;
      c_logit[i] = 0.0f;
    } else {
      c_id[i] = static_cast<int32_t>(id);
      c_logit[i] = __uint_as_float(
          static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
    }
  }
  for (int r = threadIdx.x; r < world; r += kVerdictThreads) {
    const uint64_t bits = decode_digits(
        row_base + static_cast<size_t>(r) * group +
            static_cast<size_t>(candidates) * kPickSlotsPerRank,
        kSampleLseDigits);
    lses[r] = detmath::double_of(bits);
  }
}

// The k-way merge in canonical order (glm_sample::merge_topk): every rank's
// list is already canonical, so the union's sorted prefix is the merge.
__device__ inline int merge_row(const float* c_logit, const int32_t* c_id,
                                int world, int candidates, float* m_logit,
                                int32_t* m_id) {
  int cursor[kPickMaxWorld];
  for (int r = 0; r < world; ++r) cursor[r] = 0;
  int held = 0;
  while (held < candidates) {
    int best_rank = -1;
    float best_logit = 0.0f;
    int32_t best_id = kNoId;
    for (int r = 0; r < world; ++r) {
      if (cursor[r] >= candidates) continue;
      const int i = r * candidates + cursor[r];
      if (c_id[i] == kNoId) continue;
      const float l = c_logit[i];
      const int32_t id = c_id[i];
      const bool before = best_rank < 0 || l > best_logit ||
                          (l == best_logit && id < best_id);
      if (before) {
        best_rank = r;
        best_logit = l;
        best_id = id;
      }
    }
    if (best_rank < 0) break;
    m_logit[held] = best_logit;
    m_id[held] = best_id;
    ++held;
    ++cursor[best_rank];
  }
  return held;
}

// glm_sample::merge_logsumexp over the slices in rank order.
__device__ inline double fold_lse(const double* lses, int world) {
  double top = lses[0];
  for (int r = 1; r < world; ++r) top = lses[r] > top ? lses[r] : top;
  double sum = 0.0;
  for (int r = 0; r < world; ++r) sum += detmath::exp_d(lses[r] - top);
  return top + detmath::log_d(sum);
}

__global__ void __launch_bounds__(kVerdictThreads) sample_verdict_kernel(
    const uint16_t* __restrict__ table, int rows, int world, int candidates,
    int vocab_size, GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ fed, const int64_t* __restrict__ positions,
    int position_stride, int32_t* __restrict__ counts,
    GlmPickVerdict* __restrict__ verdicts,
    GlmPickVerdict* __restrict__ device_verdicts,
    GlmSampleOutcome* __restrict__ outcomes) {
  __shared__ float c_logit[2][kPickMaxWorld * kSampleMaxCandidates];
  __shared__ int32_t c_id[2][kPickMaxWorld * kSampleMaxCandidates];
  __shared__ double lses[2][kPickMaxWorld];
  __shared__ float m_logit[2][kSampleMaxCandidates];
  __shared__ int32_t m_id[2][kSampleMaxCandidates];
  __shared__ float exps[kSampleMaxCandidates];

  const int q = blockIdx.x;
  const int row0 = q * rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const bool active =
      positions == nullptr || positions[q * position_stride] >= 0;
  for (int t = 0; t < rows_per_request; ++t)
    decode_row(table + static_cast<size_t>(row0 + t) * world * group, world,
               candidates, vocab_size, c_logit[t], c_id[t], lses[t]);
  __syncthreads();
  if (tid != 0) return;

  GlmPickVerdict v;
  GlmSampleOutcome o;
  if (!active) {
    v.rows = 0;
    v.accepted = 0;
    v.next = kNoId;
    verdicts[q] = v;
    if (device_verdicts != nullptr) device_verdicts[q] = v;
    outcomes[q] = o;
    return;
  }
  const GlmSampleSpec spec = specs[q];
  const bool penalized = spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  int held[2] = {0, 0};
  for (int t = 0; t < rows_per_request; ++t)
    held[t] = merge_row(c_logit[t], c_id[t], world, candidates, m_logit[t], m_id[t]);
  const TopReport top0{spec.logprobs, o.top_ids[0], o.top_logprobs[0], &o.top_count[0]};
  const TopReport top1{spec.logprobs, o.top_ids[1], o.top_logprobs[1], &o.top_count[1]};
  const TopReport* rep0 = spec.logprobs >= 0 ? &top0 : nullptr;
  const TopReport* rep1 = spec.logprobs >= 0 ? &top1 : nullptr;

  v.rows = rows_per_request;
  if (spec.temperature <= 0.0f || held[0] == 0) {
    // The greedy request: merge_greedy per row, the greedy judge. With
    // logprobs (or penalties) the rows came through the full path at
    // temperature 1 and report under the raw normalizer
    // (glm_sample::greedy_from_prefix).
    for (int t = 0; t < rows_per_request; ++t)
      v.winners[t] = held[t] > 0 ? m_id[t][0] : kNoId;
    v.accepted = 1;
    while (v.accepted < rows_per_request &&
           v.winners[v.accepted - 1] == fed[row0 + v.accepted])
      ++v.accepted;
    v.next = v.winners[v.accepted - 1];
    o.sampled = 0;
    o.counter = spec.counter;
    if ((spec.logprobs >= 0 || penalized) && held[0] > 0) {
      const double Z0 = fold_lse(lses[0], world);
      o.normalizer = Z0;
      o.logprob = __fsub_rn(m_logit[0][0], static_cast<float>(Z0));
      if (rep0 != nullptr)
        report_top(m_logit[0], m_id[0], held[0], 1.0f, static_cast<float>(Z0),
                   spec.logprobs, o.top_ids[0], o.top_logprobs[0], &o.top_count[0]);
      if (rows_per_request == 2 && held[1] > 0) {
        const double Z1 = fold_lse(lses[1], world);
        o.normalizer1 = Z1;
        o.logprob1 = __fsub_rn(m_logit[1][0], static_cast<float>(Z1));
        if (rep1 != nullptr)
          report_top(m_logit[1], m_id[1], held[1], 1.0f, static_cast<float>(Z1),
                     spec.logprobs, o.top_ids[1], o.top_logprobs[1], &o.top_count[1]);
      }
      // The draft leaves the count table on a greedy reject too.
      if (rows_per_request == 2 && v.accepted == 1) {
        const int32_t draft = static_cast<int32_t>(fed[row0 + 1]);
        if (draft >= 0 && draft < vocab_size)
          counts[static_cast<size_t>(q) * vocab_size + draft] -= 1;
      }
    }
  } else {
    uint64_t counter = spec.counter;
    o.sampled = 1;
    const double Z0 = fold_lse(lses[0], world);
    o.normalizer = Z0;
    if (rows_per_request == 1) {
      const Decision d = decide_prefix(m_logit[0], m_id[0], held[0], vocab_size,
                                       Z0, spec, exps, &counter, rep0);
      o.covered_mass = d.covered;
      v.accepted = 1;
      if (d.resolved) {
        v.next = d.token;
        o.logprob = d.logprob;
      } else {
        o.fallback = 1;
        o.fallback_row = 0;
        v.next = m_id[0][0];
      }
      v.winners[0] = v.next;
    } else {
      // The T=2 verify: row 0's accept test against the fed draft.
      const int32_t draft = static_cast<int32_t>(fed[row0 + 1]);
      const Decision d0 = spec_decide_prefix(m_logit[0], m_id[0], held[0],
                                             vocab_size, Z0, draft, spec, exps,
                                             &counter, rep0);
      o.covered_mass = d0.covered;
      if (!d0.resolved) {
        // Provisional REJECT: the commit keeps the post-row-0 state, the
        // host decides row 0 (and row 1 if the draft stands) between windows.
        o.fallback = 1;
        o.fallback_row = 0;
        v.accepted = 1;
        v.winners[0] = m_id[0][0];
        v.next = v.winners[0];
      } else if (!d0.accepted) {
        v.accepted = 1;
        v.winners[0] = d0.token;
        v.next = d0.token;
        o.logprob = d0.logprob;
        // The draft leaves the context it was counted into for row 1.
        if (draft >= 0 && draft < vocab_size)
          counts[static_cast<size_t>(q) * vocab_size + draft] -= 1;
      } else {
        o.accepted_draft = 1;
        v.accepted = 2;
        v.winners[0] = draft;
        o.logprob = d0.logprob;
        const double Z1 = held[1] > 0 ? fold_lse(lses[1], world) : 0.0;
        o.normalizer1 = Z1;
        const Decision d1 =
            held[1] > 0 ? decide_prefix(m_logit[1], m_id[1], held[1],
                                        vocab_size, Z1, spec, exps, &counter,
                                        rep1)
                        : Decision{false, false, kNoId, 0.0f, 0.0};
        if (d1.resolved) {
          v.winners[1] = d1.token;
          o.logprob1 = d1.logprob;
        } else {
          o.fallback = 1;
          o.fallback_row = 1;
          v.winners[1] = held[1] > 0 ? m_id[1][0] : kNoId;
        }
        v.next = v.winners[1];
      }
    }
    o.counter = counter;
    specs[q].counter = counter;
  }
  verdicts[q] = v;
  if (device_verdicts != nullptr) device_verdicts[q] = v;
  outcomes[q] = o;
}

// ---------------------------------------------------------------------------
// Kernel 2b: the digest chain over every request (glm_pick_verdict's), the
// digest group's agreement, and the carry.
// ---------------------------------------------------------------------------
__global__ void sample_digest_kernel(const uint16_t* __restrict__ digests,
                                     int world, int rank, int requests,
                                     int rows_per_request,
                                     GlmPickVerdict* __restrict__ verdicts,
                                     GlmPickVerdict* __restrict__ device_verdicts,
                                     uint64_t* __restrict__ carry_digest) {
  GlmPickVerdict* src = device_verdicts != nullptr ? device_verdicts : verdicts;
  uint64_t digest = requests == 1 ? 0 : splitmix64(requests);
  for (int q = 0; q < requests; ++q) {
    const uint64_t one = verdict_digest(src[q].rows, src[q].accepted, src[q].winners);
    if (requests == 1)
      digest = one;
    else
      digest = splitmix64(digest ^ one);
  }
  digest &= (1ull << kPickDigestBits) - 1;
  const uint64_t mine =
      decode_digits(digests + rank * kPickSlotsPerRank, kPickSlotsPerRank);
  uint32_t mismatch = 0;
  uint64_t peers[kPickMaxWorld] = {};
  for (int k = 0; k < world; ++k) {
    const uint64_t d =
        decode_digits(digests + k * kPickSlotsPerRank, kPickSlotsPerRank);
    peers[k] = d;
    if (d != mine) mismatch |= 1u << k;
  }
  for (int q = 0; q < requests; ++q) {
    verdicts[q].digest = digest;
    verdicts[q].digest_mismatch = mismatch;
    for (int k = 0; k < world; ++k) verdicts[q].peer_digests[k] = peers[k];
    if (device_verdicts != nullptr) {
      device_verdicts[q].digest = digest;
      device_verdicts[q].digest_mismatch = mismatch;
      for (int k = 0; k < world; ++k)
        device_verdicts[q].peer_digests[k] = peers[k];
    }
  }
  *carry_digest = digest;
  (void)rows_per_request;
}

__global__ void count_tokens_kernel(int32_t* __restrict__ counts,
                                    const int64_t* __restrict__ ids, int n,
                                    int vocab_size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t t = ids[i];
  if (t >= 0 && t < vocab_size) atomicAdd(counts + t, 1);
}

void check_common(int rows, int world, int rank, int candidates,
                  int rows_per_request, const char* what) {
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument(std::string(what) + ": rows must be in [1, " +
                                std::to_string(kPickMaxRows) + "]");
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument(std::string(what) + ": rank/world");
  if (candidates < 1 || candidates > kSampleMaxCandidates)
    throw std::invalid_argument(std::string(what) + ": candidates must be in [1, " +
                                std::to_string(kSampleMaxCandidates) + "]");
  if (rows_per_request != 1 && rows_per_request != 2)
    throw std::invalid_argument(std::string(what) +
                                ": the device sampler decides T=1 rows or the "
                                "MTP T=2 verify");
}

__global__ void adjust_count_kernel(int32_t* __restrict__ counts, int64_t token,
                                    int delta, int vocab_size) {
  if (threadIdx.x == 0 && token >= 0 && token < vocab_size)
    counts[token] += delta;
}

}  // namespace

void glm_sample_local(float* logits, int rows, int vocab_count,
                      int vocab_begin, int vocab_size, int rank, int world,
                      int candidates, const GlmSampleSpec* specs,
                      int rows_per_request, const int64_t* fed,
                      const int64_t* positions, int position_stride,
                      int32_t* counts, const uint64_t* carry_digest,
                      uint16_t* table, GlmPickLocal* locals,
                      cudaStream_t stream) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_local");
  if (logits == nullptr || specs == nullptr || fed == nullptr ||
      counts == nullptr || carry_digest == nullptr || table == nullptr ||
      locals == nullptr)
    throw std::invalid_argument("glm_sample_local: null argument");
  if (vocab_count < 1 || vocab_begin < 0 || vocab_size < 1 ||
      vocab_begin + vocab_count > vocab_size ||
      vocab_size > (1 << (6 * kPickIdDigits)) || vocab_size > (1 << kKeyIdxBits))
    throw std::invalid_argument(
        "glm_sample_local: vocab slice outside the id encodings");
  if (vocab_count > kSampleLseChunk * kLocalThreads)
    throw std::invalid_argument(
        "glm_sample_local: the slice has more normalizer chunks than threads");
  if (positions != nullptr && position_stride < 1)
    throw std::invalid_argument("glm_sample_local: position stride");
  if (rows % rows_per_request != 0)
    throw std::invalid_argument("glm_sample_local: rows per request");
  sample_local_kernel<<<rows / rows_per_request, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, vocab_size, rank, world,
      candidates, specs, rows_per_request, fed, positions, position_stride,
      counts, carry_digest, table, locals);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_sample_verdict(const uint16_t* table, int rows, int world, int rank,
                        int candidates, int vocab_size, GlmSampleSpec* specs,
                        int requests, int rows_per_request, const int64_t* fed,
                        const int64_t* positions, int position_stride,
                        int32_t* counts, GlmPickVerdict* verdicts,
                        GlmPickVerdict* device_verdicts,
                        GlmSampleOutcome* outcomes, uint64_t* carry_digest,
                        cudaStream_t stream) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_verdict");
  if (requests < 1 || requests > kPickMaxRequests ||
      requests * rows_per_request != rows)
    throw std::invalid_argument("glm_sample_verdict: request shape");
  if (table == nullptr || specs == nullptr || verdicts == nullptr ||
      outcomes == nullptr || carry_digest == nullptr || fed == nullptr ||
      counts == nullptr)
    throw std::invalid_argument("glm_sample_verdict: null argument");
  if (positions != nullptr && position_stride < rows_per_request)
    throw std::invalid_argument("glm_sample_verdict: position stride");
  sample_verdict_kernel<<<requests, kVerdictThreads, 0, stream>>>(
      table, rows, world, candidates, vocab_size, specs, rows_per_request,
      fed, positions, position_stride, counts, verdicts, device_verdicts,
      outcomes);
  DGPP_CUDA_OK(cudaGetLastError());
  const uint16_t* digests =
      table + static_cast<size_t>(rows) * world *
                  glm_sample_rank_group_slots(candidates);
  sample_digest_kernel<<<1, 1, 0, stream>>>(digests, world, rank, requests,
                                            rows_per_request, verdicts,
                                            device_verdicts, carry_digest);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_sample_adjust_count(int32_t* counts_row, int64_t token, int delta,
                             int vocab_size, cudaStream_t stream) {
  if (counts_row == nullptr || vocab_size < 1)
    throw std::invalid_argument("glm_sample_adjust_count: arguments");
  adjust_count_kernel<<<1, 32, 0, stream>>>(counts_row, token, delta,
                                            vocab_size);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_sample_count_tokens(int32_t* counts_row, const int64_t* ids, int n,
                             int vocab_size, cudaStream_t stream) {
  if (counts_row == nullptr || (n > 0 && ids == nullptr) || vocab_size < 1)
    throw std::invalid_argument("glm_sample_count_tokens: arguments");
  if (n <= 0) return;
  count_tokens_kernel<<<(n + 255) / 256, 256, 0, stream>>>(counts_row, ids, n,
                                                           vocab_size);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
