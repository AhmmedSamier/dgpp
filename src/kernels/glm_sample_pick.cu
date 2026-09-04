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

  const int row = blockIdx.x;
  const int q = row / rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const size_t row_slots = static_cast<size_t>(world) * group;
  uint16_t* row_base = table + static_cast<size_t>(row) * row_slots;
  uint16_t* mine = row_base + static_cast<size_t>(rank) * group;

  // Zero this row's candidate region (every rank's group: the fold needs
  // zeros in the foreign slots); block 0 also owns the digest group and the
  // even-count pad.
  for (size_t i = tid; i < row_slots; i += kLocalThreads) row_base[i] = 0;
  if (row == 0) {
    const size_t total = glm_sample_table_elems(rows, world, candidates);
    const size_t begin = static_cast<size_t>(rows) * row_slots;
    for (size_t i = begin + tid; i < total; i += kLocalThreads) table[i] = 0;
  }
  __syncthreads();

  const bool active =
      positions == nullptr || positions[q * position_stride] >= 0;
  const GlmSampleSpec spec = specs[q];
  const bool sampled = active && spec.temperature > 0.0f;
  float* slice = logits + static_cast<size_t>(row) * vocab_count;

  if (!active) {
    if (tid == 0) {
      locals[row].best_id = kNoId;
      locals[row].best_logit = -INFINITY;
      locals[row].second_logit = -INFINITY;
      if (row == 0)
        encode_digits(table + static_cast<size_t>(rows) * row_slots +
                          static_cast<size_t>(rank) * kPickSlotsPerRank,
                      *carry_digest, kPickSlotsPerRank);
    }
    return;
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
      if (row == 0)
        encode_digits(table + static_cast<size_t>(rows) * row_slots +
                          static_cast<size_t>(rank) * kPickSlotsPerRank,
                      *carry_digest, kPickSlotsPerRank);
    }
    return;
  }

  // ---- the sampled row ----------------------------------------------------
  // 1. The fed token joins the request's context, then the penalties apply
  //    in place (glm_sample::apply_penalties over this slice).
  int32_t* my_counts = counts + static_cast<size_t>(q) * vocab_size;
  if (tid == 0) {
    const int64_t t = fed[row];
    if (t >= 0 && t < vocab_size) my_counts[t] += 1;
  }
  __syncthreads();
  for (int i = tid; i < vocab_count; i += kLocalThreads) {
    const int32_t c = my_counts[vocab_begin + i];
    if (c != 0) slice[i] = penalize(slice[i], c, spec);
  }
  __syncthreads();

  // 2. The slice's temperature-scaled log-sum-exp in the host's chunked
  //    order: the max, then per-chunk sequential fp64 sums, then the chunk
  //    partials folded in order by one thread.
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
    if (row == 0)
      encode_digits(table + static_cast<size_t>(rows) * row_slots +
                        static_cast<size_t>(rank) * kPickSlotsPerRank,
                    *carry_digest, kPickSlotsPerRank);
  }
}

// ---------------------------------------------------------------------------
// Kernel 2a: the decision, one block per request (thread 0 decides; the
// block decodes the table cooperatively).
// ---------------------------------------------------------------------------
struct Decision {
  bool resolved;
  bool drew;
  int32_t token;
  float logprob;
  double covered;
};

// glm_sample::select_from_sorted's stochastic branch over prefix [0, n) of
// the merged candidates (logit[], id[] in shared memory), the request's
// spec fields overridden by the caller as the host does. exps[] is scratch.
__device__ inline void selector(const float* logit, const int32_t* id, int n,
                                float temperature, int top_k, float min_p,
                                float top_p, float* exps, uint64_t seed,
                                uint64_t* counter, int32_t* token,
                                float* logprob) {
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
  int final_count = survivors;
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
    final_count = cut;
  }
  float final_den = 0.0f;
  for (int i = 0; i < final_count; ++i) final_den = __fadd_rn(final_den, exps[i]);
  const float lse = __fadd_rn(scaled0, detmath::log_f(final_den));
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

// glm_sample::sample_from_prefix over the merged prefix [0, held).
__device__ inline Decision decide_prefix(const float* logit, const int32_t* id,
                                         int held, int vocab_size, double Z,
                                         const GlmSampleSpec& s, float* exps,
                                         uint64_t* counter) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const bool complete = held == vocab_size;
  const float T = s.temperature;
  for (int i = 0; i < held; ++i)
    d.covered += detmath::exp_d(static_cast<double>(__fdiv_rn(logit[i], T)) - Z);

  if (s.top_k > 0) {
    const int required = min(s.top_k, vocab_size);
    if (held < required) return d;
    selector(logit, id, required, T, s.top_k, s.min_p, s.top_p, exps, s.seed,
             counter, &d.token, &d.logprob);
    d.resolved = d.drew = true;
    return d;
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
    if (!bounded) return d;
    selector(logit, id, survivors, T, 0, 0.0f, s.top_p, exps, s.seed, counter,
             &d.token, &d.logprob);
    d.resolved = d.drew = true;
    return d;
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
      if (!complete) return d;
      nucleus = held;
    }
    selector(logit, id, nucleus, T, 0, 0.0f, 1.0f, exps, s.seed, counter,
             &d.token, &d.logprob);
    d.resolved = d.drew = true;
    return d;
  }
  // Pure temperature sampling: the fp64 walk over the fold masses.
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
  d.resolved = d.drew = true;
  return d;
}

__global__ void __launch_bounds__(kVerdictThreads) sample_verdict_kernel(
    const uint16_t* __restrict__ table, int rows, int world, int candidates,
    int vocab_size, GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ positions, int position_stride,
    GlmPickVerdict* __restrict__ verdicts,
    GlmPickVerdict* __restrict__ device_verdicts,
    GlmSampleOutcome* __restrict__ outcomes) {
  __shared__ float c_logit[kPickMaxWorld * kSampleMaxCandidates];
  __shared__ int32_t c_id[kPickMaxWorld * kSampleMaxCandidates];
  __shared__ double lses[kPickMaxWorld];
  __shared__ float m_logit[kSampleMaxCandidates];
  __shared__ int32_t m_id[kSampleMaxCandidates];
  __shared__ float exps[kSampleMaxCandidates];

  const int q = blockIdx.x;
  const int row = q * rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const uint16_t* row_base = table + static_cast<size_t>(row) * world * group;
  const bool active =
      positions == nullptr || positions[q * position_stride] >= 0;

  for (int i = tid; i < world * candidates; i += kVerdictThreads) {
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
  for (int r = tid; r < world; r += kVerdictThreads) {
    const uint64_t bits = decode_digits(
        row_base + static_cast<size_t>(r) * group +
            static_cast<size_t>(candidates) * kPickSlotsPerRank,
        kSampleLseDigits);
    lses[r] = detmath::double_of(bits);
  }
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

  // The k-way merge in canonical order (glm_sample::merge_topk): every rank's
  // list is already canonical, so the union's sorted prefix is the merge.
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

  v.rows = 1;
  v.accepted = 1;
  if (spec.temperature <= 0.0f || held == 0) {
    // The greedy row: merge_greedy over each rank's argmax.
    v.next = held > 0 ? m_id[0] : kNoId;
    v.winners[0] = v.next;
    o.sampled = 0;
    o.counter = spec.counter;
  } else {
    // glm_sample::merge_logsumexp over the slices in rank order.
    double top = lses[0];
    for (int r = 1; r < world; ++r) top = lses[r] > top ? lses[r] : top;
    double sum = 0.0;
    for (int r = 0; r < world; ++r) sum += detmath::exp_d(lses[r] - top);
    const double Z = top + detmath::log_d(sum);
    uint64_t counter = spec.counter;
    const Decision d = decide_prefix(m_logit, m_id, held, vocab_size, Z, spec,
                                     exps, &counter);
    o.sampled = 1;
    o.normalizer = Z;
    o.covered_mass = d.covered;
    o.counter = counter;
    if (d.resolved) {
      v.next = d.token;
      o.logprob = d.logprob;
    } else {
      // The fallback: the host decides with the same counter; the token feed
      // carries the provisional argmax until it does.
      o.fallback = 1;
      v.next = m_id[0];
    }
    v.winners[0] = v.next;
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
  if (rows_per_request != 1)
    throw std::invalid_argument(std::string(what) +
                                ": the device sampler decides T=1 rows only "
                                "(sampling under MTP is the next slice)");
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
  sample_local_kernel<<<rows, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, vocab_size, rank, world,
      candidates, specs, rows_per_request, fed, positions, position_stride,
      counts, carry_digest, table, locals);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_sample_verdict(const uint16_t* table, int rows, int world, int rank,
                        int candidates, int vocab_size, GlmSampleSpec* specs,
                        int requests, int rows_per_request,
                        const int64_t* positions, int position_stride,
                        GlmPickVerdict* verdicts,
                        GlmPickVerdict* device_verdicts,
                        GlmSampleOutcome* outcomes, uint64_t* carry_digest,
                        cudaStream_t stream) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_verdict");
  if (requests < 1 || requests > kPickMaxRequests ||
      requests * rows_per_request != rows)
    throw std::invalid_argument("glm_sample_verdict: request shape");
  if (table == nullptr || specs == nullptr || verdicts == nullptr ||
      outcomes == nullptr || carry_digest == nullptr)
    throw std::invalid_argument("glm_sample_verdict: null argument");
  if (positions != nullptr && position_stride < rows_per_request)
    throw std::invalid_argument("glm_sample_verdict: position stride");
  sample_verdict_kernel<<<requests, kVerdictThreads, 0, stream>>>(
      table, rows, world, candidates, vocab_size, specs, rows_per_request,
      positions, position_stride, verdicts, device_verdicts, outcomes);
  DGPP_CUDA_OK(cudaGetLastError());
  const uint16_t* digests =
      table + static_cast<size_t>(rows) * world *
                  glm_sample_rank_group_slots(candidates);
  sample_digest_kernel<<<1, 1, 0, stream>>>(digests, world, rank, requests,
                                            rows_per_request, verdicts,
                                            device_verdicts, carry_digest);
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
