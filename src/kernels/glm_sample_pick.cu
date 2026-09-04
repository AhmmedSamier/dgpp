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
//
// Cost shape (GB10, 38,720-id slices, k = 128, the kernel bench). The
// deterministic fp64 exp is a 13-term polynomial and this part runs fp64
// at 1/64 rate: a slice's 38,720 normalizer terms are ~250 µs of one SM's
// time however a single block spreads them, so they run on a chunk grid
// (one block per 256 ids, every SM busy: ~16 µs for the batch). A
// dependent fp64 add is ~90 ns, so the normalizer's summation order is a
// fixed pairwise tree (glm_sample::chunked_exp_sum) rather than the
// sequential chunk sum that cost ~30 µs per slice. The local top-k reads
// the slice twice: each thread's largest key, whose k-th largest over the
// block is a lower bound only ~k(1 + a few percent) ids reach, then those
// ids into shared memory to sort (or, tie-heavy, to radix-select) — in
// place of the streaming 2,048-key bitonic select, which cost ~680 µs per
// slice. The verdict decodes, merges (by rank counting) and evaluates
// every candidate's fp64 mass in parallel; only the decision's ordered
// walks (one fp64 prefix chain per row) stay on one thread. Measured:
// the three local launches 2 + 16 + 26 µs over two rows (was ~900 µs per
// row, rows serial within a request), the verdict 25/33 µs at T=1/T=2
// (was ~250/290).

namespace dgpp {

namespace {

constexpr int kLocalThreads = 1024;             // the per-row select block
constexpr int kChunkThreads = kSampleLseChunk;  // one id per thread
constexpr int kVerdictThreads = 256;
constexpr int kRadixBins = 256;
constexpr int kKeyIdxBits = 21;       // vocab ids < 2^21
constexpr uint64_t kKeyIdxMask = (1ull << kKeyIdxBits) - 1;
constexpr uint64_t kKeyMax = ~0ull;
constexpr uint32_t kSplitMax = 0xFFFFFFFFu;
constexpr int32_t kNoId = -1;

static_assert(kSampleLseChunk == 256, "the host's glm_sample::kLseChunk");
static_assert(kSampleMaxChunks == kLocalThreads,
              "the row block folds one chunk partial per thread");
static_assert(kSampleMaxCandidates <= kLocalThreads,
              "the row block sorts the survivors with one thread per pair");

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

// The order key of a logit: sortable_f32 of the value with -0 folded onto
// +0, so that the integer order is exactly glm_sample::candidate_before's
// float comparison (which cannot tell the zeros apart and breaks their tie
// by id). Higher key, higher logit.
__device__ inline uint32_t primary_key(float logit) {
  return sortable_f32_dev(logit == 0.0f ? 0.0f : logit);
}
// (~primary << 21) | id: the SMALLEST key is the canonical first candidate.
// A masked id is absent (glm_sample::apply_mask writes -inf): its key is 0,
// below every present key (a finite float never maps to 0), so it is never
// listed, never selected and never counted.
__device__ inline uint32_t key_of(float logit) {
  return logit == -INFINITY ? 0u : primary_key(logit);
}

__device__ inline uint64_t composite_key(float logit, int32_t id) {
  return (static_cast<uint64_t>(~primary_key(logit)) << kKeyIdxBits) |
         static_cast<uint64_t>(id);
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

// The request's spec as a row sees it. The full path (stochastic rows, and
// greedy rows that report logprobs or carry penalties: their normalizer and
// scaled logits are the raw distribution's, temperature 1) is `sampled`.
struct RowSpec {
  bool active;
  bool sampled;
  bool penalized;
  bool constrained;       // the row carries a token mask (M6 6g)
  int32_t allowed;        // the mask's allowed count (the row's vocabulary)
  const uint32_t* mask;   // the mask's bitmask words (after the header)
  GlmSampleSpec spec;
};

// The row's mask, when the host wrote one: masks[row * mask_stride] is the
// allowed count (0: unconstrained), the words after it the bitmask over
// [0, vocab_size). A constrained row takes the full path — its greedy
// argmax must respect the mask too.
__device__ inline RowSpec row_spec(const GlmSampleSpec* specs, int q,
                                   const int64_t* positions,
                                   int position_stride,
                                   const uint32_t* masks, int mask_stride,
                                   int row) {
  RowSpec rs;
  rs.active = positions == nullptr || positions[q * position_stride] >= 0;
  rs.spec = specs[q];
  rs.penalized = rs.spec.repetition_penalty != 1.0f ||
                 rs.spec.frequency_penalty != 0.0f ||
                 rs.spec.presence_penalty != 0.0f;
  rs.mask = nullptr;
  rs.allowed = 0;
  rs.constrained = false;
  if (masks != nullptr) {
    const uint32_t* m = masks + static_cast<size_t>(row) * mask_stride;
    if (m[0] != 0u) {
      rs.constrained = true;
      rs.allowed = static_cast<int32_t>(m[0]);
      rs.mask = m + 1;
    }
  }
  rs.sampled = rs.active && (rs.spec.temperature > 0.0f ||
                             rs.spec.logprobs >= 0 || rs.penalized ||
                             rs.constrained);
  if (rs.spec.temperature <= 0.0f) rs.spec.temperature = 1.0f;
  return rs;
}

__device__ inline bool mask_allows(const uint32_t* mask, int id) {
  return ((mask[id >> 5] >> (id & 31)) & 1u) != 0u;
}

// Row t's context count of id v: the request's table (every token
// committed through the previous step) plus this step's fed tokens up to
// and including row t's — the table itself is untouched until the verdict
// commits (glm_sample::SampledSpeculator's rule: the draft joins the
// context only when it stands).
__device__ inline int32_t context_count(const int32_t* counts_q, int32_t v,
                                        const int64_t* fed, int row0, int t) {
  int32_t c = counts_q[v];
  for (int j = 0; j <= t; ++j)
    if (fed[row0 + j] == static_cast<int64_t>(v)) ++c;
  return c;
}

// ---------------------------------------------------------------------------
// Kernel 1a, one block per 256-id chunk of every row: the penalties in
// place (glm_sample::apply_penalties) and the chunk's max of the
// temperature-scaled slice.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kChunkThreads) sample_prepare_kernel(
    float* __restrict__ logits, int vocab_count, int vocab_begin,
    int vocab_size, const GlmSampleSpec* __restrict__ specs,
    int rows_per_request, const int64_t* __restrict__ fed,
    const int64_t* __restrict__ positions, int position_stride,
    const int32_t* __restrict__ counts, const uint32_t* __restrict__ masks,
    int mask_stride, double* __restrict__ maxes) {
  __shared__ float fred[32];
  const int c = blockIdx.x;
  const int row = blockIdx.y;
  const int q = row / rows_per_request;
  const int t = row % rows_per_request;
  const RowSpec rs =
      row_spec(specs, q, positions, position_stride, masks, mask_stride, row);
  if (!rs.sampled) return;
  const int nchunks = gridDim.x;
  float* slice = logits + static_cast<size_t>(row) * vocab_count;
  const int i = c * kChunkThreads + threadIdx.x;
  float scaled = -INFINITY;
  if (i < vocab_count) {
    float l = slice[i];
    if (rs.penalized) {
      const int32_t cnt =
          context_count(counts + static_cast<size_t>(q) * vocab_size,
                        vocab_begin + i, fed, q * rows_per_request, t);
      if (cnt != 0) {
        l = penalize(l, cnt, rs.spec);
        slice[i] = l;
      }
    }
    // The mask: an excluded id becomes -inf in place (absent to every
    // later stage, the host's gather fallback included).
    if (rs.constrained && !mask_allows(rs.mask, vocab_begin + i)) {
      l = -INFINITY;
      slice[i] = l;
    }
    scaled = __fdiv_rn(l, rs.spec.temperature);
  }
  const float top = block_max_f(scaled, fred);
  if (threadIdx.x == 0)
    maxes[static_cast<size_t>(row) * nchunks + c] = static_cast<double>(top);
}

// ---------------------------------------------------------------------------
// Kernel 1b, the same grid: each chunk's normalizer partial in the host's
// order — the slice max, the chunk's terms exp_d(scaled - max) summed by
// recursive halving (glm_sample::chunked_exp_sum's inner tree).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kChunkThreads) sample_partials_kernel(
    const float* __restrict__ logits, int vocab_count,
    const GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ positions, int position_stride,
    const uint32_t* __restrict__ masks, int mask_stride,
    const double* __restrict__ maxes, double* __restrict__ partials) {
  __shared__ double terms[kChunkThreads];
  __shared__ float fred[32];
  const int c = blockIdx.x;
  const int row = blockIdx.y;
  const int q = row / rows_per_request;
  const RowSpec rs =
      row_spec(specs, q, positions, position_stride, masks, mask_stride, row);
  if (!rs.sampled) return;
  const int nchunks = gridDim.x;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < nchunks; j += kChunkThreads)
    m = fmaxf(m, static_cast<float>(maxes[static_cast<size_t>(row) * nchunks + j]));
  const float top = block_max_f(m, fred);  // exact: any order
  const float* slice = logits + static_cast<size_t>(row) * vocab_count;
  const int c0 = c * kChunkThreads;
  const int i = c0 + threadIdx.x;
  // A slice with every id masked has no mass (top is -inf): zero terms, so
  // the row's lse folds as -inf rather than NaN.
  terms[threadIdx.x] =
      i < vocab_count && top != -INFINITY
          ? detmath::exp_d(
                static_cast<double>(__fdiv_rn(slice[i], rs.spec.temperature)) -
                static_cast<double>(top))
          : 0.0;
  __syncthreads();
  // glm_sample::halving_sum over the chunk: eight dependent adds.
  for (int h = kChunkThreads >> 1; h >= 1; h >>= 1) {
    if (threadIdx.x < h) terms[threadIdx.x] += terms[threadIdx.x + h];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partials[static_cast<size_t>(row) * nchunks + c] = terms[0];
}

// ---------------------------------------------------------------------------
// The block-wide radix select: the threshold key of the `need` largest
// 32-bit keys over [0, n) — four 8-bit passes from the top, a shared
// histogram per pass over the keys still matching the resolved prefix.
// Returns the threshold, how many of the keys EQUAL to it are wanted
// (>= 1) and how many there are; keys above it are all wanted. All threads
// of the block call it. Cheap for the shared-memory key sets it normally
// runs on (one or two keys per thread); the whole-slice form behind
// SliceKeyFn is the fallback for tie-heavy slices.
// ---------------------------------------------------------------------------
constexpr int kLoadBatch = 8;  // independent loads in flight per thread

template <typename KeyFn>
__device__ inline void radix_threshold(KeyFn key_of, int n, int need,
                                       uint32_t* hist, int* found,
                                       uint32_t* threshold, int* need_at,
                                       int* count_at) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint32_t prefix = 0, mask = 0;
  int count = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = tid; b < kRadixBins; b += blockDim.x) hist[b] = 0;
    __syncthreads();
    for (int base = tid; base < n; base += blockDim.x * kLoadBatch) {
      uint32_t key[kLoadBatch];
      bool have[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * blockDim.x;
        have[b] = i < n;
        key[b] = have[b] ? key_of(i) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b)
        if (have[b] && (key[b] & mask) == prefix)
          atomicAdd(&hist[(key[b] >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (warp == 0) {
      // Lane l owns bins 255-8l .. 248-8l; the cumulative count from the
      // top bin down locates the bin holding the need-th largest key.
      int local = 0;
      for (int j = 0; j < 8; ++j) local += static_cast<int>(hist[255 - 8 * lane - j]);
      int incl = local;
      for (int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += v;
      }
      const int excl = incl - local;
      if (excl < need && need <= incl) {
        int cum = excl;
        for (int j = 0; j < 8; ++j) {
          const int b = 255 - 8 * lane - j;
          const int h = static_cast<int>(hist[b]);
          if (cum + h >= need) {
            found[0] = b;
            found[1] = need - cum;
            found[2] = h;
            break;
          }
          cum += h;
        }
      }
    }
    __syncthreads();
    const int bin = found[0];
    need = found[1];
    count = found[2];
    prefix |= static_cast<uint32_t>(bin) << shift;
    mask |= 255u << shift;
    __syncthreads();
  }
  *threshold = prefix;
  *need_at = need;
  *count_at = count;
}

// Key sets: a shared-memory list of keys (with the ids' local indices for
// the tie pass), and the whole slice. Among keys equal to the value
// threshold the LOWER index wins: the tie pass selects on ~index, every
// other key 0 (below every tie).
struct ListKeyFn {
  const uint32_t* keys;
  __device__ uint32_t operator()(int i) const { return keys[i]; }
};
struct ListTieKeyFn {
  const uint32_t* keys;
  const int32_t* idx;
  uint32_t threshold;
  __device__ uint32_t operator()(int i) const {
    return keys[i] == threshold ? ~static_cast<uint32_t>(idx[i]) : 0u;
  }
};
struct SliceKeyFn {
  const float* slice;
  __device__ uint32_t operator()(int i) const { return key_of(slice[i]); }
};
struct SliceTieKeyFn {
  const float* slice;
  uint32_t threshold;
  __device__ uint32_t operator()(int i) const {
    return key_of(slice[i]) == threshold ? ~static_cast<uint32_t>(i) : 0u;
  }
};

// bitonic_sort_asc for the survivors (n <= kSampleMaxCandidates) on the
// block's first kSortThreads threads only, synchronized by a named barrier:
// a 1024-thread __syncthreads per stage cost more than the stage.
constexpr int kSortThreads = 128;
static_assert(kSampleMaxCandidates <= 2 * kSortThreads, "two pairs per thread");
__device__ inline void sort_survivors(uint32_t* hi, uint32_t* lo, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += kSortThreads) {
        const int j = i ^ stride;
        if (j > i) cx_split(hi, lo, i, j, (i & size) == 0);
      }
      asm volatile("bar.sync 1, %0;" ::"r"(kSortThreads) : "memory");
    }
  }
}

// The ids at or above the per-thread-maxima bound that the row block
// keeps in shared memory before the exact select; more than this (a
// tie-heavy slice) falls back to the whole-slice radix passes.
constexpr int kListCap = 2048;

// ---------------------------------------------------------------------------
// Kernel 1c, one block per ROW: the row's wire group — the normalizer
// folded from the chunk partials in chunk order, the exact local top-k in
// canonical order (radix select, then one bitonic sort of the survivors) —
// or the greedy row's argmax; the digest carry.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kLocalThreads) sample_local_kernel(
    const float* __restrict__ logits, int rows, int vocab_count,
    int vocab_begin, int rank, int world, int candidates,
    const GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ positions, int position_stride,
    const uint32_t* __restrict__ masks, int mask_stride,
    const double* __restrict__ maxes, const double* __restrict__ partials,
    int nchunks, const uint64_t* __restrict__ carry_digest,
    uint16_t* __restrict__ table, GlmPickLocal* __restrict__ locals) {
  __shared__ uint32_t hist[kRadixBins];
  __shared__ int found[3];
  __shared__ int sel_count;
  __shared__ int list_count;
  __shared__ int present_count;
  __shared__ uint32_t tmax[kLocalThreads];
  __shared__ uint32_t list_key[kListCap];
  __shared__ int32_t list_idx[kListCap];
  __shared__ uint32_t sel_hi[kSampleMaxCandidates];
  __shared__ uint32_t sel_lo[kSampleMaxCandidates];
  __shared__ double fold[kSampleMaxChunks];
  __shared__ float fred[32];
  __shared__ uint64_t kred[32];

  const int row = blockIdx.x;
  const int q = row / rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const size_t row_slots = static_cast<size_t>(world) * group;
  const RowSpec rs =
      row_spec(specs, q, positions, position_stride, masks, mask_stride, row);
  uint16_t* row_base = table + static_cast<size_t>(row) * row_slots;
  uint16_t* mine = row_base + static_cast<size_t>(rank) * group;
  const float* slice = logits + static_cast<size_t>(row) * vocab_count;

  // Zero this row's candidate region (every rank's group: the fold needs
  // zeros in the foreign slots); block 0 also owns the digest group and
  // the even-count pad.
  for (size_t i = tid; i < row_slots; i += kLocalThreads) row_base[i] = 0;
  if (row == 0) {
    const size_t total = glm_sample_table_elems(rows, world, candidates);
    const size_t begin = static_cast<size_t>(rows) * row_slots;
    for (size_t i = begin + tid; i < total; i += kLocalThreads) table[i] = 0;
  }
  __syncthreads();

  if (!rs.active) {
    if (tid == 0) {
      locals[row].best_id = kNoId;
      locals[row].best_logit = -INFINITY;
      locals[row].second_logit = -INFINITY;
    }
  } else if (!rs.sampled) {
    // The greedy row: the canonical argmax (the smallest composite key),
    // written as the one candidate; every other slot carries the empty id.
    uint64_t best = kKeyMax;
    for (int i = tid; i < vocab_count; i += kLocalThreads) {
      const uint64_t key = composite_key(slice[i], vocab_begin + i);
      best = key < best ? key : best;
    }
    best = block_min_key(best, kred);
    const int32_t best_id = key_id(best);
    const float best_logit = slice[best_id - vocab_begin];
    for (int j = tid; j < candidates; j += kLocalThreads) {
      uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
      if (j == 0) {
        encode_digits(slot, static_cast<uint64_t>(__float_as_uint(best_logit)),
                      kPickLogitDigits);
        encode_digits(slot + kPickLogitDigits, static_cast<uint64_t>(best_id),
                      kPickIdDigits);
      } else {
        encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
      }
    }
    if (tid == 0) {
      locals[row].best_id = best_id;
      locals[row].best_logit = best_logit;
      locals[row].second_logit = -INFINITY;
    }
  } else {
    // ---- the sampled row ------------------------------------------------
    // 1. The slice's temperature-scaled log-sum-exp: the max over the chunk
    //    maxes, the chunk partials (zero-padded to a power of two) summed
    //    by recursive halving, then log (glm_sample::slice_logsumexp).
    const float m = tid < nchunks
                        ? static_cast<float>(maxes[static_cast<size_t>(row) * nchunks + tid])
                        : -INFINITY;
    const float top = block_max_f(m, fred);
    fold[tid] = tid < nchunks ? partials[static_cast<size_t>(row) * nchunks + tid]
                              : 0.0;
    __syncthreads();
    int fold_width = 1;
    while (fold_width < nchunks) fold_width <<= 1;
    for (int h = fold_width >> 1; h >= 1; h >>= 1) {
      if (tid < h) fold[tid] += fold[tid + h];
      __syncthreads();
    }
    if (tid == 0) {
      // A slice with every id masked carries no mass: lse -inf (the fold
      // skips it), never log(0).
      const double lse = top == -INFINITY
                             ? -INFINITY
                             : static_cast<double>(top) + detmath::log_d(fold[0]);
      encode_digits(mine + static_cast<size_t>(candidates) * kPickSlotsPerRank,
                    detmath::bits_of(lse), kSampleLseDigits);
    }

    // 2. The exact local top-k (glm_sample::local_topk) over the PRESENT
    //    ids (a masked id's key is 0 and it is never listed).
    //    a. Each thread's largest key over its strided share of the slice,
    //       and the count of present ids. The k-th largest of the maxima
    //       is a lower bound on the k-th largest key of the slice (the k
    //       largest thread maxima are k distinct ids at or above it) and a
    //       tight one: with ~38 ids per thread, the ids reaching it number
    //       ~k(1 + a few percent). The slice does not fit L1 beside the
    //       block's shared memory, so its loads are issued kLoadBatch at a
    //       time from L2.
    uint32_t my_max = 0;
    int my_present = 0;
    if (tid == 0) present_count = 0;
    for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
      uint32_t key[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        my_max = max(my_max, key[b]);
        my_present += key[b] != 0u ? 1 : 0;
      }
    }
    tmax[tid] = my_max;
    if (tid == 0) list_count = 0;
    __syncthreads();
    if (my_present != 0) atomicAdd(&present_count, my_present);
    __syncthreads();
    const int present = present_count;
    const int k = min(candidates, present);
    uint32_t bound = 0;
    int bound_need = 0, bound_count = 0;
    if (k > 0)
      radix_threshold(ListKeyFn{tmax}, kLocalThreads, k, hist, found, &bound,
                      &bound_need, &bound_count);
    // Fewer present ids than threads hold maxima: the k-th maximum may be
    // 0 (an absent id); the bound is then 1 — every present id, and none
    // of the absent ones.
    bound = max(bound, 1u);
    //    b. The ids at or above the bound, with their keys, in shared
    //       memory.
    for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
      uint32_t key[kLoadBatch];
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
      }
#pragma unroll
      for (int b = 0; b < kLoadBatch; ++b) {
        const int i = base + b * kLocalThreads;
        if (i < vocab_count && key[b] >= bound) {
          const int pos = atomicAdd(&list_count, 1);
          if (pos < kListCap) {
            list_key[pos] = key[b];
            list_idx[pos] = i;
          }
        }
      }
    }
    if (tid == 0) sel_count = 0;
    for (int i = tid; i < kSampleMaxCandidates; i += kLocalThreads) {
      sel_hi[i] = kSplitMax;
      sel_lo[i] = kSplitMax;
    }
    __syncthreads();
    const int listed = list_count;  // >= k by construction
    //    c. The survivors into sel (hi = ~key, lo = id; ascending is the
    //       canonical order): the whole list when it is no wider than the
    //       sort, else the exact threshold over the list — or, past the
    //       list's capacity, over the whole slice — and its compaction.
    int survivors = k;
    if (k == 0) {
      survivors = 0;  // no present id: every slot empty
    } else if (listed <= kSampleMaxCandidates) {
      survivors = listed;
      for (int i = tid; i < listed; i += kLocalThreads) {
        sel_hi[i] = ~list_key[i];
        sel_lo[i] = static_cast<uint32_t>(vocab_begin + list_idx[i]);
      }
    } else if (listed <= kListCap) {
      uint32_t threshold = 0;
      int need = 0, count = 0;
      radix_threshold(ListKeyFn{list_key}, listed, k, hist, found, &threshold,
                      &need, &count);
      const bool every_tie = need == count;
      uint32_t tie_threshold = 0;
      if (!every_tie) {
        int need2 = 0, count2 = 0;
        radix_threshold(ListTieKeyFn{list_key, list_idx, threshold}, listed,
                        need, hist, found, &tie_threshold, &need2, &count2);
      }
      for (int i = tid; i < listed; i += kLocalThreads) {
        const uint32_t u = list_key[i];
        const int32_t idx = list_idx[i];
        const bool take =
            u > threshold ||
            (u == threshold &&
             (every_tie || ~static_cast<uint32_t>(idx) >= tie_threshold));
        if (take) {
          const int pos = atomicAdd(&sel_count, 1);
          if (pos < kSampleMaxCandidates) {
            sel_hi[pos] = ~u;
            sel_lo[pos] = static_cast<uint32_t>(vocab_begin + idx);
          }
        }
      }
    } else {
      uint32_t threshold = 0;
      int need = 0, count = 0;
      radix_threshold(SliceKeyFn{slice}, vocab_count, k, hist, found,
                      &threshold, &need, &count);
      const bool every_tie = need == count;
      uint32_t tie_threshold = 0;
      if (!every_tie) {
        int need2 = 0, count2 = 0;
        radix_threshold(SliceTieKeyFn{slice, threshold}, vocab_count, need,
                        hist, found, &tie_threshold, &need2, &count2);
      }
      for (int base = tid; base < vocab_count; base += kLocalThreads * kLoadBatch) {
        uint32_t key[kLoadBatch];
#pragma unroll
        for (int b = 0; b < kLoadBatch; ++b) {
          const int i = base + b * kLocalThreads;
          key[b] = i < vocab_count ? key_of(slice[i]) : 0u;
        }
#pragma unroll
        for (int b = 0; b < kLoadBatch; ++b) {
          const int i = base + b * kLocalThreads;
          const uint32_t u = key[b];
          const bool take =
              i < vocab_count && u != 0u &&
              (u > threshold ||
               (u == threshold &&
                (every_tie || ~static_cast<uint32_t>(i) >= tie_threshold)));
          if (take) {
            const int pos = atomicAdd(&sel_count, 1);
            if (pos < kSampleMaxCandidates) {
              sel_hi[pos] = ~u;
              sel_lo[pos] = static_cast<uint32_t>(vocab_begin + i);
            }
          }
        }
      }
    }
    __syncthreads();
    //    d. Sorted; the first k are the canonical local top-k.
    int width = 2;
    while (width < survivors) width <<= 1;
    if (tid < kSortThreads && survivors > 0) sort_survivors(sel_hi, sel_lo, width);
    __syncthreads();
    for (int j = tid; j < candidates; j += kLocalThreads) {
      uint16_t* slot = mine + static_cast<size_t>(j) * kPickSlotsPerRank;
      if (j < k) {
        const int32_t id = static_cast<int32_t>(sel_lo[j]);
        const float logit = slice[id - vocab_begin];
        encode_digits(slot, static_cast<uint64_t>(__float_as_uint(logit)),
                      kPickLogitDigits);
        encode_digits(slot + kPickLogitDigits, static_cast<uint64_t>(id),
                      kPickIdDigits);
      } else {
        encode_digits(slot + kPickLogitDigits, kSampleEmptyId, kPickIdDigits);
      }
    }
    if (tid == 0) {
      if (k > 0) {
        const int32_t id0 = static_cast<int32_t>(sel_lo[0]);
        locals[row].best_id = id0;
        locals[row].best_logit = slice[id0 - vocab_begin];
        locals[row].second_logit =
            k > 1 ? slice[static_cast<int32_t>(sel_lo[1]) - vocab_begin]
                  : -INFINITY;
      } else {
        locals[row].best_id = kNoId;
        locals[row].best_logit = -INFINITY;
        locals[row].second_logit = -INFINITY;
      }
    }
  }
  if (row == 0 && tid == 0)
    encode_digits(table + static_cast<size_t>(rows) * row_slots +
                      static_cast<size_t>(rank) * kPickSlotsPerRank,
                  *carry_digest, kPickSlotsPerRank);
}

// ---------------------------------------------------------------------------
// Kernel 2a: the decision, one block per request. The block decodes,
// merges and evaluates the candidates' masses cooperatively; thread 0 then
// makes the decision, whose ordered walks are the host's arithmetic.
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
// candidates. `expsrc[i]` is exp_f(scaled_i - scaled_0), precomputed in
// parallel (the same deterministic value the host computes in its loop);
// exps[] receives the survivors' exps. Returns final_count, final_den, lse
// through the out-params.
__device__ inline void selector_state(const float* logit, const float* expsrc,
                                      int n, float temperature, int top_k,
                                      float min_p, float top_p, float* exps,
                                      int* final_count, float* final_den,
                                      float* lse) {
  const float scaled0 = __fdiv_rn(logit[0], temperature);
  int kept = n;
  if (top_k > 0) kept = min(kept, top_k);
  for (int i = 0; i < kept; ++i) exps[i] = expsrc[i];
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

__device__ inline void selector(const float* logit, const int32_t* id,
                                const float* expsrc, int n, float temperature,
                                int top_k, float min_p, float top_p,
                                float* exps, uint64_t seed, uint64_t* counter,
                                int32_t* token, float* logprob,
                                const TopReport* top = nullptr) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, expsrc, n, temperature, top_k, min_p, top_p, exps,
                 &final_count, &final_den, &lse);
  if (top != nullptr)
    report_top(logit, id, final_count, temperature, lse, top->N, top->ids,
               top->lps, top->count);
  const double r = uniform01(seed, *counter);
  *counter += 1;
  double cum = 0.0;
  int chosen = final_count;
  int last_positive = 0;  // the rounding guard: the last token with mass
  for (int i = 0; i < final_count; ++i) {
    if (exps[i] > 0.0f) last_positive = i;
    cum += static_cast<double>(__fdiv_rn(exps[i], final_den));
    if (cum > r) {
      chosen = i;
      break;
    }
  }
  if (chosen == final_count) chosen = last_positive;
  *token = id[chosen];
  *logprob = __fsub_rn(__fdiv_rn(logit[chosen], temperature), lse);
}

// glm_sample::spec_select_from_sorted: accept the draft with its exact
// probability under the final set, else the residual walk.
__device__ inline bool spec_select(const float* logit, const int32_t* id,
                                   const float* expsrc, int n,
                                   float temperature, int top_k, float min_p,
                                   float top_p, int32_t draft, float* exps,
                                   uint64_t seed, uint64_t* counter,
                                   int32_t* token, float* logprob,
                                   const TopReport* top = nullptr) {
  int final_count = 0;
  float final_den = 0.0f, lse = 0.0f;
  selector_state(logit, expsrc, n, temperature, top_k, min_p, top_p, exps,
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
  int last = final_count;  // the last residual token with mass
  for (int i = 0; i < final_count; ++i) {
    if (i == j) continue;
    if (exps[i] > 0.0f || last == final_count) last = i;
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

// glm_sample::resolve_support over the merged prefix [0, held). `mass[i]`
// is exp_d(scaled_i - Z), precomputed in parallel; `prefix[i]` receives
// the running fp64 sum of the masses through i in the host's order — the
// covered mass, the nucleus crossing and the pure walk all read the same
// chain, computed once (a dependent fp64 add is ~90 ns on GB10).
struct Support {
  int kind;  // 0 fallback, 1 materialized, 2 pure
  int n;
  int top_k;
  float min_p;
  float top_p;
  double covered;
};

__device__ inline Support resolve_support(const float* logit,
                                          const double* mass, double* prefix,
                                          int held, int vocab_size,
                                          const GlmSampleSpec& s) {
  Support out{0, 0, s.top_k, s.min_p, s.top_p, 0.0};
  const bool complete = held == vocab_size;
  const float T = s.temperature;
  {
    double cumulative = 0.0;
    for (int i = 0; i < held; ++i) {
      cumulative += mass[i];
      prefix[i] = cumulative;
    }
    out.covered = cumulative;
  }
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
    int nucleus = 0;
    for (int i = 0; i < held; ++i) {
      if (prefix[i] >= top_p) {
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
                                         const double* mass, double* prefix,
                                         const float* expsrc, int held,
                                         int vocab_size, double Z,
                                         const GlmSampleSpec& s, float* exps,
                                         uint64_t* counter,
                                         const TopReport* top = nullptr) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup =
      resolve_support(logit, mass, prefix, held, vocab_size, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    selector(logit, id, expsrc, sup.n, T, sup.top_k, sup.min_p, sup.top_p,
             exps, s.seed, counter, &d.token, &d.logprob, top);
    d.resolved = true;
    return d;
  }
  // Pure temperature sampling: the fp64 walk over the fold masses.
  const bool complete = held == vocab_size;
  if (top != nullptr && !complete && top->N > held) return d;  // the host's rule
  const double draw = uniform01(s.seed, *counter);
  int chosen = held;
  int last_positive = 0;  // the rounding guard: the last token with mass
  for (int i = 0; i < held; ++i) {
    if (mass[i] > 0.0) last_positive = i;
    if (prefix[i] > draw) {
      chosen = i;
      break;
    }
  }
  if (chosen == held) {
    if (!complete) return d;
    chosen = last_positive;
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
// `draft_excluded`: the draft is a masked id — probability 0 without the
// prefix having to show it (glm_sample::spec_accept_from_prefix).
__device__ inline Decision spec_decide_prefix(
    const float* logit, const int32_t* id, const double* mass, double* prefix,
    const float* expsrc, int held, int vocab_size, double Z, int32_t draft,
    const GlmSampleSpec& s, float* exps, uint64_t* counter,
    const TopReport* top = nullptr, bool draft_excluded = false) {
  Decision d{false, false, kNoId, 0.0f, 0.0};
  const Support sup =
      resolve_support(logit, mass, prefix, held, vocab_size, s);
  d.covered = sup.covered;
  if (sup.kind == 0) return d;
  const float T = s.temperature;
  if (sup.kind == 1) {
    d.accepted = spec_select(logit, id, expsrc, sup.n, T, sup.top_k,
                             sup.min_p, sup.top_p, draft, exps, s.seed,
                             counter, &d.token, &d.logprob, top);
    d.resolved = true;
    return d;
  }
  const bool complete = held == vocab_size;
  int j = held;
  if (!draft_excluded)
    for (int i = 0; i < held; ++i)
      if (id[i] == draft) {
        j = i;
        break;
      }
  if (j == held && !complete && !draft_excluded) return d;
  const double p_draft = j < held ? mass[j] : 0.0;
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
  int last = held;  // the last residual token with mass
  for (int i = 0; i < held; ++i) {
    if (i == j) continue;
    if (mass[i] > 0.0 || last == held) last = i;
    cumulative += mass[i];
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

// glm_sample::merge_logsumexp's pivot: the largest slice lse.
__device__ inline double lse_top(const double* lses, int world) {
  double top = lses[0];
  for (int r = 1; r < world; ++r) top = lses[r] > top ? lses[r] : top;
  return top;
}

// The number of keys below (hi, lo) in an ascending split-key list.
__device__ inline int lower_bound_split(const uint32_t* hi, const uint32_t* lo,
                                        int n, uint32_t khi, uint32_t klo) {
  int l = 0, r = n;
  while (l < r) {
    const int m = (l + r) >> 1;
    if (key_less(hi[m], lo[m], khi, klo))
      l = m + 1;
    else
      r = m;
  }
  return l;
}

__global__ void __launch_bounds__(kVerdictThreads) sample_verdict_kernel(
    const uint16_t* __restrict__ table, int rows, int world, int candidates,
    int vocab_size, GlmSampleSpec* __restrict__ specs, int rows_per_request,
    const int64_t* __restrict__ fed, const int64_t* __restrict__ positions,
    int position_stride, int32_t* __restrict__ counts,
    const uint32_t* __restrict__ masks, int mask_stride,
    GlmPickVerdict* __restrict__ verdicts,
    GlmPickVerdict* __restrict__ device_verdicts,
    GlmSampleOutcome* __restrict__ outcomes) {
  // Every rank's list as split composite keys (hi = ~primary, lo = id;
  // the empty slot is the maximal key), the merged prefix, its masses.
  __shared__ uint32_t c_hi[2][kPickMaxWorld * kSampleMaxCandidates];
  __shared__ uint32_t c_lo[2][kPickMaxWorld * kSampleMaxCandidates];
  __shared__ double lses[2][kPickMaxWorld];
  __shared__ float m_logit[2][kSampleMaxCandidates];
  __shared__ int32_t m_id[2][kSampleMaxCandidates];
  __shared__ double mass[2][kSampleMaxCandidates];
  __shared__ float expsrc[2][kSampleMaxCandidates];
  __shared__ float exps[kSampleMaxCandidates];
  __shared__ double prefix[kSampleMaxCandidates];
  __shared__ double lse_terms[2][kPickMaxWorld];
  __shared__ int total[2];
  __shared__ double Z[2];

  const int q = blockIdx.x;
  const int row0 = q * rows_per_request;
  const int tid = threadIdx.x;
  const size_t group = glm_sample_rank_group_slots(candidates);
  const bool active =
      positions == nullptr || positions[q * position_stride] >= 0;
  if (!active) {
    if (tid == 0) {
      GlmPickVerdict v;
      v.rows = 0;
      v.accepted = 0;
      v.next = kNoId;
      verdicts[q] = v;
      if (device_verdicts != nullptr) device_verdicts[q] = v;
      outcomes[q] = GlmSampleOutcome{};
    }
    return;
  }
  const GlmSampleSpec spec = specs[q];
  const int entries = world * candidates;
  if (tid < 2) total[tid] = 0;
  __syncthreads();
  // A constrained row's vocabulary is its allowed count (the complete list
  // is the allowed set); a masked draft is excluded from row 0 outright.
  int row_vocab[2] = {vocab_size, vocab_size};
  bool constrained[2] = {false, false};
  const uint32_t* mask0 = nullptr;
  for (int t = 0; t < rows_per_request; ++t) {
    if (masks == nullptr) continue;
    const uint32_t* m = masks + static_cast<size_t>(row0 + t) * mask_stride;
    if (m[0] != 0u) {
      constrained[t] = true;
      row_vocab[t] = static_cast<int>(m[0]);
      if (t == 0) mask0 = m + 1;
    }
  }

  // 1. Decode every rank's group of every row (cooperative).
  for (int t = 0; t < rows_per_request; ++t) {
    const uint16_t* row_base =
        table + static_cast<size_t>(row0 + t) * world * group;
    for (int i = tid; i < entries; i += kVerdictThreads) {
      const int r = i / candidates;
      const int j = i % candidates;
      const uint16_t* slot = row_base + static_cast<size_t>(r) * group +
                             static_cast<size_t>(j) * kPickSlotsPerRank;
      const uint32_t id = static_cast<uint32_t>(
          decode_digits(slot + kPickLogitDigits, kPickIdDigits));
      if (id >= static_cast<uint32_t>(vocab_size)) {
        c_hi[t][i] = kSplitMax;
        c_lo[t][i] = kSplitMax;
      } else {
        const float l = __uint_as_float(
            static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
        c_hi[t][i] = ~primary_key(l);
        c_lo[t][i] = id;
        atomicAdd(&total[t], 1);
      }
    }
    for (int r = tid; r < world; r += kVerdictThreads)
      lses[t][r] = detmath::double_of(decode_digits(
          row_base + static_cast<size_t>(r) * group +
              static_cast<size_t>(candidates) * kPickSlotsPerRank,
          kSampleLseDigits));
  }
  __syncthreads();

  // 2. The k-way merge in canonical order (glm_sample::merge_topk): every
  //    rank's list is canonical with its empties last, so a candidate's
  //    rank in the union is its index in its own list plus the count of
  //    smaller keys in every other list; the ids are disjoint across
  //    ranks, so the ranks are a permutation and the first `held` land.
  int held[2] = {0, 0};
  for (int t = 0; t < rows_per_request; ++t) {
    held[t] = min(candidates, total[t]);
    const uint16_t* row_base =
        table + static_cast<size_t>(row0 + t) * world * group;
    for (int i = tid; i < entries; i += kVerdictThreads) {
      const uint32_t khi = c_hi[t][i];
      const uint32_t klo = c_lo[t][i];
      if (khi == kSplitMax && klo == kSplitMax) continue;
      const int r = i / candidates;
      const int j = i % candidates;
      int position = j;
      for (int o = 0; o < world; ++o)
        if (o != r)
          position += lower_bound_split(c_hi[t] + o * candidates,
                                        c_lo[t] + o * candidates, candidates,
                                        khi, klo);
      if (position < held[t]) {
        const uint16_t* slot = row_base + static_cast<size_t>(r) * group +
                               static_cast<size_t>(j) * kPickSlotsPerRank;
        m_logit[t][position] = __uint_as_float(
            static_cast<uint32_t>(decode_digits(slot, kPickLogitDigits)));
        m_id[t][position] = static_cast<int32_t>(klo);
      }
    }
  }
  __syncthreads();

  // 3. The fold normalizers (glm_sample::merge_logsumexp: the slices'
  //    terms in parallel, summed in rank order), and for the stochastic
  //    decision every candidate's fold mass and selector exp in parallel
  //    — the very values the host's loops compute one by one.
  if (tid < rows_per_request * world) {
    const int t = tid / world;
    const int r = tid % world;
    lse_terms[t][r] = detmath::exp_d(lses[t][r] - lse_top(lses[t], world));
  }
  __syncthreads();
  if (tid < rows_per_request) {
    double sum = 0.0;
    for (int r = 0; r < world; ++r) sum += lse_terms[tid][r];
    Z[tid] = held[tid] > 0 ? lse_top(lses[tid], world) + detmath::log_d(sum)
                           : 0.0;
  }
  __syncthreads();
  const bool stochastic = spec.temperature > 0.0f && held[0] > 0;
  const bool full_path = stochastic || spec.logprobs >= 0 || constrained[0] ||
                         spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  (void)full_path;
  if (stochastic) {
    const float T = spec.temperature;
    for (int t = 0; t < rows_per_request; ++t) {
      if (held[t] == 0) continue;
      const float scaled0 = __fdiv_rn(m_logit[t][0], T);
      for (int i = tid; i < held[t]; i += kVerdictThreads) {
        const float scaled = __fdiv_rn(m_logit[t][i], T);
        mass[t][i] = detmath::exp_d(static_cast<double>(scaled) - Z[t]);
        expsrc[t][i] = detmath::exp_f(__fsub_rn(scaled, scaled0));
      }
    }
  }
  __syncthreads();
  if (tid != 0) return;

  // 4. The decision (thread 0).
  GlmPickVerdict v;
  GlmSampleOutcome o;
  const bool penalized = spec.repetition_penalty != 1.0f ||
                         spec.frequency_penalty != 0.0f ||
                         spec.presence_penalty != 0.0f;
  const TopReport top0{spec.logprobs, o.top_ids[0], o.top_logprobs[0], &o.top_count[0]};
  const TopReport top1{spec.logprobs, o.top_ids[1], o.top_logprobs[1], &o.top_count[1]};
  const TopReport* rep0 = spec.logprobs >= 0 ? &top0 : nullptr;
  const TopReport* rep1 = spec.logprobs >= 0 ? &top1 : nullptr;

  v.rows = rows_per_request;
  if (!stochastic) {
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
      const double Z0 = Z[0];
      o.normalizer = Z0;
      o.logprob = __fsub_rn(m_logit[0][0], static_cast<float>(Z0));
      if (rep0 != nullptr)
        report_top(m_logit[0], m_id[0], held[0], 1.0f, static_cast<float>(Z0),
                   spec.logprobs, o.top_ids[0], o.top_logprobs[0], &o.top_count[0]);
      if (rows_per_request == 2 && held[1] > 0) {
        const double Z1 = Z[1];
        o.normalizer1 = Z1;
        o.logprob1 = __fsub_rn(m_logit[1][0], static_cast<float>(Z1));
        if (rep1 != nullptr)
          report_top(m_logit[1], m_id[1], held[1], 1.0f, static_cast<float>(Z1),
                     spec.logprobs, o.top_ids[1], o.top_logprobs[1], &o.top_count[1]);
      }
    }
  } else {
    uint64_t counter = spec.counter;
    o.sampled = 1;
    const double Z0 = Z[0];
    o.normalizer = Z0;
    if (rows_per_request == 1) {
      const Decision d = decide_prefix(m_logit[0], m_id[0], mass[0], prefix,
                                       expsrc[0], held[0], row_vocab[0], Z0,
                                       spec, exps, &counter, rep0);
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
      const bool draft_excluded =
          mask0 != nullptr &&
          (draft < 0 || draft >= vocab_size || !mask_allows(mask0, draft));
      const Decision d0 = spec_decide_prefix(m_logit[0], m_id[0], mass[0],
                                             prefix, expsrc[0], held[0],
                                             row_vocab[0], Z0, draft, spec, exps,
                                             &counter, rep0, draft_excluded);
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
      } else {
        o.accepted_draft = 1;
        v.accepted = 2;
        v.winners[0] = draft;
        o.logprob = d0.logprob;
        const double Z1 = Z[1];
        o.normalizer1 = Z1;
        const Decision d1 =
            held[1] > 0 ? decide_prefix(m_logit[1], m_id[1], mass[1], prefix,
                                        expsrc[1], held[1], row_vocab[1], Z1,
                                        spec, exps, &counter, rep1)
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

  // 5. The commit of the step's fed tokens into the request's context:
  //    the token this step consumed always, the draft only when it stood
  //    (a provisional reject leaves it out; the host adds it back if its
  //    decision accepts — glm_sample_adjust_count).
  int32_t* counts_q = counts + static_cast<size_t>(q) * vocab_size;
  const int64_t consumed = fed[row0];
  if (consumed >= 0 && consumed < vocab_size) counts_q[consumed] += 1;
  if (rows_per_request == 2 && v.accepted == 2) {
    const int64_t draft = fed[row0 + 1];
    if (draft >= 0 && draft < vocab_size) counts_q[draft] += 1;
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
                      const int32_t* counts, const uint32_t* masks,
                      int mask_stride, const uint64_t* carry_digest,
                      uint16_t* table, GlmPickLocal* locals, double* scratch,
                      cudaStream_t stream) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_local");
  if (masks != nullptr && mask_stride < glm_sample_mask_words(vocab_size))
    throw std::invalid_argument("glm_sample_local: mask stride");
  if (logits == nullptr || specs == nullptr || fed == nullptr ||
      counts == nullptr || carry_digest == nullptr || table == nullptr ||
      locals == nullptr || scratch == nullptr)
    throw std::invalid_argument("glm_sample_local: null argument");
  if (vocab_count < 1 || vocab_begin < 0 || vocab_size < 1 ||
      vocab_begin + vocab_count > vocab_size ||
      vocab_size > (1 << (6 * kPickIdDigits)) || vocab_size > (1 << kKeyIdxBits))
    throw std::invalid_argument(
        "glm_sample_local: vocab slice outside the id encodings");
  const int nchunks = (vocab_count + kSampleLseChunk - 1) / kSampleLseChunk;
  if (nchunks > kSampleMaxChunks)
    throw std::invalid_argument(
        "glm_sample_local: the slice has more normalizer chunks than the "
        "row block has threads");
  if (positions != nullptr && position_stride < 1)
    throw std::invalid_argument("glm_sample_local: position stride");
  if (rows % rows_per_request != 0)
    throw std::invalid_argument("glm_sample_local: rows per request");
  double* maxes = scratch;
  double* partials = scratch + static_cast<size_t>(rows) * nchunks;
  const dim3 chunk_grid(static_cast<unsigned>(nchunks), static_cast<unsigned>(rows));
  sample_prepare_kernel<<<chunk_grid, kChunkThreads, 0, stream>>>(
      logits, vocab_count, vocab_begin, vocab_size, specs, rows_per_request,
      fed, positions, position_stride, counts, masks, mask_stride, maxes);
  DGPP_CUDA_OK(cudaGetLastError());
  sample_partials_kernel<<<chunk_grid, kChunkThreads, 0, stream>>>(
      logits, vocab_count, specs, rows_per_request, positions,
      position_stride, masks, mask_stride, maxes, partials);
  DGPP_CUDA_OK(cudaGetLastError());
  sample_local_kernel<<<rows, kLocalThreads, 0, stream>>>(
      logits, rows, vocab_count, vocab_begin, rank, world, candidates, specs,
      rows_per_request, positions, position_stride, masks, mask_stride, maxes,
      partials, nchunks, carry_digest, table, locals);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_sample_verdict(const uint16_t* table, int rows, int world, int rank,
                        int candidates, int vocab_size, GlmSampleSpec* specs,
                        int requests, int rows_per_request, const int64_t* fed,
                        const int64_t* positions, int position_stride,
                        int32_t* counts, const uint32_t* masks, int mask_stride,
                        GlmPickVerdict* verdicts,
                        GlmPickVerdict* device_verdicts,
                        GlmSampleOutcome* outcomes, uint64_t* carry_digest,
                        cudaStream_t stream) {
  check_common(rows, world, rank, candidates, rows_per_request,
               "glm_sample_verdict");
  if (masks != nullptr && mask_stride < glm_sample_mask_words(vocab_size))
    throw std::invalid_argument("glm_sample_verdict: mask stride");
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
      fed, positions, position_stride, counts, masks, mask_stride, verdicts,
      device_verdicts, outcomes);
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
