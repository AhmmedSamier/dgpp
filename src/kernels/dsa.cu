// DSA/MLA forward kernels (see dsa.hpp for the layout and determinism
// contract). The streaming top-k selection is shared by the decode, prefill,
// and merge paths through a key-source functor: every path produces the same
// composite keys ((sortable_fp32 << 21) | pool_idx), so the selection spec —
// highest logit, exact ties to the lower pool index, ascending output — is
// implemented exactly once.
//
// expf (not __expf) everywhere exp appears: reference parity comes before
// the ~2 ulp a fast intrinsic would save; revisit only with M9 profiling
// evidence.
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"

namespace dgpp {

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;

constexpr uint64_t kKeyMax = ~0ull;
constexpr int kIdxBits = 21;  // pool ids < 2^21 (validated at launch)
constexpr uint64_t kIdxMask = (1ull << kIdxBits) - 1;
constexpr int kSelectTile = 2048;  // pools per selection tile (>= 2*select_k)
constexpr float kFp8Max = 448.0f;
constexpr float kAbsmaxFloor = 1e-4f;
constexpr int kAttnTile = 32;  // latent rows per attention tile

__device__ inline uint32_t sortable_f32_dev(float f) {
  uint32_t u = __float_as_uint(f);
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

// Smallest power of two >= v; exact powers map to themselves (see the
// reference header for why this replaces exp2f(ceilf(log2f(v)))).
__device__ inline float next_pow2_dev(float v) {
  uint32_t b = __float_as_uint(v);
  int e = int((b >> 23) & 0xFFu) - 127;
  bool exact = (b & 0x7FFFFFu) == 0;
  return exp2f(float(exact ? e : e + 1));
}

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}

// Block-wide max over one value per thread (blockDim <= 1024).
__device__ inline float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) / 32;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(~0u, v, off));
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// In-place Hadamard-128 over smem, cooperative across the block.
__device__ inline void fwht128_smem(float* x) {
  for (int stride = 1; stride < 128; stride <<= 1) {
    for (int t = threadIdx.x; t < 64; t += blockDim.x) {
      const int p = (t / stride) * (2 * stride) + (t % stride);
      const float a = x[p];
      const float b = x[p + stride];
      x[p] = a + b;
      x[p + stride] = a - b;
    }
    __syncthreads();
  }
  if (threadIdx.x < 128) x[threadIdx.x] *= 0.08838834764831845f;  // 1/sqrt(128)
  __syncthreads();
}

// Bitonic sort (ascending) of n keys in smem; n a power of two.
template <typename T>
__device__ inline void bitonic_sort_asc(T* a, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool asc = (i & size) == 0;
          if (asc ? a[i] > a[j] : a[i] < a[j]) {
            const T t = a[i];
            a[i] = a[j];
            a[j] = t;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Bitonic merge of a bitonic sequence to ascending; n a power of two.
template <typename T>
__device__ inline void bitonic_merge_asc(T* a, int n) {
  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ stride;
      if (j > i && a[i] > a[j]) {
        const T t = a[i];
        a[i] = a[j];
        a[j] = t;
      }
    }
    __syncthreads();
  }
}

// Streaming top-select_k over key_fn(pool) -> composite key, restricted to
// pools [lo, hi). best[select_k] (smem) must be pre-initialized to kKeyMax;
// on return it holds the smallest select_k keys seen, sorted ascending.
// tile (smem) needs kSelectTile entries; kSelectTile >= 2*select_k required.
template <typename KeyFn>
__device__ inline void select_topk_stream(KeyFn key_fn, int64_t lo, int64_t hi,
                                          uint64_t* best, uint64_t* tile,
                                          int select_k) {
  const int nthreads = blockDim.x;
  const int warp = threadIdx.x >> 5;
  const int nwarp = nthreads >> 5;
  for (int64_t base = lo; base < hi; base += kSelectTile) {
    const int n = int(min((int64_t)kSelectTile, hi - base));
    for (int p = threadIdx.x; p < kSelectTile; p += nthreads)
      tile[p] = kKeyMax;  // pads; [0, n) is overwritten by the warps below
    __syncthreads();
    for (int p = warp; p < n; p += nwarp) {
      const uint64_t key = key_fn(base + p);
      if ((threadIdx.x & 31) == 0) tile[p] = key;
    }
    __syncthreads();
    bitonic_sort_asc(tile, kSelectTile);
    // Merge best (asc) with the tile's smallest select_k reversed (desc):
    // [asc][desc] is bitonic, so a 2*select_k bitonic merge yields the new
    // best. Elements beyond the tile's first select_k are dominated by
    // select_k better tile keys and can never enter the global top-k.
    for (int i = threadIdx.x; i < select_k; i += nthreads)
      tile[select_k + i] = tile[select_k - 1 - i];
    __syncthreads();
    for (int i = threadIdx.x; i < select_k; i += nthreads) tile[i] = best[i];
    __syncthreads();
    bitonic_merge_asc(tile, 2 * select_k);
    for (int i = threadIdx.x; i < select_k; i += nthreads) best[i] = tile[i];
    __syncthreads();
  }
}

// Extract pool ids from sorted best keys, sort ascending, expand to token
// positions, append the query's incomplete tail. Writes out_row[0,
// max_selected) (-1 padded); returns the token count. scratch: smem int32
// [select_k]; smem_count: smem int32 [1].
__device__ inline int expand_from_best(const uint64_t* best, int select_k,
                                       int64_t pos, int kpool,
                                       int max_selected, int32_t* out_row,
                                       int32_t* scratch, int* smem_count) {
  const int nthreads = blockDim.x;
  if (threadIdx.x == 0) *smem_count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < select_k; i += nthreads) {
    if (best[i] != kKeyMax) {
      const int slot = atomicAdd(smem_count, 1);
      scratch[slot] = int32_t(best[i] & kIdxMask);
    }
  }
  __syncthreads();
  const int n_sel = *smem_count;
  for (int i = n_sel + threadIdx.x; i < select_k; i += nthreads)
    scratch[i] = INT32_MAX;
  __syncthreads();
  bitonic_sort_asc(scratch, select_k);

  const int64_t seq_len = pos + 1;
  const int64_t tail_start = (seq_len / kpool) * kpool;
  const int tail_cnt = int(seq_len - tail_start);
  const int hist = n_sel * kpool;
  for (int col = threadIdx.x; col < max_selected; col += nthreads) {
    int32_t tok;
    if (col < hist)
      tok = scratch[col / kpool] * kpool + (col % kpool);
    else if (col - hist < tail_cnt)
      tok = int32_t(tail_start + (col - hist));
    else
      tok = -1;
    out_row[col] = tok;
  }
  return hist + tail_cnt;
}

// ---------------------------------------------------------------------
// Elementwise paths
// ---------------------------------------------------------------------

__global__ void fwht_quant_rows_kernel(const uint16_t* q, uint8_t* q_fp8,
                                       float* q_scale) {
  extern __shared__ float xs[];  // [128]
  const int64_t r = blockIdx.x;
  const uint16_t* row = q + r * 128;
  if (threadIdx.x < 128)
    xs[threadIdx.x] = bf16_bits_to_float(row[threadIdx.x]);
  __syncthreads();
  fwht128_smem(xs);
  // bf16 round before quantization (pinned boundary).
  if (threadIdx.x < 128)
    xs[threadIdx.x] =
        bf16_bits_to_float(float_to_bf16_bits(xs[threadIdx.x]));
  __syncthreads();
  float absmax = (threadIdx.x < 128) ? fabsf(xs[threadIdx.x]) : 0.0f;
  __shared__ float red[32];
  absmax = block_max(absmax, red);
  absmax = fmaxf(absmax, kAbsmaxFloor);
  const float scale = next_pow2_dev(absmax * (1.0f / kFp8Max));
  if (threadIdx.x == 0) q_scale[r] = scale;
  if (threadIdx.x < 128)
    q_fp8[r * 128 + threadIdx.x] =
        float_to_fp8_e4m3_bits(xs[threadIdx.x] / scale);
}

__global__ void fold_weights_kernel(const float* weights, const float* q_scale,
                                    float* out, int64_t n, float scale) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (weights[i] * q_scale[i]) * scale;
}

__global__ void k_layernorm_kernel(const uint16_t* k_raw, int64_t k_stride,
                                   const uint16_t* w, const uint16_t* b,
                                   uint16_t* k_out, int dim, float eps) {
  const int64_t r = blockIdx.x;
  const uint16_t* row = k_raw + r * k_stride;
  __shared__ float red[32];
  float v = (threadIdx.x < dim) ? bf16_bits_to_float(row[threadIdx.x]) : 0.0f;
  float sum = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (dim + 31) / 32; ++i) t += red[i];
    red[0] = t / dim;
  }
  __syncthreads();
  const float d0 = (threadIdx.x < dim) ? v - red[0] : 0.0f;
  // Everyone has the mean now; red is reused for the variance partials.
  __syncthreads();
  float var = warp_sum(d0 * d0);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = var;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (dim + 31) / 32; ++i) t += red[i];
    red[0] = t / dim;
  }
  __syncthreads();
  if (threadIdx.x < dim) {
    const float inv = rsqrtf(red[0] + eps);
    const float y = d0 * inv * bf16_bits_to_float(w[threadIdx.x]) +
                    bf16_bits_to_float(b[threadIdx.x]);
    k_out[r * dim + threadIdx.x] = float_to_bf16_bits(y);
  }
}

__global__ void fused_qkv_rmsnorm_kernel(const uint16_t* qkv, uint16_t* q_c,
                                         uint16_t* kv_c, int q_dim, int kv_dim,
                                         const uint16_t* q_w,
                                         const uint16_t* kv_w, float eps) {
  const int64_t r = blockIdx.x;
  const uint16_t* row = qkv + r * int64_t(q_dim + kv_dim);
  __shared__ float red[32];
  __shared__ float inv[2];
  float ss = 0.0f;
  for (int d = threadIdx.x; d < q_dim; d += blockDim.x) {
    const float v = bf16_bits_to_float(row[d]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) / 32); ++i) t += red[i];
    inv[0] = rsqrtf(t / q_dim + eps);
  }
  // red is reused for the kv-half partials only after thread 0 has read it.
  __syncthreads();
  ss = 0.0f;
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
    const float v = bf16_bits_to_float(row[q_dim + d]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) / 32); ++i) t += red[i];
    inv[1] = rsqrtf(t / kv_dim + eps);
  }
  __syncthreads();
  for (int d = threadIdx.x; d < q_dim; d += blockDim.x)
    q_c[r * q_dim + d] = float_to_bf16_bits(
        bf16_bits_to_float(row[d]) * inv[0] * bf16_bits_to_float(q_w[d]));
  for (int d = threadIdx.x; d < kv_dim; d += blockDim.x)
    kv_c[r * kv_dim + d] =
        float_to_bf16_bits(bf16_bits_to_float(row[q_dim + d]) * inv[1] *
                           bf16_bits_to_float(kv_w[d]));
}

// ---------------------------------------------------------------------
// Index cache + tail machinery
// ---------------------------------------------------------------------

// Shared compression tail: bf16 round -> Hadamard-128 -> bf16 round ->
// absmax fp8 quant, writing the fp8 row and its power-of-two scale.
// xs (smem, [dim]) holds the per-dim softmax-weighted sum on entry.
__device__ inline void compress_write_quant(float* xs, uint8_t* slot_k,
                                            float* slot_scale, int dim) {
  __shared__ float red[32];
  fwht128_smem(xs);
  if (threadIdx.x < 128)
    xs[threadIdx.x] =
        bf16_bits_to_float(float_to_bf16_bits(xs[threadIdx.x]));
  __syncthreads();
  float absmax = (threadIdx.x < dim) ? fabsf(xs[threadIdx.x]) : 0.0f;
  absmax = block_max(absmax, red);
  absmax = fmaxf(absmax, kAbsmaxFloor);
  const float scale = next_pow2_dev(absmax * (1.0f / kFp8Max));
  if (threadIdx.x == 0) *slot_scale = scale;
  if (threadIdx.x < dim)
    slot_k[threadIdx.x] = float_to_fp8_e4m3_bits(xs[threadIdx.x] / scale);
}

__global__ void kpool_compress_write_kernel(
    const uint16_t* k, int64_t k_stride, const uint16_t* gate,
    int64_t gate_stride, const float* ape, const int32_t* block_table,
    int pools_per_block, int64_t first_pool, int n_pools, uint8_t* index_k,
    float* index_scale, int kpool, int dim) {
  const int i = blockIdx.x;
  if (i >= n_pools) return;
  const int64_t pool = first_pool + i;
  const int32_t blk = block_table[pool / pools_per_block];
  const int64_t slot =
      int64_t(blk) * pools_per_block + (pool % pools_per_block);

  extern __shared__ float xs[];  // [dim]
  const int d = threadIdx.x;
  float scores[8];
  float maxs = -INFINITY;
  if (d < dim) {
    for (int s = 0; s < kpool; ++s) {
      const float score =
          bf16_bits_to_float(gate[int64_t(i * kpool + s) * gate_stride + d]) +
          ape[int64_t(s) * dim + d];
      scores[s] = score;
      maxs = fmaxf(maxs, score);
    }
  }
  float acc = 0.0f;
  float denom = 0.0f;
  if (d < dim) {
    for (int s = 0; s < kpool; ++s) {
      const float prob = expf(scores[s] - maxs);
      denom += prob;
      acc +=
          bf16_bits_to_float(k[int64_t(i * kpool + s) * k_stride + d]) * prob;
    }
    xs[d] = acc / denom;
  }
  __syncthreads();
  compress_write_quant(xs, index_k + slot * dim, index_scale + slot, dim);
}

__global__ void kpool_tail_seed_kernel(const uint16_t* k, int64_t k_stride,
                                       const uint16_t* gate,
                                       int64_t gate_stride,
                                       const int32_t* req_ids,
                                       const int64_t* pos, int64_t tokens,
                                       uint16_t* tail, int kpool, int dim) {
  const int64_t i = blockIdx.x;
  if (i >= tokens) return;
  const int32_t req = req_ids[i];
  const int64_t p = pos[i];
  if (p < 0) return;
  // Seed iff the token kpool ahead is the same request still in this batch
  // (the reference's ahead-check): only the request's last kpool tokens
  // survive in the ring. The ahead index is arithmetically clamped — the
  // compiler speculates both ternary arms' loads past any bounds check, and
  // a speculated OOB read of garbage would corrupt the seeding decision.
  const int64_t ahead_idx = min(i + kpool, tokens - 1);
  const bool ahead_in_batch = (i + kpool < tokens);
  const bool ahead_same = ahead_in_batch &&
                          req_ids[ahead_idx] == req &&
                          pos[ahead_idx] >= 0;
  if (ahead_same) return;
  const int slot = int(p % kpool);
  const int64_t kbase = (int64_t(req) * 2 * kpool + slot) * dim;
  const int64_t gbase = (int64_t(req) * 2 * kpool + kpool + slot) * dim;
  const int d = threadIdx.x;
  if (d < dim) {
    tail[kbase + d] = k[i * k_stride + d];
    tail[gbase + d] = gate[i * gate_stride + d];
  }
}

__global__ void kpool_decode_update_kernel(
    const uint16_t* k, int64_t k_stride, const uint16_t* gate,
    int64_t gate_stride, const float* ape, const int64_t* pos,
    const int32_t* req_spans, const int32_t* block_tables,
    int blocks_per_request, uint16_t* tail, uint8_t* index_k,
    float* index_scale, int pools_per_block, int kpool, int dim) {
  const int req = blockIdx.x;
  const int t0 = req_spans[req * 2];
  const int t1 = t0 + req_spans[req * 2 + 1];
  const int d = threadIdx.x;
  extern __shared__ float xs[];  // [dim]

  for (int t = t0; t < t1; ++t) {
    const int64_t p = pos[t];
    if (p < 0) continue;  // uniform across threads (p is batch metadata)
    const int slot = int(p % kpool);
    const bool completing = (slot == kpool - 1);

    if (completing) {
      // The pool spans [p-kpool+1, p]; ring slot of member s is
      // (pool_start+s) % kpool; the current token overrides its own slot
      // (which still holds one pool's stale stash — the reference's
      // is_current rule).
      const int64_t pool_start = p - (kpool - 1);
      float scores[8];
      float maxs = -INFINITY;
      if (d < dim) {
        for (int s = 0; s < kpool; ++s) {
          const int ring = int((pool_start + s) % kpool);
          const float g =
              (s == kpool - 1)
                  ? bf16_bits_to_float(gate[int64_t(t) * gate_stride + d])
                  : bf16_bits_to_float(
                        tail[(int64_t(req) * 2 * kpool + kpool + ring) * dim +
                             d]);
          const float score = g + ape[int64_t(s) * dim + d];
          scores[s] = score;
          maxs = fmaxf(maxs, score);
        }
      }
      float acc = 0.0f;
      float denom = 0.0f;
      if (d < dim) {
        for (int s = 0; s < kpool; ++s) {
          const int ring = int((pool_start + s) % kpool);
          const float kk =
              (s == kpool - 1)
                  ? bf16_bits_to_float(k[int64_t(t) * k_stride + d])
                  : bf16_bits_to_float(
                        tail[(int64_t(req) * 2 * kpool + ring) * dim + d]);
          const float prob = expf(scores[s] - maxs);
          denom += prob;
          acc += kk * prob;
        }
        xs[d] = acc / denom;
      }
      __syncthreads();
      const int64_t pool = p / kpool;
      const int32_t blk = block_tables[int64_t(req) * blocks_per_request +
                                       pool / pools_per_block];
      const int64_t phys =
          int64_t(blk) * pools_per_block + (pool % pools_per_block);
      compress_write_quant(xs, index_k + phys * dim, index_scale + phys, dim);
      __syncthreads();  // ring reads done; the stash may overwrite
    }

    // Stash AFTER the completion read (ordering pinned by the reference).
    if (d < dim) {
      tail[(int64_t(req) * 2 * kpool + slot) * dim + d] =
          k[int64_t(t) * k_stride + d];
      tail[(int64_t(req) * 2 * kpool + kpool + slot) * dim + d] =
          gate[int64_t(t) * gate_stride + d];
    }
  }
}

__global__ void latent_append_kernel(const uint16_t* latent_rows,
                                     const int32_t* req_ids,
                                     const int64_t* pos,
                                     const int32_t* block_tables,
                                     int blocks_per_request, int block_tokens,
                                     uint16_t* latent_cache, int kv_lora) {
  const int64_t i = blockIdx.x;
  const int64_t p = pos[i];
  if (p < 0) return;
  const int32_t req = req_ids[i];
  const int32_t blk =
      block_tables[int64_t(req) * blocks_per_request + p / block_tokens];
  const int64_t phys = int64_t(blk) * block_tokens + (p % block_tokens);
  const uint4* s4 = reinterpret_cast<const uint4*>(latent_rows + i * kv_lora);
  uint4* d4 = reinterpret_cast<uint4*>(latent_cache + phys * kv_lora);
  for (int j = threadIdx.x; j < kv_lora / 8; j += blockDim.x) d4[j] = s4[j];
}

__global__ void gather_index_pools_kernel(const int32_t* block_table,
                                          int pools_per_block,
                                          const uint8_t* index_k,
                                          const float* index_scale,
                                          int64_t n_pools, uint8_t* out_k,
                                          float* out_scale, int dim) {
  const int64_t j = blockIdx.x;
  if (j >= n_pools) return;
  const int32_t blk = block_table[j / pools_per_block];
  const int64_t slot =
      int64_t(blk) * pools_per_block + (j % pools_per_block);
  const uint4* s = reinterpret_cast<const uint4*>(index_k + slot * dim);
  uint4* d = reinterpret_cast<uint4*>(out_k + j * dim);
  for (int i = threadIdx.x; i < dim / 16; i += blockDim.x) d[i] = s[i];
  if (threadIdx.x == 0) out_scale[j] = index_scale[slot];
}

// ---------------------------------------------------------------------
// Selection kernels
// ---------------------------------------------------------------------

// Warp-cooperative: lane h computes head h's contribution (heads == 32).
// q stays FP8 in shared memory (4 KB/row instead of 16 KB as fp32) — the
// dot loop converts on access and is memory-bound regardless; the saving
// is what lets an 8-row MTP decode batch fit the 99 KB GB10 smem optin.
struct DecodeKeyFn {
  const uint8_t* q8;   // smem [heads * 128] fp8 bits (this row)
  const float* w;      // smem [heads] (this row)
  const uint8_t* index_k;
  const float* index_scale;
  const int32_t* block_table;  // this row's request
  int pools_per_block;
  int dim;

  __device__ uint64_t operator()(int64_t pool) const {
    const int lane = threadIdx.x & 31;
    const int32_t blk = block_table[pool / pools_per_block];
    const int64_t slot =
        int64_t(blk) * pools_per_block + (pool % pools_per_block);
    const uint8_t* krow = index_k + slot * dim;
    const float ks = index_scale[slot];
    // Contraction-proof arithmetic (__fmul_rn/__fadd_rn, no FMA fusion):
    // the host oracle mirrors this exact sequence so pool-logit parity —
    // and therefore top-k position parity — is bitwise, not statistical.
    float partial = 0.0f;
#pragma unroll 4
    for (int d = 0; d < 128; ++d)
      partial = __fadd_rn(
          partial,
          __fmul_rn(fp8_e4m3_bits_to_float(q8[lane * 128 + d]),
                    fp8_e4m3_bits_to_float(krow[d])));
    const float contrib = __fmul_rn(__fmul_rn(w[lane], ks), partial);
    const float total = warp_sum(contrib);
    // ~sortable reverses the ascending float order: the smallest composite
    // key is then the HIGHEST logit (ties -> lower pool index from the idx
    // bits). Without the inversion the selection picks the worst pools.
    return (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
  }
};

struct PrefillKeyFn {
  const float* dot;       // [rows * heads, dot_stride]
  const float* w_folded;  // [rows, heads]
  const float* k_scale;   // [n_pools]
  int64_t dot_stride;
  int heads;
  int row;

  __device__ uint64_t operator()(int64_t pool) const {
    const int lane = threadIdx.x & 31;
    const float dv = dot[int64_t(row * heads + lane) * dot_stride + pool];
    const float contrib = (w_folded[row * heads + lane] * k_scale[pool]) * dv;
    const float total = warp_sum(contrib);
    // Inverted sortable: smallest composite key == highest logit (see
    // DecodeKeyFn).
    return (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
  }
};

struct MergeKeyFn {
  const uint64_t* partials;  // [grid_blocks, rows, select_k]
  int rows;
  int select_k;
  int row;

  __device__ uint64_t operator()(int64_t linear) const {
    const int64_t b = linear / select_k;
    const int p = int(linear % select_k);
    return partials[(b * rows + row) * select_k + p];
  }
};

// Fixed grid, grid-striped over each row's visible pools; the last block to
// finish merges the partials and writes the expanded token rows.
__global__ void select_decode_kernel(
    const uint8_t* q_fp8, const float* w_folded, const int32_t* req_ids,
    const int64_t* pos, int rows, const int32_t* block_tables,
    int blocks_per_request, const uint8_t* index_k, const float* index_scale,
    int pools_per_block, int heads, int select_k, int kpool, int max_selected,
    int32_t* topk_out, int32_t* out_counts, uint64_t* partial_ws,
    int32_t* counter_ws) {
  extern __shared__ uint64_t smem_u64[];
  // Layout: [best: rows*select_k][tile: kSelectTile][q8: rows*heads*128
  // bytes][w: rows*heads][scratch: select_k + 1 (i32)].
  uint64_t* best = smem_u64;
  uint64_t* tile = best + int64_t(rows) * select_k;
  uint8_t* q8 = reinterpret_cast<uint8_t*>(tile + kSelectTile);
  float* w = reinterpret_cast<float*>(q8 + int64_t(rows) * heads * 128);
  int32_t* scratch = reinterpret_cast<int32_t*>(w + int64_t(rows) * heads);

  for (int64_t i = threadIdx.x;
       i < int64_t(rows) * heads * 128; i += blockDim.x)
    q8[i] = q_fp8[i];
  for (int64_t i = threadIdx.x; i < int64_t(rows) * heads; i += blockDim.x)
    w[i] = w_folded[i];
  __syncthreads();

  for (int r = 0; r < rows; ++r) {
    const int64_t visible = (pos[r] + 1) / kpool;
    const int64_t stripe =
        (visible + gridDim.x - 1) / gridDim.x;  // >= 0; 0 when visible == 0
    const int64_t lo = min(visible, int64_t(blockIdx.x) * stripe);
    const int64_t hi = min(visible, lo + stripe);
    for (int i = threadIdx.x; i < select_k; i += blockDim.x)
      best[int64_t(r) * select_k + i] = kKeyMax;
    __syncthreads();
    DecodeKeyFn fn{q8 + int64_t(r) * heads * 128, w + int64_t(r) * heads,
                   index_k, index_scale,
                   block_tables + int64_t(req_ids[r]) * blocks_per_request,
                   pools_per_block, 128};
    select_topk_stream(fn, lo, hi, best + int64_t(r) * select_k, tile,
                       select_k);
  }
  __syncthreads();

  // Publish this block's partials, then the last block merges.
  for (int64_t i = threadIdx.x; i < int64_t(rows) * select_k; i += blockDim.x)
    partial_ws[int64_t(blockIdx.x) * rows * select_k + i] = best[i];
  __threadfence();
  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    const int ticket = atomicAdd(counter_ws, 1);
    is_last = (ticket == int(gridDim.x) - 1);
  }
  __syncthreads();
  if (!is_last) return;

  for (int r = 0; r < rows; ++r) {
    for (int i = threadIdx.x; i < select_k; i += blockDim.x)
      best[int64_t(r) * select_k + i] = kKeyMax;
    __syncthreads();
    MergeKeyFn fn{partial_ws, rows, select_k, r};
    select_topk_stream(fn, 0, int64_t(gridDim.x) * select_k,
                       best + int64_t(r) * select_k, tile, select_k);
    __syncthreads();
    int* smem_count = reinterpret_cast<int*>(scratch + select_k);
    const int cnt = expand_from_best(
        best + int64_t(r) * select_k, select_k, pos[r], kpool, max_selected,
        topk_out + int64_t(r) * max_selected, scratch, smem_count);
    if (threadIdx.x == 0) out_counts[r] = cnt;
    __syncthreads();
  }
  if (threadIdx.x == 0) *counter_ws = 0;  // self-reset for graph replay
}

__global__ void select_prefill_kernel(const float* dot, int64_t dot_stride,
                                      const float* w_folded,
                                      const float* k_scale, const int64_t* pos,
                                      int rows, int64_t n_pools, int heads,
                                      int select_k, int kpool,
                                      int max_selected, int32_t* topk_out,
                                      int32_t* out_counts) {
  extern __shared__ uint64_t smem_u64[];
  uint64_t* best = smem_u64;                // [select_k]
  uint64_t* tile = best + select_k;         // [kSelectTile]
  int32_t* scratch = reinterpret_cast<int32_t*>(tile + kSelectTile);
  int* smem_count = reinterpret_cast<int*>(scratch + select_k);

  const int r = blockIdx.x;
  const int64_t visible = (pos[r] + 1) / kpool;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) best[i] = kKeyMax;
  __syncthreads();
  PrefillKeyFn fn{dot, w_folded, k_scale, dot_stride, heads, r};
  select_topk_stream(fn, 0, min(visible, n_pools), best, tile, select_k);
  __syncthreads();
  const int cnt = expand_from_best(best, select_k, pos[r], kpool, max_selected,
                                   topk_out + int64_t(r) * max_selected,
                                   scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// ---------------------------------------------------------------------
// MLA absorbed attention
// ---------------------------------------------------------------------

__global__ void absorb_q_kernel(const uint16_t* q, const uint16_t* kv_b,
                                uint16_t* q_tilde, int local_heads, int nope,
                                int v, int kv_lora) {
  // One block per (row, head); each thread owns 4 output columns, so a warp
  // reads 32*4 consecutive bf16 of every W row (coalesced). kv_b is the
  // checkpoint's interleaved layout: head h owns rows
  // [h*(nope+v), h*(nope+v)+nope) of W_uk.
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const uint16_t* qh = q + (r * local_heads + h) * nope;
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;
  uint16_t* out = q_tilde + (r * local_heads + h) * kv_lora;
  __shared__ uint16_t qs[256];
  for (int d = threadIdx.x; d < nope; d += blockDim.x) qs[d] = qh[d];
  __syncthreads();
  const int col = threadIdx.x * 4;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int d = 0; d < nope; ++d) {
    const float qv = bf16_bits_to_float(qs[d]);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      acc[j] += qv * bf16_bits_to_float(wuk[int64_t(d) * kv_lora + col + j]);
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) out[col + j] = float_to_bf16_bits(acc[j]);
}

// One block per (row, split, head-group). Thread (h, g) within the block:
// head h, dim group g (contiguous lanes, so the group reduce is shuffles).
__global__ void attn_partial_kernel(
    const uint16_t* q_tilde, const uint16_t* latent_cache,
    const int32_t* req_ids, const int32_t* topk, int topk_stride,
    const int32_t* counts, int n_split, int local_heads, int kv_lora,
    int block_tokens, const int32_t* block_tables, int blocks_per_request,
    float scale, float* m_ws, float* l_ws, float* c_ws) {
  const int64_t r = blockIdx.x;
  const int s = blockIdx.y;
  const int hpb = local_heads / gridDim.z;  // heads per block
  const int h0 = blockIdx.z * hpb;
  const int groups = blockDim.x / hpb;
  const int h = h0 + threadIdx.x / groups;
  const int g = threadIdx.x % groups;
  const int d0 = g * (kv_lora / groups);
  const int dslice = kv_lora / groups;

  const int cnt = counts[r];
  const int chunk = (cnt + n_split - 1) / n_split;
  const int t_begin = s * chunk;
  const int t_end = min(cnt, t_begin + chunk);

  extern __shared__ uint16_t sm16[];
  uint16_t* qt = sm16;  // [hpb * kv_lora] absorbed queries (head-local rows)
  uint16_t* lat = qt + hpb * kv_lora;  // [kAttnTile * kv_lora] latent tile
  float* c =
      reinterpret_cast<float*>(lat + kAttnTile * kv_lora);  // [hpb * kv_lora]
  float* scores = c + hpb * kv_lora;  // [hpb * kAttnTile]
  float* l = scores + hpb * kAttnTile;  // [hpb], single-writer (g == 0)

  // The running softmax max is per-lane register state: the group's
  // butterfly reduction leaves identical values in every lane, so no
  // shared m exists to order (this replaces a smem m/m_new pair whose
  // update needed a barrier between the readers and the writer).
  float m_reg = -INFINITY;

  const int hl = h - h0;  // head index local to this block
  for (int64_t i = threadIdx.x; i < int64_t(hpb) * kv_lora; i += blockDim.x)
    qt[i] = q_tilde[(r * local_heads + h0) * kv_lora + i];
  for (int i = threadIdx.x; i < hpb; i += blockDim.x) l[i] = 0.0f;
  for (int64_t i = threadIdx.x; i < int64_t(hpb) * kv_lora; i += blockDim.x)
    c[i] = 0.0f;
  __syncthreads();

  const int32_t req = req_ids[r];
  const int32_t* bt = block_tables + int64_t(req) * blocks_per_request;
  const int32_t* toks = topk + int64_t(r) * topk_stride;

  for (int t0 = t_begin; t0 < t_end; t0 += kAttnTile) {
    const int n = min(kAttnTile, t_end - t0);
    // Load the latent tile (bf16 rows gathered through the block table).
    for (int i = threadIdx.x; i < n * kv_lora; i += blockDim.x) {
      const int tt = i / kv_lora;
      const int cc = i % kv_lora;
      const int64_t tok = toks[t0 + tt];
      const int32_t blk = bt[tok / block_tokens];
      const int64_t phys =
          int64_t(blk) * block_tokens + (tok % block_tokens);
      lat[tt * kv_lora + cc] = latent_cache[phys * kv_lora + cc];
    }
    __syncthreads();

    // Scores for this thread's head over the tile: every dim group computes
    // its partial dot, then a butterfly group reduce — unlike a down-reduce,
    // the butterfly leaves the identical sum in EVERY lane of the group, so
    // the running max below is consistent per-lane with no shared state.
    float tile_max = -INFINITY;
    for (int tt = 0; tt < n; ++tt) {
      float partial = 0.0f;
      for (int dd = 0; dd < dslice; ++dd)
        partial += bf16_bits_to_float(qt[hl * kv_lora + d0 + dd]) *
                   bf16_bits_to_float(lat[tt * kv_lora + d0 + dd]);
#pragma unroll
      for (int off = groups / 2; off > 0; off >>= 1)
        partial += __shfl_xor_sync(~0u, partial, off);
      const float score = partial * scale;
      if (g == 0) scores[hl * kAttnTile + tt] = score;
      tile_max = fmaxf(tile_max, score);
    }
    __syncthreads();

    // Online softmax update; m lives in registers, l in smem under its
    // single writer — no barrier needed between reading the old max and
    // committing the new one.
    const float m_new = fmaxf(m_reg, tile_max);
    const float rescale = expf(m_reg - m_new);
    for (int dd = 0; dd < dslice; ++dd)
      c[hl * kv_lora + d0 + dd] *= rescale;
    if (g == 0) {
      float ladd = 0.0f;
      for (int tt = 0; tt < n; ++tt)
        ladd += expf(scores[hl * kAttnTile + tt] - m_new);
      l[hl] = l[hl] * rescale + ladd;
    }
    // c accumulation: probs round to bf16 (pinned; l stays unrounded).
    for (int tt = 0; tt < n; ++tt) {
      const float p = bf16_bits_to_float(float_to_bf16_bits(
          expf(scores[hl * kAttnTile + tt] - m_new)));
      for (int dd = 0; dd < dslice; ++dd)
        c[hl * kv_lora + d0 + dd] +=
            p * bf16_bits_to_float(lat[tt * kv_lora + d0 + dd]);
    }
    m_reg = m_new;
    __syncthreads();
  }

  // Publish partials.
  const int64_t base_m = (r * n_split + s) * local_heads + h;
  if (g == 0) {
    m_ws[base_m] = m_reg;
    l_ws[base_m] = l[hl];
  }
  for (int dd = threadIdx.x; dd < hpb * kv_lora; dd += blockDim.x) {
    const int hh = dd / kv_lora;
    const int cc = dd % kv_lora;
    c_ws[((r * n_split + s) * local_heads + h0 + hh) * kv_lora + cc] =
        c[dd];
  }
}

__global__ void attn_combine_kernel(const float* m_ws, const float* l_ws,
                                    const float* c_ws, int n_split,
                                    int local_heads, int kv_lora,
                                    float* c_out) {
  const int64_t r = blockIdx.x;
  const int64_t total = int64_t(local_heads) * kv_lora;
  for (int64_t i = int64_t(threadIdx.x); i < total; i += blockDim.x) {
    const int h = int(i / kv_lora);
    const int cc = int(i % kv_lora);
    float mhat = -INFINITY;
    for (int s = 0; s < n_split; ++s)
      mhat = fmaxf(mhat, m_ws[(r * n_split + s) * local_heads + h]);
    if (mhat == -INFINITY) {
      c_out[r * total + i] = 0.0f;  // empty row (padding)
      continue;
    }
    float num = 0.0f;
    float den = 0.0f;
    for (int s = 0; s < n_split; ++s) {
      const float p = expf(m_ws[(r * n_split + s) * local_heads + h] - mhat);
      num += p * c_ws[((r * n_split + s) * local_heads + h) * kv_lora + cc];
      den += p * l_ws[(r * n_split + s) * local_heads + h];
    }
    c_out[r * total + i] = (den > 0.0f) ? num / den : 0.0f;
  }
}

__global__ void vout_gemm_kernel(const float* c, const uint16_t* kv_b,
                                 uint16_t* out, int local_heads, int nope,
                                 int v, int kv_lora) {
  // One block per (row, head); thread owns output rows, streaming W rows
  // through L1. kv_b interleaved: head h's W_uv rows are
  // [h*(nope+v)+nope, (h+1)*(nope+v)).
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  __shared__ float cs[512];
  for (int cc = threadIdx.x; cc < kv_lora; cc += blockDim.x)
    cs[cc] = c[(r * local_heads + h) * kv_lora + cc];
  __syncthreads();
  const uint16_t* wuv =
      kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
  uint16_t* orow = out + (r * local_heads + h) * v;
  for (int d = threadIdx.x; d < v; d += blockDim.x) {
    float acc = 0.0f;
    for (int cc = 0; cc < kv_lora; ++cc)
      acc += bf16_bits_to_float(wuv[int64_t(d) * kv_lora + cc]) * cs[cc];
    orow[d] = float_to_bf16_bits(acc);
  }
}

}  // namespace

// ---------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------

void dsa_fwht_quant_rows(const void* q_bf16, int64_t rows, void* q_fp8,
                         float* q_scale, cudaStream_t stream) {
  fwht_quant_rows_kernel<<<unsigned(rows), 128, 128 * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(q_bf16),
      static_cast<uint8_t*>(q_fp8), q_scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_fold_weights(const float* weights, const float* q_scale, float* out,
                      int64_t n, float scale, cudaStream_t stream) {
  const int blocks = unsigned((n + 255) / 256);
  fold_weights_kernel<<<blocks, 256, 0, stream>>>(weights, q_scale, out, n,
                                                  scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_k_layernorm(const void* k_raw, int64_t k_stride, const void* w,
                     const void* b, void* k_out, int64_t rows, int dim,
                     float eps, cudaStream_t stream) {
  k_layernorm_kernel<<<unsigned(rows), 128, 0, stream>>>(
      static_cast<const uint16_t*>(k_raw), k_stride,
      static_cast<const uint16_t*>(w), static_cast<const uint16_t*>(b),
      static_cast<uint16_t*>(k_out), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_fused_qkv_rmsnorm(const void* qkv, void* q_c, void* kv_c, int q_dim,
                           int kv_dim, int64_t rows, const void* q_w,
                           const void* kv_w, float eps, cudaStream_t stream) {
  fused_qkv_rmsnorm_kernel<<<unsigned(rows), 256, 0, stream>>>(
      static_cast<const uint16_t*>(qkv), static_cast<uint16_t*>(q_c),
      static_cast<uint16_t*>(kv_c), q_dim, kv_dim,
      static_cast<const uint16_t*>(q_w), static_cast<const uint16_t*>(kv_w),
      eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_compress_write(const void* k, int64_t k_stride,
                              const void* gate, int64_t gate_stride,
                              const float* ape, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool,
                              int n_pools, void* index_k, float* index_scale,
                              int kpool, int dim, cudaStream_t stream) {
  if (n_pools <= 0) return;
  kpool_compress_write_kernel<<<unsigned(n_pools), 128,
                                dim * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, ape, block_table,
      pools_per_block, first_pool, n_pools, static_cast<uint8_t*>(index_k),
      index_scale, kpool, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_tail_seed(const void* k, int64_t k_stride, const void* gate,
                         int64_t gate_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, void* tail,
                         int kpool, int dim, cudaStream_t stream) {
  if (tokens <= 0) return;
  kpool_tail_seed_kernel<<<unsigned(tokens), 128, 0, stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, req_ids, pos, tokens,
      static_cast<uint16_t*>(tail), kpool, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_kpool_decode_update(const void* k, int64_t k_stride,
                             const void* gate, int64_t gate_stride,
                             const float* ape, const int64_t* pos,
                             const int32_t* req_spans, int num_requests,
                             const int32_t* block_tables,
                             int blocks_per_request, void* tail,
                             void* index_k, float* index_scale,
                             int pools_per_block, int kpool, int dim,
                             cudaStream_t stream) {
  if (num_requests <= 0) return;
  kpool_decode_update_kernel<<<unsigned(num_requests), 128,
                               dim * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, ape, pos, req_spans,
      block_tables, blocks_per_request, static_cast<uint16_t*>(tail),
      static_cast<uint8_t*>(index_k), index_scale, pools_per_block, kpool,
      dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_latent_append(const void* latent_rows, const int32_t* req_ids,
                       const int64_t* pos, int64_t tokens,
                       const int32_t* block_tables, int blocks_per_request,
                       int block_tokens, void* latent_cache, int kv_lora,
                       cudaStream_t stream) {
  if (tokens <= 0) return;
  latent_append_kernel<<<unsigned(tokens), 128, 0, stream>>>(
      static_cast<const uint16_t*>(latent_rows), req_ids, pos, block_tables,
      blocks_per_request, block_tokens,
      static_cast<uint16_t*>(latent_cache), kv_lora);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_gather_index_pools(const int32_t* block_table, int pools_per_block,
                            const void* index_k, const float* index_scale,
                            int64_t n_pools, void* out_k, float* out_scale,
                            int dim, cudaStream_t stream) {
  if (n_pools <= 0) return;
  gather_index_pools_kernel<<<unsigned(n_pools), 128, 0, stream>>>(
      block_table, pools_per_block, static_cast<const uint8_t*>(index_k),
      index_scale, n_pools, static_cast<uint8_t*>(out_k), out_scale, dim);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_select_decode(const void* q_fp8, const float* w_folded,
                       const int32_t* req_ids, const int64_t* pos, int rows,
                       const int32_t* block_tables, int blocks_per_request,
                       const void* index_k, const float* index_scale,
                       int pools_per_block, int heads, int dim, int select_k,
                       int kpool, int max_selected, int32_t* topk_out,
                       int32_t* out_counts, uint64_t* partial_ws,
                       int32_t* counter_ws, int grid_blocks,
                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads != 32) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (select_k > kSelectTile / 2) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (rows > 8) DGPP_CUDA_OK(cudaErrorInvalidValue);
  const size_t smem = size_t(rows) * select_k * 8 + kSelectTile * 8 +
                      size_t(rows) * heads * 128 +
                      size_t(rows) * heads * 4 + (select_k + 1) * 4;
  static int smem_cap = -1;
  if (smem_cap < 0) {
    cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           0);
    smem_cap -= 1024;  // leave room for static smem + alignment
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_cap));
  }
  if (smem > size_t(smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
  const int blocks = grid_blocks > 0 ? grid_blocks : 48;
  DGPP_CUDA_OK(cudaMemsetAsync(counter_ws, 0, sizeof(int32_t), stream));
  select_decode_kernel<<<blocks, 128, smem, stream>>>(
      static_cast<const uint8_t*>(q_fp8), w_folded, req_ids, pos, rows,
      block_tables, blocks_per_request,
      static_cast<const uint8_t*>(index_k), index_scale, pools_per_block,
      heads, select_k, kpool, max_selected, topk_out, out_counts, partial_ws,
      counter_ws);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_select_prefill(const float* dot, int64_t dot_stride,
                        const float* w_folded, const float* k_scale,
                        const int64_t* pos, int rows, int64_t n_pools,
                        int heads, int select_k, int kpool, int max_selected,
                        int32_t* topk_out, int32_t* out_counts,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (heads != 32) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (select_k > kSelectTile / 2) DGPP_CUDA_OK(cudaErrorInvalidValue);
  const size_t smem =
      size_t(select_k) * 8 + kSelectTile * 8 + (select_k + 1) * 4;
  select_prefill_kernel<<<unsigned(rows), 128, smem, stream>>>(
      dot, dot_stride, w_folded, k_scale, pos, rows, n_pools, heads, select_k,
      kpool, max_selected, topk_out, out_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_absorb_q(const void* q, const void* kv_b, void* q_tilde,
                  int64_t rows, int local_heads, int nope, int v, int kv_lora,
                  cudaStream_t stream) {
  if (rows <= 0) return;
  if (nope > 256 || kv_lora % 4 != 0 || 512 % kv_lora != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);  // static q smem + column mapping
  dim3 grid{unsigned(rows), unsigned(local_heads)};
  absorb_q_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const uint16_t*>(q), static_cast<const uint16_t*>(kv_b),
      static_cast<uint16_t*>(q_tilde), local_heads, nope, v, kv_lora);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_attn_partial(const void* q_tilde, const void* latent_cache,
                      const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows,
                      int n_split, int local_heads, int kv_lora,
                      int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale, float* m_ws,
                      float* l_ws, float* c_ws, cudaStream_t stream) {
  if (rows <= 0) return;
  const int hpb = local_heads < 16 ? local_heads : 16;  // smem bound
  if (local_heads % hpb != 0 || 128 % hpb != 0 || 128 / hpb > 32)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora % 8 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  const size_t smem = (size_t(hpb) * kv_lora * 2 +
                       kAttnTile * kv_lora * 2 + hpb * kv_lora * 4 +
                       hpb * kAttnTile * 4 + hpb * 4 + 15) &
                      ~size_t(15);
  static int smem_cap = -1;
  if (smem_cap < 0) {
    cudaDeviceGetAttribute(&smem_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           0);
    smem_cap -= 1024;  // static smem + alignment margin
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        attn_partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_cap));
  }
  if (smem > size_t(smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
  dim3 grid{unsigned(rows), unsigned(n_split),
            unsigned(local_heads / hpb)};
  attn_partial_kernel<<<grid, 128, smem, stream>>>(
      static_cast<const uint16_t*>(q_tilde),
      static_cast<const uint16_t*>(latent_cache), req_ids, topk, topk_stride,
      counts, n_split, local_heads, kv_lora, block_tokens, block_tables,
      blocks_per_request, scale, m_ws, l_ws, c_ws);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws,
                      int rows, int n_split, int local_heads, int kv_lora,
                      float* c_out, cudaStream_t stream) {
  if (rows <= 0) return;
  attn_combine_kernel<<<unsigned(rows), 128, 0, stream>>>(
      m_ws, l_ws, c_ws, n_split, local_heads, kv_lora, c_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_vout_gemm(const void* c, const void* kv_b, void* out,
                   int64_t rows, int local_heads, int nope, int v,
                   int kv_lora, cudaStream_t stream) {
  if (rows <= 0) return;
  if (kv_lora > 512) DGPP_CUDA_OK(cudaErrorInvalidValue);  // static c smem
  dim3 grid{unsigned(rows), unsigned(local_heads)};
  vout_gemm_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const float*>(c), static_cast<const uint16_t*>(kv_b),
      static_cast<uint16_t*>(out), local_heads, nope, v, kv_lora);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
