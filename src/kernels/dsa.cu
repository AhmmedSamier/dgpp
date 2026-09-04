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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/topk_select.cuh"

namespace dgpp {

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;

// ---- native conversion helpers ----------------------------------------
// The GB10 has hardware e4m3->f16 and bf16->f32 conversions; the software
// decoders in dtypes.hpp are branchy (NaN/denormal special cases) and were
// ~90% of the select kernel's cycles. Both paths are EXACT — e4m3 fits
// entirely inside f16, bf16 inside f32 — so these are bit-identical
// replacements, verified by the bitwise select fuzz.

__device__ inline float2 fp8x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
  return __half22float2(__half2(h));
}

__device__ inline float2 bf16x2_to_float2(uint32_t v) {
  __nv_bfloat16_raw lo, hi;
  lo.x = static_cast<unsigned short>(v & 0xFFFFu);
  hi.x = static_cast<unsigned short>(v >> 16);
  return __bfloat1622float2(
      __nv_bfloat162(__nv_bfloat16(lo), __nv_bfloat16(hi)));
}

// acc += sum(element-wise products of 8 bf16 pairs), accumulated in element
// order (same sequence as the scalar loop it replaces — FFMA-friendly; the
// attention path is tolerance-pinned, not bitwise).
__device__ inline float dot8_bf16(uint4 a, uint4 b, float acc) {
  const uint32_t* a32 = reinterpret_cast<const uint32_t*>(&a);
  const uint32_t* b32 = reinterpret_cast<const uint32_t*>(&b);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 x = bf16x2_to_float2(a32[j]);
    const float2 y = bf16x2_to_float2(b32[j]);
    acc += x.x * y.x;
    acc += x.y * y.y;
  }
  return acc;
}

constexpr int kIdxBits = 21;  // pool ids < 2^21 (validated at launch)
constexpr uint64_t kIdxMask = (1ull << kIdxBits) - 1;
constexpr float kFp8Max = 448.0f;
constexpr float kAbsmaxFloor = 1e-4f;
constexpr int kAttnTile = 32;  // latent rows per attention tile

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

// Extract pool ids from sorted best keys, sort ascending, expand to token
// positions, append the query's incomplete tail. Writes out_row[0,
// max_selected) (-1 padded); returns the token count. scratch: smem int32
// [select_k]; smem_count: smem int32 [1].
__device__ inline int expand_from_best(const uint32_t* best_hi,
                                       const uint32_t* best_lo, int select_k,
                                       int64_t pos, int kpool,
                                       int max_selected, int32_t* out_row,
                                       int32_t* scratch, int* smem_count) {
  const int nthreads = blockDim.x;
  if (threadIdx.x == 0) *smem_count = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < select_k; i += nthreads) {
    const bool real = best_hi[i] != 0xFFFFFFFFu || best_lo[i] != 0xFFFFFFFFu;
    if (real) {
      const int slot = atomicAdd(smem_count, 1);
      scratch[slot] = int32_t(
          (uint64_t(best_hi[i]) << 32 | best_lo[i]) & kIdxMask);
    }
  }
  __syncthreads();
  const int n_sel = *smem_count;
  for (int i = n_sel + threadIdx.x; i < select_k; i += nthreads)
    scratch[i] = INT32_MAX;
  __syncthreads();
  // Sort the selected pool ids ascending (single 32-bit array; positive
  // ids, so int32 ordering == u32 ordering).
  for (int size = 2; size <= select_k; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool asc = (i & size) == 0;
          if (asc ? scratch[i] > scratch[j] : scratch[i] < scratch[j]) {
            const int32_t t = scratch[i];
            scratch[i] = scratch[j];
            scratch[j] = t;
          }
        }
      }
      __syncthreads();
    }
  }
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
    int64_t gate_stride, const float* ape, const int32_t* req_ids,
    const int64_t* pos, const int32_t* req_spans,
    const int32_t* block_tables,
    int blocks_per_request, uint16_t* tail, uint8_t* index_k,
    float* index_scale, int pools_per_block, int kpool, int dim,
    uint16_t* tail_snapshots) {
  const int span = blockIdx.x;
  const int t0 = req_spans[span * 2];
  const int t1 = t0 + req_spans[span * 2 + 1];
  int first_real = t0;
  while (first_real < t1 && pos[first_real] < 0) ++first_real;
  if (first_real == t1) return;
  const int req = req_ids[first_real];
  const int d = threadIdx.x;
  extern __shared__ float xs[];  // [dim]
  const int ring_elems = 2 * kpool * dim;

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
    // Speculative rows: the ring after batch row t is the state to restore
    // if rows > t are rejected (the last row's ring stays in place). The
    // ring is the ONE non-idempotent DSA write — latent rows and completed
    // pools are positional and a rewound position simply overwrites them.
    if (tail_snapshots && t + 1 < t1) {
      __syncthreads();  // every thread's stash is visible before the copy
      const uint16_t* ring = tail + int64_t(req) * ring_elems;
      uint16_t* snap = tail_snapshots + int64_t(t) * ring_elems;
      for (int e = threadIdx.x; e < ring_elems; e += blockDim.x)
        snap[e] = ring[e];
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
  static constexpr bool kWarpCooperative = true;  // warp computes one pool
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
    // The element order matches the scalar form exactly (d = 0..127); only
    // the fp8 decode is vectorized (native hardware, bit-exact — see the
    // conversion helpers above).
    float partial = 0.0f;
    const uint2* q2 = reinterpret_cast<const uint2*>(q8 + lane * 128);
    const uint2* k2 = reinterpret_cast<const uint2*>(krow);
#pragma unroll 8
    for (int p = 0; p < 16; ++p) {
      const uint2 qv = q2[p];  // 8 fp8 values of this lane's head
      const uint2 kv = k2[p];  // 8 fp8 values of the pool row
      const uint16_t* qq = reinterpret_cast<const uint16_t*>(&qv);
      const uint16_t* kk = reinterpret_cast<const uint16_t*>(&kv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 a = fp8x2_to_float2(qq[j]);
        const float2 b = fp8x2_to_float2(kk[j]);
        partial = __fadd_rn(partial, __fmul_rn(a.x, b.x));
        partial = __fadd_rn(partial, __fmul_rn(a.y, b.y));
      }
    }
    const float contrib = __fmul_rn(__fmul_rn(w[lane], ks), partial);
    const float total = warp_sum(contrib);
    // ~sortable reverses the ascending float order: the smallest composite
    // key is then the HIGHEST logit (ties -> lower pool index from the idx
    // bits). Without the inversion the selection picks the worst pools.
    return (uint64_t(~sortable_f32_dev(total)) << kIdxBits) | uint64_t(pool);
  }
};

struct PrefillKeyFn {
  static constexpr bool kWarpCooperative = true;  // warp computes one pool
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
  static constexpr bool kWarpCooperative = false;  // one load per key
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
// The select counter's reset is a KERNEL, not cudaMemsetAsync. A memset
// node in the captured decode graph executes on the copy-engine queue, an
// in-order queue shared by every stream in the process; a queued node's
// dependency wait blocks everything behind it. In a one-process multi-rank
// world (the loopback gates) a peer rank's queued post-collective memset
// held this reset behind it while that peer's collective spun waiting on
// ours — the batched-MTP graph stall (docs/batched_mtp_graph_stall.md).
// Kernel nodes never share that queue; the graph engine rejects any
// non-kernel node at capture.
__global__ void select_counter_reset_kernel(int32_t* counter) {
  if (threadIdx.x == 0) *counter = 0;
}

__global__ void select_decode_kernel(
    const uint8_t* q_fp8, const float* w_folded, const int32_t* req_ids,
    const int64_t* pos, int rows, const int32_t* block_tables,
    int blocks_per_request, const uint8_t* index_k, const float* index_scale,
    int pools_per_block, int heads, int select_k, int kpool, int max_selected,
    int32_t* topk_out, int32_t* out_counts, uint64_t* partial_ws,
    int32_t* counter_ws) {
  extern __shared__ uint64_t smem_u64[];
  // Layout (split 32-bit key arrays; see the bitonic networks above):
  // [best_hi/best_lo: rows*select_k each][tile_hi/tile_lo: kSelectTile each]
  // [q8: rows*heads*128 bytes][w: rows*heads][scratch: select_k + 1 (i32)].
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_u64);
  uint32_t* best_lo = best_hi + int64_t(rows) * select_k;
  uint32_t* tile_hi = best_lo + int64_t(rows) * select_k;
  uint32_t* tile_lo = tile_hi + kSelectTile;
  uint8_t* q8 = reinterpret_cast<uint8_t*>(tile_lo + kSelectTile);
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
    // Short contexts (visible <= select_k pools) select EVERY pool: the
    // top-k of at most k candidates is all of them, and the expansion
    // sorts ids ascending anyway — the streaming selection and the merge
    // are no-ops here, and at 2K-token contexts they were the whole 263us
    // of this kernel (the T=1 profile). The merge below rebuilds the same
    // best[] directly; the result is the general path's, bit for bit.
    // (Sentinels keep the published partials initialized — initcheck
    // discipline; the merge never reads them for these rows.)
    if (visible <= select_k) {
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        best_hi[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
        best_lo[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
      }
      continue;
    }
    const int64_t stripe =
        (visible + gridDim.x - 1) / gridDim.x;  // >= 0; 0 when visible == 0
    const int64_t lo = min(visible, int64_t(blockIdx.x) * stripe);
    const int64_t hi = min(visible, lo + stripe);
    for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
      best_hi[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
      best_lo[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
    }
    __syncthreads();
    DecodeKeyFn fn{q8 + int64_t(r) * heads * 128, w + int64_t(r) * heads,
                   index_k, index_scale,
                   block_tables + int64_t(req_ids[r]) * blocks_per_request,
                   pools_per_block, 128};
    select_topk_stream(fn, lo, hi, best_hi + int64_t(r) * select_k,
                       best_lo + int64_t(r) * select_k, tile_hi, tile_lo,
                       select_k);
  }
  __syncthreads();

  // Publish this block's partials, then the last block merges.
  for (int64_t i = threadIdx.x; i < int64_t(rows) * select_k; i += blockDim.x)
    partial_ws[int64_t(blockIdx.x) * rows * select_k + i] =
        uint64_t(best_hi[i]) << 32 | best_lo[i];
  __threadfence();
  __shared__ bool is_last;
  if (threadIdx.x == 0) {
    const int ticket = atomicAdd(counter_ws, 1);
    is_last = (ticket == int(gridDim.x) - 1);
  }
  __syncthreads();
  if (!is_last) return;

  for (int r = 0; r < rows; ++r) {
    const int64_t visible = (pos[r] + 1) / kpool;
    if (visible <= select_k) {
      // Every visible pool is selected (see above): keys carry only the
      // pool id in their low kIdxBits — the expansion reads nothing else.
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        const bool real = i < visible;
        best_hi[int64_t(r) * select_k + i] = real ? 0u : 0xFFFFFFFFu;
        best_lo[int64_t(r) * select_k + i] =
            real ? uint32_t(i) : 0xFFFFFFFFu;
      }
      __syncthreads();
    } else {
      for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
        best_hi[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
        best_lo[int64_t(r) * select_k + i] = 0xFFFFFFFFu;
      }
      __syncthreads();
      // Only blocks whose stripe was non-empty published real keys; the
      // rest hold sentinels the merge can skip (block b's stripe starts at
      // b*stripe, so the non-empty ones are the first ceil(visible/stripe)).
      const int64_t stripe = (visible + gridDim.x - 1) / gridDim.x;
      const int64_t live_blocks =
          min(int64_t(gridDim.x), (visible + stripe - 1) / stripe);
      MergeKeyFn fn{partial_ws, rows, select_k, r};
      select_topk_stream(fn, 0, live_blocks * select_k,
                         best_hi + int64_t(r) * select_k,
                         best_lo + int64_t(r) * select_k, tile_hi, tile_lo,
                         select_k);
      __syncthreads();
    }
    int* smem_count = reinterpret_cast<int*>(scratch + select_k);
    const int cnt = expand_from_best(
        best_hi + int64_t(r) * select_k, best_lo + int64_t(r) * select_k,
        select_k, pos[r], kpool, max_selected,
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
  uint32_t* best_hi = reinterpret_cast<uint32_t*>(smem_u64);  // [select_k]
  uint32_t* best_lo = best_hi + select_k;                     // [select_k]
  uint32_t* tile_hi = best_lo + select_k;                     // [kSelectTile]
  uint32_t* tile_lo = tile_hi + kSelectTile;                  // [kSelectTile]
  int32_t* scratch = reinterpret_cast<int32_t*>(tile_lo + kSelectTile);
  int* smem_count = reinterpret_cast<int*>(scratch + select_k);

  const int r = blockIdx.x;
  const int64_t visible = (pos[r] + 1) / kpool;
  for (int i = threadIdx.x; i < select_k; i += blockDim.x) {
    best_hi[i] = 0xFFFFFFFFu;
    best_lo[i] = 0xFFFFFFFFu;
  }
  __syncthreads();
  PrefillKeyFn fn{dot, w_folded, k_scale, dot_stride, heads, r};
  select_topk_stream(fn, 0, min(visible, n_pools), best_hi, best_lo, tile_hi,
                     tile_lo, select_k);
  __syncthreads();
  const int cnt = expand_from_best(best_hi, best_lo, select_k, pos[r], kpool,
                                   max_selected,
                                   topk_out + int64_t(r) * max_selected,
                                   scratch, smem_count);
  if (threadIdx.x == 0) out_counts[r] = cnt;
}

// ---------------------------------------------------------------------
// MLA absorbed attention
// ---------------------------------------------------------------------

// absorb_q: q_tilde[r, h, :] = q[r, h, :] (nope) x W_uk[h] (nope x kv_lora).
// One block of kAbsorbGroups x 128 threads per (row, head). Every thread
// owns 8 output columns (one uint4 of W per row it visits); the groups
// split the nope rows round-robin, so a block keeps kAbsorbGroups x 128 x
// (rows in flight) loads outstanding instead of 128 x 4 — the previous
// one-group form was latency-bound at ~40 us per layer for 2 MB of
// weights. The groups' partials meet in shared memory and are summed in
// group order (deterministic; the reassociation vs. the single-chain
// version is fp32 rounding, accepted 2026-09-02). kv_b is the
// checkpoint's interleaved layout: head h owns rows [h*(nope+v),
// h*(nope+v)+nope) of W_uk.
constexpr int kAbsorbGroups = 8;
constexpr int kAbsorbGroupThreads = 128;
constexpr int kAbsorbThreads = kAbsorbGroups * kAbsorbGroupThreads;

__global__ __launch_bounds__(kAbsorbThreads) void absorb_q_kernel(
    const uint16_t* q, const uint16_t* kv_b, uint16_t* q_tilde,
    int local_heads, int nope, int v, int kv_lora) {
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const uint16_t* qh = q + (r * local_heads + h) * nope;
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;
  uint16_t* out = q_tilde + (r * local_heads + h) * kv_lora;
  __shared__ float qs[256];
  __shared__ __align__(16) float partial[kAbsorbGroups][512];
  for (int d = threadIdx.x; d < nope; d += blockDim.x)
    qs[d] = bf16_bits_to_float(qh[d]);
  __syncthreads();
  const int group = threadIdx.x / kAbsorbGroupThreads;
  const int col = (threadIdx.x % kAbsorbGroupThreads) * 8;
  // kv_lora % 8 == 0 and kv_lora <= 512 validated at launch; smaller ranks
  // idle their excess threads (an unguarded write would land in the next
  // head's row — silent corruption, not an error).
  if (col < kv_lora) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
    for (int d = group; d < nope; d += kAbsorbGroups) {
      const float qv = qs[d];
      const uint4 wv =
          *reinterpret_cast<const uint4*>(wuk + int64_t(d) * kv_lora + col);
      const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 wf = bf16x2_to_float2(w32[j]);
        acc[2 * j] += qv * wf.x;
        acc[2 * j + 1] += qv * wf.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) partial[group][col + j] = acc[j];
  }
  __syncthreads();
  if (group == 0 && col < kv_lora) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float total = partial[0][col + j];
#pragma unroll
      for (int g = 1; g < kAbsorbGroups; ++g) total += partial[g][col + j];
      out[col + j] = float_to_bf16_bits(total);
    }
  }
}

// One block per (row, split, head-group). Thread (h, g) within the block:
// head h, dim group g (contiguous lanes, so the group reduce is shuffles).
//
// smem strides are PADDED to break bank conflicts: with the natural strides
// (row 512 floats, group 64) every thread's address is congruent mod 32
// banks — a 32-way conflict on every access (measured: 85% of this kernel's
// cycles). Padding (16B-aligned for uint4/float4 vector access):
//   gstride    = dslice + 8            (group windows in a padded row)
//   row_stride = groups*gstride - 8 + 16
//   scores_stride = kAttnTile + 1
// A group's logical dims [g*dslice, g*dslice+dslice) live at padded offset
// [g*gstride, g*gstride+dslice); the pad words between groups are never
// read as data. All loads/stores/publish use this one mapping.
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
  const int dslice = kv_lora / groups;
  const int gstride = dslice + 8;
  const int row_stride = groups * gstride + 8;
  const int hl = h - h0;              // head local to this block
  const int d0 = g * gstride;         // group window start (padded row)

  const int cnt = counts[r];
  const int chunk = (cnt + n_split - 1) / n_split;
  const int t_begin = s * chunk;
  const int t_end = min(cnt, t_begin + chunk);

  // Empty split (padding rows, short contexts, counts far below the split
  // granularity): publish zeroed partials and exit before any smem traffic
  // — combine requires initialized values for every (row, split, head).
  if (t_begin >= t_end) {
    const int64_t base = (r * n_split + s) * local_heads;
    for (int dd = threadIdx.x; dd < hpb * kv_lora; dd += blockDim.x) {
      const int hh = dd / kv_lora;
      const int cc = dd % kv_lora;
      c_ws[(base + h0 + hh) * kv_lora + cc] = 0.0f;
    }
    if (threadIdx.x < hpb) {
      m_ws[base + h0 + threadIdx.x] = -INFINITY;
      l_ws[base + h0 + threadIdx.x] = 0.0f;
    }
    return;
  }

  extern __shared__ uint16_t sm16[];
  uint16_t* qt = sm16;  // [hpb * row_stride] padded
  uint16_t* lat = qt + hpb * row_stride;  // [kAttnTile * row_stride] padded
  float* c = reinterpret_cast<float*>(lat + kAttnTile * row_stride);
  float* scores = c + hpb * row_stride;  // [hpb * (kAttnTile + 1)]
  float* l = scores + hpb * (kAttnTile + 1);  // [hpb], single writer
  const int scores_stride = kAttnTile + 1;
  const bool vec = (dslice % 8) == 0;  // vector paths need 8-wide groups

  // The running softmax max is per-lane register state: the group's
  // butterfly reduction leaves identical values in every lane, so no
  // shared m exists to order.
  float m_reg = -INFINITY;

  // Load q_tilde: logical dim gg*dslice+cc -> padded gg*gstride+cc.
  if (vec) {
    const int ds8 = dslice / 8;
    for (int idx = threadIdx.x; idx < hpb * groups * ds8; idx += blockDim.x) {
      const int hh = idx / (groups * ds8);
      const int gidx = (idx / ds8) % groups;
      const int c8 = idx % ds8;
      *reinterpret_cast<uint4*>(qt + hh * row_stride + gidx * gstride +
                                c8 * 8) =
          *reinterpret_cast<const uint4*>(
              &q_tilde[(r * local_heads + h0 + hh) * kv_lora +
                       gidx * dslice + c8 * 8]);
    }
  } else {
    for (int idx = threadIdx.x; idx < hpb * groups * dslice;
         idx += blockDim.x) {
      const int hh = idx / (groups * dslice);
      const int gidx = (idx / dslice) % groups;
      const int cc = idx % dslice;
      qt[hh * row_stride + gidx * gstride + cc] =
          q_tilde[(r * local_heads + h0 + hh) * kv_lora + gidx * dslice + cc];
    }
  }
  for (int i = threadIdx.x; i < hpb; i += blockDim.x) l[i] = 0.0f;
  for (int64_t i = threadIdx.x; i < int64_t(hpb) * row_stride;
       i += blockDim.x)
    c[i] = 0.0f;
  __syncthreads();

  const int32_t req = req_ids[r];
  const int32_t* bt = block_tables + int64_t(req) * blocks_per_request;
  const int32_t* toks = topk + int64_t(r) * topk_stride;

  for (int t0 = t_begin; t0 < t_end; t0 += kAttnTile) {
    const int n = min(kAttnTile, t_end - t0);
    // Latent tile gather with the same padded mapping.
    if (vec) {
      const int ds8 = dslice / 8;
      for (int idx = threadIdx.x; idx < n * groups * ds8; idx += blockDim.x) {
        const int tt = idx / (groups * ds8);
        const int gidx = (idx / ds8) % groups;
        const int c8 = idx % ds8;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys =
            int64_t(blk) * block_tokens + (tok % block_tokens);
        *reinterpret_cast<uint4*>(&lat[int64_t(tt) * row_stride +
                                       gidx * gstride + c8 * 8]) =
            *reinterpret_cast<const uint4*>(
                latent_cache + phys * kv_lora + gidx * dslice + c8 * 8);
      }
    } else {
      for (int i = threadIdx.x; i < n * groups * dslice; i += blockDim.x) {
        const int tt = i / (groups * dslice);
        const int gidx = (i / dslice) % groups;
        const int cc = i % dslice;
        const int64_t tok = toks[t0 + tt];
        const int32_t blk = bt[tok / block_tokens];
        const int64_t phys =
            int64_t(blk) * block_tokens + (tok % block_tokens);
        lat[tt * row_stride + gidx * gstride + cc] =
            latent_cache[phys * kv_lora + gidx * dslice + cc];
      }
    }
    __syncthreads();

    // Scores: each group lane dots its window, then a butterfly leaves the
    // full sum in every lane (keeps the running max per-lane consistent).
    float tile_max = -INFINITY;
    for (int tt = 0; tt < n; ++tt) {
      float partial = 0.0f;
      if (vec) {
        const uint4* q4 =
            reinterpret_cast<const uint4*>(qt + hl * row_stride + d0);
        const uint4* l4 =
            reinterpret_cast<const uint4*>(lat + tt * row_stride + d0);
#pragma unroll 2
        for (int u = 0; u < dslice / 8; ++u)
          partial = dot8_bf16(q4[u], l4[u], partial);
      } else {
        for (int dd = 0; dd < dslice; ++dd)
          partial += bf16_bits_to_float(qt[hl * row_stride + d0 + dd]) *
                     bf16_bits_to_float(lat[tt * row_stride + d0 + dd]);
      }
#pragma unroll
      for (int off = groups / 2; off > 0; off >>= 1)
        partial += __shfl_xor_sync(~0u, partial, off);
      const float score = partial * scale;
      if (g == 0) scores[hl * scores_stride + tt] = score;
      tile_max = fmaxf(tile_max, score);
    }
    __syncthreads();

    // Online softmax update; m in registers, l under its single writer.
    const float m_new = fmaxf(m_reg, tile_max);
    const float rescale = expf(m_reg - m_new);
    for (int dd = 0; dd < dslice; ++dd)
      c[hl * row_stride + d0 + dd] *= rescale;
    if (g == 0) {
      float ladd = 0.0f;
      for (int tt = 0; tt < n; ++tt)
        ladd += expf(scores[hl * scores_stride + tt] - m_new);
      l[hl] = l[hl] * rescale + ladd;
    }
    // c accumulation: probs round to bf16 (pinned; l stays unrounded).
    // float4 RMW over the group window; element order preserved.
    for (int tt = 0; tt < n; ++tt) {
      const float p = bf16_bits_to_float(float_to_bf16_bits(
          expf(scores[hl * scores_stride + tt] - m_new)));
      if (vec) {
        const uint4* l4 =
            reinterpret_cast<const uint4*>(lat + tt * row_stride + d0);
        float* crow = c + hl * row_stride + d0;
#pragma unroll 2
        for (int u = 0; u < dslice / 8; ++u) {
          const uint4 lv = l4[u];
          const uint32_t* l32 = reinterpret_cast<const uint32_t*>(&lv);
          const float2 w0 = bf16x2_to_float2(l32[0]);  // elems 8u+0,1
          const float2 w1 = bf16x2_to_float2(l32[1]);  // 8u+2,3
          const float2 w2 = bf16x2_to_float2(l32[2]);  // 8u+4,5
          const float2 w3 = bf16x2_to_float2(l32[3]);  // 8u+6,7
          float4 c4 = *reinterpret_cast<const float4*>(crow + u * 8);
          c4.x += p * w0.x;
          c4.y += p * w0.y;
          c4.z += p * w1.x;
          c4.w += p * w1.y;
          float4 c4b = *reinterpret_cast<const float4*>(crow + u * 8 + 4);
          c4b.x += p * w2.x;
          c4b.y += p * w2.y;
          c4b.z += p * w3.x;
          c4b.w += p * w3.y;
          *reinterpret_cast<float4*>(crow + u * 8) = c4;
          *reinterpret_cast<float4*>(crow + u * 8 + 4) = c4b;
        }
      } else {
        for (int dd = 0; dd < dslice; ++dd)
          c[hl * row_stride + d0 + dd] +=
              p * bf16_bits_to_float(lat[tt * row_stride + d0 + dd]);
      }
    }
    m_reg = m_new;
    __syncthreads();
  }

  // Publish partials: m/l by the group-0 lane, c gathered window by window
  // back to logical column order.
  const int64_t base_m = (r * n_split + s) * local_heads + h;
  if (g == 0) {
    m_ws[base_m] = m_reg;
    l_ws[base_m] = l[hl];
  }
  // Each head's row is published by its OWN `groups` lanes (strided by
  // groups — the head has only `groups` threads, so a blockDim.x stride
  // would copy 8x4=32 of the row's 512 columns and leave the rest stale).
  float* crow_ws =
      c_ws + (r * n_split + s) * local_heads * kv_lora + int64_t(h) * kv_lora;
  for (int idx = g; idx < groups * dslice; idx += groups) {
    const int gg = idx / dslice;
    const int cc = idx % dslice;
    crow_ws[gg * dslice + cc] = c[hl * row_stride + gg * gstride + cc];
  }
}

__global__ void attn_combine_kernel(const float* m_ws, const float* l_ws,
                                    const float* c_ws, int n_split,
                                    int local_heads, int kv_lora,
                                    float* c_out) {
  // One block per (row, head): the old (row)-block form launched ONE block
  // for single-row decode — a single SM merging 64 heads x 512 dims.
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int64_t base_m = (r * n_split) * local_heads + h;
  float mhat = -INFINITY;
  for (int s = 0; s < n_split; ++s)
    mhat = fmaxf(mhat, m_ws[base_m + int64_t(s) * local_heads]);
  const int64_t total = int64_t(local_heads) * kv_lora;
  for (int64_t i = int64_t(h) * kv_lora + threadIdx.x;
       i < int64_t(h + 1) * kv_lora; i += blockDim.x) {
    if (mhat == -INFINITY) {
      c_out[r * total + i] = 0.0f;  // empty row (padding)
      continue;
    }
    float num = 0.0f;
    float den = 0.0f;
    for (int s = 0; s < n_split; ++s) {
      const float p =
          expf(m_ws[base_m + int64_t(s) * local_heads] - mhat);
      num += p * c_ws[(r * n_split + s) * total + i];
      den += p * l_ws[base_m + int64_t(s) * local_heads];
    }
    c_out[r * total + i] = (den > 0.0f) ? num / den : 0.0f;
  }
}

// vout: out[r, h, d] = <c[r, h, :], W_uv[h][d, :]> over kv_lora, for the v
// output rows of every head. One WARP per output row: the lanes read the
// row's kv_lora bf16 as consecutive uint4s (512 contiguous bytes per warp
// instruction), multiply by the shared c row, and a shuffle tree sums the
// 32 partials. Blocks are kVoutRowsPerBlock warps of one head. The
// previous thread-per-row form streamed each row through one thread's
// L1 (34 us per layer for 4 MB); the reassociation is fp32 rounding
// (accepted 2026-09-02). kv_b interleaved: head h's W_uv rows are
// [h*(nope+v)+nope, (h+1)*(nope+v)).
constexpr int kVoutRowsPerBlock = 8;
constexpr int kVoutThreads = 32 * kVoutRowsPerBlock;

__global__ __launch_bounds__(kVoutThreads) void vout_gemm_kernel(
    const float* c, const uint16_t* kv_b, uint16_t* out, int local_heads,
    int nope, int v, int kv_lora) {
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int d0 = blockIdx.z * kVoutRowsPerBlock;
  const int head_rows = nope + v;
  __shared__ __align__(16) float cs[512];
  for (int cc = threadIdx.x; cc < kv_lora; cc += blockDim.x)
    cs[cc] = c[(r * local_heads + h) * kv_lora + cc];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int d = d0 + warp;
  if (d >= v) return;
  const uint16_t* wuv =
      kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
  const uint4* w4 = reinterpret_cast<const uint4*>(wuv + int64_t(d) * kv_lora);
  float acc = 0.0f;
  for (int u = lane; u < kv_lora / 8; u += 32) {
    const uint4 wv = w4[u];
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
    const float4 cs4 = *reinterpret_cast<const float4*>(cs + u * 8);
    const float4 cs4b = *reinterpret_cast<const float4*>(cs + u * 8 + 4);
    const float2 w0 = bf16x2_to_float2(w32[0]);
    const float2 w1 = bf16x2_to_float2(w32[1]);
    const float2 w2 = bf16x2_to_float2(w32[2]);
    const float2 w3 = bf16x2_to_float2(w32[3]);
    acc += w0.x * cs4.x;
    acc += w0.y * cs4.y;
    acc += w1.x * cs4.z;
    acc += w1.y * cs4.w;
    acc += w2.x * cs4b.x;
    acc += w2.y * cs4b.y;
    acc += w3.x * cs4b.z;
    acc += w3.y * cs4b.w;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) out[(r * local_heads + h) * v + d] = float_to_bf16_bits(acc);
}

}  // namespace

// ---------------------------------------------------------------------
// Shared-memory opt-in
// ---------------------------------------------------------------------

namespace {

// Populated by dsa_prepare_kernel_smem(); the launchers only read them.
int g_select_smem_cap = -1;
int g_attn_smem_cap = -1;

}  // namespace

void dsa_prepare_kernel_smem() {
  if (g_select_smem_cap < 0) {
    int cap = 0;
    DGPP_CUDA_OK(cudaDeviceGetAttribute(
        &cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    g_select_smem_cap = cap - 1024;  // leave room for static smem + alignment
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        select_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_select_smem_cap));
  }
  if (g_attn_smem_cap < 0) {
    int cap = 0;
    DGPP_CUDA_OK(cudaDeviceGetAttribute(
        &cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    g_attn_smem_cap = cap - 1024;
    DGPP_CUDA_OK(cudaFuncSetAttribute(
        attn_partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        g_attn_smem_cap));
  }
}

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
                             const float* ape, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans,
                             int num_requests,
                             const int32_t* block_tables,
                             int blocks_per_request, void* tail,
                             void* index_k, float* index_scale,
                             int pools_per_block, int kpool, int dim,
                             cudaStream_t stream, void* tail_snapshots) {
  if (num_requests <= 0) return;
  kpool_decode_update_kernel<<<unsigned(num_requests), 128,
                               dim * sizeof(float), stream>>>(
      static_cast<const uint16_t*>(k), k_stride,
      static_cast<const uint16_t*>(gate), gate_stride, ape, req_ids, pos,
      req_spans, block_tables, blocks_per_request,
      static_cast<uint16_t*>(tail), static_cast<uint8_t*>(index_k),
      index_scale, pools_per_block, kpool, dim,
      static_cast<uint16_t*>(tail_snapshots));
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ void dsa_zero_padding_rows_kernel(const int64_t* __restrict__ pos,
                                             uint16_t* __restrict__ out,
                                             int hidden) {
  const int t = blockIdx.x;
  if (pos[t] >= 0) return;
  uint16_t* row = out + static_cast<size_t>(t) * hidden;
  for (int h = threadIdx.x; h < hidden; h += blockDim.x) row[h] = 0;
}

void dsa_zero_padding_rows(void* out, const int64_t* pos, int tokens,
                           int hidden, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0) return;
  if (out == nullptr || pos == nullptr)
    throw std::invalid_argument("dsa_zero_padding_rows: null buffer");
  dsa_zero_padding_rows_kernel<<<unsigned(tokens), 256, 0, stream>>>(
      pos, static_cast<uint16_t*>(out), hidden);
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
  dsa_prepare_kernel_smem();
  if (smem > size_t(g_select_smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
  const int blocks = grid_blocks > 0 ? grid_blocks : 48;
  select_counter_reset_kernel<<<1, 32, 0, stream>>>(counter_ws);
  select_decode_kernel<<<blocks, 256, smem, stream>>>(
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
  select_prefill_kernel<<<unsigned(rows), 256, smem, stream>>>(
      dot, dot_stride, w_folded, k_scale, pos, rows, n_pools, heads, select_k,
      kpool, max_selected, topk_out, out_counts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_absorb_q(const void* q, const void* kv_b, void* q_tilde,
                  int64_t rows, int local_heads, int nope, int v, int kv_lora,
                  cudaStream_t stream) {
  if (rows <= 0) return;
  // kv_lora % 8: uint4 weight streaming (8 bf16 per load); 512 % kv_lora:
  // static q smem + column mapping.
  if (nope > 256 || kv_lora % 8 != 0 || 512 % kv_lora != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  dim3 grid{unsigned(rows), unsigned(local_heads)};
  absorb_q_kernel<<<grid, kAbsorbThreads, 0, stream>>>(
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
  // hpb in {1,2,4,8,16}: local_heads must divide across head-group blocks,
  // and blockDim(128)/hpb gives the per-head dim groups (the butterfly
  // reduce handles any power-of-two group count up to blockDim). kv_lora
  // must give every group a nonempty 8-multiple slice: kv_lora >= groups
  // and kv_lora % groups must keep the vec path's 8-wide windows aligned.
  const int hpb = local_heads < 16 ? local_heads : 16;
  if (local_heads % hpb != 0 || 128 % hpb != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora % 8 != 0) DGPP_CUDA_OK(cudaErrorInvalidValue);
  if (kv_lora < 128 / hpb) DGPP_CUDA_OK(cudaErrorInvalidValue);
  // Padded strides (see the kernel's bank-conflict note): must mirror the
  // kernel's gstride/row_stride arithmetic exactly.
  const int groups = 128 / hpb;
  const int dslice = kv_lora / groups;
  const int gstride = dslice + 8;
  const int row_stride = groups * gstride + 8;
  const size_t smem = (size_t(hpb) * row_stride * 2 +
                        kAttnTile * row_stride * 2 + hpb * row_stride * 4 +
                        hpb * (kAttnTile + 1) * 4 + hpb * 4 + 15) &
                      ~size_t(15);
  dsa_prepare_kernel_smem();
  if (smem > size_t(g_attn_smem_cap)) DGPP_CUDA_OK(cudaErrorInvalidValue);
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
  // One block per (row, head): 64+ blocks instead of `rows`.
  dim3 grid{unsigned(rows), unsigned(local_heads)};
  attn_combine_kernel<<<grid, 128, 0, stream>>>(
      m_ws, l_ws, c_ws, n_split, local_heads, kv_lora, c_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsa_vout_gemm(const void* c, const void* kv_b, void* out,
                   int64_t rows, int local_heads, int nope, int v,
                   int kv_lora, cudaStream_t stream) {
  if (rows <= 0) return;
  // kv_lora % 8: uint4 weight streaming; <= 512: static c smem.
  if (kv_lora > 512 || kv_lora % 8 != 0)
    DGPP_CUDA_OK(cudaErrorInvalidValue);
  dim3 grid{unsigned(rows), unsigned(local_heads),
            unsigned((v + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock)};
  vout_gemm_kernel<<<grid, kVoutThreads, 0, stream>>>(
      static_cast<const float*>(c), static_cast<const uint16_t*>(kv_b),
      static_cast<uint16_t*>(out), local_heads, nope, v, kv_lora);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
