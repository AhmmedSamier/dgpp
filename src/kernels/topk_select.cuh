#pragma once
// Block-cooperative exact top-k over 64-bit composite keys (the DSA decode
// select's machinery, lifted so the sampler's local top-k shares it):
//   key = (~sortable_f32(logit) << idx_bits) | idx
// makes the SMALLEST key the highest logit with ties broken toward the
// lower index — sample::candidate_before, as an integer order. The
// streaming selection keeps the running best select_k keys in shared
// memory (split hi/lo 32-bit arrays: one bank per thread at every bitonic
// stride) and returns them sorted ascending, i.e. in canonical order.
// Deterministic on any correct implementation: the result is a set under a
// total order, so no reduction shape can change it.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kSelectTile = 2048;  // keys per selection tile (>= 2*select_k)

__device__ inline uint32_t sortable_f32_dev(float f) {
  uint32_t u = __float_as_uint(f);
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

// Bitonic networks over 64-bit composite keys stored as SPLIT 32-bit arrays
// (hi/lo). A u64 key spans two 4-byte banks, so a u64 array with XOR indexing
// bank-conflicts ~16 ways in the sort stages (measured: the merge phase was
// 97% of the select kernel). Split arrays give every thread its own bank for
// every stride: stride >= 32 leaves i%32 untouched; smaller strides XOR-
// permute the banks. Same keys, same comparisons, identical selection.
__device__ inline bool key_less(uint32_t a_hi, uint32_t a_lo, uint32_t b_hi,
                                uint32_t b_lo) {
  return (a_hi < b_hi) || (a_hi == b_hi && a_lo < b_lo);
}

__device__ inline void cx_split(uint32_t* hi, uint32_t* lo, int i, int j,
                                bool asc) {
  const uint32_t a_hi = hi[i], a_lo = lo[i];
  const uint32_t b_hi = hi[j], b_lo = lo[j];
  const bool swap = asc ? key_less(b_hi, b_lo, a_hi, a_lo)
                        : key_less(a_hi, a_lo, b_hi, b_lo);
  if (swap) {
    hi[i] = b_hi; lo[i] = b_lo;
    hi[j] = a_hi; lo[j] = a_lo;
  }
}

// Ascending bitonic sort of n keys (power of two).
__device__ inline void bitonic_sort_asc(uint32_t* hi, uint32_t* lo, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) cx_split(hi, lo, i, j, (i & size) == 0);
      }
      __syncthreads();
    }
  }
}

// Ascending bitonic merge of a bitonic sequence of n (power of two).
__device__ inline void bitonic_merge_asc(uint32_t* hi, uint32_t* lo, int n) {
  for (int stride = n >> 1; stride > 0; stride >>= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const int j = i ^ stride;
      if (j > i) cx_split(hi, lo, i, j, true);
    }
    __syncthreads();
  }
}

// Streaming top-select_k over key_fn(pool) -> composite key, restricted to
// pools [lo, hi). best_hi/best_lo (smem, [select_k]) must be pre-initialized
// to kKeyMax; on return they hold the smallest select_k keys seen, sorted
// ascending. tile arrays (smem) need kSelectTile entries each;
// kSelectTile >= 2*select_k required.
template <typename KeyFn>
__device__ inline void select_topk_stream(KeyFn key_fn, int64_t lo, int64_t hi,
                                          uint32_t* best_hi, uint32_t* best_lo,
                                          uint32_t* tile_hi, uint32_t* tile_lo,
                                          int select_k) {
  const int nthreads = blockDim.x;
  const int warp = threadIdx.x >> 5;
  const int nwarp = nthreads >> 5;
  for (int64_t base = lo; base < hi; base += kSelectTile) {
    const int n = int(min((int64_t)kSelectTile, hi - base));
    for (int p = threadIdx.x; p < kSelectTile; p += nthreads) {
      tile_hi[p] = 0xFFFFFFFFu;  // kKeyMax padding; [0, n) overwritten below
      tile_lo[p] = 0xFFFFFFFFu;
    }
    __syncthreads();
    if constexpr (KeyFn::kWarpCooperative) {
      // Warp-cooperative keys (pool logits): one warp evaluates one pool;
      // lane 0 publishes. All lanes load the same cache row — broadcast.
      for (int p = warp; p < n; p += nwarp) {
        const uint64_t key = key_fn(base + p);
        if ((threadIdx.x & 31) == 0) {
          tile_hi[p] = uint32_t(key >> 32);
          tile_lo[p] = uint32_t(key);
        }
      }
    } else {
      // Per-thread keys (merge partials): thread-strided with unrolled
      // independent loads — the warp-strided form serialized 512 dependent
      // global loads per warp and was 57% of the select kernel.
      for (int p = threadIdx.x; p < n; p += nthreads) {
        const uint64_t key = key_fn(base + p);
        tile_hi[p] = uint32_t(key >> 32);
        tile_lo[p] = uint32_t(key);
      }
    }
    __syncthreads();
    bitonic_sort_asc(tile_hi, tile_lo, kSelectTile);
    // Merge best (asc) with the tile's smallest select_k reversed (desc):
    // [asc][desc] is bitonic, so a 2*select_k bitonic merge yields the new
    // best. Elements beyond the tile's first select_k are dominated by
    // select_k better tile keys and can never enter the global top-k.
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      tile_hi[select_k + i] = tile_hi[select_k - 1 - i];
      tile_lo[select_k + i] = tile_lo[select_k - 1 - i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      tile_hi[i] = best_hi[i];
      tile_lo[i] = best_lo[i];
    }
    __syncthreads();
    bitonic_merge_asc(tile_hi, tile_lo, 2 * select_k);
    for (int i = threadIdx.x; i < select_k; i += nthreads) {
      best_hi[i] = tile_hi[i];
      best_lo[i] = tile_lo[i];
    }
    __syncthreads();
  }
}

}  // namespace dgpp
