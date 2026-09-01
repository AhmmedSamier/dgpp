#pragma once
// Shared shape of the decode GEMV cores (fp8_gemv.cuh, bf16_gemv.cu): one
// block = kWarps warps = kWarps weight rows; up to kMaxRows activation rows
// staged bf16 in dynamic shared memory. Both cores are bandwidth kernels —
// the shape exists to keep bytes in flight (see fp8_gemv.cuh for the why).
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dgpp {
namespace gemv {

constexpr int kWarps = 8;             // weight rows per block
constexpr int kThreads = kWarps * 32;
constexpr int kMaxRows = 4;           // activation rows per pass (smem: rows * k * 2B)
constexpr int kChunkBytes = 16;       // one uint4 per lane per step
constexpr int kWarpSpan = 32 * kChunkBytes;  // 512 weight bytes per warp step
constexpr int kBatch = 8;             // chunks in flight per lane

// Shared-memory bytes the caller must provide for `rows` activation rows.
__host__ __device__ constexpr size_t smem_bytes(int rows, int k) {
  return static_cast<size_t>(rows) * static_cast<size_t>(k) * 2;
}

// The default dynamic-smem ceiling (no cudaFuncSetAttribute opt-in — a
// context mutation the graph-capture paths must never make). Shapes past
// it (world-1 o_proj at m>=2, dense down at m>=2) keep their GEMM path.
constexpr size_t kMaxSmemBytes = 48 * 1024;
__host__ inline bool smem_fits(int rows, int k) {
  return smem_bytes(rows, k) <= kMaxSmemBytes;
}

__host__ inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// Stage `kRows` bf16 activation rows (row stride x_stride elements) into
// shared memory as [kRows][k]; the whole block participates, the caller
// syncs. 16-byte vectors when the source allows, scalar otherwise.
template <int kRows>
__device__ __forceinline__ void stage_activations(const uint16_t* __restrict__ x,
                                                  size_t x_stride, int k,
                                                  uint16_t* __restrict__ sx) {
  const bool vec = ((reinterpret_cast<uintptr_t>(x) & 15u) == 0) &&
                   ((x_stride * 2) % 16 == 0) && (k % 8 == 0);
  if (vec) {
    const int vecs_per_row = k / 8;
    for (int i = threadIdx.x; i < kRows * vecs_per_row; i += blockDim.x) {
      const int r = i / vecs_per_row, c = i - r * vecs_per_row;
      reinterpret_cast<uint4*>(sx + static_cast<size_t>(r) * k)[c] =
          reinterpret_cast<const uint4*>(x + static_cast<size_t>(r) * x_stride)[c];
    }
  } else {
    for (int i = threadIdx.x; i < kRows * k; i += blockDim.x) {
      const int r = i / k, c = i - r * k;
      sx[static_cast<size_t>(r) * k + c] = x[static_cast<size_t>(r) * x_stride + c];
    }
  }
}

// Fixed xor-shuffle tree: deterministic, launch-shape independent.
template <int kRows>
__device__ __forceinline__ void warp_reduce(float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
  }
}

}  // namespace gemv
}  // namespace dgpp
