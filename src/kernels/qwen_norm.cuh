#pragma once
// The Qwen norms' block reduction (2026-09-09, factored out of qwen_norm.cu
// so the GR site's norm-staged GEMV runs the same reduction — one order,
// one result, whichever kernel computes it).
#include <cstdint>

#include "common/dtypes.hpp"

namespace dgpp {
namespace qwen_norm {

// Block-wide sum of squares in a fixed order: each thread's strided fma
// chain, a warp butterfly, then the warps in index order — the same on
// every launch, so the norms are deterministic.
__device__ __forceinline__ float block_sum_squares(const uint16_t* __restrict__ xr,
                                                   int dim, float* __restrict__ staged,
                                                   float* __restrict__ warp_sums) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  float partial = 0.0f;
  // The thread's strided chain in i order; the loads issued four ahead of
  // the chain (2026-09-09: each was an L2 round trip on the chain's
  // critical path) — the same fmaf sequence, so the sum is unchanged.
  int i = tid;
  for (; i + 3 * nthreads < dim; i += 4 * nthreads) {
    const uint16_t b0 = xr[i], b1 = xr[i + nthreads], b2 = xr[i + 2 * nthreads],
                   b3 = xr[i + 3 * nthreads];
    const float v0 = bf16_bits_to_float(b0), v1 = bf16_bits_to_float(b1),
                v2 = bf16_bits_to_float(b2), v3 = bf16_bits_to_float(b3);
    staged[i] = v0;
    staged[i + nthreads] = v1;
    staged[i + 2 * nthreads] = v2;
    staged[i + 3 * nthreads] = v3;
    partial = fmaf(v0, v0, partial);
    partial = fmaf(v1, v1, partial);
    partial = fmaf(v2, v2, partial);
    partial = fmaf(v3, v3, partial);
  }
  for (; i < dim; i += nthreads) {
    const float v = bf16_bits_to_float(xr[i]);
    staged[i] = v;
    partial = fmaf(v, v, partial);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = partial;
  __syncthreads();
  float total = 0.0f;
  const int nwarps = (nthreads + 31) / 32;
  for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
  return total;
}

}  // namespace qwen_norm
}  // namespace dgpp
