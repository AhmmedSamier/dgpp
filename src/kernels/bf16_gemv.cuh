#pragma once
// The bf16 GEMV's row chain (2026-09-09, factored out of bf16_gemv.cu so
// the fused decode kernels — the Qwen shared expert's gate/up/swiglu and
// down/accumulate/round, the GR site's norm-staged GEMVs — run the SAME
// code: a row's chain is the same sequence of FMAs on the same values
// whichever launcher issued its loads, which is what keeps every fusion
// bitwise the plain launch).
#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace bf16_gemv {

// bf16 x bf16 products are exact in fp32 (8-bit x 8-bit significands), so
// the chain's only roundings are the FMA accumulations.
__device__ __forceinline__ void fma_pair(uint32_t w2, uint32_t x2, float& acc) {
  const float w0 = bf16_bits_to_float(static_cast<uint16_t>(w2 & 0xFFFFu));
  const float w1 = bf16_bits_to_float(static_cast<uint16_t>(w2 >> 16));
  const float x0 = bf16_bits_to_float(static_cast<uint16_t>(x2 & 0xFFFFu));
  const float x1 = bf16_bits_to_float(static_cast<uint16_t>(x2 >> 16));
  acc = __fmaf_rn(w0, x0, acc);
  acc = __fmaf_rn(w1, x1, acc);
}

// One warp, one bf16 weight row (k elements, 16B aligned), kRows staged
// activation rows: acc[r] = dot(w_row, sx[r]). Chunk = 8 bf16 (16 bytes).
template <int kRows>
__device__ __forceinline__ void row_dots(const uint16_t* __restrict__ w_row,
                                         const uint16_t* __restrict__ sx, int k,
                                         int lane, float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  constexpr int kChunkElems = gemv::kChunkBytes / 2;  // 8
  constexpr int kSpanElems = gemv::kWarpSpan / 2;     // 256
  const int lane_off = lane * kChunkElems;
  for (int base = 0; base < k; base += kSpanElems * gemv::kBatch) {
    uint4 w[gemv::kBatch];
#pragma unroll
    for (int b = 0; b < gemv::kBatch; ++b) {
      const int c0 = base + b * kSpanElems + lane_off;
      w[b] = (c0 < k) ? *reinterpret_cast<const uint4*>(w_row + c0)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int b = 0; b < gemv::kBatch; ++b) {
      const int c0 = base + b * kSpanElems + lane_off;
      if (c0 >= k) break;  // per lane: its later chunks are past k too
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 xv = *reinterpret_cast<const uint4*>(
            sx + static_cast<size_t>(r) * k + c0);
        fma_pair(w[b].x, xv.x, acc[r]);
        fma_pair(w[b].y, xv.y, acc[r]);
        fma_pair(w[b].z, xv.z, acc[r]);
        fma_pair(w[b].w, xv.w, acc[r]);
      }
    }
  }
  gemv::warp_reduce<kRows>(acc);
}

}  // namespace bf16_gemv
}  // namespace dgpp
