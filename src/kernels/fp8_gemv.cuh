#pragma once
// The block-scaled fp8 GEMV core (2026-09-01, M6 Stage 2 round 2).
//
// WHY THIS EXISTS: the decode step's expert matrices are read at m=1, and
// the mma-tile GEMM (scale_gemm.cu) rides that shape badly — 1-byte weight
// loads, two block barriers per 32-wide k stage, 128 serialized stages per
// 64-row tile: latency-bound at ~50 GB/s on a 233 GB/s part (the T=1
// profile: 47.9 ms of a 190 ms step for 2.38 GB of weights). A GEMV is a
// bandwidth problem, so this core is shaped for bytes in flight: one WARP
// per weight row, each lane owning 16-byte fp8 chunks strided 512 bytes
// apart (a warp load instruction covers four full 128-byte lines), a whole
// batch of chunks issued before any is consumed, no barrier in the k loop.
//
// NUMERICS: the dequantization is the dequant bridge's, op for op —
// bf16(e4m3(w) * scale) — so the weight VALUES are identical to the tile
// path's; the fp32 ACCUMULATION ORDER is not (per-lane sequential chain in k
// order, then a fixed xor-shuffle tree), so outputs differ from the tile
// path at the fp32-rounding level (well inside the oracle ULP budgets) and
// are deterministic. Every activation row's chain is the same sequence of
// FMAs on the same values whatever kRows is, which is what lets the
// slot path (kRows=1) and scale_gemm at m<=kMaxRows agree bit for bit.
//
// CONTRACT: k % 16 == 0 and 16-byte-aligned weight rows (a 16B chunk then
// lies inside ONE 128-wide scale block, 128 % 16 == 0). The activation rows
// are staged bf16 in shared memory by the caller (kRows * k elements).
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace fp8_gemv {

using gemv::kBatch;
using gemv::kChunkBytes;
using gemv::kMaxRows;
using gemv::kThreads;
using gemv::kWarps;
using gemv::kWarpSpan;
using gemv::smem_bytes;
using gemv::stage_activations;

// k a multiple of 16 (a chunk sits inside one 128-wide scale block) and a
// 16-byte-aligned payload.
__host__ inline bool shape_ok(const void* payload, int rows, int k) {
  return k > 0 && (k % kChunkBytes) == 0 && gemv::aligned16(payload) &&
         gemv::smem_fits(rows, k);
}

// e4m3 -> f32, exact (every finite e4m3 value is an fp16 value; the two
// NaN codes come back NaN). Matches fp8_e4m3_bits_to_float on every code
// except the NaN sign, which no consumer observes (NaN-ness propagates).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t packed) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
  return __half22float2(__half2(h));
}

// The dequant bridge's rounding: bf16(w * s), back to f32 for the FMA.
__device__ __forceinline__ float dequant(float w, float s) {
  return bf16_bits_to_float(float_to_bf16_bits(w * s));
}

// One 16-byte weight chunk (columns [c0, c0+16) of a row, scale s) into
// kRows accumulators — the core's arithmetic, factored (2026-09-06) so
// row_dots and block_rows_multi run the same code: a row's chain is the
// same sequence of FMAs on the same values whichever launcher issued its
// loads, which is what keeps the multi-row down projection bitwise.
template <int kRows>
__device__ __forceinline__ void consume_chunk(const uint4& wchunk, float s,
                                              const uint16_t* __restrict__ sx,
                                              int k, int c0,
                                              float (&acc)[kRows]) {
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
  // 16 activation elements per row for this chunk: two uint4 of bf16.
  uint4 xa[kRows], xb[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4* xp = reinterpret_cast<const uint4*>(
        sx + static_cast<size_t>(r) * k + c0);
    xa[r] = xp[0];
    xb[r] = xp[1];
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t word = words[q];
    // Weights q*4 .. q*4+3 of the chunk.
    const float2 w01 = e4m3x2_to_float2(static_cast<uint16_t>(word & 0xFFFFu));
    const float2 w23 = e4m3x2_to_float2(static_cast<uint16_t>(word >> 16));
    const float d0 = dequant(w01.x, s), d1 = dequant(w01.y, s);
    const float d2 = dequant(w23.x, s), d3 = dequant(w23.y, s);
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      // Activation elements q*4..q*4+3: words q*2, q*2+1 of the 16-elem run.
      const uint32_t xw0 = (q < 2) ? ((q == 0) ? xa[r].x : xa[r].z)
                                   : ((q == 2) ? xb[r].x : xb[r].z);
      const uint32_t xw1 = (q < 2) ? ((q == 0) ? xa[r].y : xa[r].w)
                                   : ((q == 2) ? xb[r].y : xb[r].w);
      const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw0 & 0xFFFFu));
      const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw0 >> 16));
      const float x2 = bf16_bits_to_float(static_cast<uint16_t>(xw1 & 0xFFFFu));
      const float x3 = bf16_bits_to_float(static_cast<uint16_t>(xw1 >> 16));
      acc[r] = __fmaf_rn(d0, x0, acc[r]);
      acc[r] = __fmaf_rn(d1, x1, acc[r]);
      acc[r] = __fmaf_rn(d2, x2, acc[r]);
      acc[r] = __fmaf_rn(d3, x3, acc[r]);
    }
  }
}

// One warp, one weight row, kRows activation rows: acc[r] = dot(w_row, sx[r]).
// Every lane returns the full reduced dot for each row.
//   w_row     fp8 row (k bytes, 16B aligned)
//   scale_row this row's scale block row ([ceil(k/128)] f32)
//   sx        staged activations [kRows][k] bf16
template <int kRows>
__device__ __forceinline__ void row_dots(const uint8_t* __restrict__ w_row,
                                         const float* __restrict__ scale_row,
                                         const uint16_t* __restrict__ sx, int k,
                                         int lane, float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

  const int lane_off = lane * kChunkBytes;
  for (int base = 0; base < k; base += kWarpSpan * kBatch) {
    // Issue the whole batch before consuming any of it: the bytes in flight
    // are the point of this kernel.
    uint4 w[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int c0 = base + b * kWarpSpan + lane_off;
      w[b] = (c0 < k) ? *reinterpret_cast<const uint4*>(w_row + c0)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int c0 = base + b * kWarpSpan + lane_off;
      if (c0 >= k) break;  // per lane: each lane only consumes its own chunks
      consume_chunk<kRows>(w[b], scale_row[c0 >> 7], sx, k, c0, acc);
    }
  }
  gemv::warp_reduce<kRows>(acc);
}

// The output policy of one dot: bf16 (the activation dtype — one rounding
// here) or raw fp32 (the MoE down projection's partials, which the fp32
// accumulation chain consumes unrounded; see glm_moe.cu).
__device__ __forceinline__ void store_dot(uint16_t* out, float v) {
  *out = float_to_bf16_bits(v);
}
__device__ __forceinline__ void store_dot(float* out, float v) { *out = v; }

// The block body shared by the dense and slot launchers: this block's kWarps
// rows [n0, n0+kWarps) of an [n, k] fp8 matrix against the staged rows.
//   out[r * out_stride + col] = store_dot(acc[r])  (bf16 or fp32 by OutT)
template <int kRows, typename OutT>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w,
                                           const float* __restrict__ scales,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n, int k,
                                           OutT* __restrict__ out,
                                           size_t out_stride) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = n0 + warp;
  if (row >= n) return;
  const int scale_cols = (k + 127) / 128;
  const float* scale_row = scales + static_cast<size_t>(row / 128) * scale_cols;
  float acc[kRows];
  row_dots<kRows>(w + static_cast<size_t>(row) * k, scale_row, sx, k, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRows; ++r)
      store_dot(out + static_cast<size_t>(r) * out_stride + row, acc[r]);
  }
}

// The block body with R weight rows per warp (2026-09-06): rows [n0, n0 +
// kWarps * R), warp w owning rows n0 + w*R .. + R-1. The R rows' chunks
// form one stream per lane issued kBatch at a time, so a short row (the
// sliced down projection's k = 512 is ONE chunk per lane) no longer
// leaves a warp with a single load in flight — the measured cause of the
// down kernel's 224 GB/s against gate_up's 295. Each row's chunks are
// consumed in k order into its own accumulator through consume_chunk, so
// every row's result is bitwise block_rows'.
template <int kRows, int R, typename OutT>
__device__ __forceinline__ void block_rows_multi(
    const uint8_t* __restrict__ w, const float* __restrict__ scales,
    const uint16_t* __restrict__ sx, int n0, int n, int k,
    OutT* __restrict__ out, size_t out_stride) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row0 = n0 + warp * R;
  if (row0 >= n) return;
  const int scale_cols = (k + 127) / 128;
  const int lane_off = lane * kChunkBytes;
  const int chunks = (k + kWarpSpan - 1) / kWarpSpan;  // per row per lane
  float acc[R][kRows];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[r][a] = 0.f;
  const int total = R * chunks;
  for (int cb = 0; cb < total; cb += kBatch) {
    uint4 wv[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int idx = cb + b;
      const int r = idx / chunks, c = idx - r * chunks;
      const int c0 = c * kWarpSpan + lane_off;
      const bool ok = idx < total && row0 + r < n && c0 < k;
      wv[b] = ok ? *reinterpret_cast<const uint4*>(
                       w + static_cast<size_t>(row0 + r) * k + c0)
                 : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int idx = cb + b;
      const int r = idx / chunks, c = idx - r * chunks;
      const int c0 = c * kWarpSpan + lane_off;
      if (idx >= total || row0 + r >= n || c0 >= k) continue;
      const float s =
          scales[static_cast<size_t>((row0 + r) / 128) * scale_cols + (c0 >> 7)];
      consume_chunk<kRows>(wv[b], s, sx, k, c0, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    if (row0 + r >= n) break;  // uniform across the warp
    gemv::warp_reduce<kRows>(acc[r]);
    if (lane == 0) {
#pragma unroll
      for (int a = 0; a < kRows; ++a)
        store_dot(out + static_cast<size_t>(a) * out_stride + row0 + r, acc[r][a]);
    }
  }
}

}  // namespace fp8_gemv
}  // namespace dgpp
