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

namespace dgpp {
namespace fp8_gemv {

constexpr int kWarps = 8;                 // rows per block
constexpr int kThreads = kWarps * 32;
constexpr int kChunkBytes = 16;           // one uint4 of fp8 per lane per step
constexpr int kWarpSpan = 32 * kChunkBytes;  // 512 weight bytes per warp step
constexpr int kBatch = 8;                 // chunks in flight per lane (k=4096 in one batch)
constexpr int kMaxRows = 4;               // activation rows per pass (smem: rows * k * 2B)

// Shared-memory bytes the caller must provide for `rows` activation rows.
__host__ __device__ constexpr size_t smem_bytes(int rows, int k) {
  return static_cast<size_t>(rows) * static_cast<size_t>(k) * 2;
}

__host__ inline bool shape_ok(const void* payload, int k) {
  return k > 0 && (k % kChunkBytes) == 0 &&
         (reinterpret_cast<uintptr_t>(payload) & 15u) == 0;
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

// Stage `rows` bf16 activation rows (row stride x_stride elements) into
// shared memory as [rows][k]; the whole block participates, the caller
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
      if (c0 >= k) break;  // uniform across the warp beyond the last chunk? No:
                           // lanes differ; the break is per lane and correct
                           // (each lane only consumes its own chunks).
      const float s = scale_row[c0 >> 7];
      const uint32_t words[4] = {w[b].x, w[b].y, w[b].z, w[b].w};
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
  }
  // Fixed xor-shuffle tree: deterministic, launch-shape independent.
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
  }
}

// The block body shared by the dense and slot launchers: this block's kWarps
// rows [n0, n0+kWarps) of an [n, k] fp8 matrix against the staged rows.
//   out[r * out_stride + col] = bf16(acc[r])
template <int kRows>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w,
                                           const float* __restrict__ scales,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n, int k,
                                           uint16_t* __restrict__ out,
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
      out[static_cast<size_t>(r) * out_stride + row] = float_to_bf16_bits(acc[r]);
  }
}

}  // namespace fp8_gemv
}  // namespace dgpp
