#pragma once
// The NVFP4 GEMV core (2026-09-08, docs/nvfp4_plan.md §3): the decode
// step's routed experts at m = 1..4 activation rows.
//
// SHAPE: the fp8 core's (fp8_gemv.cuh) — a bandwidth kernel shaped for
// bytes in flight: 16-byte weight chunks per lane, a whole batch issued
// before any is consumed, no barrier in the k loop, a fixed xor-shuffle
// tree. A chunk holds 32 e2m1 codes (two per byte, low nibble = even
// element) and spans two 16-wide scale blocks. The row geometry is a
// COMPILE-TIME function of K (one of 32, 64, ..., 4096): a lane owns up to
// four chunks of a row, a row is covered by K/128 lanes (32 at K = 4096,
// four at the world-4 down projection's K = 512), and a warp advances
// rows_per_step = 32 / lanes_per_row rows per step, two steps per warp —
// so every lane keeps eight 16-byte loads in flight whatever the row
// length, and every loop below unrolls with constant indices (the first
// draft indexed its accumulators by a lane-dependent runtime value, which
// put them in local memory: 14 GB/s). Which lane owns which chunk is a
// function of K alone, and every row's chain is the same sequence of FMAs
// whatever kRows or which launcher issued it — the property the slot path
// (kRows = 1) and the host grouped path (kRows <= 4) are pinned on, bitwise.
//
// NUMERICS (the plan's §3.1): an e2m1 code has <= 2 significant bits and an
// e4m3 scale <= 4, so the scaled weight fp4 * s is EXACT in fp32 (and in
// bf16); its product with a bf16 activation is exact in fp32; the
// accumulation is fp32 FMA in a fixed order; the one inexact operation of
// the format — the division by the tensor's global scale — is applied once
// to the finished dot. Nothing is rounded to bf16 before the epilogue.
//
// DECODE: e2m1 -> fp16 by bit placement, no table and no arch-specific
// instruction: the code's three magnitude bits (e1 e0 m) land in fp16 bits
// 11..9 (the low two exponent bits and the top mantissa bit), the sign in
// bit 15, and the value read back is the e2m1 value times 2^-14 for every
// code, subnormals included (e = 0: fp16 subnormal m * 2^-15 = m/2 *
// 2^-14). The 2^14 folds into the block scale (s * 16384, exact), so the
// per-code cost is the bit placement, one f16 -> f32 conversion and the
// multiply — no fp16 arithmetic anywhere.
//
// CONTRACT: K in {32, 64, 128, 256, 512, 1024, 2048, 4096}; payload rows
// 16-byte aligned. Scales are e4m3 bytes [n, K/16] row-major; NaN scale
// codes (0x7F/0xFF) propagate as NaN.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace fp4_gemv {

using gemv::kChunkBytes;
using gemv::kMaxRows;
using gemv::kThreads;
using gemv::kWarps;
using gemv::smem_bytes;
using gemv::stage_activations;

constexpr int kCodesPerChunk = 2 * kChunkBytes;   // 32
constexpr int kGroup = 16;                        // codes per e4m3 scale
constexpr int kMaxChunksPerLane = 4;
constexpr int kSteps = 2;                         // row steps per warp
constexpr int kMaxK = 4096;

__host__ __device__ constexpr bool k_supported(int k) {
  return k >= 32 && k <= kMaxK && (k & (k - 1)) == 0;
}

__host__ inline bool shape_ok(const void* payload, int k) {
  return k_supported(k) && gemv::aligned16(payload);
}

// The compile-time row geometry.
template <int K>
struct Geom {
  static_assert(k_supported(K), "fp4_gemv: K must be a power of two in [32, 4096]");
  static constexpr int row_chunks = K / kCodesPerChunk;
  static constexpr int chunks = row_chunks < kMaxChunksPerLane ? row_chunks : kMaxChunksPerLane;
  static constexpr int lanes_per_row = row_chunks / chunks;   // 1 .. 32
  static constexpr int rows_per_step = 32 / lanes_per_row;
  static constexpr int rows_per_warp = kSteps * rows_per_step;
  static constexpr int rows_per_block = kWarps * rows_per_warp;
  static constexpr int row_bytes = K / 2;
  static constexpr int scale_cols = K / kGroup;
  static constexpr int loads = kSteps * chunks;                // per lane, in flight
};

// The runtime twins for launch arithmetic (the host dispatches on k).
__host__ __device__ constexpr int rows_per_step_of(int k) {
  const int rc = k / kCodesPerChunk;
  const int ch = rc < kMaxChunksPerLane ? rc : kMaxChunksPerLane;
  return 32 / (rc / ch);
}
__host__ __device__ constexpr int rows_per_warp_of(int k) {
  return kSteps * rows_per_step_of(k);
}
__host__ __device__ constexpr int rows_per_block_of(int k) {
  return kWarps * rows_per_warp_of(k);
}

// e4m3 -> f32, exact (the fp8 core's conversion), for the two scales a
// chunk spans, pre-multiplied by 2^14 (exact) — the decode's correction.
__device__ __forceinline__ float2 scales_x16384(uint16_t packed) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed), __NV_E4M3);
  float2 s = __half22float2(__half2(h));
  s.x *= 16384.f;
  s.y *= 16384.f;
  return s;
}

// One e2m1 code -> its value times 2^-14, as fp32, exactly.
__device__ __forceinline__ float e2m1_scaled(uint32_t code) {
  const uint32_t bits = ((code & 7u) << 9) | ((code & 8u) << 12);
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// One 16-byte chunk (codes [e0, e0+32) of a row, scales s01 = {s(e0), s(e0+16)}
// times 2^14) into kRows accumulators. Element order within the chunk is
// the storage order: byte j of word q holds elements e0+8q+2j (low nibble)
// and e0+8q+2j+1.
template <int kRows>
__device__ __forceinline__ void consume_chunk(const uint4& wchunk, float2 s01,
                                              const uint16_t* __restrict__ sx,
                                              int k, int e0,
                                              float (&acc)[kRows]) {
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
  uint4 xv[kRows][4];  // 32 activation elements per row: four uint4
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4* xp = reinterpret_cast<const uint4*>(
        sx + static_cast<size_t>(r) * k + e0);
#pragma unroll
    for (int q = 0; q < 4; ++q) xv[r][q] = xp[q];
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t word = words[q];
    const float s = q < 2 ? s01.x : s01.y;  // codes 0-15 | 16-31
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (word >> (8 * j)) & 0xFFu;
      const float w0 = e2m1_scaled(byte & 0xFu) * s;
      const float w1 = e2m1_scaled(byte >> 4) * s;
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint32_t xw = j == 0 ? xv[r][q].x : j == 1 ? xv[r][q].y
                          : j == 2 ? xv[r][q].z : xv[r][q].w;
        const float x0 = bf16_bits_to_float(static_cast<uint16_t>(xw & 0xFFFFu));
        const float x1 = bf16_bits_to_float(static_cast<uint16_t>(xw >> 16));
        acc[r] = __fmaf_rn(w0, x0, acc[r]);
        acc[r] = __fmaf_rn(w1, x1, acc[r]);
      }
    }
  }
}

// Reduce kRows accumulators across a lane group of Lanes lanes (a power of
// two, aligned): the fixed xor tree, launch-shape independent.
template <int kRows, int Lanes>
__device__ __forceinline__ void group_reduce(float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int off = Lanes >> 1; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
}

// The output policy of one dot (the fp8 core's): bf16 or raw fp32.
__device__ __forceinline__ void store_dot(uint16_t* out, float v) {
  *out = float_to_bf16_bits(v);
}
__device__ __forceinline__ void store_dot(float* out, float v) { *out = v; }

// The dots of a warp's rows_per_warp rows (rows [n0 + warp*rows_per_warp,
// +rows_per_warp) of an [n, K] matrix) against kRows staged activation
// rows, UNDIVIDED by the global scale and reduced across each row's lane
// group. acc[st][a] is this lane's row of step st (row_base + st *
// rows_per_step + group); lane `lig == 0` of the group holds the reduced
// value. Each row's chunks are consumed in k order into its own
// accumulator, so a row's result is the same whatever kRows or the
// launcher. No warp-uniform early return: callers may barrier after this.
template <int K, int kRows>
__device__ __forceinline__ void warp_row_dots(const uint8_t* __restrict__ w,
                                              const uint8_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx,
                                              int n0, int n,
                                              float (&acc)[kSteps][kRows]) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[st][a] = 0.f;

  // Issue every load of the warp's rows (kSteps x chunks per lane), then
  // consume them in (step, chunk) order — chunk order is k order per row.
  uint4 wv[G::loads];
  uint16_t sv[G::loads];
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < G::chunks; ++c) {
      const int boff = (c * G::lanes_per_row + lig) * kChunkBytes;
      const int i = st * G::chunks + c;
      wv[i] = ok ? *reinterpret_cast<const uint4*>(
                       w + static_cast<size_t>(row) * G::row_bytes + boff)
                 : make_uint4(0u, 0u, 0u, 0u);
      sv[i] = ok ? *reinterpret_cast<const uint16_t*>(
                       scales + static_cast<size_t>(row) * G::scale_cols + boff / 8)
                 : static_cast<uint16_t>(0);
    }
  }
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    if (row >= n) continue;  // per lane; no barrier inside
#pragma unroll
    for (int c = 0; c < G::chunks; ++c) {
      const int e0 = (c * G::lanes_per_row + lig) * kCodesPerChunk;
      const int i = st * G::chunks + c;
      consume_chunk<kRows>(wv[i], scales_x16384(sv[i]), sx, K, e0, acc[st]);
    }
  }
#pragma unroll
  for (int st = 0; st < kSteps; ++st) group_reduce<kRows, G::lanes_per_row>(acc[st]);
}

// Which lane owns the reduced dot of the warp's step-st row, and that
// row's index: lane `lig == 0` of its group.
template <int K>
__device__ __forceinline__ int owned_row(int n0, int st, bool& mine) {
  using G = Geom<K>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  mine = (lane % G::lanes_per_row) == 0;
  return n0 + warp * G::rows_per_warp + st * G::rows_per_step + lane / G::lanes_per_row;
}

// The block body: kWarps warps x rows_per_warp rows each of an [n, K]
// NVFP4 matrix (payload [n, K/2], scales [n, K/16], global scale g):
//   out[a * out_stride + row] = store_dot(dot(row, sx[a]) / g)
template <int K, int kRows, typename OutT>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w,
                                           const uint8_t* __restrict__ scales,
                                           float g,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n,
                                           OutT* __restrict__ out,
                                           size_t out_stride) {
  float acc[kSteps][kRows];
  warp_row_dots<K, kRows>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    bool mine = false;
    const int row = owned_row<K>(n0, st, mine);
    if (mine && row < n) {
#pragma unroll
      for (int a = 0; a < kRows; ++a)
        store_dot(out + static_cast<size_t>(a) * out_stride + row,
                  __fdiv_rn(acc[st][a], g));
    }
  }
}

// Dispatch over the supported K: f(std::integral_constant<int, K>{}).
template <typename F>
__host__ inline void dispatch_k(int k, F&& f) {
  switch (k) {
    case 32: f(std::integral_constant<int, 32>{}); return;
    case 64: f(std::integral_constant<int, 64>{}); return;
    case 128: f(std::integral_constant<int, 128>{}); return;
    case 256: f(std::integral_constant<int, 256>{}); return;
    case 512: f(std::integral_constant<int, 512>{}); return;
    case 1024: f(std::integral_constant<int, 1024>{}); return;
    case 2048: f(std::integral_constant<int, 2048>{}); return;
    case 4096: f(std::integral_constant<int, 4096>{}); return;
    default:
      throw std::invalid_argument(
          "fp4_gemv: K must be a power of two in [32, 4096]");
  }
}

}  // namespace fp4_gemv
}  // namespace dgpp
