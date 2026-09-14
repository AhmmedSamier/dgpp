// The streaming tensor-core decode GEMM — see mma_gemv.hpp.
#include "kernels/mma_gemv.hpp"

#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemv.cuh"

namespace dgpp {
namespace {

constexpr int kWarps = 8;              // weight rows per block: kWarps x 8
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerWarp = 8;        // the mma's n
constexpr int kBatch = 8;              // 16-byte vectors in flight per lane
// A lane's run: kVecs consecutive 16-byte vectors of its weight row, so a
// quad's loads cover 4 x kVecs x 16 B contiguous of a row per group: fp8
// one vector (a quad = 64 B; 64-byte runs, four vectors per lane with the
// lines filled across instructions, halved the stream to 112 GB/s), bf16
// two (a quad = one 128-byte line).
template <bool kFp8> struct Fmt {
  static constexpr int kVecs = kFp8 ? 1 : 2;         // 16-byte vectors per lane per group
  static constexpr int kPerVec = kFp8 ? 16 : 8;      // k per vector
  static constexpr int kLaneK = kVecs * kPerVec;     // k per lane per group: 16
  static constexpr int kQuadSpan = 4 * kLaneK;       // k per quad per group: 64
  static constexpr int kLaneUnits = kLaneK / 8;      // the lane's A run in 16-byte units: 2
};
// A's shared-memory unit (16 B = 8 bf16) swizzle: fp8 lanes read 128-byte
// runs 128 B apart (a 4-way bank conflict per LDS.128 phase); XOR-ing the
// unit's low bits with its run index spreads the four lanes over the
// banks. bf16's 32-byte runs already interleave. Stage and read agree.
template <int kLaneUnits>
__device__ __forceinline__ int swz(int u) { return kLaneUnits == 8 ? (u ^ ((u >> 3) & 3)) : u; }
// The activation window [tiles*16 rows, kWindowK(tiles) k] staged in
// dynamic shared memory per batch of chunks, double-buffered (window i+1
// staged while window i is consumed: one barrier per window, and the
// weight loads of the next batch issued before it): every warp of the
// block reads the same A fragments from shared memory instead of a
// dependent global load per k slice (that stall halved the m = 1 rate).
// The window shrinks with the tile count so two buffers stay inside the
// 48 KB the capture paths allow without an attribute opt-in: 1/2/4/8
// tiles (16/32/64/128 rows) take 512/256/128/64-k windows, 8/4/2/1
// chunks in flight per lane. The wider forms are the prefill's (33..128
// rows per launch, the weights read once per 128 rows instead of once
// per 4-row chunk); a row's chain is the same in every form (the set of
// physical k folded per slice is a function of the 64-k group, whatever
// the window), so the forms are bitwise each other's.
template <int kTiles> struct Win {
  static constexpr int kBatchW = kBatch / kTiles;                     // vectors per lane per window
  static constexpr int kK = 64 * kBatchW;                             // k per window: 512 / 256 / 128 / 64
  static constexpr int kStride = kK + 8;                              // staged row stride (rows shift 16 B: no bank conflicts)
  static constexpr int kRows = kTiles * 16;
  static constexpr size_t kBufBytes = static_cast<size_t>(kRows) * kStride * 2;
  static constexpr size_t kSmemBytes = 2 * kBufBytes;                 // double-buffered
};
static_assert(Win<1>::kSmemBytes <= 48 * 1024 && Win<2>::kSmemBytes <= 48 * 1024 &&
              Win<4>::kSmemBytes <= 48 * 1024 && Win<8>::kSmemBytes <= 48 * 1024, "the A windows fit the default smem");

// Stage rows [0, tiles*16) x [base, base + W::kK) of act into sA (rows
// past m and columns past k read as zero); the whole block participates,
// the caller syncs.
template <int kTiles, int kLaneUnits>
__device__ __forceinline__ void stage_a(const uint16_t* __restrict__ act, size_t act_stride, int m,
                                        int k, int base, uint16_t* __restrict__ sA) {
  using W = Win<kTiles>;
  constexpr int kUnits = W::kK / 8;  // 16-byte units per row
  if (base >= k) return;
  for (int i = threadIdx.x; i < W::kRows * kUnits; i += kThreads) {
    const int row = i / kUnits, u = i - row * kUnits;
    const int col = base + u * 8;
    uint4 x = make_uint4(0u, 0u, 0u, 0u);
    if (row < m && col + 8 <= k)
      x = *reinterpret_cast<const uint4*>(act + static_cast<size_t>(row) * act_stride + col);
    *reinterpret_cast<uint4*>(sA + static_cast<size_t>(row) * W::kStride + swz<kLaneUnits>(u) * 8) = x;
  }
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16-byte fp8 chunk (16 k of one weight row, scale s) -> 8 packed bf16
// pairs (k 0..15 in order), the dequant bridge's values bf16(w * s) — exact
// for the e8m0 (power-of-two) scales the fp8 matrices carry.
__device__ __forceinline__ void chunk_to_bf16x8(const uint4& c, float s, uint32_t (&w)[8]) {
  const uint32_t words[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 w01 = fp8_gemv::e4m3x2_to_float2(static_cast<uint16_t>(words[q] & 0xFFFFu));
    const float2 w23 = fp8_gemv::e4m3x2_to_float2(static_cast<uint16_t>(words[q] >> 16));
    const uint32_t d0 = float_to_bf16_bits(w01.x * s), d1 = float_to_bf16_bits(w01.y * s);
    const uint32_t d2 = float_to_bf16_bits(w23.x * s), d3 = float_to_bf16_bits(w23.y * s);
    w[2 * q] = d0 | (d1 << 16);
    w[2 * q + 1] = d2 | (d3 << 16);
  }
}

// The k permutation. The mma's B fragment gives lane (r = lane/4, t =
// lane%4) weight row r at the slice's k 2t, 2t+1 (b0) and 2t+8, 2t+9 (b1);
// its A fragment gives the same lane rows r and r+8 at the same k. A dot
// product is invariant to which physical k sits in which slot as long as
// A and B agree, so lane t's own 16-element chunk (physical k c0_t..+15,
// c0_t = window + 16 t) fills its four slots of four slices: slice q takes
// chunk elements 4q..4q+3 as b0 = (4q, 4q+1), b1 = (4q+2, 4q+3), and the
// A words are the same four activation elements — 32 contiguous bytes per
// row per chunk, two 16-byte shared loads. No shuffles: the earlier
// owner-lane form (8 shuffles per slice) was issue-bound at ~4 GB/s per SM.
// The chain of a weight row (the set of k folded per slice, the order of
// slices) is a function of (k, the window layout) only: rows batched and
// rows alone see the same arithmetic.

// kTiles: 1 (m <= 16) or 2 (m <= 32). kFp8: the weight format.
template <int kTiles, bool kFp8, typename OutT>
__global__ __launch_bounds__(kThreads, 2) void mma_gemv_kernel(const uint16_t* __restrict__ act,
                                                            size_t act_stride,
                                                            const void* __restrict__ wv,
                                                            const float* __restrict__ scales,
                                                            OutT* __restrict__ out, int m, int n,
                                                            int k, size_t out_stride, int rs,
                                                            int cs) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = lane / 4, t = lane % 4;
  const int n0 = (blockIdx.x * kWarps + warp) * kRowsPerWarp;  // this warp's first weight row
  const int row = n0 + r;                                        // this lane's weight row
  const bool row_live = row < n;
  const int scale_cols = (k + (1 << cs) - 1) >> cs;
  const float* scale_row = scales + (row_live ? static_cast<size_t>(row >> rs) * scale_cols : 0);
  const uint8_t* w8 = static_cast<const uint8_t*>(wv);
  const uint16_t* w16 = static_cast<const uint16_t*>(wv);

  float c[kTiles][4];
#pragma unroll
  for (int tt = 0; tt < kTiles; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[tt][i] = 0.f;

  using W = Win<kTiles>;
  using F = Fmt<kFp8>;
  constexpr int kGroups = W::kK / F::kQuadSpan;  // lane runs per window
  extern __shared__ __align__(16) uint16_t sA_raw[];
  uint16_t* sA[2] = {sA_raw, sA_raw + W::kBufBytes / 2};
  // The first window's activations, then per window: issue the weight
  // loads, stage the NEXT window's activations, consume, one barrier.
  stage_a<kTiles, F::kLaneUnits>(act, act_stride, m, k, 0, sA[0]);
  __syncthreads();
  int buf = 0;
  for (int base = 0; base < k; base += W::kK, buf ^= 1) {
    uint4 wv4[kGroups][F::kVecs];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      const int c0 = base + g * F::kQuadSpan + t * F::kLaneK;  // this lane's run (k % 64 == 0: whole or none)
      const bool in = row_live && c0 < k;
      const uint4* p = kFp8 ? reinterpret_cast<const uint4*>(w8 + static_cast<size_t>(row) * k + c0)
                            : reinterpret_cast<const uint4*>(w16 + static_cast<size_t>(row) * k + c0);
#pragma unroll
      for (int v = 0; v < F::kVecs; ++v) wv4[g][v] = in ? p[v] : make_uint4(0u, 0u, 0u, 0u);
    }
    stage_a<kTiles, F::kLaneUnits>(act, act_stride, m, k, base + W::kK, sA[buf ^ 1]);
    const uint16_t* sAc = sA[buf];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      if (base + g * F::kQuadSpan >= k) break;
      const int c0 = base + g * F::kQuadSpan + t * F::kLaneK;
      const int u0 = (c0 - base) / 8;  // the run's first A unit (window-local)
#pragma unroll
      for (int v = 0; v < F::kVecs; ++v) {
        // This vector's k (16 fp8 or 8 bf16) as bf16 words, the B words of
        // its slices (4 k per slice); the A words are the same k of rows r
        // and r+8: one 16-byte unit per 8 k.
        uint32_t w[8];
        if (kFp8) {
          const float s = (row_live && c0 < k) ? scale_row[(c0 + v * F::kPerVec) >> cs] : 0.f;
          chunk_to_bf16x8(wv4[g][v], s, w);
        } else {
          w[0] = wv4[g][v].x; w[1] = wv4[g][v].y; w[2] = wv4[g][v].z; w[3] = wv4[g][v].w;
        }
        constexpr int kUnitsPerVec = F::kPerVec / 8;  // 2 or 1
#pragma unroll
        for (int tt = 0; tt < kTiles; ++tt) {
          const uint16_t* row0 = sAc + static_cast<size_t>(tt * 16 + r) * W::kStride;
          const uint16_t* row8 = row0 + static_cast<size_t>(8) * W::kStride;
#pragma unroll
          for (int j = 0; j < kUnitsPerVec; ++j) {
            const int u = swz<F::kLaneUnits>(u0 + v * kUnitsPerVec + j) * 8;
            const uint4 a0 = *reinterpret_cast<const uint4*>(row0 + u);
            const uint4 a8 = *reinterpret_cast<const uint4*>(row8 + u);
            mma_bf16(c[tt], a0.x, a8.x, a0.y, a8.y, w[4 * j], w[4 * j + 1]);
            mma_bf16(c[tt], a0.z, a8.z, a0.w, a8.w, w[4 * j + 2], w[4 * j + 3]);
          }
        }
      }
    }
    __syncthreads();  // this window consumed, the next one staged
  }
  // Epilogue: C fragment (m16n8): (r, 2t), (r, 2t+1), (r+8, 2t), (r+8, 2t+1)
  // of the warp's [16 x 8] slice; the slice's n columns are the warp's rows.
  const int col0 = n0 + 2 * t;
#pragma unroll
  for (int tt = 0; tt < kTiles; ++tt) {
    const int mrow = tt * 16 + r;
    if (mrow < m) {
      if (col0 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col0, c[tt][0]);
      if (col0 + 1 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow) * out_stride + col0 + 1, c[tt][1]);
    }
    if (mrow + 8 < m) {
      if (col0 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow + 8) * out_stride + col0, c[tt][2]);
      if (col0 + 1 < n) fp8_gemv::store_dot(out + static_cast<size_t>(mrow + 8) * out_stride + col0 + 1, c[tt][3]);
    }
  }
}

template <bool kFp8, typename OutT>
void launch(const uint16_t* act, size_t act_stride, const void* w, const float* scales, OutT* out,
            int m, int n, int k, size_t out_stride, int rs, int cs, cudaStream_t stream) {
  if (m < 1) throw std::invalid_argument("mma_gemv: m outside [1, ...)");
  if (n <= 0 || k <= 0 || (k % 64) != 0) throw std::invalid_argument("mma_gemv: k a multiple of 64");
  if (!mma_gemv_shape_ok(w, act, act_stride, m, k)) throw std::invalid_argument("mma_gemv: alignment");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  if (out_stride < static_cast<size_t>(n)) throw std::invalid_argument("mma_gemv: out stride");
  if (kFp8 && (scales == nullptr || cs < 4)) throw std::invalid_argument("mma_gemv: fp8 scales");
  const dim3 grid((n + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp));
  // Groups of at most kMmaGemvMaxRowsPerLaunch rows; the form by the group's rows.
  for (int row0 = 0; row0 < m; row0 += kMmaGemvMaxRowsPerLaunch) {
    const int rows = std::min(kMmaGemvMaxRowsPerLaunch, m - row0);
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    OutT* o = out + static_cast<size_t>(row0) * out_stride;
    if (rows <= 16)
      mma_gemv_kernel<1, kFp8, OutT><<<grid, kThreads, Win<1>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs);
    else if (rows <= 32)
      mma_gemv_kernel<2, kFp8, OutT><<<grid, kThreads, Win<2>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs);
    else if (rows <= 64)
      mma_gemv_kernel<4, kFp8, OutT><<<grid, kThreads, Win<4>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs);
    else
      mma_gemv_kernel<8, kFp8, OutT><<<grid, kThreads, Win<8>::kSmemBytes, stream>>>(a, act_stride, w, scales, o, rows, n, k, out_stride, rs, cs);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}
}  // namespace

bool mma_gemv_shape_ok(const void* w, const void* act, size_t act_stride, int m, int k) {
  return m >= 1 && k > 0 && (k % 64) == 0 &&
         (reinterpret_cast<uintptr_t>(w) & 15u) == 0 && (reinterpret_cast<uintptr_t>(act) & 15u) == 0 &&
         ((act_stride * 2) % 16) == 0;
}

void launch_mma_gemv_fp8_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                              const float* scales, uint16_t* out, int m, int n, int k,
                              size_t out_stride, int rs, int cs, cudaStream_t stream) {
  launch<true, uint16_t>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream);
}
void launch_mma_gemv_fp8_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                             const float* scales, float* out, int m, int n, int k,
                             size_t out_stride, int rs, int cs, cudaStream_t stream) {
  launch<true, float>(act, act_stride, w, scales, out, m, n, k, out_stride, rs, cs, stream);
}
void launch_mma_gemv_bf16_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w,
                               uint16_t* out, int m, int n, int k, size_t out_stride,
                               cudaStream_t stream) {
  launch<false, uint16_t>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream);
}
void launch_mma_gemv_bf16_f32(const uint16_t* act, size_t act_stride, const uint16_t* w,
                              float* out, int m, int n, int k, size_t out_stride,
                              cudaStream_t stream) {
  launch<false, float>(act, act_stride, w, nullptr, out, m, n, k, out_stride, 7, 7, stream);
}

}  // namespace dgpp
