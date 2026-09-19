#include "kernels/bf12_gemv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.cuh"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace {

// One step of one lane: eight elements at columns [c0, c0 + 8). lo/hi carry
// their sign+mantissa bytes, ew their exponent nibbles. The weight's f32
// bits are its bf16 bits << 16 (bf16_bits_to_float), rebuilt from the byte
// and base + code; an escaped element takes its bits from the row's table.
// The FMA order is bf16_gemv::row_dots' (elements ascending per row).
// sx holds the staged activations at row stride sk, the step's elements at
// staged column cs (the narrow kernel stages whole rows: sk = k, cs = c0;
// the wide one a window of them).
template <int kRows>
__device__ __forceinline__ void consume_step(uint32_t lo, uint32_t hi, uint32_t ew,
                                             uint32_t base23,
                                             const uint16_t* __restrict__ sx, int sk, int cs,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e,
                                             float (&acc)[kRows]) {
  uint32_t wbits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t code = (ew >> (4 * j)) & 0xFu;
    wbits[j] = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 16) | (base23 + (code << 23));
  }
  // Any nibble == 15 (about one step in 25 on a trained matrix).
  const uint32_t any15 = ew & (ew >> 1) & (ew >> 2) & (ew >> 3) & 0x11111111u;
  if (any15 != 0u) {
    for (int j = 0; j < 8; ++j) {
      if (((ew >> (4 * j)) & 0xFu) != 0xFu) continue;
      const uint32_t col = static_cast<uint32_t>(c0 + j);
      for (uint32_t e = esc_b; e < esc_e; ++e) {
        const uint32_t ent = esc[e];
        if ((ent >> 16) == col) {
          wbits[j] = (ent & 0xFFFFu) << 16;
          break;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4 xv = *reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * sk + cs);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = std::bit_cast<float>(wbits[j]);
      const float x =
          bf16_bits_to_float(static_cast<uint16_t>((xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu));
      acc[r] = __fmaf_rn(w, x, acc[r]);
    }
  }
}

// kSB super-blocks (3 * kSB warp loads) in flight per lane before any is
// consumed — the bf16 core's reason (bytes in flight), the same shape.
template <int kRows, bool kOutF32, int kSB>
__global__ void bf12_gemv_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                 const uint8_t* __restrict__ w,
                                 const uint32_t* __restrict__ rows,
                                 const uint32_t* __restrict__ esc,
                                 const uint16_t* __restrict__ raw, void* __restrict__ out, int n,
                                 int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  const int nsb = k / kBf12Super;
  const uint8_t* wr = w + static_cast<size_t>(row) * nsb * kBf12SuperBytes;
  const uint32_t eb = rows[2 * row];
  const uint32_t base23 = rows[2 * row + 1];
  const uint32_t ee = rows[2 * row + 2];
  float acc[kRows];
  if ((base23 & kBf12RawRow) != 0u) {
    // A row kept bf16 (uniform across the warp): the production chain.
    bf16_gemv::row_dots<kRows>(raw + static_cast<size_t>(base23 & ~kBf12RawRow) * k, sx, k, lane,
                               acc);
  } else {
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
    for (int s0 = 0; s0 < nsb; s0 += kSB) {
      uint4 a[kSB], b[kSB], e[kSB];
#pragma unroll
      for (int s = 0; s < kSB; ++s) {
        // Past the row's end a lane re-reads a block it holds (uniform
        // across the warp; never consumed).
        const int sb = s0 + s < nsb ? s0 + s : s0;
        const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
        a[s] = *reinterpret_cast<const uint4*>(p);
        b[s] = *reinterpret_cast<const uint4*>(p + 512);
        e[s] = *reinterpret_cast<const uint4*>(p + 1024);
      }
#pragma unroll
      for (int s = 0; s < kSB; ++s) {
        if (s0 + s >= nsb) break;
        const int cb = (s0 + s) * kBf12Super + lane * 8;
        consume_step<kRows>(a[s].x, a[s].y, e[s].x, base23, sx, k, cb, cb, esc, eb, ee, acc);
        consume_step<kRows>(a[s].z, a[s].w, e[s].y, base23, sx, k, cb + 256, cb + 256, esc, eb, ee, acc);
        consume_step<kRows>(b[s].x, b[s].y, e[s].z, base23, sx, k, cb + 512, cb + 512, esc, eb, ee, acc);
        consume_step<kRows>(b[s].z, b[s].w, e[s].w, base23, sx, k, cb + 768, cb + 768, esc, eb, ee, acc);
      }
    }
    gemv::warp_reduce<kRows>(acc);
  }
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

// One super-block of a row kept bf16 (the wide kernel's raw rows): the
// production chain over columns [col0, col0 + 1024) — bf16_gemv::row_dots'
// four steps of it, in its order, into the running accumulators.
template <int kRows>
__device__ __forceinline__ void raw_super_block(const uint16_t* __restrict__ w_row, int col0,
                                                const uint16_t* __restrict__ sx, int lane,
                                                float (&acc)[kRows]) {
  uint4 wv[4];
#pragma unroll
  for (int t = 0; t < 4; ++t)
    wv[t] = *reinterpret_cast<const uint4*>(w_row + col0 + t * 256 + lane * 8);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const uint4 xv = *reinterpret_cast<const uint4*>(
          sx + static_cast<size_t>(r) * kBf12Super + t * 256 + lane * 8);
      bf16_gemv::fma_pair(wv[t].x, xv.x, acc[r]);
      bf16_gemv::fma_pair(wv[t].y, xv.y, acc[r]);
      bf16_gemv::fma_pair(wv[t].z, xv.z, acc[r]);
      bf16_gemv::fma_pair(wv[t].w, xv.w, acc[r]);
    }
  }
}

// The WIDE form (2026-09-19): five to eight activation rows — the batch the
// bf16 sites hand to cuBLASLt today, which reads the bf16 bytes once at
// ~210 GB/s — and the narrow rows whose whole staged rows pass the smem
// bound (k = 8192 from four rows). The activations are staged one
// super-block (1024 columns) at a time, 16 KB at eight rows, with a barrier
// on both sides of every restage; a lane's accumulators live across the
// windows, so a row's chain is still the scalar one: bitwise the GEMV
// chunks at any m. Measured at eight rows against cuBLASLt (real weights,
// cold): 80 -> 57 us on the KDA q slice, 1,400 -> 940 us on the lm head.
template <int kRows, bool kOutF32>
__global__ void bf12_gemv_wide_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                      const uint8_t* __restrict__ w,
                                      const uint32_t* __restrict__ rows,
                                      const uint32_t* __restrict__ esc,
                                      const uint16_t* __restrict__ raw, void* __restrict__ out,
                                      int n, int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  // A warp past n keeps staging and meeting the barriers (they are the
  // block's), and consumes nothing.
  const bool live = row < n;
  const int nsb = k / kBf12Super;
  const uint8_t* wr = w + static_cast<size_t>(live ? row : 0) * nsb * kBf12SuperBytes;
  const uint32_t eb = live ? rows[2 * row] : 0u;
  const uint32_t base23 = live ? rows[2 * row + 1] : 0u;
  const uint32_t ee = live ? rows[2 * row + 2] : 0u;
  const bool raw_row = (base23 & kBf12RawRow) != 0u;
  const uint16_t* raw_w = raw + static_cast<size_t>(base23 & ~kBf12RawRow) * k;
  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  for (int sb = 0; sb < nsb; ++sb) {
    gemv::stage_activations<kRows>(act + static_cast<size_t>(sb) * kBf12Super, act_stride,
                                   kBf12Super, sx);
    __syncthreads();
    if (live) {
      if (raw_row) {
        raw_super_block<kRows>(raw_w, sb * kBf12Super, sx, lane, acc);
      } else {
        const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
        const uint4 a = *reinterpret_cast<const uint4*>(p);
        const uint4 b = *reinterpret_cast<const uint4*>(p + 512);
        const uint4 e = *reinterpret_cast<const uint4*>(p + 1024);
        const int cw = lane * 8;                 // the window's column
        const int cb = sb * kBf12Super + cw;     // the row's
        consume_step<kRows>(a.x, a.y, e.x, base23, sx, kBf12Super, cw, cb, esc, eb, ee, acc);
        consume_step<kRows>(a.z, a.w, e.y, base23, sx, kBf12Super, cw + 256, cb + 256, esc, eb, ee, acc);
        consume_step<kRows>(b.x, b.y, e.z, base23, sx, kBf12Super, cw + 512, cb + 512, esc, eb, ee, acc);
        consume_step<kRows>(b.z, b.w, e.w, base23, sx, kBf12Super, cw + 768, cb + 768, esc, eb, ee, acc);
      }
    }
    // Everyone is done with sx before it is restaged.
    if (sb + 1 < nsb) __syncthreads();
  }
  if (!live) return;
  gemv::warp_reduce<kRows>(acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

template <int kRows>
void launch_wide(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                 bool out_f32, cudaStream_t stream) {
  const dim3 grid((w.n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, kBf12Super);
  if (out_f32)
    bf12_gemv_wide_kernel<kRows, true><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k);
  else
    bf12_gemv_wide_kernel<kRows, false><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows, int kSB>
void launch_rows(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                 bool out_f32, cudaStream_t stream) {
  const dim3 grid((w.n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, w.k);
  if (out_f32)
    bf12_gemv_kernel<kRows, true, kSB><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k);
  else
    bf12_gemv_kernel<kRows, false, kSB><<<grid, gemv::kThreads, smem, stream>>>(
        act, act_stride, w.packed, w.rows, w.esc, w.raw, out, w.n, w.k);
  DGPP_CUDA_OK(cudaGetLastError());
}

template <int kRows>
void launch_depth(const uint16_t* act, size_t act_stride, const Bf12Matrix& w, void* out,
                  bool out_f32, cudaStream_t stream) {
  // Whole rows in flight when the row is a multiple of four super-blocks
  // (k = 4096: twelve loads), pairs otherwise (k = 2048: six).
  if ((w.k / kBf12Super) % 4 == 0)
    launch_rows<kRows, 4>(act, act_stride, w, out, out_f32, stream);
  else
    launch_rows<kRows, 2>(act, act_stride, w, out, out_f32, stream);
}

// The packed position of element c of a row: byte and nibble offsets.
struct Slot {
  size_t sm;   // the sign+mantissa byte
  size_t exp;  // the byte holding the nibble
  int shift;   // 0 (even element) or 4
};
inline Slot slot_of(int c) {
  const int sb = c / kBf12Super, in = c % kBf12Super;
  const int t = in / 256, lane = (in % 256) / 8, j = in % 8;
  const size_t blk = static_cast<size_t>(sb) * kBf12SuperBytes;
  return {blk + (t < 2 ? 0 : 512) + static_cast<size_t>(lane) * 16 + (t & 1) * 8 + j,
          blk + 1024 + static_cast<size_t>(lane) * 16 + t * 4 + j / 2, 4 * (j & 1)};
}

}  // namespace

Bf12Host bf12_encode(const uint16_t* w, int n, int k, int threads) {
  constexpr uint32_t kRawMark = 0xFFFFFFFFu;
  Bf12Host h;
  if (w == nullptr || !bf12_shape_ok(n, k)) return h;
  if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  threads = std::min(threads, n);
  const auto range = [&](int t) {
    return std::pair<int, int>{static_cast<int>(static_cast<int64_t>(n) * t / threads),
                               static_cast<int>(static_cast<int64_t>(n) * (t + 1) / threads)};
  };
  const size_t row_bytes = bf12_packed_bytes(1, k);
  h.packed.assign(static_cast<size_t>(n) * row_bytes, 0);
  h.rows.assign((static_cast<size_t>(n) + 1) * 2, 0);
  std::vector<std::vector<uint32_t>> escs(threads);
  std::vector<uint32_t> counts(static_cast<size_t>(n), 0);
  // The slot of every column, once (the rows share it).
  std::vector<Slot> slots(static_cast<size_t>(k));
  for (int c = 0; c < k; ++c) slots[c] = slot_of(c);
  {
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        const auto [r0, r1] = range(t);
        std::array<uint32_t, 256> hist;
        for (int row = r0; row < r1; ++row) {
          const uint16_t* src = w + static_cast<size_t>(row) * k;
          // The row's window: the fifteen consecutive exponents holding the
          // most of its weights.
          hist.fill(0);
          for (int c = 0; c < k; ++c) ++hist[(src[c] >> 7) & 0xFF];
          uint32_t in = 0;
          for (int i = 0; i < kBf12Window; ++i) in += hist[i];
          uint32_t best = in;
          int base = 0;
          for (int b = 1; b + kBf12Window <= 256; ++b) {
            in += hist[b + kBf12Window - 1] - hist[b - 1];
            if (in > best) {
              best = in;
              base = b;
            }
          }
          h.rows[2 * static_cast<size_t>(row) + 1] = static_cast<uint32_t>(base) << 23;
          if (static_cast<uint32_t>(k) - best > static_cast<uint32_t>(kBf12MaxRowEscapes)) {
            counts[row] = kRawMark;  // kept bf16: indexed after the join
            continue;
          }
          uint8_t* dst = h.packed.data() + static_cast<size_t>(row) * row_bytes;
          for (int c = 0; c < k; ++c) {
            const uint16_t v = src[c];
            int code = static_cast<int>((v >> 7) & 0xFF) - base;
            if (code < 0 || code >= kBf12Window) {
              code = 15;
              escs[t].push_back((static_cast<uint32_t>(c) << 16) | v);
              ++counts[row];
            }
            const Slot& s = slots[c];
            dst[s.sm] = static_cast<uint8_t>(((v >> 8) & 0x80) | (v & 0x7F));
            dst[s.exp] = static_cast<uint8_t>(dst[s.exp] | (code << s.shift));
          }
        }
      });
    for (auto& th : pool) th.join();
  }
  uint32_t at = 0;
  for (int row = 0; row < n; ++row) {
    h.rows[2 * static_cast<size_t>(row)] = at;
    if (counts[row] == kRawMark) {
      h.rows[2 * static_cast<size_t>(row) + 1] = kBf12RawRow | static_cast<uint32_t>(h.raw_rows++);
      const uint16_t* src = w + static_cast<size_t>(row) * k;
      h.raw.insert(h.raw.end(), src, src + k);
      continue;
    }
    at += counts[row];
    h.max_row_escapes = std::max(h.max_row_escapes, static_cast<int>(counts[row]));
  }
  if (h.raw.empty()) h.raw.assign(static_cast<size_t>(k), 0);  // a valid device array
  h.rows[2 * static_cast<size_t>(n)] = at;
  h.escapes = at;
  h.esc.reserve(static_cast<size_t>(at) + 1);
  for (const auto& part : escs) h.esc.insert(h.esc.end(), part.begin(), part.end());
  if (h.esc.empty()) h.esc.push_back(0);  // a valid device array either way
  h.ok = static_cast<int64_t>(h.raw_rows) * kBf12MaxRawFraction <= n;
  return h;
}

void bf12_decode(const Bf12Host& h, int n, int k, uint16_t* out) {
  if (!bf12_shape_ok(n, k) || h.packed.size() != bf12_packed_bytes(n, k))
    throw std::invalid_argument("bf12_decode: shape does not match the packed form");
  const size_t row_bytes = bf12_packed_bytes(1, k);
  for (int row = 0; row < n; ++row) {
    const uint8_t* src = h.packed.data() + static_cast<size_t>(row) * row_bytes;
    uint32_t e = h.rows[2 * static_cast<size_t>(row)];
    const uint32_t end = h.rows[2 * static_cast<size_t>(row) + 2];
    const uint32_t word = h.rows[2 * static_cast<size_t>(row) + 1];
    if ((word & kBf12RawRow) != 0u) {
      std::memcpy(out + static_cast<size_t>(row) * k,
                  h.raw.data() + static_cast<size_t>(word & ~kBf12RawRow) * k,
                  static_cast<size_t>(k) * 2);
      continue;
    }
    const uint32_t base = word >> 23;
    for (int c = 0; c < k; ++c) {
      const Slot s = slot_of(c);
      const uint32_t b = src[s.sm];
      const uint32_t code = (src[s.exp] >> s.shift) & 0xFu;
      if (code == 15u) {
        // Column-sorted within the row: the next entry is this column's.
        if (e >= end || (h.esc[e] >> 16) != static_cast<uint32_t>(c))
          throw std::runtime_error("bf12_decode: escape table out of step");
        out[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(h.esc[e++] & 0xFFFFu);
      } else {
        out[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(
            ((b & 0x80u) << 8) | ((base + code) << 7) | (b & 0x7Fu));
      }
    }
  }
}

bool bf12_gemv_accepts(const Bf12Matrix& w, int m) {
  return w.packed != nullptr && w.rows != nullptr && w.esc != nullptr && w.raw != nullptr &&
         gemv::aligned16(w.raw) && bf12_shape_ok(w.n, w.k) && m >= 1 && m <= kBf12MaxRows &&
         gemv::aligned16(w.packed);
}

void launch_bf12_gemv(const uint16_t* act, size_t act_row_stride, const Bf12Matrix& w,
                      void* out, bool out_f32, int m, cudaStream_t stream) {
  if (!act || !out) throw std::invalid_argument("bf12_gemv: null pointer");
  if (!bf12_gemv_accepts(w, m))
    throw std::invalid_argument("bf12_gemv: shape outside the GEMV contract");
  if (act_row_stride < static_cast<size_t>(w.k))
    throw std::invalid_argument("bf12_gemv: activation stride narrower than k");
  // Whole staged rows while they fit the smem bound and the register
  // budget (four rows), the windowed form past either.
  const bool narrow = m <= gemv::kMaxRows && gemv::smem_fits(m, w.k);
  switch (m) {
    case 1: launch_depth<1>(act, act_row_stride, w, out, out_f32, stream); break;
    case 2:
      if (narrow) launch_depth<2>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<2>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 3:
      if (narrow) launch_depth<3>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<3>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 4:
      if (narrow) launch_depth<4>(act, act_row_stride, w, out, out_f32, stream);
      else launch_wide<4>(act, act_row_stride, w, out, out_f32, stream);
      break;
    case 5: launch_wide<5>(act, act_row_stride, w, out, out_f32, stream); break;
    case 6: launch_wide<6>(act, act_row_stride, w, out, out_f32, stream); break;
    case 7: launch_wide<7>(act, act_row_stride, w, out, out_f32, stream); break;
    case 8: launch_wide<8>(act, act_row_stride, w, out, out_f32, stream); break;
    default: throw std::invalid_argument("bf12_gemv: m outside 1..8");
  }
}

}  // namespace dgpp
