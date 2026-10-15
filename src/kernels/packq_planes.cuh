#pragma once
// The int8 head's plane layout (kernels/packq_head.hpp): the device
// loaders of one 16-code chunk's three fragments and the assembler that
// puts the chunk back in the packed core's word order, shared by the
// bit-plane head kernels (packq_head.cu) and the draft head's screen
// (head_screen.cu), whose exact rescore runs the same chain on the
// candidate rows.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Bit position of code j (0..15 of a chunk) in the chunk's two-bit plane
// words: pairs (j, j + 4) and (8 + j, 12 + j) 16 bits apart, as the nibble
// plane's word pairs are, so a pair is one shift-and-mask (the fp16 trick).
__host__ __device__ constexpr int packq_planes_pos(int j) {
  constexpr int base[4] = {0, 16, 8, 24};
  return 2 * (j & 3) + base[j >> 2];
}

namespace packq_planes {

// The three fragments of chunk ci (16 codes) of a row in the plane layout.
struct PlaneChunk {
  uint2 nib;      // codes' high nibbles: code j at bits 4j of the 64-bit pair
  uint32_t mid;   // bits 3..2 of code j at bits 2j
  uint32_t lo;    // bits 1..0 of code j at bits 2j
};
// The row's three planes: nibbles [0, K/2), bits 3..2 [K/2, 3K/4), bits
// 1..0 [3K/4, K); chunk ci's 16 codes are 8 B of the first and 4 B of each
// of the others.
__device__ __forceinline__ void load_hi6(const uint8_t* __restrict__ w, int row, int K, int ci, uint2& nib,
                                         uint32_t& mid) {
  const uint8_t* rb = w + static_cast<size_t>(row) * K;
  nib = *reinterpret_cast<const uint2*>(rb + ci * 8);
  mid = *reinterpret_cast<const uint32_t*>(rb + K / 2 + ci * 4);
}
__device__ __forceinline__ uint32_t load_lo2(const uint8_t* __restrict__ w, int row, int K, int ci) {
  const uint8_t* rb = w + static_cast<size_t>(row) * K;
  return *reinterpret_cast<const uint32_t*>(rb + K / 2 + K / 4 + ci * 4);
}
// The chunk's 16 codes back in the packed core's word order (word q holds
// codes 4q..4q+3, code j at bits 8 (j % 4)) — consume_chunk<8> then runs
// the production chain on them.
__device__ __forceinline__ uint4 assemble(const uint2& nib, uint32_t mid, uint32_t lo) {
  uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t nb = ((j < 8 ? nib.x : nib.y) >> (4 * (j & 7))) & 0xFu;
    const uint32_t md = (mid >> packq_planes_pos(j)) & 3u;
    const uint32_t lw = (lo >> packq_planes_pos(j)) & 3u;
    w[j >> 2] |= ((nb << 4) | (md << 2) | lw) << (8 * (j & 3));
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}


// One code of the plane layout (the build of a screen copy reads rows
// element by element): row `row`, element k of K.
__device__ __forceinline__ uint32_t plane_code(const uint8_t* __restrict__ w, int row, int K, int k) {
  const uint8_t* rb = w + static_cast<size_t>(row) * K;
  const uint32_t nb = (rb[k >> 1] >> (4 * (k & 1))) & 0xFu;
  const int chunk = k >> 4, pos = packq_planes_pos(k & 15);
  const uint32_t md = (*reinterpret_cast<const uint32_t*>(rb + K / 2 + chunk * 4) >> pos) & 3u;
  const uint32_t lw = (*reinterpret_cast<const uint32_t*>(rb + K / 2 + K / 4 + chunk * 4) >> pos) & 3u;
  return (nb << 4) | (md << 2) | lw;
}

}  // namespace packq_planes
}  // namespace dgpp
