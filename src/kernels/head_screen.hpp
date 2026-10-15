#pragma once
// The draft head's screen (2026-10-13): an MTP draft step needs only the
// argmax (or the top of the distribution) of the vocabulary head, not every
// logit, yet it streams the whole head — on Qwen3.8-Flash-Next 636 MB per
// draft token, 2.6 ms of a 4 ms draft step at one node, read once per draft
// in the chain. The screen replaces that read with three kernel nodes:
//
//   prep     the hidden row through a fixed random-sign block-128 Walsh-
//            Hadamard rotation, then int8 per row (the rotation spreads the
//            hidden state's outlier channels, |x| max 44 -> 14 at the
//            median, so the rounding noise below is Gaussian);
//   screen   dp4a of that int8 row against a 2-bit copy of the SAME rotated
//            head built once at load (one global four-level quantizer per
//            vocab row, step 0.9957 x the row's std; 25 % of the fp8 bytes),
//            one CTA per 128-row block, the block's approximate maximum out;
//   select   the blocks whose approximate maximum lies within delta sigmas
//            of the best block's (sigma = |x|_2 x the block's worst column
//            rounding rms), the top `cap` of them by value;
//   rescore  those blocks' logits computed EXACTLY from the served head by
//            the production row chains (fp8_gemv / bf16_gemv / bf12_gemv
//            row_dots), every other logit -inf. The pick kernels then run
//            unchanged on the masked row.
//
// Exactness: the draft only proposes; the verify reads the full head, so a
// greedy transcript is bit for bit the exact head's whatever the screen
// picks, and a sampled request's output distribution is exact for any
// proposal set (the ratio rule verifies against the same set). What can
// change is WHICH token is drafted. Measured on 5,571 real head inputs
// (2026-10-13 study, delta 3, cap 32, 2 bits, rotated): the draft equals the
// exact head's argmax on 99.3 % of draft rows and the remaining rows lose
// 0.001 % of the accepted probability mass; the sampled proposal's top-64
// mass trails the exact one by 0.7 points. 3 bits reached 100 % at 1.5x the
// bytes; 1 bit 97.7 % at half.
//
// Layout of the codes: for vocab block b (rows [128 b, 128 b + 128)), chunk
// g (k in [16 g, 16 g + 16)) and column c in the block, one u32 at
// index (b * (k / 16) + g) * 128 + c holds the sixteen codes with
//   byte lane L, bits [2 j, 2 j + 2)  =  code of k = 16 g + 4 j + L,
// so ((word >> 2 j) & 0x03030303) puts the four codes of k = 16 g + 4 j ..
// + 3 in the four byte lanes, where the int8 row's u32 for those k sits,
// and one dp4a consumes them. A CTA's warp reads 32 consecutive columns'
// words: 128 contiguous bytes; the block's whole stream is contiguous.
// The dot is Sum x (2 c - 3) step / 2 = step / 2 (Sum x 2 c - 3 Sum x), the
// first sum the dp4a's (unsigned 2 c against signed x), the second the
// row's once.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/bf12_gemv.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

constexpr int kHeadScreenBlock = 128;     // vocab rows per screen block
constexpr int kHeadScreenMaxRows = 16;    // activation rows per screen launch chunk
constexpr int kHeadScreenMaxBlocks = 64;  // the candidate cap's ceiling
constexpr int kHeadScreenMaxK = 8192;     // hidden widths the kernels take (k % 128 == 0; 6144 = the full GLM-5.3)
constexpr float kHeadScreenDelta = 3.0f;  // the study's margin, in screen-noise sigmas

// The screen copy of one rank's vocab slice: n rows (any count; a partial
// last block masks its missing columns), k hidden.
struct HeadScreen {
  uint32_t* codes = nullptr;  // [nb][k / 16][128]
  float* step = nullptr;      // [n] the row's quantizer step (0.9957 x its rotated std)
  float* rms_blk = nullptr;   // [nb] max over the block's rows of the rounding rms
  int8_t* sign = nullptr;     // [k] the rotation's signs (+-1)
  float* rms_row = nullptr;   // [n] build scratch (each row's rounding rms)
  int n = 0, k = 0, nb = 0;
};
size_t head_screen_bytes(int n, int k);
// Carves the struct's arrays from `storage` (head_screen_bytes of device memory the caller owns).
HeadScreen head_screen_carve(void* storage, int n, int k);
// Builds the copy from the served head (fp8 rows + block scales, or bf16
// rows); `seed` fixes the rotation's signs. Enqueued on `stream`.
void head_screen_build_fp8(const HeadScreen& s, const GlmQuantMatrix& w, uint64_t seed, cudaStream_t stream);
void head_screen_build_bf16(const HeadScreen& s, const uint16_t* w, uint64_t seed, cudaStream_t stream);
// The int8 g128 head in its plane layout (kernels/packq_head.hpp; the
// AutoRound hybrid): value = (code - 128) x the row's f16 group scale.
// Refuses any other packed form (bits, scale format, row layout).
bool head_screen_packq_supported(const GlmPackedMatrix& w);
// The bf16 head's 12-bit companion as the build's source (a residency that
// keeps only the companion): every element rebuilt from its sign+mantissa
// byte, exponent code and the row's window base, escapes and raw rows
// included — the values are the bf16 ones exactly, so the codes come out
// as the bf16 build's. k must be a whole number of 1024-column super-blocks
// (the tail's layout is not decoded here).
bool head_screen_bf12_supported(const Bf12Matrix& w);
void head_screen_build_bf12(const HeadScreen& s, const Bf12Matrix& w, uint64_t seed, cudaStream_t stream);
void head_screen_build_packq(const HeadScreen& s, const GlmPackedMatrix& w, uint64_t seed, cudaStream_t stream);

// Per-launch scratch for up to `rows` activation rows.
struct HeadScreenScratch {
  int8_t* xq = nullptr;     // [rows][k] the rotated int8 rows
  float* xmeta = nullptr;   // [rows][4]: scale, sum of codes, |x|_2, 0
  float* bmax = nullptr;    // [rows][nb] the blocks' approximate maxima
  int32_t* cand = nullptr;  // [rows][cap] candidate blocks by value, -1 past the last
  int rows = 0, cap = 0;
};
size_t head_screen_scratch_bytes(int rows, int k, int nb, int cap);
HeadScreenScratch head_screen_scratch_carve(void* storage, int rows, int k, int nb, int cap);

// prep + screen + select for `rows` bf16 activation rows (row stride
// act_stride elements): sc.cand holds each row's candidate blocks after.
// Kernel nodes only (capture-safe).
void launch_head_screen(const HeadScreen& s, const HeadScreenScratch& sc, const uint16_t* act,
                        size_t act_stride, int rows, float delta, cudaStream_t stream);

// The candidate blocks' exact logits into out [rows][out_stride] (f32), -inf
// in every other column below n. One launch; the weight form picks the
// production row chain (fp8_gemv::row_dots on the fp8 payload with its
// scale grid, bf16_gemv::row_dots, bf12_gemv::row_dots).
void launch_head_rescore_fp8(const HeadScreen& s, const HeadScreenScratch& sc, const GlmQuantMatrix& w,
                             const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                             cudaStream_t stream);
void launch_head_rescore_bf16(const HeadScreen& s, const HeadScreenScratch& sc, const uint16_t* w,
                              const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                              cudaStream_t stream);
void launch_head_rescore_bf12(const HeadScreen& s, const HeadScreenScratch& sc, const Bf12Matrix& w,
                              const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                              cudaStream_t stream);
void launch_head_rescore_packq(const HeadScreen& s, const HeadScreenScratch& sc, const GlmPackedMatrix& w,
                               const uint16_t* act, size_t act_stride, int rows, float* out, size_t out_stride,
                               cudaStream_t stream);

// The rotation's signs for `seed` (host; the build and the tests share it).
void head_screen_signs(uint64_t seed, int k, int8_t* out);

}  // namespace dgpp
