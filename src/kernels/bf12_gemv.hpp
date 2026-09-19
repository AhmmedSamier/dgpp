#pragma once
// bf12: a LOSSLESS 12-bit resident form of a bf16 weight matrix for the
// decode GEMV (2026-09-19).
//
// WHY THIS EXISTS: the decode step is a bandwidth problem and the native
// bf16 tensors (the KDA projections, the lm head) are over half of the bytes
// a GLM-5.3-Flash step reads. A bf16 weight's sign and mantissa are
// incompressible, but its exponent is not: a trained matrix spends ~2.6 bits
// of entropy on it and 99.98 % of the weights sit inside a window of fifteen
// consecutive exponents. The packed form stores the sign+mantissa byte and a
// 4-bit exponent code against a per-ROW base (a fused projection's rows do
// not share a scale: the KDA in_proj stacks f_a|g_a|q|k|v|b) — 12 bits a
// weight, 0.75 of the bytes — and the kernel rebuilds the exact bf16 bits in
// registers.
// Code 15 is the escape: the weight's true bits live in a per-row side table
// (column-sorted, a few entries per thousand rows) and the lane that meets
// the code fetches them before the weight enters its chain.
//
// NUMERICS: none. Every weight value is bit-for-bit the bf16 weight, and the
// lane/step ownership and the FMA order are bf16_gemv's (bf16_gemv.cuh), so
// every output is bitwise launch_bf16_gemv's — the unit test pins it, and a
// served transcript cannot tell the two apart. This is a storage format, not
// a quantization.
//
// LAYOUT: a row is k / 1024 super-blocks of 1536 bytes: [sm A | sm B | exp],
// 512 bytes each. Lane l owns 16 bytes of each segment at offset 16 * l:
// A holds the sign+mantissa bytes of its eight elements of steps 0 and 1 (a
// step is the production kernel's 256-element warp step, a lane's share of
// it eight elements), B steps 2 and 3, exp the four steps' exponent nibbles
// (low nibble = even element). Every warp load is a contiguous 512 bytes —
// four whole lines, as the bf16 core's.
//
// RAW ROWS: the escape lookup is a linear scan, so a row with more than
// kBf12MaxRowEscapes escapes (an outlier channel: a handful of q/k rows in
// a few layers) keeps its bf16 bits in a side array, and its warp — a warp
// is one weight row, so the branch is uniform across its lanes — runs the
// production row chain on them (bf16_gemv::row_dots itself).
//
// CONTRACT: k % 1024 == 0, k <= 65536 (an escape's column is 16 bits), at
// most one row in kBf12MaxRawFraction kept raw (past that the matrix keeps
// its bf16 form: the packing would not pay).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kBf12Super = 1024;       // elements per super-block
constexpr int kBf12SuperBytes = 1536;  // their packed bytes
constexpr int kBf12Window = 15;        // exponent codes 0..14; 15 escapes
constexpr int kBf12MaxRowEscapes = 64;
constexpr int kBf12MaxRows = 8;        // activation rows per launch
constexpr int kBf12MaxRawFraction = 8;
constexpr uint32_t kBf12RawRow = 0x80000000u;  // a rows[] base word: raw row | index

// Device views of one packed matrix (the owner keeps the allocations).
struct Bf12Matrix {
  const uint8_t* packed = nullptr;  // [n][k / 1024 * 1536]
  // [n + 1] pairs {first escape of the row, the row's window base << 23 —
  // or kBf12RawRow | its index in raw}; the pair after the last row closes
  // its escape range.
  const uint32_t* rows = nullptr;
  const uint32_t* esc = nullptr;    // (column << 16) | bf16 bits
  const uint16_t* raw = nullptr;    // [raw rows][k] bf16, 16-byte aligned
  int n = 0, k = 0;
};

// The host form bf12_encode produces (upload the three arrays as they are).
struct Bf12Host {
  std::vector<uint8_t> packed;
  std::vector<uint32_t> rows, esc;  // Bf12Matrix's arrays
  std::vector<uint16_t> raw;        // the raw rows (never empty: one pad row)
  size_t escapes = 0;
  int raw_rows = 0;
  int max_row_escapes = 0;  // the widest packed row's escapes
  bool ok = false;          // false: the matrix keeps its bf16 form
};

inline bool bf12_shape_ok(int n, int k) {
  return n > 0 && k > 0 && (k % kBf12Super) == 0 && k <= 65536;
}
inline size_t bf12_packed_bytes(int n, int k) {
  return static_cast<size_t>(n) * static_cast<size_t>(k / kBf12Super) * kBf12SuperBytes;
}

// Packs w[n, k] (bf16 bits). threads <= 0 picks the hardware's. ok == false
// when the shape is outside the contract or too many rows would stay raw.
Bf12Host bf12_encode(const uint16_t* w, int n, int k, int threads = 0);
// The inverse (tests): out[n, k] bf16 bits.
void bf12_decode(const Bf12Host& h, int n, int k, uint16_t* out);

// m activation rows this launch can take: 1..kBf12MaxRows. Up to four rows
// whose staged rows fit the smem bound run the narrow kernel (whole rows
// staged, the row's loads in flight at once); five to eight rows — and the
// narrow counts past the bound — the wide one (the activations staged a
// super-block at a time, 16 KB at eight rows). Both keep the scalar chain,
// so a row's bits do not depend on the rows sharing its launch.
bool bf12_gemv_accepts(const Bf12Matrix& w, int m);

// out[m, n] = act[m, k] x W^T, bitwise launch_bf16_gemv's rows on the
// unpacked weights (in chunks, past four rows). act_row_stride in elements;
// out bf16 or f32 by out_f32.
void launch_bf12_gemv(const uint16_t* act, size_t act_row_stride, const Bf12Matrix& w,
                      void* out, bool out_f32, int m, cudaStream_t stream);

}  // namespace dgpp
