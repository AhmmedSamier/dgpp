#pragma once
// MiMo-V2.6-Flash attention kernels (2026-09-22, docs/mimo_v26_flash_plan.md
// §1.3, D3; the reference is the release's MiMoV2Attention with the
// engine's flash-style attention numerics — the GLM-4.7 kernels' pin, which
// these fork for the 192-wide qk / 128-wide v heads, the fused pre-sharded
// projection, the sliding window and the attention sink). The fused qkv
// projection runs through the fp8 GEMV / tile cores with fp32 outputs;
// these kernels are the rest:
//
//   qkv finish   per (row, head): bf16(dot) — the Linear's one rounding —
//                then, for the q and k heads, the partial RoPE on the first
//                rotary_dim = 64 dims with the reference's bf16 ops (cos/sin
//                bf16 from fp32 pos x inv_freq; pair (i, i + 32) —
//                transformers' rotate_half over the rotary slice: x*cos and
//                rotate(x)*sin rounded, the sum rounded), the q heads written
//                to a bf16 buffer and the k heads appended to the paged K
//                cache; the v heads scaled — bf16(bf16(dot) x value_scale),
//                the reference's `value_states * v_scale` before the cache —
//                and appended to the V cache at each row's slot. The fp32
//                source is the fused projection's output: `chunks` chunks of
//                `chunk_stride` columns, each [Q (q_per_chunk x 192) | K
//                (kv_per_chunk x 192) | V (kv_per_chunk x 128)] — the
//                checkpoint's pre-sharded layout, so no reordering is loaded;
//   attention    split-KV paged GQA attention: one block per (row, split,
//                kv head), the kv head's query heads sharing every K/V
//                tile, fp32 scores and online softmax, probabilities
//                rounded to bf16 for the V accumulation, the denominator
//                unrounded; a row attends [max(0, pos - window + 1), pos] of
//                its request (window 0: [0, pos]; its own row already
//                appended); the sink of query head h — one more softmax
//                column with logit sink[h] and no value, the reference's
//                `cat([scores, sink])` then `probs[..., :-1]` — enters split
//                0's running (max, sum) before its first tile, so the
//                combine's algebra needs nothing else; rows with pos < 0 are
//                padding;
//   combine      the splits merged, out = bf16(C / L).
//
// Every kernel is deterministic and capturable (fixed grids, no host
// reads). Cache layout: bf16 K [slots, kv_heads * 192] and V [slots,
// kv_heads * 128] per layer, slot = block_tables[req][pos / block_tokens]
// * block_tokens + pos % block_tokens.
//
// The fp8 cache (k_scale / v_scale non-null; engine.kv_dtype fp8): the
// same rows as e4m3 codes (one byte per element at the same address
// arithmetic) with one fp32 scale per (slot, kv head) — the fp8 row form of
// kernels/latent_format.hpp: scale = absmax / 448 over the head's finished
// bf16 values, code = e4m3(x x 448 / absmax). The finish quantizes the
// values the bf16 cache would have stored; the attention dequantizes a
// tile as bf16(e4m3 x scale) into the same shared tiles, so its chain over
// the dequantized rows is the bf16 kernel's, bitwise.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kMimoQkDim = 192;
constexpr int kMimoVDim = 128;
constexpr int kMimoRotaryDim = 64;
// Tiles of the attention kernel (the split geometry the host derives).
constexpr int kMimoAttnTile = 32;

// Where the fused projection's fp32 output keeps each head on this rank.
struct MimoQkvLayout {
  int chunks = 1;              // the checkpoint chunks the rank holds
  int64_t chunk_stride = 0;    // fp32 columns per chunk (the padded chunk rows)
  int q_per_chunk = 16;        // query heads per chunk
  int kv_per_chunk = 1;        // kv heads per chunk
};

// Rows [0, rows) of the fused projection's fp32 output `qkv` (row stride
// qkv_stride elements, the layout above). inv_freq fp32 [32] (the rotary
// slice's). value_scale: the reference's attention_value_scale (1: none).
// pos / req_ids: device [rows]; a row with pos < 0 writes nothing.
// k_scale / v_scale: the fp8 cache's scale planes [slots, kv_heads] (both
// null: the bf16 cache).
void mimo_qkv_finish(const float* qkv, int64_t qkv_stride, const MimoQkvLayout& layout,
                     const float* inv_freq, float value_scale, const int32_t* req_ids,
                     const int64_t* pos, int rows, int local_heads, int kv_heads,
                     const int32_t* block_tables, int blocks_per_request, int block_tokens,
                     uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache, uint16_t* v_cache,
                     cudaStream_t stream, float* k_scale = nullptr, float* v_scale = nullptr);

// The partials of row r, split s, head h: m_ws / l_ws fp32 [rows, n_split,
// local_heads], c_ws fp32 [rows, n_split, local_heads, 128]. q: bf16 rows
// of local_heads x 192 (row stride q_stride, heads contiguous). Every
// query head h reads kv head h / (local_heads / kv_heads). A row's visible
// tokens are [max(0, pos - window + 1), pos] (window <= 0: from 0); split
// s of a row covers tiles [s * chunk, (s + 1) * chunk) of its 32-token
// tiles counted from the first visible token (chunk = ceil(tiles /
// n_split)); an empty split writes m = -inf, l = 0. sink: fp32
// [local_heads] or null.
void mimo_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                       int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, int window,
                       float scale, const float* sink, float* m_ws, float* l_ws, float* c_ws,
                       cudaStream_t stream, const float* k_scale = nullptr,
                       const float* v_scale = nullptr);

// out[r, h * 128 + d] = bf16(sum_s e^(m_s - M) c_s / sum_s e^(m_s - M) l_s)
// (0 when every split is empty); out rows local_heads * 128 wide.
void mimo_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                       int n_split, int local_heads, uint16_t* out, cudaStream_t stream);

// The three kernels fused for a decode batch (plan §7.1): one launch that
// finishes the queries in place, appends the rows' K/V, runs the split
// partials and combines them (the last block of each (row, kv head) to
// arrive on `counters`, an int per (row, kv head) that the kernel leaves
// zero again — allocate zeroed, never touch between launches). The batch
// rows of one request may attend each other's tokens: the kernel finishes
// those K/V from `qkv` itself instead of reading the cache (which other
// blocks of the launch are writing). Output bitwise the chain's. Rows at
// most kMimoAttnFusedMaxRows; four to 16 query heads per kv head.
constexpr int kMimoAttnFusedMaxRows = 64;

struct MimoAttnFusedArgs {
  const float* qkv = nullptr;
  int64_t qkv_stride = 0;
  MimoQkvLayout layout;
  const float* inv_freq = nullptr;
  float value_scale = 1.0f;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int n_split = 1;
  int local_heads = 0;
  int kv_heads = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int block_tokens = 0;
  int window = 0;
  float scale = 1.0f;
  const float* sink = nullptr;
  uint16_t* k_cache = nullptr;
  uint16_t* v_cache = nullptr;
  float* k_scale = nullptr;  // fp8 cache: both planes; bf16: both null
  float* v_scale = nullptr;
  float* m_ws = nullptr;     // [rows, n_split, local_heads]
  float* l_ws = nullptr;
  float* c_ws = nullptr;     // [rows, n_split, local_heads, 128]
  int* counters = nullptr;   // [rows, kv_heads], zero between launches
  uint16_t* out = nullptr;   // [rows, local_heads * 128]
};

size_t mimo_attn_fused_smem_bytes(int local_heads, int kv_heads);
void mimo_attn_fused(const MimoAttnFusedArgs& a, cudaStream_t stream);

// The query-tiled prefill attention (plan §7.2; kernels/mimo_attn_prefill.cu):
// a block per (tile of 64 / hpk consecutive rows, kv head) stages each
// 32-token K/V tile once for its 64 query vectors and runs the scores and
// the PV products on the tensor cores (bf16 x bf16, fp32 accumulation —
// the tensor core's summation order, so not bitwise the split-KV
// kernel's rows; the terms are the same exact products). Rows attend
// [max(0, pos - window + 1), pos] of their request; a row with pos < 0 is
// padding (zero output); rows of several requests in one tile run
// separately. q: the finish's bf16 rows (local_heads x 192, stride
// q_stride); out rows local_heads x 128. hpk must divide 64.
struct MimoAttnPrefillArgs {
  const uint16_t* q = nullptr;
  int64_t q_stride = 0;
  const uint16_t* k_cache = nullptr;
  const uint16_t* v_cache = nullptr;
  const float* k_scale = nullptr;  // fp8 cache: both planes; bf16: both null
  const float* v_scale = nullptr;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  int rows = 0;
  int local_heads = 0;
  int kv_heads = 0;
  int block_tokens = 0;
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int window = 0;
  float scale = 1.0f;
  const float* sink = nullptr;
  uint16_t* out = nullptr;
};

size_t mimo_attn_prefill_smem_bytes();
int mimo_attn_prefill_rows_per_block(int local_heads, int kv_heads);
void mimo_attn_prefill(const MimoAttnPrefillArgs& a, cudaStream_t stream);

}  // namespace dgpp
