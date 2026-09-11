#pragma once
// GLM-4.7 attention kernels (2026-09-09, docs/glm47_plan.md §1.2, D4; the
// reference is transformers Glm4MoeAttention with the engine's flash-style
// attention numerics, the QSA/DSA kernels' pin). The projections run
// through the GEMM interface with fp32 outputs; these kernels are the rest:
//
//   qkv finish   per (row, head): bf16(dot + bias) — the Linear's one
//                rounding — then the two-rounding GLM RMSNorm of q and k
//                heads (u = bf16(x * rstd), y = bf16(w * u)), the partial
//                RoPE on the first rotary_dim dims with the reference's
//                bf16 ops (cos/sin bf16 from fp32 pos x inv_freq; pair
//                (i, i + rotary_dim/2) — transformers' rotate_half, not
//                the older glm/glm4 interleaved pairs: x*cos and
//                rotate(x)*sin rounded, the sum rounded), the q heads
//                written to a bf16 buffer and the k/v
//                heads appended to the paged caches at each row's slot;
//   attention    split-KV paged GQA attention: one block per (row, split,
//                kv head), the kv head's query heads sharing every K/V
//                tile, fp32 scores and online softmax, probabilities
//                rounded to bf16 for the V accumulation, the denominator
//                unrounded; a row attends [0, pos] of its request (its own
//                row already appended); rows with pos < 0 are padding;
//   combine      the splits merged, out = bf16(C / L).
//
// Every kernel is deterministic and capturable (fixed grids, no host
// reads). Cache layout: bf16 [slots, kv_heads * dim] per layer (the QSA
// pool's), slot = block_tables[req][pos / block_tokens] * block_tokens +
// pos % block_tokens. dim is 128.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kGlm4HeadDim = 128;

// Rows [0, rows) of the projections' fp32 outputs: q [rows, local_heads *
// dim] (row stride q_stride elements), k / v [rows, kv_heads * dim] (row
// strides k_stride / v_stride). Biases bf16 (null: none), norms bf16 [dim]
// (null: no norm), inv_freq fp32 [rotary_dim / 2] (rotary_dim 0: no RoPE).
// pos / req_ids: device [rows]; a row with pos < 0 writes nothing.
void glm4_qkv_finish(const float* q_dot, int64_t q_stride, const float* k_dot, int64_t k_stride,
                     const float* v_dot, int64_t v_stride, const uint16_t* q_bias,
                     const uint16_t* k_bias, const uint16_t* v_bias, const uint16_t* q_norm,
                     const uint16_t* k_norm, float eps, const float* inv_freq, int rotary_dim,
                     const int32_t* req_ids, const int64_t* pos, int rows, int local_heads,
                     int kv_heads, const int32_t* block_tables, int blocks_per_request,
                     int block_tokens, uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache,
                     uint16_t* v_cache, cudaStream_t stream);

// The partials of row r, split s, head h: m_ws / l_ws fp32 [rows, n_split,
// local_heads], c_ws fp32 [rows, n_split, local_heads, dim]. q: bf16 rows
// of local_heads x dim (row stride q_stride, heads contiguous). Every
// query head h reads kv head h / (local_heads / kv_heads). Split s of a
// row covers tiles [s * chunk, (s + 1) * chunk) of its 32-token tiles
// (chunk = ceil(tiles / n_split)); an empty split writes m = -inf, l = 0.
void glm4_attn_partial(const uint16_t* q, int64_t q_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                       int rows, int n_split, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, float scale,
                       float* m_ws, float* l_ws, float* c_ws, cudaStream_t stream);

// out[r, h * dim + d] = bf16(sum_s e^(m_s - M) c_s / sum_s e^(m_s - M) l_s)
// (0 when every split is empty); out rows local_heads * dim wide.
void glm4_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws, int rows,
                       int n_split, int local_heads, uint16_t* out, cudaStream_t stream);

// Tiles of the attention kernel (the split geometry the host derives).
constexpr int kGlm4AttnTile = 32;

}  // namespace dgpp
