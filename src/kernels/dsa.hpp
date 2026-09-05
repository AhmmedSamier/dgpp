#pragma once
// DSA/MLA sparse-attention forward kernels, DESIGN §7.2.
//
// Semantics are pinned to the vLLM GLM-5 reference (glm-release fork,
// kpool_compress.py / sparse_attn_indexer_kpool.py); see dsa_reference.hpp
// for the numeric contract this mirrors and the documented power-of-two
// scale divergence.
//
// Cache layout (DGPP's own, not the reference's packed 132-byte pages):
//   * index K cache: planar. `index_k` is FP8-E4M3 [total_slots, 128] and
//     `index_scale` is FP32 [total_slots], one slot per pool. Rows are
//     128-byte aligned, so the decode streaming path reads perfectly aligned
//     lines.
//   * latent cache: BF16 [total_token_slots, kv_lora_rank] rows.
//   * tail cache: BF16 [max_requests, 2, kpool, 128] — half 0 is raw K,
//     half 1 the gate score; a per-request ring indexed by pos % kpool.
//   * one shared block table per request maps logical to physical slots:
//     block_tables[req][token_block] for the latent cache, and the same
//     entry's pool range [token_block*kpool .. +kpool) for the index cache
//     (co-located blocks, DESIGN §8).
//
// Decode is the hot path (single-stream goal): the fused select kernel
// streams the index cache once, keeps a running top-select_k composite-key
// selection in shared memory ((sortable_fp32 << 21) | pool_idx — a total
// order, so exact ties resolve to the lower pool index by construction),
// and merges block partials with a last-block reduction. No logits are
// materialized and no per-step allocation happens; grid sizes are fixed
// with grid-stride loops over device-visible counts, so the whole decode
// path is CUDA-graph capturable.
//
// Prefill materializes per-(row,head) fp8 dots through the IGemm seam
// (FP8 x FP8 -> F32, unit scales) into a bounded dot buffer and runs the
// same streaming selection over it per row.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// ---- indexer elementwise paths --------------------------------------

// Hadamard-128 + per-row absmax FP8 quant (the indexer q path).
//   q_bf16: [rows, 128] bf16 bits; q_fp8: [rows, 128] fp8 bits;
//   q_scale: [rows] fp32 (power-of-two scales).
void dsa_fwht_quant_rows(const void* q_bf16, int64_t rows, void* q_fp8,
                         float* q_scale, cudaStream_t stream);

// w_folded[i] = (weights[i] * q_scale[i]) * scale, elementwise fp32.
void dsa_fold_weights(const float* weights, const float* q_scale, float* out,
                      int64_t n, float scale, cudaStream_t stream);

// Full LayerNorm (fp32 compute, weight+bias upcast) on a strided input —
// the indexer's k_norm (eps 1e-6).
//   k_raw: [rows, dim] with row stride k_stride elements; k_out: [rows, dim]
//   contiguous bf16.
void dsa_k_layernorm(const void* k_raw, int64_t k_stride, const void* w,
                     const void* b, void* k_out, int64_t rows, int dim,
                     float eps, cudaStream_t stream);

// Fused q_a/kv_a RMSNorm over the split halves of the fused [q_a|kv_a]
// projection (eps 1e-5).
//   qkv: [rows, q_dim + kv_dim] bf16; q_c/kv_c: [rows, dim] contiguous.
void dsa_fused_qkv_rmsnorm(const void* qkv, void* q_c, void* kv_c, int q_dim,
                           int kv_dim, int64_t rows, const void* q_w,
                           const void* kv_w, float eps, cudaStream_t stream);

// ---- index cache + tail state machinery ------------------------------

// Prefill pool compression: one block per pool. Pools [first_pool,
// first_pool + n_pools) are complete pools whose kpool tokens start at
// chunk row (i * kpool) for i in [0, n_pools). Physical slot via the
// request's block table.
//   k/gate: bf16 [tokens, dim], row strides in elements;
//   block_table: int32 [blocks_per_request] (single request);
//   index_k/index_scale: planar cache, physical slot indexing.
void dsa_kpool_compress_write(const void* k, int64_t k_stride,
                              const void* gate, int64_t gate_stride,
                              const float* ape, const int32_t* block_table,
                              int pools_per_block, int64_t first_pool,
                              int n_pools, void* index_k, float* index_scale,
                              int kpool, int dim, cudaStream_t stream);

// Seed the per-request tail ring with each request's last kpool tokens of
// the batch (the reference's ahead-check rule). One block per token.
//   tail: bf16 [max_requests, 2, kpool, dim].
void dsa_kpool_tail_seed(const void* k, int64_t k_stride, const void* gate,
                         int64_t gate_stride, const int32_t* req_ids,
                         const int64_t* pos, int64_t tokens, void* tail,
                         int kpool, int dim, cudaStream_t stream);

// Decode update: one block per request, tokens processed in position order
// (read-after-stash within the call). Each token stashes raw K + gate into
// the ring; a token completing its pool (pos % kpool == kpool-1) compresses
// the ring (with the current token overriding its ring slot, exactly the
// reference's is_current rule) and writes the pool to the block-mapped slot.
//   req_spans: int32 [num_requests, 2] (start, len) into the token batch;
//   eager callers may pass only active spans, while a fixed-shape graph may
//   include an all-padding span for every unoccupied configured slot. req_ids
//   selects the actual state slot (active slots need not be
//   0..num_requests-1). Tokens of a request must be batch-contiguous; pos:
//   [tokens] (may be -1 for padding rows — skipped). An all-padding span
//   touches nothing; its req_ids may be a sentinel.
//   tail_snapshots (optional, speculative decode): bf16 [tokens, 2, kpool,
//   dim]; after every batch row t that is not its request's last, the
//   request's ring as it stands is copied to row t. Rolling a request back
//   to `a` accepted rows = copying row (start + a - 1) over its ring.
void dsa_kpool_decode_update(const void* k, int64_t k_stride,
                             const void* gate, int64_t gate_stride,
                             const float* ape, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans,
                             int num_requests,
                             const int32_t* block_tables,
                             int blocks_per_request, void* tail,
                             void* index_k, float* index_scale,
                             int pools_per_block, int kpool, int dim,
                             cudaStream_t stream,
                             void* tail_snapshots = nullptr);

// Writes zeros to every output row whose position is negative (a fixed-shape
// padding row). The decode kernels skip such rows' state writes and leave
// their attention output unwritten, so without this the row's block output
// is whatever the scratch held; zeroing makes a padding row inert by
// construction (deterministic, finite, rank-identical) rather than merely
// unobserved downstream.
void dsa_zero_padding_rows(void* out, const int64_t* pos, int tokens,
                           int hidden, cudaStream_t stream);

// Append normed latent rows to the blocked latent cache. One block per row.
//   latent_rows: bf16 [tokens, kv_lora]; block_tables: int32
//   [max_requests, blocks_per_request].
void dsa_latent_append(const void* latent_rows, const int32_t* req_ids,
                       const int64_t* pos, int64_t tokens,
                       const int32_t* block_tables, int blocks_per_request,
                       int block_tokens, void* latent_cache, int kv_lora,
                       cudaStream_t stream);

// Gather a request's pools [0, n_pools) into a contiguous buffer for the
// prefill logits GEMM.
void dsa_gather_index_pools(const int32_t* block_table, int pools_per_block,
                            const void* index_k, const float* index_scale,
                            int64_t n_pools, void* out_k, float* out_scale,
                            int dim, cudaStream_t stream);

// ---- top-k selection --------------------------------------------------
//
// Pinned spec: the select_k pools with the highest fp32 logits, exact ties
// broken to the lower pool index, output ascending in pool index. The
// composite sort key (~sortable_fp32 << 21) | pool_idx is a total order, so
// the selection is deterministic on any correct implementation and identical
// across the decode, prefill, and merge paths. Finite logits are assumed:
// real quantized cache rows are always finite (the saturating fp8 encoder
// never mints NaN), and NaN logits would sort as if near +infinity.

// Opts the select-decode and attention kernels into large dynamic shared
// memory (idempotent, cheap after the first call). Call it once per process
// before CUDA graph capture of the decode path — cudaFuncSetAttribute is a
// context mutation, not a stream operation, and must not happen inside
// capture. DsaLayer::prepare()/enqueue*() call this automatically; a bare
// kernel consumer that captures graphs should call it explicitly.
void dsa_prepare_kernel_smem();


// Fused decode select: streams pools [0, visible(pos[r])) per row straight
// from the blocked index cache, computes pool logits inline (warp per pool,
// lane per head), and keeps a running top-select_k composite-key selection.
// Fixed grid, grid-stride over pools; the last block merges partials,
// extracts ids ascending, expands pools to tokens, and appends each row's
// incomplete tail. Workspace must be zeroed once at allocation
// (counter_ws self-resets after each call).
//   q_fp8: [rows, heads, dim] fp8; w_folded: [rows, heads] fp32;
//   block_tables: int32 [max_requests, blocks_per_request]; req_ids: [rows];
//   pos: [rows]; topk_out: int32 [rows, max_selected] (-1 padded);
//   out_counts: int32 [rows];
//   partial_ws: uint64 [grid_blocks * rows * select_k];
//   counter_ws: int32 [1].
// rows <= 8 (decode/MTP batch bound).
void dsa_select_decode(const void* q_fp8, const float* w_folded,
                       const int32_t* req_ids, const int64_t* pos, int rows,
                       const int32_t* block_tables, int blocks_per_request,
                       const void* index_k, const float* index_scale,
                       int pools_per_block, int heads, int dim, int select_k,
                       int kpool, int max_selected, int32_t* topk_out,
                       int32_t* out_counts, uint64_t* partial_ws,
                       int32_t* counter_ws, int grid_blocks,
                       cudaStream_t stream);

// Prefill select: one block per row over the materialized dot buffer.
//   dot: fp32 [rows * heads, dot_stride] (row r head h at
//   dot[r*heads + h]); k_scale: fp32 [n_pools] contiguous (gathered);
//   pos: [rows]; visible per row is derived on device.
void dsa_select_prefill(const float* dot, int64_t dot_stride,
                        const float* w_folded, const float* k_scale,
                        const int64_t* pos, int rows, int64_t n_pools,
                        int heads, int select_k, int kpool, int max_selected,
                        int32_t* topk_out, int32_t* out_counts,
                        cudaStream_t stream);

// ---- MLA absorbed attention -------------------------------------------

// Absorbed query: q_tilde[r,h,c] = sum_d q[r,h,d] * W_uk[h,d][c], bf16
// output rounding (the production absorbed-MLA prep). kv_b is the
// checkpoint's interleaved layout [local_heads*(nope+v), kv_lora]: head h
// owns rows [h*(nope+v), h*(nope+v)+nope) for W_uk.
//   q: bf16 [rows, local_heads * nope];
//   q_tilde: bf16 [rows, local_heads * kv_lora].
void dsa_absorb_q(const void* q, const void* kv_b, void* q_tilde,
                  int64_t rows, int local_heads, int nope, int v, int kv_lora,
                  cudaStream_t stream);

// Split-KV sparse attention over gathered latent rows (flash-decoding
// shape: the decode hot path). One block per (row, split, head-group).
//   q_tilde: bf16 [rows, local_heads * kv_lora];
//   topk: int32 [rows, topk_stride]; counts: int32 [rows];
//   m_ws/l_ws: fp32 [rows, n_split, local_heads];
//   c_ws: fp32 [rows, n_split, local_heads, kv_lora].
void dsa_attn_partial(const void* q_tilde, const void* latent_cache,
                      const int32_t* req_ids, const int32_t* topk,
                      int topk_stride, const int32_t* counts, int rows,
                      int n_split, int local_heads, int kv_lora,
                      int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale,
                      float* m_ws, float* l_ws, float* c_ws,
                      cudaStream_t stream);

// Dense causal attention on tensor cores (2026-09-05): the prefill path
// below index_topk tokens of context, where the selection is provably
// dense (every visible pool selected, the tail appended) and a query at
// position p attends to tokens [0, p]. Same partial layout and split
// semantics as dsa_attn_partial — combine with dsa_attn_combine. The M
// dimension is (row, head) pairs; 32 per block. Returns false without
// launching when kv_lora is not 512 or 256 (the caller keeps the split
// kernel). Tolerance-equal to the split kernel (a different summation
// order), deterministic.
//   pos: int64 [rows] — the rows' token positions (causal bound per row).
bool dsa_attn_dense(const void* q_tilde, const void* latent_cache,
                    const int32_t* req_ids, const int64_t* pos, int rows,
                    int n_split, int local_heads, int kv_lora, int block_tokens,
                    const int32_t* block_tables, int blocks_per_request,
                    float scale, float* m_ws, float* l_ws, float* c_ws,
                    cudaStream_t stream);

// The same kernel over each row's SELECTED tokens (2026-09-05, the sparse
// regime past index_topk tokens of context): topk/counts as
// dsa_attn_partial takes them, each 16-row slab (one query row) walking its
// own list. Requires local_heads a multiple of 16 (returns false otherwise,
// as for kv_lora outside 512/256); combine with dsa_attn_combine.
bool dsa_attn_listed(const void* q_tilde, const void* latent_cache,
                     const int32_t* req_ids, const int32_t* topk, int topk_stride,
                     const int32_t* counts, int rows, int n_split, int local_heads,
                     int kv_lora, int block_tokens, const int32_t* block_tables,
                     int blocks_per_request, float scale, float* m_ws, float* l_ws,
                     float* c_ws, cudaStream_t stream);

// Merge the split partials into normalized c rows: c_out[r,h,:] =
// (sum_s p_s * c_s) / (sum_s p_s * l_s), p_s = exp(m_s - max m).
void dsa_attn_combine(const float* m_ws, const float* l_ws, const float* c_ws,
                      int rows, int n_split, int local_heads, int kv_lora,
                      float* c_out, cudaStream_t stream);

// v-absorbed output: out[r, h*v+d] = sum_c W_uv[h,d][c] * c[r,h,c]. bf16
// out; c is fp32. kv_b interleaved: head h's W_uv rows are
// [h*(nope+v)+nope, (h+1)*(nope+v)).
//   c: fp32 [rows, local_heads * kv_lora];
//   out: bf16 [rows, local_heads * v].
void dsa_vout_gemm(const void* c, const void* kv_b, void* out,
                   int64_t rows, int local_heads, int nope, int v,
                   int kv_lora, cudaStream_t stream);

}  // namespace dgpp
