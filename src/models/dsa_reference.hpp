#pragma once
// Host reference implementation of the DSA/MLA sparse-attention layer.
//
// Two roles:
//  1. test oracle for the CUDA kernels (Acc=float mirrors device fp32 math;
//     Acc=double measures fp32 accumulation drift);
//  2. executable specification of the pinned reference numerics — the Python
//     dump harness mirrors the same equations independently, so a
//     disagreement between the two references is itself a bug.
//
// All tensors are host memory. bf16 is carried as raw uint16 bits; fp8-e4m3
// as raw uint8 bits. Boundary rounding points match the device path exactly.
//
// Pinned numerics (vLLM GLM-5 glm-release, GLM-5.3-Flash config):
//   * indexer k: LayerNorm over the wk projection, fp32 compute, eps 1e-6,
//     bf16 output (weight+bias upcast from bf16);
//   * indexer weights: fp32(hidden) @ fp32(wp^T), no bf16 rounding; then
//     weights' = (weights * q_scale) * (128^-0.5 * 32^-0.5) — the q scale
//     folds in first, the combined scale 1/64 is exact in fp32;
//   * indexer q: wq_b(q_lora_normed), Hadamard-128 rotation in fp32,
//     rounded to bf16, then absmax fp8-e4m3 quant with power-of-two scale
//     2^ceil(log2(absmax/448)) and absmax floor 1e-4;
//   * gate: hidden @ gate^T, bf16 GEMM output;
//   * pool compression: per-dimension softmax over the pool's kpool slots
//     with scores fp32(gate) + ape (fp32 table [kpool, 128]); weighted sum
//     of fp32(k) divided by the softmax denominator; bf16 round; Hadamard-128
//     (fp32 butterflies + 1/sqrt(128)); bf16 round; absmax fp8 quant
//     (power-of-two scale, floor 1e-4);
//   * logits: fp32 accumulation of q_fp8 * k_fp8 products, dequantized by
//     the per-pool scale and weighted per head by weights';
//   * selection: the select_k pools with the highest logits; exact ties
//     break to the lower pool index; output ascending (DGPP's deterministic
//     pin — the reference's radix path is atomicAdd-ordered at exact ties);
//   * expansion: selected pools expand to their kpool tokens, then the
//     query's own incomplete tail [kpool*floor((p+1)/kpool), p] is appended;
//     all attended positions are causal by construction;
//   * MLA: q_lora RMSNorm (eps 1e-5) -> q_b; latent = RMSNorm(kv_a, eps
//     1e-5), stored bf16; k-absorbed scores q~=W_uk_h^T q_h (bf16 GEMM
//     output rounding), score = q~ . latent in fp32, softmax fp32, probs
//     rounded to bf16 for the v-accumulation MMA, c_h = sum probs*latent in
//     fp32, out_h = W_uv_h c_h (bf16 GEMM); attention scale 256^-0.5; o_proj
//     bf16 GEMM. Zero RoPE: no positional encoding anywhere.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "models/dsa_geometry.hpp"

namespace dgpp::dsa_ref {

// All-M2-style weight views: host pointers, bf16 as uint16 bits, fp32 as
// float. Indexer weights are replicated (global dims); MLA core weights are
// the local TP views (local_heads from DsaGeometry).
struct HostWeights {
  // Indexer (replicated across TP).
  const uint16_t* wq_b = nullptr;    // [index_n_heads*128, q_lora_rank]
  const uint16_t* wk = nullptr;      // [index_head_dim, hidden]
  const uint16_t* wp = nullptr;      // [index_n_heads, hidden]
  const uint16_t* gate = nullptr;    // [index_head_dim, hidden]
  const uint16_t* k_norm_w = nullptr;  // [index_head_dim]
  const uint16_t* k_norm_b = nullptr;  // [index_head_dim]
  const float* ape = nullptr;        // [kpool, index_head_dim] fp32

  // MLA core (local TP views).
  const uint16_t* qkv_a = nullptr;   // [q_lora+kv_lora, hidden] fused [q_a|kv_a]
  const uint16_t* q_aln = nullptr;   // [q_lora_rank]
  const uint16_t* kv_aln = nullptr;  // [kv_lora_rank]
  const uint16_t* q_b = nullptr;     // [local_heads*nope, q_lora_rank]
  const uint16_t* kv_b = nullptr;    // [local_heads*(nope+v), kv_lora_rank]
  const uint16_t* o_proj = nullptr;  // [hidden, local_heads*v]
};

// Per-request cache state in logical (flat) indexing. The device path's
// block tables map these to physical slots; parity tests walk the block
// table and compare logical content.
struct HostState {
  std::vector<uint8_t> index_k;    // [max_pools, index_head_dim] fp8 bits
  std::vector<float> index_scale;  // [max_pools]
  std::vector<uint16_t> latent;    // [max_tokens, kv_lora_rank]
  std::vector<uint16_t> tail;      // [2, kpool, index_head_dim]
  int64_t num_pools = 0;           // complete pools written
  int64_t num_tokens = 0;          // latent rows written

  void reset(const DsaConfig& cfg, int64_t max_tokens) {
    const int64_t max_pools = (max_tokens + cfg.index_kpool - 1) /
                              cfg.index_kpool;
    index_k.assign(max_pools * cfg.index_head_dim, 0);
    index_scale.assign(max_pools, 0.0f);
    latent.assign(max_tokens * cfg.kv_lora_rank, 0);
    tail.assign(2ull * cfg.index_kpool * cfg.index_head_dim, 0);
    num_pools = 0;
    num_tokens = 0;
  }
};

// out[m,n] = bf16(sum_k act[m,k] * w[n,k]), accumulation in Acc.
template <typename Acc>
void gemm_bf16(const uint16_t* act, int64_t act_row_stride, const uint16_t* w,
               uint16_t* out, int m, int n, int k);

// Hadamard-128 rotation (fp32 butterflies, x * 1/sqrt(128) exact in fp32).
template <typename Acc>
void fwht128(Acc* x);

// The indexer q path: bf16 rows -> fp8 + power-of-two scale.
// q [rows, dim] bf16 bits -> q_fp8 [rows, dim] fp8 bits, q_scale [rows] fp32.
// dim must be 128.
template <typename Acc>
void fwht128_quant_fp8(const uint16_t* q, int rows, int dim, uint8_t* q_fp8,
                       float* q_scale);

// The selection-side indexer inputs for `tokens` hidden rows (the exact
// steps of layer_forward up to and including the weight fold):
// q_fp8 [tokens, heads, dim] fp8 bits and w_folded [tokens, heads] fp32.
// Exported so parity tests can audit divergences with the reference's OWN
// quantized rows (host GEMMs are m-independent, so a row's values do not
// depend on the chunk it was computed in).
template <typename Acc>
void indexer_query_inputs(const HostWeights& w, const DsaConfig& cfg,
                          const uint16_t* hidden_in, int tokens,
                          uint8_t* q_fp8, float* w_folded);

// Compress one pool of kpool tokens into the index cache.
// k/gate: [kpool, dim] bf16 bits; ape: [kpool, dim] fp32;
// writes k_out [dim] fp8 bits + scale_out (single value).
template <typename Acc>
void compress_pool(const uint16_t* k, const uint16_t* gate, const float* ape,
                   int kpool, int dim, uint8_t* k_out, float* scale_out);

// One query's pool logits: logits[j] = sum_h w'[h] * k_scale[j] *
// (sum_d q_fp8[h,d] * k_fp8[j,d]) for j in [0, num_pools). All fp32/Acc.
// q_fp8: [heads, dim] fp8 bits; w: [heads] fp32 (already scale-folded).
template <typename Acc>
void pool_logits(const uint8_t* q_fp8, const float* w, const uint8_t* k_fp8,
                 const float* k_scale, int64_t num_pools, int heads, int dim,
                 float* logits);

// Deterministic selection: the k = min(select_k, num_pools) highest logits,
// exact ties -> lower pool index, output ascending. Returns the count.
// This is DGPP's pinned spec (see file header).
int select_pools(const float* logits, int64_t num_pools, int select_k,
                 int32_t* out_pool_ids);

// Expand selected pools to tokens and append the query's incomplete tail.
// pool_ids: [n_sel] ascending; query position p; writes token ids to
// out_tokens [max_selected], -1 padded; returns the written count.
int expand_append_tail(const int32_t* pool_ids, int n_sel, int64_t pos,
                       int kpool, int max_selected, int32_t* out_tokens);

// Dense causal path (visible_pools <= select_k): all tokens [0, p].
int causal_all_tokens(int64_t pos, int max_selected, int32_t* out_tokens);

// Absorbed MLA attention for one query over selected token positions.
// q: [local_heads, nope] bf16 bits; latent_rows: gathers latent[token] for
// each selected position; kv_b: [local_heads*(nope+v), kv_lora] bf16;
// tokens: [n_sel] positions (-1 skipped); writes out [local_heads, v] bf16.
template <typename Acc>
void absorbed_attn(const uint16_t* q, const uint16_t* latent,
                   int64_t latent_stride, const int32_t* tokens, int n_sel,
                   const uint16_t* kv_b, int local_heads, int nope, int v,
                   int kv_lora, float scale, uint16_t* out);

// Full layer forward over one pool-aligned chunk [token_start,
// token_start + tokens). Updates state (index cache, latent, tail) and
// writes layer_out [tokens, hidden] bf16 and topk [tokens, max_selected]
// int32 (-1 padded). chunk_starts_pool_aligned must hold for continuation
// chunks (the final chunk may end mid-pool; its trailing tokens persist in
// the tail for decode).
template <typename Acc>
void layer_forward(const HostWeights& w, const DsaConfig& cfg,
                   const uint16_t* hidden_in, HostState& state,
                   int64_t token_start, int tokens, uint16_t* layer_out,
                   int32_t* topk_out);

}  // namespace dgpp::dsa_ref
