#pragma once
// One DSA layer's full forward (M3 layer phase, DESIGN §7.2): fused
// [q_a|kv_a] projection, MLA + indexer heads, pool compression, pinned
// top-k selection, absorbed attention, output projection.
//
// Weight layout (device pointers, caller-owned; mirrors dsa_reference.hpp's
// HostWeights field-for-field — the M4 loader materializes them from the
// checkpoint, BF16 except the fp32 APE):
//   Indexer (replicated across TP):
//     wq_b [index_n_heads*128, q_lora_rank], wk [128, hidden],
//     wp [index_n_heads, hidden], gate [128, hidden],
//     k_norm_w/k_norm_b [128] (LayerNorm, eps 1e-6), ape fp32 [kpool, 128].
//   MLA core (local TP views):
//     qkv_a [q_lora+kv_lora, hidden] fused [q_a|kv_a],
//     q_aln [q_lora], kv_aln [kv_lora] (RMSNorm, eps rms_norm_eps),
//     q_b [local_heads*nope, q_lora],
//     kv_b [local_heads*(nope+v), kv_lora] — the checkpoint's interleaved
//       per-head layout (head h owns rows [h*(nope+v), h*(nope+v)+nope) of
//       W_uk and the next v rows of W_uv),
//     o_proj [hidden, local_heads*v] (row-parallel).
//
// Scratch is caller-provided and SHARED by all DSA layers: they run
// sequentially on one stream, so the transient buffers (q, q_tilde, c, the
// attention workspace, the prefill dot buffer) are sized once for the
// largest layer use and reused — per-layer ownership would multiply the
// footprint by num_dsa_layers for zero benefit. Scratch contents are only
// valid until the next enqueue on ANY layer sharing the buffer.
//
// The forward is allocation-free. The decode path is CUDA-graph capturable:
//   * call prepare(tokens) for every row count you will capture BEFORE
//     capture (GEMM plans + the shared-memory opt-in are context state);
//   * decode batch metadata (req_ids/pos/req_spans) must live in
//     caller-owned DEVICE buffers whose contents are re-uploaded between
//     replays — visible-pool counts are derived on device, so one graph
//     serves any position;
//   * block-table growth is host-side admission control: the pool's
//     ensure_request_blocks() must already cover every position a captured
//     batch will write (enqueue_decode cannot check it — pos is device
//     state). enqueue_prefill grows the table itself and throws on pool
//     exhaustion (prefill is a control-path operation).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/dsa_geometry.hpp"

namespace dgpp {

class DsaStatePool;

struct DsaLayerWeights {
  // Indexer (replicated across TP ranks).
  const void* wq_b = nullptr;      // bf16 [index_n_heads*128, q_lora_rank]
  const void* wk = nullptr;        // bf16 [index_head_dim, hidden]
  const void* wp = nullptr;        // bf16 [index_n_heads, hidden]
  const void* gate = nullptr;      // bf16 [index_head_dim, hidden]
  const void* k_norm_w = nullptr;  // bf16 [index_head_dim]
  const void* k_norm_b = nullptr;  // bf16 [index_head_dim]
  const float* ape = nullptr;      // fp32 [kpool, index_head_dim]

  // MLA core (local TP views).
  const void* qkv_a = nullptr;   // bf16 [q_lora+kv_lora, hidden] fused
  const void* q_aln = nullptr;   // bf16 [q_lora_rank]
  const void* kv_aln = nullptr;  // bf16 [kv_lora_rank]
  const void* q_b = nullptr;     // bf16 [local_heads*nope, q_lora_rank]
  const void* kv_b = nullptr;    // bf16 [local_heads*(nope+v), kv_lora_rank]
  const void* o_proj = nullptr;  // bf16 [hidden, local_heads*v]
};

class DsaLayer {
 public:
  // Carves the fixed scratch layout (see scratch_bytes) out of `scratch`,
  // which must be at least scratch_bytes(cfg, ...) bytes. `max_tokens`
  // bounds both prefill chunks and decode batches; `max_cache_tokens`
  // bounds the context length any request will reach (sizes the prefill
  // gather buffer); `max_decode_rows` (<= 8, the select kernel's bound)
  // and `decode_n_split` shape the decode attention split. `dot_budget`
  // bounds the prefill dot buffer; prefill query tiles shrink to fit it,
  // trading K-cache re-reads for a bounded footprint.
  DsaLayer(IGemm& gemm, const DsaLayerWeights& w, const DsaConfig& cfg,
           int max_tokens, int64_t max_cache_tokens, void* scratch,
           size_t scratch_capacity, void* gemm_workspace,
           size_t gemm_ws_bytes, int max_decode_rows = 8,
           int decode_n_split = 32, size_t dot_budget = 64ull << 20);

  // Prebuilds projection-GEMM plans for `tokens` rows and performs the
  // shared-memory opt-in. Must run outside graph capture; returns false
  // when a heuristic is unavailable.
  bool prepare(int tokens);

  // Prebuilds the fp8 dot-GEMM plan for a prefill query tile of
  // `tile_rows` rows against `visible_pools` gathered pools (the plan's n
  // is the internally padded pool count). Eager prefill works without
  // this — matmul builds plans lazily — but a cold plan costs a heuristic
  // query mid-chunk.
  bool prepare_prefill(int tile_rows, int64_t visible_pools);

  // Prefills one pool-aligned chunk [token_start, token_start+tokens) of
  // request `req` (a single request per call: the prefill gather/select
  // path is per-request; the engine loops requests). token_start must be a
  // multiple of kpool (continuation chunks stay pool-aligned; the final
  // chunk may end mid-pool, leaving the tail for decode).
  //   hidden_in: bf16 [tokens, hidden] (must not alias `out`)
  //   out:       bf16 [tokens, hidden]
  // Grows the request's block table via the pool; throws on exhaustion.
  void enqueue_prefill(const void* hidden_in, DsaStatePool& state, int layer,
                       int req, int64_t token_start, int tokens, void* out,
                       cudaStream_t stream);

  // Decodes a batch of `tokens` rows over `num_requests` requests.
  //   hidden_in: bf16 [tokens, hidden] (padding rows: garbage in, garbage
  //              out — kernels skip latent/ring writes for pos < 0)
  //   req_ids:   device int32 [tokens] — request id per row
  //   pos:       device int64 [tokens] — absolute position (-1 = padding)
  //   req_spans: device int32 [num_requests, 2] — (start, len) row spans
  //              into the token batch; a request's rows must be contiguous
  //   out:       bf16 [tokens, hidden]
  // The caller must already have grown the block tables to cover pos+1.
  void enqueue_decode(const void* hidden_in, DsaStatePool& state, int layer,
                      const int32_t* req_ids, const int64_t* pos,
                      const int32_t* req_spans, int num_requests, int tokens,
                      void* out, cudaStream_t stream);

  const DsaConfig& config() const { return cfg_; }
  const DsaGeometry& geometry() const { return geo_; }
  int max_tokens() const { return max_tokens_; }

  // Test/diagnostic probes into the shared scratch; valid until the next
  // enqueue on any layer sharing it.
  const void* debug_attn_out() const { return attn_out_; }   // [rows, local_v_rows] bf16
  const int32_t* debug_topk() const { return topk_; }        // [rows, max_selected]
  const int32_t* debug_counts() const { return counts_; }    // [rows]
  const void* debug_q_fp8() const { return q_fp8_; }         // [rows*heads, 128] fp8
  const float* debug_w_folded() const { return w_folded_; }  // [rows, heads]
  const void* debug_k_rows() const { return k_rows_; }       // [rows, 128] bf16
  const void* debug_gate_rows() const { return gate_rows_; } // [rows, 128] bf16
  // The last prefill tile's fp8-dot buffer [tile_rows*heads, padded_n] and
  // its pool stride — the exact dots the prefill select consumed for that
  // tile's rows (parity audits consume these bit-exact; the host cannot
  // reproduce a tensor-core reduction order). Zero rows for decode paths.
  const float* debug_dots() const { return dot_; }
  int64_t debug_dot_stride() const { return dot_stride_last_; }

  // Scratch bytes for the given shape (each region 256-byte aligned). One
  // buffer of this size serves every DSA layer.
  static size_t scratch_bytes(const DsaConfig& cfg, int max_tokens,
                              int64_t max_cache_tokens, int max_decode_rows = 8,
                              int decode_n_split = 32,
                              size_t dot_budget = 64ull << 20);

 private:
  // Offsets of every scratch region; computed once and shared by the
  // constructor and scratch_bytes so the formula cannot drift.
  struct ScratchLayout {
    size_t total = 0;
    size_t off_qkv = 0, off_q_c = 0, off_kv_c = 0, off_q = 0, off_q_idx = 0;
    size_t off_k_raw = 0, off_k_rows = 0, off_gate = 0, off_weights = 0;
    size_t off_q_fp8 = 0, off_q_scale = 0, off_w_folded = 0, off_topk = 0;
    size_t off_counts = 0, off_pos = 0, off_req_ids = 0, off_attn_out = 0;
    size_t off_q_tilde = 0, off_c = 0, off_m = 0, off_l = 0, off_cws = 0;
    size_t off_dot = 0, off_gather_k = 0, off_gather_scale = 0;
    size_t off_partial = 0, off_counter = 0;
    int64_t max_pools = 0;   // gather buffer capacity, in pools
    int tile_cap = 0;        // max prefill query-tile rows
  };
  static ScratchLayout scratch_layout(const DsaConfig& cfg, int max_tokens,
                                      int64_t max_cache_tokens,
                                      int max_decode_rows, int decode_n_split,
                                      size_t dot_budget);

  // The projection chain shared by both paths: fused qkv GEMM, split
  // RMSNorms, q_b/wq_b projections, indexer k (LayerNorm) + gate + fp32
  // weights, Hadamard quant, weight fold. Fills q_, q_fp8_, w_folded_,
  // kv_c_, k_rows_, gate_. Latent append and cache writes are path-specific.
  void project_common(const void* hidden_in, int tokens, cudaStream_t stream);

  // absorb + split-KV attention + v-absorb for `rows` query rows starting
  // at chunk-relative row `row0`, in attention tiles of attn_rows_ rows;
  // topk_/counts_ must already hold the tile's selection. `req_ids` is
  // indexed tile-locally (entry r = request of row row0 + r). n_split is
  // the split-KV parallelism for this call (decode: rows are scarce, split
  // wide; prefill: 8-row tiles already parallelize, split narrow).
  void attend_tile(DsaStatePool& state, int layer, const int32_t* req_ids,
                   int64_t row0, int rows, int n_split, cudaStream_t stream);

  void validate_pool(const DsaStatePool& state, int layer) const;

  IGemm& gemm_;
  DsaLayerWeights w_;
  DsaConfig cfg_;
  DsaGeometry geo_;
  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int max_decode_rows_ = 8;
  int decode_n_split_ = 32;  // split-KV parallelism: rows x n_split x 4 head-groups ~ 128 blocks/row
  int attn_rows_ = 8;         // attention tile rows = max(max_decode_rows_, 8)
  int tile_cap_ = 0;          // prefill dot-tile rows (dot-budget bound)
  int64_t max_pools_ = 0;     // gather/dot capacity, in padded pools
  int64_t gather_zeroed_ = 0; // gather high-water mark already zeroed
  float logit_scale_ = 0.f;   // 128^-0.5 * heads^-0.5, reference-pinned
  float attn_scale_ = 0.f;    // nope^-0.5
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;

  // Fixed-address scratch (carved from the caller's shared buffer).
  uint8_t* scratch_ = nullptr;
  uint16_t* qkv_ = nullptr;      // [T, q_lora+kv_lora]
  uint16_t* q_c_ = nullptr;      // [T, q_lora]
  uint16_t* kv_c_ = nullptr;     // [T, kv_lora]
  uint16_t* q_ = nullptr;        // [T, local_heads*nope]
  uint16_t* q_idx_ = nullptr;    // [T, index_heads*128]
  uint16_t* k_raw_ = nullptr;    // [T, 128]
  uint16_t* k_rows_ = nullptr;   // [T, 128] (LayerNormed)
  uint16_t* gate_rows_ = nullptr;  // [T, 128]
  float* weights_ = nullptr;     // [T, index_heads]
  uint8_t* q_fp8_ = nullptr;     // [T*index_heads, 128]
  float* q_scale_ = nullptr;     // [T*index_heads]
  float* w_folded_ = nullptr;    // [T*index_heads]
  int32_t* topk_ = nullptr;      // [T, max_selected]
  int32_t* counts_ = nullptr;    // [T]
  int64_t* pos_dev_ = nullptr;   // [T] (prefill staging target)
  int32_t* req_ids_dev_ = nullptr;  // [T] (prefill staging target)
  uint16_t* attn_out_ = nullptr;  // [T, local_heads*v]
  uint16_t* q_tilde_ = nullptr;  // [tile_cap, local_heads*kv_lora]
  float* c_ = nullptr;           // [tile_cap, local_heads*kv_lora]
  float* m_ws_ = nullptr;        // [tile_cap, n_split, local_heads]
  float* l_ws_ = nullptr;        // [tile_cap, n_split, local_heads]
  float* c_ws_ = nullptr;        // [tile_cap, n_split, local_heads, kv_lora]
  float* dot_ = nullptr;         // [tile_rows*index_heads, padded_n]
  uint8_t* gather_k_ = nullptr;  // [max_pools, 128] fp8
  float* gather_scale_ = nullptr;  // [max_pools]
  uint64_t* partial_ws_ = nullptr;  // [grid, max_decode_rows, select_k]
  int32_t* counter_ws_ = nullptr;   // [1]
  int64_t dot_stride_last_ = 0;     // padded pool count of the last dot tile

  // Prefill host staging (pos / req_ids uploads; prefill is never captured).
  std::vector<int64_t> pos_host_;
  std::vector<int32_t> req_ids_host_;
};

}  // namespace dgpp
