#pragma once
// DSA/MLA sparse-attention geometry contract, DESIGN §7.2.
//
// Shapes and byte formulas below are the single source of truth for cache
// sizing, admission accounting, and the "measured bytes within 2% of the
// formula" M3 exit criterion. They reproduce the DESIGN §7.2 table exactly:
// at 300,000 cached tokens the per-rank budget is ~3.379 GB MLA latent +
// ~0.109 GB pooled index + 22.5 KB/request incomplete tails.
//
// Reference semantics pinned to the vLLM GLM-5 kernels (glm-release fork):
//   * index_kpool=4 — one compressed index entry per four input tokens;
//     pools are scored per query, selected pools expand back to token
//     positions, and the valid incomplete tail is always appended
//     (index_kpool_always_select_tail=true).
//   * qk_rope_head_dim=0 for GLM-5.3-Flash: the MLA q/k path is entirely
//     nope — there is no RoPE anywhere in a Flash DSA layer.
//   * visible pools for a query at position p: floor((p+1)/kpool); the tail
//     [kpool*floor(L/kpool), L) is never scored, only appended.
//
// The full GLM-5.3 (2026-09-12, docs/glm53_plan.md D3/D4/D8) is the same
// contract with three more knobs: a 64-wide decoupled RoPE key beside each
// latent row (qk_rope_head_dim 64: the cache row is [kv_lora in the cache
// format | rope bf16], the score runs over both, the value over kv_lora),
// per-token selection (index_kpool 1: a pool IS a token, no compression
// gate, no tail), a ReLU on each indexer head's score (index_relu), and
// index caches on a subset of the DSA layers (num_index_layers: the
// "shared" layers attend with the last "full" layer's selection and own a
// latent cache only). Flash is rope 0 / kpool 4 / relu 0 / every layer
// indexed — the same code paths with a zero-width tail.
//
// Layout decisions that are ours (not the reference's): the index cache is
// two planar arrays — FP8-E4M3 K rows and FP32 scales — instead of the
// reference's packed 132-byte pages (a DeepGEMM block_kv constraint we do
// not inherit). One shared block table co-locates latent blocks and index
// blocks so prefix attachment shares both by reference.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/latent_format.hpp"

namespace dgpp {

// Checkpoint-level DSA/MLA configuration. Defaults are the GLM-5.3-Flash
// values from config.json (text_config): 64 attention heads, q_lora 1536,
// kv_lora 512, nope 256, rope 0, v 256, indexer 32 heads x 128 dim,
// index_topk 2048, index_kpool 4.
struct DsaConfig {
  int hidden = 4096;
  int num_heads = 64;        // MLA heads, head-sharded across TP
  int q_lora_rank = 1536;
  int kv_lora_rank = 512;
  int qk_nope_head_dim = 256;
  int qk_rope_head_dim = 0;  // 0 (Flash) or 64 (the full model's decoupled RoPE)
  int v_head_dim = 256;
  int index_n_heads = 32;    // indexer heads, replicated across TP
  int index_head_dim = 128;
  int index_topk = 2048;     // selected TOKENS per query (topk_tokens)
  int index_kpool = 4;       // tokens per compressed index pool (1 = per-token)
  int always_select_tail = 1;  // index_kpool_always_select_tail
  int index_relu = 0;        // relu(q_h . k) per indexer head before the weighted sum
  int num_dsa_layers = 11;   // of the 45 text layers ([3,7,...,43])
  // DSA layers that own an index cache (and run a selection); 0 = every
  // layer. The others reuse the last indexed layer's selection (plan D4)
  // and hold a latent cache only.
  int num_index_layers = 0;
  int block_tokens = 128;    // shared-block granularity, in tokens
  int tp_size = 1;
  float rms_norm_eps = 1e-5f;  // q_a/kv_a layernorms (config rms_norm_eps)
  // The latent cache's storage format (2026-09-06, the KV dtype knob):
  // bf16 (the parity gates' format), fp8 (e4m3 + a row scale) or fp4
  // (e2m1 in blocks of 16 + a row scale) — kernels/latent_format.hpp. The
  // rope tail (when present) is bf16 in every format.
  LatentFormat latent_format = LatentFormat::kBf16;
  // The fp8 projections' lowering (o_proj, q_a / kv_a, q_b in the
  // quantized form): rows from this count take the streaming tensor-core
  // GEMM (scale_gemm.hpp's mma_from_rows; 0: the GEMV chunks to 128 rows).
  int gemm_mma_from_rows = 0;

  int index_layers() const {
    return num_index_layers > 0 ? num_index_layers : num_dsa_layers;
  }

  static void validate_config(const DsaConfig& c) {
    auto fail = [](const char* what) {
      throw std::invalid_argument(std::string("DsaConfig: ") + what);
    };
    if (c.hidden <= 0) fail("hidden must be positive");
    if (c.num_heads <= 0) fail("num_heads must be positive");
    if (c.q_lora_rank <= 0) fail("q_lora_rank must be positive");
    if (c.kv_lora_rank <= 0) fail("kv_lora_rank must be positive");
    if (c.qk_nope_head_dim <= 0) fail("qk_nope_head_dim must be positive");
    if (c.v_head_dim <= 0) fail("v_head_dim must be positive");
    if (c.index_n_heads <= 0) fail("index_n_heads must be positive");
    if (c.index_head_dim != 128)
      fail("index_head_dim must be 128 (Hadamard-128 is pinned)");
    if (c.index_topk <= 0) fail("index_topk must be positive");
    if (c.index_kpool != 1 && c.index_kpool != 2 && c.index_kpool != 4 &&
        c.index_kpool != 8)
      fail("index_kpool must be 1, 2, 4, or 8 (kernel template dispatch)");
    if (c.index_topk % c.index_kpool != 0)
      fail("index_topk must be a multiple of index_kpool");
    if (c.num_dsa_layers <= 0) fail("num_dsa_layers must be positive");
    if (c.num_index_layers < 0 || c.num_index_layers > c.num_dsa_layers)
      fail("num_index_layers must be in [0, num_dsa_layers]");
    if (c.tp_size <= 0) fail("tp_size must be positive");
    if (c.num_heads % c.tp_size != 0) fail("num_heads must divide by tp_size");
    // The rope tail rides the latent row as bf16 and the indexer rotates
    // its first rope dims: the kernels tile both at 64 (the attention
    // kernels' k-quarters need score_width a multiple of 64; the indexer's
    // rope slice must fit its 128-wide head).
    if (c.qk_rope_head_dim != 0 && c.qk_rope_head_dim != 64)
      fail("qk_rope_head_dim must be 0 (GLM-5.3-Flash) or 64 (the full "
           "GLM-5.3's decoupled RoPE)");
    if (c.qk_rope_head_dim > 0 && c.index_kpool != 1)
      fail("a roped indexer key is per-token (index_kpool must be 1 with "
           "qk_rope_head_dim > 0)");
    if (c.block_tokens % c.index_kpool != 0)
      fail("block_tokens must be a multiple of index_kpool (pool-aligned "
           "blocks keep chunk boundaries pool-aligned, DESIGN §7.2)");
    // The fused [q_a | kv_a] projection keeps kv_a's column offset at
    // q_lora_rank elements; 16-byte alignment for the strided kv_a GEMM
    // input requires q_lora_rank to be a multiple of 8 BF16 elements.
    if (c.q_lora_rank % 8 != 0)
      fail("q_lora_rank must be a multiple of 8 (fused layout alignment)");
    // The quantized rows: the append kernel quantizes a row with one
    // block (8 elements per thread for fp8, one 16-element block per
    // thread for fp4) and the attention tiles load 8 elements at a time.
    if (c.latent_format == LatentFormat::kFp8 &&
        (c.kv_lora_rank % 8 != 0 || c.kv_lora_rank > 1024))
      fail("an fp8 latent cache needs kv_lora_rank a multiple of 8 and <= 1024");
    if (c.latent_format == LatentFormat::kFp4 &&
        (c.kv_lora_rank % kLatentFp4Block != 0 || c.kv_lora_rank > 2048))
      fail("an fp4 latent cache needs kv_lora_rank a multiple of 16 and <= 2048");
    if (c.latent_format == LatentFormat::kFp8Block &&
        (c.kv_lora_rank % kLatentFp8BlockGroup != 0 || c.kv_lora_rank > 1024))
      fail("an fp8_block latent cache needs kv_lora_rank a multiple of 32 and <= 1024");
    if (c.latent_format == LatentFormat::kFp4Block &&
        (c.kv_lora_rank % kLatentFp4Block != 0 || c.kv_lora_rank > 2048))
      fail("an fp4_block latent cache needs kv_lora_rank a multiple of 16 and <= 2048");
    // The rope tail's 8-wide loads sit right after the format's payload.
    if (c.qk_rope_head_dim > 0 && c.kv_lora_rank % 8 != 0)
      fail("a rope tail needs kv_lora_rank a multiple of 8");
  }
};

// Derived per-rank geometry. All byte fields are exact; the totals reproduce
// the DESIGN §7.2 table to the last byte.
struct DsaGeometry {
  int local_heads = 0;     // num_heads / tp_size
  int local_q_rows = 0;    // local_heads * (nope + rope) (q_b output rows)
  int local_v_rows = 0;    // local_heads * v_head_dim (attention output rows)
  int rope_dim = 0;        // qk_rope_head_dim
  int score_width = 0;     // kv_lora + rope: the absorbed query / cache row width
  int select_k = 0;        // pools selected per query (topk / kpool)
  int pools_per_block = 0; // block_tokens / kpool
  int max_selected = 0;    // expanded width: topk + kpool - 1 (pools + tail)
  int index_layers = 0;    // DSA layers with an index cache

  // Per-layer, per-rank cache formulas (bytes).
  size_t latent_payload_bytes = 0;     // the kv_lora part of a row, in the cache's format
  size_t latent_bytes_per_token = 0;   // payload + the bf16 rope tail
  size_t latent_scale_bytes_per_token = 0;  // the row's fp32 scale (fp8/fp4)
  size_t index_k_bytes_per_pool = 0;   // FP8 K row
  size_t index_scale_bytes_per_pool = 0;  // FP32 scale
  size_t index_bytes_per_token = 0;    // (K + scale) / kpool
  size_t tail_bytes_per_request = 0;   // [2, kpool, head_dim] BF16

  // Per-request totals across all DSA layers.
  size_t latent_bytes_per_token_all = 0;  // rows AND their scales
  size_t index_bytes_per_token_all = 0;
  size_t tail_bytes_per_request_all = 0;

  // Block-level allocation sizes (the shared block table's unit).
  size_t latent_block_bytes = 0;       // [block_tokens] latent rows
  size_t index_k_block_bytes = 0;      // [pools_per_block, head_dim] FP8
  size_t index_scale_block_bytes = 0;  // [pools_per_block] FP32

  // Whole-cache totals for a given token capacity and request capacity.
  size_t latent_total_bytes(int64_t tokens) const {
    return latent_bytes_per_token_all * static_cast<size_t>(tokens);
  }
  size_t index_total_bytes(int64_t tokens) const {
    return index_bytes_per_token_all * static_cast<size_t>(tokens);
  }
  size_t tail_total_bytes(int64_t requests) const {
    return tail_bytes_per_request_all * static_cast<size_t>(requests);
  }

  // Causal bounds for a query at absolute position p (reference semantics:
  // cu_seqlen_ke = floor((p+1)/kpool) visible pools; the incomplete tail is
  // appended by expansion, never scored).
  int visible_pools(int64_t pos) const { return int((pos + 1) / kpool); }
  int tail_len(int64_t seq_len) const { return int(seq_len % kpool); }

  int kpool = 4;  // carried for the formula helpers
  LatentFormat latent_format = LatentFormat::kBf16;  // carried for the kernels

  static DsaGeometry from_config(const DsaConfig& c) {
    DsaConfig::validate_config(c);
    DsaGeometry g;
    g.local_heads = c.num_heads / c.tp_size;
    g.local_q_rows = g.local_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim);
    g.local_v_rows = g.local_heads * c.v_head_dim;
    g.rope_dim = c.qk_rope_head_dim;
    g.score_width = c.kv_lora_rank + c.qk_rope_head_dim;
    g.select_k = c.index_topk / c.index_kpool;
    g.pools_per_block = c.block_tokens / c.index_kpool;
    g.max_selected = c.index_topk + c.index_kpool - 1;
    g.index_layers = c.index_layers();
    g.kpool = c.index_kpool;
    g.latent_format = c.latent_format;

    g.latent_payload_bytes = latent_row_bytes(c.latent_format, c.kv_lora_rank);
    g.latent_bytes_per_token =
        g.latent_payload_bytes + static_cast<size_t>(c.qk_rope_head_dim) * 2;
    g.latent_scale_bytes_per_token =
        latent_format_has_row_scale(c.latent_format) ? sizeof(float) : 0;
    g.index_k_bytes_per_pool = static_cast<size_t>(c.index_head_dim);
    g.index_scale_bytes_per_pool = 4;
    g.index_bytes_per_token =
        (g.index_k_bytes_per_pool + g.index_scale_bytes_per_pool) /
        static_cast<size_t>(c.index_kpool);
    g.tail_bytes_per_request =
        2 * static_cast<size_t>(c.index_kpool) * c.index_head_dim * 2;

    g.latent_bytes_per_token_all =
        (g.latent_bytes_per_token + g.latent_scale_bytes_per_token) *
        c.num_dsa_layers;
    g.index_bytes_per_token_all =
        g.index_bytes_per_token * static_cast<size_t>(g.index_layers);
    g.tail_bytes_per_request_all =
        g.tail_bytes_per_request * static_cast<size_t>(g.index_layers);

    g.latent_block_bytes =
        static_cast<size_t>(c.block_tokens) * g.latent_bytes_per_token;
    g.index_k_block_bytes =
        static_cast<size_t>(g.pools_per_block) * c.index_head_dim;
    g.index_scale_block_bytes =
        static_cast<size_t>(g.pools_per_block) * 4;
    return g;
  }
};

}  // namespace dgpp
