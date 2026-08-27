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
//   * qk_rope_head_dim=0 for this checkpoint: the MLA q/k path is entirely
//     nope — there is no RoPE anywhere in a DSA layer, and the engine only
//     implements the rope-free path.
//   * visible pools for a query at position p: floor((p+1)/kpool); the tail
//     [kpool*floor(L/kpool), L) is never scored, only appended.
//
// Layout decisions that are ours (not the reference's): the index cache is
// two planar arrays — FP8-E4M3 K rows and FP32 scales — instead of the
// reference's packed 132-byte pages (a DeepGEMM block_kv constraint we do
// not inherit). One shared block table co-locates latent blocks and index
// blocks so prefix attachment shares both by reference.
#include <cstdint>
#include <stdexcept>
#include <string>

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
  int qk_rope_head_dim = 0;  // pinned to 0 for this checkpoint
  int v_head_dim = 256;
  int index_n_heads = 32;    // indexer heads, replicated across TP
  int index_head_dim = 128;
  int index_topk = 2048;     // selected TOKENS per query (topk_tokens)
  int index_kpool = 4;       // tokens per compressed index pool
  int always_select_tail = 1;  // index_kpool_always_select_tail
  int num_dsa_layers = 11;   // of the 45 text layers ([3,7,...,43])
  int block_tokens = 128;    // shared-block granularity, in tokens
  int tp_size = 1;
  float rms_norm_eps = 1e-5f;  // q_a/kv_a layernorms (config rms_norm_eps)

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
    if (c.index_kpool != 2 && c.index_kpool != 4 && c.index_kpool != 8)
      fail("index_kpool must be 2, 4, or 8 (kernel template dispatch)");
    if (c.index_topk % c.index_kpool != 0)
      fail("index_topk must be a multiple of index_kpool");
    if (c.num_dsa_layers <= 0) fail("num_dsa_layers must be positive");
    if (c.tp_size <= 0) fail("tp_size must be positive");
    if (c.num_heads % c.tp_size != 0) fail("num_heads must divide by tp_size");
    if (c.qk_rope_head_dim != 0)
      fail("qk_rope_head_dim must be 0 (this engine implements the "
           "rope-free GLM-5.3-Flash MLA path only)");
    if (c.block_tokens % c.index_kpool != 0)
      fail("block_tokens must be a multiple of index_kpool (pool-aligned "
           "blocks keep chunk boundaries pool-aligned, DESIGN §7.2)");
    // The fused [q_a | kv_a] projection keeps kv_a's column offset at
    // q_lora_rank elements; 16-byte alignment for the strided kv_a GEMM
    // input requires q_lora_rank to be a multiple of 8 BF16 elements.
    if (c.q_lora_rank % 8 != 0)
      fail("q_lora_rank must be a multiple of 8 (fused layout alignment)");
  }
};

// Derived per-rank geometry. All byte fields are exact; the totals reproduce
// the DESIGN §7.2 table to the last byte.
struct DsaGeometry {
  int local_heads = 0;     // num_heads / tp_size
  int local_q_rows = 0;    // local_heads * qk_nope_head_dim (q_b output rows)
  int local_v_rows = 0;    // local_heads * v_head_dim (attention output rows)
  int select_k = 0;        // pools selected per query (topk / kpool)
  int pools_per_block = 0; // block_tokens / kpool
  int max_selected = 0;    // expanded width: topk + kpool - 1 (pools + tail)

  // Per-layer, per-rank cache formulas (bytes).
  size_t latent_bytes_per_token = 0;   // kv_lora_rank BF16
  size_t index_k_bytes_per_pool = 0;   // FP8 K row
  size_t index_scale_bytes_per_pool = 0;  // FP32 scale
  size_t index_bytes_per_token = 0;    // (K + scale) / kpool
  size_t tail_bytes_per_request = 0;   // [2, kpool, head_dim] BF16

  // Per-request totals across all DSA layers.
  size_t latent_bytes_per_token_all = 0;
  size_t index_bytes_per_token_all = 0;
  size_t tail_bytes_per_request_all = 0;

  // Block-level allocation sizes (the shared block table's unit).
  size_t latent_block_bytes = 0;       // [block_tokens, kv_lora] BF16
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

  static DsaGeometry from_config(const DsaConfig& c) {
    DsaConfig::validate_config(c);
    DsaGeometry g;
    g.local_heads = c.num_heads / c.tp_size;
    g.local_q_rows = g.local_heads * c.qk_nope_head_dim;
    g.local_v_rows = g.local_heads * c.v_head_dim;
    g.select_k = c.index_topk / c.index_kpool;
    g.pools_per_block = c.block_tokens / c.index_kpool;
    g.max_selected = c.index_topk + c.index_kpool - 1;
    g.kpool = c.index_kpool;

    g.latent_bytes_per_token =
        static_cast<size_t>(c.kv_lora_rank) * 2;
    g.index_k_bytes_per_pool = static_cast<size_t>(c.index_head_dim);
    g.index_scale_bytes_per_pool = 4;
    g.index_bytes_per_token =
        (g.index_k_bytes_per_pool + g.index_scale_bytes_per_pool) /
        static_cast<size_t>(c.index_kpool);
    g.tail_bytes_per_request =
        2 * static_cast<size_t>(c.index_kpool) * c.index_head_dim * 2;

    g.latent_bytes_per_token_all =
        g.latent_bytes_per_token * c.num_dsa_layers;
    g.index_bytes_per_token_all =
        g.index_bytes_per_token * c.num_dsa_layers;
    g.tail_bytes_per_request_all =
        g.tail_bytes_per_request * c.num_dsa_layers;

    g.latent_block_bytes =
        static_cast<size_t>(c.block_tokens) * c.kv_lora_rank * 2;
    g.index_k_block_bytes =
        static_cast<size_t>(g.pools_per_block) * c.index_head_dim;
    g.index_scale_block_bytes =
        static_cast<size_t>(g.pools_per_block) * 4;
    return g;
  }
};

}  // namespace dgpp
