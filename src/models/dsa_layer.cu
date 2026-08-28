#include "models/dsa_layer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/dsa.hpp"
#include "models/dsa_state.hpp"

namespace dgpp {

namespace {

// Fixed fused-select grid (grid-invariant selection; partial_ws is sized
// for exactly this many blocks).
constexpr int kSelectGridBlocks = 16;

// Dot-GEMM n granularity, in pools: plans are keyed per multiple of 256
// pools instead of per exact pool count, bounding the plan cache over
// arbitrary chunk boundaries (a 300k-token context yields ~293 shapes).
constexpr int64_t kPoolPadGranularity = 256;

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

int64_t round_up_to(int64_t v, int64_t gran) {
  return (v + gran - 1) / gran * gran;
}

}  // namespace

// ---------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------

DsaLayer::ScratchLayout DsaLayer::scratch_layout(
    const DsaConfig& cfg, int max_tokens, int64_t max_cache_tokens,
    int max_decode_rows, int decode_n_split, size_t dot_budget) {
  DsaConfig::validate_config(cfg);
  if (cfg.index_n_heads != 32)
    throw std::invalid_argument(
        "dsa layer: the selection kernels pin index_n_heads to 32");
  if (max_tokens <= 0 || max_cache_tokens <= 0)
    throw std::invalid_argument(
        "dsa layer: max_tokens and max_cache_tokens must be positive");
  if (max_decode_rows <= 0 || max_decode_rows > 8)
    throw std::invalid_argument("dsa layer: max_decode_rows must be in [1, 8]");
  if (max_decode_rows > max_tokens)
    throw std::invalid_argument(
        "dsa layer: max_decode_rows must not exceed max_tokens");
  if (decode_n_split <= 0)
    throw std::invalid_argument("dsa layer: decode_n_split must be positive");
  if (dot_budget == 0)
    throw std::invalid_argument("dsa layer: dot_budget must be positive");

  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int heads = cfg.index_n_heads;
  const int dim = cfg.index_head_dim;
  const int A = std::max(max_decode_rows, 8);  // attention tile rows
  // Attention workspace capacity: decode splits wide (rows are scarce),
  // prefill splits by a fixed 8 — size for whichever is larger so either
  // path is always in bounds.
  const int split_cap = std::max(decode_n_split, 8);

  const int64_t max_pools = round_up_to(
      (max_cache_tokens + g.kpool - 1) / g.kpool, kPoolPadGranularity);
  int tile_cap =
      int(dot_budget / (size_t(heads) * size_t(max_pools) * 4));
  if (tile_cap <= 0) tile_cap = 1;  // degenerate budget: 1-row tiles
  tile_cap = std::min(tile_cap, max_tokens);

  ScratchLayout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += bytes;
    return at;
  };
  const size_t T = size_t(max_tokens);
  const size_t Ta = std::max(T, size_t(A));
  L.off_qkv = alloc(T * (cfg.q_lora_rank + cfg.kv_lora_rank) * 2);
  L.off_q_c = alloc(T * size_t(cfg.q_lora_rank) * 2);
  L.off_kv_c = alloc(T * size_t(cfg.kv_lora_rank) * 2);
  L.off_q = alloc(T * size_t(g.local_q_rows) * 2);
  L.off_q_idx = alloc(T * size_t(heads) * size_t(dim) * 2);
  L.off_k_raw = alloc(T * size_t(dim) * 2);
  L.off_k_rows = alloc(T * size_t(dim) * 2);
  L.off_gate = alloc(T * size_t(dim) * 2);
  L.off_weights = alloc(T * size_t(heads) * 4);
  L.off_q_fp8 = alloc(T * size_t(heads) * size_t(dim));
  L.off_q_scale = alloc(T * size_t(heads) * 4);
  L.off_w_folded = alloc(T * size_t(heads) * 4);
  L.off_topk = alloc(T * size_t(g.max_selected) * 4);
  L.off_counts = alloc(T * 4);
  L.off_pos = alloc(T * 8);
  L.off_req_ids = alloc(Ta * 4);
  L.off_attn_out = alloc(T * size_t(g.local_v_rows) * 2);
  L.off_q_tilde = alloc(size_t(A) * size_t(g.local_heads) *
                        size_t(cfg.kv_lora_rank) * 2);
  L.off_c = alloc(size_t(A) * size_t(g.local_heads) *
                  size_t(cfg.kv_lora_rank) * 4);
  L.off_m = alloc(size_t(A) * size_t(split_cap) * size_t(g.local_heads) * 4);
  L.off_l = alloc(size_t(A) * size_t(split_cap) * size_t(g.local_heads) * 4);
  L.off_cws = alloc(size_t(A) * size_t(split_cap) * size_t(g.local_heads) *
                    size_t(cfg.kv_lora_rank) * 4);
  L.off_dot = alloc(size_t(tile_cap) * size_t(heads) * size_t(max_pools) * 4);
  L.off_gather_k = alloc(size_t(max_pools) * size_t(dim));
  L.off_gather_scale = alloc(size_t(max_pools) * 4);
  L.off_partial =
      alloc(size_t(kSelectGridBlocks) * size_t(max_decode_rows) *
            size_t(g.select_k) * 8);
  L.off_counter = alloc(4);
  L.total = align256(off);
  L.max_pools = max_pools;
  L.tile_cap = tile_cap;
  return L;
}

// ---------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------

DsaLayer::DsaLayer(IGemm& gemm, const DsaLayerWeights& w, const DsaConfig& cfg,
                   int max_tokens, int64_t max_cache_tokens, void* scratch,
                   size_t scratch_capacity, void* gemm_workspace,
                   size_t gemm_ws_bytes, int max_decode_rows,
                   int decode_n_split, size_t dot_budget)
    : gemm_(gemm),
      w_(w),
      cfg_(cfg),
      geo_(DsaGeometry::from_config(cfg)),
      max_tokens_(max_tokens),
      max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows),
      decode_n_split_(decode_n_split),
      gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const ScratchLayout L =
      scratch_layout(cfg, max_tokens, max_cache_tokens, max_decode_rows,
                     decode_n_split, dot_budget);
  if (!scratch)
    throw std::invalid_argument("dsa layer: scratch buffer required");
  if (scratch_capacity < L.total)
    throw std::invalid_argument(
        "dsa layer: scratch buffer too small (" +
        std::to_string(scratch_capacity) + " < " + std::to_string(L.total) +
        "; use scratch_bytes()");
  if (!gemm_workspace || gemm_ws_bytes == 0)
    throw std::invalid_argument("dsa layer: GEMM workspace required");
  const auto require = [](const void* p, const char* what) {
    if (!p)
      throw std::invalid_argument(std::string("dsa layer: missing ") + what);
  };
  require(w_.wq_b, "wq_b");
  require(w_.wk, "wk");
  require(w_.wp, "wp");
  require(w_.gate, "gate");
  require(w_.k_norm_w, "k_norm_w");
  require(w_.k_norm_b, "k_norm_b");
  require(w_.ape, "ape");
  require(w_.qkv_a, "qkv_a");
  require(w_.q_aln, "q_aln");
  require(w_.kv_aln, "kv_aln");
  require(w_.q_b, "q_b");
  require(w_.kv_b, "kv_b");
  require(w_.o_proj, "o_proj");

  attn_rows_ = std::max(max_decode_rows, 8);
  tile_cap_ = L.tile_cap;
  max_pools_ = L.max_pools;

  // Reference computes both scales in python float64 and rounds to fp32;
  // reproduce the double-precision products exactly (1/64 and 1/16 here).
  logit_scale_ = static_cast<float>(
      std::pow(double(cfg.index_head_dim), -0.5) *
      std::pow(double(cfg.index_n_heads), -0.5));
  attn_scale_ = 1.0f / std::sqrt(float(cfg.qk_nope_head_dim));

  scratch_ = static_cast<uint8_t*>(scratch);
  const auto at = [&](size_t o) { return scratch_ + o; };
  qkv_ = reinterpret_cast<uint16_t*>(at(L.off_qkv));
  q_c_ = reinterpret_cast<uint16_t*>(at(L.off_q_c));
  kv_c_ = reinterpret_cast<uint16_t*>(at(L.off_kv_c));
  q_ = reinterpret_cast<uint16_t*>(at(L.off_q));
  q_idx_ = reinterpret_cast<uint16_t*>(at(L.off_q_idx));
  k_raw_ = reinterpret_cast<uint16_t*>(at(L.off_k_raw));
  k_rows_ = reinterpret_cast<uint16_t*>(at(L.off_k_rows));
  gate_rows_ = reinterpret_cast<uint16_t*>(at(L.off_gate));
  weights_ = reinterpret_cast<float*>(at(L.off_weights));
  q_fp8_ = at(L.off_q_fp8);
  q_scale_ = reinterpret_cast<float*>(at(L.off_q_scale));
  w_folded_ = reinterpret_cast<float*>(at(L.off_w_folded));
  topk_ = reinterpret_cast<int32_t*>(at(L.off_topk));
  counts_ = reinterpret_cast<int32_t*>(at(L.off_counts));
  pos_dev_ = reinterpret_cast<int64_t*>(at(L.off_pos));
  req_ids_dev_ = reinterpret_cast<int32_t*>(at(L.off_req_ids));
  attn_out_ = reinterpret_cast<uint16_t*>(at(L.off_attn_out));
  q_tilde_ = reinterpret_cast<uint16_t*>(at(L.off_q_tilde));
  c_ = reinterpret_cast<float*>(at(L.off_c));
  m_ws_ = reinterpret_cast<float*>(at(L.off_m));
  l_ws_ = reinterpret_cast<float*>(at(L.off_l));
  c_ws_ = reinterpret_cast<float*>(at(L.off_cws));
  dot_ = reinterpret_cast<float*>(at(L.off_dot));
  gather_k_ = at(L.off_gather_k);
  gather_scale_ = reinterpret_cast<float*>(at(L.off_gather_scale));
  partial_ws_ = reinterpret_cast<uint64_t*>(at(L.off_partial));
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.off_counter));
}

size_t DsaLayer::scratch_bytes(const DsaConfig& cfg, int max_tokens,
                               int64_t max_cache_tokens, int max_decode_rows,
                               int decode_n_split, size_t dot_budget) {
  return scratch_layout(cfg, max_tokens, max_cache_tokens, max_decode_rows,
                        decode_n_split, dot_budget)
      .total;
}

// ---------------------------------------------------------------------
// Plan preparation
// ---------------------------------------------------------------------

bool DsaLayer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("dsa layer: token count out of range");
  const int hid = cfg_.hidden;
  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int qkv_cols = cfg_.q_lora_rank + cfg_.kv_lora_rank;
  const bool ok =
      gemm_.ensure_plan(tokens, qkv_cols, hid, DType::BF16, GemmOut::BF16,
                        size_t(hid)) &&
      gemm_.ensure_plan(tokens, geo_.local_q_rows, cfg_.q_lora_rank,
                        DType::BF16, GemmOut::BF16,
                        size_t(cfg_.q_lora_rank)) &&
      gemm_.ensure_plan(tokens, heads * dim, cfg_.q_lora_rank, DType::BF16,
                        GemmOut::BF16, size_t(cfg_.q_lora_rank)) &&
      gemm_.ensure_plan(tokens, dim, hid, DType::BF16, GemmOut::BF16,
                        size_t(hid)) &&
      gemm_.ensure_plan(tokens, heads, hid, DType::BF16, GemmOut::F32,
                        size_t(hid)) &&
      gemm_.ensure_plan(tokens, hid, geo_.local_v_rows, DType::BF16,
                        GemmOut::BF16, size_t(geo_.local_v_rows));
  // The kernel shared-memory opt-in is context state; do it now so the
  // first enqueue (which may be inside graph capture) never mutates it.
  dsa_prepare_kernel_smem();
  return ok;
}

bool DsaLayer::prepare_prefill(int tile_rows, int64_t visible_pools) {
  if (tile_rows <= 0 || tile_rows > tile_cap_)
    throw std::invalid_argument("dsa layer: prefill tile rows out of range");
  if (visible_pools < 0 || visible_pools > max_pools_)
    throw std::invalid_argument("dsa layer: visible pools out of range");
  const int64_t padded = round_up_to(visible_pools, kPoolPadGranularity);
  return gemm_.ensure_plan(tile_rows * cfg_.index_n_heads, int(padded),
                           cfg_.index_head_dim, DType::F8_E4M3, GemmOut::F32,
                           size_t(cfg_.index_head_dim));
}

// ---------------------------------------------------------------------
// Shared projection chain
// ---------------------------------------------------------------------

void DsaLayer::project_common(const void* hidden_in, int tokens,
                              cudaStream_t stream) {
  const int hid = cfg_.hidden;
  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int qkv_cols = cfg_.q_lora_rank + cfg_.kv_lora_rank;

  // 1) fused [q_a | kv_a] projection, then RMSNorms on the split halves.
  gemm_.matmul(hidden_in, w_.qkv_a, qkv_, tokens, qkv_cols, hid, DType::BF16,
               GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  dsa_fused_qkv_rmsnorm(qkv_, q_c_, kv_c_, cfg_.q_lora_rank,
                        cfg_.kv_lora_rank, tokens, w_.q_aln, w_.kv_aln,
                        cfg_.rms_norm_eps, stream);
  // 2) MLA q and indexer q, both from the normed q-lora rows.
  gemm_.matmul(q_c_, w_.q_b, q_, tokens, geo_.local_q_rows, cfg_.q_lora_rank,
               DType::BF16, GemmOut::BF16, size_t(cfg_.q_lora_rank), gemm_ws_,
               gemm_ws_bytes_, stream);
  gemm_.matmul(q_c_, w_.wq_b, q_idx_, tokens, heads * dim, cfg_.q_lora_rank,
               DType::BF16, GemmOut::BF16, size_t(cfg_.q_lora_rank), gemm_ws_,
               gemm_ws_bytes_, stream);
  // 3) indexer k (LayerNorm, eps 1e-6 — the indexer's own norm, not
  // rms_norm_eps) and gate, both straight from hidden.
  gemm_.matmul(hidden_in, w_.wk, k_raw_, tokens, dim, hid, DType::BF16,
               GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  dsa_k_layernorm(k_raw_, dim, w_.k_norm_w, w_.k_norm_b, k_rows_, tokens, dim,
                  1e-6f, stream);
  gemm_.matmul(hidden_in, w_.gate, gate_rows_, tokens, dim, hid, DType::BF16,
               GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  // 4) indexer weights in fp32 with no bf16 rounding (reference pins this):
  // bf16 inputs, fp32 accumulate, fp32 out.
  gemm_.matmul(hidden_in, w_.wp, weights_, tokens, heads, hid, DType::BF16,
               GemmOut::F32, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  // 5) Hadamard-128 + fp8 quant of the indexer q, then fold the q scale and
  // the combined logit scale into the weights.
  dsa_fwht_quant_rows(q_idx_, int64_t(tokens) * heads, q_fp8_, q_scale_,
                      stream);
  dsa_fold_weights(weights_, q_scale_, w_folded_, int64_t(tokens) * heads,
                   logit_scale_, stream);
}

// ---------------------------------------------------------------------
// Attention (shared by both paths)
// ---------------------------------------------------------------------

void DsaLayer::attend_tile(DsaStatePool& state, int layer,
                           const int32_t* req_ids, int64_t row0, int rows,
                           int n_split, cudaStream_t stream) {
  for (int64_t a0 = row0; a0 < row0 + rows; a0 += attn_rows_) {
    const int arows = int(std::min<int64_t>(attn_rows_, row0 + rows - a0));
    dsa_absorb_q(q_ + a0 * geo_.local_q_rows, w_.kv_b, q_tilde_, arows,
                 geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                 cfg_.kv_lora_rank, stream);
    dsa_attn_partial(q_tilde_, state.latent(layer), req_ids + a0,
                     topk_ + a0 * geo_.max_selected, geo_.max_selected,
                     counts_ + a0, arows, n_split, geo_.local_heads,
                     cfg_.kv_lora_rank, cfg_.block_tokens, state.block_tables(),
                     int(state.total_blocks()), attn_scale_, m_ws_, l_ws_,
                     c_ws_, stream);
    dsa_attn_combine(m_ws_, l_ws_, c_ws_, arows, n_split, geo_.local_heads,
                     cfg_.kv_lora_rank, c_, stream);
    dsa_vout_gemm(c_, w_.kv_b, attn_out_ + a0 * geo_.local_v_rows, arows,
                  geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                  cfg_.kv_lora_rank, stream);
  }
}

// ---------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------

void DsaLayer::enqueue_prefill(const void* hidden_in, DsaStatePool& state,
                               int layer, int req, int64_t token_start,
                               int tokens, void* out, cudaStream_t stream) {
  validate_pool(state, layer);
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("dsa layer: token count out of range");
  if (token_start < 0 || token_start % cfg_.index_kpool != 0)
    throw std::invalid_argument(
        "dsa layer: prefill chunks must start pool-aligned");
  if (token_start + tokens > max_cache_tokens_)
    throw std::invalid_argument("dsa layer: chunk exceeds max_cache_tokens");
  // The reference tail-seed reads only in-chunk k rows; a continuation
  // chunk shorter than kpool would read before the chunk. (The device ring
  // would handle it, but parity is pinned to the reference.)
  if (token_start > 0 && tokens < cfg_.index_kpool)
    throw std::invalid_argument(
        "dsa layer: continuation chunks must be at least kpool tokens");
  if (!hidden_in || !out)
    throw std::invalid_argument("dsa layer: null buffer");

  if (!state.ensure_request_blocks(req, token_start + tokens, stream))
    throw std::runtime_error("dsa layer: cache pool exhausted during prefill");

  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int kpool = cfg_.index_kpool;
  const int blocks_per_req = int(state.total_blocks());

  project_common(hidden_in, tokens, stream);

  // Per-token metadata (host staging — prefill is never graph-captured).
  pos_host_.resize(size_t(tokens));
  for (int i = 0; i < tokens; ++i) pos_host_[size_t(i)] = token_start + i;
  req_ids_host_.assign(std::max(size_t(tokens), size_t(attn_rows_)), req);
  DGPP_CUDA_OK(cudaMemcpyAsync(pos_dev_, pos_host_.data(),
                               size_t(tokens) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(req_ids_dev_, req_ids_host_.data(),
                               req_ids_host_.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

  // Latent cache append.
  dsa_latent_append(kv_c_, req_ids_dev_, pos_dev_, tokens, state.block_tables(),
                    blocks_per_req, cfg_.block_tokens, state.latent(layer),
                    cfg_.kv_lora_rank, stream);

  // Complete pools fully inside this chunk -> compressed index cache.
  const int64_t pool_lo = token_start / kpool;
  const int64_t pool_hi = (token_start + tokens) / kpool;
  dsa_kpool_compress_write(k_rows_, dim, gate_rows_, dim, w_.ape,
                           state.block_tables() + size_t(req) * blocks_per_req,
                           geo_.pools_per_block, pool_lo, pool_hi - pool_lo,
                           state.index_k(layer), state.index_scale(layer),
                           kpool, dim, stream);

  // Tail ring: the last kpool tokens of the request so far.
  dsa_kpool_tail_seed(k_rows_, dim, gate_rows_, dim, req_ids_dev_, pos_dev_,
                      tokens, state.tail(layer), kpool, dim, stream);

  // Selection + attention over dot tiles. The gather is per-chunk (every
  // tile reads the same pools); only the dot buffer is per-tile.
  const int64_t n_gather = pool_hi;  // pools visible to the chunk's last row
  const int64_t padded_n =
      n_gather > 0 ? round_up_to(n_gather, kPoolPadGranularity) : 0;
  if (padded_n > 0) {
    // The dot GEMM reads the padded tail of the gather buffer; keep every
    // byte it will ever read initialized (compute-sanitizer initcheck
    // discipline — the select kernel itself never reads those pools).
    if (padded_n > gather_zeroed_) {
      DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * dim, 0,
                                   size_t(padded_n - gather_zeroed_) * dim,
                                   stream));
      DGPP_CUDA_OK(cudaMemsetAsync(gather_scale_ + gather_zeroed_, 0,
                                   size_t(padded_n - gather_zeroed_) * 4,
                                   stream));
      gather_zeroed_ = padded_n;
    }
    dsa_gather_index_pools(
        state.block_tables() + size_t(req) * blocks_per_req,
        geo_.pools_per_block, state.index_k(layer), state.index_scale(layer),
        n_gather, gather_k_, gather_scale_, dim, stream);
  }

  for (int row0 = 0; row0 < tokens; row0 += tile_cap_) {
    const int rows = std::min(tile_cap_, tokens - row0);
    if (padded_n > 0) {
      // Dots for this tile's (row, head) pairs against the gathered pools.
      gemm_.matmul(q_fp8_ + size_t(row0) * heads * dim, gather_k_, dot_,
                   rows * heads, int(padded_n), dim, DType::F8_E4M3,
                   GemmOut::F32, size_t(dim), gemm_ws_, gemm_ws_bytes_,
                   stream);
      dot_stride_last_ = padded_n;
    } else {
      dot_stride_last_ = 0;
    }
    dsa_select_prefill(dot_, std::max<int64_t>(padded_n, 1),
                       w_folded_ + size_t(row0) * heads, gather_scale_,
                       pos_dev_ + row0, rows, n_gather, heads, geo_.select_k,
                       kpool, geo_.max_selected,
                       topk_ + size_t(row0) * geo_.max_selected, counts_ + row0,
                       stream);
    // Prefill's 8-row tiles already give rows*4 blocks of parallelism;
    // split by a fixed 8 (<= the scratch capacity computed at construction)
    // so early-row groups with small counts don't pay wide-split c_ws
    // round trips.
    attend_tile(state, layer, req_ids_dev_, row0, rows, 8, stream);
  }

  // Output projection.
  gemm_.matmul(attn_out_, w_.o_proj, out, tokens, cfg_.hidden,
               geo_.local_v_rows, DType::BF16, GemmOut::BF16,
               size_t(geo_.local_v_rows), gemm_ws_, gemm_ws_bytes_, stream);
}

// ---------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------

void DsaLayer::enqueue_decode(const void* hidden_in, DsaStatePool& state,
                              int layer, const int32_t* req_ids,
                              const int64_t* pos, const int32_t* req_spans,
                              int num_requests, int tokens, void* out,
                              cudaStream_t stream) {
  validate_pool(state, layer);
  if (tokens <= 0 || tokens > max_decode_rows_)
    throw std::invalid_argument("dsa layer: decode rows out of range");
  if (num_requests <= 0 || num_requests > state.max_requests())
    throw std::invalid_argument("dsa layer: request count out of range");
  if (!hidden_in || !req_ids || !pos || !req_spans || !out)
    throw std::invalid_argument("dsa layer: null buffer");

  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int kpool = cfg_.index_kpool;
  dot_stride_last_ = 0;  // decode selects consume no dot buffer (debug probe)

  project_common(hidden_in, tokens, stream);

  // Latent rows first (this batch's own tokens are readable by this
  // batch's attention — causal self-include, reference semantics).
  dsa_latent_append(kv_c_, req_ids, pos, tokens, state.block_tables(),
                    int(state.total_blocks()), cfg_.block_tokens,
                    state.latent(layer), cfg_.kv_lora_rank, stream);
  // Ring update + pool compression for any pool completed by this batch.
  dsa_kpool_decode_update(k_rows_, dim, gate_rows_, dim, w_.ape, pos,
                          req_spans, num_requests, state.block_tables(),
                          int(state.total_blocks()), state.tail(layer),
                          state.index_k(layer), state.index_scale(layer),
                          geo_.pools_per_block, kpool, dim, stream);
  // Fused select straight from the blocked index cache.
  dsa_select_decode(q_fp8_, w_folded_, req_ids, pos, tokens,
                    state.block_tables(), int(state.total_blocks()),
                    state.index_k(layer), state.index_scale(layer),
                    geo_.pools_per_block, heads, dim, geo_.select_k, kpool,
                    geo_.max_selected, topk_, counts_, partial_ws_,
                    counter_ws_, kSelectGridBlocks, stream);
  // Absorbed attention + v-absorb + output projection.
  attend_tile(state, layer, req_ids, 0, tokens, decode_n_split_, stream);
  gemm_.matmul(attn_out_, w_.o_proj, out, tokens, cfg_.hidden,
               geo_.local_v_rows, DType::BF16, GemmOut::BF16,
               size_t(geo_.local_v_rows), gemm_ws_, gemm_ws_bytes_, stream);
}

// ---------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------

void DsaLayer::validate_pool(const DsaStatePool& state, int layer) const {
  const DsaConfig& c = state.config();
  const bool same = c.hidden == cfg_.hidden && c.num_heads == cfg_.num_heads &&
                    c.q_lora_rank == cfg_.q_lora_rank &&
                    c.kv_lora_rank == cfg_.kv_lora_rank &&
                    c.qk_nope_head_dim == cfg_.qk_nope_head_dim &&
                    c.qk_rope_head_dim == cfg_.qk_rope_head_dim &&
                    c.v_head_dim == cfg_.v_head_dim &&
                    c.index_n_heads == cfg_.index_n_heads &&
                    c.index_head_dim == cfg_.index_head_dim &&
                    c.index_topk == cfg_.index_topk &&
                    c.index_kpool == cfg_.index_kpool &&
                    c.block_tokens == cfg_.block_tokens &&
                    c.tp_size == cfg_.tp_size &&
                    c.num_dsa_layers == cfg_.num_dsa_layers;
  if (!same)
    throw std::invalid_argument(
        "dsa layer: state pool was built for a different configuration");
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa layer: layer " + std::to_string(layer));
}

}  // namespace dgpp
