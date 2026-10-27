#include "models/llama/layers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/qsa.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

}  // namespace

// ---- LlamaAttentionLayer ----------------------------------------------------------

int LlamaAttentionLayer::default_decode_splits() {
  // 32 splits: at TP=1 (32 query heads over 8 kv heads) a T=1 row is 64
  // blocks of 12 warps; a short context leaves splits empty (they write
  // m = -inf and cost nothing), a 32K context gives each split 32 tiles.
  if (const char* v = std::getenv("DGPP_LLAMA_ATTN_SPLITS"); v && *v) {
    const int n = std::atoi(v);
    if (n > 0) return n;
  }
  return 32;
}

LlamaAttentionLayer::LlamaAttentionLayer(const LlamaAttnResident& w, const LlamaGemmWorkspace& gemm,
                                         const LlamaTextConfig& cfg, int max_tokens, int decode_rows,
                                         int n_split_decode)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), lh_(w.local_heads), lkv_(w.local_kv_heads), dim_(cfg.head_dim),
      rotary_(cfg.head_dim), max_tokens_(max_tokens), decode_rows_(decode_rows), n_split_(n_split_decode),
      eps_(cfg.rms_norm_eps) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("LlamaAttentionLayer: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("LlamaAttentionLayer: max_tokens must be positive");
  if (decode_rows_ < 0) throw std::invalid_argument("LlamaAttentionLayer: decode_rows");
  if (n_split_ <= 0) throw std::invalid_argument("LlamaAttentionLayer: n_split_decode must be positive");
  if (lh_ <= 0 || lkv_ <= 0 || lh_ % lkv_ != 0)
    throw std::invalid_argument("LlamaAttentionLayer: query heads must be a multiple of kv heads");
  if (dim_ != kGlm4HeadDim) throw std::invalid_argument("LlamaAttentionLayer: 128-wide heads (the kernels' shape)");
  // The reference's scale: head_dim ** -0.5 in double, narrowed to fp32.
  scale_ = static_cast<float>(std::pow(static_cast<double>(dim_), -0.5));
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t Q = static_cast<size_t>(lh_) * dim_, KV = static_cast<size_t>(lkv_) * dim_;
  {
    std::vector<float> inv(static_cast<size_t>(rotary_ / 2));
    qsa_rope_inv_freq(cfg.rope_theta, rotary_, inv.data());
    d_inv_freq_ = dev_alloc<float>(inv.size());
    DGPP_CUDA_OK(cudaMemcpy(d_inv_freq_, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice));
  }
  qd_ = dev_alloc<float>(M * Q);
  kd_ = dev_alloc<float>(M * KV);
  vd_ = dev_alloc<float>(M * KV);
  q_ = dev_alloc<uint16_t>(M * Q);
  part_rows_ = std::max(M, static_cast<size_t>(decode_rows_) * static_cast<size_t>(n_split_));
  const size_t part = part_rows_ * lh_;
  m_ws_ = dev_alloc<float>(part);
  l_ws_ = dev_alloc<float>(part);
  c_ws_ = dev_alloc<float>(part * dim_);
  o_ = dev_alloc<uint16_t>(M * Q);
}

LlamaAttentionLayer::~LlamaAttentionLayer() {
  cudaFree(d_inv_freq_);
  cudaFree(qd_);
  cudaFree(kd_);
  cudaFree(vd_);
  cudaFree(q_);
  cudaFree(m_ws_);
  cudaFree(l_ws_);
  cudaFree(c_ws_);
  cudaFree(o_);
}

void LlamaAttentionLayer::rebind(const LlamaAttnResident& w) {
  if (w.local_heads != lh_ || w.local_kv_heads != lkv_)
    throw std::invalid_argument("LlamaAttentionLayer: rebind changes the head geometry");
  w_ = w;
}

size_t LlamaAttentionLayer::scratch_bytes(const LlamaTextConfig& cfg, int local_heads, int local_kv_heads,
                                          int max_tokens, int decode_rows, int n_split_decode) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t Q = static_cast<size_t>(local_heads) * cfg.head_dim, KV = static_cast<size_t>(local_kv_heads) * cfg.head_dim;
  const size_t part_rows = std::max(M, static_cast<size_t>(std::max(decode_rows, 0)) * static_cast<size_t>(n_split_decode));
  const size_t part = part_rows * local_heads;
  return static_cast<size_t>(cfg.head_dim / 2) * 4 + M * Q * 4 + 2 * M * KV * 4 + M * Q * 2 + part * 4 * 2 +
         part * cfg.head_dim * 4 + M * Q * 2;
}

void LlamaAttentionLayer::enqueue(const uint16_t* x, int tokens, const LlamaAttnRows& rows, const LlamaKvCache& cache,
                                  uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("LlamaAttentionLayer: tokens exceed max_tokens");
  if (!x || !out || !rows.req_ids || !rows.pos || !cache.k_cache || !cache.v_cache || !cache.block_tables)
    throw std::invalid_argument("LlamaAttentionLayer: null pointer");
  if (rows.split(x, tokens, hidden_, out, [&](const uint16_t* input, int n,
      const auto& map, uint16_t* output) { enqueue(input, n, map, cache, output, stream); })) return;
  const int H = hidden_;
  const int Q = lh_ * dim_, KV = lkv_ * dim_;
  // The three projections, fp32 out (decode rows take the bf16 GEMV inside
  // the interface). Under engine.dense_weights = "fp8" they run the scale
  // GEMM instead — E4M3 codes + 128x128 scales, bf16 activations:
  // weight-only quantization (kernels/scale_gemm.hpp).
  if (w_.q_proj_fp8.payload) {
    launch_scale_gemm_f32(x, static_cast<size_t>(H), w_.q_proj_fp8.payload, w_.q_proj_fp8.scales, qd_, tokens, Q, H,
                          stream, static_cast<size_t>(Q), g_.mma_from_rows, /*last_row_only=*/false, g_.ws,
                          g_.ws_bytes);
    launch_scale_gemm_f32(x, static_cast<size_t>(H), w_.k_proj_fp8.payload, w_.k_proj_fp8.scales, kd_, tokens, KV, H,
                          stream, static_cast<size_t>(KV), g_.mma_from_rows, /*last_row_only=*/false, g_.ws,
                          g_.ws_bytes);
    launch_scale_gemm_f32(x, static_cast<size_t>(H), w_.v_proj_fp8.payload, w_.v_proj_fp8.scales, vd_, tokens, KV, H,
                          stream, static_cast<size_t>(KV), g_.mma_from_rows, /*last_row_only=*/false, g_.ws,
                          g_.ws_bytes);
  } else {
    g_.gemm->matmul(x, w_.q_proj, qd_, tokens, Q, H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), g_.ws,
                    g_.ws_bytes, stream);
    g_.gemm->matmul(x, w_.k_proj, kd_, tokens, KV, H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), g_.ws,
                    g_.ws_bytes, stream);
    g_.gemm->matmul(x, w_.v_proj, vd_, tokens, KV, H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), g_.ws,
                    g_.ws_bytes, stream);
  }
  // The full RoPE (no bias, no head norm) and the K/V append.
  glm4_qkv_finish(qd_, Q, kd_, KV, vd_, KV, nullptr, nullptr, nullptr, nullptr, nullptr, eps_, d_inv_freq_,
                  rotary_, rows.req_ids, rows.pos, tokens, lh_, lkv_, cache.block_tables,
                  cache.blocks_per_request, cache.block_tokens, q_, Q, cache.k_cache, cache.v_cache, stream);
  // The attention: decode rows split the visible range (the capture's
  // geometry), prefill rows run one split each (T x kv-head blocks).
  const int n_split = rows.decode ? n_split_ : 1;
  if (static_cast<size_t>(tokens) * static_cast<size_t>(n_split) > part_rows_)
    throw std::invalid_argument("LlamaAttentionLayer: decode rows exceed the layer's decode_rows");
  glm4_attn_partial(q_, Q, cache.k_cache, cache.v_cache, rows.req_ids, rows.pos, tokens, n_split, lh_, lkv_,
                    cache.block_tokens, cache.block_tables, cache.blocks_per_request, scale_, m_ws_, l_ws_, c_ws_,
                    stream);
  glm4_attn_combine(m_ws_, l_ws_, c_ws_, tokens, n_split, lh_, o_, stream);
  if (w_.o_proj_fp8.payload)
    launch_scale_gemm_bf16(o_, static_cast<size_t>(Q), w_.o_proj_fp8.payload, w_.o_proj_fp8.scales, out, tokens, H, Q,
                           stream, static_cast<size_t>(H), g_.mma_from_rows, g_.ws, g_.ws_bytes);
  else
    g_.gemm->matmul(o_, w_.o_proj, out, tokens, H, Q, DType::BF16, GemmOut::BF16, static_cast<size_t>(Q), g_.ws,
                    g_.ws_bytes, stream);
}

// ---- LlamaDenseMlp ------------------------------------------------------------------

LlamaDenseMlp::LlamaDenseMlp(const LlamaDenseMlpResident& w, const LlamaGemmWorkspace& gemm,
                             const LlamaTextConfig& cfg, int max_tokens)
    : w_(w), g_(gemm), hidden_(cfg.hidden_size), max_tokens_(max_tokens), inter_(w.local_inter) {
  if (!g_.gemm || !g_.ws) throw std::invalid_argument("LlamaDenseMlp: GEMM workspace required");
  if (max_tokens_ <= 0) throw std::invalid_argument("LlamaDenseMlp: max_tokens must be positive");
  if (inter_ <= 0) throw std::invalid_argument("LlamaDenseMlp: the inter slice must be positive");
  const bool fp8 = w_.gate_fp8.payload != nullptr;
  if (fp8 != static_cast<bool>(w_.up_fp8.payload) || fp8 != static_cast<bool>(w_.down_fp8.payload))
    throw std::invalid_argument("LlamaDenseMlp: mixed fp8 and bf16 weights");
  if (!fp8 && (!w_.gate || !w_.up || !w_.down)) throw std::invalid_argument("LlamaDenseMlp: unbound weights");
  const size_t M = static_cast<size_t>(max_tokens_), I = static_cast<size_t>(inter_);
  gate_ = dev_alloc<uint16_t>(M * I);
  up_ = dev_alloc<uint16_t>(M * I);
  act_ = dev_alloc<uint16_t>(M * I);
}

LlamaDenseMlp::~LlamaDenseMlp() {
  cudaFree(gate_);
  cudaFree(up_);
  cudaFree(act_);
}

void LlamaDenseMlp::rebind(const LlamaDenseMlpResident& w) {
  if (w.local_inter != inter_) throw std::invalid_argument("LlamaDenseMlp: rebind changes the inter slice");
  w_ = w;
}

size_t LlamaDenseMlp::scratch_bytes(const LlamaTextConfig&, int64_t local_inter, int max_tokens) {
  return 3 * static_cast<size_t>(std::max(max_tokens, 0)) * static_cast<size_t>(local_inter) * 2;
}

void LlamaDenseMlp::enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_) throw std::invalid_argument("LlamaDenseMlp: tokens exceed max_tokens");
  if (!x || !out) throw std::invalid_argument("LlamaDenseMlp: null pointer");
  const int H = hidden_, I = static_cast<int>(inter_);
  if (w_.gate_fp8.payload) {
    launch_scale_gemm_bf16(x, static_cast<size_t>(H), w_.gate_fp8.payload, w_.gate_fp8.scales, gate_, tokens, I, H,
                           stream, static_cast<size_t>(I), g_.mma_from_rows, g_.ws, g_.ws_bytes);
    launch_scale_gemm_bf16(x, static_cast<size_t>(H), w_.up_fp8.payload, w_.up_fp8.scales, up_, tokens, I, H, stream,
                           static_cast<size_t>(I), g_.mma_from_rows, g_.ws, g_.ws_bytes);
    launch_moe_swiglu_clamp(gate_, up_, act_, static_cast<int64_t>(tokens) * I,
                            std::numeric_limits<float>::infinity(), stream);
    launch_scale_gemm_bf16(act_, static_cast<size_t>(I), w_.down_fp8.payload, w_.down_fp8.scales, out, tokens, H, I,
                           stream, static_cast<size_t>(H), g_.mma_from_rows, g_.ws, g_.ws_bytes);
    return;
  }
  g_.gemm->matmul(x, w_.gate, gate_, tokens, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), g_.ws,
                  g_.ws_bytes, stream);
  g_.gemm->matmul(x, w_.up, up_, tokens, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H), g_.ws,
                  g_.ws_bytes, stream);
  launch_moe_swiglu_clamp(gate_, up_, act_, static_cast<int64_t>(tokens) * I,
                          std::numeric_limits<float>::infinity(), stream);
  g_.gemm->matmul(act_, w_.down, out, tokens, H, I, DType::BF16, GemmOut::BF16, static_cast<size_t>(I), g_.ws,
                  g_.ws_bytes, stream);
}

}  // namespace dgpp
