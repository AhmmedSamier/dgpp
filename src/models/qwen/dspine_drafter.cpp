#include "models/qwen/dspine_drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "engine/decode_outputs.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/dspine.hpp"
#include "kernels/kernels.hpp"
#include "kernels/qsa.hpp"

namespace dgpp {

DSpineDrafter::~DSpineDrafter() { free(); }

void DSpineDrafter::init(const DFlash2Host& host, const DFlashConfig& cfg,
                         const std::string& dir) {
  if (on_) throw std::logic_error("DSpineDrafter: init called twice");
  if (!cfg.dspine) throw std::invalid_argument("DSpineDrafter: DSpine config required");
  if (host.world != 1 || host.rank != 0)
    throw std::invalid_argument("DSpineDrafter: this checkpoint currently requires tp=1");
  if (!host.lm_head_bf16 || host.vocab_begin != 0 || host.vocab_count != cfg.vocab_size)
    throw std::invalid_argument("DSpineDrafter: an unsharded BF16 target lm_head is required");
  if (host.weights_fp8 || host.rope)
    throw std::invalid_argument("DSpineDrafter: BF16 weights and default draft RoPE are required");
  if (cfg.hidden_size != host.hidden || cfg.vocab_size != host.target_vocab ||
      cfg.num_hidden_layers < 1 || cfg.block_size > kSpecRows)
    throw std::invalid_argument("DSpineDrafter: target geometry mismatch");
  if (!cfg.tp_divisible(host.world)) throw std::invalid_argument("DSpineDrafter: invalid TP geometry");
  host_ = host;
  cfg_ = cfg;
  weights_ = load_dflash_weights(cfg_, dir, host_.stream, host_.rank, host_.world);
  tap_map_.assign(static_cast<size_t>(host_.num_target_layers), -1);
  for (size_t i = 0; i < cfg_.target_layer_ids.size(); ++i)
    tap_map_.at(static_cast<size_t>(cfg_.target_layer_ids[i])) = static_cast<int>(i);

  const int H = host_.hidden, QR = cfg_.query_rows(), D = cfg_.drafts();
  const int V = cfg_.vocab_size, C = cfg_.code_size, R = cfg_.code_rank;
  const int K = cfg_.code_head_topk, N = D * K;
  q_width_ = cfg_.q_row();
  kv_width_ = cfg_.kv_row();
  inter_ = cfg_.intermediate_size;
  heads_ = cfg_.num_attention_heads;
  kv_heads_ = cfg_.num_key_value_heads;
  auto alloc = [](auto** p, size_t n) { DGPP_CUDA_OK(cudaMalloc(p, n)); };
  alloc(&acc_, static_cast<size_t>(host_.max_tokens) * H * sizeof(float));
  alloc(&tap_out_, static_cast<size_t>(host_.max_tokens) * H * sizeof(float));
  alloc(&x_, static_cast<size_t>(QR) * H * 2);
  alloc(&resid_, static_cast<size_t>(QR) * H * 2);
  alloc(&q_, static_cast<size_t>(QR) * q_width_ * 2);
  alloc(&k_, static_cast<size_t>(QR) * kv_width_ * 2);
  alloc(&v_, static_cast<size_t>(QR) * kv_width_ * 2);
  alloc(&attn_, static_cast<size_t>(QR) * q_width_ * 2);
  alloc(&proj_, static_cast<size_t>(QR) * H * 2);
  alloc(&gate_, static_cast<size_t>(QR) * inter_ * 2);
  alloc(&up_, static_cast<size_t>(QR) * inter_ * 2);
  alloc(&mlp_, static_cast<size_t>(QR) * inter_ * 2);
  alloc(&hidden_, static_cast<size_t>(QR) * H * 2);
  alloc(&raw_, static_cast<size_t>(QR) * H * 2);
  alloc(&normalized_, static_cast<size_t>(QR) * H * 2);
  alloc(&codes_, static_cast<size_t>(QR) * C * 2);
  alloc(&packed_, static_cast<size_t>(QR) * 3 * R * 2);
  alloc(&message_, static_cast<size_t>(QR) * R * 2);
  alloc(&delta_, static_cast<size_t>(QR) * H * 2);
  alloc(&scaled_, static_cast<size_t>(N) * std::max(H, C) * 2);
  alloc(&candidate_head_, static_cast<size_t>(N) * H * 2);
  alloc(&sender_codes_, static_cast<size_t>(N) * C * 2);
  alloc(&tg_, static_cast<size_t>(N) * R * 2);
  alloc(&tsv_gate_, static_cast<size_t>(N) * R * 2);
  alloc(&tsv_value_, static_cast<size_t>(N) * R * 2);
  alloc(&tsv_, static_cast<size_t>(N) * 2 * R * 2);
  alloc(&transposed_write_, static_cast<size_t>(R) * H * 2);
  alloc(&rms_, static_cast<size_t>(QR) * sizeof(float));
  alloc(&logits_, static_cast<size_t>(QR) * V * sizeof(float));
  alloc(&prewrite_logits_, static_cast<size_t>(N) * sizeof(float));
  alloc(&scaled_scores_, static_cast<size_t>(N) * sizeof(float));
  alloc(&attn_partials_, dflash2_block_attn_partials_bytes(QR, heads_));
  topk_workspace_bytes_ = dflash2_topk_ws_bytes(V, D, K);
  alloc(&topk_workspace_, topk_workspace_bytes_);
  alloc(&ids_, static_cast<size_t>(N) * sizeof(int32_t));
  alloc(&sender_ids_, static_cast<size_t>(N) * sizeof(int32_t));
  alloc(&chain_index_, static_cast<size_t>(N) * sizeof(int32_t));
  alloc(&drafts_, static_cast<size_t>(D) * sizeof(int32_t));
  alloc(&positions_, static_cast<size_t>(QR) * sizeof(int64_t));
  alloc(&tokens_, static_cast<size_t>(QR) * sizeof(int64_t));
  DGPP_CUDA_OK(cudaMallocHost(&host_io_, static_cast<size_t>(2 * QR) * sizeof(int64_t)));
  std::vector<float> inv(static_cast<size_t>(cfg_.head_dim / 2));
  for (size_t i = 0; i < inv.size(); ++i)
    inv[i] = static_cast<float>(std::pow(cfg_.rope_theta,
        -static_cast<double>(2 * i) / static_cast<double>(cfg_.head_dim)));
  DGPP_CUDA_OK(cudaMalloc(&inv_freq_, inv.size() * sizeof(float)));
  DGPP_CUDA_OK(cudaMemcpyAsync(inv_freq_, inv.data(), inv.size() * sizeof(float),
                               cudaMemcpyHostToDevice, host_.stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(host_.stream));
  on_ = true;
}

void DSpineDrafter::free() {
  cudaFree(acc_); cudaFree(tap_out_); cudaFree(x_); cudaFree(resid_); cudaFree(q_);
  cudaFree(k_); cudaFree(v_); cudaFree(attn_); cudaFree(proj_); cudaFree(gate_);
  cudaFree(up_); cudaFree(mlp_); cudaFree(hidden_); cudaFree(raw_);
  cudaFree(normalized_); cudaFree(codes_); cudaFree(packed_); cudaFree(message_);
  cudaFree(delta_); cudaFree(scaled_); cudaFree(candidate_head_); cudaFree(sender_codes_);
  cudaFree(tg_); cudaFree(tsv_gate_); cudaFree(tsv_value_); cudaFree(tsv_); cudaFree(rms_);
  cudaFree(transposed_write_); cudaFree(logits_); cudaFree(prewrite_logits_); cudaFree(scaled_scores_); cudaFree(attn_partials_);
  cudaFree(topk_workspace_); cudaFree(ids_); cudaFree(sender_ids_); cudaFree(chain_index_);
  cudaFree(drafts_); cudaFree(positions_); cudaFree(tokens_); cudaFree(inv_freq_);
  cudaFreeHost(host_io_);
  acc_ = tap_out_ = nullptr; x_ = resid_ = q_ = k_ = v_ = attn_ = proj_ = gate_ = up_ = mlp_ = nullptr;
  hidden_ = raw_ = normalized_ = codes_ = packed_ = message_ = delta_ = scaled_ = candidate_head_ = nullptr;
  sender_codes_ = tg_ = tsv_gate_ = tsv_value_ = tsv_ = transposed_write_ = nullptr;
  rms_ = logits_ = prewrite_logits_ = scaled_scores_ = attn_partials_ = nullptr;
  topk_workspace_ = nullptr; ids_ = sender_ids_ = chain_index_ = drafts_ = nullptr;
  positions_ = tokens_ = nullptr; inv_freq_ = nullptr; host_io_ = nullptr; on_ = false;
}

int DSpineDrafter::tap(int target_layer) const {
  if (target_layer < 0 || static_cast<size_t>(target_layer) >= tap_map_.size()) return -1;
  return tap_map_[static_cast<size_t>(target_layer)];
}

void DSpineDrafter::store_features(int rows, const int32_t* req_ids,
                                  const int64_t* positions) {
  if (!on_ || rows <= 0) return;
  const int H = host_.hidden, QR = cfg_.query_rows();
  const int KV = kv_width_, KVH = kv_heads_, HD = cfg_.head_dim;
  const int32_t* table = host_.device_tables;
  const int bpr = host_.total_blocks, bt = host_.block_tokens;
  for (int off = 0; off < rows; off += QR) {
    const int count = std::min(QR, rows - off);
    host_.gemm_rows(count, false);
    dspine_context_norm_f32_bf16(acc_ + static_cast<size_t>(off) * H, weights_.hidden_norm,
                                 x_, count, H, cfg_.rms_norm_eps, host_.stream);
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
      const auto& w = weights_.layers[static_cast<size_t>(l)];
      host_.gemm->matmul(x_, w.k, k_, count, KV, H, DType::BF16, GemmOut::BF16,
                         static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
      host_.gemm->matmul(x_, w.v, v_, count, KV, H, DType::BF16, GemmOut::BF16,
                         static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
      dflash2_norm_rope_bf16(k_, KV, w.k_norm, positions + off, inv_freq_, k_, KV,
                             count, KVH, HD, cfg_.rms_norm_eps, host_.stream);
      const auto plane = host_.plane(l);
      qsa_kv_append(k_, KV, v_, KV, req_ids + off, positions + off, count, table, bpr, bt,
                    KVH, HD, plane.k, plane.v, nullptr, nullptr, host_.stream);
    }
  }
}

void DSpineDrafter::block_layers(int request) {
  const int H = host_.hidden, T = cfg_.query_rows(), QW = q_width_, KV = kv_width_;
  const int HD = cfg_.head_dim, I = inter_, V = cfg_.vocab_size;
  const float eps = cfg_.rms_norm_eps, scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const int32_t* table = host_.device_tables + static_cast<size_t>(request) * host_.total_blocks;
  const int bpr = host_.total_blocks, bt = host_.block_tokens;
  int32_t* zero = reinterpret_cast<int32_t*>(scaled_); // aligned scratch prefix
  DGPP_CUDA_OK(cudaMemsetAsync(zero, 0, static_cast<size_t>(T) * sizeof(int32_t), host_.stream));
  bool input_normed = false;
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    const auto& w = weights_.layers[static_cast<size_t>(l)];
    if (!input_normed)
      dflash2_rmsnorm_bf16(resid_, w.input_norm, x_, T, H, eps, host_.stream);
    host_.gemm->matmul(x_, w.q, q_, T, QW, H, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    host_.gemm->matmul(x_, w.k, k_, T, KV, H, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    host_.gemm->matmul(x_, w.v, v_, T, KV, H, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    dflash2_norm_rope_bf16(q_, QW, w.q_norm, positions_, inv_freq_, q_, QW, T,
                           heads_, HD, eps, host_.stream);
    dflash2_norm_rope_bf16(k_, KV, w.k_norm, positions_, inv_freq_, k_, KV, T,
                           kv_heads_, HD, eps, host_.stream);
    const auto plane = host_.plane(l);
    qsa_kv_append(k_, KV, v_, KV, zero, positions_, T, table,
                  bpr, bt, kv_heads_, HD, plane.k, plane.v, nullptr, nullptr, host_.stream);
    dflash2_block_attn_split(q_, QW, plane.k, plane.v, table,
                            bt, bpr, T, cfg_.attention_window(l), positions_, T,
                            heads_, kv_heads_, HD, scale, attn_partials_, attn_, host_.stream,
                            false, true);
    host_.gemm->matmul(attn_, w.o, proj_, T, H, QW, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(QW), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    dflash2_add_rmsnorm_bf16(resid_, proj_, w.post_norm, x_, T, H, eps, host_.stream);
    host_.gemm->matmul(x_, w.gate, gate_, T, I, H, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    host_.gemm->matmul(x_, w.up, up_, T, I, H, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(H), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    swiglu_limit_bf16(gate_, up_, mlp_, static_cast<int64_t>(T) * I, INFINITY, host_.stream);
    host_.gemm->matmul(mlp_, w.down, proj_, T, H, I, DType::BF16, GemmOut::BF16,
                       static_cast<size_t>(I), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
    add_inplace_bf16(resid_, proj_, static_cast<int64_t>(T) * H, host_.stream);

    if (std::binary_search(cfg_.code_feature_layers.begin(), cfg_.code_feature_layers.end(), l + 1)) {
      const uint16_t* read = l + 1 == cfg_.num_hidden_layers
          ? weights_.read_weight
          : weights_.early_read_weights + static_cast<size_t>(l) * cfg_.code_size * H;
      dspine_prepare_read(resid_, nullptr, raw_, normalized_, rms_, T, H, cfg_.rms_norm_eps,
                          false, host_.stream);
      host_.gemm->matmul(normalized_, read, codes_, T, cfg_.code_size, H, DType::BF16,
                         GemmOut::BF16, static_cast<size_t>(H), host_.gemm_ws,
                         host_.gemm_ws_bytes, host_.stream);
      dspine_normalize_codes(codes_, weights_.codebook, tokens_, codes_, T, T, V,
                             cfg_.code_size, cfg_.code_eps, host_.stream);
      const int C = cfg_.code_size, R = cfg_.code_rank;
      // feature_message consumes three row-major planes [q; sender-gate; value].
      host_.gemm->matmul(codes_, weights_.bridge_gate, packed_, T, R, C, DType::BF16,
                         GemmOut::BF16, static_cast<size_t>(C), host_.gemm_ws,
                         host_.gemm_ws_bytes, host_.stream);
      host_.gemm->matmul(codes_, weights_.predecessor_gate,
                         packed_ + static_cast<size_t>(T) * R, T, R, C,
                         DType::BF16, GemmOut::BF16, static_cast<size_t>(C), host_.gemm_ws,
                         host_.gemm_ws_bytes, host_.stream);
      host_.gemm->matmul(codes_, weights_.previous_value,
                         packed_ + static_cast<size_t>(2) * T * R, T, R, C,
                         DType::BF16, GemmOut::BF16, static_cast<size_t>(C), host_.gemm_ws,
                         host_.gemm_ws_bytes, host_.stream);
      dspine_feature_message(packed_, weights_.bridge_gate_bias, message_, T, T, R, host_.stream);
      const auto* write = weights_.feature_write_weights + static_cast<size_t>(l) * H * R;
      host_.gemm->matmul(message_, write, delta_, T, H, R, DType::BF16, GemmOut::BF16,
                         static_cast<size_t>(R), host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
      const uint16_t* next_norm = l + 1 == cfg_.num_hidden_layers
          ? weights_.norm : weights_.layers[static_cast<size_t>(l + 1)].input_norm;
      dspine_feature_finish(raw_, delta_, rms_, positions_, next_norm, resid_, x_, T, T,
                            H, eps, host_.stream);
      input_normed = true;
    } else {
      input_normed = false;
    }
  }
  // The checkpoint validates that its last layer has a bridge write.
  copy_rows_bf16(hidden_, x_, T, H, host_.stream);
}

void DSpineDrafter::select_chain() {
  const int D = cfg_.drafts(), K = cfg_.code_head_topk, H = host_.hidden;
  const int V = cfg_.vocab_size, C = cfg_.code_size, R = cfg_.code_rank, N = D * K;
  host_.gemm->matmul(hidden_, host_.lm_head_bf16, logits_, cfg_.query_rows(), V, H,
                     DType::BF16, GemmOut::F32, static_cast<size_t>(H),
                     host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
  dflash2_topk_f32(logits_ + V, ids_, scaled_scores_, V, D, K, host_.stream,
                   topk_workspace_, topk_workspace_bytes_);
  host_.gemm_rows(N, false);
  dspine_gather_rows(host_.lm_head_bf16, ids_, candidate_head_, N, H, host_.stream);
  dspine_scale_rows(candidate_head_, weights_.norm, scaled_, N, H, 1.0f, host_.stream);
  const auto* last_write = weights_.feature_write_weights +
      static_cast<size_t>(cfg_.num_hidden_layers - 1) * H * R;
  dspine_transpose_bf16(last_write, transposed_write_, H, R, host_.stream);
  host_.gemm->matmul(scaled_, transposed_write_, tg_, N, R, H, DType::BF16,
                     GemmOut::BF16, static_cast<size_t>(H), host_.gemm_ws,
                     host_.gemm_ws_bytes, host_.stream);
  dspine_sender_ids(ids_, tokens_, sender_ids_, D, K, host_.stream);
  dspine_gather_rows(weights_.codebook, sender_ids_, sender_codes_, N, C, host_.stream);
  dspine_scale_rows(sender_codes_, nullptr, scaled_, N, C,
                    std::sqrt(static_cast<float>(C)), host_.stream);
  host_.gemm->matmul(scaled_, weights_.predecessor_gate, tsv_gate_, N, R, C,
                     DType::BF16, GemmOut::BF16, static_cast<size_t>(C),
                     host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
  host_.gemm->matmul(scaled_, weights_.previous_value, tsv_value_, N, R, C,
                     DType::BF16, GemmOut::BF16, static_cast<size_t>(C),
                     host_.gemm_ws, host_.gemm_ws_bytes, host_.stream);
  dspine_pack_pair(tsv_gate_, tsv_value_, tsv_, N, R, host_.stream);
  dspine_prewrite_scores(raw_, weights_.norm, host_.lm_head_bf16, ids_, prewrite_logits_,
                         D, K, H, V, cfg_.query_rows(), host_.stream);
  // One request owns this draft block. D is the number of proposal rows,
  // not the number of independent blocks; passing D here made the kernel
  // walk D * (T - 1) rows through buffers sized for only T - 1.
  dspine_serial_table(ids_, prewrite_logits_, tokens_, packed_, rms_,
                      weights_.bridge_gate_bias, tg_, tsv_, 1, cfg_.query_rows(),
                      R, K, V, cfg_.code_refine_weight, chain_index_, host_.stream);
  dspine_serial_chain(ids_, chain_index_, 1, cfg_.query_rows(), K, drafts_, host_.stream);
}

bool DSpineDrafter::draft(int request, int64_t anchor,
                          std::vector<int32_t>* out) {
  if (!on_ || !out) throw std::logic_error("DSpineDrafter: not initialized or null output");
  const int T = cfg_.query_rows();
  const int64_t pos = host_.session_position(request);
  if (pos < 0 || pos + T > host_.max_context) return false;
  if (!host_.ensure_request_blocks(request, pos + T, host_.stream)) return false;
  for (int i = 0; i < T; ++i) {
    host_io_[i] = pos + i;
    host_io_[T + i] = i == 0 ? anchor : cfg_.mask_token_id;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(positions_, host_io_, static_cast<size_t>(T) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, host_.stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(tokens_, host_io_ + T, static_cast<size_t>(T) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, host_.stream));
  embed_gather_bf16(host_.embed, tokens_, resid_, T, host_.hidden, host_.stream);
  host_.gemm_rows(T, true);
  block_layers(request);
  select_chain();
  if (const char* trace = std::getenv("DGPP_DFLASH2_TRACE");
      trace && *trace && *trace != '0') {
    static int traced = 0;
    if (traced++ < 4) {
      int32_t candidates[64]{};
      const int k = cfg_.code_head_topk;
      DGPP_CUDA_OK(cudaMemcpy(candidates, ids_, static_cast<size_t>(k) * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
      std::string detail;
      for (int i = 0; i < k; ++i) {
        if (!detail.empty()) detail += ' ';
        detail += std::to_string(candidates[i]);
      }
      logf(LogLevel::Info, "dflash trace: DSpine anchor={} pos={} row-1 top{} candidates [{}]", anchor, pos, k, detail);
    }
  }
  out->resize(static_cast<size_t>(cfg_.drafts()));
  DGPP_CUDA_OK(cudaMemcpyAsync(out->data(), drafts_, out->size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, host_.stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(host_.stream));
  if (const char* trace = std::getenv("DGPP_DFLASH2_TRACE");
      trace && *trace && *trace != '0') {
    static int traced_block = 0;
    if (traced_block++ < 4) {
      std::string detail;
      for (const int32_t token : *out) {
        if (!detail.empty()) detail += ' ';
        detail += std::to_string(token);
      }
      logf(LogLevel::Info, "dflash trace: DSpine proposal [{}]", detail);
    }
  }
  return true;
}

void DSpineDrafter::plan_entries(const DFlashConfig& cfg, int max_tokens,
                                int64_t vocab_count,
                                std::vector<std::pair<std::string, size_t>>* out) {
  if (!out) throw std::invalid_argument("DSpineDrafter::plan_entries: null output");
  const size_t H = cfg.hidden_size, T = cfg.query_rows(), D = cfg.drafts();
  const size_t V = static_cast<size_t>(vocab_count), C = cfg.code_size, R = cfg.code_rank;
  const size_t N = D * static_cast<size_t>(cfg.code_head_topk);
  const size_t I = static_cast<size_t>(cfg.local_intermediate(1));
  const size_t Q = static_cast<size_t>(cfg.q_row()), KV = static_cast<size_t>(cfg.kv_row());
  const size_t K = static_cast<size_t>(cfg.code_head_topk);
  const size_t device = 2 * static_cast<size_t>(max_tokens) * H * sizeof(float) +
      (2 * T * H + 2 * T * Q + 2 * T * KV + T * H + 3 * T * I +
       3 * T * H + T * C + 3 * T * R + T * R + T * H) * 2 +
      N * std::max(H, C) * 2 + N * H * 2 + N * C * 2 + 5 * N * R * 2 + R * H * 2 +
      T * sizeof(float) + T * V * sizeof(float) + 2 * N * sizeof(float) +
      dflash2_block_attn_partials_bytes(static_cast<int>(T), cfg.num_attention_heads) +
      dflash2_topk_ws_bytes(static_cast<int64_t>(V), static_cast<int>(D), static_cast<int>(K)) +
      (3 * N + D) * sizeof(int32_t) + 2 * T * sizeof(int64_t) +
      static_cast<size_t>(cfg.head_dim / 2) * sizeof(float);
  out->emplace_back("DSpine BF16 checkpoint weights", dflash_weights_bytes(cfg));
  out->emplace_back("DSpine draft, head and selected-prefix scratch",
                    device);
}

}  // namespace dgpp
