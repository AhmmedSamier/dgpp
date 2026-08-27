#include "models/gpt_doll.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/doll_attention.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {

namespace {

constexpr uint64_t kKeyDecode = 0xD3C0DE0000000001ull;
constexpr uint64_t kKeyPrefillBase = 0x9E37000000000000ull;
constexpr uint64_t kKeyRolloutBase = 0x7072210000000000ull;
constexpr int kLogitRowsCap = 8;

size_t bf16_bytes(size_t n) { return n * 2; }

}  // namespace

DollModel::DollModel(Arena& arena, StreamPool& streams, GraphCache& graphs,
                     IGemm& gemm, const DollConfig& cfg)
    : arena_(arena), sp_(streams), graphs_(graphs), gemm_(gemm), cfg_(cfg) {
  if (cfg_.heads % cfg_.kv_heads != 0)
    throw std::runtime_error("doll: heads not divisible by kv_heads");
  if (cfg_.head_dim != 128)
    throw std::runtime_error("doll: head_dim fast path requires 128");

  const int mc = cfg_.max_chunk;
  const size_t hid = static_cast<size_t>(cfg_.hidden);
  const size_t qkv_dim =
      (static_cast<size_t>(cfg_.heads) + 2 * cfg_.kv_heads) * cfg_.head_dim;
  const size_t kv_slots =
      static_cast<size_t>(cfg_.max_seq) * cfg_.kv_heads * cfg_.head_dim;

  auto alloc_hot_bytes = [&](size_t bytes) -> void* {
    return arena_.alloc_persistent(MemClass::DeviceHot, bytes);
  };
  auto b16 = [&](size_t elems) -> uint16_t* {
    return static_cast<uint16_t*>(alloc_hot_bytes(bf16_bytes(elems)));
  };

  embed_ = b16(static_cast<size_t>(cfg_.vocab) * hid);
  ln1_.resize(cfg_.layers);
  ln2_.resize(cfg_.layers);
  w_qkv_.resize(cfg_.layers);
  w_o_.resize(cfg_.layers);
  w13_.resize(cfg_.layers);
  w2_.resize(cfg_.layers);
  kc_.resize(cfg_.layers);
  vc_.resize(cfg_.layers);
  for (int l = 0; l < cfg_.layers; ++l) {
    ln1_[l] = b16(hid);
    ln2_[l] = b16(hid);
    w_qkv_[l] = b16(qkv_dim * hid);
    w_o_[l] = b16(static_cast<size_t>(hid) * hid);
    w13_[l] = b16(2ull * cfg_.inter * hid);
    w2_[l] = b16(static_cast<size_t>(hid) * cfg_.inter);
    kc_[l] = b16(kv_slots);
    vc_[l] = b16(kv_slots);
  }
  final_ln_ = b16(hid);
  lm_head_fp8_ = static_cast<uint8_t*>(alloc_hot_bytes(
      static_cast<size_t>(cfg_.vocab) * hid));

  x_ = b16(static_cast<size_t>(mc) * hid);
  normed_ = b16(static_cast<size_t>(mc) * hid);
  qkv_buf_ = b16(static_cast<size_t>(mc) * qkv_dim);
  attn_out_ = b16(static_cast<size_t>(mc) * hid);
  normed2_ = b16(static_cast<size_t>(mc) * hid);
  gateup_ = b16(static_cast<size_t>(mc) * 2 * cfg_.inter);
  act_ = b16(static_cast<size_t>(mc) * cfg_.inter);
  ffn_out_ = b16(static_cast<size_t>(mc) * hid);
  act8_ = static_cast<uint8_t*>(alloc_hot_bytes(kLogitRowsCap * hid));
  bf16_logits_ =
      b16(kLogitRowsCap * static_cast<size_t>(cfg_.vocab));
  logits32_ = static_cast<float*>(
      alloc_hot_bytes(4ull * kLogitRowsCap * cfg_.vocab));
  ids_hist_dev_ = static_cast<int64_t*>(alloc_hot_bytes(
      sizeof(int64_t) * kMaxRolloutSteps));

  tok_dev_ = static_cast<int64_t*>(alloc_hot_bytes(64));
  abs_first_dev_ = static_cast<int32_t*>(alloc_hot_bytes(64));
  nq_dev_ = static_cast<int32_t*>(alloc_hot_bytes(64));
  tok_out_dev_ = static_cast<int64_t*>(alloc_hot_bytes(64));

  gemm_ws_ = alloc_hot_bytes(64ull << 20);

  // Pinned step-interface slab: scalars then max_chunk prompt staging.
  const size_t pin_bytes =
      (4 * sizeof(int64_t) + 8 + static_cast<size_t>(mc) * sizeof(int64_t) +
       256);
  uint8_t* pinslab = static_cast<uint8_t*>(
      arena_.alloc_persistent(MemClass::HostPinned, pin_bytes));
  pin_tok_ = reinterpret_cast<int64_t*>(pinslab);
  pin_pos_ = reinterpret_cast<int32_t*>(pin_tok_ + 2);
  pin_nq_ = pin_pos_ + 2;
  pin_tok_out_ = reinterpret_cast<int64_t*>(pin_nq_ + 4);
  pin_prompt_ = pin_tok_out_ + 1;

  DGPP_CUDA_OK(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&copy_done_, cudaEventDisableTiming));

  reset_generation_state();
}

DollModel::~DollModel() {
  cudaEventDestroy(step_done_);
  cudaEventDestroy(copy_done_);
}

void DollModel::reset_generation_state() { cur_len_ = 0; }

uint64_t DollModel::decode_step_bytes(int seq_len) const {
  const size_t hid = cfg_.hidden;
  const size_t qkv_dim =
      (static_cast<size_t>(cfg_.heads) + 2 * cfg_.kv_heads) * cfg_.head_dim;
  const double per_layer =
      bf16_bytes(qkv_dim * hid) + bf16_bytes(hid * hid) +
      bf16_bytes(2ull * cfg_.inter * hid) +
      bf16_bytes(static_cast<size_t>(hid) * cfg_.inter) +
      static_cast<double>(seq_len) * cfg_.kv_heads * cfg_.head_dim * 2u *
          (1 + cfg_.heads / cfg_.kv_heads);
  const double lm_head_bytes = static_cast<double>(cfg_.vocab) * hid;
  return static_cast<uint64_t>(per_layer * cfg_.layers + lm_head_bytes + hid);
}

size_t DollModel::persistent_hot_bytes(const DollConfig& cfg,
                                       int rollout_steps_cap) {
  const size_t hid = cfg.hidden;
  const size_t qkv_dim =
      (static_cast<size_t>(cfg.heads) + 2 * cfg.kv_heads) * cfg.head_dim;
  const size_t L = cfg.layers;
  const size_t per_layer =
      bf16_bytes(hid) * 2 +                              // ln1+ln2
      bf16_bytes(qkv_dim * hid) +                        // w_qkv
      bf16_bytes(static_cast<size_t>(hid) * hid) +       // w_o
      bf16_bytes(2ull * cfg.inter * hid) +               // w13
      bf16_bytes(static_cast<size_t>(hid) * cfg.inter);  // w2
  const size_t kv_per_layer = bf16_bytes(static_cast<size_t>(cfg.max_seq) *
                                         cfg.kv_heads * cfg.head_dim) * 2;
  const size_t chunk_rows = static_cast<size_t>(cfg.max_chunk);
  const size_t activations =
      bf16_bytes(chunk_rows * (5ull * hid + qkv_dim + 3ull * cfg.inter)) +
      kLogitRowsCap * (2ull * cfg.vocab /* bf16 logits */ +
                       4ull * cfg.vocab /* fp32 logits */ + hid /* act8 */);
  constexpr size_t kScalars = 512;
  return per_layer * L + kv_per_layer * L +
         cfg.vocab * hid * 2ull /* embed */ +
         static_cast<size_t>(cfg.vocab) * hid /* lm_head fp8 */ +
         bf16_bytes(hid) /* final_ln */ + activations + kScalars +
         (64ull << 20) /* gemm workspace */ +
         sizeof(int64_t) * static_cast<size_t>(rollout_steps_cap);
}

void DollModel::init_weights() {
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  const uint64_t seed = cfg_.seed;
  const size_t hid = cfg_.hidden;
  const size_t qkv_dim =
      (static_cast<size_t>(cfg_.heads) + 2 * cfg_.kv_heads) * cfg_.head_dim;
  const float inv_sqrt_hid = 0.02f / std::sqrt(static_cast<float>(hid));
  const float inv_sqrt_inter =
      0.02f / std::sqrt(static_cast<float>(cfg_.inter));

  fill_random_normal_bf16(embed_, static_cast<uint64_t>(cfg_.vocab) * hid,
                          seed ^ 0xA1A1A1u, 0.02f, s);
  for (int l = 0; l < cfg_.layers; ++l) {
    fill_random_normal_bf16(ln1_[l], hid, seed ^ (0xB0000u + l), 0.05f, s);
    fill_random_normal_bf16(ln2_[l], hid, seed ^ (0xC0000u + l), 0.05f, s);
    fill_random_normal_bf16(w_qkv_[l], qkv_dim * hid,
                            seed ^ (0xD00000u + l), inv_sqrt_hid, s);
    fill_random_normal_bf16(w_o_[l], static_cast<uint64_t>(hid) * hid,
                            seed ^ (0xE00000u + l), inv_sqrt_hid, s);
    fill_random_normal_bf16(w13_[l], 2ull * cfg_.inter * hid,
                            seed ^ (0xF00000u + l), inv_sqrt_hid, s);
    fill_random_normal_bf16(w2_[l],
                            static_cast<uint64_t>(hid) * cfg_.inter,
                            seed ^ (0x1000000u + l), inv_sqrt_inter, s);
    const size_t kv_bytes = bf16_bytes(
        static_cast<size_t>(cfg_.max_seq) * cfg_.kv_heads * cfg_.head_dim);
    DGPP_CUDA_OK(cudaMemsetAsync(kc_[l], 0, kv_bytes, s));
    DGPP_CUDA_OK(cudaMemsetAsync(vc_[l], 0, kv_bytes, s));
  }
  fill_random_normal_bf16(final_ln_, hid, seed ^ 0xF10F1F, 0.05f, s);
  // lm_head sigma 0.05: logit spread ~= sigma*sqrt(hidden) >> bf16
  // ulp so greedy argmax does not collapse onto low-id ties.
  fill_random_normal_fp8(lm_head_fp8_,
                         static_cast<uint64_t>(cfg_.vocab) * hid, seed ^ 0x7EA5,
                         0.05f, s);

  f32_lm_head_ok_ =
      gemm_.ensure_plan(1, cfg_.vocab, cfg_.hidden, DType::F8_E4M3,
                        GemmOut::F32, static_cast<size_t>(cfg_.hidden));
  lm_head_out_ = f32_lm_head_ok_ ? GemmOut::F32 : GemmOut::BF16;
  DGPP_LOG_INFO("doll weights seeded; lm_head out={} (fp32 plan exists: {})",
                f32_lm_head_ok_ ? "f32" : "bf16+cast", f32_lm_head_ok_);
  sp_.sync_all();
}

void DollModel::launch_meta_copies(cudaStream_t s, int n_rows, int abs_first,
                                   int64_t token) {
  *pin_tok_ = token;
  *pin_pos_ = abs_first;
  *pin_nq_ = n_rows;
  DGPP_CUDA_OK(cudaMemcpyAsync(tok_dev_, pin_tok_, sizeof(int64_t),
                               cudaMemcpyHostToDevice, s));
  DGPP_CUDA_OK(cudaMemcpyAsync(abs_first_dev_, pin_pos_, sizeof(int32_t),
                               cudaMemcpyHostToDevice, s));
  DGPP_CUDA_OK(cudaMemcpyAsync(nq_dev_, pin_nq_, sizeof(int32_t),
                               cudaMemcpyHostToDevice, s));
}

void DollModel::record_body(cudaStream_t s, const BodyCtx& b) {
  const int rows = b.n_rows;
  if (rows <= 0 || rows > cfg_.max_chunk)
    throw std::runtime_error("doll: row count out of range");
  // Debug aid: DGPP_STOP_STAGE aborts record_body after a chosen stage so
  // self-tests can inspect intermediate buffers.
  static const int kStopStage = [] {
    const char* e = std::getenv("DGPP_STOP_STAGE");
    return e ? std::atoi(e) : 0;
  }();
  const int hid = cfg_.hidden;
  const int hq_dh = cfg_.heads * cfg_.head_dim;
  const int kv_dim = cfg_.kv_heads * cfg_.head_dim;
  const int qkv_dim = hq_dh + 2 * kv_dim;
  constexpr size_t kWsBytes = 64ull << 20;
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

  embed_gather_bf16(embed_, tok_dev_, x_, rows, hid, s);
  if (kStopStage == 1) return;
  for (int l = 0; l < cfg_.layers; ++l) {
    rmsnorm_bf16(x_, ln1_[l], normed_, rows, hid, cfg_.eps, s);
    if (kStopStage == 2) return;
    gemm_.matmul(normed_, w_qkv_[l], qkv_buf_, rows, qkv_dim, hid,
                 DType::BF16, GemmOut::BF16, static_cast<size_t>(hid),
                 gemm_ws_, kWsBytes, s);
    if (kStopStage == 3) return;
    kv_append_from_pairs(qkv_buf_ + hq_dh, qkv_dim, kc_[l], vc_[l],
                         abs_first_dev_, rows, kv_dim, s);
    gqa_causal_attention(qkv_buf_, qkv_dim, kc_[l], vc_[l], attn_out_,
                         nq_dev_, abs_first_dev_, rows, cfg_.heads,
                         cfg_.kv_heads, cfg_.head_dim, attn_scale, s);
    add_inplace_bf16(x_, attn_out_, static_cast<int64_t>(rows) * hid, s);
    rmsnorm_bf16(x_, ln2_[l], normed2_, rows, hid, cfg_.eps, s);
    gemm_.matmul(normed2_, w13_[l], gateup_, rows, 2 * cfg_.inter, hid,
                 DType::BF16, GemmOut::BF16, static_cast<size_t>(hid),
                 gemm_ws_, kWsBytes, s);
    swiglu_gateup_pairs_bf16(gateup_, act_, rows, cfg_.inter,
                             cfg_.swiglu_limit, s);
    gemm_.matmul(act_, w2_[l], ffn_out_, rows, hid, cfg_.inter, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(cfg_.inter), gemm_ws_,
                 kWsBytes, s);
    add_inplace_bf16(x_, ffn_out_, static_cast<int64_t>(rows) * hid, s);
  }
  rmsnorm_bf16(x_ + static_cast<int64_t>(b.logit_row_idx) * hid, final_ln_,
               normed_, 1, hid, cfg_.eps, s);
  cast_bf16_to_fp8_rows(normed_, act8_, hid, s);
  if (kStopStage == 4) return;
  if (lm_head_out_ == GemmOut::F32) {
    gemm_.matmul(act8_, lm_head_fp8_, logits32_, 1, cfg_.vocab, hid,
                 DType::F8_E4M3, GemmOut::F32, static_cast<size_t>(hid),
                 gemm_ws_, kWsBytes, s);
  } else {
    gemm_.matmul(act8_, lm_head_fp8_, bf16_logits_, 1, cfg_.vocab, hid,
                 DType::F8_E4M3, GemmOut::BF16, static_cast<size_t>(hid),
                 gemm_ws_, kWsBytes, s);
    cast_bf16_to_f32_rows(bf16_logits_, logits32_, 1, cfg_.vocab, s);
  }
  argmax_rows_f32(logits32_, b.ids_out, nullptr, 1, cfg_.vocab, s);
  bump_i32_device(abs_first_dev_, rows, s);
}

void DollModel::finish_step_outputs(cudaStream_t s) {
  // Result copy + fence live OUTSIDE every captured graph: events recorded
  // during capture are absorbed as graph nodes and misbehave when
  // synchronized from the host afterwards.
  DGPP_CUDA_OK(cudaMemcpyAsync(pin_tok_out_, tok_out_dev_, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, s));
  DGPP_CUDA_OK(cudaEventRecord(copy_done_, s));
}

void DollModel::enqueue_prefill(const std::vector<int64_t>& tokens) {
  const int n = static_cast<int>(tokens.size());
  if (n <= 0) throw std::runtime_error("doll: empty prompt");
  if (cur_len_ + n > cfg_.max_seq)
    throw std::runtime_error("doll: sequence overflow");
  if (n > cfg_.max_chunk)
    throw std::runtime_error("doll: prompt exceeds max_chunk");

  cudaStream_t s = sp_.stream(StreamPool::Compute);
  std::memcpy(pin_prompt_, tokens.data(), sizeof(int64_t) * n);
  launch_meta_copies(s, n, cur_len_, tokens.front());
  // Async from pinned staging keeps capture-safe semantics; the body's gather
  // consumes all n rows via the device copy below.
  DGPP_CUDA_OK(cudaMemcpyAsync(tok_dev_, pin_prompt_, sizeof(int64_t) * n,
                               cudaMemcpyHostToDevice, s));

  GraphCache::Key key = kKeyPrefillBase | static_cast<uint64_t>(n);
  graphs_.replay_or_capture(
      key, "prefill_n" + std::to_string(n), s,
      [this, n](cudaStream_t cs) {
        BodyCtx bc{Mode::Prefill, n, n - 1, tok_out_dev_};
        record_body(cs, bc);
      });
  finish_step_outputs(s);
  cur_len_ += n;
}

int64_t DollModel::sample_prefill_result() {
  DGPP_CUDA_OK(cudaEventSynchronize(copy_done_));
  return *pin_tok_out_;
}

void DollModel::enqueue_decode_step(int64_t token) {
  if (cur_len_ >= cfg_.max_seq)
    throw std::runtime_error("doll: sequence overflow");
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  launch_meta_copies(s, 1, cur_len_, token);
  graphs_.replay_or_capture(kKeyDecode, "decode_bs1", s,
                            [this](cudaStream_t cs) {
                              BodyCtx b{Mode::Decode, 1, 0, tok_out_dev_};
                              record_body(cs, b);
                            });
  finish_step_outputs(s);
  cur_len_ += 1;
}

int64_t DollModel::poll_decode_sample() {
  DGPP_CUDA_OK(cudaEventSynchronize(copy_done_));
  return *pin_tok_out_;
}

int64_t DollModel::eager_decode_step(int64_t token) {
  if (cur_len_ >= cfg_.max_seq)
    throw std::runtime_error("doll: sequence overflow");
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  launch_meta_copies(s, 1, cur_len_, token);
  BodyCtx b{Mode::Decode, 1, 0, tok_out_dev_};
  record_body(s, b);
  DGPP_CUDA_OK(cudaMemcpyAsync(pin_tok_out_, tok_out_dev_, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, s));
  cur_len_ += 1;
  DGPP_CUDA_OK(cudaEventRecord(copy_done_, s));
  return poll_decode_sample();
}

int64_t DollModel::eager_prefill_result(const std::vector<int64_t>& tokens) {
  const int n = static_cast<int>(tokens.size());
  if (n <= 0 || cur_len_ + n > cfg_.max_seq || n > cfg_.max_chunk)
    throw std::runtime_error("doll: bad eager prefill request");
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  std::memcpy(pin_prompt_, tokens.data(), sizeof(int64_t) * n);
  launch_meta_copies(s, n, cur_len_, tokens.front());
  DGPP_CUDA_OK(cudaMemcpyAsync(tok_dev_, pin_prompt_, sizeof(int64_t) * n,
                               cudaMemcpyHostToDevice, s));
  BodyCtx b{Mode::Prefill, n, n - 1, tok_out_dev_};
  record_body(s, b);
  DGPP_CUDA_OK(cudaMemcpyAsync(pin_tok_out_, tok_out_dev_, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, s));
  cur_len_ += n;
  DGPP_CUDA_OK(cudaEventRecord(copy_done_, s));
  return poll_decode_sample();
}

void DollModel::reset_kv() {
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  const size_t kv_bytes = bf16_bytes(static_cast<size_t>(cfg_.max_seq) *
                                     cfg_.kv_heads * cfg_.head_dim);
  for (int l = 0; l < cfg_.layers; ++l) {
    DGPP_CUDA_OK(cudaMemsetAsync(kc_[l], 0, kv_bytes, s));
    DGPP_CUDA_OK(cudaMemsetAsync(vc_[l], 0, kv_bytes, s));
  }
  cur_len_ = 0;
}

void DollModel::capture_rollout(int steps) {
  if (steps <= 0 || steps > kMaxRolloutSteps)
    throw std::runtime_error("doll: rollout steps out of range");
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  rollout_steps_ = steps;
  pinned_ids_.assign(steps, -1);
  GraphCache::Key key = kKeyRolloutBase | static_cast<uint64_t>(steps);
  graphs_.replay_or_capture(key, "rollout_" + std::to_string(steps), s,
                            [this, steps](cudaStream_t cs) {
                              for (int i = 0; i < steps; ++i) {
                                if (i > 0)
                                  copy_i64_device_to_device(&ids_hist_dev_[i - 1],
                                                            tok_dev_, cs);
                                BodyCtx b{Mode::Decode, 1, 0, &ids_hist_dev_[i]};
                                record_body(cs, b);
                              }
                            });
}

void DollModel::run_rollout(int64_t seed_token) {
  cudaStream_t s = sp_.stream(StreamPool::Compute);
  if (rollout_steps_ <= 0)
    throw std::runtime_error("doll: rollout not captured");
  const size_t kv_bytes = bf16_bytes(static_cast<size_t>(cfg_.max_seq) *
                                     cfg_.kv_heads * cfg_.head_dim);
  for (int l = 0; l < cfg_.layers; ++l) {
    DGPP_CUDA_OK(cudaMemsetAsync(kc_[l], 0, kv_bytes, s));
    DGPP_CUDA_OK(cudaMemsetAsync(vc_[l], 0, kv_bytes, s));
  }
  DGPP_CUDA_OK(cudaGetLastError());
  launch_meta_copies(s, 1, 0, seed_token);
  GraphCache::Key key = kKeyRolloutBase | static_cast<uint64_t>(rollout_steps_);
  graphs_.launch(key, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));  // surface graph errors here
  DGPP_CUDA_OK(cudaMemcpy(pinned_ids_.data(), ids_hist_dev_,
                          sizeof(int64_t) * rollout_steps_,
                          cudaMemcpyDeviceToHost));
  cur_len_ += rollout_steps_;
}

}  // namespace dgpp