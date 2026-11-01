// Nemotron-3.5-Transcribe model (milestone 4): the SessionModel that runs
// the 52-slot Nemotron-H hybrid walk on the GB10. MTP is not yet wired.

#include "models/nemotron_transcribe/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "engine/boundary_reducer.hpp"
#include "kernels/kernels.hpp"
#include "models/nemotron_transcribe/forward.hpp"

namespace dgpp {
namespace {

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

}  // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

NemotronModel::NemotronModel(const NemotronTranscribeConfig& cfg,
                             const std::string& checkpoint_dir, int max_tokens,
                             int64_t max_cache_tokens, NemotronResidency residency,
                             BoundaryReducer* boundary, int tp_rank, int tp_world,
                             int max_requests, bool /*mtp*/, int decode_rows)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              NemotronHeadSharding::Full, false),
      arena_(),
      stream_(nullptr),
      max_tokens_(max_tokens),
      max_cache_tokens_(max_cache_tokens),
      max_requests_(max_requests),
      hidden_(cfg.lm.hidden_size),
      vocab_size_(cfg.lm.vocab_size) {
  // Capture the primary CUDA context on the construction (main) thread so
  // run_rows() — which the engine thread executes — can switch back to it.
  // cudaSetDevice(0) alone does not migrate to the primary context when a
  // non-primary context is already current on a thread, and every device
  // pointer (weights + activations) was allocated under this context.
  // IMPORTANT: cudaSetDevice(0) must run BEFORE cuCtxGetCurrent so we capture
  // the same primary context under which all subsequent allocations happen.
  DGPP_CUDA_OK(cudaSetDevice(0));
  CUcontext ctx = nullptr;
  cuCtxGetCurrent(&ctx);
  primary_ctx_ = ctx;
  if (max_tokens <= 0) throw std::invalid_argument("NemotronModel: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > 32)
    throw std::invalid_argument("NemotronModel: max_requests must be in [1, 32]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument("NemotronModel: boundary reducer required exactly when tp_world > 1");
  if (cfg_.lm.eos_token_ids.empty())
    throw std::invalid_argument("NemotronModel: the config names no EOS token");

  const auto& lm = cfg_.lm;
  mamba_d_inner_ = lm.mamba_num_heads * lm.mamba_head_dim;
  mamba_groups_state_ = lm.mamba_num_groups * lm.mamba_state_size;
  mamba_heads_ = lm.mamba_num_heads;
  mamba_head_dim_ = lm.mamba_head_dim;
  mamba_state_size_ = lm.mamba_state_size;
  mamba_conv_kernel_ = lm.conv_kernel;
  attn_q_rows_ = lm.num_attention_heads * lm.head_dim;
  attn_kv_rows_ = lm.num_key_value_heads * lm.head_dim;
  attn_q_heads_ = lm.num_attention_heads;
  attn_kv_heads_ = lm.num_key_value_heads;
  attn_head_dim_ = lm.head_dim;

  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
  log_memory_ledger("nemotron: before load");

  globals_ = loader_.load_globals();
  log_memory_ledger("nemotron: globals resident");

  init_activations(max_tokens);
  log_memory_ledger("nemotron: activations allocated");

  // The mamba state pool: one region per request, across all mamba slots.
  NemotronMambaGeometry g;
  g.num_mamba_slots = 0;
  for (auto s : lm.slots)
    if (s == NemotronSlot::Mamba) ++g.num_mamba_slots;
  g.heads = lm.mamba_num_heads;
  g.head_dim = lm.mamba_head_dim;
  g.groups = lm.mamba_num_groups;
  g.state_size = lm.mamba_state_size;
  g.conv_dim = nemotron_mamba_geometry(lm).conv_channels;
  g.conv_kernel = lm.conv_kernel;

  // Initialize the arena slab before the pool allocates from it.
  Arena::Config ac;
  ac.persistent_hot = static_cast<size_t>(g.request_bytes()) * static_cast<size_t>(max_requests);
  arena_.init(ac);
  pool_.init(arena_, g, max_requests);
  log_memory_ledger("nemotron: state pool allocated");

  // Pre-load all layers eagerly on the main thread so weight loading
  // never happens on request-handling threads (which may have a
  // different CUDA context).
  for (int slot = 0; slot < lm.num_layers; ++slot)
    (void)loader_.load_layer(slot);
  log_memory_ledger("nemotron: all layers loaded");

  init_session(SessionParams{
      .max_tokens = max_tokens,
      .max_cache_tokens = max_cache_tokens,
      .rank = tp_rank,
      .world = tp_world,
      .boundary = boundary,
      .max_requests = max_requests,
      .decode_rows = decode_rows,
      .vocab_size = vocab_size_,
      .hidden = hidden_,
      .max_position_embeddings = cfg_.lm.max_position_embeddings,
      .block_tokens = 128,
      .snapshot_align = 1,
      .eos = static_cast<int>(cfg_.lm.eos_token_ids[0]),
  });
  log_memory_ledger("nemotron: session initialized");
}

NemotronModel::~NemotronModel() {
  free_activations();
  if (stream_) cudaStreamDestroy(stream_);
}

MemoryPlan NemotronModel::plan_memory(const NemotronTranscribeConfig& cfg, int max_tokens,
                                      int64_t max_cache_tokens, int /*tp_rank*/, int /*tp_world*/,
                                      NemotronResidency /*residency*/, int /*max_requests*/,
                                      bool /*mtp*/, int /*decode_rows*/) {
  MemoryPlan p;
  const auto& lm = cfg.lm;
  const int64_t H = lm.hidden_size;
  const int64_t V = lm.vocab_size;
  // Globals.
  p.add("embed", static_cast<size_t>(V) * static_cast<size_t>(H) * 2);
  p.add("final_norm", static_cast<size_t>(H) * 2);
  p.add("lm_head", static_cast<size_t>(V) * static_cast<size_t>(H) * 2);
  // Activations (per max_tokens rows).
  p.add("resid", static_cast<size_t>(max_tokens) * static_cast<size_t>(H) * 2);
  p.add("x", static_cast<size_t>(max_tokens) * static_cast<size_t>(H) * 2);
  p.add("y", static_cast<size_t>(max_tokens) * static_cast<size_t>(H) * 2);
  // Scratch for mamba in_proj (2*d_inner + 2*groups*state + heads).
  const auto mg = nemotron_mamba_geometry(lm);
  const int64_t aq_rows = lm.num_attention_heads * lm.head_dim;
  const int64_t akv_rows = lm.num_key_value_heads * lm.head_dim;
  p.add("xbc", static_cast<size_t>(max_tokens) * static_cast<size_t>(mg.in_proj_rows) * 2);
  p.add("mamba_conv_x", static_cast<size_t>(max_tokens) * static_cast<size_t>(mg.d_inner) * 2);
  p.add("mamba_conv_bc",
        2 * static_cast<size_t>(max_tokens) * static_cast<size_t>(mg.conv_channels - mg.d_inner) * 2);
  p.add("mamba_scan_out", static_cast<size_t>(max_tokens) * static_cast<size_t>(mg.d_inner) * 2);
  // Attention projections.
  p.add("attn_q", static_cast<size_t>(max_tokens) * static_cast<size_t>(aq_rows) * 2);
  p.add("attn_kv",
        2 * static_cast<size_t>(max_tokens) * static_cast<size_t>(akv_rows) * 2);
  p.add("attn_out", static_cast<size_t>(max_tokens) * static_cast<size_t>(aq_rows) * 2);
  // Logits.
  p.add("logits", static_cast<size_t>(max_tokens) * static_cast<size_t>(V) * 4);
  // Mamba state pool.
  int num_mamba = 0;
  for (auto s : lm.slots)
    if (s == NemotronSlot::Mamba) ++num_mamba;
  size_t slot_bytes = 0;
  if (num_mamba > 0) {
    slot_bytes = align256(static_cast<size_t>(lm.mamba_num_heads) * lm.mamba_head_dim *
                          lm.mamba_state_size * 4) +
                 align256(static_cast<size_t>(mg.conv_channels) * (lm.conv_kernel - 1) * 2);
    p.add("mamba_state_pool", slot_bytes * num_mamba * static_cast<size_t>(max_tokens));
  }
  return p;
}

void NemotronModel::init_activations(int max_tokens) {
  const size_t H = static_cast<size_t>(hidden_);
  const size_t V = static_cast<size_t>(vocab_size_);
  const size_t rows = static_cast<size_t>(max_tokens);
  const auto& lm = cfg_.lm;
  const auto mg = nemotron_mamba_geometry(lm);
  const size_t in_proj_rows = static_cast<size_t>(mg.in_proj_rows);
  const size_t d_inner = static_cast<size_t>(mg.d_inner);
  const size_t bc_channels = static_cast<size_t>(mg.conv_channels - mg.d_inner);
  const size_t q_rows = static_cast<size_t>(attn_q_rows_);
  const size_t kv_rows = static_cast<size_t>(attn_kv_rows_);

  DGPP_CUDA_OK(cudaMalloc(&resid_, rows * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&x_, rows * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&y_, rows * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&xbc_, rows * in_proj_rows * 2));
  DGPP_CUDA_OK(cudaMalloc(&x_conv_, rows * d_inner * 2));
  DGPP_CUDA_OK(cudaMalloc(&b_conv_, rows * bc_channels * 2));
  DGPP_CUDA_OK(cudaMalloc(&c_conv_, rows * bc_channels * 2));
  DGPP_CUDA_OK(cudaMalloc(&mamba_out_, rows * d_inner * 2));
  DGPP_CUDA_OK(cudaMalloc(&mamba_normed_, rows * d_inner * 2));
  DGPP_CUDA_OK(cudaMalloc(&q_, rows * q_rows * 2));
  DGPP_CUDA_OK(cudaMalloc(&k_, rows * kv_rows * 2));
  DGPP_CUDA_OK(cudaMalloc(&v_, rows * kv_rows * 2));
  DGPP_CUDA_OK(cudaMalloc(&attn_out_, rows * q_rows * 2));
  DGPP_CUDA_OK(cudaMalloc(&logits_, rows * V * 4));
  DGPP_CUDA_OK(cudaMalloc(&h_, rows * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_tokens_, rows * sizeof(int64_t)));
  DGPP_CUDA_OK(cudaMalloc(&d_positions_, rows * sizeof(int64_t)));
  DGPP_CUDA_OK(cudaMemset(resid_, 0, rows * H * 2));
  DGPP_CUDA_OK(cudaMemset(x_, 0, rows * H * 2));
  DGPP_CUDA_OK(cudaMemset(y_, 0, rows * H * 2));
  DGPP_CUDA_OK(cudaMemset(logits_, 0, rows * V * 4));
  DGPP_CUDA_OK(cudaMemset(h_, 0, rows * H * 2));
}

void NemotronModel::free_activations() {
  cudaFree(resid_); cudaFree(x_); cudaFree(y_);
  cudaFree(xbc_); cudaFree(x_conv_); cudaFree(b_conv_); cudaFree(c_conv_);
  cudaFree(mamba_out_); cudaFree(mamba_normed_);
  cudaFree(q_); cudaFree(k_); cudaFree(v_); cudaFree(attn_out_);
  cudaFree(logits_); cudaFree(h_);
  cudaFree(d_tokens_); cudaFree(d_positions_);
  resid_ = x_ = y_ = nullptr;
  xbc_ = x_conv_ = b_conv_ = c_conv_ = nullptr;
  mamba_out_ = mamba_normed_ = nullptr;
  q_ = k_ = v_ = attn_out_ = nullptr;
  logits_ = nullptr;
  h_ = nullptr;
  d_tokens_ = d_positions_ = nullptr;
}

// ---------------------------------------------------------------------------
// Diagnostic forward
// ---------------------------------------------------------------------------

SessionModel<NemotronModel>::Outputs NemotronModel::forward(const std::vector<int64_t>& token_ids, bool /*capture_layers*/) {
  RowRun run;
  run.ids = token_ids.data();
  run.T = static_cast<int>(token_ids.size());
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture = false;
  return run_rows(run);
}

// ---------------------------------------------------------------------------
// The layer walk
// ---------------------------------------------------------------------------

void NemotronModel::walk(const RowRun& run) {
  const int T = run.T;
  const auto& lm = cfg_.lm;
  const float eps = static_cast<float>(lm.norm_eps);

  // 1. Copy token ids to device.
  DGPP_CUDA_OK(cudaMemcpy(d_tokens_, run.ids, static_cast<size_t>(T) * sizeof(int64_t),
                          cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaGetLastError());

  // 2. Set up positions on device: pos0, pos0+1, ..., pos0+T-1.
  // (Positions are not used by position-free attention, but we stage them
  //  for consistency.)
  {
    std::vector<int64_t> positions(T);
    for (int i = 0; i < T; ++i) positions[i] = run.pos0 + i;
    DGPP_CUDA_OK(cudaMemcpy(d_positions_, positions.data(), static_cast<size_t>(T) * sizeof(int64_t),
                            cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaGetLastError());
  }

  // 3. Embed lookup: resid_[t] = embed[token[t]].
  nemotron_embed(globals_.embed, d_tokens_, resid_, T, hidden_, stream_);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  // 4. Walk the slots.
  for (int slot = 0; slot < lm.num_layers; ++slot) {
    walk_slot(slot, T, run.pos0);
    DGPP_CUDA_OK(cudaGetLastError());
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }

  // 5. norm_f + head.
  nemotron_head(resid_, globals_.final_norm, globals_.lm_head, logits_, T, hidden_, vocab_size_,
                eps, stream_);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

void NemotronModel::walk_slot(int slot, int T, int64_t /*pos0*/) {
  const auto& lm = cfg_.lm;
  const float eps = static_cast<float>(lm.norm_eps);
  const auto mg = nemotron_mamba_geometry(lm);
  const int d_inner = mg.d_inner;
  const int bc_channels = mg.conv_channels - mg.d_inner;
  const NemotronSlot kind = lm.slots[slot];
  const std::string p = std::format("llm.model.layers.{}.mixer.", slot);
  const NemotronLayerResident& r = loader_.load_layer(slot);

  // Count how many mamba slots precede this one, to index into the state pool.
  int mamba_slot_idx = 0;
  for (int s = 0; s < slot; ++s)
    if (lm.slots[s] == NemotronSlot::Mamba) ++mamba_slot_idx;

  // 4a. RMSNorm(resid_) → x_.
  nemotron_rmsnorm(r.input_norm, resid_, x_, T, hidden_, eps, stream_);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  // 4b. Mixer(x_) → y_.
  switch (kind) {
    case NemotronSlot::Mamba: {
      // in_proj: x_ [T,H] @ in_proj^T [in_proj_rows,H] → xbc_ [T,in_proj_rows].
      nemotron_gemm_bf16(x_, r.mamba_in_proj, xbc_, T, mg.in_proj_rows, hidden_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      // Split xbc_: x [0:d_inner], z [d_inner:2d_inner], B [2d_inner:2d_inner+bc/2],
      // C [2d_inner+bc/2:2d_inner+bc], dt [2d_inner+bc:...].
      const int64_t x_off = 0;
      const int64_t z_off = d_inner;
      const int64_t b_off = 2 * d_inner;
      const int64_t c_off = 2 * d_inner + bc_channels / 2;
      const int64_t dt_off = 2 * d_inner + bc_channels;

      // conv_silu over x, B, C.
      nemotron_mamba_conv_silu(xbc_ + x_off, mg.in_proj_rows, r.mamba_conv_weight,
                               r.mamba_conv_bias, x_conv_, d_inner, T, d_inner,
                               mamba_conv_kernel_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      nemotron_mamba_conv_silu(xbc_ + b_off, mg.in_proj_rows,
                               r.mamba_conv_weight + d_inner * mamba_conv_kernel_,
                               r.mamba_conv_bias + d_inner, b_conv_, bc_channels / 2, T,
                               bc_channels / 2, mamba_conv_kernel_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      nemotron_mamba_conv_silu(xbc_ + c_off, mg.in_proj_rows,
                               r.mamba_conv_weight + (d_inner + bc_channels / 2) * mamba_conv_kernel_,
                               r.mamba_conv_bias + d_inner + bc_channels / 2, c_conv_, bc_channels / 2,
                               T, bc_channels / 2, mamba_conv_kernel_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // Get the recurrent state for this mamba slot from the pool.
      float* state = pool_.recurrent(0, mamba_slot_idx);

      // Selective scan.
      nemotron_mamba_scan(x_conv_, d_inner, b_conv_, bc_channels / 2, c_conv_, bc_channels / 2,
                          xbc_ + dt_off, mg.in_proj_rows, r.mamba_A_log, r.mamba_D,
                          r.mamba_dt_bias, state, mamba_out_, d_inner, T, mamba_heads_,
                          mamba_head_dim_, mamba_groups_state_ / mamba_state_size_,
                          mamba_state_size_,
                          static_cast<float>(lm.time_step_min), stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // gate = silu(z).
      nemotron_silu(xbc_ + z_off, mamba_normed_, T * d_inner, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // gated norm: mamba_out_ [T,d_inner] gated by mamba_normed_ [T,d_inner] (silu(z)).
      nemotron_mamba_gated_norm(mamba_out_, d_inner, mamba_normed_, d_inner, r.mamba_norm,
                                mamba_normed_, d_inner, T, d_inner, mg.d_inner / (mamba_groups_state_ / mamba_state_size_),
                                eps, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // out_proj: mamba_normed_ [T,d_inner] @ out_proj^T [H,d_inner] → y_ [T,H].
      nemotron_gemm_bf16(mamba_normed_, r.mamba_out_proj, y_, T, hidden_, d_inner, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      break;
    }
    case NemotronSlot::Attention: {
      // q/k/v projections.
      nemotron_gemm_bf16(x_, r.attn_q, q_, T, attn_q_rows_, hidden_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      nemotron_gemm_bf16(x_, r.attn_k, k_, T, attn_kv_rows_, hidden_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      nemotron_gemm_bf16(x_, r.attn_v, v_, T, attn_kv_rows_, hidden_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // GQA attention (position-free, causal).
      float scale = 1.0f / sqrtf(static_cast<float>(attn_head_dim_));
      nemotron_gqa_attention(q_, k_, v_, attn_out_, T, attn_q_heads_, attn_kv_heads_,
                             attn_head_dim_, scale, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());

      // o_proj: attn_out_ [T,q_rows] @ o_proj^T [H,q_rows] → y_ [T,H].
      nemotron_gemm_bf16(attn_out_, r.attn_o, y_, T, hidden_, attn_q_rows_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      break;
    }
    case NemotronSlot::Moe: {
      nemotron_gemm_bf16(x_, r.moe_gate, xbc_, T, lm.num_routed_experts, hidden_, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      nemotron_moe(x_, r.moe_gate, r.moe_gate_bias, r.moe_experts_gate_and_up,
                   r.moe_experts_down, r.moe_shared_up, r.moe_shared_down, y_, T, hidden_,
                   lm.num_routed_experts, lm.moe_intermediate_size,
                   lm.moe_shared_expert_intermediate_size, lm.num_experts_per_tok,
                   static_cast<float>(lm.routed_scaling_factor), lm.norm_topk_prob, stream_);
      DGPP_CUDA_OK(cudaGetLastError());
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      break;
    }
  }

  // 4c. resid_ += y_.
  nemotron_residual_add(resid_, resid_, y_, T * hidden_, stream_);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// SessionModel hooks
// ---------------------------------------------------------------------------

SessionModel<NemotronModel>::Outputs NemotronModel::run_rows(const RowRun& run) {
  // All device memory was allocated under primary_ctx_ (cudaMalloc is
  // context-local). Switch to primary_ctx_ so worker-thread access is valid.
  if (primary_ctx_ != nullptr) {
    CUresult cu_res = cuCtxPushCurrent(primary_ctx_);
    if (cu_res != CUDA_SUCCESS)
      throw std::runtime_error("NemotronModel: cuCtxPushCurrent failed: " +
                               std::to_string(static_cast<int>(cu_res)));
  }
  const int T = run.T;
  if (T <= 0 || T > max_tokens_)
    throw std::invalid_argument("NemotronModel::run_rows: invalid row count");

  walk(run);

  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaGetLastError());

  Outputs out;
  out.lm_vocab_begin = 0;
  out.lm_vocab_count = vocab_size_;
  out.final_hidden_bits.resize(static_cast<size_t>(T) * static_cast<size_t>(hidden_));
  out.logits.resize(static_cast<size_t>(T) * static_cast<size_t>(vocab_size_));

  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(out.final_hidden_bits.data(), h_,
                          static_cast<size_t>(T) * static_cast<size_t>(hidden_) * 2,
                          cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(out.logits.data(), logits_,
                          static_cast<size_t>(T) * static_cast<size_t>(vocab_size_) * 4,
                          cudaMemcpyDeviceToHost));

  if (primary_ctx_ != nullptr) cuCtxPopCurrent(nullptr);
  return out;
}

void NemotronModel::reset_slot_state(int req) { /* state pool zeroing deferred */ }

GlmSpecSegments NemotronModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void NemotronModel::write_state_snapshot(int, uint8_t*, int) {}

void NemotronModel::graph_prepare() {}

}  // namespace dgpp