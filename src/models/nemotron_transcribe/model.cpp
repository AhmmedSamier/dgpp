// Nemotron-3.5-Transcribe model (milestone 4): the SessionModel that runs
// the 52-slot Nemotron-H hybrid walk on the GB10. MTP is not yet wired.

#include "models/nemotron_transcribe/model.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "engine/boundary_reducer.hpp"
#include "kernels/kernels.hpp"
#include "loaders/safetensors.hpp"
#include "models/nemotron_transcribe/forward.hpp"
#include "models/nemotron_transcribe/perception.hpp"

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
      vocab_size_(cfg.lm.vocab_size),
      checkpoint_dir_(checkpoint_dir) {
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
  if (d_audio_embeds_) cudaFree(d_audio_embeds_);
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

  // 3b. Splice audio embeddings at <|audio|> locator positions (milestone 6).
  // Each locator was expanded to T_tok placeholder tokens (id 0) in the
  // prompt; the recorded span tells us where to overwrite resid_ with the
  // clip's perception rows, one row per placeholder position.
  if (run.req >= 0 && run.req < static_cast<int>(audio_spans_.size()) &&
      audio_spans_[static_cast<size_t>(run.req)].count > 0) {
    const auto& span = audio_spans_[static_cast<size_t>(run.req)];
    for (int64_t i = 0; i < span.count; ++i) {
      DGPP_CUDA_OK(cudaMemcpy(resid_ + (static_cast<size_t>(span.prompt_offset) + i) *
                                      static_cast<size_t>(hidden_),
                              d_audio_embeds_ +
                                  (static_cast<size_t>(span.embed_offset) + i) *
                                      static_cast<size_t>(hidden_),
                              static_cast<size_t>(hidden_) * 2,
                              cudaMemcpyDeviceToDevice));
    }
    DGPP_CUDA_OK(cudaGetLastError());
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }

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
      // Split xbc_: z [0:d_inner] (gate), x_B_C [d_inner : d_inner+conv_channels]
      // with x [d_inner : 2*d_inner), B [2*d_inner : 2*d_inner+bc/2),
      // C [2*d_inner+bc/2 : 2*d_inner+bc), dt [2*d_inner+bc : ...].
      // (HF: gate, hidden_states_B_C, dt = in_proj(h).split([d_inner, conv_dim, nheads]))
      const int64_t z_off = 0;
      const int64_t x_off = d_inner;
      const int64_t b_off = 2 * d_inner;
      const int64_t c_off = 2 * d_inner + bc_channels / 2;
      const int64_t dt_off = 2 * d_inner + bc_channels;

      // conv state for this request/slot: [conv_dim][hist] uint16, hist = kernel-1,
      // laid out as [x region][B region][C region] within each time step.
      const int hist = mamba_conv_kernel_ - 1;
      uint16_t* conv_state = pool_.conv(0, mamba_slot_idx);
      for (int t = 0; t < T; ++t) {
        const uint16_t* xrow = xbc_ + static_cast<size_t>(t) * static_cast<size_t>(mg.in_proj_rows);
        uint16_t* xout = x_conv_ + static_cast<size_t>(t) * static_cast<size_t>(d_inner);
        uint16_t* bout = b_conv_ + static_cast<size_t>(t) * static_cast<size_t>(bc_channels / 2);
        uint16_t* cout = c_conv_ + static_cast<size_t>(t) * static_cast<size_t>(bc_channels / 2);
        // x region conv step.
        mamba2_conv_silu_step(xrow + x_off, conv_state,
                              r.mamba_conv_weight, r.mamba_conv_bias,
                              xout, d_inner, mamba_conv_kernel_, stream_);
        DGPP_CUDA_OK(cudaGetLastError());
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        // B region conv step.
        mamba2_conv_silu_step(xrow + b_off, conv_state + static_cast<size_t>(d_inner) * static_cast<size_t>(hist),
                              r.mamba_conv_weight + d_inner * static_cast<size_t>(mamba_conv_kernel_),
                              r.mamba_conv_bias + d_inner,
                              bout, bc_channels / 2, mamba_conv_kernel_, stream_);
        DGPP_CUDA_OK(cudaGetLastError());
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        // C region conv step.
        mamba2_conv_silu_step(xrow + c_off,
                              conv_state + (static_cast<size_t>(d_inner) + bc_channels / 2) * static_cast<size_t>(hist),
                              r.mamba_conv_weight + (d_inner + bc_channels / 2) * static_cast<size_t>(mamba_conv_kernel_),
                              r.mamba_conv_bias + d_inner + bc_channels / 2,
                              cout, bc_channels / 2, mamba_conv_kernel_, stream_);
        DGPP_CUDA_OK(cudaGetLastError());
        DGPP_CUDA_OK(cudaDeviceSynchronize());
      }

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
      // gate = z (raw); the gated-norm kernel applies silu itself (HF
      // RMSNormGated: h = silu(z) * x, then group RMS).
      nemotron_mamba_gated_norm(mamba_out_, d_inner, xbc_ + z_off, mg.in_proj_rows, r.mamba_norm,
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

SessionModel<NemotronModel>::Outputs NemotronModel::session_prefill_audios(
    int req, const std::vector<int64_t>& prompt,
    const std::vector<dgpp::AudioInput>& audios) {
  if (req < 0 || req >= max_requests_)
    throw std::invalid_argument("NemotronModel::session_prefill_audios: bad req");
  if (audios.empty())
    throw std::invalid_argument("NemotronModel::session_prefill_audios: no audios");

  // --- 1. Build perception weights from the checkpoint (CPU side) ---
  nemotron_ref::Weights w;
  {
    const std::string safetensors_path = checkpoint_dir_ + "/model.safetensors";
    auto sf = SafetensorsFile::open(safetensors_path);
    auto perc_keys = sf->grep("perception.");
    for (const auto* ti : perc_keys) {
      nemotron_ref::Tensor t;
      if (ti->dtype == DType::BF16) {
        t.bf16.resize(ti->numel());
        std::memcpy(t.bf16.data(), ti->data, ti->nbytes());
      } else if (ti->dtype == DType::F32) {
        t.f32.resize(ti->numel());
        std::memcpy(t.f32.data(), ti->data, ti->nbytes());
      } else {
        throw std::runtime_error(
            "NemotronModel: unsupported perception tensor dtype " + ti->name);
      }
      w[ti->name] = std::move(t);
    }
  }

  // --- 2. Run perception on each audio clip ---
  const auto& pc = cfg_.perception;
  const int H = hidden_;
  if (pc.output_dim != H)
    throw std::runtime_error(
        "NemotronModel: perception output_dim != LM hidden");

  std::vector<uint16_t> all_audio_embeds;  // concatenated [sum T_tok, H] bf16
  int64_t embed_row = 0;

  // Resize per-request audio span storage.
  if (static_cast<size_t>(req) >= audio_spans_.size())
    audio_spans_.resize(req + 1);

  size_t total_rows = 0;
  for (const auto& au : audios) {
    auto out = nemotron_perc::perceive(pc, w, au.pcm.data(), au.pcm.size(), nullptr);
    const size_t T_tok = out.tokens;
    if (T_tok == 0)
      throw std::runtime_error("NemotronModel: perception produced no tokens");
    for (size_t r = 0; r < T_tok; ++r) {
      for (int c = 0; c < H; ++c)
        all_audio_embeds.push_back(
            float_to_bf16_bits(out.rows[r * pc.output_dim + c]));
    }
    audio_spans_[static_cast<size_t>(req)] = AudioSpan{
        .prompt_offset = au.offset,
        .embed_offset = embed_row,
        .count = static_cast<int>(T_tok),
    };
    embed_row += static_cast<int64_t>(T_tok);
    total_rows += T_tok;
  }

  // --- 3. Upload audio embeddings to device ---
  const size_t needed_bytes = total_rows * static_cast<size_t>(H) * 2;
  if (needed_bytes > audio_capacity_ * static_cast<size_t>(H) * 2) {
    if (d_audio_embeds_) cudaFree(d_audio_embeds_);
    DGPP_CUDA_OK(cudaMalloc(&d_audio_embeds_, needed_bytes));
    audio_capacity_ = total_rows;
  }
  if (!all_audio_embeds.empty()) {
    DGPP_CUDA_OK(cudaMemcpy(d_audio_embeds_, all_audio_embeds.data(),
                            all_audio_embeds.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaGetLastError());
  }

  // --- 4. Run the model forward pass (prompt already has placeholder ids) ---
  // Open the session so the scheduler's decode path finds an active slot.
  open_slot(req);
  RowRun run;
  run.req = req;
  run.ids = prompt.data();
  run.T = static_cast<int>(prompt.size());
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = false;
  run.capture_layers = false;
  run.capture = false;
  run.first_chunk = true;
  run.last_chunk = true;
  Outputs out = run_rows(run);
  session_pos_[static_cast<size_t>(req)] = static_cast<int64_t>(prompt.size());
  push_position(req);

  // --- 5. Clear audio span for this request (one-shot) ---
  audio_spans_[static_cast<size_t>(req)] = AudioSpan{-1, 0, 0};

  return out;
}

void NemotronModel::reset_slot_state(int req) { /* state pool zeroing deferred */ }

GlmSpecSegments NemotronModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void NemotronModel::write_state_snapshot(int, uint8_t*, int) {}

void NemotronModel::graph_prepare() {}

}  // namespace dgpp