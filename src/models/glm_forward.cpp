#include "models/glm_forward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {

namespace {

constexpr size_t kGemmWsBase = 64ull << 20;

void* alloc_managed(size_t bytes) {
  void* p = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&p, bytes));
  return p;
}

// TP configs: the geometry validators run inside from_config, so the
// head-divisibility rejection happens at construction, before any load.
template <typename Cfg>
Cfg with_tp(Cfg c, int world) {
  c.tp_size = world;
  return c;
}

}  // namespace

GlmDiagnosticModel::GlmDiagnosticModel(const GlmTextConfig& cfg,
                                        const std::string& checkpoint_dir,
                                        int max_tokens,
                                        int64_t max_cache_tokens,
                                        GlmBoundaryReducer* boundary,
                                        int tp_rank, int tp_world,
                                        GlmResidency residency,
                                        GlmHeadSharding head,
                                        int max_requests)
    : cfg_(cfg),
      kda_cfg_(with_tp(cfg.kda_config(), tp_world)),
      dsa_cfg_(with_tp(cfg.dsa_config(), tp_world)),
      mhc_cfg_(cfg.mhc_config()),
      moe_cfg_(cfg.moe_config()),
      kda_geo_(KdaGeometry::from_config(kda_cfg_)),
      max_tokens_(max_tokens),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency, head) {
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmDiagnosticModel: max_tokens must be positive");
  if (max_cache_tokens < max_tokens_)
    max_cache_tokens = max_tokens_;
  if (max_requests <= 0)
    throw std::invalid_argument(
        "GlmDiagnosticModel: max_requests must be positive");
  max_requests_ = max_requests;
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "GlmDiagnosticModel: a boundary reducer is required exactly when "
        "tp_world > 1 — without one the block-boundary partials would be "
        "returned silently as results");
  boundary_ = boundary;

  // Boot check (§5.2): hash every replicated tensor BEFORE anything else
  // runs — runners exchange this across ranks at startup, and a mismatch
  // pinpoints the diverging layer. Pure mmap reads; no residency needed.
  if (tp_world > 1) boot_digest_ = loader_.hash_replicated();

  globals_ = loader_.load_globals();
  lm_vocab_begin_ = globals_.lm_vocab_begin;
  lm_vocab_count_ =
      globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;

  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
  // The model's kernels are the resident layers' readers: load boundaries
  // synchronize exactly this stream (+ the loader's dequant stream), not
  // the whole device — a device-wide wait in a one-process multi-rank
  // world deadlocks against a peer's spinning collective kernel (the
  // loopback first-collective stall; see set_reader_stream).
  loader_.set_reader_stream(stream_);

  if (tp_world > 1)
    tp_ = std::make_unique<GlmTpViews>(cfg, tp_rank, tp_world, stream_);

  // GEMM workspace: 64 MB covers every M2/M3 shape; the lm head's N
  // (this rank's vocab slice in sharded mode) is the only dimension
  // larger than anything tested there.
  gemm_ws_bytes_ = kGemmWsBase;
  {
    const size_t head_ws = gemm_.query_workspace_bytes(
        max_tokens_, lm_vocab_count_, cfg_.hidden_size, DType::BF16);
    if (head_ws > gemm_ws_bytes_) gemm_ws_bytes_ = head_ws;
  }
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));

  // The DSA pool shares the arena with the KDA layer scratch (both take
  // persistent-hot grants at construction/init).
  const int64_t dsa_slots =
      ((max_cache_tokens + dsa_cfg_.block_tokens - 1) /
       dsa_cfg_.block_tokens) *
      dsa_cfg_.block_tokens;
  Arena::Config ac;
  ac.persistent_hot = KdaLayer::persistent_hot_bytes(kda_cfg_, max_tokens_);
  if (dsa_cfg_.num_dsa_layers > 0) {
    ac.persistent_hot += DsaStatePool::cache_bytes(dsa_cfg_, max_requests_,
                                                   dsa_slots);
    arena_.init(ac);
    pool_.init(arena_, dsa_cfg_, max_requests_, dsa_slots);
    dsa_scratch_ = static_cast<uint8_t*>(
        alloc_managed(DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                              dsa_slots)));
    // Decode-session metadata (enqueue_decode's caller-owned device
    // buffers): allocated HERE, at construction — never mid-session (the
    // synchronizing-call discipline applies between collectives).
    d_req_ids_ = static_cast<int32_t*>(alloc_managed(sizeof(int32_t) *
                                                     kDecodeRows));
    d_step_pos_ = static_cast<int64_t*>(alloc_managed(sizeof(int64_t) *
                                                      kDecodeRows));
    d_req_spans_ = static_cast<int32_t*>(alloc_managed(sizeof(int32_t) * 2 *
                                                       kDecodeRows));
  } else {
    arena_.init(ac);
  }

  // The decode path's H2D upload sources, PINNED at construction
  // (pageable async copies stream-sync before initiating — a per-step
  // pipeline drain the decode path refuses; and the graph era's memcpy
  // nodes require page-locked sources). Addresses are stable for the
  // model's lifetime — the graph bakes them.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_req_ids_),
                             sizeof(int32_t) * kDecodeRows,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_step_pos_),
                             sizeof(int64_t) * kDecodeRows,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_req_spans_),
                             sizeof(int32_t) * 2 * kDecodeRows,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_token_),
                             sizeof(int64_t) * kDecodeRows,
                             cudaHostAllocDefault));

  // Per-request, per-layer KDA state (slot-major: one memset pair per
  // request open — see the header's layout note).
  if (kda_cfg_.num_kda_layers > 0) {
    kda_rec_ = static_cast<float*>(alloc_managed(
        static_cast<size_t>(kda_cfg_.num_kda_layers) * max_requests_ *
        kda_geo_.recurrent_bytes));
    kda_conv_ = static_cast<uint16_t*>(alloc_managed(
        static_cast<size_t>(kda_cfg_.num_kda_layers) * max_requests_ *
        kda_geo_.conv_committed_bytes));
  }
  session_pos_.assign(static_cast<size_t>(max_requests_), 0);

  // Decode-path route traces: pinned staging the decode MoE's async
  // copies land in (see glm_moe_layer.hpp's MoeTraceStaging). Pinned —
  // the copies are issued mid-step with no sync, and pinned destinations
  // keep them true async D2H.
  n_moe_layers_ = static_cast<int>(std::count_if(
      cfg_.mlps.begin(), cfg_.mlps.end(),
      [](GlmMlpKind k) { return k == GlmMlpKind::Moe; }));
  if (n_moe_layers_ > 0) {
    const size_t rows = static_cast<size_t>(n_moe_layers_) * kDecodeRows;
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_ids_),
                               rows * moe_cfg_.top_k * sizeof(int32_t),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_weights_),
                               rows * moe_cfg_.top_k * sizeof(float),
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&moe_trace_biased_),
                               rows * moe_cfg_.n_experts * sizeof(float),
                               cudaHostAllocDefault));
  }

  // Activations.
  const size_t T = static_cast<size_t>(max_tokens_);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  d_tokens_ = static_cast<int64_t*>(alloc_managed(T * 8));
  streams_[0] = static_cast<uint16_t*>(alloc_managed(T * 4 * H * 2));
  streams_[1] = static_cast<uint16_t*>(alloc_managed(T * 4 * H * 2));
  post_ = static_cast<uint16_t*>(alloc_managed(T * 4 * 2));
  comb_ = static_cast<uint16_t*>(alloc_managed(T * 16 * 2));
  collapsed_ = static_cast<uint16_t*>(alloc_managed(T * H * 2));
  normed_ = static_cast<uint16_t*>(alloc_managed(T * H * 2));
  sub_out_ = static_cast<uint16_t*>(alloc_managed(T * H * 2));
  if (cfg_.first_k_dense_replace > 0) {
    const size_t I = static_cast<size_t>(cfg_.intermediate_size);
    dense_g_ = static_cast<uint16_t*>(alloc_managed(T * I * 2));
    dense_u_ = static_cast<uint16_t*>(alloc_managed(T * I * 2));
    dense_act_ = static_cast<uint16_t*>(alloc_managed(T * I * 2));
  }
  logits_ =
      static_cast<uint16_t*>(alloc_managed(T * lm_vocab_count_ * 2));

  // Every device allocation happens above (see preconstruct_layers): the
  // TP runners barrier after construction so no rank's first collective
  // can spin while a peer is still inside allocation-phase device syncs.
  preconstruct_layers();
}

GlmDiagnosticModel::~GlmDiagnosticModel() {
  cudaFree(gemm_ws_);
  cudaFree(dsa_scratch_);
  cudaFree(d_req_ids_);
  cudaFree(d_step_pos_);
  cudaFree(d_req_spans_);
  cudaFreeHost(h_req_ids_);
  cudaFreeHost(h_step_pos_);
  cudaFreeHost(h_req_spans_);
  cudaFreeHost(h_token_);
  cudaFree(kda_rec_);
  cudaFree(kda_conv_);
  cudaFree(d_tokens_);
  cudaFree(streams_[0]);
  cudaFree(streams_[1]);
  cudaFree(post_);
  cudaFree(comb_);
  cudaFree(collapsed_);
  cudaFree(normed_);
  cudaFree(sub_out_);
  cudaFree(dense_g_);
  cudaFree(dense_u_);
  cudaFree(dense_act_);
  cudaFree(logits_);
  cudaFreeHost(moe_trace_ids_);
  cudaFreeHost(moe_trace_weights_);
  cudaFreeHost(moe_trace_biased_);
  if (stream_) cudaStreamDestroy(stream_);
}

// ---- DSA admission meters (the scheduler's budget seam, Stage 2b) ------
// A no-DSA model reports an unbounded pool: admission then keys on the
// engine slot count alone. INT64_MAX (not "huge") so the scheduler's
// subtraction arithmetic cannot overflow a real capacity.
int64_t GlmDiagnosticModel::dsa_blocks_total() const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.total_blocks() : INT64_MAX;
}
int64_t GlmDiagnosticModel::dsa_blocks_in_use() const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.blocks_in_use() : int64_t(0);
}
int64_t GlmDiagnosticModel::dsa_blocks_for_tokens(int64_t tokens) const {
  return dsa_cfg_.num_dsa_layers > 0 ? pool_.block_count_for_tokens(tokens)
                                    : int64_t(0);
}

GlmMoeWeights GlmDiagnosticModel::moe_weights(const GlmMoeResident& r) {
  GlmMoeWeights w;
  w.router_gate = r.router_gate;
  w.router_bias = r.router_bias;
  w.experts = r.experts.data();
  for (int i = 0; i < 3; ++i) w.shared[i] = r.shared[i];
  return w;
}

GlmLayerBound GlmDiagnosticModel::bind_layer(const GlmLayerResident& r,
                                              bool dense_mlp) {
  // The sharded path (d4): the resident layer is already this rank's
  // geometry, so binding is identity wiring. The views' full-load bind()
  // remains the parity reference the shard test pins bitwise.
  if (tp_) return tp_->bind_sharded(r, dense_mlp);
  // World=1: direct resident views, byte-identical to the M4 path.
  full_ = GlmLayerBound{};
  full_.mhc = &r.mhc;
  full_.ln1 = r.ln1;
  full_.ln2 = r.ln2;
  if (r.kind == GlmLayerKind::Kda) full_.kda = &r.kda;
  else full_.dsa = &r.dsa;
  if (dense_mlp) {
    full_.dense = r.dense;
  } else {
    moe_full_ = moe_weights(r.moe);  // partition defaults: every expert
    full_.moe = &moe_full_;
  }
  return full_;
}

// Constructs every layer object (KDA/DSA/MoE) with throwaway layer-0..k
// loads so NO allocation ever happens inside run_stack. Allocations are
// implicit device syncs, and a rank constructing lazily while a peer's
// first collective kernel spins on doorbells is the process-wide
// deadlock the M5 loopback bring-up measured: the lagging rank's
// cudaMallocManaged waits for the spinning kernel, which waits for the
// lagging rank's post, which requires the lagging rank to finish
// constructing. Post-startup (all objects resident, engines posting from
// free threads) the shape is safe — staggered per-layer loads and
// collectives coexist by construction.
void GlmDiagnosticModel::preconstruct_layers() {
  bool need_kda = kda_cfg_.num_kda_layers > 0;
  bool need_dsa = dsa_cfg_.num_dsa_layers > 0;
  bool need_moe =
      std::any_of(cfg_.mlps.begin(), cfg_.mlps.end(),
                  [](GlmMlpKind k) { return k == GlmMlpKind::Moe; });
  for (int layer = 0; layer < cfg_.num_hidden_layers &&
                       (need_kda || need_dsa || need_moe);
       ++layer) {
    const GlmLayerResident& r = loader_.load_layer(layer);
    const GlmLayerBound b =
        bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);
    if (r.kind == GlmLayerKind::Kda && need_kda && !kda_) {
      kda_ = std::make_unique<KdaLayer>(arena_, gemm_, *b.kda, kda_cfg_,
                                        max_tokens_, gemm_ws_,
                                        gemm_ws_bytes_, cfg_.rms_norm_eps);
      need_kda = false;
    }
    if (r.kind == GlmLayerKind::Dsa && need_dsa && !dsa_) {
      dsa_ = std::make_unique<DsaLayer>(
          gemm_, *b.dsa, dsa_cfg_, max_tokens_, pool_.max_token_slots(),
          dsa_scratch_,
          DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                  pool_.max_token_slots()),
          gemm_ws_, gemm_ws_bytes_);
      need_dsa = false;
    }
    if (need_moe && !moe_ && cfg_.mlps[layer] == GlmMlpKind::Moe) {
      // kDecodeRows: the decode fast path's slot bound (session_step's
      // time-multiplexed rows; the batched-decode ceiling). The graph
      // table slots (one per MoE layer) provision the capture path's
      // per-layer pinned expert-table sources unconditionally — a few
      // hundred KB of pinned memory; the eager path never touches them.
      moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                           kDecodeRows, n_moe_layers_);
      need_moe = false;
    }
  }
  loader_.release_layer();
}

void GlmDiagnosticModel::enqueue_dense_mlp(
    const uint16_t* x, uint16_t* out, const GlmQuantMatrix* dense,
    int tokens, cudaStream_t stream) {
  // The inter dim comes from the views: the full matrix at world=1, this
  // rank's column/row shard under TP (scratch is sized for the full I).
  const int H = cfg_.hidden_size;
  const int I = static_cast<int>(dense[0].rows);
  if (dense[1].rows != dense[0].rows || dense[2].cols != dense[0].rows)
    throw std::runtime_error("forward: inconsistent dense matrices");
  launch_scale_gemm_bf16(x, H, dense[0].payload, dense[0].scales, dense_g_,
                          tokens, I, H, stream);
  launch_scale_gemm_bf16(x, H, dense[1].payload, dense[1].scales, dense_u_,
                          tokens, I, H, stream);
  // Same asymmetric swiglu clamps as the experts (the reference MLP and
  // experts share the clamp choreography; DESIGN §7.4).
  launch_moe_swiglu_clamp(dense_g_, dense_u_, dense_act_,
                          static_cast<int64_t>(tokens) * I,
                          cfg_.swiglu_limit, stream);
  launch_scale_gemm_bf16(dense_act_, I, dense[2].payload, dense[2].scales,
                          out, tokens, H, I, stream);
}

// Shared stack runner. `layer_inputs` (isolated mode) overrides the stream
// state entering EVERY layer (index L feeds layer L); `capture` (isolated
// mode) receives each layer's output streams plus the initial state at
// index 0; `boundary_capture` (isolated mode) additionally receives the two
// post-fold block-boundary outputs per layer (attn, FFN) — the raw surface
// where slicing errors surface unattenuated (the mHC stream update in
// `capture`'s snapshots compresses boundary errors below assertion budgets;
// the dense scale-grid slice bug hid exactly there). Free-run forward
// passes null for all three.
GlmDiagnosticModel::Outputs GlmDiagnosticModel::run_stack(
    const std::vector<int64_t>& token_ids, const uint16_t* const* layer_inputs,
    std::vector<std::vector<uint16_t>>* capture,
    std::vector<std::vector<uint16_t>>* boundary_capture) {
  // The re-forward and an open decode session SHARE the KDA/DSA state
  // pools — running one mid-session silently clobbers the other's state.
  // That failure mode is exactly the kind a gate would absorb as noise;
  // it throws loudly instead (a parity harness uses separate model
  // instances for engine and reference).
  for (int64_t pos : session_pos_) {
    if (pos > 0)
      throw std::runtime_error(
          "forward: a decode session is open on this model instance — a "
          "re-forward would clobber its state (use a second model for the "
          "reference)");
  }
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_)
    throw std::invalid_argument("forward: tokens exceed max_tokens");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("forward: token id out of range");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;

  // Shape-keyed plans (cache hits after the first forward of this size).
  if (!gemm_.ensure_plan(T, lm_vocab_count_, H, DType::BF16, GemmOut::BF16,
                         H))
    throw std::runtime_error("forward: lm head GEMM plan unavailable");

  // Fresh per-request state.
  if (kda_rec_) {
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_rec_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.recurrent_bytes,
        stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_conv_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.conv_committed_bytes,
        stream_));
  }
  if (dsa_cfg_.num_dsa_layers > 0) {
    // Cold start every forward: fresh block tables and zeroed caches (the
    // diagnostic model owns no cross-call state).
    pool_.reset_all(stream_);
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, token_ids.data(),
                               static_cast<size_t>(T) * 8,
                               cudaMemcpyHostToDevice, stream_));
  glm_embed_bcast_streams(globals_.embed, d_tokens_, streams_[0], T, H,
                          stream_);
  if (layer_inputs) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    std::memcpy(streams_[0], layer_inputs[0],
                static_cast<size_t>(T) * 4 * H * 2);
    if (capture)
      capture->push_back(std::vector<uint16_t>(
          streams_[0], streams_[0] + static_cast<size_t>(T) * 4 * H));
  }

  Outputs out;
  out.routes.reserve(static_cast<size_t>(cfg_.num_hidden_layers));
  uint16_t* cur = streams_[0];
  uint16_t* nxt = streams_[1];
  int dsa_ordinal = 0;
  int kda_ordinal = 0;

  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    if (layer_inputs && layer > 0) {
      // Isolated mode: every layer starts from the reference trajectory.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::memcpy(cur, layer_inputs[layer],
                  static_cast<size_t>(T) * 4 * H * 2);
    }
    const GlmLayerResident& r = loader_.load_layer(layer);
    const GlmLayerBound b = bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = b.mhc->attn_fn;
    hw.base = b.mhc->attn_base;
    hw.scale = b.mhc->attn_scale;
    launch_mhc_compute(cur, hw, mhc_cfg_, collapsed_, post_, comb_, T,
                        stream_);    glm_rmsnorm_bf16(collapsed_, b.ln1, normed_, T, H, eps, stream_);
    // Block boundary 1 (DESIGN §5.1): the attention output projection is
    // row-parallel over this rank's heads, so the block output is a partial
    // sum until folded. With pre-stage support the attention writes the
    // pinned staging buffer directly (the §6.3 seam — no staging copy;
    // decode T qualifies, prefill-sized T falls back to the device
    // buffer). The decision precedes the enqueue so the GEMM's destination
    // is the transport's send source.
    uint16_t* attn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) attn_out = staged;
    }
    if (r.kind == GlmLayerKind::Kda) {
      if (!kda_) {
        kda_ = std::make_unique<KdaLayer>(arena_, gemm_, *b.kda, kda_cfg_,
                                          max_tokens_, gemm_ws_,
                                          gemm_ws_bytes_, eps);
      } else {
        kda_->rebind(*b.kda);
      }
      if (!kda_->prepare(T))
        throw std::runtime_error("forward: KDA GEMM plans unavailable");
      float* rec = kda_rec_ +
                   static_cast<size_t>(kda_ordinal) *
                       kda_geo_.recurrent_elems;
      // Conv state stride is the COMMITTED width (conv_hist): the
      // diagnostic forward reserves no MTP spec region, so the slot width
      // and the committed width coincide.
      uint16_t* conv =
          kda_conv_ + static_cast<size_t>(kda_ordinal) *
                          (kda_geo_.conv_committed_bytes / 2);
      kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, attn_out, T,
                    stream_);
      ++kda_ordinal;
    } else {
      if (!dsa_) {
        dsa_ = std::make_unique<DsaLayer>(
            gemm_, *b.dsa, dsa_cfg_, max_tokens_,
            pool_.max_token_slots(), dsa_scratch_,
            DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                    pool_.max_token_slots()),
            gemm_ws_, gemm_ws_bytes_);
      } else {
        dsa_->rebind(*b.dsa);
      }
      if (!dsa_->prepare(T))
        throw std::runtime_error("forward: DSA GEMM plans unavailable");
      dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, 0, 0, T, attn_out,
                             stream_);
      ++dsa_ordinal;
    }
    // The collective kernel runs on the bus's stream, so the producer
    // quiesces first (stream order cannot cover a cross-stream consumer;
    // the graph-mode decode path restores this as a stream-ordered node).
    if (boundary_) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      DGPP_LOG_DEBUG("TP boundary (attn) layer={} rows={} — reducing", layer,
                     T);
      boundary_->reduce(attn_out, T, H);
    }
    if (boundary_capture) {
      // Post-fold attention boundary (world=1: the unfolded full output —
      // the comparison target). Captured before the FFN site reuses the
      // device buffer.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_capture->push_back(std::vector<uint16_t>(
          attn_out, attn_out + static_cast<size_t>(T) * H));
    }
    launch_mhc_stream_update(post_, comb_, attn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);

    // ---- feed-forward site -----------------------------------------
    GlmMhcWeights fw;
    fw.fn = b.mhc->ffn_fn;
    fw.base = b.mhc->ffn_base;
    fw.scale = b.mhc->ffn_scale;
    launch_mhc_compute(cur, fw, mhc_cfg_, collapsed_, post_, comb_, T,
                        stream_);
    glm_rmsnorm_bf16(collapsed_, b.ln2, normed_, T, H, eps, stream_);
    // Block boundary 2 destination: same staging-seam decision before the
    // FFN enqueue (dense down-proj, shared expert, and this rank's routed
    // experts are all partial until the fold).
    uint16_t* ffn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) ffn_out = staged;
    }
    if (cfg_.mlps[layer] == GlmMlpKind::Dense) {
      enqueue_dense_mlp(normed_, ffn_out, b.dense, T, stream_);    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_,
                                             max_tokens_);
      } else {
        moe_->rebind(*b.moe);
      }
      moe_->enqueue(normed_, ffn_out, T, stream_);
      GlmRouteTraceLayer route;
      route.layer_idx = static_cast<uint32_t>(layer);
      route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
      route.tokens = static_cast<uint64_t>(T);
      route.ids = moe_->last_ids();
      route.weights = moe_->last_weights();
      out.routes.push_back(std::move(route));
      out.route_biased.push_back(moe_->last_biased());
    }
    if (boundary_) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      DGPP_LOG_DEBUG("TP boundary (ffn)  layer={} rows={} — reducing", layer,
                     T);
      boundary_->reduce(ffn_out, T, H);
    }
    if (boundary_capture) {
      // Post-fold FFN boundary (world=1: the unfolded full output). The
      // raw fold surface — dense/shared-expert slicing errors appear here
      // at full magnitude.
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_capture->push_back(std::vector<uint16_t>(
          ffn_out, ffn_out + static_cast<size_t>(T) * H));
    }
    launch_mhc_stream_update(post_, comb_, ffn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);
    if (capture) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      capture->push_back(std::vector<uint16_t>(
          cur, cur + static_cast<size_t>(T) * 4 * H));
    }
  }

  // ---- head: mean over streams, final norm, lm head -----------------
  launch_mhc_final_mean(cur, collapsed_, mhc_cfg_, T, stream_);
  glm_rmsnorm_bf16(collapsed_, globals_.final_norm, normed_, T, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, lm_vocab_count_, H,
               DType::BF16, GemmOut::BF16, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

    const size_t TH = static_cast<size_t>(T) * H;
  const size_t TV = static_cast<size_t>(T) * lm_vocab_count_;
  out.final_hidden_bits.assign(normed_, normed_ + TH);
  out.logits_bits.assign(logits_, logits_ + TV);
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward(
    const std::vector<int64_t>& token_ids) {
  return run_stack(token_ids, nullptr, nullptr, nullptr);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward_isolated(
    const std::vector<int64_t>& token_ids,
    const std::vector<const uint16_t*>& layer_inputs,
    std::vector<std::vector<uint16_t>>& capture,
    std::vector<std::vector<uint16_t>>* boundary_capture) {
  if (layer_inputs.size() != static_cast<size_t>(cfg_.num_hidden_layers) + 1)
    throw std::invalid_argument(
        "forward_isolated: needs num_layers+1 input snapshots");
  return run_stack(token_ids, layer_inputs.data(), &capture,
                   boundary_capture);
}

std::vector<std::vector<std::pair<int32_t, float>>> GlmDiagnosticModel::topk(
    const std::vector<uint16_t>& logits_bits, int64_t rows, int vocab,
    int k) {
  if (k <= 0 || k > 64 || k > vocab)
    throw std::invalid_argument("topk: k out of range");
  std::vector<std::vector<std::pair<int32_t, float>>> out(
      static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* row = logits_bits.data() + static_cast<size_t>(r) * vocab;
    std::vector<std::pair<float, int32_t>> best;
    best.reserve(static_cast<size_t>(k));
    for (int c = 0; c < vocab; ++c) {
      const float v = bf16_bits_to_float(row[c]);
      // Lowest-id tie-break: a later column only displaces on a strictly
      // greater value (insertion into the sorted-descending prefix).
      bool placed = false;
      for (size_t i = 0; i < best.size(); ++i) {
        if (v > best[i].first) {
          if (best.size() < static_cast<size_t>(k))
            best.emplace_back(0.f, 0);
          for (size_t j = best.size() - 1; j > i; --j) best[j] = best[j - 1];
          best[i] = {v, c};
          placed = true;
          break;
        }
      }
      if (!placed && best.size() < static_cast<size_t>(k))
        best.emplace_back(v, c);
    }
    auto& row_out = out[static_cast<size_t>(r)];
    row_out.reserve(best.size());
    for (const auto& [v, c] : best) row_out.emplace_back(c, v);
  }
  return out;
}

}  // namespace dgpp
