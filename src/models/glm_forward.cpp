#include "models/glm_forward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
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

}  // namespace

GlmDiagnosticModel::GlmDiagnosticModel(const GlmTextConfig& cfg,
                                       const std::string& checkpoint_dir,
                                       int max_tokens,
                                       int64_t max_cache_tokens)
    : cfg_(cfg),
      kda_cfg_(cfg.kda_config()),
      dsa_cfg_(cfg.dsa_config()),
      mhc_cfg_(cfg.mhc_config()),
      moe_cfg_(cfg.moe_config()),
      kda_geo_(KdaGeometry::from_config(kda_cfg_)),
      max_tokens_(max_tokens),
      loader_(cfg, checkpoint_dir) {
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmDiagnosticModel: max_tokens must be positive");
  if (max_cache_tokens < max_tokens_)
    max_cache_tokens = max_tokens_;

  globals_ = loader_.load_globals();

  DGPP_CUDA_OK(cudaStreamCreate(&stream_));

  // GEMM workspace: 64 MB covers every M2/M3 shape; the lm head's N
  // (vocab) is the only dimension larger than anything tested there.
  gemm_ws_bytes_ = kGemmWsBase;
  {
    const size_t head_ws = gemm_.query_workspace_bytes(
        max_tokens_, cfg_.vocab_size, cfg_.hidden_size, DType::BF16);
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
    ac.persistent_hot += DsaStatePool::cache_bytes(dsa_cfg_, 1, dsa_slots);
    arena_.init(ac);
    pool_.init(arena_, dsa_cfg_, 1, dsa_slots);
    dsa_scratch_ = static_cast<uint8_t*>(
        alloc_managed(DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                              dsa_slots)));
  } else {
    arena_.init(ac);
  }

  // Per-layer KDA state, contiguous so one forward zeros it all.
  if (kda_cfg_.num_kda_layers > 0) {
    kda_rec_ = static_cast<float*>(alloc_managed(
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
        kda_geo_.recurrent_bytes));
    kda_conv_ = static_cast<uint16_t*>(alloc_managed(
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
        kda_geo_.conv_committed_bytes));
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
      static_cast<uint16_t*>(alloc_managed(T * cfg_.vocab_size * 2));
}

GlmDiagnosticModel::~GlmDiagnosticModel() {
  cudaFree(gemm_ws_);
  cudaFree(dsa_scratch_);
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
  if (stream_) cudaStreamDestroy(stream_);
}

GlmMoeWeights GlmDiagnosticModel::moe_weights(const GlmMoeResident& r) {
  GlmMoeWeights w;
  w.router_gate = r.router_gate;
  w.router_bias = r.router_bias;
  w.experts = r.experts.data();
  for (int i = 0; i < 3; ++i) w.shared[i] = r.shared[i];
  return w;
}

void GlmDiagnosticModel::enqueue_dense_mlp(
    const uint16_t* x, uint16_t* out, const GlmQuantMatrix (&dense)[3],
    int tokens, cudaStream_t stream) {
  const int H = cfg_.hidden_size;
  const int I = cfg_.intermediate_size;
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
// index 0. Free-run forward passes null for both.
GlmDiagnosticModel::Outputs GlmDiagnosticModel::run_stack(
    const std::vector<int64_t>& token_ids, const uint16_t* const* layer_inputs,
    std::vector<std::vector<uint16_t>>* capture) {
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
  if (!gemm_.ensure_plan(T, cfg_.vocab_size, H, DType::BF16, GemmOut::BF16,
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

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = r.mhc.attn_fn;
    hw.base = r.mhc.attn_base;
    hw.scale = r.mhc.attn_scale;
    launch_mhc_compute(cur, hw, mhc_cfg_, collapsed_, post_, comb_, T,
                       stream_);    glm_rmsnorm_bf16(collapsed_, r.ln1, normed_, T, H, eps, stream_);    if (r.kind == GlmLayerKind::Kda) {
      if (!kda_) {
        kda_ = std::make_unique<KdaLayer>(arena_, gemm_, r.kda, kda_cfg_,
                                          max_tokens_, gemm_ws_,
                                          gemm_ws_bytes_, eps);
      } else {
        kda_->rebind(r.kda);
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
      kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, sub_out_, T,
                    stream_);      ++kda_ordinal;
    } else {
      if (!dsa_) {
        dsa_ = std::make_unique<DsaLayer>(
            gemm_, r.dsa, dsa_cfg_, max_tokens_,
            pool_.max_token_slots(), dsa_scratch_,
            DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                    pool_.max_token_slots()),
            gemm_ws_, gemm_ws_bytes_);
      } else {
        dsa_->rebind(r.dsa);
      }
      if (!dsa_->prepare(T))
        throw std::runtime_error("forward: DSA GEMM plans unavailable");
      dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, 0, 0, T, sub_out_,
                            stream_);
      ++dsa_ordinal;
    }
    launch_mhc_stream_update(post_, comb_, sub_out_, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);

    // ---- feed-forward site -----------------------------------------
    GlmMhcWeights fw;
    fw.fn = r.mhc.ffn_fn;
    fw.base = r.mhc.ffn_base;
    fw.scale = r.mhc.ffn_scale;
    launch_mhc_compute(cur, fw, mhc_cfg_, collapsed_, post_, comb_, T,
                       stream_);
    glm_rmsnorm_bf16(collapsed_, r.ln2, normed_, T, H, eps, stream_);
    if (cfg_.mlps[layer] == GlmMlpKind::Dense) {
      enqueue_dense_mlp(normed_, sub_out_, r.dense, T, stream_);    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(moe_weights(r.moe), moe_cfg_,
                                             max_tokens_);
      } else {
        moe_->rebind(moe_weights(r.moe));
      }
      moe_->enqueue(normed_, sub_out_, T, stream_);
      GlmRouteTraceLayer route;
      route.layer_idx = static_cast<uint32_t>(layer);
      route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
      route.tokens = static_cast<uint64_t>(T);
      route.ids = moe_->last_ids();
      route.weights = moe_->last_weights();
      out.routes.push_back(std::move(route));
      out.route_biased.push_back(moe_->last_biased());
    }
    launch_mhc_stream_update(post_, comb_, sub_out_, cur, nxt, mhc_cfg_, T,
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
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, cfg_.vocab_size, H,
               DType::BF16, GemmOut::BF16, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

    const size_t TH = static_cast<size_t>(T) * H;
  const size_t TV = static_cast<size_t>(T) * cfg_.vocab_size;
  out.final_hidden_bits.assign(normed_, normed_ + TH);
  out.logits_bits.assign(logits_, logits_ + TV);
  return out;
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward(
    const std::vector<int64_t>& token_ids) {
  return run_stack(token_ids, nullptr, nullptr);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::forward_isolated(
    const std::vector<int64_t>& token_ids,
    const std::vector<const uint16_t*>& layer_inputs,
    std::vector<std::vector<uint16_t>>& capture) {
  if (layer_inputs.size() != static_cast<size_t>(cfg_.num_hidden_layers) + 1)
    throw std::invalid_argument(
        "forward_isolated: needs num_layers+1 input snapshots");
  return run_stack(token_ids, layer_inputs.data(), &capture);
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
