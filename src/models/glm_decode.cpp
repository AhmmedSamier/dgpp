#include "models/glm_forward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {

// DESIGN §7.1: pool-aligned prefill chunks (initially 2048). Must be a
// multiple of DSA's kpool so continuation chunks stay pool-aligned; the
// FINAL chunk may end mid-pool (the tail persists into decode).
constexpr int kPrefillChunkTokens = 2048;

}  // namespace

// ---------------------------------------------------------------------------
// session_prefill: opens a fresh request and processes the prompt.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill(
    const std::vector<int64_t>& prompt_ids) {
  const int64_t P = static_cast<int64_t>(prompt_ids.size());
  if (P <= 0) throw std::invalid_argument("session_prefill: empty prompt");
  if (P > max_tokens_)
    throw std::invalid_argument("session_prefill: prompt exceeds max_tokens");
  for (int64_t id : prompt_ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_prefill: token id out of range");
  if (dsa_cfg_.num_dsa_layers > 0 &&
      kPrefillChunkTokens % dsa_cfg_.index_kpool != 0)
    throw std::runtime_error(
        "session_prefill: the chunk size broke the kpool-alignment contract");

  // Fresh request state — the SAME starting state run_stack builds, so a
  // single-chunk prefill runs the exact reference op sequence (the
  // bitwise tier of the parity gate).
  if (kda_rec_) {
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_rec_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.recurrent_bytes,
        stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(
        kda_conv_, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.conv_committed_bytes,
        stream_));
  }
  if (dsa_cfg_.num_dsa_layers > 0) pool_.reset_all(stream_);
  session_pos_ = 0;

  Outputs out;  // last row's logits/final_hidden; routes cover ALL rows
  // Chunking: boundaries stay pool-aligned (starts ≡ 0 mod kpool) and a
  // CONTINUATION chunk must carry at least kpool tokens — the DSA
  // tail-seed read is pinned to in-chunk k rows, and enqueue_prefill
  // rejects shorter continuations. A 1..kpool-1 token tail therefore
  // borrows one pool from its predecessor (which stays pool-aligned:
  // the chunk size is a multiple of kpool). First chunk: any size.
  const int64_t kpool =
      dsa_cfg_.num_dsa_layers > 0 ? dsa_cfg_.index_kpool : 1;
  int64_t c0 = 0;
  while (c0 < static_cast<int64_t>(P)) {
    int64_t n = std::min<int64_t>(kPrefillChunkTokens, P - c0);
    if (n < kpool && c0 > 0) {
      c0 -= kpool;
      n += kpool;
    }
    Outputs chunk = session_run_rows(
        std::vector<int64_t>(prompt_ids.begin() + c0,
                             prompt_ids.begin() + c0 + n),
        c0, /*decode_row=*/false);
    out.logits_bits = std::move(chunk.logits_bits);
    out.final_hidden_bits = std::move(chunk.final_hidden_bits);
    out.lm_vocab_begin = chunk.lm_vocab_begin;
    out.lm_vocab_count = chunk.lm_vocab_count;
    session_merge_routes(&out, std::move(chunk));
    c0 += n;
  }
  session_pos_ = P;
  return out;
}

// ---------------------------------------------------------------------------
// session_step: one token at the next position.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_step(int64_t token_id) {
  if (session_pos_ <= 0)
    throw std::invalid_argument("session_step: no open session");
  if (token_id < 0 || token_id >= cfg_.vocab_size)
    throw std::invalid_argument("session_step: token id out of range");
  if (session_pos_ + 1 > max_tokens_)
    throw std::invalid_argument("session_step: position exceeds max_tokens");

  // DSA admission: the block table must cover this position BEFORE
  // enqueue_decode (its pos is device state; growth is host control).
  if (dsa_cfg_.num_dsa_layers > 0 &&
      !pool_.ensure_request_blocks(0, session_pos_ + 1, stream_))
    throw std::runtime_error("session_step: DSA pool exhausted (admission "
                             "budget) — grow the pool or shed requests");

  // Decode-batch metadata (device; re-uploaded per step — mid-session
  // uploads are async, never device syncs, per the decode-path discipline).
  h_req_ids_[0] = 0;
  h_step_pos_[0] = session_pos_;
  h_req_spans_[0] = 0;
  h_req_spans_[1] = 1;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_req_ids_, h_req_ids_, sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_step_pos_, h_step_pos_, sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_req_spans_, h_req_spans_, 2 * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));

  Outputs out =
      session_run_rows(std::vector<int64_t>{token_id}, session_pos_,
                       /*decode_row=*/true);
  session_pos_ += 1;
  return out;
}

// ---------------------------------------------------------------------------
// The session's row runner. Modeled on run_stack with three deltas: the
// state is NEVER reset here (prefill owns the one reset, at open), the DSA
// path is chosen by `decode_row` (prefill chunks vs decode positions), and
// only the LAST row's logits/final_hidden are copied out (a prompt-sized
// logits matrix is 100s of MB at real dims; the last row is what greedy
// consumes, and the parity gate compares rows, not matrices).
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_run_rows(
    const std::vector<int64_t>& ids, int64_t token_start, bool decode_row) {
  const int T = static_cast<int>(ids.size());
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;

  if (!gemm_.ensure_plan(T, lm_vocab_count_, H, DType::BF16, GemmOut::BF16, H))
    throw std::runtime_error("session: lm head GEMM plan unavailable");

  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, ids.data(),
                               static_cast<size_t>(T) * 8,
                               cudaMemcpyHostToDevice, stream_));
  glm_embed_bcast_streams(globals_.embed, d_tokens_, streams_[0], T, H,
                          stream_);

  Outputs out;
  out.routes.reserve(static_cast<size_t>(cfg_.num_hidden_layers));
  uint16_t* cur = streams_[0];
  uint16_t* nxt = streams_[1];
  int dsa_ordinal = 0;
  int kda_ordinal = 0;

  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const GlmLayerResident& r = loader_.load_layer(layer);
    const GlmLayerBound b = bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = b.mhc->attn_fn;
    hw.base = b.mhc->attn_base;
    hw.scale = b.mhc->attn_scale;
    launch_mhc_compute(cur, hw, mhc_cfg_, collapsed_, post_, comb_, T,
                       stream_);
    glm_rmsnorm_bf16(collapsed_, b.ln1, normed_, T, H, eps, stream_);
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
        throw std::runtime_error("session: KDA GEMM plans unavailable");
      float* rec = kda_rec_ +
                   static_cast<size_t>(kda_ordinal) * kda_geo_.recurrent_elems;
      uint16_t* conv =
          kda_conv_ +
          static_cast<size_t>(kda_ordinal) *
              (kda_geo_.conv_committed_bytes / 2);
      // In-place state update: prefill chunks and steps share ONE
      // recurrence implementation, so the state this enqueue leaves is
      // exactly the state the next row needs (DESIGN §7.1).
      kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, attn_out, T,
                    stream_);
      ++kda_ordinal;
    } else {
      if (!dsa_) {
        dsa_ = std::make_unique<DsaLayer>(
            gemm_, *b.dsa, dsa_cfg_, max_tokens_, pool_.max_token_slots(),
            dsa_scratch_,
            DsaLayer::scratch_bytes(dsa_cfg_, max_tokens_,
                                    pool_.max_token_slots()),
            gemm_ws_, gemm_ws_bytes_);
      } else {
        dsa_->rebind(*b.dsa);
      }
      if (!dsa_->prepare(T))
        throw std::runtime_error("session: DSA GEMM plans unavailable");
      if (decode_row) {
        dsa_->enqueue_decode(normed_, pool_, dsa_ordinal, d_req_ids_,
                             d_step_pos_, d_req_spans_, /*num_requests=*/1, T,
                             attn_out, stream_);
      } else {
        dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, /*req=*/0,
                              token_start, T, attn_out, stream_);
      }
      ++dsa_ordinal;
    }
    if (boundary_) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_->reduce(attn_out, T, H);
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
    uint16_t* ffn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) ffn_out = staged;
    }
    if (cfg_.mlps[layer] == GlmMlpKind::Dense) {
      enqueue_dense_mlp(normed_, ffn_out, b.dense, T, stream_);
    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_);
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
      boundary_->reduce(ffn_out, T, H);
    }
    launch_mhc_stream_update(post_, comb_, ffn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);
  }

  // ---- head: mean over streams, final norm, lm head -----------------
  launch_mhc_final_mean(cur, collapsed_, mhc_cfg_, T, stream_);
  glm_rmsnorm_bf16(collapsed_, globals_.final_norm, normed_, T, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, lm_vocab_count_, H,
               DType::BF16, GemmOut::BF16, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

  // Last row only (see the runner's header note).
  const uint16_t* last_hidden =
      normed_ + static_cast<size_t>(T - 1) * H;
  const uint16_t* last_logits =
      logits_ + static_cast<size_t>(T - 1) * lm_vocab_count_;
  out.final_hidden_bits.assign(last_hidden, last_hidden + H);
  out.logits_bits.assign(last_logits, last_logits + lm_vocab_count_);
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

// Prefill chunks emit one route entry per MoE layer per chunk; the
// reference emits one per layer per forward. Merge same-layer entries
// along the token axis so Outputs.routes has the reference's shape.
void GlmDiagnosticModel::session_merge_routes(Outputs* out,
                                              Outputs&& chunk) const {
  for (size_t i = 0; i < chunk.routes.size(); ++i) {
    if (!out->routes.empty() &&
        out->routes.back().layer_idx == chunk.routes[i].layer_idx) {
      GlmRouteTraceLayer& dst = out->routes.back();
      const GlmRouteTraceLayer& src = chunk.routes[i];
      dst.ids.insert(dst.ids.end(), src.ids.begin(), src.ids.end());
      dst.weights.insert(dst.weights.end(), src.weights.begin(),
                         src.weights.end());
      dst.tokens += src.tokens;
      out->route_biased.back().insert(out->route_biased.back().end(),
                                      chunk.route_biased[i].begin(),
                                      chunk.route_biased[i].end());
    } else {
      out->routes.push_back(std::move(chunk.routes[i]));
      out->route_biased.push_back(std::move(chunk.route_biased[i]));
    }
  }
}

}  // namespace dgpp
