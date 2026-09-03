#include "models/glm_forward.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/kernels.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm_step_timing.hpp"

namespace dgpp {
namespace {


// A quantized matrix's two device ranges (payload and block scales are
// separate allocations), handed to the prefetcher as such.
void prefetch_quant(WeightPrefetcher& pf, const GlmQuantMatrix& m) {
  pf.add(m.payload, static_cast<size_t>(m.rows) * m.cols);
  pf.add(m.scales,
         static_cast<size_t>((m.rows + 127) / 128) * ((m.cols + 127) / 128) *
             sizeof(float));
}

}  // namespace

// ---------------------------------------------------------------------------
// The boundary prefetch windows. Each block boundary is a bus all-reduce
// the chain waits on (~30 us) followed by the mHC site and a norm (~30 us
// more of latency-bound kernels) before the next weight-streaming kernel
// starts: ~60 us during which DRAM would idle. The window opened here
// runs through all of it, so the other side's first ~12 MB of weights are
// in L2 when their kernels arrive. Only the weights that exist regardless
// of routing are prefetchable — the routed experts wait for the router.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::prefetch_ffn_side(const GlmLayerBound& b,
                                           bool dense_mlp) {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(b.mhc->ffn_fn,
                static_cast<size_t>(mhc_cfg_.coeff_rows()) * mhc_cfg_.hc_mult *
                    H * 2);
  prefetch_.add(b.ln2, H * 2);
  if (dense_mlp) {
    for (int i = 0; i < 3; ++i) prefetch_quant(prefetch_, b.dense[i]);
    return;
  }
  prefetch_.add(b.moe->router_gate,
                static_cast<size_t>(moe_cfg_.n_experts) * H * 2);
  prefetch_.add(b.moe->router_bias,
                static_cast<size_t>(moe_cfg_.n_experts) * sizeof(float));
  for (int i = 0; i < 3; ++i) prefetch_quant(prefetch_, b.moe->shared[i]);
}

void GlmDiagnosticModel::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled()) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  // Resident stacks only: load_layer is a lookup there. A streaming stack
  // loads on demand, and touching layer N+1 during layer N would reorder
  // the loader's one-layer-at-a-time contract.
  if (loader_.residency() != GlmResidency::Resident) return;
  const GlmLayerResident& r = loader_.load_layer(layer);
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(r.mhc.attn_fn,
                static_cast<size_t>(mhc_cfg_.coeff_rows()) * mhc_cfg_.hc_mult *
                    H * 2);
  prefetch_.add(r.ln1, H * 2);
  // The first projection is far larger than the window; add() clamps to
  // the budget and the GEMV's leading blocks are the ones that hit.
  if (r.kind == GlmLayerKind::Kda) {
    if (kda_) prefetch_.add(r.kda.in_proj, kda_->in_proj_bytes());
  } else {
    if (dsa_) prefetch_.add(r.dsa.qkv_a, dsa_->qkv_a_bytes());
  }
}

void GlmDiagnosticModel::prefetch_head() {
  if (!prefetch_.enabled()) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, 0, prefetch_.boundary_rate());
  prefetch_.add(globals_.final_norm, H * 2);
  prefetch_.add(globals_.lm_head, static_cast<size_t>(lm_vocab_count_) * H * 2);
}

// ---------------------------------------------------------------------------
// session_prefill: opens request slot `req` and processes the prompt.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_prefill(
    int req, const std::vector<int64_t>& prompt_ids) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_prefill: request slot " +
                           std::to_string(req));
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

  // Open THIS slot only — other slots' sessions are untouched (the Stage
  // 2b concurrency contract). Slot 0's zeroed state is the SAME starting
  // state run_stack builds, so a single-chunk prefill there still runs
  // the exact reference op sequence (the bitwise tier of the parity gate).
  if (kda_rec_) {
    float* slot_rec =
        kda_rec_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers *
                       kda_geo_.recurrent_elems;
    uint16_t* slot_conv =
        kda_conv_ + static_cast<size_t>(req) * kda_cfg_.num_kda_layers *
                        (kda_geo_.conv_committed_bytes / 2);
    DGPP_CUDA_OK(cudaMemsetAsync(
        slot_rec, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) * kda_geo_.recurrent_bytes,
        stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(
        slot_conv, 0,
        static_cast<size_t>(kda_cfg_.num_kda_layers) *
            kda_geo_.conv_committed_bytes,
        stream_));
  }
  if (dsa_cfg_.num_dsa_layers > 0) pool_.reset_request(req, stream_);
  session_pos_[static_cast<size_t>(req)] = 0;
  if (mtp_) mtp_pos_[static_cast<size_t>(req)] = 0;

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
        req,
        std::vector<int64_t>(prompt_ids.begin() + c0,
                             prompt_ids.begin() + c0 + n),
        c0, /*decode_row=*/false);
    out.logits = std::move(chunk.logits);
    out.final_hidden_bits = std::move(chunk.final_hidden_bits);
    out.lm_vocab_begin = chunk.lm_vocab_begin;
    out.lm_vocab_count = chunk.lm_vocab_count;
    session_merge_routes(&out, std::move(chunk));
    c0 += n;
  }
  session_pos_[static_cast<size_t>(req)] = P;
  // The draft block over the prompt (its cache must cover every position
  // before the first draft); the chunks above left h_q in the cache.
  if (mtp_) mtp_prefill(req, prompt_ids);
  return out;
}
// ---------------------------------------------------------------------------
// session_step: one token at slot `req`'s next position.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_step(int req,
                                                             int64_t token_id) {
  return session_verify(req, std::vector<int64_t>{token_id});
}

// ---------------------------------------------------------------------------
// session_verify / session_rollback: T rows in one call, retract the tail.
// ---------------------------------------------------------------------------
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_verify(
    int req, const std::vector<int64_t>& token_ids) {
  step_timing::Scope tick(step_timing::kStep);
  session_decode_host_prep(req, token_ids, /*upload=*/true);
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  Outputs out = session_run_rows(req, token_ids, pos, /*decode_row=*/true);
  session_pos_[static_cast<size_t>(req)] += static_cast<int64_t>(token_ids.size());
  return out;
}

size_t GlmDiagnosticModel::spec_tail_ring_elems() const {
  return static_cast<size_t>(2) * dsa_cfg_.index_kpool * dsa_cfg_.index_head_dim;
}

void GlmDiagnosticModel::session_rollback(int req, int accepted) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_rollback: request slot " +
                            std::to_string(req));
  const int T = decode_rows_;
  if (accepted < 1 || accepted > T)
    throw std::invalid_argument("session_rollback: accepted rows must be in "
                                "[1, " + std::to_string(T) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos < T)
    throw std::invalid_argument("session_rollback: no verify to retract");
  if (accepted == T) return;  // every row landed in place already

  // The state after row `accepted-1` lives in snapshot row accepted-1;
  // the request's committed layer slots are contiguous, so each state
  // family rolls back in one copy (the DSA rings per layer).
  const size_t row = static_cast<size_t>(accepted - 1);
  if (kda_cfg_.num_kda_layers > 0) {
    const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
    DGPP_CUDA_OK(cudaMemcpyAsync(
        kda_rec_ + static_cast<size_t>(req) * layers * kda_geo_.recurrent_elems,
        spec_rec_ + row * layers * kda_geo_.recurrent_elems,
        layers * kda_geo_.recurrent_bytes, cudaMemcpyDeviceToDevice, stream_));
    const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
    DGPP_CUDA_OK(cudaMemcpyAsync(
        kda_conv_ + static_cast<size_t>(req) * layers * conv_elems,
        spec_conv_ + row * layers * conv_elems,
        layers * kda_geo_.conv_committed_bytes, cudaMemcpyDeviceToDevice,
        stream_));
  }
  const size_t ring = spec_tail_ring_elems();
  for (int layer = 0; layer < main_dsa_layers_; ++layer) {
    uint16_t* tail = static_cast<uint16_t*>(pool_.tail(layer)) +
                     static_cast<size_t>(req) * ring;
    const uint16_t* snap =
        spec_tail_ + (static_cast<size_t>(layer) * kSpecRows + row) * ring;
    DGPP_CUDA_OK(cudaMemcpyAsync(tail, snap, ring * 2,
                                 cudaMemcpyDeviceToDevice, stream_));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  session_pos_[static_cast<size_t>(req)] = pos - (T - accepted);
}

// The decode step's host-side half (see the header): one implementation
// so the eager, capture, and replay paths validate and stage IDENTICALLY.
void GlmDiagnosticModel::session_decode_host_prep(
    int req, const std::vector<int64_t>& ids, bool upload) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_decode: request slot " +
                            std::to_string(req));
  const int T = static_cast<int>(ids.size());
  if (T < 1 || T > kSpecRows)
    throw std::invalid_argument("session_decode: row count must be in [1, " +
                                std::to_string(kSpecRows) + "]");
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0)
    throw std::invalid_argument("session_decode: no open session on slot " +
                                std::to_string(req));
  for (int64_t id : ids)
    if (id < 0 || id >= cfg_.vocab_size)
      throw std::invalid_argument("session_decode: token id out of range");
  if (pos + T > max_tokens_)
    throw std::invalid_argument("session_decode: position exceeds max_tokens");

  // DSA admission: the block table must cover every row's position
  // BEFORE enqueue_decode (its pos is device state; growth is host
  // control). Blocks a rolled-back row reserved stay reserved — harmless,
  // the scheduler budgets prompt + max_steps up front anyway.
  if (dsa_cfg_.num_dsa_layers > 0 &&
      !pool_.ensure_request_blocks(req, pos + T, stream_))
    throw std::runtime_error("session_decode: DSA pool exhausted (admission "
                            "budget) — grow the pool or shed requests");

  // Decode-batch metadata: T consecutive rows of ONE request (time-
  // multiplexed requests). The PINNED members are the upload sources —
  // eager issues the H2Ds, capture records them as memcpy nodes, the
  // replay stage only writes the members (its graph re-uploads).
  for (int r = 0; r < T; ++r) {
    h_req_ids_[r] = req;
    h_step_pos_[r] = pos + r;
    h_token_[r] = ids[static_cast<size_t>(r)];
  }
  h_req_spans_[0] = 0;
  h_req_spans_[1] = T;
  decode_rows_ = T;
  if (upload) {
    DGPP_CUDA_OK(cudaMemcpyAsync(d_req_ids_, h_req_ids_, sizeof(int32_t) * T,
                                 cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_step_pos_, h_step_pos_, sizeof(int64_t) * T,
                                 cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_req_spans_, h_req_spans_, 2 * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream_));
  }
}

// ---------------------------------------------------------------------------
// The graph era (DESIGN §6.2). The three replay-side halves the caller
// sequences around ITS bus arm/launch/finish dance — see the header.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::session_graph_prepare() {
  if (loader_.residency() != GlmResidency::Resident)
    throw std::logic_error(
        "session_graph_prepare: the decode graph needs a resident stack "
        "(a streaming stack rebinds every layer through one slot)");
  int moe_ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    if (cfg_.mlps[layer] != GlmMlpKind::Moe) continue;
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b = bind_layer(r, /*dense_mlp=*/false);
    if (!moe_)
      moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                           kDecodeRows, moe_graph_slots());
    moe_->rebind(*b.moe);
    moe_->prepare_graph_table(moe_ordinal++, stream_);
  }
  if (mtp_) {
    // The draft layer's MoE takes the slot after the main stack's.
    const GlmLayerResident& r = stack_layer(cfg_.mtp_layer());
    const GlmLayerBound b = bind_layer(r, /*dense_mlp=*/false);
    moe_->rebind(*b.moe);
    moe_->prepare_graph_table(n_moe_layers_, stream_);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

void GlmDiagnosticModel::session_graph_capture_step(int req,
                                                    int64_t token_id) {
  session_graph_capture_step(req, std::vector<int64_t>{token_id});
}

void GlmDiagnosticModel::session_graph_capture_step(
    int req, const std::vector<int64_t>& ids) {
  session_decode_host_prep(req, ids, /*upload=*/true);
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  // The uploads and every launch record; the walk's syncs are skipped
  // inside (capture_mode). NOTHING EXECUTES — no state, no position.
  Outputs out = session_run_rows(req, ids, pos, /*decode_row=*/true,
                                 /*capture_mode=*/true);
  (void)out;  // empty by contract; the caller instantiates the graph
}

void GlmDiagnosticModel::session_graph_stage(int req, int64_t token_id) {
  session_graph_stage(req, std::vector<int64_t>{token_id});
}

void GlmDiagnosticModel::session_graph_stage(int req,
                                             const std::vector<int64_t>& ids) {
  session_decode_host_prep(req, ids, /*upload=*/false);
}

GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_graph_collect(
    int req) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_graph_collect: request slot " +
                           std::to_string(req));
  const int64_t pos = session_pos_[static_cast<size_t>(req)];
  if (pos <= 0)
    throw std::invalid_argument("session_graph_collect: no open session");
  // The caller synced the stream and finished the bus window: the
  // graph's D2H nodes joined, the state advanced in place, the logits
  // are stable. Materialize exactly as the eager tail does.
  Outputs out = session_decode_tail(decode_rows_);
  session_pos_[static_cast<size_t>(req)] = pos + decode_rows_;
  return out;
}

// ---------------------------------------------------------------------------
// session_close: retires the slot — blocks return to the free pool (the
// scheduler's admission meters see the capacity again) and the slot may be
// reopened by a later prefill. No collective; safe between any two ops.
// ---------------------------------------------------------------------------
void GlmDiagnosticModel::session_close(int req) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_close: request slot " +
                            std::to_string(req));
  if (dsa_cfg_.num_dsa_layers > 0)
    pool_.release_request_blocks(req, stream_);
  session_pos_[static_cast<size_t>(req)] = 0;
  if (mtp_) mtp_pos_[static_cast<size_t>(req)] = 0;
}

int64_t GlmDiagnosticModel::session_position(int req) const {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("session_position: request slot " +
                            std::to_string(req));
  return session_pos_[static_cast<size_t>(req)];
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
    int req, const std::vector<int64_t>& ids, int64_t token_start,
    bool decode_row, bool capture_mode) {
  const int T = static_cast<int>(ids.size());
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;

  // The head runs on every row of a prefill chunk although greedy reads
  // only the last: a last-row head would come off the m=1 GEMV while the
  // re-forward reference's comes off the m=T GEMM, and the prefill ==
  // re-forward BITWISE gate (glm_tp_test) is worth more than the ~6 ms
  // and 300 MB a 2048-row head costs per chunk.
  if (!gemm_.ensure_plan(T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
                         H))
    throw std::runtime_error("session: lm head GEMM plan unavailable");

  // The decode rows' token id rides the PINNED member (a memcpy node's
  // baked source; a pageable async copy would stream-sync anyway).
  // Prefill keeps the caller's vector (its syncs amortize over chunks).
  const int64_t* ids_src = decode_row ? h_token_ : ids.data();
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, ids_src,
                               static_cast<size_t>(T) * 8,
                               cudaMemcpyHostToDevice, stream_));
  // The step's first node: when did the GPU actually start this replay?
  // (The bus logs arm -> first collective; this splits it at the graph's
  // own start.) One 1-thread kernel; decode rows only.
  if (decode_row) launch_globaltimer_stamp(h_graph_start_gt_, stream_);
  glm_embed_bcast_streams(globals_.embed, d_tokens_, streams_[0], T, H,
                          stream_);

  Outputs out;
  out.routes.reserve(static_cast<size_t>(cfg_.num_hidden_layers));
  uint16_t* cur = streams_[0];
  uint16_t* nxt = streams_[1];
  int dsa_ordinal = 0;
  int kda_ordinal = 0;
  // Decode-path route-trace bookkeeping: enqueue_decode defers its
  // traces (async pinned copies — no round-trip); the entries pushed
  // during the loop are materialized from staging after the final sync.
  int moe_decode_calls = 0;

  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const GlmLayerResident& r = stack_layer(layer);
    const GlmLayerBound b = bind_layer(r, cfg_.mlps[layer] == GlmMlpKind::Dense);

    // ---- attention site --------------------------------------------
    GlmMhcWeights hw;
    hw.fn = b.mhc->attn_fn;
    hw.base = b.mhc->attn_base;
    hw.scale = b.mhc->attn_scale;
    launch_mhc_compute_normed(cur, hw, mhc_cfg_, collapsed_, post_, comb_,
                              mhc_logits_, b.ln1, normed_, eps, T, stream_,
                              mhc_counters_);
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
      // Slot-major state: request `req`'s layer-`kda_ordinal` slice.
      float* rec =
          kda_rec_ +
          (static_cast<size_t>(req) * kda_cfg_.num_kda_layers +
           static_cast<size_t>(kda_ordinal)) *
              kda_geo_.recurrent_elems;
      uint16_t* conv =
          kda_conv_ +
          (static_cast<size_t>(req) * kda_cfg_.num_kda_layers +
           static_cast<size_t>(kda_ordinal)) *
              (kda_geo_.conv_committed_bytes / 2);
      // In-place state update: prefill chunks and steps share ONE
      // recurrence implementation, so the state this enqueue leaves is
      // exactly the state the next row needs (DESIGN §7.1). Speculative
      // rows (decode, T > 1) also leave post-row snapshots for
      // session_rollback; the snapshot row stride spans every KDA layer.
      KdaSpeculativeSinks spec;
      if (decode_row && T > 1) {
        const size_t layers = static_cast<size_t>(kda_cfg_.num_kda_layers);
        spec.recurrent.states =
            spec_rec_ + static_cast<size_t>(kda_ordinal) * kda_geo_.recurrent_elems;
        spec.recurrent.stride_elems =
            static_cast<int64_t>(layers * kda_geo_.recurrent_elems);
        const size_t conv_elems = kda_geo_.conv_committed_bytes / 2;
        spec.conv.states =
            spec_conv_ + static_cast<size_t>(kda_ordinal) * conv_elems;
        spec.conv.stride_elems = static_cast<int64_t>(layers * conv_elems);
      }
      kda_->enqueue(normed_, rec, conv, kda_geo_.conv_hist, attn_out, T,
                    stream_, decode_row ? &prefetch_ : nullptr, spec);
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
        // The decode table's T rows all serve `req` (staged in
        // session_decode_host_prep); num_requests=1 — time-multiplexed
        // steps. Speculative rows leave post-row ring snapshots.
        void* tail_snaps =
            T > 1 ? spec_tail_ + static_cast<size_t>(dsa_ordinal) *
                                     kSpecRows * spec_tail_ring_elems()
                  : nullptr;
        dsa_->enqueue_decode(normed_, pool_, dsa_ordinal, d_req_ids_,
                             d_step_pos_, d_req_spans_, /*num_requests=*/1, T,
                             attn_out, stream_, &prefetch_, tail_snaps);
      } else {
        dsa_->enqueue_prefill(normed_, pool_, dsa_ordinal, /*req=*/req,
                              token_start, T, attn_out, stream_);
      }
      ++dsa_ordinal;
    }
    const bool dense_mlp = cfg_.mlps[layer] == GlmMlpKind::Dense;
    if (decode_row) prefetch_ffn_side(b, dense_mlp);
    if (boundary_) {
      // A captured sync is an error — under capture the fold is a
      // recorded node and the stream order IS the drain.
      if (!capture_mode) {
        step_timing::Scope drain(step_timing::kFoldDrain);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      }
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
    launch_mhc_compute_normed(cur, fw, mhc_cfg_, collapsed_, post_, comb_,
                              mhc_logits_, b.ln2, normed_, eps, T, stream_,
                              mhc_counters_);
    uint16_t* ffn_out = sub_out_;
    if (boundary_) {
      if (uint16_t* staged = boundary_->stage(T, H)) ffn_out = staged;
    }
    if (dense_mlp) {
      enqueue_dense_mlp(normed_, ffn_out, b.dense, T, stream_);
    } else {
      if (!moe_) {
        moe_ = std::make_unique<GlmMoeLayer>(*b.moe, moe_cfg_, max_tokens_,
                                             kDecodeRows);
      } else {
        moe_->rebind(*b.moe);
      }
      if (decode_row) {
        // The decode fast path (2026-09-01): the MoE runs from the
        // DEVICE-side route — no router round-trip, no host
        // segmentation, no per-segment H2D. Traces ride async copies
        // into this layer's pinned staging slot; the placeholder route
        // entry is filled after the final sync below.
        MoeTraceStaging trace;
        const size_t lay = static_cast<size_t>(moe_decode_calls);
        trace.ids = moe_trace_ids_ + lay * kDecodeRows * moe_cfg_.top_k;
        trace.weights =
            moe_trace_weights_ + lay * kDecodeRows * moe_cfg_.top_k;
        trace.biased =
            moe_trace_biased_ + lay * kDecodeRows * moe_cfg_.n_experts;
        // Capture passes the layer's OWN graph table slot so the
        // recorded upload node replays THIS layer's expert views; the
        // eager path's shared pinned buffer serves both callers.
        moe_->enqueue_decode(normed_, ffn_out, T,
                             decode_route_traces_ ? &trace : nullptr, stream_,
                             capture_mode ? moe_decode_calls : -1);
        GlmRouteTraceLayer route;
        route.layer_idx = static_cast<uint32_t>(layer);
        route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
        route.tokens = static_cast<uint64_t>(T);
        out.routes.push_back(std::move(route));  // ids/weights: post-sync
        out.route_biased.emplace_back();
        ++moe_decode_calls;
      } else {
        // Prefill keeps the host-orchestrated path (the sync amortizes
        // over pool-aligned chunks) — and warms the decode path's
        // device expert tables on the way through.
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
    }
    if (decode_row) prefetch_attention_side(layer + 1);
    if (boundary_) {
      // A captured sync is an error — under capture the fold is a
      // recorded node and the stream order IS the drain.
      if (!capture_mode) {
        step_timing::Scope drain(step_timing::kFoldDrain);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      }
      boundary_->reduce(ffn_out, T, H);
    }
    launch_mhc_stream_update(post_, comb_, ffn_out, cur, nxt, mhc_cfg_, T,
                             stream_);
    std::swap(cur, nxt);
  }

  // ---- head: mean over streams, final norm, lm head -----------------
  launch_mhc_final_mean(cur, collapsed_, mhc_cfg_, T, stream_);
  // The draft block's hnorm input is THIS (pre-final-norm) hidden: keep
  // it per position. Decode rows scatter by device position (the graph
  // replays at moving positions); prefill chunks are contiguous.
  if (mtp_) {
    uint16_t* cache = mtp_hidden_cache(req);
    if (decode_row)
      glm_rows_scatter_bf16(collapsed_, d_step_pos_, cache, T, H, stream_);
    else
      DGPP_CUDA_OK(cudaMemcpyAsync(
          cache + static_cast<size_t>(token_start) * H, collapsed_,
          static_cast<size_t>(T) * H * 2, cudaMemcpyDeviceToDevice, stream_));
  }
  glm_rmsnorm_bf16(collapsed_, globals_.final_norm, normed_, T, H, eps,
                   stream_);
  gemm_.matmul(normed_, globals_.lm_head, logits_, T, lm_vocab_count_, H,
               DType::BF16, GemmOut::F32, H, gemm_ws_, gemm_ws_bytes_,
               stream_);
  // The prefetch side stream rejoins here: a capture must end with every
  // forked stream joined, and the eager tail's sync below should cover
  // the prefetches too (they read weights, nothing else).
  if (decode_row) prefetch_.join(stream_);
  // The decode tail's rows ride D2H into pinned mirrors (graph nodes when
  // capturing) so the host never touches the managed activations — see
  // h_tail_logits_'s comment for the 9 ms stall that bought this.
  {
    // Decode rows: all T rows (T <= kDecodeRows). Prefill chunks: the
    // LAST row only, into mirror row 0 (a prompt-sized logits matrix is
    // 100s of MB; greedy needs one row).
    const size_t first = decode_row ? 0 : static_cast<size_t>(T - 1);
    const size_t rows = decode_row ? static_cast<size_t>(T) : 1;
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_,
                                 logits_ + first * lm_vocab_count_,
                                 rows * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_hidden_, normed_ + first * H,
                                 rows * H * 2, cudaMemcpyDeviceToHost,
                                 stream_));
  }

  // Capture ends HERE: nothing executed, so there is nothing to sync
  // or materialize — the caller ends the capture, instantiates, and the
  // first replay performs this step for real.
  if (capture_mode) return Outputs{};

  {
    step_timing::Scope sync_tick(step_timing::kFinalSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }

  // The decode tail (route traces from the pinned staging the loop's
  // async copies just joined, last-row logits/hidden) is one shared
  // materializer — the eager step and the graph-era collect produce
  // byte-identical Outputs through it.
  if (decode_row) return session_decode_tail(T);

  // Last row only (see the runner's header note) — from the pinned
  // mirrors' row 0, which the D2H above filled with row T-1.
  out.final_hidden_bits.assign(h_tail_hidden_, h_tail_hidden_ + H);
  out.logits.assign(h_tail_logits_, h_tail_logits_ + lm_vocab_count_);
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  return out;
}

// The decode tail shared by the eager step and the graph-era collect
// (see the header): routes materialized from the per-MoE-layer pinned
// staging — the walk (eager) or the replay's D2H nodes (graph) filled
// them — plus the last row's logits/final_hidden off the stable device
// buffers. The route shape (one entry per MoE layer, actual layer
// indices) is exactly the eager path's.
GlmDiagnosticModel::Outputs GlmDiagnosticModel::session_decode_tail(int T) {
  const int H = cfg_.hidden_size;
  const size_t K = static_cast<size_t>(moe_cfg_.top_k);
  const size_t E = static_cast<size_t>(moe_cfg_.n_experts);
  const size_t rows = static_cast<size_t>(T);
  Outputs out;
  int moe_ordinal = 0;
  // Traces off: the staging was never written this step; report no routes
  // rather than stale ones.
  for (int layer = 0; decode_route_traces_ && layer < cfg_.num_hidden_layers;
       ++layer) {
    if (cfg_.mlps[layer] != GlmMlpKind::Moe) continue;
    const size_t lay = static_cast<size_t>(moe_ordinal);
    const int32_t* ids =
        moe_trace_ids_ + lay * kDecodeRows * moe_cfg_.top_k;
    const float* ws =
        moe_trace_weights_ + lay * kDecodeRows * moe_cfg_.top_k;
    const float* bs =
        moe_trace_biased_ + lay * kDecodeRows * moe_cfg_.n_experts;
    GlmRouteTraceLayer route;
    route.layer_idx = static_cast<uint32_t>(layer);
    route.top_k = static_cast<uint32_t>(moe_cfg_.top_k);
    route.tokens = static_cast<uint64_t>(T);
    route.ids.assign(ids, ids + rows * K);
    route.weights.assign(ws, ws + rows * K);
    out.routes.push_back(std::move(route));
    out.route_biased.emplace_back(bs, bs + rows * E);
    ++moe_ordinal;
  }
  // From the pinned mirrors the step's D2H copies filled (the caller
  // synced), never from the managed activations. Every row: a verify's
  // consumer compares each row's argmax with the next row's token.
  out.final_hidden_bits.assign(h_tail_hidden_, h_tail_hidden_ + rows * H);
  out.logits.assign(h_tail_logits_,
                    h_tail_logits_ + rows * static_cast<size_t>(lm_vocab_count_));
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
