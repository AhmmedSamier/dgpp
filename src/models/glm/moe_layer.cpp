#include "models/glm/moe_layer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm/step_timing.hpp"

namespace dgpp {

size_t GlmMoeLayer::scratch_bytes(const GlmMoeConfig& cfg, int max_tokens,
                                  int decode_slots, int graph_table_slots,
                                  size_t* pinned_bytes) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t H = static_cast<size_t>(cfg.hidden);
  const size_t I = static_cast<size_t>(cfg.inter);
  const size_t K = static_cast<size_t>(cfg.top_k);
  const size_t E = static_cast<size_t>(cfg.n_experts);
  const size_t rows_total = M * (K + 1);
  const size_t tk_max = M * K;
  const size_t segs_max = E + 1;
  size_t dev = 0, pin = 0;
  dev += M * K * 4 * 2;              // d_ids_, d_weights_
  dev += M * E * 4 * 2;              // d_biased_, d_scores_
  dev += rows_total * 4;             // d_rows_
  dev += tk_max * 4 * 2;             // d_row_w_, d_slot_row_
  dev += segs_max * sizeof(MoeSegment);
  dev += segs_max * 3 * sizeof(MoeExpertView);
  pin += tk_max * 4 * 2;             // h_ids_pinned_, h_weights_pinned_
  pin += M * E * 4;                  // h_biased_pinned_
  pin += rows_total * 4;             // h_seg_rows_
  pin += tk_max * 4;                 // h_slot_row_
  pin += segs_max * sizeof(MoeSegment);
  pin += static_cast<size_t>(kViewRing) * segs_max * 3 * sizeof(MoeExpertView);
  dev += rows_total * H * 2;         // d_gather_
  dev += rows_total * I * 2 * 3;     // d_gate_, d_up_, d_act_
  dev += rows_total * H * 4;         // d_down_
  dev += M * H * 4;                  // d_acc_
  if (decode_slots > 0) {
    const size_t rows = static_cast<size_t>(decode_slots) * (K + 1);
    dev += rows * I * 2 + rows * H * 4 + rows * 4;
    dev += static_cast<size_t>(decode_slots) * sizeof(int);
    dev += sizeof(MoeExpertView) * E * 3;
    if (graph_table_slots > 0) {
      const size_t table = sizeof(MoeExpertView) * E * 3 * static_cast<size_t>(graph_table_slots);
      pin += table;
      dev += table;
    }
  }
  if (pinned_bytes) *pinned_bytes = pin;
  return dev;
}

GlmMoeLayer::GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
                          int max_tokens, int decode_slots,
                          int graph_table_slots)
    : w_(weights), cfg_(cfg), max_tokens_(max_tokens),
      decode_slots_(decode_slots), graph_table_slots_(graph_table_slots) {
  GlmMoeConfig::validate_config(cfg_);
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmMoeLayer: max_tokens must be positive");
  if (decode_slots_ < 0)
    throw std::invalid_argument("GlmMoeLayer: decode_slots must be >= 0");
  if (graph_table_slots_ < 0)
    throw std::invalid_argument(
        "GlmMoeLayer: graph_table_slots must be >= 0");
  if (!w_.router_gate || !w_.router_bias ||
      (!w_.experts && !w_.experts_fp4) || !w_.shared[0].payload)
    throw std::invalid_argument("GlmMoeLayer: null weight pointer");
  if (w_.experts && w_.experts_fp4)
    throw std::invalid_argument(
        "GlmMoeLayer: routed experts bound in both formats at once");

  const int M = max_tokens_;
  const size_t H = static_cast<size_t>(cfg_.hidden);
  const size_t I = static_cast<size_t>(cfg_.inter);
  // Plain device memory, not managed (2026-09-02): nothing on the host
  // ever dereferences these (the traces and the diagnostic path leave via
  // cudaMemcpyAsync), and on the GB10 managed pages are the slow
  // translation path for every kernel that touches them — the decode
  // slot chain touches them ~700 times per token.
  DGPP_CUDA_OK(cudaMalloc(&d_ids_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_weights_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_biased_,
                                  static_cast<size_t>(M) * cfg_.n_experts * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_scores_,
                                  static_cast<size_t>(M) * cfg_.n_experts * 4));
  // The grouped prefill path's rows: every routed (token, slot) plus the
  // tokens once more for the shared expert.
  const size_t rows_total = static_cast<size_t>(M) * (cfg_.top_k + 1);
  const size_t tk_max = static_cast<size_t>(M) * cfg_.top_k;
  const size_t segs_max = static_cast<size_t>(cfg_.n_experts) + 1;
  DGPP_CUDA_OK(cudaMalloc(&d_rows_, rows_total * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_row_w_, tk_max * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_slot_row_, tk_max * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_segs_, segs_max * sizeof(MoeSegment)));
  DGPP_CUDA_OK(cudaMalloc(&d_views_prefill_, segs_max * 3 * sizeof(MoeExpertView)));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_pinned_), tk_max * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_weights_pinned_),
                             tk_max * 4, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_biased_pinned_),
                             static_cast<size_t>(M) * cfg_.n_experts * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_seg_rows_),
                             rows_total * 4, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_slot_row_), tk_max * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_segs_),
                             segs_max * sizeof(MoeSegment), cudaHostAllocDefault));
  // The expert-view upload ring (see the member's comment).
  view_table_entries_ = segs_max * 3;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_view_ring_),
                             static_cast<size_t>(kViewRing) * view_table_entries_ *
                                 sizeof(MoeExpertView),
                             cudaHostAllocDefault));
  for (int i = 0; i < kViewRing; ++i)
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&view_ring_event_[i],
                                          cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaMalloc(&d_gather_, rows_total * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_gate_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_up_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_act_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_down_, rows_total * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_acc_, M * H * sizeof(float)));
  h_counts_.assign(cfg_.n_experts, 0);

  // Decode-slot scratch: tokens*(top_k+1) rows — routed slots plus the
  // shared expert's, per token. Sized by the decode-row bound, not
  // max_tokens: a short-prompt model still decodes full slots.
  if (decode_slots_ > 0) {
    const size_t rows =
        static_cast<size_t>(decode_slots_) * (cfg_.top_k + 1);
    DGPP_CUDA_OK(cudaMalloc(&d_slot_act_, rows * I * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_slot_down_, rows * H * sizeof(float)));
    DGPP_CUDA_OK(cudaMalloc(&d_slot_order_, rows * sizeof(int32_t)));
    // The fused router selection's tickets: one per decode row, zero at
    // rest (the last block of each launch resets its own).
    DGPP_CUDA_OK(cudaMalloc(&d_router_counters_,
                            static_cast<size_t>(decode_slots_) * sizeof(int)));
    DGPP_CUDA_OK(cudaMemset(d_router_counters_, 0,
                            static_cast<size_t>(decode_slots_) * sizeof(int)));
    // One table of every expert's three views, re-uploaded per eager
    // enqueue_decode from the upload ring (see h_view_ring_'s comment).
    DGPP_CUDA_OK(cudaMalloc(
        &d_expert_views_,
        sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts) * 3));
    // Per-slot capture sources: each recorded upload node bakes its
    // slot's address, whose contents freeze at capture time (resident
    // bindings). The eager path never touches these.
    if (graph_table_slots_ > 0) {
      const size_t table_bytes =
          sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts) * 3 *
          static_cast<size_t>(graph_table_slots_);
      DGPP_CUDA_OK(cudaHostAlloc(
          reinterpret_cast<void**>(&h_expert_views_graph_), table_bytes,
          cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaMalloc(&d_expert_views_graph_, table_bytes));
      graph_table_ready_.assign(static_cast<size_t>(graph_table_slots_),
                                false);
    }
  }
}

GlmMoeLayer::~GlmMoeLayer() {
  cudaFree(d_ids_);
  cudaFree(d_slot_row_);
  cudaFree(d_segs_);
  cudaFree(d_views_prefill_);
  if (h_ids_pinned_) cudaFreeHost(h_ids_pinned_);
  if (h_weights_pinned_) cudaFreeHost(h_weights_pinned_);
  if (h_biased_pinned_) cudaFreeHost(h_biased_pinned_);
  if (h_seg_rows_) cudaFreeHost(h_seg_rows_);
  if (h_slot_row_) cudaFreeHost(h_slot_row_);
  if (h_segs_) cudaFreeHost(h_segs_);
  if (h_view_ring_) cudaFreeHost(h_view_ring_);
  for (int i = 0; i < kViewRing; ++i)
    if (view_ring_event_[i]) cudaEventDestroy(view_ring_event_[i]);
  cudaFree(d_weights_);
  cudaFree(d_biased_);
  cudaFree(d_scores_);
  cudaFree(d_rows_);
  cudaFree(d_row_w_);
  cudaFree(d_gather_);
  cudaFree(d_gate_);
  cudaFree(d_up_);
  cudaFree(d_act_);
  cudaFree(d_down_);
  cudaFree(d_acc_);
  cudaFree(d_slot_act_);
  cudaFree(d_slot_down_);
  cudaFree(d_slot_order_);
  cudaFree(d_router_counters_);
  cudaFree(d_expert_views_);
  cudaFree(d_expert_views_graph_);
  cudaFreeHost(h_expert_views_graph_);
}

void GlmMoeLayer::prepare_graph_table(int table_slot, cudaStream_t stream) {
  if (table_slot < 0 || table_slot >= graph_table_slots_ ||
      h_expert_views_graph_ == nullptr)
    throw std::invalid_argument(
        "GlmMoeLayer: graph table slot out of range (construct with "
        "graph_table_slots)");
  const size_t E3 = static_cast<size_t>(cfg_.n_experts) * 3;
  const size_t off = static_cast<size_t>(table_slot) * E3;
  MoeExpertView* src = h_expert_views_graph_ + off;
  for (size_t i = 0; i < E3; ++i)
    src[i] = w_.nvfp4() ? MoeExpertView::of(w_.experts_fp4[i])
                        : MoeExpertView::of(w_.experts[i]);
  DGPP_CUDA_OK(cudaMemcpyAsync(d_expert_views_graph_ + off, src,
                               sizeof(MoeExpertView) * E3,
                               cudaMemcpyHostToDevice, stream));
  graph_table_ready_[static_cast<size_t>(table_slot)] = true;
}


// Every expert triple must share the routed geometry (the loader's
// contract: one slice width per rank); the shared triple has its own inter
// but the same hidden. Checked once per enqueue — the views can rebind.
void GlmMoeLayer::check_expert_geometry() const {
  const int H = cfg_.hidden, E = cfg_.n_experts;
  if (w_.nvfp4()) {
    const GlmFp4Matrix& g0 = w_.experts_fp4[0];
    for (int e = 0; e < E; ++e) {
      const GlmFp4Matrix* m = w_.experts_fp4 + static_cast<size_t>(e) * 3;
      if (m[0].cols != H || m[1].cols != H || m[0].rows != g0.rows ||
          m[1].rows != g0.rows || m[2].rows != H || m[2].cols != g0.rows ||
          !m[0].global_scale || !m[1].global_scale || !m[2].global_scale)
        throw std::runtime_error(
            "GlmMoeLayer: inconsistent NVFP4 routed expert matrices (expert " +
            std::to_string(e) + ")");
    }
    if (w_.shared[0].rows != w_.shared[1].rows ||
        w_.shared[2].cols != w_.shared[0].rows || w_.shared[2].rows != H ||
        w_.shared[0].cols != H || w_.shared[1].cols != H)
      throw std::runtime_error("GlmMoeLayer: inconsistent shared matrices");
    return;
  }
  const GlmQuantMatrix& g0 = w_.experts[0];
  for (int e = 0; e < E; ++e) {
    const GlmQuantMatrix* m = w_.experts + static_cast<size_t>(e) * 3;
    if (m[0].cols != H || m[1].cols != H || m[0].rows != g0.rows ||
        m[1].rows != g0.rows || m[2].rows != H || m[2].cols != g0.rows)
      throw std::runtime_error(
          "GlmMoeLayer: inconsistent routed expert matrices (expert " +
          std::to_string(e) + ")");
  }
  if (w_.shared[0].rows != w_.shared[1].rows ||
      w_.shared[2].cols != w_.shared[0].rows || w_.shared[2].rows != H ||
      w_.shared[0].cols != H || w_.shared[1].cols != H)
    throw std::runtime_error("GlmMoeLayer: inconsistent shared matrices");
}

void GlmMoeLayer::enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
                          cudaStream_t stream, MoeExpertKernel kernel) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int H = cfg_.hidden, E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();

  // 1. Router + one sync: the ids/weights round-trip is the diagnostic
  //    mode's cost; the production path keeps segmentation device-side.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
  const size_t tk = static_cast<size_t>(tokens) * K;
  DGPP_CUDA_OK(cudaMemcpyAsync(h_ids_pinned_, d_ids_, tk * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_weights_pinned_, d_weights_, tk * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_biased_pinned_, d_biased_,
                               static_cast<size_t>(tokens) * E * 4,
                               cudaMemcpyDeviceToHost, stream));
  {
    step_timing::Scope sync_tick(step_timing::kMoeSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  h_ids_.assign(h_ids_pinned_, h_ids_pinned_ + tk);
  h_weights_.assign(h_weights_pinned_, h_weights_pinned_ + tk);
  h_biased_.assign(h_biased_pinned_,
                   h_biased_pinned_ + static_cast<size_t>(tokens) * E);

  // 2. Segment by expert (ascending expert id — the accumulation order)
  //    straight into the pinned staging: the routed rows in segment order,
  //    then every token once more for the shared expert; per (token, slot)
  //    its gathered row; the segment table; the expert views (the shared
  //    triple last). ONE upload of each per layer.
  std::fill(h_counts_.begin(), h_counts_.end(), 0);
  for (size_t i = 0; i < tk; ++i) ++h_counts_[h_ids_[i]];
  std::vector<int> seg_begin(E, 0);
  for (int e = 1; e < E; ++e) seg_begin[e] = seg_begin[e - 1] + h_counts_[e - 1];
  std::vector<int> fill(seg_begin.begin(), seg_begin.end());
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      const int e = h_ids_[static_cast<size_t>(t) * K + i];
      h_seg_rows_[fill[e]] = t;
      h_slot_row_[static_cast<size_t>(t) * K + i] = fill[e];
      ++fill[e];
    }
  const int shared_row0 = static_cast<int>(tk);
  for (int t = 0; t < tokens; ++t) h_seg_rows_[tk + t] = t;
  int n_segs = 0, max_rows = 1;
  for (int e = 0; e < E; ++e) {
    if (h_counts_[e] == 0) continue;
    h_segs_[n_segs++] = MoeSegment{seg_begin[e], h_counts_[e], e};
    max_rows = std::max(max_rows, h_counts_[e]);
  }
  h_segs_[n_segs] = MoeSegment{shared_row0, tokens, E};  // the shared segment
  const size_t rows_total = tk + static_cast<size_t>(tokens);
  DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_seg_rows_, rows_total * 4,
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_slot_row_, h_slot_row_, tk * 4,
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_segs_, h_segs_,
                               static_cast<size_t>(n_segs + 1) * sizeof(MoeSegment),
                               cudaMemcpyHostToDevice, stream));
  upload_expert_views(d_views_prefill_, /*with_shared=*/true, stream);

  // 3. The grouped chain: gather every row once; gate and up over the
  //    routed segments in one launch each and the shared segment in one
  //    more (its inter may differ); swiglu over every row; the down
  //    projection the same way, fp32 (the chain rounds once, at the end).
  //    The inter dims come from the matrix views (the rank's slices).
  grouped_expert_chain(kernel, hidden, d_segs_, n_segs, max_rows,
                       d_segs_ + n_segs, tokens, rows_total, stream);

  // 4. The ordered accumulation: per token its K slots in ascending expert
  //    id, then the shared row (weight 1), the fmaf chain from zero, one
  //    rounding onto the wire buffer.
  launch_moe_accum_ordered(out, d_down_, H, d_slot_row_, d_ids_, d_weights_,
                           shared_row0, tokens, K, static_cast<int>(H), stream);
}

void GlmMoeLayer::upload_expert_views(MoeExpertView* d_dst, bool with_shared,
                                      cudaStream_t stream) {
  const int E = cfg_.n_experts;
  const int slot = view_ring_next_;
  view_ring_next_ = (view_ring_next_ + 1) % kViewRing;
  // The entry's previous upload must have executed before the fill
  // overwrites its source; the host is only ever made to wait here when
  // it is kViewRing uploads ahead of the stream.
  if (view_ring_armed_[slot])
    DGPP_CUDA_OK(cudaEventSynchronize(view_ring_event_[slot]));
  MoeExpertView* h = h_view_ring_ + static_cast<size_t>(slot) * view_table_entries_;
  for (size_t i = 0; i < static_cast<size_t>(E) * 3; ++i)
    h[i] = w_.nvfp4() ? MoeExpertView::of(w_.experts_fp4[i])
                      : MoeExpertView::of(w_.experts[i]);
  size_t n = static_cast<size_t>(E) * 3;
  if (with_shared) {
    for (int w = 0; w < 3; ++w)
      h[n + static_cast<size_t>(w)] = MoeExpertView::of(w_.shared[w]);
    n += 3;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(d_dst, h, n * sizeof(MoeExpertView),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaEventRecord(view_ring_event_[slot], stream));
  view_ring_armed_[slot] = true;
}

void GlmMoeLayer::enqueue_prefill(const uint16_t* hidden, uint16_t* out,
                                  int tokens, MoeTraceStaging* trace,
                                  cudaStream_t stream) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int H = cfg_.hidden, E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();
  const size_t tk = static_cast<size_t>(tokens) * K;

  // 1. Router; the traces ride async copies into the caller's pinned
  //    staging (no round trip).
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
  if (trace) {
    if (!trace->ids || !trace->weights || !trace->biased)
      throw std::invalid_argument("GlmMoeLayer: incomplete trace staging");
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->ids, d_ids_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->weights, d_weights_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->biased, d_biased_,
                                 static_cast<size_t>(tokens) * E * 4,
                                 cudaMemcpyDeviceToHost, stream));
  }

  // 2. Segmentation on the device: rows, slot map, segment table (every
  //    expert, empty ones included; the shared segment last).
  launch_moe_segment(d_ids_, tokens, K, E, d_rows_, d_slot_row_, d_segs_,
                     stream);
  // The expert views: the same table the host path uploads, from the
  // upload ring (the host runs ahead of the stream here — no per-layer
  // sync — so the source must not be a single table; see h_view_ring_).
  upload_expert_views(d_views_prefill_, /*with_shared=*/true, stream);

  // 3. The grouped chain over every expert's segment (an empty one's
  //    blocks exit at once) and the shared segment.
  const int shared_row0 = static_cast<int>(tk);
  const size_t rows_total = tk + static_cast<size_t>(tokens);
  grouped_expert_chain(MoeExpertKernel::kMma, hidden, d_segs_, E,
                       /*max_rows=*/std::max(tokens, 1), d_segs_ + E, tokens,
                       rows_total, stream);
  launch_moe_accum_ordered(out, d_down_, H, d_slot_row_, d_ids_, d_weights_,
                           shared_row0, tokens, K, H, stream);
}

void GlmMoeLayer::grouped_expert_chain(MoeExpertKernel kernel,
                                       const uint16_t* hidden,
                                       const MoeSegment* segs, int n_segs,
                                       int max_rows, const MoeSegment* shared_seg,
                                       int tokens, size_t rows_total,
                                       cudaStream_t stream) {
  const int H = static_cast<int>(cfg_.hidden);
  const bool fp4 = w_.nvfp4();
  const int I_r = static_cast<int>(fp4 ? w_.experts_fp4[0].rows : w_.experts[0].rows);
  const int I_s = static_cast<int>(w_.shared[0].rows);
  const size_t I_max = static_cast<size_t>(std::max(I_r, I_s));
  // The shared segment is every token: split across blocks along z (the
  // GEMV core in 16-row pieces, the tensor-core kernel in whole m-tiles).
  const bool mma = kernel == MoeExpertKernel::kMma;
  const int shared_split = mma ? 128 : 16;
  // The GEMV core reads a gathered copy of the rows; the tensor-core kernel
  // reads the hidden rows through the row map directly (2026-09-05: the
  // gather was 1.26 ms per layer at 2048 tokens).
  if (!mma)
    launch_moe_gather_rows(hidden, d_rows_, d_gather_, static_cast<int>(rows_total),
                           H, stream);
  // `routed` selects the routed experts' kernel family: the fp4 kernels
  // (tensor-core or GEMV) for NVFP4 tables, the fp8 ones otherwise; the
  // shared segment (FP8 under both formats) always takes the fp8 kernels.
  auto gemm_bf16 = [&](const MoeSegment* sg, int ns, int mr, int split, int which,
                       uint16_t* out, int n, bool routed) {
    if (mma && routed && fp4)
      launch_moe_grouped_mma_fp4_bf16(hidden, H, sg, ns, mr, split, d_views_prefill_,
                                      which, out, I_max, n, H, stream, d_rows_);
    else if (mma)
      launch_moe_grouped_mma_bf16(hidden, H, sg, ns, mr, split, d_views_prefill_,
                                  which, out, I_max, n, H, stream, d_rows_);
    else if (routed && fp4)
      launch_moe_grouped_gemv_fp4_bf16(d_gather_, H, sg, ns, mr, split, d_views_prefill_,
                                       which, out, I_max, n, H, stream);
    else
      launch_moe_grouped_gemv_bf16(d_gather_, H, sg, ns, mr, split, d_views_prefill_,
                                   which, out, I_max, n, H, stream);
  };
  auto gemm_f32 = [&](const MoeSegment* sg, int ns, int mr, int split, int k,
                      bool routed) {
    if (mma && routed && fp4)
      launch_moe_grouped_mma_fp4_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                     2, d_down_, H, H, k, stream);
    else if (mma)
      launch_moe_grouped_mma_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                 2, d_down_, H, H, k, stream);
    else if (routed && fp4)
      launch_moe_grouped_gemv_fp4_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                      2, d_down_, H, H, k, stream);
    else
      launch_moe_grouped_gemv_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                  2, d_down_, H, H, k, stream);
  };
  gemm_bf16(segs, n_segs, max_rows, 0, 0, d_gate_, I_r, true);
  gemm_bf16(shared_seg, 1, tokens, shared_split, 0, d_gate_, I_s, false);
  gemm_bf16(segs, n_segs, max_rows, 0, 1, d_up_, I_r, true);
  gemm_bf16(shared_seg, 1, tokens, shared_split, 1, d_up_, I_s, false);
  launch_moe_swiglu_clamp(d_gate_, d_up_, d_act_,
                          static_cast<int64_t>(rows_total) * I_max,
                          cfg_.swiglu_limit, stream);
  gemm_f32(segs, n_segs, max_rows, 0, I_r, true);
  gemm_f32(shared_seg, 1, tokens, shared_split, I_s, false);
  if (std::getenv("DGPP_MOE_CHAIN_DUMP") != nullptr) {
    // Hunt instrument: per-stage checksums of the chain's buffers.
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    auto sum_bf16 = [&](const uint16_t* d, size_t n) {
      std::vector<uint16_t> h(n);
      DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
      double acc = 0;
      for (uint16_t v : h) acc += std::fabs(bf16_bits_to_float(v));
      return acc;
    };
    auto sum_f32 = [&](const float* d, size_t n) {
      std::vector<float> h(n);
      DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
      double acc = 0;
      for (float v : h) acc += std::fabs(v);
      return acc;
    };
    std::vector<MoeSegment> hs(static_cast<size_t>(n_segs) + 1);
    DGPP_CUDA_OK(cudaMemcpy(hs.data(), segs, n_segs * sizeof(MoeSegment),
                            cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(&hs[n_segs], shared_seg, sizeof(MoeSegment),
                            cudaMemcpyDeviceToHost));
    std::string segtxt;
    for (const MoeSegment& sg : hs)
      if (sg.rows > 0)
        segtxt += " (" + std::to_string(sg.row0) + "," + std::to_string(sg.rows) + ",e" +
                  std::to_string(sg.expert) + ")";
    DGPP_LOG_INFO(
        "[chain {}] rows_total={} I_r={} I_s={} segs:{} | gather {:.6g} gate {:.6g} "
        "up {:.6g} act {:.6g} down {:.6g}",
        mma ? "mma" : "gemv", rows_total, I_r, I_s, segtxt,
        sum_bf16(d_gather_, rows_total * H), sum_bf16(d_gate_, rows_total * I_max),
        sum_bf16(d_up_, rows_total * I_max), sum_bf16(d_act_, rows_total * I_max),
        sum_f32(d_down_, rows_total * H));
  }
}

void GlmMoeLayer::enqueue_decode(const uint16_t* hidden, uint16_t* out,
                                 int tokens, MoeTraceStaging* trace,
                                 cudaStream_t stream, int table_slot) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (decode_slots_ <= 0)
    throw std::runtime_error(
        "GlmMoeLayer: decode path not provisioned (construct with "
        "decode_slots > 0)");
  if (tokens > decode_slots_)
    throw std::invalid_argument(
        "GlmMoeLayer: decode rows exceed decode_slots");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int H = cfg_.hidden, E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();

  // 1. Router, dots and selection in one launch — ids ASCENDING per row,
  //    on device.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream,
                    d_router_counters_);
  // 2. Route traces ride ASYNC copies into the caller's pinned staging;
  //    the caller materializes them after its next stream sync (the
  //    decode step's final sync). No round-trip on the hot path.
  if (trace) {
    if (!trace->ids || !trace->weights || !trace->biased)
      throw std::invalid_argument("GlmMoeLayer: incomplete trace staging");
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->ids, d_ids_,
                                  static_cast<size_t>(tokens) * K * 4,
                                  cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->weights, d_weights_,
                                  static_cast<size_t>(tokens) * K * 4,
                                  cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->biased, d_biased_,
                                  static_cast<size_t>(tokens) * E * 4,
                                  cudaMemcpyDeviceToHost, stream));
  }

  // 3. The slot chain. Slot layout: tokens*(K+1); slot t*(K+1)+j is row
  //    t's routed expert j (ascending id — the router's contract) and
  //    slot ..+K is the shared expert. EAGER: the expert-view table is
  //    RE-UPLOADED EVERY CALL (see the member's comment: the streaming
  //    loader makes binding-keyed caching a wrong-weights factory; one
  //    small async upload per layer per step is the honest price).
  //    CAPTURE (table_slot >= 0): the kernels read the slot's OWN device
  //    table, prepared before the capture — no upload node at all. A
  //    slot never prepared is refused (the wrong-weights class the 4c
  //    cache bug taught, caught at capture time instead of in the
  //    transcript).
  const MoeExpertView* table = d_expert_views_;
  if (table_slot >= 0) {
    if (table_slot >= graph_table_slots_ || d_expert_views_graph_ == nullptr)
      throw std::invalid_argument(
          "GlmMoeLayer: graph table slot out of range (construct with "
          "graph_table_slots)");
    if (!graph_table_ready_[static_cast<size_t>(table_slot)])
      throw std::logic_error(
          "GlmMoeLayer: graph table slot " + std::to_string(table_slot) +
          " was not prepared (call prepare_graph_table before capturing)");
    table = d_expert_views_graph_ +
            static_cast<size_t>(table_slot) * static_cast<size_t>(E) * 3;
  } else {
    upload_expert_views(d_expert_views_, /*with_shared=*/false, stream);
  }

  const int slots = tokens * (K + 1);
  const bool fp4 = w_.nvfp4();
  const int I_r = static_cast<int>(fp4 ? w_.experts_fp4[0].rows
                                       : w_.experts[0].rows);  // routed inter slice
  const int I_s = static_cast<int>(w_.shared[0].rows);   // shared inter slice
  // Multi-token batches (a speculative verify) run their slots in
  // expert order so an expert two rows share is read from DRAM once (see
  // launch_moe_slot_order); one token has nothing to share.
  const int32_t* order = nullptr;
  if (tokens > 1) {
    launch_moe_slot_order(d_ids_, d_slot_order_, slots, K, E, stream);
    order = d_slot_order_;
  }
  // Gate + up + swiglu in one launch (bit-identical to the three-launch
  // chain — see the launcher). Per-slot bounds are consumed downstream
  // (the down GEMV reads only k=I_s of the shared slot).
  if (fp4) {
    launch_moe_slot_gate_up_swiglu_fp4(
        hidden, H, d_ids_, order, table, I_r, H, I_s, H, w_.shared[0].payload,
        w_.shared[0].scales, w_.shared[1].payload, w_.shared[1].scales,
        d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream);
    launch_moe_slot_down_fp4(d_slot_act_, I_r, d_ids_, order, table, H, I_r, H,
                             I_s, w_.shared[2].payload, w_.shared[2].scales,
                             d_slot_down_, H, slots, K, stream);
  } else {
    launch_moe_slot_gate_up_swiglu(
        hidden, H, d_ids_, order, table, I_r, H, I_s, H, w_.shared[0].payload,
        w_.shared[0].scales, w_.shared[1].payload, w_.shared[1].scales,
        d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream);
    launch_moe_slot_down(d_slot_act_, I_r, d_ids_, order, table, H, I_r, H,
                         I_s, w_.shared[2].payload, w_.shared[2].scales,
                         d_slot_down_, H, slots, K, stream);
  }
  launch_moe_slot_accum(out, d_slot_down_, d_weights_, tokens, H, K, stream);
}

}  // namespace dgpp
