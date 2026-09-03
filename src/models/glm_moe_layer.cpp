#include "models/glm_moe_layer.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm_step_timing.hpp"

namespace dgpp {

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
  if (!w_.router_gate || !w_.router_bias || !w_.experts ||
      !w_.shared[0].payload)
    throw std::invalid_argument("GlmMoeLayer: null weight pointer");

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
  DGPP_CUDA_OK(cudaMalloc(&d_rows_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_row_w_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_gather_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_gate_, M * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_up_, M * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_act_, M * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_down_, M * H * sizeof(float)));
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
    // One table of every expert's three views, re-uploaded per
    // enqueue_decode. The source is PINNED (see the member's comment):
    // pageable async H2D syncs the stream before initiating, which would
    // drain the pipeline once per MoE layer per step.
    DGPP_CUDA_OK(cudaMalloc(
        &d_expert_views_,
        sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts) * 3));
    DGPP_CUDA_OK(cudaHostAlloc(
        reinterpret_cast<void**>(&h_expert_views_pinned_),
        sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts) * 3,
        cudaHostAllocDefault));
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
  cudaFree(d_expert_views_);
  cudaFree(d_expert_views_graph_);
  cudaFreeHost(h_expert_views_pinned_);
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
    src[i] = MoeExpertView{w_.experts[i].payload, w_.experts[i].scales};
  DGPP_CUDA_OK(cudaMemcpyAsync(d_expert_views_graph_ + off, src,
                               sizeof(MoeExpertView) * E3,
                               cudaMemcpyHostToDevice, stream));
  graph_table_ready_[static_cast<size_t>(table_slot)] = true;
}

void GlmMoeLayer::run_expert_segment(
    const uint16_t* x, const int32_t* rows_dev, const float* row_w_dev,
    int n_rows, const GlmQuantMatrix& gate, const GlmQuantMatrix& up,
    const GlmQuantMatrix& down, cudaStream_t stream) {
  // The inter dim comes from the matrix views, not the config: at world>1
  // every expert (routed and shared) is the rank's slice, I = cfg.inter /
  // world — the views know, the config does not.
  const int H = cfg_.hidden, I = static_cast<int>(gate.rows);
  if (gate.rows != up.rows || down.cols != gate.rows)
    throw std::runtime_error("GlmMoeLayer: inconsistent expert matrices");
  // Gather this segment's rows; the shared expert passes rows = identity
  // with x itself, but the gather is cheap and uniform — one code path.
  launch_moe_gather_rows(x, rows_dev, d_gather_, n_rows, H, stream);
  launch_scale_gemm_bf16(d_gather_, H, gate.payload, gate.scales, d_gate_,
                          n_rows, I, H, stream);
  launch_scale_gemm_bf16(d_gather_, H, up.payload, up.scales, d_up_, n_rows,
                          I, H, stream);
  launch_moe_swiglu_clamp(d_gate_, d_up_, d_act_,
                          static_cast<int64_t>(n_rows) * I,
                          cfg_.swiglu_limit, stream);
  // The down projection stays fp32: the chain rounds once, at the end.
  launch_scale_gemm_f32(d_act_, I, down.payload, down.scales, d_down_, n_rows,
                        H, I, stream);
  launch_moe_accum(d_acc_, d_down_, rows_dev, row_w_dev, n_rows, H, stream);
}

// Every expert triple must share the routed geometry (the loader's
// contract: one slice width per rank); the shared triple has its own inter
// but the same hidden. Checked once per enqueue — the views can rebind.
void GlmMoeLayer::check_expert_geometry() const {
  const int H = cfg_.hidden, E = cfg_.n_experts;
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
                          cudaStream_t stream) {
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
  h_ids_.resize(static_cast<size_t>(tokens) * K);
  h_weights_.resize(static_cast<size_t>(tokens) * K);
  h_biased_.resize(static_cast<size_t>(tokens) * E);
  DGPP_CUDA_OK(cudaMemcpyAsync(h_ids_.data(), d_ids_,
                               static_cast<size_t>(tokens) * K * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_weights_.data(), d_weights_,
                               static_cast<size_t>(tokens) * K * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_biased_.data(), d_biased_,
                               static_cast<size_t>(tokens) * E * 4,
                               cudaMemcpyDeviceToHost, stream));
  {
    step_timing::Scope sync_tick(step_timing::kMoeSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }

  // 2. Segment by expert (ascending expert id — the accumulation order).
  //    The fp32 chain starts at zero.
  DGPP_CUDA_OK(cudaMemsetAsync(d_acc_, 0,
                               static_cast<size_t>(tokens) * H * sizeof(float),
                               stream));
  std::fill(h_counts_.begin(), h_counts_.end(), 0);
  h_rows_.assign(static_cast<size_t>(tokens) * K, 0);
  h_row_w_.assign(static_cast<size_t>(tokens) * K, 0.f);
  // First pass: counts per expert; second pass: stable row placement.
  for (size_t i = 0; i < h_ids_.size(); ++i) ++h_counts_[h_ids_[i]];
  std::vector<int> seg_begin(E, 0);
  for (int e = 1; e < E; ++e) seg_begin[e] = seg_begin[e - 1] + h_counts_[e - 1];
  std::vector<int> fill(seg_begin.begin(), seg_begin.end());
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      const int e = h_ids_[static_cast<size_t>(t) * K + i];
      h_rows_[fill[e]] = t;
      h_row_w_[fill[e]] = h_weights_[static_cast<size_t>(t) * K + i];
      ++fill[e];
    }

  // 3. Per-expert segments in ascending order (skipping empty ones). Every
  //    expert is local — this rank's slice of it; the FFN all-reduce folds
  //    the ranks' partial sums.
  for (int e = 0; e < E; ++e) {
    const int n = h_counts_[e];
    if (n == 0) continue;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_rows_.data() + seg_begin[e],
                                  static_cast<size_t>(n) * 4,
                                  cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_row_w_, h_row_w_.data() + seg_begin[e],
                                  static_cast<size_t>(n) * 4,
                                  cudaMemcpyHostToDevice, stream));
    run_expert_segment(hidden, d_rows_, d_row_w_, n,
                       w_.experts[static_cast<size_t>(e) * 3 + 0],
                       w_.experts[static_cast<size_t>(e) * 3 + 1],
                       w_.experts[static_cast<size_t>(e) * 3 + 2], stream);
  }

  // 4. Shared expert: all tokens, weight 1, added last; then the chain's
  //    single rounding onto the wire buffer.
  h_rows_.resize(tokens);
  for (int t = 0; t < tokens; ++t) h_rows_[t] = t;
  h_row_w_.assign(tokens, 1.0f);
  DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_rows_.data(),
                                static_cast<size_t>(tokens) * 4,
                                cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_row_w_, h_row_w_.data(),
                                static_cast<size_t>(tokens) * 4,
                                cudaMemcpyHostToDevice, stream));
  run_expert_segment(hidden, d_rows_, d_row_w_, tokens, w_.shared[0],
                     w_.shared[1], w_.shared[2], stream);
  launch_moe_round_bf16(out, d_acc_, static_cast<int64_t>(tokens) * H, stream);
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

  // 1. Router: unchanged kernel — ids ASCENDING per row, on device.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
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
    for (size_t i = 0; i < static_cast<size_t>(E) * 3; ++i)
      h_expert_views_pinned_[i] =
          MoeExpertView{w_.experts[i].payload, w_.experts[i].scales};
    DGPP_CUDA_OK(cudaMemcpyAsync(
        d_expert_views_, h_expert_views_pinned_,
        sizeof(MoeExpertView) * static_cast<size_t>(E) * 3,
        cudaMemcpyHostToDevice, stream));
  }

  const int slots = tokens * (K + 1);
  const int I_r = static_cast<int>(w_.experts[0].rows);  // routed inter slice
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
  launch_moe_slot_gate_up_swiglu(
      hidden, H, d_ids_, order, table, I_r, H, I_s, H, w_.shared[0].payload,
      w_.shared[0].scales, w_.shared[1].payload, w_.shared[1].scales,
      d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream);
  launch_moe_slot_down(d_slot_act_, I_r, d_ids_, order, table, H, I_r, H,
                       I_s, w_.shared[2].payload, w_.shared[2].scales,
                       d_slot_down_, H, slots, K, stream);
  launch_moe_slot_accum(out, d_slot_down_, d_weights_, tokens, H, K, stream);
}

}  // namespace dgpp
