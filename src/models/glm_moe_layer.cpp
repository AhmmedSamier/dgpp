#include "models/glm_moe_layer.hpp"

#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {

GlmMoeLayer::GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
                         int max_tokens)
    : w_(weights), cfg_(cfg), max_tokens_(max_tokens) {
  GlmMoeConfig::validate_config(cfg_);
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmMoeLayer: max_tokens must be positive");
  if (!w_.router_gate || !w_.router_bias || !w_.experts ||
      !w_.shared[0].payload)
    throw std::invalid_argument("GlmMoeLayer: null weight pointer");

  const int M = max_tokens_;
  const size_t H = static_cast<size_t>(cfg_.hidden);
  const size_t I = static_cast<size_t>(cfg_.inter);
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_weights_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_biased_,
                                 static_cast<size_t>(M) * cfg_.n_experts * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_rows_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_row_w_, static_cast<size_t>(M) * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_gather_, M * H * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_gate_, M * I * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_up_, M * I * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_act_, M * I * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_down_, M * H * 2));
  h_counts_.assign(cfg_.n_experts, 0);
}

GlmMoeLayer::~GlmMoeLayer() {
  cudaFree(d_ids_);
  cudaFree(d_weights_);
  cudaFree(d_biased_);
  cudaFree(d_rows_);
  cudaFree(d_row_w_);
  cudaFree(d_gather_);
  cudaFree(d_gate_);
  cudaFree(d_up_);
  cudaFree(d_act_);
  cudaFree(d_down_);
}

void GlmMoeLayer::run_expert_segment(
    const uint16_t* x, const int32_t* rows_dev, const float* row_w_dev,
    uint16_t* acc, int n_rows, const GlmQuantMatrix& gate,
    const GlmQuantMatrix& up, const GlmQuantMatrix& down,
    cudaStream_t stream) {
  const int H = cfg_.hidden, I = cfg_.inter;
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
  launch_scale_gemm_bf16(d_act_, I, down.payload, down.scales, d_down_,
                         n_rows, H, I, stream);
  launch_moe_accum(acc, d_down_, rows_dev, row_w_dev, n_rows, H, stream);
}

void GlmMoeLayer::enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
                          cudaStream_t stream) {
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int H = cfg_.hidden, E = cfg_.n_experts, K = cfg_.top_k;

  // 1. Router + one sync: the ids/weights round-trip is the diagnostic
  //    mode's cost; the production path keeps segmentation device-side.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, cfg_, tokens, stream, d_biased_);
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
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));

  // 2. Segment by expert (ascending expert id — the accumulation order).
  DGPP_CUDA_OK(cudaMemsetAsync(out, 0, static_cast<size_t>(tokens) * H * 2,
                               stream));
  std::fill(h_counts_.begin(), h_counts_.end(), 0);
  h_rows_.assign(static_cast<size_t>(tokens) * K, 0);
  h_row_w_.assign(static_cast<size_t>(tokens) * K, 0.f);
  // First pass: counts per expert; second pass: stable row placement.
  for (size_t i = 0; i < h_ids_.size(); ++i) ++h_counts_[h_ids_[i]];
  std::vector<int> begin(E, 0);
  for (int e = 1; e < E; ++e) begin[e] = begin[e - 1] + h_counts_[e - 1];
  std::vector<int> fill(begin.begin(), begin.end());
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      const int e = h_ids_[static_cast<size_t>(t) * K + i];
      h_rows_[fill[e]] = t;
      h_row_w_[fill[e]] = h_weights_[static_cast<size_t>(t) * K + i];
      ++fill[e];
    }

  // 3. Per-expert segments in ascending order (skipping empty ones).
  for (int e = 0; e < E; ++e) {
    const int n = h_counts_[e];
    if (n == 0) continue;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_rows_.data() + begin[e],
                                 static_cast<size_t>(n) * 4,
                                 cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(d_row_w_, h_row_w_.data() + begin[e],
                                 static_cast<size_t>(n) * 4,
                                 cudaMemcpyHostToDevice, stream));
    run_expert_segment(hidden, d_rows_, d_row_w_, out, n,
                       w_.experts[static_cast<size_t>(e) * 3 + 0],
                       w_.experts[static_cast<size_t>(e) * 3 + 1],
                       w_.experts[static_cast<size_t>(e) * 3 + 2], stream);
  }

  // 4. Shared expert: all tokens, weight 1, added last (the reference's
  //    routed + shared single bf16 add).
  h_rows_.resize(tokens);
  for (int t = 0; t < tokens; ++t) h_rows_[t] = t;
  h_row_w_.assign(tokens, 1.0f);
  DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_rows_.data(),
                               static_cast<size_t>(tokens) * 4,
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_row_w_, h_row_w_.data(),
                               static_cast<size_t>(tokens) * 4,
                               cudaMemcpyHostToDevice, stream));
  run_expert_segment(hidden, d_rows_, d_row_w_, out, tokens, w_.shared[0],
                     w_.shared[1], w_.shared[2], stream);
}

}  // namespace dgpp
