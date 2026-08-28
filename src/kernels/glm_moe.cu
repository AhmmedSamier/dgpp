#include "kernels/glm_moe_launch.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kRouterThreads = 256;
constexpr int kElemThreads = 256;

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// One block per token; one thread fully owns each expert's dot (experts are
// strided by thread count), so every expert's logit has one fixed sequential
// reduction order regardless of launch shape. Thread 0 then selects top-k
// (biased score descending, ties to the lower id), sorts the k ids ascending
// — the accumulation order — and normalizes weights with per-element
// division, matching the reference's elementwise ops.
__global__ void moe_router_kernel(const uint16_t* __restrict__ hidden,
                                  const uint16_t* __restrict__ gate,
                                  const float* __restrict__ bias,
                                  int32_t* __restrict__ ids,
                                  float* __restrict__ weights, int tokens,
                                  int hidden_dim, int n_experts, int top_k,
                                  float routed_scaling_factor,
                                  int norm_topk) {
  extern __shared__ float smem[];
  float* scores = smem;            // sigmoid scores
  float* biased = smem + n_experts;

  const int token = blockIdx.x;
  if (token >= tokens) return;
  const uint16_t* x = hidden + static_cast<size_t>(token) * hidden_dim;

  for (int e = threadIdx.x; e < n_experts; e += kRouterThreads) {
    const uint16_t* w = gate + static_cast<size_t>(e) * hidden_dim;
    float dot = 0.f;
    for (int k = 0; k < hidden_dim; ++k)
      dot = __fmaf_rn(bf16_bits_to_float(x[k]), bf16_bits_to_float(w[k]),
                      dot);
    const float s = 1.0f / (1.0f + expf(-dot));
    scores[e] = s;
    biased[e] = s + bias[e];
  }
  __syncthreads();

  if (threadIdx.x != 0) return;
  int sel[16];
  float wsel[16];
  for (int r = 0; r < top_k; ++r) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < n_experts; ++e) {
      // strict >: equal biased scores keep the LOWER expert id.
      if (biased[e] > bv) {
        bv = biased[e];
        best = e;
      }
    }
    sel[r] = best;
    wsel[r] = scores[best];
    biased[best] = -INFINITY;
  }
  // Ascending expert order (insertion sort; top_k <= 16).
  for (int i = 1; i < top_k; ++i) {
    const int id = sel[i];
    const float w = wsel[i];
    int j = i - 1;
    while (j >= 0 && sel[j] > id) {
      sel[j + 1] = sel[j];
      wsel[j + 1] = wsel[j];
      --j;
    }
    sel[j + 1] = id;
    wsel[j + 1] = w;
  }
  // Normalize with per-element division (the reference's elementwise op,
  // not reciprocal-multiply), then scale.
  float denom = 0.f;
  for (int i = 0; i < top_k; ++i) denom = __fadd_rn(denom, wsel[i]);
  denom = __fadd_rn(denom, 1e-20f);
  for (int i = 0; i < top_k; ++i) {
    const float w =
        norm_topk ? __fdiv_rn(wsel[i], denom) * routed_scaling_factor
                  : wsel[i] * routed_scaling_factor;
    ids[static_cast<size_t>(token) * top_k + i] = sel[i];
    weights[static_cast<size_t>(token) * top_k + i] = w;
  }
}

__global__ void moe_swiglu_clamp_kernel(const uint16_t* __restrict__ gate,
                                        const uint16_t* __restrict__ up,
                                        uint16_t* __restrict__ out,
                                        int64_t n, float limit) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  float g = bf16_bits_to_float(gate[i]);
  float u = bf16_bits_to_float(up[i]);
  if (g > limit) g = limit;  // gate: NO lower clamp (reference asymmetry)
  u = fminf(fmaxf(u, -limit), limit);
  const uint16_t t =
      float_to_bf16_bits(g * sigmoidf_acc(g));  // rounding 1 (silu)
  out[i] =
      float_to_bf16_bits(bf16_bits_to_float(t) * u);  // rounding 2 (product)
}

__global__ void moe_gather_rows_kernel(const uint16_t* __restrict__ src,
                                       const int32_t* __restrict__ rows,
                                       uint16_t* __restrict__ dst, int64_t n,
                                       int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  dst[i] = src[static_cast<int64_t>(rows[r]) * hidden + (i - r * hidden)];
}

__global__ void moe_accum_kernel(uint16_t* __restrict__ acc,
                                 const uint16_t* __restrict__ y,
                                 const int32_t* __restrict__ rows,
                                 const float* __restrict__ row_w, int64_t n,
                                 int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  // contribution = bf16(w * y) — the reference's .to(bf16) before the add.
  const uint16_t contrib =
      float_to_bf16_bits(row_w[r] * bf16_bits_to_float(y[i]));
  uint16_t* dst = acc + static_cast<int64_t>(rows[r]) * hidden +
                  (i - r * hidden);
  // bf16 accumulation, exactly one rounding per expert add.
  *dst = float_to_bf16_bits(bf16_bits_to_float(*dst) +
                            bf16_bits_to_float(contrib));
}

void check_router_args(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights) {
  if (!hidden || !gate || !bias || !ids || !weights)
    throw std::invalid_argument("moe_router: null pointer");
}

}  // namespace

void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       const GlmMoeConfig& cfg, int tokens,
                       cudaStream_t stream) {
  GlmMoeConfig::validate_config(cfg);
  if (tokens <= 0) return;
  check_router_args(hidden, gate, bias, ids, weights);
  const size_t smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
  moe_router_kernel<<<tokens, kRouterThreads, smem, stream>>>(
      hidden, gate, bias, ids, weights, tokens, cfg.hidden, cfg.n_experts,
      cfg.top_k, cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream) {
  if (n <= 0) return;
  if (!gate || !up || !out)
    throw std::invalid_argument("moe_swiglu: null pointer");
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_swiglu_clamp_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                            stream>>>(gate, up, out, n, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!src || !rows || !dst)
    throw std::invalid_argument("moe_gather: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_gather_rows_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                           stream>>>(src, rows, dst, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_accum(uint16_t* acc, const uint16_t* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!acc || !y || !rows || !row_weights)
    throw std::invalid_argument("moe_accum: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      acc, y, rows, row_weights, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
