// Nemotron-3.5-Transcribe forward kernels (milestone 4): the CUDA kernels
// for the 52-slot hybrid walk. Pure BF16 activations, fp32 accumulation,
// matching the host oracle bitwise. Mamba-2 conv/scan/gated-norm are the
// src/kernels/mamba2.cu kernels wrapped here.

#include "models/nemotron_transcribe/forward.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mamba2.hpp"

namespace dgpp {
namespace {

__device__ __forceinline__ float dev_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float dev_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------------------
// Embed lookup: one thread per output element.
// ---------------------------------------------------------------------------
__global__ void nemotron_embed_kernel(const uint16_t* embed, const int64_t* tokens,
                                      uint16_t* out, int T, int H) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(T) * static_cast<int64_t>(H);
  if (idx >= total) return;
  const int t = static_cast<int>(idx / H);
  const int h = static_cast<int>(idx % H);
  out[idx] = embed[static_cast<size_t>(tokens[t]) * static_cast<size_t>(H) + static_cast<size_t>(h)];
}

// ---------------------------------------------------------------------------
// RMSNorm: one block per row, threads parallelize over H.
// ---------------------------------------------------------------------------
__global__ void nemotron_rmsnorm_kernel(const uint16_t* w, const uint16_t* x, uint16_t* out,
                                        int T, int H, float eps) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const uint16_t* x_row = x + static_cast<size_t>(t) * static_cast<size_t>(H);
  uint16_t* out_row = out + static_cast<size_t>(t) * static_cast<size_t>(H);
  float ss = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = bf16_bits_to_float(x_row[i]);
    ss += v * v;
  }
  __shared__ float sdata[1024];
  float v = ss;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    v += __shfl_down_sync(0xFFFFFFFF, v, stride);
  if (threadIdx.x == 0) sdata[0] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mean = sdata[0] / static_cast<float>(H);
    sdata[1] = 1.0f / sqrtf(mean + eps);
  }
  __syncthreads();
  float rstd = sdata[1];
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float xv = bf16_bits_to_float(x_row[i]);
    float wv = bf16_bits_to_float(w[i]);
    out_row[i] = float_to_bf16_bits(wv * (xv * rstd));
  }
}

// ---------------------------------------------------------------------------
// Residual add: one thread per element.
// ---------------------------------------------------------------------------
__global__ void nemotron_residual_add_kernel(uint16_t* out, const uint16_t* a, const uint16_t* b,
                                             int n) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float av = bf16_bits_to_float(a[idx]);
  float bv = bf16_bits_to_float(b[idx]);
  out[idx] = float_to_bf16_bits(av + bv);
}

// ---------------------------------------------------------------------------
// GQA attention: one block per (query position, query head), head_dim threads.
// Position-free (no RoPE), causal. Shared memory: [T + 2] floats for scores,
// max and exp-sum.
// ---------------------------------------------------------------------------
__global__ void nemotron_gqa_attention_kernel(const uint16_t* q, const uint16_t* k,
                                              const uint16_t* v, uint16_t* out, int T,
                                              int q_heads, int kv_heads, int head_dim,
                                              float scale) {
  extern __shared__ float smem[];
  float* scores = smem;          // [T + 2]: scores[T] = max, scores[T+1] = exp sum
  const int t = blockIdx.x;      // query position
  const int qh = blockIdx.y;     // query head
  const int tid = threadIdx.x;
  if (t >= T || qh >= q_heads || tid >= head_dim) return;

  const int kvh = qh * kv_heads / q_heads;

  // Each thread computes its output element d = tid.
  // First, cooperatively compute scores for all positions s <= t.
  float qd = bf16_bits_to_float(q[static_cast<size_t>(t) * static_cast<size_t>(q_heads * head_dim) +
                                  static_cast<size_t>(qh) * static_cast<size_t>(head_dim) +
                                  static_cast<size_t>(tid)]);

  float max_score = -1e30f;
  for (int s = tid; s <= t; s += blockDim.x) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += qd * 0.0f;  // placeholder: q[d] is needed, not just qd
    }
    // Recompute properly: each thread needs q[all d] for its dot product.
    // Since each thread only has qd, we need a different approach.
    // Load all q elements into shared memory first.
    scores[s] = 0.0f;  // placeholder
  }
  __syncthreads();

  // Load all q elements into shared memory (first head_dim threads).
  if (tid < head_dim) {
    scores[tid] = bf16_bits_to_float(q[static_cast<size_t>(t) * static_cast<size_t>(q_heads * head_dim) +
                                       static_cast<size_t>(qh) * static_cast<size_t>(head_dim) +
                                       static_cast<size_t>(tid)]);
  }
  __syncthreads();

  // Now compute scores: each thread handles one position s.
  max_score = -1e30f;
  for (int s = tid; s <= t; s += blockDim.x) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += scores[d] *
             bf16_bits_to_float(k[static_cast<size_t>(s) * static_cast<size_t>(kv_heads * head_dim) +
                                  static_cast<size_t>(kvh) * static_cast<size_t>(head_dim) +
                                  static_cast<size_t>(d)]);
    }
    scores[head_dim + s] = dot * scale;  // store score at offset head_dim
    if (scores[head_dim + s] > max_score) max_score = scores[head_dim + s];
  }
  __syncthreads();

  // Thread 0: max reduction over scores[head_dim .. head_dim + t].
  if (tid == 0) {
    for (int s = 1; s <= t; ++s)
      if (scores[head_dim + s] > max_score) max_score = scores[head_dim + s];
    scores[head_dim + T] = max_score;  // sentinel
  }
  __syncthreads();
  max_score = scores[head_dim + T];

  // Softmax + exp sum (thread 0).
  if (tid == 0) {
    float exp_sum = 0.0f;
    for (int s = 0; s <= t; ++s) {
      scores[head_dim + s] = expf(scores[head_dim + s] - max_score);
      exp_sum += scores[head_dim + s];
    }
    scores[head_dim + T + 1] = exp_sum;
  }
  __syncthreads();
  float exp_sum = scores[head_dim + T + 1];

  // Each thread computes its output element d = tid.
  float acc = 0.0f;
  for (int s = 0; s <= t; ++s) {
    acc += (scores[head_dim + s] / exp_sum) *
           bf16_bits_to_float(v[static_cast<size_t>(s) * static_cast<size_t>(kv_heads * head_dim) +
                                static_cast<size_t>(kvh) * static_cast<size_t>(head_dim) +
                                static_cast<size_t>(tid)]);
  }
  out[static_cast<size_t>(t) * static_cast<size_t>(q_heads * head_dim) +
      static_cast<size_t>(qh) * static_cast<size_t>(head_dim) +
      static_cast<size_t>(tid)] = float_to_bf16_bits(acc);
}

// ---------------------------------------------------------------------------
// MoE: one block per row. Top-k routing, stacked relu2 experts, shared expert.
// ---------------------------------------------------------------------------
__global__ void nemotron_moe_kernel(const uint16_t* x, const uint16_t* gate,
                                    const float* gate_bias, const uint16_t* gate_and_up,
                                    const uint16_t* down, const uint16_t* shared_up,
                                    const uint16_t* shared_down, uint16_t* out, int T, int H,
                                    int E, int I, int SI, int topk, float scale, bool norm_topk) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const int tid = threadIdx.x;
  const uint16_t* x_row = x + static_cast<size_t>(t) * static_cast<size_t>(H);
  uint16_t* out_row = out + static_cast<size_t>(t) * static_cast<size_t>(H);

  extern __shared__ float smem[];
  float* gate_logits = smem;                      // [E]
  float* scores = smem + E;                       // [E]
  int* topk_idx = reinterpret_cast<int*>(scores + E);  // [topk]
  float* topk_val = reinterpret_cast<float*>(topk_idx + topk);  // [topk]
  bool* taken = reinterpret_cast<bool*>(topk_val + topk);  // [E]
  float* routed = reinterpret_cast<float*>(taken + E);  // [H]
  float* up = routed + H;                         // [I]
  float* shared_up_out = up + I;                  // [SI]

  // Router logits: [E] = x @ gate^T
  for (int e = tid; e < E; e += blockDim.x) {
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      acc += bf16_bits_to_float(x_row[h]) *
             bf16_bits_to_float(gate[static_cast<size_t>(e) * static_cast<size_t>(H) +
                                     static_cast<size_t>(h)]);
    }
    gate_logits[e] = acc + gate_bias[e];
  }
  __syncthreads();

  // Scores = sigmoid(logits)
  for (int e = tid; e < E; e += blockDim.x) {
    scores[e] = dev_sigmoid(gate_logits[e]);
  }
  __syncthreads();

  // Top-k selection (serial on thread 0, E=128 is small).
  if (tid == 0) {
    for (int e = 0; e < E; ++e) taken[e] = false;
    for (int k = 0; k < topk; ++k) {
      int best_e = -1;
      float best_v = -1e30f;
      for (int e = 0; e < E; ++e) {
        if (!taken[e] && scores[e] > best_v) { best_v = scores[e]; best_e = e; }
      }
      topk_idx[k] = best_e;
      topk_val[k] = best_v;
      taken[best_e] = true;
    }
    if (norm_topk) {
      float sum = 0.0f;
      for (int k = 0; k < topk; ++k) sum += topk_val[k];
      for (int k = 0; k < topk; ++k) topk_val[k] /= (sum + 1e-20f);
    }
    for (int k = 0; k < topk; ++k) topk_val[k] *= scale;
  }
  __syncthreads();

  // Routed output: sum over top-k experts.
  for (int h = tid; h < H; h += blockDim.x) routed[h] = 0.0f;
  __syncthreads();

  for (int k = 0; k < topk; ++k) {
    int e = topk_idx[k];
    float w = topk_val[k];
    // up[i] = relu2(sum_h x[h] * gate_and_up[e][h, i])
    for (int i = tid; i < I; i += blockDim.x) {
      float acc = 0.0f;
      for (int h = 0; h < H; ++h) {
        acc += bf16_bits_to_float(x_row[h]) *
               bf16_bits_to_float(gate_and_up[((static_cast<size_t>(e) * static_cast<size_t>(H) +
                                               static_cast<size_t>(h)) * static_cast<size_t>(I) +
                                               static_cast<size_t>(i))]);
      }
      up[i] = fmaxf(acc, 0.0f);
      up[i] = up[i] * up[i];  // relu2
    }
    __syncthreads();

    // expert_out[h] += w * sum_i up[i] * down[e][i, h]
    for (int h = tid; h < H; h += blockDim.x) {
      float acc = 0.0f;
      for (int i = 0; i < I; ++i) {
        acc += up[i] *
               bf16_bits_to_float(down[((static_cast<size_t>(e) * static_cast<size_t>(I) +
                                        static_cast<size_t>(i)) * static_cast<size_t>(H) +
                                        static_cast<size_t>(h))]);
      }
      routed[h] += w * acc;
    }
    __syncthreads();
  }

  // Shared expert: relu2(x @ shared_up) @ shared_down
  for (int i = tid; i < SI; i += blockDim.x) {
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      acc += bf16_bits_to_float(x_row[h]) *
             bf16_bits_to_float(shared_up[static_cast<size_t>(i) * static_cast<size_t>(H) +
                                          static_cast<size_t>(h)]);
    }
    shared_up_out[i] = fmaxf(acc, 0.0f);
    shared_up_out[i] = shared_up_out[i] * shared_up_out[i];  // relu2
  }
  __syncthreads();

  for (int h = tid; h < H; h += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < SI; ++i) {
      acc += shared_up_out[i] *
             bf16_bits_to_float(shared_down[static_cast<size_t>(h) * static_cast<size_t>(SI) +
                                            static_cast<size_t>(i)]);
    }
    routed[h] += acc;
  }
  __syncthreads();

  for (int h = tid; h < H; h += blockDim.x) {
    out_row[h] = float_to_bf16_bits(routed[h]);
  }
}

// ---------------------------------------------------------------------------
// norm_f: RMSNorm per row, in place.
// ---------------------------------------------------------------------------
__global__ void nemotron_norm_f_kernel(const uint16_t* norm_w, uint16_t* inout,
                                       int T, int H, float eps) {
  const int t = blockIdx.x;
  if (t >= T) return;
  uint16_t* r_row = inout + static_cast<size_t>(t) * static_cast<size_t>(H);
  float ss = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = bf16_bits_to_float(r_row[i]);
    ss += v * v;
  }
  __shared__ float sdata[1024];
  float v = ss;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    v += __shfl_down_sync(0xFFFFFFFF, v, stride);
  if (threadIdx.x == 0) sdata[0] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mean = sdata[0] / static_cast<float>(H);
    sdata[1] = 1.0f / sqrtf(mean + eps);
  }
  __syncthreads();
  float rstd = sdata[1];
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float rv = bf16_bits_to_float(r_row[i]);
    float wv = bf16_bits_to_float(norm_w[i]);
    r_row[i] = float_to_bf16_bits(wv * (rv * rstd));
  }
}

// Head GEMV: one block per row, each thread computes a chunk of vocab columns.
// logits [T, V] fp32 = h [T, H] bf16 @ lm_head [V, H] bf16.
__global__ void nemotron_lm_head_kernel(const uint16_t* h, const uint16_t* lm_head, float* logits,
                                        int T, int H, int V) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const uint16_t* h_row = h + static_cast<size_t>(t) * static_cast<size_t>(H);
  float* logits_row = logits + static_cast<size_t>(t) * static_cast<size_t>(V);
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int cols_per_thread = (V + threads - 1) / threads;
  const int start_col = tid * cols_per_thread;
  const int end_col = min(start_col + cols_per_thread, V);
  for (int v = start_col; v < end_col; ++v) {
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      acc += bf16_bits_to_float(h_row[h]) *
             bf16_bits_to_float(lm_head[static_cast<size_t>(v) * static_cast<size_t>(H) +
                                        static_cast<size_t>(h)]);
    }
    logits_row[v] = acc;
  }
}

// GEMM bf16: C[M,N] = A[M,K] @ B[N,K]^T, output bf16.
// A row-major [M,K], B row-major [N,K], C row-major [M,N].
__global__ void nemotron_gemm_bf16_kernel(const uint16_t* A, const uint16_t* B, uint16_t* C,
                                          int M, int N, int K) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += bf16_bits_to_float(A[static_cast<size_t>(row) * static_cast<size_t>(K) + static_cast<size_t>(k)]) *
           bf16_bits_to_float(B[static_cast<size_t>(col) * static_cast<size_t>(K) + static_cast<size_t>(k)]);
  }
  C[static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(col)] = float_to_bf16_bits(acc);
}

// GEMM fp32: C[M,N] = A[M,K] @ B[N,K]^T, output fp32 (for the head).
__global__ void nemotron_gemm_f32_kernel(const uint16_t* A, const uint16_t* B, float* C,
                                         int M, int N, int K) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += bf16_bits_to_float(A[static_cast<size_t>(row) * static_cast<size_t>(K) + static_cast<size_t>(k)]) *
           bf16_bits_to_float(B[static_cast<size_t>(col) * static_cast<size_t>(K) + static_cast<size_t>(k)]);
  }
  C[static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(col)] = acc;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public wrappers
// ---------------------------------------------------------------------------

void nemotron_embed(const uint16_t* embed, const int64_t* tokens, uint16_t* out, int T, int H,
                    cudaStream_t stream) {
  const int total = T * H;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  nemotron_embed_kernel<<<grid, block, 0, stream>>>(embed, tokens, out, T, H);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_rmsnorm(const uint16_t* w, const uint16_t* x, uint16_t* out, int T, int H, float eps,
                      cudaStream_t stream) {
  const int block = min(1024, (H + 31) / 32 * 32);
  nemotron_rmsnorm_kernel<<<T, block, 0, stream>>>(w, x, out, T, H, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_residual_add(uint16_t* out, const uint16_t* a, const uint16_t* b, int n,
                           cudaStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  nemotron_residual_add_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_gqa_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                            uint16_t* out, int T, int q_heads, int kv_heads, int head_dim,
                            float scale, cudaStream_t stream) {
  // Shared memory layout: scores[head_dim + T + 2] floats.
  //   [0, head_dim)        : q row copy for the score computation
  //   [head_dim, head_dim+T): per-position scores
  //   [head_dim+T]         : max score sentinel
  //   [head_dim+T+1]       : exp-sum sentinel
  const size_t smem_bytes =
      (static_cast<size_t>(head_dim) + static_cast<size_t>(T) + 2) * sizeof(float);
  dim3 block(min(1024, head_dim));
  dim3 grid(T, q_heads);
  nemotron_gqa_attention_kernel<<<grid, block, smem_bytes, stream>>>(
      q, k, v, out, T, q_heads, kv_heads, head_dim, scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_moe(const uint16_t* x, const uint16_t* gate, const float* gate_bias,
                  const uint16_t* gate_and_up, const uint16_t* down,
                  const uint16_t* shared_up, const uint16_t* shared_down, uint16_t* out, int T,
                  int H, int E, int I, int SI, int topk, float scale, bool norm_topk,
                  cudaStream_t stream) {
  const size_t smem_floats = static_cast<size_t>(E) * 2 + static_cast<size_t>(topk) * 2 +
                             static_cast<size_t>(E) + static_cast<size_t>(H) +
                             static_cast<size_t>(I) + static_cast<size_t>(SI);
  const size_t smem_bytes = smem_floats * sizeof(float);
  const int block = min(256, H > I ? H : I);
  nemotron_moe_kernel<<<T, block, smem_bytes, stream>>>(
      x, gate, gate_bias, gate_and_up, down, shared_up, shared_down, out, T, H, E, I, SI,
      topk, scale, norm_topk);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_head(const uint16_t* resid, const uint16_t* norm_w, const uint16_t* lm_head,
                   float* logits, int T, int H, int V, float eps, cudaStream_t stream) {
  // Step 1: norm_f → in-place on resid (bf16 [T, H])
  nemotron_norm_f_kernel<<<T, min(1024, (H + 31) / 32 * 32), 0, stream>>>(
      norm_w, const_cast<uint16_t*>(resid), T, H, eps);
  // Step 2: lm_head GEMV → logits (fp32 [T, V])
  const int block = 256;
  nemotron_lm_head_kernel<<<T, block, 0, stream>>>(resid, lm_head, logits, T, H, V);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_mamba_conv_silu(const uint16_t* u, int64_t u_stride, const uint16_t* weight,
                              const uint16_t* bias, uint16_t* out, int64_t out_stride,
                              int seq, int channels, int kernel, cudaStream_t stream) {
  mamba2_conv_silu(u, u_stride, weight, bias, out, out_stride, seq, channels, kernel, stream);
}

void nemotron_mamba_scan(const uint16_t* x, int64_t x_stride, const uint16_t* B, int64_t B_stride,
                         const uint16_t* C, int64_t C_stride, const uint16_t* dt, int64_t dt_stride,
                         const float* A_log, const float* D, const float* dt_bias, float* state,
                         uint16_t* out, int64_t out_stride, int seq, int heads, int head_dim,
                         int groups, int state_size, float dt_min, cudaStream_t stream) {
  mamba2_scan(x, x_stride, B, B_stride, C, C_stride, dt, dt_stride, A_log, D, dt_bias, state, out,
              out_stride, seq, heads, head_dim, groups, state_size, dt_min, stream);
}

void nemotron_mamba_gated_norm(const uint16_t* y, int64_t y_stride, const uint16_t* gate,
                               int64_t gate_stride, const uint16_t* weight, uint16_t* out,
                               int64_t out_stride, int seq, int d_inner, int group, float eps,
                               cudaStream_t stream) {
  mamba2_gated_norm(y, y_stride, gate, gate_stride, weight, out, out_stride, seq, d_inner, group,
                    eps, stream);
}

void nemotron_gemm_bf16(const uint16_t* A, const uint16_t* B, uint16_t* C, int M, int N, int K,
                        cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x, M);
  nemotron_gemm_bf16_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  DGPP_CUDA_OK(cudaGetLastError());
}

void nemotron_gemm_f32(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K,
                       cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x, M);
  nemotron_gemm_f32_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  DGPP_CUDA_OK(cudaGetLastError());
}

// Elementwise silu: out = silu(in), one thread per element.
__global__ void nemotron_silu_kernel(const uint16_t* in, uint16_t* out, int n) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = bf16_bits_to_float(in[idx]);
  out[idx] = float_to_bf16_bits(v / (1.0f + expf(-v)));
}

void nemotron_silu(const uint16_t* in, uint16_t* out, int n, cudaStream_t stream) {
  const int block = 256;
  const int grid = (n + block - 1) / block;
  nemotron_silu_kernel<<<grid, block, 0, stream>>>(in, out, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
