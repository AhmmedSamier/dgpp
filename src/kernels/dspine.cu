#include "kernels/dspine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

__device__ float reduce_sum(float v, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  return shared[0];
}

__global__ void context_norm_kernel(const float* projected, const uint16_t* norm_weight,
                                    uint16_t* normalized, int hidden, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const uint16_t rounded = float_to_bf16_bits(projected[static_cast<size_t>(row) * hidden + col]);
    const float value = bf16_bits_to_float(rounded);
    sum += value * value;
  }
  __shared__ float shared[kThreads];
  const float inv = rsqrtf(reduce_sum(sum, shared) / hidden + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const uint16_t rounded = float_to_bf16_bits(projected[static_cast<size_t>(row) * hidden + col]);
    const float value = bf16_bits_to_float(rounded);
    normalized[static_cast<size_t>(row) * hidden + col] =
        float_to_bf16_bits(value * inv * bf16_bits_to_float(norm_weight[col]));
  }
}

__global__ void prepare_read_kernel(const uint16_t* x, const uint16_t* residual,
                                    uint16_t* raw, uint16_t* normalized, float* rms,
                                    int hidden, float eps, bool add_residual) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    float value = bf16_bits_to_float(x[static_cast<size_t>(row) * hidden + col]);
    if (add_residual)
      value = bf16_bits_to_float(float_to_bf16_bits(
          value + bf16_bits_to_float(residual[static_cast<size_t>(row) * hidden + col])));
    raw[static_cast<size_t>(row) * hidden + col] = float_to_bf16_bits(value);
    sum += value * value;
  }
  __shared__ float shared[kThreads];
  const float total = reduce_sum(sum, shared);
  const float root = sqrtf(fmaxf(total / hidden, 1.1754943508222875e-38f));
  if (tid == 0) rms[row] = root;
  __syncthreads();
  const float inverse = 1.0f / (root + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const float value = bf16_bits_to_float(raw[static_cast<size_t>(row) * hidden + col]);
    normalized[static_cast<size_t>(row) * hidden + col] = float_to_bf16_bits(value * inverse);
  }
}

__global__ void normalize_codes_kernel(const uint16_t* projected, const uint16_t* codebook,
                                       const int64_t* tokens, uint16_t* codes, int block_rows,
                                       int vocab, int code_size, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const size_t out = static_cast<size_t>(row) * code_size;
  if (row % block_rows == 0) {
    const int64_t token = tokens[row];
    if (token < 0 || token >= vocab) {
      for (int c = tid; c < code_size; c += blockDim.x) codes[out + c] = 0;
      return;
    }
    const float scale = sqrtf(static_cast<float>(code_size));
    for (int c = tid; c < code_size; c += blockDim.x) {
      const float z = bf16_bits_to_float(codebook[static_cast<size_t>(token) * code_size + c]);
      codes[out + c] = float_to_bf16_bits(z * scale);
    }
    return;
  }
  float sum = 0.0f;
  for (int c = tid; c < code_size; c += blockDim.x) {
    const float z = bf16_bits_to_float(projected[out + c]);
    sum += z * z;
  }
  __shared__ float shared[kThreads];
  const float norm = sqrtf(reduce_sum(sum, shared));
  const float scale = sqrtf(static_cast<float>(code_size)) / (norm + eps);
  for (int c = tid; c < code_size; c += blockDim.x) {
    const float z = bf16_bits_to_float(projected[out + c]);
    codes[out + c] = float_to_bf16_bits(z * scale);
  }
}

__global__ void feature_message_kernel(const uint16_t* packed, const uint16_t* gate_bias,
                                       uint16_t* message, int rows, int block_rows, int rank) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int prev = row - (row % block_rows == 0 ? 0 : 1);
  for (int r = tid; r < rank; r += blockDim.x) {
    const size_t row_offset = static_cast<size_t>(row) * rank;
    const size_t prev_offset = static_cast<size_t>(prev) * rank;
    const float query = bf16_bits_to_float(packed[row_offset + r]);
    const float sender = bf16_bits_to_float(packed[static_cast<size_t>(rows) * rank + prev_offset + r]);
    const float value = bf16_bits_to_float(packed[static_cast<size_t>(2) * rows * rank + prev_offset + r]);
    const float bias = bf16_bits_to_float(gate_bias[r]);
    const float gate = 1.0f / (1.0f + expf(-(query + sender + bias)));
    message[static_cast<size_t>(row) * rank + r] =
        row % block_rows == 0 ? 0 : float_to_bf16_bits(gate * value);
  }
}

__global__ void feature_finish_kernel(const uint16_t* raw, const uint16_t* delta,
                                      const float* rms, const int64_t* positions,
                                      const uint16_t* norm_weight, uint16_t* output,
                                      uint16_t* normed, int block_rows, int hidden, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int previous = row - (row % block_rows == 0 ? 0 : 1);
  const bool valid = positions[row] >= 0 && positions[previous] >= 0 && row % block_rows != 0;
  float sum = 0.0f;
  for (int col = tid; col < hidden; col += blockDim.x) {
    const size_t i = static_cast<size_t>(row) * hidden + col;
    float value = bf16_bits_to_float(raw[i]);
    if (valid) value += rms[row] * bf16_bits_to_float(delta[i]);
    const uint16_t rounded = float_to_bf16_bits(value);
    output[i] = rounded;
    const float f = bf16_bits_to_float(rounded);
    sum += f * f;
  }
  __shared__ float shared[kThreads];
  const float variance = reduce_sum(sum, shared) / hidden;
  const float inv = rsqrtf(variance + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    const size_t i = static_cast<size_t>(row) * hidden + col;
    // Match DSpine's reference boundary: RMS-normalize and round to BF16,
    // then apply the learned norm weight and round again. Keeping the first
    // multiply in FP32 changes the hidden stream passed to every later layer.
    const uint16_t normalized = float_to_bf16_bits(
        bf16_bits_to_float(output[i]) * inv);
    normed[i] = float_to_bf16_bits(
        bf16_bits_to_float(normalized) * bf16_bits_to_float(norm_weight[col]));
  }
}

__global__ void serial_table_kernel(const int32_t* candidate_ids,
                                    const float* prewrite_logits,
                                    const int64_t* anchors, const uint16_t* query,
                                    const float* rms, const uint16_t* gate_bias,
                                    const uint16_t* tg, const uint16_t* tsv,
                                    int block_rows, int rank, int topk, int vocab,
                                    float refine_weight, int32_t* indices) {
  const int n = blockIdx.x;
  const int step = n % (block_rows - 1);
  const int slot = n / (block_rows - 1);
  const int row = slot * block_rows + step + 1;
  const int tid = threadIdx.x;
  __shared__ float scores[64 * 64];
  for (int pair = tid; pair < topk * topk; pair += blockDim.x) {
    const int predecessor = pair / topk;
    const int candidate = pair % topk;
    const int32_t token = candidate_ids[n * topk + candidate];
    const int32_t sender = step == 0 ? static_cast<int32_t>(anchors[slot])
                                     : candidate_ids[(n - 1) * topk + predecessor];
    float score = prewrite_logits[n * topk + candidate];
    if (token < 0 || token >= vocab || sender < 0 || sender >= vocab) {
      scores[pair] = -INFINITY;
      continue;
    }
    float refinement = 0.0f;
    for (int r = 0; r < rank; ++r) {
      const float q = bf16_bits_to_float(query[static_cast<size_t>(row) * rank + r]);
      const int sender_lane = step == 0 ? 0 : predecessor;
      const size_t send = static_cast<size_t>(n) * topk * 2 * rank +
                          static_cast<size_t>(sender_lane) * 2 * rank;
      const float gate_sender = bf16_bits_to_float(tsv[send + r]);
      const float value_sender = bf16_bits_to_float(tsv[send + rank + r]);
      const float b = bf16_bits_to_float(gate_bias[r]);
      const float gate = 1.0f / (1.0f + expf(-(q + gate_sender + b)));
      refinement += gate * value_sender *
                    bf16_bits_to_float(tg[(static_cast<size_t>(n) * topk + candidate) * rank + r]);
    }
    scores[pair] = score + rms[row] * refine_weight * refinement;
  }
  __syncthreads();
  for (int predecessor = tid; predecessor < topk; predecessor += blockDim.x) {
    int best = 0;
    float best_score = scores[predecessor * topk];
    for (int candidate = 1; candidate < topk; ++candidate) {
      const float value = scores[predecessor * topk + candidate];
      if (value > best_score) {
        best = candidate;
        best_score = value;
      }
    }
    indices[n * topk + predecessor] = best;
  }
}

__global__ void scale_rows_kernel(const uint16_t* input, const uint16_t* scale,
                                  uint16_t* output, int64_t n, int width,
                                  float multiplier) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int col = static_cast<int>(i % width);
    const float x = bf16_bits_to_float(input[i]);
    const float s = scale ? bf16_bits_to_float(scale[col]) : 1.0f;
    output[i] = float_to_bf16_bits(x * s * multiplier);
  }
}

__global__ void gather_rows_kernel(const uint16_t* matrix, const int32_t* ids,
                                   uint16_t* output, int count, int width) {
  const int64_t n = static_cast<int64_t>(count) * width;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(i / width);
    output[i] = matrix[static_cast<size_t>(ids[row]) * width + (i % width)];
  }
}

__global__ void gather_logits_kernel(const float* logits, const int32_t* ids,
                                     float* values, int steps, int topk, int vocab) {
  const int64_t n = static_cast<int64_t>(steps) * topk;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int32_t id = ids[i];
    values[i] = logits[static_cast<size_t>(i / topk) * vocab + id];
  }
}

__global__ void pack_pair_kernel(const uint16_t* a, const uint16_t* b,
                                 uint16_t* output, int rows, int rank) {
  const int64_t n = static_cast<int64_t>(rows) * 2 * rank;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int row = static_cast<int>(i / (2 * rank));
    const int col = static_cast<int>(i % (2 * rank));
    output[i] = col < rank ? a[static_cast<size_t>(row) * rank + col]
                           : b[static_cast<size_t>(row) * rank + (col - rank)];
  }
}

__global__ void transpose_bf16_kernel(const uint16_t* input, uint16_t* output,
                                      int rows, int cols) {
  const int64_t n = static_cast<int64_t>(rows) * cols;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / cols);
    const int c = static_cast<int>(i % cols);
    output[static_cast<size_t>(c) * rows + r] = input[i];
  }
}

__global__ void sender_ids_kernel(const int32_t* candidates, const int64_t* anchors,
                                  int32_t* senders, int steps, int topk) {
  const int64_t n = static_cast<int64_t>(steps) * topk;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int step = static_cast<int>(i / topk);
    const int lane = static_cast<int>(i % topk);
    senders[i] = step == 0 ? static_cast<int32_t>(anchors[0])
                            : candidates[static_cast<size_t>(step - 1) * topk + lane];
  }
}

__global__ void prewrite_scores_kernel(const uint16_t* raw, const uint16_t* norm,
                                       const uint16_t* lm_head, const int32_t* candidates,
                                       float* scores, int steps, int topk, int hidden,
                                       int vocab, int block_rows) {
  const int n = blockIdx.x;
  const int lane = threadIdx.x;
  const int step = n / topk;
  const int row = step + 1;
  const int32_t token = candidates[n];
  float part = 0.0f;
  for (int h = lane; h < hidden; h += blockDim.x) {
    const float x = bf16_bits_to_float(raw[static_cast<size_t>(row) * hidden + h]);
    const float w = bf16_bits_to_float(norm[h]);
    const float y = bf16_bits_to_float(lm_head[static_cast<size_t>(token) * hidden + h]);
    part += x * w * y;
  }
  __shared__ float shared[kThreads];
  const float total = reduce_sum(part, shared);
  if (lane == 0) scores[n] = total;
}

__global__ void serial_chain_kernel(const int32_t* candidate_ids, const int32_t* indices,
                                    int block_rows, int topk, int32_t* drafts) {
  const int slot = blockIdx.x;
  if (threadIdx.x != 0) return;
  int previous = 0;
  for (int step = 0; step < block_rows - 1; ++step) {
    const int n = slot * (block_rows - 1) + step;
    const int selected = indices[n * topk + previous];
    drafts[n] = candidate_ids[n * topk + selected];
    previous = selected;
  }
}

}  // namespace

void dspine_prepare_read(const uint16_t* x, const uint16_t* residual, uint16_t* raw,
                         uint16_t* normalized, float* rms, int rows, int hidden,
                         float eps, bool add_residual, cudaStream_t stream) {
  if (rows <= 0) return;
  if (hidden <= 0 || eps <= 0 || !x || !raw || !normalized || !rms || (add_residual && !residual))
    throw std::invalid_argument("dspine_prepare_read: invalid shape or pointer");
  prepare_read_kernel<<<rows, kThreads, 0, stream>>>(x, residual, raw, normalized, rms,
                                                     hidden, eps, add_residual);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_context_norm_f32_bf16(const float* projected, const uint16_t* norm_weight,
                                  uint16_t* normalized, int rows, int hidden,
                                  float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (hidden <= 0 || eps <= 0 || !projected || !norm_weight || !normalized)
    throw std::invalid_argument("dspine_context_norm_f32_bf16: invalid shape or pointer");
  context_norm_kernel<<<rows, kThreads, 0, stream>>>(projected, norm_weight, normalized, hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_normalize_codes(const uint16_t* projected, const uint16_t* codebook,
                            const int64_t* tokens, uint16_t* codes, int rows,
                            int block_rows, int vocab, int code_size, float eps,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (block_rows <= 0 || vocab <= 0 || code_size <= 0 || eps <= 0 || !projected ||
      !codebook || !tokens || !codes)
    throw std::invalid_argument("dspine_normalize_codes: invalid shape or pointer");
  normalize_codes_kernel<<<rows, kThreads, 0, stream>>>(projected, codebook, tokens, codes,
                                                        block_rows, vocab, code_size, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_feature_message(const uint16_t* packed, const uint16_t* gate_bias,
                            uint16_t* message, int rows, int block_rows, int rank,
                            cudaStream_t stream) {
  if (rows <= 0) return;
  if (block_rows <= 0 || rows % block_rows || rank <= 0 || !packed || !gate_bias || !message)
    throw std::invalid_argument("dspine_feature_message: invalid shape or pointer");
  feature_message_kernel<<<rows, kThreads, 0, stream>>>(packed, gate_bias, message,
                                                        rows, block_rows, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_feature_finish(const uint16_t* raw, const uint16_t* delta,
                           const float* rms, const int64_t* positions,
                           const uint16_t* norm_weight, uint16_t* output,
                           uint16_t* normed, int rows, int block_rows, int hidden,
                           float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  if (block_rows <= 0 || rows % block_rows || hidden <= 0 || eps <= 0 || !raw || !delta ||
      !rms || !positions || !norm_weight || !output || !normed)
    throw std::invalid_argument("dspine_feature_finish: invalid shape or pointer");
  feature_finish_kernel<<<rows, kThreads, 0, stream>>>(raw, delta, rms, positions, norm_weight,
                                                       output, normed, block_rows, hidden, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_scale_rows(const uint16_t* input, const uint16_t* scale,
                       uint16_t* output, int rows, int width,
                       float multiplier, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!input || !output || width <= 0 || !std::isfinite(multiplier))
    throw std::invalid_argument("dspine_scale_rows: invalid shape or pointer");
  const int64_t n = static_cast<int64_t>(rows) * width;
  scale_rows_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      input, scale, output, n, width, multiplier);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_gather_rows(const uint16_t* matrix, const int32_t* ids,
                        uint16_t* rows_out, int count, int width,
                        cudaStream_t stream) {
  if (count <= 0) return;
  if (!matrix || !ids || !rows_out || width <= 0)
    throw std::invalid_argument("dspine_gather_rows: invalid shape or pointer");
  const int64_t n = static_cast<int64_t>(count) * width;
  gather_rows_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      matrix, ids, rows_out, count, width);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_gather_logits(const float* logits, const int32_t* ids,
                          float* values, int steps, int topk, int vocab,
                          cudaStream_t stream) {
  if (steps <= 0) return;
  if (!logits || !ids || !values || topk <= 0 || vocab <= 0)
    throw std::invalid_argument("dspine_gather_logits: invalid shape or pointer");
  const int64_t n = static_cast<int64_t>(steps) * topk;
  gather_logits_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      logits, ids, values, steps, topk, vocab);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_pack_pair(const uint16_t* a, const uint16_t* b,
                      uint16_t* packed, int rows, int rank,
                      cudaStream_t stream) {
  if (rows <= 0) return;
  if (!a || !b || !packed || rank <= 0)
    throw std::invalid_argument("dspine_pack_pair: invalid shape or pointer");
  const int64_t n = static_cast<int64_t>(rows) * 2 * rank;
  pack_pair_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      a, b, packed, rows, rank);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_transpose_bf16(const uint16_t* input, uint16_t* output,
                           int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return;
  if (!input || !output) throw std::invalid_argument("dspine_transpose_bf16: null pointer");
  const int64_t n = static_cast<int64_t>(rows) * cols;
  transpose_bf16_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      input, output, rows, cols);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_sender_ids(const int32_t* candidates, const int64_t* anchors,
                       int32_t* senders, int steps, int topk,
                       cudaStream_t stream) {
  if (steps <= 0) return;
  if (!candidates || !anchors || !senders || topk <= 0)
    throw std::invalid_argument("dspine_sender_ids: invalid shape or pointer");
  const int64_t n = static_cast<int64_t>(steps) * topk;
  sender_ids_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      candidates, anchors, senders, steps, topk);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_prewrite_scores(const uint16_t* raw, const uint16_t* norm,
                            const uint16_t* lm_head, const int32_t* candidates,
                            float* scores, int steps, int topk, int hidden,
                            int vocab, int block_rows, cudaStream_t stream) {
  if (steps <= 0) return;
  if (!raw || !norm || !lm_head || !candidates || !scores || topk <= 0 ||
      hidden <= 0 || vocab <= 0 || block_rows != steps + 1 || topk > kThreads)
    throw std::invalid_argument("dspine_prewrite_scores: invalid shape or pointer");
  prewrite_scores_kernel<<<steps * topk, kThreads, 0, stream>>>(raw, norm, lm_head, candidates,
      scores, steps, topk, hidden, vocab, block_rows);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_serial_table(const int32_t* candidate_ids, const float* prewrite_logits,
                         const int64_t* anchors, const uint16_t* query,
                         const float* rms, const uint16_t* gate_bias,
                         const uint16_t* tg, const uint16_t* tsv,
                         int slots, int block_rows, int rank, int topk,
                         int vocab, float refine_weight, int32_t* indices, cudaStream_t stream) {
  if (slots <= 0) return;
  if (block_rows < 2 || rank <= 0 || topk < 1 || topk > 64 || vocab <= 0 ||
      !candidate_ids || !prewrite_logits || !anchors || !query || !rms || !gate_bias ||
      !tg || !tsv || !indices || !std::isfinite(refine_weight) || refine_weight < 0)
    throw std::invalid_argument("dspine_serial_table: invalid shape or pointer");
  const int rows = slots * (block_rows - 1);
  serial_table_kernel<<<rows, kThreads, 0, stream>>>(candidate_ids, prewrite_logits, anchors,
                                                    query, rms, gate_bias, tg, tsv,
                                                    block_rows, rank, topk, vocab, refine_weight, indices);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dspine_serial_chain(const int32_t* candidate_ids, const int32_t* indices,
                         int slots, int block_rows, int topk, int32_t* drafts,
                         cudaStream_t stream) {
  if (slots <= 0) return;
  if (block_rows < 2 || topk < 1 || topk > 64 || !candidate_ids || !indices || !drafts)
    throw std::invalid_argument("dspine_serial_chain: invalid shape or pointer");
  serial_chain_kernel<<<slots, 32, 0, stream>>>(candidate_ids, indices, block_rows, topk, drafts);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
