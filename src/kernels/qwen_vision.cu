#include <cstdint>

#include <math_constants.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qwen_vision.hpp"

namespace dgpp {
namespace {
constexpr int B = 256;
constexpr int kWarp = 32;
constexpr int kTemporal = 2;  // the tower's temporal_patch_size

__device__ inline float bf(uint16_t v) { return bf16_bits_to_float(v); }
__device__ inline uint16_t fb(float v) { return float_to_bf16_bits(v); }

__device__ float block_sum(float v, float* smem) {
  for (int off = kWarp / 2; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (threadIdx.x % kWarp == 0) smem[threadIdx.x / kWarp] = v;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x) / kWarp; ++w) total += smem[w];
  return total;
}

__device__ float block_max(float v, float* smem) {
  for (int off = kWarp / 2; off; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  if (threadIdx.x % kWarp == 0) smem[threadIdx.x / kWarp] = v;
  __syncthreads();
  float total = -CUDART_INF_F;
  for (int w = 0; w < static_cast<int>(blockDim.x) / kWarp; ++w) total = fmaxf(total, smem[w]);
  return total;
}

// The checkpoint's image normalization is mean = std = 0.5, i.e. a linear map
// of the byte onto [-1, 1].
__device__ inline uint16_t normalize(uint8_t byte) {
  return fb(byte * (1.0f / 255.0f) * 2.0f - 1.0f);
}

// RGB [H,W,3] -> [n, 3*temporal*patch*patch] in merge-block token order:
// patch `token` sits at block (token/4 / wb, token/4 % wb) at merge offset
// (token%4)/2, token%2. Both temporal frames of a still image are the same
// pixels, so the frame index selects the same byte.
__global__ void patchify(const uint8_t* rgb, uint16_t* out, int width, int patch, int grid, int in,
                         int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t token = i / in;
  const int f = static_cast<int>(i % in);
  const int per_channel = kTemporal * patch * patch;
  const int c = f / per_channel;
  const int r = f % per_channel;
  const int ph = (r / patch) % patch;  // r runs (frame, row, column)
  const int pw = r % patch;
  const int block = static_cast<int>(token / 4), iy = (static_cast<int>(token) % 4) / 2,
              ix = static_cast<int>(token) % 2;
  const int wb = width / grid;
  const int y = (block / wb) * grid + iy * patch + ph;
  const int x = (block % wb) * grid + ix * patch + pw;
  out[i] = normalize(rgb[(static_cast<int64_t>(y) * width + x) * 3 + c]);
}

// The bilinear resample of the learned position table: four table rows per
// output row, weighted by the tensor product of the two axis fractions.
// Accumulating in fp32 and rounding once is one rounding better than the
// reference's bf16 products.
__global__ void pos_embed_gather(const uint16_t* table, const int32_t* index, const float* weight,
                                 uint16_t* out, int hidden, int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t t = i / hidden;
  const int e = static_cast<int>(i % hidden);
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k)
    sum += weight[t * 4 + k] * bf(table[static_cast<int64_t>(index[t * 4 + k]) * hidden + e]);
  out[i] = fb(sum);
}

// One block per row. The tower's widest LayerNorm is the merger's input
// (4 * hidden), which the strided pass covers for any hidden size.
__global__ void layernorm(const uint16_t* x, const uint16_t* gamma, const uint16_t* beta,
                          uint16_t* y, int dim, float eps) {
  __shared__ float smem[B / kWarp];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  float sum = 0.0f, sq = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    sum += v;
    sq += v * v;
  }
  __shared__ float s2[B / kWarp];
  sum = block_sum(sum, smem);
  __syncthreads();
  sq = block_sum(sq, s2);
  const float mean = sum / static_cast<float>(dim);
  const float var = fmaxf(sq / static_cast<float>(dim) - mean * mean, 0.0f);
  const float rstd = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    y[row * dim + i] = fb((bf(xr[i]) - mean) * rstd * bf(gamma[i]) + bf(beta[i]));
}

// hidden_act gelu_pytorch_tanh: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
__global__ void gelu(uint16_t* x, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float v = bf(x[i]);
  x[i] = fb(0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
}

__global__ void residual_add(uint16_t* x, const uint16_t* y, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  x[i] = fb(bf(x[i]) + bf(y[i]));
}

// qkv [n, 3*hidden] (token-major, bias already fused into the GEMM) into
// head-major q/k/v [heads, n, dim], with the 2-D axial RoPE on q and k. The
// reference's rotate_half convention reads a full head-width angle vector:
// the low half of the head turns by the h grid, the high half by the w grid,
// and each dimension pairs with the one half a head away.
__global__ void split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                           const float* cos_tab, const float* sin_tab, int n, int hidden, int dim,
                           int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t t = i / (3 * hidden);
  const int rem = static_cast<int>(i % (3 * hidden));
  const int part = rem / hidden;
  const int col = rem % hidden;
  const int j = col / dim, e = col % dim;
  uint16_t* dst = part == 0 ? q : part == 1 ? k : v;
  const int64_t at = (static_cast<int64_t>(j) * n + t) * dim + e;
  if (part == 2) {
    dst[at] = qkv[i];
    return;
  }
  const int half = dim / 2;
  const int pair = e < half ? e + half : e - half;
  const float self = bf(qkv[i]);
  const float other =
      bf(qkv[t * 3 * hidden + part * hidden + j * dim + pair]);
  const float sign = e < half ? -1.0f : 1.0f;
  dst[at] = fb(self * cos_tab[t * dim + e] + sign * other * sin_tab[t * dim + e]);
}

// head-major [heads, n, dim] -> token-major [n, hidden].
__global__ void unhead(const uint16_t* attn, uint16_t* out, int n, int hidden, int dim,
                       int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int64_t t = i / hidden;
  const int col = static_cast<int>(i % hidden);
  out[i] = attn[(static_cast<int64_t>(col / dim) * n + t) * dim + col % dim];
}

// One query tile of one head's P@V result [heads, tile, dim] into the
// head-major attention buffer at rows [first, first + tile).
__global__ void store_attention(const float* tile, uint16_t* attn, int first, int n, int tile_rows,
                                int dim, int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int head = static_cast<int>(i / (static_cast<int64_t>(tile_rows) * dim));
  const int r = static_cast<int>((i / dim) % tile_rows), e = static_cast<int>(i % dim);
  attn[((static_cast<int64_t>(head) * n) + first + r) * dim + e] = fb(tile[i]);
}

// Softmax over the score row with the 1/sqrt(dim) scale folded in.
__global__ void softmax(const float* scores, uint16_t* probs, int n, float scale) {
  __shared__ float smem[B / kWarp];
  const int64_t row = blockIdx.x;
  const float* s = scores + row * n;
  float max_v = -CUDART_INF_F;
  for (int i = threadIdx.x; i < n; i += blockDim.x) max_v = fmaxf(max_v, s[i]);
  max_v = block_max(max_v, smem) * scale;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(s[i] * scale - max_v);
  sum = block_sum(sum, smem);
  const float inv = 1.0f / sum;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    probs[row * n + i] = fb(expf(s[i] * scale - max_v) * inv);
}

// Image rows into the hyper-connection embedding: the [row][branch][hidden]
// layout glm_embed_bcast_streams writes, with every branch the same row.
__global__ void image_broadcast(const uint16_t* rows, uint16_t* out, int hidden, int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  out[i] = rows[(i / (4 * hidden)) * hidden + i % hidden];
}

unsigned grid_for(int64_t total) {
  return static_cast<unsigned>((total + B - 1) / B);
}
}  // namespace

void qwen_vision_patchify(const uint8_t* rgb, uint16_t* out, int width, int height, int patch,
                          int grid, cudaStream_t stream) {
  const int64_t n = static_cast<int64_t>(width / grid) * (height / grid) * 4;
  const int in = 3 * kTemporal * patch * patch;
  const int64_t total = n * in;
  patchify<<<grid_for(total), B, 0, stream>>>(rgb, out, width, patch, grid, in, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_pos_embed_gather(const uint16_t* table, const int32_t* index, const float* weight,
                           uint16_t* out, int n, int hidden, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(n) * hidden;
  pos_embed_gather<<<grid_for(total), B, 0, stream>>>(table, index, weight, out, hidden, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_layernorm(const uint16_t* x, const uint16_t* gamma, const uint16_t* beta,
                           uint16_t* y, int rows, int dim, float eps, cudaStream_t stream) {
  layernorm<<<static_cast<unsigned>(rows), B, 0, stream>>>(x, gamma, beta, y, dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_gelu(uint16_t* x, int64_t count, cudaStream_t stream) {
  if (count <= 0) return;
  gelu<<<grid_for(count), B, 0, stream>>>(x, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_residual_add(uint16_t* x, const uint16_t* y, int64_t count, cudaStream_t stream) {
  if (count <= 0) return;
  residual_add<<<grid_for(count), B, 0, stream>>>(x, y, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                            const float* cos_tab, const float* sin_tab, int n, int hidden, int heads,
                            cudaStream_t stream) {
  const int dim = hidden / heads;
  const int64_t total = static_cast<int64_t>(n) * hidden * 3;
  split_rope<<<grid_for(total), B, 0, stream>>>(qkv, q, k, v, cos_tab, sin_tab, n, hidden, dim, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_unhead(const uint16_t* attn, uint16_t* out, int n, int hidden, int heads,
                        cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(n) * hidden;
  unhead<<<grid_for(total), B, 0, stream>>>(attn, out, n, hidden, hidden / heads, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_softmax(const float* scores, uint16_t* probs, int rows, int n, float scale,
                         cudaStream_t stream) {
  softmax<<<static_cast<unsigned>(rows), B, 0, stream>>>(scores, probs, n, scale);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_vision_store_attention(const float* tile, uint16_t* attn, int tile_rows, int n, int hidden,
                                 int heads, int first, cudaStream_t stream) {
  const int dim = hidden / heads;
  const int64_t total = static_cast<int64_t>(heads) * tile_rows * dim;
  store_attention<<<grid_for(total), B, 0, stream>>>(tile, attn, first, n, tile_rows, dim, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_image_broadcast(const uint16_t* rows, uint16_t* streams, int n, int hidden,
                          cudaStream_t stream) {
  if (n <= 0) return;
  const int64_t total = static_cast<int64_t>(n) * 4 * hidden;
  image_broadcast<<<grid_for(total), B, 0, stream>>>(rows, streams, hidden, total);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_image_copy(const uint16_t* rows, uint16_t* dst, int n, int hidden, cudaStream_t stream) {
  if (n <= 0) return;
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, rows, static_cast<size_t>(n) * hidden * sizeof(uint16_t),
                               cudaMemcpyDeviceToDevice, stream));
}
}  // namespace dgpp
