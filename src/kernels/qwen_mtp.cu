#include "kernels/qwen_mtp.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

__global__ void embed_gather_kernel(const uint16_t* __restrict__ embed,
                                    const int64_t* __restrict__ tokens, uint16_t* __restrict__ out,
                                    int hidden) {
  const int64_t t = blockIdx.x;
  const int64_t tok = tokens[t];
  for (int d = threadIdx.x; d < hidden; d += blockDim.x)
    out[t * hidden + d] = embed[tok * hidden + d];
}

__global__ void fuse_kernel(const uint16_t* __restrict__ ein, const uint16_t* __restrict__ enc,
                            uint16_t* __restrict__ out, int64_t n, int hc, int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t width = static_cast<int64_t>(hc) * hidden;
  const int64_t t = i / width;
  const int d = static_cast<int>((i - t * width) % hidden);
  const float a = bf16_bits_to_float(ein[t * hidden + d]);
  const float b = bf16_bits_to_float(enc[i]);
  out[i] = float_to_bf16_bits(a + b);
}

__global__ void hidden_store_kernel(const uint16_t* __restrict__ r,
                                    const int32_t* __restrict__ req_ids,
                                    const int64_t* __restrict__ pos, uint16_t* __restrict__ window,
                                    int64_t req_stride, int window_rows, int width) {
  const int64_t row = blockIdx.x;
  const int64_t p = pos[row];
  if (p < 0) return;
  uint16_t* dst = window + static_cast<int64_t>(req_ids[row]) * req_stride +
                  (p % window_rows) * static_cast<int64_t>(width);
  for (int d = threadIdx.x; d < width; d += blockDim.x) dst[d] = r[row * width + d];
}

__global__ void hidden_gather_kernel(const uint16_t* __restrict__ window, int64_t req_stride,
                                     int window_rows, const int32_t* __restrict__ req_ids,
                                     const int64_t* __restrict__ pos, uint16_t* __restrict__ out,
                                     int width) {
  const int64_t row = blockIdx.x;
  const int64_t p = pos[row];
  if (p < 0) {
    for (int d = threadIdx.x; d < width; d += blockDim.x) out[row * width + d] = 0;
    return;
  }
  const uint16_t* src = window + static_cast<int64_t>(req_ids[row]) * req_stride +
                        (p % window_rows) * static_cast<int64_t>(width);
  for (int d = threadIdx.x; d < width; d += blockDim.x) out[row * width + d] = src[d];
}

}  // namespace

void qwen_mtp_embed_gather_bf16(const uint16_t* embed, const int64_t* tokens, uint16_t* out,
                                int rows, int hidden, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!embed || !tokens || !out || hidden <= 0)
    throw std::invalid_argument("qwen_mtp_embed_gather: bad arguments");
  embed_gather_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(embed, tokens, out, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_mtp_fuse_bf16(const uint16_t* ein, const uint16_t* enc, uint16_t* out, int rows, int hc,
                        int hidden, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!ein || !enc || !out || hc <= 0 || hidden <= 0)
    throw std::invalid_argument("qwen_mtp_fuse: bad arguments");
  const int64_t n = static_cast<int64_t>(rows) * hc * hidden;
  const int blocks = static_cast<int>((n + 255) / 256);
  fuse_kernel<<<blocks, 256, 0, stream>>>(ein, enc, out, n, hc, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_mtp_hidden_store_bf16(const uint16_t* r, const int32_t* req_ids, const int64_t* pos,
                                uint16_t* window, int64_t req_stride, int window_rows, int rows,
                                int width, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!r || !req_ids || !pos || !window || window_rows <= 0 || width <= 0 ||
      req_stride < static_cast<int64_t>(window_rows) * width)
    throw std::invalid_argument("qwen_mtp_hidden_store: bad arguments");
  hidden_store_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(r, req_ids, pos, window,
                                                                        req_stride, window_rows, width);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_mtp_hidden_gather_bf16(const uint16_t* window, int64_t req_stride, int window_rows,
                                 const int32_t* req_ids, const int64_t* pos, uint16_t* out, int rows,
                                 int width, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!window || !req_ids || !pos || !out || window_rows <= 0 || width <= 0 ||
      req_stride < static_cast<int64_t>(window_rows) * width)
    throw std::invalid_argument("qwen_mtp_hidden_gather: bad arguments");
  hidden_gather_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(window, req_stride, window_rows,
                                                                         req_ids, pos, out, width);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
