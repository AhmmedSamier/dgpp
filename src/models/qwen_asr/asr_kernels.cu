// Qwen3-ASR dense SwiGLU (CUDA TU: __global__ cannot live in model.cpp).
#include "models/qwen_asr/asr_kernels.hpp"

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {
__global__ void qwen_asr_swiglu_kernel(const uint16_t* __restrict__ gate,
                                       const uint16_t* __restrict__ up, uint16_t* __restrict__ out,
                                       int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = bf16_bits_to_float(gate[i]);
  out[i] = float_to_bf16_bits(g / (1.0f + expf(-g)) * bf16_bits_to_float(up[i]));
}
}  // namespace

void qwen_asr_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                     cudaStream_t stream) {
  if (n <= 0) return;
  const int blocks = static_cast<int>((static_cast<uint64_t>(n) + 255) / 256);
  qwen_asr_swiglu_kernel<<<blocks, 256, 0, stream>>>(gate, up, out, n);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
