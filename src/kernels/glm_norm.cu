#include "kernels/glm_norm.hpp"

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

constexpr int kBlock = 512;

// Two-rounding RMSNorm (the Glm5NextTextRMSNorm choreography): fp32 mean of
// squares, u = bf16(x * rstd), y = bf16(w * u). The intermediate bf16 round
// of u is semantics, not noise — the reference multiplies the bf16 tensor.
__global__ void glm_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                   const uint16_t* __restrict__ w,
                                   uint16_t* __restrict__ y, int dim,
                                   float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  extern __shared__ float smem[];  // [dim] staging for the second pass
  float* sx = smem;

  const uint16_t* xr = x + static_cast<size_t>(row) * dim;
  uint16_t* yr = y + static_cast<size_t>(row) * dim;

  double ssq = 0.0;
  for (int i = tid; i < dim; i += nthreads) {
    const float v = bf16_bits_to_float(xr[i]);
    sx[i] = v;
    ssq += static_cast<double>(v) * v;
  }
  __syncthreads();
  __shared__ double part[kBlock];
  part[tid] = ssq;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride && tid + stride < blockDim.x)
      part[tid] += part[tid + stride];
    __syncthreads();
  }
  const float rstd = rsqrtf(static_cast<float>(part[0] / dim) + eps);

  for (int i = tid; i < dim; i += nthreads) {
    const uint16_t u = float_to_bf16_bits(sx[i] * rstd);
    yr[i] = float_to_bf16_bits(bf16_bits_to_float(w[i]) *
                               bf16_bits_to_float(u));
  }
}

// streams[t, s, :] = embed[tokens[t], :] for the four residual streams —
// the reference's unsqueeze+expand of the embedding.
__global__ void glm_embed_bcast_kernel(const uint16_t* __restrict__ table,
                                       const int64_t* __restrict__ tokens,
                                       uint16_t* __restrict__ streams,
                                       int hidden) {
  const int t = blockIdx.x;
  const uint16_t* src =
      table + static_cast<int64_t>(tokens[t]) * hidden;
  uint16_t* dst = streams + static_cast<size_t>(t) * 4 * hidden;
  for (int h = threadIdx.x + blockIdx.y * blockDim.x; h < hidden;
       h += blockDim.x * gridDim.y) {
    const uint16_t v = src[h];
    dst[h] = v;
    dst[hidden + h] = v;
    dst[2 * hidden + h] = v;
    dst[3 * hidden + h] = v;
  }
}

}  // namespace

void glm_rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                      int dim, float eps, cudaStream_t stream) {
  if (rows <= 0) return;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  // One-time opt-in for >48 KB dynamic smem (hidden beyond 12288). The
  // request is 64 KB, not the device's 99 KB opt-in ceiling: GB10 rejects
  // larger dynamic requests for this kernel's footprint with
  // cudaErrorInvalidValue, and the tolerance fallback below would silently
  // cap us at 48 KB instead. 64 KB covers hidden <= 16384.
  static const bool attr_ok = [] {
    constexpr size_t kWant = 64ull << 10;
    const cudaError_t e = cudaFuncSetAttribute(
        glm_rmsnorm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kWant));
    if (e != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }();
  if (!attr_ok && shmem > 49152)
    throw std::runtime_error(
        "glm_rmsnorm: dynamic smem exceeds driver-accepted limit");
  cudaGetLastError();
  glm_rmsnorm_kernel<<<rows, kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x),
      static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_embed_bcast_streams(const void* embed_table, const int64_t* tokens,
                             void* streams, int num_tokens, int hidden,
                             cudaStream_t stream) {
  if (num_tokens <= 0) return;
  const int blocks_y = (hidden + kBlock - 1) / kBlock;
  dim3 grid(static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(blocks_y));
  glm_embed_bcast_kernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(embed_table), tokens,
      static_cast<uint16_t*>(streams), hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
