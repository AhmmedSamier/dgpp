#include "kernels/fp4_gemv.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "kernels/fp4_gemv.cuh"

namespace dgpp {
namespace {

// One block = kWarps warps x rows_per_warp(K) rows. The activation rows are
// staged bf16 in dynamic shared memory.
template <int K, int kRows, typename OutT>
__global__ void fp4_gemv_kernel(const uint16_t* __restrict__ act,
                                size_t act_stride,
                                const uint8_t* __restrict__ w,
                                const uint8_t* __restrict__ scales,
                                const float* __restrict__ global_scale,
                                OutT* __restrict__ out, int n) {
  extern __shared__ __align__(16) uint16_t sx[];
  fp4_gemv::stage_activations<kRows>(act, act_stride, K, sx);
  __syncthreads();
  fp4_gemv::block_rows<K, kRows>(w, scales, *global_scale, sx,
                                 blockIdx.x * fp4_gemv::Geom<K>::rows_per_block,
                                 n, out, static_cast<size_t>(n));
}

template <int kRows, typename OutT>
void launch_rows(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w,
                 OutT* out, int n, int k, cudaStream_t stream) {
  fp4_gemv::dispatch_k(k, [&](auto kc) {
    constexpr int K = decltype(kc)::value;
    const int rows_per_block = fp4_gemv::Geom<K>::rows_per_block;
    const dim3 grid((n + rows_per_block - 1) / rows_per_block);
    fp4_gemv_kernel<K, kRows, OutT>
        <<<grid, fp4_gemv::kThreads, fp4_gemv::smem_bytes(kRows, K), stream>>>(
            act, act_stride, w.payload, w.scales, w.global_scale, out, n);
    DGPP_CUDA_OK(cudaGetLastError());
  });
}

template <typename OutT>
void launch(const uint16_t* act, size_t act_stride, const GlmFp4Matrix& w,
            OutT* out, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0) return;
  if (!act || !w.payload || !w.scales || !w.global_scale || !out)
    throw std::invalid_argument("fp4_gemv: null pointer");
  if (w.rows < n || w.cols != k)
    throw std::invalid_argument("fp4_gemv: matrix geometry does not match n, k");
  if (!fp4_gemv::shape_ok(w.payload, k))
    throw std::invalid_argument(
        "fp4_gemv: K must be a power of two in [32, 4096] and the payload "
        "16-byte aligned");
  for (int row0 = 0; row0 < m;) {
    int rows = std::min(fp4_gemv::kMaxRows, m - row0);
    while (!gemv::smem_fits(rows, k)) --rows;
    const uint16_t* a = act + static_cast<size_t>(row0) * act_stride;
    OutT* o = out + static_cast<size_t>(row0) * n;
    switch (rows) {
      case 4: launch_rows<4, OutT>(a, act_stride, w, o, n, k, stream); break;
      case 3: launch_rows<3, OutT>(a, act_stride, w, o, n, k, stream); break;
      case 2: launch_rows<2, OutT>(a, act_stride, w, o, n, k, stream); break;
      default: launch_rows<1, OutT>(a, act_stride, w, o, n, k, stream); break;
    }
    row0 += rows;
  }
}

}  // namespace

void launch_fp4_gemv_bf16(const uint16_t* act, size_t act_row_stride_elems,
                          const GlmFp4Matrix& w, uint16_t* out, int m, int n,
                          int k, cudaStream_t stream) {
  launch<uint16_t>(act, act_row_stride_elems, w, out, m, n, k, stream);
}

void launch_fp4_gemv_f32(const uint16_t* act, size_t act_row_stride_elems,
                         const GlmFp4Matrix& w, float* out, int m, int n,
                         int k, cudaStream_t stream) {
  launch<float>(act, act_row_stride_elems, w, out, m, n, k, stream);
}

bool fp4_gemv_accepts(const GlmFp4Matrix& w) {
  return w.payload && w.scales && w.global_scale && w.cols > 0 &&
         w.cols <= fp4_gemv::kMaxK &&
         fp4_gemv::shape_ok(w.payload, static_cast<int>(w.cols));
}

}  // namespace dgpp
