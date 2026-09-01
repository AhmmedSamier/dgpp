#include "kernels/bf16_gemv.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace {

// bf16 x bf16 products are exact in fp32 (8-bit x 8-bit significands), so
// the chain's only roundings are the FMA accumulations.
__device__ __forceinline__ void fma_pair(uint32_t w2, uint32_t x2, float& acc) {
  const float w0 = bf16_bits_to_float(static_cast<uint16_t>(w2 & 0xFFFFu));
  const float w1 = bf16_bits_to_float(static_cast<uint16_t>(w2 >> 16));
  const float x0 = bf16_bits_to_float(static_cast<uint16_t>(x2 & 0xFFFFu));
  const float x1 = bf16_bits_to_float(static_cast<uint16_t>(x2 >> 16));
  acc = __fmaf_rn(w0, x0, acc);
  acc = __fmaf_rn(w1, x1, acc);
}

// One warp, one bf16 weight row (k elements, 16B aligned), kRows staged
// activation rows: acc[r] = dot(w_row, sx[r]). Chunk = 8 bf16 (16 bytes).
template <int kRows>
__device__ __forceinline__ void row_dots(const uint16_t* __restrict__ w_row,
                                         const uint16_t* __restrict__ sx, int k,
                                         int lane, float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  constexpr int kChunkElems = gemv::kChunkBytes / 2;  // 8
  constexpr int kSpanElems = gemv::kWarpSpan / 2;     // 256
  const int lane_off = lane * kChunkElems;
  for (int base = 0; base < k; base += kSpanElems * gemv::kBatch) {
    uint4 w[gemv::kBatch];
#pragma unroll
    for (int b = 0; b < gemv::kBatch; ++b) {
      const int c0 = base + b * kSpanElems + lane_off;
      w[b] = (c0 < k) ? *reinterpret_cast<const uint4*>(w_row + c0)
                      : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int b = 0; b < gemv::kBatch; ++b) {
      const int c0 = base + b * kSpanElems + lane_off;
      if (c0 >= k) break;  // per lane: its later chunks are past k too
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4 xv = *reinterpret_cast<const uint4*>(
            sx + static_cast<size_t>(r) * k + c0);
        fma_pair(w[b].x, xv.x, acc[r]);
        fma_pair(w[b].y, xv.y, acc[r]);
        fma_pair(w[b].z, xv.z, acc[r]);
        fma_pair(w[b].w, xv.w, acc[r]);
      }
    }
  }
  gemv::warp_reduce<kRows>(acc);
}

template <int kRows, bool kOutF32>
__global__ void bf16_gemv_kernel(const uint16_t* __restrict__ act,
                                 size_t act_stride,
                                 const uint16_t* __restrict__ w,
                                 void* __restrict__ out, int n, int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  float acc[kRows];
  row_dots<kRows>(w + static_cast<size_t>(row) * k, sx, k, lane, acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

template <int kRows>
void launch_rows(const uint16_t* act, size_t act_stride, const uint16_t* w,
                 void* out, bool out_f32, int n, int k, cudaStream_t stream) {
  const dim3 grid((n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = gemv::smem_bytes(kRows, k);
  if (out_f32)
    bf16_gemv_kernel<kRows, true>
        <<<grid, gemv::kThreads, smem, stream>>>(act, act_stride, w, out, n, k);
  else
    bf16_gemv_kernel<kRows, false>
        <<<grid, gemv::kThreads, smem, stream>>>(act, act_stride, w, out, n, k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool bf16_gemv_accepts(const void* weight, int m, int k) {
  return m >= 1 && m <= gemv::kMaxRows && k > 0 && (k % 8) == 0 &&
         gemv::aligned16(weight) && gemv::smem_fits(m, k);
}

void launch_bf16_gemv(const uint16_t* act, size_t act_row_stride,
                      const uint16_t* weight, void* out, bool out_f32, int m,
                      int n, int k, cudaStream_t stream) {
  if (n <= 0) return;
  if (!act || !weight || !out)
    throw std::invalid_argument("bf16_gemv: null pointer");
  if (!bf16_gemv_accepts(weight, m, k))
    throw std::invalid_argument("bf16_gemv: shape outside the GEMV contract");
  if (act_row_stride < static_cast<size_t>(k))
    throw std::invalid_argument("bf16_gemv: activation stride narrower than k");
  switch (m) {
    case 1: launch_rows<1>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 2: launch_rows<2>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 3: launch_rows<3>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    case 4: launch_rows<4>(act, act_row_stride, weight, out, out_f32, n, k, stream); break;
    default: throw std::invalid_argument("bf16_gemv: m outside 1..4");
  }
}

}  // namespace dgpp
