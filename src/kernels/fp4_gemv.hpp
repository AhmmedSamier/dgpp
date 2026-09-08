#pragma once
// The NVFP4 GEMV as a single-matrix launcher (docs/nvfp4_plan.md §2.3):
//   D[m, n] = Act[m, k] x W[n, k]^T / g
// with W an NVFP4 matrix (e2m1 pairs [n, k/2], e4m3 scales [n, k/16], one
// F32 global scale g on the device) and bf16 activations. Rows are chunked
// four at a time through the same core the MoE slot kernels use
// (fp4_gemv.cuh), so every output row is bitwise invariant to m and to the
// launcher — the test's reference for the grouped and slot paths, and the
// oracle's subject.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// act: bf16 row-major [m, act_row_stride_elems]; out: bf16 row-major [m, n]
// (or f32 with the accumulators stored unrounded — bf16(out_f32) ==
// out_bf16 bit for bit). Contract: fp4_gemv::shape_ok(w.payload, k).
void launch_fp4_gemv_bf16(const uint16_t* act, size_t act_row_stride_elems,
                          const GlmFp4Matrix& w, uint16_t* out, int m, int n,
                          int k, cudaStream_t stream);
void launch_fp4_gemv_f32(const uint16_t* act, size_t act_row_stride_elems,
                         const GlmFp4Matrix& w, float* out, int m, int n,
                         int k, cudaStream_t stream);

// True when the core accepts this matrix (k % 32 == 0, k | 1024 or
// 1024 | k, 16-byte aligned payload, k within the smem budget at one row).
bool fp4_gemv_accepts(const GlmFp4Matrix& w);

}  // namespace dgpp
