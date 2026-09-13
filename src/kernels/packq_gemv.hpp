#pragma once
// The packed-int GEMV as a single-matrix launcher (docs/glm53_plan.md D2):
//   D[m, n] = Act[m, k] x W[n, k]^T
// with W a compressed-tensors pack-quantized matrix (int4 or int8 codes
// packed in I32 words [n, k*bits/32], one bf16 scale per 64 codes along k)
// and bf16 activations. Rows are chunked four at a time through the same
// core the MoE slot kernels use (packq_gemv.cuh), so every output row is
// bitwise invariant to m and to the launcher — the test's reference for
// the grouped and slot paths, and the oracle's subject.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// act: bf16 row-major [m, act_row_stride_elems]; out: bf16 row-major [m, n]
// (or f32 with the accumulators stored unrounded — bf16(out_f32) ==
// out_bf16 bit for bit). Contract: packq_gemv::shape_ok(w.packed, w.bits, k).
void launch_packq_gemv_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const GlmPackedMatrix& w, uint16_t* out, int m, int n,
                            int k, cudaStream_t stream);
void launch_packq_gemv_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const GlmPackedMatrix& w, float* out, int m, int n,
                           int k, cudaStream_t stream);

// True when the core accepts this matrix (k a multiple of 64 in the
// compiled set, the code width 4 or 8, a 16-byte aligned payload, k within
// the smem budget at one row).
bool packq_gemv_accepts(const GlmPackedMatrix& w);

}  // namespace dgpp
