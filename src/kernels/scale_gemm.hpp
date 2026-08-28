#pragma once
// Scale-aware GEMM (DESIGN §4): consumes the resident compressed form of a
// quantized weight — E4M3 payload [N, K] plus F32 128x128 block scales —
// natively, with no BF16 materialization of the weight. The block scale is
// applied inside the weight-tile load path (decode x scale, one BF16
// round), producing weight tiles bit-identical to the transient dequant
// bridge; the MMA runs bf16 x bf16 with fp32 accumulation — the pinned
// reference numerics (dequantized weights, bf16 math), not DeepSeek-style
// dynamic activation quantization, which would change the semantics the
// M2/M3 references pinned.
//
//   D[M, N] = Act[M, K] x W[N, K]^T
//
// Tile geometry (BM=16, BN=64, BK=32) divides the 128-wide scale block
// exactly, so every stage applies ONE scalar scale (no per-element scale
// gather) and ragged N/K tails — anything not a multiple of 128 — read the
// true last block row/col with masked tile loads. Deterministic at fixed
// shapes (fixed k-loop order) and CUDA-graph capturable.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// act: bf16 row-major [M, act_row_stride_elems] (K-column slices allowed,
// matching the IGemm seam's fused-buffer views); w_payload: E4M3 row-major
// [N, K] contiguous; w_scales: F32 [ceil(N/128), ceil(K/128)] row-major;
// out: bf16 row-major [M, N].
void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream);

}  // namespace dgpp
