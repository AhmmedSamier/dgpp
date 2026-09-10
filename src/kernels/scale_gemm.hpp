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
// true last block row/col with masked tile loads. Decode shapes up to eight
// rows use row-independent GEMV chunks, keeping each row bitwise invariant
// to serving occupancy. Both paths are deterministic and graph-capturable.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "models/quant_matrix.hpp"

namespace dgpp {

// act: bf16 row-major [M, act_row_stride_elems] (K-column slices allowed,
// matching the IGemm seam's fused-buffer views); w_payload: E4M3 row-major
// [N, K] contiguous; w_scales: F32 [ceil(N/128), ceil(K/128)] row-major;
// out: bf16 row-major [M, N], or [M, out_row_stride_elems] with the product
// in its first N columns when the stride is given (0: N) — a projection
// written into a column range of a wider buffer (the DSA layer's fused
// [q_a | kv_a] output from two fp8 pairs, 2026-09-08).
void launch_scale_gemm_bf16(const uint16_t* act, size_t act_row_stride_elems,
                            const uint8_t* w_payload, const float* w_scales,
                            uint16_t* out, int m, int n, int k,
                            cudaStream_t stream, size_t out_row_stride_elems = 0);

// The same product with the fp32 accumulators stored UNROUNDED: out is f32
// row-major [M, N]. bf16(out_f32[i]) == out_bf16[i] bit for bit — the two
// launchers differ only in the epilogue store. This is the MoE down
// projection's output (its partials feed an fp32 accumulation chain that
// rounds to bf16 once, at the end — see models/glm_moe_layer.hpp).
void launch_scale_gemm_f32(const uint16_t* act, size_t act_row_stride_elems,
                           const uint8_t* w_payload, const float* w_scales,
                           float* out, int m, int n, int k,
                           cudaStream_t stream, size_t out_row_stride_elems = 0);

// The tile kernel regardless of m (the bf16 mma.sync m16n8k16 path the
// large-m route takes): the reference the grouped tensor-core MoE kernel is
// pinned bitwise against (glm_moe_test) — the routed launcher above would
// send small m to the GEMV core instead.
// rs / cs (2026-09-09): the scale grid as log2 block sizes — 7 the
// checkpoint's 128 x 128, a TP slice's re-blocked axis 6 or 5 (plan D2);
// every row reads its own scale row, a 32-deep stage its one column.
void launch_scale_gemm_tile_bf16(const uint16_t* act, size_t act_row_stride_elems,
                                 const uint8_t* w_payload, const float* w_scales,
                                 uint16_t* out, int m, int n, int k,
                                 cudaStream_t stream, int rs = 7, int cs = 7);
void launch_scale_gemm_tile_f32(const uint16_t* act, size_t act_row_stride_elems,
                                const uint8_t* w_payload, const float* w_scales,
                                float* out, int m, int n, int k,
                                cudaStream_t stream, int rs = 7, int cs = 7);

}  // namespace dgpp
