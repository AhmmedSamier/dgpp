#pragma once
// W4A4 NVFP4 grouped MoE GEMM on sm_121a's block-scaled tensor cores
// (mma kind::mxf4nvf4, OMMA.SF.16864). See moe_w4a4.cu.
#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels/glm_moe_launch.hpp"

namespace dgpp {

// The activation scale buffer's row stride in bytes: k/16 rounded up to 8.
size_t nvfp4_act_scale_stride(int k);

// bf16 rows [rows, k] (stride x_stride elements) -> NVFP4: codes [rows, k/2]
// (e2m1 pairs, low nibble first), scales [rows, nvfp4_act_scale_stride(k)]
// (e4m3, one per 16), gs[rows] (fp32 per-row global). x ~= e2m1 * s * gs.
// A static global (the checkpoint's input_scale; over-range values clip) is
// *static_gs_dev when that pointer is given, else static_gs; a global <= 0
// means a dynamic amax / (6 * 448) per row.
void launch_quantize_rows_nvfp4(const uint16_t* x, size_t x_stride, int rows, int k, uint8_t* codes,
                                uint8_t* scales, float* gs, cudaStream_t stream, float static_gs = 0.f,
                                const float* static_gs_dev = nullptr);
// The MoE activation fused in (moe_swiglu_clamp's math and roundings): gate/up rows
// [rows, stride] -> NVFP4 codes of silu(gate) * up, bitwise the swiglu -> quantize chain.
// qwen_rounding (2026-10-16): the Qwen3.5 dense MLP's SwiGLU instead — silu as
// g / (1 + exp(-g)), one bf16 rounding of the product, no clamps (bitwise the
// model's qwen35_swiglu_bf16 kernel followed by the row quantizer).
void launch_swiglu_quantize_rows_nvfp4(const uint16_t* gate, const uint16_t* up, size_t stride, int rows, int k,
                                       float limit, uint8_t* codes, uint8_t* scales, float* gs,
                                       cudaStream_t stream, float static_gs,
                                       const float* static_gs_dev = nullptr, bool qwen_rounding = false);

// out[seg.row0 + m, :n] = act(row) . W(expert, which)^T for every segment,
// act(row) the quantized row act_rows[seg.row0 + m] (or seg.row0 + m when
// act_rows is null). Same segment/view/out conventions as
// launch_moe_grouped_mma_fp4_{bf16,f32}; k % 64 == 0.
void launch_moe_grouped_w4a4_bf16(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                  const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                  const MoeExpertView* views, int which, uint16_t* out, size_t out_stride, int n,
                                  int k, cudaStream_t stream);
void launch_moe_grouped_w4a4_f32(const uint8_t* codes, const uint8_t* scales, const float* gs,
                                 const int32_t* act_rows, const MoeSegment* segs, int n_segs, int max_rows,
                                 const MoeExpertView* views, int which, float* out, size_t out_stride, int n, int k,
                                 cudaStream_t stream);

// The dense W4A4 form staged by TMA (2026-10-16): 128 x 256 x 128 (fp4 k) a
// stage — the codes by two SWIZZLE_64B boxes (64-byte rows), the per-row
// 8-byte scale words by one 8-byte cp.async a thread — three stages, eight
// 64 x 64 warp tiles, m-tiles fastest so the CTAs in flight share a weight
// tile from L2. Bitwise the grouped kernel over one segment (the same k64
// chain per output): 218-228 against 168-173 TFLOP/s at the Qwen3.8-27B
// MLP shapes (3.2 bytes a kFLOP on the GB10's ~800 GB/s L2-to-SM feed).
// A map covers a code matrix [rows, k/2 bytes] (k % 128 == 0, 16-byte
// aligned); box rows 128 for the activation codes, 256 for the weight codes.
bool dense_w4a4_tma_supported(int k);
CUtensorMap nvfp4_code_tma_map(const uint8_t* codes, int64_t rows, int64_t k, int box_rows);
// out[m, :n] = act(row) . W^T over the maps; a_scales [rows, a_scale_stride]
// e4m3 (the quantizer's), a_gs [rows] the per-row globals, w_scales [n, k/16]
// e4m3, w_global the matrix's F32 global (device).
void launch_dense_w4a4_tma_bf16(const CUtensorMap& amap, const CUtensorMap& wmap, const uint8_t* a_scales,
                                size_t a_scale_stride, const float* a_gs, const uint8_t* w_scales,
                                const float* w_global, uint16_t* out, size_t out_stride, int m, int n, int k,
                                cudaStream_t stream);

}  // namespace dgpp
