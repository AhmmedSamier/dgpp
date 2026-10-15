#pragma once
// The opt-in fp8 prefill GEMM (2026-09-30, engine.prefill_fp8_gemm; NOT
// bitwise the dequantized bf16 chain — docs/qwen38_autoround_int4_plan.md
// §6.15): the block-FP8 dense stack's prefill-shaped products on the fp8
// tensor cores, the reference stack's blockwise recipe (per-token 1 x 128
// e4m3 activations, the checkpoint's 128 x 128 weight scales, fp32
// promotion per k group) in place of dequantizing each matrix to bf16 for
// the cuBLASLt GEMM.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dgpp {

// The per-token 1 x 128 e4m3 activation quantizer: act [m][act_stride]
// bf16 -> q [m][k] e4m3 and scales [m][k / 128] f32 (row-major); scale =
// amax / 448 over each row's 128-wide group (1 for an all-zero group), code
// = e4m3(x / scale) rounded to nearest, saturating. k % 128 == 0; act
// 8-byte aligned with act_stride % 4 == 0.
void launch_fp8_quantize_rows(const uint16_t* act, size_t act_stride, int m, int k, uint8_t* q,
                              float* scales, cudaStream_t stream);

// out[m][n] = A8 x W8^T: a [m][k] e4m3 with the quantizer's scales; w [n][k]
// e4m3 (the checkpoint's payload) with f32 scales
// [ceil(n / w_scale_block_rows)][k / 128] (the checkpoint's 128 x 128 grid,
// or a TP slice's re-blocked rows). Per 128-wide k group the e4m3 products
// accumulate in fp32 (mma m16n8k32), then acc += a_scale[row] *
// w_scale[col] * partial. k % 128 == 0; out_stride 0 = n. The bf16 output
// is the fp32 sum rounded once (bitwise bf16(the f32 launch)).
void launch_fp8_gemm_bf16(const uint8_t* a, const float* a_scales, const uint8_t* w,
                          const float* w_scales, int w_scale_block_rows, uint16_t* out, int m, int n,
                          int k, cudaStream_t stream, size_t out_stride = 0);
void launch_fp8_gemm_f32(const uint8_t* a, const float* a_scales, const uint8_t* w,
                         const float* w_scales, int w_scale_block_rows, float* out, int m, int n,
                         int k, cudaStream_t stream, size_t out_stride = 0);

// The checkpoint's per-token recipe (engine.prefill_activations = native,
// 2026-10-15; the Qwen3.8-27B NVFP4 release's channel-FP8 projections and
// late-layer MLPs: compressed-tensors "float-quantized, channel" weights with
// "token, dynamic" activations). The quantizer: one scale per activation row
// over all of k (amax / 448; 1 for an all-zero row), codes e4m3 round-to-
// nearest saturating; act 16-byte aligned rows, act_stride % 8 == 0, k % 8.
void launch_fp8_quantize_tokens(const uint16_t* act, size_t act_stride, int m, int k, uint8_t* q,
                                float* scales, cudaStream_t stream);
// out[m][n] = (a_scale[m] * w_scale[n]) * (A8 x W8^T): the e4m3 products
// accumulate in fp32 over the whole k (mma m16n8k32) and the two scales
// apply once in the epilogue (the reference stack's per-token x per-channel
// epilogue). a [m][k] e4m3 with a_scales [m]; w [n][k] e4m3 with w_scales
// [n] (the checkpoint's channel grid as F32). k % 64 == 0; out_stride 0 = n.
void launch_fp8_gemm_channel_bf16(const uint8_t* a, const float* a_scales, const uint8_t* w,
                                  const float* w_scales, uint16_t* out, int m, int n, int k,
                                  cudaStream_t stream, size_t out_stride = 0);
void launch_fp8_gemm_channel_f32(const uint8_t* a, const float* a_scales, const uint8_t* w,
                                 const float* w_scales, float* out, int m, int n, int k,
                                 cudaStream_t stream, size_t out_stride = 0);
// The channel form staged by 2-D TMA boxes (2026-10-15): 128 x 256 x 128 a
// stage (A 16 KB + W 32 KB, SWIZZLE_128B), two stages, eight 64 x 64 warp
// tiles. Bitwise the cp.async channel form (the same k order per output);
// 139 against 100 TFLOP/s at the 27B's shapes, the L2-to-SM feed being the
// limit at ~800 GB/s (the wider tile moves 5.9 against 7.8 bytes a kFLOP).
// A map covers an e4m3 [rows, k] matrix (k % 128 == 0, 16-byte aligned);
// box rows 128 for the activation rows, 256 for the weight rows.
bool fp8_channel_tma_supported(int k);
CUtensorMap fp8_channel_tma_map(const uint8_t* base, int64_t rows, int64_t k, int box_rows);
void launch_fp8_gemm_channel_tma_bf16(const CUtensorMap& amap, const CUtensorMap& wmap, const float* a_scales,
                                      const float* w_scales, uint16_t* out, int m, int n, int k,
                                      cudaStream_t stream, size_t out_stride = 0);
void launch_fp8_gemm_channel_tma_f32(const CUtensorMap& amap, const CUtensorMap& wmap, const float* a_scales,
                                     const float* w_scales, float* out, int m, int n, int k, cudaStream_t stream,
                                     size_t out_stride = 0);

}  // namespace dgpp
