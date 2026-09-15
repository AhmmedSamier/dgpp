#pragma once
// Packed int4/int8 prefill: exact integer codes on BF16 tensor cores,
// then FP32 scaling of each 64-element group's partial dot. Weights are
// never rounded to BF16. The summation order differs from packq_gemv.
#include "kernels/glm_moe_launch.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// Short-prefill reassociation changes MTP acceptance and can reduce C1
// throughput even when accuracy gates pass. Keep the established GEMV
// path there and amortize packed tiles over bulk-prefill batches.
constexpr int kPackqMmaFromRows = 128;

// D[m,n] = Act[m,k] W[n,k]^T. K is a positive multiple of 64; packed
// weight rows are 16-byte aligned. Activations may have a padded stride.
// These explicit GEMM entry points do not change the GEMV decode launchers.
void launch_packq_gemm_bf16(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                            uint16_t* out, int m, int n, int k, cudaStream_t stream);
void launch_packq_gemm_f32(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                           float* out, int m, int n, int k, cudaStream_t stream);

// Device-resident expert segments, including empty/ragged segments.
// max_rows bounds the largest segment; act_rows optionally maps gathered
// row indices to original activation rows. Output uses segment row indices.
// Each selected view must have the supplied bit width and N/K geometry.
void launch_moe_grouped_mma_packq_bf16(const uint16_t* act, size_t act_stride,
                                       const MoeSegment* segs, int n_segs, int max_rows,
                                       const MoeExpertView* views, int which, uint16_t* out,
                                       size_t out_stride, int n, int k, int bits,
                                       cudaStream_t stream, const int32_t* act_rows = nullptr);
void launch_moe_grouped_mma_packq_f32(const uint16_t* act, size_t act_stride,
                                      const MoeSegment* segs, int n_segs, int max_rows,
                                      const MoeExpertView* views, int which, float* out,
                                      size_t out_stride, int n, int k, int bits,
                                      cudaStream_t stream, const int32_t* act_rows = nullptr);

}  // namespace dgpp
