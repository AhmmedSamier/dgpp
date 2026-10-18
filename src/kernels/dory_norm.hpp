#pragma once
// The Dory nGPT normalizations (docs/dory_plan.md §1.2, §4): `justnorm` and
// the fused residual that uses it twice, one launch per site instead of a
// chain of elementwise passes.
//
// The arithmetic is the host reference's (models/dory/reference.cpp), op for
// op, because that reference is the only executable description of the model:
//
//   justnorm(x)           = bf16( fp32(x) * rstd ),  rstd = 1/sqrt(sum fp32(x)^2 + eps)
//   slerp_alpha(a, b, α, s) = justnorm( bf16(a·ra) + lr · (bf16(b·rb) − bf16(a·ra)) )
//
// with ra, rb the two input rows' rstds and lr = |fp32(α) · s| per column
// (α is the layer's `attn_alpha` / `mlp_alpha`, broadcast along rows). There
// are two roundings: the combine is rounded back to bf16 before the output
// row's own normalization, exactly as the reference stores it (`cast_back`).
//
// The row sum is accumulated in fp64 (the add_rmsnorm trick): every bf16
// square is exact in fp64, so the sum is the exact one and `float(sum) + eps`
// rounds once. The reference's sequential fp32 fma chain differs only in the
// rounding of the additions, which the bf16 output absorbs — pinned on random
// rows of every width the model uses by tests/cuda/dory_norm_test.cu.
//
// WIDTHS (plan §6): the attention-output justnorm spans ALL query heads
// (dim = q_size = 4096) — a per-head implementation is a plausible silent
// error; the one before the `sqk` square spans one head (dim = head_dim =
// 256); the MLP's spans the intermediate (7168).
//
// Contract: dim a multiple of 8, at most 8192; rows 16-byte aligned (row
// stride dim). Deterministic and capturable.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kDoryNormMaxDim = 8192;

// y[r, :] = bf16(fp32(x[r, :]) * rstd), rstd = 1/sqrt(sum_k fp32(x)^2 + eps).
void dory_justnorm_bf16(const uint16_t* x, uint16_t* y, int rows, int dim, float eps,
                        cudaStream_t stream);

// y[r, :] = justnorm(bf16(a·ra) + lr·(bf16(b·rb) − bf16(a·ra))).
void dory_slerp_alpha_bf16(const uint16_t* a, const uint16_t* b, const uint16_t* alpha, float scale,
                           uint16_t* y, int rows, int dim, float eps, cudaStream_t stream);

}  // namespace dgpp
