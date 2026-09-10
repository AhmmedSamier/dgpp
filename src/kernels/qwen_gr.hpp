#pragma once
// The gated residual's elementwise kernels (Q3, 2026-09-09;
// docs/qwen38_flash_next_plan.md §1.5, transformers Qwen4ExpTextGatedResidual).
// The two gate matmuls (down [r, hc*H] and up [hc*H, r]) and the inject
// dots go through the GEMM/GEMV seam; these kernels are what sits between
// them, each pinned to the reference's bf16 rounding points:
//   mix:     Rn = group_rmsnorm(R)                    (qwen_norm.hpp)
//            t  = bf16(silu(bf16(bf16(W_down Rn) / hc)))     gate_act
//            G  = bf16(sigmoid(bf16(W_up t)))                 |
//            x  = bf16(mean_i bf16(G_i * Rn_i))               | mix_finish
//   combine: s_i = 2 * bf16(sigmoid(bf16(W_inj Rn)_i / hc))   |
//            R_i = bf16(R_i + bf16(y * s_i))                   | combine
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// In place on t [rows, r] bf16: t = silu(t / hc).
void qwen_gr_gate_act_bf16(void* t, int64_t rows, int r, int hc, cudaStream_t stream);

// x[rows, H] from the gate logits [rows, hc*H] and Rn [rows, hc*H].
void qwen_gr_mix_finish_bf16(const void* logits, const void* rn, void* x, int64_t rows,
                             int hc, int hidden, cudaStream_t stream);

// R[rows, hc*H] += y[rows, H] * s_i, with s from W_inj [hc, hc*H] and
// Rn [rows, hc*H] (the inject dots computed here: hc dots of hc*H per row,
// staged through `gates` — F32 [rows, hc] device scratch — between the
// dots kernel and the row-wide apply kernel, 2026-09-09).
void qwen_gr_combine_bf16(void* r_state, const void* rn, const void* w_inject,
                          const void* y, float* gates, int64_t rows, int hc, int hidden,
                          cudaStream_t stream);

// The combine's two halves as their own launches, so the dots — which read
// only Rn, and are therefore ready as soon as the mix has normalized the
// row — can be issued on a side stream at mix time and joined before the
// apply (2026-09-10, the decode profile: one block per row walking the
// hc*H-wide chain cost 21 us per site at T=1 and 23 at two rows, 96 sites
// per step, all of it on the chain between the boundary collective and the
// next mix). Same kernels, same order, same values: bitwise
// qwen_gr_combine_bf16 either way.
void qwen_gr_combine_dots_bf16(const void* rn, const void* w_inject, float* gates,
                               int64_t rows, int hc, int hidden, cudaStream_t stream);
void qwen_gr_combine_apply_bf16(void* r_state, const float* gates, const void* y,
                                int64_t rows, int hc, int hidden, cudaStream_t stream);

// The mix's two GEMVs with their producers folded into the staging
// (2026-09-09), each bitwise the two-launch chain it replaces, for the
// decode rows (rows <= 8, chunked by four): norm_down = the group norm
// (block_sum_squares' order per group; `rn` gets the normalized rows when
// given) staged into the down GEMV, t = bf16(Rn . down_w^T); act_up = the
// gate activation silu(t / hc) staged into the up GEMV, logits = bf16(act
// . up_w^T). R bf16 [rows, hc*hidden] (row stride r_stride), norm_w
// [hc*hidden], down_w [lowrank, hc*hidden], up_w [hc*hidden, lowrank].
bool qwen_gr_fused_mix_accepts(int hc, int hidden, int lowrank);
// `w_inject` and `gates` (both or neither) add the combine's inject dots to
// the same launch — one extra block reads the normalized row out of the
// staged copy every block already holds, in combine_dots_kernel's order, so
// the gates are bitwise that kernel's and the chain loses a 21 us launch
// (2026-09-10).
void qwen_gr_norm_down_bf16(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden,
                            float eps, void* rn, const void* down_w, void* t, int lowrank,
                            int64_t rows, cudaStream_t stream, const void* w_inject = nullptr,
                            float* gates = nullptr);
void qwen_gr_act_up_bf16(const void* t, int lowrank, int hc, const void* up_w, void* logits, int hidden,
                         int64_t rows, cudaStream_t stream);

}  // namespace dgpp
