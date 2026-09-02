#pragma once
// Launchers for the MoE kernels (DESIGN §7.4; semantics in
// models/glm_moe.hpp). All deterministic (fixed reduction/selection order)
// and CUDA-graph capturable: no scratch, no host reads.
#include <cuda_runtime.h>

#include "models/glm_moe.hpp"

namespace dgpp {

// Router: hidden bf16 [tokens, hidden] -> ids int32 [tokens, top_k] in
// ASCENDING expert order (the accumulation order the reference's index_add
// produces), weights f32 [tokens, top_k] normalized and scaled. Two
// kernels: per-(expert, token) dots, then per-token selection. `scores`
// and `biased` are f32 [tokens, n_experts] device buffers the caller owns:
// scores is kernel-internal scratch (the sigmoid scores the selection reads
// back), biased receives the full biased score row — the hand-off between
// the two kernels AND the exported selection inputs (near-tie
// certification needs every expert's true biased score).
void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream);

// swiglu with asymmetric clamps: gate clamp_max only, up clamp both; two
// bf16 rounding points (silu result, then the product). n = rows*inter.
void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream);

// dst[r, :] = src[rows[r], :] for n rows of width hidden.
void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream);

// acc[rows[r], c] = bf16(acc + bf16(row_w[r] * y[r, c])) — the reference's
// per-expert contribution rounding and bf16 accumulation, one expert's
// segment at a time (host loops experts in ascending order).
void launch_moe_accum(uint16_t* acc, const uint16_t* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream);

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// The decode step's MoE without host round-trips: the router leaves
// ids/weights on the device (ids ASCENDING per row — the router kernel's
// contract), the slot kernels read the route from device memory and
// early-exit foreign experts, and the accumulation reproduces the host
// path's exact op order. Slot layout: tokens*(top_k+1) slots, slot
// s = t*(K+1)+j; j<K is row t's routed expert j (ascending expert id),
// j==K is the shared expert (all rows, weight 1, accumulated last).
//
// slot_gemv: one m=1 fp8 GEMV per slot (the fp8_gemv core: warp per weight
// row, 16-byte loads). `views` is the device expert table [count, 3]
// (gate,up,down); `which` selects the matrix. Routed dims (n_routed,
// k_routed) vs the shared expert's (n_shared, k_shared — the TP-sliced
// inter differs); shared matrices arrive as args (they are host-known
// constants, not table entries). Contract: both k a multiple of 16,
// payloads 16B-aligned (the loader's). The arithmetic is exactly
// launch_scale_gemm_bf16's at m<=4 (same core) — glm_moe_test's bitwise
// gate pins the equivalence.
void launch_moe_slot_gemv(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const MoeExpertView* views, int which, int n_routed, int k_routed,
    int n_shared, int k_shared, const uint8_t* sh_payload,
    const float* sh_scales, uint16_t* out, int out_stride, int slots,
    int top_k, int begin, int count, cudaStream_t stream);

// slot_gate_up_swiglu: slot_gemv(which=0), slot_gemv(which=1) and
// swiglu_clamp in one launch — act[slot, row] = swiglu(gate_dot, up_dot)
// with the same bf16 rounding points the three-launch chain has (the gate
// and up dots are rounded to bf16 exactly where the intermediate buffers
// rounded them), so the result is bit-identical; the intermediates never
// touch memory. Gate and up share n and k (routed: the expert matrices'
// contract; shared: the caller's).
void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const MoeExpertView* views, int n_routed, int k_routed, int n_shared,
    int k_shared, const uint8_t* sh_gate_payload, const float* sh_gate_scales,
    const uint8_t* sh_up_payload, const float* sh_up_scales, uint16_t* act,
    int act_stride, int slots, int top_k, int begin, int count, float limit,
    cudaStream_t stream);

// slot_accum: per (token, element), the ordered chain
//   out = bf16( ... bf16(bf16(0) + bf16(w_j * y_j)) ... ) + shared last
// — exactly the host path's ascending-expert accumulation with the same
// per-add rounding; foreign experts contribute nothing on this rank
// (their partials arrive via the FFN all-reduce, which folds in rank
// order after).
void launch_moe_slot_accum(uint16_t* acc, const uint16_t* contrib,
                           const int32_t* ids, const float* weights,
                           int tokens, int hidden, int top_k, int begin,
                           int count, cudaStream_t stream);

}  // namespace dgpp
