#pragma once
// Launchers for the MoE kernels (DESIGN §7.4; semantics in
// models/glm_moe.hpp). All deterministic (fixed reduction/selection order)
// and CUDA-graph capturable: no scratch, no host reads.
#include <cuda_runtime.h>

#include "models/glm_moe.hpp"

namespace dgpp {

// Router: hidden bf16 [tokens, hidden] -> ids int32 [tokens, top_k] in
// ASCENDING expert order (the accumulation order the reference's index_add
// produces), weights f32 [tokens, top_k] normalized and scaled. Per-(expert,
// token) dots, then per-token selection — two kernels, or ONE when
// `counters` (an int per token, zeroed once; the kernel leaves them zero)
// is given: the last dots block of a token runs the selection, bitwise the
// two-kernel form. `scores` and `biased` are f32 [tokens, n_experts] device
// buffers the caller owns: scores is kernel-internal scratch (the sigmoid
// scores the selection reads back), biased receives the full biased score
// row — the hand-off between the two phases AND the exported selection
// inputs (near-tie certification needs every expert's true biased score).
void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream,
                       int* counters = nullptr);

// swiglu with asymmetric clamps: gate clamp_max only, up clamp both; two
// bf16 rounding points (silu result, then the product). n = rows*inter.
void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream);

// dst[r, :] = src[rows[r], :] for n rows of width hidden.
void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream);

// acc[rows[r], c] = fma(row_w[r], y[r, c], acc[rows[r], c]) — the fp32
// accumulation chain, one expert's segment at a time (the host loops
// experts in ascending order, the shared expert last with weight 1). y is
// the down projection's UNROUNDED fp32 output (launch_scale_gemm_f32).
void launch_moe_accum(float* acc, const float* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream);

// out[i] = bf16(acc[i]) — the chain's single rounding, as the sum leaves for
// the FFN all-reduce (bf16 on the wire).
void launch_moe_round_bf16(uint16_t* out, const float* acc, int64_t n,
                           cudaStream_t stream);

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// The decode step's MoE without host round-trips: the router leaves
// ids/weights on the device (ids ASCENDING per row — the router kernel's
// contract), the slot kernels read the route from device memory, and the
// accumulation reproduces the host path's exact op order. Slot layout:
// tokens*(top_k+1) slots, slot s = t*(K+1)+j; j<K is row t's routed expert
// j (ascending expert id), j==K is the shared expert (all rows, weight 1,
// accumulated last). Every expert is local: each rank holds a slice of all
// of them (gate/up rows, down columns of the intermediate dim), so `views`
// is the device expert table [n_experts, 3] (gate,up,down). Routed dims
// (n_routed, k_routed) vs the shared expert's (n_shared, k_shared); shared
// matrices arrive as args (host-known constants, not table entries).
// Contract: both k a multiple of 16, payloads 16B-aligned (the loader's).
// The arithmetic is exactly launch_scale_gemm_*'s at m<=4 (same core) —
// glm_moe_test's bitwise gate pins the equivalence.
//
// slot_gate_up_swiglu: the gate and up GEMVs and the swiglu in one launch —
// act[slot, row] = swiglu(gate_dot, up_dot) with the bf16 rounding points of
// the three-launch chain (the dots round to bf16 exactly where the
// intermediate buffers rounded them), so the result is bit-identical; the
// intermediates never touch memory. Gate and up share n and k.
//
// `order` (nullable): the slots' EXECUTION order, blockIdx.y -> logical
// slot, from launch_moe_slot_order — a multi-token batch sorted by expert
// so a shared expert's second read is an L2 hit. Null = identity. Results
// are indexed by logical slot either way (bitwise identical).
void launch_moe_slot_order(const int32_t* ids, int32_t* order, int slots,
                           int top_k, int n_experts, cudaStream_t stream);
void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream);

// slot_down: out[slot, :] = fp32 dot(down rows, act[slot]) per slot,
// UNROUNDED (the accumulation owns the single rounding). Routed slots read
// act row 0..k_routed, the shared slot 0..k_shared.
void launch_moe_slot_down(const uint16_t* act, size_t act_stride,
                          const int32_t* ids, const int32_t* order,
                          const MoeExpertView* views,
                          int n_routed, int k_routed, int n_shared,
                          int k_shared, const uint8_t* sh_payload,
                          const float* sh_scales, float* out, int out_stride,
                          int slots, int top_k, cudaStream_t stream);

// slot_accum: per (token, element), the ordered fp32 chain
//   out = bf16( fma(1, y_shared, fma(w_{K-1}, y_{K-1}, ... fma(w_0, y_0, 0))) )
// — exactly the host path's ascending-expert accumulation (launch_moe_accum
// per segment, then launch_moe_round_bf16).
void launch_moe_slot_accum(uint16_t* out, const float* contrib,
                           const float* weights, int tokens, int hidden,
                           int top_k, cudaStream_t stream);

}  // namespace dgpp
