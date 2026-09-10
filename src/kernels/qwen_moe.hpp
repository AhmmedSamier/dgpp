#pragma once
// Qwen3.8-Flash-Next MoE pieces that are not the shared MoE kernels
// (kernels/glm_moe_launch.hpp carries the router — its SoftmaxTopk mode —
// the expert chain and the accumulation): the BF16 shared expert's scalar
// gate. Deterministic, capturable.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// w[t] = bf16(sigmoid(bf16(x[t] . g))) as an fp32 value: the reference's
// `sigmoid(shared_expert_gate(x))` — a bf16 Linear (one rounding of the
// logit) then a bf16 sigmoid (one rounding of the gate). x bf16
// [tokens, hidden], g bf16 [hidden]; hidden % 8 == 0, x and g 16-byte
// aligned. One warp per token; the lanes' strided fp32 partials and a
// fixed xor tree (the router dots' reduction shape).
void qwen_moe_shared_gate_bf16(const uint16_t* x, const uint16_t* g, float* w,
                               int tokens, int hidden, cudaStream_t stream);

// The shared expert's decode tail in two launches (2026-09-09), bitwise
// the seven-launch chain (the GEMM seam's bf16 GEMVs at m <= 4, the
// swiglu with no clamps, the shared gate, the accumulate's fma, the
// round): act = swiglu(x . gate_w, x . up_w) [tokens, S] and sw[t] =
// sigmoid(x[t] . g), then out = bf16(fma(sw, act . down_w, acc)). x bf16
// [tokens, H] with row stride x_stride; gate_w/up_w bf16 [S, H]; down_w
// bf16 [H, S]; acc fp32 [tokens, H] (the routed chain's); tokens <= 8
// (rows in chunks of four); H, S multiples of 8, 16-byte aligned operands.
void qwen_moe_shared_tail_decode(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                 const uint16_t* up_w, const uint16_t* down_w,
                                 const uint16_t* g, uint16_t* act, float* sw, const float* acc,
                                 uint16_t* out, int tokens, int H, int S, cudaStream_t stream);

}  // namespace dgpp
