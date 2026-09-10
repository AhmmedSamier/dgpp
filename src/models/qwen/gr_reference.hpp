#pragma once
// Host reference of the gated residual (Q3, 2026-09-09; the reference is
// transformers Qwen4ExpTextGatedResidual): each stage on its own, so a
// device kernel is checked against its exact math with the device's own
// inputs, and the whole site end to end. bf16 io; fp32 interior with the
// reference's rounding points (kernels/qwen_gr.hpp).
#include <cstdint>

namespace dgpp::qwen_ref {

// out[m, n] = bf16(sum_k act[m, k] * w[n, k]), fp32 sequential chain.
void gemv_bf16(const uint16_t* act, const uint16_t* w, uint16_t* out, int64_t m, int n, int k);
// t = bf16(silu(t / hc)) in place.
void gr_gate_act(uint16_t* t, int64_t rows, int r, int hc);
// x[rows, H] from the gate logits and Rn (both [rows, hc*H]).
void gr_mix_finish(const uint16_t* logits, const uint16_t* rn, uint16_t* x, int64_t rows, int hc,
                   int hidden);
// R += y * s with s from W_inj [hc, hc*H] and Rn.
void gr_combine(uint16_t* r_state, const uint16_t* rn, const uint16_t* w_inject,
                const uint16_t* y, int64_t rows, int hc, int hidden);
// The whole read: Rn and x from R (group norm with w_norm, the two gate
// matrices of rank r).
void gr_mix(const uint16_t* r_state, const uint16_t* w_norm, const uint16_t* w_down,
            const uint16_t* w_up, uint16_t* rn, uint16_t* x, int64_t rows, int hc, int hidden,
            int rank, float eps);

}  // namespace dgpp::qwen_ref
