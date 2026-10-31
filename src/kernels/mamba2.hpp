#pragma once
// Mamba-2 device paths for the Nemotron-Transcribe LM walk (milestone 3 of
// docs/nemotron_transcribe_plan.md): the causal depthwise conv + SiLU over
// the xBC stream, the selective scan (prefill and single-row decode over a
// carried fp32 state — mamba_ssm_cache_dtype float32), and the grouped
// gated RMSNorm that rides before out_proj. The numerics contract is the
// host fp32 oracle, src/models/nemotron_transcribe/reference.{hpp,cpp}:
// dt = clamp(softplus(dt_raw + dt_bias), min = dt_min) and A = -exp(A_log)
// inside the scan; the conv bias and the SiLU gate after it; the gate
// BEFORE the group variance in the norm. Activations are BF16 (the device
// dtype), computed with fp32 accumulation; the recurrent state is F32
// [heads, head_dim, state_size] per slot, the conv state BF16
// [conv_dim, kernel - 1] (column 0 the oldest committed token, column
// kernel - 2 the newest). Row strides are in elements. tests/cuda/
// mamba2_test.cu gates every kernel against the oracle, and gates the
// stateful composition (conv/scan step-by-step after a resume) bitwise
// against the fused prefill.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// out[t] = silu(bias[c] + sum_k w[c, k] * u[t - (K-1) + k][c]), zero left
// of the sequence. weight is the checkpoint's [c, 1, K] flattening.
void mamba2_conv_silu(const void* u, int64_t u_stride, const void* weight, const void* bias,
                      void* out, int64_t out_stride, int seq, int channels, int kernel,
                      cudaStream_t stream);

// The decode-row form: folds the committed [channels, kernel-1] bf16
// history, emits one row, and advances the state (shift-left, append u).
void mamba2_conv_silu_step(const void* u_row, void* state, const void* weight, const void* bias,
                           void* out_row, int channels, int kernel, cudaStream_t stream);

// The selective scan, one block per head, state carried in registers and
// written back to `state` [heads, head_dim, state_size] fp32 (in AND out:
// zeroed for the cold start). `state` may be null for a stateless probe
// (the kernel then uses scratch and leaves nothing behind). Requires
// head_dim <= 256, 256 % head_dim == 0, and state_size % (256 / head_dim)
// == 0 — the real geometry (64, 128) and the fixture's small ones qualify;
// anything else throws std::invalid_argument.
void mamba2_scan(const void* x, int64_t x_stride, const void* B, int64_t B_stride, const void* C,
                 int64_t C_stride, const void* dt, int64_t dt_stride, const float* A_log,
                 const float* D, const float* dt_bias, float* state, void* out, int64_t out_stride,
                 int seq, int heads, int head_dim, int groups, int state_size, float dt_min,
                 cudaStream_t stream);

// The decode-row form of the scan: identical math with seq = 1 (the
// single-row decode path of the milestone).
void mamba2_scan_step(const void* x_row, const void* B_row, const void* C_row, const void* dt_row,
                      const float* A_log, const float* D, const float* dt_bias, float* state,
                      void* out_row, int heads, int head_dim, int groups, int state_size,
                      float dt_min, cudaStream_t stream);

// Grouped gated RMSNorm: each block owns one (row, group); t = y * silu
// (gate), out = w * t * rsqrt(mean(t^2 over the group) + eps). The block
// width is `group`, so it must be <= 1024 and a multiple of 32.
void mamba2_gated_norm(const void* y, int64_t y_stride, const void* gate, int64_t gate_stride,
                       const void* weight, void* out, int64_t out_stride, int seq, int d_inner,
                       int group, float eps, cudaStream_t stream);

}  // namespace dgpp
