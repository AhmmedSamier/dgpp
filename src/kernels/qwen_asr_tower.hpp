#pragma once
// Kernels for the Qwen3-ASR audio tower's convolutional frontend and row
// plumbing (models/qwen_asr/tower.hpp). The transformer's linears reuse
// CublasLtGemm::matmul_linear_bf16 and its LayerNorm/GELU/residual/softmax
// reuse the vision tower's kernels; what is ASR-specific is the strided
// conv (kernel 3, stride 2, pad 1 over mel frames), the conv_out channel
// gather, the per-chunk sinusoidal add, the valid-row pack, and the
// RoPE-less token-major to head-major transpose. All elementwise math is
// fp32 with a single bf16 rounding at the store, the tower-wide numerics
// convention. The convs run as direct kernels (one thread per output,
// fp32 accumulation over the 3x3 taps) rather than im2col+GEMM: at 480
// channels the im2col cells would cost 200+ MB for no accuracy gain.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Mel [128, frames] bin-major fp32, chunk c covering frames [c*100,
// c*100+100) -> conv1 acts [(chunks*ho*wo), co] cell-first bf16
// (weight [co, 9], bias [co]).
void qwen_asr_conv_mel(const float* feats, const uint16_t* weight, const uint16_t* bias,
                       uint16_t* out, int chunks, int frames, int ho, int wo, int co,
                       cudaStream_t stream);

// Cell-first acts [chunks, hi*wi, ci] bf16 -> [(chunks*ho*wo), co] bf16
// (weight [co, ci*9], bias [co]).
void qwen_asr_conv(const uint16_t* act, const uint16_t* weight, const uint16_t* bias,
                   uint16_t* out, int chunks, int ci, int hi, int wi, int ho, int wo, int co,
                   cudaStream_t stream);

// h3 [chunks, 16*13, 480] cell-first -> conv_out rows [(chunks*13), 7680]:
// row (c, t) is the [channel, freq] plane flattened channel-major, i.e.
// dst[(c*13+t)*7680 + ch*16+fq] = h3[(c*208 + fq*13+t)*480 + ch].
void qwen_asr_conv_out_gather(const uint16_t* h3, uint16_t* rows, int chunks,
                              cudaStream_t stream);

// rows [(chunks*13), 1024] += pos[t] (the 13-row sinusoidal table),
// one fp32 add per element, one bf16 rounding at the store.
void qwen_asr_add_pos(uint16_t* rows, const uint16_t* pos, int chunks, cudaStream_t stream);

// conv rows [(chunks*13), 1024] -> packed [rows, 1024] in chunk-major order:
// chunk c contributes its first valid[c] rows at packed row base[c].
void qwen_asr_pack_rows(const uint16_t* conv_rows, uint16_t* packed, const int* valid,
                        const int* base, int chunks, cudaStream_t stream);

// Token-major [rows, heads*hd] -> head-major [heads, rows, hd]. The tower has
// no RoPE, so this is a pure transpose (the vision split_rope minus rope).
void qwen_asr_to_head_major(const uint16_t* in, uint16_t* out, int rows, int heads, int hd,
                            cudaStream_t stream);
}  // namespace dgpp
