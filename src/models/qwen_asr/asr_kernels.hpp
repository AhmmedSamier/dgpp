#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace dgpp {
// out[i] = silu(gate[i]) * up[i]: fp32 math, one bf16 rounding at the store
// (the qwen35 dense convention).
void qwen_asr_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                     cudaStream_t stream);
}  // namespace dgpp
