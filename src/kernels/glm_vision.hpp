#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {
void vision_patchify(const uint8_t* rgb, uint16_t* patches, int width, int height,
                     cudaStream_t stream);
void vision_swiglu(uint16_t* gate, const uint16_t* up, int count, float limit, cudaStream_t stream);
void vision_rmsnorm(const uint16_t* x, const uint16_t* weight, uint16_t* y, int rows, int dim,
                    float eps, cudaStream_t stream);
void vision_layernorm_gelu(uint16_t* x, const uint16_t* weight, const uint16_t* bias, int rows,
                           int dim, cudaStream_t stream, uint16_t* normalized = nullptr);
// qkv is token-major [N,3,H]; packed Q/K/V are [heads,N,D].
void vision_qkv(uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v, const uint16_t* qnorm,
                const uint16_t* knorm, int n, int hidden, int heads, int grid_w, float eps,
                cudaStream_t stream);
void vision_softmax(const float* scores, uint16_t* probs, int rows, int n, int dim,
                    cudaStream_t stream);
void vision_store_attention(const float* tile, uint16_t* out, int rows, int n, int dim, int heads,
                            int first, cudaStream_t stream);
void vision_unhead(const uint16_t* heads, uint16_t* tokens, int n, int hidden, int head_count,
                   cudaStream_t stream);
void vision_merge(const uint16_t* x, uint16_t* y, int n, int hidden, cudaStream_t stream);
void vision_broadcast(const uint16_t* rows, uint16_t* streams, int n, int hidden,
                      cudaStream_t stream);
}  // namespace dgpp
