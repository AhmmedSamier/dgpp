#pragma once
// Kernels for the Qwen3.8-Flash-Next vision tower (models/qwen/vision.hpp).
// The tower's token order is Qwen3-VL's: patches are listed merge-block
// first — for a 2x2 spatial merge, the four patches of each 32x32-pixel
// block are consecutive — which is also the order the merger concatenates
// them in, so the merger's shuffle is a free re-view of the rows.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// RGB [H,W,3] -> patches [n, patch_in] with the checkpoint's normalization,
// in merge-block order. `grid` is pixels per visual token (32), `patch` the
// tower's patch side (16); patch_in = 3 * temporal(2) * patch * patch. Both
// temporal frames of a still image are the same pixels.
void qwen_vision_patchify(const uint8_t* rgb, uint16_t* out, int width, int height, int patch,
                          int grid, cudaStream_t stream);

// Bilinear resample of the learned 48x48 position table. `index` is [n,4]
// table rows and `weight` [n,4] the tensor-product barycentric weights, both
// computed on the host from the image grid (models/qwen/vision.hpp) so every
// rank and every run agree bit for bit.
void qwen_pos_embed_gather(const uint16_t* table, const int32_t* index, const float* weight,
                           uint16_t* out, int n, int hidden, cudaStream_t stream);

// LayerNorm with affine weight/bias (the tower uses LayerNorm, not RMSNorm).
void qwen_vision_layernorm(const uint16_t* x, const uint16_t* gamma, const uint16_t* beta,
                           uint16_t* y, int rows, int dim, float eps, cudaStream_t stream);

// The tanh approximation, 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))):
// hidden_act gelu_pytorch_tanh is nn.GELU(approximate="tanh") in
// transformers. (exllamav3's port of this tower uses the erf form, which
// differs by ~1e-3 relative; a parity run against it needs that switched.)
void qwen_vision_gelu(uint16_t* x, int64_t count, cudaStream_t stream);

void qwen_vision_residual_add(uint16_t* x, const uint16_t* y, int64_t count, cudaStream_t stream);

// qkv [n, 3*hidden] (token-major, bias already applied by the GEMM) into
// head-major q/k/v [heads, n, head_dim], applying the 2-D axial RoPE to q
// and k. `cos_tab`/`sin_tab` are [n, head_dim], holding the dim/2 angles
// ([h grid, w grid] -- dim/4 frequencies each) stored twice, because the
// reference's rotate_half reads a full-head-width angle vector: the angle at
// index j turns the pair (j, j + head_dim/2).
void qwen_vision_split_rope(const uint16_t* qkv, uint16_t* q, uint16_t* k, uint16_t* v,
                            const float* cos_tab, const float* sin_tab, int n, int hidden,
                            int heads, cudaStream_t stream);

// [heads, n, head_dim] -> [n, hidden].
void qwen_vision_unhead(const uint16_t* attn, uint16_t* out, int n, int hidden, int heads,
                        cudaStream_t stream);

// Softmax over the score rows, with the 1/sqrt(head_dim) scale folded in.
void qwen_vision_softmax(const float* scores, uint16_t* probs, int rows, int n, float scale,
                         cudaStream_t stream);

// Attention output [tile, n, head_dim] (one row per head) into head-major
// attn at rows [first, first+tile).
void qwen_vision_store_attention(const float* tile, uint16_t* attn, int tile_rows, int n,
                                 int hidden, int heads, int first, cudaStream_t stream);

// Image rows into the language model's hyper-connection embedding: every
// row is written to all four branches, matching glm_embed_bcast_streams'
// [row][branch][hidden] layout.
void qwen_image_broadcast(const uint16_t* rows, uint16_t* streams, int n, int hidden,
                          cudaStream_t stream);

// The same rows for a single-copy buffer (MTP's shifted embedding).
void qwen_image_copy(const uint16_t* rows, uint16_t* dst, int n, int hidden, cudaStream_t stream);
}  // namespace dgpp
