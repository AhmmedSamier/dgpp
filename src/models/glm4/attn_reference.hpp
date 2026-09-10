#pragma once
// Host reference of the GLM-4.7 attention pieces (2026-09-09;
// kernels/glm4_attn.hpp): the qkv finish — bf16(dot + bias), the two-
// rounding head norm, the half-split partial RoPE with the reference's bf16 ops
// — and the GQA attention (fp32 scores, fp32 softmax, probabilities
// rounded to bf16 for the value sum, the denominator unrounded).
#include <cstdint>
#include <vector>

namespace dgpp::glm4_ref {

// out[dim] bf16 bits: the finished head. dot fp32 [dim]; bias bf16 [dim]
// or null; norm bf16 [dim] or null (no norm, no RoPE); inv_freq
// [rotary_dim / 2].
void qkv_finish(const float* dot, const uint16_t* bias, const uint16_t* norm, float eps,
                const float* inv_freq, int rotary_dim, int64_t pos, uint16_t* out, int dim);

// c_out[h * dim + d] = attention of q[h] (bf16 [local_heads, dim]) over the
// n K/V rows (bf16 [n, kv_heads * dim]) of its kv head, unrounded (the
// combine's C / L); the caller rounds to bf16. `tile` > 0 models the
// kernel's chain exactly: the rows in tiles of `tile`, each tile's
// probabilities rounded to bf16 against the running max and the sums
// rescaled, `n_split` splits of ceil(tiles / n_split) tiles merged as the
// combine merges them. tile 0: the plain softmax (every probability
// rounded against the final max).
void attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n,
               int local_heads, int kv_heads, int dim, float scale, std::vector<float>& c_out,
               int tile = 0, int n_split = 1);

}  // namespace dgpp::glm4_ref
