#pragma once
// Host reference of the MiMo-V2.6-Flash attention pieces (2026-09-22;
// kernels/mimo_attn.hpp): the qkv finish — bf16(dot), the half-split
// partial RoPE over the 64-wide rotary slice with the reference's bf16
// ops, the value scale's second rounding — and the windowed GQA attention
// with its sink (fp32 scores, fp32 softmax with the sink as one more
// column, probabilities rounded to bf16 for the value sum, the
// denominator unrounded).
#include <cstdint>
#include <vector>

namespace dgpp::mimo_ref {

// out[192] bf16 bits: a finished q or k head. dot fp32 [192]; inv_freq
// [32]; the pair (i, i + 32) for i < 32 rotated by pos x inv_freq[i].
void qk_finish(const float* dot, const float* inv_freq, int64_t pos, uint16_t* out);
// out[128] bf16 bits: a finished v head — bf16(bf16(dot) x scale).
void v_finish(const float* dot, float value_scale, uint16_t* out);

// c_out[h * 128 + d] = attention of q[h] (bf16 [local_heads, 192]) over
// the n K rows (bf16 [n, kv_heads * 192]) / V rows (bf16 [n, kv_heads *
// 128]) of its kv head — the rows the query sees, in order — unrounded
// (the combine's C / L); the caller rounds to bf16. sink: fp32
// [local_heads] or null — one more softmax column with logit sink[h] and
// no value. `tile` > 0 models the kernel's chain exactly: the rows in
// tiles of `tile`, each tile's probabilities rounded to bf16 against the
// running max and the sums rescaled, `n_split` splits of ceil(tiles /
// n_split) tiles merged as the combine merges them (the sink in split 0's
// running pair before its first tile). tile 0: the plain softmax (every
// probability rounded against the final max, the sink in the denominator).
void attention(const uint16_t* q, const uint16_t* k_rows, const uint16_t* v_rows, int n,
               int local_heads, int kv_heads, float scale, const float* sink,
               std::vector<float>& c_out, int tile = 0, int n_split = 1);

}  // namespace dgpp::mimo_ref
