#pragma once
// Host reference of the PLE layer (Q3, 2026-09-09; transformers
// Qwen4ExpTextPLELayer / Qwen4ExpTextNGramEmbedding): each stage on its
// own with the reference's rounding points (kernels/qwen_ple.hpp), and the
// whole layer end to end at world 1 (every hash head local).
#include <cstdint>
#include <vector>

namespace dgpp::qwen_ref {

// ids [n, heads] from the sequence and its two preceding ids (EOS rule).
void ple_hash_ids(const int32_t* tokens, int n, int32_t ctx_prev1, int32_t ctx_prev2,
                  int32_t eos, const int64_t* multipliers, const int64_t* head_vocab,
                  const int64_t* head_offset, int heads, int heads_per_ngram, int32_t* ids);
// out [n, heads_local * head_dim] = bf16(e4m3 x scale) of the rank's heads.
void ple_gather(const uint8_t* table, int64_t row_begin, float scale, const int32_t* ids, int n,
                int heads, int head_begin, int heads_local, int head_dim, uint16_t* out);
// gated [n, hc * H] from the normalized key / query [n, hc * H] and value [n, H].
void ple_gate(const uint16_t* key_n, const uint16_t* query_n, const uint16_t* value,
              uint16_t* gated, int n, int hc, int hidden);
// The dilated conv over un with state [C, (width-1)*dilation] (in/out),
// silu, + gv, (+ residual). out may alias residual.
void ple_conv(const uint16_t* un, const uint16_t* gv, uint16_t* state, const uint16_t* weight,
              const uint16_t* residual, uint16_t* out, int n, int channels, int width,
              int dilation);

struct PleWeights {
  const uint16_t* key_proj = nullptr;    // bf16 [hc*H, E]
  const uint16_t* value_proj = nullptr;  // bf16 [H, E]
  const uint16_t* norm_key = nullptr;    // bf16 [hc*H]
  const uint16_t* norm_query = nullptr;  // bf16 [hc*H]
  const uint16_t* norm_conv = nullptr;   // bf16 [hc*H]
  const uint16_t* conv = nullptr;        // bf16 [hc*H, width]
};

// The layer at world 1: r_out = bf16(r + ple(r, e)) for e = the gathered
// embedding [n, E]; conv_state [hc*H, (width-1)*dilation] in/out.
void ple_forward(const uint16_t* r, const uint16_t* e, const PleWeights& w, uint16_t* conv_state,
                 uint16_t* r_out, int n, int hc, int hidden, int embed_dim, int width,
                 int dilation, float eps);

}  // namespace dgpp::qwen_ref
