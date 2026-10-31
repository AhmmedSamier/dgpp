#pragma once
// Host fp32 oracle for the Nemotron-3.5-Transcribe perception module
// (docs/nemotron_transcribe_plan.md milestone 5): the frozen mel frontend,
// the 32-layer ASR TransformerEncoder, the ParallelExpertEncoder fusion and
// the connector — every formula transcribed from NeMo
// (nemo/collections/asr/parts/preprocessing/features.py,
// modules/transformer_encoder.py, parts/submodules/subsampling.py,
// parts/submodules/multi_head_attention.py,
// modules/parallel_expert_encoder.py) and pinned to it in fp32 by
// tests/python/nemotron_transcribe_perception_oracle.py (the real NeMo
// classes, CPU, eval mode). The Sortformer branch is pinned separately
// (milestone 5b): the fusion here consumes its activity as an input.
//
// Contract notes, all settled against the NeMo source (2026-10-31):
//  - Frontend: dither is training-only (eval is deterministic even though
//    the config ships 1e-5); preemph 0.97 (the config omits it, so the
//    NeMo default holds — it is APPLIED, on the unpadded signal, before
//    the center-padded framing); the shipped `featurizer.window` [400]
//    (a symmetric Hann) and `featurizer.fb` [1, 128, 257] override the
//    computed buffers and are the truth; frames from center-padded STFT:
//    T = floor(N / 160) + 1; magnitude = sqrt(re^2 + im^2) then pow 2
//    (mag_power 2.0 — the square of the rounded sqrt, not the raw sum);
//    log(x + 2^-24) (log_zero_guard 'add', the default).
//  - Encoder: FeatureStacking pads the 1024-wide rows to a multiple of 8
//    frames with zeros and stacks frame-major (row[f * 128 + c]); the
//    stacking Linear has no bias. Blocks are pre-norm (LayerNorm eps 1e-5,
//    affine), attention is FUSED w_qkv [3D, D] split token-major into
//    (q, k, v) per head, per-head LayerNorm on q and k BEFORE the half
//    RoPE (rotary_fraction 0.5 rotates lanes [0, 64) of the 128-lane head
//    in pairs (i, i + 32) with duplicated cos/sin, lanes 64.. pass
//    through), bidirectional softmax at 1/sqrt(128), biased out_proj;
//    FFN Linear -> GELU(erf) -> Linear, both biased; then final_norm.
//  - Fusion (ParallelExpertEncoder._fuse_diar_and_asr): activity rows
//    binarized at (a > speaker_activity_threshold) [thresholded mode],
//    LayerNorm(n_spk), times the BUILT-IN sinusoidal kernel (8 x 1280:
//    column 2j is sin(s * 10000^(-2j/1280)), column 2j+1 is its cos —
//    note sin on the EVEN columns), plus LayerNorm(d_model) of the ASR
//    states, times spk_kernel_scale on the infusion only (1.0); then the
//    connector perception.proj [2688, 1280] + bias.
//
// Everything is fp32, unfused, fixed-order (-ffp-contract=off): the CUDA
// kernels of milestone 6 are gated against these numbers, and the shipped
// BF16 weights arrive through nemotron_ref::Tensor's dequant-at-use path.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "models/nemotron_transcribe/config.hpp"
#include "models/nemotron_transcribe/reference.hpp"  // nemotron_ref::Tensor/Weights/gemv

namespace dgpp {
namespace nemotron_perc {

using nemotron_ref::Tensor;
using nemotron_ref::Weights;

// One NeMo TransformerEncoder branch: the ASR stack (q/k LayerNorms, half
// rope) and the Sortformer stack (no q/k norms, FULL rope) share every
// formula and differ only here.
struct EncSpec {
  size_t d_model = 0, heads = 0, layers = 0, ff_expansion = 0;
  double rope_base = 10000.0, rotary_fraction = 1.0;
  bool qk_norm = true, qkv_bias = false, pre_block_norm = true;
};
EncSpec enc_spec_asr(const NemotronTranscribePerceptionConfig& pc);
EncSpec enc_spec_diar(const NemotronTranscribePerceptionConfig& pc);

// LayerNorm row: y = (x - mean) / sqrt(var + eps) * w + b (var is the
// biased population variance, torch's convention), sequential sums.
void layernorm_row(const float* x, const Tensor& w, const Tensor& b, size_t n, float eps,
                   float* y);

// NeMo normalize_batch(mel, length, 'per_feature') at full length: per
// channel c over the T frames, (x - mean) / (sqrt(var_unbiased) + 1e-5)
// with the biased-reference mean of features.py (reference = frame 0; at
// full length the masking is a no-op, and var is the population variance
// divided by T - 1 — torch.std semantics), computed in fp32.
void normalize_per_feature(const float* mel, size_t C, size_t T, std::vector<float>& out);

// The frozen frontend: `samples` [n] float in [-1, 1] (the endpoint's PCM
// contract), `window` [400] and `fb` [1, 128, 257] dequantized from the
// shipped tensors, `preemph` 0.97 from the NeMo default (pass < 0 to
// disable). Writes mel [128, T] with T = (n + 112) / 160 + 1 (center-
// padded framing); returns T.
size_t mel_frontend(const float* samples, size_t n, const Tensor& window, const Tensor& fb,
                    float preemph, std::vector<float>& mel);

// 512-point real FFT of a windowed frame: re/im out, 257 bins. Exposed so
// the kernels' twiddle-free direct form can be pinned element-by-element.
void rfft512(const float* frame, float* re, float* im);

// FeatureStacking + the bias-free stacking Linear: mel [C, T] ->
// [T_tok, d_model] (d_model from the proj [d_model, 8*C] width),
// T_tok = ceil(T / 8); short tails pad with zero rows.
void stack_pre_encode(const NemotronTranscribePerceptionConfig& pc, const float* mel, size_t T,
                      const Tensor& proj_w /*[d_model, 8*128]*/, std::vector<float>& out);

// One encoder block on the [T, d_model] stream (bidirectional, full rows):
//   x += attn(w_qkv -> per-head LN(q,k) -> rope -> softmax -> out_proj+b)
//   x += ffn(Linear+b -> GELU(erf) -> Linear+b)
// `prefix` names the block ("perception.encoder.asr_encoder.layers.7.").
void enc_block(const EncSpec& es, const Weights& w, const std::string& prefix, float* x,
               size_t T);

// A whole TransformerEncoder branch: stack -> embed_norm -> spec.layers
// blocks -> final_norm; writes [T_tok, d_model]. `root` names the encoder
// ("perception.encoder.asr_encoder." / "…diarization_model.encoder.").
void encoder_run(const EncSpec& es, const NemotronTranscribePerceptionConfig& pc,
                 const Weights& w, const std::string& root, const float* mel, size_t T,
                 std::vector<float>& states);

// The ASR branch exactly as ParallelExpertEncoder._run_asr: the mel is
// normalized per feature FIRST (pc.asr_normalize_per_feature), then the
// encoder; writes [T_tok, enc_d_model].
void asr_encoder(const NemotronTranscribePerceptionConfig& pc, const Weights& w, const float* mel,
                 size_t T, std::vector<float>& states);

// The diarization branch (Sortformer, milestone 5b) on the SINGLE-PASS
// route (eval, T <= chunk_feat_len): frontend_encoder (the diar encoder
// over the RAW mel) -> encoder_proj (d_model -> tf_d_model + bias) ->
// sub-pixel conv1d upsample x subsampling -> relu ->
// first_hidden_to_hidden + bias -> relu -> single_hidden_to_spks + bias
// -> sigmoid -> masked length-average downsample back onto the ASR grid.
// Writes preds [T_tok, n_spk] (the probabilities the thresholded fusion
// consumes). mask_padding is the full-length no-op; the streaming
// spkcache/fifo and the tf-transformer head are not on this route.
void diar_activity(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                   const float* mel, size_t T, std::vector<float>& preds);

// The built-in fusion speaker kernel: [n_spk, d_model], sin on even
// columns, cos on odd (NeMo _build_sinusoid_position_encoding).
void build_spk_kernel(const NemotronTranscribePerceptionConfig& pc, std::vector<float>& kernel);

// The fusion + connector: `activity` [T_tok, n_spk] (the Sortformer branch's
// sigmoid scores), states [T_tok, d_model] -> rows [T_tok, output_dim]
// (the LM's audio embeddings).
void fuse_and_connect(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                      const float* states, const float* activity, size_t T_tok,
                      std::vector<float>& rows);

// The whole perception module: samples -> mel -> {ASR states (normalized
// mel), diar activity (raw mel)} -> fusion -> rows. `activity` overrides
// the diar branch (external scores); with a null activity the module runs
// Sortformer itself and reports it in PerceptionOutput::activity.
struct PerceptionOutput {
  std::vector<float> mel;       // [128, T] (raw frontend output — the diar input)
  size_t frames = 0;            // T
  std::vector<float> states;    // [T_tok, d_model]
  std::vector<float> activity;  // [T_tok, n_spk] (set only for the internal branch)
  std::vector<float> rows;      // [T_tok, output_dim]
  size_t tokens = 0;            // T_tok
};
PerceptionOutput perceive(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                          const float* samples, size_t n, const float* activity);

}  // namespace nemotron_perc
}  // namespace dgpp
