#pragma once
// Nemotron-3.5-Transcribe-30B-A4B config (docs/nemotron_transcribe_plan.md).
// Two documents: the checkpoint root config.json (NeMo speechlm2 shape: the
// perception module, the prompt contract, the MTP block) and the HF
// `nemotron_h` LLM config (inline `llm_config`, mirrored in
// `llm_backbone/config.json`). This parser reads the root and cross-checks
// the mirrored LLM fields, so a checkpoint whose two copies disagree is a
// loud failure, not a silent geometry mix.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

// One entry of the Nemotron-H `layers_block_type` list. The slots are not
// whole transformer layers: Mamba/Attention name the mixer of a block and
// Moe the MLP that follows the group's mixers (the walk order is decided in
// the model; the census lives in the plan).
enum class NemotronSlot : unsigned char { Mamba, Attention, Moe };

struct NemotronTranscribeLmConfig {
  int hidden_size = 0;
  int vocab_size = 0;
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1, pad_token_id = -1;
  int num_layers = 0;                        // the 52-slot count
  std::vector<NemotronSlot> slots;           // layers_block_type
  std::vector<NemotronSlot> mtp_slots;       // mtp_layers_block_type
  // Attention (the 6 slots).
  int num_attention_heads = 0, num_key_value_heads = 0, head_dim = 0;
  double rope_theta = 10000.0;
  double partial_rotary_factor = 1.0;
  bool attention_bias = false;
  // Mamba-2 (the 23 slots): d_inner = num_heads * head_dim; the in_proj
  // width is 2*d_inner + 2*num_groups*state_size + num_heads (x, z, B, C,
  // dt) — the shapes pin it, not the `expand` key.
  int mamba_num_heads = 0, mamba_head_dim = 0, mamba_num_groups = 0, mamba_state_size = 0;
  int conv_kernel = 0;
  bool use_conv_bias = true, mamba_proj_bias = false;
  std::string mamba_hidden_act = "silu";
  double time_step_min = 0.001, time_step_max = 0.1, time_step_floor = 1e-4;
  // MoE (the 23 slots): stacked experts, relu2, sigmoid router with the
  // learned score-correction bias (NeMo Automodel semantics pin the group
  // policy — the oracle gate decides no-aux vs group-limited).
  int num_routed_experts = 0, num_experts_per_tok = 0;
  int moe_intermediate_size = 0, moe_shared_expert_intermediate_size = 0;
  double routed_scaling_factor = 1.0;
  int moe_n_groups = 1, moe_topk_group = 1;
  bool norm_topk_prob = true;
  bool mlp_bias = false;
  std::string mlp_hidden_act = "relu2";
  // Norms and residuals.
  double norm_eps = 1e-5;
  bool rescale_prenorm_residual = true;
  bool tie_word_embeddings = false;
  int max_position_embeddings = 0;
  // The serving contract: trained to 65 536 positions despite the 262 144
  // the checkpoint advertises (the card's context limit).
  int trained_context_limit = 65536;

  static NemotronTranscribeLmConfig parse(const minijson::Value& root);
};

struct NemotronTranscribePerceptionConfig {
  // Shared mel front end (16 kHz, 25 ms / 10 ms, n_fft 512, 128 mels from
  // the shipped filterbank tensors, natural-log domain per NeMo).
  int sample_rate = 16000;
  int fft_size = 512, window_size = 400, window_stride = 160;
  int mel_features = 128;
  double dither = 0.0;          // must be applied off at inference; see plan
  int frame_splicing = 1;
  bool log_mel = true;
  int stft_pad_amount = 256;    // center padding, from the token estimator
  // Frame stacking: `subsampling_factor` mel frames per encoder token.
  int subsampling_factor = 8;
  // ASR branch: the NeMo TransformerEncoder (prenorm, GELU FF, qk-norm,
  // RoPE on half the head dims, full attention inside a chunk).
  int enc_layers = 0, enc_d_model = 0, enc_heads = 0, enc_ff_expansion = 0;
  bool enc_qk_norm = true, enc_qkv_bias = false, enc_pre_block_norm = true;
  double enc_rope_base = 10000.0, enc_rotary_fraction = 0.5;
  // Independent-branch chunking on the encoder grid (30 s -> 375 tokens).
  double encoder_chunk_size_seconds = 30.0;
  // Diarization branch (Sortformer, milestone 5b): a second NeMo
  // TransformerEncoder with its OWN dims and rotary policy — the release's
  // config block omits `rotary_fraction`/`rope_base`, so the
  // TransformerEncoder defaults hold: FULL rope at base 10000 — and no
  // q/k norms (the 11-tensor blocks pin it). The mask head is the
  // single-head path the ParallelExpertEncoder runs
  // (frontend_encoder + forward_infer): encoder_proj (d -> tf), the
  // sub-pixel x-subsampling conv1d (k3 p1), relu -> first_hidden_to_hidden
  // -> relu -> single_hidden_to_spks -> sigmoid, then the 8-frame
  // length-masked mean downsample onto the ASR grid.
  int diar_d_model = 512, diar_heads = 8, diar_layers = 31, diar_ff_expansion = 4;
  double diar_rope_base = 10000.0, diar_rotary_fraction = 1.0;
  bool diar_qk_norm = false;
  int tf_d_model = 192;
  // The PEE normalizes the ASR branch's mel per feature
  // (`asr_normalize_type: per_feature`); the diar branch consumes the raw
  // frontend mel (`diar_normalize_type: "NA"` — normalize_batch's no-op).
  bool asr_normalize_per_feature = true;
  // Speaker branch (Sortformer): the activity grid the fusion consumes.
  int n_spk = 8;
  int diar_chunk_len = 264, diar_spkcache_len = 264, diar_spkcache_update_period = 264;
  double diar_pred_score_threshold = 0.25, diar_sil_threshold = 0.2;
  // Fusion: thresholded continuous activity (sigmoid > threshold), then
  // LayerNorm(n_spk) against the built-in sinusoidal kernel.
  bool speaker_feature_thresholded = true;
  double speaker_activity_threshold = 0.5;
  double spk_kernel_scale = 1.0;
  // Connector: identity into the projection the checkpoint ships.
  int output_dim = 0;           // 2688 = the LM hidden
  int connector_d_model = 1280; // the IdentityConnector width

  static NemotronTranscribePerceptionConfig parse(const minijson::Value& root);

  // Encoder tokens for a clip of `samples` samples at sample_rate: the
  // center-padded STFT frames ceil'd onto the stride grid, stacked.
  int64_t audio_token_count(int64_t samples) const;
};

struct NemotronTranscribeConfig {
  NemotronTranscribeLmConfig lm;
  NemotronTranscribePerceptionConfig perception;
  // The prompt contract: one `<|audio|>` locator per clip (the id is
  // resolved from the tokenizer at serve time, the way the vision path
  // resolves image markers from the config).
  std::string audio_locator_tag = "<|audio|>";
  std::string prompt_format = "nemotron3p5";

  static NemotronTranscribeConfig parse(const minijson::Value& root);
  static NemotronTranscribeConfig from_json_file(const std::string& path);
};

}  // namespace dgpp
