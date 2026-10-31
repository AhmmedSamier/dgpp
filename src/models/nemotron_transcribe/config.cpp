#include "models/nemotron_transcribe/config.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <string>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Nemotron-Transcribe config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}

int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}

int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}

double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}

double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}

bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a boolean");
  return f->as_bool();
}

std::string optional_string(const minijson::Value& v, std::string_view field, const char* dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}

std::vector<NemotronSlot> parse_block_type(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<NemotronSlot> out;
  for (const auto& e : f.items()) {
    if (!e.is_string()) reject(field, "entry is not a string");
    const std::string_view s = e.as_string();
    if (s == "mamba") out.push_back(NemotronSlot::Mamba);
    else if (s == "attention") out.push_back(NemotronSlot::Attention);
    else if (s == "moe") out.push_back(NemotronSlot::Moe);
    else reject(field, std::string("unknown slot '") + std::string(s) + "'");
  }
  return out;
}

std::vector<int64_t> parse_eos(const minijson::Value& root) {
  std::vector<int64_t> out;
  const minijson::Value* f = root.find("eos_token_id");
  if (!f) reject("eos_token_id", "missing");
  if (f->is_number()) {
    out.push_back(f->as_int());
  } else if (f->is_array()) {
    for (const auto& e : f->items()) {
      if (!e.is_number()) reject("eos_token_id", "entry is not a number");
      out.push_back(e.as_int());
    }
  } else {
    reject("eos_token_id", "not a number or array");
  }
  if (out.empty()) reject("eos_token_id", "empty");
  return out;
}

const minijson::Value* find_path(const minijson::Value& root, std::initializer_list<const char*> path) {
  const minijson::Value* v = &root;
  for (const char* key : path) {
    v = v->find(key);
    if (!v) return nullptr;
  }
  return v;
}

void check_positive(int v, std::string_view field) {
  if (v <= 0) reject(field, "must be positive");
}

}  // namespace

NemotronTranscribeLmConfig NemotronTranscribeLmConfig::parse(const minijson::Value& root) {
  NemotronTranscribeLmConfig c;
  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.eos_token_ids = parse_eos(root);
  c.bos_token_id = optional_int(root, "bos_token_id", -1);
  c.pad_token_id = optional_int(root, "pad_token_id", -1);
  c.slots = parse_block_type(root, "layers_block_type");
  c.num_layers = static_cast<int>(c.slots.size());
  check_positive(c.num_layers, "layers_block_type");
  if (c.slots.size() < 2 || c.slots.front() != NemotronSlot::Mamba)
    reject("layers_block_type", "the first slot must be a Mamba mixer");
  c.mtp_slots = parse_block_type(root, "mtp_layers_block_type");
  if (c.mtp_slots.empty()) reject("mtp_layers_block_type", "the checkpoint trains an MTP block");

  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = require_int(root, "head_dim");
  if (c.head_dim <= 0) c.head_dim = c.hidden_size / c.num_attention_heads;
  c.rope_theta = optional_double(root, "rope_theta", 10000.0);
  c.partial_rotary_factor = optional_double(root, "partial_rotary_factor", 1.0);
  c.attention_bias = optional_bool(root, "attention_bias", false);
  if (c.attention_bias) reject("attention_bias", "the family ships no qkv biases");
  for (int v : {c.num_attention_heads, c.num_key_value_heads}) check_positive(v, "num_*_heads");
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide the query heads");

  c.mamba_num_heads = require_int(root, "mamba_num_heads");
  c.mamba_head_dim = require_int(root, "mamba_head_dim");
  c.mamba_num_groups = require_int(root, "n_groups");
  c.mamba_state_size = require_int(root, "ssm_state_size");
  c.conv_kernel = require_int(root, "conv_kernel");
  c.use_conv_bias = optional_bool(root, "use_conv_bias", true);
  c.mamba_proj_bias = optional_bool(root, "mamba_proj_bias", false);
  c.mamba_hidden_act = optional_string(root, "mamba_hidden_act", "silu");
  if (c.mamba_hidden_act != "silu") reject("mamba_hidden_act", "only silu is implemented");
  c.time_step_min = optional_double(root, "time_step_min", 0.001);
  c.time_step_max = optional_double(root, "time_step_max", 0.1);
  c.time_step_floor = optional_double(root, "time_step_floor", 1e-4);
  // The tl;dr of the in_proj split, pinned against the checkpoint shapes in
  // the binding test: x + z + B + C + dt.
  const int64_t d_inner = static_cast<int64_t>(c.mamba_num_heads) * c.mamba_head_dim;
  const int64_t in_proj = 2 * d_inner + 2LL * c.mamba_num_groups * c.mamba_state_size + c.mamba_num_heads;
  if (in_proj > 1u << 24) reject("mamba", "in_proj width out of range");

  c.num_routed_experts = require_int(root, "n_routed_experts");
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.moe_shared_expert_intermediate_size = require_int(root, "moe_shared_expert_intermediate_size");
  c.routed_scaling_factor = optional_double(root, "routed_scaling_factor", 1.0);
  c.moe_n_groups = optional_int(root, "n_groups", 1);
  // The LLM config reuses n_groups for both the Mamba groups and the
  // router group limit; topk_group carries the router side.
  c.moe_topk_group = optional_int(root, "topk_group", 1);
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  c.mlp_bias = optional_bool(root, "mlp_bias", false);
  c.mlp_hidden_act = optional_string(root, "mlp_hidden_act", "relu2");
  if (c.mlp_hidden_act != "relu2") reject("mlp_hidden_act", "the family ships relu2 experts");
  for (int v : {c.num_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size})
    check_positive(v, "moe sizes");

  c.norm_eps = optional_double(root, "norm_eps", 1e-5);
  c.rescale_prenorm_residual = optional_bool(root, "rescale_prenorm_residual", true);
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  check_positive(c.max_position_embeddings, "max_position_embeddings");
  return c;
}

NemotronTranscribePerceptionConfig NemotronTranscribePerceptionConfig::parse(
    const minijson::Value& root) {
  const minijson::Value* p = root.find("perception");
  if (!p || !p->is_object()) reject("perception", "missing");
  NemotronTranscribePerceptionConfig c;

  c.output_dim = require_int(*p, "output_dim");
  const minijson::Value* pre = find_path(*p, {"preprocessor"});
  if (!pre) reject("perception.preprocessor", "missing");
  c.sample_rate = require_int(*pre, "sample_rate");
  c.fft_size = require_int(*pre, "n_fft");
  c.mel_features = require_int(*pre, "features");
  c.window_size = static_cast<int>(require_double(*pre, "window_size") * c.sample_rate + 0.5);
  c.window_stride = static_cast<int>(require_double(*pre, "window_stride") * c.sample_rate + 0.5);
  c.dither = optional_double(*pre, "dither", 0.0);
  c.frame_splicing = optional_int(*pre, "frame_splicing", 1);
  c.log_mel = optional_bool(*pre, "log", true);
  if (const minijson::Value* w = pre->find("window"); w && (!w->is_string() || w->as_string() != "hann"))
    reject("perception.preprocessor.window", "only hann is implemented");
  if (c.frame_splicing != 1) reject("perception.preprocessor.frame_splicing", "only 1 is implemented");
  if (!c.log_mel) reject("perception.preprocessor.log", "the shipped model trains on log mels");
  if (c.window_size <= 0 || c.window_stride <= 0 || c.fft_size < c.window_size)
    reject("perception.preprocessor", "window/stride/nfft are not a usable shape");

  const minijson::Value* est = root.find("audio_token_estimator");
  if (est && est->is_object()) {
    if (const minijson::Value* pp = est->find("preprocessor")) {
      c.stft_pad_amount = optional_int(*pp, "stft_pad_amount", c.fft_size / 2);
      if (const minijson::Value* hl = pp->find("hop_length"); hl && hl->is_number() &&
          hl->as_int() != c.window_stride)
        reject("audio_token_estimator.preprocessor.hop_length", "disagrees with the preprocessor stride");
    }
    if (const minijson::Value* sub = est->find("subsampling")) {
      const std::string ty = optional_string(*sub, "type", "feature_stacking");
      if (ty != "feature_stacking") reject("audio_token_estimator.subsampling.type", "only feature_stacking");
      c.subsampling_factor = require_int(*sub, "factor");
    }
  }
  if (c.subsampling_factor <= 0) reject("subsampling_factor", "must be positive");

  const minijson::Value* enc = find_path(*p, {"encoder"});
  if (!enc) reject("perception.encoder", "missing");
  c.enc_layers = require_int(*enc, "n_layers");
  c.enc_d_model = require_int(*enc, "d_model");
  c.enc_heads = require_int(*enc, "n_heads");
  c.enc_ff_expansion = require_int(*enc, "ff_expansion");
  c.enc_qk_norm = optional_bool(*enc, "qk_norm", true);
  c.enc_qkv_bias = optional_bool(*enc, "qkv_bias", false);
  c.enc_pre_block_norm = optional_bool(*enc, "pre_block_norm", true);
  c.enc_rope_base = optional_double(*enc, "rope_base", 10000.0);
  c.enc_rotary_fraction = optional_double(*enc, "rotary_fraction", 1.0);
  if (const minijson::Value* am = enc->find("attn_mode"); am && am->as_string() != "full")
    reject("perception.encoder.attn_mode", "only full attention is implemented");
  for (int v : {c.enc_layers, c.enc_d_model, c.enc_heads, c.enc_ff_expansion})
    check_positive(v, "perception.encoder sizes");
  if (c.enc_d_model % c.enc_heads != 0) reject("perception.encoder.n_heads", "must divide d_model");
  c.encoder_chunk_size_seconds = optional_double(root, "encoder_chunk_size_seconds", 30.0);
  if (c.encoder_chunk_size_seconds <= 0)
    reject("encoder_chunk_size_seconds", "must be positive");

  // The ParallelExpertEncoder bundle: the fusion contract and the Sortformer
  // window sizes the diarization branch streams with.
  const minijson::Value* pe = root.find("pe_encoder_config");
  if (!pe || !pe->is_object()) reject("pe_encoder_config", "missing");
  c.spk_kernel_scale = optional_double(root, "spk_kernel_scale", 1.0);
  if (const minijson::Value* ma = find_path(*p, {"modality_adapter"}); ma) {
    const int d = optional_int(*ma, "d_model", c.connector_d_model);
    if (d != c.enc_d_model) reject("perception.modality_adapter.d_model", "disagrees with the encoder d_model");
    c.connector_d_model = d;
  }
  const minijson::Value* dia = pe->find("diarization_model_cfg");
  if (!dia) reject("pe_encoder_config.diarization_model_cfg", "missing");
  if (const minijson::Value* sf = dia->find("sortformer_modules"); sf && sf->is_object()) {
    c.n_spk = require_int(*sf, "num_spks");
    c.diar_chunk_len = require_int(*sf, "chunk_len");
    c.diar_spkcache_len = require_int(*sf, "spkcache_len");
    c.diar_spkcache_update_period = require_int(*sf, "spkcache_update_period");
    c.diar_pred_score_threshold = optional_double(*sf, "pred_score_threshold", 0.25);
    c.diar_sil_threshold = optional_double(*sf, "sil_threshold", 0.2);
  }
  // The diarization branch (milestone 5b): a second NeMo TransformerEncoder
  // with its own dims — and the mask head it actually runs. The release
  // route is the single-pass forward_infer (no streaming spkcache/fifo);
  // everything outside it is refused, not silently skipped.
  if (dia->find("transformer_encoder"))
    reject("pe_encoder_config.diarization_model_cfg.transformer_encoder",
           "the Sortformer tf-transformer over the mask head is not the release route");
  const minijson::Value* denc = dia->find("encoder");
  if (!denc || !denc->is_object()) reject("pe_encoder_config.diarization_model_cfg.encoder", "missing");
  c.diar_layers = require_int(*denc, "n_layers");
  c.diar_d_model = require_int(*denc, "d_model");
  c.diar_heads = require_int(*denc, "n_heads");
  c.diar_ff_expansion = static_cast<int>(require_int(*denc, "ff_expansion"));
  c.diar_qk_norm = optional_bool(*denc, "qk_norm", false);
  c.diar_rope_base = optional_double(*denc, "rope_base", 10000.0);
  c.diar_rotary_fraction = optional_double(*denc, "rotary_fraction", 1.0);
  if (optional_string(*denc, "self_attention_model", "rope") != "rope")
    reject("diarization_model_cfg.encoder.self_attention_model", "only rope");
  if (optional_string(*denc, "attn_mode", "full") != "full")
    reject("diarization_model_cfg.encoder.attn_mode", "only full attention");
  if (!optional_bool(*denc, "pre_block_norm", true))
    reject("diarization_model_cfg.encoder.pre_block_norm", "post-norm is not this release");
  if (optional_bool(*denc, "qkv_bias", false))
    reject("diarization_model_cfg.encoder.qkv_bias", "biased qkv is not this release");
  if (require_int(*denc, "subsampling_factor") != c.subsampling_factor)
    reject("diarization_model_cfg.encoder.subsampling_factor",
           "the diar grid must match the ASR grid");
  for (int v : {c.diar_layers, c.diar_d_model, c.diar_heads, c.diar_ff_expansion})
    check_positive(v, "diarization_model_cfg.encoder sizes");
  if (c.diar_d_model % c.diar_heads != 0)
    reject("diarization_model_cfg.encoder.n_heads", "must divide d_model");
  if (const minijson::Value* md = dia->find("model_defaults")) {
    c.tf_d_model = optional_int(*md, "tf_d_model", c.diar_d_model);
  }
  if (c.tf_d_model <= 0 || c.tf_d_model == c.diar_d_model)
    reject("diarization_model_cfg.model_defaults.tf_d_model",
           "fc == tf selects the tf-transformer path, not this release's");
  // The PEE's branch normalizations: ASR per_feature, diar raw ("NA").
  const std::string asr_norm = optional_string(*pe, "asr_normalize_type", "per_feature");
  if (asr_norm != "per_feature" && asr_norm != "NA")
    reject("pe_encoder_config.asr_normalize_type", "only per_feature is implemented");
  c.asr_normalize_per_feature = asr_norm == "per_feature";
  const std::string dia_norm = optional_string(*pe, "diar_normalize_type", "NA");
  if (dia_norm != "NA")
    reject("pe_encoder_config.diar_normalize_type",
           "the diarization branch consumes the raw frontend mel on this route");
  c.speaker_feature_thresholded =
      optional_string(*pe, "speaker_feature_mode", "continuous") == "thresholded";
  if (const minijson::Value* th = pe->find("speaker_activity_threshold"); th && th->is_number())
    c.speaker_activity_threshold = th->as_double();
  if (c.speaker_feature_thresholded && (c.speaker_activity_threshold < 0.0 ||
                                        c.speaker_activity_threshold > 1.0))
    reject("pe_encoder_config.speaker_activity_threshold", "must sit in [0, 1]");
  if (const minijson::Value* hs = dia->find("high_resolution"); hs && hs->is_bool() && hs->as_bool()) {
    // Sortformer runs its activity at 2x the encoder grid (the conv_transpose
    // upsampler) and the fusion downsamples: the frame_shift the bundle
    // names is the pre-upsample grid.
  }
  check_positive(c.n_spk, "sortformer_modules.num_spks");
  return c;
}

int64_t NemotronTranscribePerceptionConfig::audio_token_count(int64_t samples) const {
  if (samples <= 0) return 0;
  // NeMo's center-padded STFT (pad n_fft/2 each side, pad_to 0) yields
  // floor(samples / hop) + 1 frames; the stacking subsamples ceil by 8 —
  // the transformer branch's boundary rule in ParallelExpertEncoder.
  const int64_t frames = samples / window_stride + 1;
  return (frames + subsampling_factor - 1) / subsampling_factor;
}

NemotronTranscribeConfig NemotronTranscribeConfig::parse(const minijson::Value& root) {
  NemotronTranscribeConfig c;
  const minijson::Value* llm = root.find("llm_config");
  if (!llm || !llm->is_object()) reject("llm_config", "missing (the inline HF config)");
  c.lm = NemotronTranscribeLmConfig::parse(*llm);
  c.perception = NemotronTranscribePerceptionConfig::parse(root);
  c.audio_locator_tag = optional_string(root, "audio_locator_tag", "<|audio|>");
  c.prompt_format = optional_string(root, "prompt_format", "nemotron3p5");
  if (c.prompt_format != "nemotron3p5") reject("prompt_format", "only nemotron3p5 is implemented");
  if (c.perception.output_dim != c.lm.hidden_size)
    reject("perception.output_dim", "must equal the LM hidden size");
  return c;
}

NemotronTranscribeConfig NemotronTranscribeConfig::from_json_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("Nemotron-Transcribe config: cannot open " + path);
  std::string buf;
  char tmp[65536];
  size_t n;
  while ((n = std::fread(tmp, 1, sizeof(tmp), f)) > 0) buf.append(tmp, n);
  std::fclose(f);
  const minijson::ParseResult doc = minijson::parse(buf);
  if (!doc.root.is_object()) reject("root", "not an object");
  return parse(doc.root);
}

}  // namespace dgpp
