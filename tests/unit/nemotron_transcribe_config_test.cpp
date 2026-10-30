// The Nemotron-3.5-Transcribe config parser (docs/nemotron_transcribe_plan.md
// §census): the release's own values parse, the slot census matches
// (23 Mamba / 6 Attention / 23 Moe over 52 entries), the perception geometry
// pins (400/160/512 mels, 8-frame stacking, the thresholded 0.5 fusion), the
// token estimator's boundary rule, and the refusals name the field. The
// landed-checkpoint case reads the cached release when present.
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/nemotron_transcribe/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The slot pattern exactly as the release ships it
// (llm_backbone/config.json layers_block_type, transcribed verbatim):
// 23 mamba / 6 attention / 23 moe slots over 52 entries.
std::string slots_array() {
  static const char* kSlots[] = {
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe",
      "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
      "mamba", "moe"};
  std::string s = "[";
  for (size_t i = 0; i < sizeof(kSlots) / sizeof(kSlots[0]); ++i) {
    if (i) s += ", ";
    s += std::string("\"") + kSlots[i] + "\"";
  }
  return s + "]";
}

std::string llm_json() {
  return R"({
    "hidden_size": 2688, "vocab_size": 131072,
    "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    "layers_block_type": )" + slots_array() + R"(,
    "mtp_layers_block_type": ["attention", "moe"],
    "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
    "rope_theta": 10000, "partial_rotary_factor": 1.0, "attention_bias": false,
    "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
    "ssm_state_size": 128, "conv_kernel": 4, "use_conv_bias": true,
    "mamba_proj_bias": false, "mamba_hidden_act": "silu",
    "time_step_min": 0.001, "time_step_max": 0.1, "time_step_floor": 0.0001,
    "n_routed_experts": 128, "num_experts_per_tok": 6,
    "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
    "routed_scaling_factor": 2.5, "topk_group": 1, "norm_topk_prob": true,
    "mlp_bias": false, "mlp_hidden_act": "relu2",
    "norm_eps": 1e-05, "rescale_prenorm_residual": true,
    "tie_word_embeddings": false, "max_position_embeddings": 262144
  })";
}

// A trimmed root config carrying the release's perception values.
std::string root_json(const std::string& needle = "", const std::string& repl = "") {
  std::string s = R"({
    "model_type": "nemo_speechlm",
    "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
    "prompt_format": "nemotron3p5",
    "audio_locator_tag": "<|audio|>",
    "encoder_chunk_size_seconds": 30.0,
    "compute_mtp": true,
    "spk_kernel_scale": 1.0,
    "llm_config": )" + llm_json() + R"(,
    "perception": {
      "output_dim": 2688,
      "modality_adapter": {"d_model": 1280},
      "preprocessor": {
        "sample_rate": 16000, "window_size": 0.025, "window_stride": 0.01,
        "window": "hann", "features": 128, "n_fft": 512, "log": true,
        "frame_splicing": 1, "dither": 1e-05, "pad_to": 0, "normalize": null
      },
      "encoder": {
        "feat_in": 128, "d_model": 1280, "n_heads": 10, "n_layers": 32,
        "qkv_bias": false, "qk_norm": true, "ff_expansion": 4,
        "pre_block_norm": true, "subsampling_factor": 8, "attn_mode": "full",
        "self_attention_model": "rope", "rope_base": 10000.0, "rotary_fraction": 0.5
      }
    },
    "audio_token_estimator": {
      "preprocessor": {"n_fft": 512, "hop_length": 160, "stft_pad_amount": 256},
      "subsampling": {"type": "feature_stacking", "factor": 8}
    },
    "pe_encoder_config": {
      "asr_encoder_type": "transformer",
      "speaker_feature_mode": "thresholded",
      "speaker_activity_threshold": 0.5,
      "chunk_size_seconds": 30.0,
      "frame_shift_seconds": 0.01,
      "diarization_model_cfg": {
        "sample_rate": 16000, "high_resolution": true,
        "sortformer_modules": {
          "num_spks": 8, "chunk_len": 264, "spkcache_len": 264,
          "spkcache_update_period": 264, "pred_score_threshold": 0.25,
          "sil_threshold": 0.2
        }
      }
    }
  })";
  if (!needle.empty()) {
    const size_t at = s.find(needle);
    require(at != std::string::npos, "test fixture carries the needle: " + needle);
    s = s.substr(0, at) + repl + s.substr(at + needle.size());
  }
  return s;
}

int count_slot(const std::vector<dgpp::NemotronSlot>& v, dgpp::NemotronSlot s) {
  int n = 0;
  for (auto e : v) if (e == s) ++n;
  return n;
}

}  // namespace

DGPP_TEST(nemotron_transcribe_config_parses_the_release_shape) {
  const auto d = dgpp::minijson::parse(root_json());
  const dgpp::NemotronTranscribeConfig c = dgpp::NemotronTranscribeConfig::parse(d.root);

  require(c.lm.num_layers == 52, "52 slots");
  require(count_slot(c.lm.slots, dgpp::NemotronSlot::Mamba) == 23, "23 mamba slots");
  require(count_slot(c.lm.slots, dgpp::NemotronSlot::Attention) == 6, "6 attention slots");
  require(count_slot(c.lm.slots, dgpp::NemotronSlot::Moe) == 23, "23 moe slots");
  require(c.lm.mtp_slots.size() == 2 && c.lm.mtp_slots[0] == dgpp::NemotronSlot::Attention,
          "the MTP block is attention + moe");
  require(c.lm.hidden_size == 2688 && c.lm.vocab_size == 131072, "lm sizes");
  require(c.lm.eos_token_ids.size() == 1 && c.lm.eos_token_ids[0] == 2, "eos 2");
  require(c.lm.num_attention_heads == 32 && c.lm.num_key_value_heads == 2 && c.lm.head_dim == 128,
          "attention geometry");
  require(c.lm.mamba_num_heads == 64 && c.lm.mamba_head_dim == 64 && c.lm.mamba_num_groups == 8 &&
              c.lm.mamba_state_size == 128 && c.lm.conv_kernel == 4,
          "mamba geometry");
  require(c.lm.num_routed_experts == 128 && c.lm.num_experts_per_tok == 6 &&
              c.lm.moe_intermediate_size == 1856 &&
              c.lm.moe_shared_expert_intermediate_size == 3712 &&
              c.lm.routed_scaling_factor == 2.5 && c.lm.norm_topk_prob,
          "moe geometry");
  require(!c.lm.tie_word_embeddings && c.lm.rescale_prenorm_residual, "lm flags");

  const auto& p = c.perception;
  require(p.sample_rate == 16000 && p.fft_size == 512 && p.window_size == 400 &&
              p.window_stride == 160 && p.mel_features == 128,
          "mel front end");
  require(p.subsampling_factor == 8 && p.stft_pad_amount == 256, "stacking and pad");
  require(p.enc_layers == 32 && p.enc_d_model == 1280 && p.enc_heads == 10 &&
              p.enc_ff_expansion == 4 && p.enc_qk_norm && !p.enc_qkv_bias,
          "asr encoder");
  require(p.enc_rotary_fraction == 0.5 && p.enc_rope_base == 10000.0, "encoder rope");
  require(p.n_spk == 8 && p.speaker_feature_thresholded && p.speaker_activity_threshold == 0.5,
          "the thresholded fusion contract");
  require(p.diar_chunk_len == 264 && p.diar_spkcache_len == 264 &&
              p.diar_spkcache_update_period == 264,
          "the Sortformer window");
  require(p.output_dim == 2688 && p.connector_d_model == 1280, "connector widths");
  require(c.audio_locator_tag == "<|audio|>" && c.prompt_format == "nemotron3p5", "prompt contract");

  // The estimator's boundary rule: floor(T/hop)+1 frames stacked by 8,
  // ceil'd — 13 tokens for the first second (the 101st frame opens a
  // partial group), 751 at 60 s, 45 001 at an hour.
  require(p.audio_token_count(16000) == 13, "1 s = 13 tokens");
  require(p.audio_token_count(96000) == 76, "6 s = 76 tokens (601 frames)");
  require(p.audio_token_count(60 * 16000) == 751, "60 s = 751 tokens");
  require(p.audio_token_count(3600 * 16000) == 45001, "1 h = 45001 tokens");
  require(p.audio_token_count(0) == 0, "empty clip");

  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::NemotronTranscribe, "arch");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::NemotronTranscribe)) ==
              "nemotron_transcribe",
          "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "nemo_speechlm"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::NemotronTranscribe,
          "by model_type");
}

DGPP_TEST(nemotron_transcribe_config_refuses_unsupported_shapes) {
  const auto refuse = [](const std::string& needle, const std::string& repl,
                         const std::string& why) {
    const auto d = dgpp::minijson::parse(root_json(needle, repl));
    bool threw = false;
    try {
      (void)dgpp::NemotronTranscribeConfig::parse(d.root);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find(why) != std::string::npos;
    }
    require(threw, "refused by name: " + why);
  };
  refuse("\"mlp_hidden_act\": \"relu2\"", "\"mlp_hidden_act\": \"silu\"", "mlp_hidden_act");
  refuse("\"mamba_hidden_act\": \"silu\"", "\"mamba_hidden_act\": \"gelu\"", "mamba_hidden_act");
  refuse("\"attn_mode\": \"full\"", "\"attn_mode\": \"windowed\"", "attn_mode");
  refuse("\"frame_splicing\": 1", "\"frame_splicing\": 2", "frame_splicing");
  refuse("\"output_dim\": 2688", "\"output_dim\": 4096", "output_dim");
  refuse("\"prompt_format\": \"nemotron3p5\"", "\"prompt_format\": \"nemotron2\"", "prompt_format");
  refuse("\"mtp_layers_block_type\": [\"attention\", \"moe\"]",
         "\"mtp_layers_block_type\": []", "mtp_layers_block_type");
  // The 52-slot census: an eighth attention slot parses (the field is the
  // release's contract, not a whitelist) — a first-slot mixer change does
  // not, and neither does an unknown kind.
  {
    const auto d = dgpp::minijson::parse(root_json(
        "\"layers_block_type\": [\"mamba\", \"moe\", \"mamba\"", "\"layers_block_type\": [\"moe\", \"mamba\", \"mamba\""));
    bool threw = false;
    try {
      (void)dgpp::NemotronTranscribeConfig::parse(d.root);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find("first slot") != std::string::npos;
    }
    require(threw, "the group's first slot must be the mixer");
  }
  {
    const auto d = dgpp::minijson::parse(root_json("\"attention\", \"moe\"]", "\"attentionx\", \"moe\"]"));
    bool threw = false;
    try {
      (void)dgpp::NemotronTranscribeConfig::parse(d.root);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find("unknown slot") != std::string::npos;
    }
    require(threw, "unknown slot kinds are refused by name");
  }
}

DGPP_TEST(nemotron_transcribe_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME is unset; checkpoint cache unavailable");
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/snapshots";
  if (!fs::is_directory(root)) dgpp::test::skip("checkpoint snapshot directory absent: " + root.string());
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path cfg = snap.path() / "config.json";
    if (!fs::exists(cfg)) continue;
    const dgpp::NemotronTranscribeConfig c = dgpp::NemotronTranscribeConfig::from_json_file(cfg.string());
    require(c.lm.num_layers == 52 && c.lm.hidden_size == 2688 && c.lm.vocab_size == 131072,
            "landed lm values");
    require(c.perception.enc_layers == 32 && c.perception.n_spk == 8 &&
                c.perception.speaker_feature_thresholded,
            "landed perception values");
    require(dgpp::detect_architecture_file(cfg.string()) ==
                dgpp::ModelArchitecture::NemotronTranscribe,
            "arch");
    return;
  }
  dgpp::test::skip("no eligible checkpoint config under " + root.string());
}
