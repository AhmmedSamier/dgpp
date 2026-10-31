// The Nemotron-3.5-Transcribe expected-tensor table
// (docs/nemotron_transcribe_plan.md §census): the table the config derives,
// the counts the release's census shows, and — against the cached
// checkpoint — a byte-exact name/dtype/shape agreement with everything the
// loader consumes, the ignored groups excluded.
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/nemotron_transcribe/binding.hpp"
#include "models/nemotron_transcribe/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string root_json() {
  // The release shape, transcribed (the config test owns the full fixture;
  // the binding only needs the derived geometry).
  std::string slots = "[";
  static const char* kSlots[] = {
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe",
      "mamba", "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe",
      "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
      "mamba", "moe"};
  for (size_t i = 0; i < sizeof(kSlots) / sizeof(kSlots[0]); ++i) {
    require(sizeof(kSlots) / sizeof(kSlots[0]) == 52, "fixture slot census");
    if (i) slots += ", ";
    slots += std::string("\"") + kSlots[i] + "\"";
  }
  slots += "]";
  return R"({
    "model_type": "nemo_speechlm",
    "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
    "prompt_format": "nemotron3p5", "audio_locator_tag": "<|audio|>",
    "encoder_chunk_size_seconds": 30.0, "spk_kernel_scale": 1.0,
    "llm_config": {
      "hidden_size": 2688, "vocab_size": 131072,
      "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
      "layers_block_type": )" + slots + R"(,
      "mtp_layers_block_type": ["attention", "moe"],
      "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
      "rope_theta": 10000, "partial_rotary_factor": 1.0,
      "mamba_num_heads": 64, "mamba_head_dim": 64, "n_groups": 8,
      "ssm_state_size": 128, "conv_kernel": 4, "use_conv_bias": true,
      "mamba_proj_bias": false, "mamba_hidden_act": "silu",
      "n_routed_experts": 128, "num_experts_per_tok": 6,
      "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
      "routed_scaling_factor": 2.5, "norm_topk_prob": true, "mlp_hidden_act": "relu2",
      "norm_eps": 1e-05, "rescale_prenorm_residual": true,
      "tie_word_embeddings": false, "max_position_embeddings": 262144
    },
    "perception": {
      "output_dim": 2688,
      "modality_adapter": {"d_model": 1280},
      "preprocessor": {"sample_rate": 16000, "window_size": 0.025, "window_stride": 0.01,
        "window": "hann", "features": 128, "n_fft": 512, "log": true,
        "frame_splicing": 1, "dither": 0.0, "pad_to": 0, "normalize": null},
      "encoder": {"feat_in": 128, "d_model": 1280, "n_heads": 10, "n_layers": 32,
        "qkv_bias": false, "qk_norm": true, "ff_expansion": 4, "pre_block_norm": true,
        "subsampling_factor": 8, "attn_mode": "full", "self_attention_model": "rope",
        "rope_base": 10000.0, "rotary_fraction": 0.5}
    },
    "audio_token_estimator": {"preprocessor": {"n_fft": 512, "hop_length": 160, "stft_pad_amount": 256},
      "subsampling": {"type": "feature_stacking", "factor": 8}},
    "pe_encoder_config": {"asr_encoder_type": "transformer",
      "asr_normalize_type": "per_feature", "diar_normalize_type": "NA",
      "speaker_feature_mode": "thresholded", "speaker_activity_threshold": 0.5,
      "diarization_model_cfg": {
        "high_resolution": true,
        "model_defaults": {"fc_d_model": 512, "tf_d_model": 192},
        "encoder": {"n_layers": 31, "d_model": 512, "n_heads": 8, "ff_expansion": 4.0,
          "subsampling_factor": 8, "self_attention_model": "rope", "qk_norm": false,
          "pre_block_norm": true, "qkv_bias": false, "attn_mode": "full"},
        "sortformer_modules": {"num_spks": 8, "chunk_len": 264,
        "spkcache_len": 264, "spkcache_update_period": 264,
        "pred_score_threshold": 0.25, "sil_threshold": 0.2}}}
  })";
}

std::unordered_map<std::string, dgpp::NemotronTensorDesc> present_from(
    const std::vector<dgpp::NemotronExpectedTensor>& expected) {
  std::unordered_map<std::string, dgpp::NemotronTensorDesc> m;
  for (const auto& t : expected) m.emplace(t.name, dgpp::NemotronTensorDesc{t.dtype, t.shape});
  return m;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") && fs::exists(snap.path() / "model.safetensors"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(nemotron_binding_table_has_the_release_shape) {
  const auto doc = dgpp::minijson::parse(root_json());
  const dgpp::NemotronTranscribeConfig cfg = dgpp::NemotronTranscribeConfig::parse(doc.root);
  const auto tensors = dgpp::nemotron_expected_tensors(cfg);

  // Census: the LM is 3 globals + 398 slot tensors (23 mamba x 9,
  // 6 attention x 5, 23 moe x 7, one norm each) + the MTP block
  // (attention 5, moe 7, four wrapper tensors).
  const auto lm = dgpp::nemotron_expected_lm_tensors(cfg);
  require(lm.size() == 3 + 398 + 12 + 4, "lm tensors " + std::to_string(lm.size()));
  const auto per = dgpp::nemotron_expected_perception_tensors(cfg);
  // Perception: preprocessor 2, stacking linear 1, four encoder globals,
  // 32 x 15 blocks, four fusion-norm tensors, connector 2 — plus the
  // Sortformer branch: own preprocessor 2, stacking 1, four globals,
  // 31 x 11 blocks (no q/k norms), mask head 11.
  require(per.size() == 2 + 1 + 4 + 32 * 15 + 4 + 2 + 2 + 1 + 4 + 31 * 11 + 11,
          "perception tensors " + std::to_string(per.size()));
  require(tensors.size() == lm.size() + per.size(), "total table");

  // The derived Mamba widths, pinned against the release's shapes.
  const auto mg = dgpp::nemotron_mamba_geometry(cfg.lm);
  require(mg.d_inner == 4096 && mg.conv_channels == 6144 && mg.in_proj_rows == 10304,
          "mamba geometry");
  const auto find = [&](const std::string& name) -> const dgpp::NemotronExpectedTensor* {
    for (const auto& t : tensors)
      if (t.name == name) return &t;
    return nullptr;
  };
  const auto* in_proj = find("llm.model.layers.0.mixer.in_proj.weight");
  require(in_proj && in_proj->shape == std::vector<int64_t>({10304, 2688}), "in_proj");
  const auto* alog = find("llm.model.layers.0.mixer._fp32_params.A_log");
  require(alog && alog->dtype == dgpp::DType::F32 && alog->shape == std::vector<int64_t>({64}), "A_log");
  const auto* gu = find("llm.model.layers.1.mixer.experts.gate_and_up_projs");
  require(gu && gu->shape == std::vector<int64_t>({128, 2688, 1856}), "gate_and_up");
  const auto* down = find("llm.model.layers.1.mixer.experts.down_projs");
  require(down && down->shape == std::vector<int64_t>({128, 1856, 2688}), "experts.down");
  const auto* qkv = find("perception.encoder.asr_encoder.layers.31.attn.w_qkv.weight");
  require(qkv && qkv->shape == std::vector<int64_t>({3840, 1280}), "encoder qkv");
  const auto* eh = find("llm.mtp.layers.0.eh_proj.weight");
  require(eh && eh->shape == std::vector<int64_t>({2688, 5376}), "mtp eh_proj");

  // Self-consistency: the table validates against itself, and the ignored
  // groups are not part of it.
  const auto rep = dgpp::nemotron_validate_binding(cfg, present_from(tensors));
  require(rep.ok, "table is self-consistent");
  require(!dgpp::nemotron_ignored_tensor("perception.encoder.diarization_model.encoder.x"),
          "the diarization branch is enumerated");
  const auto* sh = find("perception.encoder.diarization_model.sortformer_modules."
                        "single_hidden_to_spks.weight");
  require(sh && sh->shape == std::vector<int64_t>({8, 192}), "diar speaker head");
  const auto* sup = find("perception.encoder.diarization_model.sortformer_modules."
                         "subpixel_upsample.weight");
  require(sup && sup->shape == std::vector<int64_t>({1536, 192, 3}), "sub-pixel upsample");
  const auto* dblk = find("perception.encoder.diarization_model.encoder.layers.30.attn.w_qkv.weight");
  require(dblk && dblk->shape == std::vector<int64_t>({1536, 512}), "diar block 30");
  require(!find("perception.encoder.diarization_model.encoder.layers.30.attn.q_norm.weight"),
          "no q/k norms on the diar blocks (the 11-tensor census)");
  require(dgpp::nemotron_ignored_tensor("ctc_timestamp.decoder.head.weight"), "ctc ignored");
  require(dgpp::nemotron_ignored_tensor("llm.model.layers.5.mixer.q_proj._extra_state"),
          "TE state ignored");
  require(!dgpp::nemotron_ignored_tensor("llm.model.embed_tokens.weight"), "globals not ignored");
}

DGPP_TEST(nemotron_binding_reports_drift) {
  const auto doc = dgpp::minijson::parse(root_json());
  const dgpp::NemotronTranscribeConfig cfg = dgpp::NemotronTranscribeConfig::parse(doc.root);
  auto tensors = dgpp::nemotron_expected_tensors(cfg);
  {
    auto present = present_from(tensors);
    present.erase("llm.model.layers.23.mixer.out_proj.weight");
    const auto rep = dgpp::nemotron_validate_binding(cfg, present);
    require(!rep.ok && rep.missing == 1, "a dropped tensor is missing");
  }
  {
    auto present = present_from(tensors);
    present["llm.model.layers.23.mixer.out_proj.weight"].dtype = dgpp::DType::F32;
    const auto rep = dgpp::nemotron_validate_binding(cfg, present);
    require(!rep.ok && rep.dtype_mismatch == 1, "a dtype flip is caught");
  }
  {
    auto present = present_from(tensors);
    present["perception.proj.weight"].shape = {2688, 2688};
    const auto rep = dgpp::nemotron_validate_binding(cfg, present);
    require(!rep.ok && rep.shape_mismatch == 1, "a shape flip is caught");
  }
  {
    auto present = present_from(tensors);
    present["llm.model.layers.99.norm.weight"] = {dgpp::DType::BF16, {2688}};
    const auto rep = dgpp::nemotron_validate_binding(cfg, present);
    require(!rep.ok && rep.unexpected == 1, "an extra tensor is refused");
  }
}

DGPP_TEST(nemotron_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty()) {
    dgpp::test::skip("NVIDIA-Nemotron-3.5-Transcribe-30B-A4B snapshot unavailable");
    return;
  }
  const dgpp::NemotronTranscribeConfig cfg =
      dgpp::NemotronTranscribeConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::NemotronTensorDesc> present;
  for (const auto& entry : std::filesystem::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::NemotronTensorDesc{t.dtype, t.shape});
    });
  }
  const auto rep = dgpp::nemotron_validate_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok, "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " +
                        std::to_string(rep.shape_mismatch) + ", unexpected " +
                        std::to_string(rep.unexpected) + errs);
}
