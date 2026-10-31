#include "models/nemotron_transcribe/binding.hpp"

#include <format>
#include <set>

namespace dgpp {
namespace {

using TensorList = std::vector<NemotronExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         NemotronWeightClass cls, int slot) {
  out.push_back({std::move(name), dtype, std::move(shape), cls, slot});
}
void add_bf16(TensorList& out, std::string name, std::vector<int64_t> shape,
              NemotronWeightClass cls, int slot) {
  add(out, std::move(name), DType::BF16, std::move(shape), cls, slot);
}
void add_f32(TensorList& out, std::string name, std::vector<int64_t> shape,
             NemotronWeightClass cls, int slot) {
  add(out, std::move(name), DType::F32, std::move(shape), cls, slot);
}

// The LM's slot walk. `prefix` names the family root of the block list
// ("llm.model.layers." or "llm.mtp.layers."); the MTP carries the same
// slot kinds per its own block list, with the fusion tensors around it.
void add_slots(TensorList& out, const NemotronTranscribeLmConfig& lm,
               const std::vector<NemotronSlot>& slots, const std::string& prefix,
               bool mtp) {
  const int64_t H = lm.hidden_size;
  const auto mg = nemotron_mamba_geometry(lm);
  const int64_t q_rows = static_cast<int64_t>(lm.num_attention_heads) * lm.head_dim;
  const int64_t kv_rows = static_cast<int64_t>(lm.num_key_value_heads) * lm.head_dim;
  const int64_t E = lm.num_routed_experts, I = lm.moe_intermediate_size;
  const int64_t SI = lm.moe_shared_expert_intermediate_size;
  for (size_t i = 0; i < slots.size(); ++i) {
    const int slot = static_cast<int>(i);
    const std::string p = prefix + std::to_string(i) + ".";
    add_bf16(out, p + "norm.weight", {H}, NemotronWeightClass::SlotNorm, slot);
    switch (slots[i]) {
      case NemotronSlot::Mamba: {
        const std::string m = p + "mixer.";
        add_bf16(out, m + "in_proj.weight", {mg.in_proj_rows, H}, NemotronWeightClass::MambaInProj, slot);
        add_bf16(out, m + "conv1d.weight", {mg.conv_channels, 1, lm.conv_kernel},
                 NemotronWeightClass::MambaConvWeight, slot);
        if (lm.use_conv_bias)
          add_bf16(out, m + "conv1d.bias", {mg.conv_channels}, NemotronWeightClass::MambaConvBias, slot);
        add_f32(out, m + "_fp32_params.A_log", {lm.mamba_num_heads}, NemotronWeightClass::MambaAlog, slot);
        add_f32(out, m + "_fp32_params.D", {lm.mamba_num_heads}, NemotronWeightClass::MambaD, slot);
        add_f32(out, m + "_fp32_params.dt_bias", {lm.mamba_num_heads}, NemotronWeightClass::MambaDtBias, slot);
        add_bf16(out, m + "norm.weight", {mg.d_inner}, NemotronWeightClass::MambaGatedNorm, slot);
        add_bf16(out, m + "out_proj.weight", {H, mg.d_inner}, NemotronWeightClass::MambaOutProj, slot);
        break;
      }
      case NemotronSlot::Attention: {
        const std::string m = p + "mixer.";
        add_bf16(out, m + "q_proj.weight", {q_rows, H}, NemotronWeightClass::AttnQ, slot);
        add_bf16(out, m + "k_proj.weight", {kv_rows, H}, NemotronWeightClass::AttnK, slot);
        add_bf16(out, m + "v_proj.weight", {kv_rows, H}, NemotronWeightClass::AttnV, slot);
        add_bf16(out, m + "o_proj.weight", {H, q_rows}, NemotronWeightClass::AttnO, slot);
        break;
      }
      case NemotronSlot::Moe: {
        const std::string m = p + "mixer.";
        add_bf16(out, m + "gate.weight", {E, H}, NemotronWeightClass::RouterGate, slot);
        add_f32(out, m + "gate.e_score_correction_bias", {E}, NemotronWeightClass::RouterBias, slot);
        add_bf16(out, m + "experts.gate_and_up_projs", {E, H, I}, NemotronWeightClass::ExpertsGateUp, slot);
        add_bf16(out, m + "experts.down_projs", {E, I, H}, NemotronWeightClass::ExpertsDown, slot);
        add_bf16(out, m + "shared_experts.up_proj.weight", {SI, H}, NemotronWeightClass::SharedUp, slot);
        add_bf16(out, m + "shared_experts.down_proj.weight", {H, SI}, NemotronWeightClass::SharedDown, slot);
        break;
      }
    }
  }
  if (mtp) {
    // The MTP wrapper tensors: the fusion pair and the embedding->hidden
    // projection ride the first slot, the output norm the last.
    add_bf16(out, prefix + "0.eh_proj.weight", {H, 2 * H}, NemotronWeightClass::MtpEhProj, 0);
    add_bf16(out, prefix + "0.enorm.weight", {H}, NemotronWeightClass::MtpEnorm, 0);
    add_bf16(out, prefix + "0.hnorm.weight", {H}, NemotronWeightClass::MtpHnorm, 0);
    add_bf16(out, prefix + std::to_string(slots.size() - 1) + ".final_layernorm.weight", {H},
             NemotronWeightClass::MtpFinalNorm, static_cast<int>(slots.size()) - 1);
  }
}

void add_perception(TensorList& out, const NemotronTranscribeConfig& cfg) {
  const auto& p = cfg.perception;
  const int64_t D = p.enc_d_model, F = p.enc_ff_expansion * D;
  const int64_t hd = D / p.enc_heads;
  add_bf16(out, "perception.preprocessor.featurizer.fb", {1, p.mel_features, p.fft_size / 2 + 1},
           NemotronWeightClass::PreprocFb, -1);
  add_bf16(out, "perception.preprocessor.featurizer.window", {p.window_size},
           NemotronWeightClass::PreprocWindow, -1);
  const std::string e = "perception.encoder.asr_encoder.";
  add_bf16(out, e + "pre_encode.proj.weight", {D, static_cast<int64_t>(p.mel_features) * p.subsampling_factor},
           NemotronWeightClass::EncPreEncode, -1);
  add_bf16(out, e + "embed_norm.weight", {D}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, e + "embed_norm.bias", {D}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, e + "final_norm.weight", {D}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, e + "final_norm.bias", {D}, NemotronWeightClass::EncNorm, -1);
  for (int l = 0; l < p.enc_layers; ++l) {
    const std::string q = e + "layers." + std::to_string(l) + ".";
    add_bf16(out, q + "attn.w_qkv.weight", {3 * D, D}, NemotronWeightClass::EncQkv, l);
    add_bf16(out, q + "attn.q_norm.weight", {hd}, NemotronWeightClass::EncQkNorm, l);
    add_bf16(out, q + "attn.q_norm.bias", {hd}, NemotronWeightClass::EncQkNorm, l);
    add_bf16(out, q + "attn.k_norm.weight", {hd}, NemotronWeightClass::EncQkNorm, l);
    add_bf16(out, q + "attn.k_norm.bias", {hd}, NemotronWeightClass::EncQkNorm, l);
    add_bf16(out, q + "attn.out_proj.weight", {D, D}, NemotronWeightClass::EncOut, l);
    add_bf16(out, q + "attn.out_proj.bias", {D}, NemotronWeightClass::EncOut, l);
    add_bf16(out, q + "ffn.net.0.weight", {F, D}, NemotronWeightClass::EncFfn1, l);
    add_bf16(out, q + "ffn.net.0.bias", {F}, NemotronWeightClass::EncFfn1, l);
    add_bf16(out, q + "ffn.net.3.weight", {D, F}, NemotronWeightClass::EncFfn2, l);
    add_bf16(out, q + "ffn.net.3.bias", {D}, NemotronWeightClass::EncFfn2, l);
    add_bf16(out, q + "norm1.weight", {D}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm1.bias", {D}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm2.weight", {D}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm2.bias", {D}, NemotronWeightClass::EncNorm, l);
  }
  add_bf16(out, "perception.encoder.asr_norm.weight", {D}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, "perception.encoder.asr_norm.bias", {D}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, "perception.encoder.diar_norm.weight", {p.n_spk}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, "perception.encoder.diar_norm.bias", {p.n_spk}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, "perception.proj.weight", {p.output_dim, p.connector_d_model},
           NemotronWeightClass::Connector, -1);
  add_bf16(out, "perception.proj.bias", {p.output_dim}, NemotronWeightClass::Connector, -1);

  // The Sortformer branch (milestone 5b): its own frozen preprocessor
  // (an artifact of the bundled module — the release route feeds it the
  // ASR-side mel), the 31-block encoder (no q/k norms on the release's
  // blocks), and the single-pass mask head. hidden_to_spks is shipped
  // and unused on this route; learnable_sil_emb likewise (the streaming
  // path's silence row).
  const std::string dm = "perception.encoder.diarization_model.";
  const int64_t DD = p.diar_d_model, TF = p.tf_d_model;
  const int64_t dff = static_cast<int64_t>(p.diar_ff_expansion) * DD;
  const int64_t dh = DD / p.diar_heads;
  add_bf16(out, dm + "preprocessor.featurizer.fb", {1, p.mel_features, p.fft_size / 2 + 1},
           NemotronWeightClass::PreprocFb, -1);
  add_bf16(out, dm + "preprocessor.featurizer.window", {p.window_size},
           NemotronWeightClass::PreprocWindow, -1);
  add_bf16(out, dm + "encoder.pre_encode.proj.weight",
           {DD, static_cast<int64_t>(p.subsampling_factor) * p.mel_features},
           NemotronWeightClass::EncPreEncode, -1);
  add_bf16(out, dm + "encoder.embed_norm.weight", {DD}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, dm + "encoder.embed_norm.bias", {DD}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, dm + "encoder.final_norm.weight", {DD}, NemotronWeightClass::EncNorm, -1);
  add_bf16(out, dm + "encoder.final_norm.bias", {DD}, NemotronWeightClass::EncNorm, -1);
  for (int l = 0; l < p.diar_layers; ++l) {
    const std::string q = dm + "encoder.layers." + std::to_string(l) + ".";
    if (p.diar_qk_norm) {
      add_bf16(out, q + "attn.q_norm.weight", {dh}, NemotronWeightClass::EncQkNorm, l);
      add_bf16(out, q + "attn.q_norm.bias", {dh}, NemotronWeightClass::EncQkNorm, l);
      add_bf16(out, q + "attn.k_norm.weight", {dh}, NemotronWeightClass::EncQkNorm, l);
      add_bf16(out, q + "attn.k_norm.bias", {dh}, NemotronWeightClass::EncQkNorm, l);
    }
    add_bf16(out, q + "attn.w_qkv.weight", {3 * DD, DD}, NemotronWeightClass::EncQkv, l);
    add_bf16(out, q + "attn.out_proj.weight", {DD, DD}, NemotronWeightClass::EncOut, l);
    add_bf16(out, q + "attn.out_proj.bias", {DD}, NemotronWeightClass::EncOut, l);
    add_bf16(out, q + "ffn.net.0.weight", {dff, DD}, NemotronWeightClass::EncFfn1, l);
    add_bf16(out, q + "ffn.net.0.bias", {dff}, NemotronWeightClass::EncFfn1, l);
    add_bf16(out, q + "ffn.net.3.weight", {DD, dff}, NemotronWeightClass::EncFfn2, l);
    add_bf16(out, q + "ffn.net.3.bias", {DD}, NemotronWeightClass::EncFfn2, l);
    add_bf16(out, q + "norm1.weight", {DD}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm1.bias", {DD}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm2.weight", {DD}, NemotronWeightClass::EncNorm, l);
    add_bf16(out, q + "norm2.bias", {DD}, NemotronWeightClass::EncNorm, l);
  }
  const std::string sm = dm + "sortformer_modules.";
  add_bf16(out, sm + "encoder_proj.weight", {TF, DD}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "encoder_proj.bias", {TF}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "first_hidden_to_hidden.weight", {TF, TF}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "first_hidden_to_hidden.bias", {TF}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "hidden_to_spks.weight", {p.n_spk, 2 * TF}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "hidden_to_spks.bias", {p.n_spk}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "single_hidden_to_spks.weight", {p.n_spk, TF},
           NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "single_hidden_to_spks.bias", {p.n_spk}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "subpixel_upsample.weight",
           {TF * p.subsampling_factor, TF, 3}, NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "subpixel_upsample.bias", {TF * p.subsampling_factor},
           NemotronWeightClass::DiarHead, -1);
  add_bf16(out, sm + "learnable_sil_emb", {DD}, NemotronWeightClass::DiarHead, -1);
}

void record(NemotronBindReport& rep, const std::string& msg) {
  if (rep.errors.size() < 24) rep.errors.push_back(msg);
}

}  // namespace

NemotronMambaGeometry nemotron_mamba_geometry(const NemotronTranscribeLmConfig& lm) {
  const int64_t nh = lm.mamba_num_heads, hd = lm.mamba_head_dim;
  const int64_t G = lm.mamba_num_groups, S = lm.mamba_state_size;
  NemotronMambaGeometry g;
  g.d_inner = nh * hd;
  g.conv_channels = g.d_inner + 2 * G * S;  // the conv runs over x, B and C
  g.in_proj_rows = 2 * g.d_inner + 2 * G * S + nh;  // x + z + B + C + dt
  return g;
}

std::vector<NemotronExpectedTensor> nemotron_expected_lm_tensors(
    const NemotronTranscribeConfig& cfg) {
  const auto& lm = cfg.lm;
  TensorList out;
  add_bf16(out, "llm.model.embed_tokens.weight", {lm.vocab_size, lm.hidden_size},
           NemotronWeightClass::Embed, -1);
  add_bf16(out, "llm.lm_head.weight", {lm.vocab_size, lm.hidden_size},
           NemotronWeightClass::LmHead, -1);
  add_bf16(out, "llm.model.norm.weight", {lm.hidden_size}, NemotronWeightClass::FinalNorm, -1);
  add_slots(out, lm, lm.slots, "llm.model.layers.", /*mtp=*/false);
  add_slots(out, lm, lm.mtp_slots, "llm.mtp.layers.", /*mtp=*/true);
  return out;
}

std::vector<NemotronExpectedTensor> nemotron_expected_perception_tensors(
    const NemotronTranscribeConfig& cfg) {
  TensorList out;
  add_perception(out, cfg);
  return out;
}

std::vector<NemotronExpectedTensor> nemotron_expected_tensors(
    const NemotronTranscribeConfig& cfg) {
  auto out = nemotron_expected_lm_tensors(cfg);
  const auto per = nemotron_expected_perception_tensors(cfg);
  out.insert(out.end(), per.begin(), per.end());
  return out;
}

bool nemotron_ignored_tensor(const std::string& name) {
  static const std::string_view kPrefixes[] = {
      "ctc_timestamp.",                         // word timestamps are deferred
  };
  for (auto pfx : kPrefixes)
    if (name.rfind(pfx, 0) == 0) return true;
  return name.rfind("._extra_state") == name.size() - 13;  // TransformerEngine noise
}

NemotronBindReport nemotron_validate_binding(
    const NemotronTranscribeConfig& cfg,
    const std::unordered_map<std::string, NemotronTensorDesc>& present) {
  NemotronBindReport rep;
  const auto expected = nemotron_expected_tensors(cfg);
  std::set<std::string> seen;
  for (const auto& t : expected) {
    if (!seen.insert(t.name).second) {
      rep.ok = false;
      record(rep, "duplicate expected tensor: " + t.name);
    }
    const auto it = present.find(t.name);
    if (it == present.end()) {
      ++rep.missing;
      rep.ok = false;
      record(rep, "missing " + t.name);
      continue;
    }
    if (it->second.dtype != t.dtype) {
      ++rep.dtype_mismatch;
      rep.ok = false;
      record(rep, std::format("dtype {}: checkpoint {} expected {}", t.name,
                              static_cast<int>(it->second.dtype), static_cast<int>(t.dtype)));
    }
    if (it->second.shape != t.shape) {
      ++rep.shape_mismatch;
      rep.ok = false;
      record(rep, t.name + " shape drift: checkpoint " + [&] {
        std::string s;
        for (auto d : it->second.shape) s += std::to_string(d) + "x";
        return s;
      }() + " expected " + [&] {
        std::string s;
        for (auto d : t.shape) s += std::to_string(d) + "x";
        return s;
      }());
    }
  }
  for (const auto& [name, desc] : present) {
    if (nemotron_ignored_tensor(name)) continue;
    if (!seen.count(name)) {
      ++rep.unexpected;
      rep.ok = false;
      record(rep, "unexpected tensor: " + name);
    }
  }
  return rep;
}

}  // namespace dgpp
