#pragma once
// Expected-tensor table for NVIDIA Nemotron-3.5-Transcribe-30B-A4B
// (docs/nemotron_transcribe_plan.md §census): every BF16-release tensor the
// loader consumes — names, dtypes, exact shapes — derived from the parsed
// config, not observed from one file.
//
// Naming is checkpoint truth (nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
// @ snapshot 9d8cd6b7, 2026-10-30): the LM rides `llm.model.layers.N.*` with
// every projection under `.mixer.` and one pre-norm per slot; the MTP block
// mirrors the walk under `llm.mtp.layers.N.*` (fusion tensors — `enorm`,
// `hnorm`, `eh_proj` — on the first slot, `final_layernorm` on the last).
// The perception module rides `perception.*`: the frozen preprocessor
// buffers, `encoder.asr_encoder.*` (NeMo TransformerEncoder: `pre_encode.proj`,
// `attn.w_qkv` + q/k norms + biased out_proj, `ffn.net.0`/`net.3`, two
// per-block LayerNorms, `embed_norm`/`final_norm`), the two fusion norms, and
// the connector `proj`.
//
// One group stays named-but-not-enumerated: the CTC timestamp head
// (`ctc_timestamp.`, deferred with word timestamps). TransformerEngine
// emits empty `*_extra_state` tensors; they are structural noise and
// ignored. The Sortformer branch (`perception.encoder.diarization_model.*`)
// joined the table with the milestone 5b oracle: its 31-block encoder
// (no q/k norms - the 11-tensor block pins it), pre_encode and norms,
// the mask head (encoder_proj, the sub-pixel upsample conv, the two
// speaker heads - the 384-wide hidden_to_spks is declared but unused on
// the single-pass route - and the learnable silence embedding), and its
// own frozen preprocessor buffers (unused on this route, shipped all the
// same).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/nemotron_transcribe/config.hpp"

namespace dgpp {

enum class NemotronWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  SlotNorm,        // the per-slot pre-norm
  MambaInProj,
  MambaConvWeight,
  MambaConvBias,
  MambaAlog,       // F32
  MambaD,          // F32
  MambaDtBias,     // F32
  MambaGatedNorm,  // mixer.norm over d_inner
  MambaOutProj,
  AttnQ,
  AttnK,
  AttnV,
  AttnO,
  RouterGate,
  RouterBias,      // F32 e_score_correction_bias
  ExpertsGateUp,   // stacked [E, H, I]
  ExpertsDown,     // stacked [E, I, H]
  SharedUp,
  SharedDown,
  MtpEnorm,
  MtpHnorm,
  MtpEhProj,
  MtpFinalNorm,
  PreprocFb,       // the shipped mel filterbank [1, mel, n_fft/2+1]
  PreprocWindow,   // [win]
  EncPreEncode,    // the 8-frame stacking Linear
  EncNorm,         // any encoder LayerNorm weight/bias (by name)
  EncQkv,
  EncQkNorm,
  EncOut,
  EncFfn1,
  EncFfn2,
  Connector,       // perception.proj weight/bias
  DiarHead,        // the Sortformer mask-head matrices and the silence embedding
};

struct NemotronExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  NemotronWeightClass cls = NemotronWeightClass::SlotNorm;
  int slot = -1;  // the slot index within the walk; -1 for the globals

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
};

// One observed tensor, keyed by name, for the validator.
struct NemotronTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct NemotronBindReport {
  bool ok = true;
  int missing = 0, dtype_mismatch = 0, shape_mismatch = 0, unexpected = 0;
  std::vector<std::string> errors;  // capped; counts are exact
};

// The Mamba-2 derived widths from the config, pinned against the release's
// shapes: d_inner = heads x head_dim; the in_proj row count is
// x + z + B + C + dt; the conv runs over x + B + C.
struct NemotronMambaGeometry {
  int64_t d_inner = 0;
  int64_t in_proj_rows = 0;
  int64_t conv_channels = 0;
};
NemotronMambaGeometry nemotron_mamba_geometry(const NemotronTranscribeLmConfig& lm);

// Tensors for the whole checkpoint (excluding the ignored groups).
std::vector<NemotronExpectedTensor> nemotron_expected_tensors(
    const NemotronTranscribeConfig& cfg);
// The LM walk only (globals, the 52 slots, the MTP block).
std::vector<NemotronExpectedTensor> nemotron_expected_lm_tensors(
    const NemotronTranscribeConfig& cfg);
// The perception module only (preprocessor, ASR encoder, fusion norms,
// connector).
std::vector<NemotronExpectedTensor> nemotron_expected_perception_tensors(
    const NemotronTranscribeConfig& cfg);

// true for names the binding neither expects nor rejects (the diarization
// branch, the CTC head, the TE `_extra_state` tensors).
bool nemotron_ignored_tensor(const std::string& name);

// Compare the expected table against the observed tensors: missing names,
// dtype and shape drift, and unexpected names (outside the ignored groups).
NemotronBindReport nemotron_validate_binding(
    const NemotronTranscribeConfig& cfg,
    const std::unordered_map<std::string, NemotronTensorDesc>& present);

}  // namespace dgpp
