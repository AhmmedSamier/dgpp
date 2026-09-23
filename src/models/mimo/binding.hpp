#pragma once
// Expected-tensor table for MiMo-V2.6-Flash (MiMoV2ForCausalLM, 2026-09-22,
// docs/mimo_v26_flash_plan.md §1.3): every text-backbone tensor the
// checkpoint must contain — names, dtypes, exact shapes — derived from the
// parsed config, not observed from one file. The table drives the offline
// validator and the resident loader, as the GLM-4.7 and DeepSeek tables do.
//
// Naming is checkpoint truth (XiaomiMiMo/MiMo-V2.6-Flash-RL): main layers
// under `model.layers.L.`, the draft layers under `model.mtp.layers.M.`
// (model_mtp.safetensors), the globals `model.embed_tokens.weight`,
// `model.norm.weight`, `lm_head.weight`. The vision encoder (`visual.*`),
// the audio encoder (`audio_encoder.*`, `speech_embeddings.*`) and the
// draft layers past the first are in the checkpoint and NOT served: the
// validator lists them as ignored, never as unexpected.
//
// Format contract (the release as shipped):
//   fp8 dense   X.weight F8_E4M3 [N, K] + X.weight_scale_inv F32
//               [ceil(N/128), K/128] — the fused qkv_proj's scale rows are
//               tiled per chunk (config.qkv_scale_rows), the MLPs' one grid;
//   MXFP4       X.weight U8 [N, K/2] (e2m1 pairs) + X.weight_scale U8
//               [N, K/32] (e8m0) — the routed experts;
//   BF16        o_proj, the router gate, the norms, the sink biases, the
//               embedding, the head, the draft's eh_proj;
//   F32         the router's e_score_correction_bias.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/mimo/config.hpp"

namespace dgpp {

enum class MimoWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // input_layernorm, post_attention_layernorm / pre_mlp_layernorm
  Attention,      // qkv_proj (fp8), o_proj (bf16), attention_sink_bias
  Router,         // mlp.gate.weight, mlp.gate.e_score_correction_bias
  DenseMlp,       // layer 0's and the draft's gate/up/down (fp8)
  RoutedExpert,   // MXFP4 triples
  MtpHead,        // enorm, hnorm, eh_proj, final_layernorm
};

enum class MimoTensorRole : uint8_t {
  Plain,
  Fp8Payload,   // F8_E4M3 [N, K]
  Fp8Scale,     // F32 [scale rows, K/128]
  Fp4Payload,   // U8 [N, K/2]
  Fp4Scale,     // U8 [N, K/32] (e8m0)
};

struct MimoExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  MimoWeightClass cls = MimoWeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals; mtp_layer() for the draft
  int expert = -1;  // routed-expert id, -1 otherwise
  MimoTensorRole role = MimoTensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const { return false; }
};

// The checkpoint name prefix of a layer: "model.layers.L." for the main
// layers, "model.mtp.layers.0." for the draft.
std::string mimo_layer_prefix(const MimoTextConfig& cfg, int layer);
// Whether a checkpoint tensor name belongs to a part of the release the
// engine does not serve (the encoders, the later draft layers).
bool mimo_ignored_tensor(const MimoTextConfig& cfg, const std::string& name);

std::vector<MimoExpectedTensor> mimo_expected_text_tensors(const MimoTextConfig& cfg);
std::vector<MimoExpectedTensor> mimo_expected_layer_tensors(const MimoTextConfig& cfg, int layer);
std::vector<MimoExpectedTensor> mimo_expected_global_tensors(const MimoTextConfig& cfg);

struct MimoTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct MimoBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t ignored = 0;       // encoder / later-draft tensors present and skipped
  size_t fp8_matrices = 0;
  size_t fp4_matrices = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

MimoBindReport mimo_validate_text_binding(
    const MimoTextConfig& cfg,
    const std::unordered_map<std::string, MimoTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/mimo_v26_flash_plan.md §2):
// throws std::invalid_argument naming the dim that does not divide. world
// must divide the qkv_proj chunk count (num_key_value_heads: the chunks a
// rank holds are whole), hence the query heads and both kv head counts;
// every expert / dense intermediate slice must be a multiple of 128 (the
// fp8 scale block) and 32 (the MXFP4 block).
void mimo_tp_validate_geometry(const MimoTextConfig& cfg, int rank, int world);

}  // namespace dgpp
