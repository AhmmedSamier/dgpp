#pragma once
// Expected-tensor table for the GLM-5.3 text model (M4 deliverables 1-2).
//
// Given a parsed GlmTextConfig, glm_expected_text_tensors() enumerates every
// tensor the checkpoint must contain for the text stack: names, dtypes, and
// exact shapes, derived from config fields — not from observation of one
// checkpoint. The table drives two consumers:
//   * glm_validate_text_binding(): offline validation of a real checkpoint
//     (the glm_bind_check app; M4 exit criterion "all quantized matrices
//     bind to validated scale tensors");
//   * the resident loader, which walks the table role by role to copy the
//     compressed E4M3+scale payloads into device memory (no BF16 expansion).
//
// Scale contract (DESIGN §4): every F8_E4M3 matrix X.weight carries an F32
// partner X.weight_scale_inv of shape [ceil(N/128), ceil(K/128)] — 128x128
// dequant blocks, MULTIPLY on dequant. The validator enforces pairing both
// ways: a missing/misshapen scale fails, and an orphan scale is unexpected.
//
// Naming is checkpoint truth, transcribed from the GLM-5.3-Flash revision
// named in docs/checkpoint_budget.md. The MTP draft layer occupies index
// num_hidden_layers and repeats the DSA + MoE layout without mHC tensors.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/glm_config.hpp"

namespace dgpp {

enum class GlmWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,
  Mhc,
  Kda,
  Dsa,
  DsaIndexer,
  DenseMlp,
  Router,
  SharedExpert,
  RoutedExpert,
  Mtp,
};

struct GlmExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  GlmWeightClass cls = GlmWeightClass::LayerNorm;
  int layer = -1;   // layer index; -1 for global tensors
  int expert = -1;  // routed-expert id within the layer; -1 otherwise

  bool quantized() const { return dtype == DType::F8_E4M3; }

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

// Scale-grid shape for a quantized payload of the given [N, K] shape:
// F32 [ceil(N/128), ceil(K/128)] — DESIGN §4's 128x128 dequant blocks.
// Single source for the table generator, the validator, and the loader.
std::vector<int64_t> glm_scale_shape(const std::vector<int64_t>& payload);

// Full text-model table (main layers + MTP layer + globals). Ordered by
// layer then class; expert tensors are grouped per layer.
std::vector<GlmExpectedTensor> glm_expected_text_tensors(
    const GlmTextConfig& cfg);

// Entries for one layer only: `layer` in [0, num_hidden_layers) or
// mtp_layer(). The resident loader sizes itself per layer from this
// (cheap) instead of materializing the 75k-entry full table per call.
std::vector<GlmExpectedTensor> glm_expected_layer_tensors(
    const GlmTextConfig& cfg, int layer);

// A tensor as observed in a checkpoint header (the app maps SafetensorsFile
// entries to this; tests synthesize them directly).
struct GlmTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct GlmBindReport {
  size_t expected = 0;
  size_t matched = 0;           // present with exact dtype+shape
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;        // present, not in table, not vision
  size_t vision = 0;            // present under model.visual.* (not validated)
  size_t quantized_matrices = 0;  // matched F8 payloads
  size_t scales_bound = 0;        // ... whose scale partner validated
  size_t scales_bad = 0;          // missing/mistyped/misshapen scales
  std::vector<std::string> errors;  // capped at max_errors

  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 &&
           unexpected == 0 && scales_bad == 0 &&
           quantized_matrices == scales_bound;
  }
};

// Validates `present` (every tensor in the checkpoint, including vision)
// against the expected table. Vision tensors are counted, not checked.
GlmBindReport glm_validate_text_binding(
    const GlmTextConfig& cfg,
    const std::unordered_map<std::string, GlmTensorDesc>& present,
    size_t max_errors = 32);

}  // namespace dgpp
