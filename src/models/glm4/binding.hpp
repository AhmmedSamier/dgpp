#pragma once
// Expected-tensor table for GLM-4.7 (Glm4MoeForCausalLM, 2026-09-09,
// docs/glm47_plan.md §1.4): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the GLM-5.3 and Qwen tables do.
//
// Naming is checkpoint truth (nvidia/GLM-4.7-NVFP4 @ 47fa7dc8): main
// layers under `model.layers.L.`, the draft layer under
// `model.layers.<num_hidden_layers>.` (in mtp.safetensors, with its own
// copies of the embedding and the head under `embed_tokens.weight` and
// `shared_head.head.weight`), the globals `model.embed_tokens.weight`,
// `model.norm.weight`, `lm_head.weight`.
//
// Format contract (modelopt NVFP4): every quantized [N, K] matrix X is the
// triple X.weight U8 [N, K/2] (e2m1 pairs), X.weight_scale F8_E4M3
// [N, K/16], X.weight_scale_2 F32 [] (the per-tensor scale; dequant =
// e2m1 x e4m3 x weight_scale_2), plus X.input_scale F32 [] (unused). The
// attention projections, norms, router, embeddings and head are BF16;
// the router bias is F32; the FP8 KV scales k_proj.k_scale /
// v_proj.v_scale are F32 [] and unused. The draft layer is BF16
// throughout (its experts are requantized at load).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/glm4/config.hpp"

namespace dgpp {

enum class Glm4WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // input_layernorm, post_attention_layernorm
  Attention,      // q/k/v/o projections, biases, q/k norms
  KvScale,        // k_scale / v_scale (unused)
  Router,         // mlp.gate.weight, mlp.gate.e_score_correction_bias
  DenseMlp,       // layers < first_k_dense_replace
  SharedExpert,
  RoutedExpert,
  MtpHead,        // enorm, hnorm, eh_proj, shared_head.norm, the draft's embed/head copies
};

enum class Glm4TensorRole : uint8_t {
  Plain,
  Fp4Payload,   // U8 [N, K/2]
  Fp4Scale,     // F8_E4M3 [N, K/16]
  Fp4Global,    // F32 [] — weight_scale_2
  InputScale,   // F32 [] — the W4A4 activation scale (unused)
  KvScale,      // F32 [] — the FP8 KV scale (unused)
  Bf16Expert,   // BF16 [N, K] — a draft-layer expert matrix, requantized at load
  Duplicate,    // BF16 — the draft's copies of the embedding / head (checked, not loaded)
};

struct Glm4ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Glm4WeightClass cls = Glm4WeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals; mtp_layer() for the draft
  int expert = -1;  // routed-expert id, -1 otherwise
  Glm4TensorRole role = Glm4TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const {
    return role == Glm4TensorRole::InputScale || role == Glm4TensorRole::KvScale ||
           role == Glm4TensorRole::Duplicate;
  }
};

// The checkpoint name prefix of a layer ("model.layers.L." for main and
// draft layers alike).
std::string glm4_layer_prefix(const Glm4TextConfig& cfg, int layer);

// The four names of one NVFP4 matrix: base + ".weight" / ".weight_scale" /
// ".weight_scale_2" / ".input_scale" (`base` is e.g. "...gate_proj").
std::vector<Glm4ExpectedTensor> glm4_expected_text_tensors(const Glm4TextConfig& cfg);
std::vector<Glm4ExpectedTensor> glm4_expected_layer_tensors(const Glm4TextConfig& cfg, int layer);
std::vector<Glm4ExpectedTensor> glm4_expected_global_tensors(const Glm4TextConfig& cfg);

struct Glm4TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Glm4BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;
  size_t bf16_expert_matrices = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Glm4BindReport glm4_validate_text_binding(
    const Glm4TextConfig& cfg,
    const std::unordered_map<std::string, Glm4TensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/glm47_plan.md §2): throws
// std::invalid_argument naming the dim that does not divide. world must
// divide the query heads; either world divides the kv heads or the kv
// heads divide world (a kv head then lives on world/kv ranks); every
// expert / dense intermediate slice must be a multiple of 32 (the NVFP4
// GEMV core's K and the 16-block column slice).
void glm4_tp_validate_geometry(const Glm4TextConfig& cfg, int rank, int world);

}  // namespace dgpp
