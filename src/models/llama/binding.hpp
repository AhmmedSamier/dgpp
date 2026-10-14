#pragma once
// Expected-tensor table for Zeta-2.1 (LlamaForCausalLM, 2026-10-14,
// docs/zeta_plan.md §1): every tensor the checkpoint must contain — names,
// dtypes, exact shapes — derived from the parsed config, not observed from
// one file. The table drives the offline validator and (next slice) the
// resident loader.
//
// Naming is checkpoint truth (zed-industries/zeta-2.1 @ 9a83b32): main
// layers under `model.layers.L.`, globals `model.embed_tokens.weight`,
// `model.norm.weight`, `lm_head.weight`. All BF16, dense.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/llama/config.hpp"

namespace dgpp {

enum class LlamaWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,  // input_layernorm, post_attention_layernorm
  Attention,  // q/k/v/o projections
  DenseMlp,   // gate/up/down projections
};

struct LlamaExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  LlamaWeightClass cls = LlamaWeightClass::Attention;
  int layer = -1;  // layer index; -1 for globals

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

std::vector<LlamaExpectedTensor> llama_expected_text_tensors(const LlamaTextConfig& cfg);
std::vector<LlamaExpectedTensor> llama_expected_layer_tensors(const LlamaTextConfig& cfg, int layer);
std::vector<LlamaExpectedTensor> llama_expected_global_tensors(const LlamaTextConfig& cfg);

struct LlamaTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct LlamaBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

LlamaBindReport llama_validate_text_binding(
    const LlamaTextConfig& cfg,
    const std::unordered_map<std::string, LlamaTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (docs/zeta_plan.md §2): throws
// std::invalid_argument naming the dim that does not divide. World must
// divide the query heads; either world divides the kv heads or the kv
// heads divide world (a kv head then lives on world/kv ranks); the MLP
// intermediate size must divide by world.
void llama_tp_validate_geometry(const LlamaTextConfig& cfg, int rank, int world);

}  // namespace dgpp
