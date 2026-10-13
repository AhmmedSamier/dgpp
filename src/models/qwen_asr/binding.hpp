#pragma once
// Expected-tensor table for the Qwen3-ASR language half (docs/audio.md),
// derived from the parsed config like every other family's: names are
// checkpoint truth - everything under `thinker.`, the LM at
// `thinker.model.layers.L.`, the globals at `thinker.model.*`, and the
// duplicated `thinker.lm_head.weight` (tie_word_embeddings: the loader
// aliases the embedding and only checks the copy's bytes). The tower's
// tensors are listed as unused so the shard scan finds nothing unexpected;
// models/qwen_asr/audio_tower_reference.hpp loads them by key.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/qwen_asr/lm_config.hpp"

namespace dgpp {
enum class QwenAsrWeightClass : int {
  Embed, LmHead, FinalNorm, LayerNorm, Attention, DenseMlp, Tower
};
enum class QwenAsrTensorRole : uint8_t { Plain, Duplicate, Tower };

struct QwenAsrExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  QwenAsrWeightClass cls = QwenAsrWeightClass::Attention;
  int layer = -1;
  QwenAsrTensorRole role = QwenAsrTensorRole::Plain;
  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const {
    return role == QwenAsrTensorRole::Duplicate || role == QwenAsrTensorRole::Tower;
  }
};

std::string qwen_asr_layer_prefix(const QwenAsrLmConfig& cfg, int layer);
std::vector<QwenAsrExpectedTensor> qwen_asr_expected_layer_tensors(const QwenAsrLmConfig& cfg,
                                                                   int layer);
std::vector<QwenAsrExpectedTensor> qwen_asr_expected_global_tensors(const QwenAsrLmConfig& cfg);

struct QwenAsrTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct QwenAsrBindReport {
  size_t expected = 0, matched = 0, missing = 0, dtype_mismatch = 0, shape_mismatch = 0,
         unexpected = 0, tower = 0;
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

QwenAsrBindReport qwen_asr_validate_text_binding(
    const QwenAsrLmConfig& cfg, const std::unordered_map<std::string, QwenAsrTensorDesc>& present,
    size_t max_errors = 32);

// True for the weight classes resident-replicated (world 1 by contract:
// every tensor is).
bool qwen_asr_is_replicated(QwenAsrWeightClass c);

}  // namespace dgpp
