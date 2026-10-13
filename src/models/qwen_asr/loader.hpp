#pragma once
// Qwen3-ASR LM loader: BF16 verbatim through the shared resident stream
// (loaders/resident_stream.hpp). World 1 only; the lm_head is aliased to
// the embedding (the checkpoint's tie_word_embeddings ships a byte-identical
// duplicate, checked at load); the audio tower tensors are listed unused
// and read by models/qwen_asr/audio_tower_reference.hpp.
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/qwen_asr/binding.hpp"

namespace dgpp {
struct QwenAsrAttnResident {
  const uint16_t* q_proj = nullptr;  // BF16 [heads * 128, hidden]
  const uint16_t* k_proj = nullptr;  // BF16 [kv_heads * 128, hidden]
  const uint16_t* v_proj = nullptr;
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, heads * 128]
  const uint16_t* q_norm = nullptr;  // BF16 [128]
  const uint16_t* k_norm = nullptr;
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
};

struct QwenAsrLayerResident {
  int layer = -1;
  const uint16_t* input_norm = nullptr;
  const uint16_t* post_norm = nullptr;
  QwenAsrAttnResident attn;
  const uint16_t* gate = nullptr;  // BF16 [inter, hidden]
  const uint16_t* up = nullptr;
  const uint16_t* down = nullptr;  // BF16 [hidden, inter]
  size_t bytes = 0;
};

struct QwenAsrGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // the embed alias
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

struct QwenAsrLocalGeometry {
  int world = 1, rank = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static QwenAsrLocalGeometry from_config(const QwenAsrLmConfig& cfg, int rank, int world,
                                          LoaderHeadSharding head);
};

struct QwenAsrLoaderFamily {
  using Config = QwenAsrLmConfig;
  using Expected = QwenAsrExpectedTensor;
  using LayerResident = QwenAsrLayerResident;
  using GlobalsResident = QwenAsrGlobalsResident;
  using Geometry = QwenAsrLocalGeometry;
  using PresentMap = std::unordered_map<std::string, QwenAsrTensorDesc>;
  struct Builder;
  static const char* who() { return "qwen_asr loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return qwen_asr_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) {
    return qwen_asr_expected_global_tensors(c);
  }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e) { return !e.unused(); }
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<QwenAsrLoaderFamily>;

class QwenAsrLayerStream : public ResidentLayerStream<QwenAsrLoaderFamily> {
 public:
  QwenAsrLayerStream(const QwenAsrLmConfig& cfg, const std::string& checkpoint_dir);
  ~QwenAsrLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};
}  // namespace dgpp
