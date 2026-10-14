#pragma once
// Zeta-2.1 weight loader using ResidentLayerStream. The family supplies
// tensor bindings and TP geometry for dense BF16 Llama; the shared stream
// owns allocations, staging, resident images, byte accounting and digests.
//
// Placement (every slice a formula in the world size W):
//   attention: 32/W query heads (q_proj rows), 8/W kv heads (k/v rows;
//              a kv head shared by W/8 ranks at W > 8), o_proj packed
//              columns; both norms replicated.
//   MLP:       gate/up rows and down columns at 14336/W.
//   globals:   embed replicated (a row gather), the final norm, lm_head
//              vocab-sharded under VocabSharded.
//
// All matrices are BF16. The projections and the head are packable (the
// model's 12-bit companions replace them under bf12 residency).
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/llama/binding.hpp"
#include "models/llama/config.hpp"

namespace dgpp {

using LlamaResidency = LoaderResidency;
using LlamaHeadSharding = LoaderHeadSharding;
using LlamaReplicatedDigest = ReplicatedDigest;

struct LlamaAttnResident {
  const uint16_t* q_proj = nullptr;  // BF16 [lh * d, hidden]
  const uint16_t* k_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* v_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * d] (packed columns)
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
};

struct LlamaDenseMlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I/W, hidden] (packed rows)
  const uint16_t* up = nullptr;    // BF16 [I/W, hidden] (packed rows)
  const uint16_t* down = nullptr;  // BF16 [hidden, I/W] (packed columns)
  int64_t local_inter = 0;
};

struct LlamaLayerResident {
  int layer = -1;
  const uint16_t* input_norm = nullptr;  // BF16 [hidden]
  const uint16_t* post_norm = nullptr;   // BF16 [hidden]
  LlamaAttnResident attn;
  LlamaDenseMlpResident mlp;
  size_t bytes = 0;
};

struct LlamaGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct LlamaLocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;
  int local_kv_heads = 0, kv_head_begin = 0;
  int64_t local_inter = 0;  // intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static LlamaLocalGeometry from_config(const LlamaTextConfig& cfg, int rank, int world,
                                        LlamaHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct LlamaLoaderFamily {
  using Config = LlamaTextConfig;
  using Expected = LlamaExpectedTensor;
  using LayerResident = LlamaLayerResident;
  using GlobalsResident = LlamaGlobalsResident;
  using Geometry = LlamaLocalGeometry;
  using PresentMap = std::unordered_map<std::string, LlamaTensorDesc>;
  struct Builder;  // models/llama/loader.cpp
  static const char* who() { return "llama loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return llama_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return llama_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t globals_side_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<LlamaLoaderFamily>;

class LlamaLayerStream : public ResidentLayerStream<LlamaLoaderFamily> {
 public:
  LlamaLayerStream(const LlamaTextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                   int world = 1, LlamaResidency residency = LlamaResidency::Streaming,
                   LlamaHeadSharding head = LlamaHeadSharding::Full, bool resident_mtp = false);
  ~LlamaLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
