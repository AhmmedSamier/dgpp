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
#include "models/quant_matrix.hpp"

namespace dgpp {

using LlamaResidency = LoaderResidency;
using LlamaHeadSharding = LoaderHeadSharding;
using LlamaReplicatedDigest = ReplicatedDigest;

struct LlamaAttnResident {
  const uint16_t* q_proj = nullptr;  // BF16 [lh * d, hidden]
  const uint16_t* k_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* v_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * d] (packed columns)
  // engine.dense_weights = "fp8": the same four in block FP8 (E4M3 codes +
  // one fp32 scale per 128 x 128 block, loaders/fp8_quant.hpp); the BF16
  // pointers are null in that mode.
  GlmQuantMatrix q_proj_fp8, k_proj_fp8, v_proj_fp8, o_proj_fp8;
  // A ModelOpt weight-only NVFP4 checkpoint: the same four as e2m1 pairs +
  // e4m3 group-16 scales + one fp32 global (its reciprocal — the kernels
  // divide by it once, models/quant_matrix.hpp); BF16 and FP8 null.
  GlmFp4Matrix q_proj_fp4, k_proj_fp4, v_proj_fp4, o_proj_fp4;
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
};

struct LlamaDenseMlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I/W, hidden] (packed rows)
  const uint16_t* up = nullptr;    // BF16 [I/W, hidden] (packed rows)
  const uint16_t* down = nullptr;  // BF16 [hidden, I/W] (packed columns)
  GlmQuantMatrix gate_fp8, up_fp8, down_fp8;  // dense_weights = "fp8"
  GlmFp4Matrix gate_fp4, up_fp4, down_fp4;    // the ModelOpt NVFP4 checkpoint
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
  GlmQuantMatrix lm_head_fp8;            // dense_weights = "fp8"
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
  static uint64_t loader_format();  // dense_weights fp8 bumps it (models/llama/loader.cpp)
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

  // engine.dense_weights = "fp8": every dense projection and the lm head
  // encode to block FP8 at load (E4M3 codes, one fp32 scale per 128 x 128
  // block; loaders/fp8_quant.hpp) and the layers run the scale GEMM —
  // weight-only quantization, bf16 activations (kernels/scale_gemm.hpp).
  // Set before any model builds (the serve app does, beside the Qwen
  // family's); the resident image format bumps so a mode switch rebuilds.
  static void set_dense_weights_fp8(bool on);
  static bool dense_weights_fp8();

  // A ModelOpt weight-only NVFP4 checkpoint (quantization_config quant_algo
  // W4A16_NVFP4): every projection loads the e2m1/e4m3/fp32-global triple.
  // Checkpoint-driven, not a knob — the static mirrors it for the resident
  // image format (3) beside the fp8 knob's (2).
  static void set_weights_nvfp4(bool on);
  static bool weights_nvfp4();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
