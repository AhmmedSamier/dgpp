#pragma once
// Nemotron-3.5-Transcribe weight loader (milestone 4 of
// docs/nemotron_transcribe_plan.md). Uses the shared
// `ResidentLayerStream` lifecycle (loaders/resident_stream.hpp): checkpoint
// mappings, binding validation against `binding.hpp`'s expected-tensor
// table, device staging, resident images and byte accounting.
//
// The checkpoint is pure BF16 (plus the Mamba/F32 scalars), no packing, no
// quantized experts — so the loader is a straightforward name->device
// pointer map. TP is not yet supported: world must be 1 and the whole
// vocabulary and every expert live on rank 0.
//
// MTP (the `--mtp` draft block) is not yet loaded by this loader; it lands
// as a follow-up once the base forward serves.
//
// Per-slot layout (checkpoint names, `llm.model.layers.<slot>.mixer.<tensor>`):
//   norm.weight                  [H]          BF16  (pre-norm, every slot)
//   Mamba slot:
//     in_proj.weight             [in_proj, H] BF16
//     conv1d.weight              [C, 1, K]    BF16
//     conv1d.bias                [C]          BF16
//     _fp32_params.{A_log,D,dt_bias} [heads]  F32
//     norm.weight                [d_inner]    BF16  (gated norm)
//     out_proj.weight            [H, d_inner] BF16
//   Attention slot:
//     q_proj.weight              [q_rows, H]  BF16
//     k_proj.weight              [kv_rows, H] BF16
//     v_proj.weight              [kv_rows, H] BF16
//     o_proj.weight              [H, q_rows]  BF16
//   MoE slot:
//     gate.weight                [E, H]       BF16
//     gate.e_score_correction_bias [E]         F32
//     experts.gate_and_up_projs  [E, H, I]    BF16
//     experts.down_projs         [E, I, H]    BF16
//     shared_experts.up_proj.weight   [SI, H] BF16
//     shared_experts.down_proj.weight [H, SI] BF16

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/nemotron_transcribe/binding.hpp"
#include "models/nemotron_transcribe/config.hpp"

namespace dgpp {

using NemotronResidency = LoaderResidency;
using NemotronHeadSharding = LoaderHeadSharding;

struct NemotronLayerResident {
  int layer = -1;
  bool moe = false;
  // The per-slot pre-norm (applied before every mixer).
  const uint16_t* input_norm = nullptr;  // BF16 [H]

  // Mamba-2 mixer.
  const uint16_t* mamba_in_proj = nullptr;      // BF16 [in_proj_rows, H]
  const uint16_t* mamba_conv_weight = nullptr;  // BF16 [conv_channels, 1, conv_kernel]
  const uint16_t* mamba_conv_bias = nullptr;    // BF16 [conv_channels]
  const float* mamba_A_log = nullptr;           // F32 [mamba_num_heads]
  const float* mamba_D = nullptr;               // F32 [mamba_num_heads]
  const float* mamba_dt_bias = nullptr;         // F32 [mamba_num_heads]
  const uint16_t* mamba_norm = nullptr;         // BF16 [d_inner] (gated norm)
  const uint16_t* mamba_out_proj = nullptr;     // BF16 [H, d_inner]

  // GQA attention mixer (position-free: no RoPE, no q/k norm).
  const uint16_t* attn_q = nullptr;  // BF16 [q_rows, H]
  const uint16_t* attn_k = nullptr;  // BF16 [kv_rows, H]
  const uint16_t* attn_v = nullptr;  // BF16 [kv_rows, H]
  const uint16_t* attn_o = nullptr;  // BF16 [H, q_rows]

  // relu2 MoE mixer.
  const uint16_t* moe_gate = nullptr;              // BF16 [E, H]
  const float* moe_gate_bias = nullptr;            // F32 [E]
  const uint16_t* moe_experts_gate_and_up = nullptr;  // BF16 [E, H, I]
  const uint16_t* moe_experts_down = nullptr;      // BF16 [E, I, H]
  const uint16_t* moe_shared_up = nullptr;         // BF16 [SI, H]
  const uint16_t* moe_shared_down = nullptr;       // BF16 [H, SI]

  size_t bytes = 0;
};

struct NemotronGlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, H]
  int embed_vocab_begin = 0;
  int embed_vocab_count = 0;
  const uint16_t* final_norm = nullptr;  // BF16 [H]
  const uint16_t* lm_head = nullptr;     // BF16 [vocab, H]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// Local TP geometry (world == 1 for now: everything local).
struct NemotronLocalGeometry {
  int world = 1, rank = 0;
  int64_t local_inter = 0;     // moe_intermediate_size (no slicing at world 1)
  int64_t local_shared_inter = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int embed_vocab_begin = 0, embed_vocab_count = 0;

  static NemotronLocalGeometry from_config(const NemotronTranscribeLmConfig& lm,
                                           int rank, int world,
                                           LoaderHeadSharding head);
  static NemotronLocalGeometry from_config(const NemotronTranscribeConfig& cfg,
                                           int rank, int world,
                                           LoaderHeadSharding head);
};

struct NemotronLoaderFamily {
  using Config = NemotronTranscribeConfig;
  using Expected = NemotronExpectedTensor;
  using LayerResident = NemotronLayerResident;
  using GlobalsResident = NemotronGlobalsResident;
  using Geometry = NemotronLocalGeometry;
  using PresentMap = std::unordered_map<std::string, NemotronTensorDesc>;
  struct Builder;  // loader.cpp

  static const char* who() { return "nemotron_transcribe loader"; }
  static uint64_t loader_format() { return 1u; }
  static int max_layer(const Config& c) { return c.lm.num_layers; }
  static int main_layers(const Config& c) { return c.lm.num_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer);
  static std::vector<Expected> global_table(const Config& c);
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected&) { return true; }
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config& c, const Geometry& geo,
                            const LoaderTensorMap& tensors, LayerBump& bump,
                            GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world,
                              LoaderHeadSharding head);
  static size_t globals_side_bytes(const Config&, int, int,
                                   LoaderHeadSharding) {
    return 0;
  }
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&,
                            LayerResident&) {}
};

extern template class ResidentLayerStream<NemotronLoaderFamily>;

class NemotronLayerStream : public ResidentLayerStream<NemotronLoaderFamily> {
 public:
  NemotronLayerStream(const NemotronTranscribeConfig& cfg,
                      const std::string& checkpoint_dir, int rank = 0,
                      int world = 1,
                      NemotronResidency residency = NemotronResidency::Streaming,
                      NemotronHeadSharding head = NemotronHeadSharding::Full,
                      bool resident_mtp = false);
  ~NemotronLayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

  const std::string& image_dir() const override;

 private:
  NemotronTranscribeConfig cfg_;
};

}  // namespace dgpp
