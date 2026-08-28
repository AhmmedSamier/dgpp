#pragma once
// Resident weight loader for the GLM text model (M4 deliverable 2,
// DESIGN §4). Loads one layer at a time from the mapped checkpoint into
// managed device memory, transformed into the exact layouts the M2/M3 layer
// kernels consume: merged KDA in_proj ([f_a|g_a|q|k|v|b] rows) and conv
// (q|k|v channels), fused DSA qkv_a, F32 indexer APE. The E4M3+scale pairs
// of the MLP/MoE matrices stay resident in compressed form — "load" never
// means a persistent BF16 expansion of the model. The four DSA attention
// matrices (q_a, kv_a, q_b, o_proj) are dequantized per layer as a
// documented TRANSIENT bridge for the M3 IGemm seam (bf16 weights); the M4
// scale-aware GEMM replaces that bridge.
//
// Streaming model (M4 deliverable 4): exactly one layer is resident at a
// time — a bump allocation reset per load, sized to the largest layer.
// Globals (embed/lm_head/final norm) load once and persist. Peak device
// memory is therefore largest-layer + globals, which is what makes
// full-model correctness testable on a single node without the TP placement.
//
// Synchronization contract: load_layer/load_globals cudaDeviceSynchronize
// before returning, so callers may read the managed buffers from the CPU
// and must not touch them while later GPU work runs.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsa_layer.hpp"
#include "models/glm_binding.hpp"
#include "models/glm_config.hpp"
#include "models/kda_layer.hpp"

namespace dgpp {

// Managed-memory bump (defined in glm_loader.cu): aligned grants, reset per
// layer; counting mode walks the same grant sequence without allocating,
// which is how the byte formula and the allocator share one code path.
struct GlmLayerBump;

// Compressed resident view of one E4M3 matrix: payload + block scales,
// nothing else (DESIGN §4). Pointers are device-visible (managed) memory.
struct GlmQuantMatrix {
  const uint8_t* payload = nullptr;  // E4M3 [rows, cols]
  const float* scales = nullptr;     // F32 [ceil(rows/128), ceil(cols/128)]
  int64_t rows = 0;
  int64_t cols = 0;
};

// One layer's routed-expert neighborhood in compressed form.
struct GlmMoeResident {
  const uint16_t* router_gate = nullptr;  // BF16 [n_routed_experts, hidden]
  const float* router_bias = nullptr;     // F32 [n_routed_experts]
  GlmQuantMatrix shared[3];               // gate, up, down
  std::vector<GlmQuantMatrix> experts;    // gate, up, down per expert
  const GlmQuantMatrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
};

// Multi-head hyper-connection coefficients (all replicated across TP).
struct GlmMhcResident {
  const float* attn_base = nullptr;   // F32 [24]
  const uint16_t* attn_fn = nullptr;  // BF16 [24, hc_mult * hidden]
  const float* attn_scale = nullptr;  // F32 [3]
  const float* ffn_base = nullptr;
  const uint16_t* ffn_fn = nullptr;
  const float* ffn_scale = nullptr;
};

struct GlmLayerResident {
  int layer = -1;
  GlmLayerKind kind = GlmLayerKind::Kda;
  GlmMhcResident mhc;  // empty on the MTP draft layer (it has none)
  const uint16_t* ln1 = nullptr;  // BF16 [hidden]
  const uint16_t* ln2 = nullptr;
  // Exactly one attention view is populated (by layer kind).
  KdaLayerWeights kda;  // merged in_proj/conv layouts (M2 kernel contract)
  DsaLayerWeights dsa;  // bf16 layouts (M3 kernel contract; transient bridge)
  // Exactly one MLP view is populated (by layer MLP class).
  GlmQuantMatrix dense[3];  // gate, up, down
  GlmMoeResident moe;
  // MTP draft head (populated only for layer == mtp_layer()).
  const uint16_t* enorm = nullptr;         // BF16 [hidden]
  const uint16_t* hnorm = nullptr;         // BF16 [hidden]
  const uint16_t* eh_proj = nullptr;       // BF16 [hidden, 2 * hidden]
  const uint16_t* shared_head_norm = nullptr;  // BF16 [hidden]
  size_t bytes = 0;  // exact device bytes this layer occupies
};

struct GlmGlobalsResident {
  const uint16_t* embed = nullptr;      // BF16 [vocab, hidden]
  const uint16_t* lm_head = nullptr;    // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  size_t bytes = 0;
};

class GlmLayerStream {
 public:
  // Opens every safetensors shard in `checkpoint_dir` (sorted by name) and
  // validates the full text binding (glm_validate_text_binding); throws on
  // any mismatch. Allocates the layer bump at the max layer size.
  GlmLayerStream(const GlmTextConfig& cfg, const std::string& checkpoint_dir);
  ~GlmLayerStream();
  GlmLayerStream(const GlmLayerStream&) = delete;
  GlmLayerStream& operator=(const GlmLayerStream&) = delete;

  // Loads layer `layer` (0..num_hidden_layers-1, or mtp_layer() when the
  // draft layer is present). Frees the previously resident layer first.
  // The returned view stays valid until the next load_layer/release_layer.
  const GlmLayerResident& load_layer(int layer);

  // Loads embed/lm_head/final norm once; persists across load_layer calls.
  const GlmGlobalsResident& load_globals();

  void release_layer();
  void release_globals();

  // Exact device bytes load_layer will use for a layer — the same formula
  // that sizes the bump; load_layer throws if actual usage ever differs,
  // so the formula and the allocator cannot silently drift apart.
  static size_t layer_bytes(const GlmTextConfig& cfg, int layer);
  static size_t globals_bytes(const GlmTextConfig& cfg);

  const GlmTextConfig& config() const { return cfg_; }
  // Capacity of the per-layer bump (the largest layer's exact size).
  size_t layer_capacity() const;

 private:
  GlmTextConfig cfg_;
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::unordered_map<std::string, const TensorInfo*> tensors_;
  std::unique_ptr<GlmLayerBump> layer_bump_;
  std::unique_ptr<GlmLayerBump> globals_bump_;
  GlmLayerResident resident_;
  GlmGlobalsResident globals_;
  cudaStream_t stream_ = nullptr;  // dedicated; synced before returning
};

}  // namespace dgpp
