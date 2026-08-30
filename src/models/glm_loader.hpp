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
// Sharded build (M5 d4): world>1 produces the resident layer directly at
// this rank's TP geometry — the same layouts GlmTpViews::bind would carve
// from a full layer, pinned bitwise by glm_tp_test's shard-parity test.
// world=1 stays the degenerate rank 0 (the M4 build, byte-for-byte).
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
#include "models/quant_matrix.hpp"
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

// One layer's routed-expert neighborhood in compressed form. At world>1
// (M5 d4) `experts` holds ONLY this rank's contiguous whole-expert range,
// [expert_begin, expert_begin + expert_count); world=1 keeps the M4
// "every expert" sentinel (count -1) so the single-rank forward is
// byte-identical.
struct GlmMoeResident {
  const uint16_t* router_gate = nullptr;  // BF16 [n_routed_experts, hidden]
  const float* router_bias = nullptr;     // F32 [n_routed_experts]
  GlmQuantMatrix shared[3];               // gate, up, down
  std::vector<GlmQuantMatrix> experts;    // gate, up, down per LOCAL expert
  int expert_begin = 0;    // first global expert id resident here
  int expert_count = -1;   // -1 = every expert (the M4 single-rank default)
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

// Boot-time digest over REPLICATED weight source bytes (DESIGN §5.2: "boot
// checks hash all replicated tensors"). One order-independent sum-hash per
// layer (MTP last) plus the globals: ranks compare element-wise, so a
// mismatch pinpoints the layer. Each tensor folds as FNV-1a over its name
// then its raw checkpoint bytes, summed per layer — commutative, so load
// order cannot change the value. Not adversarial: this is a
// checksum-class agreement check between ranks loading the same files,
// not a content-authentication hash.
//
// Replicated in v1: mHC, both layer norms, routers, the DSA indexer and
// APE, the DSA latents (q_a/kv_a/their norms), KDA f_a/g_a/o_norm, the
// MTP draft head, and the globals (embed/lm_head/final_norm load full on
// every rank until the M6 vocabulary-sharded lm-head seam). The DSA q_b/
// o_proj dequant bridges are NOT here — they are read in full by every
// rank (the bf16 seam), but their resident outputs are sharded; they
// count as full-read in the byte reconcile instead.
struct GlmReplicatedDigest {
  std::vector<uint64_t> layer;  // per layer 0..max_layer-1
  uint64_t globals = 0;
  uint64_t bytes = 0;    // total folded source bytes
  uint64_t tensors = 0;  // total folded tensors
};

class GlmLayerStream {
 public:
  // Opens every safetensors shard in `checkpoint_dir` (sorted by name) and
  // validates the full text binding (glm_validate_text_binding); throws on
  // any mismatch. Allocates the layer bump at the max layer size.
  //
  // Sharded load (M5 d4): `world` > 1 builds each resident layer DIRECTLY
  // at this rank's local geometry — head row ranges, contiguous
  // whole-expert ranges, and 128-aligned quantized row/column slices (the
  // §5.2 scale-grid contract) — so only rank-local checkpoint bytes are
  // ever read. world=1 is the degenerate rank 0: the M4 full-geometry
  // build, byte-for-byte (same grant sequence, same bytes — the layer
  // formula cannot drift). TP geometry is validated BEFORE any shard is
  // opened; misaligned inter quotients fail there, loudly.
  GlmLayerStream(const GlmTextConfig& cfg, const std::string& checkpoint_dir,
                 int rank = 0, int world = 1);
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
  // so the formula and the allocator cannot silently drift apart. At
  // world>1 this is the LOCAL geometry's formula (the sharded bump).
  static size_t layer_bytes(const GlmTextConfig& cfg, int layer,
                            int rank = 0, int world = 1);
  static size_t globals_bytes(const GlmTextConfig& cfg);

  // Registers the stream whose kernels READ resident layers (the model's
  // compute stream). When set, load boundaries synchronize ONLY that
  // stream plus the loader's own dequant stream — the complete set of
  // bump readers — instead of the whole device. The device-wide wait is
  // correct for standalone callers (conservative default), but in a
  // ONE-PROCESS multi-rank world it deadlocks by construction: rank A's
  // spinning collective kernel never completes, so rank B's
  // cudaDeviceSynchronize inside a layer load never returns, so B never
  // posts the doorbell A spins on (measured: first-collective stall, CI
  // under post-build load — ~15ms of thread skew is enough). Peers'
  // kernels never touch this rank's bump; the precise sync is strictly
  // safer than the device-wide one everywhere, including the fabric.
  void set_reader_stream(cudaStream_t reader) { reader_ = reader; }

  const GlmTextConfig& config() const { return cfg_; }
  // Capacity of the per-layer bump (the largest layer's exact size at
  // this rank's geometry).
  size_t layer_capacity() const;

  int rank() const { return rank_; }
  int world() const { return world_; }

  // Folds every replicated tensor's raw source bytes (all layers + the
  // globals) straight from the mmaps — no layer residency required, which
  // is what makes it a BOOT check. Rank-invariant by construction.
  GlmReplicatedDigest hash_replicated() const;

  // Checkpoint source bytes this rank has TOUCHED across all loads — the
  // "reconcile per-rank byte totals" input (§5.2): the sharded-class bytes
  // partition across ranks exactly once, the replicated+bridge bytes are
  // re-read by every rank. verbatim_source_bytes() is that rank-invariant
  // re-read subset, so sum_r(source) == world1_total + (world-1) *
  // verbatim reconciles the shard coverage arithmetically.
  uint64_t source_bytes_read() const { return source_bytes_; }
  uint64_t verbatim_source_bytes() const { return verbatim_bytes_; }

 private:
  GlmTextConfig cfg_;
  int rank_ = 0;
  int world_ = 1;
  cudaStream_t reader_ = nullptr;  // bump readers' stream (see above)
  uint64_t source_bytes_ = 0;
  uint64_t verbatim_bytes_ = 0;
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::unordered_map<std::string, const TensorInfo*> tensors_;
  std::unique_ptr<GlmLayerBump> layer_bump_;
  std::unique_ptr<GlmLayerBump> globals_bump_;
  GlmLayerResident resident_;
  GlmGlobalsResident globals_;
  cudaStream_t stream_ = nullptr;  // dedicated; synced before returning
};

}  // namespace dgpp
