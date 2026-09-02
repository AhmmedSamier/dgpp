#pragma once
// M5 tensor-parallel weight slicing (DESIGN §5.2): builds this rank's
// local views of one RESIDENT layer. The loader's transformed layouts
// are world=1 (full); the TP shard rules are applied on top:
//
//   replicated (full views): mHC, both layer norms, routers, DSA indexer
//     (wq_b/wk/wp/gate/k_norm/ape), DSA latents (qkv_a, q_aln, kv_aln —
//     the latent caches are replicated so every rank selects the same
//     sparse tokens), KDA f_a/g_a, o_norm.
//   head/inter sharded: KDA in_proj/conv (per-section row ranges packed
//     into the layer's fused layout), f_b/g_b/a_log/dt_bias contiguous
//     row ranges, o_proj column packs; DSA q_b/kv_b contiguous head
//     blocks, o_proj column pack; dense, shared-expert AND every routed
//     expert's gate/up row views with the down projection column-packed
//     (the routed experts were whole-per-rank until 2026-09-02; slicing
//     them equalizes the ranks' per-token expert bytes — the router still
//     scores all experts on every rank).
//
// Quantized (E4M3 + 128x128 block scale) row/column slices must start
// 128-ALIGNED in the sliced dimension: the local-frame consumer re-anchors
// the scale grid at the slice origin, which is exact only for aligned
// starts. Misaligned starts throw (quant_rows_view here; the column packs
// in bind check the same contract). Non-multiple TAILS at world=1 remain
// the kernel's own masked-tile case — correct and separately covered.
//
// The slice scratch is one slab of FIXED region slots reused per layer
// (exactly one layer is resident at a time — the loader's contract), so
// bind's arithmetic and slice_bytes' sizing share one layout. Pack copies
// enqueue on the model's stream: a bound view is valid once the stream
// drains, and the layer's enqueues follow on the same stream behind them.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "models/glm_loader.hpp"
#include "models/glm_moe.hpp"

namespace dgpp {

// What the forward consumes per layer: full views (world=1) or this
// rank's slice (GlmTpViews). Exactly one attention and one MLP view is
// populated, mirroring GlmLayerResident's convention.
struct GlmLayerBound {
  const GlmMhcResident* mhc = nullptr;
  const uint16_t* ln1 = nullptr;
  const uint16_t* ln2 = nullptr;
  const KdaLayerWeights* kda = nullptr;   // KDA layers only
  const DsaLayerWeights* dsa = nullptr;   // DSA layers only
  const GlmQuantMatrix* dense = nullptr;  // dense-MLP layers only
  const GlmMoeWeights* moe = nullptr;     // MoE layers only
};

// quant_rows_view (the row-range view with the 128-aligned scale-grid
// contract) lives in models/quant_matrix.hpp, next to the type.

class GlmTpViews {
 public:
  // `stream` is the model's compute stream (pack copies enqueue there).
  // Throws legibly when the geometry does not divide by `world`: heads
  // (via the geometry validators), routed experts, and the two inter
  // dims — before any load.
  GlmTpViews(const GlmTextConfig& cfg, int rank, int world,
             cudaStream_t stream);
  ~GlmTpViews();
  GlmTpViews(const GlmTpViews&) = delete;
  GlmTpViews& operator=(const GlmTpViews&) = delete;

  // Binds this rank's views of the resident layer. `dense_mlp` selects
  // the dense-vs-MoE branch exactly as the forward does (the resident
  // layer populates one MLP view). Views stay valid until the next bind.
  GlmLayerBound bind(const GlmLayerResident& r, bool dense_mlp);

  // Binds a SHARDED resident layer (M5 d4): a GlmLayerStream constructed
  // with the SAME rank/world already built the local geometry and did the
  // packs at load time, so this is wholesale pointer wiring plus the
  // expert-partition stamps the loader recorded. No slab copies. The
  // shard-parity test pins this path BITWISE against bind() of a full
  // resident — the two are alternative implementations of one slicing
  // spec (§5.2), and the test is what keeps them from drifting.
  GlmLayerBound bind_sharded(const GlmLayerResident& r, bool dense_mlp);

  // Exact device bytes of the slice slab (same layout bind carves). The
  // routed experts' down packs are NOT in it: bind() allocates that region
  // on first use (see expert_pack_bytes) — it is the parity reference's
  // cost, and production ranks (bind_sharded) never bind a full layer.
  static size_t slice_bytes(const GlmTextConfig& cfg, int world);
  // Device bytes of the routed experts' column-packed down slices.
  static size_t expert_pack_bytes(const GlmTextConfig& cfg, int world);

  int rank() const { return rank_; }
  int world() const { return world_; }

 private:
  // Fixed region slots, 256-byte aligned; one layout feeds the allocator
  // and slice_bytes so the formula cannot drift from the carve.
  struct SlabLayout {
    size_t total = 0;
    size_t off_kda_in_proj = 0, off_kda_conv = 0, off_kda_o_proj = 0;
    size_t off_dsa_o_proj = 0;
    size_t off_dense_down = 0, off_dense_scales = 0;
    size_t off_shared_down = 0, off_shared_scales = 0;
  };
  static SlabLayout layout(const GlmTextConfig& cfg, int world);

  // Strided bf16 block pack on the model stream: `rows` x `width_elems`
  // from a `src_stride_elems`-wide matrix (row or column slice — the
  // caller sets pointer/pitch) into a contiguous destination region.
  void pack2d_bf16(const void* src, size_t src_pitch_bytes, uint16_t* dst,
                   size_t dst_pitch_bytes, size_t width_bytes, size_t rows);
  // Column-slices a quantized [rows, full_cols] matrix into a packed
  // [rows, cols] payload + scale grid at `payload`/`scales` (device).
  GlmQuantMatrix pack_quant_cols(const GlmQuantMatrix& m, int64_t col_start,
                                 int64_t cols, uint8_t* payload,
                                 float* scales);
  void ensure_expert_pack();

  GlmTextConfig cfg_;
  KdaConfig kda_cfg_;
  DsaConfig dsa_cfg_;
  KdaGeometry kda_geo_;
  DsaGeometry dsa_geo_;
  GlmMoeConfig moe_cfg_;
  int rank_ = 0;
  int world_ = 1;
  int64_t dense_inter_ = 0;  // this rank's dense inter slice
  int64_t moe_inter_ = 0;    // this rank's shared/routed inter slice
  cudaStream_t stream_ = nullptr;
  uint8_t* slab_ = nullptr;
  size_t slab_bytes_ = 0;
  uint8_t* expert_pack_ = nullptr;  // lazily allocated (bind() only)

  // Last-bound views (owned; rebind overwrites them).
  GlmLayerBound bound_{};
  KdaLayerWeights kda_{};
  DsaLayerWeights dsa_{};
  GlmMoeWeights moe_{};
  GlmQuantMatrix dense_[3]{};
  GlmQuantMatrix shared_[3]{};
  std::vector<GlmQuantMatrix> experts_;  // [n_experts * 3] sliced views
};

}  // namespace dgpp
