#pragma once
// M5 tensor-parallel weight slicing (DESIGN §5.2): builds this rank's
// local views of one RESIDENT layer. The loader's transformed layouts
// are world=1 (full); the TP shard rules are applied on top:
//
//   replicated (full views): mHC, both layer norms, routers, DSA indexer
//     (wq_b/wk/wp/gate/k_norm/ape), DSA latents (qkv_a, q_aln, kv_aln —
//     the latent caches are replicated so every rank selects the same
//     sparse tokens), KDA f_a/g_a, o_norm.
//   head/expert sharded: KDA in_proj/conv (per-section row ranges packed
//     into the layer's fused layout), f_b/g_b/a_log/dt_bias contiguous
//     row ranges, o_proj column packs; DSA q_b/kv_b contiguous head
//     blocks, o_proj column pack; dense and shared-expert gate/up row
//     views with the down projection column-packed.
//   whole experts: routed experts are never split — a rank owns a
//     contiguous expert id range and GlmMoeLayer executes only that
//     partition (the router still scores all experts on every rank).
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

// Quant-matrix row-range view: payload rows are contiguous, so this is a
// pure pointer view — no copy. A 128-ALIGNED row_start re-anchors the scale
// grid exactly (local row r's true block is row_start/128 + r/128, which is
// what the local-frame consumer computes); a MISALIGNED start that crosses
// a block boundary would need two different scale values inside one local
// block — unrepresentable — and is rejected here, loudly. The real
// checkpoint's inter dims (12288 dense / 2048 shared) are 128-multiples at
// every TP world; the loader-era fixture's 200 was this contract's
// counterexample and produced silently wrong scales (measured: layer folds
// 0.30 l2-wrong, invisible to the stream-state metric that attenuated it
// 300x — see the M5 record).
inline GlmQuantMatrix quant_rows_view(const GlmQuantMatrix& m,
                                       int64_t row_start, int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("quant_rows_view: range out of bounds");
  if (row_start % 128 != 0)
    throw std::invalid_argument(
        "quant_rows_view: row_start must be 128-aligned (quantized "
        "scale-grid slice contract; misaligned slices cannot re-anchor "
        "the block scales)");
  const int64_t sb = (m.cols + 127) / 128;  // scale blocks per scale row
  GlmQuantMatrix v;
  v.payload = m.payload + row_start * m.cols;
  v.scales = m.scales + (row_start / 128) * sb;
  v.rows = rows;
  v.cols = m.cols;
  return v;
}

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

  // Exact device bytes of the slice slab (same layout bind carves).
  static size_t slice_bytes(const GlmTextConfig& cfg, int world);

  int rank() const { return rank_; }
  int world() const { return world_; }
  int64_t local_experts() const { return local_experts_; }

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

  GlmTextConfig cfg_;
  KdaConfig kda_cfg_;
  DsaConfig dsa_cfg_;
  KdaGeometry kda_geo_;
  DsaGeometry dsa_geo_;
  GlmMoeConfig moe_cfg_;
  int rank_ = 0;
  int world_ = 1;
  int64_t local_experts_ = 0;
  int64_t dense_inter_ = 0;  // this rank's dense/shared inter slice
  cudaStream_t stream_ = nullptr;
  uint8_t* slab_ = nullptr;
  size_t slab_bytes_ = 0;

  // Last-bound views (owned; rebind overwrites them).
  GlmLayerBound bound_{};
  KdaLayerWeights kda_{};
  DsaLayerWeights dsa_{};
  GlmMoeWeights moe_{};
  GlmQuantMatrix dense_[3]{};
  GlmQuantMatrix shared_[3]{};
};

}  // namespace dgpp
