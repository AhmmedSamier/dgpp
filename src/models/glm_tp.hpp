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
// Quantized (E4M3 + 128x128 block scale) views offset the scale grid by
// floor(start/128), not start/128: a rank boundary inside a scale block
// shares that block's scale VALUE on both ranks — partial blocks are the
// fixture's deliberate worst case (intermediate_size 200).
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
// pure pointer view — no copy. `row_start` may land inside a 128-row
// scale block; the scale row offset is floor(start/128) and the shared
// block value is correct on both ranks that straddle it.
inline GlmQuantMatrix quant_rows_view(const GlmQuantMatrix& m,
                                      int64_t row_start, int64_t rows) {
  if (row_start < 0 || rows < 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("quant_rows_view: range out of bounds");
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
