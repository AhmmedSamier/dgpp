#pragma once
// One KDA layer's forward path (M2 deliverable 1): BF16 projections, causal
// depthwise conv, FP32 recurrent update, gated RMSNorm, output projection.
//
// Weight layout (device pointers, caller-owned; the M4 loader materializes
// them from the checkpoint):
//   in_proj: bf16 [in_proj_cols, hidden], rows ordered [f_a | g_a | q | k |
//            v | b] — the runtime-merged projection of the reference
//            (in_proj_qkvbfg_a) with the two replicated shards first so the
//            strided f_b/g_b GEMM inputs land on 16-byte-aligned column
//            offsets. q/k/v/b rows are the head-sharded column-parallel
//            pieces; f_a/g_a rows are replicated across ranks.
//   f_b/g_b: bf16 [local_proj, head_dim] (column-parallel).
//   conv:    bf16 [conv_channels, conv_width] — q|k|v conv weights merged
//            along channels, matching cat([q_conv, k_conv, v_conv], dim=0).
//   a_log:   fp32 [local_heads]; dt_bias: fp32 [local_proj].
//   o_norm:  bf16 [head_dim]; o_proj: bf16 [hidden, local_proj] (row-parallel).
//
// The forward is allocation-free and graph-capturable: scratch lives at
// fixed arena addresses acquired at construction; call prepare() for the
// token counts you will capture BEFORE capture.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "kernels/gemm.hpp"
#include "kernels/kda.hpp"
#include "kernels/l2_prefetch.hpp"
#include "models/kda_geometry.hpp"

namespace dgpp {

// Where a speculative (T > 1 decode) enqueue leaves its post-row state
// snapshots — see KdaStateSnapshots in kernels/kda.hpp. Default: none.
struct KdaSpeculativeSinks {
  KdaStateSnapshots recurrent;
  KdaConvSnapshots conv;
};

struct KdaLayerWeights {
  const void* in_proj = nullptr;
  const void* f_b = nullptr;
  const void* g_b = nullptr;
  const void* conv = nullptr;
  const float* a_log = nullptr;
  const float* dt_bias = nullptr;
  const void* o_norm = nullptr;
  const void* o_proj = nullptr;
};

class KdaLayer {
 public:
  // Acquires fixed scratch for up to max_tokens rows and remembers the
  // shared GEMM workspace (one such buffer can serve every KDA layer).
  KdaLayer(Arena& arena, IGemm& gemm, const KdaLayerWeights& weights,
           const KdaConfig& cfg, int max_tokens, void* gemm_workspace,
           size_t gemm_ws_bytes, float o_norm_eps = 1e-5f);

  // Prebuilds cuBLASLt plans for a token count. Must run outside graph
  // capture; returns false when a heuristic is unavailable.
  bool prepare(int tokens);

  // Streaming-weight seam (M4 diagnostic forward): swap the device weight
  // view this layer enqueues against. Scratch and GEMM plans are
  // shape-keyed, so a same-geometry rebind costs a struct copy. The new
  // view must outlive the next enqueue.
  void rebind(const KdaLayerWeights& w) { w_ = w; }

  // Enqueues the full layer for `tokens` rows on `stream`.
  //   hidden_in:       bf16 [tokens, hidden]
  //   recurrent_state: fp32 [local_heads, head_dim, head_dim], updated in place
  //   conv_state:      bf16 [conv_channels, state_width] (state pool slot
  //                    block); committed history columns are updated in place
  //   out:             bf16 [tokens, hidden]
  //   prefetch:        optional L2 weight prefetcher; after the fused
  //                    in-projection it is pointed at o_proj, so the
  //                    latency-bound middle of the layer (f_b/g_b, conv,
  //                    recurrence, norm) pulls the output projection's
  //                    bytes into L2 instead of leaving DRAM idle.
  //   spec:            optional post-row state snapshot sinks (speculative
  //                    verify rows; rows > accepted are rolled back by the
  //                    caller from these).
  void enqueue(const void* hidden_in, float* recurrent_state,
               uint16_t* conv_state, int conv_state_width, void* out,
               int tokens, cudaStream_t stream,
               WeightPrefetcher* prefetch = nullptr,
               const KdaSpeculativeSinks& spec = {});

  // Bytes of the bf16 output projection [hidden, local_proj].
  size_t o_proj_bytes() const {
    return static_cast<size_t>(cfg_.hidden) * geo_.local_proj * 2;
  }
  // Bytes of the fused bf16 in-projection [in_proj_cols, hidden].
  size_t in_proj_bytes() const {
    return static_cast<size_t>(geo_.in_proj_cols) * cfg_.hidden * 2;
  }

  const KdaConfig& config() const { return cfg_; }
  const KdaGeometry& geometry() const { return geo_; }
  int max_tokens() const { return max_tokens_; }
  // Test/diagnostic probe: the recurrent output buffer [tokens, local_proj]
  // after the most recent enqueue (pre-o_norm core attention values).
  const void* debug_core() const { return core_; }

  // Scratch bytes this layer needs from the arena's persistent hot region.
  static size_t persistent_hot_bytes(const KdaConfig& cfg, int max_tokens);

 private:
  IGemm& gemm_;
  KdaLayerWeights w_;
  KdaConfig cfg_;
  KdaGeometry geo_;
  int max_tokens_ = 0;
  float o_norm_eps_ = 1e-5f;
  float scale_ = 0.f;  // head_dim ** -0.5, computed like the reference (f64 -> f32)
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;

  // Fixed-address scratch (row counts sized to max_tokens).
  uint16_t* proj_ = nullptr;      // [max_tokens, in_proj_cols]
  uint16_t* g1_ = nullptr;        // [max_tokens, local_proj] decay logits
  uint16_t* g2_ = nullptr;        // [max_tokens, local_proj] o_norm gate
  uint16_t* qkv_conv_ = nullptr;  // [max_tokens, conv_channels]
  uint16_t* core_ = nullptr;      // [max_tokens, local_proj] recurrent out
  uint16_t* normed_ = nullptr;    // [max_tokens, local_proj]
};

}  // namespace dgpp
