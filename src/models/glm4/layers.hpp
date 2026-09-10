#pragma once
// The GLM-4.7 layer objects (2026-09-09, docs/glm47_plan.md §1.2, D4):
// each one wires the kernels and the GEMM seam into one site of the
// layer, owns shape-keyed scratch, and is REBOUND to each layer's resident
// weight views (the streaming diagnostic forward keeps one layer resident
// at a time). Host-orchestrated and graph-capturable alike (fixed grids,
// no host reads on the decode rows).
//
//   Glm4AttentionLayer  q/k/v projections (fp32 out), the fused bias +
//                       head norm + half-split RoPE + paged K/V append,
//                       the split-KV GQA attention and its combine, o_proj
//   Glm4DenseMlp        the NVFP4 dense MLP of the first layers (gate/up,
//                       swiglu, down) on the fp4 GEMV core (decode rows)
//                       or the fp4 tile kernel (prefill)
// The MoE layers run GlmMoeLayer directly (the GLM router and chain, the
// NVFP4 shared expert as view-table entry E).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/glm4/config.hpp"
#include "models/glm4/kv_pool.hpp"
#include "models/glm4/loader.hpp"

namespace dgpp {

// The GEMM seam's workspace, shared by every layer object of a model.
struct Glm4GemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

// The rows one attention enqueue serves: device [T] request ids and
// positions (a padding row has pos < 0), and whether they are decode rows
// (every row's own split geometry, kernels-only) or a prefill chunk of one
// request (positions pos0 .. pos0 + T - 1, one split).
struct Glm4AttnRows {
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  bool decode = false;
};

class Glm4AttentionLayer {
 public:
  // decode_rows: the most rows a decode enqueue carries (the engine's
  // kDecodeRows); n_split_decode: the split-KV geometry of the decode rows
  // (a capture bakes it in) — the partials' workspace holds max(max_tokens,
  // decode_rows * n_split_decode) row-splits; prefill rows run one split.
  Glm4AttentionLayer(const Glm4AttnResident& w, const Glm4GemmWorkspace& gemm, const Glm4TextConfig& cfg,
                     int max_tokens, int decode_rows, int n_split_decode);
  ~Glm4AttentionLayer();
  Glm4AttentionLayer(const Glm4AttentionLayer&) = delete;
  Glm4AttentionLayer& operator=(const Glm4AttentionLayer&) = delete;
  void rebind(const Glm4AttnResident& w);

  // out[T, H] = o_proj(attention(x[T, H])): the projections, the finish
  // (K/V of every row appended to `cache` at its slot), the attention over
  // [0, pos] per row, the output projection into `out` (a partial at
  // world > 1).
  void enqueue(const uint16_t* x, int tokens, const Glm4AttnRows& rows, const Glm4KvCache& cache,
               uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const Glm4TextConfig& cfg, int local_heads, int local_kv_heads, int max_tokens,
                              int decode_rows, int n_split_decode);
  // The default decode split count (DGPP_GLM4_ATTN_SPLITS overrides).
  static int default_decode_splits();
  int local_heads() const { return lh_; }
  int local_kv_heads() const { return lkv_; }
  int n_split_decode() const { return n_split_; }

 private:
  Glm4AttnResident w_;
  Glm4GemmWorkspace g_;
  int hidden_, lh_, lkv_, dim_, rotary_, max_tokens_, decode_rows_, n_split_;
  size_t part_rows_ = 0;  // row-splits the partial workspaces hold
  float eps_, scale_;
  float* d_inv_freq_ = nullptr;  // [rotary/2]
  float* qd_ = nullptr;          // [M, lh * D] fp32
  float* kd_ = nullptr;          // [M, lkv * D] fp32
  float* vd_ = nullptr;          // [M, lkv * D] fp32
  uint16_t* q_ = nullptr;        // [M, lh * D] bf16 (finished)
  float* m_ws_ = nullptr;        // [part_rows, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;        // [part_rows, lh, D]
  uint16_t* o_ = nullptr;        // [M, lh * D] bf16
};

class Glm4DenseMlp {
 public:
  Glm4DenseMlp(const Glm4DenseMlpResident& w, const Glm4TextConfig& cfg, int max_tokens, int decode_rows);
  ~Glm4DenseMlp();
  Glm4DenseMlp(const Glm4DenseMlp&) = delete;
  Glm4DenseMlp& operator=(const Glm4DenseMlp&) = delete;
  void rebind(const Glm4DenseMlpResident& w);
  // out[T, H] = down(silu(gate(x)) * up(x)); decode rows (T <= decode_rows)
  // on the fp4 GEMV core, more on the fp4 tile kernel.
  void enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream);
  static size_t scratch_bytes(const Glm4TextConfig& cfg, int64_t local_inter, int max_tokens);

 private:
  Glm4DenseMlpResident w_;
  int hidden_, max_tokens_, decode_rows_;
  int64_t inter_;
  uint16_t* gate_ = nullptr;  // [M, I]
  uint16_t* up_ = nullptr;    // [M, I]
  uint16_t* act_ = nullptr;   // [M, I]
};

}  // namespace dgpp
