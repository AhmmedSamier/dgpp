#pragma once
// MiMo-V2.6-Flash layers composed from CUDA kernels, with shape-specific
// scratch and rebindable weight views. Decode uses fixed grids without
// host readbacks, allowing graph capture; streaming mode rebinds the
// layers as weights are loaded.
//
//   MimoAttentionLayer  the fused fp8 qkv projection (fp32 out: the
//                       streaming tensor-core GEMM at decode rows, the
//                       GEMV chunks / tile kernel on prefill chunks), the
//                       finish (partial RoPE, the value scale, the paged
//                       K/V append), the split-KV GQA attention with its
//                       window and sink, the combine, the bf16 o_proj
//                       through the GEMM interface (its 12-bit companion at
//                       decode rows)
//   MimoDenseMlp        the fp8 dense MLP of layer 0 and the draft (gate/up,
//                       swiglu, down) on the same fp8 cores
// The MoE layers run GlmMoeLayer directly (the sigmoid router, the MXFP4
// routed chain, no shared expert).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/mimo/config.hpp"
#include "models/mimo/kv_pool.hpp"
#include "models/mimo/loader.hpp"

namespace dgpp {

// The GEMM interface's workspace, shared by every layer object of a model.
struct MimoGemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

// The rows one attention enqueue serves: device [T] request ids and
// positions (a padding row has pos < 0), and whether they are decode rows
// (every row's own split geometry, kernels-only) or a prefill chunk of one
// request (positions pos0 .. pos0 + T - 1, one split).
struct MimoAttnRows {
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  bool decode = false;
};

class MimoAttentionLayer {
 public:
  // decode_rows: the most rows a decode enqueue carries (the engine's
  // kDecodeRows); n_split_decode: the split-KV geometry of a global
  // layer's decode rows (a sliding-window layer's rows split at most per
  // tile of its window: window / 32) — the partials' workspace holds
  // max(max_tokens, decode_rows * n_split_decode) row-splits; prefill rows
  // run one split. The layer serves both kinds: rebind() switches the
  // window, the sink and the kv head count with the weights.
  MimoAttentionLayer(const MimoAttnResident& w, const MimoGemmWorkspace& gemm, const MimoTextConfig& cfg,
                     int max_tokens, int decode_rows, int n_split_decode, bool decode_mma);
  ~MimoAttentionLayer();
  MimoAttentionLayer(const MimoAttentionLayer&) = delete;
  MimoAttentionLayer& operator=(const MimoAttentionLayer&) = delete;
  void rebind(const MimoAttnResident& w);

  // out[T, H] = o_proj(attention(x[T, H])): the projection, the finish
  // (K/V of every row appended to `cache` at its slot), the attention over
  // the row's visible range, the output projection into `out` (a partial
  // at world > 1).
  void enqueue(const uint16_t* x, int tokens, const MimoAttnRows& rows, const MimoKvCache& cache,
               uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const MimoTextConfig& cfg, int local_heads, int max_tokens, int decode_rows,
                              int n_split_decode);
  // The default decode split count of the global layers
  // (DGPP_MIMO_ATTN_SPLITS overrides).
  static int default_decode_splits();
  int local_heads() const { return lh_; }
  int n_split_decode() const { return n_split_; }
  // The split count a layer's decode rows take: the window's tiles for a
  // sliding-window layer, n_split_decode for a global one.
  int decode_splits_of(const MimoAttnResident& w) const;

 private:
  MimoAttnResident w_;
  MimoGemmWorkspace g_;
  int hidden_, lh_, max_tokens_, decode_rows_, n_split_;
  int window_ = 0;
  bool decode_mma_ = true;
  size_t part_rows_ = 0;  // row-splits the partial workspaces hold
  int64_t qkv_cols_ = 0;  // the fp32 projection buffer's row stride
  float scale_;
  float value_scale_;
  float* d_inv_freq_ga_ = nullptr;  // [32] the global layers' rotary slice
  float* d_inv_freq_swa_ = nullptr; // [32] the sliding-window layers'
  float* qkv_ = nullptr;            // [M, qkv_cols] fp32
  uint16_t* q_ = nullptr;           // [M, lh * 192] bf16 (finished)
  float* m_ws_ = nullptr;           // [part_rows, lh]
  float* l_ws_ = nullptr;
  float* c_ws_ = nullptr;           // [part_rows, lh, 128]
  int* counters_ = nullptr;         // [decode_rows, kv heads] the fused kernel's arrival counters (zero at rest)
  bool fused_decode_ = true;        // DGPP_MIMO_ATTN_FUSED=0: the three-kernel chain for decode rows too
  bool tiled_prefill_ = true;       // DGPP_MIMO_PREFILL_ATTN=chain: the split-KV kernel for prefill rows
  uint16_t* o_ = nullptr;           // [M, lh * 128] bf16
};

class MimoDenseMlp {
 public:
  MimoDenseMlp(const MimoDenseMlpResident& w, const MimoTextConfig& cfg, int max_tokens, int decode_rows,
               bool decode_mma);
  ~MimoDenseMlp();
  MimoDenseMlp(const MimoDenseMlp&) = delete;
  MimoDenseMlp& operator=(const MimoDenseMlp&) = delete;
  void rebind(const MimoDenseMlpResident& w);
  // out[T, H] = down(silu(gate(x)) * up(x)) (a partial at world > 1).
  void enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream);
  static size_t scratch_bytes(const MimoTextConfig& cfg, int64_t local_inter, int max_tokens);

 private:
  MimoDenseMlpResident w_;
  int hidden_, max_tokens_, decode_rows_;
  int64_t inter_;
  bool decode_mma_ = true;
  uint16_t* gate_ = nullptr;  // [M, I]
  uint16_t* up_ = nullptr;    // [M, I]
  uint16_t* act_ = nullptr;   // [M, I]
};

}  // namespace dgpp
