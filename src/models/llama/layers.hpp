#pragma once
// Llama layers composed from CUDA kernels and GEMM interfaces, with
// shape-specific scratch and rebindable weight views. Decode uses fixed
// grids without host readbacks, allowing graph capture; streaming mode
// rebinds the layers as weights are loaded.
//
//   LlamaAttentionLayer  q/k/v projections (fp32 out), the half-split
//                        full RoPE + paged K/V append (no bias, no head
//                        norm — kernels/glm4_attn.hpp at its no-norm
//                        point), the split-KV GQA attention and its
//                        combine, o_proj
//   LlamaDenseMlp        gate/up, swiglu, down on the GEMM interface
//                        (BF16 throughout)
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/paged_attention_rows.hpp"
#include "kernels/gemm.hpp"
#include "models/llama/config.hpp"
#include "models/llama/kv_pool.hpp"
#include "models/llama/loader.hpp"

namespace dgpp {

// The GEMM interface's workspace, shared by every layer object of a model.
struct LlamaGemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

// The rows one attention enqueue serves: device [T] request ids and
// positions (a padding row has pos < 0), and whether they are decode rows
// (every row's own split geometry, kernels-only) or a prefill chunk of one
// request (positions pos0 .. pos0 + T - 1, one split).
using LlamaAttnRows = PagedAttentionRows;

class LlamaAttentionLayer {
 public:
  // decode_rows: the most rows a decode enqueue carries (the engine's
  // kDecodeRows); n_split_decode: the split-KV geometry of the decode rows
  // (a capture bakes it in) — the partials' workspace holds max(max_tokens,
  // decode_rows * n_split_decode) row-splits; prefill rows run one split.
  LlamaAttentionLayer(const LlamaAttnResident& w, const LlamaGemmWorkspace& gemm, const LlamaTextConfig& cfg,
                      int max_tokens, int decode_rows, int n_split_decode);
  ~LlamaAttentionLayer();
  LlamaAttentionLayer(const LlamaAttentionLayer&) = delete;
  LlamaAttentionLayer& operator=(const LlamaAttentionLayer&) = delete;
  void rebind(const LlamaAttnResident& w);

  // out[T, H] = o_proj(attention(x[T, H])): the projections, the finish
  // (full RoPE, the K/V of every row appended to `cache` at its slot),
  // the attention over [0, pos] per row, the output projection into `out`
  // (a partial at world > 1).
  void enqueue(const uint16_t* x, int tokens, const LlamaAttnRows& rows, const LlamaKvCache& cache,
               uint16_t* out, cudaStream_t stream);

  static size_t scratch_bytes(const LlamaTextConfig& cfg, int local_heads, int local_kv_heads, int max_tokens,
                              int decode_rows, int n_split_decode);
  // The default decode split count (DGPP_LLAMA_ATTN_SPLITS overrides).
  static int default_decode_splits();
  int local_heads() const { return lh_; }
  int local_kv_heads() const { return lkv_; }
  int n_split_decode() const { return n_split_; }

 private:
  LlamaAttnResident w_;
  LlamaGemmWorkspace g_;
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

class LlamaDenseMlp {
 public:
  LlamaDenseMlp(const LlamaDenseMlpResident& w, const LlamaGemmWorkspace& gemm, const LlamaTextConfig& cfg,
                int max_tokens);
  ~LlamaDenseMlp();
  LlamaDenseMlp(const LlamaDenseMlp&) = delete;
  LlamaDenseMlp& operator=(const LlamaDenseMlp&) = delete;
  void rebind(const LlamaDenseMlpResident& w);
  // out[T, H] = down(silu(gate(x)) * up(x)) on the GEMM interface (BF16).
  void enqueue(const uint16_t* x, int tokens, uint16_t* out, cudaStream_t stream);
  static size_t scratch_bytes(const LlamaTextConfig&, int64_t local_inter, int max_tokens);

 private:
  LlamaDenseMlpResident w_;
  LlamaGemmWorkspace g_;
  int hidden_, max_tokens_;
  int64_t inter_;
  uint16_t* gate_ = nullptr;  // [M, I]
  uint16_t* up_ = nullptr;    // [M, I]
  uint16_t* act_ = nullptr;   // [M, I]
};

}  // namespace dgpp
