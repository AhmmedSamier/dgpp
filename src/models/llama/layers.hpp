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
//
// Weight forms per projection: BF16 on the GEMM interface, block FP8 on
// the scale GEMM (engine.dense_weights = "fp8"), or the ModelOpt NVFP4
// triple (a W4A16_NVFP4 checkpoint) on the fp4 GEMV core (decode rows)
// and the fp4 tile GEMM (prefill; bf16 outs — the attention's q/k/v widen
// to fp32 through cast_bf16_to_f32_rows).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/paged_attention_rows.hpp"
#include "kernels/gemm.hpp"
#include "kernels/fp4_gemv.hpp"
#include "models/llama/config.hpp"
#include "models/llama/kv_pool.hpp"
#include "models/llama/loader.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The GEMM interface's workspace, shared by every layer object of a model.
struct LlamaGemmWorkspace {
  IGemm* gemm = nullptr;
  void* ws = nullptr;
  size_t ws_bytes = 0;
  // Rows from which the scale GEMM takes the streaming tensor-core form
  // (kernels/scale_gemm.hpp): 0 keeps the chunked GEMV; the model sets 1
  // under engine.dense_weights = "fp8" (every row's chain m-invariant).
  int mma_from_rows = 0;
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
                              int decode_rows, int n_split_decode, bool fp4_weights = false);
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
  // The NVFP4 prefill's bf16 GEMM outs for k/v (the fp4 tile kernel has no
  // fp32 epilogue; q reuses q_ as its scratch — the projections run before
  // the RoPE overwrites it), widened to fp32 by cast_bf16_to_f32_rows.
  uint16_t* k4b_ = nullptr;
  uint16_t* v4b_ = nullptr;
};

class LlamaDenseMlp {
 public:
  LlamaDenseMlp(const LlamaDenseMlpResident& w, const LlamaGemmWorkspace& gemm, const LlamaTextConfig& cfg,
                int max_tokens, int decode_rows = 0);
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
  int decode_rows_ = 0;  // the GEMV/mma boundary of the fp4 path (0: always mma... fp4 requires > 0)
  int64_t inter_;
  uint16_t* gate_ = nullptr;  // [M, I]
  uint16_t* up_ = nullptr;    // [M, I]
  uint16_t* act_ = nullptr;   // [M, I]
};

}  // namespace dgpp
