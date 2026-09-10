#pragma once
// The Qwen3.8-Flash-Next MoE (Q3, 2026-09-09; docs/qwen38_flash_next_plan.md
// §1.6): 512 FP8 experts, top-10 by softmax with the picked probabilities
// renormalized, plus ONE BF16 shared expert scaled by a sigmoid gate of the
// input. Composition, not a new expert path: the routed experts run the
// shared MoE layer (GlmMoeLayer — its router in SoftmaxTopk mode, no shared
// segment, no swiglu clamps) and hand back the fp32 chain unrounded; this
// layer runs the BF16 shared expert through the GEMM seam, weighs it by
// the gate and continues the same fmaf chain, which rounds to bf16 exactly
// once — the GLM chain's numerics (models/glm/moe_layer.hpp), the shared
// expert's weight sigma(x . g) instead of 1.
//
// Host-orchestrated (the diagnostic forward's path; one stream sync in the
// routed layer). The production decode/prefill variants — the slot kernels
// without a shared slot, the BF16 shared expert fused beside them — are
// the engine milestone's (Q6/Q7); GlmMoeLayer refuses those paths without
// its shared expert until then.
//
// TP (plan D1/D2): every rank holds every expert sliced on the intermediate
// dim (I/W rows of gate/up, columns of down, the scale grid re-blocked at
// gcd(128, I/W)), the shared expert sliced the same way (S/W), the router
// and the shared gate replicated; the layer's output is the rank's partial
// for the FFN all-reduce. The kernels' sub-128 scale grid lands with the
// TP forward (plan §5 Q4).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// Device-visible views (the loader's QwenMoeResident wires them).
struct QwenMoeWeights {
  const uint16_t* router = nullptr;            // bf16 [n_experts, hidden]
  const uint16_t* shared_gate = nullptr;       // bf16 [hidden] (shared_expert_gate)
  const uint16_t* shared_gate_proj = nullptr;  // bf16 [S, hidden]
  const uint16_t* shared_up_proj = nullptr;    // bf16 [S, hidden]
  const uint16_t* shared_down_proj = nullptr;  // bf16 [hidden, S]
  const GlmQuantMatrix* shared_fp8 = nullptr;  // dense_weights fp8: gate, up, down (the bf16 three null)
  int64_t shared_inter = 0;                    // S: this rank's shared slice
  const GlmQuantMatrix* experts = nullptr;     // [n_experts * 3] gate, up, down (FP8 block form)
  const GlmFp4Matrix* experts_fp4 = nullptr;   // the NVFP4 form instead (one of the two is set)
};

class QwenMoeLayer {
 public:
  // The routed chain's configuration: SoftmaxTopk router, no shared
  // expert in the chain, no scaling factor, no swiglu clamps.
  static GlmMoeConfig routed_config(int hidden, int inter, int n_experts,
                                    int top_k, bool norm_topk_prob);

  // gemm: the model's GEMM seam (the BF16 shared expert's three products;
  // decode shapes take the in-house GEMV inside it). max_tokens bounds
  // enqueue()'s rows.
  // decode_slots / graph_table_slots: the routed layer's decode-path
  // provisioning (models/glm/moe_layer.hpp) — 0 keeps the host path only.
  QwenMoeLayer(const QwenMoeWeights& weights, const GlmMoeConfig& cfg,
               IGemm& gemm, int max_tokens, int decode_slots = 0,
               int graph_table_slots = 0);
  ~QwenMoeLayer();
  QwenMoeLayer(const QwenMoeLayer&) = delete;
  QwenMoeLayer& operator=(const QwenMoeLayer&) = delete;

  // out[tokens, hidden] = bf16(sum_e w_e down_e(...) + sigma(x.g) shared(x)),
  // the fp32 chain in ascending expert order with the shared expert last.
  // Synchronizes the stream once (the routed layer's host segmentation).
  void enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
               cudaStream_t stream,
               MoeExpertKernel kernel = MoeExpertKernel::kGemv);

  // The routed layer (its last_ids/last_weights/last_biased are this
  // enqueue's routing decision; last_biased holds the bf16 logits).
  // The decode fast path (Q6): the routed chain off the device route (no
  // host round-trip; table_slot >= 0 reads a graph slot's prepared table),
  // then the BF16 shared expert and the single rounding — bitwise
  // enqueue() at the same routing. tokens <= decode_slots.
  void enqueue_decode(const uint16_t* hidden, uint16_t* out, int tokens,
                      cudaStream_t stream, int table_slot = -1);
  // The prefill path (Q7): the routed chain device-segmented on the
  // tensor-core kernel (the slice's 32/64 scale grid included, no host
  // round-trip), then the BF16 shared expert and the one rounding.
  // The routing (ids, weights) lands async in `trace` (pinned; read after
  // the stream's next sync) when given; last_ids()/last_weights() are NOT
  // updated by this path.
  void enqueue_prefill(const uint16_t* hidden, uint16_t* out, int tokens,
                       cudaStream_t stream, MoeTraceStaging* trace = nullptr);
  void prepare_graph_table(int table_slot, cudaStream_t stream) {
    routed_.prepare_graph_table(table_slot, stream);
  }
  const GlmMoeLayer& routed() const { return routed_; }
  GlmMoeLayer& routed() { return routed_; }
  const GlmMoeConfig& config() const { return cfg_; }

  // Streaming-weight seam: swap the views (shapes unchanged).
  void rebind(const QwenMoeWeights& w);

  // Bytes the constructor allocates (device; pinned in *pinned_bytes),
  // the routed layer's included — the memory plan's line.
  static size_t scratch_bytes(const GlmMoeConfig& cfg, int64_t shared_inter, int max_tokens,
                              size_t* pinned_bytes = nullptr, int decode_slots = 0,
                              int graph_table_slots = 0);

 private:
  static GlmMoeWeights routed_view(const QwenMoeWeights& w);
  void check_weights() const;
  // The chain's tail after the routed experts left d_acc_: the BF16 shared
  // expert, its sigmoid weight, the last fma and the one rounding.
  void shared_tail(const uint16_t* hidden, uint16_t* out, int tokens, cudaStream_t stream);

  QwenMoeWeights w_;
  GlmMoeConfig cfg_;
  IGemm& gemm_;
  int max_tokens_;
  GlmMoeLayer routed_;

  float* d_acc_ = nullptr;      // [M, H] the fp32 chain
  uint16_t* d_sgate_ = nullptr; // [M, S]
  uint16_t* d_sup_ = nullptr;   // [M, S]
  uint16_t* d_sact_ = nullptr;  // [M, S]
  float* d_sdown_ = nullptr;    // [M, H] the shared down projection, unrounded
  float* d_sw_ = nullptr;       // [M] sigma(x . g), a bf16 value
  int32_t* d_rows_ = nullptr;   // [M] identity (the accumulation's row map)
  void* gemm_ws_ = nullptr;
  uint16_t* d_shared_bridge_ = nullptr;  // dense_weights fp8: a shared matrix dequantized for prefill
  size_t gemm_ws_bytes_ = 0;
};

}  // namespace dgpp
