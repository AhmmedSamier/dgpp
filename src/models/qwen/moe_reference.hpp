#pragma once
// Double-precision oracle for the Qwen3.8-Flash-Next MoE (Q3, 2026-09-09):
// the SoftmaxTopk router (glm_moe_ref_router in that mode), the FP8 routed
// experts with the strict bf16 rounding points, the BF16 shared expert
// weighed by its sigmoid gate, and the engine's fp32-chain semantics in
// double — unrounded down dots, one fma per expert ascending, the shared
// expert last, ONE bf16 rounding (models/qwen/moe_layer.hpp).
#include <cstdint>
#include <vector>

#include "models/glm/moe.hpp"
#include "models/glm/moe_reference.hpp"

namespace dgpp {

struct QwenMoeHostWeights {
  int hidden = 0;
  int inter = 0;          // the routed slice width on this rank (I/W)
  int n_experts = 0;
  int shared_inter = 0;   // the shared slice width (S/W)
  // The experts' scale grid: `scale_block` on the sliced axis (gate/up
  // rows, down columns), 128 on the other (plan D2).
  int scale_block = 128;
  std::vector<uint16_t> router;       // bf16 [E, H]
  std::vector<uint16_t> shared_gate;  // bf16 [H]
  std::vector<uint16_t> shared_w[3];  // bf16 gate [S, H], up [S, H], down [H, S]
  // E triples of e4m3 matrices (gate [I, H], up [I, H], down [H, I]) and
  // their F32 scale grids, matrix after matrix.
  std::vector<uint8_t> payloads;
  std::vector<float> scales;

  int64_t rows(int m) const { return m == 2 ? hidden : inter; }
  int64_t cols(int m) const { return m == 2 ? inter : hidden; }
  int scale_block_rows(int m) const { return m == 2 ? 128 : scale_block; }
  int scale_block_cols(int m) const { return m == 2 ? scale_block : 128; }
  int64_t scale_rows(int m) const {
    return (rows(m) + scale_block_rows(m) - 1) / scale_block_rows(m);
  }
  int64_t scale_cols(int m) const {
    return (cols(m) + scale_block_cols(m) - 1) / scale_block_cols(m);
  }
  size_t payload_bytes(int m) const { return static_cast<size_t>(rows(m) * cols(m)); }
  size_t scale_count(int m) const { return static_cast<size_t>(scale_rows(m) * scale_cols(m)); }
  // Offsets of expert e's matrix m (index e*3+m) in payloads / scales.
  size_t payload_offset(int index) const;
  size_t scale_offset(int index) const;
  // Sizes the two vectors to the geometry (zeros).
  void allocate();
};

// hidden [tokens, H] bf16 in; out [tokens, H] bf16 bits. `route` (optional)
// receives the router's decision (ids ascending, bf16 weights, the logit
// row in `biased`).
void qwen_moe_ref_forward(const uint16_t* hidden, const QwenMoeHostWeights& w,
                          const GlmMoeConfig& cfg, int tokens,
                          std::vector<uint16_t>& out,
                          GlmMoeRouterRef* route = nullptr);

}  // namespace dgpp
