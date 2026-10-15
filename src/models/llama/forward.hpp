#pragma once
// Zeta-2.1 Llama model with resident or streaming weights on the shared
// session core. The layer walk follows transformers' LlamaDecoderLayer:
//
//   h = embed(x)
//   per layer: h += attn(input_norm(h))   (GQA, full RoPE, no bias,
//                                          no q/k norm)
//              h += mlp(post_norm(h))     (dense SwiGLU MLP, BF16)
//   logits = lm_head(final_norm(h))
//
// one row walk (run_rows) serves every entry point — the cold diagnostic
// forward, a session's prefill chunks, the eager decode rows and the
// captured decode graphs (engine/graph_engine.hpp's contract) — the
// engine/session_model.hpp core owns everything around it (positions,
// feeds, chunking, snapshots, the graph era).
//
// State per request slot: its row of the paged K/V pool (models/llama/
// kv_pool.hpp, 64-token blocks). Nothing else: a rejected verify row
// leaves stale K/V rows past the position that no later row reads, so the
// speculative commit table is empty and the prefix snapshot is the block
// list. There is no draft block (the checkpoint ships none): mtp paths
// refuse.
//
// TP: `tp_world` > 1 loads this rank's slices (32/W query heads, 8/W kv
// heads, the MLP's intermediate slice, the lm-head vocab slice) and folds
// the attention and MLP block outputs per layer through `boundary` (the
// canonical rank-order sum). The norms are replicated; every rank's
// residual is bitwise the others'.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "models/llama/config.hpp"
#include "models/llama/kv_pool.hpp"
#include "models/llama/layers.hpp"
#include "models/llama/loader.hpp"

namespace dgpp {

class LlamaModel : public SessionModel<LlamaModel> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group,
  // 2026-09-14, the group prefill ported from DeepSeek): the attention rows
  // carry their own request ids and positions, so a span attends to its
  // own request's cache only; every span within max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<LlamaModel>;
  static constexpr bool kPrefillChunkGrid = false;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot (max_context()). max_requests:
  // the session slots. decode_rows: the fixed decode batch's row ceiling —
  // the slots times the verify rows per request (1 without a draft), the
  // serving app's derivation (engine/decode_outputs.hpp); 0 = kDecodeRows,
  // the floor.
  LlamaModel(const LlamaTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
             int64_t max_cache_tokens, LlamaResidency residency = LlamaResidency::Streaming,
             BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
             int max_requests = 1, bool mtp = false, int decode_rows = 0,
             bool serving_logits = false);
  ~LlamaModel();
  LlamaModel(const LlamaModel&) = delete;
  LlamaModel& operator=(const LlamaModel&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for
  // a shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const LlamaTextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                LlamaResidency residency = LlamaResidency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0,
                                bool serving_logits = false);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // capture_layers: every layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The same number for a shape that is not built yet (the memory plan).
  static size_t session_snapshot_bytes(const LlamaTextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;

  const LlamaTextConfig& config() const { return cfg_; }
  const LlamaKvPool& kv_pool() const { return pool_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  LlamaKvPool& pool() { return pool_; }
  const LlamaKvPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests, int hidden_row = 0);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // No draft block ships with the checkpoint: the chain members exist for
  // the session core's template, depth >= 2 paths refuse through mtp_.
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;

  void build_layer_objects(const LlamaLayerResident& r);
  // The layer's two blocks over `resid` (the residual, updated in place):
  // the attention block and the dense MLP block with their folds.
  struct WalkRows {
    const RowRun* run = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    int req = 0;
    bool decode = false;
    bool capture = false;
  };
  void enqueue_layer(const LlamaLayerResident& r, int pool_layer, uint16_t* resid, int T, const WalkRows& rows);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  LlamaTextConfig cfg_;
  LlamaLayerStream loader_;
  CublasLtGemm gemm_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  LlamaGemmWorkspace gw_;
  LlamaGlobalsResident globals_;
  int n_split_ = 32;

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<LlamaAttentionLayer> attn_;
  std::unique_ptr<LlamaDenseMlp> mlp_;

  // The bf16 decode weights' 12-bit companions (kernels/bf12_companions.hpp):
  // every layer's q/k/v/o projections and gate/up/down matrices plus the lm
  // head, packed as each layer lands and before any capture. The loader
  // grants those matrices aside (models/llama/loader.hpp), so under
  // bf12-only residency their bf16 bytes go back as each companion exists.
  static constexpr int kBf12ExpandSlots = 1;
  void pack_layer_companions(int layer, const LlamaLayerResident& r);
  void finish_companions();
  Bf12Companions bf12_;
  bool bf12_built_ = false;
  double bf12_s_ = 0.0;

  LlamaKvPool pool_;  // every layer's paged K/V

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
};

}  // namespace dgpp
