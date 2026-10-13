#pragma once
// Qwen3-ASR: the dense Qwen3 language half (models/glm4 attention layers'
// geometry with 128-wide heads, no biases, per-head q/k norms, full RoPE)
// plus the audio tower (models/qwen_asr/audio_tower_reference.hpp) whose
// rows replace the prompt's audio pad spans during prefill (docs/audio.md).
//
// The walk:
//   h = embed(x)               (audio spans overwritten from the tower rows)
//   per layer: h += attn(input_norm(h)); h += mlp(post_norm(h))
//   logits = lm_head(final_norm(h))
//
// The tower runs the fp32 CPU reference on the scheduler thread inside
// session_prefill_audios, its rows staged once into a device buffer the
// chunk loop splices from; CUDA-izing it is a pure speed-up behind the same
// numbers. No MTP, no MoE, no TP, resident weights only.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/audio_input.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "loaders/minijson.hpp"
#include "kernels/glm_spec.hpp"
#include "models/glm4/kv_pool.hpp"
#include "models/glm4/layers.hpp"  // Glm4AttentionLayer (the shared GQA+qk-norm walk)
#include "models/qwen_asr/audio_config.hpp"
#include "models/qwen_asr/audio_tower_reference.hpp"
#include "models/qwen_asr/lm_config.hpp"
#include "models/qwen_asr/loader.hpp"

namespace dgpp {
struct QwenAsrConfig {
  QwenAsrLmConfig lm;
  QwenAsrAudioConfig audio;
  static QwenAsrConfig parse(const minijson::Value& root) {
    QwenAsrConfig c;
    c.lm = QwenAsrLmConfig::parse(root);
    c.audio = QwenAsrAudioConfig::parse(root);
    c.audio.require_output(c.lm.hidden_size);
    return c;
  }
  static QwenAsrConfig from_json_file(const std::string& path);
};

class QwenAsrModel : public SessionModel<QwenAsrModel> {
 public:
  using Base = SessionModel<QwenAsrModel>;
  using Outputs = Base::Outputs;
  using RowRun = Base::RowRun;

  QwenAsrModel(const QwenAsrConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
               int64_t max_cache_tokens, int max_requests = 1, int decode_rows = 0,
               bool serving_logits = false);
  ~QwenAsrModel();
  QwenAsrModel(const QwenAsrModel&) = delete;
  QwenAsrModel& operator=(const QwenAsrModel&) = delete;

  static MemoryPlan plan_memory(const QwenAsrConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int max_requests = 1, int decode_rows = 0,
                                bool serving_logits = false);

  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return 2048; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  static size_t session_snapshot_bytes(const QwenAsrConfig& cfg);
  using Base::session_snapshot_bytes;

  const QwenAsrConfig& config() const { return cfg_; }
  Glm4KvPool& pool() { return pool_; }
  const Glm4KvPool& pool() const { return pool_; }
  bool has_pool() const { return true; }

  // The audio seam (engine/*_engine.hpp probes these names).
  bool supports_audios() const { return true; }
  AudioTokens audio_tokens() const { return cfg_.audio.tokens; }
  Outputs session_prefill_audios(int req, const std::vector<int64_t>& prompt,
                                 const std::vector<AudioInput>& audios);

  // ---- the session core's hooks ---------------------------------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int, int) const { return {}; }
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int, uint8_t* dst, int) {
    DGPP_CUDA_OK(cudaMemsetAsync(dst, 0, kSnapshotStamp, stream_));
  }
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  void graph_prepare();
  void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int) {
    throw std::logic_error("Qwen ASR: no draft block");
  }
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  static constexpr bool kDraftChain = false;
  // The session core instantiates its draft-chain helpers for every
  // family; these are never reached (kDraftChain is false) but must name
  // a well-formed thing.
  static constexpr bool kBatchedDraftChain = false;
  const uint16_t* draft_hidden_rows() const { return nullptr; }

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  static constexpr size_t kSnapshotStamp = 16;
  static constexpr int64_t kSegmentSamples = 16000ll * 30;  // the 30 s featurizer window

  struct DenseMlp;  // the BF16 SwiGLU (gate/up GEMMs, fused act, down)

  void build_layer_objects(const QwenAsrLayerResident& r);
  void enqueue_layer(const QwenAsrLayerResident& r, int layer, uint16_t* resid, int T,
                     const int32_t* req_ids, const int64_t* pos, bool decode, bool capture);
  // The current prefill's tower rows: [row][hidden] bf16, the audios' spans
  // in prompt order.
  void stage_audio_rows(const std::vector<AudioInput>& audios);
  void apply_audio_embeddings(uint16_t* resid, int64_t pos0, int T);

  QwenAsrConfig cfg_;
  Glm4TextConfig geom_;  // the shared attention layer objects' view of lm
  QwenAsrLayerStream loader_;
  CublasLtGemm gemm_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Glm4GemmWorkspace gw_;
  QwenAsrGlobalsResident globals_;
  std::unique_ptr<Glm4AttentionLayer> attn_;
  std::unique_ptr<DenseMlp> dense_;
  Glm4KvPool pool_;

  uint16_t* resid_ = nullptr;  // [M, H]
  uint16_t* x_ = nullptr;
  uint16_t* y_ = nullptr;

  qwen_asr_ref::AudioTowerWeights tower_;
  uint16_t* audio_rows_ = nullptr;  // [<= M rows, H] bf16, device
  std::vector<float> audio_scratch_;
  std::vector<uint16_t> audio_bf16_;
  const std::vector<AudioInput>* audio_spans_ = nullptr;  // the walk's scope
  std::vector<int64_t> audio_row_base_;                    // parallel to audio_spans_
};
}  // namespace dgpp
