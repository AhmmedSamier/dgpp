#pragma once
// Nemotron-3.5-Transcribe-30B-A4B model (milestone 4 of
// docs/nemotron_transcribe_plan.md): the SessionModel that runs the 52-slot
// Nemotron-H hybrid walk (Mamba-2 + GQA attention + relu2 MoE) on the GB10,
// plus the host fp32 oracle cross-check. MTP (`--mtp`) is not yet wired;
// it lands as a follow-up once the base forward serves.
//
// The walk: embed → per-slot [RMSNorm → mixer → residual add] → norm_f →
// lm_head. Attention is position-free (no RoPE). The Mamba-2 slots use the
// device kernels in src/kernels/mamba2.cu, gated against the host oracle
// in src/models/nemotron_transcribe/reference.cpp.
//
// Derives from SessionModel<NemotronModel>; the base owns sessions, the
// block pool, prefix snapshots, verify/rollback and graph capture. The
// derived supplies run_rows() (the layer walk) and the state-family hooks.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/kernels.hpp"
#include "models/nemotron_transcribe/binding.hpp"
#include "models/nemotron_transcribe/config.hpp"
#include "models/nemotron_transcribe/loader.hpp"
#include "models/nemotron_transcribe/state.hpp"
#include "common/audio_input.hpp"

namespace dgpp {

class NemotronModel : public SessionModel<NemotronModel> {
 public:
  using Base = SessionModel<NemotronModel>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  NemotronModel(const NemotronTranscribeConfig& cfg, const std::string& checkpoint_dir,
                int max_tokens, int64_t max_cache_tokens,
                NemotronResidency residency = NemotronResidency::Streaming,
                BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
                int max_requests = 1, bool mtp = false, int decode_rows = 0);
  ~NemotronModel();
  NemotronModel(const NemotronModel&) = delete;
  NemotronModel& operator=(const NemotronModel&) = delete;

  static MemoryPlan plan_memory(const NemotronTranscribeConfig& cfg, int max_tokens,
                                int64_t max_cache_tokens, int tp_rank = 0, int tp_world = 1,
                                NemotronResidency residency = NemotronResidency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0);

  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  static constexpr int prefill_chunk_tokens() { return 2048; }
  static constexpr int kv_block_tokens_static() { return 128; }
  static constexpr int decode_rows_cap() { return 32; }

  const NemotronTranscribeConfig& config() const { return cfg_; }
  const NemotronGlobalsResident& globals() const { return globals_; }
  NemotronMambaStatePool& pool() { return pool_; }
  const NemotronMambaStatePool& pool() const { return pool_; }
  cudaStream_t stream() const { return stream_; }
  bool mtp_enabled() const { return false; }
  int max_session_requests() const { return max_requests_; }
  int lm_vocab_begin() const { return 0; }
  int lm_vocab_count() const { return cfg_.lm.vocab_size; }

  // ---- SessionModel hooks --------------------------------------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  static constexpr bool kDraftChain = false;
  static constexpr bool kVerifyConfidence = false;
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void mtp_select_block(int) {}
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}

  size_t snapshot_state_bytes() const { return pool_.geometry().request_bytes(); }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return false; }
  static constexpr bool kBatchedDraftChain = false;
  size_t draft_state_bytes() const { return 0; }
  void graph_prepare();
  void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int) {}

  // ---- Audio support (milestone 6) ---------------------------------------
  // The Nemotron contract: one `<|audio|>` locator (id == vocab_size) per
  // clip, a pure splice marker never embedding-looked-up.
  bool supports_audios() const { return true; }
  dgpp::AudioTokens audio_tokens() const { return {-1, -1, -1}; }  // single locator
  int64_t prefill_audio_token_limit() const { return 4096; }
  Outputs session_prefill_audios(int req, const std::vector<int64_t>& prompt,
                                 const std::vector<dgpp::AudioInput>& audios);

 private:
  static constexpr size_t kSnapshotStamp = 16;

  void init_activations(int max_tokens);
  void free_activations();
  // The full walk: embed → per-slot [norm → mixer → residual] → norm_f → head.
  // Writes device logits_ and h_; run_rows copies them host-side.
  void walk(const RowRun& run);
  // One slot: RMSNorm(resid) → x_, mixer(x_) → y_, resid += y_.
  void walk_slot(int slot, int T, int64_t pos0);

  NemotronTranscribeConfig cfg_;
  NemotronLayerStream loader_;
  NemotronGlobalsResident globals_;
  Arena arena_;
  NemotronMambaStatePool pool_;
  cudaStream_t stream_ = nullptr;
  // The primary CUDA context captured on the construction (main) thread.
  // run_rows() may be invoked from request-handling threads whose implicit
  // context differs; this handle lets us switch back to the context under
  // which every device pointer (weights + activations) was allocated.
  CUcontext primary_ctx_ = nullptr;

  // Activations, bf16 [max_tokens, hidden] unless noted.
  uint16_t* resid_ = nullptr;   // [M, H] the residual
  uint16_t* x_ = nullptr;       // [M, H] normed block input
  uint16_t* y_ = nullptr;       // [M, H] block output
  // Scratch for per-slot mixers, bf16 [max_tokens, width].
  uint16_t* xbc_ = nullptr;     // [M, in_proj_rows] mamba in_proj output
  uint16_t* x_conv_ = nullptr;  // [M, d_inner] mamba conv output (x)
  uint16_t* b_conv_ = nullptr;  // [M, groups*state] mamba conv output (B)
  uint16_t* c_conv_ = nullptr;  // [M, groups*state] mamba conv output (C)
  uint16_t* mamba_out_ = nullptr;  // [M, d_inner] mamba scan output
  uint16_t* mamba_normed_ = nullptr;  // [M, d_inner] mamba gated-norm output
  uint16_t* q_ = nullptr;       // [M, q_rows] attention q
  uint16_t* k_ = nullptr;       // [M, kv_rows] attention k
  uint16_t* v_ = nullptr;       // [M, kv_rows] attention v
  uint16_t* attn_out_ = nullptr; // [M, q_rows] attention output
  // Device logits [rows, vocab] fp32 and hidden [rows, H] bf16.
  float* logits_ = nullptr;
  uint16_t* h_ = nullptr;
  // Device token ids and positions for the current run.
  int64_t* d_tokens_ = nullptr;
  int64_t* d_positions_ = nullptr;

  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int max_requests_ = 1;
  int hidden_ = 0;
  int vocab_size_ = 0;
  int mamba_d_inner_ = 0;
  int mamba_groups_state_ = 0;
  int mamba_heads_ = 0;
  int mamba_head_dim_ = 0;
  int mamba_state_size_ = 0;
  int mamba_conv_kernel_ = 0;
  int attn_q_rows_ = 0;
  int attn_kv_rows_ = 0;
  int attn_q_heads_ = 0;
  int attn_kv_heads_ = 0;
  int attn_head_dim_ = 0;

  // Audio (milestone 6): per-request audio embedding spans spliced into the
  // residual at <|audio|> locator positions during the walk.
  struct AudioSpan {
    int64_t prompt_offset = -1;  // position in prompt where audio span starts
    int64_t embed_offset = 0;    // offset into d_audio_embeds_ (bf16 rows)
    int count = 0;               // number of audio token rows (T_tok)
  };
  std::vector<AudioSpan> audio_spans_;   // indexed by req
  uint16_t* d_audio_embeds_ = nullptr;   // [capacity, H] bf16 device buffer
  size_t audio_capacity_ = 0;            // capacity in rows

  std::string checkpoint_dir_;  // for on-demand perception weight reads
};

}  // namespace dgpp
