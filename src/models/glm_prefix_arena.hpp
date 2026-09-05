#pragma once
// The prefix cache's snapshot arena (M7 stage B, DESIGN §8): device slots
// of session_snapshot_bytes() each, one per cache entry, over the model's
// stage-A primitives — session_snapshot / session_attach /
// session_release_snapshot. Both engine adapters own one and serve the
// scheduler's prefix ops through it; the scheduler's PrefixCache is the
// ledger of which slot holds what, this is the state itself.
//
// A slot's contents: the KDA recurrent + conv state of every KDA layer,
// every DSA layer's tail ring, the draft block's h_q, and (in the pool,
// not the arena) the entry's block references — the full blocks pinned by
// reference and its private copy of the partial block. Snapshots and
// attaches are stream-ordered on the model stream; their device time is
// measured with events, harvested lazily (no sync on the decode path).
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "models/glm_forward.hpp"

namespace dgpp {

class PrefixArena {
 public:
  PrefixArena(GlmDiagnosticModel* model, int slots)
      : model_(model), slots_(slots) {
    if (model_ == nullptr) throw std::invalid_argument("PrefixArena: null model");
    if (slots_ < 0) throw std::invalid_argument("PrefixArena: negative slots");
    bytes_ = model_->session_snapshot_bytes();
    if (slots_ > 0) {
      if (bytes_ == 0)
        throw std::invalid_argument(
            "PrefixArena: the model has no session state to snapshot");
      DGPP_CUDA_OK(cudaMalloc(&base_, bytes_ * static_cast<size_t>(slots_)));
      metas_.resize(static_cast<size_t>(slots_));
      filled_.assign(static_cast<size_t>(slots_), false);
      for (Timer& t : timers_) {  // timing events (the default flags)
        DGPP_CUDA_OK(cudaEventCreate(&t.start));
        DGPP_CUDA_OK(cudaEventCreate(&t.end));
      }
    }
  }
  ~PrefixArena() {
    for (int s = 0; s < slots_; ++s)
      if (filled_[static_cast<size_t>(s)]) {
        try {
          model_->session_release_snapshot(metas_[static_cast<size_t>(s)]);
        } catch (...) {
        }
      }
    if (slots_ > 0)
      for (Timer& t : timers_) {
        cudaEventDestroy(t.start);
        cudaEventDestroy(t.end);
      }
    if (base_) cudaFree(base_);
  }
  PrefixArena(const PrefixArena&) = delete;
  PrefixArena& operator=(const PrefixArena&) = delete;

  int slots() const { return slots_; }
  size_t bytes() const { return bytes_; }
  bool filled(int slot) const { return filled_.at(static_cast<size_t>(slot)); }
  int64_t position(int slot) const {
    check(slot);
    if (!filled_[static_cast<size_t>(slot)])
      throw std::logic_error("PrefixArena: slot " + std::to_string(slot) + " is empty");
    return metas_[static_cast<size_t>(slot)].position;
  }

  // A live session's state at its current (pool-aligned) position into
  // `slot`, replacing what the slot held. `expected_position` (>= 0) is
  // the caller's view of the session position, verified.
  void snapshot(int req, int slot, int64_t expected_position) {
    check(slot);
    if (expected_position >= 0 &&
        model_->session_position(req) != expected_position)
      throw std::logic_error(
          "PrefixArena: session " + std::to_string(req) + " sits at " +
          std::to_string(model_->session_position(req)) + ", the snapshot expects " +
          std::to_string(expected_position));
    release(slot);
    Timer& t = begin_timer();
    metas_[static_cast<size_t>(slot)] = model_->session_snapshot(req, ptr(slot));
    end_timer(t, &snapshot_ms_, &snapshots_);
    filled_[static_cast<size_t>(slot)] = true;
  }

  // The hop snapshot (M7 under the two-row step): the state after the last
  // step's FIRST row — the aligned position `expected_position` the step
  // committed past — into `slot`, from the model's spec snapshot rows
  // (`spec_row` the slot's first row of that step).
  void snapshot_post_row0(int req, int slot, int64_t expected_position, int spec_row) {
    check(slot);
    if (model_->session_position(req) != expected_position + 1)
      throw std::logic_error(
          "PrefixArena: session " + std::to_string(req) + " sits at " +
          std::to_string(model_->session_position(req)) + ", the hop snapshot expects " +
          std::to_string(expected_position + 1) + " (one past the position)");
    release(slot);
    Timer& t = begin_timer();
    metas_[static_cast<size_t>(slot)] =
        model_->session_snapshot_post_row0(req, ptr(slot), spec_row);
    end_timer(t, &snapshot_ms_, &snapshots_);
    filled_[static_cast<size_t>(slot)] = true;
  }
  // The slot's bytes and metadata (the gates compare them).
  const void* slot_data(int slot) const {
    check(slot);
    return ptr(slot);
  }
  const GlmDiagnosticModel::SessionSnapshotMeta& meta(int slot) const {
    check(slot);
    return metas_.at(static_cast<size_t>(slot));
  }

  // A snapshot request for the model's prefill (taken mid-prefill when a
  // chunk ends at `position`); commit() afterwards records whether it was.
  GlmDiagnosticModel::SnapshotRequest request(int slot, int64_t position) {
    check(slot);
    release(slot);
    GlmDiagnosticModel::SnapshotRequest r;
    r.position = position;
    r.dst = ptr(slot);
    r.meta = &metas_[static_cast<size_t>(slot)];
    return r;
  }
  void commit(int slot, const GlmDiagnosticModel::SnapshotRequest& r) {
    check(slot);
    filled_[static_cast<size_t>(slot)] = r.taken;
    if (r.taken) ++snapshots_;
  }

  // Opens closed session `req` from `slot`.
  void attach(int req, int slot) {
    check(slot);
    if (!filled_[static_cast<size_t>(slot)])
      throw std::logic_error("PrefixArena: attach from empty slot " +
                             std::to_string(slot));
    Timer& t = begin_timer();
    model_->session_attach(req, ptr(slot), metas_[static_cast<size_t>(slot)]);
    end_timer(t, &attach_ms_, &attaches_);
  }

  // Drops the slot's block references; the slot is empty afterwards.
  void release(int slot) {
    check(slot);
    if (!filled_[static_cast<size_t>(slot)]) return;
    model_->session_release_snapshot(metas_[static_cast<size_t>(slot)]);
    metas_[static_cast<size_t>(slot)] = GlmDiagnosticModel::SessionSnapshotMeta{};
    filled_[static_cast<size_t>(slot)] = false;
  }

  // The measured device time so far (events harvested as they complete).
  int64_t snapshots() const { harvest(); return snapshots_; }
  double snapshot_ms() const { harvest(); return snapshot_ms_; }
  int64_t attaches() const { harvest(); return attaches_; }
  double attach_ms() const { harvest(); return attach_ms_; }

 private:
  struct Timer {
    cudaEvent_t start = nullptr, end = nullptr;
    bool armed = false;
    double* sink = nullptr;
  };
  void check(int slot) const {
    if (slot < 0 || slot >= slots_)
      throw std::out_of_range("PrefixArena: slot " + std::to_string(slot) +
                              " outside [0, " + std::to_string(slots_) + ")");
  }
  void* ptr(int slot) const {
    return static_cast<uint8_t*>(base_) + bytes_ * static_cast<size_t>(slot);
  }
  // A ring of event pairs: the oldest armed one is harvested (or waited
  // for, if still in flight — four ops later it never is) before reuse.
  Timer& begin_timer() {
    Timer& t = timers_[next_timer_];
    next_timer_ = (next_timer_ + 1) % kTimers;
    if (t.armed) {
      DGPP_CUDA_OK(cudaEventSynchronize(t.end));
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, t.start, t.end));
      *t.sink += ms;
      t.armed = false;
    }
    DGPP_CUDA_OK(cudaEventRecord(t.start, model_->stream()));
    return t;
  }
  void end_timer(Timer& t, double* sink, int64_t* count) {
    DGPP_CUDA_OK(cudaEventRecord(t.end, model_->stream()));
    t.sink = sink;
    t.armed = true;
    ++*count;
  }
  void harvest() const {
    for (Timer& t : timers_) {
      if (!t.armed) continue;
      if (cudaEventQuery(t.end) != cudaSuccess) continue;
      float ms = 0;
      if (cudaEventElapsedTime(&ms, t.start, t.end) == cudaSuccess) *t.sink += ms;
      t.armed = false;
    }
  }

  GlmDiagnosticModel* model_;
  int slots_ = 0;
  size_t bytes_ = 0;
  void* base_ = nullptr;
  std::vector<GlmDiagnosticModel::SessionSnapshotMeta> metas_;
  std::vector<bool> filled_;
  static constexpr int kTimers = 4;
  mutable Timer timers_[kTimers];
  int next_timer_ = 0;
  int64_t snapshots_ = 0, attaches_ = 0;
  mutable double snapshot_ms_ = 0, attach_ms_ = 0;
};

}  // namespace dgpp
