#pragma once
// The fabric engine closure for serving (Stage 4b) and glm_gen_check's
// scheduler path — ONE seam, two apps, no drift. Real-mesh BusOptions
// (the budgets the first fabric gate run found, 2026-08-30) + the
// distributed greedy pick over the vocab-sharded head.
//
// CUDA-app-only header: it drags the bus (verbs) headers. Host gates
// fake the engine instead; nothing in dgpp_service includes this.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "common/process_memory.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_graph_check.hpp"
#include "models/glm_loader.hpp"
#include "models/glm_step_timing.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

// Pins the process's current pages BEFORE the model is constructed — the
// decode loop's host state (tokenizer tables, the bus, this binary) once
// had to survive a load phase that drove every box to its memory
// watermark and had the kernel swapping exactly those pages out
// (2026-09-02: a ~10 ms swap-in fault per shared token, in lockstep across
// ranks). The one-pass loader removed that pressure, and a 1000-step run
// with the pin OFF was indistinguishable (p99 46, 0 stalls, no swap
// traffic) — so this is a belt-and-braces safety net, NOT a requirement:
// a box whose RLIMIT_MEMLOCK refuses it serves exactly as well, and the
// refusal is informational. Before construction on purpose: the loader's
// checkpoint mmaps do not exist yet, so MCL_CURRENT cannot try to pin
// them. DGPP_MLOCK=off skips the attempt.
inline void pin_serving_process(int rank) {
  if (const char* m = std::getenv("DGPP_MLOCK"); m && std::string(m) == "off") {
    DGPP_LOG_INFO("rank {}: process memory not locked (DGPP_MLOCK=off)", rank);
    return;
  }
  std::string why;
  size_t locked = 0;
  if (lock_process_memory(&why, &locked))
    DGPP_LOG_INFO("rank {}: process memory locked (mlockall MCL_CURRENT, "
                  "{:.0f} MiB)",
                  rank, static_cast<double>(locked) / (1024.0 * 1024.0));
  else
    DGPP_LOG_INFO("rank {}: process memory not locked ({}); serving "
                  "proceeds — the pin is optional",
                  rank, why);
}

// The resident image cache's location (GlmLayerStream::set_resident_image_dir):
//   DGPP_RESIDENT_CACHE=off        disabled
//   DGPP_RESIDENT_CACHE_DIR=DIR    explicit directory
//   otherwise                      $XDG_CACHE_HOME/dgpp/resident, or
//                                  $HOME/.cache/dgpp/resident
// On by default on purpose: a serving box's disk exists to make the next
// start fast, and the key (checkpoint headers + config + world/rank +
// format version) makes a stale image impossible to load by accident.
inline void configure_resident_image_cache(int rank) {
  std::string dir;
  if (const char* mode = std::getenv("DGPP_RESIDENT_CACHE");
      mode && std::string(mode) == "off") {
    DGPP_LOG_INFO("rank {}: resident image cache off (DGPP_RESIDENT_CACHE)",
                  rank);
  } else if (const char* d = std::getenv("DGPP_RESIDENT_CACHE_DIR"); d && *d) {
    dir = d;
  } else if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
    dir = std::string(x) + "/dgpp/resident";
  } else if (const char* h = std::getenv("HOME"); h && *h) {
    dir = std::string(h) + "/.cache/dgpp/resident";
  }
  GlmLayerStream::set_resident_image_dir(dir);
  if (!dir.empty())
    DGPP_LOG_INFO("rank {}: resident image cache at {}", rank, dir);
}

// Everything a serving process does before it constructs its model.
inline void prepare_serving_process(int rank) {
  pin_serving_process(rank);
  configure_resident_image_cache(rank);
}

// Real-mesh budgets, not loopback budgets (found by the first fabric
// gate run, 2026-08-30: a cold peer's first boundary waits behind
// seconds of cold NVMe weight streaming). The lane watchdog arms from
// each POST, so these bounds measure genuine in-flight stalls only.
// launch_consumers=false: the app drives every collective itself.
inline net::BusOptions fabric_bus_options(int rank, int world, uint16_t port,
                                          const std::string& peer,
                                          int rendezvous_timeout_ms) {
  net::BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : peer;
  o.rendezvous_timeout_ms = rendezvous_timeout_ms;
  o.lat_slots = 8;
  // Phase 2 folds the full fixed row batch in one collective: eight bf16
  // hidden-4096 rows = 64 KiB. This is only a transport sizing knob; the
  // latency protocol and element-driven kernels are unchanged.
  o.lat_slot_bytes = GlmDiagnosticModel::kDecodeRows * 4096 * 2;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 120000;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

// The fabric pick: this rank's vocab-slice argmax, then the winner
// through bus_greedy_pick — the collective every rank joins in the
// same tick, with the readback invariant that makes a corrupt
// broadcast LOUD on the exact rank. The float row is hoisted inside
// the returned closure (no per-token device-adjacent allocation; the
// pick scratch is caller-pinned BEFORE the world forms).
inline GenEngineAdapter::Pick make_fabric_pick(net::CollectiveBus* bus,
                                               int rank, int world,
                                               uint16_t* pick_scratch,
                                               int64_t vocab,
                                               int pick_timeout_ms = 60000) {
  return [bus, rank, world, pick_scratch, vocab, pick_timeout_ms](
             const GlmDiagnosticModel::Outputs& out) -> int32_t {
    const glm_sample::Candidate local = glm_sample::local_max(
        out.logits.data(), static_cast<int>(out.lm_vocab_count),
        out.lm_vocab_begin);
    const int32_t t = bus_greedy_pick(*bus, rank, world, local,
                                      pick_scratch, pick_timeout_ms);
    if (t < 0 || t >= vocab)
      throw std::runtime_error("fabric pick out of range: " +
                               std::to_string(t));
    return t;
  };
}

// The fabric SAMPLER (M6 6b, the eager engines): the request's penalized
// slice through the exact candidate/LSE fold (bus_sampling_prefix at
// kSamplingCandidates per rank); when the prefix cannot decide, the exact
// fallback — the penalized slices gathered as one bulk collective and the
// SAME decision over the complete list under the transported normalizer,
// with the same draw — and rank 0's digest echoed either way. Both scratch
// buffers are caller-pinned BEFORE the world forms:
// fabric_sampling_prefix_scratch_elems(world) and
// sampling_gather_scratch_elems(vocab) words.
inline size_t fabric_sampling_prefix_scratch_elems(int world) {
  return sampling_prefix_scratch_elems(world, kSamplingCandidates);
}

inline GenEngineAdapter::Sample make_fabric_sample(
    net::CollectiveBus* bus, int rank, int world, uint16_t* prefix_scratch,
    uint16_t* gather_scratch, int64_t vocab, int pick_timeout_ms = 60000) {
  if (bus == nullptr || prefix_scratch == nullptr || gather_scratch == nullptr)
    throw std::invalid_argument("fabric sample: null bus/scratch");
  return [bus, rank, world, prefix_scratch, gather_scratch, vocab,
          pick_timeout_ms](const GlmDiagnosticModel::Outputs& out,
                           const glm_sample::Params& p, glm_sample::Rng& rng,
                           const std::vector<int32_t>& context)
             -> glm_sample::Result {
    step_timing::Scope tick(step_timing::kPick);
    const int count = static_cast<int>(out.lm_vocab_count);
    const int begin = out.lm_vocab_begin;
    const glm_sample::PrefixDecision d = bus_sampling_prefix(
        *bus, rank, world, out.logits.data(), count, begin,
        static_cast<int>(vocab), p, rng, context, kSamplingCandidates,
        prefix_scratch, pick_timeout_ms);
    glm_sample::Result r;
    if (d.resolved) {
      r = d.result;
    } else {
      // The exact fallback (DESIGN §10): the penalized slices to every rank,
      // the complete decision under the normalizer the prefix step already
      // folded, the draw the prefix left untouched.
      std::vector<float> adjusted(out.logits.begin(),
                                  out.logits.begin() + count);
      glm_sample::apply_penalties(adjusted.data(), count, begin, p,
                                  glm_sample::count_context(context));
      std::vector<float> full;
      bus_gather_logits(*bus, rank, world, adjusted.data(), count, begin,
                        static_cast<int>(vocab), gather_scratch,
                        pick_timeout_ms, &full);
      r = glm_sample::sample_complete_logits(full.data(),
                                             static_cast<int>(vocab),
                                             d.normalizer, p, rng);
      bus_check_decision_digest(*bus, rank, /*resolved=*/true, r,
                                d.normalizer, prefix_scratch,
                                pick_timeout_ms, "fabric sample fallback");
    }
    if (r.token < 0 || r.token >= vocab)
      throw std::runtime_error("fabric sample out of range: " +
                               std::to_string(r.token));
    return r;
  };
}

// M6.6a Phase 2: adaptive scalar/row-batched graphs. Each physical request
// slot lazily gets the exact Phase-1 scalar capture; one fixed batch covers
// every configured slot, with rows [slot, speculative-row] (T=1 plain, T=2
// MTP). Below the measured crossover the adapter replays each live scalar
// variant; above it, one batch replay advances them all. Closed batch slots
// derive position -1 and remain padding. Both paths return one independently
// judged token vector per live slot.
class GlmGraphEngineAdapter final : public glm::SchedulerEngine {
 public:
  GlmGraphEngineAdapter(GlmDiagnosticModel* model, net::CollectiveBus* bus,
                        int rank, int world, uint16_t* pick_scratch,
                        int64_t vocab, int pick_timeout_ms = 60000,
                        int batch_min_live = 4)
      : model_(model),
        bus_(bus),
        rank_(rank),
        vocab_(vocab),
        pick_timeout_ms_(pick_timeout_ms) {
    if (model_ == nullptr || bus_ == nullptr || pick_scratch == nullptr)
      throw std::invalid_argument("graph engine: null model/bus/pick scratch");
    slots_ = model_->max_session_requests();
    rows_per_request_ = model_->mtp_enabled() ? 2 : 1;
    if (slots_ < 1 || slots_ * rows_per_request_ >
                          GlmDiagnosticModel::kDecodeRows)
      throw std::invalid_argument(
          "graph engine: request slots * speculative rows exceed the fixed "
          "decode-row ceiling");
    prefill_pick_ = make_fabric_pick(bus_, rank_, world, pick_scratch, vocab_,
                                     pick_timeout_ms_);
    picker_ = std::make_unique<GlmDevicePicker>(*bus_, rank_, world,
                                                pick_timeout_ms_);
    pending_.assign(static_cast<size_t>(slots_), -1);
    draft_.assign(static_cast<size_t>(slots_), -1);
    live_.assign(static_cast<size_t>(slots_), false);
    reserved_.assign(static_cast<size_t>(slots_), false);
    scalar_execs_.assign(static_cast<size_t>(slots_), nullptr);
    batch_min_live_ = slots_ == 1
                          ? 1
                          : std::clamp(batch_min_live, 1, slots_);
    if (batch_min_live_ != batch_min_live)
      DGPP_LOG_INFO(
          "rank {}: graph batch crossover {} clamped to {} — the fixed batch "
          "has {} slot{}, so the batch is selected only at full occupancy",
          rank_, batch_min_live, batch_min_live_, slots_,
          slots_ == 1 ? "" : "s");
  }

  ~GlmGraphEngineAdapter() override {
    for (cudaGraphExec_t exec : scalar_execs_)
      if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (batch_exec_ != nullptr) cudaGraphExecDestroy(batch_exec_);
  }
  GlmGraphEngineAdapter(const GlmGraphEngineAdapter&) = delete;
  GlmGraphEngineAdapter& operator=(const GlmGraphEngineAdapter&) = delete;

  int max_concurrent_requests() const override { return slots_; }
  int decode_batch_capacity() const override { return slots_; }
  int batch_min_live() const { return batch_min_live_; }

  // Startup warm-up: records every scalar variant and, above one slot, the
  // row batch BEFORE the first client request, so no capture (record +
  // instantiate, ~10 ms each, up to slots+1 of them) lands on a live
  // stream. Opens ONE throwaway session at a time — slot by slot, with
  // `warm_prompt` (eager prefill: collectives on the bus, so every rank must
  // call this at the same point) — captures that slot's scalar variant (and
  // the row batch while the first slot is open; a capture bakes addresses,
  // occupancy is derived at replay), then closes it. One live warm session
  // keeps the pool footprint at a single prompt. Nothing of the warm
  // sessions survives: close pushes position 0 to the device and releases
  // the blocks, and a slot's next prefill rewrites its state. Requires an
  // idle engine.
  void warm_captures(const std::vector<int64_t>& warm_prompt) {
    if (warm_prompt.empty())
      throw std::invalid_argument("graph engine: warm prompt must not be empty");
    for (int req = 0; req < slots_; ++req)
      if (live_[static_cast<size_t>(req)])
        throw std::logic_error("graph engine: warm_captures needs an idle "
                               "engine");
    for (int req = 0; req < slots_; ++req) {
      (void)prefill(req, warm_prompt);
      try {
        ensure_scalar_graph(req);
        if (req == 0 && slots_ > 1) ensure_batch_graph();
      } catch (...) {
        close(req);
        throw;
      }
      close(req);
    }
    batch_feeds_dirty_ = true;
    DGPP_LOG_INFO(
        "rank {}: warm capture complete — {} scalar variant{}{} recorded "
        "before the first request",
        rank_, slots_, slots_ == 1 ? "" : "s",
        slots_ > 1 ? " and the row batch" : "");
  }
  int64_t pool_blocks_total() const override {
    return model_->dsa_blocks_total();
  }
  int64_t pool_blocks_in_use() const override {
    return model_->dsa_blocks_in_use();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return model_->dsa_blocks_for_tokens(tokens);
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    check_req(req);
    if (live_[static_cast<size_t>(req)])
      throw std::logic_error("graph engine: prefill on a live request");
    try {
      const int32_t first = prefill_pick_(model_->session_prefill(req, prompt));
      pending_[static_cast<size_t>(req)] = first;
      if (model_->mtp_enabled()) {
        // session_prefill filled the draft cache through the prompt. Advance
        // its one-row lag over the first generated token and pick the initial
        // proposal directly from device logits (decode mirrors may already be
        // disabled after this graph's first capture).
        (void)model_->session_draft(req, {first});
        draft_[static_cast<size_t>(req)] =
            picker_->run(model_->stream(), scalar_pick_inputs(1)).next;
      }
      reserved_[static_cast<size_t>(req)] = false;
      live_[static_cast<size_t>(req)] = true;
      // Prefill and the eager initial draft use d_tokens_ as scratch. Once
      // the fixed batch exists, restore every live slot's persistent feed,
      // not just the newly admitted one. Scalar variants stage/seed their
      // compact row-zero feed immediately before each replay.
      if (batch_exec_ != nullptr) seed_all_live_batch();
      return first;
    } catch (...) {
      // session_prefill opens the slot before any later pick/draft can fail.
      // Make a failed admission recoverable rather than leaking its blocks.
      model_->session_close(req);
      pending_[static_cast<size_t>(req)] = -1;
      draft_[static_cast<size_t>(req)] = -1;
      live_[static_cast<size_t>(req)] = false;
      reserved_[static_cast<size_t>(req)] = false;
      throw;
    }
  }

  void reserve(int req, int64_t tokens) override {
    check_live(req, "reserve");
    model_->session_reserve_blocks(req, tokens);
    reserved_[static_cast<size_t>(req)] = true;
  }

  std::vector<int32_t> step(int req) override {
    std::vector<std::vector<int32_t>> out = step_batch({req});
    return std::move(out[0]);
  }

  std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs) override {
    validate_batch(reqs);
    const bool batched = slots_ > 1 &&
                         reqs.size() >= static_cast<size_t>(batch_min_live_);
    log_mode_change(batched, reqs.size());
    if (!batched) {
      std::vector<std::vector<int32_t>> batches;
      batches.reserve(reqs.size());
      for (const int req : reqs) batches.push_back(step_scalar(req));
      if (batch_exec_ != nullptr) batch_feeds_dirty_ = true;
      return batches;
    }

    ensure_batch_graph();
    model_->session_graph_use_batch_contract(rows_per_request_);
    if (batch_feeds_dirty_) seed_all_live_batch();
    model_->session_graph_stage_batch();
    replay(batch_exec_, batch_variant());

    std::vector<std::vector<int32_t>> batches;
    batches.reserve(reqs.size());
    for (const int req : reqs)
      batches.push_back(collect_verdict(req, /*verdict_request=*/req));
    batch_feeds_dirty_ = false;
    return batches;
  }

  void close(int req) override {
    check_live(req, "close");
    model_->session_close(req);
    live_[static_cast<size_t>(req)] = false;
    reserved_[static_cast<size_t>(req)] = false;
    pending_[static_cast<size_t>(req)] = -1;
    draft_[static_cast<size_t>(req)] = -1;
  }

 private:
  void check_req(int req) const {
    if (req < 0 || req >= slots_)
      throw std::out_of_range("graph engine: request slot " +
                              std::to_string(req));
  }
  void check_live(int req, const char* op) const {
    check_req(req);
    if (!live_[static_cast<size_t>(req)])
      throw std::logic_error(std::string("graph engine: ") + op +
                             " on a closed request");
  }

  GlmDevicePicker::Inputs scalar_pick_inputs(int slot) const {
    GlmDevicePicker::Inputs in;
    in.logits = model_->device_logits();
    in.rows = 1;
    in.vocab_count = model_->lm_vocab_count();
    in.vocab_begin = model_->lm_vocab_begin();
    in.fed = model_->device_tokens();
    in.slot = slot;
    return in;
  }

  GlmDevicePicker::Inputs verify_pick_inputs() const {
    GlmDevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/0);
    in.rows = slots_ * rows_per_request_;
    in.requests = slots_;
    in.rows_per_request = rows_per_request_;
    in.positions = model_->device_positions();
    in.position_stride = rows_per_request_;
    return in;
  }

  GlmDevicePicker::Inputs draft_pick_inputs(
      const GlmPickVerdict* verify_verdicts) const {
    GlmDevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/1);
    in.rows = slots_;
    in.requests = slots_;
    in.rows_per_request = 1;
    in.positions = model_->device_positions();
    in.position_stride = rows_per_request_;
    in.row_select = verify_verdicts;
    in.source_row_stride = rows_per_request_;
    return in;
  }

  void validate_batch(const std::vector<int>& reqs) const {
    if (reqs.empty() || reqs.size() > static_cast<size_t>(slots_))
      throw std::invalid_argument("graph engine: invalid decode batch size");
    std::vector<bool> seen(static_cast<size_t>(slots_), false);
    for (const int req : reqs) {
      check_live(req, "step");
      if (!reserved_[static_cast<size_t>(req)])
        throw std::logic_error(
            "graph engine: step before lifetime reservation on slot " +
            std::to_string(req));
      if (seen[static_cast<size_t>(req)])
        throw std::invalid_argument("graph engine: duplicate request slot");
      seen[static_cast<size_t>(req)] = true;
    }
    for (int req = 0; req < slots_; ++req) {
      if (live_[static_cast<size_t>(req)] != seen[static_cast<size_t>(req)])
        throw std::logic_error(
            "graph engine: a physical replay must include every live slot");
    }
  }

  int batch_variant() const { return slots_; }

  void seed_all_live_batch() {
    model_->session_graph_use_batch_contract(rows_per_request_);
    for (int req = 0; req < slots_; ++req) {
      if (!live_[static_cast<size_t>(req)]) continue;
      if (model_->mtp_enabled())
        model_->session_graph_seed_tokens(
            req, {pending_[static_cast<size_t>(req)],
                  draft_[static_cast<size_t>(req)]});
      else
        model_->session_graph_seed_tokens(
            req, {pending_[static_cast<size_t>(req)]});
    }
    batch_feeds_dirty_ = false;
  }

  void ensure_capture_resources() {
    if (recorder_ != nullptr) return;
    model_->set_decode_route_traces(false);
    model_->set_decode_tail_mirrors(false);
    model_->session_graph_prepare();
    recorder_ =
        std::make_unique<GlmGraphRecordReducer>(*bus_, model_->stream());
  }

  template <typename Build>
  cudaGraphExec_t capture_variant(int variant, Build&& build) {
    ensure_capture_resources();
    GlmBoundaryReducer* eager = model_->set_boundary(recorder_.get());
    bool record_open = false;
    bool capture_open = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    try {
      std::string err;
      if (!bus_->graph_record_begin(&err, variant))
        throw std::runtime_error("graph engine record begin: " + err);
      record_open = true;
      DGPP_CUDA_OK(cudaStreamBeginCapture(model_->stream(),
                                          cudaStreamCaptureModeThreadLocal));
      capture_open = true;
      build();
      const cudaError_t capture_status =
          cudaStreamEndCapture(model_->stream(), &graph);
      capture_open = false;
      DGPP_CUDA_OK(capture_status);
      if (graph == nullptr)
        throw std::runtime_error("graph engine capture produced no graph");
      if (!bus_->graph_record_end(&err))
        throw std::runtime_error("graph engine record end: " + err);
      record_open = false;
      model_->set_boundary(eager);
      eager = nullptr;
      glm_check_decode_graph(graph, rank_,
                             "graph variant " + std::to_string(variant));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      cudaGraphDestroy(graph);
      return exec;
    } catch (...) {
      if (capture_open) {
        cudaGraph_t abandoned = nullptr;
        (void)cudaStreamEndCapture(model_->stream(), &abandoned);
        if (abandoned != nullptr) cudaGraphDestroy(abandoned);
      }
      if (record_open) {
        std::string ignored;
        (void)bus_->graph_record_end(&ignored);
      }
      if (eager != nullptr) model_->set_boundary(eager);
      if (graph != nullptr) cudaGraphDestroy(graph);
      if (exec != nullptr) cudaGraphExecDestroy(exec);
      throw;
    }
  }

  void ensure_scalar_graph(int req) {
    cudaGraphExec_t& exec = scalar_execs_[static_cast<size_t>(req)];
    if (exec != nullptr) return;
    const bool mtp = model_->mtp_enabled();
    exec = capture_variant(req, [&] {
      if (mtp) {
        model_->session_graph_capture_step(
            req,
            std::vector<int64_t>{pending_[static_cast<size_t>(req)],
                                 draft_[static_cast<size_t>(req)]},
            /*device_positions=*/true, /*device_tokens=*/true);
        GlmDevicePicker::Inputs verify = scalar_pick_inputs(/*slot=*/0);
        verify.rows = 2;
        picker_->record(model_->stream(), verify);
        model_->session_graph_capture_commit(
            req, picker_->device_verdict(0));
        model_->session_graph_capture_draft(
            req, picker_->device_verdict(0));
        GlmDevicePicker::Inputs draft = scalar_pick_inputs(/*slot=*/1);
        draft.row_select = picker_->device_verdict(0);
        picker_->record(model_->stream(), draft);
        model_->session_graph_capture_next_tokens(
            req, picker_->device_verdict(1));
      } else {
        model_->session_graph_capture_step(
            req, std::vector<int64_t>{pending_[static_cast<size_t>(req)]},
            /*device_positions=*/true, /*device_tokens=*/false);
        picker_->record(model_->stream(), scalar_pick_inputs(/*slot=*/0));
        model_->session_graph_capture_commit(
            req, picker_->device_verdict(0));
      }
    });
    if (batch_exec_ != nullptr)
      model_->session_graph_use_batch_contract(rows_per_request_);
    DGPP_LOG_INFO(
        "rank {}: serving scalar graph variant {} captured for request slot "
        "{} ({} rows{})",
        rank_, req, req, rows_per_request_, mtp ? ", MTP" : "");
  }

  void ensure_batch_graph() {
    if (batch_exec_ != nullptr) return;
    const bool mtp = model_->mtp_enabled();
    batch_exec_ = capture_variant(batch_variant(), [&] {
      model_->session_graph_capture_batch(rows_per_request_);
      picker_->record(model_->stream(), verify_pick_inputs());
      model_->session_graph_capture_commit_batch(picker_->device_verdict(0));
      if (mtp) {
        model_->session_graph_capture_draft_batch(
            picker_->device_verdict(0));
        picker_->record(model_->stream(),
                        draft_pick_inputs(picker_->device_verdict(0)));
        model_->session_graph_capture_next_tokens_batch(
            picker_->device_verdict(1));
      } else {
        model_->session_graph_capture_verify_next_tokens_batch(
            picker_->device_verdict(0));
      }
    });
    model_->session_graph_use_batch_contract(rows_per_request_);
    seed_all_live_batch();
    DGPP_LOG_INFO(
        "rank {}: serving row-batched graph variant {} captured ({} slots x "
        "{} rows = {} fixed rows{}, selected at {}+ live requests)",
        rank_, batch_variant(), slots_, rows_per_request_,
        slots_ * rows_per_request_, mtp ? ", MTP" : "", batch_min_live_);
  }

  void replay(cudaGraphExec_t exec, int variant) {
    std::string err;
    if (!bus_->graph_replay_arm(&err, variant))
      throw std::runtime_error("graph engine replay arm: " + err);
    DGPP_CUDA_OK(cudaGraphLaunch(exec, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    if (!bus_->graph_replay_finish(pick_timeout_ms_, &err))
      throw std::runtime_error("graph engine replay finish: " + err);
  }

  std::vector<int32_t> collect_verdict(int req, int verdict_request) {
    const GlmPickVerdict verify = picker_->verdict(0, verdict_request);
    if (verify.rows != rows_per_request_ || verify.accepted < 1 ||
        verify.accepted > rows_per_request_)
      throw std::runtime_error(
          "graph engine: invalid verdict for slot " + std::to_string(req) +
          " (rows " + std::to_string(verify.rows) + ", accepted " +
          std::to_string(verify.accepted) + ")");
    model_->session_graph_settle(req, verify.accepted);

    std::vector<int32_t> decided;
    decided.reserve(static_cast<size_t>(verify.accepted));
    for (int row = 0; row < verify.accepted; ++row) {
      const int32_t token = verify.winners[row];
      if (token < 0 || token >= vocab_)
        throw std::runtime_error(
            "graph engine pick out of range for slot " +
            std::to_string(req) + ": " + std::to_string(token));
      decided.push_back(token);
    }
    pending_[static_cast<size_t>(req)] = verify.next;
    if (model_->mtp_enabled()) {
      const GlmPickVerdict draft = picker_->verdict(1, verdict_request);
      if (draft.rows != 1 || draft.accepted != 1 || draft.next < 0 ||
          draft.next >= vocab_)
        throw std::runtime_error(
            "graph engine: invalid draft verdict for slot " +
            std::to_string(req));
      draft_[static_cast<size_t>(req)] = draft.next;
    }
    return decided;
  }

  std::vector<int32_t> step_scalar(int req) {
    ensure_scalar_graph(req);
    if (model_->mtp_enabled()) {
      const std::vector<int64_t> feed = {
          pending_[static_cast<size_t>(req)],
          draft_[static_cast<size_t>(req)]};
      model_->session_graph_stage(req, feed);
      model_->session_graph_seed_scalar_tokens(feed);
    } else {
      model_->session_graph_stage(req,
                                  pending_[static_cast<size_t>(req)]);
    }
    replay(scalar_execs_[static_cast<size_t>(req)], req);
    return collect_verdict(req, /*verdict_request=*/0);
  }

  void log_mode_change(bool batched, size_t live) {
    const int mode = batched ? 1 : 0;
    if (last_mode_ == mode) return;
    last_mode_ = mode;
    DGPP_LOG_INFO(
        "rank {}: adaptive decode selected {} graph at {} live request{} "
        "(batch crossover {})",
        rank_, batched ? "row-batched" : "scalar", live,
        live == 1 ? "" : "s", batch_min_live_);
  }

  GlmDiagnosticModel* model_ = nullptr;
  net::CollectiveBus* bus_ = nullptr;
  int rank_ = 0;
  int64_t vocab_ = 0;
  int pick_timeout_ms_ = 60000;
  int slots_ = 0;
  int rows_per_request_ = 1;
  int batch_min_live_ = 1;
  int last_mode_ = -1;  // 0 scalar variants, 1 fixed row batch
  bool batch_feeds_dirty_ = true;
  GenEngineAdapter::Pick prefill_pick_;
  std::unique_ptr<GlmDevicePicker> picker_;
  // Recorder storage is referenced by graph nodes and must outlive the exec.
  std::unique_ptr<GlmGraphRecordReducer> recorder_;
  std::vector<cudaGraphExec_t> scalar_execs_;
  cudaGraphExec_t batch_exec_ = nullptr;
  std::vector<int64_t> pending_;
  std::vector<int64_t> draft_;
  std::vector<bool> live_;
  std::vector<bool> reserved_;
};

}  // namespace dgpp
