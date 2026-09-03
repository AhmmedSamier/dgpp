#pragma once
// The fabric engine closure for serving (Stage 4b) and glm_gen_check's
// scheduler path — ONE seam, two apps, no drift. Real-mesh BusOptions
// (the budgets the first fabric gate run found, 2026-08-30) + the
// distributed greedy pick over the vocab-sharded head.
//
// CUDA-app-only header: it drags the bus (verbs) headers. Host gates
// fake the engine instead; nothing in dgpp_service includes this.
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
  // One slot holds a speculative verify's kSpecRows rows of hidden 4096
  // (the collective kernels are element-driven; the slot only strides).
  o.lat_slot_bytes = GlmDiagnosticModel::kSpecRows * 4096 * 2;
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

// M6.6a phase 1: the request-slot-0 graph engine. The scheduler still sees
// the same prefill token, but step() now returns every token newly decided by
// one replay. With MTP, the replay verifies [pending, draft] and returns the
// verify winners through the accepted row: one token on a miss, two on a hit.
// Those are precisely the tokens after the already-emitted `pending` token.
// Without MTP this is the T=1 form and returns one winner per replay.
//
// The bus owns one graph-record era per process, so this adapter deliberately
// exposes one slot and records once. A closed request releases its model state;
// the next request prefills slot 0, reseeds the baked graph addresses, and
// reuses the graph unchanged.
class GlmGraphEngineAdapter final : public glm::SchedulerEngine {
 public:
  GlmGraphEngineAdapter(GlmDiagnosticModel* model, net::CollectiveBus* bus,
                        int rank, int world, uint16_t* pick_scratch,
                        int64_t vocab, int pick_timeout_ms = 60000)
      : model_(model),
        bus_(bus),
        rank_(rank),
        vocab_(vocab),
        pick_timeout_ms_(pick_timeout_ms) {
    if (model_ == nullptr || bus_ == nullptr || pick_scratch == nullptr)
      throw std::invalid_argument("graph engine: null model/bus/pick scratch");
    if (model_->max_session_requests() != 1)
      throw std::invalid_argument(
          "graph engine phase 1 requires a model with exactly one request "
          "slot");
    prefill_pick_ = make_fabric_pick(bus_, rank_, world, pick_scratch, vocab_,
                                     pick_timeout_ms_);
    picker_ = std::make_unique<GlmDevicePicker>(*bus_, rank_, world,
                                                pick_timeout_ms_);
  }

  ~GlmGraphEngineAdapter() override {
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
  }
  GlmGraphEngineAdapter(const GlmGraphEngineAdapter&) = delete;
  GlmGraphEngineAdapter& operator=(const GlmGraphEngineAdapter&) = delete;

  int max_concurrent_requests() const override { return 1; }
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
    if (live_)
      throw std::logic_error("graph engine: prefill on a live request");
    try {
      const int32_t first = prefill_pick_(model_->session_prefill(req, prompt));
      pending_ = first;
      if (model_->mtp_enabled()) {
        // session_prefill filled the draft cache through the prompt. Advance
        // its one-row lag over the first generated token and pick the initial
        // proposal directly from device logits (decode mirrors may already be
        // disabled after this graph's first capture).
        (void)model_->session_draft(req, {first});
        draft_ = picker_->run(model_->stream(), pick_inputs(1, 1, nullptr)).next;
        if (graph_exec_ != nullptr)
          model_->session_graph_seed_tokens(req, {pending_, draft_});
      }
      reserved_ = false;
      live_ = true;
      return first;
    } catch (...) {
      // session_prefill opens the slot before any later pick/draft can fail.
      // Make a failed admission recoverable rather than leaking its blocks.
      model_->session_close(req);
      pending_ = -1;
      draft_ = -1;
      throw;
    }
  }

  void reserve(int req, int64_t tokens) override {
    check_live(req, "reserve");
    model_->session_reserve_blocks(req, tokens);
    reserved_ = true;
  }

  std::vector<int32_t> step(int req) override {
    check_live(req, "step");
    if (!reserved_)
      throw std::logic_error("graph engine: step before lifetime reservation");
    if (graph_exec_ == nullptr) capture(req);

    if (!model_->mtp_enabled())
      model_->session_graph_stage(req, pending_);
    std::string err;
    if (!bus_->graph_replay_arm(&err))
      throw std::runtime_error("graph engine replay arm: " + err);
    DGPP_CUDA_OK(cudaGraphLaunch(graph_exec_, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    if (!bus_->graph_replay_finish(pick_timeout_ms_, &err))
      throw std::runtime_error("graph engine replay finish: " + err);

    const GlmPickVerdict verify = picker_->verdict(0);
    if (verify.accepted < 1 ||
        verify.accepted > (model_->mtp_enabled() ? 2 : 1))
      throw std::runtime_error("graph engine: invalid accepted-row count " +
                               std::to_string(verify.accepted));
    model_->session_graph_settle(req, verify.accepted);

    std::vector<int32_t> decided;
    decided.reserve(static_cast<size_t>(verify.accepted));
    for (int row = 0; row < verify.accepted; ++row) {
      const int32_t token = verify.winners[row];
      if (token < 0 || token >= vocab_)
        throw std::runtime_error("graph engine pick out of range: " +
                                 std::to_string(token));
      decided.push_back(token);
    }
    pending_ = verify.next;
    if (model_->mtp_enabled()) draft_ = picker_->verdict(1).next;
    return decided;
  }

  void close(int req) override {
    check_live(req, "close");
    model_->session_close(req);
    live_ = false;
    reserved_ = false;
    pending_ = -1;
    draft_ = -1;
  }

 private:
  void check_req(int req) const {
    if (req != 0)
      throw std::out_of_range("graph engine phase 1 only owns request slot 0");
  }
  void check_live(int req, const char* op) const {
    check_req(req);
    if (!live_)
      throw std::logic_error(std::string("graph engine: ") + op +
                             " on a closed request");
  }

  GlmDevicePicker::Inputs pick_inputs(
      int rows, int slot, const GlmPickVerdict* row_select) const {
    GlmDevicePicker::Inputs in;
    in.logits = model_->device_logits();
    in.rows = rows;
    in.vocab_count = model_->lm_vocab_count();
    in.vocab_begin = model_->lm_vocab_begin();
    in.fed = model_->device_tokens();
    in.slot = slot;
    in.row_select = row_select;
    return in;
  }

  void capture(int req) {
    const bool mtp = model_->mtp_enabled();
    model_->set_decode_route_traces(false);
    model_->set_decode_tail_mirrors(false);
    model_->session_graph_prepare();
    recorder_ =
        std::make_unique<GlmGraphRecordReducer>(*bus_, model_->stream());
    GlmBoundaryReducer* eager = model_->set_boundary(recorder_.get());
    bool record_open = false;
    bool capture_open = false;
    cudaGraph_t graph = nullptr;
    try {
      std::string err;
      if (!bus_->graph_record_begin(&err))
        throw std::runtime_error("graph engine record begin: " + err);
      record_open = true;
      DGPP_CUDA_OK(cudaStreamBeginCapture(model_->stream(),
                                          cudaStreamCaptureModeThreadLocal));
      capture_open = true;
      if (mtp) {
        model_->session_graph_capture_step(
            req, std::vector<int64_t>{pending_, draft_},
            /*device_positions=*/true, /*device_tokens=*/true);
        picker_->record(model_->stream(), pick_inputs(2, 0, nullptr));
        model_->session_graph_capture_commit(req, picker_->device_verdict(0));
        model_->session_graph_capture_draft(req, picker_->device_verdict(0));
        picker_->record(model_->stream(),
                        pick_inputs(1, 1, picker_->device_verdict(0)));
        model_->session_graph_capture_next_tokens(
            req, picker_->device_verdict(1));
      } else {
        model_->session_graph_capture_step(
            req, std::vector<int64_t>{pending_},
            /*device_positions=*/true, /*device_tokens=*/false);
        picker_->record(model_->stream(), pick_inputs(1, 0, nullptr));
        model_->session_graph_capture_commit(req, picker_->device_verdict(0));
      }
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
      DGPP_CUDA_OK(
          cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0));
      cudaGraphDestroy(graph);
      graph = nullptr;
      if (mtp) model_->session_graph_seed_tokens(req, {pending_, draft_});
      DGPP_LOG_INFO("rank {}: serving decode graph captured ({} rows{})",
                    rank_, mtp ? 2 : 1, mtp ? ", MTP" : "");
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
      recorder_.reset();
      throw;
    }
  }

  GlmDiagnosticModel* model_ = nullptr;
  net::CollectiveBus* bus_ = nullptr;
  int rank_ = 0;
  int64_t vocab_ = 0;
  int pick_timeout_ms_ = 60000;
  GenEngineAdapter::Pick prefill_pick_;
  std::unique_ptr<GlmDevicePicker> picker_;
  // Recorder storage is referenced by graph nodes and must outlive the exec.
  std::unique_ptr<GlmGraphRecordReducer> recorder_;
  cudaGraphExec_t graph_exec_ = nullptr;
  int64_t pending_ = -1;
  int64_t draft_ = -1;
  bool live_ = false;
  bool reserved_ = false;
};

}  // namespace dgpp
