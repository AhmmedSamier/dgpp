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
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "common/process_memory.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_prefix_arena.hpp"
#include "models/glm_graph_check.hpp"
#include "models/glm_speculative.hpp"
#include "models/glm_loader.hpp"
#include "models/glm_step_timing.hpp"
#include "kernels/glm_spec.hpp"
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
  auto gather_buffer = std::make_shared<std::vector<float>>();
  return [bus, rank, world, prefix_scratch, gather_scratch, vocab,
          pick_timeout_ms, gather_buffer](
             const GlmDiagnosticModel::Outputs& out,
             const glm_sample::Params& p, glm_sample::Rng& rng,
             const std::vector<int32_t>& context,
             const glm::TokenMask* mask, const float* bias) -> glm_sample::Result {
    step_timing::Scope tick(step_timing::kPick);
    const glm_sample::Result r = bus_sample_row(
        *bus, rank, world, out.logits.data(),
        static_cast<int>(out.lm_vocab_count), out.lm_vocab_begin,
        static_cast<int>(vocab), p, rng, context, kSamplingCandidates,
        prefix_scratch, gather_scratch, pick_timeout_ms, gather_buffer.get(),
        mask, bias);
    if (r.token < 0 || r.token >= vocab)
      throw std::runtime_error("fabric sample out of range: " +
                               std::to_string(r.token));
    return r;
  };
}

// The fabric row deciders of the eager SAMPLED speculator (glm_speculative.hpp
// SampledSpeculator): row 0's accept/residual and row 1's sample, each one
// fold plus the gather fallback plus rank 0's digest, in the same order on
// every rank.
inline SampledSpeculator::Row0 make_fabric_spec_row0(
    net::CollectiveBus* bus, int rank, int world, uint16_t* prefix_scratch,
    uint16_t* gather_scratch, int64_t vocab, int pick_timeout_ms = 60000) {
  auto gather_buffer = std::make_shared<std::vector<float>>();
  return [bus, rank, world, prefix_scratch, gather_scratch, vocab,
          pick_timeout_ms, gather_buffer](
             const GlmDiagnosticModel::Outputs& row0, int32_t draft,
             const glm_sample::Params& p, glm_sample::Rng& rng,
             const std::vector<int32_t>& context)
             -> glm_sample::SpecPrefixDecision {
    step_timing::Scope tick(step_timing::kPick);
    return bus_spec_accept(*bus, rank, world, row0.logits.data(),
                           static_cast<int>(row0.lm_vocab_count),
                           row0.lm_vocab_begin, static_cast<int>(vocab), draft,
                           p, rng, context, kSamplingCandidates,
                           prefix_scratch, gather_scratch, pick_timeout_ms,
                           gather_buffer.get());
  };
}

// M6.6a Phase 2: adaptive scalar/row-batched graphs. Each physical request
// slot lazily gets the exact Phase-1 scalar capture; one fixed batch covers
// every configured slot, with rows [slot, speculative-row] (T=1 plain, T=2
// MTP). Below the measured crossover the adapter replays each live scalar
// variant; above it, one batch replay advances them all. Closed batch slots
// derive position -1 and remain padding. Both paths return one independently
// judged token vector per live slot.
// SAMPLING (M6 6b, the device path): with both sampler scratch tables the
// plain (T=1) graphs carry the on-device sampling pick
// (kernels/glm_sample_pick.hpp) — per-slot device specs and count tables,
// the exact local top-k + slice normalizer, and the verdict that decides
// bit for bit what the host oracle decides or flags a fallback. The
// candidate width is the widest that fits the fixed batch's rows in one
// latency slot (at most kSamplingCandidates). A fallback is served between
// windows exactly as the eager engine's: the penalized row to the host, the
// bulk gather, the complete decision under the transported normalizer with
// the reserved draw, rank 0's digest — and the true token overrides the
// provisional one in the graph's token feed before the next replay. The
// MTP graphs sample too (the T=2 verify's accept test and row-1 sample on
// the device); a fallback there rolls the in-graph draft back to its ring
// snapshot, decides on the host (re-running the verify's second row eagerly
// when a provisionally rejected draft turns out to stand), re-drafts
// eagerly on the true rows and reseeds the [next, draft] feed.
class GlmGraphEngineAdapter final : public glm::SchedulerEngine {
 public:
  // `grammar_vocab` (optional, M6 6g): the tokenizer's token table; with
  // it and the device sampler the adapter constrains the pick per slot
  // (configure_constraint) — the masks live in a device table the captures
  // bake in, staged before every replay.
  GlmGraphEngineAdapter(GlmDiagnosticModel* model, net::CollectiveBus* bus,
                        int rank, int world, uint16_t* pick_scratch,
                        int64_t vocab, int pick_timeout_ms = 60000,
                        int batch_min_live = 4,
                        uint16_t* sample_prefix_scratch = nullptr,
                        uint16_t* sample_gather_scratch = nullptr,
                        int sampling_candidates_cap = kSamplingCandidates,
                        const glm::GrammarVocab* grammar_vocab = nullptr,
                        int prefix_slots = 0)
      : model_(model),
        bus_(bus),
        rank_(rank),
        world_(world),
        vocab_(vocab),
        pick_timeout_ms_(pick_timeout_ms),
        sample_prefix_scratch_(sample_prefix_scratch),
        sample_gather_scratch_(sample_gather_scratch),
        grammar_vocab_(grammar_vocab),
        arena_(model, model == nullptr ? 0 : prefix_slots) {
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
    if (sample_prefix_scratch_ != nullptr && sample_gather_scratch_ != nullptr) {
      // The planned width, or a lower cap (the loopback gates narrow it so
      // the tiny fixture vocabulary still exercises the fallback).
      if (sampling_candidates_cap < 1 ||
          sampling_candidates_cap > kSampleMaxCandidates)
        throw std::invalid_argument("graph engine: sampling candidates cap");
      candidates_ = glm_sample_candidates_that_fit(
          slots_ * rows_per_request_, world_,
          bus_->slot_bytes(net::BusMessageClass::kLatency),
          sampling_candidates_cap);
      if (candidates_ > 0) {
        sampling_ = true;
        prefill_sample_ = make_fabric_sample(
            bus_, rank_, world_, sample_prefix_scratch_,
            sample_gather_scratch_, vocab_, pick_timeout_ms_);
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_specs_),
                                sizeof(GlmSampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMemset(d_specs_, 0, sizeof(GlmSampleSpec) * slots_));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_specs_),
                                    sizeof(GlmSampleSpec) * slots_));
        for (int i = 0; i < slots_; ++i) h_specs_[i] = GlmSampleSpec{};
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_counts_),
                                sizeof(int32_t) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMemset(d_counts_, 0, sizeof(int32_t) * slots_ * vocab_));
        // The logit bias table (2026-09-06): one dense row per slot.
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bias_),
                                sizeof(float) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMemset(d_bias_, 0, sizeof(float) * slots_ * vocab_));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_bias_row_),
                                    sizeof(float) * vocab_));
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_prompt_ids_),
                                sizeof(int64_t) * model_->max_tokens()));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_prompt_ids_),
                                    sizeof(int64_t) * model_->max_tokens()));
        DGPP_CUDA_OK(cudaMallocHost(
            reinterpret_cast<void**>(&h_fallback_row_),
            sizeof(float) * model_->lm_vocab_count()));
        // The verify rows as the pick left them (penalized, masked), kept
        // for the host's MTP fallback: the in-graph draft's head reuses the
        // logits buffer, so after a replay the buffer holds the DRAFT's
        // rows — a fallback deciding over those decides over the wrong
        // distribution. A copy kernel node after the verify pick, before
        // the draft, preserves them ([slots][rows_per_request][count]).
        if (model_->mtp_enabled())
          DGPP_CUDA_OK(cudaMalloc(
              reinterpret_cast<void**>(&d_verify_logits_),
              sizeof(float) * static_cast<size_t>(slots_) * rows_per_request_ *
                  model_->lm_vocab_count()));
        // The token masks: one row per physical decode row, the header
        // word 0 (unconstrained) until a grammar stages one.
        mask_stride_ = glm_sample_mask_words(static_cast<int>(vocab_));
        const size_t mask_words =
            static_cast<size_t>(slots_) * rows_per_request_ * mask_stride_;
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_masks_),
                                sizeof(uint32_t) * mask_words));
        DGPP_CUDA_OK(cudaMemset(d_masks_, 0, sizeof(uint32_t) * mask_words));
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&h_masks_),
                                    sizeof(uint32_t) * mask_words));
        std::fill_n(h_masks_, mask_words, 0u);
        DGPP_LOG_INFO(
            "rank {}: graph engine samples on the device — {} candidates per "
            "rank ({} rows x {} ranks in a {}-byte latency slot; the planned "
            "width is {}); constrained decoding {}",
            rank_, candidates_, slots_ * rows_per_request_, world_,
            bus_->slot_bytes(net::BusMessageClass::kLatency),
            sampling_candidates_cap,
            grammar_vocab_ != nullptr && grammar_vocab_->usable()
                ? "available"
                : "unavailable (no grammar vocabulary)");
      } else {
        DGPP_LOG_WARN(
            "rank {}: graph engine cannot sample — not even one candidate per "
            "rank fits the latency slot for {} rows; greedy only",
            rank_, slots_ * rows_per_request_);
      }
    }
    picker_ = std::make_unique<GlmDevicePicker>(*bus_, rank_, world,
                                                pick_timeout_ms_, candidates_);
    params_.assign(static_cast<size_t>(slots_), glm_sample::greedy_params());
    rng_.assign(static_cast<size_t>(slots_), glm_sample::Rng{});
    grammar_.resize(static_cast<size_t>(slots_));
    bias_.resize(static_cast<size_t>(slots_));
    masks_.assign(static_cast<size_t>(slots_) * 2, glm::TokenMask{});
    context_.assign(static_cast<size_t>(slots_), {});
    report_.assign(static_cast<size_t>(slots_), false);
    pending_logprobs_.assign(static_cast<size_t>(slots_), {});
    slot_sampled_.assign(static_cast<size_t>(slots_), 0);
    slot_fallbacks_.assign(static_cast<size_t>(slots_), 0);
    pending_.assign(static_cast<size_t>(slots_), -1);
    draft_.assign(static_cast<size_t>(slots_), -1);
    hop_slot_.assign(static_cast<size_t>(slots_), -1);
    hop_position_.assign(static_cast<size_t>(slots_), 0);
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
    if (sampling_)
      DGPP_LOG_INFO(
          "rank {}: graph engine sampling summary — {} sampled decode steps, "
          "{} served by the exact gather fallback ({:.1f}%), {} candidates "
          "per rank",
          rank_, sampled_steps_, fallbacks_,
          sampled_steps_ ? 100.0 * static_cast<double>(fallbacks_) /
                               static_cast<double>(sampled_steps_)
                         : 0.0,
          candidates_);
    for (cudaGraphExec_t exec : scalar_execs_)
      if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (batch_exec_ != nullptr) cudaGraphExecDestroy(batch_exec_);
    if (d_specs_) cudaFree(d_specs_);
    if (h_specs_) cudaFreeHost(h_specs_);
    if (d_counts_) cudaFree(d_counts_);
    if (d_bias_) cudaFree(d_bias_);
    if (h_bias_row_) cudaFreeHost(h_bias_row_);
    if (d_prompt_ids_) cudaFree(d_prompt_ids_);
    if (h_prompt_ids_) cudaFreeHost(h_prompt_ids_);
    if (h_fallback_row_) cudaFreeHost(h_fallback_row_);
    if (d_masks_) cudaFree(d_masks_);
    if (h_masks_) cudaFreeHost(h_masks_);
    if (d_verify_logits_) cudaFree(d_verify_logits_);
  }
  GlmGraphEngineAdapter(const GlmGraphEngineAdapter&) = delete;
  GlmGraphEngineAdapter& operator=(const GlmGraphEngineAdapter&) = delete;

  int max_concurrent_requests() const override { return slots_; }
  int decode_batch_capacity() const override { return slots_; }
  // The MTP verify writes two rows per step (next and the draft).
  int max_tokens_per_step() const override { return rows_per_request_; }
  int batch_min_live() const { return batch_min_live_; }
  int sampling_candidates() const { return candidates_; }

  bool supports_sampling() const override { return sampling_; }
  bool supports_logprobs() const override { return sampling_; }
  void configure_sampling(int req, const glm_sample::Params& sampling,
                          uint64_t seed) override {
    check_req(req);
    glm_sample::validate_params(sampling);
    if (sampling.temperature > 0.0f && !sampling_)
      throw std::logic_error(
          "graph engine: no device sampler (engines without the sampler "
          "scratch are greedy-only)");
    params_[static_cast<size_t>(req)] = sampling;
    rng_[static_cast<size_t>(req)] = glm_sample::Rng{seed, 0};
    context_[static_cast<size_t>(req)].clear();
    report_[static_cast<size_t>(req)] = false;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    if (!sampling_) return;
    GlmSampleSpec spec;
    spec.temperature = sampling.temperature;
    spec.top_p = sampling.top_p;
    spec.min_p = sampling.min_p;
    spec.repetition_penalty = sampling.repetition_penalty;
    spec.frequency_penalty = sampling.frequency_penalty;
    spec.presence_penalty = sampling.presence_penalty;
    spec.top_k = sampling.top_k;
    spec.logprobs = -1;
    spec.seed = seed;
    spec.counter = 0;
    push_spec(req, spec);
    // A fresh context: the prompt's counts arrive with the prefill.
    DGPP_CUDA_OK(cudaMemsetAsync(d_counts_ + static_cast<size_t>(req) * vocab_,
                                 0, sizeof(int32_t) * vocab_,
                                 model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }
  void configure_logprobs(int req, int logprobs) override {
    check_req(req);
    if (logprobs >= 0 && !sampling_)
      throw std::logic_error("graph engine: no device sampler — no logprobs");
    report_[static_cast<size_t>(req)] = logprobs >= 0;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    if (!sampling_) return;
    GlmSampleSpec spec = h_specs_[req];
    spec.logprobs = logprobs;
    push_spec(req, spec);
  }
  std::vector<glm_sample::Result> take_logprobs(int req) override {
    check_req(req);
    std::vector<glm_sample::Result> out;
    out.swap(pending_logprobs_[static_cast<size_t>(req)]);
    return out;
  }
  // The logit bias (OpenAI's logit_bias, 2026-09-06): the slot's dense row
  // on the device (the pick kernels add it after the penalties, in place,
  // for the rows whose spec says biased — the graphs carry the table's
  // pointer, the spec flag turns a row on) and on the host (the prefill's
  // decision and the MTP fallback's row-1 sample go through the host
  // sampler). Needs the device sampler: a biased slot takes the full path.
  bool supports_logit_bias() const override { return sampling_; }
  void configure_logit_bias(int req,
                            const std::vector<glm::LogitBias>& bias) override {
    check_req(req);
    std::vector<float>& row = bias_[static_cast<size_t>(req)];
    if (bias.empty()) {
      if (row.empty()) return;
      row.clear();
      clear_bias_row(req);
      return;
    }
    if (!sampling_)
      throw std::logic_error(
          "graph engine: no device sampler — this engine cannot bias the pick");
    row.assign(static_cast<size_t>(vocab_), 0.0f);
    for (const glm::LogitBias& b : bias) {
      if (b.token < 0 || b.token >= vocab_)
        throw std::invalid_argument(
            "graph engine: logit_bias token outside the vocabulary");
      row[static_cast<size_t>(b.token)] = b.bias;
    }
    std::copy(row.begin(), row.end(), h_bias_row_);
    DGPP_CUDA_OK(cudaMemcpyAsync(d_bias_ + static_cast<size_t>(req) * vocab_,
                                 h_bias_row_, sizeof(float) * vocab_,
                                 cudaMemcpyHostToDevice, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    GlmSampleSpec spec = h_specs_[req];
    spec.biased = 1;
    push_spec(req, spec);
  }
  void clear_bias_row(int req) {
    if (!sampling_ || d_bias_ == nullptr) return;
    DGPP_CUDA_OK(cudaMemsetAsync(d_bias_ + static_cast<size_t>(req) * vocab_, 0,
                                 sizeof(float) * vocab_, model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    GlmSampleSpec spec = h_specs_[req];
    spec.biased = 0;
    push_spec(req, spec);
  }
  const float* bias_row(int req) const {
    const std::vector<float>& row = bias_[static_cast<size_t>(req)];
    return row.empty() ? nullptr : row.data();
  }
  bool supports_constraints() const override {
    return sampling_ && grammar_vocab_ != nullptr && grammar_vocab_->usable();
  }
  void configure_constraint(int req, const glm::GrammarSpec& grammar) override {
    check_req(req);
    grammar_[static_cast<size_t>(req)].reset();
    if (sampling_) clear_masks(req);
    if (!grammar.active()) return;
    if (!supports_constraints())
      throw std::logic_error(
          "graph engine: no grammar vocabulary — this engine cannot "
          "constrain the pick");
    // The opening state is re-derived at the prefill from the prompt.
    grammar_[static_cast<size_t>(req)] = std::make_unique<glm::GrammarState>(
        grammar_vocab_, grammar, /*prompt_opens_thinking=*/true);
  }

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
    return open_slot(req, prompt,
                     [&] { return model_->session_prefill(req, prompt); });
  }

  // ---- prefix cache (M7) --------------------------------------------------
  glm::SchedulerEngine::PrefixInfo prefix_info() const override {
    glm::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_.slots();
    info.step_tokens_max = rows_per_request_;
    info.align = model_->session_kpool();
    info.block_tokens = model_->dsa_block_tokens();
    info.chunk_tokens = GlmDiagnosticModel::prefill_chunk_tokens();
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         glm::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan == nullptr || plan->boundaries == nullptr)
      throw std::invalid_argument("graph engine: prefill_cached without a plan");
    return open_slot(req, prompt, [&] {
      GlmDiagnosticModel::SnapshotRequest snap;
      GlmDiagnosticModel::SnapshotRequest* snap_ptr = nullptr;
      if (plan->snap_slot >= 0) {
        snap = arena_.request(plan->snap_slot, plan->snap_position);
        snap_ptr = &snap;
      }
      GlmDiagnosticModel::Outputs out;
      if (plan->attach_slot >= 0) {
        if (arena_.position(plan->attach_slot) != plan->attach_position)
          throw std::logic_error(
              "graph engine: the attach slot's position differs from the plan");
        arena_.attach(req, plan->attach_slot);
        const std::vector<int64_t> suffix(prompt.begin() + plan->attach_position,
                                          prompt.end());
        out = model_->session_prefill_resume(req, suffix, *plan->boundaries,
                                             snap_ptr);
      } else {
        out = model_->session_prefill(req, prompt, *plan->boundaries, snap_ptr);
      }
      if (snap_ptr != nullptr) {
        arena_.commit(plan->snap_slot, snap);
        plan->snap_taken = snap.taken;
      }
      return out;
    });
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    check_live(req, "prefix_snapshot");
    arena_.snapshot(req, slot, position);
  }
  void prefix_arm_hop(int req, int slot, int64_t position) override {
    check_live(req, "prefix_arm_hop");
    hop_slot_[static_cast<size_t>(req)] = slot;
    hop_position_[static_cast<size_t>(req)] = position;
  }
  void prefix_release(int slot) override { arena_.release(slot); }
  glm::SchedulerEngine::PrefixEngineStats prefix_engine_stats() const override {
    glm::SchedulerEngine::PrefixEngineStats st;
    st.snapshots = arena_.snapshots();
    st.snapshot_ms = arena_.snapshot_ms();
    st.attaches = arena_.attaches();
    st.attach_ms = arena_.attach_ms();
    st.snapshot_bytes = static_cast<int64_t>(arena_.bytes());
    return st;
  }

 private:
  // The prefill's slot-side work around the model call that yields the
  // last row's logits (a cold prefill, or an attach + resume): the grammar's
  // opening state, the pick or the sampled draw, the sampled context and
  // its device count table, the draft block's first proposal, the batch
  // feeds. `run` opens the slot on the model; a failure after it closes
  // the slot again (a failed admission must not leak its blocks).
  template <typename Run>
  int32_t open_slot(int req, const std::vector<int64_t>& prompt, Run&& run) {
    check_req(req);
    if (live_[static_cast<size_t>(req)])
      throw std::logic_error("graph engine: prefill on a live request");
    try {
      std::unique_ptr<glm::GrammarState>& grammar =
          grammar_[static_cast<size_t>(req)];
      if (grammar) {
        // The grammar's opening state: thinking iff the prompt ends in
        // <think> (the template's generation prompt does).
        const glm::ChatMarkers& m = grammar_vocab_->markers();
        const bool opens = !prompt.empty() && m.think_open.available() &&
                           prompt.back() == m.think_open.id;
        grammar = std::make_unique<glm::GrammarState>(grammar_vocab_,
                                                      grammar->spec(), opens);
      }
      const bool sampled = full_path_slot(req);
      int32_t first = -1;
      {
        const GlmDiagnosticModel::Outputs out = run();
        if (sampled) {
          std::vector<int32_t> context(prompt.begin(), prompt.end());
          const glm::TokenMask* mask = nullptr;
          if (grammar && grammar->active()) {
            grammar->mask(&masks_[static_cast<size_t>(req) * 2]);
            if (masks_[static_cast<size_t>(req) * 2].constrained())
              mask = &masks_[static_cast<size_t>(req) * 2];
          }
          const glm_sample::Result r =
              prefill_sample_(out, params_[static_cast<size_t>(req)],
                              rng_[static_cast<size_t>(req)], context, mask,
                              bias_row(req));
          first = r.token;
          if (report_[static_cast<size_t>(req)])
            pending_logprobs_[static_cast<size_t>(req)].push_back(r);
        } else {
          first = prefill_pick_(out);
        }
        if (grammar) grammar->advance(first);
      }
      if (sampled) {
        // The prompt is the request's context on the device (the first
        // token is the next step's fed token and counts itself there), and
        // the prefill's draw moved the counter. The host mirror of the
        // context serves the MTP fallbacks' penalties.
        std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
        context.assign(prompt.begin(), prompt.end());
        context.push_back(first);
        const int n = static_cast<int>(prompt.size());
        if (n > model_->max_tokens())
          throw std::invalid_argument("graph engine: prompt exceeds max_tokens");
        std::copy(prompt.begin(), prompt.end(), h_prompt_ids_);
        DGPP_CUDA_OK(cudaMemcpyAsync(d_prompt_ids_, h_prompt_ids_,
                                     sizeof(int64_t) * n,
                                     cudaMemcpyHostToDevice, model_->stream()));
        glm_sample_count_tokens(d_counts_ + static_cast<size_t>(req) * vocab_,
                                d_prompt_ids_, n, static_cast<int>(vocab_),
                                model_->stream());
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
        push_counter(req);
      }
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

 public:
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
    for (const int req : reqs) stage_masks(req);
    replay(batch_exec_, batch_variant());

    std::vector<std::vector<int32_t>> batches;
    batches.reserve(reqs.size());
    for (const int req : reqs)
      batches.push_back(
          collect_verdict(req, /*verdict_request=*/req, /*batched=*/true));
    batch_feeds_dirty_ = false;
    return batches;
  }

  void close(int req) override {
    check_live(req, "close");
    // The slot's sampling tally, for the width sweep and the record: how
    // many of its stochastic steps the exact gather fallback served.
    if (sampling_ && slot_sampled_[static_cast<size_t>(req)] > 0)
      DGPP_LOG_INFO(
          "rank {}: slot {} closed: {} sampled decode steps, {} fallbacks",
          rank_, req, slot_sampled_[static_cast<size_t>(req)],
          slot_fallbacks_[static_cast<size_t>(req)]);
    slot_sampled_[static_cast<size_t>(req)] = 0;
    slot_fallbacks_[static_cast<size_t>(req)] = 0;
    model_->session_close(req);
    live_[static_cast<size_t>(req)] = false;
    reserved_[static_cast<size_t>(req)] = false;
    pending_[static_cast<size_t>(req)] = -1;
    draft_[static_cast<size_t>(req)] = -1;
    hop_slot_[static_cast<size_t>(req)] = -1;
    // A reopened slot is greedy until the scheduler arms it again — on the
    // device too, so a padded replay of this slot never draws.
    params_[static_cast<size_t>(req)] = glm_sample::greedy_params();
    rng_[static_cast<size_t>(req)] = glm_sample::Rng{};
    context_[static_cast<size_t>(req)].clear();
    report_[static_cast<size_t>(req)] = false;
    pending_logprobs_[static_cast<size_t>(req)].clear();
    grammar_[static_cast<size_t>(req)].reset();
    if (sampling_) {
      push_spec(req, GlmSampleSpec{});
      clear_masks(req);
    }
    if (!bias_[static_cast<size_t>(req)].empty()) {
      bias_[static_cast<size_t>(req)].clear();
      if (sampling_ && d_bias_ != nullptr) {
        DGPP_CUDA_OK(cudaMemsetAsync(d_bias_ + static_cast<size_t>(req) * vocab_,
                                     0, sizeof(float) * vocab_, model_->stream()));
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
      }
    }
  }

  // The slot's RNG state — the audit's view of the draws consumed.
  const glm_sample::Rng& rng(int req) const {
    check_req(req);
    return rng_[static_cast<size_t>(req)];
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

  // The scalar variant of request slot `req`: its own spec and count table
  // are baked into the recorded nodes (the verdict indexes request 0).
  GlmDevicePicker::Inputs scalar_sampling_inputs(int req, int rows = 1) const {
    GlmDevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/0);
    in.rows = rows;
    if (sampling_) {
      in.specs = d_specs_ + req;
      in.counts = d_counts_ + static_cast<size_t>(req) * vocab_;
      in.vocab_size = static_cast<int>(vocab_);
      in.masks = d_masks_ + static_cast<size_t>(req) * rows_per_request_ *
                                mask_stride_;
      in.mask_stride = mask_stride_;
      in.bias = d_bias_ + static_cast<size_t>(req) * vocab_;
    }
    return in;
  }

  GlmDevicePicker::Inputs verify_pick_inputs() const {
    GlmDevicePicker::Inputs in = scalar_pick_inputs(/*slot=*/0);
    in.rows = slots_ * rows_per_request_;
    in.requests = slots_;
    in.rows_per_request = rows_per_request_;
    in.positions = model_->device_positions();
    in.position_stride = rows_per_request_;
    if (sampling_) {
      in.specs = d_specs_;
      in.counts = d_counts_;
      in.vocab_size = static_cast<int>(vocab_);
      in.masks = d_masks_;
      in.mask_stride = mask_stride_;
      in.bias = d_bias_;
    }
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
        picker_->record(model_->stream(), scalar_sampling_inputs(req, /*rows=*/2));
        snapshot_verify_rows(req, /*rows=*/2, /*first_row=*/0);
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
        picker_->record(model_->stream(), scalar_sampling_inputs(req));
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
      if (mtp)
        snapshot_verify_rows(/*req=*/0, slots_ * rows_per_request_,
                             /*first_row=*/0);
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

  std::vector<int32_t> collect_verdict(int req, int verdict_request,
                                       bool batched) {
    const GlmPickVerdict verify = picker_->verdict(0, verdict_request);
    if (verify.rows != rows_per_request_ || verify.accepted < 1 ||
        verify.accepted > rows_per_request_)
      throw std::runtime_error(
          "graph engine: invalid verdict for slot " + std::to_string(req) +
          " (rows " + std::to_string(verify.rows) + ", accepted " +
          std::to_string(verify.accepted) + ")");
    model_->session_graph_settle(req, verify.accepted);
    // The prefix cache's hop snapshot (M7 under the two-row step): armed by
    // the scheduler for the aligned position row 1 sat on; taken from the
    // state after row 0 when both rows stood — here, before another slot's
    // scalar step can reuse the spec snapshot rows. A one-row verdict landed
    // ON the position; the scheduler's rolling snapshot follows.
    if (hop_slot_[static_cast<size_t>(req)] >= 0) {
      const int slot = hop_slot_[static_cast<size_t>(req)];
      const int64_t hop = hop_position_[static_cast<size_t>(req)];
      hop_slot_[static_cast<size_t>(req)] = -1;
      if (verify.accepted == 2) {
        if (model_->session_position(req) != hop + 1)
          throw std::runtime_error(
              "graph engine: slot " + std::to_string(req) + " sits at " +
              std::to_string(model_->session_position(req)) +
              " after a two-row step, the armed hop expects " +
              std::to_string(hop + 1));
        arena_.snapshot_post_row0(req, slot, hop,
                                  batched ? req * rows_per_request_ : 0);
      }
    }

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
    int32_t next = verify.next;
    const bool stochastic = sampled_slot(req);
    const bool full_path = full_path_slot(req);
    if (stochastic) {
      ++sampled_steps_;
      ++slot_sampled_[static_cast<size_t>(req)];
    }
    std::vector<glm_sample::Result>& report =
        pending_logprobs_[static_cast<size_t>(req)];
    if (full_path && !stochastic) {
      // The greedy request through the full path (logprobs or penalties):
      // the device decided the argmax rows under the raw normalizer; the
      // context mirror and the report follow.
      const GlmSampleOutcome& o = picker_->outcome(0, verdict_request);
      std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
      if (model_->mtp_enabled() && verify.accepted == 2)
        context.push_back(static_cast<int32_t>(draft_[static_cast<size_t>(req)]));
      context.push_back(next);
      if (report_[static_cast<size_t>(req)])
        for (int row = 0; row < verify.accepted; ++row)
          report.push_back(device_result(o, row, verify.winners[row]));
    }
    if (stochastic && model_->mtp_enabled()) {
      // The T=2 verify's sampled verdict. The context mirror follows the
      // device count table: the draft joins it only when it stands.
      const GlmSampleOutcome& o = picker_->outcome(0, verdict_request);
      glm_sample::Rng& rng = rng_[static_cast<size_t>(req)];
      std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
      const int32_t fed_draft =
          static_cast<int32_t>(draft_[static_cast<size_t>(req)]);
      if (!o.sampled)
        throw std::runtime_error(
            "graph engine: the device made no stochastic decision for a "
            "sampled MTP slot " + std::to_string(req));
      DGPP_LOG_DEBUG(
          "rank {}: slot {} device MTP sampling outcome: fallback_row {} "
          "accepted_draft {} counter {} winners {}/{} accepted {}",
          rank_, req, o.fallback_row, o.accepted_draft, o.counter,
          verify.winners[0], verify.winners[1], verify.accepted);
      if (o.fallback_row < 0) {
        if (o.counter != rng.counter + 2)
          throw std::runtime_error(
              "graph engine: the device consumed " +
              std::to_string(o.counter - rng.counter) +
              " draws for one T=2 step");
        rng.counter = o.counter;
        if (verify.accepted == 2) context.push_back(fed_draft);
        context.push_back(next);
        if (report_[static_cast<size_t>(req)])
          for (int row = 0; row < verify.accepted; ++row)
            report.push_back(device_result(o, row, verify.winners[row]));
      } else {
        next = serve_mtp_fallback(req, verdict_request, o, verify, fed_draft,
                                  &decided, batched, &report);
      }
    } else if (stochastic) {
      const GlmSampleOutcome& o = picker_->outcome(0, verdict_request);
      glm_sample::Rng& rng = rng_[static_cast<size_t>(req)];
      if (!o.sampled)
        throw std::runtime_error(
            "graph engine: the device made no stochastic decision for a "
            "sampled slot " + std::to_string(req));
      DGPP_LOG_DEBUG(
          "rank {}: slot {} device sampling outcome: fallback {} counter {} "
          "normalizer {:.6f} covered {:.6f} next {} logprob {:.4f}",
          rank_, req, o.fallback, o.counter, o.normalizer, o.covered_mass,
          verify.next, o.logprob);
      if (o.fallback) {
        if (o.counter != rng.counter)
          throw std::runtime_error(
              "graph engine: the device's counter drifted from the host's "
              "on a fallback");
        const glm_sample::Result r = serve_fallback(req, verdict_request, o);
        next = r.token;
        decided.back() = next;
        if (report_[static_cast<size_t>(req)]) report.push_back(r);
        // The graph fed itself the provisional token; the true one replaces
        // it before the next replay (the scalar variant stages pending_,
        // the batch keeps its persistent feed on the device).
        if (batched) model_->session_graph_seed_tokens(req, {next});
      } else {
        if (o.counter != rng.counter + 1)
          throw std::runtime_error(
              "graph engine: the device consumed " +
              std::to_string(o.counter - rng.counter) +
              " draws for one T=1 step");
        rng.counter = o.counter;
        if (report_[static_cast<size_t>(req)])
          report.push_back(device_result(o, 0, next));
      }
      context_[static_cast<size_t>(req)].push_back(next);
    }
    pending_[static_cast<size_t>(req)] = next;
    if (std::unique_ptr<glm::GrammarState>& grammar =
            grammar_[static_cast<size_t>(req)]) {
      // The committed tokens advance the grammar in transcript order; the
      // sampler never produced one outside its mask, so the state stays
      // live (a dead state would mean a contract breach upstream).
      for (const int32_t token : decided) grammar->advance(token);
      if (!grammar->active() && grammar->spec().active())
        DGPP_LOG_WARN(
            "rank {}: slot {} grammar died on a committed token — the pick "
            "left its mask (state {})",
            rank_, req, grammar->state_name());
    }
    if (model_->mtp_enabled() && !mtp_redrafted_) {
      const GlmPickVerdict draft = picker_->verdict(1, verdict_request);
      if (draft.rows != 1 || draft.accepted != 1 || draft.next < 0 ||
          draft.next >= vocab_)
        throw std::runtime_error(
            "graph engine: invalid draft verdict for slot " +
            std::to_string(req));
      draft_[static_cast<size_t>(req)] = draft.next;
    }
    mtp_redrafted_ = false;
    return decided;
  }

  // The sampled MTP step's fallback (DESIGN §9/§10): the in-graph draft ran
  // on provisional rows, so it rolls back to its ring snapshot first. Row 0
  // undecided (the device rejected provisionally, the commit kept the
  // post-row-0 state): the host gathers row 0 and decides; a reject is the
  // residual token; an accept means the draft stood after all, so the
  // verify's second row re-runs eagerly and its token is sampled on the
  // host. Row 1 undecided (the draft stood on the device): the host gathers
  // row 1 and samples it. Either way the true rows re-draft eagerly, the
  // [next, draft] feed is reseeded, the counter is pushed. Returns the new
  // next token and rewrites `decided`.
  int32_t serve_mtp_fallback(int req, int verdict_request,
                             const GlmSampleOutcome& o,
                             const GlmPickVerdict& verify, int32_t fed_draft,
                             std::vector<int32_t>* decided, bool batched,
                             std::vector<glm_sample::Result>* report) {
    const bool reporting = report_[static_cast<size_t>(req)];
    glm_sample::Rng& rng = rng_[static_cast<size_t>(req)];
    std::vector<int32_t>& context = context_[static_cast<size_t>(req)];
    const glm_sample::Params& p = params_[static_cast<size_t>(req)];
    const int count = model_->lm_vocab_count();
    const int begin = model_->lm_vocab_begin();
    (void)verdict_request;
    // The draft block ran on the provisional rows: back to its snapshot.
    model_->session_draft_rollback(req, verify.accepted);
    // The verify rows from the snapshot the graph took before its draft
    // (the logits buffer itself holds the draft head's rows now); the
    // snapshot is indexed [slot][row], the scalar and the batch alike.
    const auto gather_row = [&](size_t t) {
      const float* src = d_verify_logits_ +
                         (static_cast<size_t>(req) * rows_per_request_ + t) * count;
      DGPP_CUDA_OK(cudaMemcpyAsync(h_fallback_row_, src, sizeof(float) * count,
                                   cudaMemcpyDeviceToHost, model_->stream()));
      DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
      bus_gather_logits(*bus_, rank_, world_, h_fallback_row_, count, begin,
                        static_cast<int>(vocab_), sample_gather_scratch_,
                        pick_timeout_ms_, &fallback_full_);
    };
    std::vector<int64_t> rows;
    int32_t next = -1;
    if (o.fallback_row == 0) {
      if (o.counter != rng.counter || verify.accepted != 1)
        throw std::runtime_error(
            "graph engine: a row-0 fallback must leave the counter and "
            "commit one row");
      gather_row(0);
      check_gathered_row(o.covered_mass, o.normalizer, p, "row 0");
      const glm_sample::SpecPrefixDecision d0 =
          glm_sample::spec_accept_complete(fallback_full_.data(),
                                           static_cast<int>(vocab_),
                                           o.normalizer, fed_draft, p, rng);
      bus_check_decision_digest(*bus_, rank_, d0.accepted, d0.result,
                                o.normalizer, sample_prefix_scratch_,
                                pick_timeout_ms_, "graph MTP fallback row 0");
      if (reporting) report->push_back(d0.result);
      if (!d0.accepted) {
        next = d0.result.token;
        *decided = {next};
        rows = {next};
      } else {
        // The draft stood after all: it joins the device count table (the
        // verdict committed only the consumed token on its provisional
        // reject), then the verify's second row runs eagerly and is sampled.
        glm_sample_adjust_count(d_counts_ + static_cast<size_t>(req) * vocab_,
                                fed_draft, +1, static_cast<int>(vocab_),
                                model_->stream());
        DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
        context.push_back(fed_draft);
        const GlmDiagnosticModel::Outputs row1 =
            model_->session_verify(req, std::vector<int64_t>{fed_draft});
        // Row 1 under the mask staged for it (the grammar advanced by the
        // draft), as the device would have applied it.
        const glm::TokenMask& m1 = masks_[static_cast<size_t>(req) * 2 + 1];
        const glm_sample::Result r1 = prefill_sample_(
            row1, p, rng, context, m1.constrained() ? &m1 : nullptr,
            bias_row(req));
        if (reporting) report->push_back(r1);
        next = r1.token;
        *decided = {fed_draft, next};
        rows = {fed_draft, next};
      }
    } else {
      if (o.counter != rng.counter + 1 || verify.accepted != 2)
        throw std::runtime_error(
            "graph engine: a row-1 fallback must consume the accept draw "
            "and commit two rows");
      rng.counter = o.counter;
      context.push_back(fed_draft);
      gather_row(1);
      const glm_sample::Result r1 = glm_sample::sample_complete_logits(
          fallback_full_.data(), static_cast<int>(vocab_), o.normalizer1, p,
          rng);
      bus_check_decision_digest(*bus_, rank_, true, r1, o.normalizer1,
                                sample_prefix_scratch_, pick_timeout_ms_,
                                "graph MTP fallback row 1");
      if (reporting) {
        // Row 0 stood on the device; its report is the device's.
        report->push_back(device_result(o, 0, fed_draft));
        report->push_back(r1);
      }
      next = r1.token;
      *decided = {fed_draft, next};
      rows = {fed_draft, next};
    }
    if (next < 0 || next >= vocab_)
      throw std::runtime_error("graph engine MTP fallback token out of range");
    context.push_back(next);
    // The true rows through the draft block, eagerly; the greedy draft pick.
    const int32_t draft_new = prefill_pick_(model_->session_draft(req, rows));
    draft_[static_cast<size_t>(req)] = draft_new;
    mtp_redrafted_ = true;
    if (batched) model_->session_graph_seed_tokens(req, {next, draft_new});
    push_counter(req);
    ++fallbacks_;
    ++slot_fallbacks_[static_cast<size_t>(req)];
    return next;
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
    stage_masks(req);
    replay(scalar_execs_[static_cast<size_t>(req)], req);
    return collect_verdict(req, /*verdict_request=*/0, /*batched=*/false);
  }

  // The slot's masks for the coming replay (M6 6g): row 0 under the
  // grammar's current state, row 1 (MTP) under the state advanced by the
  // pending draft (row 1 is used only when the draft stands, in which case
  // that is exactly its position; a draft outside the mask kills the copy
  // and leaves row 1 unconstrained — it is discarded either way). Written
  // to the device table on the model stream ahead of the graph launch; an
  // unconstrained position writes a zero header.
  void stage_masks(int req) {
    if (!sampling_) return;
    std::unique_ptr<glm::GrammarState>& grammar = grammar_[static_cast<size_t>(req)];
    if (!grammar) return;  // the headers are zero (configure/close)
    glm::TokenMask& m0 = masks_[static_cast<size_t>(req) * 2];
    glm::TokenMask& m1 = masks_[static_cast<size_t>(req) * 2 + 1];
    grammar->mask(&m0);
    if (rows_per_request_ == 2) {
      glm::GrammarState after = *grammar;
      after.advance(draft_[static_cast<size_t>(req)]);
      after.mask(&m1);
    } else {
      m1.allowed = 0;
    }
    uint32_t* h = h_masks_ + static_cast<size_t>(req) * rows_per_request_ * mask_stride_;
    for (int t = 0; t < rows_per_request_; ++t) {
      const glm::TokenMask& m = t == 0 ? m0 : m1;
      uint32_t* row = h + static_cast<size_t>(t) * mask_stride_;
      row[0] = m.constrained() ? static_cast<uint32_t>(m.allowed) : 0u;
      if (m.constrained())
        std::copy(m.words.begin(), m.words.end(), row + 1);
    }
    const size_t words = static_cast<size_t>(rows_per_request_) * mask_stride_;
    DGPP_CUDA_OK(cudaMemcpyAsync(
        d_masks_ + static_cast<size_t>(req) * rows_per_request_ * mask_stride_,
        h, sizeof(uint32_t) * words, cudaMemcpyHostToDevice, model_->stream()));
  }
  // The gathered row is the row the device decided over iff its prefix's
  // covered mass under the device's normalizer is the device's, bit for
  // bit (the same candidates, the same masses, the same order) — the
  // invariant a wrong row breaks (the draft head's row did, silently,
  // until the verify snapshot). Loud, never papered over.
  void check_gathered_row(double device_covered, double normalizer,
                          const glm_sample::Params& p, const char* what) const {
    const std::vector<glm_sample::Candidate> top = glm_sample::local_topk(
        fallback_full_.data(), static_cast<int>(vocab_), 0, candidates_);
    double covered = 0.0;
    for (const glm_sample::Candidate& c : top)
      covered += detmath::exp_d(
          static_cast<double>(c.logit / p.temperature) - normalizer);
    if (std::memcmp(&covered, &device_covered, sizeof(double)) != 0)
      throw std::runtime_error(std::format(
          "graph engine: the gathered fallback {} is not the row the device "
          "decided over (covered mass {:.9g} vs the device's {:.9g}) — the "
          "verify snapshot and the pick disagree",
          what, covered, device_covered));
  }

  // Records the copy of the verify's logits rows [first_row, first_row +
  // rows) — as the pick left them — into the snapshot at slot `req`'s
  // rows, before the draft block overwrites the buffer (a kernel node:
  // the decode graph is kernels-only).
  void snapshot_verify_rows(int req, int rows, int first_row) {
    if (d_verify_logits_ == nullptr) return;
    const size_t count = static_cast<size_t>(model_->lm_vocab_count());
    glm_device_copy(
        d_verify_logits_ + static_cast<size_t>(req) * rows_per_request_ * count,
        model_->device_logits() + static_cast<size_t>(first_row) * count,
        sizeof(float) * static_cast<size_t>(rows) * count, model_->stream());
  }
  // Zero headers for the slot's rows: unconstrained until staged again.
  void clear_masks(int req) {
    if (!sampling_ || d_masks_ == nullptr) return;
    uint32_t* h = h_masks_ + static_cast<size_t>(req) * rows_per_request_ * mask_stride_;
    for (int t = 0; t < rows_per_request_; ++t) h[static_cast<size_t>(t) * mask_stride_] = 0u;
    for (int t = 0; t < rows_per_request_; ++t) {
      const size_t off =
          (static_cast<size_t>(req) * rows_per_request_ + t) * mask_stride_;
      DGPP_CUDA_OK(cudaMemcpyAsync(d_masks_ + off, h_masks_ + off,
                                   sizeof(uint32_t), cudaMemcpyHostToDevice,
                                   model_->stream()));
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }

  bool sampled_slot(int req) const {
    return sampling_ && params_[static_cast<size_t>(req)].temperature > 0.0f;
  }
  // The device outcome's report for one decided row, as the host's Result.
  static glm_sample::Result device_result(const GlmSampleOutcome& o, int row,
                                          int32_t token) {
    glm_sample::Result r;
    r.token = token;
    r.logprob = row == 0 ? o.logprob : o.logprob1;
    for (int i = 0; i < o.top_count[row]; ++i)
      r.top_logprobs.emplace_back(o.top_ids[row][i], o.top_logprobs[row][i]);
    return r;
  }
  // Slots that take the full sampling path on the device and the host
  // sampler at the prefill: stochastic ones, and greedy ones that report
  // logprobs or carry penalties (decided as the argmax under the raw
  // distribution, bitwise the greedy pick's token).
  bool full_path_slot(int req) const {
    if (!sampling_) return false;
    const glm_sample::Params& p = params_[static_cast<size_t>(req)];
    return p.temperature > 0.0f || report_[static_cast<size_t>(req)] ||
           p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f ||
           p.presence_penalty != 0.0f ||
           grammar_[static_cast<size_t>(req)] != nullptr ||
           !bias_[static_cast<size_t>(req)].empty();
  }

  void push_spec(int req, const GlmSampleSpec& spec) {
    h_specs_[req] = spec;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_specs_ + req, h_specs_ + req,
                                 sizeof(GlmSampleSpec), cudaMemcpyHostToDevice,
                                 model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
  }
  void push_counter(int req) {
    GlmSampleSpec spec = h_specs_[req];
    spec.counter = rng_[static_cast<size_t>(req)].counter;
    push_spec(req, spec);
  }

  // The exact fallback between windows (DESIGN §10): the device penalized
  // the request's logits row in place and folded the normalizer; the host
  // gathers the row from every rank, decides over the complete list with
  // the reserved draw, echoes rank 0's digest, and pushes the advanced
  // counter back to the device spec.
  glm_sample::Result serve_fallback(int req, int verdict_request,
                                    const GlmSampleOutcome& o) {
    const int count = model_->lm_vocab_count();
    const int begin = model_->lm_vocab_begin();
    const size_t row = static_cast<size_t>(verdict_request) * rows_per_request_;
    DGPP_CUDA_OK(cudaMemcpyAsync(h_fallback_row_,
                                 model_->device_logits() + row * count,
                                 sizeof(float) * count, cudaMemcpyDeviceToHost,
                                 model_->stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model_->stream()));
    bus_gather_logits(*bus_, rank_, world_, h_fallback_row_, count, begin,
                      static_cast<int>(vocab_), sample_gather_scratch_,
                      pick_timeout_ms_, &fallback_full_);
    glm_sample::Rng& rng = rng_[static_cast<size_t>(req)];
    check_gathered_row(o.covered_mass, o.normalizer,
                       params_[static_cast<size_t>(req)], "row");
    const glm_sample::Result r = glm_sample::sample_complete_logits(
        fallback_full_.data(), static_cast<int>(vocab_), o.normalizer,
        params_[static_cast<size_t>(req)], rng);
    bus_check_decision_digest(*bus_, rank_, /*resolved=*/true, r,
                              o.normalizer, sample_prefix_scratch_,
                              pick_timeout_ms_, "graph sample fallback");
    if (r.token < 0 || r.token >= vocab_)
      throw std::runtime_error("graph engine fallback token out of range: " +
                               std::to_string(r.token));
    push_counter(req);
    ++fallbacks_;
    ++slot_fallbacks_[static_cast<size_t>(req)];
    return r;
  }

 public:
  // Fallbacks served so far (the measurement record's fallback rate).
  uint64_t fallbacks() const { return fallbacks_; }

 private:

  void log_mode_change(bool batched, size_t live) {
    const int mode = batched ? 1 : 0;
    if (last_mode_ == mode) return;
    last_mode_ = mode;
    DGPP_LOG_DEBUG(
        "rank {}: adaptive decode selected {} graph at {} live request{} "
        "(batch crossover {})",
        rank_, batched ? "row-batched" : "scalar", live,
        live == 1 ? "" : "s", batch_min_live_);
  }

  GlmDiagnosticModel* model_ = nullptr;
  net::CollectiveBus* bus_ = nullptr;
  int rank_ = 0;
  int world_ = 1;
  int64_t vocab_ = 0;
  int pick_timeout_ms_ = 60000;
  uint16_t* sample_prefix_scratch_ = nullptr;
  uint16_t* sample_gather_scratch_ = nullptr;
  bool sampling_ = false;
  int candidates_ = 0;
  GlmSampleSpec* d_specs_ = nullptr;   // device [slots]
  GlmSampleSpec* h_specs_ = nullptr;   // pinned mirror
  int32_t* d_counts_ = nullptr;        // device [slots][vocab]
  int64_t* d_prompt_ids_ = nullptr;    // device [max_tokens]
  int64_t* h_prompt_ids_ = nullptr;    // pinned [max_tokens]
  float* h_fallback_row_ = nullptr;    // pinned [lm_vocab_count]
  std::vector<float> fallback_full_;
  // Constrained decoding (M6 6g): the grammar per slot, the two staged
  // masks per slot, the device/pinned mask table [slots*rows][stride].
  const glm::GrammarVocab* grammar_vocab_ = nullptr;
  std::vector<std::unique_ptr<glm::GrammarState>> grammar_;
  std::vector<glm::TokenMask> masks_;
  uint32_t* d_masks_ = nullptr;
  uint32_t* h_masks_ = nullptr;
  int mask_stride_ = 0;
  // The logit bias (2026-09-06): the device table [slots][vocab] the pick
  // reads for the rows whose spec says biased, a pinned row for uploads,
  // and the host copy per slot for the host-side decisions.
  float* d_bias_ = nullptr;
  float* h_bias_row_ = nullptr;
  std::vector<std::vector<float>> bias_;
  float* d_verify_logits_ = nullptr;  // device [slots][rows][count]: the
                                      // verify rows the MTP fallback decides over
  std::vector<glm_sample::Params> params_;
  std::vector<glm_sample::Rng> rng_;
  std::vector<std::vector<int32_t>> context_;  // prompt + decided, per slot
  std::vector<bool> report_;                    // logprobs asked, per slot
  std::vector<std::vector<glm_sample::Result>> pending_logprobs_;
  GenEngineAdapter::Sample prefill_sample_;
  uint64_t fallbacks_ = 0;
  uint64_t sampled_steps_ = 0;  // stochastic collects (the fallback rate's base)
  std::vector<uint64_t> slot_sampled_, slot_fallbacks_;  // per slot, reset at close
  bool mtp_redrafted_ = false;  // this collect re-drafted on the host
  int slots_ = 0;
  int rows_per_request_ = 1;
  std::vector<int> hop_slot_;          // per slot: the armed hop's arena slot, -1 none
  std::vector<int64_t> hop_position_;  // per slot: the armed hop's position
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
  PrefixArena arena_;  // the prefix cache's snapshot slots (M7)
};

}  // namespace dgpp
