#pragma once
// The M6 Stage 4 generation service (PLAN M6 deliverable 4): the
// OpenAI-compatible chat/completions API over the deterministic
// scheduler. This header is the contract; the shapes live below.
//
// COMPATIBILITY CONTRACT (the loud-refusal discipline): every field the
// service ACCEPTS behaves exactly as the OpenAI API does; every field
// it does not implement yet answers 400 with the OpenAI error object
// naming the offending parameter. Silently ignoring a sampling knob
// would be the one dishonest behavior on the menu.
//
//   POST /v1/chat/completions   messages[] (string content), max_tokens
//                               or max_completion_tokens, stream,
//                               stream_options.include_usage; sampling:
//                               temperature absent-or-0 and top_p
//                               absent-or-1 only (greedy is the
//                               implemented sampler; the sampling stage
//                               is M6 deliverable 3).
//   POST /v1/completions        the legacy prompt API (string prompt).
//   GET  /v1/models, /v1/models/{id}
//   GET  /health               liveness (the fabric harnesses' probe).
//   GET  /v1/metrics           scheduler + service counters (ours).
//
// THREADING (two threads, one lock, tiny critical sections):
//   * HTTP thread — HttpServer::serve() calls handle()/idle()/
//     on_disconnect(). Parses, validates, tokenizes, renders the chat
//     template (rank 0 only — never on the fabric critical path),
//     creates request records, and formats SSE chunks from the rings.
//   * Engine thread — the app calls engine_pass() in a loop: it drains
//     the admission/cancel queue into the scheduler and runs ONE
//     scheduler quantum (sched.tick()). The scheduler observer (which
//     IS this service) appends token/retire events to the request
//     records under the lock.
//   * The single mutex covers the event queue and the record list;
//     sockets are only ever written by the HTTP thread (idle()).
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm_scheduler.hpp"
#include "service/http_server.hpp"

namespace dgpp::service {

// The tokenizer + chat-template seam. The real binding (GlmFrontend)
// wraps the Stage 3/3b exact tokenizer and template; tests bind a fake
// so the whole HTTP/SSE/lifecycle stack runs without a model cache.
class ModelFrontend {
 public:
  virtual ~ModelFrontend() = default;
  // text -> token ids (the exact ByteLevel-BPE encode).
  virtual std::vector<int64_t> encode_text(std::string_view text) const = 0;
  // ids -> text with special tokens skipped (the stream's content
  // semantics; decode(ids) == decode(ids[0..n-1]) + suffix deltas, so
  // incremental text is the suffix diff of successive full decodes).
  virtual std::string decode_ids(const std::vector<int64_t>& ids) const = 0;
  // The OpenAI messages array (minijson DOM: [{role, content}, ...])
  // rendered through the checkpoint's chat template, generation prompt
  // appended — the prompt the scheduler will prefill.
  virtual std::string render_chat(const minijson::Value& messages) const = 0;
};

struct ServiceConfig {
  // What /v1/models reports and what requests must name in "model".
  std::string model_id;
  int default_max_tokens = 256;  // when the request omits max_tokens
  int queue_limit = 64;          // admission bound; beyond → 503
};

class GenerationService : public HttpHandler,
                          public dgpp::glm::SchedulerObserver {
 public:
  GenerationService(const ServiceConfig& cfg,
                    dgpp::glm::SchedulerEngine* engine,
                    const ModelFrontend* frontend,
                    std::vector<int64_t> eos_token_ids);

  // ---- HttpHandler (HTTP thread) --------------------------------------
  void handle(const HttpRequest& req, HttpResponseWriter& w) override;
  void idle() override;
  void on_disconnect(uint64_t tag) override;

  // ---- engine-loop side (the app's engine thread) ---------------------
  // Applies every queued admission/cancel, then runs ONE scheduler
  // quantum. Returns whether work remains pending (the app may idle-
  // sleep when false; the pending-admission queue is drained first, so
  // arrivals always make the next pass productive).
  bool engine_pass();

  // Stops accepting: every pending record is answered 503 and the
  // queues drained. Called by the app on shutdown (before stopping the
  // HTTP server, so the answers can actually flush).
  void begin_shutdown();

  struct Stats {
    uint64_t requests_total = 0;
    uint64_t requests_shed = 0;      // 503s at the door or at admission
    uint64_t requests_cancelled = 0;  // client disconnects
    uint64_t tokens_out = 0;
    uint64_t rejects_bad = 0;        // 400-class refusals
  };
  Stats stats() const;

 private:
  struct StreamRecord {
    uint64_t tag = 0;        // on_disconnect correlation
    std::string id;          // response id == the scheduler request id
    std::string model;
    int64_t created_unix = 0;
    bool stream = false;
    bool include_usage = false;
    bool first_chunk_sent = false;
    bool done = false;         // retired or rejected — ready to finish
    bool reject_overloaded = false;
    bool writer_dead = false;  // on_disconnect fired; never touch it
    bool cancel_armed = false; // disconnect seen; cancel enqueued
    dgpp::glm::Scheduler::Result::Reason reason =
        dgpp::glm::Scheduler::Result::Reason::kNone;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::vector<int64_t> ids;  // generated so far
    std::string text;          // decoded so far (the suffix-diff base)
    std::string delta;         // unflushed text delta (the ring)
    HttpResponseWriter* writer = nullptr;  // HTTP thread only
  };

  // Routes (HTTP thread). Each parses, validates, and either responds
  // directly or creates a record + enqueues an admission.
  void route_chat_completions(const HttpRequest& req,
                              HttpResponseWriter& w);
  void route_completions(const HttpRequest& req, HttpResponseWriter& w);
  void route_models(const HttpRequest& req, HttpResponseWriter& w);
  void route_health(HttpResponseWriter& w) const;
  void route_metrics(HttpResponseWriter& w);

  // The OpenAI error body (never a bare string).
  void respond_error(HttpResponseWriter& w, int status,
                     const std::string& message, const std::string& type,
                     const std::string& param = "",
                     const std::string& code = "") const;

  // Admits a validated request: creates the record, hands the writer,
  // and enqueues the submit (the engine thread applies it).
  void enqueue_admission(std::shared_ptr<StreamRecord> record,
                         dgpp::glm::SchedulerRequest request);

  // Shared by both streaming and one-shot records: appends one token
  // (the observer, engine thread) and decodes the text suffix.
  void on_token(const std::string& id, int64_t token,
                int steps_done) override;
  void on_retire(const std::string& id,
                 const dgpp::glm::Scheduler::Result& result) override;

  // idle()'s record pump: flushes deltas, finishes done records.
  void pump_records();

  ServiceConfig cfg_;
  dgpp::glm::SchedulerEngine* engine_;
  const ModelFrontend* frontend_;
  dgpp::glm::Scheduler sched_;  // engine thread only (except try_submit
                                // under the lock via engine_pass)

  // Everything below lives under mutex_ (the one lock, held briefly).
  struct PendingAdmission {
    std::shared_ptr<StreamRecord> record;
    dgpp::glm::SchedulerRequest request;
  };
  struct PendingCancel {
    std::string scheduler_id;
  };
  std::vector<PendingAdmission> pending_admissions_;
  std::vector<PendingCancel> pending_cancels_;
  std::vector<std::shared_ptr<StreamRecord>> records_;
  dgpp::glm::Scheduler::Meters meters_;  // engine-published, mutex-guarded
  bool shutdown_ = false;
  mutable std::mutex mutex_;

  // HTTP-thread-only counters guarded by std::atomic where cross-thread.
  std::atomic<uint64_t> next_tag_{1};
  Stats stats_;  // written under mutex_ (cheap, exact)
};

}  // namespace dgpp::service
