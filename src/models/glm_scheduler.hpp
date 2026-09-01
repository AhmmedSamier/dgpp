#pragma once
// The Stage 2b scheduler (DESIGN §11): admission, ordering, and
// cancellation for concurrent requests over the session engine.
//
// THE INVARIANT THAT MAKES THIS WORK AT TP>1: every rank runs the SAME
// deterministic policy over the SAME request set, so every rank issues
// the same engine ops in the same order on the same slots. The engine's
// ops embed boundary folds and the distributed pick — collectives — so
// this identical-rank-order property (§11: "preserves identical rank
// order") is what keeps the fabric aligned. Concretely: no clocks, no
// thread arrival order, no unordered-container iteration may influence a
// decision — run_to_completion() is a pure function of (requests,
// engine meters, scripted cancellations).
//
// THE ISOLATION PROPERTY (what the gates pin and the fabric smoke
// proves): slots are independent and every op touches exactly one
// request's state, so a request's generated ids are invariant to
// whatever else is interleaved — including the cancellation of other
// requests. Concurrent execution is per-request identical to solo
// execution.
//
// POLICY (v1, deliberately small and fully deterministic):
//   * strict alternation — each tick admits AT MOST one queued request
//     and executes EXACTLY one decode step, so mid-answer requests
//     never wait behind a burst of prompt read-ins (the user's call);
//   * FCFS admission without head-of-line blocking — the OLDEST queued
//     request that FITS admits; a large stuck request does not block a
//     small one behind it (documented trade: an unbounded stream of
//     small requests can starve a large one — bounded queues, Stage 4);
//   * full-reserve admission — a request is admitted only when its
//     lifetime reservation (blocks_for(prompt + max_steps)) fits the
//     free pool AND a slot is open. No request can exhaust the pool
//     mid-generation; the cost is reserved-but-unused blocks when a
//     request EOSes early (the grow-on-demand + shed evolution is
//     documented in the Stage 2b record);
//   * round-robin decode over active requests, by arrival order;
//   * cancellation is deterministic (cancel_after N tokens) — Stage 4's
//     client disconnects map onto the same retire path;
//   * retirement frees the slot and blocks immediately; a queued
//     request admits on a later tick (live on the fabric smoke);
//   * no admissible request, no active requests, queue nonempty →
//     LOUD admission deadlock (the pool is sized below the smallest
//     reservation — an operator error, never papered over).
//
// WHAT THIS IS NOT (yet): batched decode. Steps are time-multiplexed
// (one request per op) — batching multiple requests' rows into one step
// needs per-row state indexing in the KDA recurrence (the DESIGN §9 MTP
// state-index surgery) and is a COMMITTED follow-up stage; the policy
// above is unchanged by op granularity, so that stage changes only what
// one step contains.
//
// STAGE 4 EVOLUTION (the service surface; additive — the manifest path
// is untouched by construction, and every gate below still pins it):
//   * tick() — one policy quantum: the run_to_completion() body
//     extracted verbatim (cancel sweep, at most one admission, exactly
//     one decode step). Dynamic arrival = submit between ticks;
//     run_to_completion() is now a tick loop, so manifest runs are the
//     degenerate "submit everything, then tick to quiet" case.
//   * SchedulerObserver — per-token and retire events fired INLINE on
//     the ticking thread (the SSE tap). Observers must not throw and
//     must not call back into the Scheduler.
//   * try_submit()/queue_limit — a bounded admission queue. A full
//     queue is a normal load-shed event (the service answers 503 and
//     the client retries); manifest errors still throw identically on
//     every rank.
//   * cancel(id) — an external retire (a client disconnect maps onto
//     the same path as scripted cancellation). Applied by a
//     fixed-position sweep at the TOP of the next tick, BEFORE any
//     admission or step: a cancelled request never pays one more engine
//     op, and the fabric journal (Stage 4b) can stamp the effect-tick
//     so every rank retires the same request at the same quantum.
//   * meters() — the /v1/metrics snapshot (queue depth, active slots,
//     pool use, cumulative tokens).
//
// THREADING: single-threaded by contract (§11 determinism leaves no
// room for lock-mediated interleavings on decision paths). The service
// funnels every mutation — submit, cancel — through its engine-loop
// queue; results(), meters(), and the observer all run on the ticking
// thread.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dgpp::glm {

// The engine seam the scheduler drives. The real binding (glm_gen_check)
// closes over GlmDiagnosticModel + the pick; the host gate binds a
// recording fake. prefill/step RETURN the picked token — the pick rides
// inside the op (at TP>1 it is the distributed greedy pick, a
// collective), which is what keeps the scheduler pure host code.
class SchedulerEngine {
 public:
  virtual ~SchedulerEngine() = default;

  // Engine session slots (== the decode-row bound at real dims: 8).
  virtual int max_concurrent_requests() const = 0;
  // DSA pool meters (a no-DSA model reports an unbounded pool).
  virtual int64_t pool_blocks_total() const = 0;
  virtual int64_t pool_blocks_in_use() const = 0;
  // Block count covering `tokens` tokens — the reserve arithmetic.
  virtual int64_t blocks_for_tokens(int64_t tokens) const = 0;

  // Opens slot `req` (fresh state), prefills `prompt`, picks the first
  // generated token. Returns a token id in [0, vocab).
  virtual int32_t prefill(int req, const std::vector<int64_t>& prompt) = 0;
  // Processes `prev_token` at slot `req`'s next position and picks the
  // next token.
  virtual int32_t step(int req, int64_t prev_token) = 0;
  // Retires the slot: blocks return to the pool; the slot may reopen.
  virtual void close(int req) = 0;
};

// One request, in arrival (manifest) order. `prompt` ids are validated by
// the ENGINE (the scheduler is model-agnostic by design).
struct SchedulerRequest {
  std::string id;
  std::vector<int64_t> prompt;
  int max_steps = 1;      // tokens to generate (prefill pick included)
  int cancel_after = 0;    // 0 = never; N = retire (Cancelled) once N
                           // tokens have been generated. N in [1, max_steps].
};

// The bounded admission queue at capacity (submit() only). A load-shed
// event, not a manifest error — the service answers 503 + Retry-After.
struct QueueFullError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Defined after Scheduler (it carries Scheduler::Result by value).
class SchedulerObserver;

class Scheduler {
 public:
  struct Result {
    enum class Status : int { kQueued, kActive, kDone, kCancelled };
    enum class Reason : int { kNone, kEos, kSteps, kCancelled };
    Status status = Status::kQueued;
    Reason reason = Reason::kNone;
    int slot = -1;          // engine slot used; -1 while queued
    int steps_done = 0;     // tokens generated (prefill pick included)
    std::vector<int64_t> generated;
  };

  // The /v1/metrics snapshot.
  struct Meters {
    int active = 0;        // requests with an open engine slot
    int queued = 0;        // admitted-not, waiting on slots/budget
    int terminal = 0;      // retired (any reason)
    int64_t pool_blocks_total = 0;
    int64_t pool_blocks_in_use = 0;
    int64_t tokens_generated = 0;  // cumulative across all requests
  };

  // `eos_token_ids` — the config's end-of-sequence set (empty disables
  // EOS retirement, e.g. --no-eos). `queue_limit` — the admission
  // queue's bound (0 = unbounded, the manifest default; a service sets
  // it so a full queue sheds load with a 503 instead of eating memory).
  Scheduler(SchedulerEngine* engine, std::vector<int64_t> eos_token_ids,
            int queue_limit = 0);

  // Arrival order = FCFS priority. Throws on an empty/duplicate id, a
  // nonpositive max_steps, or a cancel_after outside [1, max_steps] —
  // manifest errors are operator errors, and they must fail identically
  // on every rank (a request that only exists on some ranks would
  // deadlock the fabric). Throws QueueFullError when the bounded queue
  // is full.
  void submit(SchedulerRequest request);

  // The service form: false ONLY on a full bounded queue (a normal,
  // load-shedding event). Validation failures still throw — they are
  // client bugs, not load.
  bool try_submit(SchedulerRequest request);

  // Flags a live request for retirement at the next tick boundary (a
  // client disconnect). Returns false when the id is unknown or already
  // terminal — a late cancel is a no-op, never an error.
  bool cancel(const std::string& id);

  // One policy quantum: the cancel sweep, at most one admission, exactly
  // one decode step. Returns false when nothing is pending AFTER the
  // tick — the final retirement may ride on the false. Throws on
  // admission deadlock exactly like run_to_completion().
  bool tick();

  // Any queued or active request remains.
  bool has_pending() const;

  // Runs every request to a terminal state: a tick loop. Single-threaded
  // and allocation-free on the hot path (the engine owns all buffers).
  void run_to_completion();

  // Parallel to submit() order; entries reach their terminal Status
  // only via run_to_completion()/tick().
  const std::vector<Result>& results() const { return results_; }

  // Result by request id (nullptr when unknown) — the service's
  // non-streaming lookup.
  const Result* find(const std::string& id) const;

  // Lifecycle events (the SSE tap). Not owned; may be null.
  void set_observer(SchedulerObserver* observer) { observer_ = observer; }

  Meters meters() const;

 private:
  enum class State : int { kQueued, kActive, kTerminal };

  struct Request {
    SchedulerRequest spec;
    State state = State::kQueued;
    int slot = -1;
    int steps_done = 0;
    std::vector<int64_t> generated;
    bool cancel_requested = false;  // external cancel, applied at the
                                    // next tick's sweep
  };

  bool is_eos(int32_t token) const;
  int64_t reserve_blocks(const Request& r) const;
  int free_slot() const;
  // The oldest queued request whose reservation fits a free slot (no
  // head-of-line blocking), or -1.
  int next_admissible() const;
  // The submit() validations, shared by submit()/try_submit().
  void validate_new(const SchedulerRequest& request) const;
  int queued_count() const;
  void admit(int arrival);
  void step_one(int arrival);
  // Retire conditions are checked in this order: natural EOS first, then
  // scripted cancellation, then the steps cap — a cancelled request that
  // had already finished naturally reports Done (client intent cannot
  // rewrite history), and a cancel_at the cap reports Cancelled.
  void retire(int arrival, Result::Status status, Result::Reason reason);

  SchedulerEngine* engine_ = nullptr;
  std::vector<int64_t> eos_ids_;
  std::vector<Request> requests_;  // arrival order — the FCFS order
  std::vector<int> slots_;         // engine slot -> arrival index, or -1
  std::vector<Result> results_;     // parallel to requests_
  int cursor_ = -1;                // last-stepped arrival (round-robin)
  int deferred_logged_ = -1;       // arrival of the current deferral log
  SchedulerObserver* observer_ = nullptr;
  int queue_limit_ = 0;            // 0 = unbounded
  int64_t tokens_generated_ = 0;   // cumulative on_token counter
};

// Streaming lifecycle events for the service (SSE). Fired inline on the
// ticking thread — see the threading note above.
class SchedulerObserver {
 public:
  virtual ~SchedulerObserver() = default;
  // One generated token (the prefill pick is steps_done == 1). Always
  // fires BEFORE the matching retire when the token ends the request.
  virtual void on_token(const std::string& id, int64_t token,
                        int steps_done) = 0;
  // The request's terminal state, exactly once (EOS, steps cap,
  // scripted or external cancellation all land here).
  virtual void on_retire(const std::string& id,
                         const Scheduler::Result& result) = 0;
};

}  // namespace dgpp::glm
