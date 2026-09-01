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
#include <cstdint>
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

  // `eos_token_ids` — the config's end-of-sequence set (empty disables
  // EOS retirement, e.g. --no-eos).
  Scheduler(SchedulerEngine* engine, std::vector<int64_t> eos_token_ids);

  // Arrival order = FCFS priority. Throws on an empty/duplicate id, a
  // nonpositive max_steps, or a cancel_after outside [1, max_steps] —
  // manifest errors are operator errors, and they must fail identically
  // on every rank (a request that only exists on some ranks would
  // deadlock the fabric).
  void submit(SchedulerRequest request);

  // Runs every request to a terminal state. Single-threaded and
  // allocation-free on the hot path (the engine owns all buffers).
  // Throws std::runtime_error on admission deadlock or an engine
  // contract violation (out-of-range token).
  void run_to_completion();

  // Parallel to submit() order; entries reach their terminal Status
  // only via run_to_completion().
  const std::vector<Result>& results() const { return results_; }

 private:
  enum class State : int { kQueued, kActive, kTerminal };

  struct Request {
    SchedulerRequest spec;
    State state = State::kQueued;
    int slot = -1;
    int steps_done = 0;
    std::vector<int64_t> generated;
  };

  bool is_eos(int32_t token) const;
  int64_t reserve_blocks(const Request& r) const;
  int free_slot() const;
  // The oldest queued request whose reservation fits a free slot (no
  // head-of-line blocking), or -1.
  int next_admissible() const;
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
  std::vector<Result> results_;    // parallel to requests_
  int cursor_ = -1;                // last-stepped arrival (round-robin)
  int deferred_logged_ = -1;       // arrival of the current deferral log
};

}  // namespace dgpp::glm
