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
//   * round-robin decode over active requests, by arrival order; engines
//     advertise how many requests one physical pass can carry (one for the
//     eager engine, all occupied rows for the row-batched graph);
//   * cancellation is deterministic (cancel_after N tokens) — Stage 4's
//     client disconnects map onto the same retire path;
//   * retirement frees the slot and blocks immediately; a queued
//     request admits on a later tick (live on the fabric smoke);
//   * no admissible request, no active requests, queue nonempty →
//     LOUD admission deadlock (the pool is sized below the smallest
//     reservation — an operator error, never papered over).
//
// BATCH GRANULARITY: the policy is independent of the engine's physical
// step width. Scalar engines advertise one and preserve the original
// time-multiplexed op stream. A row-batched engine advertises a larger
// bound; one tick then hands it the next round-robin slice in one call and
// applies the returned token batches in that same canonical order.
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

#include "models/glm_prefix_cache.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_tool_grammar.hpp"

namespace dgpp::glm {

// One logit_bias entry (OpenAI's logit_bias, 2026-09-06): the token's id
// and the bias added to its logit before the pick, on every rank
// identically (it rides the journal with the request).
struct LogitBias {
  int32_t token = 0;
  float bias = 0.0f;
};

// The engine seam the scheduler drives. The real binding (glm_gen_check)
// closes over GlmDiagnosticModel + the pick; the host gate binds a
// recording fake. The pick rides inside each op (at TP>1 it is a
// distributed collective), which is what keeps the scheduler pure host
// code. A step returns every token newly decided by that engine pass: one
// for ordinary decode, or 1..T for a speculative step.
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
  // Pins the slot's lifetime block reservation. The scheduler calls this
  // immediately after prefill and before the first step; `tokens` is
  // prompt.size() + max_steps. Device-driven graphs rely on this because
  // they cannot grow a host-owned block table during replay.
  virtual void reserve(int req, int64_t tokens) = 0;
  // Advances slot `req` and returns the newly decided tokens in transcript
  // order. Must return at least one nonnegative token. The engine owns its
  // pending input token(s), which lets a recorded graph feed itself.
  virtual std::vector<int32_t> step(int req) = 0;
  // Maximum number of independent request slots one physical decode pass
  // can advance. Scalar engines inherit one. A row-batched graph overrides
  // this together with step_batch(); the value must stay fixed for the
  // engine's lifetime and cannot exceed max_concurrent_requests().
  virtual int decode_batch_capacity() const { return 1; }
  // Advances `reqs` in order and returns one token vector per request in the
  // same order. The default deliberately lowers to scalar step() calls, so
  // existing engines keep their exact op stream. Batch-capable engines
  // override this method with one physical pass.
  virtual std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs);
  // Retires the slot: blocks return to the pool; the slot may reopen.
  virtual void close(int req) = 0;
  // The most tokens one step() can write into the slot's KV (1 for plain
  // decode; T for a speculative engine whose verify writes T rows). The
  // grow-on-demand policy sizes each reservation's headroom by it.
  virtual int max_tokens_per_step() const { return 1; }

  // ---- sampling (M6 6b) ---------------------------------------------------
  // An engine that can draw stochastically advertises it; the scheduler
  // then hands every admitted request's spec to its slot immediately BEFORE
  // prefill (the prefill pick is the first draw). The default engine is
  // greedy-only: the scheduler refuses a stochastic request at submit —
  // identically on every rank — so this default only ever sees greedy specs,
  // and treats anything else as the contract violation it is.
  virtual bool supports_sampling() const { return false; }
  virtual void configure_sampling(int req, const glm_sample::Params& sampling,
                                  uint64_t seed) {
    (void)req;
    (void)seed;
    if (sampling.temperature > 0.0f)
      throw std::logic_error(
          "SchedulerEngine: this engine samples greedily only");
  }
  // Logprobs: an engine that reports them returns, after each prefill/step,
  // one Result per token that op returned (in order) for a slot whose spec
  // asked (sampling.logprobs >= 0 through configure_sampling with
  // `logprobs`). The default reports none; the scheduler refuses a request
  // that asks at submit.
  virtual bool supports_logprobs() const { return false; }
  // Arms slot `req` to report logprobs (`logprobs` >= 0: the top-N count,
  // also carried in the spec's sampling.logprobs) or not (-1). Called right
  // after configure_sampling, before the prefill pick.
  virtual void configure_logprobs(int req, int logprobs) {
    (void)req;
    if (logprobs >= 0)
      throw std::logic_error("SchedulerEngine: this engine reports no logprobs");
  }
  virtual std::vector<glm_sample::Result> take_logprobs(int req) {
    (void)req;
    return {};
  }

  // ---- constrained decoding (M6 6g) ----------------------------------------
  // An engine that can mask the pick advertises it; the scheduler then hands
  // every admitted request's grammar spec to its slot right after the
  // sampling spec, before the prefill pick (the first constrained position).
  // The engine keeps the grammar state per slot, advances it with every
  // token it commits, and applies the next position's mask on every rank
  // identically (the mask is a pure function of the spec, the committed
  // ids and the tokenizer). The default engine has no masks: the scheduler
  // refuses an active spec at submit, identically on every rank.
  virtual bool supports_constraints() const { return false; }
  virtual void configure_constraint(int req, const GrammarSpec& grammar) {
    (void)req;
    if (grammar.active())
      throw std::logic_error(
          "SchedulerEngine: this engine cannot constrain the pick");
  }
  // The logit bias (2026-09-06): an engine that can add a per-request
  // bias to the logits before the pick advertises it; the scheduler hands
  // every admitted request's entries to its slot right after the grammar
  // (an empty list clears the slot's bias). The default engine has none.
  virtual bool supports_logit_bias() const { return false; }
  virtual void configure_logit_bias(int req,
                                    const std::vector<LogitBias>& bias) {
    (void)req;
    if (!bias.empty())
      throw std::logic_error(
          "SchedulerEngine: this engine cannot bias the pick (logit_bias)");
  }

  // ---- prefix cache (M7 stage B, DESIGN §8) ----------------------------------
  // An engine with a snapshot arena advertises it; the scheduler then keeps
  // the index (glm_prefix_cache.hpp) and drives these ops — identically on
  // every rank, because every decision is a function of the journaled
  // request stream. The default engine has no arena: the scheduler never
  // calls any of them and the op stream is exactly the pre-cache one.
  struct PrefixInfo {
    int arena_slots = 0;       // snapshot slots the engine holds (0: none)
    int64_t align = 1;         // every snapshot position is a multiple (kpool)
    int64_t block_tokens = 0;  // the DSA block; 0 = no pool (nothing pinned)
    int64_t chunk_tokens = 2048;  // the prefill's chunk (the cold cuts)
    int step_tokens_max = 1;   // tokens one step may commit (2: the MTP step)
  };
  virtual PrefixInfo prefix_info() const { return {}; }
  // The hop (M7 under a multi-token step): the request's next step may
  // commit past `position` (= its committed count + 1, pool-aligned) without
  // stopping there; if it commits two tokens, the engine takes the state
  // after the step's first row — the state at `position` — into arena slot
  // `slot` before returning from that step. A one-token step lands ON the
  // position and the scheduler's rolling snapshot follows at the next tick.
  // The arm holds until the slot's next step (or its close).
  virtual void prefix_arm_hop(int req, int slot, int64_t position) {
    (void)req;
    (void)slot;
    (void)position;
  }
  // The prefill with the cache: `boundaries` (absolute positions, ascending)
  // are the request's structural cut positions — the cold chunking cuts at
  // every chunk multiple and at the aligned image of each; attach_slot >= 0
  // opens the slot from that arena entry (at attach_position tokens, one
  // of the prompt's cuts) and prefills prompt[attach_position..] only;
  // snap_slot >= 0 asks for a snapshot into that arena slot when a chunk
  // ends at snap_position (a cut, > attach_position), reported in
  // snap_taken. Returns the first token, exactly as prefill() does.
  struct PrefixPrefill {
    const std::vector<int64_t>* boundaries = nullptr;
    int attach_slot = -1;
    int64_t attach_position = 0;
    int snap_slot = -1;
    int64_t snap_position = 0;
    bool snap_taken = false;  // out
  };
  virtual int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                                 PrefixPrefill* plan) {
    (void)req;
    (void)prompt;
    (void)plan;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // The live slot's state at its current position (`position` tokens
  // committed, a multiple of align — the scheduler's view; the engine
  // verifies) into arena slot `slot`, replacing whatever the slot held.
  virtual void prefix_snapshot(int req, int slot, int64_t position) {
    (void)req;
    (void)slot;
    (void)position;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // Drops arena slot `slot`'s references (the blocks its entry pinned).
  virtual void prefix_release(int slot) {
    (void)slot;
    throw std::logic_error("SchedulerEngine: this engine has no prefix cache");
  }
  // The arena's measured costs, for the metrics.
  struct PrefixEngineStats {
    int64_t snapshots = 0;   // arena snapshots taken (prefill and rolling)
    double snapshot_ms = 0;  // their summed device time, where measured
    int64_t attaches = 0;
    double attach_ms = 0;
    int64_t snapshot_bytes = 0;  // one slot's state bytes
  };
  virtual PrefixEngineStats prefix_engine_stats() const { return {}; }
};

// One request, in arrival (manifest) order. `prompt` ids are validated by
// the ENGINE (the scheduler is model-agnostic by design).
struct SchedulerRequest {
  std::string id;
  std::vector<int64_t> prompt;
  int max_steps = 1;      // tokens to generate (prefill pick included)
  int cancel_after = 0;    // 0 = never; N = retire (Cancelled) once N
                           // tokens have been generated. N in [1, max_steps].
  // The sampling spec (glm_sampler.hpp's warper contract). The default is
  // GREEDY — temperature 0, no draw, no seed consumed — so every manifest
  // and gate that predates sampling keeps its exact op stream.
  glm_sample::Params sampling = glm_sample::greedy_params();
  // The counter RNG's seed. Rank 0 assigns one when the client omits it and
  // the journal carries it, so every rank draws the same sequence.
  uint64_t seed = 0;
  // Logprobs on the wire: -1 = none; N >= 0 = report every generated
  // token's log-probability and its top-N alternatives (sampling.logprobs
  // carries N to the sampler). Greedy requests report under the raw
  // distribution.
  int logprobs = -1;
  // Constrained decoding (M6 6g): the tool-call grammar the pick obeys
  // (tool_choice required / named / none, parallel_tool_calls false). The
  // default is inactive — unconstrained, the exact op stream every gate
  // pins. Rides the journal with the request.
  GrammarSpec grammar;
  // The prefix cache (M7): the prompt's structural boundaries — absolute
  // positions in [1, prompt.size()), ascending — where the cold prefill
  // cuts (at their pool-aligned images) so a snapshot of one turn is a cut
  // of the next; the service derives them from the role-marker tokens in
  // the prompt ids. Empty: cuts at chunk multiples only. `no_cache` opts
  // the request out: no attach, no snapshot. Both ride the journal.
  std::vector<int64_t> boundaries;
  bool no_cache = false;
  // The logit bias (2026-09-06): the request's entries, applied by the
  // engine on every rank; rides the journal.
  std::vector<LogitBias> logit_bias;
};

// The bounded admission queue at capacity (submit() only). A load-shed
// event, not a manifest error — the service answers 503 + Retry-After.
struct QueueFullError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Defined after Scheduler (it carries Scheduler::Result by value).
class SchedulerObserver;

// The admission policy (M6 6d). Full-reserve pins blocks_for(prompt +
// max_steps) at admission — a request never stalls mid-generation, at the
// cost of blocks held for tokens an early EOS never writes. Grow-on-demand
// reserves blocks_for(prompt + min(max_steps, window)) at admission and
// grows the reservation at tick top, BEFORE any engine op, whenever the
// next step would write past it — by `window_tokens` at a time, or by the
// minimum when the pool is short — and when even the minimum does not fit,
// SHEDS the youngest active request (retired Done / kPoolExhausted, the
// service's finish_reason "length") until it does; a request with no
// younger peer sheds itself. Every decision is a pure function of the
// scheduler's state, so the journal keeps it identical on every rank. The
// trade is explicit and the policy opt-in: optimistic admission for
// early-EOS workloads against a truncated answer for the youngest when
// everyone runs long (preemption by recompute — vLLM's answer — waits on a
// prefill fast enough to re-read a partial answer).
struct AdmissionPolicy {
  enum class Mode : int { kFullReserve = 0, kGrowOnDemand = 1 };
  Mode mode = Mode::kFullReserve;
  int window_tokens = 256;  // grow: the initial headroom and the growth step
  bool operator==(const AdmissionPolicy& o) const {
    return mode == o.mode && window_tokens == o.window_tokens;
  }
  bool operator!=(const AdmissionPolicy& o) const { return !(*this == o); }
  static const char* name(Mode m) {
    return m == Mode::kGrowOnDemand ? "grow" : "full";
  }
};

class Scheduler {
 public:
  struct Result {
    enum class Status : int { kQueued, kActive, kDone, kCancelled };
    // kStop (2026-09-06): the service's stop string matched — a natural end
    // for the client (finish_reason "stop"), journaled like a cancel.
    enum class Reason : int { kNone, kEos, kSteps, kCancelled, kPoolExhausted, kStop };
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
    int64_t reservations_grown = 0;  // grow-on-demand: growth events
    int64_t requests_shed_pool = 0;  // grow-on-demand: shed at exhaustion
    // The throughput line's counters (2026-09-06): prompts prefilled, their
    // tokens (all, and the ones actually computed — an attach skips the
    // prefix), decode passes and the request-rows they carried, and the
    // wall time spent inside the engine's prefill and step calls.
    int64_t prompts_prefilled = 0;
    int64_t prompt_tokens = 0;
    int64_t prompt_tokens_computed = 0;
    int64_t decode_steps = 0;
    int64_t decode_rows = 0;
    double prefill_ms = 0.0;
    double step_ms = 0.0;
    // The prefix cache (M7): its slots and live entries, the attach and
    // miss counts, the prompt tokens attaches skipped, the entries taken
    // (at prefill cuts / from rolling snapshots at close), rolling
    // snapshots, evictions, duplicates, snapshots skipped for want of a
    // slot, and the DSA blocks entries pin against the pool.
    int prefix_slots = 0;
    int prefix_entries = 0;
    int64_t prefix_hits = 0;
    int64_t prefix_misses = 0;
    int64_t prefix_tokens_saved = 0;
    int64_t prefix_snapshots = 0;
    int64_t prefix_close_entries = 0;
    int64_t prefix_rolling = 0;
    int64_t prefix_hops = 0;
    int64_t prefix_evictions = 0;
    int64_t prefix_duplicates = 0;
    int64_t prefix_skipped = 0;
    int64_t prefix_skipped_no_block = 0;
    int64_t prefix_blocks_pinned = 0;
  };

  // `eos_token_ids` — the config's end-of-sequence set (empty disables
  // EOS retirement, e.g. --no-eos). `queue_limit` — the admission
  // queue's bound (0 = unbounded, the manifest default; a service sets
  // it so a full queue sheds load with a 503 instead of eating memory).
  // `prefix_slots` (M7): the snapshot slots the prefix cache may use — -1
  // = every slot the engine's arena holds, 0 = the cache off, N = at most
  // N (must not exceed the arena: every rank must run the same count, so
  // rank 0's resolved value rides the warm record and the peers pass it).
  Scheduler(SchedulerEngine* engine, std::vector<int64_t> eos_token_ids,
            int queue_limit = 0, AdmissionPolicy policy = AdmissionPolicy{},
            int prefix_slots = -1);
  const AdmissionPolicy& admission_policy() const { return policy_; }
  // The prefix cache as configured: its slot count (0 = off) and the
  // running digest of every decision it made (the journal's cross-rank
  // check — identical streams, identical digests).
  int prefix_slots() const { return cache_.slots(); }
  uint64_t prefix_digest() const { return cache_.digest(); }
  const PrefixCache& prefix_cache() const { return cache_; }

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
  // The service's stop (2026-09-06): the client's stop string matched on
  // rank 0 — journaled like a cancel, applied at the next tick's sweep on
  // every rank, retiring the request as Done with Reason::kStop (the
  // service cuts the text at the match; the tokens after it are dropped).
  // A cancel that arrives with it wins.
  bool stop(const std::string& id);

  // One policy quantum: the cancel sweep, at most one admission, then one
  // engine pass over up to decode_batch_capacity() active requests. Returns
  // false when nothing is pending AFTER the tick — the final retirement may
  // ride on the false. Throws on admission deadlock exactly like
  // run_to_completion().
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
    bool stop_requested = false;    // the stop-string retire (Reason::kStop)
    int64_t reserved_tokens = 0;    // the slot's current reservation
    // The prefix cache (M7): the prompt's cuts and their prefix hashes
    // (computed once at submit), the entry the request attached to (its
    // index; -1 cold), the position it attached at, and the rolling
    // snapshot slot with the position it holds (-1: none yet).
    std::vector<int64_t> cuts;
    std::vector<uint64_t> cut_hashes;
    int attach_entry = -1;
    int64_t attach_position = 0;
    int rolling_slot = -1;
    int64_t rolling_position = -1;
    int64_t hop_armed = -1;    // the aligned position armed for the next step
    bool cache_off = false;    // the pool cannot hold the cache's blocks for it
  };
  // The admission plan the cache proposes for a queued request: the entry
  // to attach (or -1) and the position, and the cut a new entry would be
  // taken at (0: none).
  struct PrefixPlan {
    int attach_entry = -1;
    int64_t attach_position = 0;
    int64_t snap_position = 0;
  };
  bool cache_on(const Request& r) const {
    return cache_.enabled() && !r.spec.no_cache && !r.cache_off;
  }
  PrefixPlan plan_prefix(const Request& r) const;
  // The pool block a snapshot's private partial-block copy takes: one when
  // the position is not block-aligned, none otherwise (or without a pool).
  int64_t snapshot_blocks(int64_t position) const;
  // Whether a snapshot at `position` into a slot whose current snapshot
  // sits at `previous` (-1: none) needs a NEW pool block — the arena
  // releases the old copy before it takes the new one, so only a first
  // partial copy does.
  bool snapshot_needs_block(int64_t position, int64_t previous) const;
  // Frees pool blocks for a snapshot by evicting LRU unattached entries
  // until `need` blocks are free; false when nothing evictable remains.
  bool ensure_free_blocks(int64_t need, const std::string& id);
  // The blocks an admission must find FREE: the reservation less the full
  // blocks an attach shares.
  int64_t new_blocks(const Request& r, const PrefixPlan& plan) const;
  // Rolling snapshots (tick, before the step): every active cached request
  // at an aligned committed position it has not snapshotted yet.
  void rolling_snapshots();
  // A free arena slot, or the LRU unattached entry's (released on the
  // engine, an "evict" event under `id`); -1 when neither exists.
  int acquire_arena_slot(const std::string& id);
  // Releases a slot on the engine and returns it to the ledger.
  void free_arena_slot(int slot);
  void emit_prefix(const std::string& id, const char* op, int64_t position,
                   int slot);

  bool is_eos(int32_t token) const;
  // The reservation an admission pins: the lifetime under full-reserve,
  // prompt + min(max_steps, window) under grow-on-demand.
  int64_t initial_reserve_tokens(const SchedulerRequest& spec) const;
  int64_t reserve_blocks(const Request& r) const;
  // Grow-on-demand's tick-top pass (after the cancel sweep, before any
  // admission or step): every active request whose next step would write
  // past its reservation grows it, shedding the youngest when the pool
  // cannot cover even the minimum.
  void grow_reservations();
  int youngest_active_after(int arrival) const;
  int free_slot() const;
  // The oldest queued request whose reservation fits a free slot (no
  // head-of-line blocking), or -1. With the prefix cache on it may EVICT
  // unattached entries (LRU) to make the blocks a request needs.
  int next_admissible();
  // The submit() validations, shared by submit()/try_submit().
  void validate_new(const SchedulerRequest& request) const;
  int queued_count() const;
  void admit(int arrival);
  void step_batch(const std::vector<int>& arrivals);
  // Appends one token and applies terminal conditions in their canonical
  // order. Returns true when the request retired. `logprobs` (optional)
  // rides to the observer with it.
  bool append_token(int arrival, int32_t token,
                    const glm_sample::Result* logprobs = nullptr);
  // The engine's logprobs for the tokens it just returned, when the request
  // asked; empty otherwise. Throws when the engine returned a different
  // count than tokens.
  std::vector<glm_sample::Result> collect_logprobs(int arrival, int slot,
                                                   size_t tokens);
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
  int decode_batch_capacity_ = 1;  // fixed engine pass width
  int64_t tokens_generated_ = 0;   // cumulative on_token counter
  AdmissionPolicy policy_;
  int64_t grows_ = 0;              // growth events (meters)
  int64_t pool_sheds_ = 0;         // requests shed at exhaustion (meters)
  int64_t prompts_prefilled_ = 0;  // the throughput line's counters (meters)
  int64_t prompt_tokens_ = 0;
  int64_t prompt_tokens_computed_ = 0;
  int64_t decode_steps_ = 0;
  int64_t decode_rows_ = 0;
  double prefill_ms_ = 0.0;
  double step_ms_ = 0.0;
  PrefixCache cache_;              // the prefix cache's index (M7)
  SchedulerEngine::PrefixInfo prefix_info_;
  uint64_t ticks_ = 0;             // the LRU clock
};

// Streaming lifecycle events for the service (SSE). Fired inline on the
// ticking thread — see the threading note above.
class SchedulerObserver {
 public:
  virtual ~SchedulerObserver() = default;
  // The observer's running fold of everything it has recorded (the
  // fabric's continuous drift check, M9): has_digest() says it keeps one,
  // digest() is its value now. The default observer keeps none.
  virtual bool has_digest() const { return false; }
  virtual uint64_t digest() const { return 0; }
  // One generated token (the prefill pick is steps_done == 1). Always
  // fires BEFORE the matching retire when the token ends the request.
  virtual void on_token(const std::string& id, int64_t token,
                        int steps_done) = 0;
  // The token's logprobs, right after its on_token, for requests that
  // asked (SchedulerRequest::logprobs >= 0). Default: ignored.
  virtual void on_token_logprobs(const std::string& id, int steps_done,
                                 const glm_sample::Result& logprobs) {
    (void)id;
    (void)steps_done;
    (void)logprobs;
  }
  // The request's terminal state, exactly once (EOS, steps cap,
  // scripted or external cancellation, pool exhaustion all land here).
  virtual void on_retire(const std::string& id,
                         const Scheduler::Result& result) = 0;
  // Grow-on-demand grew the request's reservation to `reserved_tokens`
  // (the op stream records it: a growth decision is rank-identical state).
  virtual void on_grow(const std::string& id, int64_t reserved_tokens) {
    (void)id;
    (void)reserved_tokens;
  }
  // A prefix-cache decision (M7): `op` is "attach" (the request opened from
  // an entry at `position`, arena `slot`), "snapshot" (an entry taken at a
  // prefill cut), "rolling" (a live request's rolling snapshot), "close"
  // (a rolling snapshot became an entry at retire), "evict" (an entry
  // freed; `id` is the request whose admission needed the slot or blocks)
  // or "drop" (a rolling slot released without becoming an entry). Rank-
  // identical state: the op stream records every one.
  virtual void on_prefix(const std::string& id, const char* op,
                         int64_t position, int slot) {
    (void)id;
    (void)op;
    (void)position;
    (void)slot;
  }
};

}  // namespace dgpp::glm
