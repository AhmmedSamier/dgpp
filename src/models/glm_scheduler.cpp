#include "models/glm_scheduler.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/log.hpp"

namespace dgpp::glm {

std::vector<std::vector<int32_t>> SchedulerEngine::step_batch(
    const std::vector<int>& reqs) {
  if (reqs.empty())
    throw std::invalid_argument("SchedulerEngine: empty decode batch");
  if (reqs.size() > static_cast<size_t>(decode_batch_capacity()))
    throw std::invalid_argument(
        "SchedulerEngine: decode batch exceeds advertised capacity");
  std::vector<std::vector<int32_t>> out;
  out.reserve(reqs.size());
  for (const int req : reqs) out.push_back(step(req));
  return out;
}

namespace {

// A request's lifetime token footprint: the prompt plus every token it
// can generate. The last generated token is never fed back, so a one-row
// decode writes no position past prompt + max_steps - 2 and this count
// carries one token of slack; a two-row speculative verify (the MTP graph
// engine, which steps only while fewer than max_steps tokens stand) writes
// its second row at prompt + max_steps - 1 at the latest, so the slack is
// exactly its second row. This count is also what SchedulerEngine::reserve
// pins in the engine, so it must not shrink.
int64_t reserve_tokens(const SchedulerRequest& r) {
  return static_cast<int64_t>(r.prompt.size()) + r.max_steps;
}

const char* reason_name(Scheduler::Result::Reason r) {
  switch (r) {
    case Scheduler::Result::Reason::kEos: return "eos";
    case Scheduler::Result::Reason::kSteps: return "steps";
    case Scheduler::Result::Reason::kCancelled: return "cancelled";
    case Scheduler::Result::Reason::kPoolExhausted: return "pool exhausted";
    default: return "none";
  }
}

}  // namespace

Scheduler::Scheduler(SchedulerEngine* engine,
                     std::vector<int64_t> eos_token_ids, int queue_limit,
                     AdmissionPolicy policy)
    : engine_(engine),
      eos_ids_(std::move(eos_token_ids)),
      queue_limit_(queue_limit),
      policy_(policy) {
  if (engine_ == nullptr)
    throw std::invalid_argument("Scheduler: engine must not be null");
  if (policy_.window_tokens < 1)
    throw std::invalid_argument(
        "Scheduler: the admission window must be at least one token");
  if (queue_limit_ < 0)
    throw std::invalid_argument(
        "Scheduler: queue_limit must be 0 (unbounded) or positive");
  slots_.assign(static_cast<size_t>(engine_->max_concurrent_requests()), -1);
  if (slots_.empty())
    throw std::invalid_argument(
        "Scheduler: engine reports zero concurrent request slots");
  decode_batch_capacity_ = engine_->decode_batch_capacity();
  if (decode_batch_capacity_ < 1 ||
      decode_batch_capacity_ > static_cast<int>(slots_.size()))
    throw std::invalid_argument(
        "Scheduler: engine decode batch capacity must be in [1, " +
        std::to_string(slots_.size()) + "]");
}

bool Scheduler::is_eos(int32_t token) const {
  // Three ids at real dims — a linear scan beats building a set.
  return std::find(eos_ids_.begin(), eos_ids_.end(),
                   static_cast<int64_t>(token)) != eos_ids_.end();
}

int64_t Scheduler::initial_reserve_tokens(const SchedulerRequest& spec) const {
  const int64_t full = reserve_tokens(spec);
  if (policy_.mode != AdmissionPolicy::Mode::kGrowOnDemand) return full;
  // The window, never less than one step's width plus the pending token
  // (a step writes at most max_tokens_per_step rows past the KV's end).
  const int64_t headroom =
      std::max<int64_t>(policy_.window_tokens, engine_->max_tokens_per_step() + 1);
  return std::min<int64_t>(full, static_cast<int64_t>(spec.prompt.size()) + headroom);
}

int64_t Scheduler::reserve_blocks(const Request& r) const {
  return engine_->blocks_for_tokens(initial_reserve_tokens(r.spec));
}

int Scheduler::youngest_active_after(int arrival) const {
  for (int i = static_cast<int>(requests_.size()) - 1; i > arrival; --i)
    if (requests_[static_cast<size_t>(i)].state == State::kActive) return i;
  return -1;
}

void Scheduler::grow_reservations() {
  if (policy_.mode != AdmissionPolicy::Mode::kGrowOnDemand) return;
  const int64_t width = engine_->max_tokens_per_step();
  for (size_t i = 0; i < requests_.size(); ++i) {  // arrival order = priority
    Request& r = requests_[i];
    if (r.state != State::kActive) continue;
    const int64_t prompt = static_cast<int64_t>(r.spec.prompt.size());
    const int64_t full = prompt + r.spec.max_steps;
    // The KV holds prompt + steps_done - 1 tokens (the last pick is pending);
    // the next step writes up to `width` more. One token of slack.
    const int64_t need = std::min<int64_t>(full, prompt + r.steps_done + width);
    if (need <= r.reserved_tokens) continue;
    int64_t target = std::min<int64_t>(
        full, std::max<int64_t>(need, r.reserved_tokens + policy_.window_tokens));
    const int64_t held = engine_->blocks_for_tokens(r.reserved_tokens);
    for (;;) {
      const int64_t free_blocks =
          engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
      if (engine_->blocks_for_tokens(target) - held <= free_blocks) break;
      if (engine_->blocks_for_tokens(need) - held <= free_blocks) {
        target = need;  // the minimum, rather than shedding for a full window
        break;
      }
      // Even the minimum does not fit: the youngest active request goes —
      // this one when nothing younger is live.
      const int victim = youngest_active_after(static_cast<int>(i));
      const int shed = victim >= 0 ? victim : static_cast<int>(i);
      Request& v = requests_[static_cast<size_t>(shed)];
      DGPP_LOG_WARN(
          "sched: pool exhausted — request '{}' shed after {} tokens so "
          "request '{}' can write its next step ({} blocks free of {}; "
          "finish_reason length)",
          v.spec.id, v.steps_done, r.spec.id, free_blocks,
          engine_->pool_blocks_total());
      ++pool_sheds_;
      retire(shed, Result::Status::kDone, Result::Reason::kPoolExhausted);
      if (shed == static_cast<int>(i)) break;
    }
    if (r.state != State::kActive) continue;
    engine_->reserve(r.slot, target);
    r.reserved_tokens = target;
    ++grows_;
    if (observer_) observer_->on_grow(r.spec.id, target);
    DGPP_LOG_INFO(
        "sched: request '{}' reservation grown to {} tokens ({} blocks; pool "
        "{}/{} blocks in use)",
        r.spec.id, target, engine_->blocks_for_tokens(target),
        engine_->pool_blocks_in_use(), engine_->pool_blocks_total());
  }
}

int Scheduler::free_slot() const {
  for (int s = 0; s < static_cast<int>(slots_.size()); ++s)
    if (slots_[static_cast<size_t>(s)] < 0) return s;
  return -1;
}

int Scheduler::next_admissible() const {
  const int64_t free_blocks =
      engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i].state != State::kQueued) continue;
    // No head-of-line blocking: the OLDEST request that FITS admits. A
    // large deferred request must not dam the queue behind it — the
    // starvation it could suffer under an unbounded small-request stream
    // is a Stage 4 bounded-queue problem, not a policy bug.
    if (free_slot() < 0) return -1;
    if (reserve_blocks(requests_[i]) > free_blocks) continue;
    return static_cast<int>(i);
  }
  return -1;
}

void Scheduler::validate_new(const SchedulerRequest& request) const {
  if (request.id.empty())
    throw std::invalid_argument("Scheduler: request id must not be empty");
  for (const Request& r : requests_) {
    if (r.spec.id == request.id)
      throw std::invalid_argument("Scheduler: duplicate request id '" +
                                  request.id + "'");
  }
  if (request.prompt.empty())
    throw std::invalid_argument("Scheduler: request '" + request.id +
                               "' has an empty prompt");
  if (request.max_steps < 1)
    throw std::invalid_argument("Scheduler: request '" + request.id +
                                "' must generate at least one token");
  try {
    glm_sample::validate_params(request.sampling);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument("Scheduler: request '" + request.id +
                                "' has an invalid sampling spec: " + e.what());
  }
  if (request.sampling.temperature > 0.0f && !engine_->supports_sampling())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for stochastic sampling but the engine is greedy-only");
  if (request.logprobs >= 0 && !engine_->supports_logprobs())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for logprobs but the engine reports none");
  if (request.logprobs >= 0 && request.sampling.logprobs != request.logprobs)
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' logprobs and sampling.logprobs disagree");
  if (request.grammar.active() && !engine_->supports_constraints())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for constrained decoding but the engine cannot mask the "
        "pick");
  if (request.cancel_after < 0 || request.cancel_after > request.max_steps)
    throw std::invalid_argument(
        "Scheduler: request '" + request.id + "' cancel_after must be in "
        "[0, max_steps] — a cancel that can never fire is a manifest error");
}

int Scheduler::queued_count() const {
  int n = 0;
  for (const Request& r : requests_)
    if (r.state == State::kQueued) ++n;
  return n;
}

bool Scheduler::try_submit(SchedulerRequest request) {
  validate_new(request);
  if (queue_limit_ > 0 && queued_count() >= queue_limit_) return false;
  Request r;
  r.spec = std::move(request);
  requests_.push_back(std::move(r));
  results_.emplace_back();
  return true;
}

void Scheduler::submit(SchedulerRequest request) {
  if (!try_submit(std::move(request)))
    throw QueueFullError(
        "Scheduler: admission queue full (" +
        std::to_string(queued_count()) + " queued, limit " +
        std::to_string(queue_limit_) + ") — shed load or raise the limit");
}

bool Scheduler::cancel(const std::string& id) {
  for (Request& r : requests_) {
    if (r.spec.id != id) continue;
    if (r.state == State::kQueued || r.state == State::kActive) {
      r.cancel_requested = true;
      return true;
    }
    return false;  // terminal: a late cancel is a no-op, never an error
  }
  return false;
}

bool Scheduler::has_pending() const {
  for (const Request& r : requests_)
    if (r.state == State::kQueued || r.state == State::kActive) return true;
  return false;
}

const Scheduler::Result* Scheduler::find(const std::string& id) const {
  for (size_t i = 0; i < requests_.size(); ++i)
    if (requests_[i].spec.id == id) return &results_[i];
  return nullptr;
}

void Scheduler::admit(int arrival) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int slot = free_slot();
  if (slot < 0)
    throw std::logic_error("Scheduler: admit without a free slot");
  const int64_t reserve = reserve_blocks(r);
  // The spec lands on the slot before its first pick (the prefill's); the
  // grammar with it, so the prefill pick is the first constrained position.
  engine_->configure_sampling(slot, r.spec.sampling, r.spec.seed);
  engine_->configure_logprobs(slot, r.spec.logprobs);
  engine_->configure_constraint(slot, r.spec.grammar);
  const int32_t token = engine_->prefill(slot, r.spec.prompt);
  if (token < 0) {
    engine_->close(slot);
    throw std::runtime_error("Scheduler: engine prefill returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  }
  // Capture/device-position decode may not grow the DSA table during a
  // replay. Admission has already proved this reservation fits, and no
  // other scheduler mutation can interleave between that proof and here.
  const int64_t reserved = initial_reserve_tokens(r.spec);
  try {
    engine_->reserve(slot, reserved);
  } catch (...) {
    engine_->close(slot);
    throw;
  }
  r.reserved_tokens = reserved;
  r.state = State::kActive;
  r.slot = slot;
  slots_[static_cast<size_t>(slot)] = arrival;
  if (arrival == deferred_logged_) deferred_logged_ = -1;
  DGPP_LOG_INFO(
      "sched: request '{}' admitted to slot {} (reserve {} blocks; pool "
      "{}/{} blocks in use) — first token {}",
      r.spec.id, slot, reserve, engine_->pool_blocks_in_use(),
      engine_->pool_blocks_total(), token);
  const std::vector<glm_sample::Result> lps = collect_logprobs(arrival, slot, 1);
  (void)append_token(arrival, token, lps.empty() ? nullptr : &lps[0]);
}

std::vector<glm_sample::Result> Scheduler::collect_logprobs(int arrival,
                                                            int slot,
                                                            size_t tokens) {
  const Request& r = requests_[static_cast<size_t>(arrival)];
  if (r.spec.logprobs < 0) return {};
  std::vector<glm_sample::Result> lps = engine_->take_logprobs(slot);
  if (lps.size() != tokens)
    throw std::runtime_error(
        "Scheduler: engine reported " + std::to_string(lps.size()) +
        " logprob entries for " + std::to_string(tokens) +
        " tokens of request '" + r.spec.id + "'");
  return lps;
}

void Scheduler::step_batch(const std::vector<int>& arrivals) {
  if (arrivals.empty())
    throw std::logic_error("Scheduler: empty decode batch");
  std::vector<int> slots;
  slots.reserve(arrivals.size());
  for (const int arrival : arrivals) {
    const Request& r = requests_[static_cast<size_t>(arrival)];
    if (r.state != State::kActive || r.slot < 0)
      throw std::logic_error("Scheduler: decode batch contains an inactive "
                             "request");
    slots.push_back(r.slot);
  }

  // Validate the outer shape before publishing any token. A malformed
  // engine result must not leave half a physical pass visible to clients.
  const std::vector<std::vector<int32_t>> batches =
      engine_->step_batch(slots);
  if (batches.size() != arrivals.size())
    throw std::runtime_error(
        "Scheduler: engine returned " + std::to_string(batches.size()) +
        " request results for a decode batch of " +
        std::to_string(arrivals.size()));
  for (size_t i = 0; i < arrivals.size(); ++i) {
    const Request& r = requests_[static_cast<size_t>(arrivals[i])];
    if (batches[i].empty())
      throw std::runtime_error("Scheduler: engine step returned no tokens for "
                               "request '" + r.spec.id + "'");
    for (const int32_t token : batches[i])
      if (token < 0)
        throw std::runtime_error("Scheduler: engine returned token " +
                                 std::to_string(token) + " for request '" +
                                 r.spec.id + "'");
  }

  cursor_ = arrivals.back();
  for (size_t i = 0; i < arrivals.size(); ++i) {
    const int arrival = arrivals[i];
    const std::vector<int32_t>& tokens = batches[i];
    const std::vector<glm_sample::Result> lps =
        collect_logprobs(arrival, slots[i], tokens.size());
    for (size_t t = 0; t < tokens.size(); ++t) {
      // A speculative pass can have advanced farther than the public request
      // survives. EOS/cap/cancel retires the slot and deliberately drops the
      // rest of this request's batch; its extra device state is never seen.
      // Other requests in the same physical pass remain independent and are
      // still published below.
      if (append_token(arrival, tokens[t], lps.empty() ? nullptr : &lps[t]))
        break;
    }
  }
}

bool Scheduler::append_token(int arrival, int32_t token,
                             const glm_sample::Result* logprobs) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  if (token < 0)
    throw std::runtime_error("Scheduler: engine returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  ++r.steps_done;
  r.generated.push_back(static_cast<int64_t>(token));
  ++tokens_generated_;
  if (observer_) {
    observer_->on_token(r.spec.id, token, r.steps_done);
    if (logprobs != nullptr)
      observer_->on_token_logprobs(r.spec.id, r.steps_done, *logprobs);
  }
  DGPP_LOG_INFO("sched: request '{}' step {}: token {}", r.spec.id,
                r.steps_done, token);
  if (is_eos(token)) {
    retire(arrival, Result::Status::kDone, Result::Reason::kEos);
  } else if (r.spec.cancel_after > 0 &&
             r.steps_done >= r.spec.cancel_after) {
    retire(arrival, Result::Status::kCancelled, Result::Reason::kCancelled);
  } else if (r.steps_done >= r.spec.max_steps) {
    retire(arrival, Result::Status::kDone, Result::Reason::kSteps);
  }
  return r.state == State::kTerminal;
}

void Scheduler::retire(int arrival, Result::Status status,
                       Result::Reason reason) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  // An externally cancelled QUEUED request never held a slot or blocks.
  if (r.slot >= 0) {
    engine_->close(r.slot);
    slots_[static_cast<size_t>(r.slot)] = -1;
  }
  r.state = State::kTerminal;
  Result& res = results_[static_cast<size_t>(arrival)];
  res.status = status;
  res.reason = reason;
  res.slot = r.slot;
  res.steps_done = r.steps_done;
  res.generated = r.generated;
  r.slot = -1;
  if (observer_) observer_->on_retire(r.spec.id, res);
  DGPP_LOG_INFO("sched: request '{}' retired ({}, {} tokens generated)",
                r.spec.id, reason_name(reason), r.steps_done);
}

bool Scheduler::tick() {
  // The external-cancel sweep — FIXED POSITION, before any admission
  // or step: a request cancelled BETWEEN ticks never pays another
  // engine op; one cancelled MID-TICK waits out the in-flight op (the
  // price of a single fixed position the fabric journal can stamp —
  // every rank retires the same request at the same quantum, and
  // arrival order keeps it deterministic). Observed live at w1: a flag
  // that arrived during a 5-minute prefill rode out the whole tick.
  for (size_t i = 0; i < requests_.size(); ++i) {
    Request& r = requests_[i];
    if (r.cancel_requested &&
        (r.state == State::kQueued || r.state == State::kActive))
      retire(static_cast<int>(i), Result::Status::kCancelled,
             Result::Reason::kCancelled);
  }
  // Grow-on-demand's fixed position: after the sweep, before any
  // admission or step — the step below never writes past a reservation,
  // and every rank grows or sheds the same requests at the same quantum.
  grow_reservations();

  const bool any_active = std::any_of(
      requests_.begin(), requests_.end(),
      [](const Request& r) { return r.state == State::kActive; });
  const bool any_queued = std::any_of(
      requests_.begin(), requests_.end(),
      [](const Request& r) { return r.state == State::kQueued; });
  if (!any_active && !any_queued) return false;

  bool progressed = false;

  // (1) Strict alternation: at most ONE admission per tick, before the
  // step, so a queued request's first token is not delayed behind a
  // step — and mid-answer requests never wait behind more than one
  // read-in.
  const int admit_arrival = next_admissible();
  if (admit_arrival >= 0) {
    admit(admit_arrival);
    progressed = true;
  } else if (any_queued) {
    // Deferral bookkeeping: log the head of the queue once per
    // deferral episode, with the numbers an operator needs.
    const auto head = std::find_if(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kQueued; });
    const int head_arrival =
        static_cast<int>(head - requests_.begin());
    if (deferred_logged_ != head_arrival) {
      deferred_logged_ = head_arrival;
      const int64_t free_blocks = engine_->pool_blocks_total() -
                                  engine_->pool_blocks_in_use();
      DGPP_LOG_INFO(
          "sched: request '{}' deferred (needs {} blocks, {} free, {} "
          "slot(s) open) — admits when a peer retires",
          head->spec.id, reserve_blocks(*head), free_blocks,
          std::count(slots_.begin(), slots_.end(), -1));
    }
  }

  // (2) One physical decode pass per tick, over the next round-robin slice.
  // Scalar engines advertise capacity one and retain the original policy
  // bit-for-bit. A batched graph normally advertises the slot count, so this
  // slice contains every active request exactly once.
  std::vector<int> step_arrivals;
  step_arrivals.reserve(static_cast<size_t>(decode_batch_capacity_));
  for (int off = 1;
       off <= static_cast<int>(requests_.size()) &&
       static_cast<int>(step_arrivals.size()) < decode_batch_capacity_;
       ++off) {
    const int i = (cursor_ + off) % static_cast<int>(requests_.size());
    if (requests_[static_cast<size_t>(i)].state == State::kActive)
      step_arrivals.push_back(i);
  }
  if (!step_arrivals.empty()) {
    step_batch(step_arrivals);
    progressed = true;
  }

  if (!progressed) {
    // No admission, no step, work remaining. With any active request
    // the rotation always yields one, so this is the admission
    // deadlock: the queue cannot fit the pool ever (its head's
    // reservation exceeds the total capacity) or every slot is held
    // by requests that can never retire (impossible under full-reserve
    // — they are bounded by max_steps). Either way: LOUD.
    const auto head = std::find_if(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kQueued; });
    throw std::runtime_error(
        "Scheduler: admission deadlock — request '" + head->spec.id +
        "' needs " + std::to_string(reserve_blocks(*head)) +
        " blocks against a pool of " +
        std::to_string(engine_->pool_blocks_total()) +
        " (grow --kv-capacity or shed requests)");
  }
  return true;
}

void Scheduler::run_to_completion() {
  while (tick()) {
  }
}

Scheduler::Meters Scheduler::meters() const {
  Meters m;
  for (const Request& r : requests_) {
    if (r.state == State::kActive) ++m.active;
    else if (r.state == State::kQueued) ++m.queued;
    else ++m.terminal;
  }
  m.pool_blocks_total = engine_->pool_blocks_total();
  m.pool_blocks_in_use = engine_->pool_blocks_in_use();
  m.tokens_generated = tokens_generated_;
  m.reservations_grown = grows_;
  m.requests_shed_pool = pool_sheds_;
  return m;
}

}  // namespace dgpp::glm
