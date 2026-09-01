#include "models/glm_scheduler.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/log.hpp"

namespace dgpp::glm {

namespace {

// A request's lifetime token footprint: the prompt plus every token it
// can generate. One token of deliberate slack (the single-request path's
// sizing convention) — the block rounding makes it matter only at exact
// block boundaries, and over-reserving is the safe direction.
int64_t reserve_tokens(const SchedulerRequest& r) {
  return static_cast<int64_t>(r.prompt.size()) + r.max_steps;
}

const char* reason_name(Scheduler::Result::Reason r) {
  switch (r) {
    case Scheduler::Result::Reason::kEos: return "eos";
    case Scheduler::Result::Reason::kSteps: return "steps";
    case Scheduler::Result::Reason::kCancelled: return "cancelled";
    default: return "none";
  }
}

}  // namespace

Scheduler::Scheduler(SchedulerEngine* engine,
                     std::vector<int64_t> eos_token_ids)
    : engine_(engine), eos_ids_(std::move(eos_token_ids)) {
  if (engine_ == nullptr)
    throw std::invalid_argument("Scheduler: engine must not be null");
  slots_.assign(static_cast<size_t>(engine_->max_concurrent_requests()), -1);
  if (slots_.empty())
    throw std::invalid_argument(
        "Scheduler: engine reports zero concurrent request slots");
}

bool Scheduler::is_eos(int32_t token) const {
  // Three ids at real dims — a linear scan beats building a set.
  return std::find(eos_ids_.begin(), eos_ids_.end(),
                   static_cast<int64_t>(token)) != eos_ids_.end();
}

int64_t Scheduler::reserve_blocks(const Request& r) const {
  return engine_->blocks_for_tokens(reserve_tokens(r.spec));
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

void Scheduler::submit(SchedulerRequest request) {
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
  if (request.cancel_after < 0 || request.cancel_after > request.max_steps)
    throw std::invalid_argument(
        "Scheduler: request '" + request.id + "' cancel_after must be in "
        "[0, max_steps] — a cancel that can never fire is a manifest error");
  Request r;
  r.spec = std::move(request);
  requests_.push_back(std::move(r));
  Result res;
  results_.push_back(res);
}

void Scheduler::admit(int arrival) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int slot = free_slot();
  if (slot < 0)
    throw std::logic_error("Scheduler: admit without a free slot");
  const int64_t reserve = reserve_blocks(r);
  const int32_t token = engine_->prefill(slot, r.spec.prompt);
  if (token < 0)
    throw std::runtime_error("Scheduler: engine prefill returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  r.state = State::kActive;
  r.slot = slot;
  r.steps_done = 1;
  r.generated.push_back(static_cast<int64_t>(token));
  slots_[static_cast<size_t>(slot)] = arrival;
  if (arrival == deferred_logged_) deferred_logged_ = -1;
  DGPP_LOG_INFO(
      "sched: request '{}' admitted to slot {} (reserve {} blocks; pool "
      "{}/{} blocks in use) — first token {}",
      r.spec.id, slot, reserve, engine_->pool_blocks_in_use(),
      engine_->pool_blocks_total(), token);
  if (is_eos(token)) {
    // EOS on the prefill pick: the model answered in one token.
    retire(arrival, Result::Status::kDone, Result::Reason::kEos);
  }
}

void Scheduler::step_one(int arrival) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int32_t token = engine_->step(r.slot, r.generated.back());
  if (token < 0)
    throw std::runtime_error("Scheduler: engine step returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  r.steps_done += 1;
  r.generated.push_back(static_cast<int64_t>(token));
  cursor_ = arrival;
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
}

void Scheduler::retire(int arrival, Result::Status status,
                       Result::Reason reason) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  engine_->close(r.slot);
  slots_[static_cast<size_t>(r.slot)] = -1;
  r.state = State::kTerminal;
  Result& res = results_[static_cast<size_t>(arrival)];
  res.status = status;
  res.reason = reason;
  res.slot = r.slot;
  res.steps_done = r.steps_done;
  res.generated = r.generated;
  r.slot = -1;
  DGPP_LOG_INFO("sched: request '{}' retired ({}, {} tokens generated)",
               r.spec.id, reason_name(reason), r.steps_done);
}

void Scheduler::run_to_completion() {
  if (requests_.empty()) return;
  while (true) {
    const bool any_active = std::any_of(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kActive; });
    const bool any_queued = std::any_of(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kQueued; });
    if (!any_active && !any_queued) break;

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

    // (2) Exactly one decode step per tick, round-robin by arrival over
    // the active set. The just-admitted request is eligible only if the
    // rotation reaches it — admission does not jump the queue.
    const auto next_active_after_cursor = [&]() -> int {
      for (int off = 1; off <= static_cast<int>(requests_.size()); ++off) {
        const int i = (cursor_ + off) % static_cast<int>(requests_.size());
        if (requests_[static_cast<size_t>(i)].state == State::kActive)
          return i;
      }
      return -1;
    };
    const int step_arrival = next_active_after_cursor();
    if (step_arrival >= 0) {
      step_one(step_arrival);
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
  }
}

}  // namespace dgpp::glm
