// M6 Stage 2b: the scheduler policy gate. A recording FakeEngine stands in
// for GlmDiagnosticModel + the pick, so every policy decision is pinned
// WITHOUT a GPU, a model cache, or the bus — this gate always runs.
//
// What is pinned here (the op stream is the contract):
//   * strict alternation — at most one admission, exactly one decode step
//     per tick, admission first;
//   * FCFS admission without head-of-line blocking, budget-deferred
//     until a peer retires, and the loud admission deadlock;
//   * round-robin decode by arrival order, slot reuse (lowest free);
//   * EOS on the prefill pick and mid-run; the steps cap;
//   * scripted cancellation, and THE ISOLATION PROPERTY: a request's
//     generated ids are identical whether it runs alone or interleaved
//     with (and cancelled around) other requests;
//   * determinism — the same scenario produces the identical op stream.
//
// The fake also enforces the engine seam: scripts are consumed in order,
// each slot owns its pending transcript, and close() must find a live
// reservation to release.
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.hpp"

#include "common/test.hpp"
#include "models/glm_scheduler.hpp"

namespace {

using dgpp::glm::Scheduler;
using dgpp::glm::SchedulerEngine;
using dgpp::glm::SchedulerRequest;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The recording engine: scripted token episodes per slot (episodes queue,
// so a retired slot's next admission consumes the next episode), mirrored
// block reservations, and the op stream the tests pin.
class FakeEngine : public SchedulerEngine {
 public:
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens,
             int batch_capacity = 1, bool can_sample = false)
      : slots_(slots),
        total_blocks_(total_blocks),
        block_tokens_(block_tokens),
        batch_capacity_(batch_capacity),
        can_sample_(can_sample) {}

  // The sampling seam (M6 6b): a sampling-capable fake records the arming
  // in the op stream ("A:slot:seed"), so its position relative to the
  // prefill is pinned; a greedy fake inherits the base refusal.
  bool supports_sampling() const override { return can_sample_; }
  void configure_sampling(int req, const dgpp::glm_sample::Params& p,
                          uint64_t seed) override {
    if (!can_sample_) {
      SchedulerEngine::configure_sampling(req, p, seed);
      return;
    }
    if (p.temperature > 0.0f)
      ops_.push_back("A:" + std::to_string(req) + ":" + std::to_string(seed));
  }
  // Logprobs: a sampling-capable fake reports one entry per token it
  // returns (logprob = -token, one alternative), only for armed slots.
  bool supports_logprobs() const override { return can_sample_; }
  void configure_logprobs(int req, int logprobs) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logprobs(req, logprobs);
      return;
    }
    report_[req] = logprobs;
    if (logprobs >= 0) ops_.push_back("L:" + std::to_string(req));
  }
  std::vector<dgpp::glm_sample::Result> take_logprobs(int req) override {
    std::vector<dgpp::glm_sample::Result> out;
    out.swap(pending_lps_[req]);
    return out;
  }
  // Constrained decoding (M6 6g): a sampling-capable fake can mask and
  // records the grammar arming ("G:slot:mode:tools") right after the
  // sampling arming; a greedy fake inherits the base refusal.
  bool supports_constraints() const override { return can_sample_; }
  void configure_constraint(int req,
                            const dgpp::glm::GrammarSpec& g) override {
    if (!can_sample_) {
      SchedulerEngine::configure_constraint(req, g);
      return;
    }
    if (g.active())
      ops_.push_back("G:" + std::to_string(req) + ":" +
                     std::to_string(static_cast<int>(g.mode)) + ":" +
                     std::to_string(g.tools.size()));
  }

  // Arms `slot`'s NEXT scalar episode: prefill returns tokens[0], then each
  // step returns one following token.
  void arm(int slot, std::vector<int32_t> tokens, int max_steps) {
    require(!tokens.empty(), "fake: cannot arm an empty token script");
    Episode e;
    e.first = tokens.front();
    e.max_steps = max_steps;
    for (size_t i = 1; i < tokens.size(); ++i)
      e.steps.push_back({tokens[i]});
    episodes_[slot].push_back(std::move(e));
  }

  // Arms the new speculative seam directly: prefill returns `first`, and
  // each subsequent engine step returns one scripted batch.
  void arm_batches(int slot, int32_t first,
                   std::vector<std::vector<int32_t>> steps, int max_steps) {
    episodes_[slot].push_back(Episode{first, std::move(steps), max_steps});
  }

  const std::vector<std::string>& ops() const { return ops_; }
  const std::vector<std::vector<int>>& batch_calls() const {
    return batch_calls_;
  }
  std::string op_stream() const {
    std::string s;
    for (const std::string& op : ops_) {
      if (!s.empty()) s += " ";
      s += op;
    }
    return s;
  }

  // The prefix cache seam (M7): an arena of `slots` snapshot slots with
  // pool alignment `align`. The fake records every op — "X:slot:pos@arena"
  // an attach, "N:slot:pos@arena" a snapshot taken at a prefill cut,
  // "RS:slot:pos@arena" a rolling snapshot, "F:arena" a release — and pins
  // the blocks an entry holds (position / block_tokens rounded up) so the
  // pool meters see them exactly as the real pool does.
  void set_prefix_arena(int slots, int64_t align, int64_t chunk_tokens = 2048) {
    arena_slots_ = slots;
    arena_align_ = align;
    arena_chunk_ = chunk_tokens;
  }
  dgpp::glm::SchedulerEngine::PrefixInfo prefix_info() const override {
    dgpp::glm::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_slots_;
    info.align = arena_align_;
    info.block_tokens = block_tokens_;
    info.chunk_tokens = arena_chunk_;
    info.step_tokens_max = step_tokens_max_;
    return info;
  }
  // The two-token step's hop (M7): the fake commits whatever its script
  // says per step; an armed hop is taken ("H:req:pos@slot") when the step
  // returns two tokens, at the position the scheduler named, which must be
  // the committed count + 1.
  void set_step_tokens_max(int n) { step_tokens_max_ = n; }
  void prefix_arm_hop(int req, int slot, int64_t position) override {
    require(live_.count(req) != 0, "fake: hop armed on an unopened slot");
    hop_armed_[req] = {slot, position};
  }
  int step_tokens_max_ = 1;
  std::map<int, std::pair<int, int64_t>> hop_armed_;
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         dgpp::glm::SchedulerEngine::PrefixPrefill* plan) override {
    require(plan != nullptr && plan->boundaries != nullptr, "fake: plan");
    if (plan->attach_slot >= 0) {
      require(pinned_.count(plan->attach_slot) != 0, "fake: attach from an empty arena slot");
      require(pinned_positions_[plan->attach_slot] == plan->attach_position,
              "fake: attach position differs from the arena slot's");
      ops_.push_back("X:" + std::to_string(req) + ":" +
                     std::to_string(plan->attach_position) + "@" +
                     std::to_string(plan->attach_slot));
    }
    const int32_t token = prefill(req, prompt);
    if (plan->snap_slot >= 0) {
      require(pinned_.count(plan->snap_slot) == 0, "fake: snapshot into an occupied arena slot");
      pin(plan->snap_slot, plan->snap_position);
      plan->snap_taken = true;
      ops_.push_back("N:" + std::to_string(req) + ":" +
                     std::to_string(plan->snap_position) + "@" +
                     std::to_string(plan->snap_slot));
    }
    return token;
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    require(live_.count(req) != 0, "fake: rolling snapshot of an unopened slot");
    const Live& live = live_.at(req);
    // The scheduler's view of the committed position must be the fake's:
    // prompt + tokens returned so far - 1 (the last is pending).
    require(position == live.prompt_tokens + static_cast<int64_t>(live.returned) - 1,
            "fake: rolling snapshot position " + std::to_string(position) +
                " differs from the slot's committed " +
                std::to_string(live.prompt_tokens + static_cast<int64_t>(live.returned) - 1));
    if (pinned_.count(slot) != 0) pinned_.erase(slot);
    pin(slot, position);
    ops_.push_back("RS:" + std::to_string(req) + ":" + std::to_string(position) +
                   "@" + std::to_string(slot));
  }
  void prefix_release(int slot) override {
    require(pinned_.count(slot) != 0, "fake: release of an empty arena slot " +
                                          std::to_string(slot));
    pinned_.erase(slot);
    pinned_positions_.erase(slot);
    ops_.push_back("F:" + std::to_string(slot));
  }
  int64_t pinned_blocks() const {
    int64_t n = 0;
    for (const auto& [slot, blocks] : pinned_) n += blocks;
    return n;
  }

  int max_concurrent_requests() const override { return slots_; }
  int decode_batch_capacity() const override { return batch_capacity_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) sum += live.held_blocks;
    return sum + pinned_blocks();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    std::vector<Episode>& queue = episodes_[req];
    require(!queue.empty(), "fake: prefill on slot " + std::to_string(req) +
                               " with no armed episode");
    Live live;
    live.episode = std::move(queue.front());
    queue.erase(queue.begin());
    live.prompt_tokens = static_cast<int64_t>(prompt.size());
    live.last_token = live.episode.first;
    live.returned = 1;
    live_[req] = live;
    ops_.push_back("P:" + std::to_string(req) + ":" +
                   std::to_string(prompt.size()));
    note_logprobs(req, {live.last_token});
    return live.last_token;
  }
  void note_logprobs(int req, const std::vector<int32_t>& tokens) {
    if (report_.count(req) == 0 || report_[req] < 0) return;
    for (const int32_t t : tokens) {
      dgpp::glm_sample::Result r;
      r.token = t;
      r.logprob = -static_cast<float>(t);
      r.top_logprobs.emplace_back(t, -static_cast<float>(t));
      pending_lps_[req].push_back(r);
    }
  }

  void reserve(int req, int64_t tokens) override {
    require(live_.count(req) != 0,
            "fake: reserve on unopened slot " + std::to_string(req));
    Live& live = live_[req];
    // Full-reserve pins the lifetime once; grow-on-demand (M6 6d) starts
    // inside it and grows monotonically, never past it.
    require(tokens >= live.prompt_tokens + 1 &&
                tokens <= live.prompt_tokens + live.episode.max_steps,
            "fake: reservation outside [prompt + 1, prompt + max_steps]");
    require(blocks_for_tokens(tokens) >= live.held_blocks,
            "fake: a reservation shrank");
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    require(live_.count(req) != 0,
            "fake: step on unopened slot " + std::to_string(req));
    Live& live = live_[req];
    require(live.next_step < live.episode.steps.size(),
            "fake: token script exhausted for slot " +
                std::to_string(req) + " (a step the policy should not "
                "have issued — script the test correctly)");
    std::vector<int32_t> out = live.episode.steps[live.next_step++];
    const int32_t prev_token = live.last_token;
    if (!out.empty()) live.last_token = out.back();
    if (hop_armed_.count(req) != 0) {
      const auto [slot, position] = hop_armed_[req];
      hop_armed_.erase(req);
      if (out.size() >= 2) {
        require(position == live.prompt_tokens + static_cast<int64_t>(live.returned),
                "fake: hop position " + std::to_string(position) +
                    " is not the committed count + 1 (" +
                    std::to_string(live.prompt_tokens + static_cast<int64_t>(live.returned)) + ")");
        if (pinned_.count(slot) != 0) pinned_.erase(slot);
        pin(slot, position);
        ops_.push_back("H:" + std::to_string(req) + ":" + std::to_string(position) +
                       "@" + std::to_string(slot));
      }
    }
    live.returned += out.size();
    ops_.push_back("S:" + std::to_string(req) + ":" +
                   std::to_string(prev_token));
    note_logprobs(req, out);
    return out;
  }

  std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs) override {
    batch_calls_.push_back(reqs);
    return SchedulerEngine::step_batch(reqs);
  }

  void close(int req) override {
    require(live_.count(req) != 0,
            "fake: close on unopened slot " + std::to_string(req));
    require(live_[req].held_blocks > 0,
            "fake: close with no reservation on slot " +
                std::to_string(req) + " (double close?)");
    live_.erase(req);
    ops_.push_back("C:" + std::to_string(req));
  }

 private:
  struct Episode {
    int32_t first = -1;
    std::vector<std::vector<int32_t>> steps;
    int max_steps = 0;
  };
  struct Live {
    Episode episode;
    int64_t held_blocks = 0;
    int64_t prompt_tokens = 0;
    size_t next_step = 0;
    int32_t last_token = -1;
    size_t returned = 0;  // tokens handed to the scheduler (the pending one included)
  };
  void pin(int slot, int64_t position) {
    pinned_[slot] = position / block_tokens_ + (position % block_tokens_ != 0 ? 1 : 0);
    pinned_positions_[slot] = position;
  }

  int slots_;
  int arena_slots_ = 0;
  int64_t arena_align_ = 1;
  int64_t arena_chunk_ = 2048;
  std::map<int, int64_t> pinned_;           // arena slot -> blocks pinned
  std::map<int, int64_t> pinned_positions_;  // arena slot -> position
  int64_t total_blocks_;
  int64_t block_tokens_;
  int batch_capacity_ = 1;
  bool can_sample_ = false;
  std::map<int, int> report_;
  std::map<int, std::vector<dgpp::glm_sample::Result>> pending_lps_;
  std::map<int, std::vector<Episode>> episodes_;
  std::map<int, Live> live_;
  std::vector<std::string> ops_;
  std::vector<std::vector<int>> batch_calls_;
};

SchedulerRequest make_request(const std::string& id, int prompt_len,
                              int max_steps, int cancel_after = 0) {
  SchedulerRequest r;
  r.id = id;
  r.prompt.assign(static_cast<size_t>(prompt_len), 7);
  r.max_steps = max_steps;
  r.cancel_after = cancel_after;
  return r;
}

// EOS id used by every script (token 999 in the streams).
constexpr int32_t kEos = 999;

std::string ids_joined(const std::vector<int64_t>& ids) {
  std::string s;
  for (int64_t t : ids) {
    if (!s.empty()) s += ",";
    s += std::to_string(t);
  }
  return s;
}

DGPP_TEST(scheduler_strictAlternation_pinnedOpSequence) {
  // GIVEN three 5-token requests (max 3 steps each), 2 slots, an ample
  // pool; slots arm in the deterministic assignment order (a->0, b->1,
  // c reuses 0 after a retires):
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);   // a
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);   // b
  engine.arm(0, {7, 8, 9}, /*max_steps=*/3);   // c (slot 0's 2nd episode)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));
  sched.submit(make_request("c", 5, 3));

  // WHEN the scheduler runs to completion,
  sched.run_to_completion();

  // THEN the op stream is exactly the pinned policy: one admission per
  // tick before the step, round-robin by arrival, retirement-triggered
  // admission of c, lowest-free-slot reuse, and close on every retire.
  const std::string expected =
      "P:0:5 S:0:1 P:1:5 S:1:4 S:0:2 C:0 P:0:5 S:1:5 C:1 S:0:7 S:0:8 C:0";
  require(engine.op_stream() == expected,
          "op stream drifted from the pinned policy:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(sched.results().size() == 3, "three results");
  for (const auto& res : sched.results()) {
    require(res.status == Scheduler::Result::Status::kDone,
            "all three retire Done (steps cap)");
    require(res.reason == Scheduler::Result::Reason::kSteps,
            "all three retire on the steps cap");
    require(res.steps_done == 3, "each generated 3 tokens");
  }
  require(ids_joined(sched.results()[0].generated) == "1,2,3", "a ids");
  require(ids_joined(sched.results()[1].generated) == "4,5,6", "b ids");
  require(ids_joined(sched.results()[2].generated) == "7,8,9", "c ids");
}

DGPP_TEST(scheduler_batchEngine_stepsRoundRobinSliceInOnePass) {
  // GIVEN a two-row engine and two requests. Strict admission still opens
  // at most one request per tick; once both are active, one physical pass
  // receives both slots in round-robin order.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/2);
  engine.arm(0, {10, 11, 12}, /*max_steps=*/3);  // a
  engine.arm(1, {20, 21, 22}, /*max_steps=*/3);  // b
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));

  sched.run_to_completion();

  // Tick 1 carries a. Tick 2 admits b, then rotates from a to [b,a] in one
  // call (a retires independently at its cap). Tick 3 carries b alone.
  const std::vector<std::vector<int>> want_batches = {{0}, {1, 0}, {1}};
  require(engine.batch_calls() == want_batches,
          "batch engine did not receive the canonical round-robin slices");
  require(ids_joined(sched.results()[0].generated) == "10,11,12",
          "batched a transcript");
  require(ids_joined(sched.results()[1].generated) == "20,21,22",
          "batched b transcript");
}

DGPP_TEST(scheduler_batchEngine_capacityBelowActive_rotatesFairly) {
  // GIVEN a three-slot engine whose physical pass carries only two rows.
  // Once three requests are active, each tick advances the next two in
  // arrival rotation and the cursor lands on the slice's last member, so
  // every request is stepped in two of every three ticks.
  FakeEngine engine(/*slots=*/3, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/2);
  engine.arm(0, {10, 11, 12, 13, 14}, /*max_steps=*/5);  // a
  engine.arm(1, {20, 21, 22, 23, 24}, /*max_steps=*/5);  // b
  engine.arm(2, {30, 31, 32, 33, 34}, /*max_steps=*/5);  // c
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 5));
  sched.submit(make_request("c", 5, 5));

  sched.run_to_completion();

  // Tick 1: admit a, step {a}. Tick 2: admit b, rotate from a: {b, a}.
  // Tick 3: admit c, rotate from a: {b, c}. Tick 4: from c: {a, b}.
  // Tick 5: from b: {c, a} — a reaches its cap here. Tick 6: {b, c}; b
  // retires. Tick 7: {c}; c retires.
  const std::vector<std::vector<int>> want_batches = {
      {0}, {1, 0}, {1, 2}, {0, 1}, {2, 0}, {1, 2}, {2}};
  require(engine.batch_calls() == want_batches,
          "capacity-2 slices did not rotate fairly over three requests");
  require(ids_joined(sched.results()[0].generated) == "10,11,12,13,14",
          "rotated a transcript");
  require(ids_joined(sched.results()[1].generated) == "20,21,22,23,24",
          "rotated b transcript");
  require(ids_joined(sched.results()[2].generated) == "30,31,32,33,34",
          "rotated c transcript");
}

DGPP_TEST(scheduler_poolBudget_defersAdmissionUntilPeerRetires) {
  // GIVEN two requests each reserving 2 blocks (5+3 tokens at 4
  // tokens/block) against a 2-block pool — only one fits at a time:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a
  engine.arm(0, {4, 5, 6}, /*max_steps=*/3);  // b (admits after a frees)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));

  // WHEN run to completion,
  sched.run_to_completion();

  // THEN b is deferred (no P until a's C), admitted on the very next
  // tick, and both finish normally — the budget, not the slot count,
  // gated b (2 slots were open the whole time).
  const std::string expected =
      "P:0:5 S:0:1 S:0:2 C:0 P:0:5 S:0:4 S:0:5 C:0";
  require(engine.op_stream() == expected,
          "budget-deferral stream drifted:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(sched.results()[0].status == Scheduler::Result::Status::kDone,
          "a Done");
  require(sched.results()[1].status == Scheduler::Result::Status::kDone,
          "b Done");
}

DGPP_TEST(scheduler_capacityBelowSmallestReservation_throwsDeadlock) {
  // GIVEN one request needing 2 blocks against a 1-block pool — it can
  // never fit and nothing else can retire to free space:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/1, /*block_tokens=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));

  // WHEN run, THEN the scheduler refuses loudly (an operator sizing
  // error — never a hang, never a silent partial run).
  bool threw = false;
  try {
    sched.run_to_completion();
  } catch (const std::runtime_error& e) {
    threw = true;
    require(std::string(e.what()).find("admission deadlock") !=
                std::string::npos,
            "the error must name the deadlock: " + std::string(e.what()));
  }
  require(threw, "run_to_completion must throw on an unfixable queue");
  require(engine.ops().empty(), "no ops may run before the deadlock");
}

DGPP_TEST(scheduler_eosOnPrefillPick_retiresImmediatelyThenPeerAdmits) {
  // GIVEN a whose FIRST picked token is EOS (the model answered in one
  // token) and b behind it:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {kEos}, /*max_steps=*/5);    // a: EOS on the prefill pick
  engine.arm(0, {7, 8}, /*max_steps=*/2);    // b: reuses the freed slot
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 2));

  // WHEN run,
  sched.run_to_completion();

  // THEN a retires with a single generated token (Done/eos — no step
  // op for it), b admits on the next tick, and b's steps cap ends it.
  const std::string expected = "P:0:5 C:0 P:0:5 S:0:7 C:0";
  require(engine.op_stream() == expected,
          "prefill-EOS stream drifted:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.status == Scheduler::Result::Status::kDone &&
             a.reason == Scheduler::Result::Reason::kEos &&
             a.steps_done == 1 && ids_joined(a.generated) == "999",
          "a: one token, Done/eos");
  const auto& b = sched.results()[1];
  require(b.status == Scheduler::Result::Status::kDone &&
             b.reason == Scheduler::Result::Reason::kSteps &&
             ids_joined(b.generated) == "7,8",
          "b: steps cap, both tokens");
}

DGPP_TEST(scheduler_cancellation_isolatedPeerTranscriptUnchanged) {
  // GIVEN the headline property: x runs twice — once solo, once
  // interleaved with y, which is cancelled after 2 of its 4 tokens.
  const auto run_pair = [&](bool with_y) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);  // x -> slot 0
    if (with_y) engine.arm(1, {5, 6, 7, 8}, /*max_steps=*/4);  // y
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    if (with_y) sched.submit(make_request("y", 5, 4, /*cancel_after=*/2));
    sched.run_to_completion();
    return std::make_pair(engine.op_stream(), sched.results()[0]);
  };

  // WHEN both runs complete,
  const auto [solo_stream, solo_x] = run_pair(false);
  const auto [pair_stream, pair_x] = run_pair(true);

  // THEN x's transcript is IDENTICAL alone vs interleaved-with-a-
  // cancelled-peer: same ids, same status, same step count. The peer's
  // cancellation released its slot mid-flight and touched nothing else.
  require(ids_joined(solo_x.generated) == "1,2,3,4", "x solo ids");
  require(ids_joined(pair_x.generated) == "1,2,3,4",
          "x interleaved ids must equal solo: " +
              ids_joined(pair_x.generated));
  require(solo_x.status == pair_x.status &&
             solo_x.reason == pair_x.reason &&
             solo_x.steps_done == pair_x.steps_done,
          "x's terminal state must be identical across runs");
  // And the pair run shows y admitted, stepped twice, cancelled, closed
  // — while x keeps decoding to its cap.
  const std::string expected =
      "P:0:5 S:0:1 P:1:5 S:1:5 C:1 S:0:2 S:0:3 C:0";
  require(pair_stream == expected,
          "pair stream drifted:\n  got:      " + pair_stream +
              "\n  expected: " + expected);
}

DGPP_TEST(scheduler_sameScenarioTwice_identicalOpStreams) {
  // GIVEN a scenario with deferral, cancellation, and EOS mixed,
  const auto run = [&]() {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/3, /*block_tokens=*/4);
    // Pool fits ONE reservation at a time; x EOSes on its 3rd token
    // (before its cap), freeing space for y early.
    engine.arm(0, {1, 2, kEos}, /*max_steps=*/4);  // x -> slot 0
    engine.arm(0, {7, 8}, /*max_steps=*/2);        // y -> reuses slot 0
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    sched.submit(make_request("y", 5, 2, /*cancel_after=*/1));
    sched.run_to_completion();
    std::vector<std::string> summary;
    for (const auto& r : sched.results())
      summary.push_back(std::to_string(static_cast<int>(r.status)) + ":" +
                        ids_joined(r.generated));
    return std::make_pair(engine.op_stream(), summary);
  };

  // WHEN run twice,
  const auto first = run();
  const auto second = run();

  // THEN the op stream AND every result are identical — the §11
  // identical-rank-order invariant in miniature (if two runs of the
  // same pure function ever disagree, the fabric would desync).
  require(first.first == second.first,
          "op streams must be identical across runs");
  require(first.second == second.second,
          "results must be identical across runs");
}

DGPP_TEST(scheduler_eosMidRun_retiresBeforeStepsCap) {
  // GIVEN a request whose 3rd token is EOS with a 5-step cap,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, kEos}, /*max_steps=*/5);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));

  // WHEN run,
  sched.run_to_completion();

  // THEN it retires Done/eos at 3 tokens — EOS dominates the cap.
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "mid-run EOS stream drifted:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.reason == Scheduler::Result::Reason::kEos && a.steps_done == 3,
          "Done/eos at 3 tokens");
}

DGPP_TEST(scheduler_oneTokenCap_retiresOnPrefillWithoutDecode) {
  // GIVEN a request whose whole output budget is the prefill pick,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {17}, /*max_steps=*/1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("one", 5, 1));

  // WHEN it runs, THEN no decode step is issued: the prefill pick is token
  // one by the SchedulerRequest contract and therefore meets the cap.
  sched.run_to_completion();
  require(engine.op_stream() == "P:0:5 C:0",
          "max_steps=1 must not issue an extra decode: " +
              engine.op_stream());
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kSteps &&
              out.steps_done == 1 && ids_joined(out.generated) == "17",
          "one-token cap must return exactly the prefill pick");
}

DGPP_TEST(scheduler_multiTokenStep_eosInMiddleDropsSuffix) {
  // GIVEN one speculative pass that returns a token, EOS, then a token the
  // device happened to decide after the public request had ended,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm_batches(0, /*first=*/1, {{2, kEos, 77}}, /*max_steps=*/8);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("spec", 5, 8));

  // WHEN it runs, THEN EOS is observable and the suffix is not.
  sched.run_to_completion();
  require(engine.op_stream() == "P:0:5 S:0:1 C:0",
          "one speculative pass then close");
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kEos &&
              out.steps_done == 3 && ids_joined(out.generated) == "1,2,999",
          "tokens after an in-batch EOS must be dropped");
}

DGPP_TEST(scheduler_multiTokenStep_capOvershootDropsSuffix) {
  // GIVEN two output slots left and a speculative pass returning three,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm_batches(0, /*first=*/1, {{2, 3, 4}}, /*max_steps=*/3);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("spec", 5, 3));

  // WHEN it runs, THEN exactly the remaining two land in the transcript.
  sched.run_to_completion();
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kSteps &&
              out.steps_done == 3 && ids_joined(out.generated) == "1,2,3",
          "a speculative batch must truncate exactly at max_steps");
}

using dgpp::glm::QueueFullError;
using dgpp::glm::SchedulerObserver;

// The recording observer: every event in global fire order (the SSE tap
// in miniature — ordering IS the contract being tested).
class RecordingObserver : public SchedulerObserver {
 public:
  struct Event {
    std::string kind;  // "tok" | "end"
    std::string id;
    int64_t token = -1;
    int steps_done = -1;
    Scheduler::Result::Status status{};
  };
  void on_token(const std::string& id, int64_t token,
                int steps_done) override {
    events_.push_back({"tok", id, token, steps_done, {}});
  }
  void on_retire(const std::string& id,
                 const Scheduler::Result& result) override {
    events_.push_back({"end", id, -1, result.steps_done, result.status});
  }
  const std::vector<Event>& events() const { return events_; }
  std::string replay() const {
    std::string s;
    for (const Event& e : events_) {
      if (!s.empty()) s += " ";
      if (e.kind == "tok") s += "T(" + e.id + "," + std::to_string(e.token) + ")";
      else s += "E(" + e.id + "," + std::to_string(e.steps_done) + ")";
    }
    return s;
  }

 private:
  std::vector<Event> events_;
};

// DGPP_TEST(scheduler_observer...) uses this; keep helpers below the
// observer so the replay formatting stays next to the contract.
size_t count_kind(const RecordingObserver& o, const std::string& kind,
                  const std::string& id) {
  size_t n = 0;
  for (const auto& e : o.events())
    if (e.kind == kind && e.id == id) ++n;
  return n;
}

DGPP_TEST(scheduler_submit_rejectsManifestErrorsLoudly) {
  // GIVEN a scheduler with one request,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));

  // WHEN malformed requests are submitted, THEN each refusal names the
  // offense (identical on every rank — a partial manifest would
  // deadlock the fabric).
  const auto throws_with = [&](SchedulerRequest r, const std::string& needle) {
    try {
      sched.submit(std::move(r));
    } catch (const std::invalid_argument& e) {
      require(std::string(e.what()).find(needle) != std::string::npos,
              "refusal must name it: " + std::string(e.what()));
      return;
    }
    throw std::runtime_error("submit should have refused: " + needle);
  };
  throws_with(make_request("a", 5, 3), "duplicate");
  throws_with(make_request("b", 0, 3), "empty prompt");
  throws_with(make_request("b", 5, 0), "at least one token");
  throws_with(make_request("b", 5, 3, /*cancel_after=*/4), "cancel_after");
}

// ---------------------------------------------------------------------------
// Stage 4 surface: tick mode, dynamic arrival, bounded queue, external
// cancel, the observer, meters.
// ---------------------------------------------------------------------------

DGPP_TEST(scheduler_tickLoop_matchesRunToCompletion_exactly) {
  // GIVEN a scenario mixing deferral, scripted cancellation, and EOS,
  const auto run = [&](bool tick_mode) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/3, /*block_tokens=*/4);
    engine.arm(0, {1, 2, kEos}, /*max_steps=*/4);  // x -> slot 0
    engine.arm(0, {7, 8}, /*max_steps=*/2);        // y -> reuses slot 0
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    sched.submit(make_request("y", 5, 2, /*cancel_after=*/1));
    // WHEN driven either by run_to_completion or by raw ticks,
    if (tick_mode)
      while (sched.tick()) {
      }
    else
      sched.run_to_completion();
    std::vector<std::string> summary;
    for (const auto& r : sched.results())
      summary.push_back(std::to_string(static_cast<int>(r.status)) + ":" +
                        ids_joined(r.generated));
    return std::make_pair(engine.op_stream(), summary);
  };

  // THEN both drivers produce the IDENTICAL op stream and results — the
  // tick extraction must be a pure refactor (the fabric's manifest path
  // rides on it staying bit-equal).
  const auto batch = run(false);
  const auto ticked = run(true);
  require(batch.first == ticked.first,
          "tick-mode op stream drifted from run_to_completion:\n  batch: " +
              batch.first + "\n  tick:  " + ticked.first);
  require(batch.second == ticked.second,
          "tick-mode results drifted from run_to_completion");
}

DGPP_TEST(scheduler_dynamicArrival_transcriptsMatchSolo) {
  // GIVEN a generating mid-answer, with b arriving mid-run,
  const auto solo = [&](const char* id, std::vector<int32_t> script,
                        int max_steps) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.arm(0, std::move(script), max_steps);
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request(id, 5, max_steps));
    sched.run_to_completion();
    return ids_joined(sched.results()[0].generated);
  };
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);  // a -> slot 0
  engine.arm(1, {5, 6}, /*max_steps=*/2);       // b -> slot 1
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 4));

  // WHEN a ticks twice alone, b submits mid-run, and the loop drives to
  // completion (the service's exact arrival shape),
  require(sched.tick(), "tick 1 works");
  require(sched.tick(), "tick 2 works");
  sched.submit(make_request("b", 5, 2));
  while (sched.tick()) {
  }

  // THEN the op stream shows b admitted at the next tick's alternation
  // slot, and BOTH transcripts equal their solo runs — dynamic arrival
  // preserves the isolation property.
  const std::string expected =
      "P:0:5 S:0:1 S:0:2 P:1:5 S:1:5 C:1 S:0:3 C:0";
  require(engine.op_stream() == expected,
          "dynamic-arrival stream drifted:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(ids_joined(sched.results()[0].generated) ==
              solo("a", {1, 2, 3, 4}, 4),
          "a's transcript must match its solo run");
  require(ids_joined(sched.results()[1].generated) == solo("b", {5, 6}, 2),
          "b's transcript must match its solo run");
  require(!sched.has_pending(), "nothing pending after completion");
}

DGPP_TEST(scheduler_boundedQueue_shedsWhenFullThenAdmits) {
  // GIVEN a scheduler with a one-deep admission queue,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a -> slot 0
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);  // b -> slot 1
  Scheduler sched(&engine, {kEos}, /*queue_limit=*/1);
  sched.submit(make_request("a", 5, 3));

  // WHEN the queue is full, try_submit sheds (no throw — this is load,
  // not an error) and submit throws QueueFullError; after a's admission
  // drains the queue, b lands and runs normally.
  require(!sched.try_submit(make_request("b", 5, 3)),
          "try_submit must shed when the queue is full");
  bool shed = false;
  try {
    sched.submit(make_request("b", 5, 3));
  } catch (const QueueFullError&) {
    shed = true;
  }
  require(shed, "submit must throw QueueFullError at the bound");
  require(sched.tick(), "tick admits a (draining the queue)");
  require(sched.try_submit(make_request("b", 5, 3)),
          "b lands once the queue has room");
  while (sched.tick()) {
  }
  require(sched.results().size() == 2,
          "exactly a and the landed b — a shed request is never recorded");
}

DGPP_TEST(scheduler_externalCancel_retiresAtNextTickWithoutExtraStep) {
  // GIVEN a mid-generation request (three tokens served, one to go),
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 4));
  require(sched.tick(), "tick 1: admit + first step");
  require(sched.tick(), "tick 2: second step");

  // WHEN the client disconnects (external cancel) before the next tick,
  require(sched.cancel("a"), "cancel flags a live request");

  // THEN the next tick's SWEEP retires a — Cancelled with the tokens
  // served so far, NO fourth engine op (a cancelled request never pays
  // one more step), and the tick reports idle-after (the retirement
  // rode on the false — the documented tick contract). A late re-cancel
  // is a no-op.
  require(!sched.tick(), "the sweep tick retires a and reports idle-after");
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "cancel must retire without an extra step:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.status == Scheduler::Result::Status::kCancelled &&
              a.reason == Scheduler::Result::Reason::kCancelled &&
              a.steps_done == 3 && ids_joined(a.generated) == "1,2,3",
          "a: Cancelled at 3 tokens");
  require(!sched.cancel("a"), "re-cancel of a terminal request is a no-op");
  require(!sched.has_pending(), "idle after the cancel");
  require(!sched.tick(), "a further tick is idle");
}

DGPP_TEST(scheduler_externalCancel_queuedRequestNeverAdmits) {
  // GIVEN a pool that fits one reservation at a time: a active, b queued,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a -> slot 0
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);  // b -> slot 1 (never used)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));
  require(sched.tick(), "tick 1 admits a; b defers on budget");

  // WHEN b is cancelled while queued,
  require(sched.cancel("b"), "cancel flags the queued request");
  while (sched.tick()) {
  }

  // THEN b retires Cancelled having never touched an engine slot (no
  // prefill, no close — the op stream is a's solo stream) and a
  // finishes untouched.
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "queued cancel must be invisible to the engine:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  const auto& b = sched.results()[1];
  require(b.status == Scheduler::Result::Status::kCancelled &&
              b.steps_done == 0 && b.generated.empty() && b.slot == -1,
          "b: Cancelled, never admitted");
  require(sched.results()[0].status == Scheduler::Result::Status::kDone &&
              ids_joined(sched.results()[0].generated) == "1,2,3",
          "a finished normally");
}

DGPP_TEST(scheduler_observer_eventsInFireOrderWithLifecycle) {
  // GIVEN a scenario with EOS, a steps cap, and a cancellation (b reuses
  // slot 0 after a's EOS retires it — lowest-free-slot policy),
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, kEos}, /*max_steps=*/5);  // a: EOS at token 2
  engine.arm(0, {4, 5, 6}, /*max_steps=*/3);  // b: reuses slot 0
  Scheduler sched(&engine, {kEos});
  RecordingObserver observer;
  sched.set_observer(&observer);
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 3, /*cancel_after=*/2));

  // WHEN run to completion,
  sched.run_to_completion();

  // THEN the observer saw every token BEFORE the retire that may follow
  // it, one retire per request, and the counts/steps match the results.
  const std::string expected =
      "T(a,1) T(a,999) E(a,2) T(b,4) T(b,5) E(b,2)";
  require(observer.replay() == expected,
          "event order drifted:\n  got:      " + observer.replay() +
              "\n  expected: " + expected);
  require(count_kind(observer, "end", "a") == 1 &&
              count_kind(observer, "end", "b") == 1,
          "exactly one retire event per request");
  const auto& a = sched.results()[0];
  const auto& b = sched.results()[1];
  require(a.reason == Scheduler::Result::Reason::kEos && a.steps_done == 2,
          "a: EOS at 2 tokens");
  require(b.status == Scheduler::Result::Status::kCancelled &&
              b.steps_done == 2,
          "b: cancelled after 2 tokens");
}

DGPP_TEST(scheduler_meters_trackQueueActiveTerminalAndTokens) {
  // GIVEN a bounded queue with one request landing,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a
  engine.arm(0, {7, 8}, /*max_steps=*/2);     // b: admits after a retires
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 2));

  // WHEN sampled mid-run and after completion,
  const auto mid = sched.meters();  // everything queued pre-tick
  while (sched.tick()) {
  }
  const auto end = sched.meters();

  // THEN the meters agree with the policy at both points (pool meters
  // read the engine's own accounting, so in-use matches its arithmetic).
  require(mid.queued == 2 && mid.active == 0 && mid.terminal == 0 &&
              mid.tokens_generated == 0 && mid.pool_blocks_total == 2,
          "pre-tick meters: two queued, nothing moving");
  require(end.queued == 0 && end.active == 0 && end.terminal == 2 &&
              end.tokens_generated == 5 && end.pool_blocks_in_use == 0,
          "post-run meters: both terminal, 5 tokens total, pool drained");
}

// Grow-on-demand (M6 6d): the observer's growth events.
struct GrowLog final : dgpp::glm::SchedulerObserver {
  std::vector<std::string> grows, retires;
  void on_token(const std::string&, int64_t, int) override {}
  void on_retire(const std::string& id, const Scheduler::Result& r) override {
    retires.push_back(id + ":" + std::to_string(static_cast<int>(r.reason)) + ":" +
                      std::to_string(r.steps_done));
  }
  void on_grow(const std::string& id, int64_t tokens) override {
    grows.push_back(id + ":" + std::to_string(tokens));
  }
};

std::vector<int32_t> long_script(int n) {
  std::vector<int32_t> t;
  for (int i = 0; i < n; ++i) t.push_back(3 + (i % 5));  // never kEos
  return t;
}

DGPP_TEST(scheduler_growOnDemand_admitsEarlyGrowsAtTickTopAndShedsTheYoungest) {
  // Pool 6 blocks of 4 tokens; two requests of prompt 4 + 12 steps (16
  // tokens = 4 blocks each). Full-reserve admits one at a time. Grow with
  // a 4-token window reserves 8 tokens = 2 blocks each: both admitted at
  // once; each grows at tick top when its next step would write past its
  // reservation; when the pool cannot cover even the minimum the YOUNGEST
  // is shed (Done / kPoolExhausted) and the oldest completes its 12 steps.
  // Every decision is a function of scheduler state alone: two identical
  // runs produce identical engine op streams and observer logs.
  using Mode = dgpp::glm::AdmissionPolicy::Mode;
  const auto run = [](Mode mode, std::string* ops, GrowLog* log) {
    FakeEngine engine(2, 6, 4);
    engine.arm(0, long_script(12), 12);
    engine.arm(mode == Mode::kGrowOnDemand ? 1 : 0, long_script(12), 12);
    dgpp::glm::AdmissionPolicy policy;
    policy.mode = mode;
    policy.window_tokens = 4;
    Scheduler sched(&engine, {kEos}, 0, policy);
    sched.set_observer(log);
    sched.submit(make_request("a", 4, 12));
    sched.submit(make_request("b", 4, 12));
    sched.run_to_completion();
    *ops = engine.op_stream();
    return std::make_pair(sched.results(), sched.meters());
  };
  std::string ops1, ops2, ops_full;
  GrowLog log1, log2, log_full;
  const auto [res1, m1] = run(Mode::kGrowOnDemand, &ops1, &log1);
  const auto [res2, m2] = run(Mode::kGrowOnDemand, &ops2, &log2);
  const auto [resf, mf] = run(Mode::kFullReserve, &ops_full, &log_full);
  // Both admitted before either retired (P:1 precedes C:0).
  require(ops1.find("P:1") != std::string::npos &&
              ops1.find("P:1") < ops1.find("C:0"),
          "grow: b admitted while a is live: " + ops1);
  require(ops_full.find("C:0") < ops_full.find("P:0:") + 1 ||
              ops_full.find("C:0") < ops_full.rfind("P:0"),
          "full: b admitted only after a retired: " + ops_full);
  require(res1[0].status == Scheduler::Result::Status::kDone &&
              res1[0].reason == Scheduler::Result::Reason::kSteps &&
              res1[0].steps_done == 12,
          "grow: the oldest completes its 12 steps");
  require(res1[1].status == Scheduler::Result::Status::kDone &&
              res1[1].reason == Scheduler::Result::Reason::kPoolExhausted &&
              res1[1].steps_done < 12 && res1[1].steps_done > 0,
          "grow: the youngest shed at exhaustion after some tokens, got " +
              std::to_string(res1[1].steps_done));
  require(!log1.grows.empty() && m1.reservations_grown ==
                                     static_cast<int64_t>(log1.grows.size()),
          "grow: growth events observed and metered");
  require(m1.requests_shed_pool == 1 && m1.pool_blocks_in_use == 0,
          "grow: one shed, the pool empty at the end");
  require(resf[0].reason == Scheduler::Result::Reason::kSteps &&
              resf[1].reason == Scheduler::Result::Reason::kSteps &&
              log_full.grows.empty() && mf.requests_shed_pool == 0,
          "full-reserve: both complete, nothing grows or sheds");
  // Determinism: the same manifest, the same streams.
  require(ops1 == ops2 && log1.grows == log2.grows && log1.retires == log2.retires,
          "grow: two runs are identical");
  // The growth targets are monotone per request and never exceed the
  // lifetime (16 tokens); the first grow lands before the 5th step (the
  // 8-token reservation covers prompt 4 + 3 written tokens + 1).
  int64_t last_a = 0;
  for (const std::string& g : log1.grows) {
    if (g.rfind("a:", 0) != 0) continue;
    const int64_t t = std::stoll(g.substr(2));
    require(t > last_a && t <= 16, "grow: a's reservation grows monotonically within its lifetime");
    last_a = t;
  }
  require(last_a == 16, "grow: a reached its full reservation");
}

DGPP_TEST(scheduler_growOnDemand_aLoneRequestShedsItselfAndAWindowNeverBelowTheStepWidth) {
  // A single request that can never fit its lifetime is admitted under
  // grow (the door check is the service's) and sheds ITSELF when the pool
  // is spent: Done / kPoolExhausted with the tokens it got. And the
  // window is clamped to the engine's step width plus one, so a step
  // never writes past a reservation.
  FakeEngine engine(1, 3, 4);  // 12 tokens of pool
  engine.arm(0, long_script(40), 40);
  dgpp::glm::AdmissionPolicy policy;
  policy.mode = dgpp::glm::AdmissionPolicy::Mode::kGrowOnDemand;
  policy.window_tokens = 1;  // below the width + 1: clamped to 2
  Scheduler sched(&engine, {kEos}, 0, policy);
  sched.submit(make_request("solo", 4, 40));
  sched.run_to_completion();
  const Scheduler::Result& r = sched.results()[0];
  require(r.status == Scheduler::Result::Status::kDone &&
              r.reason == Scheduler::Result::Reason::kPoolExhausted,
          "solo: shed itself at exhaustion");
  // 12 tokens of pool: prompt 4 + the generated tokens the KV holds; the
  // pending pick is the last one — at most 9 tokens generated.
  require(r.steps_done >= 7 && r.steps_done <= 9,
          "solo: generated what the pool allowed, got " + std::to_string(r.steps_done));
  require(sched.meters().requests_shed_pool == 1 && sched.meters().pool_blocks_in_use == 0,
          "solo: metered and released");
  bool threw = false;
  try {
    dgpp::glm::AdmissionPolicy bad;
    bad.window_tokens = 0;
    Scheduler s(&engine, {kEos}, 0, bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "a zero window is refused");
}

}  // namespace

DGPP_TEST(scheduler_sampling_specArmsTheSlotBeforeItsPrefillPick) {
  // GIVEN a sampling-capable engine and one stochastic request,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/1, /*can_sample=*/true);
  engine.arm(0, {1, 2}, /*max_steps=*/2);
  Scheduler sched(&engine, {kEos});
  SchedulerRequest r = make_request("a", 5, 2);
  r.sampling.temperature = 1.0f;
  r.sampling.top_p = 0.95f;
  r.seed = 99;
  sched.submit(std::move(r));

  // WHEN it runs,
  sched.run_to_completion();

  // THEN the slot was armed with the request's seed immediately before its
  // prefill (the prefill pick is the first draw), and nothing else moved.
  require(engine.op_stream() == "A:0:99 P:0:5 S:0:1 C:0",
          "arming must precede the prefill: " + engine.op_stream());

  // AND a greedy request on the same engine arms nothing.
  FakeEngine greedy_engine(1, 100, 4, 1, /*can_sample=*/true);
  greedy_engine.arm(0, {1, 2}, 2);
  Scheduler greedy(&greedy_engine, {kEos});
  greedy.submit(make_request("g", 5, 2));
  greedy.run_to_completion();
  require(greedy_engine.op_stream() == "P:0:5 S:0:1 C:0",
          "greedy requests keep the exact op stream: " +
              greedy_engine.op_stream());

  // AND a constrained request arms its grammar after the sampling spec and
  // before the prefill — the prefill pick is the first masked position.
  FakeEngine constrained_engine(1, 100, 4, 1, /*can_sample=*/true);
  constrained_engine.arm(0, {1, 2}, 2);
  Scheduler constrained(&constrained_engine, {kEos});
  SchedulerRequest c = make_request("c", 5, 2);
  c.sampling.temperature = 1.0f;
  c.seed = 7;
  c.grammar.mode = dgpp::glm::GrammarSpec::Mode::kRequired;
  c.grammar.tools.push_back(dgpp::glm::GrammarTool{"f", false, {}, {}, {}, false});
  constrained.submit(std::move(c));
  constrained.run_to_completion();
  require(constrained_engine.op_stream() == "A:0:7 G:0:3:1 P:0:5 S:0:1 C:0",
          "the grammar arms before the prefill: " +
              constrained_engine.op_stream());
}

DGPP_TEST(scheduler_sampling_refusalsAreManifestErrors) {
  // A stochastic request against a greedy-only engine, and an invalid spec
  // against any engine, throw at submit — identically on every rank, never
  // reaching the engine.
  FakeEngine greedy_engine(1, 100, 4);
  Scheduler greedy(&greedy_engine, {kEos});
  SchedulerRequest stochastic = make_request("s", 5, 2);
  stochastic.sampling.temperature = 0.8f;
  bool threw = false;
  try {
    greedy.submit(stochastic);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "a greedy engine must refuse a stochastic request");
  require(greedy_engine.op_stream().empty(), "nothing reached the engine");

  // A constrained request against an engine without masks: refused at
  // submit too, and identically so on every rank.
  SchedulerRequest constrained = make_request("c", 5, 2);
  constrained.grammar.mode = dgpp::glm::GrammarSpec::Mode::kForbidCalls;
  threw = false;
  try {
    greedy.submit(constrained);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an engine without masks must refuse a constrained request");
  require(greedy_engine.op_stream().empty(), "nothing reached the engine");

  FakeEngine sampling_engine(1, 100, 4, 1, /*can_sample=*/true);
  Scheduler sampler(&sampling_engine, {kEos});
  SchedulerRequest bad = make_request("b", 5, 2);
  bad.sampling.temperature = 1.0f;
  bad.sampling.top_p = 2.0f;
  threw = false;
  try {
    sampler.submit(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an invalid spec must be refused at submit");
  require(!sampler.has_pending(), "the refused request was never queued");
}

// The logprobs channel: a request that asks arms the slot ("L") before the
// prefill, every token's entry reaches the observer right after its
// on_token — including the two tokens of a speculative step — and a
// request that asks on an engine that reports none is refused at submit.
class LogprobObserver : public dgpp::glm::SchedulerObserver {
 public:
  void on_token(const std::string& id, int64_t token, int steps_done) override {
    events.push_back("T:" + id + ":" + std::to_string(token) + ":" +
                     std::to_string(steps_done));
  }
  void on_token_logprobs(const std::string& id, int steps_done,
                         const dgpp::glm_sample::Result& lp) override {
    events.push_back("L:" + id + ":" + std::to_string(lp.token) + ":" +
                     std::to_string(steps_done) + ":" +
                     std::to_string(lp.top_logprobs.size()));
  }
  void on_retire(const std::string&, const Scheduler::Result&) override {}
  std::vector<std::string> events;
};

DGPP_TEST(scheduler_logprobs_rideWithEveryTokenWhenAsked) {
  FakeEngine engine(1, 100, 4, 1, /*can_sample=*/true);
  engine.arm_batches(0, 10, {{11, 12}, {13}}, /*max_steps=*/4);
  Scheduler sched(&engine, {kEos});
  LogprobObserver observer;
  sched.set_observer(&observer);
  SchedulerRequest r = make_request("a", 5, 4);
  r.logprobs = 1;
  r.sampling.logprobs = 1;
  sched.submit(std::move(r));
  sched.run_to_completion();
  require(engine.op_stream() == "L:0 P:0:5 S:0:10 S:0:12 C:0",
          "arming precedes the prefill: " + engine.op_stream());
  const std::vector<std::string> want{
      "T:a:10:1", "L:a:10:1:1", "T:a:11:2", "L:a:11:2:1", "T:a:12:3",
      "L:a:12:3:1", "T:a:13:4", "L:a:13:4:1"};
  require(observer.events == want, "every token's logprobs follow its on_token");

  FakeEngine greedy_engine(1, 100, 4);
  Scheduler greedy(&greedy_engine, {kEos});
  SchedulerRequest ask = make_request("b", 5, 2);
  ask.logprobs = 0;
  ask.sampling.logprobs = 0;
  bool threw = false;
  try {
    greedy.submit(ask);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an engine without logprobs refuses at submit");
}

// ---- the prefix cache (M7 stage B) ------------------------------------------
// The fake's geometry below: blocks of 4 tokens, pool alignment 4, chunks
// of 2048 (never reached), so a prompt's cuts are the aligned images of its
// boundaries only.

SchedulerRequest make_cached_request(const std::string& id,
                                     std::vector<int64_t> prompt,
                                     std::vector<int64_t> boundaries,
                                     int max_steps) {
  SchedulerRequest r;
  r.id = id;
  r.prompt = std::move(prompt);
  r.boundaries = std::move(boundaries);
  r.max_steps = max_steps;
  return r;
}

std::vector<int64_t> counted_prompt(int n, int64_t base = 100) {
  std::vector<int64_t> p(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) p[static_cast<size_t>(i)] = base + i;
  return p;
}

DGPP_TEST(scheduler_prefixCache_secondIdenticalPromptAttachesAtTheDeepestCut) {
  // GIVEN a 21-token prompt with boundaries at 5 and 13 (cuts 4 and 12) run
  // twice, back to back, on an engine with a four-slot arena:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(1, {1, 2, 3}, 3);
  Scheduler sched(&engine, {kEos});
  require(sched.prefix_slots() == 4, "the cache takes the engine's arena");
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  sched.submit(make_cached_request("b", counted_prompt(21), {5, 13}, 3));
  sched.run_to_completion();

  // THEN the first prefill takes an entry at the deepest cut (12), the
  // second attaches to it there and takes none (the cut is the entry's);
  // both answers are the scripts'. The rolling snapshots: committed
  // positions 21, 22 — never aligned before the cap — so none.
  const std::string expected =
      "P:0:21 N:0:12@0 S:0:1 X:1:12@0 P:1:21 S:1:1 S:0:2 C:0 S:1:2 C:1";
  require(engine.op_stream() == expected,
          "prefix cache op stream:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const Scheduler::Meters m = sched.meters();
  require(m.prefix_hits == 1 && m.prefix_misses == 1 && m.prefix_snapshots == 1 &&
              m.prefix_tokens_saved == 12 && m.prefix_entries == 1 &&
              m.prefix_blocks_pinned == 3,
          "prefix meters: one miss, one hit of 12 tokens, one entry pinning 3 blocks");
  require(ids_joined(sched.results()[1].generated) == "1,2,3", "b's ids");
}

DGPP_TEST(scheduler_prefixCache_rollingSnapshotBecomesTheCloseEntryTheNextTurnAttachesTo) {
  // GIVEN a 21-token prompt whose answer runs six tokens and ends in EOS:
  // committed positions 21 (after the prefill pick), 22, 23, 24 <- aligned,
  // 25, 26 (the EOS pending) — one rolling snapshot at 24; at retire the
  // entry sits at floor((27 - 1) / 4) * 4 = 24, the aligned image of the
  // EOS token's position 26.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34, 35, kEos}, 8);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 8));
  sched.run_to_completion();
  const std::string expected_a =
      "P:0:21 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:24@1 S:0:34 S:0:35 C:0";
  require(engine.op_stream() == expected_a,
          "turn one:\n  got:      " + engine.op_stream() + "\n  expected: " + expected_a);
  Scheduler::Meters m = sched.meters();
  require(m.prefix_rolling == 1 && m.prefix_close_entries == 1 && m.prefix_entries == 2,
          "one rolling snapshot became the close entry; two entries live");

  // WHEN the next turn arrives: the conversation so far (the prompt, the
  // five answer tokens, the EOS as the next message's marker) and a new
  // message, with boundaries at the markers (5, 13, 26 -> cuts 4, 12, 24):
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, 34, 35, kEos});
  next.insert(next.end(), {200, 201, 202, 203});
  engine.arm(0, {41, 42}, 2);
  sched.submit(make_cached_request("b", next, {5, 13, 26}, 2));
  sched.run_to_completion();

  // THEN it attaches at 24 (the close entry, the deepest cut) and prefills
  // the seven remaining tokens; no new entry at a cut (24 is the deepest),
  // and its own close entry at 32 — the retire-time snapshot, its committed
  // position 31 + 2 - 1 being aligned.
  const std::string expected_b = expected_a + " X:0:24@1 P:0:31 S:0:41 RS:0:32@2 C:0";
  require(engine.op_stream() == expected_b,
          "turn two:\n  got:      " + engine.op_stream() + "\n  expected: " + expected_b);
  m = sched.meters();
  require(m.prefix_hits == 1 && m.prefix_tokens_saved == 24, "the second turn hit at 24");
}

DGPP_TEST(scheduler_prefixCache_retireAtAnAlignedPositionSnapshotsTheLiveState) {
  // GIVEN a 21-token prompt whose answer ends at the cap after four tokens:
  // committed positions 21, 22, 23 — never aligned before the cap — and 24
  // at retire, aligned. THEN the retire takes the snapshot from the live
  // slot (before the close) and it becomes the close entry at 24, where
  // the next turn (a marker at the fourth answer token's position, 24)
  // cuts and attaches.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34}, 4);
  engine.arm(0, {41}, 1);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 4));
  sched.run_to_completion();
  const std::string expected_a = "P:0:21 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:24@1 C:0";
  require(engine.op_stream() == expected_a,
          "retire-time snapshot:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_a);
  require(sched.meters().prefix_close_entries == 1, "the close entry at 24");
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, kEos, 200, 201});
  sched.submit(make_cached_request("b", next, {5, 13, 24}, 1));
  sched.run_to_completion();
  const std::string expected_b = expected_a + " X:0:24@1 P:0:27 C:0";
  require(engine.op_stream() == expected_b,
          "the next turn attaches at 24:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_b);
}

DGPP_TEST(scheduler_prefixCache_twoTokenStepsHopOverAnAlignedPositionAndSnapshotIt) {
  // The MTP graph's two-token steps (M7's measured limit, closed
  // 2026-09-05): a 21-token prompt's committed count runs 21, 23, 25 —
  // never ON 24 — so no rolling snapshot could land there. The scheduler
  // arms the engine at 23 for the hop over 24 (the fake takes the state
  // after the step's first row when the step commits two), and the hop
  // becomes the close entry the next turn (a marker at 24) attaches to.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.set_step_tokens_max(2);
  engine.arm_batches(0, 31, {{32, 33}, {34, 35}, {36, 37}}, 6);
  engine.arm(0, {41}, 1);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 6));
  sched.run_to_completion();
  // Committed 21 (22 is not aligned: no arm) → [32,33] → 23, armed for 24 →
  // [34,35] commits two: the hop at 24 → 25 (26 not aligned) → [36,37] →
  // the cap at six tokens (37 dropped) → retire with committed 26: the
  // close entry is the hop's, at 24.
  const std::string expected_a = "P:0:21 N:0:12@0 S:0:31 H:0:24@1 S:0:33 S:0:35 C:0";
  require(engine.op_stream() == expected_a,
          "the hop:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_a);
  require(sched.meters().prefix_hops == 1 && sched.meters().prefix_rolling == 1 &&
              sched.meters().prefix_close_entries == 1,
          "one hop, counted as a rolling snapshot, one close entry");
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, kEos, 200, 201});
  sched.submit(make_cached_request("b", next, {5, 13, 24}, 1));
  sched.run_to_completion();
  const std::string expected_b = expected_a + " X:0:24@1 P:0:27 C:0";
  require(engine.op_stream() == expected_b,
          "the next turn attaches at the hop's entry:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_b);
  // A one-token engine never arms (the plain stream), and a two-token step
  // that commits ONE token lands on the position: the regular rolling
  // snapshot takes it at the next tick.
  FakeEngine one(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  one.set_prefix_arena(/*slots=*/4, /*align=*/4);
  one.set_step_tokens_max(2);
  one.arm_batches(0, 31, {{32, 33}, {34}, {35, 36}}, 6);
  Scheduler sched1(&one, {kEos});
  sched1.submit(make_cached_request("a", prompt, {5, 13}, 6));
  sched1.run_to_completion();
  // 21 → [32,33] → 23 (armed for 24) → [34] commits one: 24, no hop → the
  // rolling snapshot at 24 before the next step → [35,36] → the cap.
  const std::string expected_one = "P:0:21 N:0:12@0 S:0:31 S:0:33 RS:0:24@1 S:0:34 C:0";
  require(one.op_stream() == expected_one,
          "a one-token step lands on the position:\n  got:      " + one.op_stream() +
              "\n  expected: " + expected_one);
  require(sched1.meters().prefix_hops == 0 && sched1.meters().prefix_rolling == 1,
          "no hop, one rolling snapshot");
}

DGPP_TEST(scheduler_prefixCache_optOutAndNoArenaKeepThePlainOpStream) {
  const auto run = [](bool arena, bool no_cache) {
    FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
    if (arena) engine.set_prefix_arena(2, 4);
    engine.arm(0, {1, 2}, 2);
    engine.arm(0, {1, 2}, 2);
    Scheduler sched(&engine, {kEos});
    SchedulerRequest a = make_cached_request("a", counted_prompt(21), {5, 13}, 2);
    SchedulerRequest b = make_cached_request("b", counted_prompt(21), {5, 13}, 2);
    a.no_cache = no_cache;
    b.no_cache = no_cache;
    sched.submit(a);
    sched.submit(b);
    sched.run_to_completion();
    return engine.op_stream();
  };
  const std::string plain = "P:0:21 S:0:1 C:0 P:0:21 S:0:1 C:0";
  require(run(false, false) == plain, "no arena: the plain op stream");
  require(run(true, true) == plain, "opted out: the plain op stream");
  require(run(true, false) != plain, "the cache on changes the stream (the control)");
}

DGPP_TEST(scheduler_prefixCache_evictsTheLeastRecentlyUsedUnattachedEntry) {
  // GIVEN a one-slot arena and three prompts with different first tokens:
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  for (int i = 0; i < 4; ++i) engine.arm(0, {1}, 1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21, 100), {5, 13}, 1));
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 1));
  sched.submit(make_cached_request("a2", counted_prompt(21, 100), {5, 13}, 1));
  sched.submit(make_cached_request("b2", counted_prompt(21, 300), {5, 13}, 1));
  sched.run_to_completion();
  // THEN each new prompt evicts the other's entry (LRU, unattached — the
  // previous request retired) before its own snapshot; a2 misses (a's
  // entry went), b2 misses (b's entry went).
  const std::string expected =
      "P:0:21 N:0:12@0 C:0 F:0 P:0:21 N:0:12@0 C:0 F:0 P:0:21 N:0:12@0 C:0 "
      "F:0 P:0:21 N:0:12@0 C:0";
  require(engine.op_stream() == expected,
          "eviction:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  const Scheduler::Meters m = sched.meters();
  require(m.prefix_evictions == 3 && m.prefix_hits == 0 && m.prefix_misses == 4 &&
              m.prefix_entries == 1,
          "three evictions, no hits, one entry live");
}

DGPP_TEST(scheduler_prefixCache_entriesGiveTheirBlocksToARequestThatNeedsThem) {
  // GIVEN a pool of 12 blocks (48 tokens): a's entry pins 3 blocks; a
  // request needing 10 blocks (37 tokens + 3 steps) cannot fit beside it
  // (12 - 3 = 9 free) — the entry is evicted for it, deterministically.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/12, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/2, /*align=*/4);
  engine.arm(0, {1}, 1);
  engine.arm(0, {2, 3, 4}, 3);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 1));
  sched.run_to_completion();
  require(sched.meters().prefix_blocks_pinned == 3, "a's entry pins 3 blocks");
  sched.submit(make_cached_request("big", counted_prompt(37, 500), {}, 3));
  sched.run_to_completion();
  const std::string expected = "P:0:21 N:0:12@0 C:0 F:0 P:0:37 S:0:2 S:0:3 C:0";
  require(engine.op_stream() == expected,
          "block pressure:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  require(sched.meters().prefix_evictions == 1 && sched.meters().prefix_blocks_pinned == 0,
          "the entry gave its blocks");
}

DGPP_TEST(scheduler_prefixCache_attachedEntriesAreNeverEvicted) {
  // GIVEN a one-slot arena: a's entry is attached by b (live, cap 4) when
  // c (a different prompt) wants a slot for its own snapshot — none is
  // evictable, so c's snapshot is skipped and counted; b's answer is
  // unaffected. When b itself retires at an aligned position (24), its
  // reference has returned first, so a's entry is evictable for b's own
  // close entry — and is evicted.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.arm(0, {1}, 1);
  engine.arm(0, {5, 6, 7, 8}, 4);
  engine.arm(1, {9}, 1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 1));
  sched.run_to_completion();
  sched.submit(make_cached_request("b", counted_prompt(21), {5, 13}, 4));
  sched.submit(make_cached_request("c", counted_prompt(21, 300), {5, 13}, 1));
  sched.run_to_completion();
  const std::string expected =
      "P:0:21 N:0:12@0 C:0 X:0:12@0 P:0:21 S:0:5 P:1:21 C:1 S:0:6 S:0:7 F:0 RS:0:24@0 C:0";
  require(engine.op_stream() == expected,
          "attached entry:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  require(sched.meters().prefix_skipped == 1 && sched.meters().prefix_evictions == 1 &&
              sched.meters().prefix_close_entries == 1,
          "c's snapshot skipped while b was attached; a's entry gave way to b's close entry");
}

DGPP_TEST(scheduler_prefixCache_sameStreamTwice_identicalOpsAndDigests) {
  // Rank identity in miniature: two schedulers over two fakes, fed the same
  // requests, produce the same op streams and digests; a third whose last
  // prompt differs (a miss where the others hit) produces another digest.
  const auto run = [](bool variant) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.set_prefix_arena(/*slots=*/3, /*align=*/4);
    engine.arm(0, {1, 2, 3, 4, 5, kEos}, 8);
    engine.arm(1, {6, 7}, 2);
    engine.arm(1, {8}, 1);  // c lands on slot 1, freed first (b's two steps)
    Scheduler sched(&engine, {kEos});
    sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 8));
    sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
    sched.submit(make_cached_request("c", counted_prompt(21, variant ? 500 : 100),
                                     {5, 13}, 1));
    sched.run_to_completion();
    return std::make_pair(engine.op_stream(), sched.prefix_digest());
  };
  const auto one = run(false), two = run(false), three = run(true);
  require(one.first == two.first && one.second == two.second,
          "identical streams: identical ops and digests");
  require(one.second != three.second, "a different stream: a different digest");
  require(one.first.find("X:1:12@0") != std::string::npos,
          "c attached to a's entry in the base run: " + one.first);
}

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}
