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
// The fake also enforces the engine seam: step()'s prev_token must equal
// the last token served for that slot (cross-request contamination fails
// loudly), scripts are consumed in order, and close() must find a live
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
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens)
      : slots_(slots),
        total_blocks_(total_blocks),
        block_tokens_(block_tokens) {}

  // Arms `slot`'s NEXT episode: the token stream (prefill picks tokens[0],
  // each step picks the next) and the max_steps the reservation mirrors.
  void arm(int slot, std::vector<int32_t> tokens, int max_steps) {
    episodes_[slot].push_back(
        Episode{std::move(tokens), max_steps});
  }

  const std::vector<std::string>& ops() const { return ops_; }
  std::string op_stream() const {
    std::string s;
    for (const std::string& op : ops_) {
      if (!s.empty()) s += " ";
      s += op;
    }
    return s;
  }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) sum += live.held_blocks;
    return sum;
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
    require(!live.episode.tokens.empty(),
            "fake: empty token script for slot " + std::to_string(req));
    // Mirror the scheduler's reserve: prompt + max_steps, block-rounded.
    live.held_blocks =
        blocks_for_tokens(static_cast<int64_t>(prompt.size()) +
                           live.episode.max_steps);
    live.served = 1;
    live.last_token = live.episode.tokens[0];
    live_[req] = live;
    ops_.push_back("P:" + std::to_string(req) + ":" +
                   std::to_string(prompt.size()));
    return live.last_token;
  }

  int32_t step(int req, int64_t prev_token) override {
    require(live_.count(req) != 0,
            "fake: step on unopened slot " + std::to_string(req));
    Live& live = live_[req];
    require(static_cast<int64_t>(live.last_token) == prev_token,
            "fake: slot " + std::to_string(req) + " stepped with prev " +
                std::to_string(prev_token) + " but was last served " +
                std::to_string(live.last_token) +
                " — cross-request contamination");
    require(live.served < live.episode.tokens.size(),
            "fake: token script exhausted for slot " +
                std::to_string(req) + " (a step the policy should not "
                "have issued — script the test correctly)");
    live.last_token = live.episode.tokens[live.served];
    ++live.served;
    ops_.push_back("S:" + std::to_string(req) + ":" +
                   std::to_string(prev_token));
    return live.last_token;
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
    std::vector<int32_t> tokens;
    int max_steps;
  };
  struct Live {
    Episode episode;
    int64_t held_blocks = 0;
    size_t served = 0;
    int32_t last_token = -1;
  };

  int slots_;
  int64_t total_blocks_;
  int64_t block_tokens_;
  std::map<int, std::vector<Episode>> episodes_;
  std::map<int, Live> live_;
  std::vector<std::string> ops_;
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

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}
