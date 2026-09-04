#pragma once
// The M6 Stage 4b admission journal — the metronome that keeps every
// fabric rank's scheduler identical while requests arrive at rank 0's
// HTTP door (DESIGN §5: rank 0 owns admission; §11: identical rank
// order).
//
// THE PROTOCOL (newline-framed JSON over the TCP star; one line, one
// record):
//   {"op":"warm"}
//   {"op":"tick","s":[{"id":"…","p":[ids],"m":N}],"c":["id"]}
//   {"op":"stop"}
// The "warm" record is the start signal for the graph engine's startup
// warm capture (GlmGraphEngineAdapter::warm_captures): a run of bus
// collectives every rank must enter together, before any tick. Rank 0
// broadcasts it once its model is built; a peer holds at the record
// instead of spinning its first collective in stall diagnostics for the
// seconds rank 0's slower construction takes (seen 2026-09-03: ~8 s of
// STALLED dumps per peer). It is the only record allowed before the
// first tick and never valid after it.
// Every rank-0 engine pass is exactly one "tick" record, broadcast
// AFTER the pass's drain and BEFORE its sched.tick() — the record and
// the tick are one atomic unit. Peers apply the record's
// submits/cancels, then tick. Tick counts are identical by
// construction: rank 0 never ticks without broadcasting; a peer never
// ticks without a record. And because every engine op is a bus
// COLLECTIVE (synchronous across ranks), a record can never interleave
// a peer's in-flight tick — when rank 0 leaves a tick, every rank has
// left the same tick, so the stop record (or any record) always
// lands between ticks, never inside one.
//
// WHAT RIDES THE JOURNAL: only scheduler-state CHANGES rank 0 made —
// try_submit-ACCEPTED requests (with full prompt ids; rank 0
// tokenized them) and cancel() requests that hit. Door sheds (503)
// and admission sheds die on rank 0 and are never journaled;
// steps/retires/tokens are DERIVED state every rank computes
// identically from the same records (§11). A peer's try_submit
// refusing what rank 0 admitted is a scheduler divergence — the one
// thing this design says cannot happen — so the peer loop makes it
// LOUD and dies rather than serving on a lie.
//
// DEATH DISCIPLINE: a peer sees journal EOF when rank 0's process is
// gone (clean stop, crash, kill -9 — TCP closes the sockets either
// way) and exits; TCP is the liveness probe, no heartbeats needed. A
// dead peer makes rank 0's broadcast fail — the fabric is broken, and
// rank 0 aborts loudly instead of serving on without it (the bus
// watchdogs would catch it at the next collective; the journal
// catches it BETWEEN collectives, the gap the bus cannot see).
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "models/glm_scheduler.hpp"
#include "net/tcp.hpp"
#include "service/generation_service.hpp"

namespace dgpp::service {

// The audit tap on the engine event stream. The fabric run attaches
// one on every rank (rank 0 through GenerationService::set_audit_
// observer, peers through Scheduler::set_observer) and the
// verification hashes the per-rank texts against each other — the
// smoke's md5 ritual, serving edition. The format is deliberately
// deterministic and rank-independent: same events in the same order
// MUST produce the same bytes on every rank. Appends happen on the
// engine thread; readers (the run's exit write, the tests' live polls)
// take copies under the lock.
class OpStreamObserver final : public dgpp::glm::SchedulerObserver {
 public:
  void on_token(const std::string& id, int64_t token,
                int steps_done) override;
  void on_retire(const std::string& id,
                 const dgpp::glm::Scheduler::Result& result) override;
  void on_grow(const std::string& id, int64_t reserved_tokens) override;
  std::string text() const;

 private:
  mutable std::mutex mutex_;
  std::string text_;
};

// ---- wire codec ----------------------------------------------------------

std::string encode_journal_tick(const GenerationService::PassEvents& events);
std::string encode_journal_stop();
// The warm record carries rank 0's admission policy (M6 6d): every rank's
// scheduler must run the same one, and the peers take it from here.
std::string encode_journal_warm(
    const dgpp::glm::AdmissionPolicy& policy = dgpp::glm::AdmissionPolicy{});

struct JournalRecord {
  bool stop = false;
  bool warm = false;
  bool has_admission = false;  // warm: the policy rode along
  dgpp::glm::AdmissionPolicy admission;
  std::vector<dgpp::glm::SchedulerRequest> submits;
  std::vector<std::string> cancels;
};
// Throws on any malformed record — a corrupt journal is a fabric
// emergency, not a condition to paper over.
JournalRecord decode_journal_line(std::string_view line);

// ---- rank 0 ----------------------------------------------------------------

class JournalWriter {
 public:
  // Binds the journal port (0 = ephemeral; see port()) but does NOT
  // wait for peers — the app binds first so peers can be launched
  // against a known port, then accepts.
  explicit JournalWriter(uint16_t listen_port);

  uint16_t port() const { return listener_.port(); }

  // Accepts world-1 peer connections in any order. Throws when the
  // full world fails to land within accept_timeout_ms — a short world
  // cannot serve, and pretending otherwise would wedge every request
  // at its first collective.
  void accept_peers(int world, int accept_timeout_ms);

  // One line to every peer. A peer that cannot take it is a dead
  // member of a live-serving world — the fabric is broken, and this
  // says so loudly.
  void broadcast(const std::string& line);

 private:
  dgpp::net::TcpListener listener_;
  std::vector<dgpp::net::TcpConn> peers_;  // rank 1..world-1, arrival order
};

// ---- peers ------------------------------------------------------------------

class JournalReader {
 public:
  // Connects to rank 0's journal port, RETRYING within the window: the
  // peer's bus rendezvous can complete before rank 0 binds the journal
  // (a race measured in tens of milliseconds), and a single-shot
  // connect loses it. Throws once the whole window expires — startup
  // is exceptional and the peer wants the reason in its log.
  JournalReader(const std::string& host, uint16_t port,
                int connect_timeout_ms);

  // The next complete record (newline stripped). false when rank 0's
  // journal closed (clean stop or crash — EOF either way) or the
  // caller's stop flag fired; the two are indistinguishable to the
  // peer on purpose: both mean "the world is over, exit now".
  bool read_line(const std::function<bool()>& should_stop,
                 std::string* line);

 private:
  dgpp::net::TcpConn conn_;
  std::string pending_;
};

// The peer's startup hold: blocks until rank 0's warm record. Returns
// false when the journal ended (EOF or the caller's stop flag) — the
// peer exits cleanly, as it would from the loop. Throws when the first
// record is anything else: rank 0 ticked before warming, a protocol
// order this design says cannot happen.
bool wait_journal_warm(JournalReader* reader,
                       const std::function<bool()>& should_stop,
                       dgpp::glm::AdmissionPolicy* policy = nullptr);

// The peer serving loop: apply each record, then tick — the exact
// mirror of rank 0's engine passes (§11). Returns on the stop record,
// journal EOF, or the caller's stop flag. Throws on journal
// corruption or scheduler divergence (both are fabric emergencies).
// Attach the observer BEFORE calling: the loop drives the scheduler,
// so whatever the observer should see, it sees here.
void run_journal_peer(dgpp::glm::Scheduler* sched, JournalReader* reader,
                      const std::function<bool()>& should_stop);

}  // namespace dgpp::service
