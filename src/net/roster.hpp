#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dgpp::net {

// One member of the current roster. Absent from the roster == not executing
// requests; eviction removes the rank entirely rather than marking it.
struct RankInfo {
  int rank = -1;
  std::string node;
  bool healthy = false;
};

// The coordinator-owned membership view. Every frame between coordinator and
// rank carries the producing epoch; a rank that observes a non-increasing
// epoch treats the current batch as fatal (DESIGN §5).
struct RosterSnapshot {
  uint64_t epoch = 0;
  std::vector<RankInfo> members;  // sorted by rank

  const RankInfo* find(int rank) const;
  bool contains(int rank) const { return find(rank) != nullptr; }
};

// ---- wire framing (version 1) ---------------------------------------------
//
// 20-byte little-endian header (host is little-endian on this cluster; the
// version byte leaves migration room):
//   u32 magic 'DGPP' | u8 version | u8 type | u16 reserved | u64 epoch |
//   u32 payload_len
//
// types: ANNOUNCE (rank -> coordinator, join), ROSTER (coordinator -> rank),
// HEARTBEAT (rank -> coordinator), HEARTBEAT_ACK (coordinator -> rank),
// SHUTDOWN (coordinator -> rank), REJECT (coordinator -> rank: announce
// refused, payload is the reason — config errors must be legible in the
// rank's log, not a bare connection reset).

enum class RosterFrameType : uint8_t {
  kAnnounce = 1,
  kRoster = 2,
  kHeartbeat = 3,
  kHeartbeatAck = 4,
  kShutdown = 5,
  kReject = 6,
};

// ---- coordinator side (rank 0's process) -----------------------------------

struct RosterServerOptions {
  int world_size = 1;
  uint16_t listen_port = 0;        // 0 = ephemeral (tests); fixed in prod
  int startup_timeout_ms = 30000;  // wait for all world_size ranks
  int heartbeat_interval_ms = 1000;  // advisory: ranks send at this cadence
  int heartbeat_deadline_ms = 3000;  // evict after this without a beat
  // The coordinator's own rank (DESIGN §5: rank 0 additionally owns
  // admission/scheduling). It joins the roster implicitly at seal and is
  // never deadline-evicted — if the coordinator dies, every rank learns via
  // connection loss, which is fatal for the batch by design. -1 disables.
  int local_rank = 0;
};

class RosterServer {
 public:
  explicit RosterServer(RosterServerOptions options);
  ~RosterServer();

  RosterServer(const RosterServer&) = delete;
  RosterServer& operator=(const RosterServer&) = delete;

  // Binds, serves, and blocks until the roster is sealed (all world_size
  // ranks announced) or the startup timeout expires. On false, *error holds
  // the reason and the server is fully stopped.
  bool start(std::string* error);

  uint16_t port() const;
  RosterSnapshot snapshot() const;
  uint64_t epoch() const;

  // Protocol shutdown: broadcasts SHUTDOWN at the current epoch so ranks
  // exit cleanly, then stops serving. Also invoked by the destructor.
  void shutdown();

  // Hard stop without the SHUTDOWN frame (simulates coordinator loss).
  void stop();

 private:
  struct Impl;
  Impl* impl_;
};

// ---- rank side (every non-coordinator process) -----------------------------

enum class RosterClientState {
  kJoining,          // connect/announce/first-roster pending
  kJoined,           // heartbeating, snapshot current
  kEvicted,          // roster no longer contains this rank; batch is fatal
  kCoordinatorLost,  // no server frames within the ack deadline; fatal
  kStopped,          // server sent SHUTDOWN, or local stop() was called
};

struct RosterClientOptions {
  int rank = -1;
  std::string node;                // this process's hostname, for the roster
  std::string coordinator_host;    // rank 0's address
  uint16_t coordinator_port = 0;
  int heartbeat_interval_ms = 1000;
  int ack_deadline_ms = 3000;  // no frame from the server within => lost
  int join_timeout_ms = 30000;  // connect retries + wait for first roster
};

class RosterClient {
 public:
  explicit RosterClient(RosterClientOptions options);
  ~RosterClient();

  RosterClient(const RosterClient&) = delete;
  RosterClient& operator=(const RosterClient&) = delete;

  // Starts connect/announce/heartbeat on a background thread. Returns
  // immediately — call this for every rank before blocking in join(),
  // or the roster can never seal.
  void start();

  // Blocks until the first sealed roster is available and returns it.
  // Throws std::runtime_error if rejected, timed out, or the coordinator
  // never sealed the roster. Safe to call again (returns the first roster).
  RosterSnapshot join();

  RosterSnapshot current() const;
  RosterClientState state() const;

  // Blocks until the snapshot's epoch reaches want. false on timeout;
  // throws if the client entered a fatal state while waiting.
  bool wait_for_epoch(uint64_t want, int timeout_ms);

  // Local teardown. Closes the connection without any protocol frame — from
  // the coordinator's perspective this process just died.
  void stop();

 private:
  struct Impl;
  Impl* impl_;
};

const char* roster_client_state_name(RosterClientState state);

}  // namespace dgpp::net
