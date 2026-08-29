// M5 control-plane tests: TCP primitives and the epoch-based roster, all on
// loopback with in-process clients. Every assertion targets a final state
// (with generous polling budgets), never a sleep-race.
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "common/log.hpp"
#include "common/test.hpp"
#include "net/roster.hpp"
#include "net/tcp.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Fn>
bool wait_until(Fn&& predicate, int timeout_ms) {
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  while (Clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return predicate();
}

dgpp::net::RosterServerOptions base_server_options(int world_size) {
  return dgpp::net::RosterServerOptions{
      .world_size = world_size,
      .listen_port = 0,  // overwritten by the fixture
      .startup_timeout_ms = 3000,
      .heartbeat_interval_ms = 40,  // advisory cadence for ranks
      .heartbeat_deadline_ms = 300,
      // In-process loopback: every rank, including 0, is a TCP client, so
      // there is no implicit coordinator self-membership here.
      .local_rank = -1,
  };
}

dgpp::net::RosterClientOptions client_options(
    int rank, uint16_t port, int heartbeat_interval_ms = 40,
    int ack_deadline_ms = 800, int join_timeout_ms = 5000) {
  return dgpp::net::RosterClientOptions{
      .rank = rank,
      .node = "test-node-r" + std::to_string(rank),
      .coordinator_host = "127.0.0.1",
      .coordinator_port = port,
      .heartbeat_interval_ms = heartbeat_interval_ms,
      .ack_deadline_ms = ack_deadline_ms,
      .join_timeout_ms = join_timeout_ms,
  };
}

// Reserves an ephemeral port by binding and releasing it, so both the server
// and the clients know the address up front and nothing polls for it (the
// alternative — racing on server.port() from the starter thread — is a data
// race TSan would rightly flag).
uint16_t reserve_ephemeral_port() {
  auto probe = dgpp::net::TcpListener::bind(0);
  const uint16_t port = probe.port();
  return port;  // probe closes on destruction
}

// A server plus the thread running start() (which blocks until the roster
// is sealed, i.e. until enough clients have joined).
struct ClusterFixture {
  const uint16_t port;
  dgpp::net::RosterServer server;
  std::thread starter;
  std::atomic<bool> started{false};
  std::string error;  // written before started flips; sync via join/atomic

  explicit ClusterFixture(dgpp::net::RosterServerOptions options)
      : port(reserve_ephemeral_port()),
        server([&] {
          options.listen_port = port;
          return options;
        }()) {
    starter = std::thread([this] { started = server.start(&error); });
  }

  // Waits for start() to finish (either way) and requires success.
  void expect_started(const char* context) {
    starter.join();
    if (!started) {
      throw std::runtime_error(std::string(context) +
                               ": server start failed: " + error);
    }
  }

  // Waits for start() to finish without requiring success.
  void await_finished() {
    starter.join();
  }

  ~ClusterFixture() {
    if (starter.joinable()) starter.join();
    server.shutdown();
  }
};

// Runs client start()+join() expecting the join to fail; returns the message.
std::string join_failure_message(uint16_t port, int rank) {
  dgpp::net::RosterClient client(client_options(rank, port));
  client.start();
  try {
    client.join();
  } catch (const std::exception& e) {
    return e.what();
  }
  throw std::runtime_error("join was expected to fail but succeeded");
}

}  // namespace

DGPP_TEST(tcp_connect_to_dead_port_throws) {
  // GIVEN a listener whose port is captured and then closed:
  const uint16_t port = reserve_ephemeral_port();

  // WHEN connecting to the now-dead port,
  // THEN connect throws instead of hanging or succeeding:
  bool threw = false;
  try {
    auto conn = dgpp::net::TcpConn::connect("127.0.0.1", port, 500);
    conn.close();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "connect to a dead port must throw");
}

DGPP_TEST(tcp_roundtrip_uses_ephemeral_port) {
  // GIVEN a listener on an ephemeral port:
  auto listener = dgpp::net::TcpListener::bind(0);
  require(listener.port() != 0, "ephemeral bind must pick a real port");
  std::thread acceptor([&listener] {
    auto server_side = listener.accept(2000);
    char buf[6]{};
    server_side.set_io_deadline_ms(2000);
    require(server_side.read_exact(buf, 5), "server read failed");
    require(std::string(buf) == "hello", "server read wrong bytes");
    require(server_side.write_all("world", 5), "server write failed");
    server_side.close();
  });

  // WHEN a client connects and exchanges a message,
  auto client = dgpp::net::TcpConn::connect("127.0.0.1", listener.port(), 2000);
  client.set_io_deadline_ms(2000);
  require(client.write_all("hello", 5), "client write failed");
  char buf[6]{};
  require(client.read_exact(buf, 5), "client read failed");

  // THEN the roundtrip is byte-exact:
  require(std::string(buf) == "world", "client read wrong bytes");
  client.close();
  acceptor.join();
}

DGPP_TEST(roster_full_startup_all_ranks_see_epoch1) {
  ClusterFixture cluster(base_server_options(2));
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  dgpp::net::RosterClient r1(client_options(1, cluster.port));
  r1.start();

  // WHEN both ranks join:
  const auto snap0 = r0.join();
  const auto snap1 = r1.join();
  cluster.expect_started("full startup");

  // THEN everyone agrees on the sealed roster at epoch 1:
  require(snap0.epoch == 1, "rank 0 epoch must be 1");
  require(snap1.epoch == 1, "rank 1 epoch must be 1");
  require(snap0.members.size() == 2, "roster must contain both ranks");
  require(snap0.contains(0) && snap0.contains(1), "both ranks are members");
  require(snap0.members[0].node == "test-node-r0" &&
              snap0.members[1].node == "test-node-r1",
          "roster must carry node names, sorted by rank");
  const auto server_snap = cluster.server.snapshot();
  require(server_snap.epoch == 1 && server_snap.members.size() == 2,
          "server snapshot must match");
  r0.stop();
  r1.stop();
}

DGPP_TEST(roster_rejects_bad_late_announces_with_reason) {
  ClusterFixture cluster(base_server_options(2));
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  dgpp::net::RosterClient r1(client_options(1, cluster.port));
  r1.start();
  r0.join();
  r1.join();
  cluster.expect_started("startup");

  // WHEN announcing with an out-of-range rank,
  // THEN the rejection reason is legible (not a bare connection reset):
  {
    const std::string msg = join_failure_message(cluster.port, 7);
    require(msg.find("out of range") != std::string::npos,
            "expected out-of-range rejection, got: " + msg);
  }
  // And a duplicate rank id is refused:
  {
    const std::string msg = join_failure_message(cluster.port, 0);
    require(msg.find("already announced") != std::string::npos,
            "expected duplicate rejection, got: " + msg);
  }
  // And after a member dies, its replacement is refused too: the v1 roster
  // is all-or-nothing and never re-seals with a substitute mid-flight.
  r1.stop();  // rank 1's process dies
  require(wait_until(
              [&] { return cluster.server.snapshot().epoch >= 2; }, 2000),
          "death must bump the epoch");
  {
    const std::string msg = join_failure_message(cluster.port, 1);
    require(msg.find("already sealed") != std::string::npos,
            "expected sealed-roster rejection, got: " + msg);
  }
  r0.stop();
}

DGPP_TEST(roster_dead_rank_is_evicted_with_epoch_bump) {
  ClusterFixture cluster(base_server_options(3));
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  dgpp::net::RosterClient r1(client_options(1, cluster.port));
  r1.start();
  dgpp::net::RosterClient r2(client_options(2, cluster.port));
  r2.start();
  r0.join();
  r1.join();
  r2.join();
  cluster.expect_started("startup");

  // WHEN rank 2's process dies (silent close, no protocol):
  r2.stop();

  // THEN the survivors see the epoch bump and the shrunken roster:
  require(r0.wait_for_epoch(2, 5000), "epoch 2 never reached rank 0");
  const auto snap = r0.current();
  require(snap.epoch == 2, "epoch must be 2 after one eviction");
  require(snap.members.size() == 2 && !snap.contains(2),
          "rank 2 must be gone from the roster");
  // Roster propagation to the other survivor is asynchronous; poll for it
  // rather than asserting an instant view of another thread's snapshot.
  require(wait_until([&] { return r1.current().epoch == 2; }, 5000),
          "rank 1 must agree on the epoch");
  require(wait_until([&] {
            const auto s = cluster.server.snapshot();
            return s.epoch == 2 && s.members.size() == 2;
          }, 2000),
          "server snapshot must reflect the eviction");
  r0.stop();
  r1.stop();
}

DGPP_TEST(roster_silent_rank_is_evicted_by_heartbeat_deadline) {
  // This pins the deadline path (connection alive, process unresponsive),
  // as opposed to the EOF path covered by the death test above.
  dgpp::net::RosterServerOptions opts = base_server_options(2);
  opts.heartbeat_deadline_ms = 150;
  ClusterFixture cluster(opts);
  // Rank 1 never heartbeats but stays connected, and its ack deadline is far
  // away so it does not abandon the coordinator first.
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  dgpp::net::RosterClient deadbeat(
      client_options(1, cluster.port, /*heartbeat_interval_ms=*/60000,
                     /*ack_deadline_ms=*/5000));
  deadbeat.start();
  r0.join();
  deadbeat.join();
  cluster.expect_started("startup");

  // WHEN the heartbeat deadline passes with no beats from rank 1,
  // THEN it is evicted and — unlike a dead process — still running to
  // observe its own eviction:
  require(r0.wait_for_epoch(2, 5000), "deadline eviction never happened");
  require(wait_until([&] {
            return deadbeat.state() ==
                   dgpp::net::RosterClientState::kEvicted;
          }, 5000),
          "deadbeat must learn it was evicted, not lost");
  require(!r0.current().contains(1), "rank 1 must be out of the roster");
  r0.stop();
}

DGPP_TEST(roster_startup_timeout_fails_with_progress_report) {
  dgpp::net::RosterServerOptions opts = base_server_options(2);
  opts.startup_timeout_ms = 200;
  ClusterFixture cluster(opts);

  // GIVEN only one of two ranks ever announces (join fails: the roster never
  // seals, and the coordinator tears the connection down with it):
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  try {
    r0.join();
  } catch (const std::exception&) {
    // expected: no roster is ever sealed
  }
  cluster.await_finished();

  // THEN start() reports failure with the exact progress:
  require(!cluster.started, "start must fail without a full roster");
  require(cluster.error.find("1 of 2") != std::string::npos,
          "error must state progress, got: " + cluster.error);
  // And the abandoned rank leaves the joined state (its connection died
  // with the failed startup):
  require(wait_until([&] {
            return r0.state() != dgpp::net::RosterClientState::kJoined &&
                   r0.state() != dgpp::net::RosterClientState::kJoining;
          }, 5000),
          "abandoned rank must leave the joined state");
  r0.stop();
}

DGPP_TEST(roster_shutdown_is_graceful) {
  ClusterFixture cluster(base_server_options(2));
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  dgpp::net::RosterClient r1(client_options(1, cluster.port));
  r1.start();
  r0.join();
  r1.join();
  cluster.expect_started("startup");

  // WHEN the coordinator shuts down gracefully,
  cluster.server.shutdown();

  // THEN ranks observe the protocol stop rather than inferring loss:
  require(wait_until([&] {
            return r0.state() == dgpp::net::RosterClientState::kStopped &&
                   r1.state() == dgpp::net::RosterClientState::kStopped;
          }, 5000),
          "both ranks must reach the stopped state");
  r0.stop();
  r1.stop();
}

DGPP_TEST(roster_hard_stop_reports_coordinator_loss) {
  ClusterFixture cluster(base_server_options(2));
  dgpp::net::RosterClient r0(client_options(0, cluster.port, 40, 200));
  r0.start();
  dgpp::net::RosterClient r1(client_options(1, cluster.port, 40, 200));
  r1.start();
  r0.join();
  r1.join();
  cluster.expect_started("startup");

  // WHEN the coordinator dies without any protocol (hard stop):
  cluster.server.stop();

  // THEN ranks report coordinator loss, which is fatal to the current batch:
  require(wait_until([&] {
            return r0.state() ==
                       dgpp::net::RosterClientState::kCoordinatorLost &&
                   r1.state() ==
                       dgpp::net::RosterClientState::kCoordinatorLost;
          }, 5000),
          "both ranks must detect coordinator loss");
  r0.stop();
  r1.stop();
}

DGPP_TEST(roster_wait_for_epoch_times_out_when_quiet) {
  ClusterFixture cluster(base_server_options(1));
  dgpp::net::RosterClient r0(client_options(0, cluster.port));
  r0.start();
  r0.join();
  cluster.expect_started("single-rank startup");

  // WHEN waiting for an epoch that will never arrive:
  const bool reached = r0.wait_for_epoch(2, 150);

  // THEN the wait times out cleanly with false, no throw:
  require(!reached, "epoch 2 must not be reached");
  require(r0.state() == dgpp::net::RosterClientState::kJoined,
          "rank must still be joined after a quiet timeout");
  r0.stop();
}
