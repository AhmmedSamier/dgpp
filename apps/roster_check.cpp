// roster_check: M5 deliverable 1 deployment driver — epoch-based
// roster/startup and rank health over TCP, on real nodes or loopback.
//
//   roster_check coordinator --world 4 [--port 29500] [--startup-ms ...]
//                              [--duration-ms ...]
//   roster_check rank --id 2 --host <coordinator> [--port 29500]
//                     [--duration-ms ...]
//   roster_check selftest
//
// The coordinator is rank 0 (DESIGN §5) and joins the roster implicitly.
// The selftest exercises the whole lifecycle in-process on loopback and is
// wired into ctest; the coordinator/rank modes are the four-node
// validation commands.
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "common/log.hpp"
#include "net/roster.hpp"
#include "net/tcp.hpp"

namespace {

std::sig_atomic_t g_interrupted = 0;

void on_sigint(int) { g_interrupted = 1; }

std::string local_hostname() {
  char hostname[256]{};
  if (::gethostname(hostname, sizeof(hostname) - 1) != 0) {
    return "unknown";
  }
  return hostname;
}

std::string describe(const dgpp::net::RosterSnapshot& snap) {
  std::string out = "epoch=" + std::to_string(snap.epoch) + " members=";
  for (size_t i = 0; i < snap.members.size(); ++i) {
    if (i > 0) out += ",";
    out += std::to_string(snap.members[i].rank) + ":" + snap.members[i].node;
  }
  return out;
}

// Returns true once the predicate holds or the deadline passes.
template <typename Fn>
bool wait_until(Fn&& predicate, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

// Reserve an ephemeral port by binding and releasing it, so both sides know
// the address before the server's blocking start() runs.
uint16_t reserve_ephemeral_port() {
  auto probe = dgpp::net::TcpListener::bind(0);
  return probe.port();  // released on destruction here
}

struct Args {
  std::vector<std::pair<std::string, int*>> ints;
  std::vector<std::pair<std::string, std::string*>> strings;
};

// "--name value" pairs; every option must be known and well-formed.
bool parse_args(int argc, char** argv, int start, const Args& spec) {
  for (int i = start; i < argc; ++i) {
    const std::string name = argv[i];
    bool matched = false;
    for (auto& [key, target] : spec.ints) {
      if (name == key && i + 1 < argc) {
        *target = std::atoi(argv[++i]);
        matched = true;
        break;
      }
    }
    if (matched) continue;
    for (auto& [key, target] : spec.strings) {
      if (name == key && i + 1 < argc) {
        *target = argv[++i];
        matched = true;
        break;
      }
    }
    if (!matched) return false;
  }
  return true;
}

int cmd_coordinator(int argc, char** argv) {
  int world = 0;
  int startup_ms = 30000;
  int port = 29500;
  int duration_ms = 10000;
  int interval_ms = 1000;
  int deadline_ms = 3000;
  const Args spec{
      .ints = {{"--world", &world},
               {"--startup-ms", &startup_ms},
               {"--port", &port},
               {"--duration-ms", &duration_ms},
               {"--heartbeat-interval-ms", &interval_ms},
               {"--heartbeat-deadline-ms", &deadline_ms}},
      .strings = {},
  };
  if (!parse_args(argc, argv, 2, spec) || world < 1 || startup_ms <= 0 ||
      port <= 0 || duration_ms <= 0 || deadline_ms <= interval_ms) {
    std::fprintf(stderr,
                 "usage: roster_check coordinator --world N [--startup-ms MS] "
                 "[--port P] [--duration-ms MS]\n"
                 "       (heartbeat deadline must exceed the interval)\n");
    return 2;
  }

  dgpp::net::RosterServerOptions opts{
      .world_size = world,
      .listen_port = static_cast<uint16_t>(port),
      .startup_timeout_ms = startup_ms,
      .heartbeat_interval_ms = interval_ms,
      .heartbeat_deadline_ms = deadline_ms,
      .local_rank = 0,
  };
  dgpp::net::RosterServer server(opts);
  std::string error;
  if (!server.start(&error)) {
    DGPP_LOG_ERROR("coordinator startup failed: {}", error);
    return 1;
  }
  DGPP_LOG_INFO("coordinator up on port {} — ROSTER {}", port,
                describe(server.snapshot()));

  std::signal(SIGINT, on_sigint);
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(duration_ms);
  uint64_t last_epoch = server.epoch();
  while (std::chrono::steady_clock::now() < end && !g_interrupted) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto snap = server.snapshot();
    if (snap.epoch != last_epoch) {
      last_epoch = snap.epoch;
      DGPP_LOG_INFO("ROSTER changed — {}", describe(snap));
    }
  }

  server.shutdown();
  const auto final_snap = server.snapshot();
  DGPP_LOG_INFO("coordinator shut down cleanly — final {}",
                describe(final_snap));
  // A healthy run keeps every rank; anything evicted during the run is a
  // finding, not a success. Ctrl+C is an intentional stop, not a failure.
  const bool full = static_cast<int>(final_snap.members.size()) == world;
  return full || g_interrupted != 0 ? 0 : 1;
}

int cmd_rank(int argc, char** argv) {
  int id = -1;
  int port = 29500;
  int duration_ms = 10000;
  int interval_ms = 1000;
  int ack_ms = 3000;
  std::string host;
  const Args spec{
      .ints = {{"--id", &id},
               {"--port", &port},
               {"--duration-ms", &duration_ms},
               {"--heartbeat-interval-ms", &interval_ms},
               {"--ack-deadline-ms", &ack_ms}},
      .strings = {{"--host", &host}},
  };
  if (!parse_args(argc, argv, 2, spec) || id < 0 || host.empty() ||
      port <= 0 || duration_ms <= 0 || ack_ms <= interval_ms) {
    std::fprintf(stderr,
                 "usage: roster_check rank --id N --host HOST [--port P] "
                 "[--duration-ms MS]\n");
    return 2;
  }

  dgpp::net::RosterClientOptions opts{
      .rank = id,
      .node = local_hostname(),
      .coordinator_host = host,
      .coordinator_port = static_cast<uint16_t>(port),
      .heartbeat_interval_ms = interval_ms,
      .ack_deadline_ms = ack_ms,
      .join_timeout_ms = 30000,
  };
  dgpp::net::RosterClient client(opts);
  client.start();
  std::signal(SIGINT, on_sigint);
  try {
    const auto snap = client.join();
    DGPP_LOG_INFO("rank {} joined via {} — ROSTER {}", id, host,
                  describe(snap));
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {} failed to join: {}", id, e.what());
    return 1;
  }

  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(duration_ms);
  uint64_t last_epoch = client.current().epoch;
  while (std::chrono::steady_clock::now() < end && !g_interrupted) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const auto state = client.state();
    if (state != dgpp::net::RosterClientState::kJoined) {
      // Stopped is the graceful end (coordinator shut down); anything else
      // is a finding for this run.
      DGPP_LOG_WARN("rank {} state: {} — ROSTER {}", id,
                    dgpp::net::roster_client_state_name(state),
                    describe(client.current()));
      client.stop();
      return state == dgpp::net::RosterClientState::kStopped ? 0 : 1;
    }
    const auto snap = client.current();
    if (snap.epoch != last_epoch) {
      last_epoch = snap.epoch;
      DGPP_LOG_INFO("rank {} sees ROSTER {}", id, describe(snap));
    }
  }
  client.stop();
  DGPP_LOG_INFO("rank {} stopped locally after its duration", id);
  return 0;
}

// In-process loopback run of the full lifecycle: seal, epoch bump on death,
// graceful shutdown. Mirrors the unit tests but through the driver's own
// wiring; ctest runs this as the app-level smoke.
int cmd_selftest() {
  const uint16_t port = reserve_ephemeral_port();

  dgpp::net::RosterServerOptions opts{
      .world_size = 3,
      .listen_port = port,
      .startup_timeout_ms = 3000,
      .heartbeat_interval_ms = 40,
      .heartbeat_deadline_ms = 300,
      .local_rank = 0,
  };
  dgpp::net::RosterServer server(opts);
  bool started = false;
  std::string error;
  std::thread starter([&] { started = server.start(&error); });

  const auto make_client = [port](int rank) {
    return dgpp::net::RosterClient(dgpp::net::RosterClientOptions{
        .rank = rank,
        .node = local_hostname(),
        .coordinator_host = "127.0.0.1",
        .coordinator_port = port,
        .heartbeat_interval_ms = 40,
        .ack_deadline_ms = 800,
        .join_timeout_ms = 3000,
    });
  };
  dgpp::net::RosterClient rank1(make_client(1));
  dgpp::net::RosterClient rank2(make_client(2));
  rank1.start();
  rank2.start();

  int failures = 0;
  const auto check = [&](bool condition, const std::string& what) {
    if (condition) {
      DGPP_LOG_INFO("selftest OK: {}", what);
    } else {
      ++failures;
      DGPP_LOG_ERROR("selftest FAILED: {}", what);
    }
  };

  try {
    const auto snap1 = rank1.join();
    check(snap1.epoch == 1 && snap1.members.size() == 3,
          "sealed roster has all three ranks");
    check(snap1.contains(0) && snap1.members[0].node == local_hostname(),
          "coordinator's own rank 0 is an implicit member");
    const auto snap2 = rank2.join();
    check(snap2.epoch == 1, "both ranks agree on epoch 1");

    rank2.stop();  // rank 2 dies silently
    check(wait_until([&] { return rank1.current().epoch >= 2; }, 5000),
          "survivor observes the epoch bump after a death");
    check(wait_until([&] {
            const auto s = server.snapshot();
            return s.epoch == 2 && s.members.size() == 2;
          }, 5000),
          "coordinator evicts the dead rank");

    server.shutdown();
    check(wait_until([&] {
            return rank1.state() == dgpp::net::RosterClientState::kStopped;
          }, 5000),
          "graceful shutdown reaches the rank");
    rank1.stop();
  } catch (const std::exception& e) {
    ++failures;
    DGPP_LOG_ERROR("selftest FAILED: unexpected exception: {}", e.what());
  }
  if (starter.joinable()) starter.join();
  check(started, "server start succeeded");

  DGPP_LOG_INFO("roster selftest: {} failures", failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: roster_check <command> [options]\n"
                 "  coordinator --world N [--port P] [--startup-ms MS]\n"
                 "               [--duration-ms MS]\n"
                 "  rank --id N --host HOST [--port P] [--duration-ms MS]\n"
                 "  selftest\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "coordinator") return cmd_coordinator(argc, argv);
  if (cmd == "rank") return cmd_rank(argc, argv);
  if (cmd == "selftest") return cmd_selftest();
  DGPP_LOG_ERROR("unknown command: {}", cmd);
  return 2;
}
