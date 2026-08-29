#include "net/roster.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "net/tcp.hpp"

namespace dgpp::net {
namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kMagic = 0x50504744;  // "DGPP" packed little-endian
constexpr uint8_t kVersion = 1;
constexpr size_t kHeaderBytes = 20;  // magic|version|type|rsv|epoch|len
constexpr uint32_t kMaxPayload = 1u << 20;  // 1 MiB; rosters are tiny
constexpr int kLoopTickMs = 20;             // max idle sleep per loop pass
constexpr int kAnnounceDeadlineMs = 5000;   // announce exchange after accept;
                                           // generous: a slow-but-healthy
                                           // rank must not be silently
                                           // dropped under load
constexpr int kConnIoDeadlineMs = 2000;     // safety net for established conns

struct ReceivedFrame {
  uint8_t type;
  uint64_t epoch;
  std::vector<uint8_t> payload;
};

// ---- little-endian payload codec ------------------------------------------

void pack_u32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

bool unpack_u32(const std::vector<uint8_t>& in, size_t& pos, uint32_t* out) {
  if (pos + 4 > in.size()) return false;
  *out = 0;
  for (int i = 3; i >= 0; --i) {
    *out = (*out << 8) | in[pos + static_cast<size_t>(i)];
  }
  pos += 4;
  return true;
}

void pack_str(std::vector<uint8_t>& out, const std::string& s) {
  out.push_back(static_cast<uint8_t>(s.size()));
  out.insert(out.end(), s.begin(), s.end());
}

bool unpack_str(const std::vector<uint8_t>& in, size_t& pos,
                size_t full_len, std::string* out) {
  if (pos + full_len > in.size()) return false;
  out->assign(reinterpret_cast<const char*>(in.data() + pos), full_len);
  pos += full_len;
  return true;
}

bool send_frame(TcpConn& conn, RosterFrameType type, uint64_t epoch,
                const std::vector<uint8_t>& payload) {
  unsigned char header[kHeaderBytes]{};
  std::memcpy(header + 0, &kMagic, 4);
  header[4] = kVersion;
  header[5] = static_cast<uint8_t>(type);
  std::memcpy(header + 8, &epoch, 8);
  uint32_t len = static_cast<uint32_t>(payload.size());
  std::memcpy(header + 16, &len, 4);
  return conn.write_all(header, kHeaderBytes) &&
         (payload.empty() ||
          conn.write_all(payload.data(), payload.size()));
}

// false on malformed framing or a dead connection; the caller treats both
// as protocol events.
bool read_frame(TcpConn& conn, ReceivedFrame* out) {
  unsigned char header[kHeaderBytes]{};
  if (!conn.read_exact(header, kHeaderBytes)) {
    DGPP_LOG_WARN("roster: frame header read failed (peer closed or "
                  "malformed stream)");
    return false;
  }
  uint32_t magic = 0, payload_len = 0;
  std::memcpy(&magic, header + 0, 4);
  std::memcpy(&payload_len, header + 16, 4);
  if (magic != kMagic || header[4] != kVersion || payload_len > kMaxPayload) {
    DGPP_LOG_WARN("roster: malformed frame (magic={:#x} version={} len={})",
                  magic, header[4], payload_len);
    return false;
  }
  out->type = header[5];
  std::memcpy(&out->epoch, header + 8, 8);
  out->payload.assign(payload_len, 0);
  if (payload_len > 0 && !conn.read_exact(out->payload.data(), payload_len)) {
    DGPP_LOG_WARN("roster: frame payload read failed ({} bytes)",
                  payload_len);
    return false;
  }
  return true;
}

std::vector<uint8_t> encode_announce(int rank, const std::string& node) {
  std::vector<uint8_t> out;
  pack_u32(out, static_cast<uint32_t>(rank));
  out.push_back(static_cast<uint8_t>(node.size()));
  out.insert(out.end(), node.begin(), node.end());
  return out;
}

bool decode_announce(const std::vector<uint8_t>& in, int* rank,
                     std::string* node) {
  size_t pos = 0;
  uint32_t r = 0;
  if (!unpack_u32(in, pos, &r)) return false;
  if (pos + 1 > in.size()) return false;
  const size_t len = in[pos++];
  if (!unpack_str(in, pos, len, node)) return false;
  *rank = static_cast<int>(r);
  return pos == in.size();
}

std::vector<uint8_t> encode_roster(const RosterSnapshot& snap) {
  std::vector<uint8_t> out;
  pack_u32(out, static_cast<uint32_t>(snap.members.size()));
  for (const auto& m : snap.members) {
    pack_u32(out, static_cast<uint32_t>(m.rank));
    out.push_back(m.healthy ? 1 : 0);
    out.push_back(static_cast<uint8_t>(m.node.size()));
    out.insert(out.end(), m.node.begin(), m.node.end());
  }
  return out;
}

bool decode_roster(const std::vector<uint8_t>& in, uint64_t epoch,
                   RosterSnapshot* snap) {
  size_t pos = 0;
  uint32_t count = 0;
  if (!unpack_u32(in, pos, &count)) return false;
  snap->epoch = epoch;
  snap->members.clear();
  snap->members.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t rank = 0;
    if (!unpack_u32(in, pos, &rank)) return false;
    if (pos + 2 > in.size()) return false;
    const bool healthy = in[pos++] != 0;
    const size_t len = in[pos++];
    std::string node;
    if (!unpack_str(in, pos, len, &node)) return false;
    snap->members.push_back({static_cast<int>(rank), std::move(node),
                             healthy});
  }
  std::sort(snap->members.begin(), snap->members.end(),
            [](const RankInfo& a, const RankInfo& b) {
              return a.rank < b.rank;
            });
  return pos == in.size();
}

std::vector<uint8_t> encode_reason(const std::string& reason) {
  std::vector<uint8_t> out;
  pack_str(out, reason);
  return out;
}

bool decode_reason(const std::vector<uint8_t>& in, std::string* reason) {
  if (in.empty()) return false;
  const size_t len = in[0];
  size_t pos = 1;
  return unpack_str(in, pos, len, reason) && pos == in.size();
}

}  // namespace

const RankInfo* RosterSnapshot::find(int rank) const {
  for (const auto& m : members) {
    if (m.rank == rank) return &m;
  }
  return nullptr;
}

const char* roster_client_state_name(RosterClientState state) {
  switch (state) {
    case RosterClientState::kJoining:
      return "joining";
    case RosterClientState::kJoined:
      return "joined";
    case RosterClientState::kEvicted:
      return "evicted";
    case RosterClientState::kCoordinatorLost:
      return "coordinator-lost";
    case RosterClientState::kStopped:
      return "stopped";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// RosterServer
// ---------------------------------------------------------------------------

struct RosterServer::Impl {
  explicit Impl(RosterServerOptions o) : opt(o) {}

  enum class StopMode { kNone, kGraceful, kHard };

  RosterServerOptions opt;
  TcpListener listener;
  std::thread loop_thread;
  std::atomic<StopMode> stop_mode{StopMode::kNone};

  std::mutex mu;  // guards snapshot/sealed/start_failed + cv
  std::condition_variable cv;
  RosterSnapshot snapshot;
  bool sealed = false;
  bool start_failed = false;
  std::string start_error;

  // Remote announces still needed before the roster can seal.
  int expected_announces() const {
    return opt.world_size - (opt.local_rank >= 0 ? 1 : 0);
  }

  // Loop-thread-owned connection state (only serve() touches these).
  struct ConnState {
    TcpConn conn;
    int rank = -1;
    std::string node;
    Clock::time_point last_beat;
  };
  std::vector<ConnState> conns;

  bool is_rank_connected(int rank) const {
    for (const auto& c : conns) {
      if (c.rank == rank) return true;
    }
    return false;
  }

  RosterSnapshot current_snapshot() {
    std::lock_guard<std::mutex> lock(mu);
    return snapshot;
  }

  void publish(const RosterSnapshot& snap) {
    std::lock_guard<std::mutex> lock(mu);
    snapshot = snap;
    cv.notify_all();
  }

  void seal_roster() {
    RosterSnapshot snap;
    snap.epoch = 1;
    if (opt.local_rank >= 0) {
      char hostname[256]{};
      ::gethostname(hostname, sizeof(hostname) - 1);
      snap.members.push_back(
          {opt.local_rank, std::string(hostname), true});
    }
    for (const auto& c : conns) {
      snap.members.push_back({c.rank, c.node, true});
    }
    std::sort(snap.members.begin(), snap.members.end(),
              [](const RankInfo& a, const RankInfo& b) {
                return a.rank < b.rank;
              });
    publish(snap);
    {
      std::lock_guard<std::mutex> lock(mu);
      sealed = true;
    }
    broadcast_roster(snap);
    cv.notify_all();
    DGPP_LOG_INFO("roster: sealed epoch 1 with {} ranks", snap.members.size());
  }

  // Broadcasts the snapshot to every connection. A conn whose write fails
  // died mid-broadcast; it stays a member until its own EOF/deadline is
  // processed on a later pass (which bumps the epoch again — correct).
  void broadcast_roster(const RosterSnapshot& snap) {
    const auto payload = encode_roster(snap);
    for (auto& c : conns) {
      if (!send_frame(c.conn, RosterFrameType::kRoster, snap.epoch, payload)) {
        DGPP_LOG_WARN("roster: rank {} unreadable during broadcast",
                      c.rank);
      }
    }
  }

  // Removes a member, bumps the epoch, rebroadcasts. The evicted rank still
  // receives its final roster (so it can tell "evicted" from "coordinator
  // lost") before its connection is closed.
  void evict(int rank) {
    RosterSnapshot snap = current_snapshot();
    snap.epoch += 1;
    snap.members.erase(
        std::remove_if(snap.members.begin(), snap.members.end(),
                       [rank](const RankInfo& m) { return m.rank == rank; }),
        snap.members.end());
    publish(snap);
    broadcast_roster(snap);
    close_rank(rank);
    DGPP_LOG_WARN("roster: evicted rank {} at epoch {} ({} members left)",
                  rank, snap.epoch, snap.members.size());
  }

  void close_rank(int rank) {
    for (auto it = conns.begin(); it != conns.end(); ++it) {
      if (it->rank == rank) {
        conns.erase(it);
        return;
      }
    }
  }

  void reject_and_close(TcpConn& conn, const std::string& reason) {
    send_frame(conn, RosterFrameType::kReject, 0, encode_reason(reason));
    conn.close();
    DGPP_LOG_WARN("roster: rejected connection: {}", reason);
  }

  // Announce exchange, run synchronously right after accept so the poll set
  // only ever contains fully-announced conns.
  void handle_accept(TcpConn incoming) {
    incoming.set_io_deadline_ms(kAnnounceDeadlineMs);
    if (!incoming.wait_readable(kAnnounceDeadlineMs)) {
      incoming.close();
      return;
    }
    ReceivedFrame frame;
    if (!read_frame(incoming, &frame) ||
        frame.type != static_cast<uint8_t>(RosterFrameType::kAnnounce) ||
        frame.epoch != 0) {
      reject_and_close(incoming, "expected an announce frame first");
      return;
    }
    int rank = -1;
    std::string node;
    if (!decode_announce(frame.payload, &rank, &node)) {
      reject_and_close(incoming, "malformed announce payload");
      return;
    }
    if (rank < 0 || rank >= opt.world_size) {
      reject_and_close(incoming, "rank " + std::to_string(rank) +
                                     " out of range [0," +
                                     std::to_string(opt.world_size) + ")");
      return;
    }
    if (rank == opt.local_rank) {
      reject_and_close(incoming,
                       "rank " + std::to_string(rank) +
                           " is the coordinator's own rank");
      return;
    }
    if (is_rank_connected(rank)) {
      reject_and_close(incoming,
                       "rank " + std::to_string(rank) + " already announced");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      if (sealed) {
        const std::string reason =
            "roster already sealed at epoch " + std::to_string(snapshot.epoch);
        reject_and_close(incoming, reason);
        return;
      }
    }
    incoming.set_io_deadline_ms(kConnIoDeadlineMs);
    conns.push_back(
        {std::move(incoming), rank, node, Clock::now()});
    DGPP_LOG_INFO("roster: rank {} (node {}) announced", rank, node);
  }

  // Returns true if the connection was removed (caller must re-scan).
  bool handle_conn_frame(ConnState& c) {
    ReceivedFrame frame;
    if (!read_frame(c.conn, &frame)) {
      // EOF or malformed: this process is gone or misbehaving.
      drop_conn(c.rank);
      return true;
    }
    if (frame.type == static_cast<uint8_t>(RosterFrameType::kHeartbeat)) {
      c.last_beat = Clock::now();
      const RosterSnapshot snap = current_snapshot();
      send_frame(c.conn, RosterFrameType::kHeartbeatAck, snap.epoch, {});
      return false;
    }
    DGPP_LOG_WARN("roster: unexpected frame type {} from rank {}", frame.type,
                  c.rank);
    drop_conn(c.rank);
    return true;
  }

  // Post-seal: eviction with epoch bump. Pre-seal: the conn never made the
  // roster, so it is dropped silently and the startup gate stays open.
  void drop_conn(int rank) {
    bool is_sealed;
    {
      std::lock_guard<std::mutex> lock(mu);
      is_sealed = sealed;
    }
    if (is_sealed) {
      evict(rank);
    } else {
      close_rank(rank);
      DGPP_LOG_INFO("roster: rank {} connection lost before seal", rank);
    }
  }

  void serve() {
    const auto startup_deadline =
        Clock::now() + std::chrono::milliseconds(opt.startup_timeout_ms);

    while (stop_mode.load(std::memory_order_relaxed) == StopMode::kNone) {
      bool is_sealed;
      {
        std::lock_guard<std::mutex> lock(mu);
        is_sealed = sealed;
      }
      if (!is_sealed) {
        if (static_cast<int>(conns.size()) == expected_announces()) {
          seal_roster();
          is_sealed = true;
        } else if (Clock::now() >= startup_deadline) {
          {
            std::lock_guard<std::mutex> lock(mu);
            start_failed = true;
            start_error =
                "startup timeout: " + std::to_string(conns.size()) + " of " +
                std::to_string(expected_announces()) +
                " remote ranks announced";
            cv.notify_all();
          }
          // Exit through the normal cleanup path so the half-joined
          // connections are closed (to them, the coordinator just died)
          // instead of lingering until the destructor.
          stop_mode.store(StopMode::kHard);
          break;
        }
      }

      // Heartbeat deadlines protect sealed batches; before the seal there
      // is no batch to protect and the startup timeout governs. Enforcing
      // them pre-seal would silently close healthy-but-slow ranks under
      // load (announce delivered, first heartbeat not yet).
      if (is_sealed) {
        // evict()/close_rank() mutate conns, so re-scan from the beginning
        // whenever a conn was removed.
        for (auto it = conns.begin(); it != conns.end();) {
          if (Clock::now() - it->last_beat >=
              std::chrono::milliseconds(opt.heartbeat_deadline_ms)) {
            evict(it->rank);
            it = conns.begin();
          } else {
            ++it;
          }
        }
      }

      bool any_event = false;
      TcpConn incoming = listener.accept(0);
      if (incoming.valid()) {
        handle_accept(std::move(incoming));
        any_event = true;
      }
      bool removed = false;
      for (auto& c : conns) {
        if (c.conn.wait_readable(0)) {
          any_event = true;
          if (handle_conn_frame(c)) removed = true;
          if (removed) break;  // conns mutated; rescan next pass
        }
      }
      if (!any_event) {
        // Single blocking wait per idle pass; existing-conn frames are
        // picked up on the next pass, bounded by kLoopTickMs.
        TcpConn sleeper = listener.accept(kLoopTickMs);
        if (sleeper.valid()) handle_accept(std::move(sleeper));
      }
    }

    // Graceful shutdown broadcasts SHUTDOWN at the current epoch; the hard
    // path just drops the connections (that is what coordinator loss looks
    // like to the ranks).
    if (stop_mode.load() == StopMode::kGraceful) {
      const RosterSnapshot snap = current_snapshot();
      for (auto& c : conns) {
        send_frame(c.conn, RosterFrameType::kShutdown, snap.epoch, {});
      }
    }
    conns.clear();
    listener.close();
  }
};

RosterServer::RosterServer(RosterServerOptions options)
    : impl_(new Impl(options)) {}

RosterServer::~RosterServer() {
  if (impl_->loop_thread.joinable()) stop();
  delete impl_;
}

bool RosterServer::start(std::string* error) {
  if (impl_->loop_thread.joinable()) {
    if (error) *error = "server already started";
    return false;
  }
  if (impl_->opt.world_size < 1) {
    if (error) *error = "world_size must be >= 1";
    return false;
  }
  if (impl_->opt.local_rank < -1 ||
      impl_->opt.local_rank >= impl_->opt.world_size) {
    if (error) {
      *error = "local_rank must be -1 (off) or inside [0, world_size)";
    }
    return false;
  }
  if (impl_->opt.heartbeat_deadline_ms <= impl_->opt.heartbeat_interval_ms) {
    if (error) {
      *error = "heartbeat_deadline_ms must exceed heartbeat_interval_ms";
    }
    return false;
  }
  try {
    impl_->listener = TcpListener::bind(impl_->opt.listen_port);
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return false;
  }
  impl_->loop_thread = std::thread([this] { impl_->serve(); });

  std::unique_lock<std::mutex> lock(impl_->mu);
  impl_->cv.wait(lock, [this] { return impl_->sealed || impl_->start_failed; });
  if (impl_->start_failed) {
    lock.unlock();
    impl_->stop_mode.store(Impl::StopMode::kHard);
    if (impl_->loop_thread.joinable()) impl_->loop_thread.join();
    if (error) *error = impl_->start_error;
    return false;
  }
  return true;
}

uint16_t RosterServer::port() const { return impl_->listener.port(); }

RosterSnapshot RosterServer::snapshot() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->snapshot;
}

uint64_t RosterServer::epoch() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->snapshot.epoch;
}

void RosterServer::shutdown() {
  if (!impl_->loop_thread.joinable()) return;
  impl_->stop_mode.store(Impl::StopMode::kGraceful);
  impl_->loop_thread.join();
}

void RosterServer::stop() {
  if (!impl_->loop_thread.joinable()) return;
  impl_->stop_mode.store(Impl::StopMode::kHard);
  impl_->loop_thread.join();
}

// ---------------------------------------------------------------------------
// RosterClient
// ---------------------------------------------------------------------------

struct RosterClient::Impl {
  explicit Impl(RosterClientOptions o) : opt(std::move(o)) {}

  RosterClientOptions opt;
  std::thread worker;
  std::atomic<bool> running{true};
  TcpConn conn;  // worker-thread-owned once run() starts

  std::mutex mu;  // guards snapshot/first_snapshot/state/join_error + cv
  std::condition_variable cv;
  RosterSnapshot snapshot;
  // The roster this rank was sealed with, frozen at receipt: join() must
  // return it even if an eviction (this rank's own, or another's) has since
  // moved `snapshot` forward.
  RosterSnapshot first_snapshot;
  bool have_snapshot = false;
  bool launched = false;
  bool joined_once = false;
  RosterClientState state = RosterClientState::kJoining;
  std::string join_error;

  void set_state(RosterClientState s) {
    std::lock_guard<std::mutex> lock(mu);
    state = s;
    cv.notify_all();
  }

  // Local stop(): nothing is wrong with the coordinator; this process is
  // just shutting down.
  void mark_local_stop() {
    std::lock_guard<std::mutex> lock(mu);
    if (state == RosterClientState::kJoining ||
        state == RosterClientState::kJoined) {
      state = RosterClientState::kStopped;
    }
    cv.notify_all();
  }

  // Failure while trying to join (or a protocol violation from a
  // coordinator we no longer trust): join() surfaces the reason.
  void fail_join(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mu);
    if (!joined_once) join_error = reason;
    state = RosterClientState::kCoordinatorLost;
    cv.notify_all();
  }

  void run() {
    const auto join_deadline =
        Clock::now() + std::chrono::milliseconds(opt.join_timeout_ms);
    for (;;) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          join_deadline - Clock::now()).count();
      if (remaining <= 0) {
        fail_join("join timeout: coordinator unreachable within " +
                  std::to_string(opt.join_timeout_ms) + " ms");
        return;
      }
      try {
        conn = TcpConn::connect(opt.coordinator_host, opt.coordinator_port,
                                 static_cast<int>(
                                     std::min<long long>(1000, remaining)));
        break;
      } catch (const std::exception& e) {
        if (!running.load(std::memory_order_relaxed)) {
          mark_local_stop();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(
            static_cast<int>(std::min<long long>(200, remaining))));
      }
    }
    conn.set_io_deadline_ms(2000);

    if (!send_frame(conn, RosterFrameType::kAnnounce, 0,
                    encode_announce(opt.rank, opt.node))) {
      fail_join("coordinator connection lost during announce");
      return;
    }

    auto last_server_frame = Clock::now();
    auto last_beat_sent = Clock::now();
    bool got_first_roster = false;

    while (running.load(std::memory_order_relaxed)) {
      if (conn.wait_readable(10)) {
        ReceivedFrame frame;
        if (!read_frame(conn, &frame)) {
          on_conn_died(got_first_roster);
          return;
        }
        last_server_frame = Clock::now();
        if (!handle_frame(frame, &got_first_roster)) return;
      }
      // The ack deadline is a POST-JOIN liveness check: before the first
      // roster, server silence is normal (the coordinator is waiting for
      // the remaining ranks to announce, bounded by its own startup gate,
      // whose failure closes every connection — which we observe as EOF).
      if (got_first_roster &&
          Clock::now() - last_server_frame >=
              std::chrono::milliseconds(opt.ack_deadline_ms)) {
        set_state(RosterClientState::kCoordinatorLost);
        conn.close();
        DGPP_LOG_WARN("roster: no coordinator frames within {} ms",
                      opt.ack_deadline_ms);
        return;
      }
      if (Clock::now() - last_beat_sent >=
          std::chrono::milliseconds(opt.heartbeat_interval_ms)) {
        uint64_t epoch;
        {
          std::lock_guard<std::mutex> lock(mu);
          epoch = snapshot.epoch;
        }
        if (!send_frame(conn, RosterFrameType::kHeartbeat, epoch, {})) {
          on_conn_died(got_first_roster);
          return;
        }
        last_beat_sent = Clock::now();
      }
    }
    // Local stop(): nothing is wrong with the coordinator; this process is
    // just shutting down.
    mark_local_stop();
    conn.close();
  }

  // Returns false when the client should stop processing frames.
  bool handle_frame(const ReceivedFrame& frame, bool* got_first_roster) {
    switch (static_cast<RosterFrameType>(frame.type)) {
      case RosterFrameType::kRoster: {
        RosterSnapshot snap;
        if (!decode_roster(frame.payload, frame.epoch, &snap)) {
          fail_join("malformed roster frame at epoch " +
                    std::to_string(frame.epoch));
          conn.close();
          return false;
        }
        bool regressed = false;
        bool first = false;
        {
          std::lock_guard<std::mutex> lock(mu);
          regressed = have_snapshot && snap.epoch < snapshot.epoch;
          first = !have_snapshot;
        }
        if (regressed) {
          fail_join("roster epoch regressed");
          conn.close();
          return false;
        }
        {
          std::lock_guard<std::mutex> lock(mu);
          snapshot = snap;
          if (first) {
            // Freeze the sealed roster join() will return; an eviction that
            // races the caller's wakeup must not rewrite history.
            first_snapshot = snap;
            have_snapshot = true;
            if (!joined_once) {
              joined_once = true;
              state = RosterClientState::kJoined;
            }
          }
          cv.notify_all();
        }
        if (!snap.contains(opt.rank)) {
          set_state(RosterClientState::kEvicted);
          conn.close();
          DGPP_LOG_WARN("roster: rank {} evicted at epoch {}", opt.rank,
                        snap.epoch);
          return false;
        }
        *got_first_roster = true;
        return true;
      }
      case RosterFrameType::kHeartbeatAck:
        return true;  // liveness only; ack deadline handles loss
      case RosterFrameType::kShutdown:
        if (!*got_first_roster) {
          fail_join("coordinator shut down before sealing the roster");
        } else {
          set_state(RosterClientState::kStopped);
          DGPP_LOG_INFO("roster: coordinator shutdown at epoch {}",
                        frame.epoch);
        }
        conn.close();
        return false;
      case RosterFrameType::kReject: {
        std::string reason;
        if (!decode_reason(frame.payload, &reason)) {
          reason = "rejected (no reason given)";
        }
        fail_join("coordinator rejected this rank: " + reason);
        conn.close();
        return false;
      }
      default:
        fail_join("unexpected frame type " + std::to_string(frame.type));
        conn.close();
        return false;
    }
  }

  // EOF with no protocol event: if the last roster included this rank, the
  // coordinator vanished mid-batch; if not, the close after the final
  // roster is the eviction itself.
  void on_conn_died(bool got_first_roster) {
    if (!got_first_roster) {
      fail_join("coordinator connection lost before the first roster");
    } else {
      std::lock_guard<std::mutex> lock(mu);
      state = snapshot.contains(opt.rank) ? RosterClientState::kCoordinatorLost
                                          : RosterClientState::kEvicted;
      cv.notify_all();
    }
    conn.close();
  }
};

RosterClient::RosterClient(RosterClientOptions options)
    : impl_(new Impl(std::move(options))) {}

RosterClient::~RosterClient() {
  stop();
  delete impl_;
}

void RosterClient::start() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->launched) {
      throw std::runtime_error("roster client: start() already called");
    }
    if (impl_->opt.rank < 0 || impl_->opt.node.empty() ||
        impl_->opt.coordinator_host.empty() ||
        impl_->opt.coordinator_port == 0) {
      throw std::runtime_error(
          "roster client: rank, node, coordinator host, and port are required");
    }
    impl_->launched = true;
  }
  impl_->worker = std::thread([this] { impl_->run(); });
}

RosterSnapshot RosterClient::join() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (!impl_->launched) {
      throw std::runtime_error(
          "roster client: start() must be called before join()");
    }
  }

  std::unique_lock<std::mutex> lock(impl_->mu);
  const bool done = impl_->cv.wait_for(
      lock, std::chrono::milliseconds(impl_->opt.join_timeout_ms + 1000),
      [this] {
        return impl_->have_snapshot ||
               impl_->state != RosterClientState::kJoining;
      });
  if (!done || !impl_->have_snapshot) {
    lock.unlock();
    stop();
    std::string reason = impl_->join_error;
    if (reason.empty()) {
      reason = "no sealed roster within " +
               std::to_string(impl_->opt.join_timeout_ms) + " ms";
    }
    throw std::runtime_error("roster join failed: " + reason);
  }
  // Defensive: the coordinator must never seal a roster that excludes the
  // rank it just accepted.
  if (!impl_->first_snapshot.contains(impl_->opt.rank)) {
    lock.unlock();
    stop();
    throw std::runtime_error(
        "roster join failed: sealed roster does not contain this rank");
  }
  return impl_->first_snapshot;
}

RosterSnapshot RosterClient::current() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->snapshot;
}

RosterClientState RosterClient::state() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->state;
}

bool RosterClient::wait_for_epoch(uint64_t want, int timeout_ms) {
  std::unique_lock<std::mutex> lock(impl_->mu);
  if (!impl_->launched) {
    throw std::runtime_error(
        "roster client: start() must be called before wait_for_epoch()");
  }
  // Timeout and terminal states converge on the same question below: did the
  // epoch actually arrive? A graceful stop short of the wanted epoch simply
  // returns false; eviction and coordinator loss are the throwing failures.
  impl_->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, want] {
    return impl_->snapshot.epoch >= want ||
           impl_->state == RosterClientState::kEvicted ||
           impl_->state == RosterClientState::kCoordinatorLost ||
           impl_->state == RosterClientState::kStopped;
  });
  if (impl_->state == RosterClientState::kEvicted ||
      impl_->state == RosterClientState::kCoordinatorLost) {
    throw std::runtime_error("roster: rank " + std::to_string(impl_->opt.rank) +
                             " is " + roster_client_state_name(impl_->state) +
                             " (epoch " + std::to_string(impl_->snapshot.epoch) +
                             ")");
  }
  return impl_->snapshot.epoch >= want;
}

void RosterClient::stop() {
  impl_->running.store(false);
  if (impl_->worker.joinable()) {
    impl_->worker.join();
  }
}

}  // namespace dgpp::net
