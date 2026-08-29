// CollectiveBus implementation (DESIGN §6/§6.1). The engine thread owns
// every verbs object and all slot/credit state; other threads only submit,
// wait, and stop. One loop iteration is bounded and ordered: latency
// intake, bulk intake, CQ drains, receive recycling, credit harvest,
// watchdogs — never an unbounded pass, never a syscall on the hot path.

#include "net/collective_bus.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/log.hpp"
#include "net/bus_types.hpp"
#include "net/tcp.hpp"
#include "net/verbs.hpp"

namespace dgpp::net {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxIntakeLatency = 64;    // latency requests per iteration
constexpr int kMaxIntakeBulkPosts = 16;  // bulk stripes per iteration
constexpr int kMaxPollPerCq = 32;        // CQEs drained per CQ per iteration
constexpr int kEngineSpinIterations = 200;
constexpr int kEngineIdleSleepUs = 50;
constexpr size_t kMaxLatencySamples = 100000;
constexpr int kConnectRetryMs = 500;
constexpr size_t kMaxErrorText = 4096;

double elapsed_us(Clock::time_point from) {
  return std::chrono::duration<double, std::micro>(Clock::now() - from)
      .count();
}

// Pinned-flag acquire load (NIC- and GPU-written cells; DESIGN §2.3/§6).
inline uint32_t acquire_u32(const volatile uint32_t* p) {
  return __atomic_load_n(const_cast<const uint32_t*>(p), __ATOMIC_ACQUIRE);
}

inline uint64_t acquire_u64(const volatile uint64_t* p) {
  return __atomic_load_n(const_cast<const uint64_t*>(p), __ATOMIC_ACQUIRE);
}

}  // namespace

struct BusStripe {
  int lane = 0;
  BusPool pool = BusPool::kLatency;
  uint32_t slot = 0;
  uint32_t seq = 0;
};

struct BusRequest {
  uint64_t id = 0;
  int peer_rank = -1;
  BusMessageClass cls = BusMessageClass::kLatency;
  const void* data = nullptr;  // caller-owned until completion
  size_t len = 0;
  size_t offset = 0;  // bytes submitted as stripes so far

  int outstanding = 0;  // stripes credited back so far are subtracted
  std::vector<BusStripe> stripes;
  std::vector<uint64_t> stripe_hashes;  // aligned with stripes, filled on credit
  Clock::time_point submitted;

  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  bool ok = false;
  std::string error;
};

struct CollectiveBus::Impl {
  struct SendSlot {
    uint32_t gen = 0;          // last generation posted into this slot
    uint32_t credit_seen = 0;  // last credit seq harvested from the cell
    bool in_flight = false;
    // Owning reference: a watchdog-failed request may still receive its
    // late credit (or a lane failure) after the waiter reaped it, so the
    // slot keeps the request alive until it is cleared.
    std::shared_ptr<BusRequest> owner;
    size_t owner_stripe = 0;
  };
  struct RecvSlot {
    uint32_t expect_seq = 1;  // next arrival generation
    bool arrived = false;     // doorbell CQE seen, GPU ack pending
    size_t len = 0;
  };
  struct LaneState {
    std::unique_ptr<RcLane> lane;
    BusLaneEndpoint endpoint;      // ours, for the exchange table
    BusLaneEndpoint peer_endpoint;  // the peer's, learned from the table
    std::vector<SendSlot> send[2];  // [latency, bulk]
    std::vector<RecvSlot> recv[2];
    // Ring cursor per pool: plain SENDs are consumed FIFO by the peer's
    // receive ring, so the i-th message must target ring slot i mod depth —
    // a free-slot-first-fit would break the ring alignment silently.
    uint32_t cursor[2] = {0, 0};
    // The receive side recycles in ring order too: reposting slot r+1
    // before r would reorder the RQ and misalign the sender's cursor.
    uint32_t recycle_cursor[2] = {0, 0};
    size_t in_flight_count = 0;
    bool failed = false;
    Clock::time_point last_progress;
    BusLaneStats stats{};
  };

  BusOptions opt;
  BusSlabLayout layout{};
  std::vector<std::unique_ptr<VerbsDevice>> devices;  // deduped by name
  std::vector<int> peer_ranks;                       // ascending
  std::vector<std::vector<LaneState>> peers;         // [peer][lane]

  std::mutex lat_q_mu;
  std::deque<std::shared_ptr<BusRequest>> lat_q;
  std::mutex bulk_q_mu;
  std::deque<std::shared_ptr<BusRequest>> bulk_q;

  std::mutex registry_mu;
  std::unordered_map<uint64_t, std::shared_ptr<BusRequest>> registry;
  uint64_t next_id = 1;

  std::atomic<bool> stopping{false};
  std::mutex stop_mu;  // serializes quiesce()/stop() for one instance
  bool quiesced = false;
  bool stopped = false;
  std::thread engine;

  // One stream per consumer kernel: a persistent kernel starves every
  // launch queued behind it on the same stream, so the lane-1 consumer
  // must not share the lane-0 consumer's stream.
  std::vector<cudaStream_t> consumer_streams;
  bool consumers_launched = false;
  std::mutex stats_mu;
  BusStats stats_store;

  BusRankExchange ex_{};  // our frame, built once during start()

  // ---- helpers -----------------------------------------------------------

  size_t peer_index(int rank) const {
    return static_cast<size_t>(
        std::find(peer_ranks.begin(), peer_ranks.end(), rank) -
        peer_ranks.begin());
  }

  static int pool_index(BusPool pool) {
    return pool == BusPool::kLatency ? 0 : 1;
  }

  size_t pool_slots(BusPool pool) const {
    return pool == BusPool::kLatency ? static_cast<size_t>(opt.lat_slots)
                                     : static_cast<size_t>(opt.bulk_slots);
  }

  size_t lane_count() const { return opt.lane_devices.size(); }

  void record_latency(BusMessageClass cls, double us) {
    std::lock_guard<std::mutex> lock(stats_mu);
    BusClassStats& s = cls == BusMessageClass::kLatency ? stats_store.latency
                                                        : stats_store.bulk;
    if (s.latency_us.size() < kMaxLatencySamples) s.latency_us.push_back(us);
  }

  void complete_request(BusRequest* req, bool ok, const std::string& error) {
    {
      std::lock_guard<std::mutex> lock(req->mu);
      if (req->done) return;
      req->ok = ok;
      req->error = error;
      req->done = true;
    }
    record_latency(req->cls, elapsed_us(req->submitted));
    req->cv.notify_all();
  }

  void fail_lane(LaneState& lane, const std::string& reason) {
    if (lane.failed) return;
    lane.failed = true;
    lane.stats.failed = true;
    const std::string text =
        "bus lane to rank " + std::to_string(lane.stats.peer_rank) +
        " lane " + std::to_string(lane.stats.lane) + " failed: " + reason;
    DGPP_LOG_ERROR("{}", text);
    for (int pool = 0; pool < 2; ++pool) {
      for (SendSlot& s : lane.send[pool]) {
        if (s.in_flight && s.owner) {
          complete_request(s.owner.get(), false, text);
          s.owner = nullptr;
          s.in_flight = false;
        }
      }
    }
    lane.in_flight_count = 0;
  }

  // ---- engine passes ------------------------------------------------------

  bool submit_stripe(const std::shared_ptr<BusRequest>& req, LaneState& lane,
                     BusPool pool, uint32_t slot, size_t chunk,
                     std::string* error) {
    RcLane& rc = *lane.lane;
    const int p = pool_index(pool);
    SendSlot& ss = lane.send[p][slot];

    uint8_t* dst = rc.layout().send_payload(rc.slab(), pool, slot);
    std::memcpy(dst, static_cast<const uint8_t*>(req->data) + req->offset,
                chunk);

    const uint32_t seq = ss.gen + 1;
    if (!rc.post_send_pair(pool, slot, seq, static_cast<uint32_t>(chunk),
                           error))
      return false;
    ss.gen = seq;
    ss.in_flight = true;
    ss.owner = req;
    ss.owner_stripe = req->stripes.size();
    ++lane.in_flight_count;
    DGPP_LOG_DEBUG("stripe: peer={} lane={} pool={} slot={} seq={} chunk={}",
                   req->peer_rank, lane.stats.lane, pool_index(pool), slot, seq,
                   chunk);

    req->stripes.push_back(BusStripe{lane.stats.lane, pool, slot, seq});
    req->offset += chunk;
    ++req->outstanding;
    ++lane.stats.posts;
    lane.stats.bytes_sent += chunk;
    return true;
  }

  // Latency intake: one stripe per message, first lane with a free latency
  // slot. Bulk intake: contiguous slot-byte chunks striped round-robin,
  // deterministic lane order so verification can recompute the chunking.
  bool intake() {
    bool worked = false;

    {
      std::unique_lock<std::mutex> lock(lat_q_mu);
      int taken = 0;
      while (!lat_q.empty() && taken < kMaxIntakeLatency) {
        std::shared_ptr<BusRequest> req = lat_q.front();
        if (req->done) {
          lat_q.pop_front();
          ++taken;
          continue;
        }
        const size_t p = peer_index(req->peer_rank);
        bool placed = false;
        std::string hard_error;
        for (LaneState& lane : peers[p]) {
          if (lane.failed) continue;
          const uint32_t slot = lane.cursor[0];
          if (lane.send[0][slot].in_flight) continue;  // ring position busy
          std::string error;
          if (submit_stripe(req, lane, BusPool::kLatency, slot, req->len,
                            &error)) {
            lane.cursor[0] =
                (slot + 1) % static_cast<uint32_t>(opt.lat_slots);
            placed = true;
          } else {
            fail_lane(lane, error);
            hard_error = error;
          }
          break;
        }
        if (placed) {
          req->stripe_hashes.resize(req->stripes.size());
          lat_q.pop_front();
          ++taken;
        } else if (!hard_error.empty()) {
          complete_request(
              req.get(), false,
              "no alive lane to rank " + std::to_string(req->peer_rank));
          lat_q.pop_front();
          ++taken;
        } else {
          break;  // every latency slot busy; retry next iteration
        }
        worked = true;
      }
    }

    {
      std::unique_lock<std::mutex> lock(bulk_q_mu);
      if (!bulk_q.empty()) {
        std::shared_ptr<BusRequest> req = bulk_q.front();
        if (req->done) {
          bulk_q.pop_front();
          worked = true;
        } else {
          const size_t p = peer_index(req->peer_rank);
          const size_t lanes = peers[p].size();
          int posts = 0;
          bool request_done = false;
          while (req->offset < req->len && posts < kMaxIntakeBulkPosts) {
            const size_t stripe_idx = req->stripes.size();
            LaneState& lane = peers[p][stripe_idx % lanes];
            if (lane.failed) {
              complete_request(
                  req.get(), false,
                  "bulk stripe lane " +
                      std::to_string(stripe_idx % lanes) + " to rank " +
                      std::to_string(req->peer_rank) +
                      " is down (no per-lane failover by design)");
              request_done = true;
              break;
            }
            // Ring discipline within the bulk pool: cursor order, gated by
            // the slot still being in flight (the credit for its previous
            // generation).
            const uint32_t slot = lane.cursor[1];
            if (lane.send[1][slot].in_flight) break;  // ring full; defer
            const size_t chunk = std::min(
                static_cast<size_t>(opt.bulk_slot_bytes), req->len - req->offset);
            std::string error;
            if (!submit_stripe(req, lane, BusPool::kBulk, slot, chunk,
                               &error)) {
              fail_lane(lane, error);
              complete_request(req.get(), false, error);
              request_done = true;
              break;
            }
            lane.cursor[1] =
                (slot + 1) % static_cast<uint32_t>(opt.bulk_slots);
            req->stripe_hashes.resize(req->stripes.size());
            ++posts;
            worked = true;
          }
          if (request_done || req->offset == req->len) bulk_q.pop_front();
          // else: partially submitted; stays at the front.
        }
      }
    }
    return worked;
  }

  bool poll_cqs() {
    bool worked = false;
    ibv_wc wcs[kMaxPollPerCq];
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          int n = lane.lane->poll_tx(pool, wcs, kMaxPollPerCq);
          if (n < 0) {
            fail_lane(lane, "tx poll failed");
            continue;
          }
          for (int i = 0; i < n; ++i) {
            if (wcs[i].status != IBV_WC_SUCCESS) {
              fail_lane(lane, bus_wc_error(wcs[i]));
              break;
            }
          }
          if (n > 0) {
            lane.last_progress = Clock::now();
            worked = true;
          }
          if (lane.failed) continue;

          n = lane.lane->poll_rx(pool, wcs, kMaxPollPerCq);
          if (n < 0) {
            fail_lane(lane, "rx poll failed");
            continue;
          }
          for (int i = 0; i < n; ++i) {
            const ibv_wc& wc = wcs[i];
            if (wc.status != IBV_WC_SUCCESS) {
              fail_lane(lane, bus_wc_error(wc));
              break;
            }
            if (bus_wr_kind(wc.wr_id) != BusWr::kDoorbell) continue;
            const uint32_t slot = bus_wr_slot(wc.wr_id);
            RecvSlot& rs = lane.recv[pool_i][slot];
            // RC ordering: the payload CQE preceded this doorbell CQE on
            // the same CQ, so this means the whole message landed.
            StartSlot* cell = lane.lane->layout().recv_doorbell(
                lane.lane->slab(), pool, slot);
            const uint32_t seq = acquire_u32(&cell->seq);
            const uint32_t len = acquire_u32(&cell->len);
            DGPP_LOG_DEBUG(
                "doorbell CE: peer={} lane={} pool={} slot={} cell_seq={} "
                "expect={} len={}",
                lane.stats.peer_rank, lane.stats.lane, pool_i, slot, seq,
                rs.expect_seq, len);
            if (rs.arrived) {
              fail_lane(lane, "duplicate doorbell arrival (protocol)");
              break;
            }
            if (seq != rs.expect_seq) {
              fail_lane(lane, "doorbell seq " + std::to_string(seq) +
                                  " != expected " +
                                  std::to_string(rs.expect_seq));
              break;
            }
            rs.arrived = true;
            rs.len = len;
            ++lane.stats.doorbell_recvs;
            lane.stats.bytes_recv += len;
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  // Receive recycling, in ring order: a slot's pair is reposted (and its
  // credit granted) only after both the GPU ack and the receive
  // completions are known — §6 step 4 — and only when every earlier ring
  // position of the current wrap has already been recycled (a repost out
  // of ring order would reorder the RQ and misalign the sender's cursor).
  bool recycle_pass() {
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        RcLane& rc = *lane.lane;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool = pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          const uint32_t depth = static_cast<uint32_t>(pool_slots(pool));
          for (;;) {
            const uint32_t slot = lane.recycle_cursor[pool_i];
            RecvSlot& rs = lane.recv[pool_i][slot];
            if (!rs.arrived) break;
            FlagAck* ack = rc.layout().ack_cell(rc.slab(), pool, slot);
            if (acquire_u32(&ack->seq) != rs.expect_seq) break;
            const uint64_t hash = acquire_u64(&ack->hash);
            DGPP_LOG_DEBUG(
                "recycle: peer={} lane={} pool={} slot={} seq={} ack_seq={}",
                lane.stats.peer_rank, lane.stats.lane, pool_index(pool), slot,
                rs.expect_seq, ack->seq);
            std::string error;
            if (!rc.post_recv_pair(pool, slot, &error) ||
                !rc.post_credit_write(pool, slot, rs.expect_seq,
                                      lane.peer_endpoint, hash, &error)) {
              fail_lane(lane, error);
              break;
            }
            ++lane.stats.credits_returned;
            rs.arrived = false;
            rs.expect_seq += 1;
            lane.recycle_cursor[pool_i] = (slot + 1) % depth;
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  // Credit harvest: the peer's completion cells, read as pinned flags.
  bool credits_pass() {
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        RcLane& rc = *lane.lane;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool = pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          for (uint32_t slot = 0;
               slot < static_cast<uint32_t>(pool_slots(pool)) && !lane.failed;
               ++slot) {
            SendSlot& ss = lane.send[pool_i][slot];
            if (!ss.in_flight) continue;
            BusCredit* cell = reinterpret_cast<BusCredit*>(
                rc.layout().completion_cell(rc.slab(), pool, slot));
            const uint32_t seq = acquire_u32(&cell->seq);
            if (seq == 0 || seq == ss.credit_seen) continue;
            if (seq > ss.gen) {
              fail_lane(lane, "credit for unsent generation " +
                                  std::to_string(seq));
              break;
            }
            const uint32_t cslot = acquire_u32(&cell->slot);
            const uint64_t hash = acquire_u64(&cell->hash);
            if (cslot != slot) {
              fail_lane(lane, "credit cell slot mismatch");
              break;
            }
            ss.credit_seen = seq;
            ++lane.stats.credits_received;
            if (seq == ss.gen && ss.owner) {
              std::shared_ptr<BusRequest> req = ss.owner;
              req->stripe_hashes[ss.owner_stripe] = hash;
              ss.owner = nullptr;
              ss.in_flight = false;
              if (lane.in_flight_count > 0) --lane.in_flight_count;
              if (--req->outstanding == 0)
                complete_request(req.get(), true, {});
            }
            // seq < ss.gen: stale credit for a request already failed by
            // the watchdog and freed — nothing to do, the cell is current.
            lane.last_progress = Clock::now();
            worked = true;
          }
        }
      }
    }
    return worked;
  }

  bool watchdog_pass() {
    const auto now = Clock::now();
    bool worked = false;
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed || lane.in_flight_count == 0) continue;
        if (now - lane.last_progress >
            std::chrono::milliseconds(opt.completion_timeout_ms)) {
          fail_lane(lane, "inactivity watchdog (no progress for " +
                              std::to_string(opt.completion_timeout_ms) +
                              " ms)");
          worked = true;
        }
      }
    }
    std::vector<BusRequest*> expired;
    {
      std::lock_guard<std::mutex> lock(registry_mu);
      for (auto& entry : registry) {
        BusRequest* req = entry.second.get();
        if (!req->done &&
            now - req->submitted >
                std::chrono::milliseconds(opt.completion_timeout_ms))
          expired.push_back(req);
      }
    }
    for (BusRequest* req : expired) {
      complete_request(req, false,
                       "completion timeout (" +
                           std::to_string(opt.completion_timeout_ms) + " ms)");
      worked = true;
    }
    return worked;
  }

  size_t total_outstanding() {
    std::lock_guard<std::mutex> lock(registry_mu);
    size_t total = 0;
    for (auto& entry : registry) {
      const BusRequest* req = entry.second.get();
      if (req->done) continue;
      total += req->outstanding;
      if (req->offset < req->len) ++total;  // still queued or partial
    }
    return total;
  }

  bool drained() {
    std::lock_guard<std::mutex> lq(lat_q_mu);
    if (!lat_q.empty()) return false;
    std::lock_guard<std::mutex> bq(bulk_q_mu);
    if (!bulk_q.empty()) return false;
    return total_outstanding() == 0;
  }

  void engine_loop() {
    int idle = 0;
    for (;;) {
      if (stopping.load(std::memory_order_relaxed) && drained()) break;
      const bool worked = intake();
      const bool polled = poll_cqs();
      const bool recycled = recycle_pass();
      const bool credited = credits_pass();
      const bool watched = watchdog_pass();
      if (worked || polled || recycled || credited || watched) {
        idle = 0;
      } else if (++idle > kEngineSpinIterations) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(kEngineIdleSleepUs));
      } else {
        std::this_thread::yield();
      }
    }
  }

  // ---- rendezvous ----------------------------------------------------------

  BusRankExchange build_exchange() const {
    BusRankExchange ex;
    ex.rank = opt.my_rank;
    ex.lane_count = static_cast<uint8_t>(opt.lane_devices.size());
    ex.peer_count = static_cast<uint8_t>(peer_ranks.size());
    ex.lat_slots = static_cast<uint32_t>(opt.lat_slots);
    ex.lat_slot_bytes = static_cast<uint32_t>(opt.lat_slot_bytes);
    ex.bulk_slots = static_cast<uint32_t>(opt.bulk_slots);
    ex.bulk_slot_bytes = static_cast<uint32_t>(opt.bulk_slot_bytes);
    for (size_t p = 0; p < peer_ranks.size(); ++p)
      for (size_t l = 0; l < peers[p].size(); ++l)
        ex.peers[p][l] = peers[p][l].endpoint;
    return ex;
  }

  static bool send_error_frame(TcpConn& conn, const std::string& text) {
    const std::string bounded = text.substr(0, kMaxErrorText);
    uint8_t head[8] = {};
    uint32_t magic = kBusErrorMagic;
    uint32_t len = static_cast<uint32_t>(bounded.size());
    std::memcpy(head, &magic, 4);
    std::memcpy(head + 4, &len, 4);
    if (!conn.write_all(head, 8)) return false;
    return bounded.empty() ||
           conn.write_all(bounded.data(), bounded.size());
  }

  static std::string read_error_frame(TcpConn& conn) {
    uint8_t len_buf[4] = {};
    if (!conn.read_exact(len_buf, 4))
      return "peer closed during error frame";
    uint32_t len = 0;
    std::memcpy(&len, len_buf, 4);
    if (len > kMaxErrorText) return "peer sent oversized error frame";
    std::string text(len, '\0');
    if (len > 0 && !conn.read_exact(text.data(), len))
      return "peer closed during error frame";
    return text;
  }

  size_t exchange_frame_bytes() const {
    return kBusExchangeHeaderBytes +
           peer_ranks.size() * lane_count() * kBusExchangeLaneBytes;
  }

  // Pairwise exchange with one connection. The caller sets io deadlines.
  bool handshake(TcpConn& conn, BusRankExchange* peer_out,
                 std::string* error) {
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> mine(kBusMaxExchangeBytes);
    const size_t mine_len =
        bus_rank_exchange_encode(ex_, mine.data(), mine.size());
    if (mine_len != frame_len) {
      *error = "internal: exchange encode size mismatch";
      return false;
    }
    if (!conn.write_all(mine.data(), mine_len)) {
      *error = "rendezvous: writing exchange frame failed";
      return false;
    }

    std::vector<uint8_t> peer_buf(kBusMaxExchangeBytes);
    if (!conn.read_exact(peer_buf.data(), 4)) {
      *error = "rendezvous: peer closed before exchange";
      return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, peer_buf.data(), 4);
    if (magic == kBusErrorMagic) {
      *error = "rendezvous rejected by peer: " + read_error_frame(conn);
      return false;
    }
    if (!conn.read_exact(peer_buf.data() + 4, frame_len - 4)) {
      *error = "rendezvous: peer exchange short read";
      return false;
    }
    if (!bus_rank_exchange_decode(peer_buf.data(), frame_len, peer_out)) {
      send_error_frame(conn, "exchange frame decode failed");
      *error = "rendezvous: peer sent a malformed exchange frame";
      return false;
    }
    if (peer_out->lane_count != ex_.lane_count ||
        peer_out->peer_count != ex_.peer_count) {
      send_error_frame(conn, "lane/peer count mismatch: mine lanes=" +
                                 std::to_string(ex_.lane_count) + " peers=" +
                                 std::to_string(ex_.peer_count));
      *error = "rendezvous: lane/peer count mismatch with rank " +
               std::to_string(peer_out->rank);
      return false;
    }
    if (peer_out->lat_slots != ex_.lat_slots ||
        peer_out->lat_slot_bytes != ex_.lat_slot_bytes ||
        peer_out->bulk_slots != ex_.bulk_slots ||
        peer_out->bulk_slot_bytes != ex_.bulk_slot_bytes) {
      send_error_frame(
          conn, "slot geometry mismatch: mine lat=" +
                    std::to_string(ex_.lat_slots) + "x" +
                    std::to_string(ex_.lat_slot_bytes) + "B bulk=" +
                    std::to_string(ex_.bulk_slots) + "x" +
                    std::to_string(ex_.bulk_slot_bytes) + "B");
      *error = "rendezvous: slot geometry mismatch with rank " +
               std::to_string(peer_out->rank);
      return false;
    }
    return true;
  }

  std::vector<uint8_t> encode_table(
      const std::vector<BusRankExchange>& table) const {
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> wire(8 + table.size() * frame_len);
    uint32_t magic = kBusTableMagic;
    std::memcpy(wire.data(), &magic, 4);
    wire[4] = kBusExchangeVersion;
    wire[5] = static_cast<uint8_t>(table.size());
    std::memset(wire.data() + 6, 0, 2);
    for (size_t r = 0; r < table.size(); ++r)
      bus_rank_exchange_encode(table[r], wire.data() + 8 + r * frame_len,
                               frame_len);
    return wire;
  }

  bool rendezvous_listen(std::string* error) {
    TcpListener listener;
    try {
      listener = TcpListener::bind(opt.rendezvous_port);
    } catch (const std::exception& e) {
      *error = std::string("rendezvous listen: ") + e.what();
      return false;
    }
    DGPP_LOG_INFO("bus: rank 0 rendezvous listening on port {}",
                  listener.port());

    std::vector<TcpConn> conns;
    std::vector<BusRankExchange> frames;
    for (size_t i = 0; i < peer_ranks.size(); ++i) {
      TcpConn conn = listener.accept(opt.rendezvous_timeout_ms);
      if (!conn.valid()) {
        *error = "rendezvous: accept timed out (" +
                 std::to_string(opt.rendezvous_timeout_ms) + " ms) with " +
                 std::to_string(i) + " of " +
                 std::to_string(peer_ranks.size()) + " peers connected";
        return false;
      }
      conn.set_io_deadline_ms(opt.rendezvous_timeout_ms);
      BusRankExchange peer;
      if (!handshake(conn, &peer, error)) return false;
      conns.push_back(std::move(conn));
      frames.push_back(peer);
    }

    // The connected ranks must be exactly 1..world-1.
    std::vector<int> seen;
    for (const BusRankExchange& f : frames) seen.push_back(f.rank);
    std::sort(seen.begin(), seen.end());
    for (size_t i = 0; i < seen.size(); ++i) {
      if (seen[i] != static_cast<int>(i + 1)) {
        *error = "rendezvous: unexpected rank set (rank " +
                 std::to_string(seen[i]) + " at position " +
                 std::to_string(i) + ")";
        return false;
      }
    }

    std::vector<BusRankExchange> table(static_cast<size_t>(opt.world_size));
    table[0] = ex_;
    for (const BusRankExchange& f : frames)
      table[static_cast<size_t>(f.rank)] = f;

    const std::vector<uint8_t> wire = encode_table(table);
    for (TcpConn& conn : conns) {
      if (!conn.write_all(wire.data(), wire.size())) {
        *error = "rendezvous: table broadcast write failed";
        return false;
      }
    }
    install_table(table);
    return true;
  }

  bool rendezvous_connect(std::string* error) {
    TcpConn conn;
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(opt.rendezvous_timeout_ms);
    for (;;) {
      try {
        conn = TcpConn::connect(opt.rendezvous_host, opt.rendezvous_port,
                                opt.rendezvous_timeout_ms);
        break;
      } catch (const std::exception& e) {
        if (Clock::now() + std::chrono::milliseconds(kConnectRetryMs) >
            deadline) {
          *error = std::string("rendezvous connect: ") + e.what();
          return false;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kConnectRetryMs));
      }
    }
    conn.set_io_deadline_ms(opt.rendezvous_timeout_ms);

    BusRankExchange rank0;
    if (!handshake(conn, &rank0, error)) return false;

    std::vector<uint8_t> head(8);
    if (!conn.read_exact(head.data(), 8)) {
      *error = "rendezvous: table read failed";
      return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, head.data(), 4);
    if (magic == kBusErrorMagic) {
      *error = "rendezvous rejected by rank 0: " + read_error_frame(conn);
      return false;
    }
    if (magic != kBusTableMagic || head[4] != kBusExchangeVersion) {
      *error = "rendezvous: malformed table header";
      return false;
    }
    const size_t world = head[5];
    if (world != static_cast<size_t>(opt.world_size)) {
      *error = "rendezvous: table world " + std::to_string(world) +
               " != configured " + std::to_string(opt.world_size);
      return false;
    }
    const size_t frame_len = exchange_frame_bytes();
    std::vector<uint8_t> wire(world * frame_len);
    if (!conn.read_exact(wire.data(), wire.size())) {
      *error = "rendezvous: table frames short read";
      return false;
    }
    std::vector<BusRankExchange> table(world);
    for (size_t r = 0; r < world; ++r) {
      if (!bus_rank_exchange_decode(wire.data() + r * frame_len, frame_len,
                                    &table[r])) {
        *error = "rendezvous: table frame decode failed";
        return false;
      }
    }
    install_table(table);
    return true;
  }

  // Connect every pool's QP from the distributed table, then post the
  // initial receive pairs — the full initial credit grant.
  void install_table(const std::vector<BusRankExchange>& table) {
    for (size_t p = 0; p < peer_ranks.size(); ++p) {
      const int peer_rank = peer_ranks[p];
      const BusRankExchange& peer_frame =
          table[static_cast<size_t>(peer_rank)];
      const size_t my_slot = bus_peer_index(opt.my_rank, peer_rank);
      for (size_t l = 0; l < peers[p].size(); ++l) {
        peers[p][l].peer_endpoint = peer_frame.peers[my_slot][l];
        DGPP_LOG_DEBUG(
            "bus: rank {} lane {} <- peer {} table qpns lat={} bulk={}",
            opt.my_rank, l, peer_rank, peer_frame.peers[my_slot][l].qpn_lat,
            peer_frame.peers[my_slot][l].qpn_bulk);
        for (int pool_i = 0; pool_i < 2; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          std::string error;
          if (!peers[p][l].lane->connect(pool, peer_frame.peers[my_slot][l],
                                          &error))
            fail_lane(peers[p][l], error);
        }
      }
    }
    for (auto& peer_lanes : peers) {
      for (LaneState& lane : peer_lanes) {
        if (lane.failed) continue;
        std::string error;
        for (int pool_i = 0; pool_i < 2 && !lane.failed; ++pool_i) {
          const BusPool pool =
              pool_i == 0 ? BusPool::kLatency : BusPool::kBulk;
          for (uint32_t s = 0;
               s < static_cast<uint32_t>(pool_slots(pool)) && !lane.failed;
               ++s) {
            if (!lane.lane->post_recv_pair(pool, s, &error))
              fail_lane(lane, error);
          }
        }
        lane.last_progress = Clock::now();
      }
    }
  }
};

CollectiveBus::CollectiveBus(BusOptions options)
    : impl_(std::make_unique<Impl>()), options_(std::move(options)) {
  impl_->opt = options_;
}

CollectiveBus::~CollectiveBus() { stop(); }

bool CollectiveBus::start(std::string* error) {
  Impl& impl = *impl_;
  const BusOptions& opt = options_;

  if (opt.world_size < 2 || opt.world_size > 4) {
    *error = "world_size must be 2..4";
    return false;
  }
  if (opt.my_rank < 0 || opt.my_rank >= opt.world_size) {
    *error = "my_rank out of range";
    return false;
  }
  if (opt.lane_devices.empty() || opt.lane_devices.size() > kBusMaxLanes) {
    *error = "1..2 lane devices required";
    return false;
  }
  if (opt.my_rank != 0 && opt.rendezvous_host.empty()) {
    *error = "rendezvous_host is required for non-zero ranks";
    return false;
  }
  if (static_cast<size_t>(opt.lat_slots) + static_cast<size_t>(opt.bulk_slots) >
          kBusMaxConsumerCells ||
      !bus_slab_layout(static_cast<size_t>(opt.lat_slots), opt.lat_slot_bytes,
                       static_cast<size_t>(opt.bulk_slots),
                       opt.bulk_slot_bytes, &impl.layout)) {
    *error = "invalid slot geometry (slot bytes must be 64-multiples >= 64; "
             "combined slots must fit the consumer)";
    return false;
  }

  // One device per distinct lane name so both lanes of a pair share nothing
  // they don't have to (the two active f0s are distinct PCI functions).
  for (const std::string& name : opt.lane_devices) {
    bool have = false;
    for (auto& d : impl.devices)
      if (d->name() == name) have = true;
    if (have) continue;
    std::string open_error;
    auto device = std::make_unique<VerbsDevice>(name, &open_error);
    if (!device->ok()) {
      *error = "bus device open failed: " + open_error;
      return false;
    }
    impl.devices.push_back(std::move(device));
  }

  impl.peer_ranks.clear();
  for (int r = 0; r < opt.world_size; ++r)
    if (r != opt.my_rank) impl.peer_ranks.push_back(r);

  impl.peers.resize(impl.peer_ranks.size());
  for (auto& peer_lanes : impl.peers)
    peer_lanes.resize(opt.lane_devices.size());
  for (size_t p = 0; p < impl.peer_ranks.size(); ++p) {
    for (size_t l = 0; l < opt.lane_devices.size(); ++l) {
      Impl::LaneState& state = impl.peers[p][l];
      VerbsDevice* dev = nullptr;
      for (auto& d : impl.devices)
        if (d->name() == opt.lane_devices[l]) dev = d.get();
      std::string lane_error;
      state.lane = std::make_unique<RcLane>(*dev, impl.layout, opt.qp_depth,
                                            &lane_error);
      if (!state.lane->ok()) {
        *error = "bus lane create failed (rank " +
                 std::to_string(impl.peer_ranks[p]) + " lane " +
                 std::to_string(l) + "): " + lane_error;
        return false;
      }
      state.endpoint = state.lane->endpoint();
      DGPP_LOG_DEBUG("bus: rank {} peer {} lane {} my qpns lat={} bulk={}",
                     opt.my_rank, impl.peer_ranks[p], l,
                     state.endpoint.qpn_lat, state.endpoint.qpn_bulk);
      state.stats.peer_rank = impl.peer_ranks[p];
      state.stats.lane = static_cast<int>(l);
      state.send[0].assign(static_cast<size_t>(opt.lat_slots), Impl::SendSlot{});
      state.send[1].assign(static_cast<size_t>(opt.bulk_slots), Impl::SendSlot{});
      state.recv[0].assign(static_cast<size_t>(opt.lat_slots), Impl::RecvSlot{});
      state.recv[1].assign(static_cast<size_t>(opt.bulk_slots), Impl::RecvSlot{});
      state.last_progress = Clock::now();
    }
  }

  impl.ex_ = impl.build_exchange();

  std::string rendezvous_error;
  const bool up = opt.my_rank == 0
                       ? impl.rendezvous_listen(&rendezvous_error)
                       : impl.rendezvous_connect(&rendezvous_error);
  if (!up) {
    *error = rendezvous_error;
    return false;
  }
  for (auto& peer_lanes : impl.peers)
    for (auto& lane : peer_lanes)
      if (lane.failed) {
        *error = "bus bring-up failed on at least one lane (see log)";
        return false;
      }

  // Receive consumers: the Phase 2 harness folds payloads and acks. The
  // engine's real per-collective consumers (deliverable 3) replace these
  // by launching with launch_consumers=false. Each consumer gets its own
  // stream — persistent kernels serialize a stream, and one lane's
  // consumer must never starve another's launch.
  if (opt.launch_consumers) {
    const uint64_t deadline =
        bus_consumer_deadline_cycles(opt.consumer_deadline_s);
    if (deadline == 0) {
      *error = "could not read device clock rate for the consumer watchdog";
      return false;
    }
    for (size_t p = 0; p < impl.peer_ranks.size(); ++p) {
      for (size_t l = 0; l < impl.peers[p].size(); ++l) {
        cudaStream_t stream = nullptr;
        const cudaError_t stream_err =
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (stream_err != cudaSuccess) {
          *error = std::string("consumer stream create failed: ") +
                   cudaGetErrorString(stream_err);
          return false;
        }
        const BusRecvView view =
            recv_view(impl.peer_ranks[p], static_cast<int>(l));
        const cudaError_t launch =
            launch_bus_consumer(view, deadline, stream);
        if (launch != cudaSuccess) {
          cudaStreamDestroy(stream);
          *error = std::string("consumer launch failed: ") +
                   cudaGetErrorString(launch);
          return false;
        }
        impl.consumer_streams.push_back(stream);
      }
    }
    impl.consumers_launched = true;
  }

  impl.engine = std::thread([&impl] { impl.engine_loop(); });
  DGPP_LOG_INFO(
      "bus: rank {} up — {} peer(s) x {} lane(s), lat {}x{}B, bulk {}x{}B",
      opt.my_rank, impl.peer_ranks.size(), opt.lane_devices.size(),
      opt.lat_slots, opt.lat_slot_bytes, opt.bulk_slots, opt.bulk_slot_bytes);
  return true;
}

uint64_t CollectiveBus::send(int peer_rank, const void* data, size_t bytes,
                             BusMessageClass cls, std::string* error) {
  Impl& impl = *impl_;
  if (impl.stopping.load(std::memory_order_relaxed)) {
    *error = "bus is stopped";
    return 0;
  }
  if (bytes == 0 || bytes % 8 != 0) {
    *error = "send size must be a positive multiple of 8";
    return 0;
  }
  if (std::find(impl.peer_ranks.begin(), impl.peer_ranks.end(), peer_rank) ==
      impl.peer_ranks.end()) {
    *error = "rank " + std::to_string(peer_rank) + " is not a bus peer";
    return 0;
  }
  if (cls == BusMessageClass::kLatency && bytes > options_.lat_slot_bytes) {
    *error = "latency message exceeds lat_slot_bytes (" +
             std::to_string(options_.lat_slot_bytes) +
             "B); route it as bulk";
    return 0;
  }

  auto req = std::make_shared<BusRequest>();
  req->peer_rank = peer_rank;
  req->cls = cls;
  req->data = data;
  req->len = bytes;
  req->submitted = Clock::now();
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    req->id = impl.next_id++;
    impl.registry[req->id] = req;
  }
  const uint64_t id = req->id;
  if (cls == BusMessageClass::kLatency) {
    std::lock_guard<std::mutex> lock(impl.lat_q_mu);
    impl.lat_q.push_back(std::move(req));
  } else {
    std::lock_guard<std::mutex> lock(impl.bulk_q_mu);
    impl.bulk_q.push_back(std::move(req));
  }
  return id;
}

BusSendResult CollectiveBus::wait(uint64_t send_id, int timeout_ms) {
  BusSendResult result;
  Impl& impl = *impl_;
  std::shared_ptr<BusRequest> req;
  {
    std::lock_guard<std::mutex> lock(impl.registry_mu);
    auto it = impl.registry.find(send_id);
    if (it == impl.registry.end()) {
      result.error = "unknown or already-waited send id";
      return result;
    }
    req = it->second;
  }
  std::unique_lock<std::mutex> lock(req->mu);
  const bool done =
      req->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                       [&req] { return req->done; });
  if (!done) {
    // Backstop only. The request is NOT removed: it may still be in flight
    // (SendSlot::owner points at it) and the engine watchdog owns its
    // lifecycle from here.
    result.error = "wait backstop timeout (engine watchdog should have "
                   "failed the request first)";
    return result;
  }
  result.ok = req->ok;
  result.error = req->error;
  result.stripe_hashes = std::move(req->stripe_hashes);
  result.elapsed_us = elapsed_us(req->submitted);
  lock.unlock();
  {
    std::lock_guard<std::mutex> rlock(impl.registry_mu);
    // Only erase if still this request (stop() reaps by completing).
    auto it = impl.registry.find(send_id);
    if (it != impl.registry.end() && it->second.get() == req.get())
      impl.registry.erase(it);
  }
  return result;
}

void CollectiveBus::quiesce() {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stop_mu);
  if (impl.quiesced || impl.stopped) return;
  impl.quiesced = true;

  impl.stopping.store(true, std::memory_order_relaxed);
  if (impl.engine.joinable()) impl.engine.join();

  // Orderly consumer stop: reserved sequence in every control cell.
  for (auto& peer_lanes : impl.peers) {
    for (auto& lane : peer_lanes) {
      if (!lane.lane) continue;
      StartSlot* control =
          lane.lane->layout().control_cell(lane.lane->slab());
      __atomic_store_n(&control->seq, kFlagStopSequence, __ATOMIC_RELEASE);
    }
  }
  for (cudaStream_t stream : impl.consumer_streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  impl.consumer_streams.clear();
  impl.consumers_launched = false;
}

void CollectiveBus::stop() {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stop_mu);
  if (impl.stopped) return;
  impl.stopped = true;

  // Inline quiesce (we already hold stop_mu).
  if (!impl.quiesced) {
    impl.quiesced = true;
    impl.stopping.store(true, std::memory_order_relaxed);
    if (impl.engine.joinable()) impl.engine.join();

    for (auto& peer_lanes : impl.peers) {
      for (auto& lane : peer_lanes) {
        if (!lane.lane) continue;
        StartSlot* control =
            lane.lane->layout().control_cell(lane.lane->slab());
        __atomic_store_n(&control->seq, kFlagStopSequence, __ATOMIC_RELEASE);
      }
    }
    for (cudaStream_t stream : impl.consumer_streams) {
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
    }
    impl.consumer_streams.clear();
    impl.consumers_launched = false;
  }

  // Waiters must never hang: fail and release anything still registered.
  std::vector<std::shared_ptr<BusRequest>> remaining;
  {
    std::lock_guard<std::mutex> rlock(impl.registry_mu);
    for (auto& entry : impl.registry) remaining.push_back(entry.second);
    impl.registry.clear();
  }
  for (auto& req : remaining)
    impl.complete_request(req.get(), false, "bus stopped");

  // Lane teardown (QP -> CQ -> MR -> slab) and device close. All consumer
  // kernels must be dead by here: cudaFreeHost synchronizes the device
  // implicitly and would otherwise wait out a live peer's consumers.
  impl.peers.clear();
  impl.devices.clear();
}

BusStats CollectiveBus::stats() const {
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.stats_mu);
  BusStats out = impl.stats_store;
  for (auto& peer_lanes : impl.peers)
    for (auto& lane : peer_lanes) out.lanes.push_back(lane.stats);
  return out;
}

BusRecvView CollectiveBus::recv_view(int peer_rank, int lane) const {
  Impl& impl = *impl_;
  BusRecvView view;
  const auto it =
      std::find(impl.peer_ranks.begin(), impl.peer_ranks.end(), peer_rank);
  if (it == impl.peer_ranks.end() ||
      lane < 0 || lane >= static_cast<int>(impl.lane_count()))
    return view;
  const Impl::LaneState& state =
      impl.peers[static_cast<size_t>(it - impl.peer_ranks.begin())]
                [static_cast<size_t>(lane)];
  if (!state.lane) return view;
  RcLane& rc = *state.lane;
  const BusSlabLayout& layout = rc.layout();
  uint8_t* slab = rc.slab();
  view.doorbell_lat = layout.recv_doorbell(slab, BusPool::kLatency, 0);
  view.doorbell_bulk = layout.recv_doorbell(slab, BusPool::kBulk, 0);
  view.payload_lat = reinterpret_cast<const uint64_t*>(
      layout.recv_payload(slab, BusPool::kLatency, 0));
  view.payload_bulk = reinterpret_cast<const uint64_t*>(
      layout.recv_payload(slab, BusPool::kBulk, 0));
  view.ack_lat = layout.ack_cell(slab, BusPool::kLatency, 0);
  view.ack_bulk = layout.ack_cell(slab, BusPool::kBulk, 0);
  view.control = layout.control_cell(slab);
  view.lat_slots = static_cast<int>(options_.lat_slots);
  view.bulk_slots = static_cast<int>(options_.bulk_slots);
  view.lat_slot_bytes = static_cast<uint32_t>(options_.lat_slot_bytes);
  view.bulk_slot_bytes = static_cast<uint32_t>(options_.bulk_slot_bytes);
  return view;
}

}  // namespace dgpp::net
