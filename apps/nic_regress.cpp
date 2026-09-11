// M5 deliverable 5: the transport regression command. After a driver or
// firmware change, rerun the DESIGN §2.3 hardware contract — NIC→GPU
// visibility — on EVERY directed node pair over both active lanes, from one
// command with one verdict.
//
// The probe is the M0 project test, protocol-verbatim: a 64-byte payload and
// a 64-byte sequence doorbell posted as ordered RC SENDs; the receiver's GPU
// kernel polls the doorbell with a system-scope acquire, folds the payload
// into a hash, and publishes a system-scope release ack. The receiver
// compares the hash against the expected fold; the sender validates every
// per-sequence result. A driver that reorders payload and doorbell, drops
// either, or breaks system-scope visibility fails HERE, loudly.
//
// GID policy: probes use the bus's deployment selection (first routable
// RoCEv2 GID, via VerbsDevice) — not the M0 bench default, which probes the
// link-local fe80:: GID that the four-node mesh measured one-way silent
// drops on. A regression must exercise the path the deployment uses.
//
// Mesh orchestration is coordinator-free: every node derives the same
// deterministic schedule (each directed pair, each lane) and processes its
// obligations in order. Senders connect with bounded retries, receivers
// accept with a deadline, and connections that arrive early are
// identity-checked and held until their step's turn — a sender that laps a
// slower pair is parked, not dropped. Results aggregate at nodes[0]; the
// verdict is broadcast so every node exits with the same code.
//
// Modes:
//   mesh     --nodes IP[,IP...] [--port N] [--lanes D,D] [--iters N]
//              [--deadline-s N]     the deliverable
//   pair     --peer HOST:PORT [--lanes D,D] [--iters N] [--deadline-s N]
//              one directed probe per lane against a serving peer
//   serve    [--port N] [--lanes D,D] [--idle-s N]
//              accepts single-pair probes (the pair-mode counterpart)
//   selftest [--iters N] [--lanes D,D]
//              the full mesh machinery, world=2, both ranks as threads on
//              this device — the CI-able regression of the regression
//
// The one-process selftest shape is the M5 bring-up lesson applied by
// design: every device-synchronizing construction (slabs, MRs, streams)
// happens BEFORE the threads start, and the run phase syncs only the rank's
// own stream — a spinning peer kernel can never wedge a host-side
// allocation again.
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <cerrno>
#include <ifaddrs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "kernels/flag_protocol_types.hpp"
#include "net/nic_visibility.hpp"
#include "net/tcp.hpp"
#include "net/verbs.hpp"

namespace {

using dgpp::net::TcpConn;
using dgpp::net::TcpListener;

// ---- protocol constants (M0 project test, verbatim) ------------------------

constexpr size_t kPayloadWords = 16;  // one 64 B cache line
constexpr size_t kPayloadBytes = kPayloadWords * 4;
constexpr size_t kPayloadOffset = 0;  // slab cells
constexpr size_t kStartOffset = 4096;
constexpr size_t kAckOffset = 8192;
constexpr size_t kSlabBytes = 3 * 4096;

constexpr int kDefaultIters = 10000;  // the documented project-test strength
constexpr uint16_t kDefaultPort = 29961;
constexpr int kMeshDeadlineS = 1200;  // bounds every wait; failures are loud
constexpr int kIoMs = 10000;          // per-op TCP bound once a peer is live
constexpr int kConnectAttemptMs = 5000;
constexpr int kAckTimeoutMs = 5000;  // GPU ack wait (host spin, M0 bound)
constexpr int kKernelIdleMs = 5000;  // visibility-kernel inactivity exit
constexpr int kPairDeadlineS = 120;
constexpr int kServeIdleS = 300;
constexpr int kSelftestIters = 256;
constexpr int kSelftestDeadlineS = 300;

const std::vector<std::string> kDefaultLanes = {"rocep1s0f0", "roceP2p1s0f0"};

// Identity tags for the control frames (uniform little-endian cluster, the
// roster's stance; a version field leaves migration room).
constexpr uint32_t kHelloMagic = 0x31524744u;     // 'D','G','R','1'
constexpr uint32_t kAckMagic = 0x414B5247u;       // 'G','R','K','A'
constexpr uint32_t kQpMagic = 0x31505144u;        // 'D','Q','P','1'
constexpr uint32_t kSeqMagic = 0x314B5344u;       // 'D','S','K','1'
constexpr uint32_t kReportMagic = 0x32524744u;    // 'D','G','R','2'
constexpr uint32_t kVerdictMagic = 0x33524744u;   // 'D','G','R','3'
constexpr uint32_t kFrameVersion = 1;

// ProbeHello: the sender's first frame on a step connection. Mesh steps fill
// every field; pair-mode probes use step 0 and rank -1. The receiver checks
// it against ITS OWN schedule, so a mesh run with diverged --nodes/--lanes/
// --iters reports an error instead of probing a wrong pairing.
struct ProbeHello {
  uint32_t magic;
  uint32_t version;
  uint32_t step;
  int32_t sender_rank;
  int32_t receiver_rank;
  int32_t lane_index;
  int32_t iters;
  char lane_name[64];  // NUL-terminated; display + agreement check
};
static_assert(sizeof(ProbeHello) == 92, "ProbeHello wire size");

struct ProbeAck {
  uint32_t magic;
  int32_t ok;  // 1 = proceed to the QP swap
  char reason[80];
};
static_assert(sizeof(ProbeAck) == 88, "ProbeAck wire size");

struct QpInfo {
  uint32_t magic;
  uint32_t qpn;
  uint32_t psn;
  uint32_t pad;
  uint8_t gid[16];
};
static_assert(sizeof(QpInfo) == 32, "QpInfo wire size");

// Per-sequence verdict from receiver to sender (M0's VerifyResult plus an
// identity tag).
struct SeqResult {
  uint32_t magic;
  uint32_t seq;
  int32_t ok;
  uint32_t pad;
  uint64_t observed_hash;
  uint64_t expected_hash;
};
static_assert(sizeof(SeqResult) == 32, "SeqResult wire size");

// Aggregation: every rank reports its per-step views to nodes[0]; a step is
// green only when both endpoint views are green. The receiver view carries
// the visibility-latency tails (µs).
struct ReportHeader {
  uint32_t magic;
  uint32_t version;
  int32_t rank;
  int32_t entry_count;
};
static_assert(sizeof(ReportHeader) == 16, "ReportHeader wire size");

struct ReportEntry {
  uint32_t step;
  uint8_t role;  // 0 = sender view, 1 = receiver view
  uint8_t ok;
  uint16_t pad;
  uint32_t p50_us;
  uint32_t p99_us;
};
static_assert(sizeof(ReportEntry) == 16, "ReportEntry wire size");

struct Verdict {
  uint32_t magic;
  uint32_t version;
  int32_t exit_code;
  uint32_t steps_total;
  uint32_t steps_passed;
};
static_assert(sizeof(Verdict) == 20, "Verdict wire size");

// ---- small utilities -------------------------------------------------------

class Deadline {
 public:
  explicit Deadline(int seconds)
      : end_(std::chrono::steady_clock::now() +
             std::chrono::seconds(seconds)) {}
  bool expired() const { return std::chrono::steady_clock::now() >= end_; }
  int remaining_ms() const {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_ - std::chrono::steady_clock::now())
                        .count();
    return ms < 0 ? 0 : static_cast<int>(ms);
  }

 private:
  std::chrono::steady_clock::time_point end_;
};

std::string to_lower(std::string s) {
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

void latency_tails(std::vector<double> samples, double* p50, double* p99) {
  *p50 = 0.0;
  *p99 = 0.0;
  if (samples.empty()) return;
  std::sort(samples.begin(), samples.end());
  const auto pick = [&](double fraction) {
    return samples[std::min(samples.size() - 1, static_cast<size_t>(
                                                   fraction * samples.size()))];
  };
  *p50 = pick(0.50);
  *p99 = pick(0.99);
}

// M0's payload generator and fold, verbatim — the hash must equal the
// kernel's fold in flag_payload_kernel.
void fill_verify_payload(uint32_t* payload, uint32_t seq) {
  for (uint32_t word = 0; word < kPayloadWords; ++word)
    payload[word] = seq * 2654435761u + word;
}

uint64_t verify_payload_hash(const uint32_t* payload) {
  uint64_t hash = 0;
  for (size_t word = 0; word + 1 < kPayloadWords; word += 2) {
    const uint64_t pair = (static_cast<uint64_t>(payload[word]) << 32) |
                          static_cast<uint64_t>(payload[word + 1]);
    hash ^= pair * 0x9E3779B97F4A7C15ull;
  }
  return hash;
}

bool write_frame(TcpConn& conn, const void* frame, size_t bytes,
                const char* what) {
  if (!conn.write_all(frame, bytes)) {
    DGPP_LOG_ERROR("{} write failed (peer gone?)", what);
    return false;
  }
  return true;
}

template <typename Frame>
bool read_frame(TcpConn& conn, Frame* out, const char* what) {
  if (!conn.read_exact(out, sizeof(*out))) {
    DGPP_LOG_ERROR("{} read failed (peer gone or deadline)", what);
    return false;
  }
  return true;
}

// ---- lanes and probe QPs ---------------------------------------------------

// One opened lane: the device (deployment GID policy), a pinned slab with
// the three M0 cells, and its MR. Constructed during the pre-thread phase so
// cudaHostAlloc can never collide with a spinning kernel (M5 bring-up
// lesson).
class Lane {
 public:
  Lane(const std::string& device_hint, std::string* error)
      : device_(device_hint, error) {
    if (!device_.ok()) return;  // *error is set
    const cudaError_t alloc = cudaHostAlloc(
        reinterpret_cast<void**>(&slab_), kSlabBytes, cudaHostAllocDefault);
    if (alloc != cudaSuccess || !slab_) {
      *error = std::string("slab cudaHostAlloc failed (") +
               cudaGetErrorString(alloc) + ") on " + device_.name();
      slab_ = nullptr;
      return;
    }
    std::memset(slab_, 0, kSlabBytes);
    mr_ = ibv_reg_mr(device_.pd(), slab_, kSlabBytes,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_) {
      *error = "slab reg_mr failed errno=" + std::to_string(errno) + " on " +
               device_.name();
    }
  }
  ~Lane() {
    if (mr_) ibv_dereg_mr(mr_);
    if (slab_) cudaFreeHost(slab_);
  }
  Lane(const Lane&) = delete;
  Lane& operator=(const Lane&) = delete;

  bool ok() const { return device_.ok() && slab_ != nullptr && mr_ != nullptr; }
  dgpp::net::VerbsDevice& device() { return device_; }
  ibv_mr* mr() { return mr_; }
  uint32_t* payload() {
    return reinterpret_cast<uint32_t*>(slab_ + kPayloadOffset);
  }
  dgpp::StartSlot* start() {
    return reinterpret_cast<dgpp::StartSlot*>(slab_ + kStartOffset);
  }
  dgpp::FlagAck* ack() {
    return reinterpret_cast<dgpp::FlagAck*>(slab_ + kAckOffset);
  }
  const std::string& name() const { return device_.name(); }

 private:
  dgpp::net::VerbsDevice device_;
  uint8_t* slab_ = nullptr;
  ibv_mr* mr_ = nullptr;
};

// One raw RC QP for a probe step, with the bus's measured transition knobs
// (verbs.hpp kQp*): the regression validates the same QP configuration the
// deployment uses. `lane_lkey` is the lane MR's lkey for posted WRs.
class ProbeQp {
 public:
  ProbeQp(dgpp::net::VerbsDevice& device, uint32_t lane_lkey,
          std::string* error)
      : device_(&device), lane_lkey_(lane_lkey) {
    tx_cq_ = ibv_create_cq(device.ctx(), kQpDepth, nullptr, nullptr, 0);
    rx_cq_ = ibv_create_cq(device.ctx(), kQpDepth, nullptr, nullptr, 0);
    if (!tx_cq_ || !rx_cq_) {
      *error = "probe create_cq failed errno=" + std::to_string(errno);
      return;
    }
    ibv_qp_init_attr init{};
    init.send_cq = tx_cq_;
    init.recv_cq = rx_cq_;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = kQpDepth;
    init.cap.max_recv_wr = kQpDepth;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = 0;
    qp_ = ibv_create_qp(device.pd(), &init);
    if (!qp_) {
      *error = "probe create_qp failed errno=" + std::to_string(errno);
      return;
    }
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = static_cast<uint8_t>(device.port());
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PORT |
                                       IBV_QP_PKEY_INDEX |
                                       IBV_QP_ACCESS_FLAGS) != 0) {
      *error = "probe qp->INIT failed errno=" + std::to_string(errno);
    }
  }
  ~ProbeQp() {
    if (qp_) ibv_destroy_qp(qp_);
    if (tx_cq_) ibv_destroy_cq(tx_cq_);
    if (rx_cq_) ibv_destroy_cq(rx_cq_);
  }
  ProbeQp(const ProbeQp&) = delete;
  ProbeQp& operator=(const ProbeQp&) = delete;

  bool ok() const { return qp_ != nullptr; }

  // RTR + RTS against the peer's endpoint, mirroring RcLane::connect.
  bool connect(const QpInfo& peer, std::string* error) {
    ibv_port_attr port{};
    if (ibv_query_port(device_->ctx(), device_->port(), &port) != 0) {
      *error = "probe query_port failed errno=" + std::to_string(errno);
      return false;
    }
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu =
        port.active_mtu < IBV_MTU_4096 ? port.active_mtu : IBV_MTU_4096;
    attr.dest_qp_num = peer.qpn;
    attr.rq_psn = peer.psn;
    attr.max_dest_rd_atomic = dgpp::net::kQpMaxRdAtomic;
    attr.min_rnr_timer = dgpp::net::kQpMinRnrTimer;
    attr.ah_attr.is_global = 1;
    std::memcpy(attr.ah_attr.grh.dgid.raw, peer.gid, 16);
    attr.ah_attr.grh.sgid_index = static_cast<uint8_t>(device_->gid_index());
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.sl = 0;
    attr.ah_attr.port_num = static_cast<uint8_t>(device_->port());
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                          IBV_QP_RQ_PSN | IBV_QP_AV |
                          IBV_QP_MAX_DEST_RD_ATOMIC |
                          IBV_QP_MIN_RNR_TIMER) != 0) {
      *error = "probe qp->RTR failed errno=" + std::to_string(errno);
      return false;
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = dgpp::net::kQpTimeout;
    attr.retry_cnt = dgpp::net::kQpRetryCount;
    attr.rnr_retry = dgpp::net::kQpRnrRetry;
    attr.sq_psn = dgpp::net::kQpPsn;
    attr.max_rd_atomic = dgpp::net::kQpMaxRdAtomic;
    if (ibv_modify_qp(qp_, &attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      *error = "probe qp->RTS failed errno=" + std::to_string(errno);
      return false;
    }
    return true;
  }

  QpInfo info() const {
    QpInfo q{};
    q.magic = kQpMagic;
    q.qpn = qp_->qp_num;
    q.psn = static_cast<uint32_t>(dgpp::net::kQpPsn);
    std::memcpy(q.gid, device_->gid().raw, 16);
    return q;
  }

  // Send side: payload first (unsignaled), doorbell second (signaled) —
  // ordered RC SENDs, the M0 protocol.
  bool post_send_pair(uint32_t* payload, dgpp::StartSlot* start,
                      uint64_t wr_id, std::string* error) {
    ibv_sge sge[2]{};
    sge[0].addr = reinterpret_cast<uint64_t>(payload);
    sge[0].lkey = lane_lkey_;
    sge[0].length = static_cast<uint32_t>(kPayloadBytes);
    sge[1].addr = reinterpret_cast<uint64_t>(start);
    sge[1].lkey = lane_lkey_;
    sge[1].length = sizeof(*start);
    ibv_send_wr wr[2]{};
    wr[0].wr_id = wr_id;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;
    wr[0].opcode = IBV_WR_SEND;
    wr[0].next = &wr[1];
    wr[1].wr_id = wr_id + 1;
    wr[1].sg_list = &sge[1];
    wr[1].num_sge = 1;
    wr[1].opcode = IBV_WR_SEND;
    wr[1].send_flags = IBV_SEND_SIGNALED;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp_, &wr[0], &bad) != 0) {
      *error = "probe post_send pair failed errno=" + std::to_string(errno);
      return false;
    }
    return true;
  }

  // Receive side: pre-post both buffers, as the M0 responder did.
  bool post_recv_pair(uint32_t* payload, dgpp::StartSlot* start,
                      uint64_t wr_id, std::string* error) {
    ibv_sge sge[2]{};
    sge[0].addr = reinterpret_cast<uint64_t>(payload);
    sge[0].lkey = lane_lkey_;
    sge[0].length = static_cast<uint32_t>(kPayloadBytes);
    sge[1].addr = reinterpret_cast<uint64_t>(start);
    sge[1].lkey = lane_lkey_;
    sge[1].length = sizeof(*start);
    for (int leg = 0; leg < 2; ++leg) {
      ibv_recv_wr wr{};
      wr.wr_id = wr_id + static_cast<uint64_t>(leg);
      wr.sg_list = &sge[leg];
      wr.num_sge = 1;
      ibv_recv_wr* bad = nullptr;
      if (ibv_post_recv(qp_, &wr, &bad) != 0) {
        *error = "probe post_recv leg " + std::to_string(leg) +
                 " failed errno=" + std::to_string(errno);
        return false;
      }
    }
    return true;
  }

  // Bounded CQ waits; bad statuses surface with the verbs error text.
  bool wait_tx(std::string* error) {
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(kAckTimeoutMs);
    for (;;) {
      ibv_wc wc{};
      const int n = ibv_poll_cq(tx_cq_, 1, &wc);
      if (n < 0) {
        *error = "probe tx poll failed";
        return false;
      }
      if (n == 1) {
        if (wc.status != IBV_WC_SUCCESS) {
          *error = "probe tx completion: " + dgpp::net::bus_wc_error(wc);
          return false;
        }
        return true;
      }
      if (std::chrono::steady_clock::now() > end) {
        *error = "probe tx completion timeout";
        return false;
      }
      std::this_thread::yield();
    }
  }

  bool wait_rx_pair(std::string* error) {
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(kAckTimeoutMs);
    int got = 0;
    while (got < 2) {
      ibv_wc wcs[2]{};
      const int n = ibv_poll_cq(rx_cq_, 2, wcs);
      if (n < 0) {
        *error = "probe rx poll failed";
        return false;
      }
      for (int i = 0; i < n; ++i) {
        if (wcs[i].status != IBV_WC_SUCCESS) {
          *error = "probe rx completion: " + dgpp::net::bus_wc_error(wcs[i]);
          return false;
        }
        if (wcs[i].byte_len != kPayloadBytes &&
            wcs[i].byte_len != sizeof(dgpp::StartSlot)) {
          *error =
              "probe rx unexpected byte_len " + std::to_string(wcs[i].byte_len);
          return false;
        }
        ++got;
      }
      if (got < 2) {
        if (std::chrono::steady_clock::now() > end) {
          *error = "probe rx completion timeout";
          return false;
        }
        std::this_thread::yield();
      }
    }
    return true;
  }

 private:
  static constexpr int kQpDepth = 64;
  dgpp::net::VerbsDevice* device_ = nullptr;
  uint32_t lane_lkey_ = 0;
  ibv_cq* tx_cq_ = nullptr;
  ibv_cq* rx_cq_ = nullptr;
  ibv_qp* qp_ = nullptr;
};

// ---- the probe step (M0 protocol, on Lane/ProbeQp/TcpConn) ----------------

struct ProbeOutcome {
  bool ok = false;
  int verified = 0;
  std::string error;  // empty when the protocol ran clean
  double p50_us = 0.0;
  double p99_us = 0.0;
};

// Host-side wait for the GPU's release ack — the ordering invariant the
// whole contract hangs on (payload fully visible when ack.seq == seq).
bool wait_ack_seq(dgpp::FlagAck* ack, uint32_t seq, uint64_t* hash_out) {
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(kAckTimeoutMs);
  while (__atomic_load_n(&ack->seq, __ATOMIC_ACQUIRE) != seq) {
    if (std::chrono::steady_clock::now() > end) return false;
    std::this_thread::yield();
  }
  // The hash was published before the seq release; safe to read after the
  // acquire that observed the release.
  *hash_out = ack->hash;
  return true;
}

bool swap_qp_info(TcpConn& conn, ProbeQp* qp, QpInfo* peer_out,
                  std::string* error) {
  const QpInfo mine = qp->info();
  if (!conn.write_all(&mine, sizeof(mine))) {
    *error = "qp-info write failed";
    return false;
  }
  QpInfo peer{};
  if (!conn.read_exact(&peer, sizeof(peer)) || peer.magic != kQpMagic) {
    *error = "qp-info read failed (bad magic or peer gone)";
    return false;
  }
  if (!qp->connect(peer, error)) return false;
  *peer_out = peer;
  return true;
}

// Sender side of one probe step. `handshake_io_ms` bounds the ack/QP-swap
// reads — a mesh sender can be parked behind a slower pair, so the mesh
// deadline, not a fixed slice, is the honest bound.
ProbeOutcome run_probe_sender(TcpConn& conn, Lane& lane,
                              const ProbeHello& hello, int handshake_io_ms) {
  ProbeOutcome out;
  std::string error;
  ProbeQp qp(lane.device(), lane.mr()->lkey, &error);
  if (!qp.ok()) {
    out.error = error;
    return out;
  }
  conn.set_io_deadline_ms(handshake_io_ms);
  if (!write_frame(conn, &hello, sizeof(hello), "hello")) {
    out.error = "hello write failed";
    return out;
  }
  ProbeAck ack{};
  if (!read_frame(conn, &ack, "probe ack") || ack.magic != kAckMagic) {
    out.error = "probe ack read failed (bad magic or refused)";
    return out;
  }
  if (!ack.ok) {
    out.error = "receiver refused the probe: " + std::string(ack.reason);
    return out;
  }
  QpInfo peer{};
  if (!swap_qp_info(conn, &qp, &peer, &error)) {
    out.error = error;
    return out;
  }

  conn.set_io_deadline_ms(kIoMs);
  for (int seq = 1; seq <= hello.iters; ++seq) {
    char ready = 0;
    if (!conn.read_exact(&ready, 1) || ready != 'R') {
      out.error = "seq " + std::to_string(seq) + ": receiver not ready";
      break;
    }
    fill_verify_payload(lane.payload(), static_cast<uint32_t>(seq));
    *lane.start() = {};
    lane.start()->seq = static_cast<uint32_t>(seq);
    std::atomic_thread_fence(std::memory_order_release);
    if (!qp.post_send_pair(lane.payload(), lane.start(),
                           2ull * static_cast<uint64_t>(seq), &error) ||
        !qp.wait_tx(&error)) {
      out.error = "seq " + std::to_string(seq) + ": " + error;
      break;
    }
    SeqResult result{};
    if (!read_frame(conn, &result, "seq result") ||
        result.magic != kSeqMagic) {
      out.error = "seq " + std::to_string(seq) + ": result read failed";
      break;
    }
    if (result.seq != static_cast<uint32_t>(seq) || !result.ok) {
      out.error = "seq " + std::to_string(seq) +
                  " rejected by receiver (observed " +
                  std::to_string(result.observed_hash) + " vs expected " +
                  std::to_string(result.expected_hash) + ")";
      break;
    }
    ++out.verified;
  }
  out.ok = out.verified == hello.iters;
  conn.write_all("BYE", 3);  // best-effort close marker
  return out;
}

// Receiver side of one probe step: run the visibility kernel over the lane's
// NIC-fed cells, verify every hash, and report per-seq results.
// `handshake_io_ms` refreshes the connection's I/O deadline — a held
// connection may have been accepted long before its step's turn.
ProbeOutcome run_probe_receiver(TcpConn& conn, Lane& lane, cudaStream_t stream,
                                int clock_khz, int iters,
                                int handshake_io_ms) {
  ProbeOutcome out;
  std::string error;
  ProbeQp qp(lane.device(), lane.mr()->lkey, &error);
  if (!qp.ok()) {
    out.error = error;
    return out;
  }
  // Cells must read idle BEFORE the kernel launches (its first acquire would
  // otherwise see a stale seq and skip or mis-ack the first message).
  *lane.start() = {};
  *lane.ack() = {};
  std::fill_n(lane.payload(), kPayloadWords, 0u);

  conn.set_io_deadline_ms(handshake_io_ms);
  ProbeAck ack{};
  ack.magic = kAckMagic;
  ack.ok = 1;
  if (!write_frame(conn, &ack, sizeof(ack), "probe ack")) {
    out.error = "probe ack write failed";
    return out;
  }
  QpInfo peer{};
  if (!swap_qp_info(conn, &qp, &peer, &error)) {
    out.error = error;
    return out;
  }

  const uint64_t deadline_cycles =
      static_cast<uint64_t>(clock_khz) * kKernelIdleMs;
  const cudaError_t launch = dgpp::net::launch_nic_visibility_kernel(
      &lane.start()->seq, lane.payload(), lane.ack(), deadline_cycles, stream);
  if (launch != cudaSuccess) {
    out.error = std::string("visibility kernel launch failed: ") +
                cudaGetErrorString(launch);
    return out;
  }

  conn.set_io_deadline_ms(kIoMs);  // per-seq bounds from here on
  std::vector<double> latency_us;
  latency_us.reserve(static_cast<size_t>(iters));
  for (int seq = 1; seq <= iters; ++seq) {
    if (!qp.post_recv_pair(lane.payload(), lane.start(),
                           2ull * static_cast<uint64_t>(seq), &error)) {
      out.error = "seq " + std::to_string(seq) + ": " + error;
      break;
    }
    const auto sent_ready = std::chrono::steady_clock::now();
    if (!conn.write_all("R", 1)) {
      out.error = "seq " + std::to_string(seq) + ": ready write failed";
      break;
    }
    uint64_t observed_hash = 0;
    const bool observed = wait_ack_seq(lane.ack(),
                                       static_cast<uint32_t>(seq),
                                       &observed_hash);
    if (observed) {
      latency_us.push_back(
          std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now() - sent_ready)
              .count());
    }
    uint32_t expected_payload[16];
    fill_verify_payload(expected_payload, static_cast<uint32_t>(seq));
    const uint64_t expected_hash = verify_payload_hash(expected_payload);
    const bool receives_ok = qp.wait_rx_pair(&error);
    const bool hash_ok = observed && receives_ok &&
                         observed_hash == expected_hash;
    SeqResult result{};
    result.magic = kSeqMagic;
    result.seq = static_cast<uint32_t>(seq);
    result.ok = hash_ok ? 1 : 0;
    result.observed_hash = observed ? observed_hash : 0;
    result.expected_hash = expected_hash;
    if (!conn.write_all(&result, sizeof(result))) {
      out.error = "seq " + std::to_string(seq) + ": result write failed";
      break;
    }
    if (!hash_ok) {
      out.error = "seq " + std::to_string(seq) + ": " +
                  (observed ? "payload hash mismatch (visibility broke)"
                            : "GPU ack missing (doorbell or kernel gone)") +
                  (receives_ok ? "" : " + rx wc failure: " + error);
      break;
    }
    ++out.verified;
  }

  __atomic_store_n(&lane.start()->seq, dgpp::kFlagStopSequence,
                   __ATOMIC_RELEASE);
  const cudaError_t sync = cudaStreamSynchronize(stream);
  if (sync != cudaSuccess) {
    out.error = std::string("visibility kernel completion failed: ") +
                cudaGetErrorString(sync);
    return out;
  }
  out.ok = out.verified == iters;
  latency_tails(std::move(latency_us), &out.p50_us, &out.p99_us);
  return out;
}

// ---- mesh schedule and orchestration ---------------------------------------

struct MeshStep {
  int lane = 0;
  int sender = 0;
  int receiver = 0;
};

// Pair-major: each directed pair is fully probed (all its lanes) before the
// next pair, so field debugging reads top-to-bottom per pair.
std::vector<MeshStep> build_schedule(int world, size_t lane_count) {
  std::vector<MeshStep> steps;
  for (int sender = 0; sender < world; ++sender)
    for (int receiver = 0; receiver < world; ++receiver) {
      if (sender == receiver) continue;
      for (int lane = 0; lane < static_cast<int>(lane_count); ++lane)
        steps.push_back({lane, sender, receiver});
    }
  return steps;
}

struct MeshConfig {
  std::vector<std::string> nodes;  // display + connect addresses
  std::vector<std::string> lanes;  // device name hints
  int iters = kDefaultIters;
  int deadline_s = kMeshDeadlineS;
};

struct MeshAddr {
  std::string display;
  std::string host;
  uint16_t port = 0;
};

// One node's verdict about one step; the error text stays local (logs),
// the wire carries the boolean.
struct StepOutcome {
  uint32_t step = 0;
  uint8_t role = 0;  // 0 = sender view, 1 = receiver view
  bool ok = false;
  std::string error;
  double p50_us = 0.0;
  double p99_us = 0.0;
};

ReportEntry to_report_entry(const StepOutcome& o) {
  ReportEntry e{};
  e.step = o.step;
  e.role = o.role;
  e.ok = o.ok ? 1 : 0;
  e.p50_us = static_cast<uint32_t>(std::lround(o.p50_us));
  e.p99_us = static_cast<uint32_t>(std::lround(o.p99_us));
  return e;
}

std::string lane_name_or(const MeshConfig& cfg, int index) {
  return index >= 0 && index < static_cast<int>(cfg.lanes.size())
             ? cfg.lanes[static_cast<size_t>(index)]
             : "?";
}

// Validates a mesh hello against MY schedule: diverged configs on different
// nodes must fail loudly, never probe a wrong pairing silently.
std::string validate_step_hello(const ProbeHello& h, int my_rank,
                                const MeshConfig& cfg,
                                const std::vector<MeshStep>& steps) {
  if (h.magic != kHelloMagic || h.version != kFrameVersion)
    return "bad magic/version";
  if (h.iters != cfg.iters) return "iters mismatch (config divergence?)";
  if (h.lane_index < 0 || h.lane_index >= static_cast<int>(cfg.lanes.size()))
    return "lane index out of range";
  char name[65];
  std::memcpy(name, h.lane_name, 64);
  name[64] = '\0';
  if (name != cfg.lanes[static_cast<size_t>(h.lane_index)])
    return "lane name disagrees with lane index";
  if (h.step >= steps.size()) return "step index out of range";
  const MeshStep& st = steps[h.step];
  if (st.lane != h.lane_index || st.sender != h.sender_rank ||
      st.receiver != my_rank)
    return "hello does not match my schedule (different --nodes/--lanes on "
           "peers?)";
  return "";
}

ProbeHello make_step_hello(uint32_t step, const MeshStep& st,
                           const MeshConfig& cfg, int sender_rank) {
  ProbeHello h{};
  h.magic = kHelloMagic;
  h.version = kFrameVersion;
  h.step = step;
  h.sender_rank = sender_rank;
  h.receiver_rank = st.receiver;
  h.lane_index = st.lane;
  h.iters = cfg.iters;
  std::snprintf(h.lane_name, sizeof(h.lane_name), "%s",
               cfg.lanes[static_cast<size_t>(st.lane)].c_str());
  return h;
}

TcpConn connect_until(const MeshAddr& addr, const Deadline& deadline,
                      std::string* error) {
  for (;;) {
    if (deadline.expired()) {
      *error = "deadline expired before connecting to " + addr.display;
      return {};
    }
    try {
      return TcpConn::connect(addr.host, addr.port, kConnectAttemptMs);
    } catch (const std::exception& e) {
      *error = e.what();
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }
}

// Pulls the connection for `want_step` out of the listener. Connections for
// future receiver steps may arrive early (a sender can lap a slower pair):
// each is identity-checked, held for its turn, and a superseded retry
// replaces the older one. Anything that is not a future step of mine is
// refused promptly so its sender fails fast instead of parking.
TcpConn take_step_connection(TcpListener& listener, uint32_t want_step,
                             int rank, const MeshConfig& cfg,
                             const std::vector<MeshStep>& steps,
                             const std::vector<uint32_t>& my_receiver_steps,
                             std::map<uint32_t, TcpConn>* pending,
                             const Deadline& deadline, std::string* error) {
  if (auto held = pending->find(want_step); held != pending->end()) {
    TcpConn conn = std::move(held->second);
    pending->erase(held);
    return conn;
  }
  for (;;) {
    const int remain = deadline.remaining_ms();
    if (remain <= 0) {
      *error = "deadline expired waiting for step " + std::to_string(want_step);
      return {};
    }
    TcpConn conn = listener.accept(std::min(remain, 1000));
    if (!conn.valid()) continue;  // slice timeout; deadline re-checked above
    conn.set_io_deadline_ms(std::min(remain, kIoMs));
    ProbeHello hello{};
    if (!conn.read_exact(&hello, sizeof(hello)) ||
        hello.magic != kHelloMagic) {
      DGPP_LOG_ERROR("rank {}: step connection closed before a valid hello",
                     rank);
      continue;
    }
    const std::string why = validate_step_hello(hello, rank, cfg, steps);
    if (!why.empty()) {
      // Not this mesh's step: a diverged config or a stray. Refuse loudly
      // and keep listening for the real one.
      DGPP_LOG_ERROR("rank {}: refusing step connection: {} (hello step={} "
                     "sender={} lane={})",
                     rank, why, hello.step, hello.sender_rank, hello.lane_name);
      ProbeAck ack{};
      ack.magic = kAckMagic;
      ack.ok = 0;
      std::snprintf(ack.reason, sizeof(ack.reason), "%s", why.c_str());
      conn.write_all(&ack, sizeof(ack));
      conn.close();
      continue;
    }
    if (hello.step == want_step) return conn;
    const bool my_future =
        hello.step > want_step &&
        std::binary_search(my_receiver_steps.begin(), my_receiver_steps.end(),
                           hello.step);
    if (!my_future) {
      // A step I already served (or a foreign one): its sender must learn
      // now, not at the deadline.
      const std::string reason = "step already served or not mine";
      DGPP_LOG_ERROR("rank {}: refusing step connection: {} (hello step={} "
                     "sender={} lane={})",
                     rank, reason, hello.step, hello.sender_rank,
                     hello.lane_name);
      ProbeAck ack{};
      ack.magic = kAckMagic;
      ack.ok = 0;
      std::snprintf(ack.reason, sizeof(ack.reason), "%s", reason.c_str());
      conn.write_all(&ack, sizeof(ack));
      conn.close();
      continue;
    }
    if (auto held = pending->find(hello.step); held != pending->end()) {
      DGPP_LOG_INFO("rank {}: replacing superseded held connection for step "
                    "{}",
                    rank, hello.step);
      pending->erase(held);
    }
    pending->emplace(hello.step, std::move(conn));
  }
}

// Runs this rank's obligations in schedule order and returns its views.
std::vector<StepOutcome> run_mesh_obligations(
    int rank, const MeshConfig& cfg, const std::vector<MeshAddr>& addrs,
    const std::vector<MeshStep>& steps,
    const std::vector<std::unique_ptr<Lane>>& lanes, cudaStream_t stream,
    int clock_khz, TcpListener& listener, const Deadline& deadline) {
  std::vector<StepOutcome> outcomes;
  std::map<uint32_t, TcpConn> pending;
  std::vector<uint32_t> my_receiver_steps;
  for (uint32_t k = 0; k < steps.size(); ++k)
    if (steps[k].receiver == rank) my_receiver_steps.push_back(k);

  for (uint32_t k = 0; k < steps.size(); ++k) {
    const MeshStep& st = steps[k];
    if (st.sender != rank && st.receiver != rank) continue;
    StepOutcome out;
    out.step = k;
    out.role = st.sender == rank ? 0 : 1;
    if (st.sender == rank) {
      std::string error;
      TcpConn conn =
          connect_until(addrs[static_cast<size_t>(st.receiver)], deadline,
                        &error);
      if (conn.valid()) {
        const ProbeOutcome probe = run_probe_sender(
            conn, *lanes[static_cast<size_t>(st.lane)],
            make_step_hello(k, st, cfg, rank),
            std::min(deadline.remaining_ms(), 60000));
        out.ok = probe.ok;
        out.error = probe.error;
      } else {
        out.error = "connect: " + error;
      }
    } else {
      std::string error;
      TcpConn conn =
          take_step_connection(listener, k, rank, cfg, steps,
                               my_receiver_steps, &pending, deadline, &error);
      if (conn.valid()) {
        const ProbeOutcome probe = run_probe_receiver(
            conn, *lanes[static_cast<size_t>(st.lane)], stream, clock_khz,
            cfg.iters, std::min(deadline.remaining_ms(), kIoMs));
        out.ok = probe.ok;
        out.error = probe.error;
        out.p50_us = probe.p50_us;
        out.p99_us = probe.p99_us;
      } else {
        out.error = "accept: " + error;
      }
    }
    DGPP_LOG_INFO("rank {}: step {} ({}->{} lane {}): {}", rank, k,
                  addrs[static_cast<size_t>(st.sender)].display,
                  addrs[static_cast<size_t>(st.receiver)].display,
                  lane_name_or(cfg, st.lane),
                  out.ok ? "PASS" : "FAIL: " + out.error);
    outcomes.push_back(std::move(out));
  }
  return outcomes;
}

// Rank 0: merge every view (a step is green only when both endpoints say
// so), print the matrix, and broadcast one verdict to all reporters.
int aggregate_and_broadcast(int world, const MeshConfig& cfg,
                            const std::vector<MeshAddr>& addrs,
                            const std::vector<MeshStep>& steps,
                            std::vector<StepOutcome> my_views,
                            TcpListener& listener, const Deadline& deadline) {
  struct Merged {
    bool have_sender = false, sender_ok = false;
    bool have_receiver = false, receiver_ok = false;
    double p50_us = 0.0, p99_us = 0.0;
  };
  std::vector<Merged> merged(steps.size());
  const auto absorb = [&merged](const ReportEntry& e) {
    if (e.step >= merged.size()) return;
    Merged& m = merged[e.step];
    if (e.role == 0) {
      m.have_sender = true;
      m.sender_ok = e.ok != 0;
    } else {
      m.have_receiver = true;
      m.receiver_ok = e.ok != 0;
      m.p50_us = e.p50_us;
      m.p99_us = e.p99_us;
    }
  };
  for (const StepOutcome& view : my_views) absorb(to_report_entry(view));

  std::vector<TcpConn> reporters;
  int reports = 0;
  while (reports < world - 1) {
    if (deadline.expired()) {
      DGPP_LOG_ERROR(
          "aggregation: deadline expired with {}/{} reports; missing ranks' "
          "steps fail",
          reports, world - 1);
      break;
    }
    TcpConn conn = listener.accept(std::min(deadline.remaining_ms(), 1000));
    if (!conn.valid()) continue;
    conn.set_io_deadline_ms(kIoMs);
    ReportHeader hdr{};
    if (!conn.read_exact(&hdr, sizeof(hdr)) || hdr.magic != kReportMagic ||
        hdr.version != kFrameVersion || hdr.rank <= 0 || hdr.rank >= world ||
        hdr.entry_count < 0 ||
        hdr.entry_count > static_cast<int32_t>(steps.size())) {
      DGPP_LOG_ERROR("aggregation: refusing malformed report");
      conn.close();
      continue;
    }
    bool clean = true;
    for (int i = 0; i < hdr.entry_count && clean; ++i) {
      ReportEntry entry{};
      if (!conn.read_exact(&entry, sizeof(entry)) || entry.role > 1) {
        DGPP_LOG_ERROR("aggregation: rank {} report truncated", hdr.rank);
        clean = false;
        break;
      }
      absorb(entry);
    }
    if (!clean) {
      conn.close();
      continue;
    }
    reporters.push_back(std::move(conn));
    ++reports;
  }

  uint32_t passed = 0;
  for (int lane = 0; lane < static_cast<int>(cfg.lanes.size()); ++lane) {
    DGPP_LOG_INFO("lane[{}] {}:", lane, cfg.lanes[static_cast<size_t>(lane)]);
    for (size_t k = 0; k < steps.size(); ++k) {
      const MeshStep& st = steps[k];
      if (st.lane != lane) continue;
      const Merged& m = merged[k];
      const bool ok = m.have_sender && m.have_receiver && m.sender_ok &&
                      m.receiver_ok;
      if (ok) {
        ++passed;
        DGPP_LOG_INFO("  {}->{}: PASS (receiver p50 {:.1f}us p99 {:.1f}us)",
                      addrs[static_cast<size_t>(st.sender)].display,
                      addrs[static_cast<size_t>(st.receiver)].display, m.p50_us,
                      m.p99_us);
      } else {
        std::string why;
        if (!m.have_sender) why = "sender view missing";
        else if (!m.sender_ok) why = "sender view failed";
        if (!m.have_receiver) why += (why.empty() ? "" : " + ") +
                                     std::string("receiver view missing");
        else if (!m.receiver_ok) why += (why.empty() ? "" : " + ") +
                                        std::string("receiver view failed");
        DGPP_LOG_ERROR("  {}->{}: FAIL ({})",
                       addrs[static_cast<size_t>(st.sender)].display,
                       addrs[static_cast<size_t>(st.receiver)].display, why);
      }
    }
  }

  const int exit_code = passed == steps.size() ? 0 : 1;
  DGPP_LOG_INFO("mesh verdict: {}/{} steps passed (each step needs both "
                "endpoint views green); exit {}",
                passed, steps.size(), exit_code);
  Verdict v{};
  v.magic = kVerdictMagic;
  v.version = kFrameVersion;
  v.exit_code = exit_code;
  v.steps_total = static_cast<uint32_t>(steps.size());
  v.steps_passed = passed;
  for (TcpConn& conn : reporters) {
    conn.write_all(&v, sizeof(v));
    conn.close();
  }
  return exit_code;
}

// Non-coordinator ranks: report views, wait for the one verdict.
int report_and_verdict(int rank, const MeshAddr& coord,
                       const std::vector<StepOutcome>& views,
                       const Deadline& deadline) {
  std::string error;
  TcpConn conn = connect_until(coord, deadline, &error);
  if (!conn.valid()) {
    DGPP_LOG_ERROR("rank {}: cannot reach coordinator: {}", rank, error);
    return 1;
  }
  conn.set_io_deadline_ms(std::max(deadline.remaining_ms(), kIoMs));
  ReportHeader hdr{};
  hdr.magic = kReportMagic;
  hdr.version = kFrameVersion;
  hdr.rank = rank;
  hdr.entry_count = static_cast<int32_t>(views.size());
  if (!conn.write_all(&hdr, sizeof(hdr))) return 1;
  for (const StepOutcome& view : views) {
    const ReportEntry entry = to_report_entry(view);
    if (!conn.write_all(&entry, sizeof(entry))) return 1;
  }
  DGPP_LOG_INFO("rank {}: reported {} step views to the coordinator", rank,
                views.size());
  Verdict v{};
  if (!conn.read_exact(&v, sizeof(v)) || v.magic != kVerdictMagic ||
      v.version != kFrameVersion) {
    DGPP_LOG_ERROR("rank {}: verdict read failed", rank);
    return 1;
  }
  DGPP_LOG_INFO("rank {}: mesh verdict {}/{} steps; exit {}", rank,
                v.steps_passed, v.steps_total, v.exit_code);
  return v.exit_code;
}

// One rank's whole mesh run: obligations, then aggregate (rank 0) or report.
int run_mesh_rank(int rank, const MeshConfig& cfg,
                  const std::vector<MeshAddr>& addrs,
                  std::vector<std::unique_ptr<Lane>> lanes,
                  cudaStream_t stream, int clock_khz, TcpListener& listener) {
  const std::vector<MeshStep> steps =
      build_schedule(static_cast<int>(addrs.size()), cfg.lanes.size());
  const Deadline deadline(cfg.deadline_s);
  DGPP_LOG_INFO("rank {}: mesh {} nodes x {} lanes = {} steps, {} iters/step",
                rank, addrs.size(), cfg.lanes.size(), steps.size(), cfg.iters);
  std::vector<StepOutcome> views = run_mesh_obligations(
      rank, cfg, addrs, steps, lanes, stream, clock_khz, listener, deadline);
  const uint32_t mine_ok = static_cast<uint32_t>(std::count_if(
      views.begin(), views.end(), [](const StepOutcome& v) { return v.ok; }));
  DGPP_LOG_INFO("rank {}: {} of {} obligations passed", rank, mine_ok,
                views.size());
  if (rank == 0)
    return aggregate_and_broadcast(static_cast<int>(addrs.size()), cfg, addrs,
                                   steps, std::move(views), listener, deadline);
  return report_and_verdict(
      rank, addrs[0], views, deadline);
}

// ---- rank resolution (production mesh) -------------------------------------

std::vector<std::string> collect_local_names() {
  std::vector<std::string> names;
  char host[256] = {};
  if (gethostname(host, sizeof(host)) == 0 && host[0] != '\0') {
    const std::string full = to_lower(host);
    names.push_back(full);
    const size_t dot = full.find('.');
    if (dot != std::string::npos && dot > 0)
      names.push_back(full.substr(0, dot));
  }
  ifaddrs* ifs = nullptr;
  if (getifaddrs(&ifs) == 0 && ifs) {
    for (const ifaddrs* it = ifs; it; it = it->ifa_next) {
      if (!it->ifa_addr || it->ifa_addr->sa_family != AF_INET) continue;
      char buf[INET_ADDRSTRLEN] = {};
      if (inet_ntop(AF_INET,
                    &reinterpret_cast<const sockaddr_in*>(it->ifa_addr)
                         ->sin_addr,
                    buf, sizeof(buf)) != nullptr)
        names.push_back(to_lower(buf));
    }
    freeifaddrs(ifs);
  }
  return names;
}

// This host's index in --nodes, by matching entries against the hostname
// (full and short) and every interface address.
int resolve_my_rank(const std::vector<std::string>& nodes, std::string* error) {
  const std::vector<std::string> mine = collect_local_names();
  int matched = -1;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const std::string entry = to_lower(nodes[i]);
    bool hit = false;
    for (const std::string& name : mine) hit = hit || name == entry;
    if (hit) {
      if (matched >= 0 && matched != static_cast<int>(i)) {
        *error = "several --nodes entries match this host (" +
                 nodes[static_cast<size_t>(matched)] + ", " + nodes[i] + ")";
        return -1;
      }
      matched = static_cast<int>(i);
    }
  }
  if (matched < 0) {
    std::string listed;
    for (const std::string& name : mine)
      listed += (listed.empty() ? "" : ", ") + name;
    *error = "this host is not in --nodes; local names are: " + listed;
  }
  return matched;
}

// ---- shared construction ----------------------------------------------------

struct RankRig {
  std::vector<std::unique_ptr<Lane>> lanes;
  cudaStream_t stream = nullptr;
  int clock_khz = 0;
};

// Constructs one rank's lanes, stream, and clock reading — the
// device-synchronizing phase that must happen before any kernel spins
// (one-process selftest) and is simply startup order elsewhere.
RankRig construct_rank(const std::vector<std::string>& lane_hints) {
  RankRig rig;
  std::string error;
  for (const std::string& hint : lane_hints) {
    auto lane = std::make_unique<Lane>(hint, &error);
    if (!lane->ok())
      throw std::runtime_error("lane " + hint + ": " + error);
    rig.lanes.push_back(std::move(lane));
  }
  const cudaError_t stream =
      cudaStreamCreateWithFlags(&rig.stream, cudaStreamNonBlocking);
  if (stream != cudaSuccess)
    throw std::runtime_error(std::string("stream create failed: ") +
                             cudaGetErrorString(stream));
  int clock_khz = 0;
  const cudaError_t clock = cudaDeviceGetAttribute(
      &clock_khz, cudaDevAttrClockRate, 0);
  if (clock != cudaSuccess || clock_khz <= 0)
    throw std::runtime_error("device clock unavailable");
  rig.clock_khz = clock_khz;
  return rig;
}

// ---- modes ------------------------------------------------------------------

int run_mesh_mode(const MeshConfig& cfg, uint16_t port) {
  std::string error;
  const int rank = resolve_my_rank(cfg.nodes, &error);
  if (rank < 0) {
    DGPP_LOG_ERROR("mesh: {}", error);
    return 2;
  }
  RankRig rig = construct_rank(cfg.lanes);
  TcpListener listener = TcpListener::bind(port);
  std::vector<MeshAddr> addrs;
  for (const std::string& node : cfg.nodes)
    addrs.push_back({node, node, port});
  DGPP_LOG_INFO("mesh: rank {} of {} (listening on {})", rank,
                cfg.nodes.size(), port);
  const int exit_code = run_mesh_rank(rank, cfg, addrs, std::move(rig.lanes),
                                       rig.stream, rig.clock_khz, listener);
  cudaStreamDestroy(rig.stream);
  return exit_code;
}

int run_pair_mode(const std::string& peer_host, uint16_t peer_port,
                  const std::vector<std::string>& lane_hints, int iters,
                  int deadline_s) {
  const MeshAddr peer{"pair", peer_host, peer_port};
  const Deadline deadline(deadline_s);
  int exit_code = 0;
  for (const std::string& hint : lane_hints) {
    std::string error;
    RankRig rig = construct_rank({hint});
    std::string connect_error;
    TcpConn conn = connect_until(peer, deadline, &connect_error);
    if (!conn.valid()) {
      DGPP_LOG_ERROR("pair: lane {}: connect: {}", hint, connect_error);
      exit_code = 1;
      continue;
    }
    ProbeHello hello{};
    hello.magic = kHelloMagic;
    hello.version = kFrameVersion;
    hello.sender_rank = -1;
    hello.receiver_rank = -1;
    hello.iters = iters;
    hello.lane_index = 0;
    std::snprintf(hello.lane_name, sizeof(hello.lane_name), "%s",
                  hint.c_str());
    const ProbeOutcome probe = run_probe_sender(
        conn, *rig.lanes[0], hello,
        std::min(deadline.remaining_ms(), 60000));
    DGPP_LOG_INFO("pair: lane {}: {}/{} verified {}",
                  hint, probe.verified, iters,
                  probe.ok ? "PASS" : "FAIL: " + probe.error);
    if (!probe.ok) exit_code = 1;
  }
  return exit_code;
}

int run_serve_mode(const std::vector<std::string>& lane_hints, uint16_t port,
                   int idle_s) {
  RankRig rig = construct_rank(lane_hints);
  TcpListener listener = TcpListener::bind(port);
  const Deadline idle(idle_s);
  DGPP_LOG_INFO("serve: {} lanes, listening on {} (idle exit {}s)",
                lane_hints.size(), port, idle_s);
  int exit_code = 0;
  for (;;) {
    TcpConn conn = listener.accept(idle.remaining_ms());
    if (!conn.valid()) {
      if (idle.expired()) {
        DGPP_LOG_INFO("serve: idle timeout");
        break;
      }
      continue;
    }
    conn.set_io_deadline_ms(kIoMs);
    ProbeHello hello{};
    if (!conn.read_exact(&hello, sizeof(hello)) ||
        hello.magic != kHelloMagic || hello.version != kFrameVersion ||
        hello.iters < 1 || hello.lane_index < 0 ||
        hello.lane_index >= static_cast<int>(rig.lanes.size())) {
      DGPP_LOG_ERROR("serve: refusing malformed probe hello");
      ProbeAck refuse{};
      refuse.magic = kAckMagic;
      refuse.ok = 0;
      std::snprintf(refuse.reason, sizeof(refuse.reason), "%s",
                    "malformed hello");
      conn.write_all(&refuse, sizeof(refuse));
      conn.close();
      continue;
    }
    const int lane_index = hello.lane_index;
    // The probe function sends the acceptance ack itself (and the QP swap
    // follows immediately).
    const ProbeOutcome probe = run_probe_receiver(
        conn, *rig.lanes[static_cast<size_t>(lane_index)], rig.stream,
        rig.clock_khz, hello.iters, kIoMs);
    DGPP_LOG_INFO("serve: lane {} ({}): {}/{} verified (receiver p50 {:.1f}us "
                  "p99 {:.1f}us){}",
                  lane_index, hello.lane_name, probe.verified, hello.iters,
                  probe.p50_us, probe.p99_us,
                  probe.ok ? "" : " FAIL: " + probe.error);
    if (!probe.ok) exit_code = 1;
    conn.close();
  }
  cudaStreamDestroy(rig.stream);
  return exit_code;
}

// The selftest runs the full mesh machinery — schedule, obligations, held
// connections, aggregation, verdict — as a world=2 mesh whose two ranks are
// threads on this one device. Construction happens before the threads (the
// M5 bring-up discipline); the run phase only launches kernels and syncs its
// own stream.
int run_selftest(int iters, const std::vector<std::string>& lane_hints) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    return 2;  // ctest: skip, no GPU
  if (lane_hints.size() < 1) return 2;

  MeshConfig cfg;
  cfg.lanes = lane_hints;
  cfg.iters = iters;
  cfg.deadline_s = kSelftestDeadlineS;

  std::vector<TcpListener> listeners;
  std::vector<MeshAddr> addrs;
  for (int r = 0; r < 2; ++r) {
    listeners.push_back(TcpListener::bind(0));
    addrs.push_back({"rank" + std::to_string(r), "127.0.0.1",
                     listeners[static_cast<size_t>(r)].port()});
  }

  std::vector<RankRig> rigs;
  for (int r = 0; r < 2; ++r) rigs.push_back(construct_rank(cfg.lanes));

  std::atomic<int> exits[2] = {0, 0};
  std::vector<std::thread> threads;
  for (int r = 0; r < 2; ++r) {
    threads.emplace_back([&, r] {
      try {
        exits[r] = run_mesh_rank(r, cfg, addrs, std::move(rigs[static_cast<size_t>(r)].lanes),
                                 rigs[static_cast<size_t>(r)].stream,
                                 rigs[static_cast<size_t>(r)].clock_khz,
                                 listeners[static_cast<size_t>(r)]);
      } catch (const std::exception& e) {
        DGPP_LOG_ERROR("selftest rank {}: {}", r, e.what());
        exits[r] = 1;
      }
    });
  }
  for (std::thread& t : threads) t.join();

  for (RankRig& rig : rigs) {
    cudaStreamDestroy(rig.stream);
  }
  const int verdict = exits[0].load() != 0 || exits[1].load() != 0 ? 1 : 0;
  DGPP_LOG_INFO("selftest: {} (ranks exited {} {})", verdict == 0 ? "PASS" : "FAIL",
                exits[0].load(), exits[1].load());
  return verdict;
}

// ---- main -------------------------------------------------------------------

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> out;
  std::string current;
  for (const char c : text) {
    if (c == ',') {
      if (!current.empty()) out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

bool parse_long(const std::string& text, long lo, long hi, long* out) {
  char* end = nullptr;
  const long v = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || v < lo || v > hi) return false;
  *out = v;
  return true;
}

bool parse_peer(const std::string& text, std::string* host, uint16_t* port) {
  const size_t colon = text.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == text.size())
    return false;
  long p = 0;
  if (!parse_long(text.substr(colon + 1), 1, 65535, &p)) return false;
  *host = text.substr(0, colon);
  *port = static_cast<uint16_t>(p);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  nic_regress mesh --nodes IP[,IP...] [--port N] "
                 "[--lanes D,D] [--iters N] [--deadline-s N]\n"
                 "  nic_regress pair --peer HOST:PORT [--lanes D,D] "
                 "[--iters N] [--deadline-s N]\n"
                 "  nic_regress serve [--port N] [--lanes D,D] [--idle-s N]\n"
                 "  nic_regress selftest [--iters N] [--lanes D,D]\n"
                 "defaults: lanes rocep1s0f0,roceP2p1s0f0; iters 10000\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode != "mesh" && mode != "pair" && mode != "serve" &&
      mode != "selftest") {
    DGPP_LOG_ERROR("unknown mode {}", mode);
    return 2;
  }

  std::vector<std::string> nodes;
  std::vector<std::string> lanes = kDefaultLanes;
  std::string peer;
  long iters = kDefaultIters;
  long port = kDefaultPort;
  long deadline_s = kMeshDeadlineS;
  long idle_s = kServeIdleS;
  bool args_ok = true;
  bool have_peer = false;
  // The selftest exercises machinery, not the project-test strength — a
  // small default keeps CI fast; --iters overrides it like everywhere else.
  if (mode == "selftest") iters = kSelftestIters;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        args_ok = false;
        return {};
      }
      return argv[++i];
    };
    long n = 0;
    if (a == "--nodes") {
      nodes = split_csv(val());
      if (nodes.size() < 2) args_ok = false;
    } else if (a == "--lanes") {
      lanes = split_csv(val());
      if (lanes.empty() || lanes.size() > 8) args_ok = false;
    } else if (a == "--iters") {
      if (!parse_long(val(), 1, 1000000, &n)) args_ok = false;
      else iters = n;
    } else if (a == "--port") {
      if (!parse_long(val(), 1024, 65535, &n)) args_ok = false;
      else port = n;
    } else if (a == "--peer") {
      peer = val();
      have_peer = !peer.empty();
    } else if (a == "--deadline-s") {
      if (!parse_long(val(), 10, 86400, &n)) args_ok = false;
      else deadline_s = n;
    } else if (a == "--idle-s") {
      if (!parse_long(val(), 10, 86400, &n)) args_ok = false;
      else idle_s = n;
    } else {
      DGPP_LOG_ERROR("unexpected argument {}", a);
      args_ok = false;
    }
  }
  if (!args_ok) return 2;
  if (mode == "mesh" && nodes.size() < 2) {
    DGPP_LOG_ERROR("mesh: --nodes needs at least two comma-separated hosts");
    return 2;
  }
  if (mode == "pair" && !have_peer) {
    DGPP_LOG_ERROR("pair: --peer HOST:PORT is required");
    return 2;
  }

  try {
    if (mode == "mesh") {
      MeshConfig cfg;
      cfg.nodes = nodes;
      cfg.lanes = lanes;
      cfg.iters = static_cast<int>(iters);
      cfg.deadline_s = static_cast<int>(deadline_s);
      return run_mesh_mode(cfg, static_cast<uint16_t>(port));
    }
    if (mode == "pair") {
      std::string host;
      uint16_t peer_port = 0;
      if (!parse_peer(peer, &host, &peer_port)) {
        DGPP_LOG_ERROR("pair: --peer must be HOST:PORT");
        return 2;
      }
      return run_pair_mode(host, peer_port, lanes, static_cast<int>(iters),
                          static_cast<int>(deadline_s));
    }
    if (mode == "serve")
      return run_serve_mode(lanes, static_cast<uint16_t>(port),
                            static_cast<int>(idle_s));
    return run_selftest(static_cast<int>(iters), lanes);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("{}: {}", mode, e.what());
    return 1;
  }
}
