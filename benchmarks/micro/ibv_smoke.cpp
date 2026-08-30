// micro_ibv_smoke: RC QP plumbing + latency/bandwidth probe over RoCEv2.
// Protocol: TCP rendezvous carries role header; data moves via SEND into a
// pre-posted receive ring that both sides backfill 1-for-1 as WCs complete
// (no synchronous re-post races, no RDMA address exchange needed).
//
// Modes:
//   info                        list verbs devices/ports
//   serve <port> [--dev D]      listener: PING-echo or BW-sink per header;
//                               add --once for one connection then exit
//   ping --peer IP:PORT         N small-message RTTs
//   bw --peer IP:PORT [-s N]    SEND-flood BW test; repeat --peer and --dev
//                               in matching order to stripe fabrics
#include <infiniband/verbs.h>

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/log.hpp"
#include "net/nic_visibility.hpp"

namespace {

struct XchgInfo {
  uint32_t qpn;
  uint32_t psn;
  uint16_t lid;
  uint8_t gid[16];
};

// TCP header exchanged BEFORE QP-info swap; decides serving behavior.
struct RoleHeader {
  int32_t kind;    // 1=ping, 2=bw, 3=NIC-DMA-to-GPU visibility
  int32_t count;   // messages expected
  int32_t msgsz;   // message bytes (bw); ignored for ping
};

constexpr int kMaxWr = 2048;
constexpr int kDefaultIters = 200;
constexpr int kRingSlots = 256;  // receive-ring depth used everywhere
constexpr size_t kBwBufferBudget = 96ull << 20;
constexpr size_t kMaxMessageBytes = 8ull << 20;
constexpr size_t kVerifyPayloadOffset = 0;
constexpr size_t kVerifyStartOffset = 4096;
constexpr size_t kVerifyAckOffset = 8192;
constexpr size_t kVerifyPayloadBytes = 16 * sizeof(uint32_t);

struct VerifyResult {
  uint32_t seq;
  uint32_t ok;
  uint64_t observed_hash;
  uint64_t expected_hash;
};

int mtu_to_val(int e) {
  switch (e) {
    case IBV_MTU_512: return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
  }
  return -1;
}

struct DevPort {
  ibv_context* ctx{};
  ibv_pd* pd{};
  ibv_cq* rx_cq{};
  ibv_cq* tx_cq{};
  ibv_qp* qp{};
  ibv_mr* mr{};
  void* buf{};
  size_t buf_bytes{};
  bool cuda_pinned{};
  int port{1};
  int gid_idx{0};
  ibv_gid gid{};

  DevPort() = default;
  DevPort(const DevPort&) = delete;
  DevPort& operator=(const DevPort&) = delete;
  DevPort(DevPort&& other) noexcept { *this = std::move(other); }
  DevPort& operator=(DevPort&& other) noexcept {
    if (this == &other) return *this;
    reset();
    ctx = std::exchange(other.ctx, nullptr);
    pd = std::exchange(other.pd, nullptr);
    rx_cq = std::exchange(other.rx_cq, nullptr);
    tx_cq = std::exchange(other.tx_cq, nullptr);
    qp = std::exchange(other.qp, nullptr);
    mr = std::exchange(other.mr, nullptr);
    buf = std::exchange(other.buf, nullptr);
    buf_bytes = std::exchange(other.buf_bytes, 0);
    cuda_pinned = std::exchange(other.cuda_pinned, false);
    port = other.port;
    gid_idx = other.gid_idx;
    gid = other.gid;
    return *this;
  }
  ~DevPort() { reset(); }

  void reset() noexcept {
    if (qp) ibv_destroy_qp(qp);
    if (rx_cq) ibv_destroy_cq(rx_cq);
    if (tx_cq) ibv_destroy_cq(tx_cq);
    if (mr) ibv_dereg_mr(mr);
    if (cuda_pinned)
      cudaFreeHost(buf);
    else
      std::free(buf);
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
    ctx = nullptr;
    pd = nullptr;
    rx_cq = nullptr;
    tx_cq = nullptr;
    qp = nullptr;
    mr = nullptr;
    buf = nullptr;
    buf_bytes = 0;
    cuda_pinned = false;
  }
};

bool sock_send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n > 0) {
    ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return false;
    c += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

std::string sock_recv_exact(int fd, size_t n) {
  std::string s(n, '\0');
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd, s.data() + got, n - got, 0);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return {};
    got += static_cast<size_t>(r);
  }
  return s;
}

bool parse_long_value(const std::string& text, long minimum, long maximum,
                      long* result) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < minimum ||
      value > maximum)
    return false;
  *result = value;
  return true;
}

bool parse_size_value(const std::string& text, size_t minimum, size_t maximum,
                      size_t* result) {
  if (text.empty() || text.front() == '-') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < minimum ||
      value > maximum)
    return false;
  *result = static_cast<size_t>(value);
  return true;
}

bool parse_peer_endpoint(const std::string& text,
                         std::pair<std::string, uint16_t>* result) {
  const size_t colon = text.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == text.size() ||
      text.find(':', colon + 1) != std::string::npos)
    return false;
  long port = 0;
  if (!parse_long_value(text.substr(colon + 1), 1, 65535, &port)) return false;
  *result = {text.substr(0, colon), static_cast<uint16_t>(port)};
  return true;
}

bool set_socket_timeouts(int fd) {
  constexpr time_t kTimeoutSeconds = 130;
  timeval timeout{kTimeoutSeconds, 0};
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
      setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
    DGPP_LOG_ERROR("socket timeout setup failed errno={}", errno);
    return false;
  }
  return true;
}

int pick_rocev2_gid(ibv_context* ctx, int port) {
  std::string base = "/sys/class/infiniband/";
  base += ctx->device->name;
  base += "/ports/" + std::to_string(port) + "/gid_attrs/types/";
  for (int i = 0; i < 64; ++i) {
    FILE* f = fopen((base + std::to_string(i)).c_str(), "r");
    if (!f) continue;
    char buf[32] = {};
    size_t r = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (r == 0) continue;
    for (char* c = buf; *c; ++c) {
      *c = static_cast<char>(tolower(static_cast<unsigned char>(*c)));
    }
    if (strstr(buf, "v2")) return i;
  }
  return 0;
}

DevPort setup(const std::string& dev_pref, int gid_override,
              size_t buf_bytes, bool cuda_pinned = false) {
  DevPort d;
  d.buf_bytes = buf_bytes;
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (!list || num == 0) {
    DGPP_LOG_ERROR("no verbs devices");
    if (list) ibv_free_device_list(list);
    std::exit(1);
  }
  for (int i = 0; i < num && !d.ctx; ++i) {
    if (!dev_pref.empty() && dev_pref != list[i]->name &&
        strstr(list[i]->name, dev_pref.c_str()) == nullptr)
      continue;
    ibv_context* c = ibv_open_device(list[i]);
    if (!c) continue;
    ibv_device_attr attr{};
    if (ibv_query_device(c, &attr) != 0) {
      ibv_close_device(c);
      continue;
    }
    bool ok = false;
    for (int p = 1; p <= attr.phys_port_cnt; ++p) {
      ibv_port_attr pa{};
      if (ibv_query_port(c, p, &pa) == 0 &&
          pa.state == IBV_PORT_ACTIVE &&
          pa.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        d.port = p;
        ok = true;
        break;
      }
    }
    if (!ok) {
      ibv_close_device(c);
      continue;
    }
    d.ctx = c;
  }
  ibv_free_device_list(list);
  if (!d.ctx) {
    DGPP_LOG_ERROR("no ACTIVE RoCE device matching \"{}\"", dev_pref);
    std::exit(1);
  }
  DGPP_LOG_INFO("device {} port {}", d.ctx->device->name, d.port);

  d.gid_idx =
      gid_override >= 0 ? gid_override : pick_rocev2_gid(d.ctx, d.port);
  if (ibv_query_gid(d.ctx, d.port, d.gid_idx, &d.gid) != 0) {
    DGPP_LOG_ERROR("query_gid failed idx={}", d.gid_idx);
    std::exit(1);
  }

  d.pd = ibv_alloc_pd(d.ctx);
  if (!d.pd) {
    DGPP_LOG_ERROR("alloc_pd failed errno={}", errno);
    std::exit(1);
  }
  d.cuda_pinned = cuda_pinned;
  if (cuda_pinned) {
    if (cudaHostAlloc(&d.buf, buf_bytes, cudaHostAllocDefault) != cudaSuccess)
      d.buf = nullptr;
  } else {
    d.buf = aligned_alloc(4096, ((buf_bytes + 4095) / 4096) * 4096);
  }
  if (!d.buf) {
    DGPP_LOG_ERROR("buffer allocation failed bytes={}", buf_bytes);
    std::exit(1);
  }
  memset(d.buf, 0xAB, buf_bytes);
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
               IBV_ACCESS_REMOTE_READ;
  d.mr = ibv_reg_mr(d.pd, d.buf, buf_bytes, access);
  if (!d.mr) {
    DGPP_LOG_ERROR("reg_mr failed errno={}", errno);
    std::exit(1);
  }
  d.rx_cq = ibv_create_cq(d.ctx, kMaxWr, nullptr, nullptr, 0);
  d.tx_cq = ibv_create_cq(d.ctx, kMaxWr, nullptr, nullptr, 0);
  if (!d.rx_cq || !d.tx_cq) {
    DGPP_LOG_ERROR("create_cq failed errno={}", errno);
    std::exit(1);
  }
  ibv_qp_init_attr qi{};
  qi.send_cq = d.tx_cq;
  qi.recv_cq = d.rx_cq;
  qi.qp_type = IBV_QPT_RC;
  qi.cap.max_send_wr = kMaxWr;
  qi.cap.max_recv_wr = kMaxWr;
  qi.cap.max_send_sge = 1;
  qi.cap.max_recv_sge = 1;
  qi.cap.max_inline_data = 0;
  d.qp = ibv_create_qp(d.pd, &qi);
  if (!d.qp) {
    DGPP_LOG_ERROR("create_qp failed errno={}", errno);
    std::exit(1);
  }
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.port_num = static_cast<uint8_t>(d.port);
  a.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;
  if (ibv_modify_qp(d.qp, &a, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
                                  IBV_QP_ACCESS_FLAGS)) {
    DGPP_LOG_ERROR("qp->INIT failed errno={}", errno);
    std::exit(1);
  }
  return d;
}

bool ready_to_rts(DevPort& d, const XchgInfo& remote) {
  ibv_port_attr pa{};
  if (ibv_query_port(d.ctx, d.port, &pa) != 0) {
    DGPP_LOG_ERROR("query_port failed errno={}", errno);
    return false;
  }

  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = std::min(pa.active_mtu, IBV_MTU_4096);
  a.dest_qp_num = remote.qpn;
  a.rq_psn = remote.psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;
  a.ah_attr.is_global = 1;
  memcpy(a.ah_attr.grh.dgid.raw, remote.gid, 16);
  a.ah_attr.grh.sgid_index = static_cast<uint8_t>(d.gid_idx);
  a.ah_attr.grh.hop_limit = 255;
  a.ah_attr.sl = 0;
  a.ah_attr.port_num = static_cast<uint8_t>(d.port);
  if (ibv_modify_qp(
          d.qp, &a,
          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
              IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
    DGPP_LOG_ERROR("qp->RTR failed errno={}", errno);
    return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;
  a.sq_psn = 0x1234;
  a.max_rd_atomic = 1;
  if (ibv_modify_qp(d.qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                  IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                  IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
    DGPP_LOG_ERROR("qp->RTS failed errno={}", errno);
    return false;
  }
  return true;
}

XchgInfo local_info(const DevPort& d) {
  XchgInfo x{};
  x.qpn = d.qp->qp_num;
  x.psn = 0x1234;
  memcpy(x.gid, d.gid.raw, 16);
  return x;
}

void check_wc_ok(const ibv_wc& wc, const char* where) {
  if (wc.status != IBV_WC_SUCCESS) {
    DGPP_LOG_ERROR("{}: bad wc status={} opcode={} vend={}", where,
                   static_cast<int>(wc.status), static_cast<int>(wc.opcode),
                   wc.vendor_err);
    std::exit(1);
  }
}

void post_ring_recvs(DevPort& d, int slots, size_t each_len) {
  for (int s = 0; s < slots; ++s) {
    ibv_sge sg{};
    sg.addr = reinterpret_cast<uintptr_t>(d.mr->addr) + s * each_len;
    sg.lkey = d.mr->lkey;
    sg.length = static_cast<uint32_t>(each_len);
    ibv_recv_wr rr{};
    rr.wr_id = static_cast<uint64_t>(s);
    rr.sg_list = &sg;
    rr.num_sge = 1;
    ibv_recv_wr* bad = nullptr;
    if (ibv_post_recv(d.qp, &rr, &bad)) {
      DGPP_LOG_ERROR("post_recv slot {} failed", s);
      std::exit(1);
    }
  }
}

void refill_recv_slot(DevPort& d, uint64_t wr_id, size_t each_len,
                      int ring_slots) {
  int s = static_cast<int>(wr_id % static_cast<uint64_t>(ring_slots));
  ibv_sge sg{};
  sg.addr = reinterpret_cast<uintptr_t>(d.mr->addr) +
            static_cast<size_t>(s) * each_len;
  sg.lkey = d.mr->lkey;
  sg.length = static_cast<uint32_t>(each_len);
  ibv_recv_wr rr{};
  rr.wr_id = wr_id;
  rr.sg_list = &sg;
  rr.num_sge = 1;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(d.qp, &rr, &bad)) {
    DGPP_LOG_ERROR("refill recv failed");
    std::exit(1);
  }
}

bool post_recv_at(DevPort& d, void* address, size_t bytes, uint64_t wr_id) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(address);
  sge.lkey = d.mr->lkey;
  sge.length = static_cast<uint32_t>(bytes);
  ibv_recv_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  ibv_recv_wr* bad = nullptr;
  if (ibv_post_recv(d.qp, &wr, &bad) != 0) {
    DGPP_LOG_ERROR("verify post_recv failed errno={}", errno);
    return false;
  }
  return true;
}

uint64_t verify_payload_hash(const uint32_t* payload) {
  uint64_t hash = 0;
  for (int word = 0; word < 16; word += 2) {
    const uint64_t pair = (static_cast<uint64_t>(payload[word]) << 32) |
                          static_cast<uint64_t>(payload[word + 1]);
    hash ^= pair * 0x9E3779B97F4A7C15ull;
  }
  return hash;
}

void fill_verify_payload(uint32_t* payload, uint32_t seq) {
  for (uint32_t word = 0; word < 16; ++word)
    payload[word] = seq * 2654435761u + word;
}

bool wait_one_tx(DevPort& d, std::chrono::seconds timeout,
                 const char* where = "tx") {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    ibv_wc completion{};
    const int count = ibv_poll_cq(d.tx_cq, 1, &completion);
    if (count < 0) {
      DGPP_LOG_ERROR("{} poll failed", where);
      return false;
    }
    if (count == 1) {
      check_wc_ok(completion, where);
      return true;
    }
    if (std::chrono::steady_clock::now() > deadline) {
      DGPP_LOG_ERROR("{} completion timeout", where);
      return false;
    }
    std::this_thread::yield();
  }
}

bool post_verify_pair(DevPort& d, const uint32_t* payload,
                      const dgpp::StartSlot* start, uint64_t wr_id) {
  ibv_sge sge[2]{};
  sge[0].addr = reinterpret_cast<uintptr_t>(payload);
  sge[0].lkey = d.mr->lkey;
  sge[0].length = kVerifyPayloadBytes;
  sge[1].addr = reinterpret_cast<uintptr_t>(start);
  sge[1].lkey = d.mr->lkey;
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
  if (ibv_post_send(d.qp, &wr[0], &bad) != 0) {
    DGPP_LOG_ERROR("verify post_send pair failed errno={}", errno);
    return false;
  }
  return wait_one_tx(d, std::chrono::seconds(5), "verify-tx");
}

bool wait_verify_receives(DevPort& d) {
  int received = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (received < 2) {
    ibv_wc completions[2]{};
    const int count = ibv_poll_cq(d.rx_cq, 2, completions);
    if (count < 0) return false;
    for (int index = 0; index < count; ++index) {
      check_wc_ok(completions[index], "verify-rx");
      if (completions[index].byte_len != kVerifyPayloadBytes &&
          completions[index].byte_len != sizeof(dgpp::StartSlot)) {
        DGPP_LOG_ERROR("verify rx unexpected byte_len={}",
                       completions[index].byte_len);
        return false;
      }
    }
    received += count;
    if (count == 0) {
      if (std::chrono::steady_clock::now() > deadline) return false;
      std::this_thread::yield();
    }
  }
  return true;
}

int cmd_info() {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (!list) {
    DGPP_LOG_ERROR("get_device_list failed errno={}", errno);
    return 1;
  }
  for (int i = 0; i < num; ++i) {
    ibv_context* c = ibv_open_device(list[i]);
    if (!c) continue;
    ibv_device_attr da{};
    if (ibv_query_device(c, &da)) {
      ibv_close_device(c);
      continue;
    }
    for (int p = 1; p <= da.phys_port_cnt; ++p) {
      ibv_port_attr pa{};
      if (ibv_query_port(c, p, &pa)) continue;
      DGPP_LOG_INFO("dev={} port={} state={}({}) active_mtu={}", list[i]->name,
                    p, ibv_port_state_str(pa.state),
                    pa.state == IBV_PORT_ACTIVE ? "ok" : "-",
                    mtu_to_val(pa.active_mtu));
    }
    ibv_close_device(c);
  }
  ibv_free_device_list(list);
  return 0;
}

int tcp_listen_and_accept(uint16_t port) {
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  if (lfd < 0) {
    DGPP_LOG_ERROR("socket failed errno={}", errno);
    return -1;
  }
  int one = 1;
  if (setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
    DGPP_LOG_ERROR("setsockopt failed errno={}", errno);
    close(lfd);
    return -1;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
      listen(lfd, 2) != 0) {
    DGPP_LOG_ERROR("bind/listen on {} failed errno={}", port, errno);
    close(lfd);
    return -1;
  }
  DGPP_LOG_INFO("waiting rendezvous on :{}", port);
  int fd = accept(lfd, nullptr, nullptr);
  if (fd < 0) {
    DGPP_LOG_ERROR("accept failed errno={}", errno);
  } else if (!set_socket_timeouts(fd)) {
    close(fd);
    fd = -1;
  }
  close(lfd);
  return fd;
}

int tcp_connect(const std::string& host, uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    DGPP_LOG_ERROR("socket failed errno={}", errno);
    return -1;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    DGPP_LOG_ERROR("invalid numeric peer address {}", host);
    close(fd);
    return -1;
  }
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    DGPP_LOG_ERROR("connect {}:{} failed errno={}", host, port, errno);
    close(fd);
    return -1;
  }
  if (!set_socket_timeouts(fd)) {
    close(fd);
    return -1;
  }
  return fd;
}

// Establishes role-specific buffers then swaps XchgInfo.
bool handshake_common(int fd, const RoleHeader& hdr, const std::string& dev,
                      int gid_idx, DevPort* result) {
  size_t buf;
  if (hdr.kind == 2) {
    // Sink needs room for a deep ring of large messages.
    buf = std::min<size_t>(kBwBufferBudget,
                           static_cast<size_t>(kRingSlots) * hdr.msgsz);
  } else {
    buf = 16 << 20;
  }
  DevPort d = setup(dev, gid_idx, buf, hdr.kind == 3);

  XchgInfo me = local_info(d);
  if (!sock_send_all(fd, &me, sizeof(me))) {
    DGPP_LOG_ERROR("handshake write failed");
    return false;
  }
  auto r = sock_recv_exact(fd, sizeof(XchgInfo));
  if (r.size() != sizeof(XchgInfo)) {
    DGPP_LOG_ERROR("handshake short read");
    return false;
  }
  XchgInfo remote{};
  memcpy(&remote, r.data(), sizeof(remote));
  if (!ready_to_rts(d, remote)) return false;
  *result = std::move(d);
  return true;
}

}  // namespace

namespace {

int run_visibility_responder(int fd, const RoleHeader& hdr,
                             const std::string& dev, int gid_idx) {
  DevPort d;
  if (!handshake_common(fd, hdr, dev, gid_idx, &d)) return 1;
  auto* bytes = static_cast<unsigned char*>(d.buf);
  auto* payload = reinterpret_cast<uint32_t*>(bytes + kVerifyPayloadOffset);
  auto* start = reinterpret_cast<dgpp::StartSlot*>(bytes + kVerifyStartOffset);
  auto* ack = reinterpret_cast<dgpp::FlagAck*>(bytes + kVerifyAckOffset);
  *start = {};
  *ack = {};
  std::fill_n(payload, 16, 0u);

  int clock_khz = 0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) !=
          cudaSuccess ||
      clock_khz <= 0) {
    DGPP_LOG_ERROR("verify could not read GPU clock");
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return 1;
  const uint64_t deadline_cycles =
      static_cast<uint64_t>(clock_khz) * 5000ull;
  const cudaError_t launch = dgpp::net::launch_nic_visibility_kernel(
      &start->seq, payload, ack, deadline_cycles, stream);
  if (launch != cudaSuccess) {
    DGPP_LOG_ERROR("verify kernel launch failed: {}",
                   cudaGetErrorString(launch));
    cudaStreamDestroy(stream);
    return 1;
  }

  int verified = 0;
  for (uint32_t seq = 1; seq <= static_cast<uint32_t>(hdr.count); ++seq) {
    const bool posted =
        post_recv_at(d, payload, kVerifyPayloadBytes, 2ull * seq) &&
        post_recv_at(d, start, sizeof(*start), 2ull * seq + 1);
    const char ready = 'R';
    if (!posted || !sock_send_all(fd, &ready, 1)) break;

    bool observed = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (__atomic_load_n(&ack->seq, __ATOMIC_ACQUIRE) != seq) {
      if (std::chrono::steady_clock::now() > deadline) break;
      std::this_thread::yield();
    }
    observed = __atomic_load_n(&ack->seq, __ATOMIC_ACQUIRE) == seq;

    uint32_t expected_payload[16]{};
    fill_verify_payload(expected_payload, seq);
    const uint64_t expected_hash = verify_payload_hash(expected_payload);
    const uint64_t observed_hash = observed ? ack->hash : 0;
    const bool receives_ok = wait_verify_receives(d);
    VerifyResult result{seq,
                        observed && receives_ok && observed_hash == expected_hash,
                        observed_hash, expected_hash};
    if (!sock_send_all(fd, &result, sizeof(result))) break;
    if (!result.ok) {
      DGPP_LOG_ERROR(
          "NIC-GPU visibility failure seq={} observed={:#x} expected={:#x} "
          "rx_ok={}",
          seq, observed_hash, expected_hash, receives_ok);
      break;
    }
    ++verified;
  }

  __atomic_store_n(&start->seq, dgpp::kFlagStopSequence, __ATOMIC_RELEASE);
  const cudaError_t sync = cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  if (sync != cudaSuccess) {
    DGPP_LOG_ERROR("verify kernel completion failed: {}",
                   cudaGetErrorString(sync));
    return 1;
  }
  DGPP_LOG_INFO("NIC-GPU-VERIFY server dev={} passed={}/{}", dev, verified,
                hdr.count);
  return verified == hdr.count ? 0 : 1;
}

int run_responder(uint16_t tcp_port, const std::string& dev, int gid_idx,
                  bool once) {
  for (;;) {  // sequential connections (two possible: fabric0/fabric1)
    int fd = tcp_listen_and_accept(tcp_port);
    if (fd < 0) return 1;
    RoleHeader hdr{-1, 0, 0};
    auto hr = sock_recv_exact(fd, sizeof(hdr));
    if (hr.size() != sizeof(hdr)) {
      close(fd);
      continue;
    }
    memcpy(&hdr, hr.data(), sizeof(hdr));

    if (hdr.count <= 0 ||
        (hdr.kind == 2 &&
         (hdr.msgsz <= 0 ||
          static_cast<size_t>(hdr.msgsz) > kMaxMessageBytes))) {
      DGPP_LOG_ERROR("invalid request kind={} count={} size={}", hdr.kind,
                     hdr.count, hdr.msgsz);
      close(fd);
      if (once) return 1;
      continue;
    }

    int connection_rc = 0;
    if (hdr.kind == 1) {  // ping responder: echo everything `count` times
      DevPort d;
      if (!handshake_common(fd, hdr, dev, gid_idx, &d)) {
        close(fd);
        if (once) return 1;
        continue;
      }
      constexpr size_t kSlot = 8;
      int slots = kRingSlots;
      post_ring_recvs(d, slots, kSlot);
      DGPP_LOG_INFO("responder: echoing {} pings", hdr.count);
      int echoed = 0;
      auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (echoed < hdr.count) {
        ibv_wc rcw[16]{};
        int n = ibv_poll_cq(d.rx_cq, 16, rcw);
        if (n < 0) {
          DGPP_LOG_ERROR("ping receive poll failed");
          connection_rc = 1;
          break;
        }
        for (int q = 0; q < n; ++q) {
          check_wc_ok(rcw[q], "ping-rx");
          // Echo from inside our own MR (second half of the buffer).
          uint64_t out = 0x0DDBA11ull;
          memcpy(reinterpret_cast<char*>(d.mr->addr) + (8 << 20), &out,
                 sizeof(out));
          ibv_sge sg{};
          sg.addr = reinterpret_cast<uintptr_t>(d.mr->addr) + (8 << 20);
          sg.lkey = d.mr->lkey;
          sg.length = sizeof(out);
          ibv_send_wr sw{};
          sw.opcode = IBV_WR_SEND;
          sw.send_flags = IBV_SEND_SIGNALED;
          sw.sg_list = &sg;
          sw.num_sge = 1;
          ibv_send_wr* bad = nullptr;
          if (ibv_post_send(d.qp, &sw, &bad)) {
            DGPP_LOG_ERROR("echo post_send failed");
            connection_rc = 1;
            break;
          }
          refill_recv_slot(d, rcw[q].wr_id, kSlot, slots);
          if (!wait_one_tx(d, std::chrono::seconds(5), "ping-tx")) {
            connection_rc = 1;
            break;
          }
          ++echoed;
          deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(60);
        }
        if (connection_rc != 0) break;
        if (n == 0 && std::chrono::steady_clock::now() > deadline) {
          DGPP_LOG_ERROR("responder timeout at {}/{}", echoed, hdr.count);
          connection_rc = 1;
          break;
        }
        if (n == 0) std::this_thread::yield();
      }
      if (connection_rc == 0) DGPP_LOG_INFO("responder echoed {}", echoed);
    } else if (hdr.kind == 2) {  // BW sink
      DevPort d;
      if (!handshake_common(fd, hdr, dev, gid_idx, &d)) {
        close(fd);
        if (once) return 1;
        continue;
      }
      int slots = static_cast<int>(std::min<long long>(
          kRingSlots,
          std::max<long long>(1, kBwBufferBudget / hdr.msgsz)));
      post_ring_recvs(d, slots, hdr.msgsz);
      DGPP_LOG_INFO("sink: awaiting {} x {}B (ring {})", hdr.count, hdr.msgsz,
                    slots);
      auto start = std::chrono::steady_clock::now();
      long long rx = 0;
      auto deadline = start + std::chrono::seconds(120);
      while (rx < hdr.count) {
        ibv_wc rcw[128]{};
        int n = ibv_poll_cq(d.rx_cq, 128, rcw);
        if (n < 0) {
          DGPP_LOG_ERROR("sink receive poll failed");
          connection_rc = 1;
          break;
        }
        for (int q = 0; q < n; ++q) {
          check_wc_ok(rcw[q], "sink-rx");
          refill_recv_slot(d, rcw[q].wr_id, hdr.msgsz, slots);
        }
        rx += n;
        if (n > 0) {
          deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(120);
        } else if (std::chrono::steady_clock::now() > deadline) {
          DGPP_LOG_ERROR("sink timeout at {}/{}", rx, hdr.count);
          connection_rc = 1;
          break;
        }
        if (n == 0) std::this_thread::yield();
      }
      double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count();
      const bool complete = connection_rc == 0 && rx == hdr.count;
      if (!sock_send_all(fd, complete ? "ACK" : "NAK", 3) ||
          !sock_send_all(fd, &secs, sizeof(secs))) {
        DGPP_LOG_ERROR("sink result write failed");
        connection_rc = 1;
      }
      DGPP_LOG_INFO("SINK-DONE rx={} {:.3f}s eff_Gbps={:.1f}", rx, secs,
                    double(rx) * hdr.msgsz * 8 / secs / 1e9);
    } else if (hdr.kind == 3) {
      connection_rc = run_visibility_responder(fd, hdr, dev, gid_idx);
    } else {
      DGPP_LOG_ERROR("unknown header kind {}", hdr.kind);
      connection_rc = 1;
    }
    char bye[8];
    (void)::recv(fd, bye, sizeof(bye), MSG_DONTWAIT);
    close(fd);
    if (once) return connection_rc;
  }
  return 0;
}

int run_ping(const std::string& host, uint16_t port, int iters,
             const std::string& dev, int gid_idx) {
  int fd = tcp_connect(host, port);
  if (fd < 0) return 1;
  RoleHeader hdr{1, iters, 0};
  if (!sock_send_all(fd, &hdr, sizeof(hdr))) {
    close(fd);
    return 1;
  }
  DevPort d;
  if (!handshake_common(fd, hdr, dev, gid_idx, &d)) {
    close(fd);
    return 1;
  }
  constexpr size_t kSlot = 8;
  post_ring_recvs(d, kRingSlots, kSlot);

  double sum = 0, best = 1e18;
  int ok = 0;
  // Send payloads live INSIDE the registered region (VA must match the MR).
  uint64_t* payload =
      reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(d.mr->addr) +
                                  (12 << 20));
  for (int i = 0; i < iters; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    ibv_sge sg{};
    sg.addr = reinterpret_cast<uintptr_t>(payload);
    sg.lkey = d.mr->lkey;
    sg.length = 8;
    *payload = 0xC0FFEEull + static_cast<uint64_t>(i);
    ibv_send_wr sw{};
    sw.opcode = IBV_WR_SEND;
    sw.send_flags = IBV_SEND_SIGNALED;
    sw.sg_list = &sg;
    sw.num_sge = 1;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(d.qp, &sw, &bad)) {
      DGPP_LOG_ERROR("ping post_send failed errno={}", errno);
      break;
    }

    bool saw_tx = false, saw_rx = false;
    uint64_t rx_id = 0;
    auto deadline = t0 + std::chrono::seconds(5);
    while (!(saw_tx && saw_rx)) {
      ibv_wc twc{};
      int ntx = ibv_poll_cq(d.tx_cq, 1, &twc);
      if (ntx < 0) {
        DGPP_LOG_ERROR("ping tx poll failed");
        close(fd);
        return 1;
      }
      if (ntx > 0) {
        check_wc_ok(twc, "init-tx");
        saw_tx = true;
      }
      ibv_wc rcw[8]{};
      int nrx = ibv_poll_cq(d.rx_cq, 8, rcw);
      if (nrx < 0) {
        DGPP_LOG_ERROR("ping rx poll failed");
        close(fd);
        return 1;
      }
      for (int q = 0; q < nrx; ++q) {
        check_wc_ok(rcw[q], "init-rx");
        saw_rx = true;
        rx_id = rcw[q].wr_id;
      }
      if (!(saw_tx && saw_rx)) {
        if (std::chrono::steady_clock::now() > deadline) {
          DGPP_LOG_ERROR("ping timeout iter {} (tx={} rx={})", i, saw_tx,
                         saw_rx);
          close(fd);
          return 1;
        }
        std::this_thread::yield();
      }
    }
    refill_recv_slot(d, rx_id, kSlot, kRingSlots);
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                2.0;
    sum += us;
    best = std::min(best, us);
    ++ok;
  }
  DGPP_LOG_INFO("PING-COMPLETE dev={} iters={} oneway_min_us={:.2f} avg={:.2f}",
                d.ctx->device->name, ok, best, ok > 0 ? sum / ok : 0.0);
  sock_send_all(fd, "BYE", 3);
  close(fd);
  return ok == iters ? 0 : 1;
}

int run_visibility_verify(const std::string& host, uint16_t port, int iters,
                          const std::string& dev, int gid_idx) {
  int fd = tcp_connect(host, port);
  if (fd < 0) return 1;
  RoleHeader hdr{3, iters, static_cast<int32_t>(kVerifyPayloadBytes)};
  if (!sock_send_all(fd, &hdr, sizeof(hdr))) {
    close(fd);
    return 1;
  }
  DevPort d;
  if (!handshake_common(fd, hdr, dev, gid_idx, &d)) {
    close(fd);
    return 1;
  }
  auto* bytes = static_cast<unsigned char*>(d.buf);
  auto* payload = reinterpret_cast<uint32_t*>(bytes + kVerifyPayloadOffset);
  auto* start = reinterpret_cast<dgpp::StartSlot*>(bytes + kVerifyStartOffset);
  *start = {};

  int verified = 0;
  for (uint32_t seq = 1; seq <= static_cast<uint32_t>(iters); ++seq) {
    const auto ready = sock_recv_exact(fd, 1);
    if (ready != "R") {
      DGPP_LOG_ERROR("verify server did not become ready at seq={}", seq);
      break;
    }
    fill_verify_payload(payload, seq);
    *start = {};
    start->seq = seq;
    std::atomic_thread_fence(std::memory_order_release);
    if (!post_verify_pair(d, payload, start, 2ull * seq)) break;

    const auto response = sock_recv_exact(fd, sizeof(VerifyResult));
    if (response.size() != sizeof(VerifyResult)) break;
    VerifyResult result{};
    memcpy(&result, response.data(), sizeof(result));
    if (result.seq != seq || !result.ok) {
      DGPP_LOG_ERROR(
          "NIC-GPU verify rejected seq={} response_seq={} observed={:#x} "
          "expected={:#x}",
          seq, result.seq, result.observed_hash, result.expected_hash);
      break;
    }
    ++verified;
  }
  sock_send_all(fd, "BYE", 3);
  close(fd);
  DGPP_LOG_INFO("NIC-GPU-VERIFY client dev={} passed={}/{}", dev, verified,
                iters);
  return verified == iters ? 0 : 1;
}

int run_bw_link(const std::string& host, uint16_t port, size_t msg_size,
                int total, int window, const std::string& dev, int gid_idx,
                double* sink_secs_out) {
  int fd = tcp_connect(host, port);
  if (fd < 0) return 1;
  RoleHeader hdr{2, total, static_cast<int32_t>(msg_size)};
  if (!sock_send_all(fd, &hdr, sizeof(hdr))) {
    close(fd);
    return 1;
  }
  DevPort d;
  if (!handshake_common(fd, hdr, dev, gid_idx, &d)) {
    close(fd);
    return 1;
  }

  long long posted = 0, done_ = 0;
  uintptr_t src = reinterpret_cast<uintptr_t>(d.mr->addr) + (8ull << 20);
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (done_ < total) {
    while (posted - done_ < window && posted < total) {
      ibv_sge sg{};
      sg.addr = src;
      sg.lkey = d.mr->lkey;
      sg.length = static_cast<uint32_t>(msg_size);
      ibv_send_wr sw{};
      sw.wr_id = static_cast<uint64_t>(200000 + posted++);
      sw.opcode = IBV_WR_SEND;
      sw.send_flags = IBV_SEND_SIGNALED;
      sw.sg_list = &sg;
      sw.num_sge = 1;
      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(d.qp, &sw, &bad)) {
        DGPP_LOG_ERROR("bw post_send failed errno={}", errno);
        close(fd);
        return 1;
      }
    }
    ibv_wc wc[256]{};
    int n = ibv_poll_cq(d.tx_cq, 256, wc);
    if (n < 0) {
      DGPP_LOG_ERROR("bw poll_cq failed");
      close(fd);
      return 1;
    }
    for (int i = 0; i < n; ++i) check_wc_ok(wc[i], "bw-tx");
    if (n > 0) {
      deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(120);
    } else {
      if (std::chrono::steady_clock::now() > deadline) {
        DGPP_LOG_ERROR("bw send completion timeout at {}/{}", done_, total);
        close(fd);
        return 1;
      }
      std::this_thread::yield();
    }
    done_ += n;
  }

  auto ar = sock_recv_exact(fd, 3);
  if (ar != "ACK") {
    DGPP_LOG_ERROR("no ACK from sink");
    close(fd);
    return 1;
  }
  auto sr = sock_recv_exact(fd, sizeof(double));
  if (sr.size() != sizeof(double)) {
    DGPP_LOG_ERROR("short sink timing response");
    close(fd);
    return 1;
  }
  memcpy(sink_secs_out, sr.data(), sizeof(double));
  sock_send_all(fd, "BYE", 3);
  close(fd);
  return 0;
}

int run_bw(const std::vector<std::string>& peers,
           const std::vector<std::string>& devs, size_t msg_size, int window,
           int gid_idx) {
  const int total_per_link = 4000;
  std::vector<std::pair<std::string, uint16_t>> targets;
  for (auto& p : peers) {
    std::pair<std::string, uint16_t> endpoint;
    if (!parse_peer_endpoint(p, &endpoint)) {
      DGPP_LOG_ERROR("invalid peer {}; expected IP:PORT", p);
      return 2;
    }
    targets.push_back(std::move(endpoint));
  }

  std::vector<std::thread> ths;
  std::vector<double> sink_secs(targets.size(), 0.0);
  std::vector<int> link_rc(targets.size(), 1);
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < targets.size(); ++i) {
    ths.emplace_back([i, &targets, &devs, msg_size, window, total_per_link,
                      gid_idx, &sink_secs, &link_rc]() {
      const std::string dev = devs.empty()
                                  ? std::string{}
                                  : devs[devs.size() == 1 ? 0 : i];
      link_rc[i] = run_bw_link(targets[i].first, targets[i].second, msg_size,
                               total_per_link, window, dev, gid_idx,
                               &sink_secs[i]);
    });
  }
  for (auto& t : ths) t.join();
  auto t1 = std::chrono::steady_clock::now();

  double wall = std::chrono::duration<double>(t1 - t0).count();
  // Use per-link measured sink time (identical HW on both ends).
  double worst_sink = 0;
  for (double s : sink_secs) worst_sink = std::max(worst_sink, s);
  double eff = worst_sink > 0 ? worst_sink : wall;
  double total_bytes =
      double(total_per_link) * double(msg_size) * double(targets.size());
  DGPP_LOG_INFO(
      "BW-COMPLETE fabrics={} msg={}B wall={:.3f}s measured={:.3f}s "
      "throughput={:.1f} Gbps",
      targets.size(), msg_size, wall, eff, total_bytes * 8.0 / eff / 1e9);
  return std::any_of(link_rc.begin(), link_rc.end(),
                     [](int rc) { return rc != 0; })
             ? 1
             : 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  micro_ibv_smoke info\n"
                 "  micro_ibv_smoke serve <port> [--dev D] [--once]\n"
                 "  micro_ibv_smoke ping --peer IP:PORT [--iters N] [--dev D]"
                 " [--gid-index N]\n"
                 "  micro_ibv_smoke verify --peer IP:PORT [--iters N] "
                 "[--dev D] [--gid-index N]\n"
                 "  micro_ibv_smoke bw --peer IP:PORT [--msg-size N] "
                 "[--window N] [--dev D] [--gid-index N]\n"
                 "    repeat --peer and --dev in matching order for "
                 "multi-fabric BW\n");
    return 2;
  }
  std::string mode = argv[1];
  std::vector<std::string> peers;
  std::vector<std::string> devs;
  int gid_idx = -1;
  int iters = kDefaultIters;
  int window = 128;
  size_t msg_size = 262144;
  uint16_t port = 4791;
  bool once = false;
  bool args_ok = true;
  bool saw_server_port = false;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        args_ok = false;
        return {};
      }
      return argv[++i];
    };
    if (a == "--peer") {
      std::string value = val();
      if (value.empty()) args_ok = false;
      peers.push_back(std::move(value));
    } else if (a == "--dev") {
      std::string value = val();
      if (value.empty()) args_ok = false;
      devs.push_back(std::move(value));
    } else if (a == "--gid-index") {
      long value = 0;
      if (!parse_long_value(val(), 0, INT_MAX, &value))
        args_ok = false;
      else
        gid_idx = static_cast<int>(value);
    } else if (a == "--iters") {
      long value = 0;
      if (!parse_long_value(val(), 1, INT_MAX, &value))
        args_ok = false;
      else
        iters = static_cast<int>(value);
    } else if (a == "--window") {
      long value = 0;
      if (!parse_long_value(val(), 1, kMaxWr, &value))
        args_ok = false;
      else
        window = static_cast<int>(value);
    } else if (a == "--msg-size") {
      if (!parse_size_value(val(), 1, kMaxMessageBytes, &msg_size))
        args_ok = false;
    } else if (a == "--once") {
      once = true;
    } else if (mode == "serve" && !saw_server_port) {
      long value = 0;
      if (!parse_long_value(a, 1, 65535, &value))
        args_ok = false;
      else
        port = static_cast<uint16_t>(value);
      saw_server_port = true;
    } else {
      DGPP_LOG_ERROR("unexpected argument {}", a);
      args_ok = false;
    }
  }

  if (!args_ok) return 2;

  if (mode == "info") {
    if (argc != 2) return 2;
    return cmd_info();
  }
  if (mode == "serve") {
    if (devs.size() > 1 || !peers.empty() || !saw_server_port) return 2;
    return run_responder(port, devs.empty() ? std::string{} : devs[0],
                         gid_idx, once);
  }
  if (mode == "ping") {
    std::pair<std::string, uint16_t> endpoint;
    if (peers.size() != 1 || devs.size() > 1 || once ||
        !parse_peer_endpoint(peers[0], &endpoint))
      return 2;
    return run_ping(endpoint.first, endpoint.second, iters,
                    devs.empty() ? std::string{} : devs[0], gid_idx);
  }
  if (mode == "verify") {
    std::pair<std::string, uint16_t> endpoint;
    if (peers.size() != 1 || devs.size() > 1 || once ||
        !parse_peer_endpoint(peers[0], &endpoint))
      return 2;
    return run_visibility_verify(endpoint.first, endpoint.second, iters,
                                 devs.empty() ? std::string{} : devs[0],
                                 gid_idx);
  }
  if (mode == "bw") {
    if (peers.empty() || (devs.size() > 1 && devs.size() != peers.size()) ||
        once)
      return 2;
    return run_bw(peers, devs, msg_size, window, gid_idx);
  }
  DGPP_LOG_ERROR("unknown mode {}", mode);
  return 2;
}
