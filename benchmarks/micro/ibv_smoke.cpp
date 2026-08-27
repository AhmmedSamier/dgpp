// micro_ibv_smoke: RC QP plumbing + latency/bandwidth probe over RoCEv2.
// Protocol: TCP rendezvous carries role header; data moves via SEND into a
// pre-posted receive ring that both sides backfill 1-for-1 as WCs complete
// (no synchronous re-post races, no RDMA address exchange needed).
//
// Modes:
//   info                        list verbs devices/ports
//   serve <port> [--dev D]      listener: PING-echo or BW-sink per header
//   ping --peer IP:PORT         N small-message RTTs
//   bw --peer IP:PORT [-s N]    SEND-flood BW test; repeat --peer (per
//                               fabric) to stripe fabrics concurrently
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/log.hpp"

namespace {

struct XchgInfo {
  uint32_t qpn;
  uint32_t psn;
  uint16_t lid;
  uint8_t gid[16];
};

// TCP header exchanged BEFORE QP-info swap; decides serving behavior.
struct RoleHeader {
  int32_t kind;    // 1=ping(echo N msgs), 2=bw(sink N msgs)
  int32_t count;   // messages expected
  int32_t msgsz;   // message bytes (bw); ignored for ping
};

constexpr int kMaxWr = 2048;
constexpr int kDefaultIters = 200;
constexpr int kRingSlots = 256;  // receive-ring depth used everywhere

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
  int port{1};
  int gid_idx{0};
  ibv_gid gid{};
};

bool sock_send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n > 0) {
    ssize_t w = ::send(fd, c, n, 0);
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
    if (r <= 0) return {};
    got += static_cast<size_t>(r);
  }
  return s;
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
    for (char* c = buf; *c; ++c) *c = static_cast<char>(tolower(*c));
    if (strstr(buf, "v2")) return i;
  }
  return 0;
}

DevPort setup(const std::string& dev_pref, int gid_override,
              size_t buf_bytes, int slot_bytes) {
  DevPort d;
  d.buf_bytes = buf_bytes;
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  if (!list || num == 0) {
    DGPP_LOG_ERROR("no verbs devices");
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
  d.buf = aligned_alloc(4096, ((buf_bytes + 4095) / 4096) * 4096);
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
  ibv_qp_init_attr qi{};
  qi.send_cq = d.tx_cq;
  qi.recv_cq = d.rx_cq;
  qi.qp_type = IBV_QPT_RC;
  qi.cap.max_send_wr = kMaxWr;
  qi.cap.max_recv_wr = kMaxWr;
  qi.cap.max_send_sge = 1;
  qi.cap.max_recv_sge = 1;
  qi.cap.max_inline_data = static_cast<uint32_t>(slot_bytes);
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

void ready_to_rts(DevPort& d, const XchgInfo& remote) {
  ibv_port_attr pa{};
  ibv_query_port(d.ctx, d.port, &pa);

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
    std::exit(1);
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
    std::exit(1);
  }
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

int cmd_info() {
  int num = 0;
  ibv_device** list = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    ibv_context* c = ibv_open_device(list[i]);
    if (!c) continue;
    ibv_device_attr da{};
    if (ibv_query_device(c, &da)) continue;
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
  return 0;
}

int tcp_listen_and_accept(uint16_t port) {
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  listen(lfd, 2);
  DGPP_LOG_INFO("waiting rendezvous on :{}", port);
  int fd = accept(lfd, nullptr, nullptr);
  close(lfd);
  return fd;
}

int tcp_connect(const std::string& host, uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  return fd;
}

// Establishes role-specific buffers then swaps XchgInfo.
DevPort handshake_common(int fd, const RoleHeader& hdr,
                         const std::string& dev, int gid_idx) {
  size_t buf;
  int slot;
  if (hdr.kind == 2) {
    // Sink needs room for a deep ring of large messages.
    size_t budget = 96ull << 20;
    slot = std::max<int32_t>(hdr.msgsz, 8);
    buf = std::min<size_t>(budget, static_cast<size_t>(kRingSlots) * slot);
    slot = std::max<int>(static_cast<int>(buf / std::max<size_t>(slot, 1)),
                         8);  // effective ring depth fits budget
  } else {
    buf = 16 << 20;
    slot = 8;
  }
  DevPort d = setup(dev, gid_idx, buf, slot);

  XchgInfo me = local_info(d);
  sock_send_all(fd, &me, sizeof(me));
  auto r = sock_recv_exact(fd, sizeof(XchgInfo));
  if (r.size() != sizeof(XchgInfo)) {
    DGPP_LOG_ERROR("handshake short read");
    std::exit(1);
  }
  XchgInfo remote{};
  memcpy(&remote, r.data(), sizeof(remote));
  ready_to_rts(d, remote);
  return d;
}

}  // namespace

namespace {

int run_responder(uint16_t tcp_port, const std::string& dev, int gid_idx) {
  for (;;) {  // sequential connections (two possible: fabric0/fabric1)
    int fd = tcp_listen_and_accept(tcp_port);
    RoleHeader hdr{-1, 0, 0};
    auto hr = sock_recv_exact(fd, sizeof(hdr));
    if (hr.size() != sizeof(hdr)) {
      close(fd);
      continue;
    }
    memcpy(&hdr, hr.data(), sizeof(hdr));

    if (hdr.kind == 1) {  // ping responder: echo everything `count` times
      DevPort d = handshake_common(fd, hdr, dev, gid_idx);
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
          sw.send_flags = 0; // plain signaled send (inline stalls tx completions here)
          sw.sg_list = &sg;
          sw.num_sge = 1;
          ibv_send_wr* bad = nullptr;
          if (ibv_post_send(d.qp, &sw, &bad)) {
            DGPP_LOG_ERROR("echo post_send failed");
            return 1;
          }
          refill_recv_slot(d, rcw[q].wr_id, kSlot, slots);
          ibv_wc twc{};
          while (ibv_poll_cq(d.tx_cq, 1, &twc) <= 0)
            std::this_thread::yield();
          check_wc_ok(twc, "ping-tx");
          ++echoed;
        }
        if (n == 0 && std::chrono::steady_clock::now() > deadline) {
          DGPP_LOG_ERROR("responder timeout at {}/{}", echoed, hdr.count);
          return 1;
        }
        if (n == 0) std::this_thread::yield();
      }
      DGPP_LOG_INFO("responder echoed {}", echoed);
    } else if (hdr.kind == 2) {  // BW sink
      DevPort d = handshake_common(fd, hdr, dev, gid_idx);
      int slots = static_cast<int>(std::min<long long>(
          kRingSlots, std::max<long long>(1, (16 << 20) / hdr.msgsz)));
      post_ring_recvs(d, slots, hdr.msgsz);
      DGPP_LOG_INFO("sink: awaiting {} x {}B (ring {})", hdr.count, hdr.msgsz,
                    slots);
      auto start = std::chrono::steady_clock::now();
      long long rx = 0;
      auto deadline = start + std::chrono::seconds(120);
      while (rx < hdr.count) {
        ibv_wc rcw[128]{};
        int n = ibv_poll_cq(d.rx_cq, 128, rcw);
        for (int q = 0; q < n; ++q) {
          check_wc_ok(rcw[q], "sink-rx");
          refill_recv_slot(d, rcw[q].wr_id, hdr.msgsz, slots);
        }
        rx += n;
        if (n == 0 && std::chrono::steady_clock::now() > deadline) break;
        if (n == 0) std::this_thread::yield();
      }
      double secs =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count();
      sock_send_all(fd, "ACK", 3);
      sock_send_all(fd, &secs, sizeof(secs));
      DGPP_LOG_INFO("SINK-DONE rx={} {:.3f}s eff_Gbps={:.1f}", rx, secs,
                    double(rx) * hdr.msgsz * 8 / secs / 1e9);
    } else {
      DGPP_LOG_WARN("unknown header kind {}; ignoring", hdr.kind);
    }
    char bye[8];
    (void)::recv(fd, bye, sizeof(bye), MSG_DONTWAIT);
    close(fd);
  }
  return 0;
}

int run_ping(const std::string& host, uint16_t port, int iters,
             const std::string& dev, int gid_idx) {
  int fd = tcp_connect(host, port);
  RoleHeader hdr{1, iters, 0};
  sock_send_all(fd, &hdr, sizeof(hdr));
  DevPort d = handshake_common(fd, hdr, dev, gid_idx);
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
    sw.send_flags = 0; // plain signaled send
    sw.sg_list = &sg;
    sw.num_sge = 1;
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(d.qp, &sw, &bad)) break;

    bool saw_tx = false, saw_rx = false;
    uint64_t rx_id = 0;
    auto deadline = t0 + std::chrono::seconds(5);
    while (!(saw_tx && saw_rx)) {
      ibv_wc twc{};
      int ntx = ibv_poll_cq(d.tx_cq, 4, &twc);
      if (ntx > 0) {
        check_wc_ok(twc, "init-tx");
        saw_tx = true;
      }
      ibv_wc rcw[8]{};
      int nrx = ibv_poll_cq(d.rx_cq, 8, rcw);
      for (int q = 0; q < nrx; ++q) {
        check_wc_ok(rcw[q], "init-rx");
        saw_rx = true;
        rx_id = rcw[q].wr_id;
      }
      if (!(saw_tx && saw_rx)) {
        if (std::chrono::steady_clock::now() > deadline) {
          DGPP_LOG_ERROR("ping timeout iter {} (tx={} rx={})", i, saw_tx,
                         saw_rx);
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
                d.ctx->device->name, ok, best, sum / ok);
  sock_send_all(fd, "BYE", 3);
  close(fd);
  return ok == iters ? 0 : 1;
}

int run_bw_link(const std::string& host, uint16_t port, size_t msg_size,
                int total, int window, const std::string& dev, int gid_idx,
                double* sink_secs_out) {
  int fd = tcp_connect(host, port);
  RoleHeader hdr{2, total, static_cast<int32_t>(msg_size)};
  sock_send_all(fd, &hdr, sizeof(hdr));
  DevPort d = handshake_common(fd, hdr, dev, gid_idx);

  long long posted = 0, done_ = 0;
  uintptr_t src = reinterpret_cast<uintptr_t>(d.mr->addr) + (8ull << 20);
  while (done_ < total) {
    while (posted - done_ < window && posted < total) {
      ibv_sge sg{};
      sg.addr = src;
      sg.lkey = d.mr->lkey;
      sg.length = static_cast<uint32_t>(msg_size);
      ibv_send_wr sw{};
      sw.wr_id = static_cast<uint64_t>(200000 + posted++);
      sw.opcode = IBV_WR_SEND;
      sw.sg_list = &sg;
      sw.num_sge = 1;
      ibv_send_wr* bad = nullptr;
      if (ibv_post_send(d.qp, &sw, &bad)) {
        DGPP_LOG_ERROR("bw post_send failed errno={}", errno);
        return 1;
      }
    }
    ibv_wc wc{};
    int n = ibv_poll_cq(d.tx_cq, 256, &wc);
    if (n > 0) check_wc_ok(wc, "bw-tx");
    else if (n == 0) std::this_thread::yield();
    done_ += n;
  }

  char ack[3] = {};
  auto ar = sock_recv_exact(fd, 3);
  memcpy(ack, ar.data(), std::min<size_t>(ar.size(), 3));
  if (ack != std::string("ACK")) {
    DGPP_LOG_ERROR("no ACK from sink");
    return 1;
  }
  auto sr = sock_recv_exact(fd, sizeof(double));
  if (sr.size() == sizeof(double)) memcpy(sink_secs_out, sr.data(), sizeof(double));
  sock_send_all(fd, "BYE", 3);
  close(fd);
  return 0;
}

int run_bw(std::vector<std::string>& peers, size_t msg_size, int window,
           const std::string& dev, int gid_idx) {
  const int total_per_link = 4000;
  std::vector<std::pair<std::string, uint16_t>> targets;
  for (auto& p : peers) {
    auto colon = p.find(':');
    targets.emplace_back(p.substr(0, colon),
                         static_cast<uint16_t>(atoi(p.c_str() + colon + 1)));
  }

  std::vector<std::thread> ths;
  std::vector<double> sink_secs(targets.size(), 0.0);
  auto t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < targets.size(); ++i) {
    ths.emplace_back([i, &targets, msg_size, window, total_per_link, dev,
                      gid_idx, &sink_secs]() {
      run_bw_link(targets[i].first, targets[i].second, msg_size, window,
                  total_per_link, dev, gid_idx, &sink_secs[i]);
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
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  micro_ibv_smoke info\n"
                 "  micro_ibv_smoke serve <port> [--dev D]\n"
                 "  micro_ibv_smoke ping --peer IP:PORT [--iters N] [--dev D]"
                 " [--gid-index N]\n"
                 "  micro_ibv_smoke bw --peer IP:PORT [--msg-size N] "
                 "[--window N]\n");
    return 2;
  }
  std::string mode = argv[1];
  std::vector<std::string> peers;
  std::string dev;
  int gid_idx = -1;
  int iters = kDefaultIters;
  int window = 128;
  size_t msg_size = 262144;
  uint16_t port = 4791;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--peer") peers.push_back(val());
    else if (a == "--dev") dev = val();
    else if (a == "--gid-index") gid_idx = atoi(val().c_str());
    else if (a == "--iters") iters = atoi(val().c_str());
    else if (a == "--window") window = atoi(val().c_str());
    else if (a == "--msg-size") msg_size = strtoul(val().c_str(), nullptr, 10);
    else if (mode == "serve") port = static_cast<uint16_t>(atoi(a.c_str()));
  }

  if (mode == "info") return cmd_info();
  if (mode == "serve") return run_responder(port, dev, gid_idx);
  if (mode == "ping") {
    if (peers.size() != 1) return 2;
    auto colon = peers[0].find(':');
    return run_ping(peers[0].substr(0, colon),
                    static_cast<uint16_t>(atoi(peers[0].c_str() + colon + 1)),
                    iters, dev, gid_idx);
  }
  if (mode == "bw") {
    if (peers.empty()) return 2;
    return run_bw(peers, msg_size, window, dev, gid_idx);
  }
  DGPP_LOG_ERROR("unknown mode {}", mode);
  return 2;
}
