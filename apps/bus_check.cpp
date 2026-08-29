// bus_check: M5 deliverable 2 deployment driver — CollectiveBus over RoCE.
//
// Modes:
//   serve   [--port N] [--dev D]...        rank 0 rendezvous listener; runs
//          [--world N] [--duration-ms N]    the receiver, prints stats
//          [--lat-slots N] [--lat-bytes B] [--bulk-slots N] [--bulk-bytes B]
//   ping   --peer HOST [--port N] ...      rank 1 connector; sends messages
//          [--iters N] [--bytes B] [--class latency|bulk]
//          [--contend] [--lat-iters N]     bulk flood while latency pings
//   selftest                               loopback, in-process (ctest)
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/log.hpp"
#include "net/bus_kernel.hpp"
#include "net/collective_bus.hpp"

namespace {

using dgpp::net::BusMessageClass;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

bool parse_long(const std::string& text, long minimum, long maximum,
                long* result) {
  if (text.empty()) return false;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || value < minimum ||
      value > maximum)
    return false;
  *result = value;
  return true;
}

bool parse_size(const std::string& text, size_t minimum, size_t maximum,
                size_t* result) {
  if (text.empty() || text.front() == '-') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      value < minimum || value > maximum)
    return false;
  *result = static_cast<size_t>(value);
  return true;
}

double percentile(std::vector<double> sorted, double frac) {
  if (sorted.empty()) return 0.0;
  std::sort(sorted.begin(), sorted.end());
  const size_t idx = std::min(sorted.size() - 1,
                              static_cast<size_t>(frac * sorted.size()));
  return sorted[idx];
}

// Deterministic payload varying per message index: stale-buffer bugs (a
// slot reused without a fresh copy, a fold of last generation's bytes) fail
// the hash instead of passing silently.
void fill_payload(uint64_t* data, size_t words, uint32_t salt) {
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * salt);
  for (size_t i = 0; i < words; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = x;
  }
}

// Verifies a completed send: each stripe's consumer hash equals the fold of
// its contiguous chunk (latency: one stripe, whole message; bulk: chunk i
// is the i-th bulk_slot_bytes slice of the payload).
bool verify_result(const dgpp::net::BusSendResult& r, const void* data,
                   size_t bytes, size_t stripe_bytes, std::string* why) {
  if (!r.ok) {
    *why = "send failed: " + r.error;
    return false;
  }
  if (r.stripe_hashes.empty()) {
    *why = "no stripes credited";
    return false;
  }
  uint64_t got_xor = 0, want_xor = 0;
  for (size_t i = 0; i < r.stripe_hashes.size(); ++i) {
    const size_t off = i * stripe_bytes;
    const size_t chunk = std::min(stripe_bytes, bytes - off);
    const uint64_t want = dgpp::net::bus_fold(
        static_cast<const uint8_t*>(data) + off, chunk);
    got_xor ^= r.stripe_hashes[i];
    want_xor ^= want;
  }
  if (got_xor != want_xor) {
    *why = "payload hash mismatch (got " + std::to_string(got_xor) +
           " want " + std::to_string(want_xor) + ")";
    return false;
  }
  return true;
}

void print_bus_stats(const dgpp::net::BusStats& s) {
  for (const dgpp::net::BusLaneStats& l : s.lanes) {
    DGPP_LOG_INFO(
        "lane peer={} lane={} posts={} sent={}B recvs={} recv={}B "
        "credits_out={} credits_in={} failed={}",
        l.peer_rank, l.lane, l.posts, l.bytes_sent, l.doorbell_recvs,
        l.bytes_recv, l.credits_returned, l.credits_received,
        l.failed ? 1 : 0);
  }
}

struct CommonArgs {
  uint16_t port = 29600;
  std::vector<std::string> devs;
  int world = 2;
  int lat_slots = 32;
  size_t lat_bytes = 8192;
  int bulk_slots = 16;
  size_t bulk_bytes = 262144;
  int timeout_ms = 5000;
};

BusOptions options_for(const CommonArgs& c, int my_rank,
                       const std::string& host) {
  BusOptions o;
  o.world_size = c.world;
  o.my_rank = my_rank;
  o.lane_devices =
      c.devs.empty()
          ? std::vector<std::string>{"rocep1s0f0", "roceP2p1s0f0"}
          : c.devs;
  o.rendezvous_port = c.port;
  o.rendezvous_host = host;
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = c.lat_slots;
  o.lat_slot_bytes = c.lat_bytes;
  o.bulk_slots = c.bulk_slots;
  o.bulk_slot_bytes = c.bulk_bytes;
  o.completion_timeout_ms = c.timeout_ms;
  o.consumer_deadline_s = 60.0;
  return o;
}

// ---- serve ------------------------------------------------------------------

int run_serve(CommonArgs c, long duration_ms) {
  CollectiveBus bus(options_for(c, 0, ""));
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("serve: {}", error);
    return 1;
  }
  DGPP_LOG_INFO("SERVE-READY port={} duration_ms={}", c.port, duration_ms);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  print_bus_stats(bus.stats());
  bus.stop();
  DGPP_LOG_INFO("serve: stopped cleanly");
  return 0;
}

// ---- ping -------------------------------------------------------------------

struct PingStats {
  int ok = 0;
  int fail = 0;
  std::vector<double> latency_us;
  double bytes = 0;
  double elapsed_s = 0;
};

PingStats ping_class(CollectiveBus& bus, int peer, BusMessageClass cls,
                     int iters, size_t bytes, int wait_ms) {
  PingStats st;
  std::vector<uint64_t> payload((bytes + 7) / 8);
  const size_t stripe_bytes = bus.slot_bytes(cls);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    std::string error;
    const uint64_t id = bus.send(peer, payload.data(), bytes, cls, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("send rejected: {}", error);
      ++st.fail;
      continue;
    }
    const dgpp::net::BusSendResult r = bus.wait(id, wait_ms);
    std::string why;
    if (!verify_result(r, payload.data(), bytes, stripe_bytes, &why)) {
      DGPP_LOG_ERROR("iter {}: {}", i, why);
      ++st.fail;
      continue;
    }
    ++st.ok;
    st.latency_us.push_back(r.elapsed_us);
    st.bytes += static_cast<double>(bytes);
  }
  st.elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  return st;
}

int run_ping(CommonArgs c, std::string peer_host, int iters, size_t bytes,
             const std::string& cls_text, bool contend, int lat_iters) {
  CollectiveBus bus(options_for(c, 1, peer_host));
  std::string error;
  if (!bus.start(&error)) {
    DGPP_LOG_ERROR("ping: {}", error);
    return 1;
  }

  int failures = 0;
  if (contend) {
    // Bulk flood on one thread, latency probes on another: the §6.1
    // concurrency contract — latency must hold budget under bulk load.
    std::atomic<int> bulk_fail{0};
    std::thread bulk_thread([&] {
      const PingStats bulk =
          ping_class(bus, 0, BusMessageClass::kBulk, iters, bytes,
                     c.timeout_ms + 2000);
      bulk_fail.store(bulk.fail);
    });
    const PingStats lat =
        ping_class(bus, 0, BusMessageClass::kLatency, lat_iters, c.lat_bytes,
                   c.timeout_ms + 2000);
    bulk_thread.join();
    DGPP_LOG_INFO(
        "CONTEND lat ok={}/{} p50_us={:.1f} p99_us={:.1f} max_us={:.1f}",
        lat.ok, lat_iters, percentile(lat.latency_us, 0.5),
        percentile(lat.latency_us, 0.99),
        lat.latency_us.empty()
            ? 0.0
            : *std::max_element(lat.latency_us.begin(),
                                lat.latency_us.end()));
    failures = lat.fail + bulk_fail.load();
  } else {
    const BusMessageClass cls =
        cls_text == "bulk" ? BusMessageClass::kBulk : BusMessageClass::kLatency;
    const PingStats st =
        ping_class(bus, 0, cls, iters, bytes, c.timeout_ms + 2000);
    if (st.fail != 0) failures += st.fail;
    const double gbps =
        st.elapsed_s > 0 ? st.bytes * 8.0 / st.elapsed_s / 1e9 : 0.0;
    DGPP_LOG_INFO(
        "PING class={} iters={} bytes={} ok={} fail={} min_us={:.1f} "
        "p50_us={:.1f} p99_us={:.1f} max_us={:.1f} gbps={:.2f}",
        cls_text, iters, bytes, st.ok, st.fail,
        st.latency_us.empty()
            ? 0.0
            : *std::min_element(st.latency_us.begin(), st.latency_us.end()),
        percentile(st.latency_us, 0.5), percentile(st.latency_us, 0.99),
        st.latency_us.empty()
            ? 0.0
            : *std::max_element(st.latency_us.begin(), st.latency_us.end()),
        gbps);
  }

  print_bus_stats(bus.stats());
  bus.stop();
  DGPP_LOG_INFO("ping: stopped cleanly (failures={})", failures);
  return failures == 0 ? 0 : 1;
}


// ---- selftest ----------------------------------------------------------------

// Runs rank-0 (listener) and rank-1 (connector) buses in-process against a
// loopback rendezvous on the fabric (validated: an RC pair through the
// switch to ourselves).
int run_selftest() {
  int failures = 0;
  CommonArgs c;
  c.port = 29710;

  auto start_pair = [&](BusOptions a_opt, BusOptions b_opt,
                        std::unique_ptr<CollectiveBus>* a_out,
                        std::unique_ptr<CollectiveBus>* b_out,
                        std::string* error) {
    *a_out = std::make_unique<CollectiveBus>(a_opt);
    *b_out = std::make_unique<CollectiveBus>(b_opt);
    std::string a_error;
    std::thread a_thread([&] {
      if (!(*a_out)->start(&a_error)) DGPP_LOG_ERROR("selftest A: {}", a_error);
    });
    const bool b_ok = (*b_out)->start(error);
    a_thread.join();
    if (!b_ok) return false;
    if (!a_error.empty()) {
      *error = a_error;
      return false;
    }
    return true;
  };

  {
    // Scenario 1: latency ping integrity.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    const PingStats st =
        ping_class(*b, 0, BusMessageClass::kLatency, 32, c.lat_bytes,
                   c.timeout_ms + 2000);
    if (st.ok != 32) {
      DGPP_LOG_ERROR("selftest latency: ok={} fail={}", st.ok, st.fail);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest latency ok (p99_us={:.1f})",
                    percentile(st.latency_us, 0.99));
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  {
    // Scenario 2: bulk striping across both lanes.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    const size_t bytes = 4u << 20;
    const PingStats st = ping_class(*b, 0, BusMessageClass::kBulk, 4, bytes,
                                    c.timeout_ms + 2000);
    const dgpp::net::BusStats sender = b->stats();
    int lanes_used = 0;
    for (const auto& l : sender.lanes)
      if (l.credits_received > 0) ++lanes_used;
    if (st.ok != 4 || lanes_used < 2) {
      DGPP_LOG_ERROR("selftest bulk: ok={} fail={} lanes_used={}", st.ok,
                     st.fail, lanes_used);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest bulk ok ({:.0f} MB, {} lanes, gbps={:.2f})",
                    4.0 * 4, lanes_used,
                    st.elapsed_s > 0 ? st.bytes * 8.0 / st.elapsed_s / 1e9 : 0.0);
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  {
    // Scenario 3: latency under concurrent bulk.
    std::unique_ptr<CollectiveBus> a, b;
    std::string error;
    if (!start_pair(options_for(c, 0, ""), options_for(c, 1, "127.0.0.1"),
                    &a, &b, &error)) {
      DGPP_LOG_ERROR("selftest startup: {}", error);
      return 1;
    }
    std::thread bulk_thread([&] {
      ping_class(*b, 0, BusMessageClass::kBulk, 8, 2u << 20,
                 c.timeout_ms + 2000);
    });
    const PingStats lat = ping_class(*b, 0, BusMessageClass::kLatency, 64,
                                     c.lat_bytes, c.timeout_ms + 2000);
    bulk_thread.join();
    if (lat.ok != 64) {
      DGPP_LOG_ERROR("selftest contention: ok={} fail={}", lat.ok, lat.fail);
      ++failures;
    } else {
      DGPP_LOG_INFO("selftest contention ok (lat p99_us={:.1f})",
                    percentile(lat.latency_us, 0.99));
    }
    a->quiesce();
    b->quiesce();
    a->stop();
    b->stop();
  }

  DGPP_LOG_INFO("bus selftest: {} failures", failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  bus_check serve [--port N] [--dev D]... [--world N] "
                 "[--duration-ms N]\n"
                 "                  [--lat-slots N] [--lat-bytes B] "
                 "[--bulk-slots N] [--bulk-bytes B]\n"
                 "  bus_check ping --peer HOST [--port N] [--dev D]... "
                 "[--iters N] [--bytes B]\n"
                 "                 [--class latency|bulk] [--contend] "
                 "[--lat-iters N] [--timeout-ms N]\n"
                 "  bus_check selftest\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode == "selftest" && argc == 2) return run_selftest();

  CommonArgs c;
  std::string peer;
  long iters = 64;
  long duration_ms = 30000;
  long lat_iters = 1000;
  size_t bytes = 8192;
  std::string cls_text = "latency";
  bool contend = false;
  bool args_ok = true;

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
    size_t sz = 0;
    if (a == "--port") {
      if (!parse_long(val(), 1, 65535, &n)) args_ok = false;
      else c.port = static_cast<uint16_t>(n);
    } else if (a == "--dev") {
      const std::string d = val();
      if (d.empty()) args_ok = false;
      else c.devs.push_back(d);
    } else if (a == "--world") {
      if (!parse_long(val(), 2, 4, &n)) args_ok = false;
      else c.world = static_cast<int>(n);
    } else if (a == "--duration-ms") {
      if (!parse_long(val(), 100, 3600000, &n)) args_ok = false;
      else duration_ms = n;
    } else if (a == "--lat-slots") {
      if (!parse_long(val(), 1, 240, &n)) args_ok = false;
      else c.lat_slots = static_cast<int>(n);
    } else if (a == "--lat-bytes") {
      if (!parse_size(val(), 64, 1 << 20, &sz)) args_ok = false;
      else c.lat_bytes = sz;
    } else if (a == "--bulk-slots") {
      if (!parse_long(val(), 1, 240, &n)) args_ok = false;
      else c.bulk_slots = static_cast<int>(n);
    } else if (a == "--bulk-bytes") {
      if (!parse_size(val(), 64, 4 << 20, &sz)) args_ok = false;
      else c.bulk_bytes = sz;
    } else if (a == "--iters") {
      if (!parse_long(val(), 1, 1000000, &n)) args_ok = false;
      else iters = n;
    } else if (a == "--bytes") {
      if (!parse_size(val(), 8, 64 << 20, &sz)) args_ok = false;
      else bytes = sz;
    } else if (a == "--class") {
      cls_text = val();
      if (cls_text != "latency" && cls_text != "bulk") args_ok = false;
    } else if (a == "--contend") {
      contend = true;
    } else if (a == "--lat-iters") {
      if (!parse_long(val(), 1, 1000000, &n)) args_ok = false;
      else lat_iters = n;
    } else if (a == "--timeout-ms") {
      if (!parse_long(val(), 100, 600000, &n)) args_ok = false;
      else c.timeout_ms = static_cast<int>(n);
    } else if (a == "--peer") {
      peer = val();
      if (peer.empty()) args_ok = false;
    } else {
      DGPP_LOG_ERROR("unexpected argument {}", a);
      args_ok = false;
    }
  }
  if (!args_ok) return 2;

  if (mode == "serve") return run_serve(c, duration_ms);
  if (mode == "ping") {
    if (peer.empty()) return 2;
    return run_ping(c, peer, static_cast<int>(iters), bytes, cls_text, contend,
                    static_cast<int>(lat_iters));
  }
  DGPP_LOG_ERROR("unknown mode {}", mode);
  return 2;
}
