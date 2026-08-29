// CollectiveBus deep scenarios (M5 deliverable 2). Everything runs against
// a real loopback pair: two bus endpoints in one process, real verbs QPs
// through the fabric switch to ourselves (validated by the M0 smoke), real
// pinned slabs, real consumer kernels. The scenarios pin the §6 contract:
// payload integrity via the fold hash, credit recycling, dual-lane striping,
// latency-under-bulk contention, watchdog behavior, config-error
// legibility, and orderly stop.

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "net/bus_kernel.hpp"
#include "net/collective_bus.hpp"

namespace {

using dgpp::net::BusMessageClass;
using dgpp::net::BusOptions;
using dgpp::net::BusRecvView;
using dgpp::net::BusSendResult;
using dgpp::net::BusStats;
using dgpp::net::CollectiveBus;

int g_failures = 0;

#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      DGPP_LOG_ERROR("FAIL {}:{} — {}", __FILE__, __LINE__, msg);     \
      ++g_failures;                                                    \
    }                                                                 \
  } while (0)

double percentile(std::vector<double> v, double frac) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(frac * v.size()))];
}

BusOptions base_options(int rank, uint16_t port) {
  BusOptions o;
  o.world_size = 2;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 5000;
  o.consumer_deadline_s = 300.0;  // scenarios pause between phases
  return o;
}

// Starts a rank-0 listener bus (in a thread) and a rank-1 connector bus,
// both fully up, or fails the scenario.
std::pair<std::unique_ptr<CollectiveBus>, std::unique_ptr<CollectiveBus>>
start_pair(const BusOptions& a_opt, const BusOptions& b_opt) {
  auto a = std::make_unique<CollectiveBus>(a_opt);
  auto b = std::make_unique<CollectiveBus>(b_opt);
  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_ERROR("rank A start: {}", a_error);
  });
  std::string b_error;
  const bool b_ok = b->start(&b_error);
  a_thread.join();
  CHECK(a_error.empty(), "rank A start failed: " + a_error);
  CHECK(b_ok, "rank B start failed: " + b_error);
  if (!a_error.empty() || !b_ok) return {nullptr, nullptr};
  return {std::move(a), std::move(b)};
}

void fill_payload(uint64_t* data, size_t words, uint32_t salt) {
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * salt);
  for (size_t i = 0; i < words; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = x;
  }
}

bool send_and_verify(CollectiveBus& bus, int peer, BusMessageClass cls,
                     const void* data, size_t bytes, int wait_ms,
                     BusSendResult* out) {
  std::string error;
  const uint64_t id = bus.send(peer, data, bytes, cls, &error);
  if (id == 0) {
    DGPP_LOG_ERROR("send rejected: {}", error);
    return false;
  }
  *out = bus.wait(id, wait_ms);
  if (!out->ok) return false;
  // Stripe i covers bytes [i*stripe_bytes, min((i+1)*stripe_bytes, bytes)).
  const size_t stripe_bytes = bus.slot_bytes(cls);
  uint64_t got_xor = 0, want_xor = 0;
  for (size_t i = 0; i < out->stripe_hashes.size(); ++i) {
    const size_t off = i * stripe_bytes;
    const size_t chunk = std::min(stripe_bytes, bytes - off);
    got_xor ^= out->stripe_hashes[i];
    want_xor ^= dgpp::net::bus_fold(
        static_cast<const uint8_t*>(data) + off, chunk);
  }
  if (got_xor != want_xor) {
    DGPP_LOG_ERROR("hash mismatch: got {:x} want {:x}", got_xor, want_xor);
    return false;
  }
  return true;
}

// ---- scenarios ----------------------------------------------------------------

void scenario_latency_ping() {
  auto [a, b] = start_pair(base_options(0, 29810), base_options(1, 29810));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  int ok = 0;
  std::vector<double> latency;
  for (int i = 0; i < 64; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, payload.data(),
                        8192, 15000, &r)) {
      ++ok;
      latency.push_back(r.elapsed_us);
    }
  }
  CHECK(ok == 64, "latency ping: only " + std::to_string(ok) + "/64 verified");
  DGPP_LOG_INFO("scenario latency_ping: p50={:.1f}us p99={:.1f}us",
                percentile(latency, 0.5), percentile(latency, 0.99));
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_credit_recycle() {
  // 4 latency slots per lane, 96 messages: recycling must keep the pool
  // flowing or the run stalls (and the watchdog would fail it loudly).
  BusOptions a_opt = base_options(0, 29820);
  BusOptions b_opt = base_options(1, 29820);
  a_opt.lat_slots = 4;
  b_opt.lat_slots = 4;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  int ok = 0;
  for (int i = 0; i < 96; ++i) {
    fill_payload(payload.data(), payload.size(), static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, payload.data(),
                        8192, 15000, &r))
      ++ok;
  }
  CHECK(ok == 96, "credit recycle: only " + std::to_string(ok) + "/96");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_bulk_dual_lane() {
  auto [a, b] = start_pair(base_options(0, 29830), base_options(1, 29830));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8u << 20);  // 64 MiB / 8
  fill_payload(payload.data(), payload.size(), 7u);
  const size_t bytes = payload.size() * 8;
  const auto t0 = std::chrono::steady_clock::now();
  BusSendResult r;
  const bool ok = send_and_verify(*b, 0, BusMessageClass::kBulk,
                                  payload.data(), bytes, 60000, &r);
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  CHECK(ok, "bulk 64MiB message failed verification");
  CHECK(r.stripe_hashes.size() >= 4,
        "expected >= 4 stripes for 64MiB, got " +
            std::to_string(r.stripe_hashes.size()));
  // Both lanes must contribute: credits received per lane on the sender,
  // doorbells received per lane on the receiver.
  int sender_lanes = 0, receiver_lanes = 0;
  for (const auto& l : b->stats().lanes)
    if (l.credits_received > 0) ++sender_lanes;
  for (const auto& l : a->stats().lanes)
    if (l.doorbell_recvs > 0) ++receiver_lanes;
  CHECK(sender_lanes == 2, "striping used " + std::to_string(sender_lanes) +
                                " lanes (sender view)");
  CHECK(receiver_lanes == 2, "striping used " +
                                 std::to_string(receiver_lanes) +
                                 " lanes (receiver view)");
  DGPP_LOG_INFO("scenario bulk_dual_lane: {:.1f} MiB in {:.3f}s = {:.2f} GB/s",
                bytes / 1048576.0, secs,
                secs > 0 ? static_cast<double>(bytes) / secs / 1e9 : 0.0);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_latency_under_bulk() {
  auto [a, b] = start_pair(base_options(0, 29840), base_options(1, 29840));
  if (!a || !b) return;

  std::vector<uint64_t> bulk(2u << 20);  // 16 MiB payloads
  std::vector<uint64_t> lat(8192 / 8);
  std::atomic<int> bulk_fail{0};
  const int bulk_iters = 24;
  const int lat_iters = 256;

  std::thread bulk_thread([&] {
    for (int i = 0; i < bulk_iters; ++i) {
      fill_payload(bulk.data(), bulk.size(), static_cast<uint32_t>(i));
      BusSendResult r;
      if (!send_and_verify(*b, 0, BusMessageClass::kBulk, bulk.data(),
                           bulk.size() * 8, 30000, &r))
        bulk_fail.fetch_add(1);
    }
  });

  std::vector<double> latency;
  int ok = 0;
  for (int i = 0; i < lat_iters; ++i) {
    fill_payload(lat.data(), lat.size(), 0xF000 + static_cast<uint32_t>(i));
    BusSendResult r;
    if (send_and_verify(*b, 0, BusMessageClass::kLatency, lat.data(), 8192,
                        30000, &r)) {
      ++ok;
      latency.push_back(r.elapsed_us);
    }
  }
  bulk_thread.join();

  CHECK(bulk_fail.load() == 0, "bulk stream had failures under contention");
  CHECK(ok == lat_iters, "latency stream lost " +
                             std::to_string(lat_iters - ok) + " messages");
  const double p99 = percentile(latency, 0.99);
  // Generous CI budget (headless node, two threads); the recorded number is
  // what matters for the exit-criterion record.
  CHECK(p99 < 20000.0, "latency p99 under bulk = " + std::to_string(p99) +
                           "us, over the 20ms budget");
  DGPP_LOG_INFO("scenario latency_under_bulk: lat p50={:.1f}us p99={:.1f}us "
                "while {}x16MiB bulk streamed",
                percentile(latency, 0.5), p99, bulk_iters);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_completion_timeout() {
  // No consumer on the RECEIVER: the sender's request must fail loudly via
  // the per-request watchdog, not hang. (The sender's own consumers are
  // irrelevant — it is the receiving rank's GPU that acks.)
  BusOptions a_opt = base_options(0, 29850);
  BusOptions b_opt = base_options(1, 29850);
  a_opt.launch_consumers = false;
  a_opt.completion_timeout_ms = 600;
  b_opt.completion_timeout_ms = 600;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  fill_payload(payload.data(), payload.size(), 1);
  std::string error;
  const uint64_t id =
      b->send(0, payload.data(), 8192, BusMessageClass::kLatency, &error);
  CHECK(id != 0, "send rejected: " + error);
  const BusSendResult r = b->wait(id, 10000);
  CHECK(!r.ok, "request should have failed without a receiver consumer");
  CHECK(r.error.find("completion timeout") != std::string::npos ||
            r.error.find("watchdog") != std::string::npos,
        "expected a watchdog reason, got: " + r.error);
  DGPP_LOG_INFO("scenario completion_timeout: failed as designed ({})",
                r.error);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_consumer_inactivity_exit() {
  // A consumer with a short deadline and no traffic must exit cleanly via
  // the inactivity watchdog; the stream sync must return promptly.
  BusOptions a_opt = base_options(0, 29860);
  BusOptions b_opt = base_options(1, 29860);
  b_opt.launch_consumers = false;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;

  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess,
        "stream create failed");
  const uint64_t deadline = dgpp::net::bus_consumer_deadline_cycles(0.3);
  CHECK(deadline > 0, "deadline cycles unavailable");
  const BusRecvView view = b->recv_view(0, 0);
  CHECK(view.control != nullptr, "recv view is empty");
  CHECK(dgpp::net::launch_bus_consumer(view, deadline, stream) ==
            cudaSuccess,
        "consumer launch failed");

  const auto t0 = std::chrono::steady_clock::now();
  CHECK(cudaStreamSynchronize(stream) == cudaSuccess,
        "consumer stream sync failed");
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  CHECK(secs < 5.0, "consumer exit took " + std::to_string(secs) +
                        "s; watchdog did not bound inactivity");
  cudaStreamDestroy(stream);
  DGPP_LOG_INFO("scenario consumer_inactivity_exit: exited after {:.2f}s",
                secs);
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_mesh_three_way() {
  // World=3 loopback mesh: rank 0 listens, ranks 1 and 2 connect, and every
  // rank exchanges with every peer. Exercises the full endpoint-table
  // distribution (the multi-peer path the two-rank scenarios never touch)
  // on one node, one log.
  const uint16_t port = 29890;
  BusOptions o0 = base_options(0, port);
  BusOptions o1 = base_options(1, port);
  BusOptions o2 = base_options(2, port);
  o0.world_size = 3;
  o1.world_size = 3;
  o2.world_size = 3;
  auto a = std::make_unique<CollectiveBus>(o0);  // rank 0, listener
  auto b = std::make_unique<CollectiveBus>(o1);
  auto c = std::make_unique<CollectiveBus>(o2);

  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_ERROR("mesh A: {}", a_error);
  });
  auto start_rank = [](CollectiveBus* bus, std::string* error) {
    if (!bus->start(error)) DGPP_LOG_ERROR("mesh: {}", *error);
  };
  std::string b_error, c_error;
  std::thread b_thread(start_rank, b.get(), &b_error);
  std::thread c_thread(start_rank, c.get(), &c_error);
  a_thread.join();
  b_thread.join();
  c_thread.join();
  const bool started =
      a_error.empty() && b_error.empty() && c_error.empty();
  CHECK(started, "mesh startup failed: A=" + a_error + " B=" + b_error +
                     " C=" + c_error);
  if (!started) {
    // Error path must quiesce all consumers before any teardown frees
    // (cudaFreeHost synchronizes the device implicitly).
    a->quiesce();
    b->quiesce();
    c->quiesce();
    a->stop();
    b->stop();
    c->stop();
    return;
  }

  // Every rank sends to every peer; verify each.
  struct Lane { int from; int to; };
  const Lane lanes[] = {{1, 0}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {1, 2}};
  int failures = 0;
  for (const Lane& l : lanes) {
    CollectiveBus* from = l.from == 0 ? a.get() : (l.from == 1 ? b.get() : c.get());
    std::vector<uint64_t> payload(8192 / 8);
    fill_payload(payload.data(), payload.size(),
                 static_cast<uint32_t>(l.from * 10 + l.to));
    for (int i = 0; i < 4; ++i) {
      BusSendResult r;
      if (!send_and_verify(*from, l.to, BusMessageClass::kLatency,
                           payload.data(), 8192, 15000, &r)) {
        DGPP_LOG_ERROR("mesh {}->{} iter {} failed", l.from, l.to, i);
        ++failures;
      }
    }
  }
  CHECK(failures == 0,
        "three-way mesh had " + std::to_string(failures) + " failures");
  DGPP_LOG_INFO("scenario mesh_three_way: {} failures over 6 directed pairs",
                failures);
  a->quiesce();
  b->quiesce();
  c->quiesce();
  a->stop();
  b->stop();
  c->stop();
}

void scenario_geometry_mismatch() {
  // Config errors must be legible over the rendezvous (roster precedent),
  // never a bare close or a hang.
  BusOptions a_opt = base_options(0, 29870);
  BusOptions b_opt = base_options(1, 29870);
  a_opt.lat_slots = 8;
  b_opt.lat_slots = 16;
  auto a = std::make_unique<CollectiveBus>(a_opt);
  auto b = std::make_unique<CollectiveBus>(b_opt);
  std::string a_error;
  std::thread a_thread([&] {
    if (!a->start(&a_error)) DGPP_LOG_INFO("A rejected: {}", a_error);
  });
  std::string b_error;
  const bool b_ok = b->start(&b_error);
  a_thread.join();
  CHECK(!b_ok, "mismatched geometry must fail startup");
  CHECK(b_error.find("geometry") != std::string::npos ||
            b_error.find("mismatch") != std::string::npos,
        "expected a geometry-mismatch reason, got: " + b_error);
  CHECK(a_error.find("geometry") != std::string::npos ||
            !a_error.empty(),
        "rank A should also fail legibly, got: " + a_error);
  DGPP_LOG_INFO("scenario geometry_mismatch: rejected as designed");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
}

void scenario_stop_releases_waiters() {
  // Stopping while a message is in flight must resolve its waiter — either
  // completed or failed — and never hang.
  auto [a, b] = start_pair(base_options(0, 29880), base_options(1, 29880));
  if (!a || !b) return;
  std::vector<uint64_t> payload(8192 / 8);
  fill_payload(payload.data(), payload.size(), 3);
  std::string error;
  const uint64_t id =
      b->send(0, payload.data(), 8192, BusMessageClass::kLatency, &error);
  CHECK(id != 0, "send rejected: " + error);
  // Quiesce first: the engine drains the in-flight request (completes it
  // via the peer, or the watchdog fails it), so the waiter resolves before
  // stop() reaps the registry — never a hang, never a lost notification.
  b->quiesce();
  a->quiesce();
  const BusSendResult r = b->wait(id, 15000);
  CHECK(r.ok || !r.error.empty(),
        "waiter neither completed nor failed cleanly: " + r.error);
  DGPP_LOG_INFO("scenario stop_releases_waiters: resolved ({})",
                r.ok ? "completed" : r.error);
  a->stop();
  b->stop();
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices == 0) {
    DGPP_LOG_INFO("no CUDA device visible; skipping bus_test");
    return 2;
  }

  scenario_latency_ping();
  scenario_credit_recycle();
  scenario_bulk_dual_lane();
  scenario_latency_under_bulk();
  scenario_completion_timeout();
  scenario_consumer_inactivity_exit();
  scenario_mesh_three_way();
  scenario_geometry_mismatch();
  scenario_stop_releases_waiters();

  DGPP_LOG_INFO("bus_test: {} failures", g_failures);
  return g_failures == 0 ? 0 : 1;
}
