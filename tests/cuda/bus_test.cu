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

// ---- bf16 helpers: shared in bus_kernel.hpp (match RNE exactly) -----------

// Deterministic per-rank bf16 pattern with arbitrary mantissas (the
// rounding path is exercised; the oracle chain matters).
void fill_rank_bf16(std::vector<uint16_t>* out, size_t elems, int rank) {
  out->assign(elems, 0);
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * (rank + 1));
  for (size_t i = 0; i < elems; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const float v =
        static_cast<float>(static_cast<int32_t>(x >> 40)) * (1.0f / 8388608.0f) - 1.0f;
    (*out)[i] = dgpp::net::bf16_from_f32_rne(v);
  }
}

// Device twin of fill_rank_bf16 — the producing-kernel half of the
// pre-stage seam: a GPU kernel writing the NIC-registered pinned buffer
// directly (what the boundary GEMM does in the model seam). Single thread
// so the sequential LCG chain matches the host pattern element-for-element.
__global__ void staged_fill_kernel(uint16_t* dst, size_t elems, int rank) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ (0x100000001B3ULL * (rank + 1));
  for (size_t i = 0; i < elems; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    const float v =
        static_cast<float>(static_cast<int32_t>(x >> 40)) * (1.0f / 8388608.0f) - 1.0f;
    const uint32_t bits = __float_as_uint(v);
    dst[i] = static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
  }
}

// Starts a loopback world: rank 0 listens, ranks 1..N-1 connect.
// Collective mode: no persistent consumers (they would race the
// per-collective kernel — and their 300s deadlines would turn any
// implicit-sync call in the workers into a hang).
std::vector<std::unique_ptr<CollectiveBus>> start_world(int world,
                                                        uint16_t port,
                                                        bool one_lane = false) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) {
    BusOptions o = base_options(r, port);
    o.world_size = world;
    o.launch_consumers = false;
    if (one_lane) o.lane_devices = {o.lane_devices.front()};
    out.push_back(std::make_unique<CollectiveBus>(o));
  }
  std::vector<std::string> errors(world);
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

// One rank's share of a collective run: `iters` collectives over the same
// buffers (slot generations advance, recycling is exercised), each result
// checked bitwise against the canonical-chain oracle. Returns the number
// of local failures (CHECK is main-thread-only; workers report home).
int allreduce_rank_work(CollectiveBus& bus, int world, int my_rank,
                        size_t elems, int iters) {
  std::vector<uint16_t> host_src, host_dst(elems, 0);
  fill_rank_bf16(&host_src, elems, my_rank);
  // Any rank can regenerate any rank's pattern — the oracle needs them all.
  std::vector<std::vector<uint16_t>> all_src(world);
  for (int r = 0; r < world; ++r) fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);

  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: device alloc failed", my_rank);
    return 1;
  }
  if (cudaMemcpy(dev_src, host_src.data(), elems * 2, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return 1;
  }

  int failures = 0;
  std::vector<uint16_t> got(elems, 0);
  for (int iter = 0; iter < iters; ++iter) {
    std::string error;
    const uint64_t id = bus.allreduce(dev_src, dev_dst, elems, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("rank {}: allreduce rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 15000);
    if (!r.ok) {
      DGPP_LOG_ERROR("rank {}: collective failed: {}", my_rank, r.error);
      ++failures;
      break;
    }
    if (cudaMemcpy(got.data(), dev_dst, elems * 2, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: D2H failed", my_rank);
      ++failures;
      break;
    }
    // Canonical chain: acc = 0; for r in 0..world-1: acc += f32(src_r[i]).
    size_t mismatches = 0;
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      const uint16_t want = dgpp::net::bf16_from_f32_rne(acc);
      if (got[i] != want) {
        if (mismatches < 3)
          DGPP_LOG_ERROR("rank {} iter {} elem {}: got {:04x} want {:04x}",
                         my_rank, iter, i, got[i], want);
        ++mismatches;
      }
    }
    if (mismatches != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: {} mismatches", my_rank, iter,
                     mismatches);
      ++failures;
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  return failures;
}

void scenario_allreduce() {
  // One-shot all-to-all all-reduce over loopback worlds: bitwise-identical
  // results on every rank, across generations (slot recycling), plus the
  // v1 contract pins (consumers-running rejection, mode-closed sends).
  const size_t elems = 4096;  // 8 KiB, the decode unit
  int failures = 0;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29895 : 29896;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems, 8);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) {
      if (rank_failures[static_cast<size_t>(r)] != 0)
        DGPP_LOG_ERROR("world {} rank {} reported failures", world, r);
      world_failures += rank_failures[static_cast<size_t>(r)];
    }
    CHECK(world_failures == 0,
          "world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " collective failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce: world {} clean", world);
  }

  // Contract pins on a dedicated pair: allreduce is rejected while harness
  // consumers run, and send() is closed once the bus is in collective mode.
  {
    auto [a, b] = start_pair(base_options(0, 29897), base_options(1, 29897));
    if (!a || !b) {
      ++failures;
    } else {
      std::string error;
      CHECK(a->allreduce(nullptr, nullptr, elems, &error) == 0 &&
                error.find("launch_consumers") != std::string::npos,
            "allreduce must be rejected while consumers run: " + error);
      CHECK(b->allreduce(nullptr, nullptr, elems, &error) == 0 &&
                error.find("launch_consumers") != std::string::npos,
            "peer allreduce must be rejected while consumers run: " + error);
      a->quiesce();
      b->quiesce();
      a->stop();
      b->stop();
    }
  }

  {
    // Mode pin: once a collective has been picked up, harness sends fail
    // legibly (their messages would be folded into a peer's reduce). Both
    // ranks participate; rank 0 probes send() mid-flight, after giving the
    // engine a moment to enter collective mode.
    std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(2, 29898);
    if (buses.empty()) {
      ++failures;
    } else {
      uint16_t* dev = nullptr;
      CHECK(cudaMalloc(&dev, elems * 2) == cudaSuccess, "device alloc failed");
      std::thread peer_worker([&, elems] {
        uint16_t* peer_dev = nullptr;
        cudaMalloc(&peer_dev, elems * 2);
        std::string peer_error;
        const uint64_t peer_id =
            buses[1]->allreduce(peer_dev, peer_dev, elems, &peer_error);
        if (peer_id != 0)
          buses[1]->wait_allreduce(peer_id, 15000);
        cudaFree(peer_dev);
      });
      std::string error;
      const uint64_t id = buses[0]->allreduce(dev, dev, elems, &error);
      CHECK(id != 0, "allreduce rejected: " + error);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      std::vector<uint64_t> payload(8192 / 8);
      const uint64_t send_id = buses[0]->send(
          1, payload.data(), 8192, BusMessageClass::kLatency, &error);
      CHECK(send_id == 0 && error.find("collective mode") != std::string::npos,
            "send() must be closed in collective mode: " + error);
      const dgpp::net::BusAllReduceResult r =
          buses[0]->wait_allreduce(id, 15000);
      CHECK(r.ok, "collective failed: " + r.error);
      peer_worker.join();
      cudaFree(dev);
      for (auto& bus : buses) bus->quiesce();
      for (auto& bus : buses) bus->stop();
    }
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce: {} total failures", g_failures);
}

// One rank's share of a pre-staged run: stage_next(), a GPU kernel writes
// the pinned buffer (the seam's producing half), allreduce_staged(), wait,
// and an in-place bitwise check against the canonical-chain oracle. The
// result never leaves the pinned buffer — exactly the model's shape.
int allreduce_staged_rank_work(CollectiveBus& bus, int world, int my_rank,
                               size_t elems, int iters,
                               std::vector<uint16_t>* want) {
  // The producing kernel runs on a private stream and the worker
  // STREAM-syncs before submit — never cudaDeviceSynchronize, which is a
  // whole-device barrier and deadlocks against peer collective kernels
  // spinning on this device in the loopback process (DESIGN §6.3's
  // allocation-phase ordering rule, same class).
  cudaStream_t fill_stream = nullptr;
  if (cudaStreamCreateWithFlags(&fill_stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: fill stream create failed", my_rank);
    return 1;
  }
  int failures = 0;
  for (int iter = 0; iter < iters; ++iter) {
    std::string error;
    uint16_t* staged = static_cast<uint16_t*>(bus.stage_next(&error));
    if (staged == nullptr) {
      DGPP_LOG_ERROR("rank {}: stage_next rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    // Contract pins (rank 0, first iteration, before consuming): one
    // handout at a time, and a device-source collective must not stomp
    // it. Counted, not CHECKed — workers report home (g_failures is
    // main-thread state).
    if (my_rank == 0 && iter == 0) {
      std::string pin_error;
      if (bus.stage_next(&pin_error) != nullptr ||
          pin_error.find("one pre-stage handout") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: double stage_next not rejected: {}", pin_error);
        ++failures;
      }
      if (bus.allreduce(nullptr, nullptr, elems, &pin_error) != 0 ||
          pin_error.find("pre-stage handout is held") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: allreduce under a held handout not rejected: {}",
                       pin_error);
        ++failures;
      }
    }
    // The producing kernel writes the send source directly; a stream sync
    // (not a device sync — see the loopback note above) orders it before
    // the submit, which is the seam's contract.
    staged_fill_kernel<<<1, 1, 0, fill_stream>>>(staged, elems, my_rank);
    if (cudaStreamSynchronize(fill_stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: staged fill kernel failed", my_rank);
      ++failures;
      break;
    }
    const uint64_t id = bus.allreduce_staged(elems, &error);
    if (id == 0) {
      DGPP_LOG_ERROR("rank {}: allreduce_staged rejected: {}", my_rank, error);
      ++failures;
      break;
    }
    const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 15000);
    if (!r.ok) {
      DGPP_LOG_ERROR("rank {}: staged collective failed: {}", my_rank,
                     r.error);
      ++failures;
      break;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < elems; ++i)
      if (staged[i] != (*want)[i]) ++mismatches;
    if (mismatches != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: {} staged mismatches", my_rank, iter,
                     mismatches);
      ++failures;
      break;
    }
    if (my_rank == 0 && iter == 0) {
      // The handout was consumed; a staged submit without one must fail.
      std::string pin_error;
      if (bus.allreduce_staged(elems, &pin_error) != 0 ||
          pin_error.find("no held pre-stage handout") == std::string::npos) {
        DGPP_LOG_ERROR("rank 0: staged submit without handout not rejected: {}",
                       pin_error);
        ++failures;
      }
    }
  }
  cudaStreamDestroy(fill_stream);
  return failures;
}

void scenario_allreduce_staged() {
  // Pre-stage seam (§6.3 evolution): the producing kernel writes the
  // transport's pinned send source directly and the collective runs with
  // zero staging copies (world 2) or only the pinned fan-out (world 4).
  // Bitwise against the canonical-chain oracle, across ring generations.
  const size_t elems = 4096;
  int failures = 0;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29901 : 29902;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("staged world {} failed to start", world);
      ++failures;
      continue;
    }
    // The oracle chain over every rank's pattern.
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_staged_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems, 6, &want);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r)
      world_failures += rank_failures[static_cast<size_t>(r)];
    CHECK(world_failures == 0,
          "staged world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_staged: world {} clean", world);
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_staged: {} total failures", g_failures);
}

// One rank's share of a bulk-collective run: the machine over N iterations
// of one size — (1) allreduce_bulk verified bitwise against the canonical
// chain, (2) the latency-chunked path over the same buffer, verified
// bitwise against BOTH the oracle and the bulk run's bytes (the two paths
// compute the identical per-element chain; any divergence is a protocol
// bug, not noise). Returns failures home.
int allreduce_bulk_rank_work(CollectiveBus& bus, int world, int my_rank,
                             size_t elems, int iters,
                             std::vector<uint16_t>* want) {
  // Loopback writer's discipline: EVERY device op in a rank worker runs on
  // a private stream. Synchronous cudaMemcpy/cudaMemset are device-wide
  // barriers (legacy default stream) and deadlock against a peer rank's
  // spinning collective kernel — the §6.3 rule, which has now bitten this
  // harness three separate ways (cudaDeviceSynchronize, sync H2D, memset).
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk stream create failed", my_rank);
    return 1;
  }
  std::vector<uint16_t> host_src;
  fill_rank_bf16(&host_src, elems, my_rank);
  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk device alloc failed", my_rank);
    return 1;
  }
  if (cudaMemcpyAsync(dev_src, host_src.data(), elems * 2,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: bulk H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    return 1;
  }

  int failures = 0;
  std::vector<uint16_t> got(elems, 0), bulk_run(elems, 0);
  for (int iter = 0; iter < iters; ++iter) {
    // ---- the bulk machine (RS + AG, segments internal) -----------------
    cudaMemsetAsync(dev_dst, 0, elems * 2, stream);
    cudaStreamSynchronize(stream);
    {
      std::string error;
      const uint64_t id = bus.allreduce_bulk(dev_src, dev_dst, elems, &error);
      if (id == 0) {
        DGPP_LOG_ERROR("rank {}: allreduce_bulk rejected: {}", my_rank,
                       error);
        ++failures;
        break;
      }
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 60000);
      if (!r.ok) {
        DGPP_LOG_ERROR("rank {}: bulk collective failed: {}", my_rank,
                       r.error);
        ++failures;
        break;
      }
    }
    if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: bulk D2H failed", my_rank);
      ++failures;
      break;
    }
    size_t mismatches = 0;
    // Bring-up fingerprint: which stripes mismatch (0=clean 1=partial
    // 2=fully wrong) plus the first/last mismatching element — the
    // distribution localizes a mapping bug faster than a count alone.
    size_t first_bad = elems, last_bad = 0;
    for (size_t i = 0; i < elems; ++i) {
      if (got[i] == (*want)[i]) continue;
      ++mismatches;
      first_bad = std::min(first_bad, i);
      last_bad = i;
    }
    if (mismatches != 0) {
      const size_t stripe = 131072;  // loopback bulk_slot_bytes / 2
      std::string map;
      for (size_t s = 0; s * stripe < elems; ++s) {
        const size_t b = s * stripe;
        const size_t e = std::min(elems, b + stripe);
        size_t bad = 0;
        for (size_t i = b; i < e; ++i)
          if (got[i] != (*want)[i]) ++bad;
        map += bad == 0 ? "0" : (bad == e - b ? "2" : "1");
      }
      DGPP_LOG_ERROR("rank {} bulk iter {}: {} oracle mismatches "
                     "[{},{}] stripes {}",
                     my_rank, iter, mismatches, first_bad, last_bad, map);
      for (size_t i = first_bad; i < first_bad + 4 && i < elems; ++i)
        DGPP_LOG_ERROR("rank {} elem {}: got {:04x} want {:04x} (src {:04x})",
                       my_rank, i, got[i], (*want)[i], host_src[i]);
      ++failures;
      break;
    }
    if (iter == 0) bulk_run = got;

    // ---- the latency-chunked path over the same buffer ------------------
    // Same chain per element, chunked into latency slots; the result must
    // be BITWISE identical to the bulk machine's.
    cudaMemsetAsync(dev_dst, 0, elems * 2, stream);
    cudaStreamSynchronize(stream);
    bool chunk_failed = false;
    for (size_t base = 0; base < elems; base += 4096) {
      const size_t n = std::min<size_t>(4096, elems - base);
      std::string error;
      const uint64_t id =
          bus.allreduce(dev_src + base, dev_dst + base, n, &error);
      if (id == 0) {
        DGPP_LOG_ERROR("rank {}: chunked allreduce rejected: {}", my_rank,
                       error);
        ++failures;
        chunk_failed = true;
        break;
      }
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
      if (!r.ok) {
        DGPP_LOG_ERROR("rank {}: chunked collective failed: {}", my_rank,
                       r.error);
        ++failures;
        chunk_failed = true;
        break;
      }
    }
    if (chunk_failed) break;
    if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: chunked D2H failed", my_rank);
      ++failures;
      break;
    }
    if (std::memcmp(got.data(), bulk_run.data(), elems * 2) != 0) {
      DGPP_LOG_ERROR("rank {} iter {}: bulk and chunked paths diverged "
                     "bitwise",
                     my_rank, iter);
      ++failures;
      break;
    }
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaStreamDestroy(stream);
  return failures;
}

void scenario_allreduce_bulk() {
  // The prefill class (§6.3): segment-quantized reduce-scatter + allgather
  // over the bulk pool. Sizes chosen to exercise the degenerate geometry:
  // below one stripe (empty shards at W=4), two stripes (more empty
  // shards), and a multi-segment buffer with a partial tail stripe and
  // shard/segment straddling — every result bitwise against the canonical
  // chain and against the latency-chunked path (identical chains).
  // DGPP_BULK_ONE_LANE=1 bisects lane-dependent failures (bring-up).
  int failures = 0;
  const size_t stripe = 262144 / 2;  // loopback option bulk_slot_bytes/2
  const bool one_lane = std::getenv("DGPP_BULK_ONE_LANE") != nullptr;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29904 : 29905;
    for (const size_t elems : {static_cast<size_t>(2048), 2 * stripe + 100,
                               41 * stripe - 32}) {
      std::vector<std::unique_ptr<CollectiveBus>> world_buses =
          start_world(world, port, one_lane);
      if (world_buses.empty()) {
        DGPP_LOG_ERROR("bulk world {} failed to start", world);
        ++failures;
        break;
      }
      // Oracle chain over every rank's pattern.
      std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
      for (int r = 0; r < world; ++r)
        fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
      std::vector<uint16_t> want(elems, 0);
      for (size_t i = 0; i < elems; ++i) {
        float acc = 0.0f;
        for (int r = 0; r < world; ++r)
          acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
        want[i] = dgpp::net::bf16_from_f32_rne(acc);
      }
      std::vector<std::thread> workers;
      std::vector<int> rank_failures(static_cast<size_t>(world), 0);
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < world; ++r)
        workers.emplace_back([&, r] {
          rank_failures[static_cast<size_t>(r)] = allreduce_bulk_rank_work(
              *world_buses[static_cast<size_t>(r)], world, r, elems, 2,
              &want);
        });
      for (auto& t : workers) t.join();
      int world_failures = 0;
      for (int r = 0; r < world; ++r)
        world_failures += rank_failures[static_cast<size_t>(r)];
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      CHECK(world_failures == 0,
            "bulk world " + std::to_string(world) + " elems " +
                std::to_string(elems) + " had " +
                std::to_string(world_failures) + " failures");
      DGPP_LOG_INFO("bulk case world={} elems={}: clean ({:.0f}ms for 2 "
                    "iterations of both paths)",
                    world, elems, ms);
      for (auto& bus : world_buses) bus->quiesce();
      for (auto& bus : world_buses) bus->stop();
    }
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_bulk: {} total failures", g_failures);
}

// One rank's share of a graph-era run (§6.2): the decode step's fixed
// launch sequence — collectives with GEMM-stand-in compute between them —
// recorded once, replayed per step. Every replay's result must be bitwise
// the canonical-chain oracle AND the eager machine's bytes (the recorded
// fold is the same chain; the doorbell/claim machinery is what differs).
// The replay loop is also the MIXED-ERA gate: eager collectives (the pick
// class, the staged seam, the prefill bulk class) run BETWEEN windows —
// consuming generations from the shared counter — and every later window
// must adopt around them and stay bitwise.
// Returns failures home (CHECK is main-thread-only).
int allreduce_graph_rank_work(CollectiveBus& bus, int world, int my_rank,
                              size_t elems, int gens_per_step, int replays,
                              const std::vector<uint16_t>* want) {
  // Loopback discipline: every device op on a private stream; the capture
  // is ThreadLocal (other rank workers capture concurrently).
  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
      cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph stream create failed", my_rank);
    return 1;
  }
  std::vector<uint16_t> host_src;
  fill_rank_bf16(&host_src, elems, my_rank);
  uint16_t* dev_src = nullptr;
  uint16_t* dev_dst = nullptr;
  float* warm_buf = nullptr;
  // The prefill-class interlude buffer: the bulk machine shards on the
  // bulk-slot grid, so it needs at least one stripe per rank (world x
  // stripe elems; 4 x 131072 = 1 MiB at the default geometry).
  const size_t bulk_elems =
      static_cast<size_t>(world) * (262144 / 2) * 4;
  uint16_t* dev_bsrc = nullptr;
  uint16_t* dev_bdst = nullptr;
  if (cudaMalloc(&dev_src, elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_dst, elems * 2) != cudaSuccess ||
      cudaMalloc(&warm_buf, 256 * 4) != cudaSuccess ||
      cudaMalloc(&dev_bsrc, bulk_elems * 2) != cudaSuccess ||
      cudaMalloc(&dev_bdst, bulk_elems * 2) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph device alloc failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return 1;
  }
  // Bulk oracle: the canonical chain over every rank's fill, computed
  // locally (each rank knows every rank's deterministic input).
  std::vector<uint16_t> bulk_want(bulk_elems, 0);
  {
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], bulk_elems, r);
    for (size_t i = 0; i < bulk_elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      bulk_want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    // My own slice up to the device source.
    if (cudaMemcpyAsync(dev_bsrc, all_src[static_cast<size_t>(my_rank)].data(),
                        bulk_elems * 2, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
      DGPP_LOG_ERROR("rank {}: bulk H2D failed", my_rank);
      cudaFree(dev_src);
      cudaFree(dev_dst);
      cudaFree(warm_buf);
      cudaFree(dev_bsrc);
      cudaFree(dev_bdst);
      cudaStreamDestroy(stream);
      return 1;
    }
  }
  if (cudaMemcpyAsync(dev_src, host_src.data(), elems * 2,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemsetAsync(warm_buf, 0, 256 * 4, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    DGPP_LOG_ERROR("rank {}: graph H2D failed", my_rank);
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return 1;
  }

  int failures = 0;
  auto fail = [&](const std::string& why) {
    DGPP_LOG_ERROR("rank {}: {}", my_rank, why);
    ++failures;
  };

  // ---- eager baseline: the same fold before the era opens ----------------
  std::vector<uint16_t> eager_bytes(elems, 0);
  {
    std::string error;
    const uint64_t id = bus.allreduce(dev_src, dev_dst, elems, &error);
    if (id == 0) {
      fail("eager baseline rejected: " + error);
    } else {
      const dgpp::net::BusAllReduceResult r = bus.wait_allreduce(id, 30000);
      if (!r.ok) {
        fail("eager baseline failed: " + r.error);
      } else if (cudaMemcpyAsync(eager_bytes.data(), dev_dst, elems * 2,
                                 cudaMemcpyDeviceToHost, stream) !=
                     cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("eager baseline D2H failed");
      }
    }
  }
  if (failures != 0) {
    cudaFree(dev_src);
    cudaFree(dev_dst);
    cudaFree(warm_buf);
    cudaFree(dev_bsrc);
    cudaFree(dev_bdst);
    cudaStreamDestroy(stream);
    return failures;
  }

  // ---- contract pins (rank 0, pre-era) -------------------------------------
  if (my_rank == 0) {
    std::string error;
    if (bus.allreduce_record(stream, dev_src, dev_dst, elems, &error) ||
        error.find("no open graph session") == std::string::npos) {
      fail("record without an open session not rejected: " + error);
    }
    if (bus.graph_replay_arm(&error) ||
        error.find("no recorded graph") == std::string::npos) {
      fail("arm without a recorded graph not rejected: " + error);
    }
  }

  // ---- capture the step ----------------------------------------------------
  // G collectives with dependent compute between them (the GEMM stand-in
  // keeps the clocks honest and forces the stream-order serialization the
  // walk's one-gen-at-a-time assumption is built on), plus trailing
  // compute — the decode step's shape.
  std::string error;
  if (!bus.graph_record_begin(&error)) {
    fail("graph_record_begin rejected: " + error);
  } else {
    if (my_rank == 0) {
      std::string pin_error;
      if (bus.graph_replay_arm(&pin_error) ||
          pin_error.find("no recorded graph") == std::string::npos)
        fail("arm during an open session not rejected: " + pin_error);
      // The eager gate while RECORDING: rejections are host-side (no CUDA
      // call, no registry entry), so they are capture-safe.
      if (bus.allreduce(dev_src, dev_dst, elems, &pin_error) != 0 ||
          pin_error.find("recording") == std::string::npos)
        fail("eager allreduce during recording not rejected: " + pin_error);
      if (bus.stage_next(&pin_error) != nullptr ||
          pin_error.find("recording") == std::string::npos)
        fail("stage_next during recording not rejected: " + pin_error);
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t cap =
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (cap != cudaSuccess) {
      fail("stream capture begin failed");
    } else {
      for (int g = 0; g < gens_per_step; ++g) {
        dgpp::net::launch_bus_warm_work(stream, warm_buf, 20000);
        if (!bus.allreduce_record(stream, dev_src, dev_dst, elems, &error)) {
          fail("allreduce_record rejected: " + error);
          break;
        }
      }
      dgpp::net::launch_bus_warm_work(stream, warm_buf, 20000);
    }
    if (cap == cudaSuccess) {
      if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || !graph) {
        fail("stream capture end failed");
      }
    }
    if (failures == 0) {
      if (!bus.graph_record_end(&error)) {
        fail("graph_record_end rejected: " + error);
      }
    } else {
      // Close the half-open session so the bus state stays defined; the
      // run reports the failures above.
      std::string close_error;
      bus.graph_record_end(&close_error);
    }

    // ---- era pins ----------------------------------------------------------
    if (my_rank == 0 && failures == 0) {
      // Harness sends stay closed for the era's lifetime (coll_mode);
      // eager collectives are the ones that reopen between windows.
      std::string pin_error;
      std::vector<uint64_t> payload(8);
      if (bus.send(world == 2 ? 1 - my_rank : 1, payload.data(), 64,
                   BusMessageClass::kLatency, &pin_error) != 0 ||
          pin_error.find("collective mode") == std::string::npos)
        fail("send during the graph era not rejected: " + pin_error);
    }

    // ---- replays: warm 2, then measured ------------------------------------
    cudaGraphExec_t exec = nullptr;
    if (failures == 0) {
      if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) !=
          cudaSuccess) {
        fail("graph instantiate failed");
      }
    }
    // A second, shorter/narrower shape gets its own cell slab. Replaying it
    // once before variant 0 pins selection in both directions and, because
    // its generation count differs, catches using the newly selected G when
    // waiting for the previous window.
    cudaGraph_t alt_graph = nullptr;
    cudaGraphExec_t alt_exec = nullptr;
    const int alt_gens = gens_per_step - 3;
    const size_t alt_elems = elems / 2;
    if (failures == 0) {
      if (!bus.graph_record_begin(&error, /*variant=*/1)) {
        fail("second graph variant record_begin rejected: " + error);
      } else {
        const cudaError_t cap = cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal);
        if (cap != cudaSuccess) {
          fail("second graph variant capture begin failed");
        } else {
          for (int g = 0; g < alt_gens; ++g) {
            dgpp::net::launch_bus_warm_work(stream, warm_buf, 12000);
            if (!bus.allreduce_record(stream, dev_src, dev_dst, alt_elems,
                                      &error)) {
              fail("second graph variant allreduce_record rejected: " +
                   error);
              break;
            }
          }
          dgpp::net::launch_bus_warm_work(stream, warm_buf, 12000);
          if (cudaStreamEndCapture(stream, &alt_graph) != cudaSuccess ||
              alt_graph == nullptr)
            fail("second graph variant capture end failed");
        }
        if (failures == 0 && !bus.graph_record_end(&error))
          fail("second graph variant record_end rejected: " + error);
      }
      if (failures == 0 &&
          cudaGraphInstantiate(&alt_exec, alt_graph, nullptr, nullptr, 0) !=
              cudaSuccess)
        fail("second graph variant instantiate failed");
    }
    if (exec != nullptr && alt_exec != nullptr) {
      std::vector<double> step_us;
      std::vector<uint16_t> got(elems, 0);
      std::vector<uint16_t> bulk_got(bulk_elems, 0);
      if (!bus.graph_replay_arm(&error, /*variant=*/1)) {
        fail("second graph variant arm rejected: " + error);
      } else if (cudaGraphLaunch(alt_exec, stream) != cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("second graph variant replay failed");
      } else if (!bus.graph_replay_finish(30000, &error)) {
        fail("second graph variant finish rejected: " + error);
      } else if (cudaMemcpyAsync(got.data(), dev_dst, alt_elems * 2,
                                 cudaMemcpyDeviceToHost, stream) !=
                     cudaSuccess ||
                 cudaStreamSynchronize(stream) != cudaSuccess) {
        fail("second graph variant D2H failed");
      } else if (std::memcmp(got.data(), eager_bytes.data(), alt_elems * 2) !=
                 0) {
        fail("second graph variant diverged from eager bitwise");
      }
      for (int replay = 0; replay < 2 + replays && failures == 0; ++replay) {
        if (!bus.graph_replay_arm(&error)) {
          fail("arm rejected: " + error);
          break;
        }
        // The eager gate while a window is ARMED (arm .. finish): rank 0
        // must be rejected loudly, mid-flight — the gate is host-side
        // and coll_mu-serialized, so pinning here is race-free.
        if (replay == 3 && my_rank == 0) {
          std::string pin_error;
          if (bus.allreduce(dev_src, dev_dst, elems, &pin_error) != 0 ||
              pin_error.find("armed") == std::string::npos)
            fail("eager allreduce during an armed window not rejected: " +
                 pin_error);
          if (bus.stage_next(&pin_error) != nullptr ||
              pin_error.find("armed") == std::string::npos)
            fail("stage_next during an armed window not rejected: " +
                 pin_error);
          if (bus.allreduce_bulk(dev_bsrc, dev_bdst, bulk_elems,
                                 &pin_error) != 0 ||
              pin_error.find("armed") == std::string::npos)
            fail("allreduce_bulk during an armed window not rejected: " +
                 pin_error);
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
          fail("graph launch failed");
          break;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("replay stream sync failed");
          break;
        }
        if (!bus.graph_replay_finish(30000, &error)) {
          fail("finish rejected: " + error);
          // TEMP bring-up: the per-gen cells as this rank's engine last
          // saw them — gen/ready/done/status localize a dead kernel vs a
          // blind engine instantly.
          bus.dump_graph_cells("finish");
          break;
        }
        const double us =
            std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0)
                .count();
        if (replay >= 2) step_us.push_back(us);
        // Bitwise: every replay against the oracle AND the eager bytes.
        if (cudaMemcpyAsync(got.data(), dev_dst, elems * 2,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
          fail("replay D2H failed");
          break;
        }
        if (std::memcmp(got.data(), eager_bytes.data(), elems * 2) != 0) {
          fail("replay " + std::to_string(replay) +
               " diverged from the eager machine bitwise");
          break;
        }
        if (std::memcmp(got.data(), want->data(), elems * 2) != 0) {
          fail("replay " + std::to_string(replay) +
               " diverged from the canonical-chain oracle");
          break;
        }
        // ---- the mixed-era interlude (eager between windows) ------------
        // After the warmups: eager collectives run between replay windows,
        // consuming generations from the shared counter exactly as the
        // serving process interleaves prefill folds and the pick between
        // decode steps. Every LATER window is the proof — it must adopt
        // around the consumed numbers and stay bitwise.
        if (replay == 2) {
          std::string eager_error;
          // (a) the pick's class: a latency one-shot.
          {
            const uint64_t id =
                bus.allreduce(dev_src, dev_dst, elems, &eager_error);
            if (id == 0) {
              fail("mixed-era eager allreduce rejected: " + eager_error);
            } else {
              const dgpp::net::BusAllReduceResult r =
                  bus.wait_allreduce(id, 30000);
              if (!r.ok)
                fail("mixed-era eager allreduce failed: " + r.error);
            }
          }
          // (b) the prefill seam: a staged handout, folded in place.
          {
            void* handout = bus.stage_next(&eager_error);
            if (handout == nullptr) {
              fail("mixed-era stage_next rejected: " + eager_error);
            } else {
              std::memcpy(handout, host_src.data(), elems * 2);
              const uint64_t id = bus.allreduce_staged(elems, &eager_error);
              if (id == 0) {
                fail("mixed-era allreduce_staged rejected: " + eager_error);
              } else {
                const dgpp::net::BusAllReduceResult r =
                    bus.wait_allreduce(id, 30000);
                if (!r.ok) {
                  fail("mixed-era allreduce_staged failed: " + r.error);
                } else if (std::memcmp(handout, eager_bytes.data(),
                                       elems * 2) != 0) {
                  fail("mixed-era staged fold diverged bitwise");
                }
              }
            }
          }
          // (c) the prefill folds' class: a bulk machine run.
          {
            const uint64_t id = bus.allreduce_bulk(dev_bsrc, dev_bdst,
                                                   bulk_elems, &eager_error);
            if (id == 0) {
              fail("mixed-era allreduce_bulk rejected: " + eager_error);
            } else {
              const dgpp::net::BusAllReduceResult r =
                  bus.wait_allreduce(id, 30000);
              if (!r.ok) {
                fail("mixed-era allreduce_bulk failed: " + r.error);
              } else if (
                  cudaMemcpyAsync(bulk_got.data(), dev_bdst, bulk_elems * 2,
                                  cudaMemcpyDeviceToHost, stream) !=
                      cudaSuccess ||
                  cudaStreamSynchronize(stream) != cudaSuccess) {
                fail("mixed-era bulk D2H failed");
              } else if (std::memcmp(bulk_got.data(), bulk_want.data(),
                                     bulk_elems * 2) != 0) {
                fail("mixed-era bulk fold diverged bitwise");
              }
            }
          }
        }
      }
      if (failures == 0) {
        DGPP_LOG_INFO(
            "rank {}: {} replays x {} gens clean (eager interlude after "
            "warmup); step p50={:.1f}us ({:.1f}us per collective)",
            my_rank, replays, gens_per_step, percentile(step_us, 0.5),
            percentile(step_us, 0.5) / gens_per_step);
      }
    }
    if (exec != nullptr) cudaGraphExecDestroy(exec);
    if (alt_exec != nullptr) cudaGraphExecDestroy(alt_exec);
    if (alt_graph) cudaGraphDestroy(alt_graph);
    if (graph) cudaGraphDestroy(graph);
  }

  cudaFree(dev_src);
  cudaFree(dev_dst);
  cudaFree(warm_buf);
  cudaFree(dev_bsrc);
  cudaFree(dev_bdst);
  cudaStreamDestroy(stream);
  return failures;
}

void scenario_allreduce_graph() {
  // The decode step (§6.2): capture the fixed launch sequence once,
  // replay it per step. Pins: bitwise vs the oracle and the eager
  // machine across replays and ring generations, the contract pins
  // (record/arm/eager/send rejections), and the walk's determinism
  // across worlds.
  const size_t elems = 4096;  // 8 KiB, the decode unit
  int failures = 0;
  // Above the staging-ring depth (8) on purpose: intra-window row reuse
  // is the walk's serialized-reuse discipline, and the decode graph is
  // 90 nodes deep in it.
  const int gens_per_step = 12;
  const int replays = 8;

  for (const int world : {2, 4}) {
    const uint16_t port = world == 2 ? 29906 : 29907;
    std::vector<std::unique_ptr<CollectiveBus>> world_buses =
        start_world(world, port);
    if (world_buses.empty()) {
      DGPP_LOG_ERROR("graph world {} failed to start", world);
      ++failures;
      continue;
    }
    std::vector<std::vector<uint16_t>> all_src(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r)
      fill_rank_bf16(&all_src[static_cast<size_t>(r)], elems, r);
    std::vector<uint16_t> want(elems, 0);
    for (size_t i = 0; i < elems; ++i) {
      float acc = 0.0f;
      for (int r = 0; r < world; ++r)
        acc += dgpp::net::bf16_to_f32(all_src[static_cast<size_t>(r)][i]);
      want[i] = dgpp::net::bf16_from_f32_rne(acc);
    }
    std::vector<std::thread> workers;
    std::vector<int> rank_failures(static_cast<size_t>(world), 0);
    for (int r = 0; r < world; ++r)
      workers.emplace_back([&, r] {
        rank_failures[static_cast<size_t>(r)] = allreduce_graph_rank_work(
            *world_buses[static_cast<size_t>(r)], world, r, elems,
            gens_per_step, replays, &want);
      });
    for (auto& t : workers) t.join();
    int world_failures = 0;
    for (int r = 0; r < world; ++r) {
      if (rank_failures[static_cast<size_t>(r)] != 0)
        DGPP_LOG_ERROR("graph world {} rank {} reported failures", world, r);
      world_failures += rank_failures[static_cast<size_t>(r)];
    }
    CHECK(world_failures == 0,
          "graph world " + std::to_string(world) + " had " +
              std::to_string(world_failures) + " failures");
    for (auto& bus : world_buses) bus->quiesce();
    for (auto& bus : world_buses) bus->stop();
    DGPP_LOG_INFO("scenario allreduce_graph: world {} clean", world);
  }

  g_failures += failures;
  DGPP_LOG_INFO("scenario allreduce_graph: {} total failures", g_failures);
}

void scenario_idle_gap_collective() {
  // The real-mesh shape, found by the M5 exit-gate fabric run (2026-08-30):
  // buses sit IDLE for seconds while a host loads weights (a cold peer's
  // first inter-collective gap — cold NVMe reads of replicated globals
  // before its first boundary), then the first collective posts against a
  // lane whose last traffic flowed before the gap. The lane watchdog must
  // arm from the POST, not inherit the idle period as a stall — otherwise
  // every cold-start fabric forward dies on its first collective.
  // Tight budget on purpose: the gaps below exceed it by a wide margin, so
  // stale arming cannot pass by accident.
  BusOptions a_opt = base_options(0, 29903);
  BusOptions b_opt = base_options(1, 29903);
  // Collective mode (the per-collective kernel owns claims) — and with it
  // no persistent consumer kernels spinning on the device, which would
  // wedge a worker's cudaMalloc mid-scenario (the bring-up hazard class).
  a_opt.launch_consumers = false;
  b_opt.launch_consumers = false;
  a_opt.completion_timeout_ms = 800;
  b_opt.completion_timeout_ms = 800;
  auto [a, b] = start_pair(a_opt, b_opt);
  if (!a || !b) return;

  const size_t elems = 4096;
  std::vector<int> failures(2, 0);
  for (int round = 0; round < 3; ++round) {
    // The idle gap — longer than the watchdog budget, before the first
    // collective AND between rounds.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    std::vector<std::thread> workers;
    workers.emplace_back(
        [&] { failures[0] += allreduce_rank_work(*a, 2, 0, elems, 1); });
    workers.emplace_back(
        [&] { failures[1] += allreduce_rank_work(*b, 2, 1, elems, 1); });
    for (auto& t : workers) t.join();
  }
  const int total = failures[0] + failures[1];
  CHECK(total == 0, "idle-gap collectives: " + std::to_string(total) +
                        " failures across 3 rounds (lane watchdog fired on "
                        "an idle gap?)");
  a->quiesce();
  b->quiesce();
  a->stop();
  b->stop();
  DGPP_LOG_INFO("scenario idle_gap_collective: 3 collectives across 2s "
                "idle gaps, clean");
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
  // One process, several ranks, each with kernels that spin on a peer's
  // doorbell: CUDA's default LAZY module loading deadlocks that shape (a
  // first launch waits for an idle device — see bus_kernel.hpp,
  // bus_preload_kernels). The bus preloads its own kernels; the model's
  // compute kernels launched beside a live collective are loaded eagerly
  // here, before the first CUDA call. Production is one rank per box.
  setenv("CUDA_MODULE_LOADING", "EAGER", /*overwrite=*/0);

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
  scenario_allreduce();
  scenario_allreduce_staged();
  scenario_allreduce_bulk();
  scenario_allreduce_graph();
  scenario_idle_gap_collective();
  scenario_geometry_mismatch();
  scenario_stop_releases_waiters();

  DGPP_LOG_INFO("bus_test: {} failures", g_failures);
  return g_failures == 0 ? 0 : 1;
}
