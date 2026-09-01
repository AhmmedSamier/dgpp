// The burst-wedge repro (M6 d3 bring-up hunt, 2026-08-31) — VERDICT: the
// engine was innocent; the wedge was THIS HARNESS's cudaMallocManaged
// between collectives, and the class is now a documented discipline.
//
// WHAT THIS APP REPRODUCED: 28 staged 2048-elem collectives + rapid small
// plains stalled "every run" with an 8-second march of releases — an
// apparent transport wedge (sent-but-never-received pairs, frozen PSN
// gaps, no RNR, no errors). The hunt's instruments (per-kernel first-
// claim stamps, per-claim payload/doorbell dumps, live QP state via
// --qp_state_dump, claim/post/recycle CE trails) exonerated the engine
// at every level: claims exact, posts real, arrivals landing, recycles
// complete, RQs armed.
//
// THE ACTUAL MECHANISM: cudaMallocManaged between collectives is a
// DEVICE-SYNCHRONIZING call. With peers' per-collective kernels spinning
// on the shared GPU — waiting for doorbells that need THIS thread's next
// submission — the alloc deadlocks host against device; each rank's
// lane watchdog (completion_timeout_ms) then poisons one spinning
// kernel per period, which is EXACTLY the observed 8-second march of
// "releases" (the march rescaled when --timeout-ms changed — the timer
// was ours, not the transport's). Pre-allocating the scratch before the
// world forms makes every previously-stalling shape pass.
//
// THE DISCIPLINE (the loopback bring-up's first-collective lesson, now
// generalized): NO synchronizing CUDA calls on the decode path —
// allocations and device-wide syncs happen before the world forms (or
// in idle windows), never between collectives. The same class as the
// set_reader_stream fix (a loader sync vs spinning collective kernels).
//
// Kept as the regression instrument: the staged-then-plain sequence at
// hostile sizes/counts, with zero host-side allocations in the loop.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "net/collective_bus.hpp"

namespace {

// The GEMM's stand-in: writes the pinned handout FROM THE DEVICE (the
// model's staged boundaries are device-written; host writes are the one
// mechanical difference the first repro lacked).
__global__ void fill_row_kernel(uint16_t* row, size_t elems,
                                uint16_t value) {
  for (size_t i = threadIdx.x; i < elems; i += blockDim.x) row[i] = value;
}
cudaStream_t g_fill_stream = nullptr;

}  // namespace

using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

struct Config {
  int world = 4;
  uint16_t port = 29930;
  int staged_iters = 14;
  size_t staged_elems = 2048;
  int plain_iters = 1;
  size_t plain_elems = 16;
  int timeout_ms = 8000;
  // --pick-race N: N iterations of [staged folds + the exact two-
  // collective pick shape] under randomized per-rank skew, with an exact
  // oracle after EVERY collective. The 2026-09-01 fabric hunt's racer.
  int pick_race_iters = 0;
  int staged_per_pick = 3;   // boundary folds per decode step
  int skew_us = 0;            // max random sleep at each skew point
  uint32_t seed = 1;          // rank r seeds with seed + r
  int lat_slots = 8;          // ring depth (fabric default 32)
  // 4-process fabric mode: my rank fixed, world formed across hosts.
  int my_rank = -1;
  std::string host = "";
  int rendezvous_ms = 20000;  // the roster window (launch scripts need
                              // more than a human's interactive launch)
};

BusOptions loop_options(const Config& c, int rank, uint16_t port,
                        int timeout_ms) {
  BusOptions o;
  o.world_size = c.world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = c.rendezvous_ms;
  o.lat_slots = c.lat_slots;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = timeout_ms;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

// The bf16 helpers the oracles need (exact small integers only — the
// fold chain is fp32 over bf16, so integer values up to 256 pass
// through bitwise-exactly; that exactness IS the oracle).
uint16_t bf16_bits_of(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1;
  return static_cast<uint16_t>((x + 0x7FFFu + lsb) >> 16);
}

struct RaceRng {
  uint64_t s = 0;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  int below(int n) { return n > 0 ? static_cast<int>(next() % static_cast<uint64_t>(n)) : 0; }
};

void skew_sleep(RaceRng& rng, const Config& c) {
  if (c.skew_us <= 0) return;
  const int us = rng.below(c.skew_us);
  if (us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// One racer iteration on one rank: the decode-step shape (boundary
// staged folds, then the pick's gather+broadcast) with an exact oracle
// after every collective. Returns "" on success.
std::string pick_race_iter(const Config& c, int rank, int world,
                           CollectiveBus* bus, uint16_t* scratch,
                           int iter, RaceRng& rng,
                           std::vector<int32_t>& expected_ids,
                           std::vector<int32_t>& local_ids) {
  const size_t elems = 16;

  // ---- the "forward": staged boundary folds, each verified exactly ----
  skew_sleep(rng, c);
  for (int j = 0; j < c.staged_per_pick; ++j) {
    std::string err;
    void* handout = bus->stage_next(&err);
    if (handout == nullptr) return "stage_next rejected: " + err;
    // The GEMM's stand-in: a HOST write of the pinned handout. The first
    // cut filled from a kernel + stream sync — a synchronizing CUDA call
    // between collectives, which the loopback shared GPU turns into the
    // documented starvation class (one rank's fill blocked 8s behind
    // three spinning peer kernels — exactly the discipline the burst-
    // wedge hunt wrote down). The handout's staged semantics (pinned,
    // in-place fold) are what the race exercises; the real GEMM writer
    // is the fabric app's problem, and it does not share a GPU.
    uint16_t* row = static_cast<uint16_t*>(handout);
    const uint16_t v = bf16_bits_of(static_cast<float>(rank + 1));
    for (size_t e = 0; e < c.staged_elems; ++e) row[e] = v;
    const uint64_t id = bus->allreduce_staged(c.staged_elems, &err);
    if (id == 0) return "staged submit rejected: " + err;
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    if (!res.ok) return "staged wait failed: " + res.error;
    // Oracle: every rank contributed (r+1); the fold is rank-ordered
    // fp32 over exact bf16 integers, so every element must be the exact
    // sum. A stale leak here is a boundary-fold corruption — logged with
    // the iteration and fold index.
    const uint16_t want = bf16_bits_of(static_cast<float>(world * (world + 1) / 2));
    if (row[0] != want || row[c.staged_elems - 1] != want) {
      return std::string("iter ") + std::to_string(iter) + " fold " +
             std::to_string(j) + ": staged fold corrupt (w0=" +
             std::to_string(row[0]) + " want=" + std::to_string(want) +
             " wN=" + std::to_string(row[c.staged_elems - 1]) + ")";
    }
  }

  // ---- the pick: exactly bus_greedy_pick's shape ---------------------
  // Deterministic per-(rank, iteration) candidates, known to every
  // rank: values vary so the winner rotates.
  skew_sleep(rng, c);
  std::vector<int32_t> values(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    const uint32_t vr = 100 + ((c.seed * 31 + static_cast<uint32_t>(r) * 7 +
                                static_cast<uint32_t>(iter) * 13) %
                               5);
    values[static_cast<size_t>(r)] = static_cast<int32_t>(vr);
    local_ids[static_cast<size_t>(r)] = static_cast<int32_t>(
        (c.seed * 101 + static_cast<uint32_t>(r) * 10007 +
         static_cast<uint32_t>(iter) * 37) %
        100000);
  }
  std::memset(scratch, 0, elems * 2);
  // Each rank contributes ONLY its own quadruple; the fold's sum
  // assembles the table (the bus_greedy_pick gather discipline — a
  // sum over disjoint per-rank slots IS a gather). The first cut wrote
  // the whole table into every rank's scratch; the oracle correctly
  // called the fold out as four copies of the same table (404 = 4*101).
  {
    const size_t r = static_cast<size_t>(rank);
    scratch[r * 4 + 0] = bf16_bits_of(static_cast<float>(values[r]));
    const int32_t idr = local_ids[r];
    scratch[r * 4 + 1] = static_cast<uint16_t>(idr & 63);
    scratch[r * 4 + 2] = static_cast<uint16_t>((idr >> 6) & 63);
    scratch[r * 4 + 3] = static_cast<uint16_t>((idr >> 12) & 63);
  }
  {
    std::string err;
    const uint64_t id = bus->allreduce(scratch, scratch, elems, &err);
    if (id == 0) return "gather submit rejected: " + err;
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    if (!res.ok) return "gather wait failed: " + res.error;
  }
  // Oracle: the gather table must be every rank's exact quadruple.
  for (int r = 0; r < world; ++r) {
    const uint16_t* q = scratch + static_cast<size_t>(r) * 4;
    const int32_t idr = local_ids[static_cast<size_t>(r)];
    if (q[0] != bf16_bits_of(static_cast<float>(values[static_cast<size_t>(r)])) ||
        q[1] != (idr & 63) || q[2] != ((idr >> 6) & 63) ||
        q[3] != ((idr >> 12) & 63)) {
      std::string words;
      for (size_t i = 0; i < elems; ++i) {
        if (i) words += ",";
        words += std::to_string(scratch[i]);
      }
      return "iter " + std::to_string(iter) +
             " gather: table corrupt at rank " + std::to_string(r) +
             " slot [" + words + "]";
    }
  }
  // The winner: glm_sample's canonical (value desc, id asc).
  int32_t winner = 0;
  for (int r = 1; r < world; ++r) {
    const size_t rr = static_cast<size_t>(r);
    const size_t bw = static_cast<size_t>(winner);
    if (values[rr] > values[bw] ||
        (values[rr] == values[bw] && local_ids[rr] < local_ids[bw]))
      winner = r;
  }
  expected_ids[static_cast<size_t>(iter) % expected_ids.size()] =
      local_ids[static_cast<size_t>(winner)];

  // ---- broadcast: rank 0's digits, verified as the decoded winner ----
  std::memset(scratch, 0, elems * 2);
  if (rank == 0) {
    scratch[0] = static_cast<uint16_t>(local_ids[static_cast<size_t>(winner)] & 63);
    scratch[1] = static_cast<uint16_t>((local_ids[static_cast<size_t>(winner)] >> 6) & 63);
    scratch[2] = static_cast<uint16_t>((local_ids[static_cast<size_t>(winner)] >> 12) & 63);
    scratch[3] = 0;
  }
  {
    std::string err;
    const uint64_t id = bus->allreduce(scratch, scratch, elems, &err);
    if (id == 0) return "broadcast submit rejected: " + err;
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    if (!res.ok) return "broadcast wait failed: " + res.error;
  }
  const int32_t decoded = static_cast<int32_t>(scratch[0]) |
                          (static_cast<int32_t>(scratch[1]) << 6) |
                          (static_cast<int32_t>(scratch[2]) << 12);
  const int32_t want = local_ids[static_cast<size_t>(winner)];
  if (decoded != want) {
    std::string words;
    for (size_t i = 0; i < elems; ++i) {
      if (i) words += ",";
      words += std::to_string(scratch[i]);
    }
    return "iter " + std::to_string(iter) +
           " broadcast: readback corrupt (winner " + std::to_string(want) +
           " decoded " + std::to_string(decoded) + ") slot [" + words + "]";
  }
  skew_sleep(rng, c);
  return "";
}

// The pick-race driver for one rank. In-process loopback: world ranks as
// threads on one bus each; fabric mode (--rank/--host): one rank, the
// world formed across processes. `scratch` is allocated by the caller
// BEFORE the world forms (the discipline: no synchronizing CUDA calls
// between collectives — a post-world managed alloc deadlocks against
// spinning peer kernels in loopback).
std::string rank_pick_race(const Config& c, int rank, CollectiveBus* bus,
                           uint16_t* scratch) {
  RaceRng rng{};
  rng.s = (static_cast<uint64_t>(c.seed) + static_cast<uint64_t>(rank)) * 6364136223846793005ULL + 1442695040888963407ULL;
  std::vector<int32_t> expected_ids(1, 0);
  std::vector<int32_t> local_ids(static_cast<size_t>(c.world), 0);
  for (int i = 0; i < c.pick_race_iters; ++i) {
    const std::string err =
        pick_race_iter(c, rank, c.world, bus, scratch, i, rng,
                       expected_ids, local_ids);
    if (!err.empty()) return err;
    if ((i + 1) % 500 == 0)
      DGPP_LOG_INFO("rank {}: pick-race {}/{} clean", rank, i + 1,
                    c.pick_race_iters);
  }
  DGPP_LOG_INFO("rank {}: pick-race {} iterations clean", rank,
                c.pick_race_iters);
  return "";
}

// One rank's phase: K staged collectives (the boundary-fold stand-in),
// then N plain small ones. Returns "" on success, the failure otherwise.
std::string rank_run(const Config& c, int rank, CollectiveBus* bus) {
  for (int i = 0; i < c.staged_iters; ++i) {
    std::string err;
    void* handout = bus->stage_next(&err);
    if (handout == nullptr) return "stage_next rejected: " + err;
    // The GEMM's stand-in: host writes the pinned staging row.
    uint16_t* row = static_cast<uint16_t*>(handout);
    fill_row_kernel<<<1, 256, 0, g_fill_stream>>>(
        row, c.staged_elems, static_cast<uint16_t>(rank + i));
    DGPP_CUDA_OK(cudaStreamSynchronize(g_fill_stream));
    const uint64_t id = bus->allreduce_staged(c.staged_elems, &err);
    if (id == 0) return "staged submit rejected: " + err;
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    if (!res.ok) return "staged wait failed: " + res.error;
  }
  DGPP_LOG_INFO("rank {}: {} staged collectives done", rank, c.staged_iters);

  // THE WEDGE WAS THE HARNESS: cudaMallocManaged between collectives is a
  // DEVICE-SYNCHRONIZING call — with this rank's peers' collective kernels
  // spinning on the shared GPU (waiting for doorbells that need THIS
  // thread's next submission), the alloc deadlocks host vs device — the
  // loopback bring-up's documented class ("a lazily constructing rank
  // deadlocks against a peer's spinning collective kernel"). Pre-allocate
  // once, before the world forms: no synchronizing calls on the decode
  // path. The staged loop's fill kernel + sync is BEFORE the world too.
  uint16_t* scratch = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&scratch, c.plain_elems * 2));
  for (int i = 0; i < c.plain_iters; ++i) {
    std::memset(scratch, 0, c.plain_elems * 2);
    scratch[0] = static_cast<uint16_t>(rank);
    std::string err;
    const uint64_t id = bus->allreduce(scratch, scratch, c.plain_elems, &err);
    if (id == 0) {
      cudaFree(scratch);
      return "plain submit rejected: " + err;
    }
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    if (!res.ok) {
      cudaFree(scratch);
      return "plain small wait failed: " + res.error;
    }
  }
  cudaFree(scratch);
  DGPP_LOG_INFO("rank {}: {} plain small collectives done", rank,
               c.plain_iters);
  return "";
}

int run(const Config& c) {
  DGPP_CUDA_OK(cudaStreamCreate(&g_fill_stream));
  // The pick-race scratch: BEFORE the world forms (the discipline at the
  // top of this file — a managed alloc between spinning collective
  // kernels deadlocks host against device in loopback). One buffer PER
  // RANK: the loopback world is four threads in ONE process, and a
  // shared scratch had every "rank" memsetting and writing the same 32
  // bytes (the first racer run folded four copies of whichever thread
  // won the write race — the bus was fine, the harness was not).
  std::vector<uint16_t*> race_scratch(static_cast<size_t>(c.world), nullptr);
  if (c.pick_race_iters > 0)
    for (auto& s : race_scratch)
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&s), 16 * 2,
                                 cudaHostAllocDefault));

  // Fabric mode: one bus, one rank, the world formed across processes.
  if (c.my_rank >= 0) {
    BusOptions o;
    o.world_size = c.world;
    o.my_rank = c.my_rank;
    o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
    o.rendezvous_port = c.port;
    o.rendezvous_host = c.host;
    o.rendezvous_timeout_ms = c.rendezvous_ms;
    o.lat_slots = c.lat_slots;
    o.lat_slot_bytes = 8192;
    o.bulk_slots = 8;
    o.bulk_slot_bytes = 262144;
    o.qp_depth = 1024;
    o.completion_timeout_ms = c.timeout_ms;
    o.consumer_deadline_s = 60.0;
    o.launch_consumers = false;
    auto bus = std::make_unique<CollectiveBus>(o);
    std::string err;
    if (!bus->start(&err)) {
      DGPP_LOG_ERROR("rank {}: bus start failed: {}", c.my_rank, err);
      for (auto& s : race_scratch) cudaFreeHost(s);
      return 2;
    }
    const std::string r =
        rank_pick_race(c, c.my_rank, bus.get(), race_scratch[0]);
    bus->quiesce();
    bus->stop();
    for (auto& s : race_scratch) cudaFreeHost(s);
    if (!r.empty()) {
      DGPP_LOG_ERROR("rank {}: {}", c.my_rank, r);
      return 1;
    }
    DGPP_LOG_INFO("REPRO: PASS (rank {} pick-race clean)", c.my_rank);
    return 0;
  }

  std::vector<std::unique_ptr<CollectiveBus>> buses;
  for (int r = 0; r < c.world; ++r)
    buses.push_back(std::make_unique<CollectiveBus>(
        loop_options(c, r, c.port, c.timeout_ms)));
  std::vector<std::string> errors(static_cast<size_t>(c.world));
  std::thread listener([&] {
    if (!buses[0]->start(&errors[0]))
      DGPP_LOG_ERROR("rank 0: bus start failed: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (size_t r = 1; r < buses.size(); ++r)
    connectors.emplace_back([&, r] {
      if (!buses[r]->start(&errors[r]))
        DGPP_LOG_ERROR("rank {}: bus start failed: {}", r, errors[r]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < c.world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) {
      DGPP_LOG_ERROR("bus world failed to start: {}", errors[r]);
      return 2;
    }

  std::vector<std::string> rank_errors(static_cast<size_t>(c.world));
  std::vector<std::thread> workers;
  for (int r = 0; r < c.world; ++r)
    workers.emplace_back([&, r] {
      try {
        rank_errors[static_cast<size_t>(r)] =
            c.pick_race_iters > 0
                ? rank_pick_race(c, r, buses[static_cast<size_t>(r)].get(),
                                 race_scratch[static_cast<size_t>(r)])
                : rank_run(c, r, buses[static_cast<size_t>(r)].get());
      } catch (const std::exception& e) {
        rank_errors[static_cast<size_t>(r)] = std::string("exception: ") + e.what();
      }
    });
  for (auto& t : workers) t.join();
  for (auto& s : race_scratch) cudaFreeHost(s);

  int rc = 0;
  for (int r = 0; r < c.world; ++r) {
    if (!rank_errors[static_cast<size_t>(r)].empty()) {
      DGPP_LOG_ERROR("rank {}: {}", r, rank_errors[static_cast<size_t>(r)]);
      rc = 1;
    }
  }
  if (rc == 0) DGPP_LOG_INFO("REPRO: PASS (sequence completed)");
  else DGPP_LOG_INFO("REPRO: FAIL (sequence did not complete)");
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--world") c.world = std::atoi(next().c_str());
    else if (a == "--port") c.port = static_cast<uint16_t>(std::atoi(next().c_str()));
    else if (a == "--staged-iters") c.staged_iters = std::atoi(next().c_str());
    else if (a == "--staged-elems") c.staged_elems = std::atoi(next().c_str());
    else if (a == "--plain-iters") c.plain_iters = std::atoi(next().c_str());
    else if (a == "--plain-elems") c.plain_elems = std::atoi(next().c_str());
    else if (a == "--timeout-ms") c.timeout_ms = std::atoi(next().c_str());
    else if (a == "--rendezvous-timeout-ms")
      c.rendezvous_ms = std::atoi(next().c_str());
    else if (a == "--pick-race") c.pick_race_iters = std::atoi(next().c_str());
    else if (a == "--staged-per-pick") c.staged_per_pick = std::atoi(next().c_str());
    else if (a == "--skew-us") c.skew_us = std::atoi(next().c_str());
    else if (a == "--seed") c.seed = static_cast<uint32_t>(std::atoi(next().c_str()));
    else if (a == "--lat-slots") c.lat_slots = std::atoi(next().c_str());
    else if (a == "--rank") c.my_rank = std::atoi(next().c_str());
    else if (a == "--host") c.host = next();
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }
  if (c.pick_race_iters > 0)
    std::fprintf(stderr,
                 "bus_small_repro: pick-race world=%d iters=%d "
                 "staged/pick=%d skew<=%dus seed=%u lat_slots=%d %s\n",
                 c.world, c.pick_race_iters, c.staged_per_pick, c.skew_us,
                 c.seed, c.lat_slots,
                 c.my_rank >= 0 ? "fabric-mode" : "loopback");
  else
    std::fprintf(stderr,
                  "bus_small_repro: world=%d staged=%dx%zu plain=%dx%zu\n",
                  c.world, c.staged_iters, c.staged_elems, c.plain_iters,
                  c.plain_elems);
  try {
    return run(c);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("repro harness: {}", e.what());
    return 2;
  }
}
