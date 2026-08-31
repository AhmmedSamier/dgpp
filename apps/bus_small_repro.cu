// The small-collective stall repro (M6 d3 bring-up, 2026-08-31) — the hunt's
// instrument, deliberately OUTSIDE the CI gate until the fix lands (a
// committed red test buys nothing; a runnable repro command buys the hunt).
//
// BUG: the first SMALL plain latency allreduce after a run of STAGED
// collectives stalls every rank — the receiver's doorbell cell is visibly
// ahead of its ack (the engine's STALLED dump shows door/ack divergence)
// yet the per-collective kernel never claims it. Sizes measured:
//   16 elems after 14x2048 staged  = stall, every run (the greedy loop)
//   16 elems standalone, fresh bus = pass
//   4 elems after a 2048 plain     = pass
// Family: the gen-1825 RNR-freeze wedge (unreproduced, fabric). Suspects:
// unsignaled-payload WR retirement at ring wrap, credit/RQ re-arm lag.
//
// This reproduces the model's exact collective sequence without the model:
// K staged boundaries (stage_next -> host write [the GEMM's stand-in] ->
// allreduce_staged -> wait), then N plain small collectives. Variants via
// flags bisect the trigger (staged count, sizes, interleavings).
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

BusOptions loop_options(int rank, int world, uint16_t port,
                        int timeout_ms) {
  BusOptions o;
  o.world_size = world;
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
  o.completion_timeout_ms = timeout_ms;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

struct Config {
  int world = 4;
  uint16_t port = 29930;
  int staged_iters = 14;
  size_t staged_elems = 2048;
  int plain_iters = 1;
  size_t plain_elems = 16;
  int timeout_ms = 8000;
};

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

  for (int i = 0; i < c.plain_iters; ++i) {
    uint16_t* scratch = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&scratch, c.plain_elems * 2));
    std::memset(scratch, 0, c.plain_elems * 2);
    scratch[0] = static_cast<uint16_t>(rank);
    std::string err;
    const uint64_t id = bus->allreduce(scratch, scratch, c.plain_elems, &err);
    if (id == 0) {
      cudaFree(scratch);
      return "plain submit rejected: " + err;
    }
    const auto res = bus->wait_allreduce(id, c.timeout_ms);
    cudaFree(scratch);
    if (!res.ok) return "plain small wait failed: " + res.error;
  }
  DGPP_LOG_INFO("rank {}: {} plain small collectives done", rank,
               c.plain_iters);
  return "";
}

int run(const Config& c) {
  DGPP_CUDA_OK(cudaStreamCreate(&g_fill_stream));
  std::vector<std::unique_ptr<CollectiveBus>> buses;
  for (int r = 0; r < c.world; ++r)
    buses.push_back(std::make_unique<CollectiveBus>(
        loop_options(r, c.world, c.port, c.timeout_ms)));
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
        rank_errors[static_cast<size_t>(r)] = rank_run(c, r, buses[static_cast<size_t>(r)].get());
      } catch (const std::exception& e) {
        rank_errors[static_cast<size_t>(r)] = std::string("exception: ") + e.what();
      }
    });
  for (auto& t : workers) t.join();

  int rc = 0;
  for (int r = 0; r < c.world; ++r) {
    if (!rank_errors[static_cast<size_t>(r)].empty()) {
      DGPP_LOG_ERROR("rank {}: {}", r, rank_errors[static_cast<size_t>(r)]);
      rc = 1;
    }
  }
  if (rc == 0) DGPP_LOG_INFO("REPRO: PASS (sequence completed)");
  else DGPP_LOG_INFO("REPRO: STALL (bug reproduced)");
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
    else {
      std::fprintf(stderr, "unknown flag %s\n", a.c_str());
      return 2;
    }
  }
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
