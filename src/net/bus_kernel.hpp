#pragma once

// Receive-side consumer for CollectiveBus validation (DESIGN §6/§6.2).
//
// Phase 2's transport harness: one persistent kernel per bus endpoint that
// watches every doorbell cell, folds the arrived payload with the golden-
// ratio mix, and publishes the per-slot acknowledgement. The engine's real
// consumers (M5 deliverable 3) are per-collective kernels with the same
// doorbell/ack contract; the transport is agnostic to who consumes.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/flag_protocol_types.hpp"
#include "net/bus_types.hpp"

namespace dgpp::net {

// Raw pinned pointers into one lane's receive slab, shared between host and
// device. Built by CollectiveBus::recv_view(lane).
struct BusRecvView {
  const StartSlot* doorbell_lat = nullptr;  // [lat_slots]
  const StartSlot* doorbell_bulk = nullptr;  // [bulk_slots]
  const uint64_t* payload_lat = nullptr;    // slot-strided lat_slot_bytes
  const uint64_t* payload_bulk = nullptr;    // slot-strided bulk_slot_bytes
  FlagAck* ack_lat = nullptr;                // [lat_slots]
  FlagAck* ack_bulk = nullptr;               // [bulk_slots]
  const StartSlot* control = nullptr;       // orderly stop cell
  int lat_slots = 0;
  int bulk_slots = 0;
  uint32_t lat_slot_bytes = 0;
  uint32_t bulk_slot_bytes = 0;
};

// Shared-memory cell budget for the consumer's per-slot last-seen table.
constexpr int kBusMaxConsumerCells = 256;

// Launches the persistent consumer on `stream` (1 block x 256 threads).
// The kernel exits on the control stop sequence or after `deadline_cycles`
// without traffic; the caller still syncs the stream.
cudaError_t launch_bus_consumer(const BusRecvView& view,
                                uint64_t deadline_cycles, cudaStream_t stream);

// Watchdog deadline in device cycles for the given inactivity seconds.
uint64_t bus_consumer_deadline_cycles(double seconds);

// Host mirror of the consumer fold (bus_types.hpp's mix); the sender uses
// it to verify the receiver's hash. `bytes` must be a multiple of 8.
uint64_t bus_fold(const void* data, size_t bytes);

// ---- per-collective all-reduce consumer (DESIGN §6.3) ------------------------
//
// One kernel per collective call: stages the source vector into each peer's
// claimed send slot, signals the engine to post, waits for every peer's
// doorbell, folds bf16 with fp32 accumulation (canonical global-rank order
// — every rank's result is bitwise identical), publishes the standard
// per-slot acks, and stamps the control cell. The engine's recycle/credit
// passes treat these messages exactly like harness traffic; the wire
// protocol is unchanged.

// Pinned 64B handoff cell, one per bus. Published fields, in order:
//   ready_bits — bit p set (release) once peer p's send slot is staged;
//   done_seq   — the collective's ctl_seq (release) once the kernel is
//                finished (status 0) or gave up (status 1). The engine
//                may also write it as a poison to hasten a failed
//                kernel's exit; the kernel treats any match as an exit.
struct alignas(64) BusAllReduceCtl {
  uint64_t ready_bits = 0;
  uint64_t done_seq = 0;
  // TEMP instrumentation: kernel phase stamps (clock64), cycles 0 if unset
  uint64_t stamp_stage = 0;
  uint64_t stamp_first_claim = 0;
  uint64_t stamp_reduce_done = 0;
  uint32_t status = 0;  // 0 = reduced, 1 = deadline/poison exit
  uint32_t pad2[5];  // 40 + 4 + 20 = 64
};
static_assert(sizeof(BusAllReduceCtl) == 64,
              "BusAllReduceCtl must occupy one cache line");

// Everything the kernel needs, built by the engine at claim time.
struct BusAllReduceView {
  // Every lane of every peer, peer-major (peer ascending, lanes inner).
  BusRecvView recv[kBusMaxPeers * kBusMaxLanes] = {};
  int recv_views = 0;       // peers * lanes
  int lanes_per_peer = 0;
  // Claimed outbound latency slot per peer (the engine posts it once the
  // ready bit lands). bf16 element storage.
  const uint16_t* send_payload[kBusMaxPeers] = {};
  int send_peers = 0;
};

// `elems` is the bf16 element count (multiple of 2). The kernel folds all
// W vectors into dst with fp32 accumulation in canonical global-rank
// order, so every rank's destination is bitwise identical and a host
// oracle of the same chain matches exactly.
cudaError_t launch_bus_allreduce(const BusAllReduceView& v, int my_rank,
                                 const __nv_bfloat16* src, __nv_bfloat16* dst,
                                 uint32_t elems, uint32_t ctl_seq,
                                 BusAllReduceCtl* ctl,
                                 uint64_t deadline_cycles,
                                 cudaStream_t stream);

// Host-side bf16 helpers matching the device intrinsics' round-to-nearest-
// even. Consumers of the bus (tests, checks, the forward oracle) verify
// against the kernel's exact arithmetic; guessing at the rounding would
// make every comparison tolerance-based for no reason.
inline uint16_t bf16_from_f32_rne(float f) {
  uint32_t x = 0;
  std::memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1;  // ties toward even
  return static_cast<uint16_t>((x + 0x7FFFu + lsb) >> 16);
}

inline float bf16_to_f32(uint16_t h) {
  const uint32_t x = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &x, 4);
  return f;
}

// Benchmark realism (bus_check): inter-collective compute standing in for
// the decode GEMMs. A bare submit/wait loop is ~1% device duty and
// measures GPU wake latency instead of the collective; real decode keeps
// the clocks boosted between collectives. `spins` iterations of dependent
// FLOPs per thread; the result lands in buf[256].
cudaError_t launch_bus_warm_work(cudaStream_t stream, float* buf, int spins);

}  // namespace dgpp::net
