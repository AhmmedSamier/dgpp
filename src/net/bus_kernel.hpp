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
// oracle of the same chain matches exactly. src may be device memory or a
// pinned pre-stage buffer the producing GEMM wrote (the §6.3 seam); when
// src aliases dst, the fold runs in place per element (one thread per
// element, read before write) after snapshotting every peer's send copy.
cudaError_t launch_bus_allreduce(const BusAllReduceView& v, int my_rank,
                                  const __nv_bfloat16* src, __nv_bfloat16* dst,
                                  uint32_t elems, uint32_t ctl_seq,
                                  BusAllReduceCtl* ctl,
                                  uint64_t deadline_cycles,
                                  cudaStream_t stream);

// ---- segment-quantized reduce-scatter/allgather (the prefill class) ------
//
// DESIGN §6.3: large boundaries flow as reduce-scatter + allgather over
// the bulk pool (one-shot's (W-1)x replication is wire-optimal only at
// W=2; RS+AG pays 2(W-1)/W). The buffer is striped on the bulk-slot grid
// (stripe = one bulk slot); shards are contiguous stripe ranges per rank
// (ceil split); the machine processes SEGMENTS of at most
// bulk_slots x lanes stripes so one kernel launch and one flight step
// cover a bounded staging/posting/claiming wave within the pool depths.
//
// Both phases share one sub-range table — each rank's owned stripes of
// the current segment: RS stages every peer's sub-range TO that peer
// (from my src) and folds MY sub-range from all peers' arrivals;
// AG broadcasts MY (now reduced) sub-range to every peer and copies each
// peer's arrivals into dst. The RS fold is the canonical ascending-rank
// fp32 chain per element — bitwise identical to the latency one-shot
// chain, so both paths against one oracle must agree exactly.
//
// Arrival identification (no engine coupling): the sender posts a
// sub-range's stripes in order, striped round-robin over lanes (stripe k
// on lane k%lanes, ring-FIFO per lane); RC delivers in order per lane, so
// the j-th arrival on a lane sits in ring slot j%depth with
// seq = j/depth + 1 — the kernel derives j from the doorbell cell itself
// and maps it to stripe k = i*lanes + lane (i = the in-segment per-lane
// index, from the per-(peer,lane) base the kernel computes by summing
// its OWN ack cells at launch; its acks are the only writer).
//
// Two invariants the mapping forces, both measured into existence:
//   * SEGMENT WINDOW: the rings are flight- and phase-agnostic FIFOs and
//     ranks do not progress in lockstep (shard geometry sees to that), so
//     a receiver's active kernel WILL see future-segment/future-phase
//     doorbells in its cells. It claims only its own window —
//     [base, base + the lane's stripe share) — and leaves the rest
//     unconsumed for the kernel whose window contains them; every post
//     maps to exactly one window of the receiving flight, in ring order,
//     at any skew. Eager claiming acks foreign doorbells away and
//     deadlocks both phases on the loss.
//   * DEFERRED ACKS: the ack certifies the payload CONSUMED — the RS fold
//     consumes at kernel exit (s_ptr reads the slots one last time
//     there), so the ack flushes after the fold. An ack at claim time
//     returns the sender's credit early and a sender a segment ahead
//     overwrites a fold input. The flush re-reads the door cell (stable
//     until its credit returns) and skips the kernel's own claims in the
//     scan meanwhile — a re-presented claim would steal the shared CAS
//     from a fresh doorbell in the same warp, forever.
constexpr int kBusMaxBulkSegStripes = 64;   // >= bulk_slots * kBusMaxLanes
constexpr int kBusMaxPeersSized = 8;        // >= kBusMaxPeers
// Flat bulk receive cells the collective kernel can track (the claimed
// bitmap behind the deferred-ack flush): peers x lanes x bulk_slots must
// fit, or claimed cells stop acking (their credits never return) — the
// bus validates this at start, config-error class.
constexpr int kBusMaxBulkCells = 96;

// Per-segment launch plan; `send_payload[p]` in the view is peer p's
// arena row (segment stripes, stripe k at row + k*bulk_slot_bytes).
struct BusBulkSegPlan {
  uint32_t total_elems = 0;      // whole collective (tails the last stripe)
  uint32_t stripe_elems = 0;     // bulk_slot_bytes / 2
  uint32_t seg_first = 0;       // first stripe of the segment (global grid)
  uint32_t seg_stripe_count = 0;  // stripes in this segment
  uint32_t my_base = 0;         // my sub-range, segment-local stripe index
  uint32_t my_count = 0;
  uint32_t out_base[kBusMaxPeersSized] = {};   // peer p's sub-range
  uint32_t out_count[kBusMaxPeersSized] = {};
};

// phase 0 = reduce-scatter (fold my sub-range from all peers' arrivals),
// phase 1 = allgather (copy each peer's sub-range arrivals into dst).
cudaError_t launch_bus_bulk_collective(const BusAllReduceView& v, int my_rank,
                                       int phase, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst,
                                       const BusBulkSegPlan& plan,
                                       uint64_t* staged_counters,
                                       uint32_t ctl_seq, BusAllReduceCtl* ctl,
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
