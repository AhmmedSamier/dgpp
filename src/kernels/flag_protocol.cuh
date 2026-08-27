// Pinned-memory flag protocol: the completion-notification contract for
// CollectiveBus on GB10's unified-memory fabric.
//
// Contract (validated by tests/cuda/flag_protocol_test.cpp):
//   * Host writes payload first, then bumps `start->seq` with a RELEASE
//     store. Device observes the new seq, consumes payload, stamps cycles
//     and payload hash, and only after __threadfence_system() publishes
//     `ack->seq`. Host observes acks with ACQUIRE loads.
//   * Consequence: if a host thread sees ack.seq == S, then the payload for
//     S was fully visible to the device when it ran — the ordering invariant
//     the bus receive path relies on.
//   * The persistent kernels are watchdog-bounded: a lost wakeup exits the
//     kernel instead of wedging the device. Callers detect this as "ack
//     never arrives" + subsequent stream sync completing.
//
// Header-only by design: included from exactly one TU each (bench, test,
// later the bus itself). Keep the kernels single-thread/single-block —
// the protocol must stay minimal; bulk data movement is the caller's job.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace dgpp {

// Start slot: the host->device doorbell. Own cache line so unrelated host
// writes cannot false-share it. Sequences start at 1; 0 means "idle".
struct alignas(64) StartSlot {
  volatile uint32_t seq = 0;
  uint32_t pad[15];
};

static_assert(sizeof(StartSlot) == 64, "StartSlot must occupy one cache line");

// Ack slot: seq (protocol), cycles (device timestamp at observation), hash
// (payload integrity for the payload variant). Padded to a full cache line
// so unrelated pinned data sharing the line cannot false-share the seq.
struct alignas(64) FlagAck {
  volatile uint32_t seq = 0;
  volatile uint64_t cycles = 0;
  volatile uint64_t hash = 0;
  uint32_t pad[10];
};

static_assert(sizeof(FlagAck) == 64, "FlagAck must occupy one cache line");

__device__ inline void flag_poll_pause() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  __nanosleep(32);
#endif
}

// Persistent heartbeat: for every new start seq S, acks with S and a device
// cycle stamp. Exits silently once `deadline_cycles` elapse without traffic
// (watchdog against lost wakeups). Sequences start at 1; 0 means "idle".
__global__ inline void flag_heartbeat_kernel(volatile uint32_t* start,
                                             FlagAck* ack,
                                             uint64_t deadline_cycles) {
  const uint64_t t0 = clock64();
  uint32_t last_seen = 0;
  for (;;) {
    uint32_t s = *start;
    while (s == last_seen) {
      if (clock64() - t0 > deadline_cycles) return;
      flag_poll_pause();
      s = *start;
    }
    last_seen = s;
    ack->cycles = clock64();
    ack->hash = 0;
    __threadfence_system();  // ack must never be observable before stamp
    ack->seq = s;
  }
}

// Persistent payload variant: on new seq, folds 16 u32 words (64 B, one
// cache line) into a checksum before acking — the receive-path shape where
// payload visibility must be proven before completion is signalled.
__global__ inline void flag_payload_kernel(volatile uint32_t* start,
                                           const uint32_t* payload,
                                           FlagAck* ack,
                                           uint64_t deadline_cycles) {
  const uint64_t t0 = clock64();
  uint32_t last_seen = 0;
  for (;;) {
    uint32_t s = *start;
    while (s == last_seen) {
      if (clock64() - t0 > deadline_cycles) return;
      flag_poll_pause();
      s = *start;
    }
    last_seen = s;
    uint64_t h = 0;
    // Checksum: xor of (hi<<32 | lo) per u32 pair, mixed with the golden
    // ratio constant. Host side MUST mirror this fold exactly (see test).
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const uint64_t pair = (static_cast<uint64_t>(payload[i]) << 32) |
                            static_cast<uint64_t>(payload[i + 1]);
      h ^= pair * 0x9E3779B97F4A7C15ull;
    }
    ack->cycles = clock64();
    ack->hash = h;
    __threadfence_system();  // payload-derived state precedes ack, always
    ack->seq = s;
  }
}

}  // namespace dgpp
