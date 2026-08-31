#pragma once

#include <cstdint>
#include <limits>

namespace dgpp {

// Reserved control value for an orderly kernel shutdown. Data-plane
// sequences are 1..UINT32_MAX-1 and wrap back to 1.
constexpr uint32_t kFlagStopSequence = std::numeric_limits<uint32_t>::max();

// Start slot: the producer-to-device doorbell. Own a cache line so unrelated
// writes cannot false-share it. Sequences start at 1; 0 means idle. `len` is
// the payload length in bytes when the bus uses the slot as a receive
// doorbell (0 for pure-flag uses); the flag kernels deliberately ignore it,
// so their pinned contract is unchanged. `ctl` is the COLLECTIVE sequence a
// doorbell belongs to (0 = harness/flag traffic): a per-collective kernel
// must never claim a doorbell of another collective — ranks run unbarriered
// between collectives, so a rank finished with collective N can post N+1's
// pair while a peer's kernel N is still scanning; a generation-blind claim
// eats the early doorbell (folding N+1's payload into N — corruption) and
// starves N+1's own kernel (the stall; at fabric skew this is the gen-1825
// wedge). The gate: door->ctl == this kernel's collective sequence.
struct alignas(64) StartSlot {
  uint32_t seq = 0;
  uint32_t len = 0;
  uint32_t ctl = 0;
  uint32_t pad[13];
};

static_assert(sizeof(StartSlot) == 64, "StartSlot must occupy one cache line");

// Ack slot: system-scope release sequence plus payload-derived result fields.
struct alignas(64) FlagAck {
  uint32_t seq = 0;
  uint64_t cycles = 0;
  uint64_t hash = 0;
  uint32_t pad[10];
};

static_assert(sizeof(FlagAck) == 64, "FlagAck must occupy one cache line");

}  // namespace dgpp
