#pragma once

#include <cstdint>
#include <limits>

namespace dgpp {

// Reserved control value for an orderly kernel shutdown. Data-plane
// sequences are 1..UINT32_MAX-1 and wrap back to 1.
constexpr uint32_t kFlagStopSequence = std::numeric_limits<uint32_t>::max();

// Start slot: the producer-to-device doorbell. Own a cache line so unrelated
// writes cannot false-share it. Sequences start at 1; 0 means idle.
struct alignas(64) StartSlot {
  uint32_t seq = 0;
  uint32_t pad[15];
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
