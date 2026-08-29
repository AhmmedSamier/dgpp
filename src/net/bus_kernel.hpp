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

#include <cuda_runtime.h>

#include "kernels/flag_protocol_types.hpp"

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

}  // namespace dgpp::net
