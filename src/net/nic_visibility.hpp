#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/flag_protocol_types.hpp"

namespace dgpp::net {

// Launches the NIC→GPU visibility kernel (flag_payload_kernel) on the
// caller's stream: the device polls `start` with a system-scope acquire,
// folds the 64 B payload into `ack->hash`, and publishes `ack->seq` with a
// system-scope release. The DESIGN §2.3 hardware contract (payload + doorbell
// RC SENDs consumed directly from NIC-DMA'd pinned memory) is driven through
// this one TU by the transport regression command (apps/nic_regress) and the
// M0 bench (micro_ibv_smoke verify).
cudaError_t launch_nic_visibility_kernel(uint32_t* start,
                                         const uint32_t* payload,
                                         FlagAck* ack,
                                         uint64_t deadline_cycles,
                                         cudaStream_t stream);

}  // namespace dgpp::net
