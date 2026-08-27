#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/flag_protocol_types.hpp"

namespace dgpp::bench {

cudaError_t launch_nic_visibility_kernel(uint32_t* start,
                                         const uint32_t* payload,
                                         FlagAck* ack,
                                         uint64_t deadline_cycles,
                                         cudaStream_t stream);

}  // namespace dgpp::bench
