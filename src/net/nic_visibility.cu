#include "net/nic_visibility.hpp"

#include "kernels/flag_protocol.cuh"

namespace dgpp::net {

cudaError_t launch_nic_visibility_kernel(uint32_t* start,
                                         const uint32_t* payload,
                                         FlagAck* ack,
                                         uint64_t deadline_cycles,
                                         cudaStream_t stream) {
  flag_payload_kernel<<<1, 1, 0, stream>>>(start, payload, ack,
                                           deadline_cycles);
  return cudaGetLastError();
}

}  // namespace dgpp::net
