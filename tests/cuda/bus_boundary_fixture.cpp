// Compile the serving adapter with the host compiler: tp_bus.hpp also
// includes sampling/parser code that is not compiled by nvcc.
#include "engine/tp_bus.hpp"

void test_stream_boundary_reduce(dgpp::net::CollectiveBus& bus, uint16_t* data, size_t elems, cudaStream_t stream) {
  dgpp::BusStreamReducer reducer(bus);
  reducer.bind_stream(stream);
  reducer.reduce(data, static_cast<int>(elems / 2), 2);
}
