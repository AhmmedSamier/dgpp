#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace dgpp {
void mimo_mtp_history_gather(const uint16_t* history, uint16_t* hidden, const int64_t* positions,
                             int64_t* layer_positions, const int32_t* request_ids, int rows,
                             int width, int history_rows, int layer, cudaStream_t stream);
void mimo_mtp_history_store(uint16_t* history, const uint16_t* hidden, const int64_t* positions,
                            const int32_t* request_ids, int rows, int width, int history_rows,
                            int layer, cudaStream_t stream);
}
