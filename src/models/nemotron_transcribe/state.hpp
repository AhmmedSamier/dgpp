#pragma once
// Mamba-2 state pool for the Nemotron-Transcribe serving path (milestone 3
// of docs/nemotron_transcribe_plan.md): the per-(request, mamba-slot)
// recurrent state, FP32 [heads, head_dim, state_size], and the BF16 conv
// history [conv_dim, kernel - 1] (column 0 the oldest committed token).
// One contiguous run per request (the KdaStatePool precedent): the whole
// request's state is one region, so snapshot/resume — branch sharing at
// admission and rollback at reject — is a single device-to-device copy,
// and slot offsets never reach into kernel internals.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "engine/cache_planes.hpp"

namespace dgpp {

struct NemotronMambaGeometry {
  int num_mamba_slots = 0;  // entries in the walk with a Mamba mixer
  int heads = 0, head_dim = 0, groups = 0, state_size = 0;
  int conv_dim = 0, conv_kernel = 0;

  int64_t recurrent_bytes() const;  // heads*head_dim*state_size*4, padded
  int64_t conv_bytes() const;       // conv_dim*(conv_kernel-1)*2, padded
  int64_t slot_bytes() const;       // recurrent + conv
  int64_t request_bytes() const;    // slot_bytes * num_mamba_slots
};

class NemotronMambaStatePool {
 public:
  NemotronMambaStatePool() = default;
  NemotronMambaStatePool(const NemotronMambaStatePool&) = delete;
  NemotronMambaStatePool& operator=(const NemotronMambaStatePool&) = delete;

  // Allocates num_requests * request_bytes from the arena's persistent
  // hot region and zeroes it.
  void init(Arena& arena, const NemotronMambaGeometry& g, int num_requests);

  int num_requests() const { return num_requests_; }
  const NemotronMambaGeometry& geometry() const { return g_; }

  float* recurrent(int request, int slot);
  const float* recurrent(int request, int slot) const;
  uint16_t* conv(int request, int slot);
  const uint16_t* conv(int request, int slot) const;

  // Cold start / re-prefill.
  void zero_request(int request, cudaStream_t stream);
  void zero_all(cudaStream_t stream);

  // Resume/branch: dst becomes a byte copy of src (snapshot semantics).
  void copy_request(int dst_request, int src_request, cudaStream_t stream);

  // Paged block table stub interface (prefix arena / NVMe tier). The
  // Nemotron-Transcribe pool is a recurrent state pool, not a paged KV
  // cache; these are no-ops until the family wires prefix caching.
  const int32_t* request_table_row(int /*req*/) const { return nullptr; }
  void pin_blocks(const int32_t* /*blocks*/, int /*n*/) {}
  int32_t acquire_pinned_block() { return -1; }
  void unpin_blocks(const int32_t* /*blocks*/, int /*n*/) {}
  void copy_block_contents(int32_t /*dst*/, int32_t /*src*/,
                           cudaStream_t /*stream*/) {}
  std::vector<CachePlane> planes() const { return {}; }
  bool ensure_request_blocks(int /*req*/, int64_t /*tokens*/,
                             cudaStream_t /*stream*/) { return true; }
  uint64_t block_identity(int32_t /*block*/) const { return 0; }
  int64_t total_blocks() const { return 0; }
  int64_t blocks_in_use() const { return 0; }
  int64_t block_count_for_tokens(int64_t /*tokens*/) const { return 0; }
  bool share_blocks_into(int /*req*/, const int32_t* /*blocks*/, int /*n*/,
                         cudaStream_t /*stream*/) { return false; }

 private:
  uint8_t* request_base(int request) const;

  NemotronMambaGeometry g_{};
  int num_requests_ = 0;
  uint8_t* base_ = nullptr;
};

}  // namespace dgpp
