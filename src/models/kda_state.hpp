#pragma once
// Per-rank/request KDA state arena (M2 deliverable 2, DESIGN §7.1/§8).
//
// Layout: pool -> [slot][layer][recurrent FP32 [H,V,K]][conv BF16 [C,W]].
// Slot memory is one contiguous run, so a state snapshot (header + slot
// bytes) is a single copy and import/export never touches layer internals.
// The speculative conv suffix is allocated and zeroed but otherwise owned by
// M8; V1 snapshots copy the whole fixed-width slot (DESIGN §8).
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "models/kda_geometry.hpp"
#include "models/kda_snapshot.hpp"

namespace dgpp {

class KdaStatePool {
 public:
  KdaStatePool() = default;
  KdaStatePool(const KdaStatePool&) = delete;
  KdaStatePool& operator=(const KdaStatePool&) = delete;

  // Allocates num_slots * slot_bytes from the arena's persistent region.
  void init(Arena& arena, const KdaConfig& cfg, int num_slots);

  int num_slots() const { return num_slots_; }
  const KdaConfig& config() const { return cfg_; }
  const KdaGeometry& geometry() const { return geo_; }

  // [local_heads, V, K] FP32, K contiguous (reference h0 layout).
  float* recurrent(int slot, int layer);
  const float* recurrent(int slot, int layer) const;

  // [conv_channels, conv_state_width] BF16, width contiguous; committed
  // history in columns [0, conv_hist), speculative reserve after that.
  uint16_t* conv(int slot, int layer);
  const uint16_t* conv(int slot, int layer) const;

  // Zeroes one slot's recurrent states and conv history (including the
  // speculative suffix) — the cold-start and re-prefill path.
  void zero_slot(int slot, cudaStream_t stream);
  void zero_all(cudaStream_t stream);

  // ---- snapshots (DESIGN §8: complete snapshots only) ---------------------
  // Header the snapshot at `slot` was exported with; callers persist header
  // + payload together.
  KdaSnapshotHeader snapshot_header(std::string_view model_revision,
                                    std::string_view numerics_mode) const {
    return KdaSnapshotHeader::make(cfg_, model_revision, numerics_mode);
  }

  // snapshot payload size == geometry().slot_bytes; the full file is
  // sizeof(KdaSnapshotHeader) + slot_bytes.
  size_t snapshot_bytes() const { return geo_.slot_bytes; }

  // Async device->host copy of the whole slot into `dst` (caller-owned,
  // >= snapshot_bytes()). `dst` should be pinned for bandwidth but any
  // host-readable memory works.
  void export_snapshot(int slot, void* dst, size_t dst_bytes,
                       cudaStream_t stream) const;

  // Async host->device copy of a validated snapshot payload into `slot`.
  // `header` must already have passed validate_against(config()).
  void import_snapshot(int slot, const KdaSnapshotHeader& header,
                       const void* src, size_t src_bytes, cudaStream_t stream);

 private:
  uint8_t* slot_base(int slot) const;

  KdaConfig cfg_{};
  KdaGeometry geo_{};
  int num_slots_ = 0;
  uint8_t* base_ = nullptr;
};

}  // namespace dgpp
