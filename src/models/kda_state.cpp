#include "models/kda_state.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

void KdaStatePool::init(Arena& arena, const KdaConfig& cfg, int num_slots) {
  if (base_) throw std::logic_error("kda state pool is already initialized");
  if (num_slots <= 0) throw std::invalid_argument("kda state pool: num_slots must be positive");
  KdaConfig::validate_config(cfg);
  cfg_ = cfg;
  geo_ = KdaGeometry::from_config(cfg, num_slots);
  num_slots_ = num_slots;
  // 256-byte alignment keeps the per-layer recurrent blocks (1 MiB at the
  // real geometry) naturally aligned for vectorized kernel access.
  base_ = static_cast<uint8_t*>(
      arena.alloc_persistent(MemClass::DeviceHot, geo_.pool_bytes, 256));
}

uint8_t* KdaStatePool::slot_base(int slot) const {
  if (slot < 0 || slot >= num_slots_)
    throw std::out_of_range("kda state pool: slot " + std::to_string(slot));
  return base_ + static_cast<size_t>(slot) * geo_.slot_bytes;
}

float* KdaStatePool::recurrent(int slot, int layer) {
  uint8_t* slotp = slot_base(slot);
  if (layer < 0 || layer >= cfg_.num_kda_layers)
    throw std::out_of_range("kda state pool: layer " + std::to_string(layer));
  return reinterpret_cast<float*>(slotp +
                                  static_cast<size_t>(layer) *
                                      (geo_.recurrent_bytes +
                                       geo_.conv_slot_bytes));
}

uint16_t* KdaStatePool::conv(int slot, int layer) {
  uint8_t* slotp = slot_base(slot);
  if (layer < 0 || layer >= cfg_.num_kda_layers)
    throw std::out_of_range("kda state pool: layer " + std::to_string(layer));
  return reinterpret_cast<uint16_t*>(
      slotp + static_cast<size_t>(layer) *
                  (geo_.recurrent_bytes + geo_.conv_slot_bytes) +
      geo_.recurrent_bytes);
}

const float* KdaStatePool::recurrent(int slot, int layer) const {
  return const_cast<KdaStatePool*>(this)->recurrent(slot, layer);
}

const uint16_t* KdaStatePool::conv(int slot, int layer) const {
  return const_cast<KdaStatePool*>(this)->conv(slot, layer);
}

void KdaStatePool::zero_slot(int slot, cudaStream_t stream) {
  (void)slot_base(slot);  // bounds check before touching CUDA
  DGPP_CUDA_OK(cudaMemsetAsync(slot_base(slot), 0, geo_.slot_bytes, stream));
}

void KdaStatePool::zero_all(cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemsetAsync(base_, 0, geo_.pool_bytes, stream));
}

void KdaStatePool::export_snapshot(int slot, void* dst, size_t dst_bytes,
                                   cudaStream_t stream) const {
  if (dst_bytes < geo_.slot_bytes)
    throw std::invalid_argument("kda snapshot export: destination too small");
  if (!dst) throw std::invalid_argument("kda snapshot export: null destination");
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, slot_base(slot), geo_.slot_bytes,
                               cudaMemcpyDeviceToHost, stream));
}

void KdaStatePool::import_snapshot(int slot, const KdaSnapshotHeader& header,
                                   const void* src, size_t src_bytes,
                                   cudaStream_t stream) {
  // Validate first: a mismatched snapshot must never partially land in a
  // live slot (DESIGN §1.4 — transactional state).
  header.validate_against(cfg_);
  if (src_bytes < geo_.slot_bytes)
    throw std::invalid_argument("kda snapshot import: source too small");
  if (!src) throw std::invalid_argument("kda snapshot import: null source");
  DGPP_CUDA_OK(cudaMemcpyAsync(slot_base(slot), src, geo_.slot_bytes,
                               cudaMemcpyHostToDevice, stream));
}

}  // namespace dgpp
