#include "models/nemotron_transcribe/state.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {
constexpr size_t kRegionAlign = 256;  // the house persistent-region align
size_t pad256(int64_t bytes) {
  return (static_cast<size_t>(bytes) + kRegionAlign - 1) / kRegionAlign * kRegionAlign;
}
}  // namespace

int64_t NemotronMambaGeometry::recurrent_bytes() const {
  return static_cast<int64_t>(pad256(static_cast<int64_t>(heads) * head_dim * state_size * 4));
}
int64_t NemotronMambaGeometry::conv_bytes() const {
  return static_cast<int64_t>(
      pad256(static_cast<int64_t>(conv_dim) * (conv_kernel - 1) * 2));
}
int64_t NemotronMambaGeometry::slot_bytes() const { return recurrent_bytes() + conv_bytes(); }
int64_t NemotronMambaGeometry::request_bytes() const {
  return slot_bytes() * num_mamba_slots;
}

void NemotronMambaStatePool::init(Arena& arena, const NemotronMambaGeometry& g, int num_requests) {
  if (base_) throw std::logic_error("NemotronMambaStatePool: already initialized");
  if (num_requests <= 0 || g.num_mamba_slots <= 0 || g.heads <= 0 || g.head_dim <= 0 ||
      g.groups <= 0 || g.heads % g.groups != 0 || g.state_size <= 0 || g.conv_dim <= 0 ||
      g.conv_kernel < 2)
    throw std::invalid_argument("NemotronMambaStatePool: geometry");
  g_ = g;
  num_requests_ = num_requests;
  const size_t bytes = static_cast<size_t>(g_.request_bytes()) * num_requests;
  base_ = static_cast<uint8_t*>(arena.alloc_persistent(MemClass::DeviceHot, bytes, kRegionAlign));
  DGPP_CUDA_OK(cudaMemset(base_, 0, bytes));
}

uint8_t* NemotronMambaStatePool::request_base(int request) const {
  if (request < 0 || request >= num_requests_)
    throw std::out_of_range("NemotronMambaStatePool: request");
  return base_ + static_cast<size_t>(request) * g_.request_bytes();
}

float* NemotronMambaStatePool::recurrent(int request, int slot) {
  if (slot < 0 || slot >= g_.num_mamba_slots)
    throw std::out_of_range("NemotronMambaStatePool: slot");
  return reinterpret_cast<float*>(request_base(request) +
                                  static_cast<size_t>(slot) * g_.slot_bytes());
}
const float* NemotronMambaStatePool::recurrent(int request, int slot) const {
  return const_cast<NemotronMambaStatePool*>(this)->recurrent(request, slot);
}
uint16_t* NemotronMambaStatePool::conv(int request, int slot) {
  if (slot < 0 || slot >= g_.num_mamba_slots)
    throw std::out_of_range("NemotronMambaStatePool: slot");
  return reinterpret_cast<uint16_t*>(request_base(request) +
                                     static_cast<size_t>(slot) * g_.slot_bytes() +
                                     g_.recurrent_bytes());
}
const uint16_t* NemotronMambaStatePool::conv(int request, int slot) const {
  return const_cast<NemotronMambaStatePool*>(this)->conv(request, slot);
}

void NemotronMambaStatePool::zero_request(int request, cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemsetAsync(request_base(request), 0,
                               static_cast<size_t>(g_.request_bytes()), stream));
}
void NemotronMambaStatePool::zero_all(cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemsetAsync(base_, 0,
                               static_cast<size_t>(g_.request_bytes()) * num_requests_, stream));
}
void NemotronMambaStatePool::copy_request(int dst, int src, cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemcpyAsync(request_base(dst), request_base(src),
                               static_cast<size_t>(g_.request_bytes()),
                               cudaMemcpyDeviceToDevice, stream));
}

}  // namespace dgpp
