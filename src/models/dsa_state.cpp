#include "models/dsa_state.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

const size_t kRegionAlign = 256;

size_t padded(size_t bytes) { return (bytes + kRegionAlign - 1) / kRegionAlign * kRegionAlign; }

}  // namespace

void DsaStatePool::init(Arena& arena, const DsaConfig& cfg, int max_requests,
                        int64_t max_token_slots) {
  if (initialized_) throw std::logic_error("dsa state pool is already initialized");
  DsaConfig::validate_config(cfg);
  if (max_requests <= 0)
    throw std::invalid_argument("dsa state pool: max_requests must be positive");
  if (max_token_slots <= 0)
    throw std::invalid_argument("dsa state pool: max_token_slots must be positive");
  if (max_token_slots % cfg.block_tokens != 0)
    throw std::invalid_argument(
        "dsa state pool: max_token_slots must be a multiple of block_tokens");

  cfg_ = cfg;
  geo_ = DsaGeometry::from_config(cfg);
  max_requests_ = max_requests;
  max_token_slots_ = max_token_slots;
  total_blocks_ = max_token_slots / cfg.block_tokens;
  max_pool_slots_ = total_blocks_ * geo_.pools_per_block;
  // The selection kernels pack pool ids into 21 composite-key bits; a pool
  // id at or above 2^21 would alias into the logit bits.
  if (max_pool_slots_ >= (int64_t(1) << 21))
    throw std::invalid_argument("dsa state pool: pool id space exceeds 2^21");

  const int layers = cfg.num_dsa_layers;
  const int64_t pools = max_pool_slots_;
  // One region per array (layer-major inside each): the caches have no
  // snapshot/copy requirement in M3 (prefix attachment shares by reference,
  // DESIGN §8), so contiguity across regions buys nothing and separate
  // regions keep each 256-byte aligned for the vectorized kernel paths.
  latent_base_ = static_cast<uint8_t*>(
      arena.alloc_persistent(MemClass::DeviceHot,
                             size_t(layers) * size_t(max_token_slots_) *
                                 geo_.latent_bytes_per_token,
                             kRegionAlign));
  index_k_base_ = static_cast<uint8_t*>(
      arena.alloc_persistent(MemClass::DeviceHot,
                             size_t(layers) * size_t(pools) *
                                 geo_.index_k_bytes_per_pool,
                             kRegionAlign));
  index_scale_base_ = static_cast<float*>(arena.alloc_persistent(
      MemClass::DeviceHot, size_t(layers) * size_t(pools) * sizeof(float),
      kRegionAlign));
  tail_base_ = static_cast<uint8_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, size_t(layers) * size_t(max_requests_) *
                               geo_.tail_bytes_per_request,
      kRegionAlign));
  block_tables_ = static_cast<int32_t*>(arena.alloc_persistent(
      MemClass::DeviceHot,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t),
      kRegionAlign));

  tables_host_.assign(size_t(max_requests_) * size_t(total_blocks_), 0);
  held_.assign(size_t(max_requests_), 0);
  free_.reserve(size_t(total_blocks_));
  for (int64_t b = int64_t(total_blocks_) - 1; b >= 0; --b)
    free_.push_back(int32_t(b));  // LIFO: low ids come out first
  // Unheld table entries read as 0 from construction (the same invariant
  // reset_all re-establishes): device consumers that honor `held`/visible
  // bounds never touch them, and any tooling that reads the full row sees
  // deterministic zeros, not arena garbage. Synchronous memset — init is
  // cold-path and stream-less by design.
  DGPP_CUDA_OK(cudaMemset(
      block_tables_, 0,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t)));
  initialized_ = true;
}

int64_t DsaStatePool::blocks_in_use() const {
  return total_blocks_ - int64_t(free_.size());
}

int64_t DsaStatePool::block_count_for_tokens(int64_t tokens) const {
  return (tokens + cfg_.block_tokens - 1) / cfg_.block_tokens;
}

void* DsaStatePool::latent(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return latent_base_ + size_t(layer) * size_t(max_token_slots_) *
                            geo_.latent_bytes_per_token;
}

void* DsaStatePool::index_k(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return index_k_base_ + size_t(layer) * size_t(max_pool_slots_) *
                             geo_.index_k_bytes_per_pool;
}

float* DsaStatePool::index_scale(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return index_scale_base_ + size_t(layer) * size_t(max_pool_slots_);
}

void* DsaStatePool::tail(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return tail_base_ + size_t(layer) * size_t(max_requests_) *
                          geo_.tail_bytes_per_request;
}

const void* DsaStatePool::latent(int layer) const {
  return const_cast<DsaStatePool*>(this)->latent(layer);
}
const void* DsaStatePool::index_k(int layer) const {
  return const_cast<DsaStatePool*>(this)->index_k(layer);
}
const float* DsaStatePool::index_scale(int layer) const {
  return const_cast<DsaStatePool*>(this)->index_scale(layer);
}
const void* DsaStatePool::tail(int layer) const {
  return const_cast<DsaStatePool*>(this)->tail(layer);
}

bool DsaStatePool::ensure_request_blocks(int req, int64_t tokens,
                                         cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  if (tokens < 0)
    throw std::invalid_argument("dsa state pool: negative token count");
  const int64_t needed = block_count_for_tokens(tokens);
  if (needed > total_blocks_) return false;  // beyond pool capacity
  const int64_t have = held_[size_t(req)];
  if (needed <= have) return true;
  const int64_t extra = needed - have;
  if (int64_t(free_.size()) < extra) return false;  // transactional: no partial claims

  int32_t* row = tables_host_.data() + size_t(req) * size_t(total_blocks_);
  for (int64_t i = 0; i < extra; ++i) {
    row[have + i] = free_.back();
    free_.pop_back();
  }
  held_[size_t(req)] = int32_t(needed);
  DGPP_CUDA_OK(cudaMemcpyAsync(block_tables_ + size_t(req) * size_t(total_blocks_) +
                                   size_t(have),
                               row + have, size_t(extra) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
  return true;
}

void DsaStatePool::release_request_blocks(int req, cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  const int64_t held = held_[size_t(req)];
  if (held == 0) return;
  int32_t* row = tables_host_.data() + size_t(req) * size_t(total_blocks_);
  for (int64_t i = held - 1; i >= 0; --i) free_.push_back(row[i]);  // LIFO
  std::fill(row, row + held, 0);
  held_[size_t(req)] = 0;
  DGPP_CUDA_OK(cudaMemcpyAsync(
      block_tables_ + size_t(req) * size_t(total_blocks_), row,
      size_t(held) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

int64_t DsaStatePool::request_blocks(int req) const {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  return held_[size_t(req)];
}

void DsaStatePool::reset_all(cudaStream_t stream) {
  if (!initialized_)
    throw std::logic_error("dsa state pool: not initialized");
  const size_t layers = size_t(cfg_.num_dsa_layers);
  DGPP_CUDA_OK(cudaMemsetAsync(
      latent_base_, 0,
      layers * size_t(max_token_slots_) * geo_.latent_bytes_per_token, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      index_k_base_, 0,
      layers * size_t(max_pool_slots_) * geo_.index_k_bytes_per_pool, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      index_scale_base_, 0, layers * size_t(max_pool_slots_) * sizeof(float),
      stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      tail_base_, 0,
      layers * size_t(max_requests_) * geo_.tail_bytes_per_request, stream));

  // Return every block to the free list and zero the whole table (rows AND
  // the unheld tail of each row, so a stale device read fails against a
  // deterministically-zero table rather than a dangling id).
  std::fill(tables_host_.begin(), tables_host_.end(), 0);
  std::fill(held_.begin(), held_.end(), 0);
  free_.clear();
  free_.reserve(size_t(total_blocks_));
  for (int64_t b = int64_t(total_blocks_) - 1; b >= 0; --b)
    free_.push_back(int32_t(b));
  DGPP_CUDA_OK(cudaMemsetAsync(
      block_tables_, 0,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t),
      stream));
}

size_t DsaStatePool::cache_bytes(const DsaConfig& cfg, int max_requests,
                                 int64_t max_token_slots) {
  if (max_requests <= 0 || max_token_slots <= 0 ||
      max_token_slots % cfg.block_tokens != 0)
    throw std::invalid_argument("dsa state pool: invalid accounting shape");
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int64_t pools =
      max_token_slots / cfg.block_tokens * g.pools_per_block;
  const size_t layers = size_t(cfg.num_dsa_layers);
  return padded(layers * size_t(max_token_slots) * g.latent_bytes_per_token) +
         padded(layers * size_t(pools) * g.index_k_bytes_per_pool) +
         padded(layers * size_t(pools) * sizeof(float)) +
         padded(layers * size_t(max_requests) * g.tail_bytes_per_request) +
         padded(size_t(max_requests) *
                    size_t(max_token_slots / cfg.block_tokens) *
                    sizeof(int32_t));
}

}  // namespace dgpp
