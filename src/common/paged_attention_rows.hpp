#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace dgpp {

// Row identity is independent of the attention architecture. A mixed walk
// retains the decoder prefix's execution geometry (split attention, fused
// decode kernels) while the remaining rows use the prefill implementation.
struct PagedAttentionRows {
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  bool decode = false;
  bool cache_only = false;
  int decode_prefix = 0;

  template <class Enqueue>
  bool split(const uint16_t* x, int count, int hidden, uint16_t* out, Enqueue enqueue) const {
    if (!decode_prefix) return false;
    if (decode || decode_prefix < 0 || decode_prefix >= count)
      throw std::invalid_argument("attention: invalid mixed decoder prefix");
    auto prefix = *this;
    prefix.decode_prefix = 0;
    prefix.decode = true;
    enqueue(x, decode_prefix, prefix, out);
    auto suffix = *this;
    suffix.decode_prefix = 0;
    suffix.req_ids += decode_prefix;
    suffix.pos += decode_prefix;
    const size_t offset = static_cast<size_t>(decode_prefix) * hidden;
    enqueue(x + offset, count - decode_prefix, suffix, out + offset);
    return true;
  }
};

}  // namespace dgpp
