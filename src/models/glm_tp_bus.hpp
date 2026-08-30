#pragma once
// Composition seam (M5 deliverable 3): adapts GlmBoundaryReducer to the
// CollectiveBus one-shot all-reduce. Header-only so the ungated models
// library never links ibverbs — only consumers that already link the bus
// (the loopback test, the fabric app) include it.
//
// Chunking: the bus all-reduce's v1 bound is one latency slot
// (lat_slot_bytes/2 bf16 elements — the 4096-hidden decode unit). A
// multi-row boundary folds row-major chunks of at most that many
// elements, sequentially (single outstanding, the v1 contract; the
// chunked bulk collective replaces the loop for prefill-sized T later).
// Row-major chunking preserves the canonical per-element fold order, so
// every rank's destination stays bitwise identical across ranks.
#include <algorithm>
#include <cstdint>

#include "common/log.hpp"
#include <stdexcept>
#include <string>

#include "models/glm_forward.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

struct GlmBusBoundaryReducer final : GlmBoundaryReducer {
  static constexpr size_t kMaxCollectiveElems = 4096;  // one latency slot

  explicit GlmBusBoundaryReducer(net::CollectiveBus& bus, int timeout_ms = 60000)
      : bus_(bus), timeout_ms_(timeout_ms) {}

  void reduce(uint16_t* partial, int rows, int hidden) override {
    if (hidden <= 0 || hidden > static_cast<int>(kMaxCollectiveElems) ||
        hidden % 2 != 0)
      throw std::invalid_argument(
          "boundary reduce: hidden must be even and fit one latency slot");
    static thread_local int chunk_no = 0;
    // Floor: rows folded per collective (hidden itself when hidden fills
    // the slot — the decode shape, one collective per boundary).
    const int rows_per = static_cast<int>(kMaxCollectiveElems / hidden);
    for (int row0 = 0; row0 < rows; row0 += rows_per) {
      const int n = std::min(rows_per, rows - row0);
      const size_t elems = static_cast<size_t>(n) * hidden;
      std::string err;
      DGPP_LOG_DEBUG("boundary chunk {}: submit (rows {}..{})", chunk_no,
                     row0, row0 + n);
      const uint64_t id =
          bus_.allreduce(partial + static_cast<size_t>(row0) * hidden,
                         partial + static_cast<size_t>(row0) * hidden, elems,
                         &err);
      if (!id)
        throw std::runtime_error("boundary reduce: allreduce rejected: " + err);
      const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
      DGPP_LOG_DEBUG("boundary chunk {}: waited (ok={})", chunk_no, res.ok);
      ++chunk_no;
      if (!res.ok)
        throw std::runtime_error("boundary reduce: " + res.error);
    }
  }

 private:
  net::CollectiveBus& bus_;
  int timeout_ms_ = 60000;
};

}  // namespace dgpp
