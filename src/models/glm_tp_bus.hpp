#pragma once
// Composition seam (M5 deliverable 3): adapts GlmBoundaryReducer to the
// CollectiveBus one-shot all-reduce. Header-only so the ungated models
// library never links ibverbs — only consumers that already link the bus
// (the loopback test, the fabric app) include it.
//
// Chunking: the bus all-reduce's v1 bound is one latency slot
// (lat_slot_bytes/2 bf16 elements — the 4096-hidden decode unit). A
// multi-row boundary folds row-major chunks of at most that many
// elements, sequentially (single outstanding, the v1 contract). Row-major
// chunking preserves the canonical per-element fold order, so every
// rank's destination stays bitwise identical across ranks.
//
// Pre-stage seam (§6.3 evolution): boundaries that fit one slot are handed
// the pinned staging buffer at stage() time — the producing GEMM writes
// the transport's send source directly and the collective runs with zero
// staging copies (the kernel only publishes ready). Boundaries above the
// slot stay on the device path (chunked, or the bulk collective class).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/log.hpp"
#include <stdexcept>
#include <string>

#include "models/glm_forward.hpp"
#include "models/glm_sampler.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

struct GlmBusBoundaryReducer final : GlmBoundaryReducer {
  static constexpr size_t kMaxCollectiveElems = 4096;  // one latency slot

  explicit GlmBusBoundaryReducer(net::CollectiveBus& bus, int timeout_ms = 60000)
      : bus_(bus), timeout_ms_(timeout_ms) {}

  uint16_t* stage(int rows, int hidden) override {
    if (hidden <= 0 || rows <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(rows) * static_cast<size_t>(hidden) >
            kMaxCollectiveElems)
      return nullptr;  // prefill-shaped boundary: device path
    std::string err;
    void* p = bus_.stage_next(&err);
    if (p == nullptr)
      throw std::runtime_error("boundary stage: handout rejected: " + err);
    staged_ = static_cast<uint16_t*>(p);
    return staged_;
  }

  void reduce(uint16_t* partial, int rows, int hidden) override {
    if (hidden <= 0 || hidden > static_cast<int>(kMaxCollectiveElems) ||
        hidden % 2 != 0)
      throw std::invalid_argument(
          "boundary reduce: hidden must be even and fit one latency slot");
    if (staged_ != nullptr) {
      if (partial != staged_)
        throw std::runtime_error(
            "boundary reduce: a staged handout is held (consume it first)");
      // The pre-staged submit consumes the handout; the fold runs in
      // place in the pinned buffer.
      std::string err;
      const size_t elems =
          static_cast<size_t>(rows) * static_cast<size_t>(hidden);
      const uint64_t id = bus_.allreduce_staged(elems, &err);
      if (id == 0)
        throw std::runtime_error("boundary reduce: staged submit rejected: " +
                                 err);
      staged_ = nullptr;
      wait_collective(id, "boundary staged reduce");
      return;
    }
    static thread_local int chunk_no = 0;
    const size_t total =
        static_cast<size_t>(rows) * static_cast<size_t>(hidden);
    // Prefill-class boundaries (well above a couple of latency chunks)
    // take the bulk machine: segment-quantized reduce-scatter + allgather,
    // the same canonical per-element chain (bitwise-equal to chunking —
    // the bus_test cross-path gate pins exactly that).
    if (total > 2 * kMaxCollectiveElems) {
      std::string err;
      const uint64_t id = bus_.allreduce_bulk(partial, partial, total, &err);
      if (id == 0)
        throw std::runtime_error("boundary reduce: bulk rejected: " + err);
      wait_collective(id, "boundary bulk");
      return;
    }
    // Floor: rows folded per collective (hidden itself when hidden fills
    // the slot — the decode shape, one collective per boundary).
    const int rows_per = static_cast<int>(kMaxCollectiveElems / hidden);
    for (int row0 = 0; row0 < rows; row0 += rows_per) {
      const int n = std::min(rows_per, rows - row0);
      const size_t elems = static_cast<size_t>(n) * hidden;
      const uint64_t id = submit_collective(
          partial + static_cast<size_t>(row0) * hidden, elems);
      DGPP_LOG_DEBUG("boundary chunk {}: submit (rows {}..{})", chunk_no,
                     row0, row0 + n);
      wait_collective(id, "boundary chunk " + std::to_string(chunk_no));
      ++chunk_no;
    }
  }

 private:
  uint64_t submit_collective(uint16_t* at, size_t elems) {
    std::string err;
    const uint64_t id = bus_.allreduce(at, at, elems, &err);
    if (!id)
      throw std::runtime_error("boundary reduce: allreduce rejected: " + err);
    return id;
  }

  void wait_collective(uint64_t id, const std::string& what) {
    const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
    if (!res.ok)
      throw std::runtime_error("boundary reduce (" + what + "): " + res.error);
  }

  net::CollectiveBus& bus_;
  int timeout_ms_ = 60000;
  uint16_t* staged_ = nullptr;  // held pre-stage handout, if any
};

// ---------------------------------------------------------------------------
// M6 d3: the distributed greedy pick over the existing all-reduce.
// ---------------------------------------------------------------------------

// Exact greedy argmax across ranks with NO new transport surface. The
// bus all-reduce is an elementwise SUM in canonical rank order — and a
// sum over disjoint per-rank slots IS a gather: each rank's quadruple
// rides its own slots, the fold accumulates zeros elsewhere (bitwise-
// stable; bf16 values and small digits pass through exactly), and every
// rank then decodes an IDENTICAL candidate table. The pick is
// glm_sample's canonical (value desc, id asc) — rank-consistent by
// construction, the same discipline the unit tests pin.
//
// Encoding: ids travel as three 6-bit digits (bf16 holds integers only
// 0..256 exactly; vocab ids run past 150k); values travel as their own
// bf16 (they were bf16 logits to begin with). Wire shape: one latency
// collective for the candidate gather, one for the winner broadcast —
// both padded to the boundary folds' 2048 elements (see the OPEN ENGINE
// BUG note inside). A proper (value, id) arg-max all-reduce is the M9
// optimization; this is exact and rides the proven collective contract
// (single outstanding, no other latency traffic in flight — the pick is
// serialized behind the forward's collectives in every consumer of it).
//
// `scratch` is a device (managed) buffer of >= 4*world bf16 elements,
// host-writable — caller-owned so this helper allocates nothing inside
// the decode loop.
inline int32_t bus_greedy_pick(net::CollectiveBus& bus, int rank, int world,
                              glm_sample::Candidate local, uint16_t* scratch,
                              int timeout_ms) {
  if (local.id < 0 || local.id >= (1 << 18)) {
    throw std::invalid_argument("bus_greedy_pick: token id outside the "
                                 "6-bit-triplet encoding range");
  }
  // OPEN ENGINE BUG (2026-08-31, glm_tp_greedy_gen_loopback bring-up):
  // the FIRST SMALL plain latency allreduce after a run of STAGED
  // collectives stalls every rank — the doorbell is visibly present in
  // the receiver's slot (doorbell seq ahead of ack) yet the per-
  // collective kernel never claims it, and a send WR from an earlier
  // ring-wrapped collective sits in_flight forever. Sizes: 16 elems
  // after 14x2048-elem staged = stall (every run); 16 standalone on a
  // fresh bus = pass; 4 elems after a 2048 plain = pass. Suspects: the
  // unsignaled-payload WR retirement bookkeeping at ring wrap and the
  // recycle/credit re-arm race — the same family as the gen-1825 RNR-
  // freeze wedge. This is exactly the decode-path collective size
  // class, so the fast paths need the hunt; until then every pick
  // collective rides at the boundary folds' known-good 2048 elements
  // (zeros elsewhere, quadruples unchanged — one latency slot).
  const size_t gather_elems = 2048;
  const auto allreduce_wait = [&](std::string* err) -> uint64_t {
    const uint64_t id = bus.allreduce(scratch, scratch, gather_elems, err);
    if (id == 0) return 0;
    const net::BusAllReduceResult r = bus.wait_allreduce(id, timeout_ms);
    if (!r.ok) {
      *err = r.error;
      return 0;
    }
    return id;
  };

  // ---- gather: every rank's (value, id) in its own slots -------------
  std::memset(scratch, 0, gather_elems * 2);
  scratch[static_cast<size_t>(rank) * 4 + 0] =
      float_to_bf16_bits(local.logit);
  scratch[static_cast<size_t>(rank) * 4 + 1] =
      static_cast<uint16_t>(local.id & 63);
  scratch[static_cast<size_t>(rank) * 4 + 2] =
      static_cast<uint16_t>((local.id >> 6) & 63);
  scratch[static_cast<size_t>(rank) * 4 + 3] =
      static_cast<uint16_t>((local.id >> 12) & 63);
  std::string err;
  if (allreduce_wait(&err) == 0)
    throw std::runtime_error("bus_greedy_pick gather: " + err);
  std::vector<glm_sample::Candidate> cands;
  cands.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    const uint16_t* q = scratch + static_cast<size_t>(r) * 4;
    glm_sample::Candidate c;
    c.logit = bf16_bits_to_float(q[0]);
    c.id = static_cast<int32_t>(q[1]) | (static_cast<int32_t>(q[2]) << 6) |
           (static_cast<int32_t>(q[3]) << 12);
    cands.push_back(c);
  }
  const int32_t winner = glm_sample::merge_greedy(cands);

  // ---- broadcast: rank 0's winner digits reach every rank -----------
  std::memset(scratch, 0, gather_elems * 2);
  if (rank == 0) {
    scratch[0] = static_cast<uint16_t>(winner & 63);
    scratch[1] = static_cast<uint16_t>((winner >> 6) & 63);
    scratch[2] = static_cast<uint16_t>((winner >> 12) & 63);
    scratch[3] = 0;
  }
  if (allreduce_wait(&err) == 0)
    throw std::runtime_error("bus_greedy_pick broadcast: " + err);
  return static_cast<int32_t>(scratch[0]) |
         (static_cast<int32_t>(scratch[1]) << 6) |
         (static_cast<int32_t>(scratch[2]) << 12);
}

}  // namespace dgpp
