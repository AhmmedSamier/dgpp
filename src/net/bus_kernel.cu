// Receive-side consumer for CollectiveBus validation (see bus_kernel.hpp).

#include "net/bus_kernel.hpp"

#include <cuda/atomic>

#include "kernels/flag_protocol.cuh"
#include "net/bus_types.hpp"

namespace dgpp::net {

namespace {

constexpr int kConsumerThreads = 256;

using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_block>;

// Persistent consumer. Every iteration: all threads scan a subset of the
// doorbell cells; one arrived message is claimed by a shared-memory CAS,
// the payload is folded cooperatively, and thread 0 publishes the ack with
// a system-scope release. Exits are only taken at the common post-scan
// barrier — a lone thread never returns while others wait (that would be a
// permanent barrier deadlock), so both the control stop and the inactivity
// deadline converge on the shared stop flag first.
__global__ __launch_bounds__(kConsumerThreads) void bus_consumer_kernel(
    BusRecvView v, uint64_t deadline_cycles) {
  __shared__ uint32_t s_last_seen[kBusMaxConsumerCells];
  __shared__ uint64_t s_hash[kConsumerThreads];
  __shared__ uint32_t s_seq;
  __shared__ uint32_t s_len;
  __shared__ int s_go;    // 0 = unclaimed, >0 = cell index + 1
  __shared__ int s_stop;  // set by any exit path

  FlagRef go_ref(s_go);
  FlagRef stop_ref(s_stop);

  const int total_cells = v.lat_slots + v.bulk_slots;
  for (int i = threadIdx.x; i < kBusMaxConsumerCells; i += blockDim.x)
    s_last_seen[i] = 0;
  if (threadIdx.x == 0) stop_ref.store(0, cuda::memory_order_relaxed);
  __syncthreads();

  uint64_t idle_since = clock64();
  for (;;) {
    if (threadIdx.x == 0) go_ref.store(0, cuda::memory_order_relaxed);
    __syncthreads();

    // Thread 0 gates on the control cell; the deadline is checked in the
    // idle branch below. Both set s_stop, which everyone observes after
    // the barrier.
    if (threadIdx.x == 0 &&
        flag_load_acquire(const_cast<uint32_t*>(&v.control->seq)) ==
            kFlagStopSequence)
      stop_ref.store(1, cuda::memory_order_relaxed);

    // Scan phase: claim exactly one arrived message.
    for (int cell = threadIdx.x;
         cell < total_cells && go_ref.load(cuda::memory_order_relaxed) == 0 &&
         stop_ref.load(cuda::memory_order_relaxed) == 0;
         cell += blockDim.x) {
      const bool is_lat = cell < v.lat_slots;
      const int slot = is_lat ? cell : cell - v.lat_slots;
      const StartSlot* door =
          is_lat ? &v.doorbell_lat[slot] : &v.doorbell_bulk[slot];
      const uint32_t seq =
          flag_load_acquire(const_cast<uint32_t*>(&door->seq));
      if (seq != 0 && seq != s_last_seen[cell]) {
        int expected = 0;
        if (go_ref.compare_exchange_strong(expected, cell + 1,
                                           cuda::memory_order_relaxed,
                                           cuda::memory_order_relaxed)) {
          // Doorbell lines are written whole by the NIC; len rides the
          // same 64-byte line as seq, behind the acquire on seq.
          s_seq = seq;
          s_len = door->len;
          s_last_seen[cell] = seq;
        }
      }
    }
    __syncthreads();

    if (stop_ref.load(cuda::memory_order_relaxed)) return;  // common exit
    const int go = go_ref.load(cuda::memory_order_relaxed);
    if (go == 0) {
      // Inactivity watchdog: thread 0 flags; everyone exits together next
      // round. Threads park briefly instead of spinning at full rate.
      if (threadIdx.x == 0 && clock64() - idle_since > deadline_cycles)
        stop_ref.store(1, cuda::memory_order_relaxed);
      flag_poll_pause();
      continue;
    }

    const int cell = go - 1;
    const bool is_lat = cell < v.lat_slots;
    const int slot = is_lat ? cell : cell - v.lat_slots;
    const uint32_t seq = s_seq;
    const size_t words = static_cast<size_t>(s_len) / 8;
    const uint64_t* base =
        is_lat
            ? v.payload_lat + static_cast<size_t>(slot) * (v.lat_slot_bytes / 8)
            : v.payload_bulk +
                  static_cast<size_t>(slot) * (v.bulk_slot_bytes / 8);

    // Cooperative fold with the golden-ratio pair mix; xor-combine after.
    uint64_t h = 0;
    for (size_t i = static_cast<size_t>(threadIdx.x); i < words;
         i += kConsumerThreads)
      h ^= base[i] * kFoldMultiplier;
    s_hash[threadIdx.x] = h;
    __syncthreads();

    if (threadIdx.x == 0) {
      uint64_t total_hash = 0;
      for (int t = 0; t < kConsumerThreads; ++t) total_hash ^= s_hash[t];
      FlagAck* ack = is_lat ? &v.ack_lat[slot] : &v.ack_bulk[slot];
      ack->cycles = clock64();
      ack->hash = total_hash;
      flag_store_release(&ack->seq, seq);
    }
    idle_since = clock64();
    __syncthreads();  // ack published and last_seen stable before next claim
  }
}

}  // namespace

// ---- per-collective all-reduce consumer (DESIGN §6.3) ------------------------

namespace {

// Deterministic rank-order fp32 accumulation. The host oracle computes the
// identical chain, so verification is bitwise, not tolerance.
__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

__global__ __launch_bounds__(kConsumerThreads) void bus_allreduce_kernel(
    BusAllReduceView v, int my_rank, const __nv_bfloat16* src,
    __nv_bfloat16* dst, uint32_t elems, uint32_t ctl_seq,
    BusAllReduceCtl* ctl, uint64_t deadline_cycles) {
  using BlockRef = cuda::atomic_ref<int, cuda::thread_scope_block>;
  using SysRef = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;

  __shared__ int s_got[kBusMaxPeers];  // 0 waiting, 1 claimed+acked
  __shared__ uint32_t s_seq[kBusMaxPeers];
  __shared__ size_t s_words[kBusMaxPeers];
  __shared__ const uint16_t* s_payload[kBusMaxPeers];
  __shared__ uint64_t s_hash[kConsumerThreads];
  __shared__ int s_go;    // 0 none, >0 = flat cell index + 1
  __shared__ int s_stop;  // any exit condition
  __shared__ int s_failed;

  BlockRef go_ref(s_go);
  BlockRef stop_ref(s_stop);

  for (int p = 0; p < kBusMaxPeers; ++p) s_got[p] = 0;
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_failed = 0;
  }
  __syncthreads();

  // Phase 1 — snapshot the source vector into every peer's staging slot.
  // One u32 per thread iteration; 4096 bf16 = 2048 words over 256 threads.
  // The snapshot is load-bearing for the pre-staged seam (src == dst == a
  // staging buffer the producing GEMM wrote): the fold below overwrites
  // src in place, so each peer's send buffer must hold a copy taken BEFORE
  // the fold — the engine's post reads these, never the folding src.
  // Pre-staged sources are pinned (the GEMM wrote them directly), so this
  // pass is a pinned fan-out; the device→slot copy is what disappears.
  const uint32_t words = elems / 2;
  for (int p = 0; p < v.send_peers; ++p) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    uint32_t* d = reinterpret_cast<uint32_t*>(
        const_cast<uint16_t*>(v.send_payload[p]));
    for (uint32_t w = threadIdx.x; w < words; w += kConsumerThreads) d[w] = s[w];
  }
  // Each thread fences its own slot writes system-wide, then thread 0's
  // release publishes them all to the engine's acquire (the canonical
  // producer handoff; a lone barrier would only order CTA-scope).
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->stamp_stage = clock64();
    SysRef ready(ctl->ready_bits);
    uint64_t bits = 0;
    for (int p = 0; p < v.send_peers; ++p) bits |= 1ULL << p;
    ready.store(bits, cuda::memory_order_release);
  }

  // Phase 2 — wait for every peer's doorbell. One claim per round, same
  // shared-CAS discipline as the persistent consumer; the ring/credit
  // contract guarantees at most one unconsumed latency cell per peer.
  const int lat_slots_per_view = v.recv_views > 0 ? v.recv[0].lat_slots : 0;
  const int total_cells = v.recv_views * lat_slots_per_view;
  const uint64_t start = clock64();
  SysRef done_ref(ctl->done_seq);
  for (;;) {
    if (threadIdx.x == 0) {
      go_ref.store(0, cuda::memory_order_relaxed);
      // Engine poison (the request already failed) or the cycle deadline:
      // either way this collective is over — exit through the common path.
      if (done_ref.load(cuda::memory_order_acquire) == ctl_seq ||
          clock64() - start > deadline_cycles) {
        stop_ref.store(1, cuda::memory_order_relaxed);
        s_failed = 1;
      }
    }
    __syncthreads();

    int missing = 0;
    for (int p = 0; p < v.send_peers; ++p) missing += (s_got[p] == 0);
    if (missing == 0 || stop_ref.load(cuda::memory_order_relaxed)) break;

    // Scan: flat cell = (peer, lane, slot). Claim only waiting peers.
    for (int cell = threadIdx.x;
         cell < total_cells && go_ref.load(cuda::memory_order_relaxed) == 0 &&
         stop_ref.load(cuda::memory_order_relaxed) == 0;
         cell += blockDim.x) {
      const int view_idx = cell / lat_slots_per_view;
      const int slot = cell % lat_slots_per_view;
      const int peer = view_idx / v.lanes_per_peer;
      if (s_got[peer] != 0) continue;
      const StartSlot* door = &v.recv[view_idx].doorbell_lat[slot];
      const uint32_t seq =
          flag_load_acquire(const_cast<uint32_t*>(&door->seq));
      if (seq == 0) continue;
      const FlagAck* ack = &v.recv[view_idx].ack_lat[slot];
      if (seq == ack->seq) continue;  // already consumed
      int expected = 0;
      if (go_ref.compare_exchange_strong(expected, cell + 1,
                                         cuda::memory_order_relaxed,
                                         cuda::memory_order_relaxed)) {
        s_seq[peer] = seq;
        s_words[peer] = door->len / 8;
      }
    }
    __syncthreads();

    const int go = go_ref.load(cuda::memory_order_relaxed);
    if (go == 0) {
      flag_poll_pause();
      continue;
    }

    // Fold the claimed payload (credit-hash continuity: the engine's
    // recycle harvests this into the peer's credit WRITE), then publish
    // the standard ack — hash/cycles first, seq last (release).
    const int cell = go - 1;
    const int view_idx = cell / lat_slots_per_view;
    const int slot = cell % lat_slots_per_view;
    const int peer = view_idx / v.lanes_per_peer;
    const BusRecvView& rv = v.recv[view_idx];
    const uint64_t* base = rv.payload_lat +
                           static_cast<size_t>(slot) * (rv.lat_slot_bytes / 8);
    uint64_t h = 0;
    for (size_t i = threadIdx.x; i < s_words[peer]; i += kConsumerThreads)
      h ^= base[i] * kFoldMultiplier;
    s_hash[threadIdx.x] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t total_hash = 0;
      for (int t = 0; t < kConsumerThreads; ++t) total_hash ^= s_hash[t];
      FlagAck* ack = &rv.ack_lat[slot];
      ack->cycles = clock64();
      ack->hash = total_hash;
      flag_store_release(&ack->seq, s_seq[peer]);
      s_payload[peer] = reinterpret_cast<const uint16_t*>(base);
      s_got[peer] = 1;
      if (ctl->stamp_first_claim == 0) ctl->stamp_first_claim = clock64();
    }
    __syncthreads();  // ack published and claim state stable before next round
  }

  // Common exit: every path converges here, so no thread leaves while
  // others still wait at a barrier (the persistent-kernel lesson).
  if (s_failed == 0) {
    // Canonical global-rank-order accumulation: rank 0's vector first,
    // always. Every rank computes the identical fp32 chain, so all
    // destinations agree bitwise — replicated consumers stay in lockstep
    // (per-rank orderings could diverge in the last ulp and, after bf16
    // rounding, not always visibly).
    const uint16_t* local = reinterpret_cast<const uint16_t*>(src);
    for (uint32_t i = threadIdx.x; i < elems; i += kConsumerThreads) {
      float acc = 0.0f;
      for (int r = 0; r < v.send_peers + 1; ++r) {
        // Views are peer-major ascending; rank r maps to peer index
        // r (below us) or r-1 (above us).
        const uint16_t* vec =
            r == my_rank ? local : s_payload[r < my_rank ? r : r - 1];
        acc += bf16_to_f32(vec[i]);
      }
      dst[i] = __float2bfloat16(acc);
    }
    if (threadIdx.x == 0) ctl->stamp_reduce_done = clock64();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->status = s_failed == 0 ? 0 : 1;
    done_ref.store(ctl_seq, cuda::memory_order_release);
  }
}

}  // namespace

cudaError_t launch_bus_consumer(const BusRecvView& view,
                                  uint64_t deadline_cycles, cudaStream_t stream) {
  bus_consumer_kernel<<<1, kConsumerThreads, 0, stream>>>(view,
                                                           deadline_cycles);
  return cudaGetLastError();
}

cudaError_t launch_bus_allreduce(const BusAllReduceView& v, int my_rank,
                                  const __nv_bfloat16* src, __nv_bfloat16* dst,
                                  uint32_t elems, uint32_t ctl_seq,
                                  BusAllReduceCtl* ctl,
                                  uint64_t deadline_cycles,
                                  cudaStream_t stream) {
  bus_allreduce_kernel<<<1, kConsumerThreads, 0, stream>>>(
      v, my_rank, src, dst, elems, ctl_seq, ctl, deadline_cycles);
  return cudaGetLastError();
}

// ---- segment-quantized bulk collective (DESIGN §6.3, prefill class) ----

namespace {

// Bring-up diagnostics (see the BKFIN/BKV*/BKCL records): the failing
// kernels must report what THEIR reads saw, not what the CPU's stall
// microscope saw — the two views have diverged, and only the kernel-side
// record adjudicates scan blindness vs mapping rejection vs CAS loss.
constexpr int kBulkScanCap = kBusMaxBulkCells;  // protocol bound, start-validated
constexpr int kBulkClaimLog = 16;  // first claims of a failing kernel

__global__ __launch_bounds__(kConsumerThreads) void bus_bulk_collective_kernel(
    BusAllReduceView v, int my_rank, int phase,
    const __nv_bfloat16* src_bf, __nv_bfloat16* dst_bf, BusBulkSegPlan plan,
    uint64_t* staged_counters, uint32_t ctl_seq, BusAllReduceCtl* ctl,
    uint64_t deadline_cycles) {
  using BlockRef = cuda::atomic_ref<int, cuda::thread_scope_block>;
  using SysRef = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;

  __shared__ int s_stop;
  __shared__ int s_failed;
  __shared__ int s_go;         // 0 none, >0 = flat bulk cell index + 1
  __shared__ uint32_t s_seq;   // claimed doorbell seq
  __shared__ uint32_t s_len;   // claimed doorbell len (bytes)
  __shared__ int s_total_arrived;
  __shared__ int s_expected[kBusMaxPeersSized];
  // Per (peer, lane): bulk arrivals consumed before this segment, summed
  // from our own ack cells — this kernel chain is their only writer, so
  // the base is self-computed and race-free.
  __shared__ uint64_t s_lane_base[kBusMaxPeersSized * 2];
  // RS claim records: per stripe of my sub-range — the arrived-peer
  // bitmask and each peer's payload pointer (filled at claim, folded at
  // the end; the validation pass below refuses to fold a partial mask).
  __shared__ uint64_t s_ready[kBusMaxBulkSegStripes];
  __shared__ const uint16_t* s_ptr[kBusMaxBulkSegStripes][kBusMaxPeersSized];
  // AG claim dedupe: per peer, the bitmask of its sub-range's stripes
  // already landed (idempotent copies, counted once).
  __shared__ uint64_t s_agot[kBusMaxPeersSized];
  // Bring-up records: the scan's own view of each cell (max door seq ever
  // returned by the acquire load; last ack seq it read) and the first
  // claims' mapping values. Written by each cell's exclusive scan owner /
  // thread 0, printed once at the failure exit.
  __shared__ uint32_t s_scan_max[kBulkScanCap];
  __shared__ uint32_t s_scan_ack[kBulkScanCap];
  // Ack deferral record: one bit per flat cell this kernel claimed. The
  // ack means "payload CONSUMED" — for RS that is the exit fold, not the
  // claim — so acks flush after the fold; until then the sender's credit
  // cannot return and the payload slots stay put (the overwrite race
  // the deferred ack exists to close).
  __shared__ uint8_t s_claimed[kBulkScanCap];
  __shared__ uint32_t s_cl_cell[kBulkClaimLog];
  __shared__ uint32_t s_cl_seq[kBulkClaimLog];
  __shared__ uint64_t s_cl_j[kBulkClaimLog];
  __shared__ uint64_t s_cl_base[kBulkClaimLog];
  __shared__ uint32_t s_cl_k[kBulkClaimLog];
  __shared__ uint32_t s_cl_acc[kBulkClaimLog];
  __shared__ uint32_t s_cl_n;

  const int peers = v.send_peers;
  const int lanes = v.lanes_per_peer;
  const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bf);
  uint16_t* dst = reinterpret_cast<uint16_t*>(dst_bf);
  const uint32_t stripe = plan.stripe_elems;

  // Global stripe byte length: uniform except the buffer's tail.
  const uint64_t total_bytes = static_cast<uint64_t>(plan.total_elems) * 2;
  auto stripe_bytes = [&](uint32_t global_stripe) -> uint64_t {
    const uint64_t left =
        total_bytes - static_cast<uint64_t>(global_stripe) * stripe * 2;
    const uint64_t full = static_cast<uint64_t>(stripe) * 2;
    return left < full ? left : full;
  };

  BlockRef go_ref(s_go);
  BlockRef stop_ref(s_stop);
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_failed = 0;
    s_total_arrived = 0;
    s_cl_n = 0;
  }
  for (int k = 0; k < kBusMaxBulkSegStripes; ++k) s_ready[k] = 0;
  for (int p = 0; p < kBusMaxPeersSized; ++p) s_agot[p] = 0;
  for (int i = threadIdx.x; i < kBulkScanCap; i += kConsumerThreads) {
    s_scan_max[i] = 0;
    s_scan_ack[i] = 0;
    s_claimed[i] = 0;
  }
  __syncthreads();

  // ---- outbound staging into the arena rows -----------------------------
  // RS: every peer's sub-range, from my src. AG: my (reduced) sub-range,
  // from dst, to every peer. The per-peer staged counter releases the
  // engine's posting of that row (stripe k lands at row + k*bulk_slot).
  {
    const uint16_t* from = phase == 0 ? src : dst;
    for (int p = 0; p < peers; ++p) {
      const uint32_t base = phase == 0 ? plan.out_base[p] : plan.my_base;
      const uint32_t count = phase == 0 ? plan.out_count[p] : plan.my_count;
      for (uint32_t k = 0; k < count; ++k) {
        const uint32_t gs = plan.seg_first + base + k;
        // u64 words throughout — an earlier cut indexed u32 pointers with a
        // u64-word bound and staged exactly half of every stripe.
        const uint64_t* s = reinterpret_cast<const uint64_t*>(
            from + static_cast<uint64_t>(gs) * stripe);
        uint64_t* d = reinterpret_cast<uint64_t*>(
            const_cast<uint16_t*>(v.send_payload[p]) +
            static_cast<uint64_t>(k) * stripe);
        const uint64_t bytes = stripe_bytes(gs);
        for (uint64_t w = threadIdx.x; w < bytes / 8; w += kConsumerThreads)
          d[w] = s[w];
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0)
        SysRef(staged_counters[p])
            .store(static_cast<uint64_t>(count), cuda::memory_order_release);
    }
  }

  // ---- expected arrivals + per-lane bases --------------------------------
  for (int p = threadIdx.x; p < peers; p += kConsumerThreads)
    s_expected[p] =
        phase == 0 ? static_cast<int>(plan.my_count)
                   : static_cast<int>(plan.out_count[p]);
  // Per (peer, lane) view: sum our ack cells (each slot's seq counts the
  // claims consumed on that slot). Spread views over threads.
  for (int idx = threadIdx.x; idx < v.recv_views; idx += kConsumerThreads) {
    const BusRecvView& rv = v.recv[idx];
    uint64_t base = 0;
    for (int c = 0; c < rv.bulk_slots; ++c)
      base += flag_load_acquire(const_cast<uint32_t*>(&rv.ack_bulk[c].seq));
    s_lane_base[idx] = base;
  }
  __shared__ int s_expected_total;
  if (threadIdx.x == 0) {
    int total = 0;
    for (int p = 0; p < peers; ++p) total += s_expected[p];
    s_expected_total = total;
  }
  __syncthreads();

  // ---- claim loop ---------------------------------------------------------
  // One cell per round (the shared-CAS discipline); every claimed arrival
  // is hashed + acked (credit continuity), then mapped to its stripe:
  // the j-th arrival on a lane sits in ring slot j%depth with seq
  // j/depth+1 (RC in-order per lane); the sender's round-robin striping
  // (stripe k on lane k%lanes) maps the in-segment index i to k=i*lanes+l.
  const int slots_per_view = v.recv_views > 0 ? v.recv[0].bulk_slots : 0;
  const int total_cells = v.recv_views * slots_per_view;
  const uint64_t start = clock64();
  SysRef done_ref(ctl->done_seq);
  for (;;) {
    if (threadIdx.x == 0) {
      go_ref.store(0, cuda::memory_order_relaxed);
      // Engine poison (the request failed) or the cycle deadline — either
      // way this segment is over; exit through the common path.
      if (done_ref.load(cuda::memory_order_acquire) == ctl_seq ||
          clock64() - start > deadline_cycles) {
        stop_ref.store(1, cuda::memory_order_relaxed);
        s_failed = 1;
      }
    }
    __syncthreads();
    if (s_total_arrived >= s_expected_total ||
        stop_ref.load(cuda::memory_order_relaxed))
      break;

    for (int cell = threadIdx.x;
         cell < total_cells && go_ref.load(cuda::memory_order_relaxed) == 0 &&
         stop_ref.load(cuda::memory_order_relaxed) == 0;
         cell += blockDim.x) {
      // This kernel's own claims re-present until the deferred ack flush
      // (the ack rides at exit), and a re-presented cell would re-enter
      // the shared CAS every round — lowest-thread-wins inside a warp —
      // stealing the claim slot from a fresh doorbell owned by a higher
      // thread of the same warp, forever (measured: arrived 1/2 with the
      // second doorbell live in its cell for 5s). Skip our own claims;
      // the CTA barrier between rounds publishes the bits.
      if (cell < kBulkScanCap && s_claimed[cell] != 0) continue;
      const int view_idx = cell / slots_per_view;
      const int slot = cell % slots_per_view;
      const StartSlot* door = &v.recv[view_idx].doorbell_bulk[slot];
      const uint32_t seq =
          flag_load_acquire(const_cast<uint32_t*>(&door->seq));
      if (seq == 0) continue;
      const uint32_t ack_seq = flag_load_acquire(
          const_cast<uint32_t*>(&v.recv[view_idx].ack_bulk[slot].seq));
      // Bring-up scan record (see BKFIN): what THIS kernel's acquire
      // loads returned, kept per exclusive cell owner.
      if (cell < kBulkScanCap) {
        if (seq > s_scan_max[cell]) s_scan_max[cell] = seq;
        s_scan_ack[cell] = ack_seq;
      }
      if (seq == ack_seq) continue;  // already consumed
      // Segment window — load-bearing, not an optimization. The doorbell
      // rings are flight- and phase-agnostic FIFOs, and senders do NOT
      // progress in lockstep (shard geometry sees to that: a rank whose
      // shard ends in segment 0 enters AG while peers still fold RS). A
      // future segment/phase's doorbell lands in an active kernel's
      // cells all the same; claiming it eagerly would ack it away from
      // the kernel that maps it (measured: an RS kernel ate 16 AG
      // doorbells — mapped past the stripe guard, acked anyway — and
      // both phases deadlocked on the loss). j is the doorbell's
      // ring-lifetime index, so this segment's stripes occupy exactly
      // [base, base + this lane's stripe share); everything else stays
      // put for the kernel whose window contains it.
      const uint64_t j_arr =
          (static_cast<uint64_t>(seq) - 1) * slots_per_view + slot;
      const uint64_t in_seg_arr = j_arr - s_lane_base[view_idx];
      const int lane_idx = view_idx % lanes;
      const int peer_idx = view_idx / lanes;
      const uint32_t seg_stripes =
          phase == 0 ? plan.my_count : plan.out_count[peer_idx];
      const uint32_t lane_stripes =
          lane_idx < seg_stripes
              ? (seg_stripes - lane_idx + static_cast<uint32_t>(lanes) - 1) /
                    static_cast<uint32_t>(lanes)
              : 0;
      if (in_seg_arr >= lane_stripes) continue;  // not ours to claim
      int expected = 0;
      if (go_ref.compare_exchange_strong(expected, cell + 1,
                                         cuda::memory_order_relaxed,
                                         cuda::memory_order_relaxed)) {
        s_seq = seq;
        s_len = door->len;
      }
    }
    __syncthreads();
    const int go = go_ref.load(cuda::memory_order_relaxed);
    if (go == 0) {
      flag_poll_pause();
      continue;
    }

    const int cell = go - 1;
    const int view_idx = cell / slots_per_view;
    const int slot = cell % slots_per_view;
    const int lane = view_idx % lanes;
    const int peer = view_idx / lanes;
    const BusRecvView& rv = v.recv[view_idx];
    const uint16_t* payload =
        reinterpret_cast<const uint16_t*>(
            rv.payload_bulk +
            static_cast<size_t>(slot) * (rv.bulk_slot_bytes / 8));

    // AG lands the arrival directly: it is peer p's sub-range stripe k,
    // a pure copy into dst (byte-identical for every receiver, so a
    // re-claimed cell's duplicate copy is harmless). The copy consumes
    // the payload here; RS defers to the exit fold (see the ack flush).
    const uint64_t j =
        (static_cast<uint64_t>(s_seq) - 1) * rv.bulk_slots + slot;
    const uint64_t in_seg = j - s_lane_base[view_idx];
    const uint32_t k =
        static_cast<uint32_t>(in_seg) * static_cast<uint32_t>(lanes) + lane;
    if (phase == 1 && k < plan.out_count[peer]) {
      const uint32_t gs = plan.seg_first + plan.out_base[peer] + k;
      uint16_t* d = dst + static_cast<uint64_t>(gs) * stripe;
      for (uint32_t e = threadIdx.x; e < s_len / 2; e += kConsumerThreads)
        d[e] = payload[e];
    }
    __syncthreads();

    // Claims are DEDUPED in shared memory (CTA-coherent): the ack no
    // longer marks a claim (it is deferred to the post-fold flush below),
    // so a claimed cell re-presents every round until then, and the
    // re-claim must be a no-op — a double-counted arrival exits the loop
    // early and leaves a stripe's contribution missing (measured: k=15
    // empty while 16 claims counted).
    if (threadIdx.x == 0) {
      s_claimed[cell] = 1;  // ack deferred; the fold still reads the slot
      const uint64_t bit = 1ULL << peer;
      bool counted = false;
      if (phase == 0) {
        // RS: the (stripe k, peer) contribution, once.
        if (k < plan.my_count && (s_ready[k] & bit) == 0) {
          s_ptr[k][peer] = payload;
          s_ready[k] |= bit;
          ++s_total_arrived;
          counted = true;
        }
      } else {
        // AG: peer p's stripe k, once (the copy above is idempotent).
        if (k < plan.out_count[peer] && (s_agot[peer] & (1ULL << k)) == 0) {
          s_agot[peer] |= 1ULL << k;
          ++s_total_arrived;
          counted = true;
        }
      }
      // Bring-up claim log (see BKFIN): the mapping values of the first
      // claims — base==0 or k past the guard here is a mapping rejection,
      // not scan blindness, and the two need different fixes.
      if (s_cl_n < kBulkClaimLog) {
        s_cl_cell[s_cl_n] = static_cast<uint32_t>(cell);
        s_cl_seq[s_cl_n] = s_seq;
        s_cl_j[s_cl_n] = j;
        s_cl_base[s_cl_n] = s_lane_base[view_idx];
        s_cl_k[s_cl_n] = k;
        s_cl_acc[s_cl_n] = counted ? 1u : 0u;
        ++s_cl_n;
      }
    }
    __syncthreads();  // claim records stable before the next claim round
  }

  // ---- reduce (RS only): canonical ascending-rank fold --------------------
  if (s_failed == 0 && phase == 0) {
    // Validation before touching dst: every stripe of my sub-range must
    // hold all peers' contributions — a mapping bug must fail loudly,
    // never fold a partial sum (the silent-corruption class).
    const uint64_t all_peers = (peers >= 64) ? ~0ULL : ((1ULL << peers) - 1);
    if (threadIdx.x == 0) {
      for (uint32_t k = 0; k < plan.my_count; ++k)
        if (s_ready[k] != all_peers) s_failed = 1;
    }
    __syncthreads();
    if (s_failed == 0) {
      // Flat parallel map over my sub-range's elements (uniform stripe
      // length except the tail; per-element bound check).
      const uint64_t span =
          static_cast<uint64_t>(plan.my_count) * stripe;
      for (uint64_t flat = threadIdx.x; flat < span;
           flat += kConsumerThreads) {
        const uint32_t k =
            static_cast<uint32_t>(flat / stripe);
        const uint32_t e = static_cast<uint32_t>(flat % stripe);
        const uint32_t gs = plan.seg_first + plan.my_base + k;
        if (e >= stripe_bytes(gs) / 2) continue;  // tail padding
        const uint16_t* local = src + static_cast<uint64_t>(gs) * stripe;
        float acc = 0.0f;
        for (int r = 0; r < peers + 1; ++r) {
          const uint16_t* vec = r == my_rank
                                    ? local
                                    : s_ptr[k][r < my_rank ? r : r - 1];
          acc += bf16_to_f32(vec[e]);
        }
        // The bit-reinterpret is load-bearing: __nv_bfloat16 converts to
        // uint16_t through its float operator (integer truncation — the
        // fold wrote literal 1s before this), never through its bits.
        dst[static_cast<uint64_t>(gs) * stripe + e] =
            __bfloat16_as_ushort(__float2bfloat16(acc));
      }
      if (threadIdx.x == 0) ctl->stamp_reduce_done = clock64();
    }
  }
  // ---- deferred ack flush -------------------------------------------------
  // The ack certifies the payload CONSUMED — for RS that is the fold above
  // (s_ptr reads the ring slots one last time there), for AG the claim-
  // time copy. Acking at claim time would return the sender's credit
  // while this kernel still reads the slot, and the sender's next stripe
  // would DMA over a fold input (measured: mid-stripe prefix/suffix
  // corruption under cross-rank skew — a sender a segment ahead wraps the
  // 8-deep ring inside one claim-to-exit gap). The door cell cannot move
  // before its credit returns, so the claimed seq is re-read here rather
  // than carried per claim.
  __syncthreads();
  for (int cell = threadIdx.x; cell < total_cells; cell += kConsumerThreads) {
    if (cell >= kBulkScanCap || s_claimed[cell] == 0) continue;
    const int view_idx = cell / slots_per_view;
    const int slot = cell % slots_per_view;
    const BusRecvView& rv = v.recv[view_idx];
    const uint32_t seq = flag_load_acquire(
        const_cast<uint32_t*>(&rv.doorbell_bulk[slot].seq));
    FlagAck* ack = &rv.ack_bulk[slot];
    ack->cycles = clock64();
    ack->hash = 0;  // credit-hash continuity is the consumer kernel's
                    // contract; the collective path verifies via the fold
    flag_store_release(&ack->seq, seq);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (s_failed != 0 && s_total_arrived < s_expected_total) {
      printf("BKFIN rank=%d phase=%d seg=%u arrived=%d/%d my_base=%u "
             "my_count=%u t_ms=%llu poison=%d masks:",
             my_rank, phase, plan.seg_first, s_total_arrived,
             s_expected_total, plan.my_base, plan.my_count,
             (unsigned long long)((clock64() - start) / 1000),
             done_ref.load(cuda::memory_order_acquire) == ctl_seq ? 1 : 0);
      for (uint32_t k = 0; k < plan.my_count; ++k)
        printf(" k%u=%llx", k, (unsigned long long)s_ready[k]);
      printf("\n");
      // Exit-state dump: the base each view computed at launch, the
      // cells' freshest state as this kernel reads them now (ld.cv),
      // and the scan's own records. base==0 on a lane with prior
      // arrivals = the launch-time base read stale; smax < true door =
      // the scan never saw the arrival; sa == smax with a lower true
      // ack = a stale-ack skip.
      for (int view = 0; view < v.recv_views; ++view)
        printf("BKVB r=%d v=%d base=%llu slots=%d\n", my_rank, view,
               (unsigned long long)s_lane_base[view], slots_per_view);
      for (int view = 0; view < v.recv_views; ++view) {
        if (slots_per_view < 8) break;  // fixed-arity dump assumes depth 8
        printf("BKVD r=%d v=%d d=%u,%u,%u,%u,%u,%u,%u,%u "
               "a=%u,%u,%u,%u,%u,%u,%u,%u\n",
               my_rank, view,
               __ldcv(&v.recv[view].doorbell_bulk[0].seq),
               __ldcv(&v.recv[view].doorbell_bulk[1].seq),
               __ldcv(&v.recv[view].doorbell_bulk[2].seq),
               __ldcv(&v.recv[view].doorbell_bulk[3].seq),
               __ldcv(&v.recv[view].doorbell_bulk[4].seq),
               __ldcv(&v.recv[view].doorbell_bulk[5].seq),
               __ldcv(&v.recv[view].doorbell_bulk[6].seq),
               __ldcv(&v.recv[view].doorbell_bulk[7].seq),
               __ldcv(&v.recv[view].ack_bulk[0].seq),
               __ldcv(&v.recv[view].ack_bulk[1].seq),
               __ldcv(&v.recv[view].ack_bulk[2].seq),
               __ldcv(&v.recv[view].ack_bulk[3].seq),
               __ldcv(&v.recv[view].ack_bulk[4].seq),
               __ldcv(&v.recv[view].ack_bulk[5].seq),
               __ldcv(&v.recv[view].ack_bulk[6].seq),
               __ldcv(&v.recv[view].ack_bulk[7].seq));
      }
      for (int view = 0; view < v.recv_views; ++view) {
        if (slots_per_view < 8) break;  // fixed-arity dump assumes depth 8
        const int f = view * slots_per_view;
        if (f + 8 > kBulkScanCap) break;
        printf("BKSR r=%d v=%d sm=%u,%u,%u,%u,%u,%u,%u,%u sa=%u,%u,%u,%u,%u,"
               "%u,%u,%u\n",
               my_rank, view, s_scan_max[f], s_scan_max[f + 1],
               s_scan_max[f + 2], s_scan_max[f + 3], s_scan_max[f + 4],
               s_scan_max[f + 5], s_scan_max[f + 6], s_scan_max[f + 7],
               s_scan_ack[f], s_scan_ack[f + 1], s_scan_ack[f + 2],
               s_scan_ack[f + 3], s_scan_ack[f + 4], s_scan_ack[f + 5],
               s_scan_ack[f + 6], s_scan_ack[f + 7]);
      }
      for (uint32_t i = 0; i < s_cl_n && i < kBulkClaimLog; ++i)
        printf("BKCL r=%d i=%u c=%u seq=%u j=%llu b=%llu k=%u acc=%u\n",
               my_rank, i, s_cl_cell[i], s_cl_seq[i],
               (unsigned long long)s_cl_j[i],
               (unsigned long long)s_cl_base[i], s_cl_k[i], s_cl_acc[i]);
    }
    ctl->status = s_failed == 0 ? 0 : 1;
    done_ref.store(ctl_seq, cuda::memory_order_release);
  }
}

}  // namespace

cudaError_t launch_bus_bulk_collective(const BusAllReduceView& v, int my_rank,
                                       int phase, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst,
                                       const BusBulkSegPlan& plan,
                                       uint64_t* staged_counters,
                                       uint32_t ctl_seq, BusAllReduceCtl* ctl,
                                       uint64_t deadline_cycles,
                                       cudaStream_t stream) {
  bus_bulk_collective_kernel<<<1, kConsumerThreads, 0, stream>>>(
      v, my_rank, phase, src, dst, plan, staged_counters, ctl_seq, ctl,
      deadline_cycles);
  return cudaGetLastError();
}

namespace {

__global__ __launch_bounds__(kConsumerThreads) void bus_warm_work_kernel(
    float* buf, int spins) {
  float acc = static_cast<float>(threadIdx.x) * 1e-6f;
  for (int i = 0; i < spins; ++i) acc = acc * 1.0001f + 0.001f;
  buf[threadIdx.x] = acc;
}

}  // namespace

cudaError_t launch_bus_warm_work(cudaStream_t stream, float* buf, int spins) {
  bus_warm_work_kernel<<<1, kConsumerThreads, 0, stream>>>(buf, spins);
  return cudaGetLastError();
}

uint64_t bus_consumer_deadline_cycles(double seconds) {
  int clock_khz = 0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) !=
          cudaSuccess ||
      clock_khz <= 0)
    return 0;
  return static_cast<uint64_t>(static_cast<double>(clock_khz) * 1000.0 *
                               seconds);
}

uint64_t bus_fold(const void* data, size_t bytes) {
  return bus_fold64(static_cast<const uint64_t*>(data), bytes / 8);
}

}  // namespace dgpp::net
