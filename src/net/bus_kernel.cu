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
