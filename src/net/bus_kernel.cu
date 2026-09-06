#include "net/bus_kernel.hpp"

#include <cooperative_groups.h>
#include <cuda/atomic>

#include "kernels/flag_protocol.cuh"
#include "net/bus_types.hpp"

namespace dgpp::net {

namespace {
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
}  // namespace

namespace {

constexpr int kConsumerThreads = 256;

// PLACEMENT-visibility loaders (the 2026-09-01 hunt's root cause): the
// NIC places doorbells and payloads into host DRAM with DMA writes the
// GPU's caches do not snoop — a PLAIN cached load of a pinned cell can
// return pre-arrival bytes forever (the placement gate spun 55M
// iterations on zeros the CPU could see beside them; the pre-gate
// kernels folded the same stale lines into garbage). Every read of
// NIC-written memory follows the doorbell polls' system-scope
// discipline. (atomic_ref wants non-const; the loads are read-only.)
__device__ inline uint32_t sys_load_u32(const uint32_t* p) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(
      *const_cast<uint32_t*>(p));
  return ref.load(cuda::memory_order_relaxed);
}
__device__ inline uint64_t sys_load_u64(const uint64_t* p) {
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(
      *const_cast<uint64_t*>(p));
  return ref.load(cuda::memory_order_relaxed);
}
__device__ inline uint16_t sys_load_u16(const uint16_t* p) {
  cuda::atomic_ref<uint16_t, cuda::thread_scope_system> ref(
      *const_cast<uint16_t*>(p));
  return ref.load(cuda::memory_order_relaxed);
}

using FlagRef = cuda::atomic_ref<int, cuda::thread_scope_block>;

// Deterministic rank-order fp32 accumulation. The host oracle computes the
// identical chain, so verification is bitwise, not tolerance.
__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

// 16-byte system-scope load of NIC-placed memory (the same discipline as
// the scalar loaders above, four words per round trip).
__device__ inline uint4 sys_load_u128(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ inline bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

// The payload fold every claim gates on — the CPU's bus_fold64, which is
// the definition: XOR over words i of (w_i + i + 1) * kFoldMultiplier. XOR
// is commutative, so which thread folds which word and how the partials
// combine are free choices; this one loads two words per 16-byte system
// load and combines through the warp (the 256-way serial loop it replaces
// was ~5 us of thread 0's time on every claim). Every thread returns the
// total. Call from all threads; contains one barrier.
// `stage` (2026-09-06, the graph kernel): a shared-memory copy of the
// payload written by the same loads the hash makes — the gate's last,
// matching pass leaves a complete copy, and the fold then reads shared
// memory instead of NIC-placed system memory (the fold was 9 of every
// collective's ~50 us). Null keeps the hash-only pass.
__device__ inline uint64_t block_fold_payload(const uint64_t* base,
                                              size_t words,
                                              uint64_t* s_warp_hash,
                                              uint64_t* stage = nullptr) {
  uint64_t h = 0;
  if (aligned16(base)) {
    const size_t pairs = words / 2;
    for (size_t j = threadIdx.x; j < pairs; j += kConsumerThreads) {
      const uint4 v = sys_load_u128(reinterpret_cast<const uint4*>(base) + j);
      if (stage != nullptr) reinterpret_cast<uint4*>(stage)[j] = v;
      const uint64_t w0 = static_cast<uint64_t>(v.x) |
                          (static_cast<uint64_t>(v.y) << 32);
      const uint64_t w1 = static_cast<uint64_t>(v.z) |
                          (static_cast<uint64_t>(v.w) << 32);
      const size_t i = 2 * j;
      h ^= (w0 + i + 1) * kFoldMultiplier;
      h ^= (w1 + i + 2) * kFoldMultiplier;
    }
    if ((words & 1) != 0 && threadIdx.x == 0) {
      const size_t i = words - 1;
      const uint64_t w = sys_load_u64(&base[i]);
      if (stage != nullptr) stage[i] = w;
      h ^= (w + i + 1) * kFoldMultiplier;
    }
  } else {
    for (size_t i = threadIdx.x; i < words; i += kConsumerThreads) {
      const uint64_t w = sys_load_u64(&base[i]);
      if (stage != nullptr) stage[i] = w;
      h ^= (w + i + 1) * kFoldMultiplier;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if ((threadIdx.x & 31) == 0) s_warp_hash[threadIdx.x / 32] = h;
  __syncthreads();
  uint64_t total = 0;
#pragma unroll
  for (int w = 0; w < kConsumerThreads / 32; ++w) total ^= s_warp_hash[w];
  return total;
}

// Phase-1 snapshot: src -> each peer's staging row. 16-byte moves when the
// geometry allows (the decode hidden: 4096 bf16 = 512 of them), u32
// otherwise. Plain loads/stores: both sides are ours.
__device__ inline void block_copy_row(const uint32_t* src, uint32_t* dst,
                                      uint32_t words) {
  if ((words & 3) == 0 && aligned16(src) && aligned16(dst)) {
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (uint32_t w = threadIdx.x; w < words / 4; w += kConsumerThreads)
      d4[w] = s4[w];
  } else {
    for (uint32_t w = threadIdx.x; w < words; w += kConsumerThreads)
      dst[w] = src[w];
  }
}

// The canonical fold: dst[i] = bf16(((0 + v_0[i]) + v_1[i]) + ...) over
// global ranks in ascending order, fp32 — every rank computes the same
// chain, so all destinations agree bitwise (and the host oracle computes
// this exact chain). `peers` holds the claimed payloads by peer index
// (peer-major ascending; rank r is index r below us, r-1 above); our own
// vector is `local`. Peer bytes are NIC-placed: system-scope loads. Eight
// elements per thread iteration through 16-byte loads when aligned; the
// per-element arithmetic is identical either way. (TRIED 2026-09-06 and
// reverted: four vectors' loads per thread issued before any is consumed —
// the graph window timeline's fold span stayed at 9.3 us, so the span is
// not this loop's load latency.)
__device__ inline void block_fold_vectors(const uint16_t* local,
                                          const uint16_t* const* peers,
                                          int send_peers, int my_rank,
                                          uint32_t elems,
                                          __nv_bfloat16* dst) {
  const int world = send_peers + 1;
  bool vec_ok = (elems % 8) == 0 && aligned16(local) && aligned16(dst);
  for (int p = 0; p < send_peers; ++p) vec_ok = vec_ok && aligned16(peers[p]);
  if (vec_ok) {
    const uint32_t vecs = elems / 8;
    for (uint32_t vi = threadIdx.x; vi < vecs; vi += kConsumerThreads) {
      uint4 in[kBusMaxPeers + 1];
      for (int r = 0; r < world; ++r) {
        if (r == my_rank) {
          in[r] = reinterpret_cast<const uint4*>(local)[vi];
        } else {
          const uint16_t* vec = peers[r < my_rank ? r : r - 1];
          in[r] = sys_load_u128(reinterpret_cast<const uint4*>(vec) + vi);
        }
      }
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
      for (int r = 0; r < world; ++r) {
        const uint32_t w[4] = {in[r].x, in[r].y, in[r].z, in[r].w};
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[e] += bf16_to_f32(static_cast<uint16_t>(
              (e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu)));
      }
      uint4 out;
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const __nv_bfloat16 lo = __float2bfloat16(acc[2 * q]);
        const __nv_bfloat16 hi = __float2bfloat16(acc[2 * q + 1]);
        o[q] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo)) |
               (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi))
                << 16);
      }
      out.x = o[0]; out.y = o[1]; out.z = o[2]; out.w = o[3];
      reinterpret_cast<uint4*>(dst)[vi] = out;
    }
    return;
  }
  for (uint32_t i = threadIdx.x; i < elems; i += kConsumerThreads) {
    float acc = 0.0f;
    for (int r = 0; r < world; ++r) {
      const uint16_t* vec = r == my_rank ? local : peers[r < my_rank ? r : r - 1];
      acc += r == my_rank ? bf16_to_f32(vec[i])
                          : bf16_to_f32(sys_load_u16(&vec[i]));
    }
    dst[i] = __float2bfloat16(acc);
  }
}

// The canonical fold over STAGED peer payloads (shared memory, plain
// loads): the same per-element chain as block_fold_vectors, so the
// destinations agree with it bitwise. `staged[p]` is peer p's copy.
__device__ inline void block_fold_vectors_staged(
    const uint16_t* local, const uint16_t* const* staged, int send_peers,
    int my_rank, uint32_t elems, __nv_bfloat16* dst) {
  const int world = send_peers + 1;
  bool vec_ok = (elems % 8) == 0 && aligned16(local) && aligned16(dst);
  for (int p = 0; p < send_peers; ++p) vec_ok = vec_ok && aligned16(staged[p]);
  if (vec_ok) {
    const uint32_t vecs = elems / 8;
    for (uint32_t vi = threadIdx.x; vi < vecs; vi += kConsumerThreads) {
      uint4 in[kBusMaxPeers + 1];
      for (int r = 0; r < world; ++r) {
        const uint16_t* vec =
            r == my_rank ? local : staged[r < my_rank ? r : r - 1];
        in[r] = reinterpret_cast<const uint4*>(vec)[vi];
      }
      float acc[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
      for (int r = 0; r < world; ++r) {
        const uint32_t w[4] = {in[r].x, in[r].y, in[r].z, in[r].w};
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[e] += bf16_to_f32(static_cast<uint16_t>(
              (e & 1) ? (w[e >> 1] >> 16) : (w[e >> 1] & 0xFFFFu)));
      }
      uint4 out;
      uint32_t o[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const __nv_bfloat16 lo = __float2bfloat16(acc[2 * q]);
        const __nv_bfloat16 hi = __float2bfloat16(acc[2 * q + 1]);
        o[q] = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo)) |
               (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi))
                << 16);
      }
      out.x = o[0]; out.y = o[1]; out.z = o[2]; out.w = o[3];
      reinterpret_cast<uint4*>(dst)[vi] = out;
    }
    return;
  }
  for (uint32_t i = threadIdx.x; i < elems; i += kConsumerThreads) {
    float acc = 0.0f;
    for (int r = 0; r < world; ++r) {
      const uint16_t* vec = r == my_rank ? local : staged[r < my_rank ? r : r - 1];
      acc += bf16_to_f32(vec[i]);
    }
    dst[i] = __float2bfloat16(acc);
  }
}

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
  __shared__ uint64_t s_hash[kConsumerThreads / 32];  // per-warp fold partials
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
    const uint64_t total_hash = block_fold_payload(base, words, s_hash);

    if (threadIdx.x == 0) {
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
  __shared__ uint64_t s_hash[kConsumerThreads / 32];  // per-warp fold partials
  __shared__ uint64_t s_hash_want;   // the claimed door's placement-gate hash
  __shared__ int s_gate_matched;     // 1 once the payload folds to s_hash_want
  if (threadIdx.x == 0) ctl->gt_start = globaltimer_ns();
  __shared__ int s_round;            // claim records filled so far
  __shared__ int s_go;    // 0 none, >0 = flat cell index + 1
  __shared__ int s_stop;  // any exit condition
  __shared__ int s_failed;

  BlockRef go_ref(s_go);
  BlockRef stop_ref(s_stop);

  for (int p = 0; p < kBusMaxPeers; ++p) s_got[p] = 0;
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_failed = 0;
    s_round = 0;
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
  for (int p = 0; p < v.send_peers; ++p)
    block_copy_row(reinterpret_cast<const uint32_t*>(src),
                   reinterpret_cast<uint32_t*>(
                       const_cast<uint16_t*>(v.send_payload[p])),
                   words);
  // Each thread fences its own slot writes system-wide, then thread 0's
  // release publishes them all to the engine's acquire (the canonical
  // producer handoff; a lone barrier would only order CTA-scope).
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->stamp_stage = clock64();
    ctl->gt_stage = globaltimer_ns();
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
      // Generation gate: claim ONLY this collective's doorbells. Ranks
      // run unbarriered between collectives, so a neighbor's NEXT
      // collective can land its pair here while this kernel scans — a
      // blind claim would fold the wrong collective's payload into ours
      // and starve its own kernel (the M6 greedy-loop corruption; at
      // fabric skew, the gen-1825 wedge).
      if (sys_load_u32(&door->ctl) != ctl_seq) continue;
      const FlagAck* ack = &v.recv[view_idx].ack_lat[slot];
      if (seq == ack->seq) continue;  // already consumed
      // TEMP hunt stamp: the go-ref CAS winner records the FIRST claim's
      // (cell, len, seq) — one winner per round, so the triple is never
      // mixed across cells; atomicCAS's return gates first-only.
      int expected = 0;
      if (go_ref.compare_exchange_strong(expected, cell + 1,
                                         cuda::memory_order_relaxed,
                                         cuda::memory_order_relaxed)) {
        if (atomicCAS(reinterpret_cast<unsigned*>(&ctl->dbg_first_cell), 0,
                      static_cast<unsigned>(cell + 1)) == 0) {
          atomicExch(reinterpret_cast<unsigned*>(&ctl->dbg_first_len),
                      door->len);
          atomicExch(reinterpret_cast<unsigned*>(&ctl->dbg_first_seq), seq);
        }
        s_seq[peer] = seq;
        s_words[peer] = sys_load_u32(&door->len) / 8;
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
    // PLACEMENT GATE: the doorbell's DMA placement can become visible
    // before the payload's own placement (RC's CQE ordering protects the
    // engine, not this cell-polling kernel — the 2026-09-01 hunt: a
    // fresh, ctl-correct doorbell whose payload buffer still read
    // virgin/stale bytes). Spin until the payload folds to the door's
    // hash; a stale buffer folds to the previous generation's value and
    // cannot pass. The wait is bounded by the collective deadline (a
    // never-landing payload is a lost message — the failure paths own
    // that), and counted for the record: dbg_gate_waits > 0 in a
    // passing run is live proof the race fired and the gate held.
    const int cell = go - 1;
    const int view_idx = cell / lat_slots_per_view;
    const int slot = cell % lat_slots_per_view;
    const int peer = view_idx / v.lanes_per_peer;
    const BusRecvView& rv = v.recv[view_idx];
    const uint64_t* base = rv.payload_lat +
                           static_cast<size_t>(slot) * (rv.lat_slot_bytes / 8);
    const StartSlot* door = &v.recv[view_idx].doorbell_lat[slot];
    if (threadIdx.x == 0) {
      s_hash_want = sys_load_u64(&door->hash);
      s_gate_matched = 0;
      // The claim record for this round (the hunt's stall dump reads it):
      // the kernel's ACTUAL cell + its door's triple.
      const int rn = s_round < 3 ? s_round : 2;
      ctl->dbg_cl_cell[rn] = static_cast<uint32_t>(cell + 1);
      ctl->dbg_cl_len[rn] = sys_load_u32(&door->len);
      ctl->dbg_cl_seq[rn] = s_seq[peer];
      ctl->dbg_cl_hash[rn] = static_cast<uint32_t>(
          sys_load_u64(&door->hash) & 0xFFFFFFFFu);
      s_round = rn + 1;
    }
    __syncthreads();
    uint64_t total_hash = 0;
    for (int spin = 0;; ++spin) {
      total_hash = block_fold_payload(base, s_words[peer], s_hash);
      if (threadIdx.x == 0) {
        if (total_hash == s_hash_want) {
          s_gate_matched = 1;
        } else {
          // thread 0 is the gate's only writer; the engine reads after the
          // done stamp (release) — plain increments are ordered and cheap.
          if (spin == 0) ctl->dbg_gate_waits += 1;
          ctl->dbg_gate_spins += 1;
        }
      }
      __syncthreads();
      if (s_gate_matched != 0) break;
      if (stop_ref.load(cuda::memory_order_relaxed) != 0 ||
          clock64() - start > deadline_cycles) {
        s_failed = 1;
        stop_ref.store(1, cuda::memory_order_relaxed);
        break;
      }
      flag_poll_pause();
    }
    if (s_gate_matched == 0) continue;  // deadline: exit via the loop top
    __syncthreads();
    if (threadIdx.x == 0) {
      FlagAck* ack = &rv.ack_lat[slot];
      ack->cycles = clock64();
      ack->hash = total_hash;
      flag_store_release(&ack->seq, s_seq[peer]);
      s_payload[peer] = reinterpret_cast<const uint16_t*>(base);
      s_got[peer] = 1;
      const uint64_t now = globaltimer_ns();
      ctl->gt_claim[peer] = now;
      if (ctl->stamp_first_claim == 0) {
        ctl->stamp_first_claim = clock64();
        ctl->gt_first = now;
      }
      ctl->gt_last = now;
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
    block_fold_vectors(reinterpret_cast<const uint16_t*>(src), s_payload,
                       v.send_peers, my_rank, elems, dst);
    if (threadIdx.x == 0) ctl->stamp_reduce_done = clock64();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->gt_done = globaltimer_ns();
    ctl->status = s_failed == 0 ? 0 : 1;
    done_ref.store(ctl_seq, cuda::memory_order_release);
  }
}

// The graph kernel's staging budget: the decode collective's 2 x 4096
// bf16 payloads from three peers (48 KB); larger vectors (the 8-row
// batch) fold from the NIC-placed payloads as before.
constexpr size_t kGraphStageBytes = size_t{48} << 10;

// The graph twin of the kernel above (§6.2, the decode step's replayed
// launch sequence). The protocol is identical — snapshot, claim/fold/ack,
// exit stamp — with three deltas a shared-body refactor forced and the
// refactor MISCOMPILED (the fold read doorbell cells as payloads at -O3
// while the mechanical diff was pure renames; the eager kernel is kept
// verbatim — a validated machine is not a refactoring test bed):
//   * the generation is READ from the cell (u64 acquire against the arm
//     step's u64 release — width-matched), never baked: the recorded
//     launch parameters stay replay-stable while every replay advances
//     the generation (monotonic, so the previous replay's stale done_seq
//     can never match this execution);
//   * the staging rows derive from the generation — row
//     (gen-1)%stage_ring of each peer's ring — instead of the engine's
//     precomputed pointers;
//   * the cell is this node's per-generation cell (the eager kernel's
//     single ar_ctl is the one-flight case of the same layout).
__global__ __launch_bounds__(kConsumerThreads) void bus_allreduce_graph_kernel(
    BusAllReduceGraphView v, int my_rank, const __nv_bfloat16* src,
    __nv_bfloat16* dst, uint32_t elems, BusAllReduceCtl* ctl,
    uint64_t deadline_cycles, uint32_t stage_words) {
  // stage_words (2026-09-06): u64 words of dynamic shared memory per peer
  // (kBusMaxPeers slots) that a claimed payload is copied into by its
  // gate's hash pass; 0 = no staging (the fold reads the NIC-placed rows).
  using BlockRef = cuda::atomic_ref<int, cuda::thread_scope_block>;
  extern __shared__ __align__(16) uint64_t stage_smem[];
  __shared__ int s_staged[kBusMaxPeers];
  using SysRef = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;

  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> gen_ref(ctl->gen_seq);
  const uint32_t gen =
      static_cast<uint32_t>(gen_ref.load(cuda::memory_order_acquire));
  const uint32_t row_off =
      (gen - 1) % v.stage_ring * (v.stage_row_bytes / 2);
  if (threadIdx.x == 0) ctl->gt_start = globaltimer_ns();

  __shared__ int s_got[kBusMaxPeers];  // 0 waiting, 1 claimed+acked
  __shared__ uint32_t s_seq[kBusMaxPeers];
  __shared__ size_t s_words[kBusMaxPeers];
  __shared__ const uint16_t* s_payload[kBusMaxPeers];
  __shared__ uint64_t s_hash[kConsumerThreads / 32];  // per-warp fold partials
  __shared__ uint64_t s_hash_want;   // the claimed door's placement-gate hash
  __shared__ int s_gate_matched;     // 1 once the payload folds to s_hash_want
  __shared__ int s_round;            // claim records filled so far
  __shared__ int s_go;    // 0 none, >0 = flat cell index + 1
  __shared__ int s_stop;  // any exit condition
  __shared__ int s_failed;

  BlockRef go_ref(s_go);
  BlockRef stop_ref(s_stop);

  for (int p = 0; p < kBusMaxPeers; ++p) s_got[p] = 0;
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_failed = 0;
    s_round = 0;
  }
  __syncthreads();

  // Phase 1 — snapshot the source vector into the generation's shared
  // staging row (one copy; every peer's post is sourced from it — three
  // 8 KB writes into pinned memory were ~4 us of a 34 us collective).
  // Same contract as the eager kernel's phase 1: the fold overwrites src
  // in place, so the row holds a copy taken BEFORE the fold.
  const uint32_t words = elems / 2;
  block_copy_row(reinterpret_cast<const uint32_t*>(src),
                 reinterpret_cast<uint32_t*>(
                     const_cast<uint16_t*>(v.stage_row_base + row_off)),
                 words);
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->stamp_stage = clock64();
    ctl->gt_stage = globaltimer_ns();
    SysRef ready(ctl->ready_bits);
    uint64_t bits = 0;
    for (int p = 0; p < v.send_peers; ++p) bits |= 1ULL << p;
    ready.store(bits, cuda::memory_order_release);
  }

  // Phase 2 — wait for every peer's doorbell (identical to eager).
  const int lat_slots_per_view = v.recv_views > 0 ? v.recv[0].lat_slots : 0;
  const int total_cells = v.recv_views * lat_slots_per_view;
  const uint64_t start = clock64();
  SysRef done_ref(ctl->done_seq);
  for (;;) {
    if (threadIdx.x == 0) {
      go_ref.store(0, cuda::memory_order_relaxed);
      if (done_ref.load(cuda::memory_order_acquire) == gen ||
          clock64() - start > deadline_cycles) {
        stop_ref.store(1, cuda::memory_order_relaxed);
        s_failed = 1;
      }
    }
    __syncthreads();

    int missing = 0;
    for (int p = 0; p < v.send_peers; ++p) missing += (s_got[p] == 0);
    if (missing == 0 || stop_ref.load(cuda::memory_order_relaxed)) break;

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
      // Generation gate (see bus_allreduce_kernel's claim — the graph
      // kernel must not eat a neighbor replay's early doorbell either).
      if (sys_load_u32(&door->ctl) != gen) continue;
      const FlagAck* ack = &v.recv[view_idx].ack_lat[slot];
      if (seq == ack->seq) continue;  // already consumed
      int expected = 0;
      if (go_ref.compare_exchange_strong(expected, cell + 1,
                                         cuda::memory_order_relaxed,
                                         cuda::memory_order_relaxed)) {
        s_seq[peer] = seq;
        s_words[peer] = sys_load_u32(&door->len) / 8;
      }
    }
    __syncthreads();

    const int go = go_ref.load(cuda::memory_order_relaxed);
    if (go == 0) {
      flag_poll_pause();
      continue;
    }

    const int cell = go - 1;
    const int view_idx = cell / lat_slots_per_view;
    const int slot = cell % lat_slots_per_view;
    const int peer = view_idx / v.lanes_per_peer;
    const BusRecvView& rv = v.recv[view_idx];
    const uint64_t* base = rv.payload_lat +
                           static_cast<size_t>(slot) * (rv.lat_slot_bytes / 8);
    // PLACEMENT gate (identical to the eager kernel's — see there).
    const StartSlot* door = &v.recv[view_idx].doorbell_lat[slot];
    if (threadIdx.x == 0) {
      s_hash_want = sys_load_u64(&door->hash);
      s_gate_matched = 0;
      // The claim record for this round (the hunt's stall dump reads it):
      // the kernel's ACTUAL cell + its door's triple.
      const int rn = s_round < 3 ? s_round : 2;
      ctl->dbg_cl_cell[rn] = static_cast<uint32_t>(cell + 1);
      ctl->dbg_cl_len[rn] = sys_load_u32(&door->len);
      ctl->dbg_cl_seq[rn] = s_seq[peer];
      ctl->dbg_cl_hash[rn] = static_cast<uint32_t>(
          sys_load_u64(&door->hash) & 0xFFFFFFFFu);
      s_round = rn + 1;
    }
    __syncthreads();
    uint64_t total_hash = 0;
    uint64_t* stage = (stage_words != 0 && (stage_words & 1) == 0 &&
                       s_words[peer] <= stage_words)
                          ? stage_smem + static_cast<size_t>(peer) * stage_words
                          : nullptr;
    for (int spin = 0;; ++spin) {
      total_hash = block_fold_payload(base, s_words[peer], s_hash, stage);
      if (threadIdx.x == 0) {
        if (total_hash == s_hash_want) {
          s_gate_matched = 1;
        } else {
          // thread 0 is the gate's only writer; the engine reads after the
          // done stamp (release) — plain increments are ordered and cheap.
          if (spin == 0) ctl->dbg_gate_waits += 1;
          ctl->dbg_gate_spins += 1;
        }
      }
      __syncthreads();
      if (s_gate_matched != 0) break;
      if (stop_ref.load(cuda::memory_order_relaxed) != 0 ||
          clock64() - start > deadline_cycles) {
        s_failed = 1;
        stop_ref.store(1, cuda::memory_order_relaxed);
        break;
      }
      flag_poll_pause();
    }
    if (s_gate_matched == 0) continue;
    __syncthreads();
    if (threadIdx.x == 0) {
      FlagAck* ack = &rv.ack_lat[slot];
      ack->cycles = clock64();
      ack->hash = total_hash;
      flag_store_release(&ack->seq, s_seq[peer]);
      s_payload[peer] = reinterpret_cast<const uint16_t*>(base);
      s_staged[peer] = stage != nullptr ? 1 : 0;
      s_got[peer] = 1;
      const uint64_t now = globaltimer_ns();
      ctl->gt_claim[peer] = now;
      if (ctl->stamp_first_claim == 0) {
        ctl->stamp_first_claim = clock64();
        ctl->gt_first = now;
      }
      ctl->gt_last = now;
    }
    __syncthreads();
  }

  // Common exit + fold (identical to eager; the canonical chain).
  if (s_failed == 0) {
    // From the staged copies when every peer's landed in shared memory.
    bool all_staged = stage_words != 0;
    for (int p = 0; p < v.send_peers; ++p) all_staged = all_staged && s_staged[p] != 0;
    if (all_staged) {
      const uint16_t* staged[kBusMaxPeers];
      for (int p = 0; p < kBusMaxPeers; ++p)
        staged[p] = reinterpret_cast<const uint16_t*>(
            stage_smem + static_cast<size_t>(p) * stage_words);
      block_fold_vectors_staged(reinterpret_cast<const uint16_t*>(src), staged,
                                v.send_peers, my_rank, elems, dst);
    } else {
      block_fold_vectors(reinterpret_cast<const uint16_t*>(src), s_payload,
                         v.send_peers, my_rank, elems, dst);
    }
    if (threadIdx.x == 0) ctl->stamp_reduce_done = clock64();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ctl->gt_done = globaltimer_ns();
    ctl->status = s_failed == 0 ? 0 : 1;
    done_ref.store(gen, cuda::memory_order_release);
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

cudaError_t launch_bus_allreduce_graph(const BusAllReduceGraphView& v,
                                       int my_rank, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst, uint32_t elems,
                                       BusAllReduceCtl* cell,
                                       uint64_t deadline_cycles,
                                       cudaStream_t stream) {
  // Staged only when every peer's slot stays 16-byte aligned (elems a
  // multiple of 8: the step's small all-gathers are not) and the three
  // slots fit the budget.
  const size_t payload = static_cast<size_t>(elems) * 2;
  const bool stage = (elems % 8) == 0 && payload * kBusMaxPeers <= kGraphStageBytes;
  const uint32_t stage_words = stage ? elems / 4 : 0u;
  bus_allreduce_graph_kernel<<<1, kConsumerThreads,
                               stage ? payload * kBusMaxPeers : 0, stream>>>(
      v, my_rank, src, dst, elems, cell, deadline_cycles, stage_words);
  return cudaGetLastError();
}

// ---- segment-quantized bulk collective (DESIGN §6.3, prefill class) ----

namespace {

// ---- the bulk collective kernel: a cooperative grid ----------------------
// One launch per (phase, segment), kBulkBlocks blocks co-resident by the
// cooperative-launch contract (2026-09-05; the single-block kernel before
// it moved every byte through 256 threads at ~1.1 GB/s — a 2 MB fold cost
// 1.4 ms, a 16 MB one 16). The work that scales with bytes is spread over
// the whole grid in 16 KB tiles of 16-byte vectors: the outbound staging,
// the RS fold and the AG landing. What must be serial stays in block 0:
// the doorbell claim loop (the shared-CAS discipline) and the deferred ack
// flush. Grid barriers separate the phases: staging | claims | consume
// rounds | acks.
//
// THE PLACEMENT PROOF MOVES INTO THE CONSUME PASS. The claim loop no longer
// hashes a payload before recording it — every tile hashes the words it
// reads anyway (bus_fold64's word-indexed XOR, commutative, so tiles and
// blocks combine through an atomic XOR per (peer, stripe)) and block 0
// compares the stripe's total against the door's hash after the pass. A
// stripe whose payload was not yet fully placed when the fold read it is
// re-folded in the next round (the fold and the landing copy are pure
// functions of their inputs, so a redo overwrites the same dst bytes with
// the right ones); dbg_gate_waits/spins count those redos. The proof's
// invariant is unchanged: no ack, and no done stamp, before every consumed
// payload has folded to its door's hash.
constexpr int kBulkScanCap = kBusMaxBulkCells;  // protocol bound, start-validated
constexpr int kBulkClaimLog = 16;  // first claims of a failing kernel
constexpr int kBulkBlocks = kBusBulkBlocks;  // the cooperative grid's width
constexpr uint32_t kBulkTileVecs = 1024;  // 16-byte vectors per tile (16 KB)
constexpr uint32_t kBulkTileBytes = kBulkTileVecs * 16;

__device__ inline uint64_t bulk_mask_of(uint32_t n) {
  return n >= 64 ? ~0ULL : ((1ULL << n) - 1);
}

// 16-byte accesses at vector index vi of a u64-aligned buffer: one v4
// access when the base is 16-byte aligned, two u64 halves otherwise (the
// ring slots, arena rows and stripe bases are 16-aligned in every real
// geometry; the halves keep odd ones correct). System-scope loads for
// NIC-placed memory, plain ones for device buffers.
__device__ inline uint4 bulk_words_to_vec(uint64_t a, uint64_t b) {
  return make_uint4(static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32),
                    static_cast<uint32_t>(b), static_cast<uint32_t>(b >> 32));
}
__device__ inline uint4 bulk_load_sys(const uint16_t* base, uint32_t vi,
                                      bool aligned) {
  if (aligned) return sys_load_u128(reinterpret_cast<const uint4*>(base) + vi);
  const uint64_t* w =
      reinterpret_cast<const uint64_t*>(base) + 2 * static_cast<uint64_t>(vi);
  return bulk_words_to_vec(sys_load_u64(w), sys_load_u64(w + 1));
}
__device__ inline uint4 bulk_load_dev(const uint16_t* base, uint32_t vi,
                                      bool aligned) {
  if (aligned) return *(reinterpret_cast<const uint4*>(base) + vi);
  const uint64_t* w =
      reinterpret_cast<const uint64_t*>(base) + 2 * static_cast<uint64_t>(vi);
  return bulk_words_to_vec(w[0], w[1]);
}
__device__ inline void bulk_store(uint16_t* base, uint32_t vi, bool aligned,
                                  uint4 x) {
  if (aligned) {
    *(reinterpret_cast<uint4*>(base) + vi) = x;
    return;
  }
  uint64_t* w = reinterpret_cast<uint64_t*>(base) + 2 * static_cast<uint64_t>(vi);
  w[0] = static_cast<uint64_t>(x.x) | (static_cast<uint64_t>(x.y) << 32);
  w[1] = static_cast<uint64_t>(x.z) | (static_cast<uint64_t>(x.w) << 32);
}
// bus_fold64's contribution of one vector = words (i, i+1) of the payload.
__device__ inline uint64_t bulk_hash_vec(uint4 x, uint64_t i) {
  const uint64_t w0 = static_cast<uint64_t>(x.x) | (static_cast<uint64_t>(x.y) << 32);
  const uint64_t w1 = static_cast<uint64_t>(x.z) | (static_cast<uint64_t>(x.w) << 32);
  return ((w0 + i + 1) * kFoldMultiplier) ^ ((w1 + i + 2) * kFoldMultiplier);
}
__device__ inline uint16_t bulk_elem(const uint4& x, int e) {
  const uint32_t w = e < 2 ? x.x : e < 4 ? x.y : e < 6 ? x.z : x.w;
  return static_cast<uint16_t>((e & 1) ? (w >> 16) : (w & 0xffffu));
}
__device__ inline uint32_t bulk_pack2(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// The block's per-peer hash partials XORed into the stripe's accumulators
// (all threads; two barriers).
__device__ inline void bulk_commit_hashes(const uint64_t (&h)[kBusMaxPeersSized],
                                          int peers, uint64_t* s_h,
                                          BusBulkScratch* sc, uint32_t k) {
#pragma unroll
  for (int p = 0; p < kBusMaxPeersSized; ++p) {
    uint64_t x = h[p];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
    if ((threadIdx.x & 31) == 0)
      s_h[(threadIdx.x / 32) * kBusMaxPeersSized + p] = x;
  }
  __syncthreads();
  if (threadIdx.x < peers) {
    uint64_t t = 0;
#pragma unroll
    for (int w = 0; w < kConsumerThreads / 32; ++w)
      t ^= s_h[w * kBusMaxPeersSized + threadIdx.x];
    atomicXor(reinterpret_cast<unsigned long long*>(&sc->hash_acc[threadIdx.x][k]),
              static_cast<unsigned long long>(t));
  }
  __syncthreads();
}
// The block's XOR of every thread's h, returned to all threads (two
// barriers; s_h is free again on return).
__device__ inline uint64_t bulk_block_xor(uint64_t h, uint64_t* s_h) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if ((threadIdx.x & 31) == 0) s_h[threadIdx.x / 32] = h;
  __syncthreads();
  uint64_t t = 0;
#pragma unroll
  for (int w = 0; w < kConsumerThreads / 32; ++w) t ^= s_h[w];
  __syncthreads();
  return t;
}
__device__ inline void bulk_commit_hash(uint64_t h, uint64_t* s_h,
                                        uint64_t* acc) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
  if ((threadIdx.x & 31) == 0) s_h[threadIdx.x / 32] = h;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t t = 0;
#pragma unroll
    for (int w = 0; w < kConsumerThreads / 32; ++w) t ^= s_h[w];
    atomicXor(reinterpret_cast<unsigned long long*>(acc),
              static_cast<unsigned long long>(t));
  }
  __syncthreads();
}

__global__ __launch_bounds__(kConsumerThreads) void bus_bulk_collective_kernel(
    BusAllReduceView v, int my_rank, int phase,
    const __nv_bfloat16* src_bf, __nv_bfloat16* dst_bf, BusBulkSegPlan plan,
    uint64_t* staged_counters, uint64_t* staged_hashes, uint32_t ctl_seq,
    BusAllReduceCtl* ctl, uint64_t deadline_cycles, BusBulkScratch* sc) {
  using BlockRef = cuda::atomic_ref<int, cuda::thread_scope_block>;
  using SysRef = cuda::atomic_ref<uint64_t, cuda::thread_scope_system>;
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  const int G = static_cast<int>(gridDim.x);
  const int B = static_cast<int>(blockIdx.x);

  // Block 0's claim-loop state (the other blocks never touch it).
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
  // Bring-up records: the scan's own view of each cell (max door seq ever
  // returned by the acquire load; last ack seq it read) and the first
  // claims' mapping values. Written by each cell's exclusive scan owner /
  // thread 0, printed once at the failure exit.
  __shared__ uint32_t s_scan_max[kBulkScanCap];
  __shared__ uint32_t s_scan_ack[kBulkScanCap];
  // Ack deferral record: one bit per flat cell this kernel claimed. The
  // ack means "payload CONSUMED" — the consume pass after the claims, for
  // RS the fold and for AG the landing copy — so acks flush after it;
  // until then the sender's credit cannot return and the payload slots
  // stay put (the overwrite race the deferred ack exists to close).
  __shared__ uint8_t s_claimed[kBulkScanCap];
  __shared__ uint32_t s_cl_cell[kBulkClaimLog];
  __shared__ uint32_t s_cl_seq[kBulkClaimLog];
  __shared__ uint64_t s_cl_j[kBulkClaimLog];
  __shared__ uint64_t s_cl_base[kBulkClaimLog];
  __shared__ uint32_t s_cl_k[kBulkClaimLog];
  __shared__ uint32_t s_cl_acc[kBulkClaimLog];
  __shared__ uint32_t s_cl_n;
  // Every block: the tile hash partials and the verify pass's flags.
  __shared__ uint64_t s_h[(kConsumerThreads / 32) * kBusMaxPeersSized];
  __shared__ int s_mis[kBusMaxPeersSized * kBusMaxBulkSegStripes];

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
  auto tiles_of = [&](uint64_t bytes) -> uint32_t {
    return static_cast<uint32_t>((bytes + kBulkTileBytes - 1) / kBulkTileBytes);
  };

  BlockRef go_ref(s_go);
  BlockRef stop_ref(s_stop);
  if (threadIdx.x == 0) {
    s_stop = 0;
    s_failed = 0;
    s_total_arrived = 0;
    s_cl_n = 0;
  }
  for (int i = threadIdx.x; i < kBulkScanCap; i += kConsumerThreads) {
    s_scan_max[i] = 0;
    s_scan_ack[i] = 0;
    s_claimed[i] = 0;
  }
  if (B == 0) {
    // The grid's shared records start clean (block 0 owns them until the
    // claim barrier; every other block reads them only after it).
    for (int i = threadIdx.x; i < kBusMaxPeersSized * kBusMaxBulkSegStripes;
         i += kConsumerThreads)
      (&sc->hash_acc[0][0])[i] = 0;
    for (int i = threadIdx.x; i < kBusMaxBulkSegStripes; i += kConsumerThreads)
      sc->ready[i] = 0;
    if (threadIdx.x < kBusMaxPeersSized) {
      sc->agot[threadIdx.x] = 0;
      sc->pending_ag[threadIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
      sc->pending = 0;
      sc->failed = 0;
      sc->total_arrived = 0;
    }
  }
  __syncthreads();

  // ---- outbound staging into the arena rows (the whole grid) -------------
  // RS: every peer's sub-range, from my src. AG: my (reduced) sub-range,
  // from dst, to every peer. Tiles round-robin over the blocks; each tile
  // folds the words it copies (bus_fold64's word-indexed XOR) and the
  // block keeps a partial per stripe. After the grid barrier block 0
  // reduces the partials into the pinned hash table and then releases
  // the per-peer staged counters, which gate the engine's posting of that
  // row (stripe k lands at row + k*bulk_slot; the door carries the hash).
  {
    const uint16_t* from = phase == 0 ? src : dst;
    uint32_t t = 0;
    for (int p = 0; p < peers; ++p) {
      const uint32_t base = phase == 0 ? plan.out_base[p] : plan.my_base;
      const uint32_t count = phase == 0 ? plan.out_count[p] : plan.my_count;
      for (uint32_t k = 0; k < count; ++k) {
        const uint32_t gs = plan.seg_first + base + k;
        const uint64_t bytes = stripe_bytes(gs);
        const uint32_t nt = tiles_of(bytes);
        const uint16_t* s = from + static_cast<uint64_t>(gs) * stripe;
        uint16_t* d = const_cast<uint16_t*>(v.send_payload[p]) +
                      static_cast<uint64_t>(k) * stripe;
        const bool al = aligned16(s) && aligned16(d);
        const uint32_t nvec = static_cast<uint32_t>(bytes / 16);
        const uint32_t words = static_cast<uint32_t>(bytes / 8);
        uint64_t hb = 0;  // this block's partial of the stripe's fold
        for (uint32_t ti = 0; ti < nt; ++ti, ++t) {
          if (static_cast<int>(t % static_cast<uint32_t>(G)) != B) continue;
          const uint32_t v0 = ti * kBulkTileVecs;
          const uint32_t v1 = v0 + kBulkTileVecs < nvec ? v0 + kBulkTileVecs : nvec;
          uint64_t h = 0;
          for (uint32_t vi = v0 + threadIdx.x; vi < v1; vi += kConsumerThreads) {
            const uint4 x = bulk_load_dev(s, vi, al);
            bulk_store(d, vi, al, x);
            h ^= bulk_hash_vec(x, 2 * static_cast<uint64_t>(vi));
          }
          if (ti + 1 == nt) {
            // The tail past the last whole vector (bytes is even): an odd
            // hash word, then the elements.
            if (threadIdx.x == 0 && (words & 1u) != 0) {
              const uint64_t i = words - 1;
              const uint64_t w = reinterpret_cast<const uint64_t*>(s)[i];
              h ^= (w + i + 1) * kFoldMultiplier;
            }
            const uint32_t e0 = nvec * 8;
            const uint32_t e1 = static_cast<uint32_t>(bytes / 2);
            for (uint32_t e = e0 + threadIdx.x; e < e1; e += kConsumerThreads)
              d[e] = s[e];
          }
          hb ^= bulk_block_xor(h, s_h);
        }
        if (threadIdx.x == 0) sc->stage_part[B][p][k] = hb;
      }
    }
    __threadfence_system();
    grid.sync();
    if (B == 0) {
      for (int i = threadIdx.x; i < peers * kBusMaxBulkSegStripes;
           i += kConsumerThreads) {
        const int p = i / kBusMaxBulkSegStripes;
        const int k = i % kBusMaxBulkSegStripes;
        const uint32_t count = phase == 0 ? plan.out_count[p] : plan.my_count;
        if (static_cast<uint32_t>(k) >= count) continue;
        uint64_t h = 0;
        for (int b = 0; b < G; ++b) h ^= sc->stage_part[b][p][k];
        staged_hashes[p * kBusMaxBulkSegStripes + k] = h;
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence_system();
        for (int p = 0; p < peers; ++p) {
          const uint32_t count = phase == 0 ? plan.out_count[p] : plan.my_count;
          SysRef(staged_counters[p])
              .store(static_cast<uint64_t>(count), cuda::memory_order_release);
        }
      }
    }
  }

// Receive-side consumer for CollectiveBus validation (see bus_kernel.hpp).

  const uint64_t start = clock64();
  SysRef done_ref(ctl->done_seq);
  const int slots_per_view = v.recv_views > 0 ? v.recv[0].bulk_slots : 0;
  const int total_cells = v.recv_views * slots_per_view;
  __shared__ int s_expected_total;

  if (B == 0) {
    // ---- expected arrivals + per-lane bases ------------------------------
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
    if (threadIdx.x == 0) {
      int total = 0;
      for (int p = 0; p < peers; ++p) total += s_expected[p];
      s_expected_total = total;
    }
    __syncthreads();

    // ---- claim loop (block 0) ----------------------------------------------
    // One cell per round (the shared-CAS discipline); every claimed
    // arrival is mapped to its stripe and RECORDED — pointer, the door's
    // len and hash — for the grid's consume pass: the j-th arrival on a
    // lane sits in ring slot j%depth with seq j/depth+1 (RC in-order per
    // lane); the sender's round-robin striping (stripe k on lane
    // k%lanes) maps the in-segment index i to k=i*lanes+l.
    for (;;) {
      if (threadIdx.x == 0) {
        go_ref.store(0, cuda::memory_order_relaxed);
        // Engine poison (the request failed) or the cycle deadline —
        // either way this segment is over; exit through the common path.
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
        // Cross-collective generation gate (the segment window below only
        // separates THIS collective's phases; a NEXT collective's early
        // bulk pair would be claimed and acked away from its own kernel
        // just as eagerly — see the one-shot kernel's gate).
        if (sys_load_u32(&door->ctl) != ctl_seq) continue;
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
          s_len = sys_load_u32(&door->len);
        }
      }
      __syncthreads();
      const int go = go_ref.load(cuda::memory_order_relaxed);
      if (go == 0) {
        flag_poll_pause();
        continue;
      }

      // Claims are DEDUPED (the ack is deferred to the post-consume
      // flush, so a claimed cell re-presents every round until then, and
      // the re-claim must be a no-op — a double-counted arrival exits the
      // loop early and leaves a stripe's contribution missing (measured:
      // k=15 empty while 16 claims counted). The record carries the
      // door's hash; the consume pass proves the payload against it.
      if (threadIdx.x == 0) {
        const int cell = go - 1;
        const int view_idx = cell / slots_per_view;
        const int slot = cell % slots_per_view;
        const int lane = view_idx % lanes;
        const int peer = view_idx / lanes;
        const BusRecvView& rv = v.recv[view_idx];
        const StartSlot* door = &rv.doorbell_bulk[slot];
        const uint16_t* payload = reinterpret_cast<const uint16_t*>(
            rv.payload_bulk +
            static_cast<size_t>(slot) * (rv.bulk_slot_bytes / 8));
        const uint64_t j =
            (static_cast<uint64_t>(s_seq) - 1) * rv.bulk_slots + slot;
        const uint64_t in_seg = j - s_lane_base[view_idx];
        const uint32_t k = static_cast<uint32_t>(in_seg) *
                               static_cast<uint32_t>(lanes) +
                           lane;
        s_claimed[cell] = 1;  // ack deferred; the consume pass reads the slot
        const uint64_t bit = 1ULL << peer;
        bool counted = false;
        if (phase == 0) {
          // RS: the (stripe k, peer) contribution, once.
          if (k < plan.my_count && (sc->ready[k] & bit) == 0) {
            sc->ready[k] |= bit;
            counted = true;
          }
        } else {
          // AG: peer p's stripe k, once.
          if (k < plan.out_count[peer] && (sc->agot[peer] & (1ULL << k)) == 0) {
            sc->agot[peer] |= 1ULL << k;
            counted = true;
          }
        }
        if (counted) {
          sc->ptr[peer][k] = payload;
          sc->len[peer][k] = s_len;
          sc->want[peer][k] = sys_load_u64(&door->hash);
          ++s_total_arrived;
        }
        // Bring-up claim log (see BKFIN): the mapping values of the first
        // claims — base==0 or k past the guard here is a mapping
        // rejection, not scan blindness, and the two need different fixes.
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

    // Validation before the grid touches dst: every stripe of my
    // sub-range must hold all peers' contributions (RS) — a mapping bug
    // must fail loudly, never fold a partial sum (the silent-corruption
    // class). The consume plan — the stripes to fold or land — is
    // published with the failure flag.
    if (threadIdx.x == 0) {
      if (s_failed == 0 && phase == 0) {
        const uint64_t all_peers = bulk_mask_of(static_cast<uint32_t>(peers));
        for (uint32_t k = 0; k < plan.my_count; ++k)
          if (sc->ready[k] != all_peers) s_failed = 1;
      }
      sc->failed = s_failed;
      sc->total_arrived = s_total_arrived;
      if (s_failed == 0) {
        if (phase == 0) {
          sc->pending = bulk_mask_of(plan.my_count);
        } else {
          for (int p = 0; p < peers; ++p)
            sc->pending_ag[p] = bulk_mask_of(plan.out_count[p]);
        }
      }
      __threadfence();
    }
    __syncthreads();
  }
  grid.sync();

  // ---- consume pass (the whole grid): RS fold / AG landing ---------------
  // Tiles of the pending stripes round-robin over the blocks; every tile
  // hashes the payload words it reads, and block 0 proves each stripe
  // against its door's hash between rounds. RS: the canonical
  // ascending-rank fp32 chain per element, bitwise the one-shot kernel's
  // (the bit-reinterpret on the way out is load-bearing: __nv_bfloat16
  // converts to uint16_t through its float operator, never its bits).
  {
    int failed = sc->failed;
    uint64_t pend = sc->pending;
    uint64_t pend_ag[kBusMaxPeersSized];
#pragma unroll
    for (int p = 0; p < kBusMaxPeersSized; ++p) pend_ag[p] = sc->pending_ag[p];

    auto fold_tile = [&](uint32_t k, uint32_t ti, uint32_t nt) {
      const uint32_t gs = plan.seg_first + plan.my_base + k;
      const uint64_t bytes = stripe_bytes(gs);
      const uint16_t* local = src + static_cast<uint64_t>(gs) * stripe;
      uint16_t* out = dst + static_cast<uint64_t>(gs) * stripe;
      const uint16_t* pp[kBusMaxPeersSized];
      uint32_t words[kBusMaxPeersSized];
      bool al = aligned16(local) && aligned16(out);
#pragma unroll
      for (int p = 0; p < kBusMaxPeersSized; ++p) {
        pp[p] = p < peers ? sc->ptr[p][k] : nullptr;
        words[p] = p < peers ? sc->len[p][k] / 8 : 0;
        if (p < peers) al = al && aligned16(pp[p]);
      }
      const uint32_t nvec = static_cast<uint32_t>(bytes / 16);
      const uint32_t v0 = ti * kBulkTileVecs;
      const uint32_t v1 = v0 + kBulkTileVecs < nvec ? v0 + kBulkTileVecs : nvec;
      uint64_t h[kBusMaxPeersSized];
#pragma unroll
      for (int p = 0; p < kBusMaxPeersSized; ++p) h[p] = 0;
      for (uint32_t vi = v0 + threadIdx.x; vi < v1; vi += kConsumerThreads) {
        const uint4 mine = bulk_load_dev(local, vi, al);
        uint4 y[kBusMaxPeersSized];
#pragma unroll
        for (int p = 0; p < kBusMaxPeersSized; ++p) {
          if (p < peers) {
            y[p] = bulk_load_sys(pp[p], vi, al);
            if (vi < words[p] / 2)
              h[p] ^= bulk_hash_vec(y[p], 2 * static_cast<uint64_t>(vi));
          } else {
            y[p] = make_uint4(0, 0, 0, 0);
          }
        }
        uint16_t res[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          // Ascending rank order: peer slot p holds rank p below mine and
          // rank p+1 above it, so my own row enters at position my_rank.
          float acc = 0.0f;
          bool mine_in = false;
#pragma unroll
          for (int p = 0; p < kBusMaxPeersSized; ++p) {
            if (p == my_rank) {
              acc += bf16_to_f32(bulk_elem(mine, e));
              mine_in = true;
            }
            if (p < peers) acc += bf16_to_f32(bulk_elem(y[p], e));
          }
          if (!mine_in) acc += bf16_to_f32(bulk_elem(mine, e));
          res[e] = __bfloat16_as_ushort(__float2bfloat16(acc));
        }
        bulk_store(out, vi, al,
                   make_uint4(bulk_pack2(res[0], res[1]), bulk_pack2(res[2], res[3]),
                              bulk_pack2(res[4], res[5]), bulk_pack2(res[6], res[7])));
      }
      if (ti + 1 == nt) {
        // The tail: an odd hash word per peer, and the elements past the
        // last whole vector (folded, not hashed past the payload's words
        // — bus_fold64 walks len/8 words exactly as the door does).
        if (threadIdx.x == 0) {
#pragma unroll
          for (int p = 0; p < kBusMaxPeersSized; ++p) {
            if (p < peers && (words[p] & 1u) != 0) {
              const uint64_t i = words[p] - 1;
              const uint64_t w = sys_load_u64(
                  reinterpret_cast<const uint64_t*>(pp[p]) + i);
              h[p] ^= (w + i + 1) * kFoldMultiplier;
            }
          }
        }
        const uint32_t e0 = nvec * 8;
        const uint32_t e1 = static_cast<uint32_t>(bytes / 2);
        for (uint32_t e = e0 + threadIdx.x; e < e1; e += kConsumerThreads) {
          float acc = 0.0f;
          bool mine_in = false;
#pragma unroll
          for (int p = 0; p < kBusMaxPeersSized; ++p) {
            if (p == my_rank) {
              acc += bf16_to_f32(local[e]);
              mine_in = true;
            }
            if (p < peers) acc += bf16_to_f32(sys_load_u16(&pp[p][e]));
          }
          if (!mine_in) acc += bf16_to_f32(local[e]);
          out[e] = __bfloat16_as_ushort(__float2bfloat16(acc));
        }
      }
      bulk_commit_hashes(h, peers, s_h, sc, k);
    };

    auto land_tile = [&](int p, uint32_t k, uint32_t ti, uint32_t nt) {
      const uint32_t gs = plan.seg_first + plan.out_base[p] + k;
      const uint16_t* payload = sc->ptr[p][k];
      const uint32_t len = sc->len[p][k];
      uint16_t* out = dst + static_cast<uint64_t>(gs) * stripe;
      const bool al = aligned16(payload) && aligned16(out);
      const uint32_t words = len / 8;
      const uint32_t nvec = len / 16;
      const uint32_t v0 = ti * kBulkTileVecs;
      const uint32_t v1 = v0 + kBulkTileVecs < nvec ? v0 + kBulkTileVecs : nvec;
      uint64_t h = 0;
      for (uint32_t vi = v0 + threadIdx.x; vi < v1; vi += kConsumerThreads) {
        const uint4 y = bulk_load_sys(payload, vi, al);
        h ^= bulk_hash_vec(y, 2 * static_cast<uint64_t>(vi));
        bulk_store(out, vi, al, y);
      }
      if (ti + 1 == nt) {
        if (threadIdx.x == 0 && (words & 1u) != 0) {
          const uint64_t i = words - 1;
          const uint64_t w =
              sys_load_u64(reinterpret_cast<const uint64_t*>(payload) + i);
          h ^= (w + i + 1) * kFoldMultiplier;
        }
        const uint32_t e0 = nvec * 8;
        const uint32_t e1 = len / 2;
        for (uint32_t e = e0 + threadIdx.x; e < e1; e += kConsumerThreads)
          out[e] = sys_load_u16(&payload[e]);
      }
      bulk_commit_hash(h, s_h, &sc->hash_acc[p][k]);
    };

    for (int round = 0;; ++round) {
      if (failed == 0) {
        uint32_t t = 0;
        if (phase == 0) {
          for (uint32_t k = 0; k < plan.my_count; ++k) {
            if ((pend >> k & 1ULL) == 0) continue;
            const uint32_t nt = tiles_of(stripe_bytes(plan.seg_first + plan.my_base + k));
            for (uint32_t ti = 0; ti < nt; ++ti, ++t)
              if (static_cast<int>(t % static_cast<uint32_t>(G)) == B)
                fold_tile(k, ti, nt);
          }
        } else {
          for (int p = 0; p < peers; ++p) {
            for (uint32_t k = 0; k < plan.out_count[p]; ++k) {
              if ((pend_ag[p] >> k & 1ULL) == 0) continue;
              const uint32_t nt = tiles_of(sc->len[p][k]);
              for (uint32_t ti = 0; ti < nt; ++ti, ++t)
                if (static_cast<int>(t % static_cast<uint32_t>(G)) == B)
                  land_tile(p, k, ti, nt);
            }
          }
        }
      }
      __threadfence();
      grid.sync();

      if (B == 0) {
        // The proof: every consumed (peer, stripe) folded to its door's
        // hash. A miss means the fold read the slot before the NIC's
        // placement was fully visible (the door's placement can lead the
        // stripe's): re-fold that stripe next round, after a pause.
        if (failed == 0) {
          for (int i = threadIdx.x; i < kBusMaxPeersSized * kBusMaxBulkSegStripes;
               i += kConsumerThreads) {
            const int p = i / kBusMaxBulkSegStripes;
            const uint32_t k = static_cast<uint32_t>(i % kBusMaxBulkSegStripes);
            int mis = 0;
            if (p < peers) {
              const bool live = phase == 0 ? (k < plan.my_count && (pend >> k & 1ULL) != 0)
                                           : (k < plan.out_count[p] &&
                                              (pend_ag[p] >> k & 1ULL) != 0);
              if (live) mis = sc->hash_acc[p][k] != sc->want[p][k] ? 1 : 0;
            }
            s_mis[i] = mis;
          }
          __syncthreads();
          if (threadIdx.x == 0) {
            uint64_t next = 0;
            uint64_t next_ag[kBusMaxPeersSized] = {};
            int redo = 0;
            if (phase == 0) {
              for (uint32_t k = 0; k < plan.my_count; ++k) {
                if ((pend >> k & 1ULL) == 0) continue;
                bool bad = false;
                for (int p = 0; p < peers; ++p)
                  bad = bad || s_mis[p * kBusMaxBulkSegStripes + static_cast<int>(k)] != 0;
                if (bad) {
                  next |= 1ULL << k;
                  ++redo;
                  for (int p = 0; p < peers; ++p) sc->hash_acc[p][k] = 0;
                }
              }
            } else {
              for (int p = 0; p < peers; ++p) {
                for (uint32_t k = 0; k < plan.out_count[p]; ++k) {
                  if ((pend_ag[p] >> k & 1ULL) == 0) continue;
                  if (s_mis[p * kBusMaxBulkSegStripes + static_cast<int>(k)] != 0) {
                    next_ag[p] |= 1ULL << k;
                    ++redo;
                    sc->hash_acc[p][k] = 0;
                  }
                }
              }
            }
            if (redo != 0) {
              // thread 0 is the telemetry's only writer; the engine reads
              // after the done stamp (release).
              if (round == 0) ctl->dbg_gate_waits += static_cast<uint32_t>(redo);
              ctl->dbg_gate_spins += static_cast<uint32_t>(redo);
              if (clock64() - start > deadline_cycles) {
                s_failed = 1;
                next = 0;
                for (int p = 0; p < kBusMaxPeersSized; ++p) next_ag[p] = 0;
              }
            } else if (phase == 0) {
              ctl->stamp_reduce_done = clock64();
            }
            sc->failed = s_failed;
            sc->pending = next;
            for (int p = 0; p < kBusMaxPeersSized; ++p) sc->pending_ag[p] = next_ag[p];
            __threadfence();
          }
          __syncthreads();
        }
      }
      grid.sync();

      failed = sc->failed;
      pend = sc->pending;
      bool more = pend != 0;
#pragma unroll
      for (int p = 0; p < kBusMaxPeersSized; ++p) {
        pend_ag[p] = sc->pending_ag[p];
        more = more || pend_ag[p] != 0;
      }
      if (failed != 0 || !more) break;
      flag_poll_pause();
    }
  }

  if (B != 0) return;

  // ---- deferred ack flush (block 0) ---------------------------------------
  // The ack certifies the payload CONSUMED — the consume pass above, whose
  // last read of the ring slots the grid barrier orders before this
  // point. Acking at claim time would return the sender's credit while
  // this kernel still reads the slot, and the sender's next stripe would
  // DMA over a fold input (measured: mid-stripe prefix/suffix corruption
  // under cross-rank skew — a sender a segment ahead wraps the 8-deep
  // ring inside one claim-to-exit gap). The door cell cannot move before
  // its credit returns, so the claimed seq is re-read here rather than
  // carried per claim.
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
    const int failed = sc->failed;
    if (failed != 0 && s_total_arrived < s_expected_total) {
      printf("BKFIN rank=%d phase=%d seg=%u arrived=%d/%d my_base=%u "
             "my_count=%u t_ms=%llu poison=%d masks:",
             my_rank, phase, plan.seg_first, s_total_arrived,
             s_expected_total, plan.my_base, plan.my_count,
             (unsigned long long)((clock64() - start) / 1000),
             done_ref.load(cuda::memory_order_acquire) == ctl_seq ? 1 : 0);
      for (uint32_t k = 0; k < plan.my_count; ++k)
        printf(" k%u=%llx", k, (unsigned long long)sc->ready[k]);
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
    } else if (failed != 0) {
      printf("BKFIN rank=%d phase=%d seg=%u arrived=%d/%d: the consume pass "
             "did not prove every stripe before the deadline (gate waits %u, "
             "spins %u)\n",
             my_rank, phase, plan.seg_first, s_total_arrived, s_expected_total,
             ctl->dbg_gate_waits, ctl->dbg_gate_spins);
    }
    ctl->status = failed == 0 ? 0 : 1;
    done_ref.store(ctl_seq, cuda::memory_order_release);
  }
}

}  // namespace

cudaError_t launch_bus_bulk_collective(const BusAllReduceView& v, int my_rank,
                                       int phase, const __nv_bfloat16* src,
                                       __nv_bfloat16* dst,
                                       const BusBulkSegPlan& plan,
                                       uint64_t* staged_counters,
                                       uint64_t* staged_hashes,
                                       uint32_t ctl_seq, BusAllReduceCtl* ctl,
                                       uint64_t deadline_cycles,
                                       BusBulkScratch* scratch,
                                       cudaStream_t stream) {
  // The cooperative grid: kBulkBlocks, or fewer if the device cannot hold
  // that many co-resident (the launch contract the grid barriers rest
  // on; a cooperative launch that cannot co-schedule fails instead of
  // deadlocking). Computed once per process — one device per process.
  static const int blocks = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess || sms < 1)
      return 1;
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, bus_bulk_collective_kernel, kConsumerThreads, 0) !=
            cudaSuccess || per_sm < 1)
      return 1;
    const int cap = sms * per_sm;
    return cap < kBulkBlocks ? cap : kBulkBlocks;
  }();
  BusAllReduceView view = v;
  BusBulkSegPlan seg = plan;
  void* args[] = {&view,           &my_rank, &phase,          &src,
                  &dst,            &seg,     &staged_counters, &staged_hashes,
                  &ctl_seq,        &ctl,     &deadline_cycles, &scratch};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(bus_bulk_collective_kernel),
      dim3(static_cast<unsigned>(blocks)), dim3(kConsumerThreads), args, 0,
      stream);
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

__global__ void bus_globaltimer_stamp_kernel(uint64_t* out) {
  *out = globaltimer_ns();
}

cudaError_t bus_globaltimer_offset(int64_t* offset_ns, uint64_t* uncertainty_ns) {
  // %globaltimer runs on its own base (measured ~40 s off CLOCK_MONOTONIC
  // on the GB10 driver), so host and device stamps are comparable only
  // through this offset: one stamp kernel bracketed by two host reads; the
  // GPU read happened somewhere in between, so offset = gt - midpoint,
  // uncertain by half the bracket (~10 us, launch + sync — fine for the
  // millisecond stalls the timelines hunt).
  uint64_t* d = nullptr;
  cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&d), sizeof(uint64_t));
  if (err != cudaSuccess) return err;
  int64_t best_offset = 0;
  uint64_t best_uncertainty = ~0ull;
  for (int i = 0; i < 8; ++i) {  // the first launches carry warm-up; keep the tightest bracket
    timespec t0{}, t1{};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    bus_globaltimer_stamp_kernel<<<1, 1>>>(d);
    err = cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (err != cudaSuccess) break;
    const uint64_t h0 = static_cast<uint64_t>(t0.tv_sec) * 1000000000ull + t0.tv_nsec;
    const uint64_t h1 = static_cast<uint64_t>(t1.tv_sec) * 1000000000ull + t1.tv_nsec;
    const uint64_t half = (h1 - h0) / 2;
    if (half < best_uncertainty) {
      best_uncertainty = half;
      best_offset = static_cast<int64_t>(*d) - static_cast<int64_t>(h0 + half);
    }
  }
  cudaFreeHost(d);
  if (err != cudaSuccess) return err;
  *offset_ns = best_offset;
  *uncertainty_ns = best_uncertainty;
  return cudaSuccess;
}

cudaError_t bus_preload_kernels() {
  // cudaFuncGetAttributes forces the lazy loader to materialize a kernel
  // NOW, while nothing spins on the device — see the header for why that
  // matters (a kernel's first launch otherwise waits for an idle device,
  // and a peer's collective kernel spinning on OUR doorbell is never idle).
  cudaFuncAttributes attr{};
  const void* kernels[] = {
      reinterpret_cast<const void*>(bus_consumer_kernel),
      reinterpret_cast<const void*>(bus_allreduce_kernel),
      reinterpret_cast<const void*>(bus_allreduce_graph_kernel),
      reinterpret_cast<const void*>(bus_bulk_collective_kernel),
      reinterpret_cast<const void*>(bus_warm_work_kernel),
      reinterpret_cast<const void*>(bus_globaltimer_stamp_kernel),
  };
  for (const void* k : kernels) {
    const cudaError_t err = cudaFuncGetAttributes(&attr, k);
    if (err != cudaSuccess) return err;
  }
  // The graph kernel's staged fold (kGraphStageBytes of dynamic smem).
  if (const cudaError_t err = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(bus_allreduce_graph_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kGraphStageBytes));
      err != cudaSuccess)
    return err;
  return cudaSuccess;
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
