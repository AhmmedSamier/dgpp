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

cudaError_t launch_bus_consumer(const BusRecvView& view,
                                uint64_t deadline_cycles, cudaStream_t stream) {
  bus_consumer_kernel<<<1, kConsumerThreads, 0, stream>>>(view,
                                                         deadline_cycles);
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
