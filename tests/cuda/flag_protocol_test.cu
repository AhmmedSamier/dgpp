// Regression tests for the pinned-memory flag protocol (flag_protocol.cuh).
// These pin the ordering contract CollectiveBus will depend on: payload
// released before a flag must be fully visible when the flag is observed,
// and lost wakeups must degrade to detectable timeouts, never a wedged
// device. A driver update silently breaking fence semantics should fail
// here before it fails in production.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "common/test.hpp"
#include "kernels/flag_protocol.cuh"

namespace {

using dgpp::FlagAck;

constexpr int kMsgs = 256;  // fast enough for CI, long enough to be meaningful
// Watchdog budget for kernels under test. Cycles are clock-dependent, so the
// tests use generous time margins (GB10 clocks keep this well under 1 s).
constexpr uint64_t kDeadlineCycles = 800'000'000ull;
// Host-side per-message bound; must comfortably exceed the kernel deadline.
constexpr double kHostTimeoutSec = 6.0;
// Quiet-period length when testing the watchdog: must exceed the deadline
// at any plausible GB10 clock (800M cycles @ 267 MHz would be 3 s).
constexpr auto kQuietPeriod = std::chrono::milliseconds{3000};

#if defined(__aarch64__)
inline void cpu_pause() { asm volatile("yield" ::: "memory"); }
#else
inline void cpu_pause() {}
#endif

struct Fixture {
  dgpp::StartSlot* start = nullptr;
  FlagAck* ack = nullptr;
  uint32_t* payload = nullptr;  // one 64 B line: 16 u32 words
  cudaStream_t stream = nullptr;

  Fixture() {
    cudaHostAlloc(&start, sizeof(dgpp::StartSlot), cudaHostAllocDefault);
    cudaHostAlloc(&ack, sizeof(FlagAck), cudaHostAllocDefault);
    cudaHostAlloc(&payload, 16 * sizeof(uint32_t), cudaHostAllocDefault);
    if (!start || !ack || !payload) throw std::runtime_error("pinned alloc");
    cudaStreamCreate(&stream);
  }
  ~Fixture() {
    cudaStreamDestroy(stream);
    cudaFreeHost(payload);
    cudaFreeHost(ack);
    cudaFreeHost(start);
  }
};

// Host-side mirror of the device checksum in flag_payload_kernel. MUST stay
// in lockstep with the fold documented there.
uint64_t host_fold(const uint32_t* p) {
  uint64_t h = 0;
  for (int i = 0; i < 16; i += 2) {
    const uint64_t pair = (static_cast<uint64_t>(p[i]) << 32) |
                          static_cast<uint64_t>(p[i + 1]);
    h ^= pair * 0x9E3779B97F4A7C15ull;
  }
  return h;
}

// Waits until ack.seq == want or times out; returns true on success.
bool wait_ack(const FlagAck& ack, uint32_t want) {
  const auto t0 = std::chrono::steady_clock::now();
  while (__atomic_load_n(&ack.seq, __ATOMIC_ACQUIRE) != want) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count() > kHostTimeoutSec)
      return false;
    cpu_pause();
  }
  return true;
}

}  // namespace

DGPP_TEST(flag_protocol_with_sequential_writes_delivers_every_ack_in_order) {
  // GIVEN a heartbeat kernel spinning on a pinned start slot:
  Fixture f;
  flag_heartbeat_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack,
                                               kDeadlineCycles);

  // WHEN we issue kMsgs monotonically increasing sequences and wait for the
  // matching ack of each:
  uint64_t last_cycles = 0;
  for (uint32_t i = 1; i <= kMsgs; ++i) {
    __atomic_store_n(&f.start->seq, i, __ATOMIC_RELEASE);
    if (!wait_ack(*f.ack, i))
      throw std::runtime_error("ack for seq not observed in order");
    // THEN every stamp is present and non-decreasing (protocol sanity):
    const uint64_t c = f.ack->cycles;
    if (c == 0) throw std::runtime_error("ack cycles missing");
    if (c < last_cycles) throw std::runtime_error("device stamps regressed");
    last_cycles = c;
  }
}

DGPP_TEST(flag_protocol_with_payload_released_before_flag_shows_payload_at_ack) {
  // GIVEN the payload variant of the protocol kernel:
  Fixture f;
  flag_payload_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.payload, f.ack,
                                             kDeadlineCycles);

  // WHEN each message writes a fresh 64 B payload BEFORE releasing the flag:
  for (uint32_t i = 1; i <= kMsgs; ++i) {
    for (int w = 0; w < 16; ++w)
      f.payload[w] = i * 2654435761u + static_cast<uint32_t>(w);
    std::atomic_thread_fence(std::memory_order_release);
    __atomic_store_n(&f.start->seq, i, __ATOMIC_RELEASE);

    // THEN the acked checksum must equal the host's independent fold —
    // proving payload-before-flag visibility end to end:
    if (!wait_ack(*f.ack, i)) throw std::runtime_error("payload ack missing");
    if (f.ack->hash != host_fold(f.payload))
      throw std::runtime_error("payload not fully visible when ack observed");
  }
}

DGPP_TEST(flag_protocol_with_dead_wakeup_times_out_and_device_remains_usable) {
  // GIVEN a heartbeat kernel whose watchdog expires after a quiet period:
  Fixture f;
  flag_heartbeat_kernel<<<1, 1, 0, f.stream>>>(&f.start->seq, f.ack,
                                               kDeadlineCycles);
  __atomic_store_n(&f.start->seq, 1, __ATOMIC_RELEASE);
  if (!wait_ack(*f.ack, 1)) throw std::runtime_error("healthy ack missing");

  // WHEN traffic stops well beyond the watchdog budget and a new sequence
  // is sent to the (now dead) kernel:
  std::this_thread::sleep_for(kQuietPeriod);
  __atomic_store_n(&f.start->seq, 2, __ATOMIC_RELEASE);

  // THEN the stream drains (kernel exited) and seq 2 is never acked —
  // detection, not an infinite hang:
  const cudaError_t err = cudaStreamSynchronize(f.stream);
  if (err != cudaSuccess)
    throw std::runtime_error("stream did not drain after watchdog exit");
  if (__atomic_load_n(&f.ack->seq, __ATOMIC_ACQUIRE) != 1)
    throw std::runtime_error("kernel failed to watchdog-exit");

  // AND the device remains fully usable afterwards:
  float* probe = nullptr;
  cudaMalloc(&probe, 1 << 20);
  cudaMemsetAsync(probe, 0x3F, 1 << 20, f.stream);
  cudaStreamSynchronize(f.stream);
  std::vector<char> back(1 << 20);
  cudaMemcpy(back.data(), probe, back.size(), cudaMemcpyDeviceToHost);
  if (back[123] != 0x3F || back.back() != 0x3F)
    throw std::runtime_error("device unusable after watchdog exit");
  cudaFree(probe);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
