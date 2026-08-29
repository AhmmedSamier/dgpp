#pragma once

// CollectiveBus: the M5 data plane (DESIGN §6). One bus per process owns
// every peer pair's RC QPs on every configured lane, the single-threaded
// engine loop that polls them (§6.1), per-class slot pools, and the
// watchdog-bounded receive consumers.
//
// Concurrency contract (§6.1): the engine thread owns all verbs objects,
// slabs, and slot/credit state. Other threads touch the bus only through
// send()/wait()/stop() — submissions go to per-class queues drained in
// priority order with bounded bulk posts, so concurrent latency-under-bulk
// traffic is a supported, tested mode. The class slot pools make the credit
// floors structural: bulk can never occupy a latency slot.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "net/bus_kernel.hpp"

namespace dgpp::net {

enum class BusMessageClass { kLatency, kBulk };

struct BusOptions {
  int world_size = 2;  // TP group size; ranks are 0..world_size-1
  int my_rank = 0;
  // Lane devices in lane order; both active f0 functions by default.
  // Lane counts and order must match on every rank.
  std::vector<std::string> lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};

  // Rendezvous: rank 0 listens on `rendezvous_port`; every other rank
  // connects to `rendezvous_host:rendezvous_port` (roster precedent: the
  // coordinator owns admission, then distributes the full endpoint table).
  uint16_t rendezvous_port = 29600;
  std::string rendezvous_host;  // required when my_rank != 0
  int rendezvous_timeout_ms = 20000;

  // Per-lane slot pools. Latency pool: decode-class messages, one slot per
  // message, no striping. Bulk pool: striped round-robin across lanes.
  int lat_slots = 32;
  size_t lat_slot_bytes = 8192;  // 4096 hidden x BF16, the decode unit
  int bulk_slots = 16;
  size_t bulk_slot_bytes = 262144;

  int qp_depth = 1024;
  int completion_timeout_ms = 5000;  // engine watchdog per request
  double consumer_deadline_s = 30.0;  // receive-kernel inactivity bound
  bool launch_consumers = true;       // Phase 2 harness consumers; the
                                      // engine's real per-collective
                                      // consumers (deliverable 3) replace
                                      // them by disabling this.
};

// Per-lane traffic counters (one entry per peer x lane, peers ascending).
struct BusLaneStats {
  int peer_rank = -1;
  int lane = -1;
  uint64_t posts = 0;             // payload+doorbell pairs posted
  uint64_t bytes_sent = 0;
  uint64_t doorbell_recvs = 0;     // inbound messages arrived
  uint64_t bytes_recv = 0;
  uint64_t credits_returned = 0;   // slots recycled + credit WRITEs posted
  uint64_t credits_received = 0;   // inbound credits observed
  bool failed = false;
};

struct BusClassStats {
  uint64_t messages = 0;
  uint64_t bytes = 0;
  // Completion latency in microseconds, submit-to-credit, capped sample.
  std::vector<double> latency_us;
};

struct BusStats {
  std::vector<BusLaneStats> lanes;
  BusClassStats latency;
  BusClassStats bulk;
};

struct BusSendResult {
  bool ok = false;
  std::string error;
  // Per-stripe consumer hashes, stripe order. The sender can verify each
  // against fold() of the corresponding contiguous chunk (latency: one
  // stripe = whole message; bulk: slot-byte chunks striped round-robin).
  std::vector<uint64_t> stripe_hashes;
  double elapsed_us = 0.0;
};

struct BusAllReduceResult {
  bool ok = false;
  std::string error;
  double elapsed_us = 0.0;
};

class CollectiveBus {
 public:
  explicit CollectiveBus(BusOptions options);
  ~CollectiveBus();

  CollectiveBus(const CollectiveBus&) = delete;
  CollectiveBus& operator=(const CollectiveBus&) = delete;

  // Rendezvous (listener = rank 0), QP bring-up, initial credits posted,
  // receive consumers launched (unless disabled), engine thread started.
  // Returns false with *error on any failure; the bus is then stopped.
  bool start(std::string* error);

  // Submits a message to one peer. `data` must stay valid until wait()
  // returns. Latency class: bytes <= lat_slot_bytes. Bulk class: any size,
  // striped in bulk_slot_bytes chunks. bytes must be a multiple of 8.
  // Returns an id for wait(), or 0 with *error set.
  uint64_t send(int peer_rank, const void* data, size_t bytes,
                BusMessageClass cls, std::string* error);

  // Blocks for the request's completion. The engine enforces the watchdog;
  // timeout_ms is a generous backstop. Removes the request record.
  BusSendResult wait(uint64_t send_id, int timeout_ms);

  // One-shot all-to-all all-reduce over the whole world (DESIGN §6.3),
  // latency pool. `device_src`/`device_dst` are device pointers to
  // `bf16_elems` bf16 elements (multiple of 2, at most lat_slot_bytes/2);
  // they may alias. bf16 loads, fp32 accumulation in rank order —
  // deterministic, so a host oracle of the same chain matches bitwise.
  //
  // Contract (v1): launch_consumers=false (the per-collective kernel owns
  // doorbell claims), at most one outstanding collective (decode is
  // dependency-serialized), and no other latency traffic in flight. After
  // any collective failure the bus rejects further collectives — inbound
  // generations can no longer be trusted to line up.
  uint64_t allreduce(const void* device_src, void* device_dst,
                     size_t bf16_elems, std::string* error);

  // Blocks for the collective; on success the destination is stream-ordered
  // for the caller (one cudaStreamSynchronize after the engine's
  // completion). timeout_ms is a backstop; the engine watchdog owns the
  // lifecycle. Removes the request record.
  BusAllReduceResult wait_allreduce(uint64_t id, int timeout_ms);

  // Phase 1 of an orderly stop: rejects new submissions, joins the engine
  // (draining outstanding requests), stops the receive consumers via the
  // control cell, and destroys their stream. Frees nothing — cudaFreeHost
  // synchronizes the device implicitly, so in-process loopback pairs must
  // quiesce BOTH sides before either stops (one bus per process never
  // notices; stop() alone is correct there).
  void quiesce();

  // quiesce() plus waiter release and full teardown (QP/MR/slab frees).
  // Also invoked by the destructor.
  void stop();

  BusStats stats() const;
  int lane_count() const { return static_cast<int>(options_.lane_devices.size()); }
  size_t slot_bytes(BusMessageClass cls) const {
    return cls == BusMessageClass::kLatency ? options_.lat_slot_bytes
                                            : options_.bulk_slot_bytes;
  }

  // Receive-side view of one peer/lane (doorbells, acks, payloads, control).
  BusRecvView recv_view(int peer_rank, int lane) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  BusOptions options_;
};

}  // namespace dgpp::net
