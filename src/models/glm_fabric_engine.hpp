#pragma once
// The fabric engine closure for serving (Stage 4b) and glm_gen_check's
// scheduler path — ONE seam, two apps, no drift. Real-mesh BusOptions
// (the budgets the first fabric gate run found, 2026-08-30) + the
// distributed greedy pick over the vocab-sharded head.
//
// CUDA-app-only header: it drags the bus (verbs) headers. Host gates
// fake the engine instead; nothing in dgpp_service includes this.
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "common/process_memory.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_loader.hpp"
#include "models/glm_step_timing.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

// Pins the process's current pages BEFORE the model is constructed — the
// decode loop's host state (tokenizer tables, the bus, this binary) once
// had to survive a load phase that drove every box to its memory
// watermark and had the kernel swapping exactly those pages out
// (2026-09-02: a ~10 ms swap-in fault per shared token, in lockstep across
// ranks). The one-pass loader removed that pressure, and a 1000-step run
// with the pin OFF was indistinguishable (p99 46, 0 stalls, no swap
// traffic) — so this is a belt-and-braces safety net, NOT a requirement:
// a box whose RLIMIT_MEMLOCK refuses it serves exactly as well, and the
// refusal is informational. Before construction on purpose: the loader's
// checkpoint mmaps do not exist yet, so MCL_CURRENT cannot try to pin
// them. DGPP_MLOCK=off skips the attempt.
inline void pin_serving_process(int rank) {
  if (const char* m = std::getenv("DGPP_MLOCK"); m && std::string(m) == "off") {
    DGPP_LOG_INFO("rank {}: process memory not locked (DGPP_MLOCK=off)", rank);
    return;
  }
  std::string why;
  size_t locked = 0;
  if (lock_process_memory(&why, &locked))
    DGPP_LOG_INFO("rank {}: process memory locked (mlockall MCL_CURRENT, "
                  "{:.0f} MiB)",
                  rank, static_cast<double>(locked) / (1024.0 * 1024.0));
  else
    DGPP_LOG_INFO("rank {}: process memory not locked ({}); serving "
                  "proceeds — the pin is optional",
                  rank, why);
}

// The resident image cache's location (GlmLayerStream::set_resident_image_dir):
//   DGPP_RESIDENT_CACHE=off        disabled
//   DGPP_RESIDENT_CACHE_DIR=DIR    explicit directory
//   otherwise                      $XDG_CACHE_HOME/dgpp/resident, or
//                                  $HOME/.cache/dgpp/resident
// On by default on purpose: a serving box's disk exists to make the next
// start fast, and the key (checkpoint headers + config + world/rank +
// format version) makes a stale image impossible to load by accident.
inline void configure_resident_image_cache(int rank) {
  std::string dir;
  if (const char* mode = std::getenv("DGPP_RESIDENT_CACHE");
      mode && std::string(mode) == "off") {
    DGPP_LOG_INFO("rank {}: resident image cache off (DGPP_RESIDENT_CACHE)",
                  rank);
  } else if (const char* d = std::getenv("DGPP_RESIDENT_CACHE_DIR"); d && *d) {
    dir = d;
  } else if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
    dir = std::string(x) + "/dgpp/resident";
  } else if (const char* h = std::getenv("HOME"); h && *h) {
    dir = std::string(h) + "/.cache/dgpp/resident";
  }
  GlmLayerStream::set_resident_image_dir(dir);
  if (!dir.empty())
    DGPP_LOG_INFO("rank {}: resident image cache at {}", rank, dir);
}

// Everything a serving process does before it constructs its model.
inline void prepare_serving_process(int rank) {
  pin_serving_process(rank);
  configure_resident_image_cache(rank);
}

// Real-mesh budgets, not loopback budgets (found by the first fabric
// gate run, 2026-08-30: a cold peer's first boundary waits behind
// seconds of cold NVMe weight streaming). The lane watchdog arms from
// each POST, so these bounds measure genuine in-flight stalls only.
// launch_consumers=false: the app drives every collective itself.
inline net::BusOptions fabric_bus_options(int rank, int world, uint16_t port,
                                          const std::string& peer,
                                          int rendezvous_timeout_ms) {
  net::BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.lane_devices = {"rocep1s0f0", "roceP2p1s0f0"};
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : peer;
  o.rendezvous_timeout_ms = rendezvous_timeout_ms;
  o.lat_slots = 8;
  o.lat_slot_bytes = 8192;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 120000;
  o.consumer_deadline_s = 60.0;
  o.launch_consumers = false;
  return o;
}

// The fabric pick: this rank's vocab-slice argmax, then the winner
// through bus_greedy_pick — the collective every rank joins in the
// same tick, with the readback invariant that makes a corrupt
// broadcast LOUD on the exact rank. The float row is hoisted inside
// the returned closure (no per-token device-adjacent allocation; the
// pick scratch is caller-pinned BEFORE the world forms).
inline GenEngineAdapter::Pick make_fabric_pick(net::CollectiveBus* bus,
                                               int rank, int world,
                                               uint16_t* pick_scratch,
                                               int64_t vocab,
                                               int pick_timeout_ms = 60000) {
  return [bus, rank, world, pick_scratch, vocab, pick_timeout_ms,
          row = std::vector<float>()](
             const GlmDiagnosticModel::Outputs& out) mutable -> int32_t {
    row.resize(static_cast<size_t>(out.lm_vocab_count));
    for (int i = 0; i < out.lm_vocab_count; ++i)
      row[static_cast<size_t>(i)] =
          bf16_bits_to_float(out.logits_bits[static_cast<size_t>(i)]);
    const glm_sample::Candidate local = glm_sample::local_max(
        row.data(), static_cast<int>(out.lm_vocab_count),
        out.lm_vocab_begin);
    const int32_t t = bus_greedy_pick(*bus, rank, world, local,
                                      pick_scratch, pick_timeout_ms);
    if (t < 0 || t >= vocab)
      throw std::runtime_error("fabric pick out of range: " +
                               std::to_string(t));
    return t;
  };
}

}  // namespace dgpp
