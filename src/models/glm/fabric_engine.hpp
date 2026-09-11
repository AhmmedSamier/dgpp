#pragma once
// GLM's face of the engine: the engine adapters live in
// engine/graph_engine.hpp as templates over the model type; this header
// binds them to GlmDiagnosticModel under the names the apps and gates use,
// and keeps the GLM-specific process preparation (the resident image cache
// is the GLM loader's).
#include <cstdlib>
#include <string>

#include "common/log.hpp"
#include "engine/graph_engine.hpp"
#include "models/glm/forward.hpp"
#include "models/glm/gen_engine.hpp"
#include "models/glm/loader.hpp"
#include "models/glm/speculative.hpp"
#include "models/glm/tp_bus.hpp"

namespace dgpp {

using GlmGraphEngineAdapter = GraphEngineAdapter<GlmDiagnosticModel>;

// GLM's latency slot: kDecodeRows bf16 rows of hidden 4096 (64 KiB).
inline net::BusOptions fabric_bus_options(int rank, int world, uint16_t port,
                                          const std::string& peer,
                                          int rendezvous_timeout_ms) {
  return fabric_bus_options(rank, world, port, peer, rendezvous_timeout_ms,
                            static_cast<size_t>(kDecodeRows) * 4096 * 2);
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

}  // namespace dgpp
