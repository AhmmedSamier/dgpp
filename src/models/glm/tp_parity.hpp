#pragma once
// M5 d4: sharded load vs full-load+bind, pinned bitwise — the shared
// driver behind both the CI shard-parity test (fixture) and the
// glm_shard_parity app (real checkpoint). The sharded GlmLayerStream
// builds each resident layer directly at the rank's local geometry (only
// rank-local checkpoint bytes ever read); GlmTpViews::bind on a full
// resident is the independent reference implementation of the same
// slicing spec (§5.2). This check is what keeps the two from drifting:
// every bound surface of every layer must match byte-for-byte, at every
// world the geometry validator accepts.
//
// Plus the §5.2 boot checks, as arithmetic:
//   * the replicated digest is rank-invariant (and equals the world=1
//     pass — same files, same replicated set);
//   * byte reconcile: sum_r(source bytes read) == world1 total +
//     (world-1) * verbatim, with verbatim (the replicated + DSA-bridge
//     re-read set) identical across ranks. Sharded bytes partition
//     across ranks exactly once — a double-owned or missing row breaks
//     the identity. With a layer subset the identity still holds: every
//     term is additive per loaded layer (and per the globals).
//
// Mismatches throw with a layer- and surface-tagged message — a failure
// names the exact surface, not "differs somewhere".
#include <cstdint>
#include <string>
#include <vector>

#include "models/glm/config.hpp"

namespace dgpp {

struct GlmShardParityReport {
  int world = 0;
  int layers_checked = 0;   // layers compared (per rank)
  int surfaces_checked = 0;  // bound surfaces compared bitwise
  uint64_t full_source_bytes = 0;    // world=1 total (the reconcile base)
  uint64_t shard_source_bytes = 0;    // one rank's total
  uint64_t verbatim_bytes = 0;       // the rank-invariant re-read set
  uint64_t digest_bytes = 0;         // replicated bytes folded into digests
  uint64_t digest_tensors = 0;
  // Resident mode only: cache-hit re-loads served after the parity loop
  // (layers x ranks), each proven pointer-identical with ZERO storage
  // reads — the residency contract's proof, folded into the parity run.
  int cache_hits = 0;
};

// Runs the parity surface at `world` over `layers` (layer indices in
// 0..num_hidden_layers-1 plus mtp_layer() when present; null = all
// layers, MTP last). One cudaStream is created internally for the
// views' slab packs. Throws on any mismatch or reconcile failure.
//
// `resident` materializes the SHARDED streams' layers once each and
// serves the parity loop's loads from the resident cache after the
// first pass — pinning resident-vs-streaming bitwise (the full side
// stays streaming; resident bytes are streaming bytes by construction)
// AND proving the residency contract (cache hits re-read nothing).
GlmShardParityReport glm_shard_parity_check(
    const GlmTextConfig& cfg, const std::string& checkpoint_dir, int world,
    const std::vector<int>* layers = nullptr, bool resident = false);

}  // namespace dgpp
