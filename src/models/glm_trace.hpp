#pragma once
// Route-trace file format (M4 deliverable 5): the record of per-layer
// top-k routing decisions that replaces the uniform-expert assumption in
// the per-rank traffic model (tools/route_trace_traffic.py consumes it).
//
// Layout (little-endian):
//   8s  magic "DGPPTC1\x01"   (T = trace; version byte 1 baked in)
//   u32 num_layers
//   per layer:
//     u32 layer_idx
//     u32 top_k
//     u64 tokens
//     i32 ids[tokens * top_k]      (ascending expert id per token)
//     f32 weights[tokens * top_k]
//
// The format is pinned by golden bytes in BOTH the C++ unit test and the
// python tool test, so the two writers/readers cannot drift apart.
#include <cstdint>
#include <string>
#include <vector>

namespace dgpp {

struct GlmRouteTraceLayer {
  uint32_t layer_idx = 0;
  uint32_t top_k = 0;
  uint64_t tokens = 0;
  std::vector<int32_t> ids;
  std::vector<float> weights;
};

void glm_trace_write(const std::string& path,
                     const std::vector<GlmRouteTraceLayer>& layers);

// Reads back what glm_trace_write produced; throws on any mismatch (magic,
// truncation, or count overrun). Exists so the C++ side can verify dumps.
std::vector<GlmRouteTraceLayer> glm_trace_read(const std::string& path);

}  // namespace dgpp
