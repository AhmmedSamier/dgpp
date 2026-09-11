#pragma once
// Moved to engine/graph_check.hpp; the GLM-era name stays.
#include <string>

#include "engine/graph_check.hpp"

namespace dgpp {
inline void glm_check_decode_graph(cudaGraph_t graph, int rank,
                                   const std::string& what) {
  check_decode_graph(graph, rank, what);
}
}  // namespace dgpp
