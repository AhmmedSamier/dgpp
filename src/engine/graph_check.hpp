#pragma once
// The captured decode graph's shape contract (2026-09-03): KERNEL nodes
// only (empty nodes tolerated). A memset or memcpy node executes on the
// copy-engine queue — one in-order queue shared by every stream in the
// process, where a queued node's dependency wait blocks everything behind
// it. In a one-process multi-rank world (the loopback gates) that formed
// the batched-MTP graph stall: rank B's replay queued its post-collective
// DSA counter memset (its dependency wait at the queue head), rank A's
// pre-collective memset queued behind it, and B's collective spun waiting
// on A's — docs/batched_mtp_graph_stall.md (nsys node trace: A's memset
// ran 992 ns after B's post-collective memset, five seconds late). Host
// nodes would block the same way and more. The decode path uploads with
// kernels (glm_upload_i32) and resets counters with kernels; every capture
// site calls this gate so a copy-engine node cannot re-enter unnoticed.
// Fabric ranks are one process each and never share the queue — the
// contract costs them nothing and is checked there too.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {

// Logs the node-type histogram (info) and throws std::runtime_error when
// the graph holds any memcpy, memset, host or other non-kernel node — a
// family that declares host nodes (Model::session_graph_host_nodes(): the
// Qwen walk's n-gram staging over the mmap'ed table, 2026-09-10) may
// carry up to `host_nodes` of them: a host node runs on a runtime thread
// and waits on nothing but its own stream, so it sits outside the copy
// engine's shared in-order queue that the cycle below needs.
inline void check_decode_graph(cudaGraph_t graph, int rank,
                                   const std::string& what, size_t host_nodes = 0) {
  size_t n = 0, e = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &n));
  DGPP_CUDA_OK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &e));
  size_t kernels = 0, empties = 0, memcpys = 0, memsets = 0, hosts = 0,
         events = 0, others = 0, max_in = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType t;
    DGPP_CUDA_OK(cudaGraphNodeGetType(node, &t));
    switch (t) {
      case cudaGraphNodeTypeKernel: ++kernels; break;
      case cudaGraphNodeTypeEmpty: ++empties; break;
      case cudaGraphNodeTypeMemcpy: ++memcpys; break;
      case cudaGraphNodeTypeMemset: ++memsets; break;
      case cudaGraphNodeTypeHost: ++hosts; break;
      // An event record node is REJECTED with the rest (2026-09-06): an
      // external event record node in the replayed decode graph stalled
      // about one relaunch in five — the graph never started on one rank
      // while its peer spun in the first collective. The pipelined
      // replay's verdict is published by a kernel node instead
      // (glm_publish_seq).
      case cudaGraphNodeTypeEventRecord: ++events; break;
      default: ++others; break;
    }
    size_t deps = 0;
    DGPP_CUDA_OK(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &deps));
    max_in = std::max(max_in, deps);
  }
  DGPP_LOG_INFO(
      "rank {}: {} shape: {} nodes, {} edges: kernel {} empty {} memcpy {} "
      "memset {} host {} event {} other {}; max in-degree {}",
      rank, what, n, e, kernels, empties, memcpys, memsets, hosts, events,
      others, max_in);
  if (memcpys != 0 || memsets != 0 || hosts > host_nodes || events != 0 ||
      others != 0)
    throw std::runtime_error(
        what + " captured " + std::to_string(memcpys) + " memcpy, " +
        std::to_string(memsets) + " memset, " + std::to_string(hosts) +
        " host, " + std::to_string(events) + " event and " +
        std::to_string(others) +
        " other node(s); the decode graph must be kernels-only (" +
        std::to_string(host_nodes) + " host node(s) declared) — a "
        "copy-engine node can deadlock the in-process multi-rank world "
        "(docs/batched_mtp_graph_stall.md)");
}

}  // namespace dgpp
