#include "kernels/glm_spec.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

constexpr int kCopyThreads = 256;
constexpr size_t kUnit = 16;  // one uint4 per thread per iteration

// grid.y = segment, grid.x = chunks of the LARGEST segment; blocks past a
// smaller segment's end exit. Block (0,0)'s thread 0 owns the position
// advance so it happens exactly once whether or not anything copies.
__global__ void spec_commit_kernel(const GlmPickVerdict* __restrict__ verdict,
                                   int rows, GlmSpecSegments segments,
                                   int64_t* __restrict__ session_pos) {
  const int accepted = verdict->accepted;
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    *session_pos += accepted;
  if (accepted >= rows) return;  // every row stood: nothing to retract

  const GlmSpecSegment& s = segments.seg[blockIdx.y];
  const size_t units = s.bytes / kUnit;
  const uint4* src = reinterpret_cast<const uint4*>(
      static_cast<const char*>(s.snapshots) +
      static_cast<size_t>(accepted - 1) * s.row_stride_bytes);
  uint4* dst = static_cast<uint4*>(s.dst);
  for (size_t i = static_cast<size_t>(blockIdx.x) * kCopyThreads + threadIdx.x;
       i < units; i += static_cast<size_t>(gridDim.x) * kCopyThreads)
    dst[i] = src[i];
}

__global__ void spec_positions_kernel(const int64_t* __restrict__ session_pos,
                                      int rows, int64_t* __restrict__ step_pos) {
  const int r = threadIdx.x;
  if (r < rows) step_pos[r] = *session_pos + r;
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

}  // namespace

void glm_spec_commit(const GlmPickVerdict* verdict, int rows,
                     const GlmSpecSegments& segments, int64_t* session_pos,
                     cudaStream_t stream) {
  if (verdict == nullptr || session_pos == nullptr)
    throw std::invalid_argument("glm_spec_commit: null verdict/position");
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument("glm_spec_commit: rows outside [1, " +
                                std::to_string(kPickMaxRows) + "]");
  if (segments.count < 0 || segments.count > kSpecMaxSegments)
    throw std::invalid_argument("glm_spec_commit: segment count");
  size_t max_units = 0;
  for (int i = 0; i < segments.count; ++i) {
    const GlmSpecSegment& s = segments.seg[i];
    if (!aligned16(s.dst) || !aligned16(s.snapshots) ||
        s.bytes % kUnit != 0 || s.row_stride_bytes % kUnit != 0)
      throw std::invalid_argument(
          "glm_spec_commit: segment " + std::to_string(i) +
          " must be 16-byte aligned with 16-byte-multiple sizes");
    max_units = std::max(max_units, s.bytes / kUnit);
  }
  // A single block still runs (the position advance) when no segment
  // exists; otherwise enough blocks to stream the largest segment.
  const unsigned chunks = static_cast<unsigned>(
      std::min<size_t>(1024, (max_units + kCopyThreads - 1) / kCopyThreads));
  const dim3 grid(std::max(1u, chunks),
                  static_cast<unsigned>(std::max(1, segments.count)));
  spec_commit_kernel<<<grid, kCopyThreads, 0, stream>>>(verdict, rows,
                                                        segments, session_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_positions(const int64_t* session_pos, int rows,
                        int64_t* step_pos, cudaStream_t stream) {
  if (rows < 1 || rows > 32)
    throw std::invalid_argument("glm_spec_positions: rows");
  spec_positions_kernel<<<1, 32, 0, stream>>>(session_pos, rows, step_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
