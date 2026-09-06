#include "kernels/glm_spec.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda/atomic>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

constexpr int kUploadMaxWords = 64;

template <typename Word>
__global__ void upload_words_kernel(const Word* __restrict__ src,
                                    Word* __restrict__ dst, int count) {
  const int i = static_cast<int>(threadIdx.x);
  if (i >= count) return;
  cuda::atomic_ref<Word, cuda::thread_scope_system> ref(
      *const_cast<Word*>(src + i));
  dst[i] = ref.load(cuda::memory_order_relaxed);
}

template <typename Word>
void launch_upload_words(const char* who, const Word* pinned_src, Word* dst,
                         int count, cudaStream_t stream) {
  if (pinned_src == nullptr || dst == nullptr)
    throw std::invalid_argument(std::string(who) + ": null argument");
  if (count < 1 || count > kUploadMaxWords)
    throw std::invalid_argument(std::string(who) + ": count outside [1, " +
                                std::to_string(kUploadMaxWords) + "]");
  upload_words_kernel<Word><<<1, kUploadMaxWords, 0, stream>>>(pinned_src,
                                                               dst, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

constexpr int kCopyThreads = 256;
constexpr size_t kUnit = 16;  // one uint4 per thread per iteration

// grid.y = segment, grid.x = chunks of the LARGEST segment; blocks past a
// smaller segment's end exit. Block (0,0)'s thread 0 owns the position
// advance so it happens exactly once whether or not anything copies.
__global__ void spec_commit_kernel(const PickVerdict* __restrict__ verdict,
                                   int rows, GlmSpecSegments segments,
                                   int64_t* __restrict__ session_pos) {
  const int accepted = verdict->accepted;
  if (accepted <= 0) return;  // fixed-batch padding slot
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    *session_pos += accepted;
  if (accepted >= rows) return;  // every row stood: nothing to retract
  if (blockIdx.y >= segments.count) return;  // position-only configuration

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

__global__ void spec_positions_batched_kernel(
    const int64_t* __restrict__ session_pos,
    const int32_t* __restrict__ request_ids, int rows, int rows_per_request,
    int64_t* __restrict__ step_pos) {
  const int r = threadIdx.x + blockIdx.x * blockDim.x;
  if (r >= rows) return;
  const int req = request_ids[r];
  const int64_t base = session_pos[req];
  step_pos[r] = base > 0 ? base + (r % rows_per_request) : -1;
}

__global__ void spec_draft_rows_kernel(const PickVerdict* __restrict__ verdict,
                                       int rows, int64_t* __restrict__ block_pos,
                                       int64_t* __restrict__ step_pos,
                                       int64_t* __restrict__ tokens,
                                       int64_t* __restrict__ next_out) {
  const int accepted = verdict->accepted;
  const int r = threadIdx.x;
  if (r < rows) {
    const bool real = r < accepted;
    step_pos[r] = real ? *block_pos + r : -1;
    tokens[r] = verdict->winners[real ? r : 0];
  }
  __syncthreads();  // every row read *block_pos before it moves
  if (r == 0) {
    *block_pos += accepted;
    *next_out = verdict->next;
  }
}

__global__ void spec_next_tokens_kernel(const int64_t* __restrict__ next,
                                        GlmSpecDrafts drafts,
                                        int64_t* __restrict__ tokens) {
  tokens[0] = *next;
  for (int c = 0; c < drafts.count; ++c) {
    const int32_t id = drafts.v[c]->next;
    tokens[1 + c] = id >= 0 ? id : 0;  // any valid id; a bad draft never stands
  }
}

__global__ void spec_chain_row_kernel(
    const PickVerdict* __restrict__ verify, int src_row,
    const PickVerdict* __restrict__ draft, const uint16_t* __restrict__ block_x,
    int hidden, uint16_t* __restrict__ hidden_cache,
    const int64_t* __restrict__ block_pos, int chain_index, int64_t max_context,
    int64_t* __restrict__ step_pos, int64_t* __restrict__ tokens,
    int32_t* __restrict__ req_spans) {
  const int row = verify != nullptr ? max(verify->accepted - 1, 0) : src_row;
  const int64_t pos = *block_pos + chain_index;
  const bool fits = pos >= 0 && pos < max_context;
  if (fits) {
    // The hidden row into the position cache, 16 bytes per thread-step.
    const uint4* src = reinterpret_cast<const uint4*>(
        block_x + static_cast<size_t>(row) * hidden);
    uint4* dst = reinterpret_cast<uint4*>(
        hidden_cache + static_cast<size_t>(pos) * hidden);
    const int units = hidden / 8;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
  }
  if (threadIdx.x == 0) {
    step_pos[0] = fits ? pos : -1;
    const int32_t id = draft->next;
    tokens[0] = id >= 0 ? id : 0;
    req_spans[0] = 0;
    req_spans[1] = 1;
  }
}

__global__ void spec_draft_rows_batched_kernel(
    const PickVerdict* __restrict__ verdicts, int rows_per_request,
    int64_t* __restrict__ block_pos, int64_t* __restrict__ step_pos,
    int64_t* __restrict__ tokens, int64_t* __restrict__ next_out) {
  const int q = blockIdx.x;
  const PickVerdict& verdict = verdicts[q];
  const int accepted = verdict.accepted;
  const bool active = accepted > 0;
  const int r = threadIdx.x;
  const int row = q * rows_per_request + r;
  if (r < rows_per_request) {
    const bool real = active && r < accepted;
    step_pos[row] = real ? block_pos[q] + r : -1;
    tokens[row] = active ? verdict.winners[real ? r : 0] : 0;
  }
  __syncthreads();
  if (r == 0 && active) {
    block_pos[q] += accepted;
    next_out[q] = verdict.next;
  }
}

__global__ void spec_verify_next_tokens_batched_kernel(
    const PickVerdict* __restrict__ verify, int rows_per_request,
    int64_t* __restrict__ tokens) {
  const int q = blockIdx.x;
  const bool active = verify[q].accepted > 0;
  for (int r = threadIdx.x; r < rows_per_request; r += blockDim.x)
    tokens[q * rows_per_request + r] =
        active && r == 0 ? verify[q].next : 0;
}

__global__ void spec_next_tokens_batched_kernel(
    const int64_t* __restrict__ next,
    const PickVerdict* __restrict__ draft, int rows_per_request,
    int64_t* __restrict__ tokens) {
  const int q = blockIdx.x;
  const bool active = draft[q].accepted > 0;
  for (int r = threadIdx.x; r < rows_per_request; r += blockDim.x) {
    int64_t token = 0;
    if (active && r == 0) token = next[q];
    if (active && r == 1) token = draft[q].next;
    tokens[q * rows_per_request + r] = token;
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__global__ void stage_wait_kernel(const uint64_t* __restrict__ pinned_seq,
                                  uint64_t* __restrict__ device_seq,
                                  uint32_t* __restrict__ pinned_late,
                                  int64_t timeout_ns) {
  if (threadIdx.x != 0) return;
  const uint64_t need = *device_seq + 1;
  *device_seq = need;
  uint64_t t0 = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (ld_acquire_sys_u64(pinned_seq) < need) {
    uint64_t now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (static_cast<int64_t>(now - t0) > timeout_ns) {
      *pinned_late = 1u;
      __threadfence_system();
      break;
    }
    __nanosleep(2000);
  }
}

__global__ void publish_seq_kernel(uint64_t* __restrict__ device_seq,
                                   uint64_t* __restrict__ pinned_out) {
  if (threadIdx.x != 0) return;
  const uint64_t v = *device_seq + 1;
  *device_seq = v;
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(pinned_out), "l"(v) : "memory");
}

__global__ void upload_words_kernel(const uint32_t* __restrict__ src,
                                    uint32_t* __restrict__ dst, size_t count) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(src + i) : "memory");
    dst[i] = v;
  }
}

}  // namespace

void glm_stage_wait(const uint64_t* pinned_stage_seq, uint64_t* device_seq,
                    uint32_t* pinned_late, int64_t timeout_ns,
                    cudaStream_t stream) {
  if (pinned_stage_seq == nullptr || device_seq == nullptr ||
      pinned_late == nullptr || timeout_ns < 1)
    throw std::invalid_argument("glm_stage_wait: null argument/timeout");
  stage_wait_kernel<<<1, 32, 0, stream>>>(pinned_stage_seq, device_seq,
                                          pinned_late, timeout_ns);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_publish_seq(uint64_t* device_seq, uint64_t* pinned_out,
                     cudaStream_t stream) {
  if (device_seq == nullptr || pinned_out == nullptr)
    throw std::invalid_argument("glm_publish_seq: null argument");
  publish_seq_kernel<<<1, 32, 0, stream>>>(device_seq, pinned_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_upload_words(const uint32_t* pinned_src, uint32_t* dst, size_t count,
                      cudaStream_t stream) {
  if (pinned_src == nullptr || dst == nullptr || count == 0)
    throw std::invalid_argument("glm_upload_words: null argument/count");
  const unsigned blocks = static_cast<unsigned>(
      std::min<size_t>((count + 255) / 256, 256));
  upload_words_kernel<<<blocks, 256, 0, stream>>>(pinned_src, dst, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_commit(const PickVerdict* verdict, int rows,
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

void glm_upload_i32(const int32_t* pinned_src, int32_t* dst, int count,
                    cudaStream_t stream) {
  launch_upload_words("glm_upload_i32", pinned_src, dst, count, stream);
}

void glm_upload_i64(const int64_t* pinned_src, int64_t* dst, int count,
                    cudaStream_t stream) {
  launch_upload_words("glm_upload_i64", pinned_src, dst, count, stream);
}

namespace {
__global__ void device_copy_kernel(uint4* __restrict__ dst,
                                   const uint4* __restrict__ src,
                                   size_t units) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < units; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}
}  // namespace

void glm_device_copy(void* dst, const void* src, size_t bytes,
                     cudaStream_t stream) {
  if (dst == nullptr || src == nullptr || bytes == 0 || bytes % 16 != 0 ||
      (reinterpret_cast<uintptr_t>(dst) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(src) & 15) != 0)
    throw std::invalid_argument("glm_device_copy: alignment/size");
  const size_t units = bytes / 16;
  const unsigned blocks = static_cast<unsigned>(
      std::min<size_t>((units + 255) / 256, 1024));
  device_copy_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<uint4*>(dst), static_cast<const uint4*>(src), units);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_positions(const int64_t* session_pos, int rows,
                        int64_t* step_pos, cudaStream_t stream) {
  if (rows < 1 || rows > 32)
    throw std::invalid_argument("glm_spec_positions: rows");
  spec_positions_kernel<<<1, 32, 0, stream>>>(session_pos, rows, step_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_positions_batched(const int64_t* session_pos,
                                const int32_t* request_ids, int rows,
                                int rows_per_request, int64_t* step_pos,
                                cudaStream_t stream) {
  if (session_pos == nullptr || request_ids == nullptr || step_pos == nullptr)
    throw std::invalid_argument("glm_spec_positions_batched: null argument");
  if (rows < 1 || rows > kPickMaxRows || rows_per_request < 1 ||
      rows % rows_per_request != 0)
    throw std::invalid_argument("glm_spec_positions_batched: row shape");
  spec_positions_batched_kernel<<<1, 32, 0, stream>>>(
      session_pos, request_ids, rows, rows_per_request, step_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_draft_rows(const PickVerdict* verdict, int rows,
                         int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                         int64_t* next_out, cudaStream_t stream) {
  if (verdict == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || next_out == nullptr)
    throw std::invalid_argument("glm_spec_draft_rows: null argument");
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument("glm_spec_draft_rows: rows outside [1, " +
                                std::to_string(kPickMaxRows) + "]");
  spec_draft_rows_kernel<<<1, 32, 0, stream>>>(verdict, rows, block_pos,
                                               step_pos, tokens, next_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_draft_rows_batched(
    const PickVerdict* verdicts, int requests, int rows_per_request,
    int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
    int64_t* next_out, cudaStream_t stream) {
  if (verdicts == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || next_out == nullptr)
    throw std::invalid_argument("glm_spec_draft_rows_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests ||
      rows_per_request < 1 || requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument("glm_spec_draft_rows_batched: request shape");
  spec_draft_rows_batched_kernel<<<requests, 32, 0, stream>>>(
      verdicts, rows_per_request, block_pos, step_pos, tokens, next_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_next_tokens(const int64_t* next, const GlmSpecDrafts& drafts,
                          int64_t* tokens, cudaStream_t stream) {
  if (next == nullptr || tokens == nullptr)
    throw std::invalid_argument("glm_spec_next_tokens: null argument");
  if (drafts.count < 1 || drafts.count > kSpecMaxDrafts)
    throw std::invalid_argument("glm_spec_next_tokens: draft count");
  for (int c = 0; c < drafts.count; ++c)
    if (drafts.v[c] == nullptr)
      throw std::invalid_argument("glm_spec_next_tokens: null draft verdict");
  spec_next_tokens_kernel<<<1, 1, 0, stream>>>(next, drafts, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_chain_row(const PickVerdict* verify_verdict, int src_row,
                        const PickVerdict* draft_verdict,
                        const uint16_t* block_x, int hidden,
                        uint16_t* hidden_cache, const int64_t* block_pos,
                        int chain_index, int64_t max_context,
                        int64_t* step_pos, int64_t* tokens, int32_t* req_spans,
                        cudaStream_t stream) {
  if (draft_verdict == nullptr || block_x == nullptr ||
      hidden_cache == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || req_spans == nullptr)
    throw std::invalid_argument("glm_spec_chain_row: null argument");
  if (hidden < 8 || hidden % 8 != 0 || !aligned16(block_x) ||
      !aligned16(hidden_cache))
    throw std::invalid_argument("glm_spec_chain_row: hidden alignment");
  if (verify_verdict == nullptr && (src_row < 0 || src_row >= kPickMaxRows))
    throw std::invalid_argument("glm_spec_chain_row: source row");
  if (chain_index < 0 || chain_index >= kSpecMaxDrafts || max_context < 1)
    throw std::invalid_argument("glm_spec_chain_row: chain index/context");
  spec_chain_row_kernel<<<1, 256, 0, stream>>>(
      verify_verdict, src_row, draft_verdict, block_x, hidden, hidden_cache,
      block_pos, chain_index, max_context, step_pos, tokens, req_spans);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_verify_next_tokens_batched(const PickVerdict* verify_verdicts,
                                         int requests, int rows_per_request,
                                         int64_t* tokens,
                                         cudaStream_t stream) {
  if (verify_verdicts == nullptr || tokens == nullptr)
    throw std::invalid_argument(
        "glm_spec_verify_next_tokens_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests ||
      rows_per_request < 1 || requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument(
        "glm_spec_verify_next_tokens_batched: request shape");
  spec_verify_next_tokens_batched_kernel<<<requests, 32, 0, stream>>>(
      verify_verdicts, rows_per_request, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_next_tokens_batched(const int64_t* next,
                                  const PickVerdict* draft_verdicts,
                                  int requests, int rows_per_request,
                                  int64_t* tokens, cudaStream_t stream) {
  if (next == nullptr || draft_verdicts == nullptr || tokens == nullptr)
    throw std::invalid_argument("glm_spec_next_tokens_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests || rows_per_request != 2 ||
      requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument(
        "glm_spec_next_tokens_batched: MTP requires two rows per request");
  spec_next_tokens_batched_kernel<<<requests, 32, 0, stream>>>(
      next, draft_verdicts, rows_per_request, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
