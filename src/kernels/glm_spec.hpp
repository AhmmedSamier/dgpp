#pragma once
// The speculative step's DEVICE-SIDE control (DESIGN §9, the on-device
// step, phase B): what the host used to do between a verify and the next
// step — read the verdict, roll the rejected rows' state back, advance the
// position, stage the next rows' positions — as kernels behind the pick's
// verdict, so a replayed graph carries its own control flow and the host
// reads the verdict only to log it.
//
// The rollback is a CONDITIONAL copy: cudaMemcpyAsync nodes cannot be
// predicated, a kernel can read `verdict->accepted` and either copy the
// post-row-(accepted-1) snapshots over the live state or return. Every
// state family the verify snapshots (glm_forward.hpp: spec_rec_, spec_conv_,
// spec_tail_ per DSA layer) is one segment of the table below; the copy is
// bitwise the host path's (session_rollback), the bytes are the same bytes.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/pick.hpp"

namespace dgpp {

// One rollback segment: `dst` is the live state; the snapshot after row a
// lives at `snapshots + a * row_stride_bytes`; `bytes` are copied. All
// three addresses 16-byte aligned, `bytes` and the stride multiples of 16.
struct GlmSpecSegment {
  void* dst = nullptr;
  const void* snapshots = nullptr;
  size_t row_stride_bytes = 0;
  size_t bytes = 0;
};

constexpr int kSpecMaxSegments = 32;  // KDA rec + KDA conv + DSA layers

struct GlmSpecSegments {
  int count = 0;
  GlmSpecSegment seg[kSpecMaxSegments];
};

// The step's commit, behind the verdict: when verdict->accepted < rows,
// copies every segment's snapshot row (accepted - 1) over its live state;
// always advances *session_pos by verdict->accepted. `rows` is the
// verify's row count (the graph's T); the kernel never trusts
// verdict->rows for the copy bound.
void glm_spec_commit(const PickVerdict* verdict, int rows,
                     const GlmSpecSegments& segments, int64_t* session_pos,
                     cudaStream_t stream);

// Uploads `count` (<= 64) int32 words from a PINNED, device-mapped host
// buffer into device memory with a kernel — the decode path's replacement
// for a cudaMemcpyAsync H2D of its row tables. A memcpy node in the
// captured decode graph executes on the copy-engine queue, which is
// in-order and shared by every stream in the process; in a one-process
// multi-rank world a peer rank's queued dependency wait at that queue's
// head blocked the upload (docs/batched_mtp_graph_stall.md). The source is
// read with system-scope loads: the host wrote it before the launch.
void glm_upload_i32(const int32_t* pinned_src, int32_t* dst, int count,
                    cudaStream_t stream);
void glm_upload_i64(const int64_t* pinned_src, int64_t* dst, int count,
                    cudaStream_t stream);

// The decode rows' metadata from the device position: step_pos[r] =
// *session_pos + r for r < rows (the replacement for the host's staged
// h_step_pos_ upload in a device-driven graph).
void glm_spec_positions(const int64_t* session_pos, int rows,
                        int64_t* step_pos, cudaStream_t stream);

// Fixed slot-major row batch: request_ids[r] selects its device position;
// the offset within that request's `rows_per_request` group is added when
// the slot is open. A closed slot has position <= 0 and emits -1 for every
// row, which is the shared KDA/DSA padding sentinel.
void glm_spec_positions_batched(const int64_t* session_pos,
                                const int32_t* request_ids, int rows,
                                int rows_per_request, int64_t* step_pos,
                                cudaStream_t stream);

// The in-graph draft's rows off the verify's verdict (phase C). The draft
// block runs a FIXED `rows` rows per step; the accepted rows are real
// (step_pos[r] = *block_pos + r, tokens[r] = winners[r]) and the rest are
// padding (step_pos[r] = -1, which the DSA decode path skips — nothing is
// written at any position — and tokens[r] = winners[0], any valid id).
// Then *block_pos += accepted and *next_out = verdict->next (the verify's
// pick is about to be overwritten by the draft's; the token the main stack
// consumes next survives here for glm_spec_next_tokens).
void glm_spec_draft_rows(const PickVerdict* verdict, int rows,
                         int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                         int64_t* next_out, cudaStream_t stream);

// One independent draft group per fixed request slot. `verdicts[q]` drives
// rows [q * rows_per_request, ...); inactive verdicts (accepted == 0) emit
// only padding and leave block_pos[q] unchanged. The group index is the
// request slot by construction of the Phase-2 fixed layout.
void glm_spec_draft_rows_batched(
    const PickVerdict* verdicts, int requests, int rows_per_request,
    int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
    int64_t* next_out, cudaStream_t stream);

// A plain device-to-device copy as a KERNEL (a memcpy node may not enter
// the captured decode graph): `bytes` a multiple of 16, both pointers
// 16-byte aligned. The in-graph draft snapshots its DSA tail ring with it
// before its rows run, so a fallback decided on the host can roll the block
// back (GlmDiagnosticModel::session_draft_rollback).
void glm_device_copy(void* dst, const void* src, size_t bytes,
                     cudaStream_t stream);

// The next replay's fed tokens, written at the end of this one (phase D):
// tokens[0] = *next (the verify's), tokens[1] = draft_verdict->next (the
// block's guess for the token after it).
void glm_spec_next_tokens(const int64_t* next,
                          const PickVerdict* draft_verdict, int64_t* tokens,
                          cudaStream_t stream);

// End-of-replay token feeds for the fixed batch. The plain T=1 graph takes
// each verify verdict's next token. The MTP T=2 graph takes the parked
// verify next plus one draft verdict per request. Inactive groups are zeroed
// so a later padded replay always embeds a valid token id.
void glm_spec_verify_next_tokens_batched(const PickVerdict* verify_verdicts,
                                         int requests, int rows_per_request,
                                         int64_t* tokens,
                                         cudaStream_t stream);
void glm_spec_next_tokens_batched(const int64_t* next,
                                  const PickVerdict* draft_verdicts,
                                  int requests, int rows_per_request,
                                  int64_t* tokens, cudaStream_t stream);

}  // namespace dgpp
