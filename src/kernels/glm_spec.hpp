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

#include "kernels/glm_pick.hpp"

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
void glm_spec_commit(const GlmPickVerdict* verdict, int rows,
                     const GlmSpecSegments& segments, int64_t* session_pos,
                     cudaStream_t stream);

// The decode rows' metadata from the device position: step_pos[r] =
// *session_pos + r for r < rows (the replacement for the host's staged
// h_step_pos_ upload in a device-driven graph).
void glm_spec_positions(const int64_t* session_pos, int rows,
                        int64_t* step_pos, cudaStream_t stream);

// The in-graph draft's rows off the verify's verdict (phase C). The draft
// block runs a FIXED `rows` rows per step; the accepted rows are real
// (step_pos[r] = *block_pos + r, tokens[r] = winners[r]) and the rest are
// padding (step_pos[r] = -1, which the DSA decode path skips — nothing is
// written at any position — and tokens[r] = winners[0], any valid id).
// Then *block_pos += accepted and *next_out = verdict->next (the verify's
// pick is about to be overwritten by the draft's; the token the main stack
// consumes next survives here for glm_spec_next_tokens).
void glm_spec_draft_rows(const GlmPickVerdict* verdict, int rows,
                         int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                         int64_t* next_out, cudaStream_t stream);

// The next replay's fed tokens, written at the end of this one (phase D):
// tokens[0] = *next (the verify's), tokens[1] = draft_verdict->next (the
// block's guess for the token after it).
void glm_spec_next_tokens(const int64_t* next,
                          const GlmPickVerdict* draft_verdict, int64_t* tokens,
                          cudaStream_t stream);

}  // namespace dgpp
