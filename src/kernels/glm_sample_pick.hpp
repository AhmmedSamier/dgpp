#pragma once
// The on-device SAMPLING pick (DESIGN §10, the device path): the greedy
// pick's two kernels (glm_pick.hpp) generalized from top-1 to top-k per
// rank plus each slice's temperature-scaled log-sum-exp, and a verdict that
// makes glm_sample::sample_from_prefix's decision BIT FOR BIT on the device
// — or flags the fallback for the host's exact gather. The host oracle is
// the definition; everything here mirrors its arithmetic: the same
// penalties (fused frequency step), the same fp32 temperature division, the
// same chunked normalizer order, the same deterministic exp/log
// (common/det_math.hpp), the same fp32 selector and fp64 walk, the same
// counter RNG. glm_pick_test pins the mirror against the host, value by
// value and bit by bit.
//
// Wire table: [rows][world][group] then the digest group [world][9] (the
// greedy pick's), even-padded. A rank's group is its k candidates (fp32
// logit bits as six digits, id as three — unused candidate slots carry the
// EMPTY id, an id no vocabulary reaches) followed by its slice lse (fp64
// bits as eleven digits). A greedy row writes one candidate (its canonical
// argmax) and no lse; a padding row writes nothing but zeros.
//
// Two shapes: T=1 (one row per request: accepted 1, next the decision)
// and the MTP T=2 verify (rows [next, draft] per request): row 0 accepts
// the draft with its exact probability or samples the residual
// (glm_sample::spec_accept_from_prefix), and when the draft stands row 1
// is sampled as any step's token (sample_from_prefix) — so accepted is 2
// or 1 and next = winners[accepted-1], the greedy judge's shape. A row
// that cannot decide inside its prefix flags the fallback: row 0 by
// provisionally REJECTING (the commit then keeps only the post-row-0
// state, which is right for a reject and recoverable for an accept), row 1
// by feeding its provisional argmax; the host serves both between windows
// (GlmGraphEngineAdapter). The request's count table takes both fed
// tokens before the penalties and drops the draft again on a reject.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/glm_pick.hpp"

namespace dgpp {

// The per-request sampling spec on the device — the host writes it between
// windows (configure_sampling), the verdict advances `counter` when it
// draws. temperature <= 0 is the greedy path: the argmax, no penalties, no
// draw — bitwise the greedy pick.
struct GlmSampleSpec {
  float temperature = 0.0f;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int32_t top_k = 0;
  int32_t logprobs = -1;  // -1: none; N >= 0: report the chosen token's
                          // logprob and its top-N alternatives. A greedy
                          // request that reports (or carries penalties)
                          // runs the full path at temperature 1 — the
                          // argmax under the raw normalizer, no draw.
  uint64_t seed = 0;
  uint64_t counter = 0;
};

constexpr int kSampleMaxTopLogprobs = 20;

// The sampling verdict's outcome per request, beside the GlmPickVerdict the
// device consumers (commit, token feeds) keep reading.
struct GlmSampleOutcome {
  int32_t fallback = 0;   // a row could not decide inside its prefix
  int32_t sampled = 0;    // a stochastic decision was made (0: greedy row)
  uint64_t counter = 0;   // the spec's counter after this pick (the draws
                          // the device consumed; a fallback row's draw is
                          // reserved for the host)
  double normalizer = 0.0;    // row 0's fold log-sum-exp
  double covered_mass = 0.0;  // row 0's prefix mass under it
  float logprob = 0.0f;       // row 0's outcome log-probability
  float logprob1 = 0.0f;      // row 1's (T=2, the draft stood)
  int32_t fallback_row = -1;  // which row fell back (0 or 1), -1 none
  int32_t accepted_draft = 0; // T=2: the draft stood (provisional 0 on a
                              // row-0 fallback)
  double normalizer1 = 0.0;   // row 1's fold log-sum-exp (T=2)
  // The reported top logprobs per row (spec.logprobs >= 0, rows the device
  // decided): min(N, the final set) entries, as the host's Result.
  int32_t top_count[2] = {0, 0};
  int32_t top_ids[2][kSampleMaxTopLogprobs] = {};
  float top_logprobs[2][kSampleMaxTopLogprobs] = {};
};

constexpr int kSampleLseDigits = 11;         // 66 bits carry the fp64 lse
constexpr int kSampleMaxCandidates = 256;    // the profiler's ceiling
constexpr int kSampleLseChunk = 256;         // == glm_sample::kLseChunk
constexpr uint32_t kSampleEmptyId = (1u << (6 * kPickIdDigits)) - 1;

constexpr size_t glm_sample_rank_group_slots(int candidates) {
  return static_cast<size_t>(candidates) * kPickSlotsPerRank +
         kSampleLseDigits;
}
constexpr size_t glm_sample_table_elems(int rows, int world, int candidates) {
  const size_t slots =
      static_cast<size_t>(rows) * static_cast<size_t>(world) *
          glm_sample_rank_group_slots(candidates) +
      static_cast<size_t>(world) * kPickSlotsPerRank;
  return slots + (slots & 1);
}
// The widest candidate table for `rows` rows at `world` ranks that fits
// `slot_bytes` (0 when even one candidate does not).
constexpr int glm_sample_candidates_that_fit(int rows, int world,
                                             size_t slot_bytes, int wanted) {
  int k = wanted;
  while (k > 0 && glm_sample_table_elems(rows, world, k) * 2 > slot_bytes)
    --k;
  return k;
}

// Kernel 1 (before the fold), one block per row. Row r belongs to request
// r / rows_per_request; `positions` (optional) marks a padding request by a
// negative first position. A sampled row (its spec's temperature > 0)
// first counts its fed token into the request's count table
// (counts[request * vocab_size + token] += 1), applies the penalties IN
// PLACE on its logits slice (glm_sample::apply_penalties), and writes its
// exact local top-k in canonical order (glm_sample::local_topk) and its
// temperature-scaled slice log-sum-exp (glm_sample::slice_logsumexp). A
// greedy row writes its canonical argmax as the one candidate. locals[row]
// carries the row's best (and, for sampled rows, second-best) for the log.
void glm_sample_local(float* logits, int rows, int vocab_count,
                      int vocab_begin, int vocab_size, int rank, int world,
                      int candidates, const GlmSampleSpec* specs,
                      int rows_per_request, const int64_t* fed,
                      const int64_t* positions, int position_stride,
                      int32_t* counts, const uint64_t* carry_digest,
                      uint16_t* table, GlmPickLocal* locals,
                      cudaStream_t stream);

// Kernel 2 (after the fold), one block per request plus the digest pass:
// decodes every rank's group, merges the k-way prefix in canonical order
// (glm_sample::merge_topk), folds the lse (glm_sample::merge_logsumexp) and
// makes sample_from_prefix's decision with the request's spec and RNG.
// Writes the GlmPickVerdict (rows/accepted 1, next and winners[0] the
// decision — the provisional argmax on a fallback — and the digest chain
// exactly as glm_pick_verdict_batched) to the pinned mirror and the device
// copy, the GlmSampleOutcome, and the spec's advanced counter.
void glm_sample_verdict(const uint16_t* table, int rows, int world, int rank,
                        int candidates, int vocab_size, GlmSampleSpec* specs,
                        int requests, int rows_per_request, const int64_t* fed,
                        const int64_t* positions, int position_stride,
                        int32_t* counts, GlmPickVerdict* verdicts,
                        GlmPickVerdict* device_verdicts,
                        GlmSampleOutcome* outcomes, uint64_t* carry_digest,
                        cudaStream_t stream);

// counts[token] += delta (the host's correction of a request's context after
// a fallback it decided: the draft leaves the table on a reject).
void glm_sample_adjust_count(int32_t* counts_row, int64_t token, int delta,
                             int vocab_size, cudaStream_t stream);

// The prefill's context: counts[ids[i]] += 1 for i < n (a request's prompt).
void glm_sample_count_tokens(int32_t* counts_row, const int64_t* ids, int n,
                             int vocab_size, cudaStream_t stream);

}  // namespace dgpp
