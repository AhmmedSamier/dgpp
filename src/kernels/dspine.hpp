#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// DSpine's selected-prefix code bridge kernels. Inputs and outputs are
// row-major BF16 except RMS, which is FP32. Each block is anchor + masks.
void dspine_prepare_read(const uint16_t* x, const uint16_t* residual, uint16_t* raw,
                         uint16_t* normalized, float* rms, int rows, int hidden,
                         float eps, bool add_residual, cudaStream_t stream);
// Match the reference FC boundary: round the FP32-accumulated context
// projection to BF16 before applying hidden_norm.
void dspine_context_norm_f32_bf16(const float* projected, const uint16_t* norm_weight,
                                  uint16_t* normalized, int rows, int hidden,
                                  float eps, cudaStream_t stream);
void dspine_normalize_codes(const uint16_t* projected, const uint16_t* codebook,
                            const int64_t* tokens, uint16_t* codes, int rows,
                            int block_rows, int vocab, int code_size, float eps,
                            cudaStream_t stream);
// packed contains three component planes, each [rows, rank]: query, predecessor gate, value.
void dspine_feature_message(const uint16_t* packed, const uint16_t* gate_bias,
                            uint16_t* message, int rows, int block_rows, int rank,
                            cudaStream_t stream);
void dspine_feature_finish(const uint16_t* raw, const uint16_t* delta,
                           const float* rms, const int64_t* positions,
                           const uint16_t* norm_weight, uint16_t* output,
                           uint16_t* normed, int rows, int block_rows, int hidden,
                           float eps, cudaStream_t stream);
void dspine_scale_rows(const uint16_t* input, const uint16_t* scale,
                       uint16_t* output, int rows, int width,
                       float multiplier, cudaStream_t stream);
void dspine_gather_rows(const uint16_t* matrix, const int32_t* ids,
                        uint16_t* rows_out, int count, int width,
                        cudaStream_t stream);
void dspine_gather_logits(const float* logits, const int32_t* ids,
                          float* values, int steps, int topk, int vocab,
                          cudaStream_t stream);
void dspine_pack_pair(const uint16_t* a, const uint16_t* b,
                      uint16_t* packed, int rows, int rank,
                      cudaStream_t stream);
void dspine_transpose_bf16(const uint16_t* input, uint16_t* output,
                           int rows, int cols, cudaStream_t stream);
void dspine_sender_ids(const int32_t* candidates, const int64_t* anchors,
                       int32_t* senders, int steps, int topk,
                       cudaStream_t stream);
void dspine_prewrite_scores(const uint16_t* raw, const uint16_t* norm,
                            const uint16_t* lm_head, const int32_t* candidates,
                            float* scores, int steps, int topk, int hidden,
                            int vocab, int block_rows, cudaStream_t stream);
// The last-write refinement: for each draft position and each possible
// predecessor candidate, select the best candidate at the current position.
// `prewrite_logits` are the shared-head logits on each row's raw state before
// the last bridge write, gathered at `candidate_ids`. `tg` and `tsv` are
// compact [steps, topk, rank] and [steps, topk, 2*rank] tables. Candidate IDs
// themselves come from the ordinary head on the post-write draft hidden states.
void dspine_serial_table(const int32_t* candidate_ids, const float* prewrite_logits,
                         const int64_t* anchors, const uint16_t* query,
                         const float* rms, const uint16_t* gate_bias,
                         const uint16_t* tg, const uint16_t* tsv,
                         int slots, int block_rows, int rank, int topk,
                         int vocab, float refine_weight, int32_t* indices, cudaStream_t stream);
void dspine_serial_chain(const int32_t* candidate_ids, const int32_t* indices,
                         int slots, int block_rows, int topk, int32_t* drafts,
                         cudaStream_t stream);

}  // namespace dgpp
