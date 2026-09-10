#pragma once
// The Qwen3.8-Flash-Next MTP draft block's glue kernels (Q6 stage C,
// 2026-09-09; docs/qwen38_flash_next_plan.md §1.8; the reference is
// SGLang's Qwen4ExpForCausalLMMTP: pre_fc_norm_embedding = GemmaRMSNorm(H),
// pre_fc_norm_hidden = GemmaRMSNorm(hc*H) over the whole hyper state,
// fc_embedding / fc_hidden [H, H], R_mtp[t, i, :] = bf16(fc_e(norm(e)) +
// fc_h(norm(R)[i])) per branch i). The norms are kernels/qwen_norm.hpp's,
// the projections the GEMM seam's; these are the gathers and the add.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// out[t, :] = embed[tokens[t], :] (bf16 rows of `hidden`).
void qwen_mtp_embed_gather_bf16(const uint16_t* embed, const int64_t* tokens, uint16_t* out,
                                int rows, int hidden, cudaStream_t stream);

// out[t, i * hidden + d] = bf16(ein[t, d] + enc[t, i * hidden + d]) — the
// embedding projection broadcast over the hc branches of the hidden one.
void qwen_mtp_fuse_bf16(const uint16_t* ein, const uint16_t* enc, uint16_t* out, int rows,
                        int hc, int hidden, cudaStream_t stream);

// The hyper-state window: request req's last `window_rows` hyper states by
// position, window[req * req_stride + (pos % window_rows) * width]. store
// writes rows with pos >= 0 (r [rows, width], req_ids / pos [rows]); gather
// reads them back for the draft rows (pos < 0: zero).
void qwen_mtp_hidden_store_bf16(const uint16_t* r, const int32_t* req_ids, const int64_t* pos,
                                uint16_t* window, int64_t req_stride, int window_rows, int rows,
                                int width, cudaStream_t stream);
void qwen_mtp_hidden_gather_bf16(const uint16_t* window, int64_t req_stride, int window_rows,
                                 const int32_t* req_ids, const int64_t* pos, uint16_t* out,
                                 int rows, int width, cudaStream_t stream);

}  // namespace dgpp
