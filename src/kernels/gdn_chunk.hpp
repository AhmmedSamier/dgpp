#pragma once
// Chunked Gated DeltaNet prefill on the tensor cores (FLA's
// chunk_gated_delta_rule, as SGLang runs it). See gdn_chunk.cu.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

bool gdn_chunked_supported(int k_dim, int v_dim);
size_t gdn_chunked_workspace_bytes(int tokens, int heads);

// The layouts of gdn_recurrent_fwd (kernels/kda.hpp) with an fp32 state and
// no snapshots: qkv bf16 [tokens, Hk*K | Hk*K | H*V], a_raw/beta_raw bf16
// [tokens, H] (row strides), state fp32 [H, V, K] in/out, out bf16 [tokens, H, V].
// The caller owns a 16-byte-aligned device workspace of at least
// gdn_chunked_workspace_bytes(tokens, heads) bytes. Keep it alive until the
// stream finishes; concurrent calls must use separate workspaces. Reuse on
// the same stream is ordered, including during graph capture and replay.
void gdn_chunked_fwd(const void* qkv, const void* a_raw, int64_t a_row_stride, const void* beta_raw,
                     int64_t beta_row_stride, const float* a_log, const float* dt_bias,
                     float* state, void* out, int tokens, int heads, int kv_ratio, int k_dim,
                     int v_dim, float scale, void* workspace, size_t workspace_bytes,
                     cudaStream_t stream);

}  // namespace dgpp
