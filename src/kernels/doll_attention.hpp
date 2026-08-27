#pragma once
#include <cuda_runtime.h>

namespace dgpp {

// Causal GQA attention over a flat per-layer KV cache.
//
// Layouts:
//   Q: [nq rows, Hq*Dh] with row stride `q_stride_elems` (lives inside the
//      fused-qkv activation buffer), row-major; K,V: [slots, Hkv*Dh] caches;
//   O: [nq, Hq*Dh] contiguous bf16.
// Query i sits at absolute position (abs_first + i) and attends keys
// [0 .. abs_first+i]. Counts arrive via DEVICE MEMORY so captured graphs
// replay at any sequence state (DESIGN §5.1).
void gqa_causal_attention(const void* q, int64_t q_stride_elems,
                          const void* k_cache, const void* v_cache, void* out,
                          const int* dev_query_count,
                          const int* dev_abs_first, int max_queries_cap,
                          int num_heads, int num_kv_heads, int head_dim,
                          float softmax_scale, cudaStream_t stream);

}  // namespace dgpp
