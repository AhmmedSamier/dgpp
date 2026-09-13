#pragma once
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"

namespace dgpp {

// y[m] = w * x[m] / sqrt(mean(x^2) + eps); bf16 io, fp32 accumulation.
void rmsnorm_bf16(const void* x, const void* weight, void* y, int rows,
                  int dim, float eps, cudaStream_t stream);

// gate[m,i]: silu(clamp(g, -lim, lim)) * clamp(u, -lim, lim) -> bf16.
// Reference-matching clamp semantics live here so parity work (M4) patches
// exactly one site (DESIGN §12 numerical gates).
void swiglu_limit_bf16(const void* gate, const void* up, void* out,
                       int64_t n_elems, float limit, cudaStream_t stream);

// Fused gateup-input variant: source holds interleaved halves per row
// ([g | u] x 2*inter wide); writes inter-wide output.
void swiglu_gateup_pairs_bf16(const void* gateup, void* out, int rows,
                              int inter, float limit, cudaStream_t stream);

// mHC elementwise mixing: out[b,s,h] = sum_j mix[b,s,j] * x[b,j,h].
// stream count S=4 baked (mhc_mult of the target family); bf16 io with fp32
// accumulate; `mix` holds per-token dynamic coefficients [B,S,S] fp32.
void mhc_mix_bf16(const void* x, const void* mix, void* out, int batch,
                  int hidden, cudaStream_t stream);

// x += y elementwise, bf16 (fp32 accumulate internally).
void add_inplace_bf16(void* x, const void* y, int64_t n_elems,
                      cudaStream_t stream);

// bf16 row range -> fp32 cast (logits path).
void cast_bf16_to_f32_rows(const void* src_bf16, float* dst_f32,
                           int rows, int cols, cudaStream_t stream);

// bf16 -> e4m3 quantizing cast, saturating (unit-scale activation prep for
// fp8 GEMMs; per-tensor/block scales arrive with the M4 quantization work).
void cast_bf16_to_fp8_rows(const void* src_bf16, void* dst_fp8,
                           int64_t n_elems, cudaStream_t stream);

// Appends projected K/V rows into two caches. Source rows live strided
// inside the fused-qkv activation buffer: row r's K block starts at
// src[r*src_stride], V block immediately after (kv_dim wide). Slots written:
// [abs_first .. abs_first+n); abs_first is read from DEVICE memory.
void kv_append_from_pairs(const void* src, int64_t src_stride_elems,
                          void* k_cache, void* v_cache,
                          const int* dev_abs_first, int n_rows, int kv_dim,
                          cudaStream_t stream);

// Device-side scalar bump for graph-self-contained rollouts: (*p) += delta.
void bump_i32_device(int32_t* p, int delta, cudaStream_t stream);
void copy_i64_device_to_device(const int64_t* src, int64_t* dst,
                               cudaStream_t stream);

// Row-wise argmax, lowest-index tie-break. logits [rows, cols] fp32.
// out_idx int64 [rows]. Also emits selected value into out_val (fp32).
void argmax_rows_f32(const float* logits, int64_t* out_idx, float* out_val,
                     int rows, int cols, cudaStream_t stream);

// Embedding row gather: out[t,h] = table[tokens[t],h], bf16.
void embed_gather_bf16(const void* table, const int64_t* tokens, void* out,
                       int num_tokens, int hidden, cudaStream_t stream);
// The vocab-sharded form: `table` holds rows [vocab_begin, vocab_begin +
// vocab_count) of the embedding; a token outside the slice writes a zero
// row, so the ranks' outputs summed by one fold are the full rows (a row
// plus zeros is exact in bf16).
void embed_gather_sliced_bf16(const void* table, const int64_t* tokens, void* out,
                              int num_tokens, int hidden, int64_t vocab_begin,
                              int64_t vocab_count, cudaStream_t stream);
// dst[t,:] = src[t,:] for num_tokens rows of `hidden` bf16 (either side may
// be pinned host memory — the folds' staged buffers).
void copy_rows_bf16(void* dst, const void* src, int num_tokens, int hidden,
                    cudaStream_t stream);

// Fills `n` bf16 elements ~N(0,stddev) deterministically from seed/index
// hashing (no external RNG dependency). Used for gpt-doll init.
void fill_random_normal_bf16(void* dst, uint64_t n, uint64_t seed,
                             float stddev, cudaStream_t stream);
void fill_random_normal_fp8(void* dst, uint64_t n, uint64_t seed,
                            float stddev, cudaStream_t stream);
void fill_random_normal_f32(void* dst, uint64_t n, uint64_t seed,
                            float stddev, cudaStream_t stream);

// Writes %globaltimer (the GPU's 64-bit ns clock) to *out when the stream
// reaches this point — a timeline probe for replayed graphs (the decode
// step's launch-to-first-kernel latency; compare against the bus's
// calibrated globaltimer offset).
void launch_globaltimer_stamp(uint64_t* out, cudaStream_t stream);

}  // namespace dgpp
