#pragma once
// DeepSeek-V4.1-Flash Engram — the hashed n-gram memory (2026-09-13,
// docs/deepseek_v41_flash_plan.md §1.6, D5; the reference is the
// checkpoint's `inference/engram.py` NgramHashState and `model.py`
// Engram / ParallelEngramEmbedding). Three kernels, each deterministic and
// capturable; the wkv projection runs through the fp8 GEMM interface and
// its partials through the boundary reducer:
//
//   hash    token ids -> the (max_ngram_size - 1) x n_heads table rows per
//           Engram layer per token (int64 multiplicative-XOR hashes over
//           COMPRESSED token ids, modulo the (layer, n-gram, head) primes,
//           plus the bucket offsets); the request's context of the
//           preceding ids, pad at the sequence start;
//   gather  the rows the host staged out of the mmap'ed tables — e4m3
//           payload + e8m0 scales per 32 — as bf16(e4m3 x 2^(s - 127));
//   gate    the stream update: per (token, stream i) a normalized dot of
//           the stream against key_i, a signed-sqrt sigmoid gate, the
//           value added in.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// The hash constants (device arrays; the loader uploads them from the
// sidecar, tools/dsv41_engram_tables.py):
//   token_map   int32 [vocab]: token id -> compressed class;
//   multipliers int64 [layers, max_ngram]: per layer, one per lookback;
//   primes /    int64 [layers, max_ngram - 1, heads]: the bucket modulus
//   offsets     and the row offset of each (n-gram size, head) range.
struct Dsv41EngramHash {
  const int32_t* token_map = nullptr;
  const int64_t* multipliers = nullptr;
  const int64_t* primes = nullptr;
  const int64_t* offsets = nullptr;
  int32_t pad_class = 0;  // the class of engram_pad_token_id
  int layers = 0;
  int max_ngram = 0;      // engram_max_ngram_size (4: 2-, 3- and 4-grams)
  int heads = 0;          // engram_n_heads
  int vocab = 0;
};
constexpr int kDsv41EngramCtx = 4;  // int32 context words per request (max_ngram - 1 used)

// The row form (the decode graphs and the prefill chunks alike): rows of
// one or more requests in span order (req_spans int32 [num_requests, 2] =
// start, len into the batch; a request's rows contiguous, in position
// order; pos < 0 marks a padding row). Each request's stored context is
// ctx[req_ids[row] * 4 + k], k < max_ngram - 1: the compressed ids of the
// k+1 preceding tokens (newest first), pad_class where the sequence has
// not reached that far — a reset request holds pad_class everywhere.
// tokens: int64 [rows]. ids: int32 [rows, layers, (max_ngram - 1) * heads],
// row r's entry (l, (n - 2) * heads + h) the table row of layer l's
// n-gram, head h. A padding row hashes its (valid) token like any other;
// nothing reads the result.
void dsv41_engram_hash_ids_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                                const int64_t* pos, const int32_t* req_spans, int num_requests,
                                const int32_t* ctx, const Dsv41EngramHash& h, int32_t* ids,
                                cudaStream_t stream);
// ctx_rows[row * 4 + k] = the request's context AFTER that row (the
// compressed ids of that token and the ones before it) — the per-row
// snapshots a rollback copies back (row accepted-1); the span's last real
// row's context lands in the request's ctx IN PLACE. A padding row repeats
// the context as it stands. Enqueue after the hash (which reads the
// incoming context).
void dsv41_engram_context_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                               const int64_t* pos, const int32_t* req_spans, int num_requests,
                               const int32_t* token_map, int max_ngram, int32_t* ctx,
                               int32_t* ctx_rows, cudaStream_t stream);

// The staged rows' conversion: `staged` [n, rows_local, head_dim +
// head_dim / 32] — for every (token, local row) the e4m3 payload then its
// e8m0 scales (one per 32 elements), as the host gathered them out of the
// mmap'ed table (pinned memory is fine); out [n, rows_local * head_dim]
// bf16 = bf16(e4m3(p) x 2^(s - 127)), exact for every finite scale (an
// e4m3 value has 4 significant bits). head_dim % 32 == 0.
void dsv41_engram_gather_staged_bf16(const uint8_t* staged, int n, int rows_local, int head_dim,
                                     uint16_t* out, cudaStream_t stream);

// The gate and the stream update (the reference's Engram.forward): for
// every token t and stream i (hc of them),
//   w      = q_weight[i] * k_weight[i]                       (fp32, exact)
//   rstd   = rsqrt(mean_d(x_i^2) + eps) * rsqrt(mean_d(key_i^2) + eps)
//   dot    = sum_d x_i[d] * w[d] * key_i[d] * rstd * hidden^-0.5
//   gate   = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
//   x_i   += gate * value                                      (one bf16 rounding)
// x: bf16 [rows, hc, hidden] in place; kv: bf16 [rows, (hc + 1) * hidden]
// = key (hc x hidden) | value (hidden), the wkv projection's (folded)
// output; q_weight / k_weight: bf16 [hc, hidden]. One block per (token,
// stream).
void dsv41_engram_gate_rows(uint16_t* x, const uint16_t* kv, const uint16_t* q_weight,
                            const uint16_t* k_weight, int rows, int hc, int hidden, float eps,
                            cudaStream_t stream);

}  // namespace dgpp
