// Parity tests for the DeepSeek-V4.1-Flash Engram kernels
// (docs/deepseek_v41_flash_plan.md G2): the row-form hash against a host
// oracle of the reference's NgramHashState (compressed ids, pad at the
// start, the per-request context across rows, multiplicative-XOR hashes
// modulo the (layer, n-gram, head) primes plus the offsets), the context
// rows, the staged gather (bf16(e4m3 x 2^(s - 127)) exactly) and the
// gate/stream update against a double oracle.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/dsv41_engram.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) / static_cast<double>(1ull << 52) - 1.0; }
};

template <typename T>
T* managed(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T)));
  return p;
}

// A small Engram geometry: 2 layers, 4-grams (three n-gram sizes), 4
// heads, a 64-token vocabulary compressed to 40 classes, primes near 1000.
struct Geo {
  int layers = 2, max_ngram = 4, heads = 4, vocab = 64, classes = 40;
  std::vector<int32_t> token_map;
  std::vector<int64_t> multipliers;  // [layers, max_ngram]
  std::vector<int64_t> primes, offsets;  // [layers, max_ngram - 1, heads]
  int32_t pad_class = 0;
  int ngrams() const { return max_ngram - 1; }
  int rows_per_layer() const { return ngrams() * heads; }
};

bool is_prime(int64_t n) {
  if (n < 2) return false;
  for (int64_t d = 2; d * d <= n; ++d)
    if (n % d == 0) return false;
  return true;
}

Geo make_geo(uint64_t seed) {
  Geo g;
  Rng rng(seed);
  g.token_map.resize(static_cast<size_t>(g.vocab));
  for (auto& c : g.token_map) c = static_cast<int32_t>(rng.next() % static_cast<uint64_t>(g.classes));
  g.pad_class = g.token_map[2];
  g.multipliers.resize(static_cast<size_t>(g.layers) * g.max_ngram);
  for (auto& m : g.multipliers) m = static_cast<int64_t>((rng.next() % (1ull << 40)) * 2 + 1);
  int64_t p = 997;
  int64_t running = 0;
  for (int l = 0; l < g.layers; ++l) {
    running = 0;
    for (int n = 0; n < g.ngrams(); ++n)
      for (int h = 0; h < g.heads; ++h) {
        do ++p; while (!is_prime(p));
        g.primes.push_back(p);
        g.offsets.push_back(running);
        running += p;
      }
  }
  return g;
}

// The reference's hash for one row: y[k] = the compressed id k tokens back
// (pad past the sequence start), products XORed one lookback at a time.
std::vector<int32_t> oracle_hash(const Geo& g, const std::vector<int64_t>& y) {
  std::vector<int32_t> out;
  for (int l = 0; l < g.layers; ++l) {
    const int64_t* mult = g.multipliers.data() + static_cast<size_t>(l) * g.max_ngram;
    uint64_t rolling = static_cast<uint64_t>(y[0]) * static_cast<uint64_t>(mult[0]);
    for (int n = 1; n < g.max_ngram; ++n) {
      rolling ^= static_cast<uint64_t>(y[static_cast<size_t>(n)]) * static_cast<uint64_t>(mult[n]);
      for (int h = 0; h < g.heads; ++h) {
        const size_t at = (static_cast<size_t>(l) * g.ngrams() + (n - 1)) * g.heads + h;
        const int64_t mixed = static_cast<int64_t>(rolling);
        int64_t r = mixed % g.primes[at];
        if (r < 0) r += g.primes[at];
        out.push_back(static_cast<int32_t>(r + g.offsets[at]));
      }
    }
  }
  return out;
}

struct DevHash {
  int32_t* token_map;
  int64_t* multipliers;
  int64_t* primes;
  int64_t* offsets;
  dgpp::Dsv41EngramHash h;
  explicit DevHash(const Geo& g) {
    token_map = managed<int32_t>(g.token_map.size());
    multipliers = managed<int64_t>(g.multipliers.size());
    primes = managed<int64_t>(g.primes.size());
    offsets = managed<int64_t>(g.offsets.size());
    std::memcpy(token_map, g.token_map.data(), g.token_map.size() * 4);
    std::memcpy(multipliers, g.multipliers.data(), g.multipliers.size() * 8);
    std::memcpy(primes, g.primes.data(), g.primes.size() * 8);
    std::memcpy(offsets, g.offsets.data(), g.offsets.size() * 8);
    h.token_map = token_map;
    h.multipliers = multipliers;
    h.primes = primes;
    h.offsets = offsets;
    h.pad_class = g.pad_class;
    h.layers = g.layers;
    h.max_ngram = g.max_ngram;
    h.heads = g.heads;
    h.vocab = g.vocab;
  }
  ~DevHash() {
    cudaFree(token_map);
    cudaFree(multipliers);
    cudaFree(primes);
    cudaFree(offsets);
  }
};

}  // namespace

DGPP_TEST(dsv41_engram_hash_rows_match_the_reference_hash) {
  // Three requests in one batch: request 0 at the sequence start (pad
  // context), request 1 mid-sequence (a stored context), request 2 with a
  // padding row; every row's ids against the oracle built from the same
  // lookbacks.
  const Geo g = make_geo(0xE6);
  DevHash dh(g);
  Rng rng(0x7A);
  const int spans_h[] = {0, 5, 5, 3, 8, 4};  // (start, len) x 3
  const int rows = 12, num_requests = 3;
  const int32_t req_ids_h[rows] = {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
  int64_t* tokens = managed<int64_t>(rows);
  int32_t* req_ids = managed<int32_t>(rows);
  int64_t* pos = managed<int64_t>(rows);
  int32_t* spans = managed<int32_t>(6);
  int32_t* ctx = managed<int32_t>(3 * dgpp::kDsv41EngramCtx);
  int32_t* ids = managed<int32_t>(static_cast<size_t>(rows) * g.layers * g.rows_per_layer());
  for (int t = 0; t < rows; ++t) {
    tokens[t] = static_cast<int64_t>(rng.next() % static_cast<uint64_t>(g.vocab));
    req_ids[t] = req_ids_h[t];
    pos[t] = t;
  }
  pos[11] = -1;  // a padding row (its ids are computed, nothing reads them)
  std::memcpy(spans, spans_h, sizeof(spans_h));
  // Request 0: fresh (pad everywhere); request 1: three prior ids; request
  // 2: one real prior id, then pad.
  for (int k = 0; k < dgpp::kDsv41EngramCtx; ++k) ctx[0 * 4 + k] = g.pad_class;
  ctx[1 * 4 + 0] = 7; ctx[1 * 4 + 1] = 11; ctx[1 * 4 + 2] = 13; ctx[1 * 4 + 3] = 0;
  ctx[2 * 4 + 0] = 21; ctx[2 * 4 + 1] = g.pad_class; ctx[2 * 4 + 2] = g.pad_class; ctx[2 * 4 + 3] = 0;
  dgpp::dsv41_engram_hash_ids_rows(tokens, rows, req_ids, pos, spans, num_requests, ctx, dh.h, ids, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int q = 0; q < num_requests; ++q) {
    const int s = spans_h[2 * q], len = spans_h[2 * q + 1];
    for (int t = s; t < s + len; ++t) {
      std::vector<int64_t> y(static_cast<size_t>(g.max_ngram));
      for (int k = 0; k < g.max_ngram; ++k) {
        const int p = t - s;
        if (k == 0) y[k] = g.token_map[static_cast<size_t>(tokens[t])];
        else if (p >= k) y[k] = g.token_map[static_cast<size_t>(tokens[t - k])];
        else y[k] = ctx[q * 4 + (k - 1 - p)];
      }
      const std::vector<int32_t> want = oracle_hash(g, y);
      const int32_t* got = ids + static_cast<size_t>(t) * g.layers * g.rows_per_layer();
      for (size_t i = 0; i < want.size(); ++i)
        require(got[i] == want[i], "row " + std::to_string(t) + " id " + std::to_string(i) + " mismatch");
      // Every id lies inside its (layer, n-gram, head) bucket range.
      for (int l = 0; l < g.layers; ++l)
        for (int n = 0; n < g.ngrams(); ++n)
          for (int h = 0; h < g.heads; ++h) {
            const size_t at = (static_cast<size_t>(l) * g.ngrams() + n) * g.heads + h;
            const int32_t id = got[static_cast<size_t>(l) * g.rows_per_layer() + n * g.heads + h];
            require(id >= g.offsets[at] && id < g.offsets[at] + g.primes[at], "id inside its bucket range");
          }
    }
  }
  std::printf("[ OK ] engram hash: %d rows x %d layers x %d entries match the reference hash\n", rows, g.layers,
              g.rows_per_layer());
  cudaFree(tokens); cudaFree(req_ids); cudaFree(pos); cudaFree(spans); cudaFree(ctx); cudaFree(ids);
}

DGPP_TEST(dsv41_engram_context_rows_advance_per_real_row) {
  const Geo g = make_geo(0xE7);
  DevHash dh(g);
  const int rows = 6, num_requests = 2;
  const int spans_h[] = {0, 4, 4, 2};
  const int32_t req_ids_h[rows] = {3, 3, 3, 3, 5, 5};
  int64_t* tokens = managed<int64_t>(rows);
  int32_t* req_ids = managed<int32_t>(rows);
  int64_t* pos = managed<int64_t>(rows);
  int32_t* spans = managed<int32_t>(4);
  int32_t* ctx = managed<int32_t>(8 * dgpp::kDsv41EngramCtx);
  int32_t* ctx_rows = managed<int32_t>(static_cast<size_t>(rows) * dgpp::kDsv41EngramCtx);
  const int64_t toks[rows] = {10, 20, 30, 40, 50, 60};
  for (int t = 0; t < rows; ++t) { tokens[t] = toks[t]; req_ids[t] = req_ids_h[t]; pos[t] = t; }
  pos[2] = -1;  // a padding row in request 3
  std::memcpy(spans, spans_h, sizeof(spans_h));
  for (int k = 0; k < dgpp::kDsv41EngramCtx; ++k) { ctx[3 * 4 + k] = g.pad_class; ctx[5 * 4 + k] = 100 + k; }
  dgpp::dsv41_engram_context_rows(tokens, rows, req_ids, pos, spans, num_requests, dh.token_map, g.max_ngram,
                                  ctx, ctx_rows, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto cls = [&](int64_t tok) { return g.token_map[static_cast<size_t>(tok)]; };
  // Request 3: after row 0 [c10, pad, pad]; row 1 [c20, c10, pad]; row 2 (padding) unchanged; row 3 [c40, c20, c10].
  require(ctx_rows[0 * 4 + 0] == cls(10) && ctx_rows[0 * 4 + 1] == g.pad_class && ctx_rows[0 * 4 + 2] == g.pad_class, "row 0");
  require(ctx_rows[1 * 4 + 0] == cls(20) && ctx_rows[1 * 4 + 1] == cls(10) && ctx_rows[1 * 4 + 2] == g.pad_class, "row 1");
  require(ctx_rows[2 * 4 + 0] == cls(20) && ctx_rows[2 * 4 + 1] == cls(10), "padding row repeats the context");
  require(ctx_rows[3 * 4 + 0] == cls(40) && ctx_rows[3 * 4 + 1] == cls(20) && ctx_rows[3 * 4 + 2] == cls(10), "row 3");
  require(ctx[3 * 4 + 0] == cls(40) && ctx[3 * 4 + 1] == cls(20) && ctx[3 * 4 + 2] == cls(10), "request 3's context in place");
  // Request 5: prior [100, 101, 102]; after row 4 [c50, 100, 101]; row 5 [c60, c50, 100].
  require(ctx_rows[4 * 4 + 0] == cls(50) && ctx_rows[4 * 4 + 1] == 100 && ctx_rows[4 * 4 + 2] == 101, "row 4");
  require(ctx_rows[5 * 4 + 0] == cls(60) && ctx_rows[5 * 4 + 1] == cls(50) && ctx_rows[5 * 4 + 2] == 100, "row 5");
  require(ctx[5 * 4 + 0] == cls(60) && ctx[5 * 4 + 2] == 100, "request 5's context in place");
  std::printf("[ OK ] engram context rows: per-row snapshots and the in-place advance\n");
  cudaFree(tokens); cudaFree(req_ids); cudaFree(pos); cudaFree(spans); cudaFree(ctx); cudaFree(ctx_rows);
}

DGPP_TEST(dsv41_engram_staged_gather_is_exact) {
  // Random e4m3 rows with random e8m0 block scales (2^-20 .. 2^8 and the
  // extremes 2^-100 / 2^0 — the products stay fp32 normals; a real table's
  // scales sit near 2^-8): every output element equals
  // bf16(e4m3 x 2^(s - 127)) computed on the host, NaN scales propagate.
  const int n = 7, rows_local = 12, head_dim = 256;
  const int row_bytes = head_dim + head_dim / 32;
  uint8_t* staged = managed<uint8_t>(static_cast<size_t>(n) * rows_local * row_bytes);
  uint16_t* out = managed<uint16_t>(static_cast<size_t>(n) * rows_local * head_dim);
  Rng rng(0x5A6E);
  for (int64_t i = 0; i < static_cast<int64_t>(n) * rows_local; ++i) {
    uint8_t* row = staged + i * row_bytes;
    for (int d = 0; d < head_dim; ++d) row[d] = static_cast<uint8_t>(rng.next() & 0xFF);
    for (int b = 0; b < head_dim / 32; ++b) {
      const int pick = static_cast<int>(rng.next() % 12);
      row[head_dim + b] = pick == 0 ? 27 : pick == 1 ? 127 : static_cast<uint8_t>(107 + rng.next() % 29);
    }
  }
  staged[3 * row_bytes + head_dim + 2] = 255;  // a NaN scale on row 3's third block
  dgpp::dsv41_engram_gather_staged_bf16(staged, n, rows_local, head_dim, out, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  size_t nans = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(n) * rows_local; ++i) {
    const uint8_t* row = staged + i * row_bytes;
    for (int d = 0; d < head_dim; ++d) {
      const uint8_t sb = row[head_dim + d / 32];
      const uint16_t got = out[i * head_dim + d];
      if (sb == 255) {
        require(std::isnan(bf16_bits_to_float(got)), "NaN scale propagates");
        ++nans;
        continue;
      }
      const float s = std::ldexp(1.0f, static_cast<int>(sb) - 127);
      if ((row[d] & 0x7F) == 0x7F) {  // an e4m3 NaN code: NaN out, whatever its payload bits
        require(std::isnan(bf16_bits_to_float(got)), "NaN code propagates");
        continue;
      }
      const uint16_t want = float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(row[d]) * s);
      require(got == want, "gathered element " + std::to_string(i) + "," + std::to_string(d) + " exact");
    }
  }
  require(nans == 32, "exactly the poisoned block is NaN (beyond the NaN codes)");
  std::printf("[ OK ] engram staged gather: %d x %d rows exact (one NaN block propagated)\n", n, rows_local);
  cudaFree(staged);
  cudaFree(out);
}

DGPP_TEST(dsv41_engram_gate_matches_double_oracle) {
  const int rows = 5, hc = 4, hidden = 5120;
  const float eps = 1e-20f;
  uint16_t* x = managed<uint16_t>(static_cast<size_t>(rows) * hc * hidden);
  uint16_t* kv = managed<uint16_t>(static_cast<size_t>(rows) * (hc + 1) * hidden);
  uint16_t* qw = managed<uint16_t>(static_cast<size_t>(hc) * hidden);
  uint16_t* kw = managed<uint16_t>(static_cast<size_t>(hc) * hidden);
  Rng rng(0x6A7E);
  std::vector<uint16_t> x0(static_cast<size_t>(rows) * hc * hidden);
  for (auto& v : x0) v = float_to_bf16_bits(0.5f * static_cast<float>(rng.unit()));
  std::memcpy(x, x0.data(), x0.size() * 2);
  for (size_t i = 0; i < static_cast<size_t>(rows) * (hc + 1) * hidden; ++i)
    kv[i] = float_to_bf16_bits(0.25f * static_cast<float>(rng.unit()));
  for (size_t i = 0; i < static_cast<size_t>(hc) * hidden; ++i) {
    qw[i] = float_to_bf16_bits(1.0f + 0.2f * static_cast<float>(rng.unit()));
    kw[i] = float_to_bf16_bits(1.0f + 0.2f * static_cast<float>(rng.unit()));
  }
  // Row 4, stream 1: a dot near zero (a zero key) exercises the 1e-6 clamp.
  for (int d = 0; d < hidden; ++d) kv[(static_cast<size_t>(4) * (hc + 1) + 1) * hidden + d] = 0;
  dgpp::dsv41_engram_gate_rows(x, kv, qw, kw, rows, hc, hidden, eps, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  size_t off_by_more = 0;
  double max_gate_err = 0;
  for (int t = 0; t < rows; ++t)
    for (int i = 0; i < hc; ++i) {
      const uint16_t* xi = x0.data() + (static_cast<size_t>(t) * hc + i) * hidden;
      const uint16_t* key = kv + static_cast<size_t>(t) * (hc + 1) * hidden + static_cast<size_t>(i) * hidden;
      const uint16_t* value = kv + static_cast<size_t>(t) * (hc + 1) * hidden + static_cast<size_t>(hc) * hidden;
      double ssx = 0, ssk = 0, dot = 0;
      for (int d = 0; d < hidden; ++d) {
        const double hv = bf16_bits_to_float(xi[d]), kvv = bf16_bits_to_float(key[d]);
        const double w = static_cast<double>(bf16_bits_to_float(qw[static_cast<size_t>(i) * hidden + d])) *
                         bf16_bits_to_float(kw[static_cast<size_t>(i) * hidden + d]);
        ssx += hv * hv;
        ssk += kvv * kvv;
        dot += hv * w * kvv;
      }
      const double rstd = 1.0 / std::sqrt(ssx / hidden + eps) / std::sqrt(ssk / hidden + eps);
      const double g0 = dot * rstd / std::sqrt(static_cast<double>(hidden));
      const double mag = std::sqrt(std::max(std::fabs(g0), 1e-6));
      const double gate = 1.0 / (1.0 + std::exp(-std::copysign(mag, g0)));
      for (int d = 0; d < hidden; ++d) {
        const double want_d = bf16_bits_to_float(xi[d]) + gate * bf16_bits_to_float(value[d]);
        const uint16_t want = float_to_bf16_bits(static_cast<float>(want_d));
        const uint16_t got = x[(static_cast<size_t>(t) * hc + i) * hidden + d];
        const double diff = std::fabs(static_cast<double>(bf16_bits_to_float(got)) - bf16_bits_to_float(want));
        const double ulp = std::max(static_cast<double>(std::fabs(bf16_bits_to_float(want))), 1e-3) / 256.0;
        max_gate_err = std::max(max_gate_err, diff / ulp);
        if (diff > 2 * ulp) ++off_by_more;
      }
    }
  require(off_by_more == 0, "engram gate update within 2 bf16 ulps of the oracle everywhere (" +
                                std::to_string(off_by_more) + " over)");
  std::printf("[ OK ] engram gate: %d rows x %d streams within budget (max %.2f ulps)\n", rows, hc, max_gate_err);
  cudaFree(x); cudaFree(kv); cudaFree(qw); cudaFree(kw);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
