// Host reference implementation of the DSA/MLA layer (see the header for the
// pinned numerics). Templated on the accumulation type: float mirrors the
// device fp32 path, double is the high-precision oracle.
//
// One deliberate divergence from the Triton reference, documented here so
// nobody has to rediscover it: the power-of-two fp8 scale is computed with
// exact bit manipulation (smallest 2^n >= absmax/448, exact powers mapping
// to themselves) instead of exp2f(ceilf(log2f(v))). The fp32 libm path can
// round across a power-of-two boundary on near-tie inputs and clamp the fp8
// row; glibc and libdevice also disagree by ulps on those inputs. The exact
// form is deterministic on host and device and matches the libm form except
// on pathological near-power-of-two magnitudes, where the reference clamps
// and we do not.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dsa_reference.hpp"

namespace dgpp::dsa_ref {

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;

// Smallest power of two >= v (v > 0, normal); exact powers map to
// themselves. Bit-exact replacement for exp2f(ceilf(log2f(v))).
float next_pow2_at_or_above(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  int e = static_cast<int>((b >> 23) & 0xFFu) - 127;  // floor(log2(v))
  bool exact = (b & 0x7FFFFFu) == 0;
  float out;
  int shift = exact ? e : e + 1;
  out = std::ldexp(1.0f, shift);
  return out;
}

// Shared fp8 quant of a bf16-exact row: absmax (floor 1e-4), power-of-two
// scale, saturating encode. Values are bf16-exact, so this is identical for
// any accumulation precision upstream.
void quant_row_fp8(const float* x, int dim, uint8_t* out_bits, float* scale) {
  float absmax = 0.0f;
  for (int d = 0; d < dim; ++d) absmax = std::max(absmax, std::fabs(x[d]));
  absmax = std::max(absmax, 1e-4f);
  *scale = next_pow2_at_or_above(absmax * (1.0f / 448.0f));
  for (int d = 0; d < dim; ++d)
    out_bits[d] = float_to_fp8_e4m3_bits(x[d] / *scale);
}

// fp32 bits -> monotone-ascending sortable uint32 (negatives flipped).
uint32_t sortable_f32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return (u >> 31) ? ~u : (u | 0x80000000u);
}

}  // namespace

template <typename Acc>
void gemm_bf16(const uint16_t* act, int64_t act_row_stride, const uint16_t* w,
               uint16_t* out, int m, int n, int k) {
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      Acc acc = 0;
      for (int i = 0; i < k; ++i)
        acc += static_cast<Acc>(bf16_bits_to_float(
                   act[static_cast<int64_t>(r) * act_row_stride + i])) *
               static_cast<Acc>(bf16_bits_to_float(
                   w[static_cast<int64_t>(c) * k + i]));
      out[static_cast<int64_t>(r) * n + c] =
          float_to_bf16_bits(static_cast<float>(acc));
    }
  }
}

template <typename Acc>
void fwht128(Acc* x) {
  // Butterfly with stride S over blocks of 2S — the reference's
  // _hadamard128_stage sequence (GROUPS,STRIDE) = (64,1)...(1,64).
  for (int stride = 1; stride < 128; stride <<= 1) {
    for (int base = 0; base < 128; base += 2 * stride) {
      for (int p = base; p < base + stride; ++p) {
        Acc a = x[p];
        Acc b = x[p + stride];
        x[p] = a + b;
        x[p + stride] = a - b;
      }
    }
  }
  const Acc inv_sqrt128 = static_cast<Acc>(0.08838834764831845);
  for (int d = 0; d < 128; ++d) x[d] *= inv_sqrt128;
}

template <typename Acc>
void fwht128_quant_fp8(const uint16_t* q, int rows, int dim, uint8_t* q_fp8,
                       float* q_scale) {
  // dim must be 128 (validated by DsaConfig).
  for (int r = 0; r < rows; ++r) {
    Acc x[128];
    for (int d = 0; d < 128; ++d)
      x[d] = static_cast<Acc>(bf16_bits_to_float(q[int64_t(r) * dim + d]));
    fwht128(x);
    // The reference rounds the rotated row to bf16 before quantizing.
    float xr[128];
    for (int d = 0; d < 128; ++d)
      xr[d] = bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(x[d])));
    quant_row_fp8(xr, 128, q_fp8 + int64_t(r) * dim, q_scale + r);
  }
}

template <typename Acc>
void compress_pool(const uint16_t* k, const uint16_t* gate, const float* ape,
                   int kpool, int dim, uint8_t* k_out, float* scale_out) {
  // Per-dimension softmax over the pool's slots, fp32/Acc accumulation.
  std::vector<Acc> x(dim);
  for (int d = 0; d < dim; ++d) {
    Acc max_score = -std::numeric_limits<Acc>::infinity();
    for (int s = 0; s < kpool; ++s) {
      Acc score =
          static_cast<Acc>(bf16_bits_to_float(gate[int64_t(s) * dim + d])) +
          static_cast<Acc>(ape[int64_t(s) * dim + d]);
      max_score = std::max(max_score, score);
    }
    Acc acc = 0;
    Acc denom = 0;
    for (int s = 0; s < kpool; ++s) {
      Acc score =
          static_cast<Acc>(bf16_bits_to_float(gate[int64_t(s) * dim + d])) +
          static_cast<Acc>(ape[int64_t(s) * dim + d]);
      Acc prob = std::exp(score - max_score);
      denom += prob;
      acc += static_cast<Acc>(bf16_bits_to_float(k[int64_t(s) * dim + d])) *
             prob;
    }
    x[d] = acc / denom;
  }

  // bf16 round, Hadamard-128, bf16 round, fp8 quant — the reference's exact
  // boundary sequence.
  float xr[128];
  for (int d = 0; d < 128; ++d)
    xr[d] = bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(x[d])));
  Acc h[128];
  for (int d = 0; d < 128; ++d) h[d] = static_cast<Acc>(xr[d]);
  fwht128(h);
  for (int d = 0; d < 128; ++d)
    xr[d] = bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(h[d])));
  quant_row_fp8(xr, 128, k_out, scale_out);
}

template <typename Acc>
void pool_logits(const uint8_t* q_fp8, const float* w, const uint8_t* k_fp8,
                 const float* k_scale, int64_t num_pools, int heads, int dim,
                 float* logits) {
  for (int64_t j = 0; j < num_pools; ++j) {
    Acc total = 0;
    for (int h = 0; h < heads; ++h) {
      Acc dot = 0;
      for (int d = 0; d < dim; ++d)
        dot += static_cast<Acc>(fp8_e4m3_bits_to_float(
                   q_fp8[(int64_t(h) * dim) + d])) *
               static_cast<Acc>(fp8_e4m3_bits_to_float(
                   k_fp8[j * dim + d]));
      total += static_cast<Acc>(w[h]) * static_cast<Acc>(k_scale[j]) * dot;
    }
    logits[j] = static_cast<float>(total);
  }
}

int select_pools(const float* logits, int64_t num_pools, int select_k,
                 int32_t* out_pool_ids) {
  const int n = static_cast<int>(std::min<int64_t>(select_k, num_pools));
  if (n <= 0) return 0;
  // Composite key: (~sortable_fp32 << 21) | pool_idx — a total order where
  // the smallest key is the highest logit (the ~ inverts the ascending
  // float order), exact ties resolve to the lower pool index, and selection
  // is deterministic on any implementation.
  std::vector<uint64_t> keys(static_cast<size_t>(num_pools));
  for (int64_t j = 0; j < num_pools; ++j)
    keys[static_cast<size_t>(j)] =
        (static_cast<uint64_t>(~sortable_f32(logits[j])) << 21) |
        static_cast<uint64_t>(j);
  std::partial_sort(keys.begin(), keys.begin() + n, keys.end());
  for (int i = 0; i < n; ++i) out_pool_ids[i] = int32_t(keys[i] & 0x1FFFFFull);
  // Output ascending in pool index (matches the sorted composite order for
  // distinct logits; ties were resolved to lower indices, so ascending
  // holds unconditionally after extraction).
  std::sort(out_pool_ids, out_pool_ids + n);
  return n;
}

int expand_append_tail(const int32_t* pool_ids, int n_sel, int64_t pos,
                       int kpool, int max_selected, int32_t* out_tokens) {
  int w = 0;
  for (int i = 0; i < n_sel; ++i)
    for (int s = 0; s < kpool; ++s) out_tokens[w++] = pool_ids[i] * kpool + s;
  const int64_t seq_len = pos + 1;
  const int64_t tail_start = (seq_len / kpool) * kpool;
  for (int64_t t = tail_start; t < seq_len; ++t) out_tokens[w++] = int32_t(t);
  for (int i = w; i < max_selected; ++i) out_tokens[i] = -1;
  return w;
}

int causal_all_tokens(int64_t pos, int max_selected, int32_t* out_tokens) {
  int w = 0;
  for (int64_t t = 0; t <= pos; ++t) out_tokens[w++] = int32_t(t);
  for (int i = w; i < max_selected; ++i) out_tokens[i] = -1;
  return w;
}

template <typename Acc>
void absorbed_attn(const uint16_t* q, const uint16_t* latent,
                   int64_t latent_stride, const int32_t* tokens, int n_sel,
                   const uint16_t* kv_b, int local_heads, int nope, int v,
                   int kv_lora, float scale, uint16_t* out) {
  const int head_rows = nope + v;  // kv_b rows per head
  for (int h = 0; h < local_heads; ++h) {
    const uint16_t* w_uk = kv_b + int64_t(h) * head_rows * kv_lora;

    // Absorbed query q~[c] = sum_d q_h[d] * W_uk[d][c], bf16 GEMM rounding.
    std::vector<uint16_t> q_tilde(kv_lora);
    for (int c = 0; c < kv_lora; ++c) {
      Acc acc = 0;
      for (int d = 0; d < nope; ++d)
        acc += static_cast<Acc>(bf16_bits_to_float(q[int64_t(h) * nope + d])) *
               static_cast<Acc>(
                   bf16_bits_to_float(w_uk[int64_t(d) * kv_lora + c]));
      q_tilde[c] = float_to_bf16_bits(static_cast<float>(acc));
    }

    // Scores over the selected tokens; -1 slots are skipped.
    std::vector<Acc> s(n_sel);
    for (int t = 0; t < n_sel; ++t) {
      if (tokens[t] < 0) {
        s[t] = -std::numeric_limits<Acc>::infinity();
        continue;
      }
      const uint16_t* lat = latent + int64_t(tokens[t]) * latent_stride;
      Acc acc = 0;
      for (int c = 0; c < kv_lora; ++c)
        acc += static_cast<Acc>(bf16_bits_to_float(q_tilde[c])) *
               static_cast<Acc>(bf16_bits_to_float(lat[c]));
      s[t] = acc * static_cast<Acc>(scale);
    }

    // Softmax over valid slots only.
    Acc m = -std::numeric_limits<Acc>::infinity();
    for (int t = 0; t < n_sel; ++t) m = std::max(m, s[t]);
    std::vector<Acc> p(n_sel);
    Acc denom = 0;
    for (int t = 0; t < n_sel; ++t) {
      p[t] = (tokens[t] < 0) ? Acc(0) : std::exp(s[t] - m);
      denom += p[t];
    }
    for (int t = 0; t < n_sel; ++t) p[t] /= denom;

    // c_h = sum_t bf16(p_t) * latent_t — probs round to bf16 for the
    // accumulation MMA, matching the device path.
    std::vector<Acc> c(kv_lora, Acc(0));
    for (int t = 0; t < n_sel; ++t) {
      if (tokens[t] < 0) continue;
      const uint16_t* lat = latent + int64_t(tokens[t]) * latent_stride;
      Acc pt = static_cast<Acc>(bf16_bits_to_float(
          float_to_bf16_bits(static_cast<float>(p[t]))));
      for (int cc = 0; cc < kv_lora; ++cc)
        c[cc] += pt * static_cast<Acc>(bf16_bits_to_float(lat[cc]));
    }

    // out_h = W_uv_h . c_h, bf16 GEMM rounding.
    const uint16_t* w_uv = kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
    for (int d = 0; d < v; ++d) {
      Acc acc = 0;
      for (int cc = 0; cc < kv_lora; ++cc)
        acc += static_cast<Acc>(bf16_bits_to_float(
                   w_uv[int64_t(d) * kv_lora + cc])) *
               static_cast<Acc>(c[cc]);
      out[int64_t(h) * v + d] =
          float_to_bf16_bits(static_cast<float>(acc));
    }
  }
}

namespace {

template <typename Acc>
void rmsnorm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int dim,
                  float eps) {
  Acc ss = 0;
  for (int d = 0; d < dim; ++d)
    ss += static_cast<Acc>(bf16_bits_to_float(x[d])) *
          static_cast<Acc>(bf16_bits_to_float(x[d]));
  Acc inv = 1 / std::sqrt(ss / dim + static_cast<Acc>(eps));
  for (int d = 0; d < dim; ++d)
    y[d] = float_to_bf16_bits(static_cast<float>(
        static_cast<Acc>(bf16_bits_to_float(x[d])) * inv *
        static_cast<Acc>(bf16_bits_to_float(w[d]))));
}

// LayerNorm (fp32 compute, weight+bias upcast, bf16 out) — the indexer's
// k_norm is a full LayerNorm, not RMSNorm.
template <typename Acc>
void layernorm_bf16(const uint16_t* x, const uint16_t* w, const uint16_t* b,
                    uint16_t* y, int dim, float eps) {
  Acc mean = 0;
  for (int d = 0; d < dim; ++d) mean += static_cast<Acc>(bf16_bits_to_float(x[d]));
  mean /= dim;
  Acc var = 0;
  for (int d = 0; d < dim; ++d) {
    Acc t = static_cast<Acc>(bf16_bits_to_float(x[d])) - mean;
    var += t * t;
  }
  var /= dim;
  Acc inv = 1 / std::sqrt(var + static_cast<Acc>(eps));
  for (int d = 0; d < dim; ++d)
    y[d] = float_to_bf16_bits(static_cast<float>(
        (static_cast<Acc>(bf16_bits_to_float(x[d])) - mean) * inv *
            static_cast<Acc>(bf16_bits_to_float(w[d])) +
        static_cast<Acc>(bf16_bits_to_float(b[d]))));
}

}  // namespace

template <typename Acc>
void layer_forward(const HostWeights& w, const DsaConfig& cfg,
                   const uint16_t* hidden_in, HostState& state,
                   int64_t token_start, int tokens, uint16_t* layer_out,
                   int32_t* topk_out) {
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int hidden = cfg.hidden;
  const int kpool = cfg.index_kpool;
  const int select_k = g.select_k;
  const int max_selected = g.max_selected;
  const int heads = cfg.index_n_heads;
  const int idx_dim = cfg.index_head_dim;
  // Combined indexer logit scale: 128^-0.5 * 32^-0.5, exact 1/64 for this
  // checkpoint (power-of-two head counts keep it exact in fp32).
  const float logit_scale =
      static_cast<float>(std::pow(double(idx_dim), -0.5) *
                         std::pow(double(heads), -0.5));

  // ---- projections ----
  // Fused [q_a | kv_a]: [tokens, 1536+512].
  const int qkv_cols = cfg.q_lora_rank + cfg.kv_lora_rank;
  std::vector<uint16_t> qkv_a(size_t(tokens) * qkv_cols);
  gemm_bf16<Acc>(hidden_in, hidden, w.qkv_a, qkv_a.data(), tokens, qkv_cols,
                 hidden);

  // RMSNorms on the split halves.
  std::vector<uint16_t> q_c(size_t(tokens) * cfg.q_lora_rank);
  std::vector<uint16_t> latent_rows(size_t(tokens) * cfg.kv_lora_rank);
  for (int t = 0; t < tokens; ++t) {
    rmsnorm_bf16<Acc>(&qkv_a[size_t(t) * qkv_cols], w.q_aln,
                      &q_c[size_t(t) * cfg.q_lora_rank], cfg.q_lora_rank,
                      cfg.rms_norm_eps);
    rmsnorm_bf16<Acc>(&qkv_a[size_t(t) * qkv_cols + cfg.q_lora_rank],
                      w.kv_aln, &latent_rows[size_t(t) * cfg.kv_lora_rank],
                      cfg.kv_lora_rank, cfg.rms_norm_eps);
  }

  // Latent cache append.
  std::memcpy(&state.latent[size_t(token_start) * cfg.kv_lora_rank],
              latent_rows.data(), size_t(tokens) * cfg.kv_lora_rank * 2);
  state.num_tokens = std::max(state.num_tokens, token_start + tokens);

  // q (MLA) and indexer q, both from the normed q-lora.
  std::vector<uint16_t> q(size_t(tokens) * g.local_q_rows);
  gemm_bf16<Acc>(q_c.data(), cfg.q_lora_rank, w.q_b, q.data(), tokens,
                 g.local_q_rows, cfg.q_lora_rank);
  std::vector<uint16_t> q_idx(size_t(tokens) * heads * idx_dim);
  gemm_bf16<Acc>(q_c.data(), cfg.q_lora_rank, w.wq_b, q_idx.data(), tokens,
                 heads * idx_dim, cfg.q_lora_rank);

  // Indexer k (LayerNorm) and gate from hidden; weights in fp32 (no bf16
  // rounding — the reference recomputes this projection in fp32).
  std::vector<uint16_t> k_rows(size_t(tokens) * idx_dim);
  {
    std::vector<uint16_t> k_raw(size_t(tokens) * idx_dim);
    gemm_bf16<Acc>(hidden_in, hidden, w.wk, k_raw.data(), tokens, idx_dim,
                   hidden);
    for (int t = 0; t < tokens; ++t)
      layernorm_bf16<Acc>(&k_raw[size_t(t) * idx_dim], w.k_norm_w, w.k_norm_b,
                          &k_rows[size_t(t) * idx_dim], idx_dim, 1e-6f);
  }
  std::vector<uint16_t> gate_rows(size_t(tokens) * idx_dim);
  gemm_bf16<Acc>(hidden_in, hidden, w.gate, gate_rows.data(), tokens, idx_dim,
                 hidden);
  std::vector<float> weights(size_t(tokens) * heads);
  for (int t = 0; t < tokens; ++t)
    for (int h = 0; h < heads; ++h) {
      Acc acc = 0;
      for (int i = 0; i < hidden; ++i)
        acc += static_cast<Acc>(bf16_bits_to_float(
                   hidden_in[size_t(t) * hidden + i])) *
               static_cast<Acc>(
                   bf16_bits_to_float(w.wp[int64_t(h) * hidden + i]));
      weights[size_t(t) * heads + h] = static_cast<float>(acc);
    }

  // Indexer q path: Hadamard-128 + fp8 quant, then fold scales into weights.
  std::vector<uint8_t> q_fp8(size_t(tokens) * heads * idx_dim);
  std::vector<float> q_scale(size_t(tokens) * heads);
  fwht128_quant_fp8<Acc>(q_idx.data(), tokens * heads, idx_dim,
                         q_fp8.data(), q_scale.data());
  std::vector<float> w_folded(size_t(tokens) * heads);
  for (int t = 0; t < tokens; ++t)
    for (int h = 0; h < heads; ++h) {
      float base = weights[size_t(t) * heads + h] *
                   q_scale[size_t(t) * heads + h];
      w_folded[size_t(t) * heads + h] = base * logit_scale;
    }

  // ---- pool writes: complete pools fully inside this chunk ----
  // Chunk starts are pool-aligned; the final chunk may end mid-pool.
  const int64_t pool_lo = token_start / kpool;
  const int64_t pool_hi = (token_start + tokens) / kpool;
  for (int64_t j = pool_lo; j < pool_hi; ++j) {
    compress_pool<Acc>(&k_rows[size_t(j * kpool - token_start) * idx_dim],
                       &gate_rows[size_t(j * kpool - token_start) * idx_dim],
                       w.ape, kpool, idx_dim,
                       &state.index_k[size_t(j) * idx_dim],
                       &state.index_scale[j]);
  }
  state.num_pools = std::max(state.num_pools, pool_hi);

  // ---- tail seed: the request's last kpool tokens so far ----
  const int64_t end = token_start + tokens;
  for (int64_t pos = std::max<int64_t>(0, end - kpool); pos < end; ++pos) {
    const int slot = int(pos % kpool);
    std::memcpy(&state.tail[size_t(slot) * idx_dim],
                &k_rows[size_t(pos - token_start) * idx_dim], idx_dim * 2);
    std::memcpy(&state.tail[size_t(kpool + slot) * idx_dim],
                &gate_rows[size_t(pos - token_start) * idx_dim], idx_dim * 2);
  }

  // ---- per-query selection and attention ----
  std::vector<float> logits;
  std::vector<int32_t> pool_ids(select_k);
  std::vector<uint16_t> attn_out(size_t(tokens) * g.local_v_rows);
  for (int t = 0; t < tokens; ++t) {
    const int64_t pos = token_start + t;
    const int64_t visible = (pos + 1) / kpool;
    int n_tokens = 0;
    if (visible <= select_k) {
      n_tokens = causal_all_tokens(pos, max_selected,
                                   topk_out + size_t(t) * max_selected);
    } else {
      logits.assign(visible, 0.0f);
      pool_logits<Acc>(&q_fp8[size_t(t) * heads * idx_dim],
                       &w_folded[size_t(t) * heads], state.index_k.data(),
                       state.index_scale.data(), visible, heads, idx_dim,
                       logits.data());
      const int n_sel = select_pools(logits.data(), visible, select_k,
                                     pool_ids.data());
      n_tokens = expand_append_tail(pool_ids.data(), n_sel, pos, kpool,
                                    max_selected,
                                    topk_out + size_t(t) * max_selected);
    }
    absorbed_attn<Acc>(&q[size_t(t) * g.local_q_rows], state.latent.data(),
                       cfg.kv_lora_rank,
                       topk_out + size_t(t) * max_selected, n_tokens, w.kv_b,
                       g.local_heads, cfg.qk_nope_head_dim, cfg.v_head_dim,
                       cfg.kv_lora_rank, g.local_heads > 0
                           ? 1.0f / std::sqrt(float(cfg.qk_nope_head_dim))
                           : 1.0f,
                       &attn_out[size_t(t) * g.local_v_rows]);
  }

  // ---- output projection ----
  gemm_bf16<Acc>(attn_out.data(), g.local_v_rows, w.o_proj, layer_out, tokens,
                 hidden, g.local_v_rows);
}

// Explicit instantiations for the two oracle precisions.
template void gemm_bf16<float>(const uint16_t*, int64_t, const uint16_t*,
                               uint16_t*, int, int, int);
template void gemm_bf16<double>(const uint16_t*, int64_t, const uint16_t*,
                                uint16_t*, int, int, int);
template void fwht128<float>(float*);
template void fwht128<double>(double*);
template void fwht128_quant_fp8<float>(const uint16_t*, int, int, uint8_t*,
                                       float*);
template void fwht128_quant_fp8<double>(const uint16_t*, int, int, uint8_t*,
                                        float*);
template void compress_pool<float>(const uint16_t*, const uint16_t*,
                                   const float*, int, int, uint8_t*, float*);
template void compress_pool<double>(const uint16_t*, const uint16_t*,
                                    const float*, int, int, uint8_t*, float*);
template void pool_logits<float>(const uint8_t*, const float*, const uint8_t*,
                                 const float*, int64_t, int, int, float*);
template void pool_logits<double>(const uint8_t*, const float*, const uint8_t*,
                                  const float*, int64_t, int, int, float*);
template void absorbed_attn<float>(const uint16_t*, const uint16_t*, int64_t,
                                   const int32_t*, int, const uint16_t*, int,
                                   int, int, int, float, uint16_t*);
template void absorbed_attn<double>(const uint16_t*, const uint16_t*, int64_t,
                                    const int32_t*, int, const uint16_t*, int,
                                    int, int, int, float, uint16_t*);
template void layer_forward<float>(const HostWeights&, const DsaConfig&,
                                   const uint16_t*, HostState&, int64_t, int,
                                   uint16_t*, int32_t*);
template void layer_forward<double>(const HostWeights&, const DsaConfig&,
                                    const uint16_t*, HostState&, int64_t, int,
                                    uint16_t*, int32_t*);

}  // namespace dgpp::dsa_ref
