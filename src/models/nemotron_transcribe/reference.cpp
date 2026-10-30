#include "models/nemotron_transcribe/reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dgpp {
namespace nemotron_ref {
namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// The HF topk(2) group score: the two largest biased scores in a group.
float top2_sum(const float* v, size_t n) {
  float a = -kInf, b = -kInf;  // a >= b
  for (size_t i = 0; i < n; ++i) {
    if (v[i] > a) { b = a; a = v[i]; } else if (v[i] > b) { b = v[i]; }
  }
  return (a == -kInf ? 0.f : a) + (b == -kInf ? 0.f : b);
}

// The largest index-first argmax over a mask-allowed pool.
size_t argmax(const std::vector<float>& v, const std::vector<uint8_t>& allowed,
              const std::vector<uint8_t>& taken) {
  size_t best = v.size();
  for (size_t i = 0; i < v.size(); ++i) {
    if (!allowed[i] || taken[i]) continue;
    if (best == v.size() || v[i] > v[best]) best = i;  // ties: lowest index
  }
  if (best == v.size()) throw std::runtime_error("router: empty pool");
  return best;
}

}  // namespace

float softplus(float x) { return x > 20.f ? x : std::log1pf(std::exp(x)); }
float silu(float x) { return x / (1.f + std::exp(-x)); }

void gemv(float* y, const Tensor& W, const float* x, size_t n, size_t k) {
  if (W.numel() != n * k)
    throw std::invalid_argument("gemv: weight numel " + std::to_string(W.numel()) + " != " +
                                std::to_string(n) + "x" + std::to_string(k));
  for (size_t i = 0; i < n; ++i) {
    double acc = 0.0;  // fp64 accumulator: the fp32 reference sums, order-pinned
    for (size_t j = 0; j < k; ++j) acc += static_cast<double>(W.at(i * k + j)) * x[j];
    y[i] = static_cast<float>(acc);
  }
}

void rmsnorm(const float* x, const Tensor& w, float eps, size_t n, float* y) {
  if (w.numel() != n) throw std::invalid_argument("rmsnorm: weight width");
  double ss = 0.0;
  for (size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float rstd = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
  for (size_t i = 0; i < n; ++i) y[i] = w.at(i) * (x[i] * rstd);
}

void rmsnorm_gated(const float* y, const float* gate, const Tensor& w, size_t n, size_t group,
                   float eps, float* out) {
  if (w.numel() != n) throw std::invalid_argument("rmsnorm_gated: weight width");
  if (group == 0 || n % group != 0) throw std::invalid_argument("rmsnorm_gated: group width");
  for (size_t g0 = 0; g0 < n; g0 += group) {
    double ss = 0.0;
    for (size_t i = g0; i < g0 + group; ++i) {
      const float t = y[i] * silu(gate[i]);
      ss += static_cast<double>(t) * t;
    }
    const float rstd = static_cast<float>(1.0 / std::sqrt(ss / group + eps));
    for (size_t i = g0; i < g0 + group; ++i) {
      out[i] = w.at(i) * ((y[i] * silu(gate[i])) * rstd);
    }
  }
}

void causal_conv_silu(const float* u, size_t seq, size_t channels, const Tensor& weight,
                      const Tensor* bias, size_t kernel, float* out) {
  if (weight.numel() != channels * kernel)
    throw std::invalid_argument("conv: weight numel");
  if (bias && bias->numel() != channels) throw std::invalid_argument("conv: bias width");
  for (size_t t = 0; t < seq; ++t) {
    for (size_t c = 0; c < channels; ++c) {
      double acc = bias ? bias->at(c) : 0.0;
      for (size_t k = 0; k < kernel; ++k) {
        const size_t j = t >= (kernel - 1 - k) ? t - (kernel - 1 - k) : SIZE_MAX;
        if (j == SIZE_MAX) continue;
        acc += static_cast<double>(weight.at(c * kernel + k)) * u[j * channels + c];
      }
      out[t * channels + c] = silu(static_cast<float>(acc));
    }
  }
}

void mamba2_walk(const MambaGeom& g, const float* x, const float* B, const float* C,
                 const float* dt, const float* A_log, const float* D, const float* dt_bias,
                 size_t seq, float* out) {
  if (g.heads == 0 || g.head_dim == 0 || g.state == 0 || g.heads % g.groups != 0)
    throw std::invalid_argument("mamba2_walk: geometry");
  const size_t d_inner = g.heads * g.head_dim;
  std::vector<float> st(g.head_dim * g.state);
  for (size_t h = 0; h < g.heads; ++h) {
    const size_t grp = h * g.groups / g.heads;
    const float a = -std::exp(A_log[h]);
    std::fill(st.begin(), st.end(), 0.f);
    for (size_t t = 0; t < seq; ++t) {
      const float* Bh = B + t * g.groups * g.state + grp * g.state;
      const float* Ch = C + t * g.groups * g.state + grp * g.state;
      float d = dt[t * g.heads + h] + dt_bias[h];
      d = softplus(d);
      if (d < g.dt_min) d = g.dt_min;
      const float dA = std::exp(d * a);
      for (size_t p = 0; p < g.head_dim; ++p) {
        const float xv = x[t * d_inner + h * g.head_dim + p];
        float* row = st.data() + p * g.state;
        float o = 0.f;
        for (size_t n = 0; n < g.state; ++n) {
          row[n] = row[n] * dA + d * Bh[n] * xv;
          o += row[n] * Ch[n];
        }
        out[t * d_inner + h * g.head_dim + p] = o + D[h] * xv;
      }
    }
  }
}

RouterResult router_select(const float* x, const Tensor& gate, const float* bias, size_t E,
                           size_t H, size_t topk, size_t n_group, size_t topk_group,
                           bool norm_topk, float scale) {
  if (gate.numel() != E * H) throw std::invalid_argument("router: gate numel");
  if (bias && E % n_group != 0) throw std::invalid_argument("router: group geometry");
  std::vector<float> logits(E), scores(E), biased(E);
  gemv(logits.data(), gate, x, E, H);
  for (size_t e = 0; e < E; ++e) {
    scores[e] = 1.f / (1.f + std::exp(-logits[e]));
    biased[e] = scores[e] + (bias ? bias[e] : 0.f);
  }
  std::vector<uint8_t> allowed(E, 1);
  if (n_group > 1) {
    const size_t per = E / n_group;
    std::vector<float> gs(n_group);
    for (size_t gr = 0; gr < n_group; ++gr) gs[gr] = top2_sum(biased.data() + gr * per, per);
    // topk_group best groups, ties by lowest group index.
    std::vector<size_t> ord(n_group);
    for (size_t i = 0; i < n_group; ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(),
                     [&](size_t a, size_t b) { return gs[a] > gs[b]; });
    std::fill(allowed.begin(), allowed.end(), 0);
    for (size_t i = 0; i < topk_group && i < n_group; ++i)
      std::fill(allowed.begin() + ord[i] * per, allowed.begin() + (ord[i] + 1) * per, 1);
  }
  RouterResult r;
  std::vector<uint8_t> taken(E, 0);
  double den = 0.0;
  for (size_t i = 0; i < topk; ++i) {
    const size_t e = argmax(biased, allowed, taken);
    taken[e] = 1;
    r.index.push_back(e);
    r.weight.push_back(scores[e]);
    den += scores[e];
  }
  if (norm_topk) {
    const float inv = static_cast<float>(1.0 / (den + 1e-20));
    for (auto& wgt : r.weight) wgt *= inv;
  }
  for (auto& wgt : r.weight) wgt *= scale;
  // Canonical order: ascending expert index (the reference's topk is
  // unsorted; the sum is order-free at this precision).
  std::vector<size_t> ord(r.index.size());
  for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
  std::stable_sort(ord.begin(), ord.end(),
                   [&](size_t a, size_t b) { return r.index[a] < r.index[b]; });
  RouterResult sorted;
  for (size_t i : ord) {
    sorted.index.push_back(r.index[i]);
    sorted.weight.push_back(r.weight[i]);
  }
  return sorted;
}

void relu2(float* v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = v[i] > 0.f ? v[i] * v[i] : 0.f;
}

void relu2_mlp(const float* x, const Tensor& up, const Tensor& down, size_t H, size_t I,
               float* out) {
  if (up.numel() != I * H || down.numel() != H * I)
    throw std::invalid_argument("relu2_mlp: weight numel");
  std::vector<float> inter(I);
  gemv(inter.data(), up, x, I, H);
  relu2(inter.data(), I);
  gemv(out, down, inter.data(), H, I);
}

void experts_sum(const float* x, const RouterResult& sel, const Tensor& gate_up,
                 const Tensor& down, size_t E, size_t H, size_t I, float* out) {
  if (gate_up.numel() != E * H * I || down.numel() != E * I * H)
    throw std::invalid_argument("experts_sum: stacked numel");
  std::fill(out, out + H, 0.f);
  std::vector<float> inter(I), row(H);
  for (size_t s = 0; s < sel.index.size(); ++s) {
    const size_t e = sel.index[s];
    // gate_and_up[e]: [H, I] — inter[j] = dot(x, gu[e][., j]) over H.
    for (size_t j = 0; j < I; ++j) {
      double acc = 0.0;
      for (size_t h = 0; h < H; ++h)
        acc += static_cast<double>(gate_up.at(e * H * I + h * I + j)) * x[h];
      const float v = static_cast<float>(acc);
      inter[j] = v > 0.f ? v * v : 0.f;
    }
    // down[e]: [I, H].
    for (size_t h = 0; h < H; ++h) {
      double acc = 0.0;
      for (size_t j = 0; j < I; ++j)
        acc += static_cast<double>(down.at(e * I * H + j * H + h)) * inter[j];
      row[h] = static_cast<float>(acc) * sel.weight[s];
    }
    for (size_t h = 0; h < H; ++h) out[h] += row[h];
  }
}

static void rope_rows(float* rows, size_t seq, size_t heads, size_t head_dim, size_t pos0,
                      double theta) {
  const size_t half = head_dim / 2;
  std::vector<float> cos_v(half), sin_v(half);
  for (size_t t = 0; t < seq; ++t) {
    const double pos = static_cast<double>(pos0 + t);
    for (size_t i = 0; i < half; ++i) {
      const double invf = std::pow(theta, -2.0 * static_cast<double>(i) /
                                                 static_cast<double>(head_dim));
      const float ang = static_cast<float>(pos * invf);
      cos_v[i] = std::cos(ang);
      sin_v[i] = std::sin(ang);
    }
    for (size_t h = 0; h < heads; ++h) {
      float* row = rows + t * heads * head_dim + h * head_dim;
      for (size_t i = 0; i < half; ++i) {
        const float x1 = row[i], x2 = row[half + i];
        row[i] = x1 * cos_v[i] - x2 * sin_v[i];
        row[half + i] = x2 * cos_v[i] + x1 * sin_v[i];
      }
    }
  }
}

void rope_neox(float* q, size_t seq, size_t heads, size_t head_dim, size_t pos0, double theta) {
  rope_rows(q, seq, heads, head_dim, pos0, theta);
}
void rope_neox_k(float* k, size_t seq, size_t kv_heads, size_t head_dim, size_t pos0,
                 double theta) {
  rope_rows(k, seq, kv_heads, head_dim, pos0, theta);
}

void gqa_attention(const float* q, const float* k, const float* v, size_t seq, size_t heads,
                   size_t kv_heads, size_t head_dim, float scale, float* out) {
  if (heads % kv_heads != 0) throw std::invalid_argument("gqa: head count");
  std::vector<float> scores(seq), probs(seq);
  for (size_t t = 0; t < seq; ++t) {
    for (size_t h = 0; h < heads; ++h) {
      const size_t kv = h * kv_heads / heads;
      const float* qh = q + t * heads * head_dim + h * head_dim;
      float mx = -kInf;
      for (size_t s = 0; s <= t; ++s) {
        double acc = 0.0;
        const float* kh = k + s * kv_heads * head_dim + kv * head_dim;
        for (size_t d = 0; d < head_dim; ++d) acc += static_cast<double>(qh[d]) * kh[d];
        scores[s] = static_cast<float>(acc) * scale;
        mx = std::max(mx, scores[s]);
      }
      double den = 0.0;
      for (size_t s = 0; s <= t; ++s) {
        probs[s] = std::exp(scores[s] - mx);
        den += probs[s];
      }
      const float inv = static_cast<float>(1.0 / den);
      float* oh = out + t * heads * head_dim + h * head_dim;
      for (size_t d = 0; d < head_dim; ++d) oh[d] = 0.f;
      for (size_t s = 0; s <= t; ++s) {
        const float p = probs[s] * inv;
        const float* vh = v + s * kv_heads * head_dim + kv * head_dim;
        for (size_t d = 0; d < head_dim; ++d) oh[d] += p * vh[d];
      }
    }
  }
}

void mixer_mamba(const NemotronTranscribeLmConfig& lm, const Weights& w,
                 const std::string& prefix, const float* x, size_t seq, float* out) {
  if (lm.mamba_hidden_act != "silu")
    throw std::invalid_argument("mixer_mamba: activation " + lm.mamba_hidden_act);
  const size_t H = lm.hidden_size;
  const size_t d_inner = static_cast<size_t>(lm.mamba_num_heads) * lm.mamba_head_dim;
  const size_t groups = lm.mamba_num_groups, state = lm.mamba_state_size;
  const size_t conv_dim = d_inner + 2 * groups * state;
  const size_t proj = 2 * d_inner + 2 * groups * state + lm.mamba_num_heads;
  const Tensor& in_proj = w.at(prefix + "in_proj.weight");
  std::vector<float> full(seq * proj);
  for (size_t t = 0; t < seq; ++t)
    gemv(full.data() + t * proj, in_proj, x + t * H, proj, H);
  std::vector<float> gate(full.size()), xbc(seq * conv_dim), conv_out(seq * conv_dim);
  // in_proj rows: [gate | x B C | dt].
  std::vector<float> dt(seq * lm.mamba_num_heads);
  for (size_t t = 0; t < seq; ++t) {
    const float* row = full.data() + t * proj;
    std::copy(row, row + d_inner, gate.data() + t * d_inner);
    std::copy(row + d_inner, row + d_inner + conv_dim, xbc.data() + t * conv_dim);
    std::copy(row + d_inner + conv_dim, row + proj, dt.data() + t * lm.mamba_num_heads);
  }
  const Tensor* conv_bias = nullptr;
  auto bit = w.find(prefix + "conv1d.bias");
  if (bit != w.end()) conv_bias = &bit->second;
  causal_conv_silu(xbc.data(), seq, conv_dim, w.at(prefix + "conv1d.weight"), conv_bias,
                   lm.conv_kernel, conv_out.data());
  MambaGeom g;
  g.heads = lm.mamba_num_heads;
  g.head_dim = lm.mamba_head_dim;
  g.groups = groups;
  g.state = state;
  g.dt_min = static_cast<float>(lm.time_step_min);
  std::vector<float> y(seq * d_inner);
  const auto f32_at = [&](const std::string& name) {
    const Tensor& T = w.at(name);
    if (T.f32.empty()) throw std::invalid_argument("mixer_mamba: " + name + " must be F32");
    return T.f32.data();
  };
  {
    // The scan wants contiguous [seq, width] rows per stream — reslice the
    // conv output's interleaved [x | B | C] rows.
    std::vector<float> xs(seq * d_inner), Bs(seq * groups * state), Cs(seq * groups * state);
    for (size_t t = 0; t < seq; ++t) {
      const float* row = conv_out.data() + t * conv_dim;
      std::copy(row, row + d_inner, xs.data() + t * d_inner);
      std::copy(row + d_inner, row + d_inner + groups * state, Bs.data() + t * groups * state);
      std::copy(row + d_inner + groups * state, row + conv_dim, Cs.data() + t * groups * state);
    }
    mamba2_walk(g, xs.data(), Bs.data(), Cs.data(), dt.data(),
                f32_at(prefix + "_fp32_params.A_log"), f32_at(prefix + "_fp32_params.D"),
                f32_at(prefix + "_fp32_params.dt_bias"), seq, y.data());
  }
  std::vector<float> normed(seq * d_inner);
  for (size_t t = 0; t < seq; ++t)
    rmsnorm_gated(y.data() + t * d_inner, gate.data() + t * d_inner, w.at(prefix + "norm.weight"),
                  d_inner, d_inner / groups, static_cast<float>(lm.norm_eps),
                  normed.data() + t * d_inner);
  const Tensor& out_proj = w.at(prefix + "out_proj.weight");
  for (size_t t = 0; t < seq; ++t)
    gemv(out + t * H, out_proj, normed.data() + t * d_inner, H, d_inner);
}

void mixer_attention(const NemotronTranscribeLmConfig& lm, const Weights& w,
                     const std::string& prefix, const float* x, size_t seq, float* out) {
  const size_t H = lm.hidden_size, hd = lm.head_dim;
  const size_t qn = static_cast<size_t>(lm.num_attention_heads) * hd;
  const size_t kn = static_cast<size_t>(lm.num_key_value_heads) * hd;
  std::vector<float> q(seq * qn), k(seq * kn), vv(seq * kn), attn(seq * qn);
  const Tensor& Wq = w.at(prefix + "q_proj.weight");
  const Tensor& Wk = w.at(prefix + "k_proj.weight");
  const Tensor& Wv = w.at(prefix + "v_proj.weight");
  for (size_t t = 0; t < seq; ++t) {
    gemv(q.data() + t * qn, Wq, x + t * H, qn, H);
    gemv(k.data() + t * kn, Wk, x + t * H, kn, H);
    gemv(vv.data() + t * kn, Wv, x + t * H, kn, H);
  }
  // No rotation: the hybrid's attention layers are position-free (paper
  // §2.1); positions reach them only through the preceding Mamba states.
  gqa_attention(q.data(), k.data(), vv.data(), seq, lm.num_attention_heads,
                lm.num_key_value_heads, hd, static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd))),
                attn.data());
  const Tensor& Wo = w.at(prefix + "o_proj.weight");
  for (size_t t = 0; t < seq; ++t) gemv(out + t * H, Wo, attn.data() + t * qn, H, qn);
}

void mixer_moe(const NemotronTranscribeLmConfig& lm, const Weights& w, const std::string& prefix,
               const float* x, size_t seq, float* out) {
  if (lm.mlp_hidden_act != "relu2")
    throw std::invalid_argument("mixer_moe: activation " + lm.mlp_hidden_act);
  if (lm.mlp_bias) throw std::invalid_argument("mixer_moe: bias form unsupported");
  const size_t H = lm.hidden_size, E = lm.num_routed_experts, I = lm.moe_intermediate_size;
  const size_t SI = lm.moe_shared_expert_intermediate_size;
  const Tensor& gate = w.at(prefix + "gate.weight");
  const Tensor& gu = w.at(prefix + "experts.gate_and_up_projs");
  const Tensor& dn = w.at(prefix + "experts.down_projs");
  // The bias lives beside the gate, F32.
  std::vector<float> bias_store;
  const float* bias = nullptr;
  auto bit = w.find(prefix + "gate.e_score_correction_bias");
  if (bit != w.end()) {
    if (bit->second.f32.empty()) throw std::invalid_argument("mixer_moe: bias must be F32");
    bias = bit->second.f32.data();
  } else {
    (void)bias_store;
  }
  for (size_t t = 0; t < seq; ++t) {
    auto sel = router_select(x + t * H, gate, bias, E, H, lm.num_experts_per_tok,
                             lm.moe_n_groups, lm.moe_topk_group, lm.norm_topk_prob,
                             static_cast<float>(lm.routed_scaling_factor));
    experts_sum(x + t * H, sel, gu, dn, E, H, I, out + t * H);
    std::vector<float> shared(H);
    relu2_mlp(x + t * H, w.at(prefix + "shared_experts.up_proj.weight"),
              w.at(prefix + "shared_experts.down_proj.weight"), H, SI, shared.data());
    for (size_t h = 0; h < H; ++h) out[t * H + h] += shared[h];
  }
}

WalkResult walk(const NemotronTranscribeLmConfig& lm, const Weights& w, const uint32_t* ids,
                size_t seq, bool record, bool last_logits_only) {
  const size_t H = lm.hidden_size;
  if (lm.slots.empty() || lm.slots.size() != static_cast<size_t>(lm.num_layers))
    throw std::invalid_argument("walk: slot census");
  if (!lm.mtp_slots.empty()) {
    // The MTP block rides separately (milestone 7): the base walk ignores it.
  }
  WalkResult res;
  const Tensor& embed = w.at("llm.model.embed_tokens.weight");
  std::vector<float> hidden(seq * H);
  for (size_t t = 0; t < seq; ++t) {
    const uint32_t id = ids[t];
    if (id >= static_cast<uint32_t>(lm.vocab_size))
      throw std::invalid_argument("walk: token id out of the vocab");
    for (size_t h = 0; h < H; ++h) hidden[t * H + h] = embed.at(static_cast<size_t>(id) * H + h);
  }
  res.normed.resize(lm.slots.size());
  res.mixer_out.resize(lm.slots.size());
  for (size_t i = 0; i < lm.slots.size(); ++i) {
    const std::string p = "llm.model.layers." + std::to_string(i) + ".";
    std::vector<float> normed(seq * H);
    for (size_t t = 0; t < seq; ++t)
      rmsnorm(hidden.data() + t * H, w.at(p + "norm.weight"), static_cast<float>(lm.norm_eps), H,
              normed.data() + t * H);
    std::vector<float> mo(seq * H);
    switch (lm.slots[i]) {
      case NemotronSlot::Mamba:
        mixer_mamba(lm, w, p + "mixer.", normed.data(), seq, mo.data());
        break;
      case NemotronSlot::Attention:
        mixer_attention(lm, w, p + "mixer.", normed.data(), seq, mo.data());
        break;
      case NemotronSlot::Moe:
        mixer_moe(lm, w, p + "mixer.", normed.data(), seq, mo.data());
        break;
    }
    if (record) {
      res.normed[i] = std::move(normed);
      res.mixer_out[i] = mo;
    }
    for (size_t j = 0; j < seq * H; ++j) hidden[j] += mo[j];
  }
  for (size_t t = 0; t < seq; ++t)
    rmsnorm(hidden.data() + t * H, w.at("llm.model.norm.weight"), static_cast<float>(lm.norm_eps),
            H, hidden.data() + t * H);
  const Tensor& head = w.at("llm.lm_head.weight");
  const size_t rows = last_logits_only ? 1 : seq;
  res.logits.resize(rows * lm.vocab_size);
  for (size_t r = 0; r < rows; ++r) {
    const size_t t = last_logits_only ? seq - 1 : r;
    gemv(res.logits.data() + r * lm.vocab_size, head, hidden.data() + t * H, lm.vocab_size, H);
  }
  return res;
}

}  // namespace nemotron_ref
}  // namespace dgpp
