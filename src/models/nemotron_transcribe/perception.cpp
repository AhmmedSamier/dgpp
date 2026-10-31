#include "models/nemotron_transcribe/perception.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace dgpp {
namespace nemotron_perc {
namespace {

// The NeMo frontend constants the release's config leaves at their module
// defaults (features.py __init__): the symmetric Hann arrives through the
// shipped tensor, so only preemph / mag_power / the log guard are here.
constexpr float kPreemph = 0.97f;
constexpr float kLogEpsilon = 0x1p-24f;  // log_zero_guard 'add', 2^-24
constexpr float kLnEps = 1e-5f;          // torch LayerNorm default

inline float deq(const Tensor& t, size_t i) {
  return t.f32.empty() ? bf16_bits_to_float(t.bf16[i]) : t.f32[i];
}

// 256-point complex FFT (radix-2, iterative, in place), the engine of
// rfft512: z[j] = x[2j] + i x[2j+1], then the standard real-FFT split.
void fft256(float* zre, float* zim) {
  // bit-reversal
  for (size_t i = 1, j = 0; i < 256; ++i) {
    size_t bit = 256 >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(zre[i], zre[j]);
      std::swap(zim[i], zim[j]);
    }
  }
  for (size_t len = 2; len <= 256; len <<= 1) {
    const double ang = -2.0 * M_PI / static_cast<double>(len);
    const double wr = std::cos(ang), wi = std::sin(ang);
    for (size_t i = 0; i < 256; i += len) {
      double cur_r = 1.0, cur_i = 0.0;
      for (size_t k = 0; k < len / 2; ++k) {
        const float vr = zre[i + k], vi = zim[i + k];
        const float tr = static_cast<float>(cur_r) * zre[i + k + len / 2] -
                         static_cast<float>(cur_i) * zim[i + k + len / 2];
        const float ti = static_cast<float>(cur_r) * zim[i + k + len / 2] +
                         static_cast<float>(cur_i) * zre[i + k + len / 2];
        zre[i + k] = vr + tr;
        zim[i + k] = vi + ti;
        zre[i + k + len / 2] = vr - tr;
        zim[i + k + len / 2] = vi - ti;
        const double nr = cur_r * wr - cur_i * wi;
        cur_i = cur_r * wi + cur_i * wr;
        cur_r = nr;
      }
    }
  }
}

}  // namespace

void layernorm_row(const float* x, const Tensor& w, const Tensor& b, size_t n, float eps,
                   float* y) {
  if (w.numel() != n || b.numel() != n) throw std::invalid_argument("layernorm: width");
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += x[i];
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(x[i]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
  for (size_t i = 0; i < n; ++i) y[i] = (x[i] - static_cast<float>(mean)) * inv * deq(w, i) + deq(b, i);
}

void rfft512(const float* frame, float* re, float* im) {
  float zre[256], zim[256];
  for (size_t j = 0; j < 256; ++j) {
    zre[j] = frame[2 * j];
    zim[j] = frame[2 * j + 1];
  }
  fft256(zre, zim);
  // Split the packed FFT (z[j] = x[2j] + i x[2j+1]) into the 512-point
  // real transform, bins 0..256: A = DFT(even), B = DFT(odd) recovered
  // from Z and its mirror, then X[k] = A[k] + e^{-2 pi i k / 512} B[k].
  for (size_t k = 0; k <= 256; ++k) {
    const size_t kn = (256 - k) & 255;
    const double z_r = zre[k & 255], z_i = zim[k & 255];
    const double zn_r = zre[kn], zn_i = zim[kn];
    const double a_r = 0.5 * (z_r + zn_r), a_i = 0.5 * (z_i - zn_i);
    const double b_r = 0.5 * (z_i + zn_i), b_i = -0.5 * (z_r - zn_r);
    const double ang = -M_PI * static_cast<double>(k) / 256.0;
    const double w_r = std::cos(ang), w_i = std::sin(ang);
    re[k] = static_cast<float>(a_r + (w_r * b_r - w_i * b_i));
    im[k] = static_cast<float>(a_i + (w_r * b_i + w_i * b_r));
  }
}

size_t mel_frontend(const float* samples, size_t n, const Tensor& window, const Tensor& fb,
                    float preemph, std::vector<float>& mel) {
  // torch.stft center=True pads n_fft/2 zeros each side; frames =
  // floor((n + 2*(n_fft/2) - win) / hop) + 1, i.e. (n + 512 - 400)/160 + 1.
  const size_t T = (n + 112) / 160 + 1;
  if (window.numel() != 400 || fb.numel() != 128 * 257)
    throw std::invalid_argument("mel_frontend: shipped tensor shapes");
  // Preemphasis on the unpadded signal (NeMo forward order: dither(train)
  // -> preemph -> center-padded stft).
  std::vector<float> x(n);
  if (preemph > 0.f) {
    x[0] = n ? samples[0] : 0.f;
    for (size_t i = 1; i < n; ++i) x[i] = samples[i] - preemph * samples[i - 1];
  } else if (n) {
    std::memcpy(x.data(), samples, n * sizeof(float));
  }
  mel.assign(128 * T, 0.f);
  std::vector<float> frame(512, 0.f), re(257), im(257);
  for (size_t t = 0; t < T; ++t) {
    std::fill(frame.begin(), frame.end(), 0.f);
    for (size_t k = 0; k < 400; ++k) {
      const long long s = static_cast<long long>(t) * 160 - 256 + static_cast<long long>(k);
      if (s >= 0 && s < static_cast<long long>(n))
        frame[k] = x[static_cast<size_t>(s)] * deq(window, k);
    }
    rfft512(frame.data(), re.data(), im.data());
    for (size_t m = 0; m < 128; ++m) {
      double acc = 0.0;
      for (size_t k = 0; k < 257; ++k) {
        const float mag = std::sqrt(re[k] * re[k] + im[k] * im[k]);
        acc += static_cast<double>(deq(fb, m * 257 + k)) * static_cast<double>(mag * mag);
      }
      mel[m * T + t] = std::log(static_cast<float>(acc) + kLogEpsilon);
    }
  }
  return T;
}

void stack_pre_encode(const NemotronTranscribePerceptionConfig& pc, const float* mel, size_t T,
                      const Tensor& proj_w, std::vector<float>& out) {
  const size_t C = pc.mel_features, sf = pc.subsampling_factor;
  if (proj_w.numel() == 0 || proj_w.numel() % (sf * C) != 0)
    throw std::invalid_argument("stack_pre_encode: proj numel");
  const size_t D = proj_w.numel() / (sf * C);
  const size_t T_tok = (T + sf - 1) / sf;
  std::vector<float> row(sf * C);
  out.assign(T_tok * D, 0.f);
  for (size_t tt = 0; tt < T_tok; ++tt) {
    for (size_t f = 0; f < sf; ++f) {
      const size_t t = tt * sf + f;
      for (size_t c = 0; c < C; ++c)
        row[f * C + c] = t < T ? mel[c * T + t] : 0.f;  // frame-major stacking
    }
    nemotron_ref::gemv(out.data() + tt * D, proj_w, row.data(), D, sf * C);
  }
}

namespace {

// Half RoPE on one [T, heads*head_dim] row set, NeMo's
// RotaryPositionalEncoding: rotate lanes [0, d_rot) in pairs (i, i +
// d_rot/2) with duplicated cos/sin; pass the rest. Positions 0..T-1.
void rope_half(float* rows, size_t T, size_t heads, size_t hd, size_t d_rot, double base) {
  if (d_rot % 2) throw std::invalid_argument("rope_half: odd rotary dim");
  const size_t half = d_rot / 2;
  std::vector<float> cos_v(half), sin_v(half);
  for (size_t t = 0; t < T; ++t) {
    for (size_t j = 0; j < half; ++j) {
      const double invf = std::pow(base, -2.0 * static_cast<double>(j) / static_cast<double>(d_rot));
      const float ang = static_cast<float>(static_cast<double>(t) * invf);
      cos_v[j] = std::cos(ang);
      sin_v[j] = std::sin(ang);
    }
    for (size_t h = 0; h < heads; ++h) {
      float* r = rows + t * heads * hd + h * hd;
      for (size_t j = 0; j < half; ++j) {
        const float x1 = r[j], x2 = r[j + half];
        r[j] = x1 * cos_v[j] - x2 * sin_v[j];
        r[j + half] = x2 * cos_v[j] + x1 * sin_v[j];
      }
    }
  }
}

void gelu_erf(float* v, size_t n) {
  for (size_t i = 0; i < n; ++i)
    v[i] = 0.5f * v[i] * (1.0f + std::erf(v[i] * static_cast<float>(M_SQRT1_2)));
}

}  // namespace

EncSpec enc_spec_asr(const NemotronTranscribePerceptionConfig& pc) {
  EncSpec es;
  es.d_model = pc.enc_d_model;
  es.heads = pc.enc_heads;
  es.layers = pc.enc_layers;
  es.ff_expansion = pc.enc_ff_expansion;
  es.rope_base = pc.enc_rope_base;
  es.rotary_fraction = pc.enc_rotary_fraction;
  es.qk_norm = pc.enc_qk_norm;
  es.qkv_bias = pc.enc_qkv_bias;
  es.pre_block_norm = pc.enc_pre_block_norm;
  return es;
}

EncSpec enc_spec_diar(const NemotronTranscribePerceptionConfig& pc) {
  EncSpec es;
  es.d_model = pc.diar_d_model;
  es.heads = pc.diar_heads;
  es.layers = pc.diar_layers;
  es.ff_expansion = pc.diar_ff_expansion;
  es.rope_base = pc.diar_rope_base;
  es.rotary_fraction = pc.diar_rotary_fraction;
  es.qk_norm = pc.diar_qk_norm;
  es.qkv_bias = false;
  es.pre_block_norm = true;
  return es;
}

void enc_block(const EncSpec& es, const Weights& w, const std::string& prefix, float* x,
               size_t T) {
  const size_t D = es.d_model, H = es.heads, hd = D / H;
  if (H == 0 || D % H) throw std::invalid_argument("enc_block: head geometry");
  const Tensor& xw = w.at(prefix + "norm1.weight");
  const Tensor& xb = w.at(prefix + "norm1.bias");
  std::vector<float> normed(T * D);
  for (size_t t = 0; t < T; ++t) layernorm_row(x + t * D, xw, xb, D, kLnEps, normed.data() + t * D);
  // Fused qkv, token-major (3, H, hd) blocks: q rows [0, D), k [D, 2D), v [2D, 3D).
  const Tensor& wqkv = w.at(prefix + "attn.w_qkv.weight");
  const Tensor* bqkv = es.qkv_bias ? &w.at(prefix + "attn.w_qkv.bias") : nullptr;
  std::vector<float> q(T * D), k(T * D), v(T * D);
  for (size_t t = 0; t < T; ++t) {
    for (size_t i = 0; i < D; ++i) {
      double sq = 0.0, sk = 0.0, sv = 0.0;
      for (size_t j = 0; j < D; ++j) {
        const double xv = normed[t * D + j];
        sq += deq(wqkv, (0 * D + i) * D + j) * xv;
        sk += deq(wqkv, (1 * D + i) * D + j) * xv;
        sv += deq(wqkv, (2 * D + i) * D + j) * xv;
      }
      if (bqkv) {
        sq += deq(*bqkv, 0 * D + i);
        sk += deq(*bqkv, 1 * D + i);
        sv += deq(*bqkv, 2 * D + i);
      }
      q[t * D + i] = static_cast<float>(sq);
      k[t * D + i] = static_cast<float>(sk);
      v[t * D + i] = static_cast<float>(sv);
    }
  }
  if (es.qk_norm) {
    const Tensor& qw = w.at(prefix + "attn.q_norm.weight");
    const Tensor& qb = w.at(prefix + "attn.q_norm.bias");
    const Tensor& kw = w.at(prefix + "attn.k_norm.weight");
    const Tensor& kb = w.at(prefix + "attn.k_norm.bias");
    for (size_t t = 0; t < T; ++t) {
      for (size_t h = 0; h < H; ++h) {
        layernorm_row(q.data() + t * D + h * hd, qw, qb, hd, kLnEps, q.data() + t * D + h * hd);
        layernorm_row(k.data() + t * D + h * hd, kw, kb, hd, kLnEps, k.data() + t * D + h * hd);
      }
    }
  }
  const size_t d_rot = static_cast<size_t>(static_cast<double>(hd) * es.rotary_fraction);
  rope_half(q.data(), T, H, hd, d_rot, es.rope_base);
  rope_half(k.data(), T, H, hd, d_rot, es.rope_base);
  // Bidirectional attention, scale 1/sqrt(hd), softmax in fp32; out_proj + bias.
  std::vector<float> attn(T * D), scores(T), probs(T);
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
  for (size_t t = 0; t < T; ++t) {
    for (size_t h = 0; h < H; ++h) {
      float mx = -INFINITY;
      for (size_t s = 0; s < T; ++s) {
        double acc = 0.0;
        const float* qh = q.data() + t * D + h * hd;
        const float* kh = k.data() + s * D + h * hd;
        for (size_t j = 0; j < hd; ++j) acc += static_cast<double>(qh[j]) * kh[j];
        scores[s] = static_cast<float>(acc) * scale;
        mx = std::max(mx, scores[s]);
      }
      double den = 0.0;
      for (size_t s = 0; s < T; ++s) {
        probs[s] = std::exp(scores[s] - mx);
        den += probs[s];
      }
      const float inv = static_cast<float>(1.0 / den);
      for (size_t j = 0; j < hd; ++j) attn[t * D + h * hd + j] = 0.f;
      for (size_t s = 0; s < T; ++s) {
        const float p = probs[s] * inv;
        const float* vh = v.data() + s * D + h * hd;
        for (size_t j = 0; j < hd; ++j) attn[t * D + h * hd + j] += p * vh[j];
      }
    }
  }
  const Tensor& wo = w.at(prefix + "attn.out_proj.weight");
  const Tensor& bo = w.at(prefix + "attn.out_proj.bias");
  for (size_t t = 0; t < T; ++t) {
    for (size_t i = 0; i < D; ++i) {
      double acc = deq(bo, i);
      for (size_t j = 0; j < D; ++j) acc += deq(wo, i * D + j) * attn[t * D + j];
      x[t * D + i] += static_cast<float>(acc);
    }
  }
  // FFN: Linear+bias, GELU(erf), Linear+bias.
  const Tensor& f1w = w.at(prefix + "ffn.net.0.weight");
  const Tensor& f1b = w.at(prefix + "ffn.net.0.bias");
  const Tensor& f2w = w.at(prefix + "ffn.net.3.weight");
  const Tensor& f2b = w.at(prefix + "ffn.net.3.bias");
  const size_t F = es.ff_expansion * D;
  std::vector<float> normed2(T * D), hid(F);
  const Tensor& x2w = w.at(prefix + "norm2.weight");
  const Tensor& x2b = w.at(prefix + "norm2.bias");
  for (size_t t = 0; t < T; ++t) {
    layernorm_row(x + t * D, x2w, x2b, D, kLnEps, normed2.data() + t * D);
    for (size_t j = 0; j < F; ++j) {
      double acc = deq(f1b, j);
      for (size_t i = 0; i < D; ++i) acc += deq(f1w, j * D + i) * normed2[t * D + i];
      hid[j] = static_cast<float>(acc);
    }
    gelu_erf(hid.data(), F);
    for (size_t i = 0; i < D; ++i) {
      double acc = deq(f2b, i);
      for (size_t j = 0; j < F; ++j) acc += deq(f2w, i * F + j) * hid[j];
      x[t * D + i] += static_cast<float>(acc);
    }
  }
}

void encoder_run(const EncSpec& es, const NemotronTranscribePerceptionConfig& pc,
                 const Weights& w, const std::string& root, const float* mel, size_t T,
                 std::vector<float>& states) {
  const size_t D = es.d_model;
  stack_pre_encode(pc, mel, T, w.at(root + "pre_encode.proj.weight"), states);
  const size_t T_tok = states.size() / D;
  if (es.pre_block_norm) {
    const Tensor& ew = w.at(root + "embed_norm.weight");
    const Tensor& eb = w.at(root + "embed_norm.bias");
    for (size_t t = 0; t < T_tok; ++t)
      layernorm_row(states.data() + t * D, ew, eb, D, kLnEps, states.data() + t * D);
  }
  for (size_t l = 0; l < es.layers; ++l)
    enc_block(es, w, root + "layers." + std::to_string(l) + ".", states.data(), T_tok);
  const Tensor& fw = w.at(root + "final_norm.weight");
  const Tensor& fb = w.at(root + "final_norm.bias");
  for (size_t t = 0; t < T_tok; ++t)
    layernorm_row(states.data() + t * D, fw, fb, D, kLnEps, states.data() + t * D);
}

void normalize_per_feature(const float* mel, size_t C, size_t T, std::vector<float>& out) {
  // features.py normalize_batch(mel, length, 'per_feature') at full length:
  // the reference frame drops out of the mean (all frames valid), var is
  // the UNBIASED variance (sum of squares over T - 1), std gets +1e-5.
  if (T < 2) throw std::invalid_argument("normalize_per_feature: T < 2");
  out.assign(C * T, 0.f);
  for (size_t c = 0; c < C; ++c) {
    const float* x = mel + c * T;
    double sum = 0.0;
    for (size_t t = 0; t < T; ++t) sum += x[t];
    const float mean = static_cast<float>(sum / static_cast<double>(T));
    double vs = 0.0;
    for (size_t t = 0; t < T; ++t) {
      const double d = static_cast<double>(x[t]) - mean;
      vs += d * d;
    }
    const float stdv = static_cast<float>(std::sqrt(vs / static_cast<double>(T - 1))) + 1e-5f;
    for (size_t t = 0; t < T; ++t) out[c * T + t] = (x[t] - mean) / stdv;
  }
}

void asr_encoder(const NemotronTranscribePerceptionConfig& pc, const Weights& w, const float* mel,
                 size_t T, std::vector<float>& states) {
  std::vector<float> norm_mel;
  const float* in = mel;
  if (pc.asr_normalize_per_feature) {
    normalize_per_feature(mel, pc.mel_features, T, norm_mel);
    in = norm_mel.data();
  }
  encoder_run(enc_spec_asr(pc), pc, w, "perception.encoder.asr_encoder.", in, T, states);
}

void diar_activity(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                   const float* mel, size_t T, std::vector<float>& preds) {
  const EncSpec es = enc_spec_diar(pc);
  const std::string root = "perception.encoder.diarization_model.";
  std::vector<float> emb;
  encoder_run(es, pc, w, root + "encoder.", mel, T, emb);  // raw mel: diar_norm "NA"
  const size_t D = es.d_model, TF = pc.tf_d_model, S = pc.n_spk;
  const size_t T_tok = emb.size() / D;
  if (TF == D) throw std::invalid_argument("diar_activity: fc == tf selects the tf path");
  // frontend_encoder's tail: encoder_proj (the SortformerModules' input
  // projection); mask_padding (length == T_tok here) would zero the rows
  // past the clip — full-length, a no-op.
  std::vector<float> hidden(T_tok * TF);
  const Tensor& epw = w.at(root + "sortformer_modules.encoder_proj.weight");
  const Tensor& epb = w.at(root + "sortformer_modules.encoder_proj.bias");
  if (epw.numel() != TF * D) throw std::invalid_argument("diar_activity: encoder_proj numel");
  for (size_t t = 0; t < T_tok; ++t) {
    nemotron_ref::gemv(hidden.data() + t * TF, epw, emb.data() + t * D, TF, D);
    for (size_t o = 0; o < TF; ++o) hidden[t * TF + o] += deq(epb, o);
  }
  // upsample_hidden: conv1d(TF, TF * sf, k3, pad 1) over the TF x T_tok
  // plane, then (T_tok, sf, TF) -> (T_tok * sf, TF). Zero left/right pad.
  const size_t sf = pc.subsampling_factor;
  const Tensor& sw = w.at(root + "sortformer_modules.subpixel_upsample.weight");
  const Tensor& sb = w.at(root + "sortformer_modules.subpixel_upsample.bias");
  if (sw.numel() != TF * sf * TF * 3 || sb.numel() != TF * sf)
    throw std::invalid_argument("diar_activity: subpixel numel");
  std::vector<float> up(T_tok * sf * TF);
  for (size_t t = 0; t < T_tok; ++t) {
    for (size_t p = 0; p < sf; ++p) {
      for (size_t lane = 0; lane < TF; ++lane) {
        const size_t oc = p * TF + lane;
        double acc = deq(sb, oc);
        for (size_t j = 0; j < 3; ++j) {
          const size_t tt = t + j - 1;  // padding=1, zero outside
          if (j == 0 && t == 0) continue;
          if (j == 2 && t + 1 == T_tok) continue;
          for (size_t c = 0; c < TF; ++c)
            acc += deq(sw, (oc * TF + c) * 3 + j) * hidden[tt * TF + c];
        }
        up[(t * sf + p) * TF + lane] = static_cast<float>(acc);
      }
    }
  }
  // forward_speaker_logits (dropout is an eval no-op) + sigmoid + the
  // full-length output_mask, then downsample_preds: masked windows average
  // to the plain mean of their `sf` frames (all valid at full length).
  const Tensor& hw = w.at(root + "sortformer_modules.first_hidden_to_hidden.weight");
  const Tensor& hb = w.at(root + "sortformer_modules.first_hidden_to_hidden.bias");
  const Tensor& uw = w.at(root + "sortformer_modules.single_hidden_to_spks.weight");
  const Tensor& ub = w.at(root + "sortformer_modules.single_hidden_to_spks.bias");
  preds.assign(T_tok * S, 0.f);
  std::vector<float> relu_up(T_tok * sf * TF), mid(TF), row(S);
  for (size_t i = 0; i < up.size(); ++i) relu_up[i] = up[i] > 0.f ? up[i] : 0.f;
  for (size_t t = 0; t < T_tok; ++t) {
    for (size_t q = 0; q < sf; ++q) {
      const size_t tu = t * sf + q;
      nemotron_ref::gemv(mid.data(), hw, relu_up.data() + tu * TF, TF, TF);
      for (size_t o = 0; o < TF; ++o) mid[o] += deq(hb, o);
      for (size_t o = 0; o < TF; ++o) if (mid[o] < 0.f) mid[o] = 0.f;
      nemotron_ref::gemv(row.data(), uw, mid.data(), S, TF);
      for (size_t k = 0; k < S; ++k) {
        const float logit = row[k] + deq(ub, k);
        preds[t * S + k] += 1.f / (1.f + std::exp(-logit)) / static_cast<float>(sf);
      }
    }
  }
}

void build_spk_kernel(const NemotronTranscribePerceptionConfig& pc, std::vector<float>& kernel) {
  const size_t S = pc.n_spk, Dm = pc.enc_d_model;
  kernel.assign(S * Dm, 0.f);
  for (size_t s = 0; s < S; ++s) {
    for (size_t j = 0; j < Dm / 2; ++j) {
      const double div = std::exp(-std::log(10000.0) * (2.0 * static_cast<double>(j)) /
                                            static_cast<double>(Dm));
      const double p = static_cast<double>(s) * div;
      kernel[s * Dm + 2 * j] = static_cast<float>(std::sin(p));
      kernel[s * Dm + 2 * j + 1] = static_cast<float>(std::cos(p));
    }
  }
}

void fuse_and_connect(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                      const float* states, const float* activity, size_t T_tok,
                      std::vector<float>& rows) {
  const size_t D = pc.enc_d_model, S = pc.n_spk;
  std::vector<float> act(T_tok * S);
  for (size_t i = 0; i < T_tok * S; ++i)
    act[i] = pc.speaker_feature_thresholded ? (activity[i] > pc.speaker_activity_threshold ? 1.f : 0.f)
                                            : activity[i];
  const Tensor& dw = w.at("perception.encoder.diar_norm.weight");
  const Tensor& db = w.at("perception.encoder.diar_norm.bias");
  const Tensor& aw = w.at("perception.encoder.asr_norm.weight");
  const Tensor& ab = w.at("perception.encoder.asr_norm.bias");
  std::vector<float> kernel;
  build_spk_kernel(pc, kernel);
  const Tensor& pw = w.at("perception.proj.weight");
  const Tensor& pb = w.at("perception.proj.bias");
  std::vector<float> dn(S), fused(D);
  rows.assign(T_tok * pc.output_dim, 0.f);
  for (size_t t = 0; t < T_tok; ++t) {
    layernorm_row(act.data() + t * S, dw, db, S, kLnEps, dn.data());
    layernorm_row(states + t * D, aw, ab, D, kLnEps, fused.data());
    for (size_t i = 0; i < D; ++i) {
      double inf = 0.0;
      for (size_t s = 0; s < S; ++s) inf += dn[s] * kernel[s * D + i];
      fused[i] += static_cast<float>(pc.spk_kernel_scale * inf);
    }
    for (size_t o = 0; o < static_cast<size_t>(pc.output_dim); ++o) {
      double acc = deq(pb, o);
      for (size_t i = 0; i < D; ++i) acc += deq(pw, o * pc.connector_d_model + i) * fused[i];
      rows[t * pc.output_dim + o] = static_cast<float>(acc);
    }
  }
}

PerceptionOutput perceive(const NemotronTranscribePerceptionConfig& pc, const Weights& w,
                          const float* samples, size_t n, const float* activity) {
  PerceptionOutput out;
  out.frames = mel_frontend(samples, n, w.at("perception.preprocessor.featurizer.window"),
                            w.at("perception.preprocessor.featurizer.fb"), kPreemph, out.mel);
  asr_encoder(pc, w, out.mel.data(), out.frames, out.states);
  out.tokens = out.states.size() / static_cast<size_t>(pc.enc_d_model);
  std::vector<float> internal;
  if (!activity) {
    // The release route: the Sortformer branch on the RAW frontend mel
    // (diar_normalize_type "NA"), its preds thresholded inside the fusion.
    diar_activity(pc, w, out.mel.data(), out.frames, internal);
    activity = internal.data();
    out.activity = internal;
  }
  fuse_and_connect(pc, w, out.states.data(), activity, out.tokens, out.rows);
  return out;
}

}  // namespace nemotron_perc
}  // namespace dgpp
