#include "models/qwen_asr/audio_tower_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

#include "common/dtypes.hpp"
#include "loaders/safetensors.hpp"

namespace dgpp::qwen_asr_ref {
namespace {
constexpr int kChunk = 2 * kAudioWindowHalf;  // 100 frames
constexpr int kSteps = QwenAsrAudioConfig::kTimeSteps;

std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("Qwen ASR tower: cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}

// C[M,N] += A[M,K] * B[N,K]^T, i-k-j with a 64-row x 64-col register block.
// Single-threaded and fixed-order: the oracle every kernel is compared to.
void sgemm_add(const float* a, const float* b, float* c, int m, int n, int k) {
  constexpr int MB = 4, NB = 8;
  for (int i0 = 0; i0 < m; i0 += MB)
    for (int j0 = 0; j0 < n; j0 += NB) {
      float acc[MB][NB] = {};
      const int iend = i0 + MB < m ? i0 + MB : m, jend = j0 + NB < n ? j0 + NB : n;
      for (int kk = 0; kk < k; ++kk) {
        float av[MB], bv[NB];
        for (int i = i0; i < iend; ++i) av[i - i0] = a[static_cast<size_t>(i) * k + kk];
        for (int j = j0; j < jend; ++j) bv[j - j0] = b[static_cast<size_t>(j) * k + kk];
        for (int i = i0; i < iend; ++i)
          for (int j = j0; j < jend; ++j) acc[i - i0][j - j0] += av[i - i0] * bv[j - j0];
      }
      for (int i = i0; i < iend; ++i)
        for (int j = j0; j < jend; ++j) c[static_cast<size_t>(i) * n + j] += acc[i - i0][j - j0];
    }
}

void linear(const float* a, const float* w, const float* bias, float* out, int m, int n, int k) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) out[static_cast<size_t>(i) * n + j] = bias ? bias[j] : 0.0f;
  sgemm_add(a, w, out, m, n, k);
}

void layernorm(const float* x, const float* g, const float* b, float* y, int rows, int dim) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * dim;
    float* yr = y + static_cast<size_t>(r) * dim;
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < dim; ++i) mean += xr[i];
    mean /= dim;
    for (int i = 0; i < dim; ++i) var += (xr[i] - mean) * (xr[i] - mean);
    var /= dim;
    const float scale =
        static_cast<float>(1.0 / std::sqrt(var + QwenAsrAudioConfig::kLayerNormEps));
    for (int i = 0; i < dim; ++i)
      yr[i] = static_cast<float>((xr[i] - mean) * static_cast<double>(scale)) * g[i] + b[i];
  }
}

void gelu_inplace(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = 0.5f * x[i] * (1.0f + std::erff(x[i] * static_cast<float>(M_SQRT1_2)));
}

struct ShardMap {
  std::map<std::string, std::string> locations;
  std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
  std::string dir;
  const TensorInfo& find(const std::string& name) {
    auto it = locations.find(name);
    if (it == locations.end()) throw std::runtime_error("Qwen ASR tower: missing " + name);
    auto& file = files[it->second];
    if (!file) file = SafetensorsFile::open((std::filesystem::path(dir) / it->second).string());
    return file->at(std::string(prefix) + name);
  }

  static constexpr std::string_view prefix = "thinker.audio_tower.";
};

std::vector<float> to_f32(const TensorInfo& t, std::initializer_list<int64_t> shape) {
  if (t.shape != std::vector<int64_t>(shape))
    throw std::runtime_error("Qwen ASR tower: unexpected shape for " + t.name);
  std::vector<float> out(t.numel());
  if (t.dtype == DType::BF16) {
    const auto* p = static_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < out.size(); ++i) out[i] = bf16_bits_to_float(p[i]);
  } else if (t.dtype == DType::F32) {
    std::memcpy(out.data(), t.data, out.size() * sizeof(float));
  } else {
    throw std::runtime_error("Qwen ASR tower: unsupported dtype for " + t.name);
  }
  return out;
}
}  // namespace

std::vector<float> sinusoidal_positions(int rows, int channels) {
  std::vector<float> out(static_cast<size_t>(rows) * channels);
  const double log_ts = std::log(10000.0) / (channels / 2 - 1);
  for (int p = 0; p < rows; ++p)
    for (int i = 0; i < channels / 2; ++i) {
      const double angle = p * std::exp(-log_ts * i);
      out[static_cast<size_t>(p) * channels + i] = static_cast<float>(std::sin(angle));
      out[static_cast<size_t>(p) * channels + channels / 2 + i] =
          static_cast<float>(std::cos(angle));
    }
  return out;
}

AudioTowerWeights load_audio_tower(const std::string& checkpoint_dir) {
  namespace fs = std::filesystem;
  ShardMap shards;
  shards.dir = checkpoint_dir;
  const fs::path dir(checkpoint_dir);
  static constexpr std::string_view prefix = "thinker.audio_tower.";
  if (fs::exists(dir / "model.safetensors.index.json")) {
    const auto index = minijson::parse(read_text(dir / "model.safetensors.index.json"));
    for (const auto& m : index.root.at("weight_map").members())
      if (m.key.starts_with(prefix))
        shards.locations.emplace(std::string(m.key.substr(prefix.size())),
                                 std::string(m.value.as_string()));
  } else if (fs::exists(dir / "model.safetensors")) {
    auto f = SafetensorsFile::open((dir / "model.safetensors").string());
    for (const auto* t : f->grep(prefix))
      shards.locations.emplace(t->name.substr(prefix.size()), "model.safetensors");
  } else {
    throw std::runtime_error("Qwen ASR tower: no safetensors checkpoint at " + dir.string());
  }

  AudioTowerWeights w;
  auto& d = shards;
  w.conv1_w = to_f32(d.find("conv2d1.weight"), {480, 1, 3, 3});
  w.conv1_b = to_f32(d.find("conv2d1.bias"), {480});
  w.conv2_w = to_f32(d.find("conv2d2.weight"), {480, 480, 3, 3});
  w.conv2_b = to_f32(d.find("conv2d2.bias"), {480});
  w.conv3_w = to_f32(d.find("conv2d3.weight"), {480, 480, 3, 3});
  w.conv3_b = to_f32(d.find("conv2d3.bias"), {480});
  w.conv_out_w = to_f32(d.find("conv_out.weight"), {1024, 7680});
  w.pos = sinusoidal_positions(kSteps, 1024);
  auto layer_at = [&](int l) {
    AudioTowerLayerWeights t;
    const std::string p = "layers." + std::to_string(l) + ".";
    t.ln1_w = to_f32(d.find(p + "self_attn_layer_norm.weight"), {1024});
    t.ln1_b = to_f32(d.find(p + "self_attn_layer_norm.bias"), {1024});
    t.q_w = to_f32(d.find(p + "self_attn.q_proj.weight"), {1024, 1024});
    t.q_b = to_f32(d.find(p + "self_attn.q_proj.bias"), {1024});
    t.k_w = to_f32(d.find(p + "self_attn.k_proj.weight"), {1024, 1024});
    t.k_b = to_f32(d.find(p + "self_attn.k_proj.bias"), {1024});
    t.v_w = to_f32(d.find(p + "self_attn.v_proj.weight"), {1024, 1024});
    t.v_b = to_f32(d.find(p + "self_attn.v_proj.bias"), {1024});
    t.o_w = to_f32(d.find(p + "self_attn.out_proj.weight"), {1024, 1024});
    t.o_b = to_f32(d.find(p + "self_attn.out_proj.bias"), {1024});
    t.ln2_w = to_f32(d.find(p + "final_layer_norm.weight"), {1024});
    t.ln2_b = to_f32(d.find(p + "final_layer_norm.bias"), {1024});
    t.fc1_w = to_f32(d.find(p + "fc1.weight"), {4096, 1024});
    t.fc1_b = to_f32(d.find(p + "fc1.bias"), {4096});
    t.fc2_w = to_f32(d.find(p + "fc2.weight"), {1024, 4096});
    t.fc2_b = to_f32(d.find(p + "fc2.bias"), {1024});
    return t;
  };
  w.layers.reserve(24);
  for (int l = 0; l < 24; ++l) w.layers.push_back(layer_at(l));
  w.ln_post_w = to_f32(d.find("ln_post.weight"), {1024});
  w.ln_post_b = to_f32(d.find("ln_post.bias"), {1024});
  w.proj1_w = to_f32(d.find("proj1.weight"), {1024, 1024});
  w.proj1_b = to_f32(d.find("proj1.bias"), {1024});
  w.proj2_w = to_f32(d.find("proj2.weight"), {2048, 1024});
  w.proj2_b = to_f32(d.find("proj2.bias"), {2048});
  return w;
}

void encode_audio(const QwenAsrAudioConfig& cfg, const AudioTowerWeights& w, const float* feats,
                  const int32_t* mask, int frames, std::vector<float>& out) {
  const int dm = cfg.d_model, nchunks = frames / kChunk;
  if (frames % kChunk != 0 || nchunks < 1)
    throw std::invalid_argument("Qwen ASR tower: frames must be a positive multiple of 100");
  const int cc = cfg.conv_channels, fb = cfg.freq_bins();

  // Per-chunk valid frames and packed-row lengths.
  std::vector<int> m_c(nchunks), l_c(nchunks);
  int rows = 0, valid = 0;
  for (int c = 0; c < nchunks; ++c) {
    for (int t = 0; t < kChunk; ++t) m_c[c] += mask[c * kChunk + t];
    valid += m_c[c];
    int l = m_c[c];
    for (int conv = 0; conv < 3 && l > 0; ++conv) l = (l - 1) / 2 + 1;
    l_c[c] = l;
    rows += l;
  }
  if (rows != audio_token_count(valid))
    throw std::logic_error("Qwen ASR tower: packed length disagrees with the token rule");

  // CNN frontend, all intermediates cell-first ([y*wo + x, channels]):
  // exactly the layout the im2col GEMM writes, so no transposes between
  // convs. Per chunk: [1,128,100] -> [3200,cc] -> [800,cc] -> [208,cc].
  std::vector<float> conv_in, x(static_cast<size_t>(nchunks) * kSteps * cfg.conv_flat()), packed,
      conv_out_buf, tmp;
  const auto conv2d = [&](const float* in, int cin, int hi, int wi, const float* weight,
                          const float* bias, float* o, int ho, int wo) {
    conv_in.resize(static_cast<size_t>(ho) * wo * cin * 9);
    for (int y = 0; y < ho; ++y)
      for (int xj = 0; xj < wo; ++xj) {
        float* dst = conv_in.data() + (static_cast<size_t>(y) * wo + xj) * cin * 9;
        for (int ci = 0; ci < cin; ++ci)
          for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
              const int iy = y * 2 + p - 1, ix = xj * 2 + q - 1;
              dst[ci * 9 + p * 3 + q] =
                  (iy >= 0 && iy < hi && ix >= 0 && ix < wi)
                      ? in[static_cast<size_t>(iy) * wi * cin + static_cast<size_t>(ix) * cin + ci]
                      : 0.0f;
            }
      }
    linear(conv_in.data(), weight, bias, o, ho * wo, cc, cin * 9);
    gelu_inplace(o, static_cast<size_t>(ho) * wo * cc);
  };

  std::vector<float> h1(64 * 50 * cc), h2(32 * 25 * cc), h3(fb * kSteps * cc);
  for (int c = 0; c < nchunks; ++c) {
    // conv1 gathers straight from the mel frames (input channel 1): the
    // cell (freq, time) grid of the [1, 128, 100] chunk.
    conv_in.resize(64 * 50 * 9);
    for (int y = 0; y < 64; ++y)
      for (int xj = 0; xj < 50; ++xj)
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q) {
            const int iy = y * 2 + p - 1, ix = xj * 2 + q - 1;
            conv_in[static_cast<size_t>(y) * 50 * 9 + xj * 9 + p * 3 + q] =
                (iy >= 0 && iy < cfg.mel_bins && ix >= 0 && ix < kChunk)
                    ? feats[static_cast<size_t>(iy) * frames + c * kChunk + ix]
                    : 0.0f;
          }
    linear(conv_in.data(), w.conv1_w.data(), w.conv1_b.data(), h1.data(), 64 * 50, cc, 9);
    gelu_inplace(h1.data(), 64 * 50 * static_cast<size_t>(cc));
    conv2d(h1.data(), cc, 64, 50, w.conv2_w.data(), w.conv2_b.data(), h2.data(), 32, 25);
    conv2d(h2.data(), cc, 32, 25, w.conv3_w.data(), w.conv3_b.data(), h3.data(), fb, kSteps);
    // The conv_out row for time t: channel-major over the 16 freq bins
    // (the permute's [chan, freq] view), gathered from cell-first h3.
    for (int t = 0; t < kSteps; ++t) {
      float* dst = x.data() + (static_cast<size_t>(c) * kSteps + t) * cfg.conv_flat();
      for (int ch = 0; ch < cc; ++ch)
        for (int fq = 0; fq < fb; ++fq)
          dst[ch * fb + fq] = h3[static_cast<size_t>(fq) * kSteps * cc + t * cc + ch];
    }
  }
  // conv_out GEMM + sinusoidal positions, then the valid-row pack.
  conv_out_buf.resize(static_cast<size_t>(nchunks) * kSteps * dm);
  linear(x.data(), w.conv_out_w.data(), nullptr, conv_out_buf.data(), nchunks * kSteps, dm,
         cfg.conv_flat());
  packed.resize(static_cast<size_t>(rows) * dm);
  {
    int r = 0;
    for (int c = 0; c < nchunks; ++c)
      for (int t = 0; t < l_c[c]; ++r, ++t)
        for (int j = 0; j < dm; ++j)
          packed[static_cast<size_t>(r) * dm + j] =
              conv_out_buf[(static_cast<size_t>(c) * kSteps + t) * dm + j] +
              w.pos[static_cast<size_t>(t) * dm + j];
  }

  if (std::getenv("DGPP_ASR_DEBUG")) {
    double e = 0.0, m = 0.0;
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < dm; ++j) {
        e += double(packed[size_t(r) * dm + j]) * packed[size_t(r) * dm + j];
        m += packed[size_t(r) * dm + j];
      }
    std::printf("packed rows %d energy %.6f mean %.6e\n", rows, e, m / (rows * dm));
  }

  // Encoder: attention windows of cfg.window_rows() packed rows, batch 1.
  const int hd = cfg.head_dim(), heads = cfg.heads, win = cfg.window_rows();
  std::vector<float> hbuf(packed), norm(static_cast<size_t>(rows) * dm),
      q(static_cast<size_t>(rows) * dm), kk(static_cast<size_t>(rows) * dm),
      v(static_cast<size_t>(rows) * dm), att(static_cast<size_t>(rows) * dm),
      hidden(static_cast<size_t>(rows) * cfg.ffn), add(static_cast<size_t>(rows) * dm);
  std::vector<float> scores;
  for (const auto& t : w.layers) {
    layernorm(hbuf.data(), t.ln1_w.data(), t.ln1_b.data(), norm.data(), rows, dm);
    linear(norm.data(), t.q_w.data(), t.q_b.data(), q.data(), rows, dm, dm);
    linear(norm.data(), t.k_w.data(), t.k_b.data(), kk.data(), rows, dm, dm);
    linear(norm.data(), t.v_w.data(), t.v_b.data(), v.data(), rows, dm, dm);
    for (int s = 0; s < rows; s += win) {
      const int e = s + win < rows ? s + win : rows, n = e - s;
      scores.resize(static_cast<size_t>(n) * n);
      std::fill_n(att.data() + static_cast<size_t>(s) * dm, static_cast<size_t>(n) * dm, 0.0f);
      const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
      for (int hh = 0; hh < heads; ++hh) {
        const float* qh = q.data() + static_cast<size_t>(s) * dm + hh * hd;
        const float* kh = kk.data() + static_cast<size_t>(s) * dm + hh * hd;
        const float* vh = v.data() + static_cast<size_t>(s) * dm + hh * hd;
        for (int i = 0; i < n; ++i) {
          float* sc = scores.data() + static_cast<size_t>(i) * n;
          for (int j = 0; j < n; ++j) {
            const float* qi = qh + static_cast<size_t>(i) * dm;
            const float* kj = kh + static_cast<size_t>(j) * dm;
            double acc = 0.0;
            for (int d = 0; d < hd; ++d) acc += static_cast<double>(qi[d]) * kj[d];
            sc[j] = static_cast<float>(acc * scale);
          }
          double mx = sc[0];
          for (int j = 1; j < n; ++j) mx = sc[j] > mx ? sc[j] : mx;
          double sum = 0.0;
          for (int j = 0; j < n; ++j) {
            sc[j] = std::exp(sc[j] - static_cast<float>(mx));
            sum += sc[j];
          }
          const float inv = static_cast<float>(1.0 / sum);
          for (int d = 0; d < hd; ++d) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j)
              acc += static_cast<double>(sc[j]) * vh[static_cast<size_t>(j) * dm + d];
            att[static_cast<size_t>(s + i) * dm + hh * hd + d] = static_cast<float>(acc * inv);
          }
        }
      }
    }
    linear(att.data(), t.o_w.data(), t.o_b.data(), add.data(), rows, dm, dm);
    for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] += add[i];
    layernorm(hbuf.data(), t.ln2_w.data(), t.ln2_b.data(), norm.data(), rows, dm);
    linear(norm.data(), t.fc1_w.data(), t.fc1_b.data(), hidden.data(), rows, cfg.ffn, dm);
    gelu_inplace(hidden.data(), hidden.size());
    linear(hidden.data(), t.fc2_w.data(), t.fc2_b.data(), add.data(), rows, dm, cfg.ffn);
    for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] += add[i];
  }
  layernorm(hbuf.data(), w.ln_post_w.data(), w.ln_post_b.data(), norm.data(), rows, dm);
  tmp.resize(static_cast<size_t>(rows) * cfg.output);
  linear(norm.data(), w.proj1_w.data(), w.proj1_b.data(), hbuf.data(), rows, dm, dm);
  gelu_inplace(hbuf.data(), hbuf.size());
  linear(hbuf.data(), w.proj2_w.data(), w.proj2_b.data(), tmp.data(), rows, cfg.output, dm);
  out = std::move(tmp);
}
}  // namespace dgpp::qwen_asr_ref
