#include "models/qwen_asr/tower.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/gemm.hpp"
#include "kernels/qwen_asr_tower.hpp"
#include "kernels/qwen_vision.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen_asr/audio_tower_reference.hpp"


namespace dgpp {
namespace {
constexpr int kChunkFrames = 100;  // 2 * n_window mel frames per conv chunk
constexpr int kMaxPieceFrames = 3000;  // the 30 s featurizer window
constexpr int kMaxChunks = kMaxPieceFrames / kChunkFrames;
constexpr int kMaxPackedRows = kMaxChunks * QwenAsrAudioConfig::kTimeSteps;
constexpr int kQueryTile = 128;  // the attention query tile (windows are <= 104 rows)

struct Buffer {
  void* p = nullptr;
  size_t bytes = 0;
  ~Buffer() { cudaFree(p); }
  void resize(size_t size) {
    if (size <= bytes) return;
    void* next = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&next, size));
    cudaFree(p);
    p = next;
    bytes = size;
  }
  uint16_t* b() const { return static_cast<uint16_t*>(p); }
  float* f() const { return static_cast<float*>(p); }
  int32_t* i() const { return static_cast<int32_t*>(p); }
};

std::string read_text(const std::filesystem::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("Qwen ASR tower: cannot read " + p.string());
  return std::string(std::istreambuf_iterator<char>(f), {});
}

// The reference's _post_cnn_length on one chunk's valid frame count: three
// (L - 1) / 2 + 1 halvings, zero staying zero.
int post_cnn_length(int frames) {
  int l = frames;
  for (int conv = 0; conv < 3 && l > 0; ++conv) l = (l - 1) / 2 + 1;
  return l;
}
}  // namespace

struct QwenAsrTowerEncoder::Impl {
  QwenAsrAudioConfig c;
  cudaStream_t stream;
  CublasLtGemm gemm;
  std::map<std::string, std::unique_ptr<Buffer>> weights;
  // Uploads, conv cells/acts, packed rows, head-major q/k/v, the windowed
  // attention tile, the MLP hidden layer, one piece's projected rows, and
  // the GEMM workspace. Sized for the 30 s / 3000-frame piece; smaller
  // pieces simply use the prefix.
  Buffer feats, act, act2, conv_rows, x, norm, tmp, q, k, v, attn_win, scores, probs,
      hidden, rows, work;
  Buffer pos, ids;  // the sinusoidal table and the tiny valid/base upload

  Impl(const QwenAsrAudioConfig& cfg, const std::string& checkpoint, cudaStream_t s)
      : c(cfg), stream(s) {
    namespace fs = std::filesystem;
    static constexpr std::string_view prefix = "thinker.audio_tower.";
    const fs::path dir(checkpoint);
    std::map<std::string, std::string> locations;
    if (fs::exists(dir / "model.safetensors.index.json")) {
      const auto index = minijson::parse(read_text(dir / "model.safetensors.index.json"));
      for (const auto& m : index.root.at("weight_map").members())
        if (m.key.starts_with(prefix))
          locations.emplace(std::string(m.key.substr(prefix.size())),
                            std::string(m.value.as_string()));
    } else if (fs::exists(dir / "model.safetensors")) {
      auto f = SafetensorsFile::open((dir / "model.safetensors").string());
      for (const auto* t : f->grep(prefix))
        locations.emplace(t->name.substr(prefix.size()), "model.safetensors");
    } else {
      throw std::runtime_error("Qwen ASR tower: no safetensors checkpoint at " + dir.string());
    }
    std::map<std::string, std::unique_ptr<SafetensorsFile>> files;
    const auto load = [&](const std::string& name, std::vector<int64_t> shape) {
      auto it = locations.find(name);
      if (it == locations.end())
        throw std::runtime_error("Qwen ASR tower: missing thinker.audio_tower." + name);
      auto& file = files[it->second];
      if (!file) file = SafetensorsFile::open((dir / it->second).string());
      const auto& tensor = file->at(std::string(prefix) + name);
      if (tensor.dtype != DType::BF16 || tensor.shape != shape)
        throw std::runtime_error("Qwen ASR tower: incompatible BF16 tensor thinker.audio_tower." +
                                 name);
      auto buffer = std::make_unique<Buffer>();
      buffer->resize(tensor.nbytes());
      DGPP_CUDA_OK(cudaMemcpyAsync(buffer->p, tensor.data, tensor.nbytes(),
                                   cudaMemcpyHostToDevice, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      weights.emplace(name, std::move(buffer));
    };
    const auto linear = [&](const std::string& name, int out, int in) {
      load(name + ".weight", {out, in});
      load(name + ".bias", {out});
    };
    const int dm = c.d_model;
    load("conv2d1.weight", {c.conv_channels, 1, 3, 3});
    load("conv2d1.bias", {c.conv_channels});
    load("conv2d2.weight", {c.conv_channels, c.conv_channels, 3, 3});
    load("conv2d2.bias", {c.conv_channels});
    load("conv2d3.weight", {c.conv_channels, c.conv_channels, 3, 3});
    load("conv2d3.bias", {c.conv_channels});
    load("conv_out.weight", {dm, c.conv_flat()});
    for (int l = 0; l < c.layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      load(p + "self_attn_layer_norm.weight", {dm});
      load(p + "self_attn_layer_norm.bias", {dm});
      linear(p + "self_attn.q_proj", dm, dm);
      linear(p + "self_attn.k_proj", dm, dm);
      linear(p + "self_attn.v_proj", dm, dm);
      linear(p + "self_attn.out_proj", dm, dm);
      load(p + "final_layer_norm.weight", {dm});
      load(p + "final_layer_norm.bias", {dm});
      linear(p + "fc1", c.ffn, dm);
      linear(p + "fc2", dm, c.ffn);
    }
    load("ln_post.weight", {dm});
    load("ln_post.bias", {dm});
    linear("proj1", dm, dm);
    linear("proj2", c.output, dm);
    // The sinusoidal table is a computed buffer, not a checkpoint tensor:
    // same formula as the oracle, one bf16 rounding at the upload.
    const auto table = qwen_asr_ref::sinusoidal_positions(QwenAsrAudioConfig::kTimeSteps, dm);
    std::vector<uint16_t> table_bf16(table.size());
    for (size_t i = 0; i < table.size(); ++i) table_bf16[i] = float_to_bf16_bits(table[i]);
    pos.resize(table_bf16.size() * 2);
    DGPP_CUDA_OK(cudaMemcpyAsync(pos.p, table_bf16.data(), table_bf16.size() * 2,
                                 cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));

    // The maximum workspace before the fabric starts (vision.cpp's policy:
    // no cudaMalloc on the request path).
    const int cc = c.conv_channels;
    feats.resize(static_cast<size_t>(c.mel_bins) * kMaxPieceFrames * sizeof(float));
    ids.resize(2 * static_cast<size_t>(kMaxChunks) * sizeof(int32_t));
    act.resize(static_cast<size_t>(kMaxChunks) * 64 * 50 * cc * 2);
    act2.resize(static_cast<size_t>(kMaxChunks) * 32 * 25 * cc * 2);
    conv_rows.resize(static_cast<size_t>(kMaxChunks) * QwenAsrAudioConfig::kTimeSteps *
                     c.conv_flat() * 2);
    const size_t packed = static_cast<size_t>(kMaxPackedRows) * dm * 2;
    for (auto* buf : {&x, &norm, &tmp, &q, &k, &v}) buf->resize(packed);
    attn_win.resize(static_cast<size_t>(c.heads) * c.window_rows() * c.head_dim() * 2);
    scores.resize(static_cast<size_t>(c.heads) * kQueryTile * c.window_rows() * sizeof(float));
    probs.resize(static_cast<size_t>(c.heads) * kQueryTile * c.window_rows() * 2);
    hidden.resize(static_cast<size_t>(kMaxPackedRows) * c.ffn * 2);
    rows.resize(static_cast<size_t>(kMaxPackedRows) * c.output * 2);
    work.resize(64ull << 20);
  }

  const uint16_t* w(const std::string& name) const { return weights.at(name)->b(); }
  void linear(const uint16_t* in, uint16_t* out, int rows, int n, int inner,
              const std::string& weight, const uint16_t* bias) {
    gemm.matmul_linear_bf16(in, w(weight), bias, out, rows, n, inner, work.p, work.bytes,
                            stream);
  }
  void linear(const uint16_t* in, uint16_t* out, int rows, int n, int inner,
              const std::string& name) {
    linear(in, out, rows, n, inner, name + ".weight", w(name + ".bias"));
  }

  int encode_piece(const float* feats_host, const int32_t* mask_host, int frames,
                   uint16_t* dst) {
    const int dm = c.d_model, hd = c.head_dim(), heads = c.heads, cc = c.conv_channels;
    if (frames <= 0 || frames % kChunkFrames != 0 || frames > kMaxPieceFrames)
      throw std::invalid_argument("Qwen ASR tower: piece frames must be a positive multiple "
                                  "of 100 within the 30 s window");
    const int chunks = frames / kChunkFrames;
    // The packed geometry is host-cheap: per-chunk valid frames, their
    // post-conv lengths, and the row bases they pack at.
    std::vector<int> valid(chunks, 0), base(chunks, 0);
    int packed_rows = 0;
    for (int ch = 0; ch < chunks; ++ch) {
      int m = 0;
      for (int t = 0; t < kChunkFrames; ++t) m += mask_host[ch * kChunkFrames + t];
      valid[ch] = post_cnn_length(m);
      base[ch] = packed_rows;
      packed_rows += valid[ch];
    }
    if (packed_rows == 0)
      throw std::invalid_argument("Qwen ASR tower: piece has no valid frames");
    DGPP_CUDA_OK(cudaMemcpyAsync(feats.p, feats_host,
                                 static_cast<size_t>(c.mel_bins) * frames * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    // The three stride-2 convs, batched over the piece's chunks: each
    // writes cell-first [(chunks*cells), cc] directly, which is the next
    // conv's input layout with no transpose. Ping-ponged: every conv reads
    // one act buffer and writes the other (same-buffer would race, the
    // output prefix overlapping input other threads still read).
    qwen_asr_conv_mel(feats.f(), w("conv2d1.weight"), w("conv2d1.bias"), act.b(), chunks,
                      frames, 64, 50, cc, stream);
    qwen_vision_gelu_exact(act.b(), static_cast<int64_t>(chunks) * 64 * 50 * cc, stream);
    qwen_asr_conv(act.b(), w("conv2d2.weight"), w("conv2d2.bias"), act2.b(), chunks, cc,
                  64, 50, 32, 25, cc, stream);
    qwen_vision_gelu_exact(act2.b(), static_cast<int64_t>(chunks) * 32 * 25 * cc, stream);
    qwen_asr_conv(act2.b(), w("conv2d3.weight"), w("conv2d3.bias"), act.b(), chunks, cc, 32,
                  25, c.freq_bins(), QwenAsrAudioConfig::kTimeSteps, cc, stream);
    qwen_vision_gelu_exact(act.b(), static_cast<int64_t>(chunks) * c.freq_bins() *
                                        QwenAsrAudioConfig::kTimeSteps * cc,
                           stream);
    qwen_asr_conv_out_gather(act.b(), conv_rows.b(), chunks, stream);
    linear(conv_rows.b(), tmp.b(), chunks * QwenAsrAudioConfig::kTimeSteps, dm, c.conv_flat(),
           "conv_out.weight", nullptr);
    qwen_asr_add_pos(tmp.b(), pos.b(), chunks, stream);
    // valid/base ride a tiny upload; the pack scatters chunk-major into x.
    DGPP_CUDA_OK(cudaMemcpyAsync(ids.p, valid.data(), static_cast<size_t>(chunks) * 4,
                                 cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(static_cast<int32_t*>(ids.p) + chunks, base.data(),
                                 static_cast<size_t>(chunks) * 4, cudaMemcpyHostToDevice,
                                 stream));
    qwen_asr_pack_rows(tmp.b(), x.b(), static_cast<const int32_t*>(ids.p),
                       static_cast<const int32_t*>(ids.p) + chunks, chunks, stream);
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
    const int win = c.window_rows();
    for (int l = 0; l < c.layers; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      qwen_vision_layernorm(x.b(), w(p + "self_attn_layer_norm.weight"),
                            w(p + "self_attn_layer_norm.bias"), norm.b(), packed_rows, dm,
                            QwenAsrAudioConfig::kLayerNormEps, stream);
      linear(norm.b(), tmp.b(), packed_rows, dm, dm, p + "self_attn.q_proj");
      qwen_asr_to_head_major(tmp.b(), q.b(), packed_rows, heads, hd, stream);
      linear(norm.b(), tmp.b(), packed_rows, dm, dm, p + "self_attn.k_proj");
      qwen_asr_to_head_major(tmp.b(), k.b(), packed_rows, heads, hd, stream);
      linear(norm.b(), tmp.b(), packed_rows, dm, dm, p + "self_attn.v_proj");
      qwen_asr_to_head_major(tmp.b(), v.b(), packed_rows, heads, hd, stream);
      // Bidirectional attention in windows of packed rows (the reference's
      // cu_seqlens at batch 1 are uniform 104-cuts): the same batched-GEMM
      // + softmax + store + unhead sequence as the vision tower, windowed.
      // tmp/norm are free inside the loop (their LN/qkv contents were
      // consumed by the head-major transpose), so each window's unhead and
      // output projection stage through their window slices.
      for (int first = 0; first < packed_rows; first += win) {
        const int tile = std::min(win, packed_rows - first);
        const uint16_t* qw = q.b() + static_cast<int64_t>(first) * hd;
        const uint16_t* kw = k.b() + static_cast<int64_t>(first) * hd;
        const uint16_t* vw = v.b() + static_cast<int64_t>(first) * hd;
        uint16_t* uw = tmp.b() + static_cast<int64_t>(first) * dm;
        uint16_t* ow = norm.b() + static_cast<int64_t>(first) * dm;
        gemm.matmul_batched_bf16(qw, kw, scores.f(), tile, tile, hd, heads,
                                 static_cast<int64_t>(packed_rows) * hd,
                                 static_cast<int64_t>(packed_rows) * hd,
                                 static_cast<int64_t>(tile) * tile, work.p, work.bytes,
                                 stream, tile);
        qwen_vision_softmax(scores.f(), probs.b(), heads * tile, tile, scale, stream);
        gemm.matmul_batched_bf16(probs.b(), vw, scores.f(), tile, hd, tile, heads,
                                 static_cast<int64_t>(tile) * tile,
                                 static_cast<int64_t>(packed_rows) * hd,
                                 static_cast<int64_t>(tile) * hd, work.p, work.bytes, stream,
                                 tile, true);
        qwen_vision_store_attention(scores.f(), attn_win.b(), tile, tile, dm, heads, 0, stream);
        qwen_vision_unhead(attn_win.b(), uw, tile, dm, heads, stream);
        linear(uw, ow, tile, dm, dm, p + "self_attn.out_proj");
        qwen_vision_residual_add(x.b() + static_cast<int64_t>(first) * dm, ow,
                                 static_cast<int64_t>(tile) * dm, stream);
      }
      qwen_vision_layernorm(x.b(), w(p + "final_layer_norm.weight"),
                            w(p + "final_layer_norm.bias"), norm.b(), packed_rows, dm,
                            QwenAsrAudioConfig::kLayerNormEps, stream);
      linear(norm.b(), hidden.b(), packed_rows, c.ffn, dm, p + "fc1");
      qwen_vision_gelu_exact(hidden.b(), static_cast<int64_t>(packed_rows) * c.ffn, stream);
      linear(hidden.b(), tmp.b(), packed_rows, dm, c.ffn, p + "fc2");
      qwen_vision_residual_add(x.b(), tmp.b(), static_cast<int64_t>(packed_rows) * dm, stream);
    }
    qwen_vision_layernorm(x.b(), w("ln_post.weight"), w("ln_post.bias"), norm.b(), packed_rows,
                          dm, QwenAsrAudioConfig::kLayerNormEps, stream);
    linear(norm.b(), hidden.b(), packed_rows, dm, dm, "proj1");
    qwen_vision_gelu_exact(hidden.b(), static_cast<int64_t>(packed_rows) * dm, stream);
    linear(hidden.b(), dst, packed_rows, c.output, dm, "proj2");
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    return packed_rows;
  }
};

size_t QwenAsrTowerEncoder::weight_bytes(const QwenAsrAudioConfig& cfg) {
  const size_t dm = cfg.d_model, cc = cfg.conv_channels;
  size_t n = 9 * cc + cc + 2 * (9 * cc * cc + cc) +  // convs
             static_cast<size_t>(cfg.conv_flat()) * dm + 13 * dm;  // conv_out + positions
  n += static_cast<size_t>(cfg.layers) *
       (4 * dm * dm + 4 * dm + 2 * 2 * dm + 2 * static_cast<size_t>(dm) * cfg.ffn + dm +
        cfg.ffn);
  n += 2 * dm + static_cast<size_t>(dm) * dm + dm + static_cast<size_t>(cfg.output) * dm +
       cfg.output;
  return n * 2;
}

size_t QwenAsrTowerEncoder::workspace_bytes(const QwenAsrAudioConfig& cfg) {
  const size_t dm = cfg.d_model, cc = cfg.conv_channels;
  size_t n = static_cast<size_t>(cfg.mel_bins) * kMaxPieceFrames * sizeof(float) +  // feats
             2 * static_cast<size_t>(kMaxChunks) * sizeof(int32_t) +  // valid/base upload
             static_cast<size_t>(kMaxChunks) * 64 * 50 * cc * 2 +  // conv1 acts
             static_cast<size_t>(kMaxChunks) * 32 * 25 * cc * 2 +  // conv2 acts
             static_cast<size_t>(kMaxChunks) * QwenAsrAudioConfig::kTimeSteps * cfg.conv_flat() *
                 2 +  // conv_out rows
             6 * static_cast<size_t>(kMaxPackedRows) * dm * 2 +  // x/norm/tmp/q/k/v
             static_cast<size_t>(cfg.heads) * cfg.window_rows() * cfg.head_dim() * 2 +  // attn
             static_cast<size_t>(cfg.heads) * kQueryTile * cfg.window_rows() *
                 (sizeof(float) + 2) +  // scores + probs
             static_cast<size_t>(kMaxPackedRows) * cfg.ffn * 2 +  // MLP hidden
             static_cast<size_t>(kMaxPackedRows) * cfg.output * 2 +  // piece rows
             (64ull << 20);  // GEMM workspace
  return n;
}

QwenAsrTowerEncoder::QwenAsrTowerEncoder(const QwenAsrAudioConfig& cfg,
                                         const std::string& checkpoint, cudaStream_t s)
    : impl_(std::make_unique<Impl>(cfg, checkpoint, s)) {}
QwenAsrTowerEncoder::~QwenAsrTowerEncoder() = default;

int QwenAsrTowerEncoder::encode_piece(const float* feats, const int32_t* mask, int frames,
                                      uint16_t* dst) {
  return impl_->encode_piece(feats, mask, frames, dst);
}
}  // namespace dgpp
