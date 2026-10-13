#include "models/qwen_asr/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "audio/mel_frontend.hpp"
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "models/qwen_asr/asr_kernels.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
namespace {
template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

}  // namespace

struct QwenAsrModel::DenseMlp {
  uint16_t* gate = nullptr;  // [M, I]
  uint16_t* up = nullptr;    // [M, I]
  int hidden = 0, inter = 0;
  ~DenseMlp() {
    cudaFree(gate);
    cudaFree(up);
  }
};

QwenAsrConfig QwenAsrConfig::from_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Qwen ASR: cannot read " + path);
  const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const auto doc = minijson::parse(text);
  try {
    return parse(doc.root);
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(std::string("Qwen ASR: ") + path + ": " + e.what());
  }
}

QwenAsrModel::QwenAsrModel(const QwenAsrConfig& cfg, const std::string& checkpoint_dir,
                           int max_tokens, int64_t max_cache_tokens, int max_requests,
                           int decode_rows, bool serving_logits)
    : cfg_(cfg), geom_(cfg.lm.glm_geometry()), loader_(cfg.lm, checkpoint_dir) {
  if (max_tokens <= 0) throw std::invalid_argument("QwenAsrModel: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("QwenAsrModel: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("QwenAsrModel: decode_rows exceeds 32");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.lm.hidden_size;
  globals_ = loader_.load_globals();
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens = ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) /
                           kBlockTokens) * kBlockTokens;
    sp.rank = 0;
    sp.world = 1;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.logits_rows = compact_logits_rows(serving_logits, decode_rows, max_requests);
    sp.mtp = false;
    sp.vocab_size = cfg_.lm.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = 0;
    sp.lm_vocab_count = cfg_.lm.vocab_size;
    sp.max_position_embeddings = cfg_.lm.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    sp.snapshot_align = 1;  // the only state is the paged pool
    sp.eos = static_cast<int32_t>(cfg_.lm.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20,
                                    gemm_.query_workspace_bytes(max_tokens_, cfg_.lm.vocab_size, H,
                                                                DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = Glm4GemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  {
    Glm4KvPoolShape shape;
    shape.layers = cfg_.lm.num_hidden_layers;
    shape.kv_heads = cfg_.lm.num_key_value_heads;
    shape.dim = QwenAsrLmConfig::kHeadDim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    pool_.init(shape);
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  audio_rows_ = dev_alloc<uint16_t>(std::min<size_t>(M, kMaxAudioTokens) * H);
  tower_ = std::make_unique<QwenAsrTowerEncoder>(cfg_.audio, checkpoint_dir, stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

QwenAsrModel::~QwenAsrModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(audio_rows_);
  cudaFree(gemm_ws_);
}

MemoryPlan QwenAsrModel::plan_memory(const QwenAsrConfig& cfg, int max_tokens,
                                     int64_t max_cache_tokens, int max_requests, int decode_rows,
                                     bool serving_logits) {
  if (max_tokens <= 0 || max_requests <= 0 || max_requests > kPickMaxRequests ||
      decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: shape");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) *
      kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.lm.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens), H = cfg.lm.hidden_size;
  plan.add("language model weights (resident, BF16 verbatim)",
           QwenAsrLayerStream::resident_bytes(cfg.lm, 0, 1, LoaderHeadSharding::Full, false),
           QwenAsrLayerStream::staging_plan_bytes(cfg.lm, 0, 1, LoaderHeadSharding::Full, false));
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    Glm4KvPoolShape shape;
    shape.layers = cfg.lm.num_hidden_layers;
    shape.kv_heads = cfg.lm.num_key_value_heads;
    shape.dim = QwenAsrLmConfig::kHeadDim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add("kv cache pool (K/V bf16)", Glm4KvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, static_cast<int>(H), cfg.lm.vocab_size, false,
                            static_cast<int>(H), &core_dev, &core_pin, rows, serving_logits);
    plan.add("activations (session core, residual, block io)", core_dev + 3 * M * H * 2, core_pin);
  }
  plan.add("audio tower rows (device staging)", std::min<size_t>(M, kMaxAudioTokens) * H * 2);
  plan.add("audio tower weights (device BF16)", QwenAsrTowerEncoder::weight_bytes(cfg.audio));
  plan.add("audio tower workspace (device)", QwenAsrTowerEncoder::workspace_bytes(cfg.audio));
  {
    Glm4TextConfig g = cfg.lm.glm_geometry();
    plan.add("layer objects (attention, dense MLP scratch)",
             Glm4AttentionLayer::scratch_bytes(g, cfg.lm.num_attention_heads,
                                               cfg.lm.num_key_value_heads, max_tokens, rows,
                                               Glm4AttentionLayer::default_decode_splits()) +
                 2 * M * static_cast<size_t>(cfg.lm.intermediate_size) * 2);
  }
  return plan;
}

size_t QwenAsrModel::session_snapshot_bytes(const QwenAsrConfig&) { return kSnapshotStamp; }

void QwenAsrModel::build_layer_objects(const QwenAsrLayerResident& r) {
  Glm4AttnResident a;
  a.q_proj = r.attn.q_proj;
  a.k_proj = r.attn.k_proj;
  a.v_proj = r.attn.v_proj;
  a.o_proj = r.attn.o_proj;
  a.q_bias = nullptr;
  a.k_bias = nullptr;
  a.v_bias = nullptr;
  a.q_norm = r.attn.q_norm;
  a.k_norm = r.attn.k_norm;
  a.local_heads = r.attn.local_heads;
  a.head_begin = 0;
  a.local_kv_heads = r.attn.local_kv_heads;
  a.kv_head_begin = 0;
  if (!attn_)
    attn_ = std::make_unique<Glm4AttentionLayer>(a, gw_, geom_, max_tokens_, max_decode_rows_,
                                                 Glm4AttentionLayer::default_decode_splits());
  else
    attn_->rebind(a);
  if (!dense_) {
    dense_ = std::make_unique<DenseMlp>();
    dense_->hidden = cfg_.lm.hidden_size;
    dense_->inter = cfg_.lm.intermediate_size;
    dense_->gate = dev_alloc<uint16_t>(static_cast<size_t>(max_tokens_) * dense_->inter);
    dense_->up = dev_alloc<uint16_t>(static_cast<size_t>(max_tokens_) * dense_->inter);
  }
}

void QwenAsrModel::enqueue_layer(const QwenAsrLayerResident& r, int layer, uint16_t* resid, int T,
                                 const int32_t* req_ids, const int64_t* pos, bool decode,
                                 bool capture) {
  const int H = cfg_.lm.hidden_size;
  const int I = cfg_.lm.intermediate_size;
  const float eps = cfg_.lm.rms_norm_eps;
  const int64_t n = static_cast<int64_t>(T) * H;
  (void)layer;
  (void)capture;
  build_layer_objects(r);
  glm_rmsnorm_bf16(resid, r.input_norm, x_, T, H, eps, stream_);
  Glm4AttnRows rows;
  rows.req_ids = req_ids;
  rows.pos = pos;
  rows.decode = decode;
  attn_->enqueue(x_, T, rows, pool_.view(layer), y_, stream_);
  glm_residual_add_bf16(resid, y_, n, stream_);
  glm_rmsnorm_bf16(resid, r.post_norm, x_, T, H, eps, stream_);
  const auto M = static_cast<size_t>(T);
  gemm_.matmul(x_, r.gate, dense_->gate, T, I, H, DType::BF16, GemmOut::BF16, H, gemm_ws_,
               gemm_ws_bytes_, stream_);
  gemm_.matmul(x_, r.up, dense_->up, T, I, H, DType::BF16, GemmOut::BF16, H, gemm_ws_,
               gemm_ws_bytes_, stream_);
  qwen_asr_swiglu(dense_->gate, dense_->up, dense_->gate, static_cast<int64_t>(M) * I, stream_);
  gemm_.matmul(dense_->gate, r.down, y_, T, H, I, DType::BF16, GemmOut::BF16, I, gemm_ws_,
               gemm_ws_bytes_, stream_);
  glm_residual_add_bf16(resid, y_, n, stream_);
}

QwenAsrModel::Outputs QwenAsrModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  const int H = cfg_.lm.hidden_size;
  gemm_.set_bf12_wide(false);
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  if (audio_spans_ && !run.decode && !run.capture) apply_audio_embeddings(resid_, run.pos0, T);
  Outputs out;
  for (int layer = 0; layer < cfg_.lm.num_hidden_layers; ++layer) {
    enqueue_layer(loader_.load_layer(layer), layer, resid_, T, in.req_ids, in.pos, run.decode,
                  run.capture);
    if (run.capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  glm_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, cfg_.lm.rms_norm_eps, stream_);
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int rows) {
    gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, rows, lm_vocab_count_,
                 H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_,
                 stream_);
  });
  out = finish_run(run, std::move(out), packed_logits);
  return out;
}

QwenAsrModel::Outputs QwenAsrModel::forward(const std::vector<int64_t>& token_ids,
                                            bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.lm.vocab_size)
      throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_))
    throw std::runtime_error("forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.all_rows = true;
  run.capture_layers = capture_layers;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

void QwenAsrModel::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

void QwenAsrModel::graph_prepare() {
  for (int layer = 0; layer < cfg_.lm.num_hidden_layers; ++layer)
    build_layer_objects(loader_.load_layer(layer));
}

// ---------------------------------------------------------------------------
// The audio path (docs/audio.md).
// ---------------------------------------------------------------------------
void QwenAsrModel::stage_audio_rows(const std::vector<AudioInput>& audios) {
  int64_t rows_total = 0;
  for (const auto& au : audios) rows_total += au.tokens;
  if (rows_total > std::min<int64_t>(max_tokens_, kMaxAudioTokens))
    throw std::invalid_argument("Qwen ASR: more audio tokens than the stage holds");
  audio_row_base_.clear();
  const int H = cfg_.lm.hidden_size;
  int64_t row = 0;
  for (const auto& au : audios) {
    audio_row_base_.push_back(row);
    int64_t samples = 0;
    int64_t emitted = 0;
    while (emitted < au.tokens) {
      // One 30 s piece: its own log-mel floor, exactly the HF processor's
      // window (a <= 30 s clip is one piece).
      const size_t take =
          std::min<int64_t>(kSegmentSamples, static_cast<int64_t>(au.pcm.size()) - samples);
      const auto fb = audio::compute_fbank(au.pcm.data() + samples, take);
      samples += static_cast<int64_t>(take);
      const int frames = static_cast<int>(fb.frames);
      const int padded = ((frames + kAudioChunkFrames - 1) / kAudioChunkFrames) * kAudioChunkFrames;
      if (padded < kAudioChunkFrames)
        throw std::logic_error("Qwen ASR: empty audio piece");
      audio_scratch_.assign(static_cast<size_t>(kAudioMelBins) * padded, 0.0f);
      for (int f = 0; f < frames; ++f)
        for (int b = 0; b < kAudioMelBins; ++b)
          audio_scratch_[static_cast<size_t>(b) * padded + f] =
              fb.features[static_cast<size_t>(f) * kAudioMelBins + b];
      std::vector<int32_t> mask(padded, 1);
      std::fill(mask.begin() + frames, mask.end(), 0);
      const int piece = tower_->encode_piece(audio_scratch_.data(), mask.data(), padded,
                                             audio_rows_ + row * H);
      if (piece > au.tokens - emitted)
        throw std::logic_error("Qwen ASR: tower row count drifted from the admission");
      row += piece;
      emitted += piece;
      if (samples >= static_cast<int64_t>(au.pcm.size())) break;
    }
    if (emitted != au.tokens || row - audio_row_base_.back() != au.tokens)
      throw std::logic_error("Qwen ASR: tower produced the wrong row count");
  }
}

void QwenAsrModel::apply_audio_embeddings(uint16_t* resid, int64_t pos0, int T) {
  const int H = cfg_.lm.hidden_size;
  size_t next = 0;
  for (const auto& au : *audio_spans_) {
    const int64_t begin = std::max(au.offset, pos0), end = std::min(au.offset + au.tokens,
                                                                    pos0 + static_cast<int64_t>(T));
    if (end > begin)
      DGPP_CUDA_OK(cudaMemcpy2DAsync(resid + (begin - pos0) * static_cast<int64_t>(H), H * 2,
                                     audio_rows_ + (audio_row_base_[next] + begin - au.offset) * H,
                                     H * 2, H * 2, static_cast<size_t>(end - begin),
                                     cudaMemcpyDeviceToDevice, stream_));
    ++next;
  }
}

QwenAsrModel::Outputs QwenAsrModel::session_prefill_audios(int req,
                                                           const std::vector<int64_t>& prompt,
                                                           const std::vector<AudioInput>& audios) {
  if (req < 0 || req >= max_requests_) throw std::out_of_range("Qwen ASR: request slot");
  validate_audio_inputs(audios, prompt.size());
  for (const auto& au : audios)
    for (int64_t pos = au.offset; pos < au.offset + au.tokens; ++pos)
      if (prompt[static_cast<size_t>(pos)] != cfg_.audio.tokens.pad)
        throw std::invalid_argument("Qwen ASR: audio span does not contain audio tokens");
  stage_audio_rows(audios);
  audio_spans_ = &audios;
  struct Scope {
    QwenAsrModel* m;
    ~Scope() { m->audio_spans_ = nullptr; }
  } scope{this};
  return Base::session_prefill(req, prompt);
}
}  // namespace dgpp
