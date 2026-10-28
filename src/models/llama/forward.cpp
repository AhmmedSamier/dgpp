#include "models/llama/forward.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

}  // namespace

LlamaModel::LlamaModel(const LlamaTextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                       int64_t max_cache_tokens, LlamaResidency residency, BoundaryReducer* boundary,
                       int tp_rank, int tp_world, int max_requests, bool mtp, int decode_rows,
                       bool serving_logits)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? LlamaHeadSharding::VocabSharded : LlamaHeadSharding::Full,
              /*resident_mtp=*/false) {
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("LlamaModel: decode_rows exceeds the limit of 32");
  if (max_tokens <= 0) throw std::invalid_argument("LlamaModel: max_tokens must be positive");
  if (mtp) throw std::invalid_argument("LlamaModel: the checkpoint ships no draft layer (mtp)");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("LlamaModel: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "LlamaModel: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("LlamaModel: the config names no EOS token");
  if (cfg_.head_dim != kGlm4HeadDim) throw std::invalid_argument("LlamaModel: 128-wide heads (the kernels' shape)");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens =
        ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
    sp.rank = tp_rank;
    sp.world = tp_world;
    sp.boundary = boundary;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.logits_rows = compact_logits_rows(serving_logits, decode_rows, max_requests);
    sp.mtp = false;
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = globals_.lm_vocab_begin;
    sp.lm_vocab_count = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    sp.snapshot_align = 1;  // no recurrent state, any position snapshots
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = LlamaGemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // engine.dense_weights = "fp8": every scale GEMM row from row one takes
  // the streaming tensor-core form (kernels/scale_gemm.hpp) — one chain
  // for every m.
  gw_.mma_from_rows = 1;
  // The attention projections' lowering (kernels/gemm.hpp dense_gemv_rows):
  // the GEMV chunks to the bound, cuBLASLt's algorithm above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  n_split_ = LlamaAttentionLayer::default_decode_splits();

  const LlamaLocalGeometry& geo = loader_.geometry();
  {
    LlamaKvPoolShape shape;
    shape.layers = cfg_.num_hidden_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg_.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    pool_.init(shape);
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

LlamaModel::~LlamaModel() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(gemm_ws_);
}

LlamaModel::MemoryPlan LlamaModel::plan_memory(const LlamaTextConfig& cfg, int max_tokens,
                                               int64_t max_cache_tokens, int tp_rank, int tp_world,
                                               LlamaResidency residency, int max_requests, bool mtp,
                                               int decode_rows, bool serving_logits) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the limit of 32");
  if (mtp) throw std::invalid_argument("plan_memory: the checkpoint ships no draft layer (mtp)");
  // The fixed batch's row ceiling, floored as the session core floors it.
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const LlamaHeadSharding head = tp_world > 1 ? LlamaHeadSharding::VocabSharded : LlamaHeadSharding::Full;
  const LlamaLocalGeometry geo = LlamaLocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(LlamaLayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int n_split = LlamaAttentionLayer::default_decode_splits();

  if (residency == LlamaResidency::Resident) {
    // bf12-only residency: the packable bf16 matrices load aside and give
    // their bytes back as each layer is packed (graph_prepare).
    const bool bf12_only = Bf12Companions::packed_only() && residency == LlamaResidency::Resident;
    size_t resident =
        LlamaLayerStream::resident_bytes(cfg, tp_rank, tp_world, head, /*with_mtp=*/false);
    if (bf12_only) resident -= LlamaLayerStream::side_bytes(cfg, tp_rank, tp_world, head, /*with_mtp=*/false);
    plan.add(bf12_only ? "model weights (resident, packed bf16 matrices released)" : "model weights (resident)",
             resident);
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             LlamaLayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, /*with_mtp=*/false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, LlamaLayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + LlamaLayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  if (Bf12Companions::enabled() && residency == LlamaResidency::Resident &&
      !LlamaLayerStream::dense_weights_fp8() && !cfg.weights_nvfp4) {
    // The lossless 12-bit companions of every layer's q/k/v/o projections
    // and gate/up/down matrices plus the lm head (kernels/bf12_companions.hpp).
    // engine.dense_weights = "fp8" plans none — its projection+head bytes
    // are already resident in the fp8 form; the NVFP4 checkpoint likewise
    // (no bf16 projection bytes, and the model refuses the pairing).
    const int64_t Hh = cfg.hidden_size;
    const int64_t Q = static_cast<int64_t>(geo.local_heads) * cfg.head_dim;
    const int64_t KV = static_cast<int64_t>(geo.local_kv_heads) * cfg.head_dim;
    const int64_t I = geo.local_inter;
    size_t packed =
        static_cast<size_t>(cfg.num_hidden_layers) *
            (Bf12Companions::planned_bytes(Q, Hh) + Bf12Companions::planned_bytes(KV, Hh) +
             Bf12Companions::planned_bytes(KV, Hh) + Bf12Companions::planned_bytes(Hh, Q) +
             Bf12Companions::planned_bytes(I, Hh) + Bf12Companions::planned_bytes(I, Hh) +
             Bf12Companions::planned_bytes(Hh, I)) +
        Bf12Companions::planned_bytes(static_cast<int64_t>(V), Hh);
    plan.add("bf16 decode packing (12-bit companions)", packed);
    if (Bf12Companions::packed_only()) {
      plan.add("bf16 prefill expansion scratch",
               static_cast<size_t>(kBf12ExpandSlots) * Bf12Companions::planned_slot_bytes(
                   {static_cast<size_t>(Q) * static_cast<size_t>(Hh) * 2,
                    static_cast<size_t>(KV) * static_cast<size_t>(Hh) * 2,
                    static_cast<size_t>(Hh) * static_cast<size_t>(Q) * 2,
                    static_cast<size_t>(I) * static_cast<size_t>(Hh) * 2,
                    static_cast<size_t>(Hh) * static_cast<size_t>(I) * 2,
                    V * static_cast<size_t>(Hh) * 2}));
    }
  }
  {
    LlamaKvPoolShape shape;
    shape.layers = cfg.num_hidden_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add("kv cache pool (K/V bf16)", LlamaKvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), /*mtp=*/false,
                            cfg.hidden_size, &core_dev, &core_pin, rows, serving_logits);
    plan.add("activations (session core, residual, block io)", core_dev + 3 * M * H * 2, core_pin);
  }
  plan.add("layer objects (attention, dense MLP)",
           LlamaAttentionLayer::scratch_bytes(cfg, geo.local_heads, geo.local_kv_heads, max_tokens, rows,
                                              n_split, cfg.weights_nvfp4) +
               LlamaDenseMlp::scratch_bytes(cfg, geo.local_inter, max_tokens));
  return plan;
}

size_t LlamaModel::session_snapshot_bytes(const LlamaTextConfig&, int, bool) { return kSnapshotStamp; }

void LlamaModel::build_layer_objects(const LlamaLayerResident& r) {
  if (!attn_) {
    attn_ = std::make_unique<LlamaAttentionLayer>(r.attn, gw_, cfg_, max_tokens_, max_decode_rows_, n_split_);
  } else {
    attn_->rebind(r.attn);
  }
  if (!mlp_) {
    mlp_ = std::make_unique<LlamaDenseMlp>(r.mlp, gw_, cfg_, max_tokens_, max_decode_rows_);
  } else {
    mlp_->rebind(r.mlp);
  }
}

// The state every row walk starts from on a fresh request: no blocks
// (the K/V rows are written before any row reads them).
void LlamaModel::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

GlmSpecSegments LlamaModel::spec_segments(int, int) const { return GlmSpecSegments{}; }

void LlamaModel::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

void LlamaModel::mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int, int) {
  refuse_mtp("mtp_run_rows");
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// The boundary folds: the producer writes its partial into the
// reducer's staged buffer when the shape fits (the collective sends
// straight from there; under capture the recorder's one stable buffer),
// else into `fallback`. The eager producer quiesces before the collective;
// under capture the fold is a recorded node and the stream order IS the
// drain.
uint16_t* LlamaModel::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void LlamaModel::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void LlamaModel::enqueue_layer(const LlamaLayerResident& r, int pool_layer, uint16_t* resid, int T,
                               const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t n = static_cast<int64_t>(T) * H;
  build_layer_objects(r);
  const auto groups = [&](bool combine_readers, auto&& enqueue) {
    if (rows.run)
      rows.run->for_each_kernel_group([&](int count, int next) {
        return combine_readers || (count > 128 && next > 128);
      }, enqueue);
    else
      enqueue(0, T, rows.decode, 0);
  };
  // ---- the attention block --------------------------------------------------
  glm_rmsnorm_bf16(resid, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  groups(false, [&](int row0, int count, bool decode, int) {
    LlamaAttnRows arows;
    arows.req_ids = rows.req_ids + row0;
    arows.pos = rows.pos + row0;
    arows.decode = decode;
    gemm_.set_bf12_wide(decode);  // the decode batch's packed launches (kernels/gemm.hpp)
    const size_t offset = static_cast<size_t>(row0) * H;
    attn_->enqueue(x_ + offset, count, arows, pool_.view(pool_layer), attn_out + offset, stream_);
  });
  gemm_.set_bf12_wide(rows.decode);
  fold(attn_out, T, H, rows.capture);  // block boundary 1: o_proj's partial
  // ---- the MLP block ----------------------------------------------------------
  // The attention residual add rides the post norm's first pass (one
  // kernel, bitwise the pair; kernels/glm_norm.hpp).
  glm_residual_add_rmsnorm_bf16(resid, attn_out, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  groups(false, [&](int row0, int count, bool decode, int) {
    gemm_.set_bf12_wide(decode);  // the decode batch's packed launches (kernels/gemm.hpp)
    const size_t offset = static_cast<size_t>(row0) * H;
    mlp_->enqueue(x_ + offset, count, ffn_out + offset, stream_);
  });
  gemm_.set_bf12_wide(rows.decode);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  glm_residual_add_bf16(resid, ffn_out, n, stream_);
}

LlamaModel::Outputs LlamaModel::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != LlamaResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  // The packed companions' wide launches are the decode batch's alone (a
  // short prefill chunk keeps its Lt algorithm: kernels/gemm.hpp).
  gemm_.set_bf12_wide(run.decode);
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  WalkRows rows;
  rows.run = &run;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.req = req;
  rows.decode = run.decode;
  rows.capture = run.capture;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const LlamaLayerResident& r = loader_.load_layer(layer);
    enqueue_layer(r, layer, resid_, T, rows);
    if (run.capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  // Keep hidden rows for the head; serving projects only each request's
  // last prefill row. All-row diagnostics and decode keep their full head.
  glm_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int rows) {
    gemm_.set_bf12_wide(run.decode || input_row < run.decode_rows());
    if (globals_.lm_head_fp8.payload) {
      // The fp8 head (engine.dense_weights = "fp8"): the scale GEMM,
      // every row from row one on the streaming tensor-core form.
      launch_scale_gemm_f32(h_ + static_cast<size_t>(input_row) * H, H, globals_.lm_head_fp8.payload,
                            globals_.lm_head_fp8.scales,
                            logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, rows, lm_vocab_count_,
                            H, stream_, static_cast<size_t>(lm_vocab_count_), gw_.mma_from_rows,
                            /*last_row_only=*/false, gemm_ws_, gemm_ws_bytes_);
      return;
    }
    gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, rows, lm_vocab_count_,
                 H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_,
                 stream_);
  });
  store_run_draft_hidden(run, in, h_);
  out = finish_run(run, std::move(out), packed_logits);
  return out;
}

// ---------------------------------------------------------------------------
// The cold diagnostic forward: slot 0, fresh state, every row.
// ---------------------------------------------------------------------------
LlamaModel::Outputs LlamaModel::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_)) throw std::runtime_error("forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture_layers = capture_layers;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

// ---------------------------------------------------------------------------
// The graph era.
// ---------------------------------------------------------------------------
void LlamaModel::graph_prepare() {
  if (loader_.residency() != LlamaResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  // The bf16 decode weights' 12-bit companions are packed as each layer
  // lands (the caches already exist: under bf12-only residency the layer's
  // bf16 bytes go back at once, so no more than one layer's sit beside
  // their companions). engine.dense_weights = "fp8" packs none — the fp8
  // form is already resident and the bf16 pointers are null. The NVFP4
  // checkpoint packs none either (no bf16 projection pointers at all).
  const bool pack = Bf12Companions::enabled() && !bf12_built_ && !LlamaLayerStream::dense_weights_fp8() &&
                    !LlamaLayerStream::weights_nvfp4();
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const LlamaLayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    if (pack) pack_layer_companions(layer, r);
  }
  if (pack) finish_companions();
}

// The lossless 12-bit companions of the decode GEMV's bf16 weights
// (engine.bf16_weights = "bf12"): every layer's q/k/v/o projections and
// gate/up/down matrices plus the lm head, packed as each layer lands and
// before any capture. Prefill and the batches past eight rows keep the
// bf16 bytes — or, under bf12-only residency (the bytes returned here,
// layer by layer), expand them and take the packed launches
// (kernels/gemm.hpp).
void LlamaModel::pack_layer_companions(int layer, const LlamaLayerResident& r) {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto release = [&](const void* w) { return loader_.release_packed(layer, w); };
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    bf12_.pack_and_release(w, n, k, gemm_, stream_, release);
  };
  const LlamaAttnResident& a = r.attn;
  const int64_t Q = static_cast<int64_t>(a.local_heads) * cfg_.head_dim;
  const int64_t KV = static_cast<int64_t>(a.local_kv_heads) * cfg_.head_dim;
  pack(a.q_proj, Q, H);
  pack(a.k_proj, KV, H);
  pack(a.v_proj, KV, H);
  pack(a.o_proj, H, Q);
  const LlamaDenseMlpResident& m = r.mlp;
  const int64_t I = m.local_inter;
  pack(m.gate, I, H);
  pack(m.up, I, H);
  pack(m.down, H, I);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The lm head's, the expansion scratch (one slot: no walk of this family
// calls a matrix twice), and the summary.
void LlamaModel::finish_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  bf12_built_ = true;
  bf12_.pack_and_release(globals_.lm_head, lm_vocab_count_, cfg_.hidden_size, gemm_, stream_,
                         [&](const void* w) { return loader_.release_packed(-1, w); });
  bf12_.finish(gemm_, kBf12ExpandSlots);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(loader_.rank(), bf12_s_);
}

}  // namespace dgpp
