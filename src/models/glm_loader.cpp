#include "models/glm_loader.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.hpp"
#include "kernels/fp8_dequant.hpp"

namespace dgpp {

namespace {

constexpr size_t kAllocAlign = 256;

size_t align_up_256(size_t b) {
  return (b + kAllocAlign - 1) & ~(kAllocAlign - 1);
}

// One dequant job gathered during the copy phase, launched after all CPU
// writes complete (host access to managed memory during kernel execution is
// the gray zone this two-phase build exists to avoid).
struct DequantJob {
  const uint8_t* payload;
  const float* scales;
  uint16_t* out;
  int64_t rows;
  int64_t cols;
};

}  // namespace

// Managed-memory bump for streamed layer weights (defined here, pimpl'd in
// the header). Counting mode walks the same grant sequence without touching
// memory — layer_bytes() and load_layer() run the SAME build code, so the
// sizing formula cannot drift.
struct GlmLayerBump {
  void* base = nullptr;
  size_t capacity = 0;
  size_t cursor = 0;
  bool counting = false;

  ~GlmLayerBump() {
    if (base) cudaFree(base);
  }
  GlmLayerBump() = default;
  GlmLayerBump(const GlmLayerBump&) = delete;
  GlmLayerBump& operator=(const GlmLayerBump&) = delete;

  void init(size_t cap) {
    DGPP_CUDA_OK(cudaMallocManaged(&base, cap));
    capacity = cap;
    cursor = 0;
  }

  // Every grant is 256-aligned, so a layer's total is exactly the sum of
  // per-tensor aligned sizes — no inter-allocation padding surprises.
  void* alloc(size_t bytes) {
    const size_t grant = align_up_256(bytes);
    if (cursor + grant > capacity)
      throw std::runtime_error("glm loader: layer bump OOM");
    void* p = counting ? nullptr : static_cast<char*>(base) + cursor;
    cursor += grant;
    return p;
  }

  void reset() { cursor = 0; }
};

namespace {

// Everything one layer build needs. `copy` false = counting pass: identical
// allocation sequence, no memcpys, no kernel launches. Names and sizes come
// from the expected table (config-derived); mmap pointers only feed copies.
struct BuildCtx {
  const GlmTextConfig& cfg;
  const std::vector<GlmExpectedTensor>& table;
  const std::unordered_map<std::string, const GlmExpectedTensor*>& by_name;
  GlmLayerBump& bump;
  GlmLayerResident& out;  // pointer stores are harmless in counting mode
  const std::unordered_map<std::string, const TensorInfo*>& tensors;
  std::vector<DequantJob>& jobs;
  bool copy;

  const GlmExpectedTensor& expected(const std::string& name) const {
    auto it = by_name.find(name);
    if (it == by_name.end())
      throw std::runtime_error("glm loader: '" + name +
                               "' missing from expected table (builder bug)");
    return *it->second;
  }

  const TensorInfo& source(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("glm loader: tensor not in checkpoint: " + name);
    return *it->second;
  }

  // Allocates one tensor verbatim (any dtype).
  const void* load_raw(const std::string& name) {
    const GlmExpectedTensor& e = expected(name);
    void* dst = bump.alloc(e.nbytes());
    if (copy) std::memcpy(dst, source(name).data, e.nbytes());
    return dst;
  }
  uint16_t* load_bf16(const std::string& name) {
    return static_cast<uint16_t*>(
        const_cast<void*>(load_raw(name)));
  }
  float* load_f32(const std::string& name) {
    return static_cast<float*>(const_cast<void*>(load_raw(name)));
  }

  // Compressed residency: payload + scales, byte-for-byte.
  GlmQuantMatrix load_quant(const std::string& payload_name) {
    const GlmExpectedTensor& e = expected(payload_name);
    GlmQuantMatrix q;
    q.rows = e.shape[0];
    q.cols = e.shape[1];
    q.payload = static_cast<const uint8_t*>(load_raw(payload_name));
    q.scales = static_cast<const float*>(load_raw(payload_name + "_scale_inv"));
    return q;
  }

  // Transient bridge for the M3 bf16 seam: compressed copies land in the
  // bump, a dequant job writes the BF16 form alongside.
  uint16_t* load_dequant_bf16(const std::string& payload_name) {
    const GlmQuantMatrix q = load_quant(payload_name);
    uint16_t* out = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols) *
                   2));
    jobs.push_back(DequantJob{q.payload, q.scales, out, q.rows, q.cols});
    return out;
  }

  void build_mhc(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.mhc.attn_base = load_f32(p + "hc_attn_base");
    out.mhc.attn_fn = load_bf16(p + "hc_attn_fn");
    out.mhc.attn_scale = load_f32(p + "hc_attn_scale");
    out.mhc.ffn_base = load_f32(p + "hc_ffn_base");
    out.mhc.ffn_fn = load_bf16(p + "hc_ffn_fn");
    out.mhc.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_norms(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.ln1 = load_bf16(p + "input_layernorm.weight");
    out.ln2 = load_bf16(p + "post_attention_layernorm.weight");
  }

  void build_kda(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    // TP=1 views (single-node M4; M5 introduces rank-local geometry).
    const int64_t heads = cfg.kda_num_heads;
    const int64_t head_dim = cfg.kda_head_dim;
    const int64_t proj = heads * head_dim;
    const int64_t hidden = cfg.hidden_size;

    // Merged in_proj rows [f_a | g_a | q | k | v | b] — the M2 kernel
    // contract; the checkpoint stores the six projections separately.
    const int64_t in_rows = 2 * head_dim + 3 * proj + heads;
    uint16_t* in_proj = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(in_rows) * static_cast<size_t>(hidden) *
                   2));
    struct Piece {
      const char* name;
      int64_t rows;
    };
    const Piece pieces[] = {
        {"f_a_proj.weight", head_dim}, {"g_a_proj.weight", head_dim},
        {"q_proj.weight", proj},       {"k_proj.weight", proj},
        {"v_proj.weight", proj},       {"b_proj.weight", heads},
    };
    int64_t row = 0;
    for (const Piece& piece : pieces) {
      const GlmExpectedTensor& e = expected(p + piece.name);
      if (e.shape[0] != piece.rows || e.shape[1] != hidden)
        throw std::runtime_error("glm loader: KDA piece geometry mismatch");
      if (copy)
        std::memcpy(in_proj + static_cast<size_t>(row) * hidden,
                    source(p + piece.name).data, e.nbytes());
      row += piece.rows;
    }
    if (row != in_rows)
      throw std::runtime_error("glm loader: KDA in_proj row count mismatch");

    // Merged causal conv channels [q | k | v].
    const int64_t conv_w = cfg.kda_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(3 * proj) * static_cast<size_t>(conv_w) *
                   2));
    static const char* convs[3] = {"q_conv1d.weight", "k_conv1d.weight",
                                   "v_conv1d.weight"};
    for (int i = 0; i < 3; ++i) {
      const GlmExpectedTensor& e = expected(p + convs[i]);
      if (e.shape[0] != proj || e.shape[2] != conv_w)
        throw std::runtime_error("glm loader: conv piece geometry mismatch");
      if (copy)
        std::memcpy(conv + static_cast<size_t>(i) * static_cast<size_t>(proj) *
                                conv_w,
                    source(p + convs[i]).data, e.nbytes());
    }

    out.kda.in_proj = in_proj;
    out.kda.f_b = load_bf16(p + "f_b_proj.weight");
    out.kda.g_b = load_bf16(p + "g_b_proj.weight");
    out.kda.conv = conv;
    out.kda.a_log = load_f32(p + "A_log");
    out.kda.dt_bias = load_f32(p + "dt_bias");
    out.kda.o_norm = load_bf16(p + "o_norm.weight");
    out.kda.o_proj = load_bf16(p + "o_proj.weight");
  }

  void build_dsa(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    const int64_t hidden = cfg.hidden_size;
    const int64_t q_lora = cfg.q_lora_rank;

    // Fused qkv_a [q_lora + kv_lora, hidden]: q_a rows then kv_a rows, both
    // dequantized (their scale grids differ — two jobs, one buffer).
    const GlmExpectedTensor& qa = expected(p + "q_a_proj.weight");
    const GlmExpectedTensor& kva = expected(p + "kv_a_proj_with_mqa.weight");
    uint16_t* qkv_a = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(qa.shape[0] + kva.shape[0]) *
                   static_cast<size_t>(hidden) * 2));
    const GlmQuantMatrix qa_q = load_quant(p + "q_a_proj.weight");
    jobs.push_back(DequantJob{qa_q.payload, qa_q.scales, qkv_a, qa_q.rows,
                              qa_q.cols});
    const GlmQuantMatrix kva_q =
        load_quant(p + "kv_a_proj_with_mqa.weight");
    jobs.push_back(DequantJob{
        kva_q.payload, kva_q.scales,
        qkv_a + static_cast<size_t>(q_lora) * static_cast<size_t>(hidden),
        kva_q.rows, kva_q.cols});

    out.dsa.qkv_a = qkv_a;
    out.dsa.q_aln = load_bf16(p + "q_a_layernorm.weight");
    out.dsa.kv_aln = load_bf16(p + "kv_a_layernorm.weight");
    out.dsa.q_b = load_dequant_bf16(p + "q_b_proj.weight");
    out.dsa.kv_b = load_bf16(p + "kv_b_proj.weight");
    out.dsa.o_proj = load_dequant_bf16(p + "o_proj.weight");
    const std::string ip = p + "indexer.";
    out.dsa.wq_b = load_bf16(ip + "wq_b.weight");
    out.dsa.wk = load_bf16(ip + "wk.weight");
    out.dsa.wp = load_bf16(ip + "weights_proj.weight");
    out.dsa.gate = load_bf16(ip + "index_kpool_compress_gate");
    out.dsa.k_norm_w = load_bf16(ip + "k_norm.weight");
    out.dsa.k_norm_b = load_bf16(ip + "k_norm.bias");
    // APE: checkpoint BF16 [kpool, index_head_dim]; the M3 kernel wants F32.
    {
      const std::string name = ip + "index_kpool_compress_ape";
      const GlmExpectedTensor& e = expected(name);
      const size_t n = e.numel();
      float* ape = static_cast<float*>(bump.alloc(n * 4));
      if (copy) {
        const uint16_t* src =
            static_cast<const uint16_t*>(source(name).data);
        for (size_t i = 0; i < n; ++i)
          ape[i] = bf16_bits_to_float(src[i]);
      }
      out.dsa.ape = ape;
    }
  }

  void build_dense_mlp(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.dense[0] = load_quant(p + "mlp.gate_proj.weight");
    out.dense[1] = load_quant(p + "mlp.up_proj.weight");
    out.dense[2] = load_quant(p + "mlp.down_proj.weight");
  }

  void build_moe(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.moe.router_gate = load_bf16(p + "mlp.gate.weight");
    out.moe.router_bias = load_f32(p + "mlp.gate.e_score_correction_bias");
    const std::string sp = p + "mlp.shared_experts.";
    out.moe.shared[0] = load_quant(sp + "gate_proj.weight");
    out.moe.shared[1] = load_quant(sp + "up_proj.weight");
    out.moe.shared[2] = load_quant(sp + "down_proj.weight");
    out.moe.experts.resize(
        static_cast<size_t>(cfg.n_routed_experts) * 3);
    for (int e = 0; e < cfg.n_routed_experts; ++e) {
      const std::string ep =
          p + "mlp.experts." + std::to_string(e) + ".";
      out.moe.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant(ep + "gate_proj.weight");
      out.moe.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant(ep + "up_proj.weight");
      out.moe.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant(ep + "down_proj.weight");
    }
  }

  void build_mtp_head(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.enorm = load_bf16(p + "enorm.weight");
    out.hnorm = load_bf16(p + "hnorm.weight");
    out.eh_proj = load_bf16(p + "eh_proj.weight");
    out.shared_head_norm = load_bf16(p + "shared_head.norm.weight");
  }

  void build_layer(int layer) {
    const bool is_mtp = layer == cfg.mtp_layer();
    const int max_layer =
        cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      throw std::invalid_argument(
          "glm loader: layer index out of range: " + std::to_string(layer));

    out.layer = layer;
    out.kind = is_mtp || cfg.layers[layer] == GlmLayerKind::Dsa
                   ? GlmLayerKind::Dsa
                   : GlmLayerKind::Kda;
    if (!is_mtp) build_mhc(layer);
    build_norms(layer);
    if (out.kind == GlmLayerKind::Kda)
      build_kda(layer);
    else
      build_dsa(layer);
    const bool dense =
        !is_mtp && cfg.mlps[layer] == GlmMlpKind::Dense;
    if (dense)
      build_dense_mlp(layer);
    else
      build_moe(layer);
    if (is_mtp) build_mtp_head(layer);
  }
};

// Runs one layer's build in counting mode; returns the exact byte total.
size_t count_layer_bytes(const GlmTextConfig& cfg, int layer) {
  GlmLayerBump bump;
  bump.counting = true;
  bump.capacity = SIZE_MAX;
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  GlmLayerResident scratch;
  std::vector<DequantJob> jobs;
  std::unordered_map<std::string, const TensorInfo*> no_tensors;
  BuildCtx ctx{cfg,        table, by_name, bump,  scratch,
               no_tensors, jobs,  false};
  ctx.build_layer(layer);
  return bump.cursor;
}

}  // namespace

size_t GlmLayerStream::layer_bytes(const GlmTextConfig& cfg, int layer) {
  return count_layer_bytes(cfg, layer);
}

size_t GlmLayerStream::globals_bytes(const GlmTextConfig& cfg) {
  const size_t vocab_bytes = align_up_256(static_cast<size_t>(cfg.vocab_size) *
                                          cfg.hidden_size * 2);
  const size_t norm_bytes = align_up_256(static_cast<size_t>(cfg.hidden_size) * 2);
  return 2 * vocab_bytes + norm_bytes;
}

GlmLayerStream::GlmLayerStream(const GlmTextConfig& cfg,
                               const std::string& checkpoint_dir)
    : cfg_(cfg),
      layer_bump_(std::make_unique<GlmLayerBump>()),
      globals_bump_(std::make_unique<GlmLayerBump>()) {
  // Open shards (sorted for determinism), index headers, validate the full
  // text binding before a single byte of payload moves.
  namespace fs = std::filesystem;
  std::vector<fs::path> shard_paths;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors")
      shard_paths.push_back(entry.path());
  if (shard_paths.empty())
    throw std::runtime_error("glm loader: no .safetensors shards in " +
                             checkpoint_dir);
  std::sort(shard_paths.begin(), shard_paths.end());

  std::unordered_map<std::string, GlmTensorDesc> present;
  for (const auto& path : shard_paths) {
    auto f = SafetensorsFile::open(path.string());
    f->for_each([&](const TensorInfo& t) {
      auto [it, inserted] =
          tensors_.emplace(t.name, &t);
      if (!inserted)
        throw std::runtime_error("glm loader: duplicate tensor '" + t.name +
                                 "' in " + path.string());
      present.emplace(t.name, GlmTensorDesc{t.dtype, t.shape});
    });
    shards_.push_back(std::move(f));
  }

  const GlmBindReport rep = glm_validate_text_binding(cfg_, present);
  if (!rep.ok()) {
    std::string msg = "glm loader: checkpoint binding failed: ";
    for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
      if (i) msg += "; ";
      msg += rep.errors[i];
    }
    throw std::runtime_error(msg);
  }

  // Size the bump at the largest layer (headers say which; formula says
  // how much) and take a dedicated stream for the dequant launches.
  size_t capacity = 0;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  for (int i = 0; i < max_layer; ++i)
    capacity = std::max(capacity, count_layer_bytes(cfg_, i));
  layer_bump_->init(capacity);
  globals_bump_->init(globals_bytes(cfg_));
  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
}

GlmLayerStream::~GlmLayerStream() {
  if (stream_) cudaStreamDestroy(stream_);
}

size_t GlmLayerStream::layer_capacity() const {
  return layer_bump_->capacity;
}

const GlmLayerResident& GlmLayerStream::load_layer(int layer) {
  if (resident_.layer == layer) return resident_;

  layer_bump_->reset();
  resident_ = GlmLayerResident{};

  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg_, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);

  std::vector<DequantJob> jobs;
  BuildCtx ctx{cfg_, table, by_name, *layer_bump_, resident_, tensors_, jobs,
               true};
  ctx.build_layer(layer);

  // Phase two: all CPU writes are done, launch the dequants and wait.
  for (const DequantJob& j : jobs)
    launch_fp8_dequant_blocks(j.payload, j.scales, j.out, j.rows, j.cols,
                              stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

  // The formula and the allocator share the build code; anything but
  // equality is a bug that must never pass silently.
  const size_t used = layer_bump_->cursor;
  const size_t expected_bytes = layer_bytes(cfg_, layer);
  if (used != expected_bytes)
    throw std::runtime_error(
        "glm loader: byte-formula drift on layer " + std::to_string(layer) +
        ": used " + std::to_string(used) + " != formula " +
        std::to_string(expected_bytes));
  resident_.bytes = used;
  return resident_;
}

const GlmGlobalsResident& GlmLayerStream::load_globals() {
  if (globals_.embed) return globals_;
  globals_bump_->reset();
  globals_ = GlmGlobalsResident{};

  auto copy_global = [&](const std::string& name) -> uint16_t* {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: global tensor missing: " + name);
    const TensorInfo& t = *it->second;
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(t.nbytes()));
    std::memcpy(dst, t.data, t.nbytes());
    return dst;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  globals_.lm_head = copy_global("lm_head.weight");
  globals_.final_norm = copy_global("model.language_model.norm.weight");
  globals_.bytes = globals_bump_->cursor;

  const size_t expected_bytes = globals_bytes(cfg_);
  if (globals_.bytes != expected_bytes)
    throw std::runtime_error(
        "glm loader: globals byte-formula drift: used " +
        std::to_string(globals_.bytes) + " != formula " +
        std::to_string(expected_bytes));
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  return globals_;
}

void GlmLayerStream::release_layer() {
  resident_ = GlmLayerResident{};
  layer_bump_->reset();
}

void GlmLayerStream::release_globals() {
  globals_ = GlmGlobalsResident{};
  globals_bump_->reset();
}

}  // namespace dgpp
