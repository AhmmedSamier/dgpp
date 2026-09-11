// OpenAI-compatible text service for GLM-5.3, Qwen3.8 and GLM-4.7.
// Rank 0 owns HTTP ingress, tokenization and template rendering. Its
// engine thread drains admissions and cancellations, journals them to
// peers, then runs one scheduler tick. Peers apply the record and tick
// without an HTTP frontend.
//
// Multi-rank execution uses resident weights and distributed picks.
// Single-node graph mode also uses resident weights, with identity bus
// collectives. Single-node mode without graph decode streams layers
// through the eager engine. The memory plan validates the chosen shape.
//
// Threads: HTTP parses requests and writes responses; the engine owns
// model execution; journal peers follow rank 0's scheduler operations.
// See DESIGN §11 and docs/operations.md.
//

// USAGE
//   dgpp-serve --model ORG/NAME | --checkpoint-dir DIR
//     [--port N (default 18080; rank 0 only)] [--kv-capacity TOKENS]
//     [--kv-dtype bf16|fp8|fp4]
//     [--max-concurrency N] [--queue-limit N] [--default-max-tokens N]
//     [--max-connections N] [--no-eos]
//   fabric: --world N --rank R (--peer HOST when rank > 0)
//     [--fabric-port N (29970)] [--journal-port N (29971)]
//     [--rendezvous-timeout-ms N (120000)]
//     [--decode-graph [--mtp]] [--graph-batch-min-live N]
//       (adaptive scalar/fixed batch; concurrency * T <= 8; N defaults to
//        min(4, max-concurrency) and must lie in [1, max-concurrency])
//   sampling (M6 6b): the defaults come from generation_config.json;
//     [--temperature X] [--top-p X] [--top-k N] [--min-p X]
//     [--repetition-penalty X] override them for the process, [--seed N]
//     fixes the seed of every request that omits one.
//   tools and reasoning (M6 6f): requests may carry tools / tool_choice /
//     reasoning_effort / chat_template_kwargs; the response splits
//     reasoning_content, content and tool_calls from the token ids on
//     rank 0 (DESIGN §11); [--reasoning-in-content] folds the reasoning
//     into content for clients that expect the raw transcript. Every engine samples
//     exactly (DESIGN §10, the device path; §9 under --mtp, the exact
//     speculative accept test on the device).
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "common/process_memory.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/hf_cache.hpp"
#include "serve/cluster_config.hpp"
#include "dgpp_version.hpp"
#include "text/chat_template.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine/memory_plan.hpp"
#include "loaders/architecture.hpp"
#include "models/glm/fabric_engine.hpp"
#include "models/glm/forward.hpp"
#include "models/glm/gen_engine.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"
#include "models/glm4/config.hpp"
#include "models/glm4/forward.hpp"
#include "sched/scheduler.hpp"
#include "text/tokenizer.hpp"
#include "text/tool_grammar.hpp"
#include "net/collective_bus.hpp"
#include "serve/fabric_serve.hpp"
#include "serve/generation_service.hpp"
#include "serve/frontend.hpp"
#include "serve/http_server.hpp"

namespace fs = std::filesystem;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// SIGINT/SIGTERM → the orderly stop (flush, then exit). Installed for
// every rank: peers poll the flag between journal reads.
std::atomic<bool> g_stop_requested{false};
std::atomic<int> g_stop_signals{0};
void on_signal(int) {
  g_stop_requested.store(true);
  g_stop_signals.fetch_add(1);
}

// A PEER never leaves on its own signal (M6 6c): it follows rank 0's
// journal to the stop record, which rank 0 sends only after its final
// pass — so the bus never comes down under a collective on either side.
// A second signal forces the exit, under whatever is in flight.
bool peer_should_stop(int rank) {
  static std::atomic<bool> warned{false};
  const int n = g_stop_signals.load();
  if (n == 1 && !warned.exchange(true))
    DGPP_LOG_WARN(
        "rank {}: signal — following rank 0's journal to its stop record "
        "(drain-on-stop); a second signal exits now, under whatever "
        "collective is in flight",
        rank);
  return n >= 2;
}

void write_ops_file(const std::string& path, const std::string& text) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    DGPP_LOG_ERROR("serve: cannot write {}", path);
    return;
  }
  const size_t n = std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
  if (n != text.size()) DGPP_LOG_ERROR("serve: short write to {}", path);
}

struct ServeKnobs {
  uint16_t http_port = 8080;
  std::string http_bind = "127.0.0.1";
  int max_connections = 64;
  int queue_limit = 64;
  int default_max_tokens = 256;
  dgpp::sample::Params sampling_defaults = dgpp::sample::greedy_params();
  std::optional<uint64_t> fixed_seed;
  bool reasoning_in_content = false;
  dgpp::sched::AdmissionPolicy admission;  // M6 6d: full (default) or grow
  double stats_interval_s = 10.0;  // the throughput line's period; 0 = off
  bool mtp = false;                // the throughput line's MTP group
};

// Pinned words for the sampler's collectives, allocated BEFORE the world
// forms (the allocation discipline) and released after the bus stops.
struct PinnedWords {
  uint16_t* data = nullptr;
  explicit PinnedWords(size_t elems) {
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&data),
                               sizeof(uint16_t) * std::max<size_t>(elems, 2),
                               cudaHostAllocDefault));
  }
  ~PinnedWords() {
    if (data) cudaFreeHost(data);
  }
  PinnedWords(const PinnedWords&) = delete;
  PinnedWords& operator=(const PinnedWords&) = delete;
};

// The pre-flight memory check: every byte the model, the
// prefix arena and the engine will allocate, from the same formulas their
// constructors use, against the node's free memory — BEFORE the first
// allocation. A configuration that does not fit is refused here with the
// plan itemized and the largest context that would fit named; the
// alternative was the node driven into its memory watermark (the 262k-token
// context that froze the fabric and needed a reboot). The measure is the
// larger of the device's free memory and the host's MemAvailable (the
// GB10's unified pool is the host's memory; the page cache is reclaimable),
// as the loader's own resident-footprint check measures.
constexpr size_t kMemoryHeadroomBytes = size_t{8} << 30;  // CUDA context, bus, cuBLAS, page cache churn

std::string gib(double bytes) {
  return std::format("{:.2f} GiB", bytes / (1024.0 * 1024.0 * 1024.0));
}

// ---- the family interface ------------------------------------
// What the boot needs from a model family — its config facts, its memory
// plan, its model and its engines — so one server boots GLM-5.3-Flash and
// Qwen3.8-Flash-Next from config.json's architecture. The engine adapters
// are templates over the model (engine/*.hpp); the family owns the model
// and hands out the adapters behind the scheduler's engine interface.
struct ServeGraphEngine {
  virtual ~ServeGraphEngine() = default;
  virtual dgpp::sched::SchedulerEngine* engine() = 0;
  virtual void warm_captures(const std::vector<int64_t>& prompt) = 0;
};

template <class Model>
struct ServeGraphEngineOf final : ServeGraphEngine {
  dgpp::GraphEngineAdapter<Model> eng;
  template <class... A>
  explicit ServeGraphEngineOf(A&&... a) : eng(std::forward<A>(a)...) {}
  dgpp::sched::SchedulerEngine* engine() override { return &eng; }
  void warm_captures(const std::vector<int64_t>& p) override { eng.warm_captures(p); }
};

struct ServeFamily {
  virtual ~ServeFamily() = default;
  virtual const char* name() const = 0;
  virtual int64_t vocab_size() const = 0;
  virtual std::vector<int64_t>& eos_token_ids() = 0;
  virtual int64_t block_tokens() const = 0;
  virtual int prefill_chunk_tokens() const = 0;
  // Empty when a pool of `pool_tokens` fits the family's id spaces.
  virtual std::string pool_check(int64_t pool_tokens) const = 0;
  virtual const char* kv_format_name() const = 0;
  // The decode rows the family serves in one fixed batch at most (2026-09-10,
  // engine/decode_outputs.hpp): the session-core families take the derived
  // shape up to kDecodeRowsMax; GLM-5.3-Flash keeps its build-time 8, and
  // Qwen3.8-Flash-Next stays at 8 until its kernels are gated past it.
  virtual int decode_rows_cap() const = 0;
  // The bus's latency slot: the family's widest recorded decode fold
  // (`decode_rows` bf16 rows of its boundary width).
  virtual size_t lat_slot_bytes(int decode_rows) const = 0;
  virtual dgpp::MemoryPlan plan(int forward_rows, int64_t context, int rank, int world, bool fabric,
                                int slots, bool mtp, int decode_rows) const = 0;
  virtual size_t snapshot_bytes(int world, bool mtp) const = 0;
  virtual void build_model(dgpp::BoundaryReducer* reducer, int rank, int world, bool fabric,
                           int forward_rows, int64_t pool_tokens, int slots, bool mtp, int decode_rows) = 0;
  virtual void destroy_model() = 0;
  virtual size_t model_snapshot_bytes() const = 0;
  virtual std::unique_ptr<ServeGraphEngine> make_graph_engine(
      dgpp::net::CollectiveBus* bus, int rank, int world, uint16_t* pick_scratch,
      int batch_min_live, uint16_t* prefix_scratch, uint16_t* gather_scratch, int candidates,
      const dgpp::text::GrammarVocab* grammar, int prefix_slots, int mtp_depth) = 0;
  virtual std::unique_ptr<dgpp::sched::SchedulerEngine> make_eager_engine(
      int slots, dgpp::DecodePick pick, dgpp::DecodeSample sample,
      const dgpp::text::GrammarVocab* grammar, int prefix_slots) = 0;
};

// GLM-5.3-Flash: the DSA pool's block and pool-id geometry, the latent
// cache format, the resident vocab-sharded fabric model / the streaming
// world-1 one.
struct GlmFamily final : ServeFamily {
  dgpp::GlmTextConfig cfg;
  std::string ckpt;
  int world = 1;
  dgpp::LatentFormat kv_format = dgpp::LatentFormat::kBf16;
  std::unique_ptr<dgpp::GlmDiagnosticModel> model;
  GlmFamily(const std::string& checkpoint, int world_, dgpp::LatentFormat fmt)
      : cfg(dgpp::GlmTextConfig::from_json_file((fs::path(checkpoint) / "config.json").string())),
        ckpt(checkpoint), world(world_), kv_format(fmt) {}
  dgpp::DsaConfig dsa() const {
    dgpp::DsaConfig d = cfg.dsa_config();
    d.tp_size = world;
    d.latent_format = kv_format;
    return d;
  }
  const char* name() const override { return "glm5"; }
  int64_t vocab_size() const override { return cfg.vocab_size; }
  std::vector<int64_t>& eos_token_ids() override { return cfg.eos_token_ids; }
  int64_t block_tokens() const override { return dsa().block_tokens; }
  int prefill_chunk_tokens() const override { return dgpp::GlmDiagnosticModel::prefill_chunk_tokens(); }
  std::string pool_check(int64_t pool_tokens) const override {
    const dgpp::DsaConfig d = dsa();
    const int64_t pools = (pool_tokens / d.block_tokens) * dgpp::DsaGeometry::from_config(d).pools_per_block;
    if (pools >= (int64_t(1) << 21)) return "exceeds the DSA pool-id space (2^21 pools)";
    return "";
  }
  const char* kv_format_name() const override { return dgpp::latent_format_name(kv_format); }
  int decode_rows_cap() const override { return dgpp::GlmDiagnosticModel::kDecodeRows; }
  size_t lat_slot_bytes(int) const override {
    return static_cast<size_t>(dgpp::GlmDiagnosticModel::kDecodeRows) * static_cast<size_t>(cfg.hidden_size) * 2;
  }
  dgpp::MemoryPlan plan(int forward_rows, int64_t context, int rank, int world_, bool fabric, int slots,
                        bool mtp, int) const override {
    return dgpp::GlmDiagnosticModel::plan_memory(
        cfg, forward_rows, context, fabric ? rank : 0, fabric ? world_ : 1,
        fabric ? dgpp::GlmResidency::Resident : dgpp::GlmResidency::Streaming,
        fabric ? dgpp::GlmHeadSharding::VocabSharded : dgpp::GlmHeadSharding::Full, slots, fabric && mtp,
        kv_format);
  }
  size_t snapshot_bytes(int world_, bool mtp) const override {
    return dgpp::GlmDiagnosticModel::session_snapshot_bytes(cfg, world_, mtp);
  }
  void build_model(dgpp::BoundaryReducer* reducer, int rank, int world_, bool fabric, int forward_rows,
                   int64_t pool_tokens, int slots, bool mtp, int) override {
    model = std::make_unique<dgpp::GlmDiagnosticModel>(
        cfg, ckpt, forward_rows, pool_tokens, reducer, fabric ? rank : 0, fabric ? world_ : 1,
        fabric ? dgpp::GlmResidency::Resident : dgpp::GlmResidency::Streaming,
        fabric ? dgpp::GlmHeadSharding::VocabSharded : dgpp::GlmHeadSharding::Full, slots, fabric && mtp,
        kv_format);
  }
  void destroy_model() override { model.reset(); }
  size_t model_snapshot_bytes() const override { return model ? model->session_snapshot_bytes() : 0; }
  std::unique_ptr<ServeGraphEngine> make_graph_engine(
      dgpp::net::CollectiveBus* bus, int rank, int world_, uint16_t* pick_scratch, int batch_min_live,
      uint16_t* prefix_scratch, uint16_t* gather_scratch, int candidates,
      const dgpp::text::GrammarVocab* grammar, int prefix_slots, int mtp_depth) override {
    return std::make_unique<ServeGraphEngineOf<dgpp::GlmDiagnosticModel>>(
        model.get(), bus, rank, world_, pick_scratch, cfg.vocab_size, /*pick_timeout_ms=*/60000,
        batch_min_live, prefix_scratch, gather_scratch, candidates, grammar, prefix_slots, mtp_depth);
  }
  std::unique_ptr<dgpp::sched::SchedulerEngine> make_eager_engine(
      int slots, dgpp::DecodePick pick, dgpp::DecodeSample sample, const dgpp::text::GrammarVocab* grammar,
      int prefix_slots) override {
    return std::make_unique<dgpp::EagerEngineAdapter<dgpp::GlmDiagnosticModel>>(
        model.get(), slots, std::move(pick), std::move(sample), grammar, prefix_slots);
  }
};

// Qwen3.8-Flash-Next: the paged K/V + compressed-key pool (64-token
// blocks, pool ids in 21 bits), bf16 caches, the resident fabric model /
// the streaming world-1 one (172 GiB does not fit one rank resident).
struct QwenFamily final : ServeFamily {
  dgpp::QwenTextConfig cfg;
  std::string ckpt;
  std::unique_ptr<dgpp::QwenModel> model;
  explicit QwenFamily(const std::string& checkpoint)
      : cfg(dgpp::QwenTextConfig::from_json_file((fs::path(checkpoint) / "config.json").string())),
        ckpt(checkpoint) {}
  const char* name() const override { return "qwen4_exp"; }
  int64_t vocab_size() const override { return cfg.vocab_size; }
  std::vector<int64_t>& eos_token_ids() override { return cfg.eos_token_ids; }
  int64_t block_tokens() const override { return dgpp::QwenModel::kv_block_tokens_static(); }
  int prefill_chunk_tokens() const override { return dgpp::QwenModel::prefill_chunk_tokens(); }
  std::string pool_check(int64_t pool_tokens) const override {
    if (pool_tokens / cfg.indexer_compress_ratio >= (int64_t(1) << 21))
      return "exceeds the QSA pool-id space (2^21 pools)";
    return "";
  }
  const char* kv_format_name() const override { return "bf16"; }
  // The decode kernels are gated at 8 rows (the 2026-09-09 rounds); a wider
  // batch waits for its own gates.
  int decode_rows_cap() const override { return dgpp::kDecodeRows; }
  // The PLE layer's key partial is hc x hidden wide (plan D4) — the widest
  // fold the decode graph records.
  size_t lat_slot_bytes(int decode_rows) const override {
    return static_cast<size_t>(decode_rows) * static_cast<size_t>(cfg.hyper_width()) * 2;
  }
  dgpp::MemoryPlan plan(int forward_rows, int64_t context, int rank, int world_, bool fabric, int slots,
                        bool mtp, int decode_rows) const override {
    return dgpp::QwenModel::plan_memory(cfg, forward_rows, context, fabric ? rank : 0, fabric ? world_ : 1,
                                        fabric ? dgpp::QwenResidency::Resident : dgpp::QwenResidency::Streaming,
                                        slots, fabric && mtp, decode_rows);
  }
  size_t snapshot_bytes(int world_, bool mtp) const override {
    return dgpp::QwenModel::session_snapshot_bytes(cfg, world_, mtp);
  }
  void build_model(dgpp::BoundaryReducer* reducer, int rank, int world_, bool fabric, int forward_rows,
                   int64_t pool_tokens, int slots, bool mtp, int decode_rows) override {
    // The resident image cache: the same directory the process configured
    // for the GLM loader (prepare_serving_process).
    dgpp::QwenLayerStream::set_resident_image_dir(dgpp::GlmLayerStream::resident_image_dir());
    model = std::make_unique<dgpp::QwenModel>(
        cfg, ckpt, forward_rows, pool_tokens,
        fabric ? dgpp::QwenResidency::Resident : dgpp::QwenResidency::Streaming, reducer, fabric ? rank : 0,
        fabric ? world_ : 1, slots, fabric && mtp, decode_rows);
  }
  void destroy_model() override { model.reset(); }
  size_t model_snapshot_bytes() const override { return model ? model->session_snapshot_bytes() : 0; }
  std::unique_ptr<ServeGraphEngine> make_graph_engine(
      dgpp::net::CollectiveBus* bus, int rank, int world_, uint16_t* pick_scratch, int batch_min_live,
      uint16_t* prefix_scratch, uint16_t* gather_scratch, int candidates,
      const dgpp::text::GrammarVocab* grammar, int prefix_slots, int mtp_depth) override {
    return std::make_unique<ServeGraphEngineOf<dgpp::QwenModel>>(
        model.get(), bus, rank, world_, pick_scratch, cfg.vocab_size, /*pick_timeout_ms=*/60000,
        batch_min_live, prefix_scratch, gather_scratch, candidates, grammar, prefix_slots, mtp_depth);
  }
  std::unique_ptr<dgpp::sched::SchedulerEngine> make_eager_engine(
      int slots, dgpp::DecodePick pick, dgpp::DecodeSample sample, const dgpp::text::GrammarVocab* grammar,
      int prefix_slots) override {
    return std::make_unique<dgpp::EagerEngineAdapter<dgpp::QwenModel>>(
        model.get(), slots, std::move(pick), std::move(sample), grammar, prefix_slots);
  }
};

// GLM-4.7 (Glm4MoeForCausalLM, NVFP4): the paged K/V pool (64-token
// blocks, bf16), no recurrent state (snapshots at any position), the
// resident fabric model / the streaming world-1 one.
struct Glm4Family final : ServeFamily {
  dgpp::Glm4TextConfig cfg;
  std::string ckpt;
  std::unique_ptr<dgpp::Glm4Model> model;
  explicit Glm4Family(const std::string& checkpoint)
      : cfg(dgpp::Glm4TextConfig::from_json_file((fs::path(checkpoint) / "config.json").string())),
        ckpt(checkpoint) {}
  const char* name() const override { return "glm4_moe"; }
  int64_t vocab_size() const override { return cfg.vocab_size; }
  std::vector<int64_t>& eos_token_ids() override { return cfg.eos_token_ids; }
  int64_t block_tokens() const override { return dgpp::Glm4Model::kv_block_tokens_static(); }
  int prefill_chunk_tokens() const override { return dgpp::Glm4Model::prefill_chunk_tokens(); }
  std::string pool_check(int64_t) const override { return ""; }
  const char* kv_format_name() const override { return "bf16"; }
  // The row walk, the attention's split scratch, the MoE slot path and the
  // draft window all take the runtime ceiling (the batched depth >= 2
  // chain, 2026-09-10).
  int decode_rows_cap() const override { return dgpp::kDecodeRowsMax; }
  // The widest fold the decode graph records: a block output [rows, hidden].
  size_t lat_slot_bytes(int decode_rows) const override {
    return static_cast<size_t>(decode_rows) * static_cast<size_t>(cfg.hidden_size) * 2;
  }
  dgpp::MemoryPlan plan(int forward_rows, int64_t context, int rank, int world_, bool fabric, int slots,
                        bool mtp, int decode_rows) const override {
    return dgpp::Glm4Model::plan_memory(cfg, forward_rows, context, fabric ? rank : 0, fabric ? world_ : 1,
                                        fabric ? dgpp::Glm4Residency::Resident : dgpp::Glm4Residency::Streaming,
                                        slots, fabric && mtp, decode_rows);
  }
  size_t snapshot_bytes(int world_, bool mtp) const override {
    return dgpp::Glm4Model::session_snapshot_bytes(cfg, world_, mtp);
  }
  void build_model(dgpp::BoundaryReducer* reducer, int rank, int world_, bool fabric, int forward_rows,
                   int64_t pool_tokens, int slots, bool mtp, int decode_rows) override {
    dgpp::Glm4LayerStream::set_resident_image_dir(dgpp::GlmLayerStream::resident_image_dir());
    model = std::make_unique<dgpp::Glm4Model>(
        cfg, ckpt, forward_rows, pool_tokens,
        fabric ? dgpp::Glm4Residency::Resident : dgpp::Glm4Residency::Streaming, reducer, fabric ? rank : 0,
        fabric ? world_ : 1, slots, fabric && mtp, decode_rows);
  }
  void destroy_model() override { model.reset(); }
  size_t model_snapshot_bytes() const override { return model ? model->session_snapshot_bytes() : 0; }
  std::unique_ptr<ServeGraphEngine> make_graph_engine(
      dgpp::net::CollectiveBus* bus, int rank, int world_, uint16_t* pick_scratch, int batch_min_live,
      uint16_t* prefix_scratch, uint16_t* gather_scratch, int candidates,
      const dgpp::text::GrammarVocab* grammar, int prefix_slots, int mtp_depth) override {
    return std::make_unique<ServeGraphEngineOf<dgpp::Glm4Model>>(
        model.get(), bus, rank, world_, pick_scratch, cfg.vocab_size, /*pick_timeout_ms=*/60000,
        batch_min_live, prefix_scratch, gather_scratch, candidates, grammar, prefix_slots, mtp_depth);
  }
  std::unique_ptr<dgpp::sched::SchedulerEngine> make_eager_engine(
      int slots, dgpp::DecodePick pick, dgpp::DecodeSample sample, const dgpp::text::GrammarVocab* grammar,
      int prefix_slots) override {
    return std::make_unique<dgpp::EagerEngineAdapter<dgpp::Glm4Model>>(
        model.get(), slots, std::move(pick), std::move(sample), grammar, prefix_slots);
  }
};

std::unique_ptr<ServeFamily> make_family(const std::string& ckpt, int world, dgpp::LatentFormat kv_format) {
  const dgpp::ModelArchitecture arch =
      dgpp::detect_architecture_file((fs::path(ckpt) / "config.json").string());
  if (arch == dgpp::ModelArchitecture::Qwen4Exp) return std::make_unique<QwenFamily>(ckpt);
  if (arch == dgpp::ModelArchitecture::Glm4Moe) return std::make_unique<Glm4Family>(ckpt);
  return std::make_unique<GlmFamily>(ckpt, world, kv_format);
}

void check_memory_plan(
    int rank, const dgpp::MemoryPlan& plan,
    size_t prefix_arena_bytes, size_t engine_bytes, int64_t block_tokens,
    const std::function<dgpp::MemoryPlan(int64_t)>& plan_at) {
  size_t free_bytes = 0, total_bytes = 0;
  DGPP_CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t available = dgpp::host_memory_available_bytes();
  const size_t budget = std::max(free_bytes, available);
  const size_t need = plan.total_bytes() + prefix_arena_bytes + engine_bytes;
  std::string items;
  for (const auto& it : plan.items) {
    if (it.device + it.pinned < (size_t{1} << 20)) continue;
    items += it.name + " " + gib(static_cast<double>(it.device + it.pinned));
    if (it.pinned) items += " (" + gib(static_cast<double>(it.pinned)) + " pinned)";
    items += "; ";
  }
  DGPP_LOG_INFO("rank {}: memory plan for a {}-token context — {}prefix cache {}; engine {}",
                rank, plan.context_tokens, items, gib(static_cast<double>(prefix_arena_bytes)),
                gib(static_cast<double>(engine_bytes)));
  DGPP_LOG_INFO(
      "rank {}: memory plan total {} ({} device + {} pinned) + {} headroom against {} free "
      "(device free {} of {}, host available {})",
      rank, gib(static_cast<double>(need)), gib(static_cast<double>(plan.device_bytes())),
      gib(static_cast<double>(plan.pinned_bytes() + prefix_arena_bytes + engine_bytes)),
      gib(static_cast<double>(kMemoryHeadroomBytes)), gib(static_cast<double>(budget)),
      gib(static_cast<double>(free_bytes)), gib(static_cast<double>(total_bytes)),
      gib(static_cast<double>(available)));
  if (need + kMemoryHeadroomBytes <= budget) return;
  // The plan is affine in the context: its slope from two points names the
  // largest context this node could hold with everything else as configured.
  std::string hint;
  if (plan.context_tokens > block_tokens) {
    const dgpp::MemoryPlan below = plan_at(plan.context_tokens - block_tokens);
    const double per_token =
        static_cast<double>(plan.total_bytes()) - static_cast<double>(below.total_bytes());
    if (per_token > 0) {
      const double per = per_token / static_cast<double>(block_tokens);
      const double fixed = static_cast<double>(plan.total_bytes()) -
                           per * static_cast<double>(plan.context_tokens) +
                           static_cast<double>(prefix_arena_bytes + engine_bytes +
                                               kMemoryHeadroomBytes);
      const double room = static_cast<double>(budget) - fixed;
      const int64_t feasible =
          room > 0 ? static_cast<int64_t>(room / per) / block_tokens * block_tokens : 0;
      hint = std::format(
          " At {:.1f} KiB per context token, the largest kv_capacity this node holds as "
          "configured is about {} tokens.",
          per / 1024.0, feasible);
    }
  }
  std::vector<const dgpp::MemoryPlan::Item*> largest;
  for (const auto& it : plan.items) largest.push_back(&it);
  std::sort(largest.begin(), largest.end(), [](const auto* a, const auto* b) {
    return a->device + a->pinned > b->device + b->pinned;
  });
  std::string top;
  for (size_t i = 0; i < largest.size() && i < 3; ++i)
    top += (i ? ", " : "") + largest[i]->name + " " +
           gib(static_cast<double>(largest[i]->device + largest[i]->pinned));
  throw std::runtime_error(std::format(
      "rank {}: the configuration needs {} plus {} headroom but this node has {} free — "
      "refusing to allocate (the largest items: {}).{} Lower engine.kv_capacity, "
      "engine.max_concurrency or engine.prefix_cache_gib, or choose a smaller "
      "engine.kv_dtype.",
      rank, gib(static_cast<double>(need)), gib(static_cast<double>(kMemoryHeadroomBytes)),
      gib(static_cast<double>(budget)), top, hint));
}

// The rank-0 serving stack, shared by both worlds: everything above
// the engine interface (tokenizer/template, service, HTTP, the engine
// loop). The fabric passes the journal — its hook rides engine_pass,
// one record per pass between the drain and the tick — plus the oplog
// audit tap (the 4-way consistency evidence). w1 passes null for both and
// the loop is exactly Stage 4a's. The caller destroys the engine adapter
// and stops the bus after this loop has joined.
// The prefix cache's arena (M7): the snapshot slots a per-rank budget of
// `gib` GiB holds at this model's session state size; 0 = off.
int prefix_arena_slots(size_t bytes, double gib) {
  if (gib <= 0.0) return 0;
  if (bytes == 0) return 0;
  const double budget = gib * 1024.0 * 1024.0 * 1024.0;
  const double slots = std::floor(budget / static_cast<double>(bytes));
  return static_cast<int>(std::min(slots, 4096.0));
}

int serve_openai(dgpp::sched::SchedulerEngine* engine, int64_t vocab_size,
                 const std::vector<int64_t>& eos_ids, const std::string& ckpt,
                 const std::string& model_display, const ServeKnobs& k,
                 bool no_eos, double boot_s,
                 dgpp::serve::JournalWriter* journal,
                 dgpp::serve::OpStreamObserver* oplog) {
  const dgpp::text::Tokenizer tok =
      dgpp::text::Tokenizer::load((fs::path(ckpt) / "tokenizer.json").string());
  const dgpp::text::ChatTemplate tpl = dgpp::text::ChatTemplate::load(
      (fs::path(ckpt) / "chat_template.jinja").string());
  DGPP_LOG_INFO("serve: tokenizer {:#x}, template {:#x} loaded",
                tok.revision_hash(), tpl.source_hash());
  dgpp::serve::TextFrontend frontend(&tok, &tpl);

  dgpp::serve::ServiceConfig scfg;
  scfg.model_id = model_display;
  scfg.default_max_tokens = k.default_max_tokens;
  scfg.queue_limit = k.queue_limit;
  scfg.sampling_defaults = k.sampling_defaults;
  scfg.fixed_seed = k.fixed_seed;
  scfg.reasoning_in_content = k.reasoning_in_content;
  scfg.admission = k.admission;
  scfg.vocab_size = vocab_size;  // logit_bias's id bound
  {
    // The prefix cache's key (M7): what the entries are bound to.
    char key[96];
    std::snprintf(key, sizeof(key), "tok:%016llx tpl:%016llx",
                  static_cast<unsigned long long>(tok.revision_hash()),
                  static_cast<unsigned long long>(tpl.source_hash()));
    scfg.prefix_key = std::string(key) + " ckpt:" + fs::path(ckpt).filename().string();
  }
  std::vector<int64_t> eos =
      no_eos ? std::vector<int64_t>{} : eos_ids;

  dgpp::serve::GenerationService service(scfg, engine, &frontend,
                                           std::move(eos));
  if (oplog) service.set_audit_observer(oplog);
  dgpp::serve::HttpServer http(k.http_port, &service, k.max_connections, k.http_bind);

  std::atomic<bool> drained{false};  // the engine thread's drain is done
  // The v1 failure semantics (DESIGN §9 item 6, PLAN M9; built 2026-09-05):
  // ANY engine or fabric failure fails the service — an engine op that
  // throws (a bus watchdog, a scheduler contract violation, a journal
  // write to a dead peer) or a peer's death seen by the journal watch
  // while the engine thread is inside a collective that rank will never
  // complete. The service answers every live stream with the engine_failure
  // error after its committed tokens, one-shots and later requests get a
  // 503, and the process exits with status 2 once the answers are out
  // (serve_run.sh or the supervisor restarts the world; the resident image
  // makes that ~25 s). No drain pass, no stop record, no bus teardown: the
  // engine thread may be stuck in the bus, and the peers see the journal
  // close when this process exits.
  std::atomic<bool> engine_failed{false};
  const auto fail_service = [&](const std::string& what) {
    if (engine_failed.exchange(true)) return;
    const int n = service.fail_engine(what);
    DGPP_LOG_ERROR(
        "serve: ENGINE FAILURE — {}; {} in-flight request(s) answered with "
        "the engine_failure error after their committed tokens; exiting "
        "nonzero once the answers are out",
        what, n);
    g_stop_requested.store(true);
  };
  if (journal)
    journal->watch_peers([&](int peer, const std::string& why) {
      fail_service("rank " + std::to_string(peer) + " died (" + why + ")");
    });
  std::thread engine_loop([&] {
    const auto pass = [&] {
      return journal
                 ? service.engine_pass(
                       [&](const dgpp::serve::GenerationService::PassEvents&
                               events) {
                         journal->broadcast(
                             dgpp::serve::encode_journal_tick(events));
                       })
                 : service.engine_pass();
    };
    // The throughput line (serve_stats.hpp): fed after every pass, idle
    // passes included, so its intervals end on time.
    dgpp::serve::ThroughputLog stats(k.stats_interval_s, /*rank=*/0, k.mtp);
    const auto observe = [&] {
      const dgpp::serve::GenerationService::Stats st = service.stats();
      const dgpp::serve::ServiceCounts sc{st.requests_total, st.requests_shed,
                                           st.requests_cancelled};
      stats.observe(service.meters(), &sc);
    };
    while (!g_stop_requested.load()) {
      try {
        const bool worked = pass();
        observe();
        if (!worked) std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } catch (const std::exception& e) {
        fail_service(e.what());
        drained.store(true);
        return;
      }
    }
    if (engine_failed.load()) {  // the watch fired: no drain, no stop record
      drained.store(true);
      return;
    }
    // The peer watch ends before the drain: the stop record releases the
    // peers, whose exits close their journal connections — an orderly
    // departure, not a death.
    if (journal) journal->stop_watch();
    // Drain-on-stop (M6 6c). This is a pass boundary: no collective is in
    // flight on any rank (a stop that lands mid-prefill waited the pass
    // out above). Close the door, shed the queue, flag every live request,
    // then one more pass: the cancels ride the journal and every rank's
    // cancel sweep retires them at this same quantum with no engine op;
    // only then the stop record, which releases the peers' read loops.
    const auto t0 = std::chrono::steady_clock::now();
    const int interrupted = service.begin_shutdown();
    try {
      pass();
      if (journal) journal->broadcast(dgpp::serve::encode_journal_stop());
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("serve: the drain pass or the stop broadcast failed: {}",
                     e.what());
    }
    DGPP_LOG_INFO(
        "serve: stop — {} in-flight request(s) retired through the drain "
        "pass{}, the queue shed, in {:.0f} ms",
        interrupted, journal ? " on every rank" : "",
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0)
            .count());
    drained.store(true);
  });
  std::thread stop_watcher([&] {
    while (!g_stop_requested.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // The final pass may be a prefill: wait it out, then let the HTTP
    // thread's pump answer every interrupted stream (the error event and
    // [DONE]) and shed one-shot (503) before the server stops — bounded,
    // so a client that never reads cannot hold the process.
    while (!drained.load() && !engine_failed.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int i = 0; i < 150 && !service.drained(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!service.drained())
      DGPP_LOG_WARN("serve: stop — answers still owed after the 3 s grace; "
                    "closing the server");
    http.stop();
  });

  DGPP_LOG_INFO(
      "serve: listening on :{} — {} (boot {:.1f}s){}; endpoints: POST "
      "/v1/chat/completions, POST /v1/completions, GET /v1/models, GET "
      "/health, GET /v1/metrics",
      http.port(), scfg.model_id, boot_s, journal ? " [fabric rank 0]" : "");
  http.serve();

  stop_watcher.join();
  if (engine_failed.load()) {
    // The failure exit: the answers are out (or the grace expired). The
    // engine thread may be inside a collective a dead rank will never
    // complete, so nothing is joined or torn down — the op stream is
    // written (the evidence of what was committed here) and the process
    // ends with status 2; the peers see the journal close and exit.
    if (journal) journal->stop_watch();
    if (oplog) write_ops_file("serve_rank0.ops", oplog->text());
    DGPP_LOG_ERROR("serve: exiting with status 2 after the engine failure");
    std::fflush(nullptr);
    std::_Exit(2);
  }
  engine_loop.join();
  if (oplog)
    write_ops_file("serve_rank0.ops", oplog->text());  // the 4-way leg
  DGPP_LOG_INFO("serve: stopped cleanly");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  // An uncaught exception or a bare std::terminate still leaves a
  // timestamped line with the reason, not libstdc++'s bare "terminate
  // called" (2026-09-06: every line the server prints carries a stamp).
  std::set_terminate([] {
    const std::exception_ptr current = std::current_exception();
    if (current) {
      try {
        std::rethrow_exception(current);
      } catch (const std::exception& e) {
        DGPP_LOG_ERROR("serve: terminating on an uncaught exception: {}", e.what());
      } catch (...) {
        DGPP_LOG_ERROR("serve: terminating on an uncaught non-standard exception");
      }
    } else {
      DGPP_LOG_ERROR("serve: std::terminate called");
    }
    std::fflush(nullptr);
    std::abort();
  });
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--version") {
      std::printf("dgpp-serve %s (git %s, cuda %d.%d)\n", DGPP_VERSION, DGPP_GIT_SHA,
                  CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
      return 0;
    }
  }
  DGPP_LOG_INFO("dgpp-serve {} (git {})", DGPP_VERSION, DGPP_GIT_SHA);

  static constexpr const char* kUsage =
      "usage: dgpp-serve --config CLUSTER.json --rank R | --model ORG/NAME | --checkpoint-dir DIR\n"
      "  [--version]: print the version and exit\n"
      "  [--memory-plan]: log the memory plan for this rank's configured shape\n"
      "    (the same check every boot runs before allocating) and exit 0 when\n"
      "    it fits the node's free memory, 1 when it does not; no world forms\n"
      "  [--config PATH]: resolved cluster JSON containing the model,\n"
      "    the world (the node list), this rank's peer, the ports and every\n"
      "    engine knob below; flags given after it override\n"
      "  [--port N (default 18080; rank 0 only)]\n"
      "  [--bind-host IPV4 (default 127.0.0.1; rank 0 only)]\n"
      "  [--kv-capacity TOKENS (default 8192)]: the KV pool per rank; a prompt\n"
      "    plus its answer must fit. Before anything is allocated the memory\n"
      "    plan (the model, the pool, the activations, the prefix cache) is\n"
      "    checked against the node's free memory and refused with the plan\n"
      "    itemized when it does not fit\n"
      "  [--kv-dtype bf16|fp8|fp4 (default bf16)]: the latent cache's storage\n"
      "    format — fp8 halves its bytes, fp4 quarters them, each at a cost in\n"
      "    attention precision; bf16 is every parity gate's format\n"
      "  [--max-concurrency N (default 8, the decode-row bound)]\n"
      "  [--queue-limit N (default 64)] [--default-max-tokens N (256)]\n"
      "  [--max-connections N (default 64)] [--no-eos]\n"
      "  fabric (Stage 4b): --world N --rank R (--peer HOST when rank>0)\n"
      "    [--fabric-port N (29970)] [--journal-port N (29971)]\n"
      "    [--rendezvous-timeout-ms N (120000)]\n"
      "    [--decode-graph [--mtp]]\n"
      "    [--graph-batch-min-live N (default min(2, max-concurrency);\n"
      "      must be in [1, max-concurrency])]\n"
      "      (the row batch needs max-concurrency * (1 + mtp depth) <= 8)\n"
      "    [--mtp-depth N]  draft tokens per step (1..3; the verify runs 1+N rows)\n"
      "    [--sampling-candidates N (default 128, in [1, 256]): the sampled\n"
      "      pick's per-rank candidate width; narrower falls back more]\n"
      "  prefix cache (M7): [--prefix-cache-gib X (default 1.5)]: the\n"
      "    snapshot arena per rank (slots = X GiB / one session's state);\n"
      "    [--no-prefix-cache] turns it off; every rank takes rank 0's slot\n"
      "    count from the warm record\n"
      "  admission (M6 6d): [--admission full|grow (default full)]\n"
      "    [--admission-window N (default 256)]: grow reserves prompt + N\n"
      "    tokens, grows at tick top, and sheds the youngest request\n"
      "    (finish_reason length) when the pool runs out; every rank takes\n"
      "    rank 0's policy from the warm record\n"
      "  bus (the prefill's bulk all-reduce): [--bulk-pace-gbps X]: sender\n"
      "    pacing per (peer, lane) queue pair (default: derived from the\n"
      "    port rate, port / ((world-1) x lanes) x 0.85; 0 = unpaced)\n"
      "    [--bulk-inflight N (default 4)]: stripes in flight per lane\n"
      "  sampling (defaults from generation_config.json; temperature 0 =\n"
      "  greedy): [--temperature X] [--top-p X] [--top-k N] [--min-p X]\n"
      "    [--repetition-penalty X] [--seed N (for requests that omit one)]\n"
      "  reasoning (M6 6f): [--reasoning-in-content] folds the ids before\n"
      "    </think> into content instead of reasoning_content\n"
      "  logging: [--stats-interval-s X (default 10; 0 = off)]: one INFO line\n"
      "    per interval with the aggregate prefill and decode throughput,\n"
      "    the live and queued counts, the pool and the prefix cache; the\n"
      "    per-token, per-window and per-cache-decision lines sit at DEBUG\n"
      "    (DGPP_LOG_LEVEL=debug)\n";

  std::string ckpt, model_id, peer;
  uint16_t port = 8080, fabric_port = 29970, journal_port = 29971;
  int64_t kv_capacity = 8192;
  std::string kv_dtype = "bf16";  // the latent cache's format
  std::string ngram_table = "resident";  // the Qwen n-gram table: resident | mmap
  std::string dense_weights = "checkpoint";  // the Qwen dense stack: checkpoint | fp8
  int max_concurrency = 8, queue_limit = 64, default_max_tokens = 256;
  int graph_batch_min_live = 0;  // 0 = min(2, max_concurrency) (the batch family, 2026-09-07)
  // The sampled pick's candidate width per rank on the graph engines (the
  // planned 128; narrower forces the exact gather fallback more often —
  // the width sweep's knob, scripts/serve_width_sweep.sh).
  int sampling_candidates = dgpp::kSamplingCandidates;
  std::string admission_mode = "full";
  int admission_window = 256;
  // The bulk collective's sender pacing (prefill all-reduces): negative
  // derives the per-QP rate from the port at bus start.
  double bulk_pace_gbps = -1.0;
  int bulk_inflight = -1;
  int max_connections = 64;
  int world = 1, rank = 0, rendezvous_timeout_ms = 120000;
  bool no_eos = false, decode_graph = false, mtp = false;
  int mtp_depth = 1;  // draft tokens per step (needs --mtp; 1..3)
  double prefix_cache_gib = 1.5;  // M7: the snapshot arena; 0 = off
  std::optional<float> temperature, top_p, min_p, repetition_penalty;
  std::optional<int> top_k;
  std::optional<uint64_t> fixed_seed;
  bool reasoning_in_content = false;
  double stats_interval_s = 10.0;  // the throughput line's period
  // The cluster config: found first, whatever its position,
  // because the flags after it override what it sets.
  std::string config_path;
  std::string http_bind = "127.0.0.1";
  int config_rank = 0;
  bool memory_plan_only = false;  // --memory-plan: the check alone, then exit
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--config") config_path = argv[i + 1];
    else if (std::string(argv[i]) == "--rank") config_rank = std::atoi(argv[i + 1]);
  }
  if (!config_path.empty()) {
    dgpp::serve::ClusterConfig c;
    try {
      c = dgpp::serve::load_cluster_config(config_path);
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("{}", e.what());
      return 2;
    }
    model_id = c.model;
    world = c.world();
    rank = config_rank;
    if (rank > 0 && rank < world) peer = c.nodes[0];
    port = static_cast<uint16_t>(c.http_port);
    http_bind = c.http_bind;
    if (!c.node_env.empty()) {
      if (rank < 0 || rank >= world) {
        DGPP_LOG_ERROR("rank is outside configured nodes");
        return 2;
      }
      for (const auto& [key, value] : c.node_env[rank])
        setenv(key.c_str(), dgpp::serve::expand_home(value).c_str(), 1);
    }
    fabric_port = static_cast<uint16_t>(c.fabric_port);
    journal_port = static_cast<uint16_t>(c.journal_port);
    const dgpp::serve::ClusterConfig::Engine& e = c.engine;
    max_concurrency = e.max_concurrency;
    kv_capacity = e.kv_capacity;
    kv_dtype = e.kv_dtype;
    ngram_table = e.ngram_table;
    dense_weights = e.dense_weights;
    default_max_tokens = e.default_max_tokens;
    queue_limit = e.queue_limit;
    max_connections = e.max_connections;
    no_eos = e.no_eos;
    decode_graph = e.decode_graph;
    mtp = e.mtp;
    mtp_depth = e.mtp_depth;
    graph_batch_min_live = e.graph_batch_min_live;
    sampling_candidates = e.sampling_candidates;
    prefix_cache_gib = e.prefix_cache_gib;
    admission_mode = e.admission;
    admission_window = e.admission_window;
    bulk_pace_gbps = e.bulk_pace_gbps;
    bulk_inflight = e.bulk_inflight;
    rendezvous_timeout_ms = e.rendezvous_timeout_ms;
    stats_interval_s = e.stats_interval_s;
    reasoning_in_content = e.reasoning_in_content;
    // The resident image cache's directory, unless the environment says.
    if (!c.paths.resident_cache.empty())
      setenv("DGPP_RESIDENT_CACHE_DIR",
             dgpp::serve::expand_home(c.paths.resident_cache).c_str(), 0);
    DGPP_LOG_INFO("config {}: model {}, world {}, rank {}, http :{}, fabric :{}, journal :{}",
                  config_path, model_id, world, rank, port, fabric_port, journal_port);
  }
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--bind-host") http_bind = next();
    else if (a == "--kv-capacity") kv_capacity = std::stoll(next());
    else if (a == "--kv-dtype") kv_dtype = next();
    else if (a == "--ngram-table") ngram_table = next();
    else if (a == "--dense-weights") dense_weights = next();
    else if (a == "--memory-plan") memory_plan_only = true;
    else if (a == "--max-concurrency") max_concurrency = std::stoi(next());
    else if (a == "--queue-limit") queue_limit = std::stoi(next());
    else if (a == "--default-max-tokens") default_max_tokens = std::stoi(next());
    else if (a == "--max-connections") max_connections = std::stoi(next());
    else if (a == "--no-eos") no_eos = true;
    else if (a == "--decode-graph") decode_graph = true;
    else if (a == "--graph-batch-min-live")
      graph_batch_min_live = std::stoi(next());
    else if (a == "--mtp") mtp = true;
    else if (a == "--mtp-depth") mtp_depth = std::stoi(next());
    else if (a == "--sampling-candidates") sampling_candidates = std::stoi(next());
    else if (a == "--prefix-cache-gib") prefix_cache_gib = std::stod(next());
    else if (a == "--no-prefix-cache") prefix_cache_gib = 0.0;
    else if (a == "--admission") admission_mode = next();
    else if (a == "--admission-window") admission_window = std::stoi(next());
    else if (a == "--bulk-pace-gbps") bulk_pace_gbps = std::stod(next());
    else if (a == "--bulk-inflight") bulk_inflight = std::stoi(next());
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--fabric-port")
      fabric_port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--journal-port")
      journal_port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--rendezvous-timeout-ms")
      rendezvous_timeout_ms = std::stoi(next());
    else if (a == "--temperature") temperature = std::stof(next());
    else if (a == "--top-p") top_p = std::stof(next());
    else if (a == "--top-k") top_k = std::stoi(next());
    else if (a == "--min-p") min_p = std::stof(next());
    else if (a == "--repetition-penalty") repetition_penalty = std::stof(next());
    else if (a == "--seed") fixed_seed = std::stoull(next());
    else if (a == "--reasoning-in-content") reasoning_in_content = true;
    else if (a == "--stats-interval-s") stats_interval_s = std::stod(next());
    else if (a == "--config") next();  // applied above, before the flags
    else {
      std::fputs(kUsage, stderr);
      return a == "--help" ? 0 : 1;
    }
  }
  // ---- the fabric's first handshake: the head pushes the
  // world's shape. Rank 0 opens the journal and accepts the full world
  // before building anything; the peers connect (retrying within the
  // rendezvous window) and take the model, the world size, the fabric port
  // and every engine knob from rank 0's settings record — their own flags
  // or file supplied only the bootstrap (rank 0's address, the journal
  // port, this rank) and the local paths. The bus world forms later, after
  // the model build, exactly as before (its connect retries too).
  std::optional<dgpp::serve::JournalWriter> journal;
  std::optional<dgpp::serve::JournalReader> reader;
  const auto canonical = [&] {
    return std::format(
        "model={} world={} fabric={} journal={} conc={} kv={} kvdt={} ngt={} dw={} maxtok={} queue={} "
        "eos={} graph={} mtp={} mtpd={} batchmin={} cand={} pcgib={} adm={} win={} "
        "pace={} inflight={} reasoning_in_content={}",
        model_id.empty() ? ckpt : model_id, world, fabric_port, journal_port,
        max_concurrency, kv_capacity, kv_dtype, ngram_table, dense_weights, default_max_tokens, queue_limit,
        no_eos ? 0 : 1, decode_graph ? 1 : 0, mtp ? 1 : 0, mtp_depth, graph_batch_min_live,
        sampling_candidates, prefix_cache_gib, admission_mode, admission_window,
        bulk_pace_gbps, bulk_inflight, reasoning_in_content ? 1 : 0);
  };
  if ((world > 1 || rank > 0) && !memory_plan_only) {
    try {
      if (rank == 0) {
        require(world > 1, "--world must be > 1 for a fabric head");
        require(journal_port != fabric_port,
                "--journal-port must differ from --fabric-port");
        journal.emplace(journal_port);
        DGPP_LOG_INFO("journal: listening on :{} for {} peer(s)",
                      journal->port(), world - 1);
        journal->accept_peers(world, rendezvous_timeout_ms);
        // The row batch's threshold resolves before the push, so the
        // settings record and the effective config print the same value
        // (2026-09-06: `batchmin=0` on one line, `=4` on the next).
        if (graph_batch_min_live == 0)
          graph_batch_min_live = std::min(2, max_concurrency);
        dgpp::serve::WorldSettings ws;
        ws.version = DGPP_VERSION;
        ws.model = model_id;
        ws.checkpoint = ckpt;
        ws.world = world;
        ws.fabric_port = fabric_port;
        ws.max_concurrency = max_concurrency;
        ws.kv_capacity = kv_capacity;
        ws.kv_dtype = kv_dtype;
        ws.ngram_table = ngram_table;
        ws.dense_weights = dense_weights;
        ws.default_max_tokens = default_max_tokens;
        ws.queue_limit = queue_limit;
        ws.no_eos = no_eos;
        ws.decode_graph = decode_graph;
        ws.mtp = mtp;
        ws.mtp_depth = mtp_depth;
        ws.graph_batch_min_live = graph_batch_min_live;
        ws.sampling_candidates = sampling_candidates;
        ws.prefix_cache_gib = prefix_cache_gib;
        ws.admission = admission_mode;
        ws.admission_window = admission_window;
        ws.bulk_pace_gbps = bulk_pace_gbps;
        ws.bulk_inflight = bulk_inflight;
        ws.rendezvous_timeout_ms = rendezvous_timeout_ms;
        ws.stats_interval_s = stats_interval_s;
        ws.reasoning_in_content = reasoning_in_content;
        journal->broadcast(dgpp::serve::encode_journal_settings(ws));
        DGPP_LOG_INFO("journal: settings pushed to {} peer(s): {}", world - 1,
                      canonical());
      } else {
        require(!peer.empty(), "ranks > 0 need --peer (rank 0's address) or --config");
        reader.emplace(peer, journal_port, rendezvous_timeout_ms, rank);
        dgpp::serve::WorldSettings ws;
        if (!dgpp::serve::wait_journal_settings(
                &*reader, [rank] { return peer_should_stop(rank); }, &ws,
                DGPP_VERSION)) {
          DGPP_LOG_INFO("rank {}: no settings from rank 0 — exiting", rank);
          return 0;
        }
        const std::string own = canonical();
        model_id = ws.model;
        if (model_id.empty()) ckpt = ws.checkpoint;
        world = ws.world;
        fabric_port = static_cast<uint16_t>(ws.fabric_port);
        max_concurrency = ws.max_concurrency;
        kv_capacity = ws.kv_capacity;
        kv_dtype = ws.kv_dtype;
        ngram_table = ws.ngram_table;
        dense_weights = ws.dense_weights;
        default_max_tokens = ws.default_max_tokens;
        queue_limit = ws.queue_limit;
        no_eos = ws.no_eos;
        decode_graph = ws.decode_graph;
        mtp = ws.mtp;
        mtp_depth = ws.mtp_depth;
        graph_batch_min_live = ws.graph_batch_min_live;
        sampling_candidates = ws.sampling_candidates;
        prefix_cache_gib = ws.prefix_cache_gib;
        admission_mode = ws.admission;
        admission_window = ws.admission_window;
        bulk_pace_gbps = ws.bulk_pace_gbps;
        bulk_inflight = ws.bulk_inflight;
        rendezvous_timeout_ms = ws.rendezvous_timeout_ms;
        stats_interval_s = ws.stats_interval_s;
        reasoning_in_content = ws.reasoning_in_content;
        const std::string now = canonical();
        if (own != now)
          DGPP_LOG_WARN(
              "rank {}: rank 0's settings override this rank's own — ran: {} ; "
              "runs: {}",
              rank, own, now);
        else
          DGPP_LOG_INFO("rank {}: settings from rank 0: {}", rank, now);
      }
    } catch (const std::exception& e) {
      DGPP_LOG_ERROR("rank {}: the settings handshake failed: {}", rank, e.what());
      return 1;
    }
  }

  if (!model_id.empty()) {
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) {
      DGPP_LOG_ERROR("--model {}: {}", model_id, err);
      return 1;
    }
    ckpt = snapshot;
    DGPP_LOG_INFO("model {} -> {}", model_id, snapshot);
  }
  if (ckpt.empty()) {
    std::fputs(kUsage, stderr);
    return 1;
  }
  if (kv_capacity < 1 || max_concurrency < 1 || queue_limit < 1 ||
      default_max_tokens < 1 || max_connections < 1) {
    DGPP_LOG_ERROR("all capacity knobs must be >= 1");
    return 1;
  }
  const std::optional<dgpp::LatentFormat> kv_format_opt =
      dgpp::latent_format_from_string(kv_dtype);
  if (!kv_format_opt) {
    DGPP_LOG_ERROR("--kv-dtype must be bf16, fp8 or fp4, got '{}'", kv_dtype);
    return 2;
  }
  const dgpp::LatentFormat kv_format = *kv_format_opt;
  if (ngram_table != "resident" && ngram_table != "mmap") {
    DGPP_LOG_ERROR("--ngram-table must be resident or mmap, got '{}'", ngram_table);
    return 2;
  }
  // The Qwen n-gram table's residency: set before the plan and the load
  // (both read it; the table's bytes leave the plan under mmap).
  dgpp::QwenLayerStream::set_ngram_table_mmap(ngram_table == "mmap");
  if (dense_weights != "checkpoint" && dense_weights != "fp8") {
    DGPP_LOG_ERROR("--dense-weights must be checkpoint or fp8, got '{}'", dense_weights);
    return 2;
  }
  dgpp::QwenLayerStream::set_dense_weights_fp8(dense_weights == "fp8");
  if (world > 1) {
    require(rank >= 0 && rank < world, "--rank outside --world");
    require(!peer.empty() || rank == 0,
            "ranks > 0 need --peer (rank 0's fabric IP)");
    require(journal_port != fabric_port,
            "--journal-port must differ from --fabric-port");
  }
  if (mtp && !decode_graph) {
    DGPP_LOG_ERROR("--mtp currently requires --decode-graph");
    return 1;
  }
  // The graph world: the fabric, or a world of one with the
  // decode graph — the resident model, the graph engine and MTP on a single
  // Spark over a bus whose collectives are the identity. Without the graph
  // a world of one streams the layers through the eager engine (below).
  const bool graph_world = world > 1 || decode_graph;
  if (max_concurrency > dgpp::kPickMaxRequests) {
    DGPP_LOG_ERROR("--max-concurrency must be at most {} request slots (got {})",
                   dgpp::kPickMaxRequests, max_concurrency);
    return 1;
  }
  // Validate sampling and admission options before constructing the model.
  // The graph batch threshold is resolved below from the configured slots.
  if (sampling_candidates < 1 || sampling_candidates > dgpp::kSampleMaxCandidates) {
    DGPP_LOG_ERROR("--sampling-candidates must be in [1, {}], got {}",
                   dgpp::kSampleMaxCandidates, sampling_candidates);
    return 2;
  }
  if (admission_mode != "full" && admission_mode != "grow") {
    DGPP_LOG_ERROR("--admission must be full or grow, got '{}'", admission_mode);
    return 2;
  }
  if (admission_window < 1) {
    DGPP_LOG_ERROR("--admission-window must be at least 1, got {}", admission_window);
    return 2;
  }
  if (prefix_cache_gib < 0.0) {
    DGPP_LOG_ERROR("--prefix-cache-gib must be >= 0, got {}", prefix_cache_gib);
    return 2;
  }
  if (!(stats_interval_s >= 0.0)) {
    DGPP_LOG_ERROR("--stats-interval-s must be >= 0, got {}", stats_interval_s);
    return 2;
  }
  if (mtp_depth < 1 || mtp_depth > 3) {
    DGPP_LOG_ERROR("--mtp-depth must be in [1, 3], got {}", mtp_depth);
    return 2;
  }
  if (!mtp && mtp_depth != 1) {
    DGPP_LOG_ERROR("--mtp-depth {} needs --mtp", mtp_depth);
    return 2;
  }
  if (graph_batch_min_live == 0) {
    // The batch family: the smallest batch that covers the
    // live slots replays, so two live requests pay four rows — the
    // crossover moves from four to two.
    graph_batch_min_live = std::min(2, max_concurrency);
  } else if (graph_batch_min_live < 1 ||
             graph_batch_min_live > max_concurrency) {
    DGPP_LOG_ERROR(
        "--graph-batch-min-live must be in [1, --max-concurrency] (got {} "
        "with {} slot(s))",
        graph_batch_min_live, max_concurrency);
    return 1;
  }

  // The effective configuration: what this rank runs after the
  // config, the flags and (on a peer) rank 0's settings — canonicalized and
  // digested; rank 0 puts the digest on the warm record and every peer
  // compares its own before it serves. With the settings pushed by the head
  // the digests agree by construction; the check stays as the assertion
  // that they did. Rank-0-only knobs (the HTTP port, the connection cap,
  // the sampling defaults) are left out.
  const std::string effective_config = canonical();
  const std::string config_digest = dgpp::serve::config_digest(effective_config);
  DGPP_LOG_INFO("config: {} (digest {})", effective_config, config_digest);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }

    const auto t_boot = std::chrono::steady_clock::now();
    // The family from config.json's architecture (loaders/architecture.hpp):
    // everything below the engine interface comes from it.
    std::unique_ptr<ServeFamily> family = make_family(ckpt, world, kv_format);
    DGPP_LOG_INFO("serve: model family {} ({})", family->name(), ckpt);
    // The decode rows (2026-09-10, engine/decode_outputs.hpp): the fixed
    // batch holds every slot's verify rows — max_concurrency x (1 + the
    // MTP depth) — floored at kDecodeRows so every existing recipe keeps
    // its exact shape (4 slots x 2 rows = 8). The family's cap bounds it:
    // GLM-4.7 supports the derived shape up to 32 rows. GLM-5.3 and
    // Qwen are capped at 8 and use scalar graphs beyond MTP depth 1.
    // Reject configurations whose depth-1 batch already exceeds the cap.
    const int graph_rows_per_request = mtp ? 1 + mtp_depth : 1;
    int decode_rows = std::max(dgpp::kDecodeRows, max_concurrency * graph_rows_per_request);
    if (decode_graph && decode_rows > family->decode_rows_cap()) {
      if (mtp_depth > 1 && max_concurrency * 2 <= family->decode_rows_cap()) {
        DGPP_LOG_INFO(
            "serve: {} slots x {} rows exceed the {} family's {}-row decode ceiling; "
            "the depth-{} steps replay scalar graphs (no batched chain on this family)",
            max_concurrency, graph_rows_per_request, family->name(), family->decode_rows_cap(),
            mtp_depth);
        decode_rows = family->decode_rows_cap();
      } else {
        DGPP_LOG_ERROR(
            "--decode-graph needs --max-concurrency * (1 + mtp depth) <= {} on the {} "
            "family (got {} * {})",
            family->decode_rows_cap(), family->name(), max_concurrency, graph_rows_per_request);
        return 1;
      }
    }
    DGPP_LOG_INFO("serve: decode rows {} ({} slot(s) x {} row(s) per request, floor {}, the {} family's cap {})",
                  decode_rows, max_concurrency, graph_rows_per_request, dgpp::kDecodeRows, family->name(),
                  family->decode_rows_cap());
    if (std::string(family->name()) != "glm5" && kv_dtype != "bf16")
      DGPP_LOG_WARN("serve: --kv-dtype {} applies to the GLM-5.3 latent cache only; the {} caches stay bf16",
                    kv_dtype, family->name());
    if (std::string(family->name()) != "qwen4_exp" && ngram_table != "resident")
      DGPP_LOG_WARN("serve: --ngram-table {} applies to the Qwen n-gram table only; the {} family has none",
                    ngram_table, family->name());
    if (std::string(family->name()) != "qwen4_exp" && dense_weights != "checkpoint")
      DGPP_LOG_WARN("serve: --dense-weights {} applies to the Qwen dense stack only; the {} family loads as shipped",
                    dense_weights, family->name());
    const dgpp::GlmGenerationDefaults generation_defaults =
        dgpp::GlmGenerationDefaults::from_checkpoint_dir(ckpt, family->vocab_size());
    // generation_config.json is the generation authority. Retain the
    // config.json value only for old checkpoints/fixtures that omit it.
    if (generation_defaults.eos_token_ids.has_value())
      family->eos_token_ids() = *generation_defaults.eos_token_ids;

    // The served sampling defaults: the file's values, then the process
    // overrides (DESIGN §10 — defaults from the model, overrides from the
    // command line). Validated here so an operator typo dies at boot.
    dgpp::sample::Params sampling_defaults;
    sampling_defaults.temperature = generation_defaults.effective_temperature();
    sampling_defaults.top_p = generation_defaults.effective_top_p();
    sampling_defaults.top_k = generation_defaults.effective_top_k();
    sampling_defaults.min_p = generation_defaults.effective_min_p();
    sampling_defaults.repetition_penalty =
        generation_defaults.effective_repetition_penalty();
    if (temperature) sampling_defaults.temperature = *temperature;
    if (top_p) sampling_defaults.top_p = *top_p;
    if (top_k) sampling_defaults.top_k = *top_k;
    if (min_p) sampling_defaults.min_p = *min_p;
    if (repetition_penalty)
      sampling_defaults.repetition_penalty = *repetition_penalty;
    try {
      dgpp::sample::validate_params(sampling_defaults);
    } catch (const std::invalid_argument& e) {
      DGPP_LOG_ERROR("serve: invalid sampling defaults: {}", e.what());
      return 1;
    }
    DGPP_LOG_INFO(
        "serve: sampling defaults temperature {} top_p {} top_k {} min_p {} "
        "repetition_penalty {} ({}; overrides: {}{}{}{}{}{})",
        sampling_defaults.temperature, sampling_defaults.top_p,
        sampling_defaults.top_k, sampling_defaults.min_p,
        sampling_defaults.repetition_penalty,
        generation_defaults.file_found ? "from generation_config.json"
                                       : "no generation_config.json: greedy",
        temperature ? "temperature " : "", top_p ? "top_p " : "",
        top_k ? "top_k " : "", min_p ? "min_p " : "",
        repetition_penalty ? "repetition_penalty " : "",
        (temperature || top_p || top_k || min_p || repetition_penalty)
            ? ""
            : "none");

    // Pool sizing: the shared DSA pool is the admission budget (the
    // same arithmetic as the scheduler receipt), sliced per rank at
    // world>1 — every rank computes the same numbers from the same
    // config. The pool IS the context bound (a prompt plus its answer must
    // fit it); the model's per-forward row bound is the prefill chunk
    // (2026-09-06: it used to be the whole context, and every activation
    // buffer grew with kv_capacity — 200 GB at 262k tokens).
    const int64_t block_tokens = family->block_tokens();
    const int64_t pool_tokens = ((kv_capacity + block_tokens - 1) /
                                 block_tokens) * block_tokens;
    if (const std::string why = family->pool_check(pool_tokens); !why.empty())
      throw std::runtime_error("--kv-capacity " + std::to_string(kv_capacity) + " " + why);
    const int forward_rows = static_cast<int>(std::min<int64_t>(
        pool_tokens, family->prefill_chunk_tokens()));
    // The pre-flight memory check's inputs (see check_memory_plan): the
    // prefix arena at this shape, and the engine's own buffers (the sampler
    // tables per slot, the prompt id buffers per context token, a margin).
    const size_t snapshot_bytes = family->snapshot_bytes(world, mtp && world > 1);
    const size_t prefix_arena_bytes =
        snapshot_bytes == 0 || prefix_cache_gib <= 0.0
            ? 0
            : static_cast<size_t>(std::min(
                  std::floor(prefix_cache_gib * 1024.0 * 1024.0 * 1024.0 /
                             static_cast<double>(snapshot_bytes)),
                  4096.0)) *
                  snapshot_bytes;
    const size_t engine_bytes =
        static_cast<size_t>(max_concurrency) * static_cast<size_t>(family->vocab_size()) * 8 +
        static_cast<size_t>(pool_tokens) * 16 + (size_t{64} << 20);
    if (memory_plan_only) {
      // The check alone, for the shape this rank would run (world > 1: the
      // resident, vocab-sharded fabric model; world 1: the streaming one).
      const bool fabric = graph_world;
      const auto plan_at = [&](int64_t context) {
        return family->plan(static_cast<int>(std::min<int64_t>(context, forward_rows)), context, rank,
                            world, fabric, max_concurrency, mtp, decode_rows);
      };
      try {
        check_memory_plan(rank, plan_at(pool_tokens), prefix_arena_bytes, engine_bytes,
                          block_tokens, plan_at);
      } catch (const std::runtime_error& e) {
        DGPP_LOG_ERROR("{}", e.what());
        return 1;
      }
      DGPP_LOG_INFO("rank {}: the memory plan fits", rank);
      return 0;
    }
    std::vector<int64_t> eos =
        no_eos ? std::vector<int64_t>{} : family->eos_token_ids();
    const std::string model_display = model_id.empty()
                                          ? fs::path(ckpt).filename().string()
                                          : model_id;
    // Constrained decoding (M6 6g): every rank builds the grammar's token
    // table from the same tokenizer.json, so the masks it derives from a
    // journal record are identical on every rank. Peers keep no tokenizer
    // otherwise (records carry ids); this table is the one thing of it
    // they need.
    const dgpp::text::GrammarVocab grammar_vocab = [&] {
      const dgpp::text::Tokenizer tok = dgpp::text::Tokenizer::load(
          (fs::path(ckpt) / "tokenizer.json").string());
      dgpp::text::GrammarVocab v = dgpp::text::GrammarVocab::from_tokenizer(
          tok, family->eos_token_ids(), static_cast<int>(family->vocab_size()));
      DGPP_LOG_INFO(
          "serve: grammar vocabulary built ({} ids, tool markers {}, "
          "call-turn EOS {})",
          family->vocab_size(),
          v.markers().tool_calls_available() ? "present" : "absent",
          v.call_turn_eos());
      // The JSON grammar's tables (M6 6h): built now, on every rank, so
      // the first response_format request pays nothing.
      const auto t0 = std::chrono::steady_clock::now();
      v.prepare_json();
      DGPP_LOG_INFO("serve: JSON grammar tables built in {:.2f} s",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      return v;
    }();
    const auto boot_s = [&] {
      return std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - t_boot)
          .count();
    };

    ServeKnobs knobs;
    knobs.http_port = port;
    knobs.http_bind = http_bind;
    knobs.max_connections = max_connections;
    knobs.queue_limit = queue_limit;
    knobs.admission.mode = admission_mode == "grow"
                               ? dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand
                               : dgpp::sched::AdmissionPolicy::Mode::kFullReserve;
    knobs.admission.window_tokens = admission_window;
    knobs.default_max_tokens = default_max_tokens;
    knobs.sampling_defaults = sampling_defaults;
    knobs.fixed_seed = fixed_seed;
    knobs.reasoning_in_content = reasoning_in_content;
    knobs.mtp = mtp;
    knobs.stats_interval_s = stats_interval_s;

    // ---- world > 1: the fabric (Stage 4b) ------------------------------
    if (graph_world) {
      // ALLOCATION DISCIPLINE (the burst-wedge lesson): the pick
      // scratch pins BEFORE the bus world forms; nothing allocates
      // between collectives.
      uint16_t* pick_scratch = nullptr;
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                                 sizeof(uint16_t) * dgpp::kPickScratchElems(world),
                                 cudaHostAllocDefault));
      // The sampler's two tables (the candidate/LSE fold and the fallback
      // gather), pinned before the world forms like the pick scratch.
      PinnedWords sample_prefix(dgpp::fabric_sampling_prefix_scratch_elems(world));
      PinnedWords sample_gather(dgpp::sampling_gather_scratch_elems(family->vocab_size()));
      std::unique_ptr<dgpp::net::CollectiveBus> bus;
      try {
        dgpp::net::BusOptions bus_options = dgpp::fabric_bus_options(
            rank, world, fabric_port, peer, rendezvous_timeout_ms, family->lat_slot_bytes(decode_rows));
        if (bulk_pace_gbps >= 0) bus_options.bulk_pace_gbps = bulk_pace_gbps;
        if (bulk_inflight >= 0) bus_options.bulk_inflight_per_lane = bulk_inflight;
        bus = std::make_unique<dgpp::net::CollectiveBus>(bus_options);
        std::string err;
        if (!bus->start(&err))
          throw std::runtime_error("rank " + std::to_string(rank) +
                                   " bus start: " + err);

        // The journal star formed first (above, before either side built a
        // model): rank 0 accepted the full world and pushed the settings
        // record, so every rank here runs the same shape. A short world was
        // refused there, with the reason in the log.

        dgpp::BusBoundaryReducer reducer(*bus);
        // A world of one folds nothing: the model runs without a boundary
        // reducer (the graph engine's recorder still binds for captures,
        // where its record hooks are the identity).
        dgpp::BoundaryReducer* const model_reducer = world > 1 ? &reducer : nullptr;
        dgpp::prepare_serving_process(rank);
        {
          const auto plan_at = [&](int64_t context) {
            return family->plan(static_cast<int>(std::min<int64_t>(context, forward_rows)), context,
                                rank, world, /*fabric=*/true, max_concurrency, mtp, decode_rows);
          };
          check_memory_plan(rank, plan_at(pool_tokens), prefix_arena_bytes, engine_bytes,
                            block_tokens, plan_at);
        }
        const auto t_model = std::chrono::steady_clock::now();
        family->build_model(model_reducer, rank, world, /*fabric=*/true, forward_rows, pool_tokens,
                            max_concurrency, mtp, decode_rows);
        DGPP_LOG_INFO(
            "rank {}: model constructed in {:.1f}s (resident, {} request "
            "slots, {}-token pool in {}, {}-row forwards)",
            rank,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t_model)
                .count(),
            max_concurrency, pool_tokens, family->kv_format_name(),
            forward_rows);

        // The prefix cache's arena (M7): as many snapshot slots as the
        // budget holds; every rank computes the same count from the same
        // geometry, and the warm record carries rank 0's for the peers to
        // check against.
        const int prefix_slots = prefix_arena_slots(family->model_snapshot_bytes(), prefix_cache_gib);
        DGPP_LOG_INFO(
            "rank {}: prefix cache {} — {} snapshot slot(s) of {:.1f} MiB "
            "({:.2f} GiB asked)",
            rank, prefix_slots > 0 ? "on" : "off", prefix_slots,
            static_cast<double>(family->model_snapshot_bytes()) / (1024.0 * 1024.0),
            prefix_cache_gib);
        // The admission policy every rank runs (M6 6d): rank 0's, carried
        // by the warm record; a peer's own flags yield to it.
        dgpp::sched::AdmissionPolicy peer_policy = knobs.admission;
        int peer_prefix_slots = prefix_slots;
        std::string rank0_config;  // the warm record's config digest
        // The engine behind the scheduler's interface: the graph engine's
        // holder (its warm-up needs the adapter) or the eager adapter. Both
        // die before the model (family->destroy_model at every exit).
        std::unique_ptr<ServeGraphEngine> graph_holder;
        std::unique_ptr<dgpp::sched::SchedulerEngine> engine;
        const auto engine_ptr = [&]() -> dgpp::sched::SchedulerEngine* {
          return graph_holder ? graph_holder->engine() : engine.get();
        };
        const auto engine_release = [&] {
          graph_holder.reset();
          engine.reset();
          family->destroy_model();
        };
        if (decode_graph) {
          std::unique_ptr<ServeGraphEngine> graph_engine = family->make_graph_engine(
              bus.get(), rank, world, pick_scratch, graph_batch_min_live, sample_prefix.data,
              sample_gather.data, sampling_candidates, &grammar_vocab, prefix_slots, mtp_depth);
          // Record every graph variant now, on every rank at this same
          // point, so no capture pauses a live stream later. The warm-up
          // is a run of collectives, so it starts on the journal's clock:
          // rank 0 announces it with the warm record once its (slower)
          // construction is done; a peer holds at that record rather than
          // spinning its first collective in stall diagnostics.
          if (rank == 0) {
            if (journal)
              journal->broadcast(dgpp::serve::encode_journal_warm(
                  knobs.admission, prefix_slots, config_digest));
          } else if (!dgpp::serve::wait_journal_warm(
                         &*reader, [rank] { return peer_should_stop(rank); },
                         &peer_policy, &peer_prefix_slots, &rank0_config)) {
            graph_engine.reset();
            family->destroy_model();
            cudaFreeHost(pick_scratch);
            bus->stop();
            DGPP_LOG_INFO("rank {}: exited cleanly", rank);
            return 0;
          }
          const auto t_warm = std::chrono::steady_clock::now();
          graph_engine->warm_captures(std::vector<int64_t>(4, 0));
          DGPP_LOG_INFO(
              "rank {}: graph variants warm-captured in {:.2f}s",
              rank,
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - t_warm)
                  .count());
          graph_holder = std::move(graph_engine);
        } else {
          engine = family->make_eager_engine(
              max_concurrency,
              dgpp::make_fabric_pick(bus.get(), rank, world, pick_scratch, family->vocab_size()),
              dgpp::make_fabric_sample(bus.get(), rank, world, sample_prefix.data, sample_gather.data,
                                       family->vocab_size()),
              &grammar_vocab, prefix_slots);
          // The eager fabric path exchanges the warm record too: it
          // carries the admission policy (no capture to start here).
          if (rank == 0) {
            if (journal)
              journal->broadcast(dgpp::serve::encode_journal_warm(
                  knobs.admission, prefix_slots, config_digest));
          } else if (!dgpp::serve::wait_journal_warm(
                         &*reader, [rank] { return peer_should_stop(rank); },
                         &peer_policy, &peer_prefix_slots, &rank0_config)) {
            engine_release();
            cudaFreeHost(pick_scratch);
            bus->stop();
            return 0;
          }
        }

        if (rank != 0 && !rank0_config.empty() && rank0_config != config_digest) {
          // The configuration check: this rank would run a
          // different world than rank 0 — a different model, world size,
          // fabric port or engine knob. The op streams could never agree;
          // refuse before the first tick, naming what this rank runs (rank
          // 0's log has its own line).
          DGPP_LOG_ERROR(
              "rank {}: configuration differs from rank 0's (digest {} vs {}); "
              "this rank runs: {} — refusing to serve",
              rank, config_digest, rank0_config, effective_config);
          return 1;
        }
        if (rank != 0) {
          // THE PEER: no HTTP, no tokenizer — journal records carry
          // prompt ids (rank 0 already tokenized). Apply, tick,
          // repeat: this loop is the whole peer (§11's mirror). The
          // scheduler is constructed with rank 0's queue_limit — the
          // streams are identical, so the bound must be too.
          if (peer_policy != knobs.admission)
            DGPP_LOG_WARN(
                "rank {}: admission policy from rank 0's warm record ({} / "
                "window {}) overrides this rank's flags ({} / {})",
                rank, dgpp::sched::AdmissionPolicy::name(peer_policy.mode),
                peer_policy.window_tokens,
                dgpp::sched::AdmissionPolicy::name(knobs.admission.mode),
                knobs.admission.window_tokens);
          if (peer_prefix_slots != prefix_slots)
            DGPP_LOG_WARN(
                "rank {}: prefix cache slots from rank 0's warm record ({}) "
                "override this rank's {} (the arena must hold them)",
                rank, peer_prefix_slots, prefix_slots);
          dgpp::sched::Scheduler sched(engine_ptr(), eos, queue_limit, peer_policy,
                                     peer_prefix_slots);
          dgpp::serve::OpStreamObserver oplog;
          sched.set_observer(&oplog);
          dgpp::serve::ThroughputLog stats(stats_interval_s, rank, mtp);
          DGPP_LOG_INFO("rank {}: following rank 0's journal (admission {}, window {})",
                        rank, dgpp::sched::AdmissionPolicy::name(peer_policy.mode),
                        peer_policy.window_tokens);
          dgpp::serve::run_journal_peer(
              &sched, &*reader, [rank] { return peer_should_stop(rank); },
              [rank, &oplog] {
                // Rank 0's journal closed while this rank is inside a tick
                // — a collective rank 0 will never complete. The v1 failure
                // semantics: write the op stream (what was committed here)
                // and exit nonzero at once, never wait on the bus watchdog.
                DGPP_LOG_ERROR(
                    "rank {}: rank 0's journal closed under a collective — "
                    "the world is over; exiting with status 3",
                    rank);
                write_ops_file("serve_rank" + std::to_string(rank) + ".ops",
                               oplog.text());
                std::fflush(nullptr);
                std::_Exit(3);
              },
              /*watch_poll_ms=*/100, &oplog, &stats);
          write_ops_file("serve_rank" + std::to_string(rank) + ".ops",
                         oplog.text());
          engine_release();
          cudaFreeHost(pick_scratch);
          bus->stop();
          DGPP_LOG_INFO("rank {}: exited cleanly", rank);
          return 0;
        }
        dgpp::serve::OpStreamObserver oplog;  // rank 0's audit leg
        const int rc = serve_openai(engine_ptr(), family->vocab_size(), family->eos_token_ids(), ckpt,
                                    model_display, knobs, no_eos, boot_s(), journal ? &*journal : nullptr, &oplog);
        engine_release();
        cudaFreeHost(pick_scratch);
        bus->stop();
        return rc;
      } catch (...) {
        if (pick_scratch) cudaFreeHost(pick_scratch);
        throw;
      }
    }

    // ---- world 1: the local reference (Stage 4a) -----------------------
    dgpp::prepare_serving_process(/*rank=*/0);
    {
      const auto plan_at = [&](int64_t context) {
        return family->plan(static_cast<int>(std::min<int64_t>(context, forward_rows)), context,
                            /*rank=*/0, /*world=*/1, /*fabric=*/false, max_concurrency, /*mtp=*/false,
                            decode_rows);
      };
      check_memory_plan(0, plan_at(pool_tokens), prefix_arena_bytes, engine_bytes,
                        block_tokens, plan_at);
    }
    const auto t_model = std::chrono::steady_clock::now();
    family->build_model(/*reducer=*/nullptr, /*rank=*/0, /*world=*/1, /*fabric=*/false, forward_rows,
                        pool_tokens, max_concurrency, /*mtp=*/false, decode_rows);
    DGPP_LOG_INFO(
        "serve: model constructed in {:.1f}s (streaming, {} request slots, "
        "{}-token pool in {}, {}-row forwards)",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_model)
            .count(),
        max_concurrency, pool_tokens, family->kv_format_name(),
        forward_rows);

    const int prefix_slots = prefix_arena_slots(family->model_snapshot_bytes(), prefix_cache_gib);
    DGPP_LOG_INFO("serve: prefix cache {} — {} snapshot slot(s) of {:.1f} MiB",
                  prefix_slots > 0 ? "on" : "off", prefix_slots,
                  static_cast<double>(family->model_snapshot_bytes()) / (1024.0 * 1024.0));
    std::unique_ptr<dgpp::sched::SchedulerEngine> engine = family->make_eager_engine(
        max_concurrency, dgpp::make_w1_pick(family->vocab_size()), dgpp::make_w1_sample(family->vocab_size()),
        &grammar_vocab, prefix_slots);
    const int rc = serve_openai(engine.get(), family->vocab_size(), family->eos_token_ids(), ckpt,
                                model_display, knobs, no_eos, boot_s(), /*journal=*/nullptr,
                                /*oplog=*/nullptr);
    engine.reset();
    family->destroy_model();
    return rc;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("serve: {}", e.what());
    return 1;
  }
}
