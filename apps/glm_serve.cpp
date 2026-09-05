// M6 Stage 4 (PLAN M6 deliverables 4+5): glm_serve — the OpenAI-
// compatible chat/completions service. The exact tokenizer + chat
// template (Stage 3/3b), the deterministic scheduler (Stage 2b), the
// incremental session engine (Stage 2), and the transport race fix
// (f695f39) under one roof.
//
// STAGE 4a (world 1): the full HTTP/SSE contract over the local
// reference engine — streaming residency (the diagnostic loader;
// resident is the TP shape, DESIGN §3's memory arithmetic), full-vocab
// greedy pick.
//
// STAGE 4b (world > 1): the fabric — resident TP by default, the
// vocab-sharded head, the distributed greedy pick over the bus, and
// THE ADMISSION JOURNAL (service/fabric_serve.hpp): rank 0 is the
// sole HTTP ingress; every engine pass's scheduler-state changes ride
// one journal record broadcast to the peers, and every rank applies
// + ticks in lockstep (§11 — identical schedulers by construction,
// the smoke's md5 discipline made structural).
//
// THREADS (DESIGN §6.1's roles + the ingress):
//   * HTTP/ingress — HttpServer::serve(): accept, parse, route,
//     tokenize/render, SSE formatting (rank 0 only; never the fabric
//     critical path);
//   * engine — engine_pass() in a loop: drain admissions, one
//     scheduler quantum per iteration (the model's stream; at world>1
//     each pass's record goes out between drain and tick);
//   * peers — run_journal_peer(): apply the record, tick, repeat. No
//     HTTP, no tokenizer (records carry prompt ids).
//
// USAGE
//   glm_serve --model ORG/NAME | --checkpoint-dir DIR
//     [--port N (default 8080; rank 0 only)] [--kv-capacity TOKENS]
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm_chat_template.hpp"
#include "models/glm_fabric_engine.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_scheduler.hpp"
#include "models/glm_tokenizer.hpp"
#include "models/glm_tool_grammar.hpp"
#include "net/collective_bus.hpp"
#include "service/fabric_serve.hpp"
#include "service/generation_service.hpp"
#include "service/glm_frontend.hpp"
#include "service/http_server.hpp"

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
  int max_connections = 64;
  int queue_limit = 64;
  int default_max_tokens = 256;
  dgpp::glm_sample::Params sampling_defaults = dgpp::glm_sample::greedy_params();
  std::optional<uint64_t> fixed_seed;
  bool reasoning_in_content = false;
  dgpp::glm::AdmissionPolicy admission;  // M6 6d: full (default) or grow
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

// The rank-0 serving stack, shared by both worlds: everything above
// the engine seam (tokenizer/template, service, HTTP, the engine
// loop). The fabric passes the journal — its hook rides engine_pass,
// one record per pass between the drain and the tick — plus the oplog
// audit tap (the 4-way consistency evidence). w1 passes null for both and
// the loop is exactly Stage 4a's. The caller destroys the engine adapter
// and stops the bus after this loop has joined.
// The prefix cache's arena (M7): the snapshot slots a per-rank budget of
// `gib` GiB holds at this model's session state size; 0 = off.
int prefix_arena_slots(const dgpp::GlmDiagnosticModel& model, double gib) {
  if (gib <= 0.0) return 0;
  const size_t bytes = model.session_snapshot_bytes();
  if (bytes == 0) return 0;
  const double budget = gib * 1024.0 * 1024.0 * 1024.0;
  const double slots = std::floor(budget / static_cast<double>(bytes));
  return static_cast<int>(std::min(slots, 4096.0));
}

int serve_openai(dgpp::glm::SchedulerEngine* engine,
                 const dgpp::GlmTextConfig& cfg,
                 const std::string& ckpt,
                 const std::string& model_display, const ServeKnobs& k,
                 bool no_eos, double boot_s,
                 dgpp::service::JournalWriter* journal,
                 dgpp::service::OpStreamObserver* oplog) {
  const dgpp::GlmTokenizer tok =
      dgpp::GlmTokenizer::load((fs::path(ckpt) / "tokenizer.json").string());
  const dgpp::glm::ChatTemplate tpl = dgpp::glm::ChatTemplate::load(
      (fs::path(ckpt) / "chat_template.jinja").string());
  DGPP_LOG_INFO("serve: tokenizer {:#x}, template {:#x} loaded",
                tok.revision_hash(), tpl.source_hash());
  dgpp::service::GlmFrontend frontend(&tok, &tpl);

  dgpp::service::ServiceConfig scfg;
  scfg.model_id = model_display;
  scfg.default_max_tokens = k.default_max_tokens;
  scfg.queue_limit = k.queue_limit;
  scfg.sampling_defaults = k.sampling_defaults;
  scfg.fixed_seed = k.fixed_seed;
  scfg.reasoning_in_content = k.reasoning_in_content;
  scfg.admission = k.admission;
  {
    // The prefix cache's key (M7): what the entries are bound to.
    char key[96];
    std::snprintf(key, sizeof(key), "tok:%016llx tpl:%016llx",
                  static_cast<unsigned long long>(tok.revision_hash()),
                  static_cast<unsigned long long>(tpl.source_hash()));
    scfg.prefix_key = std::string(key) + " ckpt:" + fs::path(ckpt).filename().string();
  }
  std::vector<int64_t> eos =
      no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;

  dgpp::service::GenerationService service(scfg, engine, &frontend,
                                           std::move(eos));
  if (oplog) service.set_audit_observer(oplog);
  dgpp::service::HttpServer http(k.http_port, &service, k.max_connections);

  std::atomic<bool> drained{false};  // the engine thread's drain is done
  std::thread engine_loop([&] {
    const auto pass = [&] {
      return journal
                 ? service.engine_pass(
                       [&](const dgpp::service::GenerationService::PassEvents&
                               events) {
                         journal->broadcast(
                             dgpp::service::encode_journal_tick(events));
                       })
                 : service.engine_pass();
    };
    while (!g_stop_requested.load()) {
      if (!pass()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // Drain-on-stop (M6 6c). This is a pass boundary: no collective is in
    // flight on any rank (a stop that lands mid-prefill waited the pass
    // out above). Close the door, shed the queue, flag every live request,
    // then ONE more pass: the cancels ride the journal and every rank's
    // cancel sweep retires them at this same quantum with no engine op;
    // only then the stop record, which releases the peers' read loops.
    const auto t0 = std::chrono::steady_clock::now();
    const int interrupted = service.begin_shutdown();
    try {
      pass();
      if (journal) journal->broadcast(dgpp::service::encode_journal_stop());
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
    while (!drained.load())
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
  engine_loop.join();
  if (oplog)
    write_ops_file("serve_rank0.ops", oplog->text());  // the 4-way leg
  DGPP_LOG_INFO("serve: stopped cleanly");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  using dgpp::GlmDiagnosticModel;
  using dgpp::GlmTextConfig;

  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  static constexpr const char* kUsage =
      "usage: glm_serve --model ORG/NAME | --checkpoint-dir DIR\n"
      "  [--port N (default 8080; rank 0 only)]\n"
      "  [--kv-capacity TOKENS (default 8192)]\n"
      "  [--max-concurrency N (default 8, the decode-row bound)]\n"
      "  [--queue-limit N (default 64)] [--default-max-tokens N (256)]\n"
      "  [--max-connections N (default 64)] [--no-eos]\n"
      "  fabric (Stage 4b): --world N --rank R (--peer HOST when rank>0)\n"
      "    [--fabric-port N (29970)] [--journal-port N (29971)]\n"
      "    [--rendezvous-timeout-ms N (120000)]\n"
      "    [--decode-graph [--mtp]]\n"
      "    [--graph-batch-min-live N (default min(4, max-concurrency);\n"
      "      must be in [1, max-concurrency])]\n"
      "      (requires max-concurrency * (mtp?2:1) <= 8)\n"
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
      "    </think> into content instead of reasoning_content\n";

  std::string ckpt, model_id, peer;
  uint16_t port = 8080, fabric_port = 29970, journal_port = 29971;
  int64_t kv_capacity = 8192;
  int max_concurrency = 8, queue_limit = 64, default_max_tokens = 256;
  int graph_batch_min_live = 0;  // 0 = min(4, max_concurrency)
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
  double prefix_cache_gib = 1.5;  // M7: the snapshot arena; 0 = off
  std::optional<float> temperature, top_p, min_p, repetition_penalty;
  std::optional<int> top_k;
  std::optional<uint64_t> fixed_seed;
  bool reasoning_in_content = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--kv-capacity") kv_capacity = std::stoll(next());
    else if (a == "--max-concurrency") max_concurrency = std::stoi(next());
    else if (a == "--queue-limit") queue_limit = std::stoi(next());
    else if (a == "--default-max-tokens") default_max_tokens = std::stoi(next());
    else if (a == "--max-connections") max_connections = std::stoi(next());
    else if (a == "--no-eos") no_eos = true;
    else if (a == "--decode-graph") decode_graph = true;
    else if (a == "--graph-batch-min-live")
      graph_batch_min_live = std::stoi(next());
    else if (a == "--mtp") mtp = true;
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
    else {
      std::fputs(kUsage, stderr);
      return a == "--help" ? 0 : 1;
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
  if (decode_graph && world == 1) {
    DGPP_LOG_ERROR(
        "--decode-graph currently requires the fabric (--world > 1)");
    return 1;
  }
  const int graph_rows_per_request = mtp ? 2 : 1;
  if (decode_graph &&
      max_concurrency >
          GlmDiagnosticModel::kDecodeRows / graph_rows_per_request) {
    DGPP_LOG_ERROR(
        "--decode-graph needs --max-concurrency * speculative rows <= {} "
        "(got {} * {})",
        GlmDiagnosticModel::kDecodeRows, max_concurrency,
        graph_rows_per_request);
    return 1;
  }
  // The crossover is a fraction of the configured slots: the default is
  // four (the measured crossover of the eight-row graph) or full occupancy
  // when fewer slots exist; an explicit value outside [1, slots] is an
  // operator error, never silently clamped.
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
  if (graph_batch_min_live == 0) {
    graph_batch_min_live = std::min(4, max_concurrency);
  } else if (graph_batch_min_live < 1 ||
             graph_batch_min_live > max_concurrency) {
    DGPP_LOG_ERROR(
        "--graph-batch-min-live must be in [1, --max-concurrency] (got {} "
        "with {} slot(s))",
        graph_batch_min_live, max_concurrency);
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }

    const auto t_boot = std::chrono::steady_clock::now();
    GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
    const dgpp::GlmGenerationDefaults generation_defaults =
        dgpp::GlmGenerationDefaults::from_checkpoint_dir(ckpt,
                                                         cfg.vocab_size);
    // generation_config.json is the generation authority. Retain the
    // config.json value only for old checkpoints/fixtures that omit it.
    if (generation_defaults.eos_token_ids.has_value())
      cfg.eos_token_ids = *generation_defaults.eos_token_ids;

    // The served sampling defaults: the file's values, then the process
    // overrides (DESIGN §10 — defaults from the model, overrides from the
    // command line). Validated here so an operator typo dies at boot.
    dgpp::glm_sample::Params sampling_defaults;
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
      dgpp::glm_sample::validate_params(sampling_defaults);
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
    // config. The context bound rides a hair above the pool — an
    // admitted request always fits.
    const dgpp::DsaConfig dsa = [&] {
      dgpp::DsaConfig d = cfg.dsa_config();
      d.tp_size = world;
      return d;
    }();
    const int64_t block_tokens = dsa.block_tokens;
    const int64_t pool_tokens = ((kv_capacity + block_tokens - 1) /
                                 block_tokens) * block_tokens;
    const int64_t pools = (pool_tokens / block_tokens) *
                          dgpp::DsaGeometry::from_config(dsa).pools_per_block;
    if (pools >= (int64_t(1) << 21))
      throw std::runtime_error(
          "--kv-capacity " + std::to_string(kv_capacity) +
          " exceeds the DSA pool-id space (2^21 pools)");
    const int context_bound = static_cast<int>(pool_tokens) + 1;
    std::vector<int64_t> eos =
        no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;
    const std::string model_display = model_id.empty()
                                          ? fs::path(ckpt).filename().string()
                                          : model_id;
    // Constrained decoding (M6 6g): every rank builds the grammar's token
    // table from the same tokenizer.json, so the masks it derives from a
    // journal record are identical on every rank. Peers keep no tokenizer
    // otherwise (records carry ids); this table is the one thing of it
    // they need.
    const dgpp::glm::GrammarVocab grammar_vocab = [&] {
      const dgpp::GlmTokenizer tok = dgpp::GlmTokenizer::load(
          (fs::path(ckpt) / "tokenizer.json").string());
      dgpp::glm::GrammarVocab v = dgpp::glm::GrammarVocab::from_tokenizer(
          tok, cfg.eos_token_ids, static_cast<int>(cfg.vocab_size));
      DGPP_LOG_INFO(
          "serve: grammar vocabulary built ({} ids, tool markers {}, "
          "call-turn EOS {})",
          cfg.vocab_size,
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
    knobs.max_connections = max_connections;
    knobs.queue_limit = queue_limit;
    knobs.admission.mode = admission_mode == "grow"
                               ? dgpp::glm::AdmissionPolicy::Mode::kGrowOnDemand
                               : dgpp::glm::AdmissionPolicy::Mode::kFullReserve;
    knobs.admission.window_tokens = admission_window;
    knobs.default_max_tokens = default_max_tokens;
    knobs.sampling_defaults = sampling_defaults;
    knobs.fixed_seed = fixed_seed;
    knobs.reasoning_in_content = reasoning_in_content;

    // ---- world > 1: the fabric (Stage 4b) ------------------------------
    if (world > 1) {
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
      PinnedWords sample_gather(dgpp::sampling_gather_scratch_elems(cfg.vocab_size));
      std::unique_ptr<dgpp::net::CollectiveBus> bus;
      try {
        dgpp::net::BusOptions bus_options = dgpp::fabric_bus_options(
            rank, world, fabric_port, peer, rendezvous_timeout_ms);
        if (bulk_pace_gbps >= 0) bus_options.bulk_pace_gbps = bulk_pace_gbps;
        if (bulk_inflight >= 0) bus_options.bulk_inflight_per_lane = bulk_inflight;
        bus = std::make_unique<dgpp::net::CollectiveBus>(bus_options);
        std::string err;
        if (!bus->start(&err))
          throw std::runtime_error("rank " + std::to_string(rank) +
                                   " bus start: " + err);

        // The journal star forms AFTER the bus world: every peer's
        // rendezvous is complete, so the journal connects land at
        // once. Rank 0 binds a known port and accepts the full world;
        // a short world is not servable (its first collective would
        // wedge — better to refuse here, with the reason in the log).
        std::optional<dgpp::service::JournalWriter> journal;
        std::optional<dgpp::service::JournalReader> reader;
        if (rank == 0) {
          journal.emplace(journal_port);
          DGPP_LOG_INFO("journal: listening on :{} for {} peer(s)",
                        journal->port(), world - 1);
          journal->accept_peers(world, rendezvous_timeout_ms);
        } else {
          reader.emplace(peer, journal_port, rendezvous_timeout_ms);
        }

        dgpp::GlmBusBoundaryReducer reducer(*bus);
        dgpp::prepare_serving_process(rank);
        const auto t_model = std::chrono::steady_clock::now();
        GlmDiagnosticModel model(cfg, ckpt, context_bound, pool_tokens,
                                 &reducer, rank, world,
                                 dgpp::GlmResidency::Resident,
                                 dgpp::GlmHeadSharding::VocabSharded,
                                 max_concurrency, mtp);
        DGPP_LOG_INFO(
            "rank {}: model constructed in {:.1f}s (resident, {} request "
            "slots, {}-token pool)",
            rank,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t_model)
                .count(),
            max_concurrency, pool_tokens);

        // The prefix cache's arena (M7): as many snapshot slots as the
        // budget holds; every rank computes the same count from the same
        // geometry, and the warm record carries rank 0's for the peers to
        // check against.
        const int prefix_slots = prefix_arena_slots(model, prefix_cache_gib);
        DGPP_LOG_INFO(
            "rank {}: prefix cache {} — {} snapshot slot(s) of {:.1f} MiB "
            "({:.2f} GiB asked)",
            rank, prefix_slots > 0 ? "on" : "off", prefix_slots,
            static_cast<double>(model.session_snapshot_bytes()) / (1024.0 * 1024.0),
            prefix_cache_gib);
        // The admission policy every rank runs (M6 6d): rank 0's, carried
        // by the warm record; a peer's own flags yield to it.
        dgpp::glm::AdmissionPolicy peer_policy = knobs.admission;
        int peer_prefix_slots = prefix_slots;
        std::unique_ptr<dgpp::glm::SchedulerEngine> engine;
        if (decode_graph) {
          auto graph_engine = std::make_unique<dgpp::GlmGraphEngineAdapter>(
              &model, bus.get(), rank, world, pick_scratch, cfg.vocab_size,
              /*pick_timeout_ms=*/60000, graph_batch_min_live,
              sample_prefix.data, sample_gather.data,
              sampling_candidates, &grammar_vocab, prefix_slots);
          // Record every graph variant now, on every rank at this same
          // point, so no capture pauses a live stream later. The warm-up
          // is a run of collectives, so it starts on the journal's clock:
          // rank 0 announces it with the warm record once its (slower)
          // construction is done; a peer holds at that record rather than
          // spinning its first collective in stall diagnostics.
          if (rank == 0) {
            journal->broadcast(dgpp::service::encode_journal_warm(
                knobs.admission, prefix_slots));
          } else if (!dgpp::service::wait_journal_warm(
                         &*reader, [rank] { return peer_should_stop(rank); },
                         &peer_policy, &peer_prefix_slots)) {
            graph_engine.reset();
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
          engine = std::move(graph_engine);
        } else {
          engine = std::make_unique<dgpp::GenEngineAdapter>(
              &model, max_concurrency,
              dgpp::make_fabric_pick(bus.get(), rank, world, pick_scratch,
                                     cfg.vocab_size),
              dgpp::make_fabric_sample(bus.get(), rank, world,
                                       sample_prefix.data, sample_gather.data,
                                       cfg.vocab_size),
              &grammar_vocab, prefix_slots);
          // The eager fabric path exchanges the warm record too: it
          // carries the admission policy (no capture to start here).
          if (rank == 0) {
            journal->broadcast(dgpp::service::encode_journal_warm(
                knobs.admission, prefix_slots));
          } else if (!dgpp::service::wait_journal_warm(
                         &*reader, [rank] { return peer_should_stop(rank); },
                         &peer_policy, &peer_prefix_slots)) {
            engine.reset();
            cudaFreeHost(pick_scratch);
            bus->stop();
            return 0;
          }
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
                rank, dgpp::glm::AdmissionPolicy::name(peer_policy.mode),
                peer_policy.window_tokens,
                dgpp::glm::AdmissionPolicy::name(knobs.admission.mode),
                knobs.admission.window_tokens);
          if (peer_prefix_slots != prefix_slots)
            DGPP_LOG_WARN(
                "rank {}: prefix cache slots from rank 0's warm record ({}) "
                "override this rank's {} (the arena must hold them)",
                rank, peer_prefix_slots, prefix_slots);
          dgpp::glm::Scheduler sched(engine.get(), eos, queue_limit, peer_policy,
                                     peer_prefix_slots);
          dgpp::service::OpStreamObserver oplog;
          sched.set_observer(&oplog);
          DGPP_LOG_INFO("rank {}: following rank 0's journal (admission {}, window {})",
                        rank, dgpp::glm::AdmissionPolicy::name(peer_policy.mode),
                        peer_policy.window_tokens);
          dgpp::service::run_journal_peer(
              &sched, &*reader, [rank] { return peer_should_stop(rank); });
          write_ops_file("serve_rank" + std::to_string(rank) + ".ops",
                         oplog.text());
          engine.reset();
          cudaFreeHost(pick_scratch);
          bus->stop();
          DGPP_LOG_INFO("rank {}: exited cleanly", rank);
          return 0;
        }
        dgpp::service::OpStreamObserver oplog;  // rank 0's audit leg
        const int rc = serve_openai(engine.get(), cfg, ckpt, model_display,
                                    knobs, no_eos, boot_s(), &*journal,
                                    &oplog);
        engine.reset();
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
    const auto t_model = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, context_bound, pool_tokens,
                             /*boundary=*/nullptr, /*tp_rank=*/0,
                             /*tp_world=*/1, dgpp::GlmResidency::Streaming,
                             dgpp::GlmHeadSharding::Full, max_concurrency);
    DGPP_LOG_INFO(
        "serve: model constructed in {:.1f}s (streaming, {} request slots, "
        "{}-token pool)",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_model)
            .count(),
        max_concurrency, pool_tokens);

    const int prefix_slots = prefix_arena_slots(model, prefix_cache_gib);
    DGPP_LOG_INFO("serve: prefix cache {} — {} snapshot slot(s) of {:.1f} MiB",
                  prefix_slots > 0 ? "on" : "off", prefix_slots,
                  static_cast<double>(model.session_snapshot_bytes()) / (1024.0 * 1024.0));
    dgpp::GenEngineAdapter engine(&model, max_concurrency,
                                  dgpp::make_w1_pick(cfg.vocab_size),
                                  dgpp::make_w1_sample(cfg.vocab_size),
                                  &grammar_vocab, prefix_slots);
    return serve_openai(&engine, cfg, ckpt, model_display, knobs, no_eos,
                         boot_s(), /*journal=*/nullptr, /*oplog=*/nullptr);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("serve: {}", e.what());
    return 1;
  }
}
