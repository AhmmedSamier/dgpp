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
#include <atomic>
#include <chrono>
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
void on_signal(int) { g_stop_requested.store(true); }

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
};

// The rank-0 serving stack, shared by both worlds: everything above
// the engine seam (tokenizer/template, service, HTTP, the engine
// loop). The fabric passes the journal — its hook rides engine_pass,
// one record per pass between the drain and the tick — plus the oplog
// audit tap (the 4-way consistency evidence) and the bus (stopped
// last on exit). w1 passes null for all three and the loop is exactly
// Stage 4a's.
int serve_openai(dgpp::GenEngineAdapter* engine, const dgpp::GlmTextConfig& cfg,
                 const std::string& ckpt,
                 const std::string& model_display, const ServeKnobs& k,
                 bool no_eos, double boot_s,
                 dgpp::service::JournalWriter* journal,
                 dgpp::net::CollectiveBus* bus,
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
  std::vector<int64_t> eos =
      no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;

  dgpp::service::GenerationService service(scfg, engine, &frontend,
                                           std::move(eos));
  if (oplog) service.set_audit_observer(oplog);
  dgpp::service::HttpServer http(k.http_port, &service, k.max_connections);

  std::thread engine_loop([&] {
    while (!g_stop_requested.load()) {
      const bool progressed =
          journal
              ? service.engine_pass(
                    [&](const dgpp::service::GenerationService::PassEvents&
                            events) {
                      journal->broadcast(
                          dgpp::service::encode_journal_tick(events));
                    })
              : service.engine_pass();
      if (!progressed)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // Answers pending requests (503), cancels in-flight ones; the
    // HTTP thread's idle() drains the answers before the server
    // stops (the stop watcher below gives it one extra grace pass).
    service.begin_shutdown();
    if (journal) {
      // In-flight requests die with the world (drain-on-stop is a
      // later refinement, noted in the record); the peers mirror
      // everything rank 0 already drained, and the stop record
      // releases their read loops.
      try {
        journal->broadcast(dgpp::service::encode_journal_stop());
      } catch (const std::exception& e) {
        DGPP_LOG_ERROR("serve: stop broadcast failed: {}", e.what());
      }
    }
  });
  std::thread stop_watcher([&] {
    while (!g_stop_requested.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Give the final pump a beat to flush shutdown answers, then
    // unblock http.serve().
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
  if (bus) bus->stop();
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
      "    [--rendezvous-timeout-ms N (120000)]\n";

  std::string ckpt, model_id, peer;
  uint16_t port = 8080, fabric_port = 29970, journal_port = 29971;
  int64_t kv_capacity = 8192;
  int max_concurrency = 8, queue_limit = 64, default_max_tokens = 256;
  int max_connections = 64;
  int world = 1, rank = 0, rendezvous_timeout_ms = 120000;
  bool no_eos = false;
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
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--fabric-port")
      fabric_port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--journal-port")
      journal_port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--rendezvous-timeout-ms")
      rendezvous_timeout_ms = std::stoi(next());
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

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }

    const auto t_boot = std::chrono::steady_clock::now();
    const GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());

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
    const auto boot_s = [&] {
      return std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - t_boot)
          .count();
    };

    ServeKnobs knobs;
    knobs.http_port = port;
    knobs.max_connections = max_connections;
    knobs.queue_limit = queue_limit;
    knobs.default_max_tokens = default_max_tokens;

    // ---- world > 1: the fabric (Stage 4b) ------------------------------
    if (world > 1) {
      // ALLOCATION DISCIPLINE (the burst-wedge lesson): the pick
      // scratch pins BEFORE the bus world forms; nothing allocates
      // between collectives.
      uint16_t* pick_scratch = nullptr;
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                                 sizeof(uint16_t) * dgpp::kPickSlotsPerRank * world,
                                 cudaHostAllocDefault));
      std::unique_ptr<dgpp::net::CollectiveBus> bus;
      try {
        bus = std::make_unique<dgpp::net::CollectiveBus>(dgpp::fabric_bus_options(
            rank, world, fabric_port, peer, rendezvous_timeout_ms));
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
                                 max_concurrency);
        DGPP_LOG_INFO(
            "rank {}: model constructed in {:.1f}s (resident, {} request "
            "slots, {}-token pool)",
            rank,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          t_model)
                .count(),
            max_concurrency, pool_tokens);

        dgpp::GenEngineAdapter engine(
            &model, max_concurrency,
            dgpp::make_fabric_pick(bus.get(), rank, world, pick_scratch,
                                   cfg.vocab_size));

        if (rank != 0) {
          // THE PEER: no HTTP, no tokenizer — journal records carry
          // prompt ids (rank 0 already tokenized). Apply, tick,
          // repeat: this loop is the whole peer (§11's mirror). The
          // scheduler is constructed with rank 0's queue_limit — the
          // streams are identical, so the bound must be too.
          dgpp::glm::Scheduler sched(&engine, eos, queue_limit);
          dgpp::service::OpStreamObserver oplog;
          sched.set_observer(&oplog);
          DGPP_LOG_INFO("rank {}: following rank 0's journal", rank);
          dgpp::service::run_journal_peer(
              &sched, &*reader, [] { return g_stop_requested.load(); });
          write_ops_file("serve_rank" + std::to_string(rank) + ".ops",
                         oplog.text());
          cudaFreeHost(pick_scratch);
          bus->stop();
          DGPP_LOG_INFO("rank {}: exited cleanly", rank);
          return 0;
        }
        dgpp::service::OpStreamObserver oplog;  // rank 0's audit leg
        const int rc = serve_openai(&engine, cfg, ckpt, model_display, knobs,
                                    no_eos, boot_s(), &*journal, bus.get(),
                                    &oplog);
        cudaFreeHost(pick_scratch);
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

    dgpp::GenEngineAdapter engine(&model, max_concurrency,
                                  dgpp::make_w1_pick(cfg.vocab_size));
    return serve_openai(&engine, cfg, ckpt, model_display, knobs, no_eos,
                         boot_s(), /*journal=*/nullptr, /*bus=*/nullptr,
                         /*oplog=*/nullptr);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("serve: {}", e.what());
    return 1;
  }
}
