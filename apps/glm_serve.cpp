// M6 Stage 4 (PLAN M6 deliverables 4+5): glm_serve — the OpenAI-
// compatible chat/completions service. The exact tokenizer + chat
// template (Stage 3/3b), the deterministic scheduler (Stage 2b), the
// incremental session engine (Stage 2), and the transport race fix
// (f695f39) under one roof.
//
// STAGE 4a SCOPE (this app, world 1): the full HTTP/SSE contract over
// the local reference engine — streaming residency (the diagnostic
// loader; resident is the TP=4 production shape, DESIGN §3's memory
// arithmetic), full-vocab greedy pick. Stage 4b adds the fabric: the
// admission journal (rank 0 broadcasts validated submit/cancel events
// with tick stamps — every rank's scheduler stays identical, §11) and
// the resident TP=4 deployment.
//
// THREADS (DESIGN §6.1's three roles + the ingress):
//   * HTTP/ingress — HttpServer::serve(): accept, parse, route,
//     tokenize/render, SSE formatting (rank 0 only; never the fabric
//     critical path);
//   * engine — engine_pass() in a loop: drain admissions, one
//     scheduler quantum per iteration (the model's stream);
//   * (no bus at world 1 — Stage 4b adds the bus poller role.)
//
// USAGE
//   glm_serve --model ORG/NAME | --checkpoint-dir DIR
//     [--port N] [--kv-capacity TOKENS] [--max-concurrency N]
//     [--queue-limit N] [--default-max-tokens N] [--max-connections N]
//     [--no-eos]
#include <atomic>
#include <chrono>
#include <cstdint>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "models/glm_chat_template.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_scheduler.hpp"
#include "models/glm_tokenizer.hpp"
#include "service/generation_service.hpp"
#include "service/glm_frontend.hpp"
#include "service/http_server.hpp"

namespace fs = std::filesystem;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// SIGINT/SIGTERM → the orderly stop (flush, then exit).
std::atomic<bool> g_stop_requested{false};
void on_signal(int) { g_stop_requested.store(true); }

}  // namespace

int main(int argc, char** argv) {
  using dgpp::GlmDiagnosticModel;
  using dgpp::GlmTextConfig;
  using dgpp::GlmTokenizer;
  using dgpp::glm::ChatTemplate;
  using dgpp::service::GenerationService;
  using dgpp::service::GlmFrontend;
  using dgpp::service::HttpServer;
  using dgpp::service::ServiceConfig;

  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  static constexpr const char* kUsage =
      "usage: glm_serve --model ORG/NAME | --checkpoint-dir DIR\n"
      "  [--port N (default 8080)] [--kv-capacity TOKENS (default 8192)]\n"
      "  [--max-concurrency N (default 8, the decode-row bound)]\n"
      "  [--queue-limit N (default 64)] [--default-max-tokens N (256)]\n"
      "  [--max-connections N (default 64)] [--no-eos]\n";

  std::string ckpt, model_id;
  uint16_t port = 8080;
  int64_t kv_capacity = 8192;
  int max_concurrency = 8, queue_limit = 64, default_max_tokens = 256;
  int max_connections = 64;
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

  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }

    const auto t_boot = std::chrono::steady_clock::now();
    const GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
    const GlmTokenizer tok =
        GlmTokenizer::load((fs::path(ckpt) / "tokenizer.json").string());
    const ChatTemplate tpl =
        ChatTemplate::load((fs::path(ckpt) / "chat_template.jinja").string());
    DGPP_LOG_INFO("serve: tokenizer {:#x}, template {:#x} loaded",
                  tok.revision_hash(), tpl.source_hash());

    // Pool sizing: the shared DSA pool is the admission budget (the
    // same arithmetic as the scheduler receipt). The context bound
    // rides a hair above it — an admitted request always fits.
    const dgpp::DsaConfig dsa = [&] {
      dgpp::DsaConfig d = cfg.dsa_config();
      d.tp_size = 1;
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

    // ALLOCATION DISCIPLINE: the model (and every scratch buffer)
    // constructs BEFORE the HTTP server accepts its first request —
    // nothing allocates on the decode path.
    const int context_bound = static_cast<int>(pool_tokens) + 1;
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
    GlmFrontend frontend(&tok, &tpl);

    ServiceConfig scfg;
    scfg.model_id = model_id.empty()
                        ? fs::path(ckpt).filename().string()
                        : model_id;
    scfg.default_max_tokens = default_max_tokens;
    scfg.queue_limit = queue_limit;
    std::vector<int64_t> eos =
        no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;

    GenerationService service(scfg, &engine, &frontend, std::move(eos));
    HttpServer http(port, &service, max_connections);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::thread engine_loop([&] {
      while (!g_stop_requested.load()) {
        if (!service.engine_pass())
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      // Answers pending requests (503), cancels in-flight ones; the
      // HTTP thread's idle() drains the answers before the server
      // stops (the stop watcher below gives it one extra grace pass).
      service.begin_shutdown();
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
        "serve: listening on :{} — {} (boot {:.1f}s); endpoints: "
        "POST /v1/chat/completions, POST /v1/completions, GET /v1/models, "
        "GET /health, GET /v1/metrics",
        http.port(), scfg.model_id,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_boot)
            .count());

    http.serve();

    stop_watcher.join();
    engine_loop.join();
    DGPP_LOG_INFO("serve: stopped cleanly");
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("serve: {}", e.what());
    return 1;
  }
}
