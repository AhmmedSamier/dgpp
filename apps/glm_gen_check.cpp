// M6 Stage 1d: glm_gen_check — greedy generation end-to-end at real dims.
// The "it speaks" instrument: prompt token-ids in, generated token-ids out,
// through the exact serving path (resident TP forward + vocab-sharded head
// + the distributed greedy pick over the bus).
//
// MODES
//   world=1 (no --world): the local reference — full head, streaming
//     residency, plain argmax. No bus, no peers; the loop logic and the
//     w1 sequence land in the record before the fabric run.
//   world>1: the fabric TP shape — every rank a process, VocabSharded
//     head, Resident residency by default (the M6 serving contract;
//     --streaming keeps the M4 diagnostic loader), the per-step winner
//     through bus_greedy_pick (the same exact pick the CI gate pins:
//     glm_tp_greedy_gen_loopback). Every rank writes its tokens file;
//     the run record cross-checks them (md5) — the pick's rank
//     consistency is by construction (canonical order + broadcast),
//     the files are the audit trail.
//
// PROMPT: either raw text (--text "...", tokenized by the Stage 3 exact
// tokenizer, whose ids match HF tokenizers 0.23.1 byte-for-byte) or
// comma-separated token ids (--prompt "1,2,3"). Generated tokens decode
// through the same tokenizer; EOS stops the loop (config's eos_token_ids,
// --no-eos disables).
//
// SCHEDULER MODE (M6 Stage 2b): --requests FILE runs a manifest of
// CONCURRENT conversations through the deterministic scheduler (strict
// alternation, full-reserve admission, round-robin decode, scripted
// cancellation) — DESIGN §11's identical-rank-order contract, so all
// ranks issue the same ops in the same order and the fabric's
// collectives stay aligned. The manifest is JSONL, one object per line:
//   {"id": "a", "text": "...", "steps": 8}
//   {"id": "b", "chat": "...", "system": "...", "steps": 16,
//    "cancel_after": 8}
// Knobs (NInfer-shaped): --max-concurrency N (engine session slots;
// default min(8, requests) — 8 is the decode-row bound) and
// --kv-capacity TOKENS (the SHARED DSA pool; default: the sum of every
// request's reservation, so all fit immediately — shrink it to force
// budget deferral). --sched-plan prints the memory receipt and the
// admission forecast WITHOUT loading the model, then exits. The file
// must exist identically on every rank (scripts/fabric_run.sh
// --stage-file stages it); its FNV-1a-64 hash is logged per rank as the
// cross-rank identity check.
//
// T² IS DIAGNOSTIC HERE: every step re-forwards the whole sequence
// (fresh KDA/DSA state per call — GlmDiagnosticModel's contract). The
// incremental decode engine is Stage 2; do not read this app's per-step
// time as serving latency. EOS-stop is likewise Stage 2 policy (the
// sampler/service seam); this loop runs exactly --steps steps.
//
// ALLOCATION DISCIPLINE (the burst-wedge lesson, now a rule): every
// device allocation — the pick scratch included — happens BEFORE the
// world forms. Nothing allocates between collectives; the per-step
// host buffers are hoisted out of the loop.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_chat_template.hpp"
#include "models/glm_fabric_engine.hpp"
#include "models/glm_forward.hpp"
#include "models/glm_gen_engine.hpp"
#include "models/glm_sampler.hpp"
#include "models/glm_scheduler.hpp"
#include "models/glm_tokenizer.hpp"
#include "models/glm_tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::DsaConfig;
using dgpp::DsaGeometry;
using dgpp::DsaStatePool;
using dgpp::GenEngineAdapter;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmTextConfig;
using dgpp::KdaConfig;
using dgpp::KdaGeometry;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Real-mesh bus budgets + the fabric pick now live in the shared seam
// (models/glm_fabric_engine.hpp) — glm_serve and this app ride the
// same closure, so the pick path cannot drift between the smoke
// instrument and the serving deployment.

std::vector<int64_t> parse_prompt_ids(const std::string& text,
                                      int64_t vocab_size) {
  std::vector<int64_t> ids;
  std::string num;
  const auto flush = [&] {
    require(!num.empty(), "empty token id in --prompt");
    char* end = nullptr;
    const long long v = std::strtoll(num.c_str(), &end, 10);
    require(end && *end == '\0', "bad token id in --prompt: " + num);
    require(v >= 0 && v < vocab_size,
            "token id out of range [0, vocab): " + num);
    ids.push_back(static_cast<int64_t>(v));
    num.clear();
  };
  for (const char c : text) {
    if (c == ',' || c == ' ') flush();
    else num.push_back(c);
  }
  flush();
  require(!ids.empty(), "--prompt produced no token ids");
  return ids;
}

void write_tokens_file(const std::string& path, int rank, int world,
                       const std::vector<int64_t>& prompt,
                       const std::vector<int64_t>& generated) {
  std::string txt = "rank " + std::to_string(rank) + " world " +
                    std::to_string(world) + "\nprompt";
  for (int64_t t : prompt) txt += " " + std::to_string(t);
  txt += "\ngenerated";
  for (int64_t t : generated) txt += " " + std::to_string(t);
  txt += "\n";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  require(f != nullptr, "cannot write " + path);
  const size_t n = std::fwrite(txt.data(), 1, txt.size(), f);
  std::fclose(f);
  require(n == txt.size(), "short write to " + path);
}

std::string ids_line(const std::vector<int64_t>& ids) {
  std::string s;
  for (int64_t t : ids) {
    if (!s.empty()) s += ",";
    s += std::to_string(t);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Scheduler mode (M6 Stage 2b)
// ---------------------------------------------------------------------------

uint64_t fnv1a64(const std::string& bytes) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : bytes) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// Parses the manifest from its bytes (read once — the FNV hash logged
// across ranks must cover EXACTLY the bytes that were parsed). One JSON
// object per line ('#' comments and blanks skipped); throws with the
// line number on any malformed entry. The SAME bytes must reach every
// rank (fabric_run.sh --stage-file stages it).
std::vector<dgpp::glm::SchedulerRequest> parse_manifest(
    const std::string& manifest, const std::string& display_path,
    int default_steps, const dgpp::GlmTokenizer& tok, const fs::path& ckpt) {
  std::vector<dgpp::glm::SchedulerRequest> requests;
  std::unique_ptr<dgpp::glm::ChatTemplate> chat_tpl;
  std::istringstream in(manifest);
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto fail = [&](const std::string& what) {
      throw std::runtime_error("--requests " + display_path + " line " +
                               std::to_string(line_no) + ": " + what);
    };
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    dgpp::minijson::ParseResult parsed;
    try {
      parsed = dgpp::minijson::parse(line);
    } catch (const std::exception& e) {
      fail(std::string("JSON: ") + e.what());
    }
    const dgpp::minijson::Value& v = parsed.root;
    if (!v.is_object()) fail("expected a JSON object");
    const dgpp::minijson::Value& id = v.at("id");
    if (!id.is_string() || id.as_string().empty())
      fail("id must be a non-empty string");
    const std::string rid(id.as_string());

    const dgpp::minijson::Value* text = v.find("text");
    const dgpp::minijson::Value* chat = v.find("chat");
    const dgpp::minijson::Value* system = v.find("system");
    if ((text != nullptr) == (chat != nullptr))
      fail("exactly one of text|chat is required");
    if (system != nullptr && chat == nullptr)
      fail("system requires chat");

    int steps = default_steps;
    if (const dgpp::minijson::Value* s = v.find("steps")) {
      steps = static_cast<int>(s->as_int(0));
      if (steps < 1) fail("steps must be >= 1");
    }
    int cancel_after = 0;
    if (const dgpp::minijson::Value* c = v.find("cancel_after")) {
      cancel_after = static_cast<int>(c->as_int(0));
      if (cancel_after < 0) fail("cancel_after must be >= 0");
      if (cancel_after > steps)
        fail("cancel_after " + std::to_string(cancel_after) +
             " can never fire (steps " + std::to_string(steps) + ")");
    }

    // The same prompt builders as the single-request path.
    std::vector<int64_t> prompt;
    if (chat != nullptr) {
      if (!chat_tpl)
        chat_tpl = std::make_unique<dgpp::glm::ChatTemplate>(
            dgpp::glm::ChatTemplate::load(
                (ckpt / "chat_template.jinja").string()));
      std::vector<dgpp::glm::Value> messages;
      if (system != nullptr && !system->as_string().empty()) {
        dgpp::glm::Value::Members sys_msg;
        sys_msg.emplace_back("role",
                             dgpp::glm::Value::string_value("system"));
        sys_msg.emplace_back(
            "content", dgpp::glm::Value::string_value(
                           std::string(system->as_string())));
        messages.push_back(dgpp::glm::Value::map_value(std::move(sys_msg)));
      }
      dgpp::glm::Value::Members user_msg;
      user_msg.emplace_back("role", dgpp::glm::Value::string_value("user"));
      user_msg.emplace_back(
          "content",
          dgpp::glm::Value::string_value(std::string(chat->as_string())));
      messages.push_back(dgpp::glm::Value::map_value(std::move(user_msg)));
      dgpp::glm::Value::Members globals;
      globals.emplace_back(
          "messages", dgpp::glm::Value::list_value(std::move(messages)));
      globals.emplace_back("add_generation_prompt",
                           dgpp::glm::Value::boolean(true));
      const std::string rendered = chat_tpl->render(
          dgpp::glm::Value::map_value(std::move(globals)));
      prompt = tok.encode(rendered);
      DGPP_LOG_INFO("request '{}': chat template {} rendered to {} ids",
                    rid, chat_tpl->source_hash(), prompt.size());
    } else {
      prompt = tok.encode(std::string(text->as_string()));
      DGPP_LOG_INFO("request '{}': text encoded to {} ids by the exact "
                    "tokenizer",
                    rid, prompt.size());
    }
    if (prompt.empty()) fail("request '" + rid + "' produced no tokens");

    dgpp::glm::SchedulerRequest r;
    r.id = rid;
    r.prompt = std::move(prompt);
    r.max_steps = steps;
    r.cancel_after = cancel_after;
    requests.push_back(std::move(r));
  }
  require(!requests.empty(), "--requests produced no requests");
  return requests;
}

std::string read_whole_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  require(static_cast<bool>(f), "cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Blocks covering `tokens` tokens — the same arithmetic as the model's
// pool (DsaStatePool::block_count_for_tokens); standalone so the plan
// paths can run without a device.
int64_t model_blocks_for(int64_t tokens, int64_t block_tokens) {
  return (tokens + block_tokens - 1) / block_tokens;
}

// The shared-pool sizing every mode agrees on (mirrors GlmDiagnosticModel's
// own arithmetic: tp-sliced sub-configs, block-rounded capacity).
struct SchedSizing {
  int64_t pool_tokens = 0;  // block-rounded token capacity
  int64_t blocks_total = 0;
  int64_t block_tokens = 0;
};

SchedSizing sched_sizing(const GlmTextConfig& cfg, int world,
                         int64_t kv_capacity) {
  DsaConfig dsa = cfg.dsa_config();
  dsa.tp_size = world;
  const int64_t bt = dsa.block_tokens;
  SchedSizing z;
  z.block_tokens = bt;
  z.pool_tokens = ((kv_capacity + bt - 1) / bt) * bt;
  z.blocks_total = z.pool_tokens / bt;
  const int64_t pools =
      z.blocks_total * DsaGeometry::from_config(dsa).pools_per_block;
  // The selection kernels pack pool ids into 21 composite-key bits —
  // DsaStatePool::init refuses this shape; the plan refuses it EARLIER.
  if (pools >= (int64_t(1) << 21))
    throw std::runtime_error(
        "--kv-capacity " + std::to_string(kv_capacity) +
        " exceeds the DSA pool-id space (2^21 pools) — shrink it");
  return z;
}

// The memory receipt: EXACTLY what the model pre-allocates for this knob
// combination, by region, plus the per-request reserve math. Runs with or
// without a GPU (--sched-plan uses it before any device work).
void print_memory_receipt(const GlmTextConfig& cfg, int world,
                          int max_requests, const SchedSizing& z,
                          const std::vector<dgpp::glm::SchedulerRequest>& reqs,
                          bool with_forecast) {
  DsaConfig dsa = cfg.dsa_config();
  dsa.tp_size = world;
  KdaConfig kda = cfg.kda_config();
  kda.tp_size = world;
  const DsaGeometry geo = DsaGeometry::from_config(dsa);
  const KdaGeometry kgeo = KdaGeometry::from_config(kda);
  const int64_t pools = z.blocks_total * geo.pools_per_block;
  const int L = dsa.num_dsa_layers;
  const size_t latent =
      static_cast<size_t>(L) * static_cast<size_t>(z.pool_tokens) *
      geo.latent_bytes_per_token;
  const size_t index_k = static_cast<size_t>(L) *
                         static_cast<size_t>(pools) *
                         geo.index_k_bytes_per_pool;
  const size_t index_scale =
      static_cast<size_t>(L) * static_cast<size_t>(pools) * sizeof(float);
  const size_t tails = static_cast<size_t>(L) *
                       static_cast<size_t>(max_requests) *
                       geo.tail_bytes_per_request;
  const size_t tables = static_cast<size_t>(max_requests) *
                        static_cast<size_t>(z.blocks_total) * sizeof(int32_t);
  const size_t pool_total =
      DsaStatePool::cache_bytes(dsa, max_requests, z.pool_tokens);
  const size_t kda_state =
      static_cast<size_t>(kda.num_kda_layers) *
      static_cast<size_t>(max_requests) *
      (kgeo.recurrent_bytes + kgeo.conv_committed_bytes);

  DGPP_LOG_INFO(
      "sched receipt: pool {} tokens ({} blocks of {}), max_concurrency {}",
      z.pool_tokens, z.blocks_total, z.block_tokens, max_requests);
  DGPP_LOG_INFO(
      "sched receipt: DSA pool {:.1f} MiB = latent {:.1f} + index_k {:.1f} "
      "+ index_scale {:.1f} + tails {:.1f} + tables {:.1f}",
      pool_total / 1048576.0, latent / 1048576.0, index_k / 1048576.0,
      index_scale / 1048576.0, tails / 1048576.0, tables / 1048576.0);
  DGPP_LOG_INFO("sched receipt: KDA state {:.1f} MiB ({} slots x {} layers)",
                kda_state / 1048576.0, max_requests, kda.num_kda_layers);
  DGPP_LOG_INFO("sched receipt: session reserve = blocks_for(prompt + "
                "steps), block {} tokens — held for the request's lifetime",
                z.block_tokens);
  if (with_forecast) {
    int64_t held = 0;
    int used_slots = 0, admitted = 0, deferred = 0;
    for (const auto& r : reqs) {
      const int64_t reserve = model_blocks_for(
          static_cast<int64_t>(r.prompt.size()) + r.max_steps, z.block_tokens);
      if (reserve > z.blocks_total)
        throw std::runtime_error(
            "request '" + r.id + "' reserves " + std::to_string(reserve) +
            " blocks against a " + std::to_string(z.blocks_total) +
            "-block pool — it can NEVER fit (raise --kv-capacity or lower "
            "its steps)");
      if (used_slots < max_requests && held + reserve <= z.blocks_total) {
        held += reserve;
        ++used_slots;
        ++admitted;
      } else {
        ++deferred;
      }
    }
    DGPP_LOG_INFO(
        "sched plan: {} of {} requests admitted at start ({} blocks "
        "reserved of {}), {} deferred until peers retire",
        admitted, reqs.size(), held, z.blocks_total, deferred);
  }
}

// The engine binding lives in src/models/glm_gen_engine.hpp (shared
// with the Stage 4 serving app): GenEngineAdapter + the w1/fabric
// picks. Both worlds keep their local pick closures below.

const char* sched_status_name(const dgpp::glm::Scheduler::Result& r) {
  using S = dgpp::glm::Scheduler::Result::Status;
  switch (r.status) {
    case S::kDone: return "done";
    case S::kCancelled: return "cancelled";
    case S::kActive: return "active";
    default: return "queued";
  }
}

// Runs the whole manifest through the scheduler. Both worlds share the
// body; only the pick differs (full-head argmax vs the bus merge).
int run_scheduler(const GlmTextConfig& cfg, const std::string& ckpt,
                  int world, int rank, uint16_t port, const std::string& peer,
                  std::vector<dgpp::glm::SchedulerRequest> requests,
                  int max_requests, int64_t kv_capacity, bool resident,
                  bool no_eos, const dgpp::GlmTokenizer& tok,
                  const std::string& out_prefix, int rendezvous_timeout_ms,
                  int64_t vocab, uint64_t manifest_hash) {
  const SchedSizing z = sched_sizing(cfg, world, kv_capacity);
  for (const auto& r : requests) {
    const int64_t reserve =
        model_blocks_for(static_cast<int64_t>(r.prompt.size()) + r.max_steps,
                         z.block_tokens);
    if (reserve > z.blocks_total)
      throw std::runtime_error(
          "request '" + r.id + "' reserves " + std::to_string(reserve) +
          " blocks against a " + std::to_string(z.blocks_total) +
          "-block pool — raise --kv-capacity or lower its steps");
  }
  int max_tokens = 0;
  for (const auto& r : requests)
    max_tokens = std::max(max_tokens,
                          static_cast<int>(r.prompt.size()) + r.max_steps + 1);
  std::vector<int64_t> eos =
      no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;

  const auto log_results = [&](dgpp::glm::Scheduler& sched) {
    for (size_t i = 0; i < requests.size(); ++i) {
      const auto& spec = requests[i];
      const auto& res = sched.results()[i];
      write_tokens_file(out_prefix + "." + spec.id + ".tokens.txt", rank,
                        world, spec.prompt, res.generated);
      std::string text;
      for (int64_t t : res.generated)
        text += tok.decode(t, /*skip_special_tokens=*/false);
      DGPP_LOG_INFO("rank {} request {} generated ids: {}", rank, spec.id,
                    ids_line(res.generated));
      DGPP_LOG_INFO("rank {} request {} generated text: {}", rank, spec.id,
                    text);
      DGPP_LOG_INFO("rank {} request {} finished: {} ({} tokens)", rank,
                    spec.id, sched_status_name(res), res.steps_done);
    }
  };

  // ---- world 1: no bus, full head, plain argmax pick ------------------
  if (world == 1) {
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, max_tokens, z.pool_tokens,
                             /*boundary=*/nullptr, /*tp_rank=*/0,
                             /*tp_world=*/1, dgpp::GlmResidency::Streaming,
                             dgpp::GlmHeadSharding::Full, max_requests);
    DGPP_LOG_INFO(
        "w1 model constructed in {:.1f}s (streaming, {} request slots)",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_construct)
                .count(),
        max_requests);
    // The shared w1 pick (full-vocab argmax; the float row is hoisted
    // inside the closure — no per-token allocation).
    GenEngineAdapter engine(&model, max_requests, dgpp::make_w1_pick(vocab));
    dgpp::glm::Scheduler sched(&engine, eos);
    // Submit COPIES: the manifest entries stay intact for the audit
    // trail below (log_results reads spec.id/spec.prompt AFTER the run —
    // moving into the scheduler would leave husks there).
    for (const auto& r : requests) {
      dgpp::glm::SchedulerRequest copy = r;
      sched.submit(std::move(copy));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sched.run_to_completion();
    DGPP_LOG_INFO(
        "w1 scheduler: {} requests in {:.1f}ms", requests.size(),
                  std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                      .count());
    log_results(sched);
    return 0;
  }

  // ---- fabric TP: bus, sharded head, resident by default ---------------
  uint16_t* pick_scratch = nullptr;
  // Pinned (see the fabric-path note below): no UVM residency dependence
  // on the decode path, no migration ping-pong per pick.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                              sizeof(uint16_t) * 4 * world,
                              cudaHostAllocDefault));
  std::unique_ptr<CollectiveBus> bus;
  try {
    bus = std::make_unique<CollectiveBus>(dgpp::fabric_bus_options(
        rank, world, port, peer, rendezvous_timeout_ms));
    std::string err;
    if (!bus->start(&err))
      throw std::runtime_error("rank " + std::to_string(rank) +
                              " bus start: " + err);
    dgpp::GlmBusBoundaryReducer reducer(*bus);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(
        cfg, ckpt, max_tokens, z.pool_tokens, &reducer, rank, world,
        resident ? dgpp::GlmResidency::Resident : dgpp::GlmResidency::Streaming,
        dgpp::GlmHeadSharding::VocabSharded, max_requests);
    DGPP_LOG_INFO(
        "rank {} model constructed in {:.1f}s ({}, {} request slots)", rank,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       t_construct)
                .count(),
        resident ? "resident" : "streaming", max_requests);

    // The shared fabric pick (glm_fabric_engine.hpp) — the same
    // closure glm_serve runs in production; this app's runs are its
    // regression gate.
    GenEngineAdapter engine(&model, max_requests,
                            dgpp::make_fabric_pick(bus.get(), rank, world,
                                                   pick_scratch, vocab));
    dgpp::glm::Scheduler sched(&engine, eos);
    // Submit COPIES (the manifest entries stay intact for the audit
    // trail below — log_results reads them after the run).
    for (const auto& r : requests) {
      dgpp::glm::SchedulerRequest copy = r;
      sched.submit(std::move(copy));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sched.run_to_completion();
    DGPP_LOG_INFO("rank {} scheduler: {} requests (manifest {:016x}) in "
                  "{:.1f}ms",
                  rank, requests.size(), manifest_hash,
                  std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                      .count());
    log_results(sched);
    const auto stats = bus->stats();
    DGPP_LOG_INFO("rank {}: lat {} bulk {} collectives served",
                  rank, stats.latency.latency_us.size(),
                  stats.bulk.latency_us.size());
    bus->stop();
  } catch (...) {
    if (pick_scratch) cudaFreeHost(pick_scratch);
    throw;
  }
  cudaFreeHost(pick_scratch);
  return 0;
}

int run(const GlmTextConfig& cfg, const std::string& ckpt, int world,
        int rank, uint16_t port, const std::string& peer,
        const std::vector<int64_t>& prompt, int steps, bool resident,
        bool incremental, bool no_eos, const dgpp::GlmTokenizer& tok,
        const std::string& out_prefix, int rendezvous_timeout_ms,
        int64_t kv_capacity) {
  const int max_tokens = static_cast<int>(prompt.size()) + steps + 1;
  // --kv-capacity overrides the pool bound here too (0 = the historical
  // default); the model rounds it up to a block multiple.
  const int64_t cache =
      kv_capacity > 0 ? kv_capacity : std::max<int64_t>(128, max_tokens);

  // ---- world 1: no bus, full head, plain argmax ----------------------
  if (world == 1) {
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, max_tokens, cache);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("w1 model constructed in {:.1f}s (streaming)", construct_s);
    std::vector<int64_t> generated;
    std::string generated_text;  // decoded via the exact tokenizer
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                 cfg.eos_token_ids.end(), id) !=
                           cfg.eos_token_ids.end();
    };
    std::vector<float> frow(static_cast<size_t>(cfg.vocab_size));
    if (incremental) {
      // The serving path (Stage 2): prefill once, then one stateful
      // step per token — constant work per step, no T^2 re-forward.
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs out = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      for (int i = 0; i < out.lm_vocab_count; ++i)
        frow[static_cast<size_t>(i)] =
            dgpp::bf16_bits_to_float(out.logits_bits[static_cast<size_t>(i)]);
      dgpp::glm_sample::Candidate best =
          dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
      DGPP_LOG_INFO("[gen] prefill: {} tokens in {:.0f}ms", prompt.size(),
                    prefill_ms);
      for (int s = 0; s < steps; ++s) {
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        if (is_eos(best.id)) {
          DGPP_LOG_INFO("[gen] eos stop at step {} (token {})", s, best.id);
          break;
        }
        if (s + 1 == steps) break;
        const auto t0s = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs step =
            model.session_step(generated.back());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0s)
                              .count();
        for (int i = 0; i < step.lm_vocab_count; ++i)
          frow[static_cast<size_t>(i)] =
              dgpp::bf16_bits_to_float(step.logits_bits[static_cast<size_t>(i)]);
        best = dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s + 1, best.id, best.logit, ms);
      }
    } else {
      // The re-forward reference (the T^2 diagnostic loop).
      std::vector<int64_t> toks = prompt;
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int T = static_cast<int>(toks.size());
        require(out.lm_vocab_count == cfg.vocab_size,
                "w1 head must be full-vocab");
        const uint16_t* row = out.logits_bits.data() +
                              static_cast<size_t>(T - 1) * cfg.vocab_size;
        for (int i = 0; i < cfg.vocab_size; ++i)
          frow[static_cast<size_t>(i)] = dgpp::bf16_bits_to_float(row[i]);
        const dgpp::glm_sample::Candidate best =
            dgpp::glm_sample::local_max(frow.data(), cfg.vocab_size, 0);
        toks.push_back(best.id);
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s, best.id, best.logit, ms);
      }
    }
    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    DGPP_LOG_INFO("w1 generated ids: {}", ids_line(generated));
    DGPP_LOG_INFO("w1 generated text: {}", generated_text);
    return 0;
  }

  // ---- fabric TP: bus, sharded head, resident by default --------------
  // The pick scratch (bus_greedy_pick's caller-owned gather table) is
  // allocated BEFORE the world forms — the discipline at the top of this
  // file. Model construction happens after bus.start like glm_tp_check:
  // separate processes put a spinning peer kernel on the PEER's device;
  // it cannot block this rank's construction-phase device syncs.
  uint16_t* pick_scratch = nullptr;
  // Pinned, not managed: the pick scratch is host-written between
  // collectives and device-read by the kernel — pinning removes the UVM
  // migration ping-pong (two faults per pick) and the coherence
  // dependence entirely; the bus's own cells are pinned for the same
  // reason (the 2026-09-01 hunt's lesson: the decode path's shared
  // buffers do not ride managed memory).
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                             sizeof(uint16_t) * 4 * world,
                             cudaHostAllocDefault));
  std::unique_ptr<CollectiveBus> bus;
  try {
    bus = std::make_unique<CollectiveBus>(dgpp::fabric_bus_options(
        rank, world, port, peer, rendezvous_timeout_ms));
    std::string err;
    if (!bus->start(&err))
      throw std::runtime_error("rank " + std::to_string(rank) +
                               " bus start: " + err);

    dgpp::GlmBusBoundaryReducer reducer(*bus);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(
        cfg, ckpt, max_tokens, cache, &reducer, rank, world,
        resident ? dgpp::GlmResidency::Resident
                 : dgpp::GlmResidency::Streaming,
        dgpp::GlmHeadSharding::VocabSharded);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("rank {} model constructed in {:.1f}s ({})", rank,
                  construct_s,
                  resident ? "resident" : "streaming");

    std::vector<int64_t> toks = prompt;
    std::vector<int64_t> generated;
    std::vector<float> fslice;  // hoisted: no per-step device-adjacent work
    double forward_ms_total = 0.0;
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                  cfg.eos_token_ids.end(), id) !=
                            cfg.eos_token_ids.end();
    };
    // The step body: run one row (forward or stateful step), pick the
    // winner through the bus, record it. Returns the picked token.
    const auto run_step = [&](const GlmDiagnosticModel::Outputs& out,
                              const char* what, int s) -> int32_t {
      const uint16_t* row = out.logits_bits.data();
      fslice.resize(static_cast<size_t>(out.lm_vocab_count));
      for (int i = 0; i < out.lm_vocab_count; ++i)
        fslice[static_cast<size_t>(i)] =
            dgpp::bf16_bits_to_float(row[static_cast<size_t>(i)]);
      const dgpp::glm_sample::Candidate local = dgpp::glm_sample::local_max(
          fslice.data(), out.lm_vocab_count, out.lm_vocab_begin);
      const int32_t token =
          dgpp::bus_greedy_pick(*bus, rank, world, local, pick_scratch, 60000);
      require(token >= 0 && token < cfg.vocab_size,
              "generated id out of range: " + std::to_string(token));
      DGPP_LOG_INFO(
          "[gen] rank {} {} {}: token {} (local slice [{},{}) best "
          "{} logit {:.4f})",
          rank, what, s, token, out.lm_vocab_begin,
          out.lm_vocab_begin + out.lm_vocab_count, local.id, local.logit);
      return token;
    };
    std::string generated_text;
    if (incremental) {
      // The serving path (Stage 2): prefill once, one stateful step per
      // token, distributed pick over the sharded head's slices.
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs pre = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      DGPP_LOG_INFO("rank {} prefill: {} tokens in {:.0f}ms", rank,
                    prompt.size(), prefill_ms);
      const auto tp0 = std::chrono::steady_clock::now();
      int32_t token = run_step(pre, "prefill+pick", 0);
      forward_ms_total =
          prefill_ms + std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - tp0)
                           .count();
      for (int s = 0; s < steps; ++s) {
        generated.push_back(token);
        toks.push_back(token);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        if (is_eos(token)) {
          DGPP_LOG_INFO("rank {} eos stop at step {} (token {})", rank, s,
                        token);
          break;
        }
        if (s + 1 == steps) break;
        const auto t0s = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs step =
            model.session_step(generated.back());
        token = run_step(step, "step", s + 1);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0s)
                              .count();
        forward_ms_total += ms;
        DGPP_LOG_INFO("rank {} step {:.0f}ms (stateful step + pick)", rank,
                      ms);
      }
    } else {
      // The re-forward reference (the T^2 loop).
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int32_t token = run_step(out, "step", s);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        forward_ms_total += ms;
        toks.push_back(token);
        generated.push_back(token);
      }
    }

    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    DGPP_LOG_INFO("rank {} generated ids: {}", rank, ids_line(generated));
    DGPP_LOG_INFO("rank {} generated text: {}", rank, generated_text);

    const auto stats = bus->stats();
    const auto summarize = [](const std::vector<double>& us) {
      std::vector<double> sorted = us;
      std::sort(sorted.begin(), sorted.end());
      const auto pct = [&](double f) {
        return sorted.empty()
                   ? 0.0
                   : sorted[std::min(sorted.size() - 1,
                                     static_cast<size_t>(f * sorted.size()))];
      };
      return std::make_pair(sorted.size(),
                            std::make_pair(pct(0.5), pct(0.99)));
    };
    const auto [lat_n, lat_tails] = summarize(stats.latency.latency_us);
    const auto [bulk_n, bulk_tails] = summarize(stats.bulk.latency_us);
    DGPP_LOG_INFO(
        "rank {}: {} of {} step cap in {:.1f}ms total ({:.0f}ms/step avg "
        "incl. prefill; {}), lat {} "
        "(p50={:.1f}us p99={:.1f}us), bulk {} (p50={:.1f}us p99={:.1f}us)",
        rank, generated.size(), steps, forward_ms_total,
        forward_ms_total / generated.size(),
        incremental ? "incremental engine — constant per-step, the serving "
                       "path (steady-state from the per-step logs)"
                     : "re-forward T^2 diagnostic",
        lat_n, lat_tails.first, lat_tails.second, bulk_n, bulk_tails.first,
        bulk_tails.second);
    bus->stop();
  } catch (...) {
    if (pick_scratch) cudaFreeHost(pick_scratch);
    throw;
  }
  cudaFreeHost(pick_scratch);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");

  static constexpr const char* kUsage =
      "usage: glm_gen_check --model ORG/NAME | --checkpoint-dir DIR\n"
      "  (--prompt ID,ID,... | --text TEXT | --chat TEXT [--system TEXT]\n"
      "   | --requests FILE)\n"
      "  [--steps N] [--world N --rank R --peer HOST --port N]\n"
      "  [--streaming] [--engine incremental|reforward] [--no-eos]\n"
      "  [--rendezvous-timeout-ms N] [--out PREFIX]\n"
      "scheduler mode (--requests): [--max-concurrency N] [--kv-capacity N]\n"
      "  [--sched-plan]\n";

  std::string config_path, ckpt, peer, prompt_text, text_prompt, out_prefix =
                                                    "glm_gen",
              model_id, requests_path;
  int world = 1, rank = 0, steps = 8, rendezvous_timeout_ms = 120000;
  int max_concurrency = 0, kv_capacity = 0;
  bool sched_plan = false;
  uint16_t port = 29970;
  bool resident = true, incremental = true, no_eos = false;
  std::string system_prompt, chat_text;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--prompt") prompt_text = next();
    else if (a == "--text") text_prompt = next();
    else if (a == "--chat") chat_text = next();
    else if (a == "--system") system_prompt = next();
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--streaming") resident = false;
    else if (a == "--engine") {
      const std::string v = next();
      if (v == "incremental") incremental = true;
      else if (v == "reforward") incremental = false;
      else {
        DGPP_LOG_ERROR("--engine must be incremental|reforward");
        return 1;
      }
    }
    else if (a == "--no-eos") no_eos = true;
    else if (a == "--requests") requests_path = next();
    else if (a == "--max-concurrency") max_concurrency = std::stoi(next());
    else if (a == "--kv-capacity") kv_capacity = std::stoll(next());
    else if (a == "--sched-plan") sched_plan = true;
    else if (a == "--rendezvous-timeout-ms") rendezvous_timeout_ms = std::stoi(next());
    else if (a == "--out") out_prefix = next();
    else {
      std::fputs(kUsage, stderr);
      return a == "--help" ? 0 : 1;
    }
  }
  if (world > 1) {
    require(rank >= 0 && rank < world, "--rank outside --world");
    require(!peer.empty() || rank == 0,
            "ranks > 0 need --peer (rank 0's fabric IP)");
  }
  if (steps < 1) {
    DGPP_LOG_ERROR("--steps must be >= 1");
    return 1;
  }
  if (sched_plan && requests_path.empty()) {
    DGPP_LOG_ERROR("--sched-plan requires --requests");
    return 1;
  }
  if (!requests_path.empty() && !incremental) {
    DGPP_LOG_ERROR(
        "--requests drives the incremental session engine (leave --engine "
        "at its incremental default)");
    return 1;
  }
  if (max_concurrency < 0 || kv_capacity < 0) {
    DGPP_LOG_ERROR("--max-concurrency and --kv-capacity must be >= 1");
    return 1;
  }
  if (!model_id.empty()) {
    if (!ckpt.empty()) {
      DGPP_LOG_ERROR("--model and --checkpoint-dir are mutually exclusive");
      return 1;
    }
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

  // --sched-plan is a sizing calculator: the receipt prints from
  // config.json + the manifest alone — no device required.
  if (!sched_plan) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }
  }
  try {
    const GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
    if (cfg.vocab_size > (1 << 18)) {
      DGPP_LOG_ERROR(
          "vocab {} exceeds the 6-bit-triplet pick encoding (2^18) — this "
          "app needs a wider id path before it can serve this checkpoint",
          cfg.vocab_size);
      return 1;
    }
    // --text goes through the exact tokenizer (Stage 3); --prompt stays
    // for raw ids; --chat renders the chat template (Stage 3b) with the
    // value as the user message body. --requests (Stage 2b) is the
    // concurrent manifest. All four are mutually exclusive.
    const int prompt_kinds = static_cast<int>(!text_prompt.empty()) +
                             static_cast<int>(!prompt_text.empty()) +
                             static_cast<int>(!chat_text.empty()) +
                             static_cast<int>(!requests_path.empty());
    if (prompt_kinds > 1) {
      DGPP_LOG_ERROR(
          "--text, --prompt, --chat and --requests are mutually exclusive");
      return 1;
    }
    std::vector<int64_t> prompt;
    const dgpp::GlmTokenizer tok = dgpp::GlmTokenizer::load(
        (fs::path(ckpt) / "tokenizer.json").string());

    // ---- scheduler mode (M6 Stage 2b) --------------------------------
    if (!requests_path.empty()) {
      // Read ONCE: the hash logged across ranks must cover exactly the
      // bytes that were parsed.
      const std::string manifest = read_whole_file(requests_path);
      const uint64_t manifest_hash = fnv1a64(manifest);
      std::vector<dgpp::glm::SchedulerRequest> requests = parse_manifest(
          manifest, requests_path, steps, tok, fs::path(ckpt));
      const int n = static_cast<int>(requests.size());
      const int slots =
          max_concurrency > 0 ? max_concurrency : std::min(8, n);
      int64_t default_cap = 0;
      for (const auto& r : requests)
        default_cap +=
            static_cast<int64_t>(r.prompt.size()) + r.max_steps;
      const int64_t cap = kv_capacity > 0 ? kv_capacity : default_cap;
      const SchedSizing z = sched_sizing(cfg, world, cap);
      // The receipt + forecast validate everything the scheduler will
      // assume (pool-id space, per-request fit) BEFORE any allocation.
      print_memory_receipt(cfg, world, slots, z, requests,
                           /*with_forecast=*/true);
      DGPP_LOG_INFO(
          "manifest {} hash {:016x} — {} requests, {} slot(s), "
          "{}-token shared pool ({} tokens reserved by default)",
          requests_path, manifest_hash, n, slots, z.pool_tokens,
          default_cap);
      if (sched_plan) {
        DGPP_LOG_INFO(
            "--sched-plan: receipt printed; no model loaded, nothing run");
        return 0;
      }
      return run_scheduler(cfg, ckpt, world, rank, port, peer,
                           std::move(requests), slots, cap, resident, no_eos,
                           tok, out_prefix, rendezvous_timeout_ms,
                           cfg.vocab_size, manifest_hash);
    }

    if (!chat_text.empty()) {
      const dgpp::glm::ChatTemplate chat_tpl = dgpp::glm::ChatTemplate::load(
          (fs::path(ckpt) / "chat_template.jinja").string());
      std::vector<dgpp::glm::Value> messages;
      if (!system_prompt.empty()) {
        dgpp::glm::Value::Members sys_msg;
        sys_msg.emplace_back("role", dgpp::glm::Value::string_value("system"));
        sys_msg.emplace_back("content", dgpp::glm::Value::string_value(system_prompt));
        messages.push_back(dgpp::glm::Value::map_value(std::move(sys_msg)));
      }
      dgpp::glm::Value::Members user_msg;
      user_msg.emplace_back("role", dgpp::glm::Value::string_value("user"));
      user_msg.emplace_back("content", dgpp::glm::Value::string_value(chat_text));
      messages.push_back(dgpp::glm::Value::map_value(std::move(user_msg)));
      dgpp::glm::Value::Members globals;
      globals.emplace_back("messages",
                           dgpp::glm::Value::list_value(std::move(messages)));
      globals.emplace_back("add_generation_prompt", dgpp::glm::Value::boolean(true));
      const std::string rendered = chat_tpl.render(
          dgpp::glm::Value::map_value(std::move(globals)));
      prompt = tok.encode(rendered);
      DGPP_LOG_INFO("chat template {} rendered to {} bytes, {} ids",
                    chat_tpl.source_hash(), rendered.size(), prompt.size());
      require(!prompt.empty(), "--chat produced no tokens");
    } else if (!text_prompt.empty()) {
      prompt = tok.encode(text_prompt);
      DGPP_LOG_INFO("prompt encoded to {} ids by the exact tokenizer",
                    prompt.size());
      require(!prompt.empty(), "--text produced no tokens");
    } else {
      prompt = parse_prompt_ids(prompt_text, cfg.vocab_size);
    }
    return run(cfg, ckpt, world, rank, port, peer, prompt, steps, resident, incremental, no_eos, tok,
               out_prefix, rendezvous_timeout_ms, kv_capacity);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {}: {}", rank, e.what());
    return 1;
  }
}
