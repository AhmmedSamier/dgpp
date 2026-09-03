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
#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <cmath>
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
#include "models/glm_step_timing.hpp"
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

// "Where did the calling thread go?" — a snapshot of the scheduler's and
// the VM's view of THIS thread: on-CPU time and runnable-but-waiting time
// (/proc/thread-self/schedstat), page faults and context switches
// (getrusage), and the core it runs on. Two snapshots bracket a code region
// that has no business taking milliseconds; their delta says whether the
// thread was preempted (wait grows), stuck in the kernel on its own behalf
// (run grows — direct reclaim, a slow syscall), paging (majflt), or moved.
// The 2026-09-02 hunt needed exactly this for the ~10 ms the peers lost
// between two log lines while every bus and GPU stamp read "fine".
class ThreadProbe {
 public:
  struct Sample {
    double run_ms = 0, wait_ms = 0;
    long majflt = 0, minflt = 0, nvcsw = 0, nivcsw = 0;
    int cpu = -1;
  };

  ThreadProbe() : fd_(::open("/proc/thread-self/schedstat", O_RDONLY)) {}
  ~ThreadProbe() {
    if (fd_ >= 0) ::close(fd_);
  }
  ThreadProbe(const ThreadProbe&) = delete;
  ThreadProbe& operator=(const ThreadProbe&) = delete;

  Sample sample() const {
    Sample s;
    // procfs regenerates the file on every read from offset 0: one pread on
    // the fd opened once, not open/read/close per step.
    char buf[96] = {};
    if (fd_ >= 0 && ::pread(fd_, buf, sizeof(buf) - 1, 0) > 0) {
      unsigned long long run_ns = 0, wait_ns = 0, slices = 0;
      if (std::sscanf(buf, "%llu %llu %llu", &run_ns, &wait_ns, &slices) >= 2) {
        s.run_ms = static_cast<double>(run_ns) / 1e6;
        s.wait_ms = static_cast<double>(wait_ns) / 1e6;
      }
    }
    rusage ru{};
    if (::getrusage(RUSAGE_THREAD, &ru) == 0) {
      s.majflt = ru.ru_majflt;
      s.minflt = ru.ru_minflt;
      s.nvcsw = ru.ru_nvcsw;
      s.nivcsw = ru.ru_nivcsw;
    }
    s.cpu = ::sched_getcpu();
    return s;
  }

  static std::string delta_text(const Sample& a, const Sample& b) {
    return std::format(
        "sched run {:.2f} wait {:.2f} majflt {} minflt {} nvcsw {} nivcsw {} "
        "cpu {}->{}",
        b.run_ms - a.run_ms, b.wait_ms - a.wait_ms, b.majflt - a.majflt,
        b.minflt - a.minflt, b.nvcsw - a.nvcsw, b.nivcsw - a.nivcsw, a.cpu,
        b.cpu);
  }

 private:
  int fd_;
};

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
    dgpp::prepare_serving_process(rank);
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
    dgpp::prepare_serving_process(rank);
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
    // trail below — log_results reads spec.id/spec.prompt after the run).
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
    dgpp::step_timing::report(("rank " + std::to_string(rank) +
                               " scheduler mode (prefill+decode mixed)")
                                  .c_str());
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

// Teacher forcing (--teacher-file): the step's logits are scored against
// the KNOWN next token instead of driving the pick. Each rank logs what it
// can compute from its slice — its max, its log-sum-exp (double, over the
// bf16 logits as the sampler would see them), and the target's logit when
// the target lives in the slice; scripts/fabric_logprob.py joins the ranks
// (logaddexp over the slices' lse) into log p(target) per position and a
// perplexity. That is the numerics gate for reassociated kernels: a
// transcript md5 flips on a 1-ulp tie and says nothing about magnitude;
// the mean NLL over a fixed text says exactly how far the distribution
// moved. World 1 logs the same line over the full head.
void log_teacher_stats(int rank, int step, const GlmDiagnosticModel::Outputs& out,
                       const std::vector<float>& slice, int64_t target,
                       int32_t argmax) {
  double lmax = -INFINITY;
  for (int i = 0; i < out.lm_vocab_count; ++i)
    lmax = std::max(lmax, static_cast<double>(slice[static_cast<size_t>(i)]));
  double sum = 0.0;
  for (int i = 0; i < out.lm_vocab_count; ++i)
    sum += std::exp(static_cast<double>(slice[static_cast<size_t>(i)]) - lmax);
  const double lse = lmax + std::log(sum);
  const int64_t local = target - out.lm_vocab_begin;
  const bool in_slice = local >= 0 && local < out.lm_vocab_count;
  DGPP_LOG_INFO("[tf] rank {} step {}: target {} argmax {} lmax {:.4f} lse "
                "{:.6f} target_logit {}",
                rank, step, target, argmax, lmax, lse,
                in_slice ? std::format("{:.4f}",
                                       slice[static_cast<size_t>(local)])
                         : std::string("nan"));
}

int run(const GlmTextConfig& cfg, const std::string& ckpt, int world,
        int rank, uint16_t port, const std::string& peer,
        const std::vector<int64_t>& prompt, int steps, bool resident,
        bool incremental, bool no_eos, bool decode_graph,
        const dgpp::GlmTokenizer& tok,
        const std::string& out_prefix, int rendezvous_timeout_ms,
        int64_t kv_capacity, const std::vector<int64_t>& teacher) {
  // Teacher forcing runs the text's length and never stops at EOS: the
  // scored positions are the text's, not the model's choices.
  const bool teaching = !teacher.empty();
  if (teaching) {
    steps = static_cast<int>(teacher.size());
    no_eos = true;
  }
  const int max_tokens = static_cast<int>(prompt.size()) + steps + 1;
  // --kv-capacity overrides the pool bound here too (0 = the historical
  // default); the model rounds it up to a block multiple.
  const int64_t cache =
      kv_capacity > 0 ? kv_capacity : std::max<int64_t>(128, max_tokens);

  // ---- world 1: no bus, full head, plain argmax ----------------------
  if (world == 1) {
    dgpp::prepare_serving_process(rank);
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
      // Under teacher forcing the fed token is the text's; the pick is
      // still computed (and logged) as the top-1 hit signal.
      const auto force = [&](int s, const GlmDiagnosticModel::Outputs& o) {
        if (!teaching) return;
        log_teacher_stats(0, s, o, frow, teacher[static_cast<size_t>(s)],
                          best.id);
        best.id = static_cast<int32_t>(teacher[static_cast<size_t>(s)]);
      };
      force(0, out);
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
        force(s + 1, step);
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
    dgpp::prepare_serving_process(rank);
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
      // The slice's runner-up: with the four ranks' lines side by side
      // (fabric_xrank), the global top-2 margin of every pick follows, and
      // a transcript that diverges between two builds can be judged — a
      // near-tie flip is rounding, a wide-margin flip is a bug.
      float second = -INFINITY;
      for (int i = 0; i < out.lm_vocab_count; ++i)
        if (out.lm_vocab_begin + i != local.id && fslice[i] > second)
          second = fslice[i];
      const int32_t token =
          dgpp::bus_greedy_pick(*bus, rank, world, local, pick_scratch, 60000);
      require(token >= 0 && token < cfg.vocab_size,
              "generated id out of range: " + std::to_string(token));
      DGPP_LOG_INFO(
          "[gen] rank {} {} {}: token {} (local slice [{},{}) best "
          "{} logit {:.4f} second {:.4f})",
          rank, what, s, token, out.lm_vocab_begin,
          out.lm_vocab_begin + out.lm_vocab_count, local.id, local.logit,
          second);
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
      // Teacher forcing: score this step's logits against the text's next
      // token, then feed THAT token (every rank holds the same text, so
      // the ranks agree without the pick; the pick still runs for its
      // argmax and to keep the step's shape identical to serving).
      const auto force = [&](int s, const GlmDiagnosticModel::Outputs& o,
                             int32_t picked) -> int32_t {
        if (!teaching) return picked;
        log_teacher_stats(rank, s, o, fslice, teacher[static_cast<size_t>(s)],
                          picked);
        return static_cast<int32_t>(teacher[static_cast<size_t>(s)]);
      };
      const auto tp0 = std::chrono::steady_clock::now();
      int32_t token = force(0, pre, run_step(pre, "prefill+pick", 0));
      forward_ms_total =
          prefill_ms + std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - tp0)
                           .count();
      // The budget below is decode-steps only: prefill's folds and
      // host-path MoE syncs are a different (amortized) story.
      dgpp::step_timing::reset();

      // ---- the graph era (--decode-graph, DESIGN §6.2) ---------------
      // Record the decode step ONCE — the whole launch sequence
      // including the bus's collective nodes — then replay per token:
      // stage -> arm -> launch -> sync -> finish -> collect, with the
      // EAGER pick riding BETWEEN windows (the mixed era). The capture
      // executes NOTHING: state, positions, and staging are untouched,
      // so the first replay performs step 1 exactly as the eager path
      // would.
      cudaGraphExec_t graph_exec = nullptr;
      // The recorder OWNS the stable boundary buffer every recorded node
      // bakes in, so it must outlive the graph exec, not just the capture
      // (a block-scoped recorder here was a use-after-free that pinned
      // memory's sticky mapping happened to forgive).
      std::unique_ptr<dgpp::GlmGraphRecordReducer> recorder;
      if (decode_graph) {
        const auto t_capture = std::chrono::steady_clock::now();
        // The serving loop reads logits only; the per-layer route traces
        // are 126 D2H nodes per token it would otherwise replay for nobody.
        model.set_decode_route_traces(false);
        // The MoE expert-view tables land in their graph slots here, once;
        // the capture below records kernels that read them in place.
        model.session_graph_prepare();
        recorder =
            std::make_unique<dgpp::GlmGraphRecordReducer>(*bus, model.stream());
        dgpp::GlmBoundaryReducer* eager_reducer =
            model.set_boundary(recorder.get());
        std::string gerr;
        require(bus->graph_record_begin(&gerr),
                "graph_record_begin: " + gerr);
        cudaGraph_t graph = nullptr;
        DGPP_CUDA_OK(cudaStreamBeginCapture(
            model.stream(), cudaStreamCaptureModeThreadLocal));
        model.session_graph_capture_step(0, token);
        DGPP_CUDA_OK(cudaStreamEndCapture(model.stream(), &graph));
        require(graph != nullptr, "decode-graph capture produced no graph");
        require(bus->graph_record_end(&gerr),
                "graph_record_end: " + gerr);
        model.set_boundary(eager_reducer);
        DGPP_CUDA_OK(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                          nullptr, 0));
        cudaGraphDestroy(graph);
        DGPP_LOG_INFO(
            "rank {} decode graph recorded+instantiated in {:.0f}ms "
            "(the bus logged the collective node count)",
            rank,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_capture)
                .count());
      }

      // The inter-step seam (t_pick of step k -> t0s of step k+1) is the
      // ONLY host region no phase below times, and it is where the peers
      // lost ~10 ms in lockstep (2026-09-02, seen only as rank 0's gen-0
      // handshake). It holds two log lines and a token decode — so the
      // probe brackets it and the next step's phases line reports it.
      using Clock = std::chrono::steady_clock;
      ThreadProbe probe;
      Clock::time_point t_pick_prev{}, t_logged_prev{};
      ThreadProbe::Sample probe_prev{};
      bool have_prev = false;

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
        // Boundary phases, timed: where a step's time goes OUTSIDE the
        // replay (stage+arm+launch, the sync, the window finish, the
        // collect, the eager pick) is exactly what the bus's in-window
        // timeline cannot see — and a stall there on one rank shows up on
        // every other rank as "everyone else was late".
        const auto t0s = Clock::now();
        const ThreadProbe::Sample probe_start = probe.sample();
        std::string gap_text;
        if (have_prev) {
          const auto ms_of = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
          };
          gap_text = std::format(
              "; gap {:.2f} = log {:.2f} + top {:.2f}; {}",
              ms_of(t_pick_prev, t0s), ms_of(t_pick_prev, t_logged_prev),
              ms_of(t_logged_prev, t0s),
              ThreadProbe::delta_text(probe_prev, probe_start));
        }
        auto t_launch = t0s, t_sync = t0s, t_finish = t0s, t_collect = t0s;
        const GlmDiagnosticModel::Outputs step = [&] {
          if (graph_exec == nullptr)
            return model.session_step(generated.back());
          // The replay path: the graph re-uploads the staged pinned
          // members, runs every recorded node, and re-folds through
          // the bus's window; the pick stays EAGER between windows.
          std::string gerr;
          model.session_graph_stage(0, generated.back());
          require(bus->graph_replay_arm(&gerr),
                  "graph_replay_arm: " + gerr);
          DGPP_CUDA_OK(cudaGraphLaunch(graph_exec, model.stream()));
          t_launch = Clock::now();
          DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
          t_sync = Clock::now();
          require(bus->graph_replay_finish(60000, &gerr),
                  "graph_replay_finish: " + gerr);
          t_finish = Clock::now();
          GlmDiagnosticModel::Outputs out = model.session_graph_collect(0);
          t_collect = Clock::now();
          return out;
        }();
        token = force(s + 1, step, run_step(step, "step", s + 1));
        const auto t_pick = Clock::now();
        probe_prev = probe.sample();
        const auto ms_between = [](Clock::time_point a, Clock::time_point b) {
          return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double ms = ms_between(t0s, t_pick);
        forward_ms_total += ms;
        DGPP_LOG_INFO("rank {} step {:.0f}ms ({} + pick)", rank, ms,
                      graph_exec ? "graph replay" : "stateful step");
        if (graph_exec != nullptr) {
          // launch -> the graph's first node, through the bus's calibrated
          // globaltimer offset (CLOCK_MONOTONIC == steady_clock on Linux).
          const int64_t launch_gt =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t_launch.time_since_epoch())
                  .count() +
              bus->globaltimer_offset_ns();
          const int64_t start_gt =
              static_cast<int64_t>(model.graph_start_globaltimer());
          DGPP_LOG_INFO(
              "rank {} step phases ms: launch {:.2f} sync {:.2f} finish {:.2f} "
              "collect {:.2f} pick {:.2f}; launch->gpu_start {:.2f}{}",
              rank, ms_between(t0s, t_launch), ms_between(t_launch, t_sync),
              ms_between(t_sync, t_finish), ms_between(t_finish, t_collect),
              ms_between(t_collect, t_pick),
              static_cast<double>(start_gt - launch_gt) / 1e6, gap_text);
        }
        t_pick_prev = t_pick;
        t_logged_prev = Clock::now();
        have_prev = true;
      }
      if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
      recorder.reset();  // after the exec: its buffer is baked into the nodes
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
    dgpp::step_timing::report(
        ("rank " + std::to_string(rank) + " gen_check").c_str());

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
      "  [--teacher-file F  score the text's tokens instead of generating:\n"
      "   per-step [tf] lines for scripts/fabric_logprob.py; steps = its\n"
      "   token count; the file must exist on every rank (--stage-file)]\n"
      "  [--decode-graph] (fabric decode step as a CUDA graph: record once,\n"
      "   replay per token, eager pick between windows; resident only)\n"
      "  [--step-timing] [--rendezvous-timeout-ms N] [--out PREFIX]\n"
      "scheduler mode (--requests): [--max-concurrency N] [--kv-capacity N]\n"
      "  [--sched-plan]\n";

  std::string config_path, ckpt, peer, prompt_text, text_prompt, out_prefix =
                                                    "glm_gen",
              model_id, requests_path;
  int world = 1, rank = 0, steps = 8, rendezvous_timeout_ms = 120000;
  int max_concurrency = 0, kv_capacity = 0;
  bool sched_plan = false, step_timing = false;
  uint16_t port = 29970;
  bool resident = true, incremental = true, no_eos = false;
  bool decode_graph = false;
  std::string system_prompt, chat_text, teacher_file;
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
    else if (a == "--step-timing") step_timing = true;
    else if (a == "--decode-graph") decode_graph = true;
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
    else if (a == "--teacher-file") teacher_file = next();
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
  if (step_timing) dgpp::step_timing::set_enabled(true);
  if (world > 1) {
    require(rank >= 0 && rank < world, "--rank outside --world");
    require(!peer.empty() || rank == 0,
            "ranks > 0 need --peer (rank 0's fabric IP)");
  }
  if (steps < 1) {
    DGPP_LOG_ERROR("--steps must be >= 1");
    return 1;
  }
  if (decode_graph) {
    require(world > 1 && incremental && resident,
            "--decode-graph is the fabric decode step's graph era: it "
            "needs --world > 1, --engine incremental, and RESIDENT "
            "weights (the streaming loader refills one resident per "
            "layer — a recorded graph would bake whichever layer was "
            "last resident; resident bindings are lifetime-stable)");
    require(requests_path.empty(),
             "--decode-graph is the single-session decode gate (no "
             "--requests scheduler mode)");
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
    std::vector<int64_t> teacher;
    if (!teacher_file.empty()) {
      require(requests_path.empty() && incremental,
              "--teacher-file needs the incremental engine and no --requests");
      std::ifstream in(teacher_file, std::ios::binary);
      require(in.good(), "--teacher-file not readable: " + teacher_file);
      std::stringstream buf;
      buf << in.rdbuf();
      teacher = tok.encode(buf.str());
      require(!teacher.empty(), "--teacher-file produced no tokens");
      DGPP_LOG_INFO("teacher text {} bytes -> {} ids (fnv1a {:016x}); "
                    "--steps ignored",
                    buf.str().size(), teacher.size(),
                    fnv1a64(buf.str()));
    }
    return run(cfg, ckpt, world, rank, port, peer, prompt, steps, resident, incremental, no_eos, decode_graph, tok,
               out_prefix, rendezvous_timeout_ms, kv_capacity, teacher);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {}: {}", rank, e.what());
    return 1;
  }
}
