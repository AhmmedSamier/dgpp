// M6 Stage 4: the generation-service gate. A deterministic FakeEngine
// (the scheduler seam) and a FakeFrontend (the tokenizer/template seam)
// stand under the REAL HTTP server + service — the whole OpenAI
// surface runs host-only, no GPU, no model cache:
//   * the exact chat.completion / chat.completion.chunk shapes;
//   * stream lifecycle: role chunk, ordered content deltas whose
//     concatenation equals the full text, finish_reason, usage,
//     [DONE];
//   * EOS → "stop" vs the steps cap → "length";
//   * the loud-refusal ladder (sampling, tools, model) with the OpenAI
//     error object naming the param;
//   * overload: 503 at the admission door;
//   * client disconnect mid-stream → scheduler cancellation (metrics);
//   * /v1/models, /health, /v1/metrics.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <mutex>
#include <optional>
#include <cstring>
#include <thread>
#include <vector>

#include "common/log.hpp"

#include "common/test.hpp"
#include "models/glm_scheduler.hpp"
#include "service/generation_service.hpp"
#include "service/http_server.hpp"

namespace {

using dgpp::glm::SchedulerEngine;
using dgpp::service::GenerationService;
using dgpp::service::HttpServer;
using dgpp::service::ModelFrontend;
using dgpp::service::ServiceConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int32_t kFakeEos = 999;  // the fake's end-of-sequence id

// The deterministic fake model: token i of a request whose rendered
// prompt is P bytes long is ((len*31 + i*7) % 250) + 1 — printable-ish,
// never 0, never kFakeEos. A prompt with len % 4 == 3 answers EOS as
// its SECOND token (an early-stop scenario the gate can target).
int32_t fake_token(size_t prompt_len, int index) {
  return static_cast<int32_t>((prompt_len * 31 + static_cast<size_t>(index) * 7) % 250) + 1;
}
bool fake_eos_second(size_t prompt_len) { return prompt_len % 4 == 3; }
bool fake_eos_prefill(size_t prompt_len) { return prompt_len % 4 == 2; }

class FakeEngine : public SchedulerEngine {
 public:
  struct Live {
    size_t prompt_len = 0;
    int64_t held_blocks = 0;
    int served = 0;
    int32_t last_token = -1;
  };
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens,
             bool can_sample = false)
      : slots_(slots), total_blocks_(total_blocks),
        block_tokens_(block_tokens), can_sample_(can_sample) {}

  // The sampling seam: a sampling-capable fake records the spec each slot
  // was armed with (the request seam's evidence); a greedy fake inherits
  // the base refusal.
  struct Armed {
    dgpp::glm_sample::Params params;
    uint64_t seed = 0;
  };
  bool supports_sampling() const override { return can_sample_; }
  void configure_sampling(int req, const dgpp::glm_sample::Params& p,
                          uint64_t seed) override {
    if (!can_sample_) {
      SchedulerEngine::configure_sampling(req, p, seed);
      return;
    }
    std::lock_guard<std::mutex> lock(armed_mu_);
    armed_.push_back(Armed{p, seed});
  }
  std::vector<Armed> armed() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return armed_;
  }
  // Logprobs: the sampling-capable fake reports every token with logprob
  // -(index+1)/4 and top-N alternatives (the token itself first).
  bool supports_logprobs() const override { return can_sample_; }
  void configure_logprobs(int req, int logprobs) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logprobs(req, logprobs);
      return;
    }
    report_[req] = logprobs;
  }
  std::vector<dgpp::glm_sample::Result> take_logprobs(int req) override {
    std::vector<dgpp::glm_sample::Result> out;
    out.swap(pending_lps_[req]);
    return out;
  }
  // Constrained decoding (M6 6g): the sampling-capable fake can mask and
  // records every active grammar it is armed with (the request seam's
  // evidence); it does not enforce it — the scripts are the outputs.
  bool supports_constraints() const override { return can_sample_; }
  void configure_constraint(int req,
                            const dgpp::glm::GrammarSpec& g) override {
    if (!can_sample_) {
      SchedulerEngine::configure_constraint(req, g);
      return;
    }
    if (!g.active()) return;
    std::lock_guard<std::mutex> lock(armed_mu_);
    grammars_.push_back(g);
  }
  std::vector<dgpp::glm::GrammarSpec> grammars() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return grammars_;
  }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) sum += live.held_blocks;
    return sum;
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  // A scripted answer for prompts of exactly `prompt_len` ids (the 6f
  // gates: tool-call and reasoning id streams); EOS once it runs out.
  void script(size_t prompt_len, std::vector<int32_t> ids) {
    scripts_[prompt_len] = std::move(ids);
  }
  int32_t next_token(const Live& live, int index) const {
    const auto s = scripts_.find(live.prompt_len);
    if (s != scripts_.end())
      return index < static_cast<int>(s->second.size()) ? s->second[index]
                                                        : kFakeEos;
    if (index == 0)
      return fake_eos_prefill(live.prompt_len) ? kFakeEos
                                               : fake_token(live.prompt_len, 0);
    return fake_eos_second(live.prompt_len) && index == 1
               ? kFakeEos
               : fake_token(live.prompt_len, index);
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    if (live_.count(req) != 0)
      throw std::runtime_error("fake: prefill on live slot");
    Live live;
    live.prompt_len = prompt.size();
    live.served = 1;
    live.last_token = next_token(live, 0);
    live_[req] = live;
    note_logprobs(req, live.last_token, 0);
    return live.last_token;
  }
  void note_logprobs(int req, int32_t token, int index) {
    if (report_.count(req) == 0 || report_[req] < 0) return;
    dgpp::glm_sample::Result r;
    r.token = token;
    r.logprob = -static_cast<float>(index + 1) / 4.0f;
    for (int j = 0; j < report_[req]; ++j)
      r.top_logprobs.emplace_back(token + j, r.logprob - static_cast<float>(j));
    pending_lps_[req].push_back(r);
  }

  void reserve(int req, int64_t tokens) override {
    Live& live = live_.at(req);
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    Live& live = live_.at(req);
    live.last_token = next_token(live, live.served);
    note_logprobs(req, live.last_token, live.served);
    ++live.served;
    return {live.last_token};
  }

  void close(int req) override { live_.erase(req); }

 private:
  std::map<size_t, std::vector<int32_t>> scripts_;
  int slots_;
  int64_t total_blocks_;
  int64_t block_tokens_;
  bool can_sample_ = false;
  std::map<int, Live> live_;
  std::map<int, int> report_;
  std::map<int, std::vector<dgpp::glm_sample::Result>> pending_lps_;
  mutable std::mutex armed_mu_;
  std::vector<Armed> armed_;
  std::vector<dgpp::glm::GrammarSpec> grammars_;
};

// The 6f markers of the fake tokenizer: 1001..1008, decoding to their
// literal text (not special, exactly like the real added tokens).
constexpr int64_t kThinkOpen = 1001, kThinkClose = 1002, kToolOpen = 1003,
                  kToolClose = 1004, kKeyOpen = 1005, kKeyClose = 1006,
                  kValueOpen = 1007, kValueClose = 1008;
const std::vector<std::pair<std::string, int64_t>>& marker_table() {
  static const std::vector<std::pair<std::string, int64_t>> t = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</arg_value>", kValueClose}, {"<arg_value>", kValueOpen},
      {"</arg_key>", kKeyClose},   {"<arg_key>", kKeyOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  return t;
}

// A minijson value back to compact JSON (the tests read what the service
// handed the template).
std::string json_of(const dgpp::minijson::Value& v) {
  using K = dgpp::minijson::Value::Kind;
  switch (v.kind()) {
    case K::Null: return "null";
    case K::Bool: return v.as_bool() ? "true" : "false";
    case K::Int: return std::to_string(v.as_int());
    case K::Double: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", v.as_double());
      return buf;
    }
    case K::String: return "\"" + std::string(v.as_string()) + "\"";
    case K::Array: {
      std::string out = "[";
      for (size_t i = 0; i < v.items().size(); ++i)
        out += (i ? "," : "") + json_of(v.items()[i]);
      return out + "]";
    }
    case K::Object: {
      std::string out = "{";
      for (size_t i = 0; i < v.members().size(); ++i)
        out += (i ? "," : "") + ("\"" + v.members()[i].key + "\":") +
               json_of(v.members()[i].value);
      return out + "}";
    }
  }
  return "?";
}

// The fake frontend: bytes ↔ ids (the marker strings map to their ids,
// leftmost-longest like the real added-token scan), and a deterministic
// chat render — the concatenation of every message's content (string, or
// the text of its parts), plus "<think>" when the fake models this
// template's generation prompt — the test computes prompt lengths from
// the same rule. It keeps the last globals the service handed it.
class FakeFrontend : public ModelFrontend {
 public:
  explicit FakeFrontend(bool with_markers = false)
      : with_markers_(with_markers) {}

  std::vector<int64_t> encode_text(std::string_view text) const override {
    std::vector<int64_t> ids;
    for (size_t i = 0; i < text.size();) {
      bool matched = false;
      for (const auto& [s, id] : marker_table()) {
        if (text.compare(i, s.size(), s) == 0) {
          ids.push_back(id);
          i += s.size();
          matched = true;
          break;
        }
      }
      if (!matched) ids.push_back(static_cast<unsigned char>(text[i++]));
    }
    return ids;
  }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    std::string out;
    for (int64_t id : ids) {
      if (id == kFakeEos) continue;
      bool matched = false;
      for (const auto& [s, mid] : marker_table())
        if (mid == id) {
          out += s;
          matched = true;
          break;
        }
      if (!matched && id >= 0 && id < 256) out.push_back(static_cast<char>(id));
    }
    return out;
  }
  std::string render_chat(const dgpp::minijson::Value& globals) const override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      last_globals_ = json_of(globals);
    }
    std::string out;
    for (const auto& msg : globals.at("messages").items()) {
      const auto* content = msg.find("content");
      if (content == nullptr) continue;
      if (content->is_string()) {
        out.append(content->as_string());
      } else {
        for (const auto& part : content->items())
          if (const auto* text = part.find("text")) out.append(text->as_string());
      }
    }
    if (with_markers_) out.append("<think>");
    return out;
  }
  dgpp::glm::ChatMarkers markers() const override {
    dgpp::glm::ChatMarkers m;
    if (!with_markers_) return m;
    m.think_open = {kThinkOpen, "<think>"};
    m.think_close = {kThinkClose, "</think>"};
    m.tool_call_open = {kToolOpen, "<tool_call>"};
    m.tool_call_close = {kToolClose, "</tool_call>"};
    m.arg_key_open = {kKeyOpen, "<arg_key>"};
    m.arg_key_close = {kKeyClose, "</arg_key>"};
    m.arg_value_open = {kValueOpen, "<arg_value>"};
    m.arg_value_close = {kValueClose, "</arg_value>"};
    return m;
  }
  std::string last_globals() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_globals_;
  }

 private:
  bool with_markers_;
  mutable std::mutex mu_;
  mutable std::string last_globals_;
};

// --- the raw-socket client (as the http gate's) ------------------------
class Client {
 public:
  explicit Client(uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "client socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1,
            "client inet_pton");
    require(::connect(fd_, reinterpret_cast<sockaddr*>(&addr),
                      sizeof(addr)) == 0,
            "client connect");
    int yes = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  }
  ~Client() {
    if (fd_ >= 0) ::close(fd_);
  }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void send_all(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
      const ssize_t put = ::send(fd_, s.data() + off, s.size() - off, 0);
      require(put > 0, "client send");
      off += static_cast<size_t>(put);
    }
  }
  // Everything that arrives within the budget (EOF sets closed_).
  std::string read_available(int timeout_ms) {
    std::string out;
    int waited = 0;
    while (true) {
      pollfd p{fd_, POLLIN, 0};
      const int r = ::poll(&p, 1, 10);
      if (r < 0) break;
      if (r == 0) {
        waited += 10;
        if (waited >= timeout_ms) break;
        continue;
      }
      char buf[4096];
      const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
      if (got > 0) {
        out.append(buf, static_cast<size_t>(got));
        waited = 0;
        continue;
      }
      if (got == 0) closed_ = true;
      break;
    }
    return out;
  }
  // Reads until `needle` shows up (or the budget dies) — for responses
  // that only complete after the engine runs.
  std::string read_until(const std::string& needle, int budget_ms) {
    std::string out;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(budget_ms);
    while (out.find(needle) == std::string::npos &&
           std::chrono::steady_clock::now() < deadline && !closed_) {
      pollfd p{fd_, POLLIN, 0};
      if (::poll(&p, 1, 25) > 0) {
        char buf[4096];
        const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
        if (got > 0) out.append(buf, static_cast<size_t>(got));
        if (got == 0) closed_ = true;
      }
    }
    return out;
  }
  void hard_close() {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }

 private:
  int fd_ = -1;
  bool closed_ = false;
};

// --- the service rig: HTTP thread + gated engine thread ---------------
struct ServiceRig {
  static constexpr int kSlots = 4;
  FakeEngine engine;
  FakeFrontend frontend;
  ServiceConfig cfg;
  GenerationService service;
  HttpServer http;

  std::thread http_loop;
  std::thread engine_loop;
  std::atomic<bool> gate{false};   // pause the engine thread (tests)
  std::atomic<int> pass_delay_ms{0};  // slow the engine to a human pace (tests)
  std::atomic<bool> stopping{false};

  // `sampling_defaults`: the served defaults (greedy unless a test hands
  // the checkpoint's stochastic ones); `can_sample`: whether the fake
  // engine advertises the sampler.
  // `with_markers`: the fake tokenizer carries the template's markers and
  // the fake render opens <think> (the 6f rigs); `reasoning_in_content`:
  // the fold knob.
  explicit ServiceRig(int queue_limit = 8,
                      dgpp::glm_sample::Params sampling_defaults =
                          dgpp::glm_sample::greedy_params(),
                      bool can_sample = false,
                      std::optional<uint64_t> fixed_seed = std::nullopt,
                      bool with_markers = false,
                      bool reasoning_in_content = false,
                      dgpp::glm::AdmissionPolicy admission = {})
      : engine(kSlots, /*total_blocks=*/100, /*block_tokens=*/4, can_sample),
        frontend(with_markers),
        cfg([&] {
          ServiceConfig c;
          c.model_id = "glm-5.3-flash-fp8";
          c.default_max_tokens = 8;
          c.queue_limit = queue_limit;
          c.sampling_defaults = sampling_defaults;
          c.fixed_seed = fixed_seed;
          c.reasoning_in_content = reasoning_in_content;
          c.admission = admission;
          return c;
        }()),
        service(cfg, &engine, &frontend, {kFakeEos}),
        http(0, &service, /*max_connections=*/64) {
    http_loop = std::thread([this] { http.serve(); });
    engine_loop = std::thread([this] {
      while (!stopping.load()) {
        if (gate.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        if (!service.engine_pass())
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        else if (pass_delay_ms.load() > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(pass_delay_ms.load()));
      }
    });
  }
  // The app's drain-on-stop (M6 6c), minus the journal: the engine thread
  // stops at a pass boundary, begin_shutdown() flags the live requests and
  // sheds the queue, one more pass retires them, and the HTTP pump answers
  // everything before the server stops.
  int drain(bool stop_http) {
    stopping = true;
    gate = false;
    if (engine_loop.joinable()) engine_loop.join();
    const int interrupted = service.begin_shutdown();
    service.engine_pass();
    for (int i = 0; i < 200 && !service.drained(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (stop_http) {
      http.stop();
      if (http_loop.joinable()) http_loop.join();
    }
    return interrupted;
  }
  ~ServiceRig() {
    stopping = true;
    http.stop();
    if (http_loop.joinable()) http_loop.join();
    if (engine_loop.joinable()) engine_loop.join();
  }
  ServiceRig(const ServiceRig&) = delete;
  ServiceRig& operator=(const ServiceRig&) = delete;

  uint16_t port() const { return http.port(); }
};

const std::string kModel = "glm-5.3-flash-fp8";

std::string chat_body(const std::string& content, int max_tokens,
                      const std::string& extra = "") {
  return "{\"model\":\"" + kModel +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" + content +
         "\"}],\"max_tokens\":" + std::to_string(max_tokens) + extra + "}";
}

// The text the fake model generates for a prompt of `prompt_len` bytes
// over `n` tokens (the same arithmetic as FakeEngine, decoded).
std::string fake_text(size_t prompt_len, int n) {
  std::string out;
  for (int i = 0; i < n; ++i)
    out.push_back(static_cast<char>(fake_token(prompt_len, i)));
  return out;
}

DGPP_TEST(serve_chatNonStream_exactCompletionShape) {
  // GIVEN the service with a 4-byte prompt ("abcd" → 4 ids, no early
  // EOS) and max_tokens 3,
  ServiceRig rig;
  Client c(rig.port());

  // WHEN the chat completion completes,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(chat_body("abcd", 3).size()) +
             "\r\n\r\n" + chat_body("abcd", 3));
  const std::string resp = c.read_until("usage", 5000);

  // THEN the body is the chat.completion object: the exact generated
  // text, finish_reason "length" (the steps cap), and usage counting
  // prompt 4 / completion 3 / total 7.
  require(resp.find("200 OK") != std::string::npos, "status: " + resp);
  require(resp.find("\"object\":\"chat.completion\"") != std::string::npos,
          "object kind");
  require(resp.find("\"role\":\"assistant\",\"content\":\"" +
                    fake_text(4, 3) + "\"") != std::string::npos,
          "content text: " + resp);
  require(resp.find("\"finish_reason\":\"length\"") != std::string::npos,
          "steps cap → length");
  require(resp.find("\"prompt_tokens\":4,\"completion_tokens\":3,"
                    "\"total_tokens\":7") != std::string::npos,
          "usage arithmetic: " + resp);
}

DGPP_TEST(serve_chatOneTokenLimit_returnsExactlyOneToken) {
  // The prefill pick is completion token one. This boundary used to fall
  // through to the same tick's decode because only step() checked the cap.
  ServiceRig rig;
  Client c(rig.port());
  const std::string body = chat_body("abcd", 1);
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("usage", 5000);

  require(resp.find("\"role\":\"assistant\",\"content\":\"" +
                    fake_text(4, 1) + "\"") != std::string::npos,
          "max_tokens=1 content: " + resp);
  require(resp.find("\"prompt_tokens\":4,\"completion_tokens\":1,"
                    "\"total_tokens\":5") != std::string::npos,
          "max_tokens=1 usage: " + resp);
  require(resp.find("\"finish_reason\":\"length\"") != std::string::npos,
          "one-token cap finishes by length");
}

DGPP_TEST(serve_chatStream_chunkLifecycleInOrder) {
  // GIVEN a streaming request with include_usage (prompt "abcde" — 5
  // bytes triggers neither EOS rule, so the run goes to the cap),
  ServiceRig rig;
  Client c(rig.port());
  const std::string body =
      chat_body("abcde", 3, ",\"stream\":true,"
                             "\"stream_options\":{\"include_usage\":true}");

  // WHEN the stream completes,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("[DONE]", 5000);

  // THEN the lifecycle is exactly OpenAI's: headers, role chunk, the
  // generated text arriving as in-order content deltas (tokens may
  // coalesce into a chunk — the client contract is their
  // concatenation), the final chunk (length), the usage chunk, [DONE].
  require(resp.find("text/event-stream") != std::string::npos,
          "SSE content type");
  const size_t role = resp.find("\"delta\":{\"role\":\"assistant\","
                                "\"content\":\"\"}");
  const size_t final =
      resp.find("\"delta\":{},\"logprobs\":null,\"finish_reason\":\"length\"");
  const size_t usage =
      resp.find("\"choices\":[],\"usage\":{\"prompt_tokens\":5,"
                "\"completion_tokens\":3,\"total_tokens\":8}");
  const size_t done = resp.find("data: [DONE]");
  require(role != std::string::npos, "role chunk present");
  // Concatenate every content payload in arrival order.
  std::string concatenated;
  size_t pos = 0;
  int content_chunks = 0;
  while ((pos = resp.find("\"content\":\"", pos)) != std::string::npos) {
    const size_t vstart = pos + std::string("\"content\":\"").size();
    const size_t vend = resp.find("\"", vstart);
    require(vend != std::string::npos, "content field terminated");
    // The role chunk's empty content carries no text.
    if (vend > vstart) {
      concatenated.append(resp, vstart, vend - vstart);
      ++content_chunks;
    }
    pos = vend;
  }
  require(content_chunks >= 1, "content deltas present");
  require(concatenated == fake_text(5, 3),
          "concatenated deltas == the generated text: got \"" +
              concatenated + "\" expected \"" + fake_text(5, 3) + "\"");
  require(final != std::string::npos && final > role, "final chunk present");
  require(usage != std::string::npos && usage > final,
          "usage chunk after the final chunk");
  require(done != std::string::npos && done > usage, "[DONE] last");
}

DGPP_TEST(serve_eosMidAnswer_finishReasonStop) {
  // GIVEN a prompt of 3 bytes (len % 4 == 3 → the fake answers EOS as
  // its second token),
  ServiceRig rig;
  Client c(rig.port());

  // WHEN the completion runs to its natural stop,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(chat_body("abc", 8).size()) +
             "\r\n\r\n" + chat_body("abc", 8));
  const std::string resp = c.read_until("usage", 5000);

  // THEN finish_reason is "stop" (EOS beat the cap), completion tokens
  // count the EOS pick (2), and the EOS id itself contributes no text.
  require(resp.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "EOS → stop: " + resp);
  require(resp.find("\"prompt_tokens\":3,\"completion_tokens\":2,"
                    "\"total_tokens\":5") != std::string::npos,
          "EOS counted as a completion token");
  require(resp.find("\"content\":\"" + fake_text(3, 1) + "\"") !=
              std::string::npos,
          "one text token, then EOS (decoded to nothing)");
}

DGPP_TEST(serve_refusalLadder_openAIErrorObjects) {
  // GIVEN the service,
  ServiceRig rig;

  // THEN every unimplemented knob refuses with the OpenAI error object
  // naming the param — never a silent ignore, never a bare string.
  const auto post_and_expect = [&](const std::string& body, int status,
                                   const std::string& needle) {
    Client c(rig.port());
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_available(800);
    require(resp.find(std::to_string(status) + " ") != std::string::npos,
            "status " + std::to_string(status) + ": " + resp.substr(0, 120));
    require(resp.find("\"error\":{") != std::string::npos,
            "error object shape: " + resp.substr(0, 200));
    require(resp.find(needle) != std::string::npos,
            "needle " + needle + ": " + resp.substr(0, 400));
  };
  // A greedy-only engine refuses stochastic requests by capability; an
  // out-of-range top_p is refused by validation regardless of engine.
  post_and_expect(chat_body("abcd", 3, ",\"temperature\":0.7"), 400,
                  "\"code\":\"sampling_unsupported\"");
  post_and_expect(chat_body("abcd", 3, ",\"top_p\":5"), 400,
                  "\"param\":\"top_p\"");
  post_and_expect(chat_body("abcd", 3, ",\"logit_bias\":{\"1\":2}"), 400,
                  "\"param\":\"logit_bias\"");
  post_and_expect(chat_body("abcd", 3,
                            ",\"tools\":[{\"type\":\"function\"}]"),
                  400, "\"param\":\"tools[0].function.name\"");
  post_and_expect(chat_body("abcd", 3, ",\"n\":2"), 400, "\"param\":\"n\"");
  post_and_expect(
      "{\"model\":\"wrong-model\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}]}",
      404, "\"code\":\"model_not_found\"");
  post_and_expect("{not json", 400, "invalid JSON body");
  // Content parts render through the template (6f); a part without a
  // type, a role the template does not know, and tools on a frontend
  // without the markers all refuse by name.
  post_and_expect(
      "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
      "\"content\":[{\"text\":\"hi\"}]}]}",
      400, "\"param\":\"messages[0].content\"");
  post_and_expect(
      "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"developer\","
      "\"content\":\"hi\"}]}",
      400, "\"param\":\"messages[0].role\"");
  post_and_expect(chat_body("abcd", 3,
                            ",\"tools\":[{\"type\":\"function\","
                            "\"function\":{\"name\":\"f\"}}]"),
                  400, "\"code\":\"tools_unsupported\"");
}

DGPP_TEST(serve_overloadedQueue_503AtTheDoor) {
  // GIVEN a one-deep admission queue and a PAUSED engine (the pending
  // admission cannot drain),
  ServiceRig rig(/*queue_limit=*/1);
  rig.gate = true;

  // WHEN two requests arrive,
  Client first(rig.port());
  first.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                 "Content-Type: application/json\r\nContent-Length: " +
                 std::to_string(chat_body("abcd", 2).size()) +
                 "\r\n\r\n" + chat_body("abcd", 2));
  // (Non-streaming requests receive NOTHING until they complete — no
  // early bytes to await; the first one just sits pending.)
  Client second(rig.port());
  second.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                  "Content-Type: application/json\r\nContent-Length: " +
                  std::to_string(chat_body("efgh", 2).size()) +
                  "\r\n\r\n" + chat_body("efgh", 2));

  // THEN the second is shed at the door with 503 + the error object
  // (the overload contract), while the first still completes once the
  // engine resumes.
  const std::string shed = second.read_available(800);
  require(shed.find("503") != std::string::npos, "503 at the door: " +
                                                      shed.substr(0, 120));
  require(shed.find("\"code\":\"overloaded\"") != std::string::npos,
          "overload error object");
  rig.gate = false;
  const std::string done = first.read_until("usage", 5000);
  require(done.find("\"finish_reason\":\"length\"") != std::string::npos,
          "the first request completes after the pause: " + done.substr(0, 200));
}

DGPP_TEST(serve_clientDisconnectMidStream_mapsToSchedulerCancel) {
  // GIVEN a paused engine and a streaming request whose connection the
  // client will kill,
  ServiceRig rig;
  rig.gate = true;
  Client c(rig.port());
  const std::string body = chat_body("abcdefgh", 6, ",\"stream\":true");
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  require(c.read_available(500).find("text/event-stream") !=
              std::string::npos,
          "stream headers arrived before the disconnect");

  // WHEN the client vanishes and the engine resumes,
  c.hard_close();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rig.gate = false;

  // THEN the disconnect became a scheduler cancellation: metrics
  // counts it, and the record settles without any writer touch (the
  // engine runs to a cancelled retire within a couple passes).
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  Client m(rig.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string metrics = m.read_until("requests_cancelled", 2000);
  require(metrics.find("\"requests_cancelled\":1") != std::string::npos,
          "the disconnect counted as one cancellation: " +
              metrics.substr(metrics.find("service")));
}

DGPP_TEST(serve_modelsHealthMetrics_theOpsSurface) {
  // GIVEN the service,
  ServiceRig rig;

  // WHEN the operational endpoints are read,
  Client models(rig.port());
  models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ml = models.read_available(800);
  Client health(rig.port());
  health.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string hl = health.read_available(800);
  Client metrics(rig.port());
  metrics.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string met = metrics.read_available(800);

  // THEN each answers its documented shape.
  require(ml.find("\"object\":\"list\"") != std::string::npos &&
              ml.find("\"id\":\"" + kModel + "\"") != ml.npos &&
              ml.find("\"object\":\"model\"") != std::string::npos,
          "models list: " + ml.substr(0, 200));
  require(hl.find("\"status\":\"ok\"") != std::string::npos, "health");
  require(met.find("\"scheduler\":{") != std::string::npos &&
              met.find("\"service\":{") != std::string::npos,
          "metrics sections: " + met.substr(0, 200));
}

DGPP_TEST(serve_legacyCompletions_theTextCompletionObject) {
  // GIVEN the legacy prompt API,
  ServiceRig rig;
  Client c(rig.port());
  const std::string body =
      "{\"model\":\"" + kModel + "\",\"prompt\":\"hello\","
      "\"max_tokens\":2}";

  // WHEN it completes,
  c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("usage", 5000);

  // THEN the body is the text_completion object with the same token
  // arithmetic.
  require(resp.find("\"object\":\"text_completion\"") != std::string::npos,
          "legacy object: " + resp.substr(0, 200));
  require(resp.find("\"text\":\"" + fake_text(5, 2) + "\"") !=
              std::string::npos,
          "legacy text");
  require(resp.find("\"prompt_tokens\":5,\"completion_tokens\":2,"
                    "\"total_tokens\":7") != std::string::npos,
          "legacy usage");
}

// The checkpoint's defaults for the sampling rigs (generation_config.json:
// temperature 1.0, top_p 0.95, nothing else).
dgpp::glm_sample::Params model_defaults() {
  dgpp::glm_sample::Params p;
  p.temperature = 1.0f;
  p.top_p = 0.95f;
  return p;
}

std::string post_chat(ServiceRig& rig, const std::string& body,
                      const std::string& until = "", int budget_ms = 3000) {
  Client c(rig.port());
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  return until.empty() ? c.read_available(800) : c.read_until(until, budget_ms);
}

bool same_float(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

DGPP_TEST(serve_sampling_defaultsFillOmittedFieldsAndReachTheEngine) {
  // GIVEN a sampling-capable engine and the checkpoint's defaults,
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);

  // THEN /v1/models advertises the surface and the exact defaults,
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"sampling\":{\"available\":true,\"defaults\":{"
                    "\"temperature\":1,\"top_p\":0.95,\"top_k\":0,"
                    "\"min_p\":0,\"repetition_penalty\":1}}") !=
                std::string::npos,
            "models sampling surface: " + ml.substr(0, 400));
  }

  // WHEN a request omits every sampling field, one names a seed and
  // overrides, and one asks for greedy explicitly,
  const std::string r1 = post_chat(rig, chat_body("abcd", 2), "usage");
  const std::string r2 = post_chat(
      rig,
      chat_body("abcd", 2,
                ",\"seed\":42,\"temperature\":0.7,\"top_p\":0.5,"
                "\"presence_penalty\":1.5,\"frequency_penalty\":-0.25,"
                "\"top_k\":40,\"min_p\":0.05,\"repetition_penalty\":1.1"),
      "usage");
  const std::string r3 =
      post_chat(rig, chat_body("abcd", 2, ",\"temperature\":0"), "usage");
  require(r1.find("\"object\":\"chat.completion\"") != std::string::npos &&
              r2.find("\"object\":\"chat.completion\"") != std::string::npos &&
              r3.find("\"object\":\"chat.completion\"") != std::string::npos,
          "all three requests complete");

  // THEN the engine was armed, in order, with the defaults + a drawn seed,
  // the explicit spec + seed 42, and the greedy spec.
  const std::vector<FakeEngine::Armed> armed = rig.engine.armed();
  require(armed.size() == 3, "three slots armed, got " +
                                 std::to_string(armed.size()));
  require(same_float(armed[0].params.temperature, 1.0f) &&
              same_float(armed[0].params.top_p, 0.95f) &&
              armed[0].params.top_k == 0,
          "omitted fields take the model defaults");
  require(same_float(armed[1].params.temperature, 0.7f) &&
              same_float(armed[1].params.top_p, 0.5f) &&
              same_float(armed[1].params.presence_penalty, 1.5f) &&
              same_float(armed[1].params.frequency_penalty, -0.25f) &&
              armed[1].params.top_k == 40 &&
              same_float(armed[1].params.min_p, 0.05f) &&
              same_float(armed[1].params.repetition_penalty, 1.1f) &&
              armed[1].seed == 42,
          "explicit fields override the defaults exactly");
  require(same_float(armed[2].params.temperature, 0.0f),
          "temperature 0 is the greedy spec");
  require(armed[0].seed != armed[2].seed,
          "seedless requests draw distinct seeds");
}

DGPP_TEST(serve_sampling_validationNamesTheField) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);
  const auto refused = [&](const std::string& extra, const std::string& param) {
    const std::string resp = post_chat(rig, chat_body("abcd", 2, extra));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"" + param + "\"") != std::string::npos,
            "expected a 400 naming " + param + ": " + resp.substr(0, 300));
  };
  refused(",\"temperature\":2.5", "temperature");
  refused(",\"temperature\":\"hot\"", "temperature");
  refused(",\"top_p\":0", "top_p");
  refused(",\"top_p\":1.01", "top_p");
  refused(",\"presence_penalty\":2.5", "presence_penalty");
  refused(",\"frequency_penalty\":-3", "frequency_penalty");
  refused(",\"top_k\":1.5", "top_k");
  refused(",\"top_k\":-1", "top_k");
  refused(",\"min_p\":2", "min_p");
  refused(",\"repetition_penalty\":0", "repetition_penalty");
  refused(",\"seed\":1.5", "seed");
  require(rig.engine.armed().empty(), "no refused request reached the engine");
}

DGPP_TEST(serve_sampling_greedyEngineCollapsesDefaultsLoudly) {
  // GIVEN the checkpoint's stochastic defaults but an engine that cannot
  // sample (today's graph engine),
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/false);

  // THEN the served defaults are greedy and say so,
  Client models(rig.port());
  models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ml = models.read_available(800);
  require(ml.find("\"sampling\":{\"available\":false,\"defaults\":{"
                  "\"temperature\":0,") != std::string::npos,
          "collapsed defaults: " + ml.substr(0, 400));
  // a field-less request is served (greedy),
  const std::string ok = post_chat(rig, chat_body("abcd", 2), "usage");
  require(ok.find("\"object\":\"chat.completion\"") != std::string::npos,
          "the greedy default is served");
  // and an explicit stochastic request is refused by capability.
  const std::string refused =
      post_chat(rig, chat_body("abcd", 2, ",\"temperature\":1"));
  require(refused.find("\"code\":\"sampling_unsupported\"") !=
              std::string::npos,
          "stochastic request refused: " + refused.substr(0, 300));
}

DGPP_TEST(serve_sampling_fixedSeedAppliesToSeedlessRequests) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 /*fixed_seed=*/uint64_t{777});
  (void)post_chat(rig, chat_body("abcd", 2), "usage");
  (void)post_chat(rig, chat_body("abcd", 2, ",\"seed\":5"), "usage");
  const std::vector<FakeEngine::Armed> armed = rig.engine.armed();
  require(armed.size() == 2 && armed[0].seed == 777 && armed[1].seed == 5,
          "the fixed seed fills seedless requests; explicit seeds win");
}

DGPP_TEST(serve_logprobs_openAIShapesOnEveryRoute) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);
  // Non-stream chat: the content array with one entry per token, each with
  // the token text, its bytes and top_logprobs.
  const std::string one =
      post_chat(rig, chat_body("abcd", 3, ",\"logprobs\":true,\"top_logprobs\":2"),
                "usage");
  require(one.find("\"logprobs\":{\"content\":[{\"token\":") != std::string::npos,
          "chat logprobs content: " + one.substr(0, 400));
  require(one.find("\"logprob\":-0.25,\"bytes\":[") != std::string::npos,
          "the first token's logprob and bytes: " + one.substr(0, 600));
  require(one.find("\"top_logprobs\":[{\"token\":") != std::string::npos,
          "top_logprobs entries");
  {
    size_t count = 0, pos = 0;
    while ((pos = one.find("\"bytes\":[", pos)) != std::string::npos) {
      ++count;
      pos += 8;
    }
    // 3 tokens, each with 2 alternatives: 9 bytes arrays.
    require(count == 9, "three entries with two alternatives each, got " +
                            std::to_string(count));
  }
  // Streaming chat: content chunks carry the entries since the last one.
  {
    Client c(rig.port());
    const std::string body =
        chat_body("abcd", 3, ",\"stream\":true,\"logprobs\":true");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    require(resp.find("\"logprobs\":{\"content\":[{\"token\":") != std::string::npos,
            "streamed logprobs: " + resp.substr(0, 400));
    require(resp.find("\"top_logprobs\":[]") != std::string::npos,
            "logprobs without top_logprobs report empty alternatives");
  }
  // Legacy completions: the tokens/token_logprobs/top_logprobs/text_offset
  // object.
  {
    Client c(rig.port());
    const std::string body =
        "{\"model\":\"" + kModel + "\",\"prompt\":\"hello\",\"max_tokens\":2,"
        "\"logprobs\":1}";
    c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("usage", 5000);
    require(resp.find("\"logprobs\":{\"tokens\":[") != std::string::npos &&
                resp.find("\"token_logprobs\":[-0.25,-0.5]") != std::string::npos &&
                resp.find("\"text_offset\":[0,1]") != std::string::npos,
            "legacy logprobs object: " + resp.substr(0, 500));
  }
  // Refusals name the field.
  const std::string no_flag =
      post_chat(rig, chat_body("abcd", 2, ",\"top_logprobs\":2"));
  require(no_flag.find("\"param\":\"top_logprobs\"") != std::string::npos,
          "top_logprobs without logprobs");
  const std::string too_many =
      post_chat(rig, chat_body("abcd", 2, ",\"logprobs\":true,\"top_logprobs\":25"));
  require(too_many.find("\"param\":\"top_logprobs\"") != std::string::npos,
          "top_logprobs above 20");
  ServiceRig greedy_rig(/*queue_limit=*/8);
  const std::string unsupported =
      post_chat(greedy_rig, chat_body("abcd", 2, ",\"logprobs\":true"));
  require(unsupported.find("\"code\":\"logprobs_unsupported\"") != std::string::npos,
          "an engine without logprobs refuses");
}


// ---- M6 6f: tools and reasoning ------------------------------------------

std::string post_until_usage(ServiceRig& rig, const std::string& body) {
  return post_chat(rig, body, "usage", 5000);
}

std::vector<int32_t> script_of(const ServiceRig& rig, const std::string& text,
                               bool eos = true) {
  std::vector<int32_t> out;
  for (const int64_t id : rig.frontend.encode_text(text))
    out.push_back(static_cast<int32_t>(id));
  if (eos) out.push_back(kFakeEos);
  return out;
}

const std::string kWeatherTools =
    ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Weather\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":"
    "\"integer\"}},\"required\":[\"city\"]}}}]";

// Concatenates every `"<field>":"..."` payload in arrival order (the SSE
// delta contract: clients concatenate fragments).
std::string concat_field(const std::string& resp, const std::string& field) {
  std::string out;
  const std::string needle = "\"" + field + "\":\"";
  size_t pos = 0;
  while ((pos = resp.find(needle, pos)) != std::string::npos) {
    const size_t vstart = pos + needle.size();
    size_t vend = vstart;
    while (vend < resp.size() && resp[vend] != '"') vend += resp[vend] == '\\' ? 2 : 1;
    out.append(resp, vstart, vend - vstart);
    pos = vend;
  }
  return out;
}

DGPP_TEST(serve_tools_requestSideRendersThroughTheTemplateAndRefusesByName) {
  // GIVEN a frontend with the template's markers (tool calls available),
  ServiceRig rig(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"tools\":{\"available\":true,\"constrained\":false},"
                    "\"response_format\":{\"json_object\":false,\"json_schema\":false},"
                    "\"reasoning\":{\"in_content\":false}") != std::string::npos,
            "models advertise the tool surface (no masks on a greedy engine): " +
                ml.substr(0, 400));
  }

  // WHEN requests carry tools and template knobs, THEN the template sees
  // exactly what OpenAI semantics prescribe.
  (void)post_until_usage(rig, chat_body("abcd", 2, kWeatherTools));
  std::string g = rig.frontend.last_globals();
  require(g.find("\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
                 "\"get_weather\"") != std::string::npos,
          "tools reach the template (tool_choice auto): " + g);
  require(g.find("\"messages\":[{\"role\":\"user\",\"content\":\"abcd\"}]") !=
              std::string::npos,
          "messages pass through: " + g);

  (void)post_until_usage(
      rig, chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"none\""));
  g = rig.frontend.last_globals();
  require(g.find("\"tools\"") == std::string::npos,
          "tool_choice none omits the tools from the render: " + g);

  (void)post_until_usage(
      rig, chat_body("abcd", 2,
                     ",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":"
                     "{\"clear_thinking\":false}"));
  g = rig.frontend.last_globals();
  require(g.find("\"reasoning_effort\":\"low\"") != std::string::npos &&
              g.find("\"clear_thinking\":false") != std::string::npos,
          "reasoning_effort and chat_template_kwargs render: " + g);

  // The assistant tool_calls wire form (arguments as a JSON string) is
  // parsed into the mapping the template iterates; a null assistant
  // content becomes ""; tool messages pass with their tool_call_id, in
  // both content forms.
  const std::string conversation =
      "{\"model\":\"" + kModel + "\",\"max_tokens\":2,\"messages\":["
      "{\"role\":\"user\",\"content\":\"hi\"},"
      "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":"
      "\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"arguments\":\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 2}\"}}]},"
      "{\"role\":\"tool\",\"content\":\"18C\",\"tool_call_id\":\"call_1\"},"
      "{\"role\":\"assistant\",\"content\":\"It is 18C.\",\"reasoning_content\":"
      "\"looked it up\"},"
      "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"and Rome?\"}]},"
      "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_2\",\"type\":"
      "\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":"
      "{\"city\":\"Rome\"}}}]},"
      "{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\",\"output\":"
      "\"20C\"}]}]" + kWeatherTools + "}";
  const std::string ok = post_until_usage(rig, conversation);
  require(ok.find("\"object\":\"chat.completion\"") != std::string::npos,
          "the tool conversation is served: " + ok.substr(0, 300));
  g = rig.frontend.last_globals();
  require(g.find("{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":"
                 "\"call_1\",\"type\":\"function\",\"function\":{\"name\":"
                 "\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2}}}]}") !=
              std::string::npos,
          "string arguments parsed to a mapping, null content to \"\": " + g);
  require(g.find("\"arguments\":{\"city\":\"Rome\"}") != std::string::npos,
          "object arguments pass through: " + g);
  require(g.find("\"reasoning_content\":\"looked it up\"") != std::string::npos,
          "reasoning_content passes through: " + g);
  require(g.find("{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\","
                 "\"output\":\"20C\"}]}") != std::string::npos,
          "the tool output list passes through: " + g);

  // The refusals name the field.
  const auto refused = [&](const std::string& body, const std::string& needle) {
    const std::string resp = post_chat(rig, body);
    require(resp.find("400 ") != std::string::npos &&
                resp.find(needle) != std::string::npos,
            "expected a 400 with " + needle + ": " + resp.substr(0, 400));
  };
  refused(chat_body("abcd", 2, ",\"tool_choice\":\"required\""),
          "\"param\":\"tool_choice\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"sometimes\""),
          "\"param\":\"tool_choice\"");
  refused(chat_body("abcd", 2,
                    kWeatherTools + ",\"tool_choice\":{\"type\":\"function\","
                                    "\"function\":{\"name\":\"nope\"}}"),
          "\"param\":\"tool_choice.function.name\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"parallel_tool_calls\":false"),
          "\"code\":\"constrained_decoding_unsupported\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"required\""),
          "\"code\":\"constrained_decoding_unsupported\"");
  refused(chat_body("abcd", 2, ",\"tools\":[{\"type\":\"function\","
                               "\"function\":{\"description\":\"x\"}}]"),
          "\"param\":\"tools[0].function.name\"");
  refused(chat_body("abcd", 2, ",\"reasoning_effort\":\"extreme\""),
          "\"param\":\"reasoning_effort\"");
  refused(chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"enable_thinking\":false}"),
          "\"param\":\"chat_template_kwargs.enable_thinking\"");
  refused(chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"foo\":1}"),
          "\"param\":\"chat_template_kwargs.foo\"");
  refused(chat_body("abcd", 2,
                    ",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":"
                    "{\"reasoning_effort\":\"high\"}"),
          "\"param\":\"chat_template_kwargs.reasoning_effort\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
          "\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":null}]}",
          "\"param\":\"messages[1].content\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
          "\"content\":\"hi\"},{\"role\":\"assistant\",\"tool_calls\":[{\"function\":"
          "{\"name\":\"f\",\"arguments\":\"[1]\"}}]}]}",
          "\"param\":\"messages[1].tool_calls[0].function.arguments\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"tool\","
          "\"content\":7}]}",
          "\"param\":\"messages[0].content\"");
}

DGPP_TEST(serve_toolCalls_oneShotMessageShapeAndFinishReason) {
  // GIVEN a scripted turn: reasoning, </think>, content, one call with a
  // string and an integer argument, EOS (prompt "abcd" + <think> = 5 ids),
  ServiceRig rig(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  const std::string turn =
      "Think</think>Sure<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3"
      "</arg_value></tool_call>";
  const std::vector<int32_t> script = script_of(rig, turn);
  rig.engine.script(5, script);

  // WHEN the chat completion completes,
  const std::string resp =
      post_until_usage(rig, chat_body("abcd", 64, kWeatherTools));

  // THEN the message carries content, reasoning_content and the parsed
  // call with json.dumps-form arguments; finish_reason is "tool_calls".
  require(resp.find("\"message\":{\"role\":\"assistant\",\"content\":\"Sure\","
                    "\"reasoning_content\":\"Think\",\"tool_calls\":[{\"id\":"
                    "\"call_") != std::string::npos,
          "message shape: " + resp);
  require(resp.find("\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                    "\"arguments\":\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 3}\"}}]}") !=
              std::string::npos,
          "tool call shape: " + resp);
  require(resp.find("\"finish_reason\":\"tool_calls\"") != std::string::npos,
          "finish_reason tool_calls: " + resp);
  require(resp.find("\"prompt_tokens\":5,\"completion_tokens\":" +
                    std::to_string(script.size())) != std::string::npos,
          "usage counts every id (EOS included): " + resp);

  // A turn with calls only reports content null.
  ServiceRig rig2(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                  /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  rig2.engine.script(
      6, script_of(rig2, "Think</think><tool_call>get_weather<arg_key>city"
                         "</arg_key><arg_value>Rome</arg_value></tool_call>"));
  const std::string only =
      post_until_usage(rig2, chat_body("abcde", 64, kWeatherTools));
  require(only.find("\"content\":null,\"reasoning_content\":\"Think\","
                    "\"tool_calls\":[") != std::string::npos,
          "content null with calls only: " + only);
  Client m(rig2.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string metrics = m.read_until("tool_calls_out", 2000);
  require(metrics.find("\"tool_calls_out\":1") != std::string::npos,
          "metrics count the call: " + metrics);
}

DGPP_TEST(serve_toolCalls_streamDeltasInOrder) {
  ServiceRig rig(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  rig.engine.script(
      5, script_of(rig, "Think</think>Sure<tool_call>get_weather<arg_key>city"
                        "</arg_key><arg_value>Paris</arg_value><arg_key>days"
                        "</arg_key><arg_value>3</arg_value></tool_call>"
                        "<tool_call>get_weather<arg_key>city</arg_key>"
                        "<arg_value>Oslo</arg_value></tool_call>"));
  Client c(rig.port());
  const std::string body =
      chat_body("abcd", 128, kWeatherTools + ",\"stream\":true");
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("[DONE]", 5000);

  // The deltas: reasoning_content fragments, content fragments, then per
  // call one announcing delta (index, id, name, empty arguments) and one
  // with the complete arguments; the final chunk says tool_calls.
  require(concat_field(resp, "reasoning_content") == "Think",
          "reasoning deltas concatenate: " + resp);
  require(concat_field(resp, "content") == "Sure",
          "content deltas concatenate: " + resp);
  const size_t role = resp.find("\"delta\":{\"role\":\"assistant\"");
  const size_t reasoning = resp.find("\"delta\":{\"reasoning_content\":\"");
  const size_t content = resp.find("\"delta\":{\"content\":\"");
  const size_t start0 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_");
  const size_t args0 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":"
      "\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 3}\"}}]}");
  const size_t start1 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_");
  const size_t args1 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":"
      "\"{\\\"city\\\": \\\"Oslo\\\"}\"}}]}");
  const size_t final = resp.find("\"delta\":{},\"logprobs\":null,"
                                 "\"finish_reason\":\"tool_calls\"");
  const size_t done = resp.find("data: [DONE]");
  require(role != std::string::npos && reasoning != std::string::npos &&
              content != std::string::npos && start0 != std::string::npos &&
              args0 != std::string::npos && start1 != std::string::npos &&
              args1 != std::string::npos && final != std::string::npos &&
              done != std::string::npos,
          "every chunk kind present: " + resp);
  require(role < reasoning && reasoning < content && content < start0 &&
              start0 < args0 && args0 < start1 && start1 < args1 &&
              args1 < final && final < done,
          "chunks in arrival order: " + resp);
  require(resp.find("\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                    "\"arguments\":\"\"}}]}") != std::string::npos,
          "the announcing delta carries the name and empty arguments");
  // The two calls carry distinct ids.
  const size_t id0 = resp.find("\"id\":\"call_", start0);
  const size_t id1 = resp.find("\"id\":\"call_", start1);
  require(resp.substr(id0, 27) != resp.substr(id1, 27), "distinct call ids");
}

DGPP_TEST(serve_toolChoice_armsTheGrammarNotThePrompt) {
  // tool_choice required / named / none and parallel_tool_calls false ride
  // the request as a grammar (M6 6g): the prompt is untouched (the model
  // reasons first), the engine is armed with the spec, and the parsed
  // turn carries the reasoning and the call.
  using Mode = dgpp::glm::GrammarSpec::Mode;
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"tools\":{\"available\":true,\"constrained\":true}") !=
                std::string::npos,
            "models advertise constrained decoding");
  }
  const std::string closed_tools =
      ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
      "\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"],"
      "\"additionalProperties\":false}}},{\"type\":\"function\",\"function\":"
      "{\"name\":\"get_time\"}}]";
  rig.engine.script(5, script_of(rig, "Think</think><tool_call>get_weather"
                                      "<arg_key>city</arg_key><arg_value>Rome"
                                      "</arg_value></tool_call>"));
  // required: the prompt keeps its 5 ids, the turn reasons then calls.
  const std::string required = post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"required\""));
  require(required.find("\"prompt_tokens\":5,") != std::string::npos,
          "no forced prefix on the prompt: " + required);
  require(required.find("\"reasoning_content\":\"Think\"") != std::string::npos &&
              required.find("\"name\":\"get_weather\",\"arguments\":"
                            "\"{\\\"city\\\": \\\"Rome\\\"}\"") != std::string::npos &&
              required.find("\"finish_reason\":\"tool_calls\"") != std::string::npos,
          "the turn reasons first, then calls: " + required);
  // named, none, auto+single, required+single: the specs the engine got.
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     closed_tools + ",\"tool_choice\":{\"type\":\"function\","
                                    "\"function\":{\"name\":\"get_time\"}}"));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"none\""));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"parallel_tool_calls\":false"));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"required\","
                                                "\"parallel_tool_calls\":false"));
  (void)post_until_usage(rig, chat_body("abcd", 64, closed_tools));  // auto
  const std::vector<dgpp::glm::GrammarSpec> g = rig.engine.grammars();
  require(g.size() == 6, "six tool requests armed a grammar (auto arms the "
                         "well-formed-call grammar too, M6 6i), got " +
                             std::to_string(g.size()));
  require(g[0].mode == Mode::kRequired && g[0].parallel &&
              g[0].tools.size() == 2 && g[0].tools[0].name == "get_weather" &&
              g[0].tools[0].constrain_keys &&
              g[0].tools[0].keys == std::vector<std::string>{"city", "days"} &&
              g[0].tools[1].name == "get_time" && !g[0].tools[1].constrain_keys,
          "required: the tools with their closed key set");
  // The typed arguments (M6 6i): city a free string, days an integer
  // under the JSON machine.
  using Kind = dgpp::glm::GrammarArg::Kind;
  require(g[0].tools[0].args.size() == 2 && g[0].tools[0].args[0].key == "city" &&
              g[0].tools[0].args[0].kind == Kind::kFree &&
              g[0].tools[0].args[1].key == "days" && g[0].tools[0].args[1].kind == Kind::kJson &&
              g[0].tools[0].args[1].schema.find("integer") != std::string::npos &&
              g[0].tools[1].args.empty(),
          "the typed arguments ride with the tools");
  require(g[1].mode == Mode::kNamed && g[1].named == "get_time", "named");
  require(g[2].mode == Mode::kForbidCalls, "none forbids calls");
  require(rig.frontend.last_globals().find("\"tools\"") != std::string::npos,
          "the last (auto) request rendered the tools");
  require(g[3].mode == Mode::kAuto && !g[3].parallel, "auto + single call");
  require(g[4].mode == Mode::kRequired && !g[4].parallel, "required + single");
  require(g[5].mode == Mode::kAuto && g[5].parallel && g[5].tools.size() == 2,
          "auto: calls at will, every call well-formed");
  // An open schema (no additionalProperties: false) leaves the keys free.
  (void)post_until_usage(
      rig, chat_body("abcd", 64, kWeatherTools + ",\"tool_choice\":\"required\""));
  const std::vector<dgpp::glm::GrammarSpec> g2 = rig.engine.grammars();
  require(g2.size() == 7 && !g2[6].tools[0].constrain_keys,
          "JSON Schema's default is open: keys unconstrained");
  // A strict function whose schema leaves the enforceable subset is a 400
  // naming the keyword path; the same schema without strict is served
  // with that value free.
  const std::string strict_tools =
      ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"strict\":STRICT,"
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"days\":{\"type\":"
      "\"integer\",\"minimum\":0}}}}}]";
  {
    std::string body = strict_tools;
    body.replace(body.find("STRICT"), 6, "true");
    const std::string resp = post_chat(rig, chat_body("abcd", 2, body));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"tools[0].function.parameters.properties.days.minimum\"") !=
                    std::string::npos &&
                resp.find("\"code\":\"unsupported_schema\"") != std::string::npos,
            "strict refuses by keyword path: " + resp.substr(0, 400));
    body = strict_tools;
    body.replace(body.find("STRICT"), 6, "false");
    (void)post_until_usage(rig, chat_body("abcd", 64, body));
    const std::vector<dgpp::glm::GrammarSpec> g3 = rig.engine.grammars();
    require(g3.size() == 8 && g3[7].tools[0].args.size() == 1 &&
                g3[7].tools[0].args[0].kind == Kind::kFree,
            "non-strict: the value stays free");
  }
}

DGPP_TEST(serve_reasoning_foldKnobAndUnterminatedCallAtTheCap) {
  // The fold knob: reasoning rides as content with the model's own
  // </think>, no reasoning_content field.
  ServiceRig fold(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                  /*can_sample=*/false, std::nullopt, /*with_markers=*/true,
                  /*reasoning_in_content=*/true);
  fold.engine.script(5, script_of(fold, "Think</think>Sure"));
  const std::string folded = post_until_usage(fold, chat_body("abcd", 64));
  require(folded.find("\"content\":\"Think</think>Sure\"") != std::string::npos &&
              folded.find("reasoning_content") == std::string::npos &&
              folded.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "folded: " + folded);
  {
    Client models(fold.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find("\"reasoning\":{\"in_content\":true}") !=
                std::string::npos,
            "models report the fold");
  }
  // Streamed, the fold's </think> is a content delta in place.
  {
    Client c(fold.port());
    const std::string body = chat_body("abcd", 64, ",\"stream\":true");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    require(concat_field(resp, "content") == "Think</think>Sure" &&
                resp.find("reasoning_content") == std::string::npos,
            "folded stream: " + resp);
  }

  // The steps cap inside a block: the literal text is content,
  // finish_reason length, no tool_calls.
  ServiceRig cap(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  cap.engine.script(5, script_of(cap, "</think><tool_call>get_weather<arg_key>"
                                      "city</arg_key><arg_value>Paris"));
  const std::string capped =
      post_until_usage(cap, chat_body("abcd", 6, kWeatherTools));
  require(capped.find("\"content\":\"<tool_call>get_\"") != std::string::npos &&
              capped.find("tool_calls") == std::string::npos &&
              capped.find("\"finish_reason\":\"length\"") != std::string::npos &&
              capped.find("\"completion_tokens\":6,") != std::string::npos,
          "capped block: " + capped);
}

DGPP_TEST(serve_responseFormat_armsTheJsonGrammar) {
  // response_format (M6 6h): json_object and json_schema ride the request
  // as the JSON grammar — the prompt untouched, the engine armed with the
  // schema text — and the content is the JSON the model produced.
  using Mode = dgpp::glm::GrammarSpec::Mode;
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"response_format\":{\"json_object\":true,\"json_schema\":true}") !=
                std::string::npos,
            "models advertise response_format");
  }
  rig.engine.script(5, script_of(rig, "Think</think>{\"city\": \"Rome\"}"));
  const std::string json_object = post_until_usage(
      rig, chat_body("abcd", 64, ",\"response_format\":{\"type\":\"json_object\"}"));
  require(json_object.find("\"prompt_tokens\":5,") != std::string::npos &&
              json_object.find("\"reasoning_content\":\"Think\"") != std::string::npos &&
              json_object.find("\"content\":\"{\\\"city\\\": \\\"Rome\\\"}\"") !=
                  std::string::npos &&
              json_object.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "json_object: reasoning then the JSON as content: " + json_object);
  const std::string schema =
      "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
      "\"required\":[\"city\"],\"additionalProperties\":false}";
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                     "{\"name\":\"place\",\"strict\":true,\"schema\":" + schema + "}}"));
  // Not strict and outside the subset: served as json_object (a warning).
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                     "{\"name\":\"p\",\"schema\":{\"type\":\"string\",\"pattern\":\"^a\"}}}"));
  // type text: no grammar.
  (void)post_until_usage(rig, chat_body("abcd", 64, ",\"response_format\":{\"type\":\"text\"}"));
  const std::vector<dgpp::glm::GrammarSpec> g = rig.engine.grammars();
  require(g.size() == 3, "three JSON requests armed a grammar (text arms none), got " +
                             std::to_string(g.size()));
  require(g[0].mode == Mode::kJson && g[0].json_schema.empty() && g[0].tools.empty(),
          "json_object: the free JSON grammar");
  require(g[1].mode == Mode::kJson && g[1].json_schema.find("\"city\"") != std::string::npos &&
              g[1].json_schema.find("additionalProperties") != std::string::npos,
          "json_schema: the schema text rides: " + g[1].json_schema);
  require(g[2].mode == Mode::kJson && g[2].json_schema.empty(),
          "non-strict unsupported schema falls back to json_object");
  require(rig.frontend.last_globals().find("response_format") == std::string::npos,
          "the prompt does not carry the format");

  // The refusals: strict + unsupported keyword names the keyword; tools
  // and JSON together; a bad type; a missing name.
  const auto refused = [&](const std::string& extra, const std::string& param,
                           const std::string& code) {
    const std::string resp = post_chat(rig, chat_body("abcd", 2, extra));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"" + param + "\"") != std::string::npos &&
                (code.empty() || resp.find("\"code\":\"" + code + "\"") != std::string::npos),
            "expected a 400 naming " + param + " / " + code + ": " + resp.substr(0, 400));
  };
  refused(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\","
          "\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"city\":"
          "{\"type\":\"string\",\"pattern\":\"^a\"}}}}}",
          "response_format.json_schema.schema.properties.city.pattern", "unsupported_schema");
  refused(kWeatherTools + ",\"response_format\":{\"type\":\"json_object\"}",
          "response_format", "unsupported_parameter");
  refused(",\"response_format\":{\"type\":\"yaml\"}", "response_format.type", "");
  refused(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{}}}",
          "response_format.json_schema.name", "");
  refused(",\"response_format\":\"json_object\"", "response_format", "");
  // A greedy engine has no masks: JSON modes are refused, text is fine.
  ServiceRig greedy(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                    /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  {
    const std::string resp = post_chat(
        greedy, chat_body("abcd", 2, ",\"response_format\":{\"type\":\"json_object\"}"));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"code\":\"constrained_decoding_unsupported\"") != std::string::npos,
            "greedy engine refuses json_object: " + resp.substr(0, 300));
    Client models(greedy.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"response_format\":{\"json_object\":false,\"json_schema\":false}") !=
                std::string::npos,
            "models report the absence");
  }
}

DGPP_TEST(serve_shutdown_drainsInFlightWorkWithTheShutdownError) {
  // Drain-on-stop (M6 6c): a stream mid-generation is retired by the drain
  // pass and answered with the server_shutdown error event and [DONE]
  // after the tokens it produced (no finish chunk); a request still in
  // the admission queue is shed with a 503 server_shutdown; a request
  // arriving after the door closed gets the same 503; the metrics count
  // the interruption as a cancellation; drained() turns true only once
  // every answer is out.
  ServiceRig rig(/*queue_limit=*/8);
  rig.engine.script(5, script_of(rig, std::string(300, 'x')));
  rig.pass_delay_ms = 2;  // ~2 ms per token: the pump sees it mid-generation
  // A stream: let it produce a few chunks, then hold the engine between
  // passes — the request is live in the scheduler, mid-generation.
  Client stream(rig.port());
  {
    const std::string body = chat_body("abcd", 300, ",\"stream\":true");
    stream.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                    "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  std::string head = stream.read_until("\"content\":\"x", 3000);
  require(head.find("\"content\":\"x") != std::string::npos,
          "the stream produced content before the stop: " + head.substr(0, 300));
  rig.gate = true;
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  // A one-shot that lands in the admission queue while the engine is held.
  Client queued(rig.port());
  {
    const std::string body = chat_body("efgh", 8);
    queued.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                    "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  require(!rig.service.drained(), "not drained while work is live");
  const int interrupted = rig.drain(/*stop_http=*/false);
  require(interrupted == 1, "one live request interrupted, got " + std::to_string(interrupted));
  const std::string tail = head + stream.read_until("[DONE]", 3000);
  require(tail.find("\"code\":\"server_shutdown\"") != std::string::npos &&
              tail.find("this response is incomplete") != std::string::npos &&
              tail.find("data: [DONE]") != std::string::npos,
          "the interrupted stream ends with the shutdown error event: " + tail.substr(tail.size() > 600 ? tail.size() - 600 : 0));
  require(tail.find("\"finish_reason\":\"stop\"") == std::string::npos &&
              tail.find("\"finish_reason\":\"length\"") == std::string::npos,
          "no finish chunk on an interrupted stream");
  const std::string shed = queued.read_until("}}", 3000);
  require(shed.find("503 ") != std::string::npos &&
              shed.find("\"code\":\"server_shutdown\"") != std::string::npos &&
              shed.find("retry on another instance") != std::string::npos,
          "the queued one-shot is shed with 503 server_shutdown: " + shed.substr(0, 400));
  // The door is closed: a new request gets the same 503.
  const std::string late = post_chat(rig, chat_body("ijkl", 4));
  require(late.find("503 ") != std::string::npos &&
              late.find("\"code\":\"server_shutdown\"") != std::string::npos,
          "a request after the stop is refused: " + late.substr(0, 300));
  require(rig.service.drained(), "drained once every answer is out");
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("requests_cancelled", 2000);
    require(metrics.find("\"requests_cancelled\":1") != std::string::npos &&
                metrics.find("\"requests_shed\":2") != std::string::npos,
            "metrics: one interruption, two sheds: " + metrics.substr(0, 400));
  }
}

DGPP_TEST(serve_admission_growPolicyShedsTheYoungestWithFinishLength) {
  // Grow-on-demand (M6 6d) on the service: two 300-token requests whose
  // lifetimes (77 blocks each) exceed the 100-block pool together are both
  // admitted under grow (window 8), grow at tick top, and when the pool is
  // spent the younger is cut short with finish_reason "length" while the
  // older completes; the metrics carry the policy, the growth count and
  // the shed. The default rig reports the full-reserve policy.
  dgpp::glm::AdmissionPolicy grow;
  grow.mode = dgpp::glm::AdmissionPolicy::Mode::kGrowOnDemand;
  grow.window_tokens = 8;
  ServiceRig rig(/*queue_limit=*/8, dgpp::glm_sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/false,
                 /*reasoning_in_content=*/false, grow);
  rig.engine.script(5, script_of(rig, std::string(300, 'x')));
  Client a(rig.port()), b(rig.port());
  for (Client* c : {&a, &b}) {
    const std::string body = chat_body("abcd", 300);
    c->send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                "Content-Type: application/json\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  const std::string ra = a.read_until("\"usage\"", 10000);
  const std::string rb = b.read_until("\"usage\"", 10000);
  const auto completion_tokens = [](const std::string& r) {
    const size_t at = r.find("\"completion_tokens\":");
    return at == std::string::npos ? -1 : std::atoi(r.c_str() + at + 20);
  };
  const int ta = completion_tokens(ra), tb = completion_tokens(rb);
  require(ra.find("\"finish_reason\":\"length\"") != std::string::npos &&
              rb.find("\"finish_reason\":\"length\"") != std::string::npos,
          "both finish length: " + ra.substr(0, 200) + " / " + rb.substr(0, 200));
  require((ta == 300 && tb > 0 && tb < 300) || (tb == 300 && ta > 0 && ta < 300),
          "one completes 300 tokens, the other is cut short: " + std::to_string(ta) +
              " / " + std::to_string(tb));
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("\"admission\"", 2000);
    require(metrics.find("\"requests_shed_pool\":1") != std::string::npos &&
                metrics.find("\"admission\":{\"mode\":\"grow\",\"window\":8}") !=
                    std::string::npos &&
                metrics.find("\"reservations_grown\":0") == std::string::npos,
            "metrics: the shed, the policy, the growth: " + metrics.substr(0, 500));
  }
  ServiceRig plain(/*queue_limit=*/8);
  Client m(plain.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  require(m.read_until("\"admission\"", 2000).find(
              "\"admission\":{\"mode\":\"full\",\"window\":256}") != std::string::npos,
          "the default policy is full-reserve");
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}
