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

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    if (live_.count(req) != 0)
      throw std::runtime_error("fake: prefill on live slot");
    Live live;
    live.prompt_len = prompt.size();
    live.served = 1;
    live.last_token = fake_eos_prefill(live.prompt_len)
                          ? kFakeEos
                          : fake_token(live.prompt_len, 0);
    live_[req] = live;
    return live.last_token;
  }

  void reserve(int req, int64_t tokens) override {
    Live& live = live_.at(req);
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    Live& live = live_.at(req);
    live.last_token =
        fake_eos_second(live.prompt_len) && live.served == 1
            ? kFakeEos
            : fake_token(live.prompt_len, static_cast<int>(live.served));
    ++live.served;
    return {live.last_token};
  }

  void close(int req) override { live_.erase(req); }

 private:
  struct Live {
    size_t prompt_len = 0;
    int64_t held_blocks = 0;
    int served = 0;
    int32_t last_token = -1;
  };
  int slots_;
  int64_t total_blocks_;
  int64_t block_tokens_;
  bool can_sample_ = false;
  std::map<int, Live> live_;
  mutable std::mutex armed_mu_;
  std::vector<Armed> armed_;
};

// The fake frontend: bytes ↔ ids, and a deterministic chat render (the
// concatenation of every message's content — the test computes prompt
// lengths from the same rule).
class FakeFrontend : public ModelFrontend {
 public:
  std::vector<int64_t> encode_text(std::string_view text) const override {
    std::vector<int64_t> ids;
    for (const char c : text) ids.push_back(static_cast<unsigned char>(c));
    return ids;
  }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    std::string out;
    for (int64_t id : ids)
      if (id != kFakeEos) out.push_back(static_cast<char>(id));
    return out;
  }
  std::string render_chat(const dgpp::minijson::Value& messages) const override {
    std::string out;
    for (const auto& msg : messages.items())
      if (const auto* content = msg.find("content"))
        out.append(content->as_string());
    return out;
  }
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
  std::atomic<bool> stopping{false};

  // `sampling_defaults`: the served defaults (greedy unless a test hands
  // the checkpoint's stochastic ones); `can_sample`: whether the fake
  // engine advertises the sampler.
  explicit ServiceRig(int queue_limit = 8,
                      dgpp::glm_sample::Params sampling_defaults =
                          dgpp::glm_sample::greedy_params(),
                      bool can_sample = false,
                      std::optional<uint64_t> fixed_seed = std::nullopt)
      : engine(kSlots, /*total_blocks=*/100, /*block_tokens=*/4, can_sample),
        cfg([&] {
          ServiceConfig c;
          c.model_id = "glm-5.3-flash-fp8";
          c.default_max_tokens = 8;
          c.queue_limit = queue_limit;
          c.sampling_defaults = sampling_defaults;
          c.fixed_seed = fixed_seed;
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
      }
    });
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
                  400, "\"param\":\"tools\"");
  post_and_expect(chat_body("abcd", 3, ",\"n\":2"), 400, "\"param\":\"n\"");
  post_and_expect(
      "{\"model\":\"wrong-model\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}]}",
      404, "\"code\":\"model_not_found\"");
  post_and_expect("{not json", 400, "invalid JSON body");
  post_and_expect(
      "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
      "\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]}",
      400, "\"param\":\"messages.content\"");
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

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}
