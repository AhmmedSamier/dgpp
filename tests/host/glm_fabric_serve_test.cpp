// M6 Stage 4b gate: the admission journal (the metronome), both loops,
// and the §11 identity — over REAL localhost TCP and REAL HTTP, with
// only the engine faked. What the 4a gate did for the OpenAI contract,
// this does for the fabric seam:
//   * rank-0 side: HttpServer + GenerationService (both real) over a
//     deterministic FakeEngine, the engine loop driving engine_pass
//     with the journal hook — exactly the app's loop;
//   * peer side (two of them — the star broadcast): a real Scheduler
//     over its own FakeEngine (same token functions), driven by
//     run_journal_peer;
//   * the §11 identity, live: every peer's op stream must equal
//     rank 0's audit stream after every scenario, including a
//     mid-stream client disconnect (the cancel must cross the journal
//     and retire on every rank at the same tick);
//   * the stop discipline: the stop record releases the peers, every
//     thread joins, nobody errored.
// The fakes are the 4a gate's (same token laws), so the shapes this
// test skims are already byte-pinned there — here the JOURNAL is the
// thing under test.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/log.hpp"
#include "common/test.hpp"
#include "models/glm_scheduler.hpp"
#include "service/fabric_serve.hpp"
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

// The 4a gate's deterministic fake model: token i of a request whose
// rendered prompt is P bytes is ((len*31 + i*7) % 250) + 1 — never 0,
// never EOS; prompt len % 4 == 3 answers EOS as its SECOND token,
// len % 4 == 2 answers EOS on the PREFILL pick.
int32_t fake_token(size_t prompt_len, int index) {
  return static_cast<int32_t>((prompt_len * 31 +
                               static_cast<size_t>(index) * 7) % 250) + 1;
}
bool fake_eos_second(size_t prompt_len) { return prompt_len % 4 == 3; }
bool fake_eos_prefill(size_t prompt_len) { return prompt_len % 4 == 2; }

class FakeEngine : public SchedulerEngine {
 public:
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens)
      : slots_(slots), total_blocks_(total_blocks),
        block_tokens_(block_tokens) {}

  // Scenario knob: a per-op sleep that keeps a request in flight long
  // enough for the disconnect test to pull the plug mid-generation
  // (the instant default retires everything before any client could).
  void set_op_delay_ms(int ms) { op_delay_ms_ = ms; }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) (void)slot, sum += live.held_blocks;
    return sum;
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    if (int d = op_delay_ms_.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(d));
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
    if (int d = op_delay_ms_.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(d));
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
  std::atomic<int> op_delay_ms_{0};
  std::map<int, Live> live_;
};

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
  std::string render_chat(const dgpp::minijson::Value& globals) const override {
    std::string out;
    for (const auto& msg : globals.at("messages").items())
      if (const auto* content = msg.find("content"))
        out.append(content->as_string());
    return out;
  }
};

// --- the raw-socket client (the 4a gate's) -----------------------------
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

// --- the codec micro-gate ------------------------------------------------

void test_journal_codec() {
  GenerationService::PassEvents events;
  dgpp::glm::SchedulerRequest submit;
  submit.id = "chatcmpl-00000000000000ff";
  submit.prompt = {1, 2, 300, 154820 - 1};
  submit.max_steps = 16;
  submit.cancel_after = 3;
  events.submits.push_back(submit);
  events.cancels.push_back("chatcmpl-dead");

  const dgpp::service::JournalRecord back = dgpp::service::decode_journal_line(
      dgpp::service::encode_journal_tick(events));
  require(!back.stop, "codec: tick decoded as stop");
  require(back.submits.size() == 1 && back.cancels.size() == 1,
          "codec: wrong event counts");
  require(back.submits[0].id == submit.id, "codec: id round-trip");
  require(back.submits[0].prompt == submit.prompt, "codec: prompt round-trip");
  require(back.submits[0].max_steps == 16, "codec: max_steps round-trip");
  require(back.submits[0].cancel_after == 3, "codec: cancel_after round-trip");
  require(back.cancels[0] == "chatcmpl-dead", "codec: cancel round-trip");
  require(!back.submits[0].grammar.active(), "codec: no grammar unless sent");

  // The grammar (M6 6g) rides with the request: mode, parallel flag, the
  // named function, the tools with and without closed key sets.
  {
    GenerationService::PassEvents ev;
    dgpp::glm::SchedulerRequest s = submit;
    s.grammar.mode = dgpp::glm::GrammarSpec::Mode::kNamed;
    s.grammar.parallel = false;
    s.grammar.named = "get_weather";
    s.grammar.tools.push_back(
        dgpp::glm::GrammarTool{"get_weather", true, {"city", "days"}});
    s.grammar.tools.push_back(dgpp::glm::GrammarTool{"get_time", false, {}});
    s.grammar.tools.push_back(dgpp::glm::GrammarTool{"ping", true, {}});
    ev.submits.push_back(s);
    const dgpp::service::JournalRecord got = dgpp::service::decode_journal_line(
        dgpp::service::encode_journal_tick(ev));
    require(got.submits.size() == 1 && got.submits[0].grammar == s.grammar,
            "codec: grammar round-trip");
    require(got.submits[0].grammar.tools[2].constrain_keys &&
                got.submits[0].grammar.tools[2].keys.empty(),
            "codec: a closed empty key set survives");
  }

  require(dgpp::service::decode_journal_line(
              dgpp::service::encode_journal_stop()).stop,
          "codec: stop round-trip");
  {
    const dgpp::service::JournalRecord warm =
        dgpp::service::decode_journal_line(
            dgpp::service::encode_journal_warm());
    require(warm.warm && !warm.stop && warm.submits.empty() &&
                warm.cancels.empty(),
            "codec: warm round-trip");
    require(!back.warm, "codec: tick decoded as warm");
  }

  bool threw = false;
  try {
    (void)dgpp::service::decode_journal_line("{\"op\":\"nonsense\"}");
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "codec: unknown op must throw");

  // The sampling spec (M6 6b): a greedy submit's record carries no spec
  // (byte-identical to the pre-sampling format); a stochastic submit's
  // spec and seed round-trip BITWISE; a corrupt spec is refused.
  {
    const std::string greedy_line =
        dgpp::service::encode_journal_tick(events);
    require(greedy_line.find("\"g\"") == std::string::npos,
            "codec: greedy submit must not carry a sampling spec");
    GenerationService::PassEvents stochastic;
    dgpp::glm::SchedulerRequest s = submit;
    s.sampling.temperature = 0.7f;
    s.sampling.top_p = 0.95f;
    s.sampling.top_k = 40;
    s.sampling.min_p = 0.0125f;
    s.sampling.repetition_penalty = 1.1f;
    s.sampling.frequency_penalty = -0.3f;
    s.sampling.presence_penalty = 1.0e-7f;
    s.sampling.logprobs = 3;
    s.seed = 0xfedcba9876543210ull;
    stochastic.submits.push_back(s);
    const dgpp::service::JournalRecord got = dgpp::service::decode_journal_line(
        dgpp::service::encode_journal_tick(stochastic));
    require(got.submits.size() == 1, "codec: stochastic submit count");
    const dgpp::glm::SchedulerRequest& b = got.submits[0];
    const auto bits_equal = [](float x, float y) {
      return std::memcmp(&x, &y, sizeof(float)) == 0;
    };
    require(bits_equal(b.sampling.temperature, 0.7f) &&
                bits_equal(b.sampling.top_p, 0.95f) &&
                b.sampling.top_k == 40 &&
                bits_equal(b.sampling.min_p, 0.0125f) &&
                bits_equal(b.sampling.repetition_penalty, 1.1f) &&
                bits_equal(b.sampling.frequency_penalty, -0.3f) &&
                bits_equal(b.sampling.presence_penalty, 1.0e-7f) &&
                b.sampling.logprobs == 3 && b.seed == 0xfedcba9876543210ull,
            "codec: sampling spec must round-trip bitwise");
    bool bad = false;
    try {
      // top_p bits of 2.0f: a spec no rank may apply.
      (void)dgpp::service::decode_journal_line(
          "{\"op\":\"tick\",\"s\":[{\"id\":\"x\",\"p\":[1],\"m\":2,"
          "\"g\":{\"t\":1065353216,\"p\":1073741824,\"k\":0,\"m\":0,"
          "\"r\":1065353216,\"f\":0,\"q\":0,\"l\":0,"
          "\"s\":\"0000000000000001\"}}]}");
    } catch (const std::exception&) {
      bad = true;
    }
    require(bad, "codec: an invalid sampling spec must throw");
    // logprobs ride as "lp" with the spec (a greedy request that asks
    // carries the spec too, at temperature 0).
    GenerationService::PassEvents asking;
    dgpp::glm::SchedulerRequest g = submit;
    g.logprobs = 3;
    g.sampling.logprobs = 3;
    asking.submits.push_back(g);
    const std::string line = dgpp::service::encode_journal_tick(asking);
    require(line.find("\"lp\":3") != std::string::npos &&
                line.find("\"g\":{") != std::string::npos,
            "codec: logprobs field and spec on a greedy request");
    const dgpp::service::JournalRecord back2 =
        dgpp::service::decode_journal_line(line);
    require(back2.submits.size() == 1 && back2.submits[0].logprobs == 3 &&
                back2.submits[0].sampling.logprobs == 3 &&
                back2.submits[0].sampling.temperature == 0.0f,
            "codec: logprobs round-trip");
  }
  std::puts("ok 1 - journal codec round-trip");
}

// --- the fabric rig: rank 0's real stack + two real peer loops ----------

struct PeerRig {
  static constexpr int kSlots = 4;
  static constexpr int kQueue = 8;  // rank 0's ServiceConfig limit —
                                    // the identity is constructive
  FakeEngine engine{kSlots, 100, 4};
  dgpp::glm::Scheduler sched;
  dgpp::service::OpStreamObserver oplog;
  std::unique_ptr<dgpp::service::JournalReader> reader;
  std::thread thread;
  std::string error;  // empty = the peer never complained
  std::atomic<bool> warmed{false};  // held at and released by the warm record

  explicit PeerRig(uint16_t journal_port, const std::atomic<bool>& stop_flag)
      : sched(&engine, {kFakeEos}, kQueue) {
    sched.set_observer(&oplog);
    thread = std::thread([this, journal_port, &stop_flag] {
      try {
        // Connect BEFORE the loop: rank 0's accept_peers is waiting
        // for the full world, and it must not wait on a reader that
        // only connects after its first record.
        reader = std::make_unique<dgpp::service::JournalReader>(
            "127.0.0.1", journal_port, 5000);
        // The production peer holds here for the graph engine's warm
        // capture start signal; the rig holds the same way so the first
        // record's order (warm, then ticks) is pinned end to end.
        if (!dgpp::service::wait_journal_warm(
                reader.get(), [&stop_flag] { return stop_flag.load(); }))
          return;
        warmed.store(true);
        dgpp::service::run_journal_peer(
            &sched, reader.get(), [&stop_flag] { return stop_flag.load(); });
      } catch (const std::exception& e) {
        error = e.what();
      }
    });
  }
  ~PeerRig() {
    if (thread.joinable()) thread.join();
  }
  PeerRig(const PeerRig&) = delete;
  PeerRig& operator=(const PeerRig&) = delete;
};

struct FabricRig {
  static constexpr int kWorld = 3;  // rank 0 + two peers (the star)
  FakeEngine engine{4, 100, 4};
  FakeFrontend frontend;
  ServiceConfig cfg;
  GenerationService service;
  HttpServer http;
  dgpp::service::OpStreamObserver oplog;  // rank 0's audit leg
  dgpp::service::JournalWriter journal{0};
  std::vector<std::unique_ptr<PeerRig>> peers;
  std::thread http_loop;
  std::thread engine_loop;
  std::atomic<bool> stopping{false};

  FabricRig()
      : cfg([] {
          ServiceConfig c;
          c.model_id = "glm-5.3-flash-fp8";
          c.default_max_tokens = 8;
          c.queue_limit = 8;
          return c;
        }()),
        service(cfg, &engine, &frontend, {kFakeEos}),
        http(0, &service, 64) {
    service.set_audit_observer(&oplog);
    // Peers connect first (they block in the journal read loop),
    // then rank 0 accepts the full world, then HTTP + the engine.
    // Nothing broadcasts before accept_peers returns.
    for (int r = 1; r < kWorld; ++r)
      peers.push_back(std::make_unique<PeerRig>(journal.port(), stopping));
    journal.accept_peers(kWorld, 5000);
    // The warm record precedes every tick (glm_serve broadcasts it
    // before its warm capture); the peers are holding for it.
    journal.broadcast(dgpp::service::encode_journal_warm());
    http_loop = std::thread([this] { http.serve(); });
    engine_loop = std::thread([this] {
      while (!stopping.load()) {
        const bool progressed = service.engine_pass(
            [this](const GenerationService::PassEvents& events) {
              journal.broadcast(dgpp::service::encode_journal_tick(events));
            });
        if (!progressed)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      service.begin_shutdown();
      try {
        journal.broadcast(dgpp::service::encode_journal_stop());
      } catch (const std::exception& e) {
        DGPP_LOG_ERROR("rig: stop broadcast failed: {}", e.what());
      }
    });
  }

  void stop() {
    if (stopping.exchange(true)) return;
    http.stop();
    if (http_loop.joinable()) http_loop.join();
    if (engine_loop.joinable()) engine_loop.join();  // broadcasts stop
    for (auto& p : peers)
      if (p->thread.joinable()) p->thread.join();
  }
  ~FabricRig() { stop(); }
  FabricRig(const FabricRig&) = delete;
  FabricRig& operator=(const FabricRig&) = delete;

  uint16_t port() const { return http.port(); }

  // The §11 identity: every peer's op stream must equal rank 0's.
  bool oplogs_agree() const {
    const std::string rank0 = oplog.text();
    for (const auto& p : peers)
      if (p->oplog.text() != rank0) return false;
    return true;
  }
  std::string oplog_diff() const {
    std::string out = "rank0:\n" + oplog.text();
    for (size_t i = 0; i < peers.size(); ++i)
      out += "\npeer " + std::to_string(i + 1) + ":\n" +
             peers[i]->oplog.text();
    return out;
  }
  void require_oplogs_agree(const std::string& what) {
    for (int i = 0; i < 2000 && !oplogs_agree(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(oplogs_agree(), "op streams diverged after " + what + ":\n" +
                                oplog_diff());
    for (size_t i = 0; i < peers.size(); ++i) {
      require(peers[i]->error.empty(), "peer " + std::to_string(i + 1) +
                                           " errored: " + peers[i]->error);
      require(peers[i]->warmed.load(),
              "peer " + std::to_string(i + 1) + " never saw the warm record");
    }
  }
};

std::string post_request(const std::string& path, const std::string& json) {
  return "POST " + path + " HTTP/1.1\r\nHost: t\r\n"
         "Content-Type: application/json\r\nContent-Length: " +
         std::to_string(json.size()) + "\r\n\r\n" + json;
}

// The concatenated content across every SSE delta in `raw` — deltas
// may coalesce per chunk or not; the client contract is concatenation
// (the fake's tokens are printable bytes, no JSON escaping involved).
std::string concat_content_deltas(const std::string& raw) {
  std::string out;
  size_t at = 0;
  while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
    at += 11;
    const size_t end = raw.find('"', at);
    if (end == std::string::npos) break;
    out.append(raw, at, end - at);
    at = end;
  }
  return out;
}

// Polls /v1/metrics until `needle` appears (the counters move on the
// engine thread; give them a beat).
std::string wait_metrics(FabricRig& rig, const std::string& needle) {
  for (int i = 0; i < 1000; ++i) {
    Client c(rig.port());
    c.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string raw = c.read_until("}", 2000);
    if (raw.find(needle) != std::string::npos) return raw;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  throw std::runtime_error("metrics never showed " + needle);
}

// --- scenario 1: non-stream completion + the §11 identity ----------------

void test_non_stream_and_identity(FabricRig& rig) {
  // "Hello world, tests!" — 19 characters exactly (13 + 5 + 1; counted
  // twice, trust it): 19 % 4 == 3 → EOS on the SECOND token (finish
  // "stop", completion_tokens 2), and the first token is
  // fake_token(19,0) = 90 = 'Z' — printable, JSON-escape-free, so the
  // raw-body substring assert below is byte-exact.
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"Hello world, tests!"})"
                          R"(],"max_tokens":8})"));
  const std::string raw =
      c.read_until("\"total_tokens\":", 5000);
  require(raw.find("200") != std::string::npos,
          "non-stream: not 200: [" + raw + "]");
  require(raw.find("\"object\":\"chat.completion\"") != std::string::npos,
          "non-stream: wrong object");
  const std::string expect_text(1, static_cast<char>(fake_token(19, 0)));
  require(raw.find("\"content\":\"" + expect_text + "\"") != std::string::npos,
          "non-stream: content mismatch: " + raw);
  require(raw.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "non-stream: EOS should finish stop");
  require(raw.find("\"completion_tokens\":2") != std::string::npos,
          "non-stream: usage mismatch: " + raw);
  rig.require_oplogs_agree("non-stream");
  std::puts("ok 2 - non-stream completion + op-stream identity");
}

// --- scenario 2: streaming lifecycle over the journal --------------------

void test_stream_lifecycle(FabricRig& rig) {
  // A 1-char prompt: the fake's tokens are ((31 + 7i) % 250) + 1 =
  // 32,39,46,53,60,67 — all printable — and len % 4 == 1 → no EOS:
  // the steps cap ends it (finish "length"), 6 tokens with
  // max_tokens 6.
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"x"})"
                          R"(],"max_tokens":6,"stream":true,)"
                          R"("stream_options":{"include_usage":true}})"));
  const std::string raw = c.read_until("data: [DONE]", 5000);
  require(raw.find("200") != std::string::npos, "stream: not 200");
  const size_t role_at = raw.find("\"role\":\"assistant\"");
  const size_t first_content = raw.find("\"content\":\"");
  require(role_at != std::string::npos && first_content != std::string::npos &&
              role_at < first_content,
          "stream: role chunk must lead");
  require(raw.find("\"finish_reason\":\"length\"") != std::string::npos,
          "stream: steps cap should finish length");
  require(raw.find("\"choices\":[],\"usage\":") != std::string::npos,
          "stream: usage chunk missing");
  require(raw.find("data: [DONE]") != std::string::npos, "stream: no DONE");
  std::string expect;
  for (int i = 0; i < 6; ++i)
    expect.push_back(static_cast<char>(fake_token(1, i)));
  require(concat_content_deltas(raw) == expect,
          "stream: concatenated deltas '" + concat_content_deltas(raw) +
              "' != '" + expect + "'");
  rig.require_oplogs_agree("stream");
  std::puts("ok 3 - streaming lifecycle over the journal");
}

// --- scenario 3: disconnect-cancel crosses the journal --------------------

void test_disconnect_cancel(FabricRig& rig) {
  // 10ms per op keeps the request alive ~640ms — the client pulls the
  // plug ~20ms in, comfortably mid-generation. "abcdefgh" (8 bytes,
  // 8 % 4 == 0 → no EOS) with a 64-token budget.
  rig.engine.set_op_delay_ms(10);
  Client c(rig.port());
  c.send_all(post_request("/v1/chat/completions",
                          R"({"model":"glm-5.3-flash-fp8","messages":[)"
                          R"({"role":"user","content":"abcdefgh"})"
                          R"(],"max_tokens":64,"stream":true})"));
  const std::string raw = c.read_until("\"id\":\"", 5000);
  const size_t id_at = raw.find("\"id\":\"chatcmpl-");
  require(id_at != std::string::npos, "disconnect: no id in first chunk");
  const size_t id_begin = id_at + 6;
  const size_t id_end = raw.find('"', id_begin);
  const std::string id = raw.substr(id_begin, id_end - id_begin);
  require(!id.empty(), "disconnect: empty id");
  // Wait for the first CONTENT chunk (the request is live on every
  // rank), then pull the plug.
  const std::string live = c.read_until("\"content\":\"", 5000);
  require(live.find("\"content\":\"") != std::string::npos,
          "disconnect: never went live");
  c.hard_close();

  wait_metrics(rig, "\"requests_cancelled\":1");
  // The retire line for this id must appear on rank 0 AND the peers —
  // with steps well under the 64 cap (cancelled, not capped; EOS is
  // impossible for this prompt).
  for (int i = 0; i < 1000; ++i) {
    const std::string rank0 = rig.oplog.text();
    if (rank0.find("R " + id + " ") != std::string::npos &&
        rig.oplogs_agree())
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const std::string rank0 = rig.oplog.text();
  const size_t retire_at = rank0.find("R " + id + " ");
  require(retire_at != std::string::npos,
          "disconnect: no retire line for " + id);
  const size_t steps_at = rank0.find(' ', rank0.find(' ', retire_at + 2) + 1);
  const int retired_steps =
      std::atoi(rank0.c_str() + steps_at + 1);
  require(retired_steps < 64, "disconnect: retired at the cap, not the cancel");
  rig.require_oplogs_agree("disconnect-cancel");
  std::puts("ok 4 - disconnect-cancel crossed the journal on every rank");
}

}  // namespace

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  try {
    test_journal_codec();
    {
      FabricRig rig;
      test_non_stream_and_identity(rig);
      test_stream_lifecycle(rig);
      test_disconnect_cancel(rig);
      // The stop discipline: the stop record releases the peers (a
      // hang here joins forever and the gate times out), and nobody
      // errored on the way down.
      rig.stop();
      for (size_t i = 0; i < rig.peers.size(); ++i)
        require(rig.peers[i]->error.empty(),
                "peer " + std::to_string(i + 1) + " errored: " +
                    rig.peers[i]->error);
      rig.require_oplogs_agree("stop");
    }
    std::puts("ok 5 - stop discipline: peers released, no peer errors");
    std::puts("glm_fabric_serve_test: ALL PASS");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
