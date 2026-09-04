#include "service/fabric_serve.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

#include "common/log.hpp"
#include "loaders/minijson.hpp"
#include "service/json_out.hpp"

namespace dgpp::service {

using dgpp::glm::Scheduler;
using dgpp::glm::SchedulerRequest;

// ---- the audit tap --------------------------------------------------------

void OpStreamObserver::on_token(const std::string& id, int64_t token,
                                int steps_done) {
  const std::lock_guard<std::mutex> lock(mutex_);
  text_ += "T " + id + " " + std::to_string(token) + " " +
           std::to_string(steps_done) + "\n";
}

void OpStreamObserver::on_retire(const std::string& id,
                                const Scheduler::Result& result) {
  const std::lock_guard<std::mutex> lock(mutex_);
  // The reason rides as its enum ordinal — same binary family, same
  // values, byte-comparable across ranks.
  text_ += "R " + id + " " + std::to_string(static_cast<int>(result.reason)) +
           " " + std::to_string(result.steps_done) + "\n";
}

std::string OpStreamObserver::text() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return text_;
}

// ---- wire codec -----------------------------------------------------------

namespace {

// Floats ride as their IEEE bit patterns: the peers must apply the EXACT
// spec rank 0 applied, and a decimal round trip is one more place for a
// rank to differ by an ulp.
uint32_t float_bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
float float_from_bits(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
std::string hex64(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

}  // namespace

std::string encode_journal_tick(const GenerationService::PassEvents& events) {
  std::string out = "{\"op\":\"tick\"";
  if (!events.submits.empty()) {
    out += ",\"s\":[";
    for (size_t i = 0; i < events.submits.size(); ++i) {
      if (i != 0) out.push_back(',');
      const SchedulerRequest& r = events.submits[i];
      out += "{\"id\":";
      append_json_string(&out, r.id);
      out += ",\"p\":[";
      for (size_t j = 0; j < r.prompt.size(); ++j) {
        if (j != 0) out.push_back(',');
        append_json_int(&out, r.prompt[j]);
      }
      out += "],\"m\":";
      append_json_int(&out, r.max_steps);
      if (r.cancel_after > 0) {
        out += ",\"ca\":";
        append_json_int(&out, r.cancel_after);
      }
      // The sampling spec rides only for stochastic requests; a greedy
      // request's record is byte-identical to the pre-sampling format.
      if (r.sampling.temperature > 0.0f) {
        const glm_sample::Params& g = r.sampling;
        out += ",\"g\":{\"t\":";
        append_json_int(&out, float_bits(g.temperature));
        out += ",\"p\":";
        append_json_int(&out, float_bits(g.top_p));
        out += ",\"k\":";
        append_json_int(&out, g.top_k);
        out += ",\"m\":";
        append_json_int(&out, float_bits(g.min_p));
        out += ",\"r\":";
        append_json_int(&out, float_bits(g.repetition_penalty));
        out += ",\"f\":";
        append_json_int(&out, float_bits(g.frequency_penalty));
        out += ",\"q\":";
        append_json_int(&out, float_bits(g.presence_penalty));
        out += ",\"l\":";
        append_json_int(&out, g.logprobs);
        out += ",\"s\":\"" + hex64(r.seed) + "\"}";
      }
      out.push_back('}');
    }
    out.push_back(']');
  }
  if (!events.cancels.empty()) {
    out += ",\"c\":[";
    for (size_t i = 0; i < events.cancels.size(); ++i) {
      if (i != 0) out.push_back(',');
      append_json_string(&out, events.cancels[i]);
    }
    out.push_back(']');
  }
  out.push_back('}');
  return out;
}

std::string encode_journal_stop() { return "{\"op\":\"stop\"}"; }
std::string encode_journal_warm() { return "{\"op\":\"warm\"}"; }

namespace {

// The peer's journal connect races rank 0's journal bind: a peer's bus
// rendezvous can complete milliseconds before rank 0 (still logging
// its own bus-up) reaches the bind — observed live 2026-09-01: peer 3
// was refused 30ms after its bus came up, and rank 0's listener bound
// 30ms after THAT. The 7-hour clock skew between boxes makes the
// timestamps look impossible; the mechanism is a plain single-shot
// connect losing a race. Retry inside the window (the same lesson the
// bus rendezvous taught the soak, learned the cheap way this time).
dgpp::net::TcpConn journal_connect_with_retry(const std::string& host,
                                              uint16_t port,
                                              int window_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(window_ms);
  std::string last_error;
  for (;;) {
    try {
      dgpp::net::TcpConn conn = dgpp::net::TcpConn::connect(host, port, 5000);
      DGPP_LOG_INFO("journal: connected to rank 0 at {}:{} ({} attempt window)",
                    host, port, window_ms);
      return conn;
    } catch (const std::exception& e) {
      last_error = e.what();
      DGPP_LOG_INFO("journal: connect to {}:{} not ready yet ({}); retrying",
                    host, port, last_error);
    }
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(
          "journal: cannot connect to " + host + ":" +
          std::to_string(port) + " within " + std::to_string(window_ms) +
          "ms (rank 0's journal never appeared?) — last: " + last_error);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

const dgpp::minijson::Value& field(const dgpp::minijson::Value& v,
                                  const char* name, const char* where) {
  const dgpp::minijson::Value* f = v.find(name);
  if (f == nullptr)
    throw std::runtime_error(std::string("journal: ") + where +
                             " is missing '" + name + "'");
  return *f;
}

}  // namespace

JournalRecord decode_journal_line(std::string_view line) {
  // minijson VIEWS ITS INPUT — ParseResult owns nothing, string
  // members are string_views into the parsed buffer. The buffer must
  // outlive every read: parse a NAMED local (a temporary std::string
  // would dangle every view the moment it dies), and copy the values
  // out into the record before returning.
  const std::string src(line);
  const dgpp::minijson::ParseResult pr = dgpp::minijson::parse(src);
  const dgpp::minijson::Value& v = pr.root;
  JournalRecord rec;
  if (!v.is_object())
    throw std::runtime_error("journal: record is not a JSON object");
  const std::string op(field(v, "op", "record").as_string());
  if (op == "stop") {
    rec.stop = true;
    return rec;
  }
  if (op == "warm") {
    rec.warm = true;
    return rec;
  }
  if (op != "tick")
    throw std::runtime_error("journal: unknown op '" + op + "'");

  if (const dgpp::minijson::Value* s = v.find("s")) {
    if (!s->is_array())
      throw std::runtime_error("journal: 's' is not an array");
    for (const dgpp::minijson::Value& item : s->items()) {
      SchedulerRequest r;
      r.id = std::string(field(item, "id", "submit").as_string());
      if (r.id.empty())
        throw std::runtime_error("journal: submit with empty id");
      const dgpp::minijson::Value& p = field(item, "p", "submit");
      if (!p.is_array() || p.items().empty())
        throw std::runtime_error("journal: submit '" + r.id +
                                 "' has no prompt ids");
      for (const dgpp::minijson::Value& t : p.items()) {
        if (!t.is_number())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-numeric prompt id");
        r.prompt.push_back(t.as_int());
      }
      const dgpp::minijson::Value& m = field(item, "m", "submit");
      if (!m.is_number() || m.as_int() < 1)
        throw std::runtime_error("journal: submit '" + r.id +
                                 "' has bad max_steps");
      r.max_steps = static_cast<int>(m.as_int());
      if (const dgpp::minijson::Value* ca = item.find("ca")) {
        if (!ca->is_number() || ca->as_int() < 0)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has bad cancel_after");
        r.cancel_after = static_cast<int>(ca->as_int());
      }
      if (const dgpp::minijson::Value* g = item.find("g")) {
        if (!g->is_object())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-object sampling spec");
        const auto bits = [&](const char* name) -> uint32_t {
          const dgpp::minijson::Value& f = field(*g, name, "sampling spec");
          if (!f.is_number() || f.as_int() < 0 || f.as_int() > 0xFFFFFFFFll)
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has bad sampling field " + name);
          return static_cast<uint32_t>(f.as_int());
        };
        glm_sample::Params& p = r.sampling;
        p.temperature = float_from_bits(bits("t"));
        p.top_p = float_from_bits(bits("p"));
        p.min_p = float_from_bits(bits("m"));
        p.repetition_penalty = float_from_bits(bits("r"));
        p.frequency_penalty = float_from_bits(bits("f"));
        p.presence_penalty = float_from_bits(bits("q"));
        const dgpp::minijson::Value& k = field(*g, "k", "sampling spec");
        const dgpp::minijson::Value& l = field(*g, "l", "sampling spec");
        if (!k.is_number() || !l.is_number())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has bad top_k/logprobs");
        p.top_k = static_cast<int>(k.as_int());
        p.logprobs = static_cast<int>(l.as_int());
        const std::string seed_hex(
            field(*g, "s", "sampling spec").as_string());
        if (seed_hex.size() != 16 ||
            seed_hex.find_first_not_of("0123456789abcdef") != std::string::npos)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a bad seed");
        r.seed = std::stoull(seed_hex, nullptr, 16);
        try {
          glm_sample::validate_params(p);
        } catch (const std::invalid_argument& e) {
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' carries an invalid sampling spec: " +
                                   e.what());
        }
        if (!(p.temperature > 0.0f))
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' carries a greedy sampling spec");
      }
      rec.submits.push_back(std::move(r));
    }
  }
  if (const dgpp::minijson::Value* c = v.find("c")) {
    if (!c->is_array())
      throw std::runtime_error("journal: 'c' is not an array");
    for (const dgpp::minijson::Value& id : c->items()) {
      if (id.as_string().empty())
        throw std::runtime_error("journal: empty cancel id");
      rec.cancels.emplace_back(id.as_string());
    }
  }
  return rec;
}

// ---- rank 0 ----------------------------------------------------------------

JournalWriter::JournalWriter(uint16_t listen_port)
    : listener_(dgpp::net::TcpListener::bind(listen_port)) {}

void JournalWriter::accept_peers(int world, int accept_timeout_ms) {
  const int peers = world - 1;
  peers_.reserve(static_cast<size_t>(peers));
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(accept_timeout_ms);
  for (int i = 1; i <= peers; ++i) {
    const int left_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count());
    dgpp::net::TcpConn conn = listener_.accept(std::max(left_ms, 0));
    if (!conn.valid())
      throw std::runtime_error(
          "journal: only " + std::to_string(i - 1) + " of " +
          std::to_string(peers) + " peers connected within " +
          std::to_string(accept_timeout_ms) +
          "ms — a short world cannot serve (check the peers' logs)");
    peers_.push_back(std::move(conn));
    DGPP_LOG_INFO("journal: peer {}/{} connected ({}ms left in window)", i,
                   peers, left_ms);
  }
  DGPP_LOG_INFO("journal: world complete ({} peer connection(s))", peers);
}

void JournalWriter::broadcast(const std::string& line) {
  const std::string framed = line + "\n";
  for (size_t i = 0; i < peers_.size(); ++i) {
    if (!peers_[i].write_all(framed.data(), framed.size()))
      throw std::runtime_error(
          "journal: peer connection " + std::to_string(i + 1) + " of " +
          std::to_string(peers_.size()) +
          " stopped reading — the fabric is broken; refusing to serve on "
          "without it");
  }
}

// ---- peers ------------------------------------------------------------------

JournalReader::JournalReader(const std::string& host, uint16_t port,
                             int connect_timeout_ms)
    : conn_(journal_connect_with_retry(host, port, connect_timeout_ms)) {}

bool JournalReader::read_line(const std::function<bool()>& should_stop,
                              std::string* line) {
  for (;;) {
    const size_t nl = pending_.find('\n');
    if (nl != std::string::npos) {
      // A complete record is honored even when the stop flag already
      // fired — the stop RECORD itself arrives this way.
      line->assign(pending_, 0, nl);
      pending_.erase(0, nl + 1);
      return true;
    }
    if (should_stop()) return false;
    // 250ms slices keep SIGINT latency bounded while idle blocking
    // stays cheap; read_some does the one-syscall-per-record work.
    if (!conn_.wait_readable(250)) continue;
    char buf[4096];
    const int n = conn_.read_some(buf, sizeof(buf));
    if (n == 0) return false;  // EOF: rank 0's journal is closed
    if (n < 0)
      return false;  // readable but recv fails (reset): world's over
    pending_.append(buf, static_cast<size_t>(n));
  }
}

bool wait_journal_warm(JournalReader* reader,
                       const std::function<bool()>& should_stop) {
  std::string line;
  if (!reader->read_line(should_stop, &line)) {
    DGPP_LOG_INFO("journal: rank 0's stream ended before the warm record — "
                  "exiting");
    return false;
  }
  const JournalRecord rec = decode_journal_line(line);
  if (rec.stop) {
    DGPP_LOG_INFO("journal: stop record before the warm record — exiting");
    return false;
  }
  if (!rec.warm)
    throw std::runtime_error(
        "journal: rank 0 ticked before the warm record — protocol order "
        "violated (§11); fabric emergency");
  return true;
}

void run_journal_peer(Scheduler* sched, JournalReader* reader,
                      const std::function<bool()>& should_stop) {
  for (;;) {
    std::string line;
    if (!reader->read_line(should_stop, &line)) {
      DGPP_LOG_INFO("journal: rank 0's stream ended — exiting");
      return;
    }
    // Not const: the submits below are std::move'd into the scheduler —
    // a const record would silently turn every "move" into a copy
    // (const rvalueref binds the copy ctor; GCC's redundant-move
    // warning is how this line got caught).
    JournalRecord rec = decode_journal_line(line);
    if (rec.stop) {
      DGPP_LOG_INFO("journal: stop record — exiting");
      return;
    }
    if (rec.warm)
      throw std::runtime_error(
          "journal: warm record inside the serving loop — protocol order "
          "violated (§11); fabric emergency");
    for (auto& r : rec.submits) {
      const std::string id = r.id;  // try_submit takes by value
      // Rank 0 admitted this against a queue state identical to ours
      // (same records, same order — §11); a refusal here means the
      // schedulers DIVERGED, which is the one bug this design exists
      // to make impossible. Loud death beats serving on a lie.
      if (!sched->try_submit(std::move(r)))
        throw std::runtime_error(
            "journal: could not admit '" + id + "' that rank 0 admitted — "
            "scheduler divergence (§11); fabric emergency");
    }
    for (const std::string& id : rec.cancels) sched->cancel(id);
    sched->tick();
  }
}

}  // namespace dgpp::service
