#include "common/log.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace dgpp {
namespace {
std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::mutex g_mu;

LogLevel level_from_string(std::string_view s) {
  static constexpr std::array<std::pair<std::string_view, LogLevel>, 5> kMap{{
      {"trace", LogLevel::Trace},
      {"debug", LogLevel::Debug},
      {"info", LogLevel::Info},
      {"warn", LogLevel::Warn},
      {"error", LogLevel::Error},
  }};
  for (auto& [k, v] : kMap) {
    if (s == k) return v;
  }
  return LogLevel::Info;
}

}  // namespace

LogLevel current_log_level() { return static_cast<LogLevel>(g_level.load()); }

void set_log_level(LogLevel lvl) { g_level.store(static_cast<int>(lvl)); }

void set_log_level_from_env(const char* env_var) {
  const char* v = getenv(env_var);
  if (!v || !*v) return;
  set_log_level(level_from_string(v));
}

const char* log_level_name(LogLevel lvl) {
  switch (lvl) {
    case LogLevel::Trace: return "TRACE";
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info: return "INFO ";
    case LogLevel::Warn: return "WARN ";
    case LogLevel::Error: return "ERROR";
  }
  return "?????";
}

void log_line(LogLevel lvl, std::string_view msg) {
  using namespace std::chrono;
  const auto now = system_clock::now();
  std::time_t tt = system_clock::to_time_t(now);
  const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
  std::tm tm{};
  gmtime_r(&tt, &tm);  // UTC on every rank: the peers' logs read side by
                       // side with rank 0's (2026-09-06; a peer used to
                       // stamp its own zone)

  std::lock_guard<std::mutex> lock(g_mu);
  std::fprintf(stderr, "%04d-%02d-%02d %02d:%02d:%02d.%03d %s %.*s\n",
               tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
               tm.tm_sec, static_cast<int>(ms), log_level_name(lvl),
               static_cast<int>(msg.size()), msg.data());
}

}  // namespace dgpp
