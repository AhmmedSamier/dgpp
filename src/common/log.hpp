#pragma once
#include <format>
#include <string>
#include <string_view>

namespace dgpp {

enum class LogLevel : int { Trace = 0, Debug = 1, Info = 2, Warn = 3, Error = 4 };

// Thread-safe, level-filtered stderr logger. Timestamped via steady wall clock.
LogLevel current_log_level();
void set_log_level(LogLevel lvl);
void set_log_level_from_env(const char* env_var);  // default "DGPP_LOG_LEVEL"
const char* log_level_name(LogLevel lvl);

void log_line(LogLevel lvl, std::string_view msg);

template <typename... Args>
inline void logf(LogLevel lvl, std::format_string<Args...> fmt, Args&&... args) {
  if (static_cast<int>(lvl) < static_cast<int>(current_log_level())) return;
  log_line(lvl, std::format(fmt, std::forward<Args>(args)...));
}

#define DGPP_LOG_TRACE(...) ::dgpp::logf(::dgpp::LogLevel::Trace, __VA_ARGS__)
#define DGPP_LOG_DEBUG(...) ::dgpp::logf(::dgpp::LogLevel::Debug, __VA_ARGS__)
#define DGPP_LOG_INFO(...) ::dgpp::logf(::dgpp::LogLevel::Info, __VA_ARGS__)
#define DGPP_LOG_WARN(...) ::dgpp::logf(::dgpp::LogLevel::Warn, __VA_ARGS__)
#define DGPP_LOG_ERROR(...) ::dgpp::logf(::dgpp::LogLevel::Error, __VA_ARGS__)

}  // namespace dgpp
