#include <stdexcept>

#include "common/log.hpp"
#include "common/test.hpp"

DGPP_TEST(log_level_parsing) {
  dgpp::set_log_level(dgpp::LogLevel::Warn);
  if (dgpp::current_log_level() != dgpp::LogLevel::Warn) {
    throw std::runtime_error("set_log_level failed");
  }
  setenv("DGPP_TEST_LOG_LEVEL", "trace", 1);
  dgpp::set_log_level_from_env("DGPP_TEST_LOG_LEVEL");
  if (dgpp::current_log_level() != dgpp::LogLevel::Trace) {
    throw std::runtime_error("env override failed");
  }
}

DGPP_TEST(log_line_smoke) {
  // Must not crash under concurrent-ish usage; message formatting path.
  DGPP_LOG_INFO("smoke value={}", 42);
}

int main() { return ::dgpp::test::run_all(); }
