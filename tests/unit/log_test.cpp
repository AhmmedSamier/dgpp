#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "common/log.hpp"
#include "common/test.hpp"
#include "core/trace.hpp"

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

DGPP_TEST(trace_is_thread_safe_and_jsonl_escapes_names) {
  setenv("DGPP_TRACE", "1", 1);
  dgpp::Trace::clear();
  {
    dgpp::ScopedTrace trace{"quote\"line\n"};
  }
  std::vector<std::thread> threads;
  for (int index = 0; index < 4; ++index) {
    threads.emplace_back([] {
      dgpp::ScopedTrace trace{"worker"};
    });
  }
  for (auto& thread : threads) thread.join();
  if (dgpp::Trace::records().size() != 5)
    throw std::runtime_error("concurrent trace records were lost");

  const auto path = std::filesystem::temp_directory_path() /
                    ("dgpp-trace-" + std::to_string(getpid()) + ".jsonl");
  if (dgpp::Trace::dump_jsonl(path.string()) != 5)
    throw std::runtime_error("trace dump count mismatch");
  std::ifstream input(path);
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  std::filesystem::remove(path);
  if (contents.find("quote\\\"line\\n") == std::string::npos)
    throw std::runtime_error("trace name was not JSON escaped");
}

int main() { return ::dgpp::test::run_all(); }
