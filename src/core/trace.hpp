#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dgpp {

// Lightweight host-side scope tracing. Enabled via DGPP_TRACE=1; bounded buffer
// dumped as JSONL on demand (later: chrome-trace export).
class Trace {
 public:
  struct Record {
    std::string name;
    double micros;
    int64_t t_begin_us;
    int depth;
  };

  static bool enabled();
  static void push(std::string_view name);
  static void pop();
  static std::vector<Record> records();
  static void clear();
  static size_t dump_jsonl(const std::string& path);
};

class ScopedTrace {
 public:
  explicit ScopedTrace(std::string_view name) {
    active_ = Trace::enabled();
    if (active_) Trace::push(name);
  }
  ~ScopedTrace() {
    if (active_) Trace::pop();
  }
  ScopedTrace(const ScopedTrace&) = delete;
  ScopedTrace& operator=(const ScopedTrace&) = delete;

 private:
  bool active_;
};

}  // namespace dgpp

#define DGPP_SCOPED_TRACE(name) \
  ::dgpp::ScopedTrace dgpp_trace_scope_##__LINE__{name}
