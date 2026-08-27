#include "core/trace.hpp"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace dgpp {

namespace {
constexpr size_t kMaxRecords = 1u << 20;

thread_local std::vector<Trace::Record> tls_stack;
std::vector<Trace::Record> g_records;
std::mutex g_records_mutex;
int64_t now_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch())
      .count();
}

void write_json_string(std::ostream& stream, const std::string& value) {
  static constexpr char kHex[] = "0123456789abcdef";
  stream << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': stream << "\\\""; break;
      case '\\': stream << "\\\\"; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (character < 0x20) {
          stream << "\\u00" << kHex[character >> 4] << kHex[character & 0x0f];
        } else {
          stream << static_cast<char>(character);
        }
    }
  }
  stream << '"';
}
}  // namespace

bool Trace::enabled() {
  static bool on = [] {
    const char* e = std::getenv("DGPP_TRACE");
    return e && e[0] && !(e[0] == '0' && e[1] == '\0');
  }();
  return on;
}

void Trace::push(std::string_view name) {
  int64_t t0 = now_us();
  tls_stack.push_back(
      {std::string(name), 0.0, t0, static_cast<int>(tls_stack.size())});
}

void Trace::pop() {
  if (tls_stack.empty()) return;
  auto rec = std::move(tls_stack.back());
  tls_stack.pop_back();
  rec.micros = static_cast<double>(now_us() - rec.t_begin_us);
  rec.depth = static_cast<int>(tls_stack.size());
  std::lock_guard lock(g_records_mutex);
  if (g_records.size() < kMaxRecords) g_records.push_back(std::move(rec));
}

std::vector<Trace::Record> Trace::records() {
  std::lock_guard lock(g_records_mutex);
  return g_records;
}

void Trace::clear() {
  std::lock_guard lock(g_records_mutex);
  g_records.clear();
}

size_t Trace::dump_jsonl(const std::string& path) {
  const auto snapshot = records();
  std::ofstream f(path);
  if (!f) throw std::runtime_error("trace: cannot open " + path);
  for (const auto& r : snapshot) {
    f << "{\"name\":";
    write_json_string(f, r.name);
    f << ",\"us\":" << r.micros << ",\"t0\":" << r.t_begin_us
      << ",\"depth\":" << r.depth << "}\n";
  }
  if (!f) throw std::runtime_error("trace: write failed for " + path);
  return snapshot.size();
}

}  // namespace dgpp
