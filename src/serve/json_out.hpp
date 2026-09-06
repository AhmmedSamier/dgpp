#pragma once
// Minimal JSON WRITER for the service's response payloads (M6 Stage 4).
// minijson is the reader; this is its writing counterpart — fixed
// hand-built shapes with correct string escaping, no DOM, no
// allocations beyond the output buffer.
//
// Escaping is RFC 8259: quote, backslash, and the control characters
// (including the two-byte \uXXXX forms for <0x20). UTF-8 payload bytes
// pass through verbatim — the tokenizer's decode() emits valid UTF-8,
// and re-encoding it would be both wrong and sad.
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace dgpp::serve {

// Appends `s` as a quoted, escaped JSON string to `out`.
inline void append_json_string(std::string* out, std::string_view s) {
  out->push_back('"');
  for (const char c : s) {
    switch (c) {
      case '"': out->append("\\\""); break;
      case '\\': out->append("\\\\"); break;
      case '\n': out->append("\\n"); break;
      case '\r': out->append("\\r"); break;
      case '\t': out->append("\\t"); break;
      case '\b': out->append("\\b"); break;
      case '\f': out->append("\\f"); break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out->append(buf, 6);
        } else {
          out->push_back(c);
        }
    }
  }
  out->push_back('"');
}

inline std::string json_string(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  append_json_string(&out, s);
  return out;
}

// Fixed-precision integers (token counts, timestamps, ids) — no float
// formatting exists in the service payload surface, deliberately.
inline void append_json_int(std::string* out, int64_t v) {
  char buf[24];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  out->append(buf, end - buf);
  (void)ec;
}

inline std::string json_int(int64_t v) {
  std::string out;
  append_json_int(&out, v);
  return out;
}

// The one float surface: the effective sampling defaults /v1/models reports.
// Shortest round-trip form (std::to_chars), so the number a client reads
// back parses to the exact value the server applies. Callers pass finite
// values only (a non-finite default is refused at load).
inline void append_json_float(std::string* out, double v) {
  char buf[32];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  (void)ec;
  out->append(buf, end - buf);
}
// The spec's fields are fp32: format at float precision, so 0.95f reads
// back as 0.95 (and parses to the same float), not as its double expansion.
inline void append_json_float(std::string* out, float v) {
  char buf[32];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  (void)ec;
  out->append(buf, end - buf);
}

}  // namespace dgpp::serve
