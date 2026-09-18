#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dgpp {
inline std::string encode_base64(std::string_view bytes) {
  constexpr char abc[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    uint32_t x = static_cast<unsigned char>(bytes[i]) << 16;
    if (i + 1 < bytes.size()) x |= static_cast<unsigned char>(bytes[i + 1]) << 8;
    if (i + 2 < bytes.size()) x |= static_cast<unsigned char>(bytes[i + 2]);
    out += abc[x >> 18];
    out += abc[(x >> 12) & 63];
    out += i + 1 < bytes.size() ? abc[(x >> 6) & 63] : '=';
    out += i + 2 < bytes.size() ? abc[x & 63] : '=';
  }
  return out;
}
inline std::string decode_base64(std::string_view s, size_t limit) {
  if (s.empty() || s.size() % 4) throw std::invalid_argument("invalid padded base64");
  const size_t size = s.size() / 4 * 3 - (s.back() == '=') - (s[s.size() - 2] == '=');
  if (size > limit) throw std::invalid_argument("base64 input exceeds byte limit");
  std::string out;
  out.reserve(size);
  for (size_t i = 0; i < s.size(); i += 4) {
    const int pad = (s[i + 2] == '=') + (s[i + 3] == '=');
    if (s[i] == '=' || s[i + 1] == '=' || (pad && i + 4 != s.size()) ||
        (s[i + 2] == '=' && s[i + 3] != '='))
      throw std::invalid_argument("invalid base64 padding");
    uint32_t x = 0;
    for (int j = 0; j < 4; ++j) {
      const unsigned char c = s[i + j];
      const int v = c >= 'A' && c <= 'Z'   ? c - 'A'
                    : c >= 'a' && c <= 'z' ? c - 'a' + 26
                    : c >= '0' && c <= '9' ? c - '0' + 52
                    : c == '+'             ? 62
                    : c == '/'             ? 63
                    : c == '='             ? 0
                                           : -1;
      if (v < 0) throw std::invalid_argument("invalid base64 character");
      x = (x << 6) | v;
    }
    if ((pad == 2 && (x & 0xffff)) || (pad == 1 && (x & 0xff)))
      throw std::invalid_argument("noncanonical base64 padding");
    out += static_cast<char>(x >> 16);
    if (pad < 2) out += static_cast<char>(x >> 8);
    if (pad == 0) out += static_cast<char>(x);
  }
  return out;
}
}  // namespace dgpp
