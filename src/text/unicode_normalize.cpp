#include "text/unicode_normalize.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "text/unicode_tables.hpp"

namespace dgpp::text::unicode {
namespace {

void decode_utf8(std::string_view s, std::vector<uint32_t>& out) {
  out.clear();
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char b = static_cast<unsigned char>(s[i]);
    int len = 0;
    uint32_t cp = 0;
    if (b < 0x80) { len = 1; cp = b; }
    else if ((b & 0xE0) == 0xC0) { len = 2; cp = b & 0x1F; }
    else if ((b & 0xF0) == 0xE0) { len = 3; cp = b & 0x0F; }
    else if ((b & 0xF8) == 0xF0) { len = 4; cp = b & 0x07; }
    else throw std::runtime_error("nfc: input is not valid UTF-8");
    if (i + static_cast<size_t>(len) > s.size()) throw std::runtime_error("nfc: truncated UTF-8 sequence");
    for (int k = 1; k < len; ++k) {
      const unsigned char c = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
      if ((c & 0xC0) != 0x80) throw std::runtime_error("nfc: input is not valid UTF-8 (continuation)");
      cp = (cp << 6) | (c & 0x3F);
    }
    out.push_back(cp);
    i += static_cast<size_t>(len);
  }
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

uint8_t ccc(uint32_t cp) {
  size_t lo = 0, hi = std::size(kCcc);
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < kCcc[mid][0]) hi = mid;
    else if (cp > kCcc[mid][1]) lo = mid + 1;
    else return static_cast<uint8_t>(kCcc[mid][2]);
  }
  return 0;
}

// The single-level canonical decomposition, or false.
bool decomposition(uint32_t cp, uint32_t& a, uint32_t& b) {
  size_t lo = 0, hi = std::size(kDecomp);
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < kDecomp[mid][0]) hi = mid;
    else if (cp > kDecomp[mid][0]) lo = mid + 1;
    else {
      a = kDecomp[mid][1];
      b = kDecomp[mid][2];
      return true;
    }
  }
  return false;
}

constexpr uint32_t kHangulBase = 0xAC00, kHangulCount = 11172;
constexpr uint32_t kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;

void decompose_into(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= kHangulBase && cp < kHangulBase + kHangulCount) {
    const uint32_t s = cp - kHangulBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    const uint32_t t = s % kTCount;
    if (t != 0) out.push_back(kTBase + t);
    return;
  }
  uint32_t a = 0, b = 0;
  if (decomposition(cp, a, b)) {
    decompose_into(a, out);
    if (b != 0) decompose_into(b, out);
    return;
  }
  out.push_back(cp);
}

// The primary composite of (a, b), or 0.
uint32_t compose_pair(uint32_t a, uint32_t b) {
  // Hangul: L + V -> LV, LV + T -> LVT.
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
    return kHangulBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  if (a >= kHangulBase && a < kHangulBase + kHangulCount && (a - kHangulBase) % kTCount == 0 &&
      b > kTBase && b < kTBase + kTCount)
    return a + (b - kTBase);
  size_t lo = 0, hi = std::size(kCompose);
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (a < kCompose[mid][0] || (a == kCompose[mid][0] && b < kCompose[mid][1])) hi = mid;
    else if (a > kCompose[mid][0] || (a == kCompose[mid][0] && b > kCompose[mid][1])) lo = mid + 1;
    else return kCompose[mid][2];
  }
  return 0;
}

}  // namespace

bool nfc_quick_yes(std::string_view utf8) {
  std::vector<uint32_t> cps;
  decode_utf8(utf8, cps);
  for (uint32_t cp : cps)
    if (cp >= 0x300 && nfc_quick_no_or_maybe(cp)) return false;
  return true;
}

std::string nfc(std::string_view utf8) {
  std::vector<uint32_t> cps;
  decode_utf8(utf8, cps);
  bool yes = true;
  for (uint32_t cp : cps)
    if (cp >= 0x300 && nfc_quick_no_or_maybe(cp)) { yes = false; break; }
  if (yes) return std::string(utf8);

  // 1. Full canonical decomposition.
  std::vector<uint32_t> d;
  d.reserve(cps.size() * 2);
  for (uint32_t cp : cps) decompose_into(cp, d);
  // 2. Canonical ordering: stable sort of every non-starter run by ccc.
  std::vector<uint8_t> cls(d.size());
  for (size_t i = 0; i < d.size(); ++i) cls[i] = ccc(d[i]);
  for (size_t i = 0; i < d.size();) {
    if (cls[i] == 0) { ++i; continue; }
    size_t j = i;
    while (j < d.size() && cls[j] != 0) ++j;
    // insertion sort (runs are short), stable
    for (size_t p = i + 1; p < j; ++p) {
      for (size_t q = p; q > i && cls[q - 1] > cls[q]; --q) {
        std::swap(d[q - 1], d[q]);
        std::swap(cls[q - 1], cls[q]);
      }
    }
    i = j;
  }
  // 3. Canonical composition.
  std::vector<uint32_t> out;
  out.reserve(d.size());
  std::vector<uint8_t> out_cls;
  out_cls.reserve(d.size());
  size_t starter = SIZE_MAX;
  uint8_t last_ccc = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    const uint32_t c = d[i];
    const uint8_t cc = cls[i];
    if (starter != SIZE_MAX) {
      const bool intervening = out.size() - 1 > starter;
      const bool blocked = intervening && (last_ccc == 0 || last_ccc >= cc);
      if (!blocked) {
        const uint32_t comp = compose_pair(out[starter], c);
        if (comp != 0) {
          out[starter] = comp;
          continue;
        }
      }
    }
    out.push_back(c);
    out_cls.push_back(cc);
    if (cc == 0) starter = out.size() - 1;
    last_ccc = cc;
  }
  std::string result;
  result.reserve(utf8.size());
  for (uint32_t cp : out) append_utf8(result, cp);
  return result;
}

}  // namespace dgpp::text::unicode
