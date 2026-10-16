// Load-time vision widening (qwen_vision_widen_mxfp8,
// qwen_vision_dequant_nvfp4): exact MXFP8 products, exactly-rounded NVFP4
// products, and refusals naming the malformed tensor.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/qwen/vision.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (cond) return;
  throw std::runtime_error(what);
}

uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

// Independent E4M3 (the engine's bias-7 OCP FN) and E2M1 decoders.
double e4m3_exact(unsigned v) {
  const unsigned exp = (v >> 3) & 0xFu, man = v & 0x7u;
  double m;
  if (exp == 0)
    m = man * 0.001953125;  // man * 2^-9
  else
    m = (1.0 + man * 0.125) * std::ldexp(1.0, static_cast<int>(exp) - 7);
  return (v & 0x80u) ? -m : m;
}
double e2m1_exact(unsigned c) {
  static const double k[16] = {0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
  return k[c & 0xFu];
}

std::string refusal_mx(const std::vector<uint8_t>& w, const std::vector<uint8_t>& s, int64_t r,
                       int64_t c) {
  try {
    (void)dgpp::qwen_vision_widen_mxfp8("t", w.data(), w.size(), s.data(), s.size(), r, c);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

std::string refusal_nv(const std::vector<uint8_t>& w, const std::vector<uint8_t>& s, float g,
                       int64_t r, int64_t c) {
  try {
    (void)dgpp::qwen_vision_dequant_nvfp4("t", w.data(), w.size(), s.data(), s.size(), g, r, c);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(qwen_vision_widen_mxfp8_is_exact) {
  const int64_t R = 16, C = 64;
  std::vector<uint8_t> w(static_cast<size_t>(R * C)), s(static_cast<size_t>(R * C / 32));
  uint32_t st = 0xABCDEu;
  for (auto& b : w) {
    do {
      b = static_cast<uint8_t>(lcg(st));
    } while (b == 0x7F || b == 0xFF);
  }
  for (auto& b : s) b = static_cast<uint8_t>(100 + lcg(st) % 60);  // 2^-27 .. 2^32
  const std::vector<uint16_t> got =
      dgpp::qwen_vision_widen_mxfp8("t", w.data(), w.size(), s.data(), s.size(), R, C);
  require(got.size() == w.size(), "one BF16 per element");
  for (int64_t i = 0; i < R; ++i)
    for (int64_t k = 0; k < C; ++k) {
      const double want = e4m3_exact(w[static_cast<size_t>(i * C + k)]) *
                          std::ldexp(1.0, static_cast<int>(s[static_cast<size_t>(i * C / 32 + k / 32)]) - 127);
      const double have = dgpp::bf16_bits_to_float(got[static_cast<size_t>(i * C + k)]);
      require(have == want, "bit-exact widening");
    }
}

DGPP_TEST(qwen_vision_dequant_nvfp4_rounds_once) {
  const int64_t R = 48, C = 64;
  std::vector<uint8_t> w(static_cast<size_t>(R * C / 2)), s(static_cast<size_t>(R * C / 16));
  uint32_t st = 0x54321u;
  for (auto& b : w) b = static_cast<uint8_t>(lcg(st));
  for (auto& b : s) {
    do {
      b = static_cast<uint8_t>(0x30 + lcg(st) % 0x40);
    } while (b == 0x7F || b == 0xFF);
  }
  const float g = 0.75f;
  const std::vector<uint16_t> got =
      dgpp::qwen_vision_dequant_nvfp4("t", w.data(), w.size(), s.data(), s.size(), g, R, C);
  require(got.size() == static_cast<size_t>(R * C), "one BF16 per element");
  for (int64_t i = 0; i < R; ++i)
    for (int64_t k = 0; k < C; ++k) {
      const unsigned pair = w[static_cast<size_t>(i * C / 2 + k / 2)];
      const unsigned code = (k % 2 == 0) ? (pair & 0xF) : (pair >> 4);
      const double want =
          e2m1_exact(code) * e4m3_exact(s[static_cast<size_t>(i * C / 16 + k / 16)]) * g;
      const double have = dgpp::bf16_bits_to_float(got[static_cast<size_t>(i * C + k)]);
      // One BF16 ulp of slack: the loader rounds the f32 product, the
      // reference rounds the exact product.
      const double ulp = std::ldexp(1.0, std::ilogb(want == 0 ? 1.0 : std::fabs(want)) - 7);
      require(std::fabs(have - want) <= ulp, "single-rounding agreement");
    }
}

DGPP_TEST(qwen_vision_widen_refuses_malformed) {
  std::vector<uint8_t> w(16 * 32, 0x38), s(16 * 1, 127);
  require(refusal_mx(w, s, 16, 32).empty(), "clean streams widen");
  std::vector<uint8_t> nan_w = w;
  nan_w[3] = 0x7F;
  require(refusal_mx(nan_w, s, 16, 32).find("NaN") != std::string::npos, "NaN payload refused");
  std::vector<uint8_t> hot_s = s;
  hot_s[0] = 241;
  require(refusal_mx(w, hot_s, 16, 32).find("240") != std::string::npos, "hot scale refused");
  std::vector<uint8_t> short_w = w;
  short_w.pop_back();
  require(refusal_mx(short_w, s, 16, 32).find("byte counts") != std::string::npos,
          "counts checked");
  require(refusal_mx(w, s, 16, 31).find("geometry") != std::string::npos, "columns multiple of 32");
  std::vector<uint8_t> nw(16 * 8, 0x22), ns(16 * 1, 0x38);
  require(refusal_nv(nw, ns, 1.0f, 16, 16).empty(), "clean NVFP4 dequants");
  std::vector<uint8_t> nan_s = ns;
  nan_s[0] = 0xFF;
  require(refusal_nv(nw, nan_s, 1.0f, 16, 16).find("NaN") != std::string::npos,
          "NaN group scale refused");
  require(refusal_nv(nw, ns, 0.0f, 16, 16).find("global") != std::string::npos,
          "non-positive global refused");
  require(refusal_nv(nw, ns, 1.0f, 16, 15).find("geometry") != std::string::npos,
          "columns multiple of 16");
}
