// ModelOpt NVFP4 CSF expansion (qwen_csf_scales_expand): round-trips a
// reference encoder (row base = row min capped at 240, out-of-window
// bytes as exceptions) to the exact source bytes, and refuses malformed
// sidecars by name.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/qwen/loader.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (cond) return;
  throw std::runtime_error(what);
}

// Deterministic filler (an LCG): values across the E4M3 byte range.
uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

// Reference encoder (the vLLM csf_scales test's rule): base = row min
// capped at 240, codes = raw - base, out-of-window positions carried as
// exceptions with their true byte.
void encode(const std::vector<uint8_t>& raw, int64_t rows, int64_t cols,
            std::vector<uint8_t>& fixed, std::vector<uint32_t>& exc) {
  const int64_t slab = 16 * (1 + cols / 2);
  fixed.assign(static_cast<size_t>(rows / 16) * static_cast<size_t>(slab), 0);
  exc.clear();
  for (int64_t g = 0; g < rows / 16; ++g) {
    uint8_t* base = fixed.data() + static_cast<size_t>(g) * static_cast<size_t>(slab);
    uint8_t* packed = base + 16;
    for (int64_t r = 0; r < 16; ++r) {
      const uint8_t* row = raw.data() + static_cast<size_t>(g * 16 + r) * static_cast<size_t>(cols);
      unsigned mn = 255;
      for (int64_t c = 0; c < cols; ++c) mn = std::min(mn, static_cast<unsigned>(row[c]));
      const unsigned b = std::min(mn, 240u);
      base[static_cast<size_t>(r)] = static_cast<uint8_t>(b);
      for (int64_t c = 0; c < cols / 2; ++c) {
        unsigned code[2];
        for (int k = 0; k < 2; ++k) {
          const int64_t cc = 2 * c + k;
          const int v = static_cast<int>(row[cc]) - static_cast<int>(b);
          if (v < 0 || v > 15) {
            const size_t pos = static_cast<size_t>((g * 16 + r) * cols + cc);
            exc.push_back((static_cast<uint32_t>(row[cc]) << 24) | static_cast<uint32_t>(pos));
            code[k] = 0;
          } else {
            code[k] = static_cast<unsigned>(v);
          }
        }
        packed[static_cast<size_t>(r) * static_cast<size_t>(cols / 2) + static_cast<size_t>(c)] =
            static_cast<uint8_t>(code[0] | (code[1] << 4));
      }
    }
  }
}

std::string refusal(const std::vector<uint8_t>& fixed, const std::vector<uint32_t>& exc, int64_t rows,
                    int64_t cols) {
  try {
    (void)dgpp::qwen_csf_scales_expand(fixed.data(), fixed.size(), exc.data(), exc.size(), rows, cols);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(qwen_csf_expand_round_trips_the_reference_encoder) {
  for (const auto& [rows, cols] : {std::pair<int64_t, int64_t>{32, 8}, {64, 20}, {640, 160}}) {
    std::vector<uint8_t> raw(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    uint32_t s = 0x12345u;
    for (auto& b : raw) b = static_cast<uint8_t>(0x30 + lcg(s) % 0x50);
    // Out-of-window bytes force exceptions.
    for (int i = 0; i < 64; ++i) raw[(lcg(s) * 2654435761u >> 8) % raw.size()] = 0x7E;
    std::vector<uint8_t> fixed;
    std::vector<uint32_t> exc;
    encode(raw, rows, cols, fixed, exc);
    require(!exc.empty(), "outliers encoded");
    const std::vector<uint8_t> got =
        dgpp::qwen_csf_scales_expand(fixed.data(), fixed.size(), exc.data(), exc.size(), rows, cols);
    require(got == raw, "exact scale bytes back");
  }
}

DGPP_TEST(qwen_csf_expand_refuses_malformed_sidecars) {
  std::vector<uint8_t> fixed(16 * (1 + 4), 0x38);
  std::vector<uint32_t> no_exc;
  require(refusal(fixed, no_exc, 16, 8).empty(), "clean streams decode");
  require(refusal(fixed, no_exc, 15, 8).find("16-row") != std::string::npos, "rows multiple of 16");
  require(refusal(fixed, no_exc, 16, 7).find("even-col") != std::string::npos, "cols even");
  std::vector<uint8_t> short_fixed = fixed;
  short_fixed.pop_back();
  require(refusal(short_fixed, no_exc, 16, 8).find("fixed stream") != std::string::npos,
          "fixed size checked");
  std::vector<uint8_t> hot_base = fixed;
  hot_base[3] = 241;
  require(refusal(hot_base, no_exc, 16, 8).find("240") != std::string::npos, "bases capped at 240");
  std::vector<uint32_t> oob = {(0x60u << 24) | 16u * 8u};
  require(refusal(fixed, oob, 16, 8).find("outside") != std::string::npos, "positions in range");
}
