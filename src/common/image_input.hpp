#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dgpp {

// Resized RGB bytes, owned by the admission record. All ranks receive the
// same pixels and token spans; no URL or process-local pointer crosses the wire.
struct ImageInput {
  int64_t offset = 0;
  int tokens = 0;
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;
};
inline constexpr int kMaxInputImages = 8;
inline constexpr int kMaxImageTokens = 1024;
inline constexpr int kMaxRequestImageTokens = 4096;
inline constexpr size_t kMaxImagePixels = 1024 * 28 * 28;

inline void validate_image_inputs(const std::vector<ImageInput>& images, size_t prompt_tokens) {
  if (images.size() > kMaxInputImages) throw std::invalid_argument("too many input images");
  int64_t end = 0;
  int total = 0;
  for (const auto& im : images) {
    if (im.offset < end || im.offset < 0 || im.tokens < 1 || im.tokens > kMaxImageTokens ||
        im.offset > static_cast<int64_t>(prompt_tokens) - im.tokens)
      throw std::invalid_argument("invalid image token span");
    if (im.width < 1 || im.height < 1 ||
        static_cast<uint64_t>(im.width) * im.height > kMaxImagePixels ||
        im.rgb.size() != static_cast<uint64_t>(im.width) * im.height * 3)
      throw std::invalid_argument("invalid image RGB dimensions");
    end = im.offset + im.tokens;
    total += im.tokens;
  }
  if (total > kMaxRequestImageTokens)
    throw std::invalid_argument("too many image tokens in request");
}
}  // namespace dgpp
