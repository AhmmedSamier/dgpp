#include "serve/image_inputs.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "common/base64.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 16384
#include "../../third_party/stb/stb_image.h"

namespace dgpp::serve {
namespace {
// Antialiased bicubic, a=-0.5, matching the processor's uint8 resize.
double cubic(double x) {
  x = std::abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}
struct Filter {
  int first;
  std::vector<double> weights;
};
std::vector<Filter> filters(int src, int dst) {
  std::vector<Filter> out;
  const double scale = static_cast<double>(src) / dst, support = std::max(1.0, scale);
  for (int i = 0; i < dst; ++i) {
    const double center = (i + 0.5) * scale;
    const int first = std::max(0, static_cast<int>(std::floor(center - 2 * support + 0.5)));
    const int last = std::min(src, static_cast<int>(std::floor(center + 2 * support + 0.5)));
    Filter f{first, {}};
    double sum = 0;
    for (int j = first; j < last; ++j) {
      const double w = cubic((j + 0.5 - center) / support);
      f.weights.push_back(w);
      sum += w;
    }
    for (auto& w : f.weights) w /= sum;
    out.push_back(std::move(f));
  }
  return out;
}
}  // namespace
ImageInput resize_glm_image(const uint8_t* rgb, int width, int height, int max_tokens) {
  if (!rgb || width < 1 || height < 1 || width > 16384 || height > 16384 ||
      static_cast<int64_t>(width) * height > 32 * 1024 * 1024 || max_tokens < 16 ||
      max_tokens > kMaxImageTokens)
    throw std::invalid_argument("image dimensions or token budget exceed supported limits");
  const auto align = [](int n) { return (n + 27) / 28 * 28; };
  int th = align(height), tw = align(width);
  if (static_cast<int64_t>(th) * tw < 16 * 28 * 28) {
    const double scale = std::sqrt(16.0 * 28 * 28 / (static_cast<double>(height) * width));
    th = align(static_cast<int>(std::ceil(height * scale)));
    tw = align(static_cast<int>(std::ceil(width * scale)));
  }
  if (static_cast<int64_t>(th) * tw > static_cast<int64_t>(max_tokens) * 28 * 28) {
    int low = 1, high = height;
    th = tw = 28;
    while (low <= high) {
      const int h = (low + high) / 2;
      const int w = std::max(1, static_cast<int>(static_cast<int64_t>(width) * h / height));
      if (static_cast<int64_t>(align(h)) * align(w) <= static_cast<int64_t>(max_tokens) * 28 * 28) {
        th = align(h);
        tw = align(w);
        low = h + 1;
      } else
        high = h - 1;
    }
  }
  double scale = std::min(static_cast<double>(th) / height, static_cast<double>(tw) / width);
  if (static_cast<int64_t>(height) * width >= 16 * 28 * 28) scale = std::min(1.0, scale);
  const int ch = std::max(1, std::min(th, static_cast<int>(std::floor(height * scale))));
  const int cw = std::max(1, std::min(tw, static_cast<int>(std::floor(width * scale))));
  ImageInput out;
  out.width = tw;
  out.height = th;
  out.tokens = (tw / 28) * (th / 28);
  out.rgb.resize(static_cast<size_t>(tw) * th * 3, 0);
  if (ch == height && cw == width) {
    for (int y = 0; y < ch; ++y)
      std::copy_n(rgb + static_cast<size_t>(y) * width * 3, cw * 3,
                  out.rgb.data() + static_cast<size_t>(y) * tw * 3);
    return out;
  }
  const auto fx = filters(width, cw), fy = filters(height, ch);
  std::vector<float> tmp(static_cast<size_t>(height) * cw * 3);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (size_t j = 0; j < fx[x].weights.size(); ++j)
          sum += fx[x].weights[j] * rgb[(static_cast<size_t>(y) * width + fx[x].first + j) * 3 + c];
        tmp[(static_cast<size_t>(y) * cw + x) * 3 + c] = static_cast<float>(sum);
      }
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (size_t j = 0; j < fy[y].weights.size(); ++j)
          sum += fy[y].weights[j] * tmp[((fy[y].first + j) * cw + x) * 3 + c];
        out.rgb[(static_cast<size_t>(y) * tw + x) * 3 + c] =
            static_cast<uint8_t>(std::clamp(std::nearbyint(sum), 0.0, 255.0));
      }
  return out;
}
ImageInput prepare_glm_image(const minijson::Value& value, const std::string& param) {
  const auto* url = value.find("url");
  if (!value.is_object() || !url || !url->is_string())
    throw ImageInputError("image_url must contain a string url", param + ".url");
  int max_tokens = kMaxImageTokens;
  if (const auto* detail = value.find("detail")) {
    if (!detail->is_string() || (detail->as_string() != "auto" && detail->as_string() != "high" &&
                                 detail->as_string() != "low"))
      throw ImageInputError("image detail must be auto, high or low", param + ".detail");
    if (detail->as_string() == "low") max_tokens = 256;
  }
  const auto s = url->as_string();
  const bool png = s.starts_with("data:image/png;base64,"),
             jpeg = s.starts_with("data:image/jpeg;base64,");
  if (!png && !jpeg)
    throw ImageInputError(
        "use a base64 data URI with image/png or image/jpeg; remote URLs are not supported",
        param + ".url");
  try {
    const std::string data = decode_base64(s.substr(s.find(',') + 1), 20 * 1024 * 1024);
    if ((png && !std::string_view(data).starts_with("\x89PNG\r\n\x1a\n")) ||
        (jpeg && !std::string_view(data).starts_with("\xff\xd8\xff")))
      throw std::invalid_argument("image MIME type does not match its bytes");
    int w = 0, h = 0, channels = 0;
    const auto* bytes = reinterpret_cast<const stbi_uc*>(data.data());
    if (!stbi_info_from_memory(bytes, static_cast<int>(data.size()), &w, &h, &channels) || w < 1 ||
        h < 1 || static_cast<int64_t>(w) * h > 32 * 1024 * 1024)
      throw std::invalid_argument("invalid image or image exceeds 32 megapixels");
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load_from_memory(bytes, static_cast<int>(data.size()), &w, &h, &channels, 3),
        stbi_image_free);
    if (!pixels) throw std::invalid_argument("cannot decode image");
    return resize_glm_image(pixels.get(), w, h, max_tokens);
  } catch (const std::invalid_argument& e) {
    throw ImageInputError(e.what(), param + ".url");
  }
}
}  // namespace dgpp::serve
