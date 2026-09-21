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
    for (int j = 0; j < last - first; ++j) {
      const double w = cubic((j + 0.5 + first - center) / support);
      f.weights.push_back(w);
      sum += w;
    }
    for (auto& w : f.weights) w /= sum;
    out.push_back(std::move(f));
  }
  return out;
}
// Fill a zeroed canvas of cw*ch pixels from the (sw*sh) source's content box,
// the padded remainder staying black.
void resample(const uint8_t* rgb, int sw, int sh, uint8_t* dst, int cw, int ch, int dw) {
  if (cw == sw && ch == sh) {
    for (int y = 0; y < ch; ++y)
      std::copy_n(rgb + static_cast<size_t>(y) * sw * 3, cw * 3,
                  dst + static_cast<size_t>(y) * dw * 3);
    return;
  }
  const auto fx = filters(sw, cw), fy = filters(sh, ch);
  std::vector<float> tmp(static_cast<size_t>(sh) * cw * 3);
  for (int y = 0; y < sh; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (size_t j = 0; j < fx[x].weights.size(); ++j)
          sum += fx[x].weights[j] * rgb[(static_cast<size_t>(y) * sw + fx[x].first + j) * 3 + c];
        tmp[(static_cast<size_t>(y) * cw + x) * 3 + c] = static_cast<float>(sum);
      }
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c) {
        double sum = 0;
        for (size_t j = 0; j < fy[y].weights.size(); ++j)
          sum += fy[y].weights[j] * tmp[((fy[y].first + j) * cw + x) * 3 + c];
        dst[(static_cast<size_t>(y) * dw + x) * 3 + c] =
            static_cast<uint8_t>(std::clamp(std::nearbyint(sum), 0.0, 255.0));
      }
}
// Python's round() on the integer/factor ratio: exact ties go to the even
// side, which std::lround would not reproduce.
int round_to_even(double ratio) {
  const double floor_value = std::floor(ratio);
  const double rest = ratio - floor_value;
  if (rest > 0.5) return static_cast<int>(floor_value) + 1;
  if (rest < 0.5) return static_cast<int>(floor_value);
  const int whole = static_cast<int>(floor_value);
  return (whole % 2) ? whole + 1 : whole;
}
}  // namespace
ImageInput resize_image(const uint8_t* rgb, int width, int height, const ImagePreprocess& pp) {
  const int g = pp.grid;
  if (!rgb || width < 1 || height < 1 || width > 16384 || height > 16384 ||
      !image_grid_supported(g) || pp.min_tokens < 1 || pp.min_tokens > pp.max_tokens ||
      pp.max_tokens > kMaxImageTokens)
    throw std::invalid_argument("image dimensions or token budget exceed supported limits");
  ImageInput out;
  out.grid = g;
  if (pp.pad_to_grid) {
    if (pp.min_tokens < 16 || pp.max_tokens < 16)
      throw std::invalid_argument("image token budget must cover a 16-token canvas");
    const auto align = [g](int n) { return (n + g - 1) / g * g; };
    const int64_t min_pixels = static_cast<int64_t>(pp.min_tokens) * g * g;
    int th = align(height), tw = align(width);
    if (static_cast<int64_t>(th) * tw < min_pixels) {
      const double scale = std::sqrt(static_cast<double>(min_pixels) /
                                     (static_cast<double>(height) * width));
      th = align(static_cast<int>(std::ceil(height * scale)));
      tw = align(static_cast<int>(std::ceil(width * scale)));
    }
    if (static_cast<int64_t>(th) * tw > static_cast<int64_t>(pp.max_tokens) * g * g) {
      int low = 1, high = height;
      th = tw = g;
      while (low <= high) {
        const int h = (low + high) / 2;
        const int w = std::max(1, static_cast<int>(static_cast<int64_t>(width) * h / height));
        if (static_cast<int64_t>(align(h)) * align(w) <=
            static_cast<int64_t>(pp.max_tokens) * g * g) {
          th = align(h);
          tw = align(w);
          low = h + 1;
        } else
          high = h - 1;
      }
    }
    double scale = std::min(static_cast<double>(th) / height, static_cast<double>(tw) / width);
    if (static_cast<int64_t>(height) * width >= min_pixels) scale = std::min(1.0, scale);
    const int ch = std::max(1, std::min(th, static_cast<int>(std::floor(height * scale))));
    const int cw = std::max(1, std::min(tw, static_cast<int>(std::floor(width * scale))));
    out.width = tw;
    out.height = th;
    out.tokens = (tw / g) * (th / g);
    out.rgb.resize(static_cast<size_t>(tw) * th * 3, 0);
    resample(rgb, width, height, out.rgb.data(), cw, ch, tw);
    return out;
  }
  // smart_resize: nearest multiple of the grid, then the pixel budgets.
  // A side below the grid needs no special case: the min_pixels branch below
  // grows it, where the reference processor raises. The aspect bound matches
  // the reference.
  if (static_cast<double>(std::max(width, height)) / std::min(width, height) > 200.0)
    throw std::invalid_argument("image aspect ratio must be below 200");
  const int64_t min_pixels = static_cast<int64_t>(pp.min_tokens) * g * g,
                max_pixels = static_cast<int64_t>(pp.max_tokens) * g * g;
  int th = round_to_even(static_cast<double>(height) / g) * g,
      tw = round_to_even(static_cast<double>(width) / g) * g;
  th = std::max(g, th);
  tw = std::max(g, tw);
  if (static_cast<int64_t>(th) * tw > max_pixels) {
    const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
    th = std::max(g, static_cast<int>(std::floor(height / beta / g)) * g);
    tw = std::max(g, static_cast<int>(std::floor(width / beta / g)) * g);
  } else if (static_cast<int64_t>(th) * tw < min_pixels) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                  (static_cast<double>(height) * width));
    th = std::max(g, static_cast<int>(std::ceil(height * beta / g)) * g);
    tw = std::max(g, static_cast<int>(std::ceil(width * beta / g)) * g);
  }
  out.width = tw;
  out.height = th;
  out.tokens = (tw / g) * (th / g);
  out.rgb.resize(static_cast<size_t>(tw) * th * 3);
  resample(rgb, width, height, out.rgb.data(), tw, th, tw);
  return out;
}
ImageInput resize_glm_image(const uint8_t* rgb, int width, int height, int max_tokens) {
  ImagePreprocess pp;
  pp.grid = kGlmImageGrid;
  pp.max_tokens = max_tokens;
  return resize_image(rgb, width, height, pp);
}
ImageInput prepare_image(const minijson::Value& value, const std::string& param,
                         const ImagePreprocess& pp) {
  const auto* url = value.find("url");
  if (!value.is_object() || !url || !url->is_string())
    throw ImageInputError("image_url must contain a string url", param + ".url");
  ImagePreprocess spec = pp;
  if (const auto* detail = value.find("detail")) {
    if (!detail->is_string() || (detail->as_string() != "auto" && detail->as_string() != "high" &&
                                 detail->as_string() != "low"))
      throw ImageInputError("image detail must be auto, high or low", param + ".detail");
    if (detail->as_string() == "low") spec.max_tokens = spec.low_detail_tokens;
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
    return resize_image(pixels.get(), w, h, spec);
  } catch (const std::invalid_argument& e) {
    throw ImageInputError(e.what(), param + ".url");
  }
}
ImageInput prepare_glm_image(const minijson::Value& value, const std::string& param) {
  return prepare_image(value, param, ImagePreprocess{});
}
}  // namespace dgpp::serve
