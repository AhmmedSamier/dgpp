#pragma once

#include "common/image_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp::serve {
// Inline PNG/JPEG only. Resolving arbitrary URLs is deliberately not part of
// the HTTP/event-loop path. Decode errors carry the content part's API field.
struct ImageInputError : std::invalid_argument {
  ImageInputError(std::string message, std::string param)
      : std::invalid_argument(std::move(message)), param(std::move(param)) {}
  std::string param;
};

// How a family's processor turns decoded RGB into the patch canvas its tower
// expects. `pad_to_grid` is the GLM5-Next shape: round the canvas *up* to the
// grid and paint the remainder black. The Qwen3-VL family instead rounds to
// the *nearest* multiple (ties to even), never pads, and rejects an extreme
// aspect ratio or a side below the grid; both bounds are pixel budgets, i.e.
// tokens * grid * grid.
struct ImagePreprocess {
  int grid = kGlmImageGrid;
  int min_tokens = 16;
  int max_tokens = kMaxImageTokens;
  bool pad_to_grid = true;
  int low_detail_tokens = 256;
};

ImageInput prepare_glm_image(const minijson::Value& image_url, const std::string& param);
ImageInput prepare_image(const minijson::Value& image_url, const std::string& param,
                         const ImagePreprocess& pp);
// Exposed separately for resize/patch-layout reference tests.
ImageInput resize_glm_image(const uint8_t* rgb, int width, int height, int max_tokens);
ImageInput resize_image(const uint8_t* rgb, int width, int height, const ImagePreprocess& pp);
}  // namespace dgpp::serve
