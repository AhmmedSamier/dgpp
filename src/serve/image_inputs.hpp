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
ImageInput prepare_glm_image(const minijson::Value& image_url, const std::string& param);
// Exposed separately for resize/patch-layout reference tests.
ImageInput resize_glm_image(const uint8_t* rgb, int width, int height, int max_tokens);
}  // namespace dgpp::serve
