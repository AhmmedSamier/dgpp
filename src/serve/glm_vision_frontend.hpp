#pragma once
// GLM-5.3-Flash's image frontend: the shared VisionFrontend with GLM's
// delimiter ids and the 28-pixel padded canvas (serve/vision_frontend.hpp).
#include "serve/vision_frontend.hpp"

namespace dgpp::serve {
// GLM-5.3-Flash: the 28-pixel canvas with black padding and GLM's trained
// <|begin_of_image|> / <|image|> / <|end_of_image|> delimiters.
VisionSpec glm_vision_spec(ImageTokens ids);

class GlmVisionFrontend : public VisionFrontend {
 public:
  GlmVisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                    ImageTokens ids = ImageTokens{})
      : VisionFrontend(tok, tpl, glm_vision_spec(ids)) {}
};
}  // namespace dgpp::serve
