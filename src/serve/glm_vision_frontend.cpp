#include "serve/glm_vision_frontend.hpp"

namespace dgpp::serve {
VisionSpec glm_vision_spec(ImageTokens ids) {
  // The GLM-5.3-Flash checkpoint config carries no image token ids, so fall
  // back to the ids its tokenizer ships.
  if (ids.pad < 0) ids = ImageTokens{154830, 154854, 154831};
  return VisionSpec{ids, "<|begin_of_image|>", "<|image|>", "<|end_of_image|>",
                    ImagePreprocess{kGlmImageGrid, 16, kMaxImageTokens, true, 256}};
}
}  // namespace dgpp::serve