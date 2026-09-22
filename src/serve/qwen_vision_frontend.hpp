#pragma once
// Qwen3.8-Flash-Next's image frontend: the shared VisionFrontend with the
// checkpoint's three vision marker ids and the 32-pixel smart-resized canvas
// (serve/vision_frontend.hpp). The markers' surface forms come from the
// tokenizer's own vocabulary, so a checkpoint that renumbers them needs no
// code change; one whose vocabulary cannot render a marker fails the
// frontend's round-trip check at startup instead of sending a garbled prompt
// to the model.
#include "serve/vision_frontend.hpp"

namespace dgpp::serve {
// Qwen3.8-Flash-Next: the 32-pixel canvas, rounded to the nearest multiple of
// the grid and never padded, delimited by the checkpoint's vision_start,
// image_token and vision_end ids. Their surface forms come from the
// tokenizer's vocabulary rather than a literal, so the ids stay authoritative;
// the frontend constructor still checks that each one round-trips to exactly
// its id.
VisionSpec qwen_vision_spec(dgpp::ImageTokens ids, const dgpp::text::Tokenizer& tok);

class QwenVisionFrontend : public VisionFrontend {
 public:
  QwenVisionFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                     ImageTokens ids)
      : VisionFrontend(tok, tpl, qwen_vision_spec(ids, *tok)) {}
};
}  // namespace dgpp::serve
