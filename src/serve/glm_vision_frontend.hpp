#pragma once

#include "serve/frontend.hpp"

namespace dgpp::serve {
// The checkpoint's text template is retained for roles/tools/reasoning. Image
// parts are rendered as GLM's trained delimiters and expanded patch tokens.
class GlmVisionFrontend : public TextFrontend {
 public:
  GlmVisionFrontend(const text::Tokenizer* tok, const text::ChatTemplate* tpl);
  bool supports_images() const override { return true; }
  ChatInput prepare_chat(const minijson::Value& globals) const override;
};
}  // namespace dgpp::serve
