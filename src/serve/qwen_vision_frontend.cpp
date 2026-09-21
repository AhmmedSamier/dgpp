#include "serve/qwen_vision_frontend.hpp"

namespace dgpp::serve {
VisionSpec qwen_vision_spec(dgpp::ImageTokens ids, const dgpp::text::Tokenizer& tok) {
  if (ids.pad < 0)
    throw std::invalid_argument("Qwen vision: the checkpoint's image token ids are unset");
  // The markers are control tokens: they live in the tokenizer's added-token
  // table, past the base vocabulary, so token_by_id() (which covers only the
  // vocab) cannot see them. Search the added tokens by id.
  const auto marker = [&](int64_t id, const char* what) {
    for (const auto& added : tok.added_tokens())
      if (added.id == id && !added.content.empty()) return added.content;
    throw std::invalid_argument(std::string("Qwen vision: ") + what +
                                " is not a tokenizer entry the template can render");
  };
  return VisionSpec{ids,
                    marker(ids.start, "vision_start_token_id"),
                    marker(ids.pad, "image_token_id"),
                    marker(ids.end, "vision_end_token_id"),
                    ImagePreprocess{kQwenImageGrid, 64, kMaxImageTokens, false, 256}};
}
}  // namespace dgpp::serve