#include "serve/glm_vision_frontend.hpp"

#include "serve/image_inputs.hpp"

namespace dgpp::serve {
GlmVisionFrontend::GlmVisionFrontend(const text::Tokenizer* tok, const text::ChatTemplate* tpl)
    : TextFrontend(tok, tpl) {
  if (encode_text("<|begin_of_image|><|image|><|end_of_image|>") !=
      std::vector<int64_t>{154830, 154854, 154831})
    throw std::invalid_argument("GLM vision: checkpoint image token ids are incompatible");
}
ModelFrontend::ChatInput GlmVisionFrontend::prepare_chat(const minijson::Value& globals) const {
  using V = minijson::Value;
  ChatInput out;
  std::vector<V> messages;
  int total_tokens = 0;
  const auto& original = globals.at("messages").items();
  for (size_t i = 0; i < original.size(); ++i) {
    const auto& msg = original[i];
    std::vector<minijson::Member> fields;
    for (const auto& field : msg.members()) {
      if (field.key != "content" || !field.value.is_array()) {
        fields.push_back(field);
        continue;
      }
      std::vector<V> parts;
      for (size_t j = 0; j < field.value.items().size(); ++j) {
        const auto& part = field.value.items()[j];
        const auto* type = part.find("type");
        if (!type || !type->is_string() || type->as_string() != "image_url") {
          parts.push_back(part);
          continue;
        }
        const std::string where =
            "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].image_url";
        if (out.images.size() >= kMaxInputImages)
          throw ImageInputError("too many input images", where);
        auto image = prepare_glm_image(part.at("image_url"), where);
        total_tokens += image.tokens;
        if (total_tokens > kMaxRequestImageTokens)
          throw ImageInputError("too many image tokens in request", where);
        std::string placeholder = "<|begin_of_image|>";
        for (int k = 0; k < image.tokens; ++k) placeholder += "<|image|>";
        placeholder += "<|end_of_image|>";
        parts.push_back(V::make_object({{"type", V::make_string("text")},
                                        {"text", V::make_owned_string(std::move(placeholder))}}));
        out.images.push_back(std::move(image));
      }
      fields.push_back({field.key, V::make_array(std::move(parts))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  out.tokens = encode_text(render_chat(V::make_object(std::move(fields))));
  if (out.images.empty()) return out;
  size_t pos = 0;
  for (auto& im : out.images) {
    while (pos < out.tokens.size() && out.tokens[pos] != 154854) ++pos;
    im.offset = static_cast<int64_t>(pos);
    if (pos == 0 || out.tokens[pos - 1] != 154830)
      throw ImageInputError("image markers in message text conflict with image inputs", "messages");
    for (int j = 0; j < im.tokens; ++j)
      if (pos >= out.tokens.size() || out.tokens[pos++] != 154854)
        throw ImageInputError("image token expansion did not match image inputs", "messages");
    if (pos >= out.tokens.size() || out.tokens[pos] != 154831)
      throw ImageInputError("image token expansion did not match image inputs", "messages");
  }
  for (; pos < out.tokens.size(); ++pos)
    if (out.tokens[pos] == 154854)
      throw ImageInputError("extra image markers in message text", "messages");
  validate_image_inputs(out.images, out.tokens.size());
  return out;
}
}  // namespace dgpp::serve
