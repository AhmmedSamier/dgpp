#pragma once
// The real ModelFrontend: the Stage 3/3b exact tokenizer and chat
// template behind the service's seam (M6 Stage 4). The service never
// links CUDA; this adapter is likewise host-only.
//
// Chat rendering: the service's template globals (minijson DOM: the
// normalized OpenAI messages array plus tools / reasoning_effort /
// clear_thinking when the request carried them) convert to the
// template's Value model via glm::Value::from_minijson — member order
// preserved, exactly what the template interpreter consumes — with
// add_generation_prompt=true (this is a generation request).
//
// Markers (M6 6f): the template's reasoning and tool-call tokens looked
// up by text among the tokenizer's added tokens at construction — the
// service's parser keys on their ids, the forced tool_choice prefix on
// their text.
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm_chat_template.hpp"
#include "models/glm_tokenizer.hpp"
#include "models/glm_tool_parser.hpp"
#include "service/generation_service.hpp"

namespace dgpp::service {

class GlmFrontend : public ModelFrontend {
 public:
  GlmFrontend(const dgpp::GlmTokenizer* tok,
              const dgpp::glm::ChatTemplate* tpl)
      : tok_(tok), tpl_(tpl) {
    if (tok_ == nullptr || tpl_ == nullptr)
      throw std::invalid_argument(
          "GlmFrontend: tokenizer and chat template must both be loaded");
    markers_ = dgpp::glm::ChatMarkers::from_tokenizer(*tok_);
  }

  std::vector<int64_t> encode_text(std::string_view text) const override {
    return tok_->encode(text);
  }

  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    // skip_special_tokens=true — the SSE content contract (EOS and the
    // other special tokens never appear in generated text).
    return tok_->decode(ids, /*skip_special_tokens=*/true);
  }

  std::string render_chat(const minijson::Value& globals) const override {
    dgpp::glm::Value::Members members;
    for (const minijson::Member& m : globals.members())
      members.emplace_back(m.key, dgpp::glm::Value::from_minijson(m.value));
    members.emplace_back("add_generation_prompt",
                         dgpp::glm::Value::boolean(true));
    return tpl_->render(dgpp::glm::Value::map_value(std::move(members)));
  }

  dgpp::glm::ChatMarkers markers() const override { return markers_; }
  std::vector<int64_t> boundary_token_ids() const override {
    std::vector<int64_t> ids;
    for (const dgpp::glm::ChatMarker& m : markers_.role_markers) ids.push_back(m.id);
    return ids;
  }

 private:
  const dgpp::GlmTokenizer* tok_;
  const dgpp::glm::ChatTemplate* tpl_;
  dgpp::glm::ChatMarkers markers_;
};

}  // namespace dgpp::service
