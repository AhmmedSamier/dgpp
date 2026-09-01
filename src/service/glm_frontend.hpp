#pragma once
// The real ModelFrontend: the Stage 3/3b exact tokenizer and chat
// template behind the service's seam (M6 Stage 4). The service never
// links CUDA; this adapter is likewise host-only.
//
// Chat rendering: the OpenAI messages array (minijson DOM) converts to
// the template's Value model via glm::Value::from_minijson — member
// order preserved, exactly what the template interpreter consumes —
// with add_generation_prompt=true (this is a generation request).
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm_chat_template.hpp"
#include "models/glm_tokenizer.hpp"
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
  }

  std::vector<int64_t> encode_text(std::string_view text) const override {
    return tok_->encode(text);
  }

  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    // skip_special_tokens=true — the SSE content contract (EOS and the
    // other special tokens never appear in generated text).
    return tok_->decode(ids, /*skip_special_tokens=*/true);
  }

  std::string render_chat(const minijson::Value& messages) const override {
    dgpp::glm::Value::Members globals;
    globals.emplace_back("messages",
                         dgpp::glm::Value::from_minijson(messages));
    globals.emplace_back("add_generation_prompt",
                         dgpp::glm::Value::boolean(true));
    return tpl_->render(dgpp::glm::Value::map_value(std::move(globals)));
  }

 private:
  const dgpp::GlmTokenizer* tok_;
  const dgpp::glm::ChatTemplate* tpl_;
};

}  // namespace dgpp::service
