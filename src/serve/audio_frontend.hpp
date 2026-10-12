#pragma once
// The shared audio frontend (docs/audio.md), the audio counterpart of
// VisionFrontend. The checkpoint's Jinja template renders roles and turns
// untouched; each user audio part is rewritten to the trained delimiters -
// start, one pad token per encoder token, end - before rendering, exactly as
// the HF processor expands its single placeholder. The family spec carries
// the three delimiter ids and the text each must tokenize to.

#include <string>

#include "common/audio_input.hpp"
#include "serve/audio_inputs.hpp"
#include "serve/frontend.hpp"

namespace dgpp::serve {
struct AudioSpec {
  AudioTokens ids;
  std::string start_text, pad_text, end_text;
};

class AudioFrontend : public TextFrontend {
 public:
  AudioFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                AudioSpec spec);
  bool supports_audios() const override { return true; }
  ChatInput prepare_chat(const minijson::Value& globals) const override;
  const AudioSpec& spec() const { return spec_; }

 private:
  AudioSpec spec_;
};
}  // namespace dgpp::serve
