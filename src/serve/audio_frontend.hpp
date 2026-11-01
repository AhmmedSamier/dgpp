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

// Nemotron-3.5-Transcribe audio frontend (milestone 6): the checkpoint's
// contract is a single <|audio|> locator token (id == vocab_size) per clip,
// a pure splice marker never embedding-looked-up. There are no start/pad/end
// delimiters — each user audio part is rewritten to the literal <|audio|>
// text before rendering, and the frontend locates the resulting locator ids
// in the token stream to report per-clip spans to the engine.
class NemotronAudioFrontend : public TextFrontend {
 public:
  NemotronAudioFrontend(const dgpp::text::Tokenizer* tok,
                        const dgpp::text::ChatTemplate* tpl)
      : TextFrontend(tok, tpl) {}
  bool supports_audios() const override { return true; }
  ChatInput prepare_chat(const minijson::Value& globals) const override;
};
}  // namespace dgpp::serve
