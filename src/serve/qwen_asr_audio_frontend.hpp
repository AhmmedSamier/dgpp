#pragma once
// The Qwen3-ASR frontend: the shared AudioFrontend (serve/audio_frontend.hpp)
// with the checkpoint's three audio marker ids (audio_start_token_id,
// audio_token_id repeated once per encoder token, audio_end_token_id). Their
// surface forms are searched in the tokenizer's added-token table by id, as
// with the vision frontends, so a checkpoint that renumbers the delimiters
// needs no code change; one whose vocabulary cannot render a marker fails the
// frontend constructor's round-trip check at startup.
//
// Unlike the shared frontend, prepare_chat is overridden: this checkpoint's
// template counts audio parts itself and emits exactly one pad token per
// part (the HF processor's replace_audio_token expands that single pad to
// the encoder row count after rendering). Audio parts therefore pass through
// the template as bare {"type": "audio"} markers and the single pad is
// expanded to the clip's row count in the encoded token stream.

#include "serve/audio_frontend.hpp"

namespace dgpp::serve {
class QwenAsrAudioFrontend : public AudioFrontend {
 public:
  QwenAsrAudioFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                        AudioTokens ids);
  ChatInput prepare_chat(const minijson::Value& globals) const override;
};
}  // namespace dgpp::serve
