#pragma once
// The Qwen3-ASR frontend: the shared AudioFrontend (serve/audio_frontend.hpp)
// with the checkpoint's three audio marker ids (audio_start_token_id,
// audio_token_id repeated once per encoder token, audio_end_token_id). Their
// surface forms are searched in the tokenizer's added-token table by id, as
// with the vision frontends, so a checkpoint that renumbers the delimiters
// needs no code change; one whose vocabulary cannot render a marker fails the
// frontend constructor's round-trip check at startup.

#include "serve/audio_frontend.hpp"

namespace dgpp::serve {
class QwenAsrAudioFrontend : public AudioFrontend {
 public:
  QwenAsrAudioFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                       AudioTokens ids);
};
}  // namespace dgpp::serve
