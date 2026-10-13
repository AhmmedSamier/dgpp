#include "serve/qwen_asr_audio_frontend.hpp"

#include <stdexcept>
#include <string>

namespace dgpp::serve {
QwenAsrAudioFrontend::QwenAsrAudioFrontend(const dgpp::text::Tokenizer* tok,
                                           const dgpp::text::ChatTemplate* tpl, AudioTokens ids)
    : AudioFrontend(tok, tpl, [tok, ids] {
        if (ids.pad < 0)
          throw std::invalid_argument("Qwen ASR: the checkpoint's audio token ids are unset");
        // The delimiters are control tokens living in the added-token table
        // past the base vocabulary, so they are searched by id, exactly as
        // qwen_vision_spec resolves its vision markers.
        const auto marker = [&](int64_t id, const char* what) {
          for (const auto& added : tok->added_tokens())
            if (added.id == id && !added.content.empty()) return added.content;
          throw std::invalid_argument(std::string("Qwen ASR: ") + what +
                                      " is not a tokenizer entry the template can render");
        };
        return AudioSpec{ids,
                         marker(ids.start, "audio_start_token_id"),
                         marker(ids.pad, "audio_token_id"),
                         marker(ids.end, "audio_end_token_id")};
      }()) {}
}  // namespace dgpp::serve
