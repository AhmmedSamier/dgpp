#include "serve/qwen_asr_audio_frontend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/audio_input.hpp"
#include "serve/audio_inputs.hpp"

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

ModelFrontend::ChatInput QwenAsrAudioFrontend::prepare_chat(const minijson::Value& globals) const {
  using V = minijson::Value;
  const AudioSpec& aspec = spec();
  ChatInput out;
  std::vector<V> messages;
  size_t audio_bytes = 0;
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
        const bool input = type && type->is_string() && type->as_string() == "input_audio",
                   url = type && type->is_string() && type->as_string() == "audio_url";
        if (!input && !url) {
          parts.push_back(part);
          continue;
        }
        const std::string where = "messages[" + std::to_string(i) + "].content[" +
                                  std::to_string(j) + "]." + (input ? "input_audio" : "audio_url");
        auto audio = prepare_audio(part, where);
        const size_t bytes = audio.pcm.size() * sizeof(float);
        if (bytes > kMaxRequestAudioBytes - audio_bytes)
          throw AudioInputError("decoded audio data exceeds the 256 MiB request byte limit", where);
        audio_bytes += bytes;
        // The template counts audio parts itself (c.type == 'audio'); the
        // bulky base64 rides out.audios, not the template globals.
        parts.push_back(V::make_object({{"type", V::make_string("audio")}}));
        out.audios.push_back(std::move(audio));
      }
      fields.push_back({field.key, V::make_array(std::move(parts))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  const std::vector<int64_t> rendered = encode_text(render_chat(V::make_object(std::move(fields))));
  if (out.audios.empty()) {
    out.tokens = rendered;
    return out;
  }
  // The template emits one pad per audio part; expand it to the clip's row
  // count in place (the HF processor's replace_audio_token, post-render).
  std::vector<int64_t> expanded;
  expanded.reserve(rendered.size());
  size_t pos = 0;
  for (auto& au : out.audios) {
    while (pos < rendered.size() && rendered[pos] != aspec.ids.start) {
      if (rendered[pos] == aspec.ids.pad)
        throw AudioInputError("extra audio markers in message text", "messages");
      expanded.push_back(rendered[pos++]);
    }
    if (pos >= rendered.size())
      throw AudioInputError("audio token expansion did not match audio inputs", "messages");
    expanded.push_back(rendered[pos++]);  // start
    if (pos >= rendered.size() || rendered[pos] != aspec.ids.pad)
      throw AudioInputError("audio token expansion did not match audio inputs", "messages");
    au.offset = static_cast<int64_t>(expanded.size());
    for (int k = 0; k < au.tokens; ++k) expanded.push_back(aspec.ids.pad);
    ++pos;  // consume the template's single pad
    if (pos >= rendered.size() || rendered[pos] != aspec.ids.end)
      throw AudioInputError("audio token expansion did not match audio inputs", "messages");
    expanded.push_back(rendered[pos++]);  // end
  }
  for (; pos < rendered.size(); ++pos) {
    if (rendered[pos] == aspec.ids.pad)
      throw AudioInputError("extra audio markers in message text", "messages");
    expanded.push_back(rendered[pos]);
  }
  out.tokens = std::move(expanded);
  validate_audio_inputs(out.audios, out.tokens.size());
  return out;
}
}  // namespace dgpp::serve
