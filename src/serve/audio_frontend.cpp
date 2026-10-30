#include "serve/audio_frontend.hpp"

#include <utility>

namespace dgpp::serve {
AudioFrontend::AudioFrontend(const dgpp::text::Tokenizer* tok, const dgpp::text::ChatTemplate* tpl,
                             AudioSpec spec)
    : TextFrontend(tok, tpl), spec_(std::move(spec)) {
  const auto single = [&](const std::string& text, int64_t id, const char* what) {
    const auto ids = encode_text(text);
    if (ids.size() != 1 || ids.front() != id)
      throw std::invalid_argument(std::string("audio frontend: ") + what +
                                  " does not tokenize to the checkpoint's audio token id");
  };
  single(spec_.start_text, spec_.ids.start, "the audio start marker");
  single(spec_.pad_text, spec_.ids.pad, "the audio pad token");
  single(spec_.end_text, spec_.ids.end, "the audio end marker");
}

ModelFrontend::ChatInput AudioFrontend::prepare_chat(const minijson::Value& globals) const {
  using V = minijson::Value;
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
        std::string placeholder = spec_.start_text;
        for (int k = 0; k < audio.tokens; ++k) placeholder += spec_.pad_text;
        placeholder += spec_.end_text;
        parts.push_back(V::make_object({{"type", V::make_string("text")},
                                        {"text", V::make_owned_string(std::move(placeholder))}}));
        out.audios.push_back(std::move(audio));
      }
      fields.push_back({field.key, V::make_array(std::move(parts))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  out.tokens = encode_text(render_chat(V::make_object(std::move(fields))));
  if (out.audios.empty()) return out;
  size_t pos = 0;
  for (auto& au : out.audios) {
    while (pos < out.tokens.size() && out.tokens[pos] != spec_.ids.pad) ++pos;
    au.offset = static_cast<int64_t>(pos);
    if (pos == 0 || out.tokens[pos - 1] != spec_.ids.start)
      throw AudioInputError("audio markers in message text conflict with audio inputs", "messages");
    for (int j = 0; j < au.tokens; ++j)
      if (pos >= out.tokens.size() || out.tokens[pos++] != spec_.ids.pad)
        throw AudioInputError("audio token expansion did not match audio inputs", "messages");
    if (pos >= out.tokens.size() || out.tokens[pos] != spec_.ids.end)
      throw AudioInputError("audio token expansion did not match audio inputs", "messages");
  }
  for (; pos < out.tokens.size(); ++pos)
    if (out.tokens[pos] == spec_.ids.pad)
      throw AudioInputError("extra audio markers in message text", "messages");
  validate_audio_inputs(out.audios, out.tokens.size());
  return out;
}
}  // namespace dgpp::serve
