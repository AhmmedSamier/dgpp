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

// ---------------------------------------------------------------------------
// Nemotron-3.5-Transcribe audio frontend (milestone 6)
// ---------------------------------------------------------------------------

namespace {

// The literal text that the checkpoint's template emits per audio clip. The
// tokenizer maps it to id == vocab_size, the splice marker the model swaps
// for perception rows.
constexpr char kAudioLocatorText[] = "<|audio|>";

}  // namespace

namespace dgpp::serve {

ModelFrontend::ChatInput NemotronAudioFrontend::prepare_chat(
    const minijson::Value& globals) const {
  using V = minijson::Value;
  ChatInput out;
  std::vector<V> messages;
  size_t audio_bytes = 0;
  const auto& original = globals.at("messages").items();
  for (size_t i = 0; i < original.size(); ++i) {
    const auto& msg = original[i];
    std::vector<minijson::Member> fields;
    for (const auto& field : msg.members()) {
      if (field.key != "content") {
        fields.push_back(field);
        continue;
      }
      // Nemotron contract: content is a single string. Concatenate text parts
      // and replace each <|audio|> part with the literal locator text.
      std::string content_text;
      if (field.value.is_array()) {
        for (size_t j = 0; j < field.value.items().size(); ++j) {
          const auto& part = field.value.items()[j];
          const auto* type = part.find("type");
          const bool input = type && type->is_string() && type->as_string() == "input_audio",
                     url = type && type->is_string() && type->as_string() == "audio_url";
          const auto* text = part.find("text");
          if (!input && !url && text && text->is_string()) {
            if (!content_text.empty() && !text->as_string().empty())
              content_text += " ";
            content_text += text->as_string();
            continue;
          }
          if (input || url) {
            const std::string where = "messages[" + std::to_string(i) + "].content[" +
                                      std::to_string(j) + "]." + (input ? "input_audio" : "audio_url");
            auto audio = prepare_audio(part, where);
            const size_t bytes = audio.pcm.size() * sizeof(float);
            if (bytes > kMaxRequestAudioBytes - audio_bytes)
              throw AudioInputError("decoded audio data exceeds the 256 MiB request byte limit", where);
            audio_bytes += bytes;
            if (!content_text.empty())
              content_text += " ";
            content_text += kAudioLocatorText;
            out.audios.push_back(std::move(audio));
            continue;
          }
          // Unknown part type: include its text if any.
          if (text && text->is_string()) {
            if (!content_text.empty())
              content_text += " ";
            content_text += text->as_string();
          }
        }
      } else if (field.value.is_string()) {
        content_text = field.value.as_string();
      }
      fields.push_back({field.key, V::make_owned_string(std::move(content_text))});
    }
    messages.push_back(V::make_object(std::move(fields)));
  }
  std::vector<minijson::Member> fields;
  for (const auto& f : globals.members())
    fields.push_back(f.key == "messages" ? minijson::Member{f.key, V::make_array(messages)} : f);
  out.tokens = encode_text(render_chat(V::make_object(std::move(fields))));
  if (out.audios.empty()) return out;
  // Expand each <|audio|> locator (1 token, id == vocab_size) into its full
  // span of T_tok placeholder tokens so that validate_audio_inputs sees
  // offset + tokens <= prompt_tokens. The model's walk() later swaps these
  // placeholder rows for the clip's perception embeddings.
  const int64_t locator_id = static_cast<int64_t>(encode_text(kAudioLocatorText).front());
  std::vector<int64_t> expanded;
  expanded.reserve(out.tokens.size());
  size_t au_idx = 0;
  for (size_t i = 0; i < out.tokens.size(); ++i) {
    if (out.tokens[i] == locator_id && au_idx < out.audios.size()) {
      const int t_tok = out.audios[au_idx].tokens;
      for (int k = 0; k < t_tok; ++k) expanded.push_back(0);  // placeholder id
      out.audios[au_idx].offset = static_cast<int64_t>(expanded.size()) - t_tok;
      ++au_idx;
    } else {
      expanded.push_back(out.tokens[i]);
    }
  }
  out.tokens = std::move(expanded);
  // Locate each <|audio|> span in the expanded token stream.
  for (auto& au : out.audios) {
    if (static_cast<size_t>(au.offset) >= out.tokens.size())
      throw AudioInputError("audio locator missing from rendered prompt", "messages");
  }
  validate_audio_inputs(out.audios, out.tokens.size());
  return out;
}

}  // namespace dgpp::serve
