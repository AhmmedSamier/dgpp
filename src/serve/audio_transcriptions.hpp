#pragma once
// POST /v1/audio/transcriptions (docs/audio.md): the OpenAI transcription
// API over the chat engine. The request is multipart/form-data carrying the
// audio file plus the transcription fields; this module parses the parts
// and builds the chat-completions body the engine already serves, so the
// transcription rides validation, scheduling, usage and the error contract
// unchanged. The only new wire shape is the response, selected by
// response_format (json: {"text"}, text: raw transcript).
//
// Language note: the checkpoint auto-detects and prefixes its output with
// "language <NAME>" itself (live-verified: system text does not condition
// this model, and the template drops assistant prefill). `language` is
// therefore validated as a string and documented, not rendered; `prompt`
// becomes the system message, exactly the HF processor's semantics.

#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp::serve {
// One multipart section. `data` views the request body (binary-safe); the
// parser bounds part counts and field sizes, so holding views is safe.
struct MultipartPart {
  std::string name;
  std::string filename;
  std::string content_type;
  std::string_view data;
};

struct MultipartError : std::invalid_argument {
  MultipartError(std::string message, std::string param, int status = 400,
                 std::string code = "")
      : std::invalid_argument(std::move(message)),
        param(std::move(param)),
        status(status),
        code(std::move(code)) {}
  std::string param;
  int status;  // the HTTP status the route answers (400, or 404 for model)
  std::string code;  // the error code the route answers ("" = none)
};

// Strict RFC 2046 framing over the exact boundary: --boundary CRLF,
// part headers, CRLF CRLF, bytes, CRLF --boundary [--] CRLF. LF-only line
// endings are accepted where curl and browsers emit them. At most 32 parts;
// text fields are capped at 8 KiB (the file part rides the server's body
// limit instead).
std::vector<MultipartPart> parse_multipart(std::string_view content_type, std::string_view body);

struct TranscriptionFields {
  std::string_view file_bytes;
  std::string filename;
  std::string model;
  std::string language;  // "" when absent
  std::string prompt;    // "" when absent
  std::string response_format = "json";  // "json" or "text"
  double temperature = 0.0;
  bool has_temperature = false;
};

// Splits the checkpoint's language tag from the transcript body:
// {"language English", "Hmm. ..."} for the model's glued tag, {"", text}
// otherwise. The model emits "language <NAME><Text>" with no separator in
// ~every transcript (2619/2620 test-clean outputs), so the split falls on
// the camelCase boundary: a capitalized word followed at once by an
// uppercase letter. Anything else — spaced tags, lowercase continuations
// (one test-clean output starts "language Englishsaid ..."), missing or
// lowercase tags — passes through unsplit rather than risk eating speech.
std::pair<std::string, std::string> split_language_tag(std::string_view text);

// Renders a transcript for the wire: tag on its own line above the body,
// or the text verbatim when there is no splittable tag.
std::string format_transcription_text(std::string_view text);

// Validates the transcription fields (400s name the multipart field)
// without touching the audio: model, language, prompt, response_format,
// temperature. Throws MultipartError.
TranscriptionFields parse_transcription_fields(const std::vector<MultipartPart>& parts,
                                               const std::string& model_id);

// Folds validated fields plus one clip's WAV bytes into the
// chat-completions body the engine serves: a user message with one
// input_audio part (base64 WAV bytes), the prompt as the system message
// when present, a duration-proportional max_tokens, and no streaming.
// Decodes the clip (throwing AudioInputError naming "file" when it is not
// servable audio) for the duration rule.
struct TranscriptionRequest {
  minijson::Value body;
  std::string response_format;  // "json" or "text"
};

TranscriptionRequest transcription_body_for_wav(const TranscriptionFields& fields,
                                                std::string_view wav_bytes);

TranscriptionRequest transcription_chat_body(const std::vector<MultipartPart>& parts,
                                             const std::string& model_id);

// Long-form transcription (docs/audio.md): splits a clip that cannot fit
// one engine forward into whole-chunk segment bodies, each sized so its
// audio tokens fit token_budget (the engine's prefill window) minus a
// prompt reserve. token_budget <= 0, or a clip that fits, yields one body
// (today's path, unchanged). Throws MultipartError naming "file" when even
// one chunk plus the reserve cannot fit, AudioInputError for undecodable
// audio. Segment token counts sum exactly (chunk-aligned splits), and each
// segment carries its own duration-proportional max_tokens.
std::vector<TranscriptionRequest> transcription_segment_bodies(
    const std::vector<MultipartPart>& parts, const std::string& model_id,
    int64_t token_budget);
}  // namespace dgpp::serve
