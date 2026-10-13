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
  MultipartError(std::string message, std::string param, int status = 400)
      : std::invalid_argument(std::move(message)), param(std::move(param)), status(status) {}
  std::string param;
  int status;  // the HTTP status the route answers (400, or 404 for model)
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

// Validates the transcription fields (400s name the multipart field) and
// folds them into the chat-completions body the engine serves: a user
// message with one input_audio part (base64 WAV bytes), the prompt as the
// system message when present, a duration-proportional max_tokens, and no
// streaming. Throws MultipartError (bad fields) or AudioInputError
// (undecodable audio, "file").
struct TranscriptionRequest {
  minijson::Value body;
  std::string response_format;  // "json" or "text"
};

TranscriptionRequest transcription_chat_body(const std::vector<MultipartPart>& parts,
                                             const std::string& model_id);
}  // namespace dgpp::serve
