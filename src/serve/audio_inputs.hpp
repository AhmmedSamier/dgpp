#pragma once
// Inline WAV audio for the ASR family (docs/audio.md): the audio counterpart
// of prepare_image. Decoding is intentionally narrow - a base64 WAV, the
// format every ASR client already has - and every step that will be redone
// on the GPU (downmix, resample, mel) is NOT done here: this produces the
// trained contract, 16 kHz hop-aligned mono and its encoder token count.
//
// Accepted content parts:
//   {"type": "input_audio", "input_audio": {"data": "<b64 wav>", "format": "wav"}}
//   {"type": "audio_url",   "audio_url":   {"url": "data:audio/wav;base64,<b64>"}}
// Remote URLs are refused, as with images: resolution does not belong on the
// event-loop path.

#include <string>

#include "common/audio_input.hpp"
#include "loaders/minijson.hpp"

namespace dgpp::serve {
struct AudioInputError : std::invalid_argument {
  AudioInputError(std::string message, std::string param)
      : std::invalid_argument(std::move(message)), param(std::move(param)) {}
  std::string param;
};

// `part` is the content-part object; the branch is by its "type" member.
// Decoded payloads are capped at 20 MiB base64 (images' cap), and clips at
// half an hour of 16 kHz audio.
AudioInput prepare_audio(const minijson::Value& part, const std::string& param);

// Raw container bytes (the transcription endpoint's uploaded file) through
// the same decode/align/count contract; errors name `param` (e.g. "file").
AudioInput prepare_audio_bytes(const uint8_t* bytes, size_t size, const std::string& param);
}  // namespace dgpp::serve
