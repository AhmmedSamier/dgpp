#pragma once
// FLAC and Ogg Vorbis decoding (docs/audio.md), beside RIFF/WAVE
// (`audio/wav_decode.hpp`) and MP3 (`audio/mp3_decode.hpp`). The decoders
// are vendored single headers — dr_flac (`third_party/dr_flac/`,
// `docs/licenses/dr_flac.md`) and stb_vorbis (`third_party/stb_vorbis/`,
// `docs/licenses/stb_vorbis.md`) — so this adds no build or runtime
// dependency. Output feeds the same contract: native-rate float samples
// that callers downmix and resample with `to_mono_16k`. Opus is refused
// by name (no single-header decoder exists; it needs libopus).

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "audio/wav_decode.hpp"

namespace dgpp::audio {

// Thrown with the offending condition in the message; the serve layer maps
// these to the 400 naming the field, like WavError.
struct FlacError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};
struct VorbisError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// Decodes a FLAC stream ("fLaC" magic) to native-rate float samples.
// Refuses undecodable bytes, absurd rates, and clips past the admission
// bound; the cap mirrors the WAV path (the fastest FLAC rates run ~24x
// the 16 kHz bound).
WavAudio decode_flac(const uint8_t* data, size_t size);

// Decodes an Ogg Vorbis stream ("OggS" magic) to native-rate float
// samples. Ogg-encapsulated Opus is refused by name rather than guessed
// at; anything else undecodable is refused as non-Vorbis.
WavAudio decode_vorbis(const uint8_t* data, size_t size);

}  // namespace dgpp::audio
