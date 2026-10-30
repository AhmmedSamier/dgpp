#pragma once
// MP3 container decoding (docs/audio.md), the second accepted upload codec
// beside RIFF/WAVE (`audio/wav_decode.hpp`). MP3 is the format OpenWebUI and
// the OpenAI SDKs submit (`POST /v1/audio/transcriptions` chunks arrive as
// mp3); the decoder is the vendored single-header minimp3
// (`third_party/minimp3/`, `docs/licenses/minimp3.md`), so this adds no build
// or runtime dependency. Output feeds the same contract as WAV: native-rate
// float samples that callers downmix and resample with `to_mono_16k`.
//
// The decode is lossy-audio tolerant but container-strict: MPEG-1/2/2.5
// layers 1-3 at 8-48 kHz mono/stereo decode; mid-stream rate/channel
// changes, corrupt frames past a bound, and inputs with no MP3 frames at
// all are refused. LAME/Xing gapless tags are honored when present (see
// gapless_trim): encoder delay and end padding are dropped so clip
// durations match their WAV masters; untagged streams keep their delay
// as audio (the frontend zero-pads short clips anyway).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "audio/wav_decode.hpp"

namespace dgpp::audio {

// Thrown with the offending condition in the message; the serve layer maps
// it to the 400 naming the field, like WavError.
struct Mp3Error : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// Decodes an MP3 stream (ID3v2-tagged or raw frames) to native-rate float
// samples in `WavAudio` (sample_rate/channels/samples). Trailing tag bytes
// (ID3v1, APE) are skipped; anything else undecodable past a small bound
// fails rather than silently truncating the clip.
WavAudio decode_mp3(const uint8_t* data, size_t size);

// Gapless metadata (the LAME tag some encoders write into the first
// frame): {encoder delay, end padding} in samples, when present. Pure
// parser over the first frame — the unit tests feed it synthetic frames.
// Absent, malformed, or insane tags yield nullopt and the stream keeps
// its delay as audio.
std::optional<std::pair<size_t, size_t>> gapless_trim(const uint8_t* data, size_t size);

}  // namespace dgpp::audio
