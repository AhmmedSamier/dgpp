#pragma once
// Container-level audio decoding (docs/audio.md). Inline audio arrives as a
// base64 WAV: the canonical interchange format for ASR, and the one format
// whose decode is exact integer/IEEE arithmetic with no library dependency.
// Other codecs (mp3, opus, flac) and non-WAV containers are refused by name
// rather than guessed at; callers convert samples to 16 kHz mono.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dgpp::audio {

struct WavAudio {
  int sample_rate = 0;
  int channels = 0;  // 1..8, interleaved in `samples`
  std::vector<float> samples;  // roughly [-1, 1]
};

// Thrown with the offending chunk or field in the message.
struct WavError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

// RIFF/WAVE with PCM integer (8/16/24/32 bit) or IEEE float (32/64 bit)
// samples. Unknown chunks are skipped; WAVE_FORMAT_EXTENSIBLE is accepted
// when its subformat is plain PCM or IEEE float.
WavAudio decode_wav(const uint8_t* data, size_t size);

// Downmixes to mono (channel average) and resamples to 16 kHz with a linear
// interpolator. Quality note: the reference frontend resamples other rates
// with torchaudio's windowed sinc; linear is a deliberate v1 simplification,
// transparent in the output (it attenuates only the 8 kHz edge).
std::vector<float> to_mono_16k(const WavAudio& wav);
}  // namespace dgpp::audio
