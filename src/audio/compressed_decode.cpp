#include "audio/compressed_decode.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "common/audio_input.hpp"

#define DR_FLAC_IMPLEMENTATION
#include "../../third_party/dr_flac/dr_flac.h"

#define STB_VORBIS_NO_STDIO
#include "../../third_party/stb_vorbis/stb_vorbis.c"

namespace dgpp::audio {
namespace {

// Native-rate frame bound shared by the lossless/container codecs: the
// fastest credible rate (384 kHz) runs 24x the 16 kHz admission bound,
// plus one bound of margin. The shared align step still enforces the
// exact cap afterwards.
constexpr int64_t kMaxLosslessFrames = kMaxAudioSamples * 25;

void check_geometry(int rate, int channels, const char* what) {
  if (channels < 1 || channels > 8)
    throw std::invalid_argument(std::string(what) + " has an unsupported channel count");
  if (rate < 8000 || rate > 384000)
    throw std::invalid_argument(std::string(what) + " has an unsupported sample rate");
}

bool sniff_opus(const uint8_t* data, size_t size) {
  // The Opus BOS page carries "OpusHead" where Vorbis carries "\1vorbis":
  // scan the first pages only, never the audio.
  const size_t window = size < 4096 ? size : 4096;
  const std::string_view head(reinterpret_cast<const char*>(data), window);
  return head.find("OpusHead") != std::string_view::npos ||
         head.find("OpusTags") != std::string_view::npos;
}

}  // namespace

WavAudio decode_flac(const uint8_t* data, size_t size) {
  if (data == nullptr || size < 4 || std::memcmp(data, "fLaC", 4) != 0)
    throw FlacError("audio is not a FLAC stream; serve accepts WAV, MP3, FLAC and Ogg Vorbis");
  drflac* flac = drflac_open_memory(data, size, nullptr);
  if (flac == nullptr)
    throw FlacError("FLAC stream is undecodable; the file may be truncated");
  const int rate = static_cast<int>(flac->sampleRate);
  const int channels = static_cast<int>(flac->channels);
  try {
    check_geometry(rate, channels, "FLAC stream");
  } catch (const std::invalid_argument& e) {
    drflac_close(flac);
    throw FlacError(e.what());
  }
  WavAudio out;
  out.sample_rate = rate;
  out.channels = channels;
  std::vector<float> frame(4096 * static_cast<size_t>(channels));
  for (;;) {
    const drflac_uint64 got =
        drflac_read_pcm_frames_f32(flac, 4096, frame.data());
    if (got == 0) break;
    const size_t decoded = static_cast<size_t>(got) * static_cast<size_t>(channels);
    if (decoded / static_cast<size_t>(channels) != static_cast<size_t>(got) ||
        (out.samples.size() + decoded) / static_cast<size_t>(channels) >
            static_cast<size_t>(kMaxLosslessFrames)) {
      drflac_close(flac);
      throw FlacError("FLAC audio exceeds the 30 minute admission bound");
    }
    out.samples.insert(out.samples.end(), frame.begin(), frame.begin() + decoded);
  }
  drflac_close(flac);
  if (out.samples.empty())
    throw FlacError("FLAC stream holds no audio frames");
  return out;
}

WavAudio decode_vorbis(const uint8_t* data, size_t size) {
  if (data == nullptr || size < 4 || std::memcmp(data, "OggS", 4) != 0)
    throw VorbisError(
        "audio is not an Ogg Vorbis stream; serve accepts WAV, MP3, FLAC and Ogg Vorbis");
  if (sniff_opus(data, size))
    throw VorbisError("serve does not decode Opus; submit WAV, MP3, FLAC or Vorbis");
  if (size > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw VorbisError("Ogg Vorbis stream exceeds the decoder's size bound");
  int error = 0;
  stb_vorbis* vorbis =
      stb_vorbis_open_memory(data, static_cast<int>(size), &error, nullptr);
  if (vorbis == nullptr)
    throw VorbisError("Ogg Vorbis stream is undecodable (code " + std::to_string(error) +
                      "); the file may be truncated or Opus");
  const stb_vorbis_info info = stb_vorbis_get_info(vorbis);
  const int rate = info.sample_rate;
  const int channels = info.channels;
  try {
    check_geometry(rate, channels, "Ogg Vorbis stream");
  } catch (const std::invalid_argument& e) {
    stb_vorbis_close(vorbis);
    throw VorbisError(e.what());
  }
  WavAudio out;
  out.sample_rate = rate;
  out.channels = channels;
  std::vector<float> frame(2048 * static_cast<size_t>(channels));
  for (;;) {
    const int got = stb_vorbis_get_samples_float_interleaved(
        vorbis, channels, frame.data(), static_cast<int>(frame.size()));
    if (got <= 0) break;
    const size_t decoded = static_cast<size_t>(got) * static_cast<size_t>(channels);
    if ((out.samples.size() + decoded) / static_cast<size_t>(channels) >
        static_cast<size_t>(kMaxLosslessFrames)) {
      stb_vorbis_close(vorbis);
      throw VorbisError("Ogg Vorbis audio exceeds the 30 minute admission bound");
    }
    out.samples.insert(out.samples.end(), frame.begin(), frame.begin() + decoded);
  }
  stb_vorbis_close(vorbis);
  if (out.samples.empty())
    throw VorbisError("Ogg Vorbis stream holds no audio frames");
  return out;
}
}  // namespace dgpp::audio
