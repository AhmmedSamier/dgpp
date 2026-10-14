#include "serve/audio_inputs.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "audio/wav_decode.hpp"
#include "audio/mp3_decode.hpp"
#include "audio/compressed_decode.hpp"
#include "common/base64.hpp"

namespace dgpp::serve {
namespace {
constexpr size_t kMaxEncodedAudioBytes = 20 * 1024 * 1024;  // base64 payload cap

// Zero-extend to the frontend's 0.5 s clip minimum and drop the sub-hop tail:
// torch.stft's frame slice discards the partial last frame, so those samples
// reach no feature.
std::vector<float> align_pcm(std::vector<float>&& pcm) {
  if (pcm.size() < static_cast<size_t>(kAudioMinSamples))
    pcm.resize(static_cast<size_t>(kAudioMinSamples), 0.f);
  pcm.resize(pcm.size() - pcm.size() % kAudioHop);
  if (pcm.size() > static_cast<size_t>(kMaxAudioSamples))
    throw AudioInputError("audio clip exceeds the 30 minute admission bound", "input_audio");
  return pcm;
}

std::vector<uint8_t> from_data_uri(std::string_view s, const std::string& param) {
  if (!s.starts_with("data:"))
    throw AudioInputError("use a base64 data URI; remote audio URLs are not supported", param);
  const size_t comma = s.find(',');
  if (comma == std::string_view::npos || !s.substr(0, comma).ends_with(";base64"))
    throw AudioInputError("audio data URI must be base64-encoded", param);
  const auto data = decode_base64(s.substr(comma + 1), kMaxEncodedAudioBytes);
  return {data.begin(), data.end()};
}
}  // namespace

AudioInput prepare_audio_bytes(const uint8_t* bytes, size_t size, const std::string& param) {
  try {
    // Container sniff, never guessed from the filename or MIME: RIFF/WAVE
    // decodes as WAV, fLaC as FLAC, OggS as Vorbis (Opus refused by name
    // inside), and everything else goes to the MP3 decoder, whose
    // no-frames refusal names every accepted codec.
    const bool riff = size >= 12 && std::memcmp(bytes, "RIFF", 4) == 0 &&
                      std::memcmp(bytes + 8, "WAVE", 4) == 0;
    const bool flac = !riff && size >= 4 && std::memcmp(bytes, "fLaC", 4) == 0;
    const bool ogg = !riff && !flac && size >= 4 && std::memcmp(bytes, "OggS", 4) == 0;
    audio::WavAudio wav = riff   ? audio::decode_wav(bytes, size)
                          : flac ? audio::decode_flac(bytes, size)
                          : ogg  ? audio::decode_vorbis(bytes, size)
                                 : audio::decode_mp3(bytes, size);
    if (wav.samples.size() / static_cast<size_t>(wav.channels) >
        static_cast<size_t>(kMaxAudioSamples) * 25)  // raw-side sanity, before the resize
      throw AudioInputError("audio clip exceeds the admission bound", param);
    auto pcm = align_pcm(audio::to_mono_16k(wav));
    AudioInput au;
    au.frames = static_cast<int64_t>(pcm.size()) / kAudioHop;
    au.tokens = audio_token_count(au.frames);
    if (au.tokens > kMaxAudioTokens)
      throw AudioInputError("audio yields more encoder tokens than the request limit", param);
    au.pcm = std::move(pcm);
    return au;
  } catch (const AudioInputError&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw AudioInputError(e.what(), param);
  }
}

AudioInput prepare_audio(const minijson::Value& part, const std::string& param) {
  const auto* type = part.find("type");
  std::vector<uint8_t> bytes;
  try {
    if (type && type->is_string() && type->as_string() == "audio_url") {
      const auto* obj = part.find("audio_url");
      const auto* url = obj && obj->is_object() ? obj->find("url") : nullptr;
      if (!url || !url->is_string())
        throw AudioInputError("audio_url must contain a string url", param + ".url");
      if (!url->as_string().starts_with("data:audio/"))
        throw AudioInputError("audio_url must be a data:audio/...;base64 URI", param + ".url");
      bytes = from_data_uri(url->as_string(), param + ".url");
    } else {
      const auto* obj = part.find("input_audio");
      if (!obj || !obj->is_object())
        throw AudioInputError("input_audio must be an object", param + ".input_audio");
      if (const auto* fmt = obj->find("format");
          fmt && (!fmt->is_string() || (fmt->as_string() != "wav" && fmt->as_string() != "mp3" &&
                                        fmt->as_string() != "flac" && fmt->as_string() != "ogg")))
        throw AudioInputError("input_audio.format: serve decodes wav, mp3, flac and ogg",
                              param + ".input_audio.format");
      const auto* data = obj->find("data");
      if (!data || !data->is_string())
        throw AudioInputError("input_audio.data must be a base64 string",
                              param + ".input_audio.data");
      const auto s = data->as_string();
      bytes = s.starts_with("data:") ? from_data_uri(s, param + ".input_audio.data")
                                     : [&] {
                                         const auto v = decode_base64(s, kMaxEncodedAudioBytes);
                                         return std::vector<uint8_t>(v.begin(), v.end());
                                       }();
    }
    return prepare_audio_bytes(bytes.data(), bytes.size(), param);
  } catch (const AudioInputError&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw AudioInputError(e.what(), param);
  }
}
}  // namespace dgpp::serve
