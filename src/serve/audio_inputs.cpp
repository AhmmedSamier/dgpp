#include "serve/audio_inputs.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "audio/wav_decode.hpp"
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
          fmt && (!fmt->is_string() || fmt->as_string() != "wav"))
        throw AudioInputError("input_audio.format: serve decodes wav only",
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
    auto wav = audio::decode_wav(bytes.data(), bytes.size());
    if (wav.samples.size() / static_cast<size_t>(wav.channels) >
        static_cast<size_t>(kMaxAudioSamples) * 25)  // raw-side sanity, before the resize
      throw AudioInputError("audio clip exceeds the admission bound", "input_audio");
    auto pcm = align_pcm(audio::to_mono_16k(wav));
    AudioInput au;
    au.frames = static_cast<int64_t>(pcm.size()) / kAudioHop;
    au.tokens = audio_token_count(au.frames);
    if (au.tokens > kMaxAudioTokens)
      throw AudioInputError("audio yields more encoder tokens than the request limit",
                            "input_audio");
    au.pcm = std::move(pcm);
    return au;
  } catch (const AudioInputError&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw AudioInputError(e.what(), param);
  }
}
}  // namespace dgpp::serve
