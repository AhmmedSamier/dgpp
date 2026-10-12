#include "audio/wav_decode.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "common/audio_input.hpp"

namespace dgpp::audio {
namespace {

uint32_t rd_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 |
         static_cast<uint32_t>(p[2]) << 16 | static_cast<uint32_t>(p[3]) << 24;
}
uint16_t rd_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | static_cast<uint16_t>(p[1]) << 8;
}
bool is_id(const uint8_t* p, const char* id) { return std::memcmp(p, id, 4) == 0; }

// One sample per channel, in [-1, 1] by construction of the scale.
void decode_block(const uint8_t* p, size_t frames, int channels, int format, int bits,
                  std::vector<float>& out) {
  const size_t samples = frames * static_cast<size_t>(channels);
  const size_t base = out.size();
  out.resize(base + samples);
  float* d = out.data() + base;
  if (format == 3) {  // IEEE float
    if (bits == 32)
      for (size_t i = 0; i < samples; ++i)
        std::memcpy(d + i, p + 4 * i, sizeof(float));
    else
      for (size_t i = 0; i < samples; ++i) {
        double v;
        std::memcpy(&v, p + 8 * i, sizeof(double));
        d[i] = static_cast<float>(v);
      }
    return;
  }
  switch (bits) {
    case 8:  // unsigned, biased at 128
      for (size_t i = 0; i < samples; ++i) d[i] = (static_cast<int>(p[i]) - 128) / 128.0f;
      break;
    case 16:
      for (size_t i = 0; i < samples; ++i)
        d[i] = static_cast<int16_t>(rd_u16(p + 2 * i)) / 32768.0f;
      break;
    case 24:
      for (size_t i = 0; i < samples; ++i) {
        const uint8_t* q = p + 3 * i;
        int32_t v = static_cast<int32_t>(q[0]) | static_cast<int32_t>(q[1]) << 8 |
                    static_cast<int32_t>(q[2]) << 16;
        if (v & 0x800000) v |= ~0xffffff;
        d[i] = static_cast<float>(v) / 8388608.0f;
      }
      break;
    default:  // 32-bit integer
      for (size_t i = 0; i < samples; ++i)
        d[i] = static_cast<float>(static_cast<int32_t>(rd_u32(p + 4 * i))) / 2147483648.0f;
      break;
  }
}
}  // namespace

WavAudio decode_wav(const uint8_t* data, size_t size) {
  if (size < 12 || !is_id(data, "RIFF") || !is_id(data + 8, "WAVE"))
    throw WavError("audio is not a RIFF/WAVE file; serve accepts inline WAV");
  WavAudio wav;
  int format = 0, bits = 0, channels = 0;
  bool have_fmt = false, got_data = false;
  size_t pos = 12;
  while (pos + 8 <= size) {
    const size_t len = rd_u32(data + pos + 4);
    const uint8_t* body = data + pos + 8;
    const size_t avail = size - pos - 8;
    if (len > avail + 1)
      throw WavError("RIFF chunk overruns the file");  // a chunk's pad byte may be the last byte
    if (is_id(data + pos, "fmt ")) {
      if (len < 16) throw WavError("fmt chunk is too short");
      format = rd_u16(body);
      channels = rd_u16(body + 2);
      wav.sample_rate = static_cast<int>(rd_u32(body + 4));
      bits = rd_u16(body + 14);
      if (len >= 40 && format == 0xfffe) {  // WAVE_FORMAT_EXTENSIBLE: trust the subformat
        if (rd_u16(body + 26) != 16 || body[32 + 15] > 3)
          throw WavError("WAVE_FORMAT_EXTENSIBLE carries an unsupported subformat");
        format = body[32 + 15];
      }
      const bool ok = (format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
                      (format == 3 && (bits == 32 || bits == 64));
      if (!ok)
        throw WavError("unsupported WAV encoding (format " + std::to_string(format) + ", " +
                       std::to_string(bits) + " bit); serve decodes PCM and IEEE float");
      if (channels < 1 || channels > 8 || wav.sample_rate < 8000 || wav.sample_rate > 384000)
        throw WavError("unsupported WAV channel count or sample rate");
      have_fmt = true;
    } else if (is_id(data + pos, "data")) {
      if (!have_fmt) throw WavError("data chunk precedes fmt");
      const size_t align = (static_cast<size_t>(channels) * bits) / 8;
      const size_t bytes = len - (len % align);
      if (bytes / align > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels))
        throw WavError("data chunk length overflows");
      decode_block(body, bytes / align, channels, format, bits, wav.samples);
      got_data = true;
      // Later "data" chunks (rare) would append a discontinuity; stop at the first.
      break;
    }
    pos += 8 + len + (len & 1);
  }
  if (!have_fmt || !got_data) throw WavError("WAV is missing a fmt or data chunk");
  wav.channels = channels;
  return wav;
}

std::vector<float> to_mono_16k(const WavAudio& wav) {
  const size_t frames = wav.samples.size() / static_cast<size_t>(wav.channels);
  if (frames == 0) return {};
  std::vector<float> mono(frames);
  if (wav.channels == 1)
    mono = wav.samples;
  else
    for (size_t i = 0; i < frames; ++i) {
      float acc = 0.f;
      for (int c = 0; c < wav.channels; ++c) acc += wav.samples[i * wav.channels + c];
      mono[i] = acc / static_cast<float>(wav.channels);
    }
  if (wav.sample_rate == kAudioSampleRate) return mono;
  // Linear resample: out[j] reads src at j * rate / 16000. The last source
  // index is clamped, so the tail decays to the final sample rather than to
  // silence.
  const double step = static_cast<double>(wav.sample_rate) / kAudioSampleRate;
  const double last = static_cast<double>(frames - 1);
  std::vector<float> out(static_cast<size_t>(last / step) + 1);
  for (size_t j = 0; j < out.size(); ++j) {
    const double x = std::min(static_cast<double>(j) * step, last);
    const size_t i = static_cast<size_t>(x);
    const double f = x - static_cast<double>(i);
    const double hi =
        i + 1 < frames ? static_cast<double>(mono[i + 1]) : static_cast<double>(mono[i]);
    out[j] = static_cast<float>(static_cast<double>(mono[i]) * (1.0 - f) + hi * f);
  }
  return out;
}
}  // namespace dgpp::audio
