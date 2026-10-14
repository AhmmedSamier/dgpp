#include "audio/mp3_decode.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "common/audio_input.hpp"

#define MINIMP3_IMPLEMENTATION
#include "../../third_party/minimp3/minimp3.h"

namespace dgpp::audio {
namespace {

// ID3v2 tag skip: "ID3" + 2 version bytes + flags + 4 syncsafe size bytes.
// The tag (album art especially) may itself contain MP3 sync patterns, so
// the frame scanner must start past it. A v2.4 footer adds 10 bytes.
size_t id3v2_skip(const uint8_t* data, size_t size) {
  if (size < 10 || std::memcmp(data, "ID3", 3) != 0) return 0;
  const uint32_t body = (static_cast<uint32_t>(data[6] & 0x7f) << 21) |
                        (static_cast<uint32_t>(data[7] & 0x7f) << 14) |
                        (static_cast<uint32_t>(data[8] & 0x7f) << 7) |
                        static_cast<uint32_t>(data[9] & 0x7f);
  size_t total = 10 + body;
  if (data[5] & 0x10) total += 10;
  return total > size ? size : total;
}

// Consecutive hard decode failures before the stream is corrupt: sync finds
// frame starts and skips tag padding without counting here, so anything
// reaching this bound is genuinely undecodable bytes, not trailing tags.
constexpr size_t kMaxDecodeErrors = 4096;
// Pre-resample frame bound: the fastest native MP3 rate (48 kHz) runs 3x
// the 16 kHz admission bound, plus one bound of margin. The shared align
// step still enforces the exact cap afterwards.
constexpr int64_t kMaxMp3Frames = kMaxAudioSamples * 4;

// Layer-III bitrate table (kbps) by MPEG version, index 1..14; index 0
// (free) and 15 (bad) never carry a Xing tag.
int lame_bitrate_kbps(int version, int index) {
  static constexpr int mpeg1[16] = {0,  32, 40, 48, 56,  64,  80,  96,
                                    112, 128, 160, 192, 224, 256, 320, 0};
  static constexpr int mpeg2[15] = {0,  8,  16, 24, 32,  40,  48,  56,
                                    64, 80, 96, 112, 128, 144, 160};
  if (index < 0 || index > 14) return 0;
  return version == 3 ? mpeg1[index] : mpeg2[index];
}

int lame_sample_rate_hz(int version, int index) {
  static constexpr int rate[4][3] = {{11025, 12000, 8000},  // 2.5
                                     {0, 0, 0},             // reserved
                                     {22050, 24000, 16000},  // 2
                                     {44100, 48000, 32000}};  // 1
  return rate[version][index];
}

}  // namespace

// Locates the LAME tag structurally (never substring-searched: audio data
// may contain the magic): the first frame's MPEG header gives the side-info
// size, the Xing/Info tag follows it, and the LAME tag follows the Xing
// fields — all bounds-checked against the first frame.
std::optional<std::pair<size_t, size_t>> gapless_trim(const uint8_t* data, size_t size) {
  if (data == nullptr || size < 10) return std::nullopt;
  size_t pos = id3v2_skip(data, size);
  if (pos + 4 > size || data[pos] != 0xff || (data[pos + 1] & 0xe0) != 0xe0)
    return std::nullopt;
  const int version = (data[pos + 1] >> 3) & 3;
  const int layer = (data[pos + 1] >> 1) & 3;
  if (layer != 1 || version == 1) return std::nullopt;  // Layer III, known MPEG
  const int bitrate_idx = (data[pos + 2] >> 4) & 0xf;
  const int rate_idx = (data[pos + 2] >> 2) & 3;
  if (bitrate_idx == 0 || bitrate_idx == 15 || rate_idx == 3) return std::nullopt;
  const int bitrate = lame_bitrate_kbps(version, bitrate_idx);
  const int rate = lame_sample_rate_hz(version, rate_idx);
  if (bitrate <= 0 || rate <= 0) return std::nullopt;
  const bool mpeg1 = version == 3;
  const bool mono = ((data[pos + 3] >> 6) & 3) == 3;
  const size_t frame_len = static_cast<size_t>((mpeg1 ? 144 : 72) * bitrate * 1000 / rate) +
                           ((data[pos + 2] >> 1) & 1);
  const size_t xing = pos + 4 + (mpeg1 ? (mono ? 17 : 32) : (mono ? 9 : 17));
  if (xing + 8 > size || xing + 8 > pos + frame_len) return std::nullopt;
  if (std::memcmp(data + xing, "Xing", 4) != 0 && std::memcmp(data + xing, "Info", 4) != 0)
    return std::nullopt;
  const uint32_t flags = (static_cast<uint32_t>(data[xing + 4]) << 24) |
                         (static_cast<uint32_t>(data[xing + 5]) << 16) |
                         (static_cast<uint32_t>(data[xing + 6]) << 8) |
                         static_cast<uint32_t>(data[xing + 7]);
  size_t tag = xing + 8;
  if (flags & 1) tag += 4;
  if (flags & 2) tag += 4;
  if (flags & 4) tag += 100;
  if (flags & 8) tag += 4;
  if (tag + 27 > size || tag + 27 > pos + frame_len) return std::nullopt;
  if (std::memcmp(data + tag, "LAME", 4) != 0) return std::nullopt;
  for (int i = 4; i < 9; ++i)
    if (data[tag + i] < 0x20 || data[tag + i] > 0x7e) return std::nullopt;
  const size_t delay = (static_cast<size_t>(data[tag + 21]) << 4) | (data[tag + 22] >> 4);
  const size_t pad = (static_cast<size_t>(data[tag + 22] & 0xf) << 8) | data[tag + 23];
  // Both are 12-bit fields (4095 max): range sanity belongs to the apply
  // site, which refuses trims that would eat the clip.
  return std::make_pair(delay, pad);
}

WavAudio decode_mp3(const uint8_t* data, size_t size) {
  if (data == nullptr || size == 0)
    throw Mp3Error("audio is empty; serve accepts WAV, MP3, FLAC and Ogg Vorbis");
  size_t pos = id3v2_skip(data, size);
  if (pos >= size)
    throw Mp3Error("audio holds no MP3 frames; serve accepts WAV, MP3, FLAC and Ogg Vorbis");

  mp3dec_t dec;
  mp3dec_init(&dec);
  WavAudio out;
  int16_t frame[MINIMP3_MAX_SAMPLES_PER_FRAME];
  size_t errors = 0;
  while (pos < size) {
    mp3dec_frame_info_t info{};
    const size_t rest = size - pos;
    const int chunk = rest > static_cast<size_t>(std::numeric_limits<int>::max())
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(rest);
    // Returns samples per channel (0: no frame decoded). info.frame_bytes
    // is the consumed byte count on success, the skipped garbage count
    // when the tail holds no complete frame, and 0 on a hard decode error
    // (the decoder re-inits itself, so stepping one byte always progresses).
    const int samples = mp3dec_decode_frame(&dec, data + pos, chunk, frame, &info);
    if (samples > 0) {
      if (info.channels < 1 || info.channels > 2)
        throw Mp3Error("MP3 frame has an unsupported channel count");
      if (info.hz < 8000 || info.hz > 48000)
        throw Mp3Error("MP3 frame has an unsupported sample rate");
      if (out.sample_rate == 0) {
        out.sample_rate = info.hz;
        out.channels = info.channels;
      } else if (info.hz != out.sample_rate || info.channels != out.channels) {
        throw Mp3Error("MP3 stream changes its sample rate or channel count mid-stream");
      }
      const size_t decoded = static_cast<size_t>(samples) * static_cast<size_t>(info.channels);
      if (decoded / static_cast<size_t>(info.channels) != static_cast<size_t>(samples) ||
          (out.samples.size() + decoded) / static_cast<size_t>(out.channels) >
              static_cast<size_t>(kMaxMp3Frames))
        throw Mp3Error("MP3 audio exceeds the 30 minute admission bound");
      const size_t base = out.samples.size();
      out.samples.resize(base + decoded);
      for (size_t i = 0; i < decoded; ++i)
        out.samples[base + i] = static_cast<float>(frame[i]) / 32768.0f;
      errors = 0;
    } else if (info.frame_bytes == 0) {
      if (++errors > kMaxDecodeErrors)
        throw Mp3Error("MP3 stream has too many corrupt frames; upload is undecodable");
    }
    const size_t advance = info.frame_bytes > 0 ? static_cast<size_t>(info.frame_bytes) : 1;
    pos += advance;
  }
  if (out.samples.empty())
    throw Mp3Error("audio holds no MP3 frames; serve accepts WAV, MP3, FLAC and Ogg Vorbis");
  // Gapless trim: drop the encoder delay and end padding the LAME tag
  // reports, keeping at least a granule of audio behind.
  if (const auto trim = gapless_trim(data, size)) {
    const size_t lead = trim->first * static_cast<size_t>(out.channels);
    const size_t trail = trim->second * static_cast<size_t>(out.channels);
    if (lead + trail + 1152 * static_cast<size_t>(out.channels) < out.samples.size()) {
      out.samples.erase(out.samples.begin(), out.samples.begin() + lead);
      out.samples.resize(out.samples.size() - trail);
    }
  }
  return out;
}
}  // namespace dgpp::audio
