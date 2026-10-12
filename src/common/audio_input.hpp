#pragma once
// The audio counterpart of common/image_input.hpp (docs/audio.md). It
// describes the ASR family's trained audio span: three delimiter ids around
// a run of pad tokens, one pad per post-convolution encoder token. The
// admission record owns 16 kHz mono PCM; every rank decodes the same log-mel
// features from the same samples, so no feature buffer (or process-local
// pointer) crosses the wire.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dgpp {

struct AudioInput {
  int64_t offset = 0;   // absolute position of the first pad token in the prompt
  int tokens = 0;       // encoder tokens after the convolution frontend
  int64_t frames = 0;   // 10 ms log-mel frames, pcm.size() / kAudioHop
  // 16 kHz mono, roughly [-1, 1]. Zero-extended to the 8000-sample frontend
  // minimum and truncated to a whole hop: the frontend drops the last (partial)
  // STFT frame, so the sub-frame tail never reaches a feature.
  std::vector<float> pcm;
};

// The checkpoint's frontend geometry: 16 kHz audio, a 400-sample frame of the
// 16 kHz stream advanced every 160 samples (25 ms / 10 ms), and 128 log-mel
// bins. The encoder chunks the frame axis at 2 * n_window = 100 frames (10 s)
// and every full chunk leaves the three stride-2 convolutions with 13 tokens.
inline constexpr int kAudioSampleRate = 16000;
inline constexpr int kAudioNfft = 400;
inline constexpr int kAudioHop = 160;
inline constexpr int kAudioMelBins = 128;
inline constexpr int kAudioWindowHalf = 50;
inline constexpr int kAudioChunkFrames = 2 * kAudioWindowHalf;
inline constexpr int kAudioChunkTokens = 13;
// transformers' Qwen3ASRFeatureExtractor zero-pads clips under 8000 samples
// (0.5 s) before featurizing; the frame count follows the padded length.
inline constexpr int64_t kAudioMinSamples = 8000;
// Admission bounds, not model geometry: half an hour decodes to 115 MB of
// float PCM (57 MB of it arrives base64-encoded), and the encoder yields at
// most ~4 tokens per audio second, so a legal request never nears either cap.
inline constexpr int64_t kMaxAudioSamples = 16000ll * 60 * 30;
inline constexpr int kMaxAudioTokens = 16384;
inline constexpr size_t kMaxRequestAudioBytes = 256ull << 20;

// The checkpoint's three audio delimiters (config.json's
// audio_start_token_id, audio_token_id repeated once per encoder token, and
// audio_end_token_id).
struct AudioTokens {
  int64_t start = -1;
  int64_t pad = -1;
  int64_t end = -1;
};

// Tokens the three stride-2 convolutions leave from `frames` valid frames:
// 13 per full 100-frame chunk plus ceil3() of the remainder, matching the
// processor's _get_audio_token_length (each conv maps a positive length L to
// (L - 1) / 2 + 1, and a zero-length tail stays zero).
inline constexpr int audio_token_count(int64_t frames) {
  int64_t tail = frames % kAudioChunkFrames;
  int64_t tokens = (frames / kAudioChunkFrames) * kAudioChunkTokens;
  for (int conv = 0; conv < 3 && tail > 0; ++conv) tail = (tail - 1) / 2 + 1;
  return static_cast<int>(tokens + tail);
}

inline void validate_audio_input(const AudioInput& au) {
  if (au.frames < 1 || au.tokens < 1 || au.tokens > kMaxAudioTokens ||
      au.pcm.size() > static_cast<size_t>(kMaxAudioSamples) ||
      au.tokens != audio_token_count(au.frames) ||
      au.pcm.size() != static_cast<size_t>(au.frames) * kAudioHop)
    throw std::invalid_argument("invalid audio geometry (frames, tokens or samples)");
}

inline void validate_audio_inputs(const std::vector<AudioInput>& audios, size_t prompt_tokens) {
  if (prompt_tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("audio prompt length exceeds the position range");
  int64_t end = 0;
  size_t bytes = 0;
  for (const auto& au : audios) {
    validate_audio_input(au);
    if (au.offset < end || au.offset < 0 ||
        static_cast<uint64_t>(au.offset) > prompt_tokens ||
        static_cast<size_t>(au.tokens) > prompt_tokens - static_cast<size_t>(au.offset))
      throw std::invalid_argument("invalid audio token span");
    const size_t n = au.pcm.size() * sizeof(float);
    if (n > kMaxRequestAudioBytes - bytes)
      throw std::invalid_argument("decoded audio data exceeds the 256 MiB request byte limit");
    end = au.offset + au.tokens;
    bytes += n;
  }
}
}  // namespace dgpp
