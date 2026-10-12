#pragma once
// The Qwen3-ASR log-mel frontend, a faithful host port of transformers'
// Qwen3ASRFeatureExtractor (docs/audio.md): a center-padded 400-point STFT
// with a periodic Hann window advanced every 160 samples, squared
// magnitudes projected through the 201x128 Slaney filter bank, log10 with a
// 1e-10 clamp and an 8-decay-per-clip floor, then (x + 4) / 4.
//
// The DFT here is the straightforward matrix product - exact but quadratic,
// about a tenth of a second per second of audio. It is the oracle the CUDA
// path must match within the feature budget; replacing it (Bluestein or a
// kernel) is a pure speed-up and must not move these numbers.

#include <cstdint>
#include <cstddef>
#include <vector>

namespace dgpp::audio {

struct Fbank {
  int64_t frames = 0;
  std::vector<float> features;  // [frames, 128] row-major
};

// The 201x128 Slaney bank (row-major, frequency bin inner) as one double
// matrix; built once per process. Exposed for the golden test.
const std::vector<double>& mel_filter_bank();

// Hann(400), periodic: 0.5 - 0.5 * cos(2 pi n / 400), torch's window.
const std::vector<float>& hann_window_400();

// `samples` must be a whole number of hops (the admission path truncates the
// sub-hop tail after zero-extending to the 8000-sample minimum), and at most
// kMaxAudioSamples long.
Fbank compute_fbank(const float* samples, size_t count);
}  // namespace dgpp::audio
