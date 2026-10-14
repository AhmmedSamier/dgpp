#include "audio/mel_frontend.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "audio/fft400.hpp"
#include "common/audio_input.hpp"

namespace dgpp::audio {
namespace {
constexpr int kBins = kAudioNfft / 2 + 1;  // 201
constexpr double kPi = 3.14159265358979323846;

double hz_to_mel_slaney(double f) {
  // transformers' hertz_to_mel(mel_scale="slaney"): linear to 1000 Hz,
  // logarithmic above, meeting at (1000 Hz, 15 mel).
  return f < 1000.0 ? 3.0 * f / 200.0 : 15.0 + std::log(f / 1000.0) * (27.0 / std::log(6.4));
}
double mel_to_hz_slaney(double m) {
  return m < 15.0 ? 200.0 * m / 3.0 : 1000.0 * std::exp((m - 15.0) * (std::log(6.4) / 27.0));
}
}  // namespace

const std::vector<double>& mel_filter_bank() {
  static const std::vector<double> filters = [] {
    std::vector<double> center(kAudioMelBins + 2);
    const double m_max = hz_to_mel_slaney(8000.0);
    for (int i = 0; i < kAudioMelBins + 2; ++i)
      center[i] = mel_to_hz_slaney(m_max * static_cast<double>(i) / (kAudioMelBins + 1));
    std::vector<double> out(static_cast<size_t>(kBins) * kAudioMelBins, 0.0);
    for (int m = 0; m < kAudioMelBins; ++m) {
      const double lo = center[m], ce = center[m + 1], hi = center[m + 2];
      const double enorm = 2.0 / (hi - lo);  // the "slaney" energy normalization
      for (int k = 0; k < kBins; ++k) {
        const double f = 8000.0 * static_cast<double>(k) / (kBins - 1);
        double w = 0.0;
        if (f >= lo && f <= ce && ce > lo) w = (f - lo) / (ce - lo);
        else if (f > ce && f <= hi && hi > ce) w = (hi - f) / (hi - ce);
        out[static_cast<size_t>(k) * kAudioMelBins + m] = w * enorm;
      }
    }
    return out;
  }();
  return filters;
}

const std::vector<float>& hann_window_400() {
  static const std::vector<float> w = [] {
    std::vector<float> t(kAudioNfft);
    for (int n = 0; n < kAudioNfft; ++n)
      t[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / kAudioNfft));
    return t;
  }();
  return w;
}

Fbank compute_fbank(const float* samples, size_t count) {
  if (count % kAudioHop != 0)
    throw std::invalid_argument("audio frontend: sample count must be a whole number of hops");
  if (count > static_cast<size_t>(kMaxAudioSamples))
    throw std::invalid_argument("audio frontend: clip exceeds the admission bound");
  if (count < static_cast<size_t>(kAudioMinSamples))
    throw std::invalid_argument("audio frontend: clip below the 0.5 s frontend minimum "
                                "(the admission path zero-extends short clips)");
  const int64_t frames = static_cast<int64_t>(count) / kAudioHop;
  Fbank out{frames, {}};
  if (frames == 0) return out;
  out.features.resize(static_cast<size_t>(frames) * kAudioMelBins);

  const std::vector<double>& filt = mel_filter_bank();
  const std::vector<float>& win = hann_window_400();

  std::vector<double> power(kBins);
  double in_re[kAudioNfft], in_im[kAudioNfft], out_re[kAudioNfft], out_im[kAudioNfft];
  double log_max = -std::numeric_limits<double>::infinity();
  auto frame_features = [&](int64_t f, auto&& per_bin) {
    // The STFT is center-padded, and the padding is not uniform: torch.stft
    // mirror-pads (reflect, edge sample excluded) while the processor
    // zero-extends the clip to its 30 s frame before the stft - so frames
    // read reflected samples left of the clip and zeros past its end.
    const int64_t base = f * kAudioHop - kAudioNfft / 2;
    for (int n = 0; n < kAudioNfft; ++n) {
      const int64_t s = base + n;
      double x;
      if (s < 0) x = static_cast<double>(samples[-s]);
      else if (s < static_cast<int64_t>(count)) x = static_cast<double>(samples[s]);
      else x = 0.0;
      in_re[n] = x * win[n];
      in_im[n] = 0.0;
    }
    fft400_forward(in_re, in_im, out_re, out_im);
    for (int k = 0; k < kBins; ++k)
      power[k] = out_re[k] * out_re[k] + out_im[k] * out_im[k];
    for (int m = 0; m < kAudioMelBins; ++m) {
      double acc = 0.0;
      for (int k = 0; k < kBins; ++k)
        acc += filt[static_cast<size_t>(k) * kAudioMelBins + m] * power[k];
      const double l = std::log10(acc < 1e-10 ? 1e-10 : acc);
      if (l > log_max) log_max = l;
      per_bin(m, l);
    }
  };
  // First pass records the log floor's reference (the clip max); the second
  // applies the floor and the (x + 4) / 4 rescale, so a frame at the very end
  // can lower nothing it was not itself floor'd against.
  std::vector<double> logs(static_cast<size_t>(frames) * kAudioMelBins);
  for (int64_t f = 0; f < frames; ++f)
    frame_features(f, [&](int m, double l) { logs[static_cast<size_t>(f) * kAudioMelBins + m] = l; });
  const double floor_v = log_max - 8.0;
  for (size_t i = 0; i < logs.size(); ++i) {
    const double l = logs[i] > floor_v ? logs[i] : floor_v;
    out.features[i] = static_cast<float>((l + 4.0) / 4.0);
  }
  return out;
}
}  // namespace dgpp::audio
