#pragma once
// 400-point complex forward DFT for the log-mel frontend
// (`audio/mel_frontend.hpp`, docs/audio.md): out[k] = sum_n in[n]
// exp(-2 pi i k n / 400), unscaled. Mixed-radix Cooley-Tukey 16x25 in
// double precision — the same arithmetic the naive matrix product does,
// regrouped — so it matches a direct DFT to ~1e-12 and the frontend
// goldens (2e-4) with room to spare, at ~1/10th the flops. Only the
// frontend uses this; it is not a general FFT.
#include <cstddef>

namespace dgpp::audio {

void fft400_forward(const double* in_re, const double* in_im, double* out_re, double* out_im);

}  // namespace dgpp::audio
