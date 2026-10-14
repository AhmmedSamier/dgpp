#include "audio/fft400.hpp"

#include <cmath>
#include <vector>

namespace dgpp::audio {
namespace {

// N = N1 * N2 with N1 = 16, N2 = 25. Index maps n = N2*n1 + n2,
// k = k1 + N1*k2, so W_N^(kn) = W_16^(n1 k1) W_N^(n2 k1) W_25^(n2 k2):
// 25 length-16 DFTs, a twiddle multiply, then 16 length-25 DFTs.
constexpr int kN1 = 16;
constexpr int kN2 = 25;
constexpr int kN = 400;
constexpr double kPi = 3.14159265358979323846;

struct Tables {
  // cos/sin of -2 pi a b / m for the three lengths used.
  std::vector<double> c16, s16;    // [16][16]
  std::vector<double> c25, s25;    // [25][25]
  std::vector<double> ct, st;      // [25][16], W_400^(n2 k1)
  Tables() : c16(256), s16(256), c25(625), s25(625), ct(400), st(400) {
    for (int k = 0; k < 16; ++k)
      for (int n = 0; n < 16; ++n) {
        const double a = 2.0 * kPi * k * n / 16;
        c16[static_cast<size_t>(k) * 16 + n] = std::cos(a);
        s16[static_cast<size_t>(k) * 16 + n] = std::sin(a);
      }
    for (int k = 0; k < 25; ++k)
      for (int n = 0; n < 25; ++n) {
        const double a = 2.0 * kPi * k * n / 25;
        c25[static_cast<size_t>(k) * 25 + n] = std::cos(a);
        s25[static_cast<size_t>(k) * 25 + n] = std::sin(a);
      }
    for (int k1 = 0; k1 < 16; ++k1)
      for (int n2 = 0; n2 < 25; ++n2) {
        const double a = 2.0 * kPi * n2 * k1 / 400;
        ct[static_cast<size_t>(n2) * 16 + k1] = std::cos(a);
        st[static_cast<size_t>(n2) * 16 + k1] = std::sin(a);
      }
  }
};

const Tables& tables() {
  static const Tables t;
  return t;
}

}  // namespace

void fft400_forward(const double* in_re, const double* in_im, double* out_re, double* out_im) {
  const Tables& tb = tables();
  // Stage 1: 25 DFTs of length 16 over the strided columns
  // y[n2][k1] = sum_n1 x[25 n1 + n2] W_16^(n1 k1).
  double y_re[25][16], y_im[25][16];
  for (int n2 = 0; n2 < 25; ++n2)
    for (int k1 = 0; k1 < 16; ++k1) {
      double re = 0.0, im = 0.0;
      for (int n1 = 0; n1 < 16; ++n1) {
        const double x_re = in_re[static_cast<size_t>(25 * n1 + n2)];
        const double x_im = in_im[static_cast<size_t>(25 * n1 + n2)];
        const double c = tb.c16[static_cast<size_t>(k1) * 16 + n1];
        const double s = tb.s16[static_cast<size_t>(k1) * 16 + n1];
        re += x_re * c + x_im * s;
        im += x_im * c - x_re * s;
      }
      // Twiddle by W_400^(n2 k1) in the same pass.
      const double tc = tb.ct[static_cast<size_t>(n2) * 16 + k1];
      const double ts = tb.st[static_cast<size_t>(n2) * 16 + k1];
      y_re[n2][k1] = re * tc + im * ts;
      y_im[n2][k1] = im * tc - re * ts;
    }
  // Stage 2: 16 DFTs of length 25, out[k1 + 16 k2].
  for (int k1 = 0; k1 < 16; ++k1)
    for (int k2 = 0; k2 < 25; ++k2) {
      double re = 0.0, im = 0.0;
      for (int n2 = 0; n2 < 25; ++n2) {
        const double c = tb.c25[static_cast<size_t>(k2) * 25 + n2];
        const double s = tb.s25[static_cast<size_t>(k2) * 25 + n2];
        re += y_re[n2][k1] * c + y_im[n2][k1] * s;
        im += y_im[n2][k1] * c - y_re[n2][k1] * s;
      }
      out_re[static_cast<size_t>(k1 + 16 * k2)] = re;
      out_im[static_cast<size_t>(k1 + 16 * k2)] = im;
    }
}
}  // namespace dgpp::audio
