// The deterministic transcendental functions (common/det_math.hpp): the
// device and the host compute exactly the same bits over millions of
// inputs (random in the sampler's ranges, dense near the reduction
// boundaries, every special value), and both sit within a handful of ulps
// of libm — the sampler's on-device verdict is bitwise the host oracle only
// if this holds.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/det_math.hpp"
#include "common/test.hpp"

namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

__global__ void eval_kernel(const double* xd, const float* xf, int n,
                            double* exp_out, double* log_out, float* expf_out,
                            float* logf_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  exp_out[i] = dgpp::detmath::exp_d(xd[i]);
  log_out[i] = dgpp::detmath::log_d(xd[i]);
  expf_out[i] = dgpp::detmath::exp_f(xf[i]);
  logf_out[i] = dgpp::detmath::log_f(xf[i]);
}

struct Xorshift {
  uint64_t s = 0x9e3779b97f4a7c15ull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) * 0x1p-53; }
};

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0 ||
         (a != a && b != b);  // any NaN pair counts as equal
}
bool same_bits(float a, float b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0 || (a != a && b != b);
}

// ulps between two finite doubles of the same sign.
double ulps_apart(double a, double b) {
  if (a == b) return 0.0;
  const double u = std::nextafter(b, a) - b;
  return std::fabs((a - b) / u);
}
double ulps_apart(float a, float b) {
  if (a == b) return 0.0;
  const float u = std::nextafter(b, a) - b;
  return std::fabs(static_cast<double>(a - b) / u);
}

}  // namespace

DGPP_TEST(det_math_host_and_device_agree_bitwise_and_track_libm) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    throw std::runtime_error("no CUDA device");
  constexpr int n = 1 << 21;
  std::vector<double> xd(n);
  std::vector<float> xf(n);
  Xorshift rng;
  for (int i = 0; i < n; ++i) {
    double v;
    switch (i % 8) {
      case 0: v = (rng.unit() - 0.5) * 60.0; break;             // logit deltas
      case 1: v = (rng.unit() - 0.5) * 1500.0; break;           // the whole exp range
      case 2: v = rng.unit() * 1e-3; break;                     // tiny arguments
      case 3: v = std::ldexp(rng.unit(), static_cast<int>(rng.next() % 2000) - 1000); break;
      case 4: v = 0.6931471805599453 * (static_cast<double>(rng.next() % 2000) - 1000.0) +
                  (rng.unit() - 0.5) * 1e-9; break;            // near k*ln2
      case 5: v = 1.0 + (rng.unit() - 0.5) * 1e-6; break;       // log near 1
      case 6: v = 0.7071067811865476 * (1.0 + (rng.unit() - 0.5) * 1e-9); break;
      default: {
        uint64_t bits = rng.next();
        std::memcpy(&v, &bits, sizeof(v));  // arbitrary bit patterns
      }
    }
    xd[static_cast<size_t>(i)] = v;
    xf[static_cast<size_t>(i)] = static_cast<float>(
        (i % 8 == 1) ? (rng.unit() - 0.5) * 200.0 : v);
  }
  const double specials[] = {0.0, -0.0, 1.0, -1.0, 4.9e-324, -4.9e-324,
                             2.2250738585072014e-308, 1.7976931348623157e308,
                             709.782712893384, 709.79, -745.1332191019412,
                             -745.14, -708.4, 88.72283f, -103.97f,
                             std::nan(""), INFINITY, -INFINITY, 2.0, 0.5,
                             0.7071067811865476, 1.4142135623730951};
  for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); ++i) {
    xd[i] = specials[i];
    xf[i] = static_cast<float>(specials[i]);
  }

  double *d_xd, *d_exp, *d_log;
  float *d_xf, *d_expf, *d_logf;
  DGPP_CUDA_OK(cudaMalloc(&d_xd, n * sizeof(double)));
  DGPP_CUDA_OK(cudaMalloc(&d_exp, n * sizeof(double)));
  DGPP_CUDA_OK(cudaMalloc(&d_log, n * sizeof(double)));
  DGPP_CUDA_OK(cudaMalloc(&d_xf, n * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_expf, n * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_logf, n * sizeof(float)));
  DGPP_CUDA_OK(cudaMemcpy(d_xd, xd.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_xf, xf.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  eval_kernel<<<(n + 255) / 256, 256>>>(d_xd, d_xf, n, d_exp, d_log, d_expf, d_logf);
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<double> g_exp(n), g_log(n);
  std::vector<float> g_expf(n), g_logf(n);
  DGPP_CUDA_OK(cudaMemcpy(g_exp.data(), d_exp, n * sizeof(double), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(g_log.data(), d_log, n * sizeof(double), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(g_expf.data(), d_expf, n * sizeof(float), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(g_logf.data(), d_logf, n * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_xd);
  cudaFree(d_exp);
  cudaFree(d_log);
  cudaFree(d_xf);
  cudaFree(d_expf);
  cudaFree(d_logf);

  double worst_exp = 0.0, worst_log = 0.0, worst_expf = 0.0, worst_logf = 0.0;
  for (int i = 0; i < n; ++i) {
    const size_t k = static_cast<size_t>(i);
    const double he = dgpp::detmath::exp_d(xd[k]);
    const double hl = dgpp::detmath::log_d(xd[k]);
    const float hef = dgpp::detmath::exp_f(xf[k]);
    const float hlf = dgpp::detmath::log_f(xf[k]);
    require(same_bits(he, g_exp[k]),
            "exp_d host/device differ at x=" + std::to_string(xd[k]));
    require(same_bits(hl, g_log[k]),
            "log_d host/device differ at x=" + std::to_string(xd[k]));
    require(same_bits(hef, g_expf[k]),
            "exp_f host/device differ at x=" + std::to_string(xf[k]));
    require(same_bits(hlf, g_logf[k]),
            "log_f host/device differ at x=" + std::to_string(xf[k]));
    // Against libm, where both are finite and nonzero (the subnormal tail
    // and the exact zero of log(1) are compared exactly below).
    const double le = std::exp(xd[k]);
    if (std::isfinite(le) && le > 2.2250738585072014e-308 && std::isfinite(he))
      worst_exp = std::max(worst_exp, ulps_apart(he, le));
    const double ll = std::log(xd[k]);
    if (std::isfinite(ll) && ll != 0.0 && std::isfinite(hl))
      worst_log = std::max(worst_log, ulps_apart(hl, ll));
    const float lef = std::exp(xf[k]);
    if (std::isfinite(lef) && lef > 1.17549435e-38f && std::isfinite(hef))
      worst_expf = std::max(worst_expf, ulps_apart(hef, lef));
    const float llf = std::log(xf[k]);
    if (std::isfinite(llf) && llf != 0.0f && std::isfinite(hlf))
      worst_logf = std::max(worst_logf, ulps_apart(hlf, llf));
  }
  // Special values, exactly.
  require(dgpp::detmath::exp_d(0.0) == 1.0 && dgpp::detmath::exp_d(-0.0) == 1.0,
          "exp(0) == 1");
  require(dgpp::detmath::log_d(1.0) == 0.0, "log(1) == 0 exactly");
  require(std::isinf(dgpp::detmath::exp_d(INFINITY)) &&
              dgpp::detmath::exp_d(-INFINITY) == 0.0 &&
              dgpp::detmath::exp_d(800.0) == INFINITY &&
              dgpp::detmath::exp_d(-800.0) == 0.0,
          "exp at the infinities and beyond the range");
  require(std::isinf(dgpp::detmath::log_d(0.0)) && dgpp::detmath::log_d(0.0) < 0 &&
              std::isnan(dgpp::detmath::log_d(-1.0)) &&
              std::isinf(dgpp::detmath::log_d(INFINITY)),
          "log at zero, negatives and infinity");
  require(std::isnan(dgpp::detmath::exp_d(std::nan(""))) &&
              std::isnan(dgpp::detmath::log_d(std::nan(""))),
          "NaN propagates");
  std::printf("det_math worst ulps vs libm: exp %.2f log %.2f expf %.2f logf %.2f\n",
              worst_exp, worst_log, worst_expf, worst_logf);
  require(worst_exp <= 4.0 && worst_log <= 4.0, "double forms within 4 ulps of libm");
  require(worst_expf <= 2.0 && worst_logf <= 2.0, "float forms within 2 ulps of libm");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::puts("no CUDA device: skipping det_math_test");
    return 2;
  }
  return dgpp::test::run_all();
}
