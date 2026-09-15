// Independent FP64 oracle for exact offset-code x BF16-scale weights.
// Also pin grouped/dense arithmetic, row maps, ragged tiles, output strides,
// BF16 epilogues, NaN propagation and cold CUDA graph capture.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {
using scale_gemm_test::require;
template <typename T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
};

struct Matrix {
  int bits, n, k;
  Buffer<uint32_t> packed;
  Buffer<uint16_t> scales;
  Matrix(int bits, int n, int k, uint64_t seed)
      : bits(bits),
        n(n),
        k(k),
        packed(static_cast<size_t>(n) * k * bits / 32),
        scales(static_cast<size_t>(n) * k / 64) {
    scale_gemm_test::Rng rng(seed);
    for (size_t i = 0; i < static_cast<size_t>(n) * k * bits / 32; ++i)
      packed.p[i] = static_cast<uint32_t>(rng.next());
    for (size_t i = 0; i < static_cast<size_t>(n) * k / 64; ++i)
      scales.p[i] =
          dgpp::float_to_bf16_bits(static_cast<float>(std::exp2(rng.unit() * 4) * 0.0031));
    scales.p[0] = 0;  // zero scales must also have exact semantics
  }
  dgpp::GlmPackedMatrix view() const { return {packed.p, scales.p, n, k, bits}; }
  double value(int row, int col) const {
    const int per = 32 / bits;
    const uint32_t w = packed.p[static_cast<size_t>(row) * (k / per) + col / per];
    const int code =
        static_cast<int>((w >> (bits * (col % per))) & ((1u << bits) - 1)) - (1 << (bits - 1));
    return code * static_cast<double>(dgpp::bf16_bits_to_float(
                      scales.p[static_cast<size_t>(row) * (k / 64) + col / 64]));
  }
};

void fill(uint16_t* a, size_t count, uint64_t seed) {
  scale_gemm_test::Rng rng(seed);
  for (size_t i = 0; i < count; ++i)
    a[i] = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit() * std::exp2(rng.unit() * 3)));
}

void oracle_check(const Matrix& w, const uint16_t* a, int stride, const float* got, int os, int m,
                  const float* baseline = nullptr) {
  double error2 = 0, norm2 = 0, max_abs = 0, max_error = 0;
  double baseline_error2 = 0, bf16_error2 = 0, baseline_bf16_error2 = 0;
  int better = 0, worse = 0, changed_bf16 = 0;
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < w.n; ++col) {
      double expected = 0;
      for (int kk = 0; kk < w.k; ++kk)
        expected +=
            dgpp::bf16_bits_to_float(a[static_cast<size_t>(row) * stride + kk]) * w.value(col, kk);
      const double actual = got[static_cast<size_t>(row) * os + col];
      if (std::isnan(expected)) {
        require(std::isnan(actual), "NaN scale must poison exactly its output column");
        continue;
      }
      require(std::isfinite(actual), "finite oracle must produce finite output");
      const double e = std::abs(actual - expected);
      error2 += e * e;
      norm2 += expected * expected;
      max_abs = std::max(max_abs, std::abs(expected));
      max_error = std::max(max_error, e);
      if (baseline) {
        const double old = baseline[static_cast<size_t>(row) * w.n + col];
        require(std::isfinite(old), "finite oracle must produce finite GEMV output");
        baseline_error2 += (old - expected) * (old - expected);
        const uint16_t b = dgpp::float_to_bf16_bits(actual);
        const uint16_t ob = dgpp::float_to_bf16_bits(old);
        const double be = dgpp::bf16_bits_to_float(b) - expected;
        const double obe = dgpp::bf16_bits_to_float(ob) - expected;
        bf16_error2 += be * be;
        baseline_bf16_error2 += obe * obe;
        better += std::abs(be) < std::abs(obe);
        worse += std::abs(be) > std::abs(obe);
        changed_bf16 += b != ob;
      }
    }
  const double relative = std::sqrt(error2 / std::max(norm2, 1e-30));
  std::printf("int%d M%d N%d K%d: fp32 l2_rel %.3g max_error/max_abs %.3g\n", w.bits, m, w.n, w.k,
              relative, max_error / std::max(max_abs, 1e-30));
  require(relative < 5e-6 && max_error < std::max(1e-8, max_abs * 4e-5),
          "packed GEMM differs from exact FP64 oracle");
  if (baseline) {
    const double old_relative = std::sqrt(baseline_error2 / std::max(norm2, 1e-30));
    std::printf(
        "  GEMV fp32 l2_rel %.3g; BF16 vs FP64 GEMV %.9g GEMM %.9g; "
        "changed %d, GEMM closer %d farther %d (remaining changes equidistant)\n",
        old_relative, std::sqrt(baseline_bf16_error2 / std::max(norm2, 1e-30)),
        std::sqrt(bf16_error2 / std::max(norm2, 1e-30)), changed_bf16, better, worse);
    require(old_relative < 5e-6, "packed GEMV differs from exact FP64 oracle");
  }
}
}  // namespace

DGPP_TEST(packq_gemm_exact_weights_and_ragged_tiles) {
  for (int bits : {4, 8})
    for (auto shape : {std::vector<int>{1, 1, 64},
                       {17, 70, 192},
                       {33, 129, 512},
                       {65, 35, 6144},
                       {32, 67, 2048},
                       {3, 17, 16384}}) {
      const int m = shape[0], n = shape[1], k = shape[2];
      Matrix w(bits, n, k, 917 + bits + k);
      const int stride = k + (m % 2 ? 3 : 0);
      Buffer<uint16_t> storage(static_cast<size_t>(m) * stride + 1), b(static_cast<size_t>(m) * n);
      uint16_t* a = storage.p + 1;  // explicitly unaligned base
      fill(a, static_cast<size_t>(m) * stride, 519);
      Buffer<float> out(static_cast<size_t>(m) * n), baseline(static_cast<size_t>(m) * n);
      Buffer<uint16_t> aligned(static_cast<size_t>(m) * k);
      for (int row = 0; row < m; ++row)
        std::memcpy(aligned.p + static_cast<size_t>(row) * k, a + static_cast<size_t>(row) * stride,
                    k * sizeof(uint16_t));
      dgpp::launch_packq_gemm_f32(a, stride, w.view(), out.p, m, n, k, nullptr);
      dgpp::launch_packq_gemm_bf16(a, stride, w.view(), b.p, m, n, k, nullptr);
      dgpp::launch_packq_gemv_f32(aligned.p, k, w.view(), baseline.p, m, n, k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      oracle_check(w, a, stride, out.p, n, m, baseline.p);
      for (int i = 0; i < m * n; ++i)
        require(b.p[i] == dgpp::float_to_bf16_bits(out.p[i]),
                "BF16 epilogue must round FP32 output exactly");
    }
}

DGPP_TEST(packq_gemm_grouped_maps_padding_and_graph) {
  constexpr int n = 70, k = 192, os = 77, tokens = 137, ns = 9;
  for (int bits : {4, 8}) {
    Matrix w(bits, n, k, 611 + bits);
    w.scales.p[9 * (k / 64) + 1] = 0x7fc0;
    Matrix other(bits, n, k, 975 + bits);
    Buffer<dgpp::MoeExpertView> views(ns * 3);
    Buffer<dgpp::MoeSegment> segs(ns);
    const int counts[ns] = {0, 1, 16, 17, 31, 32, 33, 65, 130};
    int total = 0;
    for (int e = 0; e < ns; ++e) {
      segs.p[e] = {total, counts[e], e};
      total += counts[e];
      for (int which = 0; which < 3; ++which)
        views.p[e * 3 + which] = dgpp::MoeExpertView::of(which == 1 ? w.view() : other.view());
    }
    Buffer<int32_t> rows(total);
    Buffer<uint16_t> a(tokens * k), gathered(static_cast<size_t>(total) * k);
    fill(a.p, tokens * k, 617);
    for (int r = 0; r < total; ++r) {
      rows.p[r] = (r * 19) % tokens;  // duplicates and nonmonotonic input rows
      std::memcpy(gathered.p + static_cast<size_t>(r) * k, a.p + rows.p[r] * k, k * 2);
    }
    Buffer<float> out(static_cast<size_t>(total) * os), ref(static_cast<size_t>(total) * n);
    Buffer<uint16_t> bf(static_cast<size_t>(total) * os);
    std::fill(out.p, out.p + static_cast<size_t>(total) * os, -12345.f);
    std::fill(bf.p, bf.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, out.p, os, n, k,
                                           bits, stream, rows.p);
    dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 1, bf.p, os, n, k,
                                            bits, stream, rows.p);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 2; ++repeat) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    dgpp::launch_packq_gemm_f32(gathered.p, k, w.view(), ref.p, total, n, k, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    oracle_check(w, gathered.p, k, out.p, os, total);
    for (int row = 0; row < total; ++row) {
      require(std::memcmp(out.p + row * os, ref.p + row * n, n * 4) == 0,
              "grouped mapped FP32 output must be bitwise dense GEMM");
      for (int col = 0; col < n; ++col) {
        const uint16_t want = dgpp::float_to_bf16_bits(out.p[row * os + col]);
        require(bf.p[row * os + col] == want ||
                    (std::isnan(dgpp::bf16_bits_to_float(want)) &&
                     std::isnan(dgpp::bf16_bits_to_float(bf.p[row * os + col]))),
                "grouped BF16 epilogue");
      }
      for (int col = n; col < os; ++col)
        require(out.p[row * os + col] == -12345.f && bf.p[row * os + col] == 0x1234,
                "grouped output padding must remain untouched");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
}

DGPP_TEST(packq_gemm_rejects_invalid_geometry) {
  Matrix w(4, 16, 64, 919);
  Buffer<uint16_t> a(64), out(16);
  auto rejects = [&](dgpp::GlmPackedMatrix matrix, int stride, int n, int k) {
    try {
      dgpp::launch_packq_gemm_bf16(a.p, stride, matrix, out.p, 1, n, k, nullptr);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto matrix = w.view();
  require(rejects(matrix, 63, 16, 64), "short activation stride");
  require(rejects(matrix, 64, 17, 64), "N exceeds matrix rows");
  require(rejects(matrix, 64, 16, 32), "K differs from matrix columns");
  matrix.bits = 3;
  require(rejects(matrix, 64, 16, 64), "invalid code width");
  matrix = w.view();
  matrix.packed += 1;
  require(rejects(matrix, 64, 16, 64), "unaligned packed row");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
