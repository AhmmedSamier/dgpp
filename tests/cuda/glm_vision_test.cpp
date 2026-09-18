#include "kernels/glm_vision.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/gemm.hpp"

namespace {
using namespace dgpp;
template <class T>
struct Managed {
  T* p = nullptr;
  explicit Managed(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  ~Managed() { cudaFree(p); }
};
void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}

DGPP_TEST(vision_layernorm_gelu_cuda_reference) {
  // Independent CUDA PyTorch golden values, including inputs near BF16
  // rounding boundaries. tools/glm_vision_norm_reference.py regenerates them.
  constexpr int rows = 16, dim = 4096;
  Managed<uint16_t> x(rows * dim), weight(dim), bias(dim), normalized(rows * dim);
  uint32_t state = 337;
  const auto value = [&] {
    state = state * 1664525u + 1013904223u;
    const float mantissa = static_cast<float>(int((state >> 8) % 251) - 125) / 128;
    return float_to_bf16_bits(std::ldexp(mantissa, int(state % 9) - 4));
  };
  for (int i = 0; i < rows * dim; ++i) x.p[i] = value();
  for (int i = 0; i < dim; ++i) weight.p[i] = value();
  for (int i = 0; i < dim; ++i) bias.p[i] = value();
  vision_layernorm_gelu(x.p, weight.p, bias.p, rows, dim, nullptr, normalized.p);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto digest = [](const uint16_t* values) {
    uint64_t hash = 14695981039346656037ull;
    for (int i = 0; i < rows * dim; ++i) {
      hash = (hash ^ (values[i] & 255)) * 1099511628211ull;
      hash = (hash ^ (values[i] >> 8)) * 1099511628211ull;
    }
    return hash;
  };
  require(digest(normalized.p) == 0x46d34846b17a3fa4ull,
          "LayerNorm differs from CUDA reference; inspect reduction order");
  require(digest(x.p) == 0x655b7684dac67aecull, "GELU differs from CUDA reference");
}

DGPP_TEST(vision_softmax_bf16_score_boundary_and_sizes) {
  // Independent FP64 host softmax of the BF16 eager scores. Fractional raw
  // scores exercise the required rounding *before* scaling and softmax.
  for (int n : {64, 96, 2048, 2052, 4096}) {
    constexpr int rows = 5;
    Managed<float> scores(rows * n);
    Managed<uint16_t> probs(rows * n);
    for (int i = 0; i < rows * n; ++i)
      scores.p[i] = static_cast<float>((i * 17) % 127) * 0.1237f - 8.1234f;
    vision_softmax(scores.p, probs.p, rows, n, 64, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int r = 0; r < rows; ++r) {
      std::vector<double> exp(n);
      double sum = 0;
      for (int j = 0; j < n; ++j) {
        const float score = bf16_bits_to_float(float_to_bf16_bits(scores.p[r * n + j])) / 8;
        exp[j] = std::exp(static_cast<double>(score));
        sum += exp[j];
      }
      for (int j = 0; j < n; ++j)
        require(probs.p[r * n + j] == float_to_bf16_bits(static_cast<float>(exp[j] / sum)),
                "softmax differs from BF16-score oracle");
    }
  }
}

DGPP_TEST(vision_batched_gemm_strides_tiles_and_plan_identity) {
  CublasLtGemm gemm;
  Managed<uint8_t> work(64ull << 20);
  constexpr int m = 7, n = 64, k = 64, batch = 16;
  // Short rows must use the batched plan, including different padded strides
  // on a second invocation of the same GEMM shape.
  for (bool weight_kn : {false, true, false})
    for (int padding : {0, 128, 0}) {
      const int as = m * k + padding, bs = n * k + 2 * padding, cs = m * n + padding;
      Managed<uint16_t> x(batch * as), w(batch * bs);
      Managed<float> y(batch * cs);
      for (int i = 0; i < batch * as; ++i) x.p[i] = float_to_bf16_bits(float(i % 5 - 2));
      for (int i = 0; i < batch * bs; ++i) w.p[i] = float_to_bf16_bits(float(i % 7 - 3));
      for (int i = 0; i < batch * cs; ++i) y.p[i] = 123456.f;
      gemm.matmul_batched_bf16(x.p, w.p, y.p, m, n, k, batch, as, bs, cs, work.p, 64ull << 20,
                               nullptr, 0, weight_kn);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < m; ++r)
          for (int c = 0; c < n; ++c) {
            double expected = 0;
            for (int j = 0; j < k; ++j)
              expected += bf16_bits_to_float(x.p[b * as + r * k + j]) *
                          bf16_bits_to_float(w.p[b * bs + (weight_kn ? j * n + c : c * k + j)]);
            require(y.p[b * cs + r * n + c] == expected, "batched GEMM indexing mismatch");
          }
        for (int i = m * n; i < cs; ++i)
          require(y.p[b * cs + i] == 123456.f, "batched GEMM overwrote padding");
      }
    }
}

DGPP_TEST(vision_query_tiles_preserve_full_matrix_reduction) {
  CublasLtGemm gemm;
  Managed<uint8_t> work(64ull << 20);
  constexpr int m = 756, n = 64, k = 756, batch = 16;
  Managed<uint16_t> x(batch * m * k), w(batch * n * k);
  Managed<float> full(batch * m * n), tile(batch * 128 * n);
  uint32_t state = 1919;
  const auto value = [&] {
    state = state * 1664525u + 1013904223u;
    const float mantissa = static_cast<float>(int((state >> 8) % 251) - 125) / 128;
    return float_to_bf16_bits(std::ldexp(mantissa, int(state % 9) - 4));
  };
  for (int i = 0; i < batch * m * k; ++i) x.p[i] = value();
  for (int i = 0; i < batch * n * k; ++i) w.p[i] = value();
  gemm.matmul_batched_bf16(x.p, w.p, full.p, m, n, k, batch, m * k, n * k, m * n, work.p,
                           64ull << 20, nullptr);
  for (int first = 0; first < m; first += 128) {
    const int rows = std::min(128, m - first);
    gemm.matmul_batched_bf16(x.p + first * k, w.p, tile.p, rows, n, k, batch, m * k, n * k,
                             rows * n, work.p, 64ull << 20, nullptr, m);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int b = 0; b < batch; ++b)
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < n; ++c)
          require(float_to_bf16_bits(tile.p[(b * rows + r) * n + c]) ==
                      float_to_bf16_bits(full.p[(b * m + first + r) * n + c]),
                  "query tiling changed BF16 attention values");
  }
}

DGPP_TEST(vision_bias_single_rounding_and_cached_pointer) {
  CublasLtGemm gemm;
  Managed<uint8_t> work(64ull << 20);
  constexpr int m = 64, n = 1024, k = 4096;
  Managed<uint16_t> x(m * k), w(n * k), y(m * n), b1(n), b2(n);
  for (int i = 0; i < m * k; ++i) x.p[i] = float_to_bf16_bits(1.f);
  for (int i = 0; i < n * k; ++i) w.p[i] = 0;
  for (int j = 0; j < n; ++j) {
    w.p[j * k] = float_to_bf16_bits(1.f);
    w.p[j * k + 1] = float_to_bf16_bits(1.f / 1024);
    b1.p[j] = float_to_bf16_bits(-1.f);
    b2.p[j] = float_to_bf16_bits(-0.5f);
  }
  for (auto* bias : {b1.p, b2.p, b1.p}) {
    gemm.matmul_linear_bf16(x.p, w.p, bias, y.p, m, n, k, work.p, 64ull << 20, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const uint16_t expected = float_to_bf16_bits(1.f + 1.f / 1024 + bf16_bits_to_float(bias[0]));
    for (int i = 0; i < m * n; ++i)
      require(y.p[i] == expected, "bias lost precision or reused another projection's pointer");
  }
}

DGPP_TEST(vision_unbiased_linear_preserves_fp32_partial_sums) {
  CublasLtGemm gemm;
  Managed<uint8_t> work(64ull << 20);
  constexpr int m = 24, n = 4096, k = 4096;
  Managed<uint16_t> x(m * k), w(n * k), y(m * n);
  for (int i = 0; i < m * k; ++i) x.p[i] = float_to_bf16_bits(1.f);
  for (int i = 0; i < n * k; ++i) w.p[i] = 0;
  for (int j = 0; j < n; ++j) {
    w.p[j * k] = float_to_bf16_bits(1.f);
    w.p[j * k + 1] = float_to_bf16_bits(1.f / 1024);
    w.p[j * k + k - 1] = float_to_bf16_bits(-1.f);
  }
  // Cache the ordinary text plan first: its reduction policy must not leak
  // into the vision plan with the same shape and destination type.
  gemm.matmul(x.p, w.p, y.p, m, n, k, DType::BF16, GemmOut::BF16, k, work.p, 64ull << 20, nullptr);
  gemm.matmul_linear_bf16(x.p, w.p, nullptr, y.p, m, n, k, work.p, 64ull << 20, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int i = 0; i < m * n; ++i)
    require(y.p[i] == float_to_bf16_bits(1.f / 1024),
            "unbiased projection rounded partial sums before cancellation");
}

DGPP_TEST(vision_attention_tile_scatter) {
  constexpr int n = 131, rows = 3, heads = 16, dim = 64, first = 128;
  Managed<float> tile(heads * rows * dim);
  Managed<uint16_t> out(heads * n * dim);
  for (int i = 0; i < heads * rows * dim; ++i) tile.p[i] = float(i % 127) / 32;
  for (int i = 0; i < heads * n * dim; ++i) out.p[i] = 0xffff;
  vision_store_attention(tile.p, out.p, rows, n, dim, heads, first, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int head = 0; head < heads; ++head)
    for (int row = 0; row < n; ++row)
      for (int d = 0; d < dim; ++d) {
        const auto want = row < first
                              ? uint16_t(0xffff)
                              : float_to_bf16_bits(tile.p[(head * rows + row - first) * dim + d]);
        require(out.p[(head * n + row) * dim + d] == want, "attention tile escaped its rows");
      }
}
}  // namespace
int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 2;
  return dgpp::test::run_all();
}
