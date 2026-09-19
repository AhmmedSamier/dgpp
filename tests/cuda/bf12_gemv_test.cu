// Gates for the lossless 12-bit decode form of bf16 weights
// (kernels/bf12_gemv.hpp): the packed form decodes back to the bf16 bits it
// came from; a registered companion's launches through the GEMM seam are
// BITWISE the bf16 GEMV's at every decode row count, both outputs and a
// strided activation view; escapes (zeros, subnormals, outliers, NaN/Inf
// exponents, a row full of them up to the bound) take the exact path; a
// pathological row keeps the matrix in its bf16 form; wide calls ignore the
// companion; the prefetch view names the bytes the launch streams.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemm.hpp"

namespace {

using dgpp::Bf12Host;
using dgpp::Bf12Matrix;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Trained-looking weights: gaussian, so the exponents carry the 2^-j tail
// the format is built around (about 1.5e-4 of them leave the window).
std::vector<uint16_t> gaussian_bf16(size_t elems, float sigma, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.f, sigma);
  std::vector<uint16_t> v(elems);
  for (auto& x : v) x = dgpp::float_to_bf16_bits(dist(rng));
  return v;
}

struct Packed {
  std::vector<void*> allocs;
  Bf12Matrix m;
  ~Packed() {
    for (void* p : allocs) cudaFree(p);
  }
};

void upload(const Bf12Host& h, int n, int k, Packed& out) {
  void* p = nullptr;
  void* b = nullptr;
  void* e = nullptr;
  void* r = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&r, h.raw.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(r, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice));
  out.m.raw = static_cast<const uint16_t*>(r);
  DGPP_CUDA_OK(cudaMalloc(&p, h.packed.size()));
  DGPP_CUDA_OK(cudaMalloc(&b, h.rows.size() * 4));
  DGPP_CUDA_OK(cudaMalloc(&e, h.esc.size() * 4));
  out.allocs = {p, b, e, r};
  DGPP_CUDA_OK(cudaMemcpy(p, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(b, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(e, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice));
  out.m.packed = static_cast<const uint8_t*>(p);
  out.m.rows = static_cast<const uint32_t*>(b);
  out.m.esc = static_cast<const uint32_t*>(e);
  out.m.n = n;
  out.m.k = k;
}

// One matmul through the seam; the raw output bytes.
std::vector<uint8_t> run(dgpp::CublasLtGemm& gemm, const uint16_t* act, size_t stride,
                         const uint16_t* w, int m, int n, int k, bool f32) {
  const size_t bytes = static_cast<size_t>(m) * n * (f32 ? 4 : 2);
  void* out = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&out, bytes));
  DGPP_CUDA_OK(cudaMemset(out, 0xA5, bytes));
  gemm.matmul(act, w, out, m, n, k, dgpp::DType::BF16,
              f32 ? dgpp::GemmOut::F32 : dgpp::GemmOut::BF16, stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(bytes);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), out, bytes, cudaMemcpyDeviceToHost));
  cudaFree(out);
  return got;
}

// The companion's launches against the plain instance's, every decode row
// count, both outputs, contiguous and strided activations.
void bitwise_gate(const std::vector<uint16_t>& w, int n, int k, const char* label,
                  size_t* escapes_out = nullptr) {
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(h.ok, std::string(label) + ": encodes");
  std::vector<uint16_t> back(w.size());
  dgpp::bf12_decode(h, n, k, back.data());
  require(std::memcmp(back.data(), w.data(), w.size() * 2) == 0,
          std::string(label) + ": the packed form decodes to the same bf16 bits");
  if (escapes_out) *escapes_out = h.escapes;

  uint16_t* dw = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dw, w.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  Packed pk;
  upload(h, n, k, pk);
  // The reference chunks every row count (the scalar chain per row); the
  // packed instance keeps the model's four decode rows and widens by itself.
  dgpp::CublasLtGemm plain, packed;
  plain.set_decode_rows(8);
  packed.set_decode_rows(4);
  packed.register_bf12(dw, pk.m);
  require(packed.bf12_registered() == 1, "one companion registered");
  {
    // Until the caller says its rows are a decode batch, a five-row call
    // keeps the weight's own bytes (a short prefill chunk's Lt algorithm).
    const void* view = nullptr;
    size_t view_bytes = 0;
    packed.resident_view(dw, w.size() * 2, 5, &view, &view_bytes);
    require(view == dw, std::string(label) + ": five rows stay bf16 outside a decode batch");
  }
  packed.set_bf12_wide(true);

  const size_t stride = static_cast<size_t>(k) + 48;  // a fused-row view
  const std::vector<uint16_t> act = gaussian_bf16(8 * stride, 1.5f, 0xAC7 + n);
  uint16_t* dx = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dx, act.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dx, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
  // One to four rows: the narrow kernel (the wide one past the smem bound,
  // k = 8192 at four rows); five to eight: the wide kernel. Every row is
  // bitwise its scalar chain.
  for (int m = 1; m <= dgpp::kBf12MaxRows; ++m) {
    for (bool f32 : {false, true}) {
      for (size_t st : {static_cast<size_t>(k), stride}) {
        const auto want = run(plain, dx, st, dw, m, n, k, f32);
        const auto got = run(packed, dx, st, dw, m, n, k, f32);
        require(want == got, std::string(label) + ": m=" + std::to_string(m) +
                                 (f32 ? " f32" : " bf16") + " bitwise the bf16 GEMV");
      }
    }
    const void* view = nullptr;
    size_t view_bytes = 0;
    packed.resident_view(dw, w.size() * 2, m, &view, &view_bytes);
    require(view == pk.m.packed && view_bytes == dgpp::bf12_packed_bytes(n, k),
            std::string(label) + ": the prefetch view is the packed bytes");
  }
  // Past a packed launch's rows the call keeps the weight's own bytes.
  const void* view = nullptr;
  size_t view_bytes = 0;
  packed.resident_view(dw, w.size() * 2, dgpp::kBf12MaxRows + 1, &view, &view_bytes);
  require(view == dw && view_bytes == w.size() * 2, "a wide call's view is the bf16 weight");
  plain.resident_view(dw, w.size() * 2, 2, &view, &view_bytes);
  require(view == dw && view_bytes == w.size() * 2, "no companion: the bf16 weight");
  cudaFree(dx);
  cudaFree(dw);
  std::printf("[ OK ] %s N%dxK%d: %zu escapes, widest row %d, bitwise at every row count\n",
              label, n, k, h.escapes, h.max_row_escapes);
}

}  // namespace

DGPP_TEST(bf12_trained_shapes_are_bitwise_the_bf16_gemv) {
  // The KDA in_proj class (k = hidden, ragged n), the o_proj class
  // (k = 2048: the two-super-block depth), the eh_proj class (k = 8192:
  // two rows fit the smem bound, three do not).
  for (auto [n, k] : std::vector<std::pair<int, int>>{{1611, 4096}, {1024, 2048}, {515, 8192},
                                                      {7, 1024}}) {
    const auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xB12 + n);
    size_t escapes = 0;
    bitwise_gate(w, n, k, "gaussian", &escapes);
    if (static_cast<size_t>(n) * k > 1000000)
      require(escapes > 0, "a trained-looking matrix exercises the escape path");
  }
}

DGPP_TEST(bf12_rows_of_different_scale_keep_their_own_window) {
  // The KDA in_proj stacks projections of different magnitude: a window
  // per tensor would escape whole rows (and refuse the matrix).
  const int n = 48, k = 4096;
  std::vector<uint16_t> w;
  for (float sigma : {0.02f, 3.0e-5f, 40.0f}) {
    const auto part = gaussian_bf16(static_cast<size_t>(n / 3) * k, sigma, 0x5CA1E + w.size());
    w.insert(w.end(), part.begin(), part.end());
  }
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(h.ok && h.max_row_escapes < 16, "every row packs inside its own window");
  bitwise_gate(w, n, k, "mixed scales");
}

DGPP_TEST(bf12_escapes_are_exact) {
  const int n = 96, k = 2048;
  auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xE5C);
  std::mt19937_64 rng(0xE5CA9E);
  const auto at = [&](int row, int col) -> uint16_t& { return w[static_cast<size_t>(row) * k + col]; };
  // Zeros of both signs, subnormal-range and huge exponents, the first and
  // last columns of a row, neighbours inside one step, a lane's every step.
  at(0, 0) = 0x0000;
  at(0, 1) = 0x8000;
  at(0, k - 1) = 0x0001;
  at(1, 7) = 0x7F00;  // exponent 254
  at(1, 8) = 0xFF7F;
  at(2, 255) = 0x0080;  // exponent 1
  at(2, 256) = 0x0081;
  for (int c = 0; c < k; c += 257) at(3, c) = 0x0000;
  // A row at the escape bound, another with every element of one step, and
  // an outlier channel past the bound (kept raw: the production chain).
  for (int i = 0; i < dgpp::kBf12MaxRowEscapes; ++i) at(4, static_cast<int>(rng() % k)) = 0x0000;
  for (int c = 0; c < k; c += 5) at(9, c) = static_cast<uint16_t>(((c % 200) + 20) << 7 | (c & 0x7F));
  for (int c = 0; c < k; c += 3) at(95, c) = static_cast<uint16_t>(0x8000 | (((c % 90) + 60) << 7));
  for (int j = 0; j < 8; ++j) at(5, 1024 + 3 * 8 + j) = static_cast<uint16_t>(0x0100 + j);
  require(dgpp::bf12_encode(w.data(), n, k).raw_rows == 2, "the two outlier rows stay raw");
  bitwise_gate(w, n, k, "adversarial");
  // Inf and NaN weights ride the table like any other exponent (their
  // products are the bf16 kernel's: the gate above is bitwise, NaN included).
  at(6, 100) = 0x7F80;  // +inf
  at(7, 200) = 0x7FC0;  // NaN
  bitwise_gate(w, n, k, "non-finite");
}

DGPP_TEST(bf12_refuses_what_it_cannot_hold) {
  const int n = 8, k = 1024;
  auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xBAD);
  // Rows whose exponents sweep 200 values: no fifteen-wide window holds
  // them. One of eight may stay raw; two of eight and the packing does not pay.
  for (int row : {3, 6})
    for (int c = 0; c < k; ++c)
      w[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(((c % 200) + 20) << 7);
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(!h.ok && h.raw_rows == 2, "too many raw rows keep the matrix bf16");
  // An all-zero row is one exponent: it packs inside its own window.
  for (int c = 0; c < k; ++c) w[static_cast<size_t>(3) * k + c] = 0;
  const Bf12Host z = dgpp::bf12_encode(w.data(), n, k);
  require(z.ok, "an all-zero row packs");
  require(!dgpp::bf12_encode(w.data(), n, 1000).ok, "k off the super-block grid is refused");
  require(!dgpp::bf12_shape_ok(4, 131072), "a column past 16 bits is refused");
  dgpp::CublasLtGemm gemm;
  Bf12Matrix empty;
  bool threw = false;
  try {
    gemm.register_bf12(w.data(), empty);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an empty companion is refused");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
