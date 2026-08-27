#include "kernels/kda.hpp"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {

namespace {

// ---------------------------------------------------------------------------
// Causal depthwise conv + silu
// ---------------------------------------------------------------------------

template <int CW>
__global__ void kda_conv_kernel(const uint16_t* __restrict__ src,
                                int64_t src_stride,
                                const uint16_t* __restrict__ weight,
                                uint16_t* __restrict__ state,
                                int state_width, uint16_t* __restrict__ dst,
                                int tokens, int channels) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const uint16_t* wc = weight + static_cast<int64_t>(c) * CW;
  uint16_t* sc = state + static_cast<int64_t>(c) * state_width;

  float wv[CW];
#pragma unroll
  for (int j = 0; j < CW; ++j) wv[j] = bf16_bits_to_float(wc[j]);

  // Rolling register history; starts as the incoming state, ends as the
  // last CW-1 inputs regardless of how few tokens arrive (decode).
  float hist[CW - 1];
#pragma unroll
  for (int j = 0; j < CW - 1; ++j) hist[j] = bf16_bits_to_float(sc[j]);

  for (int t = 0; t < tokens; ++t) {
    const float x =
        bf16_bits_to_float(src[static_cast<int64_t>(t) * src_stride + c]);
    // Same association order as the host reference so the fp32 fma chains
    // agree; only expf/silu can differ by an ulp.
    float acc = wv[CW - 1] * x;
#pragma unroll
    for (int j = 0; j < CW - 1; ++j) acc = fmaf(wv[j], hist[j], acc);
    const float y = acc / (1.0f + expf(-acc));  // silu, fp32
    dst[static_cast<int64_t>(t) * channels + c] = float_to_bf16_bits(y);
#pragma unroll
    for (int j = 0; j < CW - 2; ++j) hist[j] = hist[j + 1];
    hist[CW - 2] = x;
  }

#pragma unroll
  for (int j = 0; j < CW - 1; ++j) sc[j] = float_to_bf16_bits(hist[j]);
}

// ---------------------------------------------------------------------------
// Gated RMSNorm (sigmoid gate)
// ---------------------------------------------------------------------------

__global__ void kda_gated_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                         const uint16_t* __restrict__ gate,
                                         const uint16_t* __restrict__ weight,
                                         uint16_t* __restrict__ y, int dim,
                                         float eps) {
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  const uint16_t* gr = gate + row * dim;
  uint16_t* yr = y + row * dim;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  extern __shared__ float sx[];  // staged x row for the second pass
  float partial = 0.0f;
  for (int i = tid; i < dim; i += nthreads) {
    const float v = bf16_bits_to_float(xr[i]);
    sx[i] = v;
    partial = fmaf(v, v, partial);
  }
  // Warp-level tree reduction, then across warps via shared memory.
  __shared__ float warp_sums[32];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = partial;
  __syncthreads();
  float total = 0.0f;
  const int nwarps = (nthreads + 31) / 32;
  for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
  // Reference: 1 / sqrt(mean(x^2) + eps), fp32.
  const float rstd = 1.0f / sqrtf(total / static_cast<float>(dim) + eps);

  for (int i = tid; i < dim; i += nthreads) {
    const float xv = sx[i];
    const float g = bf16_bits_to_float(gr[i]);
    const float w = bf16_bits_to_float(weight[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    yr[i] = float_to_bf16_bits(xv * rstd * w * sig);
  }
}

// ---------------------------------------------------------------------------
// Recurrent KDA update
// ---------------------------------------------------------------------------

constexpr int kRecurrentLanes = 4;   // k-dim lanes cooperating on one v-row
constexpr int kRecurrentRows = 32;   // v-rows per block
constexpr int kRecurrentBlock = kRecurrentLanes * kRecurrentRows;  // 128

template <int K>
__global__ __launch_bounds__(kRecurrentBlock) void kda_recurrent_kernel(
    const uint16_t* __restrict__ qkv, const uint16_t* __restrict__ g_raw,
    const uint16_t* __restrict__ beta_raw, int64_t beta_stride,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    float* __restrict__ state, uint16_t* __restrict__ out, int tokens,
    int heads, int v_dim, float lower_bound, float scale) {
  constexpr int kCols = K / kRecurrentLanes;  // columns owned per lane
  static_assert(K % kRecurrentLanes == 0, "K must split across 4 lanes");

  const int h = blockIdx.y;
  const int v = blockIdx.x * kRecurrentRows + threadIdx.x / kRecurrentLanes;
  const int lane = threadIdx.x % kRecurrentLanes;
  const int c0 = lane * kCols;
  const bool row_valid = v < v_dim;

  // State slice S[v, c0:c0+kCols] lives in registers for the whole call:
  // one load + one store per head per chunk — the decode hot path is pure
  // state bandwidth by design.
  float s[kCols];
  float* st = state + (static_cast<int64_t>(h) * v_dim + v) * K + c0;
#pragma unroll
  for (int i = 0; i < kCols; ++i) s[i] = row_valid ? st[i] : 0.0f;

  const float a = expf(a_log[h]);
  // Fused qkv row: [q (H*K) | k (H*K) | v (H*V)].
  const int64_t qkv_stride = static_cast<int64_t>(2) * heads * K +
                             static_cast<int64_t>(heads) * v_dim;

  for (int t = 0; t < tokens; ++t) {
    const uint16_t* qrow =
        qkv + static_cast<int64_t>(t) * qkv_stride + static_cast<int64_t>(h) * K;
    const uint16_t* krow = qrow + static_cast<int64_t>(heads) * K;
    const uint16_t* vrow =
        qkv + static_cast<int64_t>(t) * qkv_stride +
        static_cast<int64_t>(2) * heads * K + static_cast<int64_t>(h) * v_dim;
    const uint16_t* grow =
        g_raw + (static_cast<int64_t>(t) * heads + h) * K + c0;
    const float* bias = dt_bias + static_cast<int64_t>(h) * K + c0;

    float q[kCols], k[kCols], u = 0.0f;
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      q[i] = bf16_bits_to_float(qrow[c0 + i]);
      k[i] = bf16_bits_to_float(krow[c0 + i]);
      qs = fmaf(q[i], q[i], qs);
      ks = fmaf(k[i], k[i], ks);
      // Gate: lower_bound / (1 + exp(-exp(A_log) * (g_raw + dt_bias))).
      // Bounded to (lower_bound, 0), so exp(gate) in (exp(lb), 1] — the
      // decay can never blow the state up. Computed fused with the decay.
      const float g = bf16_bits_to_float(grow[i]) + bias[i];
      const float gate = lower_bound / (1.0f + expf(-(a * g)));
      s[i] *= expf(gate);
    }

    // l2norm(q, k) with eps inside the sqrt; butterfly across the row's
    // lanes so every lane ends with the full sum (reference tl.sum).
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1) {
      qs += __shfl_xor_sync(0xffffffff, qs, mask);
      ks += __shfl_xor_sync(0xffffffff, ks, mask);
    }
    const float qn = 1.0f / sqrtf(qs + 1e-6f);
    const float kn = 1.0f / sqrtf(ks + 1e-6f);

#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      q[i] = q[i] * qn * scale;
      k[i] = k[i] * kn;
      u = fmaf(s[i], k[i], u);  // partial <S[v,:], k>
    }
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1)
      u += __shfl_xor_sync(0xffffffff, u, mask);

    const float beta =
        1.0f / (1.0f + expf(-bf16_bits_to_float(
                            beta_raw[static_cast<int64_t>(t) * beta_stride +
                                     h])));
    // u <- beta * (v - <S k>); the delta error.
    const float vin = row_valid ? bf16_bits_to_float(vrow[v]) : 0.0f;
    u = (vin - u) * beta;

    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < kCols; ++i) {
      s[i] = fmaf(u, k[i], s[i]);  // rank-1 write
      o = fmaf(s[i], q[i], o);     // post-update read
    }
#pragma unroll
    for (int mask = 1; mask < kRecurrentLanes; mask <<= 1)
      o += __shfl_xor_sync(0xffffffff, o, mask);

    if (row_valid && lane == 0)
      out[(static_cast<int64_t>(t) * heads + h) * v_dim + v] =
          float_to_bf16_bits(o);
  }

  if (row_valid) {
#pragma unroll
    for (int i = 0; i < kCols; ++i) st[i] = s[i];
  }
}

template <int CW>
void conv_launch(const void* src, int64_t src_stride, const void* weight,
                 void* conv_state, int state_width, void* dst, int tokens,
                 int channels, cudaStream_t stream) {
  constexpr int kBlock = 256;
  const unsigned grid =
      static_cast<unsigned>((channels + kBlock - 1) / kBlock);
  kda_conv_kernel<CW><<<grid, kBlock, 0, stream>>>(
      static_cast<const uint16_t*>(src), src_stride,
      static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(conv_state), state_width,
      static_cast<uint16_t*>(dst), tokens, channels);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void kda_causal_conv_silu_bf16(const void* src, int64_t src_row_stride,
                               const void* weight, void* conv_state,
                               int state_width, void* dst, int tokens,
                               int channels, int conv_width,
                               cudaStream_t stream) {
  if (tokens <= 0 || channels <= 0)
    throw std::invalid_argument("kda conv: empty problem");
  if (conv_width < 2 || conv_width > 8)
    throw std::invalid_argument("kda conv: conv_width must be in [2, 8]");
  if (state_width < conv_width - 1)
    throw std::invalid_argument("kda conv: state narrower than history");
  if (src_row_stride < channels)
    throw std::invalid_argument("kda conv: src row stride smaller than row");
  switch (conv_width) {
    case 2:
      conv_launch<2>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    case 3:
      conv_launch<3>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    case 4:
      conv_launch<4>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    case 5:
      conv_launch<5>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    case 6:
      conv_launch<6>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    case 7:
      conv_launch<7>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
    default:
      conv_launch<8>(src, src_row_stride, weight, conv_state, state_width, dst,
                     tokens, channels, stream);
      return;
  }
}

void kda_gated_rmsnorm_sigmoid_bf16(const void* x, const void* gate,
                                    const void* weight, void* y, int64_t rows,
                                    int dim, float eps, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0)
    throw std::invalid_argument("kda gated rmsnorm: empty problem");
  if (dim > 4096)
    throw std::invalid_argument("kda gated rmsnorm: dim exceeds smem stage");
  constexpr int kBlock = 256;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  if (shmem > 48 * 1024)
    throw std::invalid_argument("kda gated rmsnorm: dim too large for smem");
  cudaGetLastError();  // clear stale errors so failures attribute correctly
  kda_gated_rmsnorm_kernel<<<static_cast<unsigned>(rows), kBlock, shmem,
                             stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(gate),
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(y), dim,
      eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void kda_recurrent_fwd(const void* qkv, const void* g_raw, const void* beta_raw,
                       int64_t beta_row_stride, const float* a_log,
                       const float* dt_bias, float* state, void* out,
                       int tokens, int heads, int k_dim, int v_dim,
                       float lower_bound, float scale, cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || v_dim <= 0)
    throw std::invalid_argument("kda recurrent: empty problem");
  if (k_dim % kRecurrentLanes != 0)
    throw std::invalid_argument("kda recurrent: k_dim must be a multiple of 4");
  if (beta_row_stride < heads)
    throw std::invalid_argument("kda recurrent: beta stride smaller than row");

  const dim3 grid(static_cast<unsigned>((v_dim + kRecurrentRows - 1) /
                                        kRecurrentRows),
                  static_cast<unsigned>(heads), 1);
  const uint16_t* qkv16 = static_cast<const uint16_t*>(qkv);
  const uint16_t* g16 = static_cast<const uint16_t*>(g_raw);
  const uint16_t* b16 = static_cast<const uint16_t*>(beta_raw);
  uint16_t* o16 = static_cast<uint16_t*>(out);

#define DGPP_KDA_RECURRENT_DISPATCH(KLIT)                                      \
  do {                                                                         \
    kda_recurrent_kernel<KLIT><<<grid, kRecurrentBlock, 0, stream>>>(          \
        qkv16, g16, b16, beta_row_stride, a_log, dt_bias, state, o16, tokens,  \
        heads, v_dim, lower_bound, scale);                                     \
    DGPP_CUDA_OK(cudaGetLastError());                                          \
  } while (0)

  switch (k_dim) {
    case 32: DGPP_KDA_RECURRENT_DISPATCH(32); return;
    case 64: DGPP_KDA_RECURRENT_DISPATCH(64); return;
    case 128: DGPP_KDA_RECURRENT_DISPATCH(128); return;
    default:
      throw std::invalid_argument(
          "kda recurrent: k_dim must be one of {32, 64, 128}");
  }
#undef DGPP_KDA_RECURRENT_DISPATCH
}

}  // namespace dgpp
