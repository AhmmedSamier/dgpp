// Mamba-2 device kernels for the Nemotron-Transcribe port (milestone 3 of
// docs/nemotron_transcribe_plan.md): causal depthwise conv + SiLU (prefill
// and the stateful decode step), the selective scan over a carried fp32
// per-head state (prefill and the single-row decode), and the grouped
// gated RMSNorm. Every kernel mirrors src/models/nemotron_transcribe/
// reference.{hpp,cpp} one fp32 operation at a time from bf16-dequantized
// inputs (the repo builds with -ffp-contract=off, so nothing fuses here
// either); the composition — conv state + recurrent state advanced row by
// row — must reproduce the fused prefill bitwise, which is what
// tests/cuda/mamba2_test.cu gates.
//
// The scan keeps the per-head state in shared memory (d_inner lanes x
// state_size fp32: 32 KB at the release geometry, inside the static smem
// budget), each thread owning a contiguous run of `state`'s lane-columns
// for one d, so the emitted sum is a fixed-order reduction.
#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mamba2.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float dev_softplus(float x) {
  return x > 20.f ? x : log1pf(expf(x));
}
__device__ __forceinline__ float dev_silu(float x) { return x / (1.f + expf(-x)); }

// Fixed-order whole-block sum: warp shuffles within a warp, then warp 0
// reduces the per-warp partials in warp order. blockDim.x % 32 == 0.
template <int MAX_WARPS>
__device__ __forceinline__ float block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float s = lane < warps ? red[lane] : 0.f;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  if (threadIdx.x == 0) red[0] = s;
  __syncthreads();
  return red[0];
}

__global__ void conv_silu_kernel(const uint16_t* __restrict__ u, int64_t u_stride,
                                 const uint16_t* __restrict__ w, const uint16_t* __restrict__ bias,
                                 uint16_t* __restrict__ out, int64_t out_stride, int seq,
                                 int channels, int kernel) {
  constexpr int ROWS = 32;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int t0 = blockIdx.y * ROWS;
  float wreg[16];
  const int kmax = kernel > 16 ? 16 : kernel;  // geometry guard below
  for (int k = 0; k < kmax; ++k) wreg[k] = bf16_bits_to_float(w[c * kernel + k]);
  const float b = bias ? bf16_bits_to_float(bias[c]) : 0.f;
  const int tend = min(t0 + ROWS, seq);
  for (int t = t0; t < tend; ++t) {
    float acc = b;
    for (int k = 0; k < kernel; ++k) {
      const int j = t - (kernel - 1) + k;
      if (j >= 0) acc += wreg[k] * bf16_bits_to_float(u[static_cast<int64_t>(j) * u_stride + c]);
    }
    out[static_cast<int64_t>(t) * out_stride + c] = float_to_bf16_bits(dev_silu(acc));
  }
}

__global__ void conv_silu_step_kernel(const uint16_t* __restrict__ u_row, uint16_t* __restrict__ state,
                                       const uint16_t* __restrict__ w, const uint16_t* __restrict__ bias,
                                       uint16_t* __restrict__ out_row, int channels, int kernel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int hist = kernel - 1;
  uint16_t* st = state + static_cast<int64_t>(c) * hist;
  float h[15];
  const int hmax = hist > 15 ? 15 : hist;
  for (int j = 0; j < hmax; ++j) h[j] = bf16_bits_to_float(st[j]);
  float acc = bias ? bf16_bits_to_float(bias[c]) : 0.f;
  for (int k = 0; k < hist; ++k) acc += bf16_bits_to_float(w[c * kernel + k]) * h[k];
  const float nu = bf16_bits_to_float(u_row[c]);
  acc += bf16_bits_to_float(w[c * kernel + hist]) * nu;
  out_row[c] = float_to_bf16_bits(dev_silu(acc));
  for (int j = 0; j < hist - 1; ++j) st[j] = st[j + 1];
  if (hist > 0) st[hist - 1] = u_row[c];
}

// One block per head. Dynamic smem: state[hd*ss], xs[hd], Bs[ss], Cs[ss],
// red[kThreads] (every thread stores its partial sum), scalars at the tail.
__global__ void scan_kernel(const uint16_t* __restrict__ x, int64_t x_stride,
                            const uint16_t* __restrict__ B, int64_t B_stride,
                            const uint16_t* __restrict__ C, int64_t C_stride,
                            const uint16_t* __restrict__ dt, int64_t dt_stride,
                            const float* __restrict__ A_log, const float* __restrict__ Dp,
                            const float* __restrict__ dt_bias, float* __restrict__ state_io,
                            uint16_t* __restrict__ out, int64_t out_stride, int seq, int heads,
                            int hd, int groups, int ss, float dt_min) {
  extern __shared__ float sm[];
  const int h = blockIdx.x;
  const int d_inner = hd * heads;
  const int grp = h * groups / heads;
  const int total = hd * ss;
  float* st = sm;                    // total
  float* xs = st + total;            // hd
  float* Bs = xs + hd;               // ss
  float* Cs = Bs + ss;               // ss
  float* red = Cs + ss;              // kThreads
  float* scal = red + kThreads;      // 2: d, dA

  for (int e = threadIdx.x; e < total; e += blockDim.x)
    st[e] = state_io ? state_io[static_cast<int64_t>(h) * total + e] : 0.f;
  const float a = -expf(A_log[h]);
  const float dtb = dt_bias[h];
  const float Dv = Dp[h];

  for (int t = 0; t < seq; ++t) {
    if (threadIdx.x < hd) xs[threadIdx.x] = bf16_bits_to_float(x[static_cast<int64_t>(t) * x_stride + h * hd + threadIdx.x]);
    if (threadIdx.x < ss) {
      const int64_t base = static_cast<int64_t>(t) * (groups * ss) + static_cast<int64_t>(grp) * ss;
      Bs[threadIdx.x] = bf16_bits_to_float(B[base + threadIdx.x]);
      Cs[threadIdx.x] = bf16_bits_to_float(C[base + threadIdx.x]);
    }
    if (threadIdx.x == 0) {
      float d = bf16_bits_to_float(dt[static_cast<int64_t>(t) * dt_stride + h]) + dtb;
      d = dev_softplus(d);
      if (d < dt_min) d = dt_min;
      scal[0] = d;
      scal[1] = expf(d * a);
    }
    __syncthreads();
    const float d = scal[0], dA = scal[1];
    // Each thread owns a contiguous EPT-sized run inside one d's row.
    const int ept = total / kThreads;
    float part = 0.f;
    for (int k = 0; k < ept; ++k) {
      const int e = threadIdx.x * ept + k;
      const int d_i = e / ss;
      const int n = e - d_i * ss;
      const float v = st[e] * dA + d * Bs[n] * xs[d_i];
      st[e] = v;
      part += v * Cs[n];
    }
    red[threadIdx.x] = part;
    __syncthreads();
    if (threadIdx.x < hd) {
      // The tpd threads whose runs tile this d's row, in n order.
      const int tpd = ss / ept;
      const int first = threadIdx.x * tpd;
      float o = 0.f;
      for (int j = 0; j < tpd; ++j) o += red[first + j];
      out[static_cast<int64_t>(t) * out_stride + h * hd + threadIdx.x] =
          float_to_bf16_bits(o + Dv * xs[threadIdx.x]);
    }
    __syncthreads();
  }
  if (state_io)
    for (int e = threadIdx.x; e < total; e += blockDim.x)
      state_io[static_cast<int64_t>(h) * total + e] = st[e];
}

__global__ void gated_norm_kernel(const uint16_t* __restrict__ y, int64_t y_stride,
                                  const uint16_t* __restrict__ gate, int64_t gate_stride,
                                  const uint16_t* __restrict__ weight, uint16_t* __restrict__ out,
                                  int64_t out_stride, int group, float eps) {
  __shared__ float red[33];
  const int g0 = blockIdx.x * group;
  const int t = blockIdx.y;
  const int i = g0 + threadIdx.x;
  const float v = bf16_bits_to_float(y[static_cast<int64_t>(t) * y_stride + i]) *
                  dev_silu(bf16_bits_to_float(gate[static_cast<int64_t>(t) * gate_stride + i]));
  float ss_sum = block_sum<32>(v * v, red);
  const float rstd = rsqrtf(ss_sum / group + eps);
  out[static_cast<int64_t>(t) * out_stride + i] =
      float_to_bf16_bits(bf16_bits_to_float(weight[i]) * (v * rstd));
}

}  // namespace

void mamba2_conv_silu(const void* u, int64_t u_stride, const void* weight, const void* bias,
                      void* out, int64_t out_stride, int seq, int channels, int kernel,
                      cudaStream_t stream) {
  if (seq <= 0 || channels <= 0) return;
  if (kernel <= 0 || kernel > 16)
    throw std::invalid_argument("mamba2_conv_silu: kernel width");
  const dim3 block(kThreads);
  const dim3 grid((channels + kThreads - 1) / kThreads, (seq + 31) / 32);
  conv_silu_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint16_t*>(u), u_stride, static_cast<const uint16_t*>(weight),
      static_cast<const uint16_t*>(bias), static_cast<uint16_t*>(out), out_stride, seq, channels,
      kernel);
  DGPP_CUDA_OK(cudaGetLastError());
}

void mamba2_conv_silu_step(const void* u_row, void* state, const void* weight, const void* bias,
                           void* out_row, int channels, int kernel, cudaStream_t stream) {
  if (channels <= 0) return;
  if (kernel < 2 || kernel > 16)
    throw std::invalid_argument("mamba2_conv_silu_step: kernel width");
  const dim3 grid((channels + kThreads - 1) / kThreads);
  conv_silu_step_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const uint16_t*>(u_row), static_cast<uint16_t*>(state),
      static_cast<const uint16_t*>(weight), static_cast<const uint16_t*>(bias),
      static_cast<uint16_t*>(out_row), channels, kernel);
  DGPP_CUDA_OK(cudaGetLastError());
}

void mamba2_scan(const void* x, int64_t x_stride, const void* B, int64_t B_stride, const void* C,
                 int64_t C_stride, const void* dt, int64_t dt_stride, const float* A_log,
                 const float* D, const float* dt_bias, float* state, void* out, int64_t out_stride,
                 int seq, int heads, int head_dim, int groups, int state_size, float dt_min,
                 cudaStream_t stream) {
  if (seq <= 0 || heads <= 0) return;
  const int total = head_dim * state_size;
  if (head_dim < 1 || head_dim > kThreads || state_size < 1 || state_size > kThreads ||
      groups <= 0 || heads % groups != 0 || total % kThreads != 0)
    throw std::invalid_argument("mamba2_scan: geometry");
  const int ept = total / kThreads;
  if (ept > state_size || state_size % ept != 0)
    throw std::invalid_argument("mamba2_scan: state tiling");
  const size_t smem = static_cast<size_t>(total + head_dim + 2 * state_size + kThreads + 2 +
                                          head_dim) *
                      sizeof(float);
  if (smem > 48u * 1024u)
    throw std::invalid_argument("mamba2_scan: state too large for static smem budget");
  scan_kernel<<<heads, kThreads, smem, stream>>>(
      static_cast<const uint16_t*>(x), x_stride, static_cast<const uint16_t*>(B), B_stride,
      static_cast<const uint16_t*>(C), C_stride, static_cast<const uint16_t*>(dt), dt_stride,
      A_log, D, dt_bias, state, static_cast<uint16_t*>(out), out_stride, seq, heads, head_dim,
      groups, state_size, dt_min);
  DGPP_CUDA_OK(cudaGetLastError());
}

void mamba2_scan_step(const void* x_row, const void* B_row, const void* C_row, const void* dt_row,
                      const float* A_log, const float* D, const float* dt_bias, float* state,
                      void* out_row, int heads, int head_dim, int groups, int state_size,
                      float dt_min, cudaStream_t stream) {
  mamba2_scan(x_row, head_dim, B_row, groups * state_size, C_row, groups * state_size, dt_row,
              heads, A_log, D, dt_bias, state, out_row, head_dim, /*seq=*/1, heads, head_dim,
              groups, state_size, dt_min, stream);
}

void mamba2_gated_norm(const void* y, int64_t y_stride, const void* gate, int64_t gate_stride,
                       const void* weight, void* out, int64_t out_stride, int seq, int d_inner,
                       int group, float eps, cudaStream_t stream) {
  if (seq <= 0 || d_inner <= 0) return;
  if (group < 32 || group > 1024 || group % 32 != 0 || d_inner % group != 0)
    throw std::invalid_argument("mamba2_gated_norm: group width");
  const dim3 grid(d_inner / group, seq);
  gated_norm_kernel<<<grid, group, 0, stream>>>(
      static_cast<const uint16_t*>(y), y_stride, static_cast<const uint16_t*>(gate), gate_stride,
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(out), out_stride, group, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
