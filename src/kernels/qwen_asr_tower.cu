#include <cstdint>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qwen_asr_tower.hpp"

namespace dgpp {
namespace {
constexpr int B = 256;
constexpr int kMelBins = 128;
constexpr int kChunkFrames = 100;
constexpr int kSteps = 13;  // conv_out time steps per chunk
constexpr int kFreqOut = 16;
constexpr int kTowerDim = 1024;

__device__ inline float bf(uint16_t v) { return bf16_bits_to_float(v); }
__device__ inline uint16_t fb(float v) { return float_to_bf16_bits(v); }

// One thread per output element: out[(c*ho*wo + y*wo + x)*co + oc] is the
// bias plus the fp32 sum over input channels and 3x3 taps (zero past the
// edges), rounded once to bf16. Taps index the weight's flattened
// [co, ci*9] rows as (ci*9 + p*3 + q), the checkpoint's [co, ci, 3, 3]
// row-major order. Input element (c, iy, ix, ch) sits at
// c*chunk_stride + iy*row_stride + ix (times ci channels for bf16 input).
template <typename In>
__global__ void __launch_bounds__(256)
    conv(const In* in, const uint16_t* weight, const uint16_t* bias, uint16_t* out, int chunks,
         int ci, int hi, int wi, int ho, int wo, int co, int row_stride, int chunk_stride) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t cells_per_chunk = static_cast<int64_t>(ho) * wo;
  const int64_t total = static_cast<int64_t>(chunks) * cells_per_chunk * co;
  if (i >= total) return;
  const int oc = static_cast<int>(i % co);
  const int64_t cell = i / co;
  const int c = static_cast<int>(cell / cells_per_chunk);
  const int64_t local = cell % cells_per_chunk;
  const int y = static_cast<int>(local / wo), x = static_cast<int>(local % wo);
  const uint16_t* wrow = weight + static_cast<int64_t>(oc) * ci * 9;
  float acc = bf(bias[oc]);
  for (int ch = 0; ch < ci; ++ch)
    for (int p = 0; p < 3; ++p)
      for (int q = 0; q < 3; ++q) {
        const int iy = y * 2 + p - 1, ix = x * 2 + q - 1;
        if (iy < 0 || iy >= hi || ix < 0 || ix >= wi) continue;
        const int64_t at =
            static_cast<int64_t>(c) * chunk_stride + iy * row_stride + ix;
        float v;
        if constexpr (sizeof(In) == 2)
          v = bf(in[at * ci + ch]);
        else
          v = in[at];
        acc += v * bf(wrow[ch * 9 + p * 3 + q]);
      }
  out[i] = fb(acc);
}

__global__ void conv_out_gather(const uint16_t* h3, uint16_t* rows, int chunks) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * 7680;
  if (i >= total) return;
  const int64_t t = (i / 7680) % kSteps;
  const int64_t c = i / (static_cast<int64_t>(kSteps) * 7680);
  const int64_t f = i % 7680;
  const int ch = static_cast<int>(f / kFreqOut), fq = static_cast<int>(f % kFreqOut);
  rows[i] = h3[(static_cast<int64_t>(c) * (kFreqOut * kSteps) + fq * kSteps + t) * 480 + ch];
}

__global__ void add_pos(uint16_t* rows, const uint16_t* pos, int chunks) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * kTowerDim;
  if (i >= total) return;
  const int64_t t = (i / kTowerDim) % kSteps;
  rows[i] = fb(bf(rows[i]) + bf(pos[t * kTowerDim + i % kTowerDim]));
}

__global__ void pack_rows(const uint16_t* conv_rows, uint16_t* packed, const int* valid,
                          const int* base, int chunks) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Over-provisioned by the valid-row bound; idle threads carry no row.
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * kTowerDim;
  if (i >= total) return;
  const int64_t c = i / (static_cast<int64_t>(kSteps) * kTowerDim);
  const int64_t t = (i / kTowerDim) % kSteps;
  if (t >= valid[c]) return;
  packed[(static_cast<int64_t>(base[c]) + t) * kTowerDim + i % kTowerDim] =
      conv_rows[i];
}

__global__ void to_head_major(const uint16_t* in, uint16_t* out, int rows, int heads, int hd) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * heads * hd;
  if (i >= total) return;
  const int d = static_cast<int>(i % hd);
  const int64_t rh = i / hd;
  const int h = static_cast<int>(rh % heads);
  const int64_t r = rh / heads;
  out[(static_cast<int64_t>(h) * rows + r) * hd + d] = in[i];
}
}  // namespace

#define DGPP_ASR_LAUNCH(kernel, total, stream, ...) \
  kernel<<<static_cast<unsigned>((total + B - 1) / B), B, 0, stream>>>(__VA_ARGS__); \
  DGPP_CUDA_OK(cudaGetLastError())

void qwen_asr_conv_mel(const float* feats, const uint16_t* weight, const uint16_t* bias,
                       uint16_t* out, int chunks, int frames, int ho, int wo, int co,
                       cudaStream_t stream) {
  // Mel is [128, frames] bin-major; chunk c starts at frame c*100.
  const int64_t total = static_cast<int64_t>(chunks) * ho * wo * co;
  DGPP_ASR_LAUNCH((conv<float>), total, stream, feats, weight, bias, out, chunks, 1, 128, 100,
                  ho, wo, co, frames, 100);
}

void qwen_asr_conv(const uint16_t* act, const uint16_t* weight, const uint16_t* bias,
                   uint16_t* out, int chunks, int ci, int hi, int wi, int ho, int wo, int co,
                   cudaStream_t stream) {
  // Cell-first acts: chunk c is hi*wi rows of ci at stride hi*wi.
  const int64_t total = static_cast<int64_t>(chunks) * ho * wo * co;
  DGPP_ASR_LAUNCH((conv<uint16_t>), total, stream, act, weight, bias, out, chunks, ci, hi, wi,
                  ho, wo, co, wi, hi * wi);
}

void qwen_asr_conv_out_gather(const uint16_t* h3, uint16_t* rows, int chunks,
                              cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * 7680;
  DGPP_ASR_LAUNCH(conv_out_gather, total, stream, h3, rows, chunks);
}

void qwen_asr_add_pos(uint16_t* rows, const uint16_t* pos, int chunks, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * kTowerDim;
  DGPP_ASR_LAUNCH(add_pos, total, stream, rows, pos, chunks);
}

void qwen_asr_pack_rows(const uint16_t* conv_rows, uint16_t* packed, const int* valid,
                        const int* base, int chunks, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(chunks) * kSteps * kTowerDim;
  DGPP_ASR_LAUNCH(pack_rows, total, stream, conv_rows, packed, valid, base, chunks);
}

void qwen_asr_to_head_major(const uint16_t* in, uint16_t* out, int rows, int heads, int hd,
                            cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(rows) * heads * hd;
  DGPP_ASR_LAUNCH(to_head_major, total, stream, in, out, rows, heads, hd);
}
}  // namespace dgpp
