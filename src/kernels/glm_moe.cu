#include "kernels/glm_moe_launch.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kRouterDotThreads = 128;
constexpr int kRouterSelectThreads = 32;
constexpr int kElemThreads = 256;

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The router in two kernels (2026-09-01, the T=1 profile). The original
// one-block-per-token design put every expert's 4096-long dot on ONE SM
// with each thread striding a different weight row (1/16 sector
// efficiency) — 617us per layer at decode, 26ms of a 190ms step for a
// 2.4MB read that costs 10us at line rate. Here each block owns ONE
// (expert, token) pair: the block stages both rows through shared memory
// with coalesced 16-byte loads, then thread 0 runs the SAME sequential
// FMA chain over k the old kernel ran — the per-expert reduction order is
// unchanged (fixed sequential, launch-shape independent), so the logits,
// the selection, and every near-tie land on the identical bits. The chain
// is ~7us of dependent FMAs; 288 blocks spread it over every SM.
__global__ void moe_router_dots_kernel(const uint16_t* __restrict__ hidden,
                                       const uint16_t* __restrict__ gate,
                                       const float* __restrict__ bias,
                                       float* __restrict__ scores,
                                       float* __restrict__ biased, int tokens,
                                       int hidden_dim, int n_experts,
                                       int vector_loads) {
  extern __shared__ uint16_t rows_smem[];  // [2][hidden_dim]: x row | w row
  uint16_t* sx = rows_smem;
  uint16_t* sw = rows_smem + hidden_dim;
  const int e = blockIdx.x;
  const int token = blockIdx.y;
  if (e >= n_experts || token >= tokens) return;
  const uint16_t* x = hidden + static_cast<size_t>(token) * hidden_dim;
  const uint16_t* w = gate + static_cast<size_t>(e) * hidden_dim;

  if (vector_loads) {
    // 16-byte vectors: the launcher verified 16B alignment of both bases
    // and hidden_dim % 8 == 0 (row strides stay aligned).
    const int vecs = hidden_dim / 8;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    uint4* sxv = reinterpret_cast<uint4*>(sx);
    uint4* swv = reinterpret_cast<uint4*>(sw);
    for (int i = threadIdx.x; i < vecs; i += kRouterDotThreads) {
      sxv[i] = xv[i];
      swv[i] = wv[i];
    }
  } else {
    for (int i = threadIdx.x; i < hidden_dim; i += kRouterDotThreads) {
      sx[i] = x[i];
      sw[i] = w[i];
    }
  }
  __syncthreads();
  if (threadIdx.x != 0) return;

  // The one fixed sequential reduction order (k ascending) — the bits the
  // old kernel produced, from the same bf16 operands.
  float dot = 0.f;
  for (int k = 0; k < hidden_dim; ++k)
    dot = __fmaf_rn(bf16_bits_to_float(sx[k]), bf16_bits_to_float(sw[k]), dot);
  const float s = 1.0f / (1.0f + expf(-dot));
  const size_t at = static_cast<size_t>(token) * n_experts + e;
  scores[at] = s;
  biased[at] = s + bias[e];
}

// One block per token: top-k over the biased scores (strict > keeps the
// LOWER expert id on ties), ids sorted ascending (the accumulation order),
// weights normalized with per-element division — the reference's
// elementwise ops, op for op. The selection scribbles -INFINITY into a
// shared-memory COPY so the exported biased row (near-tie certification
// reads every expert's true score) survives intact.
__global__ void moe_router_select_kernel(const float* __restrict__ scores,
                                         const float* __restrict__ biased,
                                         int32_t* __restrict__ ids,
                                         float* __restrict__ weights,
                                         int tokens, int n_experts, int top_k,
                                         float routed_scaling_factor,
                                         int norm_topk) {
  extern __shared__ float sel_smem[];  // [2][n_experts]: scores | biased
  float* s_scores = sel_smem;
  float* s_biased = sel_smem + n_experts;
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const size_t row = static_cast<size_t>(token) * n_experts;
  for (int e = threadIdx.x; e < n_experts; e += kRouterSelectThreads) {
    s_scores[e] = scores[row + e];
    s_biased[e] = biased[row + e];
  }
  __syncthreads();
  if (threadIdx.x != 0) return;

  int sel[16];
  float wsel[16];
  for (int r = 0; r < top_k; ++r) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < n_experts; ++e) {
      // strict >: equal biased scores keep the LOWER expert id.
      if (s_biased[e] > bv) {
        bv = s_biased[e];
        best = e;
      }
    }
    sel[r] = best;
    wsel[r] = s_scores[best];
    s_biased[best] = -INFINITY;
  }
  // Ascending expert order (insertion sort; top_k <= 16).
  for (int i = 1; i < top_k; ++i) {
    const int id = sel[i];
    const float w = wsel[i];
    int j = i - 1;
    while (j >= 0 && sel[j] > id) {
      sel[j + 1] = sel[j];
      wsel[j + 1] = wsel[j];
      --j;
    }
    sel[j + 1] = id;
    wsel[j + 1] = w;
  }
  // Normalize with per-element division (the reference's elementwise op,
  // not reciprocal-multiply), then scale.
  float denom = 0.f;
  for (int i = 0; i < top_k; ++i) denom = __fadd_rn(denom, wsel[i]);
  denom = __fadd_rn(denom, 1e-20f);
  for (int i = 0; i < top_k; ++i) {
    const float w =
        norm_topk ? __fdiv_rn(wsel[i], denom) * routed_scaling_factor
                  : wsel[i] * routed_scaling_factor;
    ids[static_cast<size_t>(token) * top_k + i] = sel[i];
    weights[static_cast<size_t>(token) * top_k + i] = w;
  }
}

__global__ void moe_swiglu_clamp_kernel(const uint16_t* __restrict__ gate,
                                        const uint16_t* __restrict__ up,
                                        uint16_t* __restrict__ out,
                                        int64_t n, float limit) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  float g = bf16_bits_to_float(gate[i]);
  float u = bf16_bits_to_float(up[i]);
  if (g > limit) g = limit;  // gate: NO lower clamp (reference asymmetry)
  u = fminf(fmaxf(u, -limit), limit);
  const uint16_t t =
      float_to_bf16_bits(g * sigmoidf_acc(g));  // rounding 1 (silu)
  out[i] =
      float_to_bf16_bits(bf16_bits_to_float(t) * u);  // rounding 2 (product)
}

__global__ void moe_gather_rows_kernel(const uint16_t* __restrict__ src,
                                       const int32_t* __restrict__ rows,
                                       uint16_t* __restrict__ dst, int64_t n,
                                       int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  dst[i] = src[static_cast<int64_t>(rows[r]) * hidden + (i - r * hidden)];
}

__global__ void moe_accum_kernel(uint16_t* __restrict__ acc,
                                 const uint16_t* __restrict__ y,
                                 const int32_t* __restrict__ rows,
                                 const float* __restrict__ row_w, int64_t n,
                                 int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  // contribution = bf16(w * y) — the reference's .to(bf16) before the add.
  const uint16_t contrib =
      float_to_bf16_bits(row_w[r] * bf16_bits_to_float(y[i]));
  uint16_t* dst = acc + static_cast<int64_t>(rows[r]) * hidden +
                  (i - r * hidden);
  // bf16 accumulation, exactly one rounding per expert add.
  *dst = float_to_bf16_bits(bf16_bits_to_float(*dst) +
                            bf16_bits_to_float(contrib));
}

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// Tile geometry: IDENTICAL to scale_gemm.cu's (BM/BN/BK, the pad, the
// thread mapping). The host-orchestrated path computes each expert's
// contribution through launch_scale_gemm_bf16 at m=1 for decode rows —
// this kernel must produce the same bits, so any change to either side's
// tile arithmetic breaks glm_moe_test's bitwise gate. That gate is the
// twin-keeping mechanism; do not "simplify" one side without the other.
constexpr int kGemvBM = 16;
constexpr int kGemvBN = 64;
constexpr int kGemvBK = 32;
constexpr int kGemvBKPad = kGemvBK + 8;
constexpr int kGemvBlockThreads = (kGemvBN / 8) * 32;

__global__ void moe_slot_gemv_kernel(
    const uint16_t* __restrict__ x, size_t x_stride,
    const int32_t* __restrict__ ids, const MoeExpertView* __restrict__ views,
    int which, int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload, const float* __restrict__ sh_scales,
    uint16_t* __restrict__ out, int out_stride, int slots, int top_k,
    int begin, int count) {
  const int n0 = blockIdx.x * kGemvBN;
  const int slot = blockIdx.y;
  if (slot >= slots) return;
  const int K = top_k;
  const int t = slot / (K + 1);
  const int j = slot - t * (K + 1);

  // The slot's expert: routed slots read the ROUTE (device data — the
  // whole point) and skip foreign experts; the shared slot (j == K)
  // uses the launch-arg matrices. `which` picks gate/up/down.
  const uint8_t* w_payload;
  const float* scales;
  int n, k;
  if (j < K) {
    const int e = ids[static_cast<size_t>(t) * K + j];
    if (e < begin || e >= begin + count) return;  // another rank's expert
    const MoeExpertView& v =
        views[static_cast<size_t>(e - begin) * 3 + which];
    w_payload = v.payload;
    scales = v.scales;
    n = n_routed;
    k = k_routed;
  } else {
    w_payload = sh_payload;
    scales = sh_scales;
    n = n_shared;
    k = k_shared;
  }
  if (n0 >= n) return;  // entirely outside (shared's shorter n)
  // The activation row: gate/up consume token t's hidden; down consumes
  // THIS SLOT's activation (each slot's act row is its own).
  const size_t act_row =
      (which == 2) ? static_cast<size_t>(slot) : static_cast<size_t>(t);
  const uint16_t* act = x + act_row * x_stride;

  __shared__ uint16_t sA[kGemvBM][kGemvBKPad];
  __shared__ uint16_t sB[kGemvBN][kGemvBKPad];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  // m16n8k16 fragment coordinates — see scale_gemm.cu.
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  const int bnr = warp * 8 + r;
  const int scale_cols = (k + 127) / 128;
  const int scale_row = n0 / 128;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (int k0 = 0; k0 < k; k0 += kGemvBK) {
    const float s = scales[(size_t)scale_row * scale_cols + (k0 / 128)];
    // Weight tile: decode + scale + one BF16 round — the dequant bridge.
    for (int idx = threadIdx.x; idx < kGemvBN * kGemvBK;
         idx += kGemvBlockThreads) {
      const int nn = idx / kGemvBK, kk = idx % kGemvBK;
      const int gn = n0 + nn, gk = k0 + kk;
      sB[nn][kk] =
          (gn < n && gk < k)
              ? float_to_bf16_bits(
                    fp8_e4m3_bits_to_float(w_payload[(size_t)gn * k + gk]) * s)
              : 0;
    }
    // Activation tile: m=1 — row 0 real, rows 1..15 zero (the host
    // path's m=1 GEMV zero-fills the same padding).
    for (int idx = threadIdx.x; idx < kGemvBM * kGemvBK;
         idx += kGemvBlockThreads) {
      const int mm = idx / kGemvBK, kk = idx % kGemvBK;
      const int gk = k0 + kk;
      sA[mm][kk] = (mm == 0 && gk < k) ? act[gk] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kGemvBK; kk += 16) {
      const uint32_t a0 = static_cast<uint32_t>(sA[r][kk + cc]) |
                          (static_cast<uint32_t>(sA[r][kk + cc + 1]) << 16);
      const uint32_t a1 =
          static_cast<uint32_t>(sA[r + 8][kk + cc]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 1]) << 16);
      const uint32_t a2 =
          static_cast<uint32_t>(sA[r][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r][kk + cc + 9]) << 16);
      const uint32_t a3 =
          static_cast<uint32_t>(sA[r + 8][kk + cc + 8]) |
          (static_cast<uint32_t>(sA[r + 8][kk + cc + 9]) << 16);
      const uint32_t b0 =
          static_cast<uint32_t>(sB[bnr][kk + cc]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 1]) << 16);
      const uint32_t b1 =
          static_cast<uint32_t>(sB[bnr][kk + cc + 8]) |
          (static_cast<uint32_t>(sB[bnr][kk + cc + 9]) << 16);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();  // tile reads done before the next stage overwrites
  }

  // Epilogue at m=1: only row 0 is real — lanes 0..3 (r == 0) hold its
  // two outputs; the host path's m=1 epilogue stores exactly these and
  // suppresses the rest on the same gm < m bound.
  const int out_col = n0 + warp * 8 + cc;
  uint16_t* dst = out + static_cast<size_t>(slot) * out_stride;
  if (r == 0) {
    if (out_col < n) dst[out_col] = float_to_bf16_bits(c0);
    if (out_col + 1 < n) dst[out_col + 1] = float_to_bf16_bits(c1);
  }
}

// The ordered decode accumulation — the host path's chain, op for op:
// start at 0 (its memset), add each locally-owned expert ascending (one
// bf16 rounding per add: bf16(acc + bf16(w * y))), shared LAST with
// weight 1. Foreign experts contribute nothing on this rank; their
// partials arrive via the FFN all-reduce (rank order after).
__global__ void moe_slot_accum_kernel(
    uint16_t* __restrict__ acc, const uint16_t* __restrict__ contrib,
    const int32_t* __restrict__ ids, const float* __restrict__ weights,
    int64_t n, int hidden, int top_k, int begin, int count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t t = i / hidden;
  const int c = static_cast<int>(i - t * hidden);
  const int K = top_k;
  const int S = K + 1;  // routed slots + shared, per token
  const int64_t base = t * S;

  uint16_t a = float_to_bf16_bits(0.f);  // the host path's memset start
  for (int j = 0; j < K; ++j) {
    const int e = ids[static_cast<size_t>(t) * K + j];
    if (e < begin || e >= begin + count) continue;  // foreign: skip
    // contribution = bf16(w * y) — moe_accum_kernel's exact math.
    const uint16_t contrib_bits = float_to_bf16_bits(
        weights[static_cast<size_t>(t) * K + j] *
        bf16_bits_to_float(
            contrib[static_cast<size_t>(base + j) * hidden + c]));
    a = float_to_bf16_bits(bf16_bits_to_float(a) +
                           bf16_bits_to_float(contrib_bits));
  }
  // Shared last, weight 1: bf16(1.0f * y) == y bits (exact), so the
  // explicit multiply is elided — value-identical to the host segment.
  const uint16_t shared_bits =
      contrib[static_cast<size_t>(base + K) * hidden + c];
  a = float_to_bf16_bits(bf16_bits_to_float(a) +
                         bf16_bits_to_float(shared_bits));
  acc[t * hidden + c] = a;
}

void check_router_args(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       const float* scores, const float* biased) {
  if (!hidden || !gate || !bias || !ids || !weights || !scores || !biased)
    throw std::invalid_argument("moe_router: null pointer");
  if (scores == biased)
    throw std::invalid_argument("moe_router: scores and biased must not alias");
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

}  // namespace

void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream) {
  GlmMoeConfig::validate_config(cfg);
  if (tokens <= 0) return;
  check_router_args(hidden, gate, bias, ids, weights, scores, biased);
  if (cfg.n_experts > 65535 || tokens > 65535)
    throw std::invalid_argument("moe_router: grid dimension overflow");
  const int vector_loads =
      (cfg.hidden % 8 == 0 && aligned16(hidden) && aligned16(gate)) ? 1 : 0;
  const size_t dots_smem = 2 * static_cast<size_t>(cfg.hidden) * sizeof(uint16_t);
  const dim3 dots_grid(static_cast<unsigned>(cfg.n_experts),
                       static_cast<unsigned>(tokens));
  moe_router_dots_kernel<<<dots_grid, kRouterDotThreads, dots_smem, stream>>>(
      hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts,
      vector_loads);
  DGPP_CUDA_OK(cudaGetLastError());
  const size_t sel_smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
  moe_router_select_kernel<<<tokens, kRouterSelectThreads, sel_smem, stream>>>(
      scores, biased, ids, weights, tokens, cfg.n_experts, cfg.top_k,
      cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_swiglu_clamp(const uint16_t* gate, const uint16_t* up,
                             uint16_t* out, int64_t n, float limit,
                             cudaStream_t stream) {
  if (n <= 0) return;
  if (!gate || !up || !out)
    throw std::invalid_argument("moe_swiglu: null pointer");
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_swiglu_clamp_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                            stream>>>(gate, up, out, n, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_gather_rows(const uint16_t* src, const int32_t* rows,
                            uint16_t* dst, int n_rows, int hidden,
                            cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!src || !rows || !dst)
    throw std::invalid_argument("moe_gather: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_gather_rows_kernel<<<static_cast<int>(blocks), kElemThreads, 0,
                           stream>>>(src, rows, dst, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_accum(uint16_t* acc, const uint16_t* y, const int32_t* rows,
                      const float* row_weights, int n_rows, int hidden,
                      cudaStream_t stream) {
  if (n_rows <= 0) return;
  if (!acc || !y || !rows || !row_weights)
    throw std::invalid_argument("moe_accum: null pointer");
  const int64_t n = static_cast<int64_t>(n_rows) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      acc, y, rows, row_weights, n, hidden);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_gemv(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const MoeExpertView* views, int which, int n_routed, int k_routed,
    int n_shared, int k_shared, const uint8_t* sh_payload,
    const float* sh_scales, uint16_t* out, int out_stride, int slots,
    int top_k, int begin, int count, cudaStream_t stream) {
  if (slots <= 0) return;
  if (!x || !ids || !out || !sh_payload || !sh_scales)
    throw std::invalid_argument("moe_slot_gemv: null pointer");
  if (count > 0 && views == nullptr)
    throw std::invalid_argument("moe_slot_gemv: null expert table");
  if (which < 0 || which > 2)
    throw std::invalid_argument("moe_slot_gemv: which must be 0..2");
  if (n_routed <= 0 || k_routed <= 0 || n_shared <= 0 || k_shared <= 0)
    throw std::invalid_argument("moe_slot_gemv: degenerate dims");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const dim3 grid((max_n + kGemvBN - 1) / kGemvBN, static_cast<unsigned>(slots));
  moe_slot_gemv_kernel<<<grid, kGemvBlockThreads, 0, stream>>>(
      x, x_stride, ids, views, which, n_routed, k_routed, n_shared, k_shared,
      sh_payload, sh_scales, out, out_stride, slots, top_k, begin, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_accum(uint16_t* acc, const uint16_t* contrib,
                           const int32_t* ids, const float* weights,
                           int tokens, int hidden, int top_k, int begin,
                           int count, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!acc || !contrib || !ids || !weights)
    throw std::invalid_argument("moe_slot_accum: null pointer");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_slot_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      acc, contrib, ids, weights, n, hidden, top_k, begin, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
