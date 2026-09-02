#include "kernels/glm_moe_launch.hpp"

#include <climits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemv.cuh"

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
  // old kernel produced, from the same bf16 operands. The chain is
  // FMA-latency bound (~7us at 4096); scalar 2-byte smem loads made it
  // load-bound (53us, 41us unrolled), so the operands stream in as
  // 16-byte vectors — 8 elements per pair of loads — and the FMAs run
  // from registers. The FMA order is untouched.
  float dot = 0.f;
  if (vector_loads) {
    const uint4* xv = reinterpret_cast<const uint4*>(sx);
    const uint4* wv = reinterpret_cast<const uint4*>(sw);
    const int vecs = hidden_dim / 8;
#pragma unroll 4
    for (int v = 0; v < vecs; ++v) {
      const uint4 xq = xv[v];
      const uint4 wq = wv[v];
      const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
      const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] & 0xFFFFu)),
                        bf16_bits_to_float(static_cast<uint16_t>(ww[i] & 0xFFFFu)),
                        dot);
        dot = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[i] >> 16)),
                        bf16_bits_to_float(static_cast<uint16_t>(ww[i] >> 16)),
                        dot);
      }
    }
  } else {
    for (int k = 0; k < hidden_dim; ++k)
      dot = __fmaf_rn(bf16_bits_to_float(sx[k]), bf16_bits_to_float(sw[k]),
                      dot);
  }
  const float s = 1.0f / (1.0f + expf(-dot));
  const size_t at = static_cast<size_t>(token) * n_experts + e;
  scores[at] = s;
  biased[at] = s + bias[e];
}

// One warp per token: top-k over the biased scores (strict > keeps the
// LOWER expert id on ties), ids sorted ascending (the accumulation order),
// weights normalized with per-element division — the reference's
// elementwise ops, op for op. The selection scribbles -INFINITY into a
// shared-memory COPY so the exported biased row (near-tie certification
// reads every expert's true score) survives intact.
//
// The argmax is exact arithmetic (compares, no rounding), so spreading it
// over the warp is bit-identical to the serial scan it replaces: each lane
// scans its strided experts with the same "strict >, from -inf" rule, and
// the shuffle reduction keeps the greater value, the LOWER id on equal
// values — the serial scan's outcome by definition. The serial version
// ran 8 x 288 dependent smem loads on one thread: 16 us per layer.
__device__ __forceinline__ void warp_argmax_lowest_id(float& v, int& id) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, off);
    const int oid = __shfl_xor_sync(0xFFFFFFFFu, id, off);
    if (ov > v || (ov == v && oid < id)) {
      v = ov;
      id = oid;
    }
  }
}

__global__ void moe_router_select_kernel(const float* __restrict__ scores,
                                         const float* __restrict__ biased,
                                         int32_t* __restrict__ ids,
                                         float* __restrict__ weights,
                                         int tokens, int n_experts, int top_k,
                                         float routed_scaling_factor,
                                         int norm_topk) {
  static_assert(kRouterSelectThreads == 32, "one warp per token");
  extern __shared__ float sel_smem[];  // [2][n_experts]: scores | biased
  float* s_scores = sel_smem;
  float* s_biased = sel_smem + n_experts;
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const size_t row = static_cast<size_t>(token) * n_experts;
  const int lane = threadIdx.x;
  for (int e = lane; e < n_experts; e += kRouterSelectThreads) {
    s_scores[e] = scores[row + e];
    s_biased[e] = biased[row + e];
  }
  __syncwarp();

  int sel[16];
  float wsel[16];
  for (int r = 0; r < top_k; ++r) {
    // Lane-local scan, the serial rule verbatim; a lane with no candidate
    // holds (-inf, INT_MAX) and loses every comparison.
    int best = INT_MAX;
    float bv = -INFINITY;
    for (int e = lane; e < n_experts; e += kRouterSelectThreads) {
      if (s_biased[e] > bv) {
        bv = s_biased[e];
        best = e;
      }
    }
    warp_argmax_lowest_id(bv, best);
    sel[r] = best;
    wsel[r] = s_scores[best];
    __syncwarp();
    if (lane == 0) s_biased[best] = -INFINITY;
    __syncwarp();
  }
  if (lane != 0) return;
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

// The fp32 accumulation chain (2026-09-02, expert slicing). Each expert's
// down projection arrives UNROUNDED (fp32 partial dots over this rank's
// slice of the intermediate dim); the chain is one fma per expert in
// ascending expert order, the shared expert last with weight 1, and the
// sum rounds to bf16 exactly once — when it leaves for the wire. The
// per-expert bf16 roundings the reference's index_add happened to perform
// are gone on purpose: fewer roundings, and a slice of an expert cannot
// reproduce the whole expert's rounding anyway. fmaf(w, y, acc) is the one
// op both paths (host segments, decode slots) issue, in the same order, so
// the decode path's pin against the host path stays bitwise.
__global__ void moe_accum_kernel(float* __restrict__ acc,
                                 const float* __restrict__ y,
                                 const int32_t* __restrict__ rows,
                                 const float* __restrict__ row_w, int64_t n,
                                 int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / hidden;
  float* dst = acc + static_cast<int64_t>(rows[r]) * hidden + (i - r * hidden);
  *dst = __fmaf_rn(row_w[r], y[i], *dst);
}

__global__ void moe_round_bf16_kernel(uint16_t* __restrict__ out,
                                      const float* __restrict__ acc,
                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  out[i] = float_to_bf16_bits(acc[i]);
}

// ---- decode-slot path (the sync-free MoE, 2026-09-01) --------------------
//
// One block per (8-row group, slot); the block resolves its slot's expert
// from the DEVICE route, stages the slot's activation row in smem, and each
// warp runs the fp8_gemv core on one weight row. The host-orchestrated path
// computes each expert's contribution through launch_scale_gemm_*, which
// dispatches m<=4 to the SAME core — glm_moe_test's bitwise gate pins the
// two (any change to one side's arithmetic breaks it; that gate is the
// twin-keeping mechanism). Every expert is local (each rank holds a slice
// of all of them), so there is no foreign-expert case.

// The slot's matrix for `which` (0 gate, 1 up, 2 down): routed slots read
// the ROUTE (device data — the whole point); the shared slot (j == top_k)
// uses the launch-arg matrices.
struct SlotMatrix {
  const uint8_t* payload;
  const float* scales;
  int n;
  int k;
};

__device__ __forceinline__ SlotMatrix resolve_slot_matrix(
    int slot, int top_k, const int32_t* __restrict__ ids,
    const MoeExpertView* __restrict__ views, int which, int n_routed,
    int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload,
    const float* __restrict__ sh_scales) {
  const int t = slot / (top_k + 1);
  const int j = slot - t * (top_k + 1);
  if (j < top_k) {
    const int e = ids[static_cast<size_t>(t) * top_k + j];
    const MoeExpertView& v = views[static_cast<size_t>(e) * 3 + which];
    return SlotMatrix{v.payload, v.scales, n_routed, k_routed};
  }
  return SlotMatrix{sh_payload, sh_scales, n_shared, k_shared};
}

// The down projection per slot: out[slot, :] = fp32 dot(down_row, act[slot]).
// Unrounded — the accumulation chain below owns the single rounding.
__global__ void moe_slot_down_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload, const float* __restrict__ sh_scales,
    float* __restrict__ out, int out_stride, int slots, int top_k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  const int slot = blockIdx.y;
  if (slot >= slots) return;
  const SlotMatrix m =
      resolve_slot_matrix(slot, top_k, ids, views, /*which=*/2, n_routed,
                          k_routed, n_shared, k_shared, sh_payload, sh_scales);
  if (n0 >= m.n) return;  // entirely outside (shared's shorter n)
  // The down projection consumes THIS SLOT's activation row.
  fp8_gemv::stage_activations<1>(act + static_cast<size_t>(slot) * act_stride,
                                 act_stride, m.k, sx);
  __syncthreads();
  fp8_gemv::block_rows<1>(m.payload, m.scales, sx, n0, m.n, m.k,
                          out + static_cast<size_t>(slot) * out_stride,
                          static_cast<size_t>(out_stride));
}

// The gate GEMV, the up GEMV and the swiglu in ONE launch: a warp computes
// its row's gate and up dots from the same staged activation and applies
// moe_swiglu_clamp_kernel's math to the bf16-rounded dots in registers.
// Bit-identical to the three-launch chain (the dots are block_rows' dots,
// rounded to bf16 exactly where the intermediate buffers rounded them);
// what disappears is two launches and the gate/up round trip through
// memory.
__global__ void moe_slot_gate_up_swiglu_kernel(
    const uint16_t* __restrict__ x, size_t x_stride,
    const int32_t* __restrict__ ids, const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_gate_payload,
    const float* __restrict__ sh_gate_scales,
    const uint8_t* __restrict__ sh_up_payload,
    const float* __restrict__ sh_up_scales, uint16_t* __restrict__ act,
    int act_stride, int slots, int top_k, float limit) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  const int slot = blockIdx.y;
  if (slot >= slots) return;
  const SlotMatrix gate = resolve_slot_matrix(
      slot, top_k, ids, views, /*which=*/0, n_routed, k_routed, n_shared,
      k_shared, sh_gate_payload, sh_gate_scales);
  const SlotMatrix up = resolve_slot_matrix(
      slot, top_k, ids, views, /*which=*/1, n_routed, k_routed, n_shared,
      k_shared, sh_up_payload, sh_up_scales);
  if (n0 >= gate.n) return;
  const size_t token = static_cast<size_t>(slot / (top_k + 1));
  fp8_gemv::stage_activations<1>(x + token * x_stride, x_stride, gate.k, sx);
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = n0 + warp;
  if (row >= gate.n) return;
  const int scale_cols = (gate.k + 127) / 128;
  const size_t scale_row = static_cast<size_t>(row / 128) * scale_cols;
  float g_acc[1], u_acc[1];
  fp8_gemv::row_dots<1>(gate.payload + static_cast<size_t>(row) * gate.k,
                        gate.scales + scale_row, sx, gate.k, lane, g_acc);
  fp8_gemv::row_dots<1>(up.payload + static_cast<size_t>(row) * up.k,
                        up.scales + scale_row, sx, up.k, lane, u_acc);
  if (lane != 0) return;
  // moe_swiglu_clamp_kernel, op for op, on the bf16-rounded dots.
  float g = bf16_bits_to_float(float_to_bf16_bits(g_acc[0]));
  float u = bf16_bits_to_float(float_to_bf16_bits(u_acc[0]));
  if (g > limit) g = limit;  // gate: NO lower clamp (reference asymmetry)
  u = fminf(fmaxf(u, -limit), limit);
  const uint16_t t = float_to_bf16_bits(g * sigmoidf_acc(g));  // rounding 1
  act[static_cast<size_t>(slot) * act_stride + row] =
      float_to_bf16_bits(bf16_bits_to_float(t) * u);  // rounding 2
}

// The ordered decode accumulation — moe_accum_kernel's chain, op for op:
// start at 0 (the host path's memset), fma each expert ascending (the
// router's id order), the shared expert last with weight 1 (fmaf(1, y, a)
// is exactly a + y), round to bf16 once.
__global__ void moe_slot_accum_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ contrib,
    const float* __restrict__ weights, int64_t n, int hidden, int top_k) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const int64_t t = i / hidden;
  const int c = static_cast<int>(i - t * hidden);
  const int K = top_k;
  const int64_t base = t * (K + 1);  // routed slots + shared, per token

  float a = 0.f;
  for (int j = 0; j < K; ++j)
    a = __fmaf_rn(weights[static_cast<size_t>(t) * K + j],
                  contrib[static_cast<size_t>(base + j) * hidden + c], a);
  a = __fmaf_rn(1.0f, contrib[static_cast<size_t>(base + K) * hidden + c], a);
  out[t * hidden + c] = float_to_bf16_bits(a);
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

void launch_moe_accum(float* acc, const float* y, const int32_t* rows,
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

void launch_moe_round_bf16(uint16_t* out, const float* acc, int64_t n,
                           cudaStream_t stream) {
  if (n <= 0) return;
  if (!out || !acc) throw std::invalid_argument("moe_round: null pointer");
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_round_bf16_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, acc, n);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {

void check_slot_args(const void* x, const int32_t* ids,
                     const MoeExpertView* views, const void* out,
                     int n_routed, int k_routed, int n_shared, int k_shared,
                     const char* who) {
  if (!x || !ids || !views || !out)
    throw std::invalid_argument(std::string(who) + ": null pointer");
  if (n_routed <= 0 || k_routed <= 0 || n_shared <= 0 || k_shared <= 0)
    throw std::invalid_argument(std::string(who) + ": degenerate dims");
  // The GEMV core's contract (fp8_gemv.cuh): k a multiple of 16 so every
  // 16-byte chunk lies inside one scale block. The routed payloads live in
  // the device table and ride the loader's 256-byte alignment contract;
  // the shared payloads are checked by the callers below.
  if (k_routed % fp8_gemv::kChunkBytes != 0 || !gemv::smem_fits(1, k_routed))
    throw std::invalid_argument(
        std::string(who) + ": k must be a multiple of 16 (16B-aligned "
                           "payloads)");
}

}  // namespace

void launch_moe_slot_down(const uint16_t* act, size_t act_stride,
                          const int32_t* ids, const MoeExpertView* views,
                          int n_routed, int k_routed, int n_shared,
                          int k_shared, const uint8_t* sh_payload,
                          const float* sh_scales, float* out, int out_stride,
                          int slots, int top_k, cudaStream_t stream) {
  if (slots <= 0) return;
  check_slot_args(act, ids, views, out, n_routed, k_routed, n_shared, k_shared,
                  "moe_slot_down");
  if (!sh_payload || !sh_scales || !fp8_gemv::shape_ok(sh_payload, 1, k_shared))
    throw std::invalid_argument(
        "moe_slot_down: shared payload must be 16B-aligned with k % 16 == 0");
  if (out_stride < n_routed || out_stride < n_shared)
    throw std::invalid_argument("moe_slot_down: out_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  const dim3 grid((max_n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps,
                  static_cast<unsigned>(slots));
  moe_slot_down_kernel<<<grid, fp8_gemv::kThreads,
                         fp8_gemv::smem_bytes(1, max_k), stream>>>(
      act, act_stride, ids, views, n_routed, k_routed, n_shared, k_shared,
      sh_payload, sh_scales, out, out_stride, slots, top_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const MoeExpertView* views, int n_routed, int k_routed, int n_shared,
    int k_shared, const uint8_t* sh_gate_payload, const float* sh_gate_scales,
    const uint8_t* sh_up_payload, const float* sh_up_scales, uint16_t* act,
    int act_stride, int slots, int top_k, float limit, cudaStream_t stream) {
  if (slots <= 0) return;
  check_slot_args(x, ids, views, act, n_routed, k_routed, n_shared, k_shared,
                  "moe_slot_gate_up");
  if (!sh_gate_payload || !sh_gate_scales || !sh_up_payload || !sh_up_scales ||
      !fp8_gemv::shape_ok(sh_gate_payload, 1, k_shared) ||
      !fp8_gemv::shape_ok(sh_up_payload, 1, k_shared))
    throw std::invalid_argument(
        "moe_slot_gate_up: shared payloads must be 16B-aligned with "
        "k % 16 == 0");
  if (act_stride < n_routed || act_stride < n_shared)
    throw std::invalid_argument("moe_slot_gate_up: act_stride below n");
  const int max_n = n_routed > n_shared ? n_routed : n_shared;
  const int max_k = k_routed > k_shared ? k_routed : k_shared;
  const dim3 grid((max_n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps,
                  static_cast<unsigned>(slots));
  moe_slot_gate_up_swiglu_kernel<<<grid, fp8_gemv::kThreads,
                                   fp8_gemv::smem_bytes(1, max_k), stream>>>(
      x, x_stride, ids, views, n_routed, k_routed, n_shared, k_shared,
      sh_gate_payload, sh_gate_scales, sh_up_payload, sh_up_scales, act,
      act_stride, slots, top_k, limit);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_accum(uint16_t* out, const float* contrib,
                           const float* weights, int tokens, int hidden,
                           int top_k, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!out || !contrib || !weights)
    throw std::invalid_argument("moe_slot_accum: null pointer");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_slot_accum_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, contrib, weights, n, hidden, top_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
