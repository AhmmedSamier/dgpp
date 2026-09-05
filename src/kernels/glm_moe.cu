#include "kernels/glm_moe_launch.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemv.cuh"

namespace dgpp {
namespace {

constexpr int kRouterSelectThreads = 32;
constexpr int kElemThreads = 256;

__device__ inline float sigmoidf_acc(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// The router dots (2026-09-02, reassociated). One WARP per (expert, token):
// each lane accumulates a strided quarter-kilobyte of the 4096-long dot in
// fp32 from 16-byte loads, then a shuffle tree sums the 32 partials. This
// is a plain bandwidth kernel (2.4 MB of gate rows at line rate ~10 us
// cold, ~3 us from L2) where the previous one reproduced the reference's
// SEQUENTIAL fp32 chain on a single thread to stay bit-identical to it —
// 4096 dependent FMAs, ~20 us whatever the memory did. The reassociation
// moves a logit by fp32 rounding (~1e-7 relative); the router test's
// oracle certifies any changed pick as a near-tie, and the transcript
// judge (scripts/fabric_xcript.py) does the same for a fabric run.
constexpr int kRouterDotWarps = 8;
constexpr int kRouterDotThreads = 32 * kRouterDotWarps;

__device__ __forceinline__ void router_dot(const uint16_t* __restrict__ hidden,
                                           const uint16_t* __restrict__ gate,
                                           const float* __restrict__ bias,
                                           float* __restrict__ scores,
                                           float* __restrict__ biased,
                                           int token, int e, int hidden_dim,
                                           int n_experts, int vector_loads,
                                           int lane);
__device__ __forceinline__ void router_select_warp(
    const float* __restrict__ scores, const float* __restrict__ biased,
    int32_t* __restrict__ ids, float* __restrict__ weights, int token,
    int n_experts, int top_k, float routed_scaling_factor, int norm_topk,
    float* s_scores, float* s_biased, int lane);

// With `sel_ids` set the select is FUSED (2026-09-03): the token's dots
// blocks take a ticket (`counters[token]`, zeroed once, reset by the last —
// replay-safe) and the last one runs router_select_warp on warp 0, so the
// separate select launch (5 us + a graph gap per MoE layer) disappears.
// The fence/ticket order is the mHC finish's: a block fences its scores
// before its ticket; the last block fences again before reading them.
__global__ void moe_router_dots_kernel(const uint16_t* __restrict__ hidden,
                                       const uint16_t* __restrict__ gate,
                                       const float* __restrict__ bias,
                                       float* __restrict__ scores,
                                       float* __restrict__ biased, int tokens,
                                       int hidden_dim, int n_experts,
                                       int vector_loads,
                                       int32_t* __restrict__ sel_ids,
                                       float* __restrict__ sel_weights,
                                       int top_k, float routed_scaling_factor,
                                       int norm_topk,
                                       int* __restrict__ counters) {
  extern __shared__ float sel_smem[];  // [2][n_experts] when fused
  __shared__ int s_last;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * kRouterDotWarps + warp;
  const int token = blockIdx.y;
  if (token >= tokens) return;
  if (e < n_experts) router_dot(hidden, gate, bias, scores, biased, token, e,
                                hidden_dim, n_experts, vector_loads, lane);
  if (sel_ids == nullptr) return;
  __syncthreads();  // every warp's score is written
  if (threadIdx.x == 0) {
    __threadfence();
    const int ticket = atomicAdd(counters + token, 1);
    s_last = (ticket == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
    if (s_last) counters[token] = 0;  // reset for the next launch
  }
  __syncthreads();
  if (!s_last || warp != 0) return;
  __threadfence();
  router_select_warp(scores, biased, sel_ids, sel_weights, token, n_experts,
                     top_k, routed_scaling_factor, norm_topk, sel_smem,
                     sel_smem + n_experts, lane);
}

// One warp's dot for (token, expert e): sigmoid(dot) and the biased copy.
__device__ __forceinline__ void router_dot(const uint16_t* __restrict__ hidden,
                                           const uint16_t* __restrict__ gate,
                                           const float* __restrict__ bias,
                                           float* __restrict__ scores,
                                           float* __restrict__ biased,
                                           int token, int e, int hidden_dim,
                                           int n_experts, int vector_loads,
                                           int lane) {
  const uint16_t* x = hidden + static_cast<size_t>(token) * hidden_dim;
  const uint16_t* w = gate + static_cast<size_t>(e) * hidden_dim;

  float dot = 0.f;
  if (vector_loads) {
    // 16-byte vectors: the launcher verified 16B alignment of both bases
    // and hidden_dim % 8 == 0 (row strides stay aligned).
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* wv = reinterpret_cast<const uint4*>(w);
    const int vecs = hidden_dim / 8;
#pragma unroll 4
    for (int v = lane; v < vecs; v += 32) {
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
    for (int k = lane; k < hidden_dim; k += 32)
      dot = __fmaf_rn(bf16_bits_to_float(x[k]), bf16_bits_to_float(w[k]), dot);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
  if (lane != 0) return;
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

// The select, as one warp's work: the token's scores/biased rows land in
// shared memory (the selection scribbles -INFINITY into the copy so the
// exported biased row survives), then top_k rounds of argmax.
__device__ __forceinline__ void router_select_warp(
    const float* __restrict__ scores, const float* __restrict__ biased,
    int32_t* __restrict__ ids, float* __restrict__ weights, int token,
    int n_experts, int top_k, float routed_scaling_factor, int norm_topk,
    float* s_scores, float* s_biased, int lane) {
  const size_t row = static_cast<size_t>(token) * n_experts;
  for (int e = lane; e < n_experts; e += 32) {
    s_scores[e] = __ldcg(scores + row + e);
    s_biased[e] = __ldcg(biased + row + e);
  }
  __syncwarp();

  int sel[16];
  float wsel[16];
  for (int r = 0; r < top_k; ++r) {
    // Lane-local scan, the serial rule verbatim; a lane with no candidate
    // holds (-inf, INT_MAX) and loses every comparison.
    int best = INT_MAX;
    float bv = -INFINITY;
    for (int e = lane; e < n_experts; e += 32) {
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

__global__ void moe_router_select_kernel(const float* __restrict__ scores,
                                         const float* __restrict__ biased,
                                         int32_t* __restrict__ ids,
                                         float* __restrict__ weights,
                                         int tokens, int n_experts, int top_k,
                                         float routed_scaling_factor,
                                         int norm_topk) {
  static_assert(kRouterSelectThreads == 32, "one warp per token");
  extern __shared__ float sel_smem[];  // [2][n_experts]: scores | biased
  const int token = blockIdx.x;
  if (token >= tokens) return;
  router_select_warp(scores, biased, ids, weights, token, n_experts, top_k,
                     routed_scaling_factor, norm_topk, sel_smem,
                     sel_smem + n_experts, threadIdx.x);
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

// Slot EXECUTION order for a multi-token batch: slots sorted by expert
// id (the shared expert last, ties by slot — stable), so an expert two
// rows share is read from DRAM once and the second time from L2 (each fp8
// expert is ~6 MB per rank, the L2 24 MB) instead of DRAM — the only
// expert traffic a speculative verify row can share with its neighbour.
// Results are written by LOGICAL slot, so the accumulation (and every bit)
// is unchanged; only the dispatch order moves. One block, one thread per
// slot: 18-72 keys. (TRIED 2026-09-03 and reverted: ranking in the
// consuming kernels' prologue instead — two barriers and a key loop per
// block cost the 9216-block down launch +10 us, six times this kernel.)
__global__ void moe_slot_order_kernel(const int32_t* __restrict__ ids,
                                      int32_t* __restrict__ order, int slots,
                                      int top_k, int n_experts) {
  extern __shared__ int32_t keys[];
  const int s = threadIdx.x;
  if (s < slots) {
    const int t = s / (top_k + 1);
    const int j = s - t * (top_k + 1);
    keys[s] = j < top_k ? ids[static_cast<size_t>(t) * top_k + j] : n_experts;
  }
  __syncthreads();
  if (s >= slots) return;
  int pos = 0;
  for (int o = 0; o < slots; ++o)
    pos += (keys[o] < keys[s]) || (keys[o] == keys[s] && o < s);
  order[pos] = s;
}

__device__ __forceinline__ int logical_slot(const int32_t* __restrict__ order) {
  return order ? order[blockIdx.y] : static_cast<int>(blockIdx.y);
}

// The down projection per slot: out[slot, :] = fp32 dot(down_row, act[slot]).
// Unrounded — the accumulation chain below owns the single rounding.
// (TRIED 2026-09-03 and reverted: the accumulation fused behind the last
// block per column chunk. The ticket's __syncthreads + fence per block and
// the last block's 18 dependent L2 loads on the kernel's tail cost +22 us
// per layer against the 1.7 us launch they replaced.)
__global__ void moe_slot_down_kernel(
    const uint16_t* __restrict__ act, size_t act_stride,
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_payload, const float* __restrict__ sh_scales,
    float* __restrict__ out, int out_stride, int slots, int top_k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
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
    const int32_t* __restrict__ ids, const int32_t* __restrict__ order,
    const MoeExpertView* __restrict__ views,
    int n_routed, int k_routed, int n_shared, int k_shared,
    const uint8_t* __restrict__ sh_gate_payload,
    const float* __restrict__ sh_gate_scales,
    const uint8_t* __restrict__ sh_up_payload,
    const float* __restrict__ sh_up_scales, uint16_t* __restrict__ act,
    int act_stride, int slots, int top_k, float limit) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  if (static_cast<int>(blockIdx.y) >= slots) return;
  const int slot = logical_slot(order);
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

namespace {

// The prefill router's dots (2026-09-05): the warp-per-(token, expert) form
// above re-streams every gate row per token and every hidden row per
// expert from L2 (~4.9 GB per 2048-token layer, 1.7 ms). This form tiles
// 16 tokens x 16 experts per block and stages both operands' K-chunks in
// shared memory, so each row is read from L2 16x less; every (token,
// expert) dot is still ONE warp with the SAME per-lane element order (lane
// l takes 16-byte vectors v = l, l + 32, ... ascending; chunks are 32-vector
// aligned) and the same shuffle tree, so scores and biased are BITWISE the
// warp form's (glm_moe_test pins it). Vector path only (the launcher keeps
// the warp form for unaligned geometry and for decode's fused select).
constexpr int kRouterTileTokens = 16;
constexpr int kRouterTileExperts = 16;
constexpr int kRouterChunkVecs = 64;  // 512 elements per staged chunk
constexpr int kRouterTileThreads = 256;

__global__ __launch_bounds__(kRouterTileThreads) void moe_router_dots_tiled_kernel(
    const uint16_t* __restrict__ hidden, const uint16_t* __restrict__ gate,
    const float* __restrict__ bias, float* __restrict__ scores,
    float* __restrict__ biased, int tokens, int hidden_dim, int n_experts) {
  __shared__ __align__(16) uint4 sH[kRouterTileTokens][kRouterChunkVecs];
  __shared__ __align__(16) uint4 sG[kRouterTileExperts][kRouterChunkVecs];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e0 = blockIdx.x * kRouterTileExperts;
  const int t0 = blockIdx.y * kRouterTileTokens;
  const int vecs = hidden_dim / 8;
  // Warp w owns tokens 2w, 2w+1 of the tile against all 16 experts.
  float dot[2][kRouterTileExperts];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < kRouterTileExperts; ++j) dot[i][j] = 0.f;

  for (int v0 = 0; v0 < vecs; v0 += kRouterChunkVecs) {
    const int nv = min(kRouterChunkVecs, vecs - v0);
    __syncthreads();  // the previous chunk's readers are done
    for (int idx = threadIdx.x; idx < (kRouterTileTokens + kRouterTileExperts) * kRouterChunkVecs;
         idx += kRouterTileThreads) {
      const int row = idx / kRouterChunkVecs, vv = idx % kRouterChunkVecs;
      if (vv >= nv) continue;
      if (row < kRouterTileTokens) {
        const int t = t0 + row;
        if (t < tokens)
          sH[row][vv] = reinterpret_cast<const uint4*>(
              hidden + static_cast<size_t>(t) * hidden_dim)[v0 + vv];
      } else {
        const int e = e0 + row - kRouterTileTokens;
        if (e < n_experts)
          sG[row - kRouterTileTokens][vv] = reinterpret_cast<const uint4*>(
              gate + static_cast<size_t>(e) * hidden_dim)[v0 + vv];
      }
    }
    __syncthreads();
    // Lane l walks vectors v0 + l, v0 + l + 32 — its share of the row, in
    // ascending order, the warp form's exact sequence.
    for (int vv = lane; vv < nv; vv += 32) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 xq = sH[warp * 2 + i][vv];
        const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
        for (int j = 0; j < kRouterTileExperts; ++j) {
          const uint4 wq = sG[j][vv];
          const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
          float d = dot[i][j];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            d = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[q] & 0xFFFFu)),
                          bf16_bits_to_float(static_cast<uint16_t>(ww[q] & 0xFFFFu)), d);
            d = __fmaf_rn(bf16_bits_to_float(static_cast<uint16_t>(xw[q] >> 16)),
                          bf16_bits_to_float(static_cast<uint16_t>(ww[q] >> 16)), d);
          }
          dot[i][j] = d;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int t = t0 + warp * 2 + i;
#pragma unroll
    for (int j = 0; j < kRouterTileExperts; ++j) {
      float d = dot[i][j];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xFFFFFFFFu, d, off);
      const int e = e0 + j;
      if (lane == 0 && t < tokens && e < n_experts) {
        const float sc = 1.0f / (1.0f + expf(-d));
        const size_t at = static_cast<size_t>(t) * n_experts + e;
        scores[at] = sc;
        biased[at] = sc + bias[e];
      }
    }
  }
}

}  // namespace

void launch_moe_router(const uint16_t* hidden, const uint16_t* gate,
                       const float* bias, int32_t* ids, float* weights,
                       float* scores, float* biased, const GlmMoeConfig& cfg,
                       int tokens, cudaStream_t stream, int* counters,
                       bool allow_tiled) {
  GlmMoeConfig::validate_config(cfg);
  if (tokens <= 0) return;
  check_router_args(hidden, gate, bias, ids, weights, scores, biased);
  if (cfg.n_experts > 65535 || tokens > 65535)
    throw std::invalid_argument("moe_router: grid dimension overflow");
  const int vector_loads =
      (cfg.hidden % 8 == 0 && aligned16(hidden) && aligned16(gate)) ? 1 : 0;
  if (counters == nullptr && allow_tiled && vector_loads &&
      tokens >= kRouterTileTokens) {
    const dim3 grid(
        static_cast<unsigned>((cfg.n_experts + kRouterTileExperts - 1) / kRouterTileExperts),
        static_cast<unsigned>((tokens + kRouterTileTokens - 1) / kRouterTileTokens));
    moe_router_dots_tiled_kernel<<<grid, kRouterTileThreads, 0, stream>>>(
        hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts);
    DGPP_CUDA_OK(cudaGetLastError());
    const size_t sel_smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
    moe_router_select_kernel<<<tokens, kRouterSelectThreads, sel_smem, stream>>>(
        scores, biased, ids, weights, tokens, cfg.n_experts, cfg.top_k,
        cfg.routed_scaling_factor, cfg.norm_topk_prob ? 1 : 0);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  const dim3 dots_grid(
      static_cast<unsigned>((cfg.n_experts + kRouterDotWarps - 1) /
                            kRouterDotWarps),
      static_cast<unsigned>(tokens));
  const size_t sel_smem = 2 * static_cast<size_t>(cfg.n_experts) * sizeof(float);
  if (counters != nullptr) {
    // Fused: the last dots block per token selects.
    moe_router_dots_kernel<<<dots_grid, kRouterDotThreads, sel_smem, stream>>>(
        hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts,
        vector_loads, ids, weights, cfg.top_k, cfg.routed_scaling_factor,
        cfg.norm_topk_prob ? 1 : 0, counters);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  moe_router_dots_kernel<<<dots_grid, kRouterDotThreads, 0, stream>>>(
      hidden, gate, bias, scores, biased, tokens, cfg.hidden, cfg.n_experts,
      vector_loads, nullptr, nullptr, 0, 0.f, 0, nullptr);
  DGPP_CUDA_OK(cudaGetLastError());
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

namespace {


// The prefill's grouped GEMV (2026-09-04): see glm_moe_launch.hpp. Every
// thread reaches every __syncthreads — the per-warp row bound is checked
// around block_rows instead of inside it (block_rows' own early return
// would strand a warp before the next group's barrier).
template <typename OutT>
__global__ void moe_grouped_gemv_kernel(const uint16_t* __restrict__ act,
                                        size_t act_stride,
                                        const MoeSegment* __restrict__ segs,
                                        const MoeExpertView* __restrict__ views,
                                        int which, OutT* __restrict__ out,
                                        size_t out_stride, int n, int k,
                                        int rows_per_block) {
  extern __shared__ __align__(16) uint16_t sx[];
  const MoeSegment seg = segs[blockIdx.y];
  // A launch may split its segments across blocks along z, rows_per_block
  // rows each (the weight slab re-read from L2 by every block) — the
  // shared expert's segment is every token, and one block walking 64
  // groups serially was that launch's latency floor. Routed launches keep
  // z = 1: their segments are short and uneven, and a z extent sized to
  // the longest one launches blocks that only exit (measured: +40 % on the
  // down launch at 256 tokens).
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v = views[seg.expert * 3 + which];
  const int n0 = blockIdx.x * fp8_gemv::kWarps;
  const bool warp_live = n0 + static_cast<int>(threadIdx.x / 32) < n;
  for (int g = z0; g < z1; g += gemv::kMaxRows) {
    const int rows = min(gemv::kMaxRows, z1 - g);
    const uint16_t* x = act + static_cast<size_t>(seg.row0 + g) * act_stride;
    OutT* o = out + static_cast<size_t>(seg.row0 + g) * out_stride;
    if (g > z0) __syncthreads();  // the previous group is done reading sx
    switch (rows) {
      case 4:
        fp8_gemv::stage_activations<4>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<4, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride);
        break;
      case 3:
        fp8_gemv::stage_activations<3>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<3, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride);
        break;
      case 2:
        fp8_gemv::stage_activations<2>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<2, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride);
        break;
      default:
        fp8_gemv::stage_activations<1>(x, act_stride, k, sx);
        __syncthreads();
        if (warp_live)
          fp8_gemv::block_rows<1, OutT>(v.payload, v.scales, sx, n0, n, k, o,
                                        out_stride);
        break;
    }
  }
}

template <typename OutT>
void launch_moe_grouped_gemv(const uint16_t* act, size_t act_stride,
                             const MoeSegment* segs, int n_segs, int max_rows,
                             int rows_per_block, const MoeExpertView* views,
                             int which, OutT* out, size_t out_stride, int n,
                             int k, cudaStream_t stream) {
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped gemv: null pointer");
  if (k <= 0 || (k % fp8_gemv::kChunkBytes) != 0)
    throw std::invalid_argument("moe grouped gemv: k must be a positive multiple of 16");
  if (!gemv::smem_fits(gemv::kMaxRows, k))
    throw std::invalid_argument("moe grouped gemv: k exceeds the smem budget");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped gemv: max_rows must be positive");
  // No split: one z block per segment walking every row (the segments'
  // lengths need not be known on the host — device segmentation).
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  const dim3 grid((n + fp8_gemv::kWarps - 1) / fp8_gemv::kWarps,
                  static_cast<unsigned>(n_segs), z_ext);
  moe_grouped_gemv_kernel<OutT>
      <<<grid, fp8_gemv::kThreads, gemv::smem_bytes(gemv::kMaxRows, k), stream>>>(
          act, act_stride, segs, views, which, out, out_stride, n, k,
          rows_per_block);
  DGPP_CUDA_OK(cudaGetLastError());
}

// ---- the prefill's grouped tensor-core GEMM (2026-09-05) -----------------
// One block per (64-column n-tile, segment): the block walks its segment in
// 128-row m-tiles (eight warps, sixteen rows each), and for every 64-deep
// k-stage stages the activation tile (bf16, 16-byte loads) and the weight
// tile (fp8 decoded, scaled and rounded to bf16 exactly as the dequant
// bridge does — the tile kernel's values, bit for bit) in shared memory,
// then runs mma.sync m16n8k16 bf16 with fp32 accumulation in ascending k16
// order — the SAME instruction sequence per output element as
// scale_gemm_kernel, so the outputs are bitwise that kernel's whatever the
// segment or tile geometry (glm_moe_test pins it). Against the grouped
// GEMV it replaces on the prefill path: that core re-reads and re-decodes
// the expert's weights once per four rows, so a 57-row segment paid for
// the weights fifteen times; here a segment up to 128 rows pays once.
namespace mma_tile {
constexpr int BM = 128;              // eight warps x m16
constexpr int BN = 64;               // eight n8 fragments per warp
constexpr int BK = 64;               // one scale column per stage (BK | 128)
constexpr int BK_PAD = BK + 8;       // u16 pad: 144-byte rows, conflict-free
constexpr int kThreads = 256;
static_assert(128 % BN == 0 && 128 % BK == 0, "one scale per stage");
static_assert(BK_PAD * 2 % 16 == 0, "16-byte aligned smem rows");
}  // namespace mma_tile

// `act_rows` (nullable): the activation row for segment row i is
// act_rows[i] (the gather folded into the tile load — the same values the
// gathered buffer would hold). `segs == nullptr` is the DENSE form: one
// segment `dense_seg` against `dense_view`, no tables in memory (the scale
// GEMM's large-m route).
template <typename OutT>
__global__ __launch_bounds__(mma_tile::kThreads) void moe_grouped_mma_kernel(
    const uint16_t* __restrict__ act, size_t act_stride, int act_vec,
    const int32_t* __restrict__ act_rows,
    const MoeSegment* __restrict__ segs, const MoeExpertView* __restrict__ views,
    int which, OutT* __restrict__ out, size_t out_stride, int n, int k,
    int rows_per_block, MoeSegment dense_seg, MoeExpertView dense_view) {
  using namespace mma_tile;
  __shared__ __align__(16) uint16_t sA[BM][BK_PAD];
  __shared__ __align__(16) uint16_t sB[BN][BK_PAD];
  const MoeSegment seg = segs != nullptr ? segs[blockIdx.y] : dense_seg;
  const int z0 = static_cast<int>(blockIdx.z) * rows_per_block;
  if (z0 >= seg.rows) return;  // beyond this segment's rows (no barrier yet)
  const int z1 = min(seg.rows, z0 + rows_per_block);
  const MoeExpertView v =
      segs != nullptr ? views[seg.expert * 3 + which] : dense_view;
  const int n0 = static_cast<int>(blockIdx.x) * BN;
  const int scale_cols = (k + 127) / 128;
  const int scale_row = n0 / 128;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int r = lane / 4;
  const int cc = (lane % 4) * 2;
  // The weight tile's load geometry: thread t decodes 16 consecutive k of
  // n-row t/4 (64 rows x 4 quads = 256 threads, one 16-byte load each).
  const int b_row = static_cast<int>(threadIdx.x) / 4;
  const int b_kq = (static_cast<int>(threadIdx.x) % 4) * 16;

  for (int m0 = z0; m0 < z1; m0 += BM) {
    const int m_rows = min(BM, z1 - m0);
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

    for (int k0 = 0; k0 < k; k0 += BK) {
      const float s = v.scales[static_cast<size_t>(scale_row) * scale_cols + (k0 / 128)];
      // Weight tile: decode + scale + one BF16 round, zero outside [n, k).
      {
        const int gn = n0 + b_row, gk = k0 + b_kq;
        uint32_t packed[8];
        if (gn < n && gk + 16 <= k) {
          const uint4 raw = *reinterpret_cast<const uint4*>(
              v.payload + static_cast<size_t>(gn) * k + gk);
          const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const uint32_t w2 = (words[i / 2] >> ((i % 2) * 16)) & 0xffffu;
            const uint16_t lo = float_to_bf16_bits(
                fp8_e4m3_bits_to_float(static_cast<uint8_t>(w2 & 0xffu)) * s);
            const uint16_t hi = float_to_bf16_bits(
                fp8_e4m3_bits_to_float(static_cast<uint8_t>(w2 >> 8)) * s);
            packed[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            uint16_t e[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int gkk = gk + 2 * i + h;
              e[h] = (gn < n && gkk < k)
                         ? float_to_bf16_bits(
                               fp8_e4m3_bits_to_float(
                                   v.payload[static_cast<size_t>(gn) * k + gkk]) * s)
                         : static_cast<uint16_t>(0);
            }
            packed[i] = static_cast<uint32_t>(e[0]) | (static_cast<uint32_t>(e[1]) << 16);
          }
        }
        uint4* dst = reinterpret_cast<uint4*>(&sB[b_row][b_kq]);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
      }
      // Activation tile: rows of this m-tile, zero-filled past the segment
      // and past k (16-byte loads when the rows are 16-byte aligned).
      for (int i = static_cast<int>(threadIdx.x); i < BM * (BK / 8); i += kThreads) {
        const int mm = i / (BK / 8), kq = (i % (BK / 8)) * 8;
        const int gk = k0 + kq;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (mm < m_rows) {
          const int srow = seg.row0 + m0 + mm;
          const uint16_t* row =
              act + static_cast<size_t>(act_rows != nullptr ? act_rows[srow] : srow) *
                        act_stride;
          if (act_vec != 0 && gk + 8 <= k) {
            val = *reinterpret_cast<const uint4*>(row + gk);
          } else {
            uint16_t e[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) e[h] = gk + h < k ? row[gk + h] : 0;
            val = make_uint4(e[0] | (e[1] << 16), e[2] | (e[3] << 16),
                             e[4] | (e[5] << 16), e[6] | (e[7] << 16));
          }
        }
        *reinterpret_cast<uint4*>(&sA[mm][kq]) = val;
      }
      __syncthreads();

#pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        const int ar = warp * 16 + r;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[ar][kk + cc + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[ar + 8][kk + cc + 8]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int bn = j * 8 + r;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc]);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[bn][kk + cc + 8]);
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
      __syncthreads();  // tile reads done before the next stage overwrites
    }

    // Epilogue: per warp a [16 x 64] slice; the thread holds rows r, r+8
    // and columns cc, cc+1 of each n8 fragment.
    const int row_lo = warp * 16 + r;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int gn = n0 + j * 8 + cc;
      auto st = [&](int row_off, int col_off, float val) {
        const int mm = row_lo + row_off;
        if (mm < m_rows && gn + col_off < n)
          fp8_gemv::store_dot(
              out + static_cast<size_t>(seg.row0 + m0 + mm) * out_stride + gn + col_off,
              val);
      };
      st(0, 0, acc[j][0]);
      st(0, 1, acc[j][1]);
      st(8, 0, acc[j][2]);
      st(8, 1, acc[j][3]);
    }
  }
}

template <typename OutT>
void launch_moe_grouped_mma(const uint16_t* act, size_t act_stride,
                            const int32_t* act_rows,
                            const MoeSegment* segs, int n_segs, int max_rows,
                            int rows_per_block, const MoeExpertView* views,
                            int which, OutT* out, size_t out_stride, int n,
                            int k, cudaStream_t stream) {
  using namespace mma_tile;
  if (n_segs <= 0 || n <= 0) return;
  if (!act || !segs || !views || !out)
    throw std::invalid_argument("moe grouped mma: null pointer");
  if (k <= 0 || (k % 16) != 0)
    throw std::invalid_argument("moe grouped mma: k must be a positive multiple of 16");
  if (max_rows <= 0)
    throw std::invalid_argument("moe grouped mma: max_rows must be positive");
  if (rows_per_block > 0 && (rows_per_block % BM) != 0)
    throw std::invalid_argument("moe grouped mma: rows_per_block must be a multiple of 128");
  const unsigned z_ext = rows_per_block > 0
                             ? static_cast<unsigned>((max_rows + rows_per_block - 1) /
                                                     rows_per_block)
                             : 1u;
  if (rows_per_block <= 0) rows_per_block = INT_MAX;
  // 16-byte activation loads need 16-byte rows: the base and the stride.
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + BN - 1) / BN, static_cast<unsigned>(n_segs), z_ext);
  moe_grouped_mma_kernel<OutT><<<grid, kThreads, 0, stream>>>(
      act, act_stride, act_vec, act_rows, segs, views, which, out, out_stride, n,
      k, rows_per_block, MoeSegment{}, MoeExpertView{});
  DGPP_CUDA_OK(cudaGetLastError());
}

inline size_t out_stride_of(int n) { return static_cast<size_t>(n); }

// The dense form: out[m, n] = act[m, k] x W[n, k]^T on the same kernel, one
// m-tile per z block (bitwise the tile kernel, like the grouped form).
template <typename OutT>
void launch_dense_mma(const uint16_t* act, size_t act_stride,
                      const uint8_t* payload, const float* scales, OutT* out,
                      int m, int n, int k, cudaStream_t stream) {
  using namespace mma_tile;
  if (m <= 0 || n <= 0) return;
  if (!act || !payload || !scales || !out)
    throw std::invalid_argument("dense mma: null pointer");
  if (k <= 0 || (k % 16) != 0)
    throw std::invalid_argument("dense mma: k must be a positive multiple of 16");
  const int act_vec =
      (reinterpret_cast<uintptr_t>(act) % 16 == 0 && (act_stride % 8) == 0) ? 1 : 0;
  const dim3 grid((n + BN - 1) / BN, 1u, static_cast<unsigned>((m + BM - 1) / BM));
  moe_grouped_mma_kernel<OutT><<<grid, kThreads, 0, stream>>>(
      act, act_stride, act_vec, nullptr, nullptr, nullptr, 0, out, out_stride_of(n),
      n, k, BM, MoeSegment{0, m, 0}, MoeExpertView{payload, scales});
  DGPP_CUDA_OK(cudaGetLastError());
}

constexpr int kAccumMaxTopK = 16;
constexpr int kSegmentMaxExperts = 1024;

// Device segmentation in three launches (2026-09-05; the single-block form
// before it had one thread per expert walking every id: 0.84 ms per layer
// at 2048 tokens). (1) counts: one block per expert, a strided count and a
// block reduction, into segs[e].rows; (2) scan: one block, the exclusive
// prefix into segs[e].row0, the shared segment and its identity rows;
// (3) place: one block per expert walks the ids in blockDim-sized chunks
// with a block exclusive scan over the match flags, so its expert's
// (token, slot) pairs land in (token, slot) order — exactly the host
// path's stable placement, exactly the single-block kernel's output.
constexpr int kSegmentThreads = 256;

__device__ __forceinline__ int block_exclusive_scan_int(int v, int* s_warp,
                                                        int* total) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int x = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = lane < (kSegmentThreads / 32) ? s_warp[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, w, off);
      if (lane >= off) w += y;
    }
    if (lane < (kSegmentThreads / 32)) s_warp[lane] = w;  // inclusive per warp
  }
  __syncthreads();
  const int warp_base = warp == 0 ? 0 : s_warp[warp - 1];
  *total = s_warp[kSegmentThreads / 32 - 1];
  const int excl = x - v + warp_base;
  __syncthreads();  // s_warp reusable by the next call
  return excl;
}

__global__ void moe_segment_count_kernel(const int32_t* __restrict__ ids, int tk,
                                         MoeSegment* __restrict__ segs) {
  __shared__ int s_warp[kSegmentThreads / 32];
  const int e = blockIdx.x;
  int c = 0;
  for (int i = threadIdx.x; i < tk; i += kSegmentThreads) c += ids[i] == e;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
  if (threadIdx.x % 32 == 0) s_warp[threadIdx.x / 32] = c;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int w = 0; w < kSegmentThreads / 32; ++w) total += s_warp[w];
    segs[e] = MoeSegment{0, total, e};
  }
}

__global__ void moe_segment_scan_kernel(int tokens, int K, int E,
                                        int32_t* __restrict__ rows,
                                        MoeSegment* __restrict__ segs) {
  const int tk = tokens * K;
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      segs[e].row0 = acc;
      acc += segs[e].rows;
    }
    segs[E] = MoeSegment{tk, tokens, E};
  }
  // The shared expert: every token once more, after the routed rows.
  for (int t = threadIdx.x; t < tokens; t += blockDim.x) rows[tk + t] = t;
}

__global__ void moe_segment_place_kernel(const int32_t* __restrict__ ids, int tk,
                                         int K, int32_t* __restrict__ rows,
                                         int32_t* __restrict__ slot_row,
                                         const MoeSegment* __restrict__ segs) {
  __shared__ int s_warp[kSegmentThreads / 32];
  const int e = blockIdx.x;
  const MoeSegment seg = segs[e];
  if (seg.rows == 0) return;
  int running = seg.row0;
  for (int base = 0; base < tk; base += kSegmentThreads) {
    const int i = base + threadIdx.x;
    const int flag = (i < tk && ids[i] == e) ? 1 : 0;
    int total = 0;
    const int excl = block_exclusive_scan_int(flag, s_warp, &total);
    if (flag) {
      const int pos = running + excl;
      rows[pos] = i / K;
      slot_row[i] = pos;
    }
    running += total;
  }
}

__global__ void moe_accum_ordered_kernel(uint16_t* __restrict__ out,
                                         const float* __restrict__ down,
                                         size_t down_stride,
                                         const int32_t* __restrict__ slot_row,
                                         const int32_t* __restrict__ slot_ids,
                                         const float* __restrict__ slot_w,
                                         int shared_row0, int tokens, int K,
                                         int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= static_cast<int64_t>(tokens) * hidden) return;
  const int t = static_cast<int>(i / hidden);
  const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
  // Ascending expert id: the chain's order (an insertion sort of K <= 16).
  int order[kAccumMaxTopK];
  for (int j = 0; j < K; ++j) order[j] = j;
  for (int j = 1; j < K; ++j) {
    const int cur = order[j];
    const int key = slot_ids[t * K + cur];
    int p = j - 1;
    while (p >= 0 && slot_ids[t * K + order[p]] > key) {
      order[p + 1] = order[p];
      --p;
    }
    order[p + 1] = cur;
  }
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    const int s = t * K + order[j];
    acc = __fmaf_rn(slot_w[s],
                    down[static_cast<size_t>(slot_row[s]) * down_stride + h], acc);
  }
  acc = __fmaf_rn(1.0f,
                  down[static_cast<size_t>(shared_row0 + t) * down_stride + h], acc);
  out[i] = float_to_bf16_bits(acc);
}

}  // namespace

void launch_moe_grouped_mma_bf16(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, uint16_t* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream, const int32_t* act_rows) {
  launch_moe_grouped_mma<uint16_t>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                   rows_per_block, views, which, out, out_stride,
                                   n, k, stream);
}

void launch_moe_grouped_mma_f32(const uint16_t* act, size_t act_stride,
                                const MoeSegment* segs, int n_segs, int max_rows,
                                int rows_per_block, const MoeExpertView* views,
                                int which, float* out, size_t out_stride, int n,
                                int k, cudaStream_t stream, const int32_t* act_rows) {
  launch_moe_grouped_mma<float>(act, act_stride, act_rows, segs, n_segs, max_rows,
                                rows_per_block, views, which, out, out_stride, n,
                                k, stream);
}

void launch_dense_mma_bf16(const uint16_t* act, size_t act_stride,
                           const uint8_t* payload, const float* scales,
                           uint16_t* out, int m, int n, int k, cudaStream_t stream) {
  launch_dense_mma<uint16_t>(act, act_stride, payload, scales, out, m, n, k, stream);
}

void launch_dense_mma_f32(const uint16_t* act, size_t act_stride,
                          const uint8_t* payload, const float* scales, float* out,
                          int m, int n, int k, cudaStream_t stream) {
  launch_dense_mma<float>(act, act_stride, payload, scales, out, m, n, k, stream);
}

void launch_moe_grouped_gemv_bf16(const uint16_t* act, size_t act_stride,
                                  const MoeSegment* segs, int n_segs,
                                  int max_rows, int rows_per_block,
                                  const MoeExpertView* views, int which,
                                  uint16_t* out, size_t out_stride, int n, int k,
                                  cudaStream_t stream) {
  launch_moe_grouped_gemv<uint16_t>(act, act_stride, segs, n_segs, max_rows,
                                    rows_per_block, views, which, out, out_stride,
                                    n, k, stream);
}

void launch_moe_grouped_gemv_f32(const uint16_t* act, size_t act_stride,
                                 const MoeSegment* segs, int n_segs, int max_rows,
                                 int rows_per_block, const MoeExpertView* views,
                                 int which, float* out, size_t out_stride, int n,
                                 int k, cudaStream_t stream) {
  launch_moe_grouped_gemv<float>(act, act_stride, segs, n_segs, max_rows,
                                 rows_per_block, views, which, out, out_stride, n,
                                 k, stream);
}

void launch_moe_segment(const int32_t* ids, int tokens, int top_k,
                        int n_experts, int32_t* rows, int32_t* slot_row,
                        MoeSegment* segs, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!ids || !rows || !slot_row || !segs)
    throw std::invalid_argument("moe segment: null pointer");
  if (top_k < 1 || top_k > kAccumMaxTopK || n_experts < 1 ||
      n_experts > kSegmentMaxExperts)
    throw std::invalid_argument("moe segment: top_k or n_experts out of range");
  const int tk = tokens * top_k;
  moe_segment_count_kernel<<<n_experts, kSegmentThreads, 0, stream>>>(ids, tk, segs);
  DGPP_CUDA_OK(cudaGetLastError());
  moe_segment_scan_kernel<<<1, 256, 0, stream>>>(tokens, top_k, n_experts, rows, segs);
  DGPP_CUDA_OK(cudaGetLastError());
  moe_segment_place_kernel<<<n_experts, kSegmentThreads, 0, stream>>>(
      ids, tk, top_k, rows, slot_row, segs);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_accum_ordered(uint16_t* out, const float* down,
                              size_t down_stride, const int32_t* slot_row,
                              const int32_t* slot_ids, const float* slot_w,
                              int shared_row0, int tokens, int top_k,
                              int hidden, cudaStream_t stream) {
  if (tokens <= 0) return;
  if (!out || !down || !slot_row || !slot_ids || !slot_w)
    throw std::invalid_argument("moe accum ordered: null pointer");
  if (top_k < 1 || top_k > kAccumMaxTopK)
    throw std::invalid_argument("moe accum ordered: top_k outside [1, 16]");
  const int64_t n = static_cast<int64_t>(tokens) * hidden;
  const int64_t blocks = (n + kElemThreads - 1) / kElemThreads;
  moe_accum_ordered_kernel<<<static_cast<int>(blocks), kElemThreads, 0, stream>>>(
      out, down, down_stride, slot_row, slot_ids, slot_w, shared_row0, tokens,
      top_k, hidden);
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

void launch_moe_slot_order(const int32_t* ids, int32_t* order, int slots,
                           int top_k, int n_experts, cudaStream_t stream) {
  if (slots <= 0) return;
  if (!ids || !order) throw std::invalid_argument("moe_slot_order: null");
  if (slots > 1024)
    throw std::invalid_argument("moe_slot_order: more slots than one block");
  moe_slot_order_kernel<<<1, static_cast<unsigned>(slots),
                          static_cast<size_t>(slots) * sizeof(int32_t),
                          stream>>>(ids, order, slots, top_k, n_experts);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_down(const uint16_t* act, size_t act_stride,
                          const int32_t* ids, const int32_t* order,
                          const MoeExpertView* views,
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
      act, act_stride, ids, order, views, n_routed, k_routed, n_shared,
      k_shared, sh_payload, sh_scales, out, out_stride, slots, top_k);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_moe_slot_gate_up_swiglu(
    const uint16_t* x, size_t x_stride, const int32_t* ids,
    const int32_t* order, const MoeExpertView* views, int n_routed,
    int k_routed, int n_shared, int k_shared, const uint8_t* sh_gate_payload,
    const float* sh_gate_scales, const uint8_t* sh_up_payload,
    const float* sh_up_scales, uint16_t* act, int act_stride, int slots,
    int top_k, float limit, cudaStream_t stream) {
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
      x, x_stride, ids, order, views, n_routed, k_routed, n_shared, k_shared,
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
