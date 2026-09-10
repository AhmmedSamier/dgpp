#pragma once
// The bf16 decode GEMV (M6 Stage 2 round 3): D[m,N] = Act[m,K] x W[N,K]^T
// for m <= gemv::kMaxRows, bf16 weights and activations, fp32 accumulate,
// bf16 or f32 output. The GEMM seam (CublasLtGemm::matmul) dispatches
// decode-shaped bf16 calls here — cuBLASLt's m=1 kernel (gemvx) ran the
// KDA/DSA projections at ~128 GB/s on a 233 GB/s part (the T=1 profile:
// 22 ms of a 190 ms step for 2.87 GB of weights).
//
// Same shape as the fp8 core (gemv_common.cuh): warp per weight row,
// 16-byte chunks 512 bytes apart, a batch in flight before any is consumed,
// no barrier in the k loop, per-lane sequential fp32 chain then a fixed
// xor tree — deterministic, launch-shape independent, every activation
// row's chain independent of the row count. The accumulation order differs
// from cuBLAS's (opaque anyway) — the layer oracles (kda_test, dsa_test,
// glm_forward_test) are the gate.
//
// Contract (the launcher checks; callers fall back to the GEMM otherwise):
// 1 <= m <= kMaxRows, k % 8 == 0, weight 16-byte aligned (rows then are:
// the row stride is k*2 bytes, a multiple of 16), and m*k*2 bytes of
// activations within the default dynamic-smem ceiling (48 KB).
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dgpp {

// True when this shape/alignment is the GEMV's (see the contract above).
bool bf16_gemv_accepts(const void* weight, int m, int k);

// out_f32: false -> bf16 [m, n] row-major; true -> f32 [m, n] row-major.
// act rows are act_row_stride elements apart (a contiguous [m,k] passes k).
void launch_bf16_gemv(const uint16_t* act, size_t act_row_stride,
                      const uint16_t* weight, void* out, bool out_f32, int m,
                      int n, int k, cudaStream_t stream);

// Two GEMVs of the same m, k and output type in ONE launch: blocks
// [0, blocks(n0)) run problem 0, the rest problem 1. Each warp's work is
// exactly what the single launch would do, so both outputs are bitwise the
// two-launch outputs; what is saved is a launch and a graph gap per pair
// (the KDA layer's f_b/g_b, 5 us each at decode, mostly fixed cost).
struct Bf16GemvProblem {
  const uint16_t* act = nullptr;
  size_t act_row_stride = 0;
  const uint16_t* weight = nullptr;
  void* out = nullptr;
  int n = 0;
};
void launch_bf16_gemv_dual(const Bf16GemvProblem& p0, const Bf16GemvProblem& p1,
                           bool out_f32, int m, int k, cudaStream_t stream);

// Up to four GEMVs of the same m, k and output type in ONE launch
// (2026-09-09, the GDN's qkv / z / a / b projections at decode): blocks
// [B_i, B_{i+1}) run problem i, each warp's work exactly the single
// launch's — every output bitwise its own launch. n == 1 is the single
// launch; n == 2 the dual.
constexpr int kBf16GemvMaxProblems = 4;
void launch_bf16_gemv_multi(const Bf16GemvProblem* problems, int n_problems,
                            bool out_f32, int m, int k, cudaStream_t stream);

}  // namespace dgpp
