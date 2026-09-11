#pragma once
// IGemm interface (DESIGN §§4 and 11): weights [N,K] fp8/bf16 against
// activations [M,K].
// Implementations must be deterministic run-to-run at fixed shapes so CUDA
// graph capture replays bitwise.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"

namespace dgpp {

enum class GemmOut : int { BF16, F32 };

class IGemm {
 public:
  virtual ~IGemm() = default;

  // Computes D[M,N] = Act[M,K] x W[N,K]^T, enqueued on `stream`.
  // act: row-major [M,K]; weight: row-major [N,K] contiguous.
  // act_row_stride: element stride between consecutive rows of `act`.
  // Pass k for a contiguous [M,K] tensor; wider strides read K-column slices
  // of a fused projection buffer (e.g. KDA f_a/g_a views).
  virtual void matmul(const void* act, const void* weight, void* out, int m,
                      int n, int k, DType io_dtype, GemmOut out_dtype,
                      size_t act_row_stride, void* workspace, size_t ws_bytes,
                      cudaStream_t stream) = 0;

  virtual size_t query_workspace_bytes(int m, int n, int k,
                                       DType io_dtype) = 0;

  // Prebuilds (and validates feasibility of) the plan for this shape without
  // executing it. Returns false when no heuristic exists — call at init, never
  // during capture.
  virtual bool ensure_plan(int m, int n, int k, DType io_dtype,
                           GemmOut out_dtype, size_t act_row_stride) = 0;
};

// The decode shapes the interface lowers to the row-independent GEMV core: m up
// to the model's decode rows (set_decode_rows; kGemmDecodeRowsDefault, the
// old fixed batch, until a model says otherwise — a wider batch's rows
// keep the scalar reduction order whatever batch they ride in: the first
// 9-row batch fell to an Lt algorithm with its own reduction order,
// 2026-09-10), in chunks of at most gemv::kMaxRows. kGemmDecodeLoweringRows
// bounds it (engine/decode_outputs.hpp's kDecodeRowsMax). The interface cannot
// tell a decode call from a short prefill chunk, so the bound is the
// model's decode shape and nothing wider: a prefill of more rows keeps
// its Lt algorithm (and its transcripts).
constexpr int kGemmDecodeRowsDefault = 8;
constexpr int kGemmDecodeLoweringRows = 32;

// cuBLASLt-backed implementation with per-shape heuristic caching. Decode-
// shaped bf16 calls (m <= the decode rows, k % 8 == 0, 16B-aligned weight)
// bypass Lt for one or more launches of the in-house row-independent
// bandwidth GEMV (bf16_gemv.hpp) — Lt's m=1 kernel runs at ~55% of the
// part's bandwidth.
class CublasLtGemm : public IGemm {
 public:
  CublasLtGemm();   // creates handle, unit-scale device constants
  ~CublasLtGemm() override;

  void matmul(const void* act, const void* weight, void* out, int m, int n,
              int k, DType io_dtype, GemmOut out_dtype,
              size_t act_row_stride, void* workspace, size_t ws_bytes,
              cudaStream_t stream) override;

  size_t query_workspace_bytes(int m, int n, int k, DType io_dtype) override;

  bool ensure_plan(int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                   size_t act_row_stride) override;

  // The widest decode shape this model runs (its fixed batch's rows): bf16
  // calls up to it take the GEMV core. [1, kGemmDecodeLoweringRows].
  void set_decode_rows(int rows);
  int decode_rows() const;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace dgpp
