#pragma once
// IGemm seam (DESIGN §§4 and 11): weights [N,K] fp8/bf16 against
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

// cuBLASLt-backed implementation with per-shape heuristic caching.
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

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace dgpp
