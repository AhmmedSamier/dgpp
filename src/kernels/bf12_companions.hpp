#pragma once
// The owner of a model's bf12 companions (bf12_gemv.hpp): packs resident
// bf16 decode matrices after the load, keeps their device memory, registers
// them with the model's GEMM, and sizes them for the memory plan. One per
// model; every family that reads bf16 weights through CublasLtGemm's GEMV
// lowering uses it the same way.
//
// The process-wide switch is the deployment's `engine.bf16_weights`
// ("checkpoint": the bf16 bytes alone, the default; "bf12": the lossless
// 12-bit companions beside them). The serving app sets it before any model
// plans or builds; DGPP_BF12=on|off overrides it (A/B runs — the launcher
// forwards DGPP_* to every rank).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"

namespace dgpp {

class Bf12Companions {
 public:
  static void set_enabled(bool on);
  static bool enabled();
  // The device bytes planned for the companion of a bf16 [n, k] matrix: the
  // packed rows, the row table, and an allowance for escapes and raw rows
  // (measured 1.5e-4 of the weights and 35 rows of a model; 1e-3 and one
  // row in 64 budgeted). 0 for a shape outside the format.
  static size_t planned_bytes(int64_t n, int64_t k);

  Bf12Companions() = default;
  ~Bf12Companions();
  Bf12Companions(const Bf12Companions&) = delete;
  Bf12Companions& operator=(const Bf12Companions&) = delete;

  // Packs the resident device matrix weight[n, k] — read back through
  // `stream`, encoded on the host, uploaded — and registers the companion
  // with `gemm`. False (and nothing registered) when the matrix keeps its
  // bf16 form: a null pointer, a shape outside the format, too many raw
  // rows. Synchronizes `stream`; call after the load, before any capture.
  bool pack(const uint16_t* weight, int64_t n, int64_t k, CublasLtGemm& gemm, cudaStream_t stream);

  size_t matrices() const { return matrices_; }
  size_t kept_bf16() const { return kept_; }
  size_t bf16_bytes() const { return raw_bytes_; }
  size_t packed_bytes() const { return packed_bytes_; }
  // One INFO line (nothing when no matrix was offered).
  void log_summary(int rank, double seconds) const;

 private:
  std::vector<void*> allocs_;
  std::vector<uint16_t> host_;
  size_t matrices_ = 0, kept_ = 0, raw_bytes_ = 0, packed_bytes_ = 0, escapes_ = 0, raw_rows_ = 0;
  int widest_ = 0;
};

}  // namespace dgpp
