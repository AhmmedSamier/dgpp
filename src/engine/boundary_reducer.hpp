#pragma once
#include <cstdint>

namespace dgpp {

// M5 tensor-parallel block-boundary seam (DESIGN §5.1): folds a partial
// hidden activation [rows, hidden] bf16 into its replicated value, in
// place. Called after the attention output projection and after the
// FFN/MoE — the two row-parallel sites whose local sums are partial —
// always with the producing kernels already quiesced on the model stream.
// world=1 constructs the model without a reducer and the seam is skipped.
struct BoundaryReducer {
  virtual ~BoundaryReducer() = default;

  // Optional staging seam (the §6.3 evolution): hands the producing GEMM
  // a pinned, device-writable destination for the boundary partial so the
  // collective sends it straight from there (no device→slot staging copy).
  // Returns nullptr when the shape does not fit (rows*hidden above the
  // latency slot — prefill-sized boundaries stay on the device path) or
  // when the transport has no pre-stage support. The returned pointer is
  // consumed by the following reduce() call.
  virtual uint16_t* stage(int /*rows*/, int /*hidden*/) { return nullptr; }

  virtual void reduce(uint16_t* partial, int rows, int hidden) = 0;

  // Measurement seam (2026-09-09, the Qwen plan's Q0): one extra collective
  // of `rows x cols` bf16 over a scratch buffer nobody reads, issued right
  // after a boundary fold — the shape and position of the all-reduce a
  // sliced GR gate would add per site. Default: nothing. Implementations
  // clamp the element count to their latency slot.
  virtual void probe(int /*rows*/, int /*cols*/) {}
};

}  // namespace dgpp
