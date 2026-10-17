#pragma once
// The wide GEMV chunk (2026-10-14): the bandwidth GEMV cores (bf16_gemv,
// the fp8 scale GEMV) stage up to kWideRows activation rows per launch
// instead of gemv::kMaxRows (4), so a decode batch of five to eight rows
// reads the weights ONCE and every row keeps the scalar chain it has alone
// — a batched row bitwise the row alone, which the cuBLASLt and streaming
// tensor-core forms above four rows do not give (their reduction order
// follows m). Eight rows of a 4096-wide activation are 64 KB of dynamic
// shared memory: past the 48 KB default, so the kernels opt in once at
// process start (enable_wide_rows, from the GEMM wrapper's constructor —
// never inside a capture) and the chunk sizers ask smem_fits_wide.
#include <cstddef>

namespace dgpp::gemv {

constexpr int kWideRows = 8;
constexpr std::size_t kWideSmemBytes = 96 * 1024;

// The opt-in: every wide instantiation's dynamic-smem ceiling raised to
// kWideSmemBytes. Idempotent; false (and narrow chunks stay) when the
// device refuses. Call before any graph capture.
bool enable_wide_rows();
bool wide_rows_enabled();
// rows x k bf16 activations fit one launch's shared memory: the default
// ceiling for one to four rows (those instantiations are not opted in, so
// every narrower chunking is exactly what it was), the wide one for five
// to eight rows once enabled.
bool smem_fits_wide(int rows, int k);

}  // namespace dgpp::gemv

namespace dgpp {
// Per-core opt-ins (the TU that owns each kernel family).
bool bf16_gemv_opt_in_wide();
bool scale_gemv_opt_in_wide();
}  // namespace dgpp
