#pragma once
// Host reference of the Qwen norms (Q3, 2026-09-09): the (1 + w) RMSNorm
// with one rounding (Qwen3_5RMSNorm) and its grouped form over the hyper
// state (group_size = hidden). The GDN gated norm's oracle is in
// gdn_reference.hpp. fp32 interior; the sum of squares is a sequential
// fma chain, so a device tree reduction may land an ulp away in rstd.
#include <cstdint>

namespace dgpp::qwen_ref {

void rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows, int dim,
             float eps);
void group_rmsnorm(const uint16_t* x, const uint16_t* w, uint16_t* y, int64_t rows,
                   int groups, int group_dim, float eps);

}  // namespace dgpp::qwen_ref
