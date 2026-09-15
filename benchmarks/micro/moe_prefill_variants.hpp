#pragma once
// Experimental kernels linked only into moe_prefill_bench. They are not
// serving alternatives: the measured variants did not clear the service gate.
#include "kernels/glm_moe_launch.hpp"

namespace dgpp::bench {
enum class Variant { kSmallTiles, kCompact, kPersistent };
void launch_variant_bf16(Variant variant, const uint16_t* act, size_t act_stride, const int32_t* act_rows,
    const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
    uint16_t* out, size_t out_stride, int n, int k, cudaStream_t stream);
void launch_variant_f32(Variant variant, const uint16_t* act, size_t act_stride, const int32_t* act_rows,
    const MoeSegment* segs, int n_segs, const MoeExpertView* views, int which,
    float* out, size_t out_stride, int n, int k, cudaStream_t stream);
}  // namespace dgpp::bench
