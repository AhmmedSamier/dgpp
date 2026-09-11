#pragma once
// GLM's eager speculators: engine/speculative.hpp's
// templates; CTAD binds them from the model argument, and the explicit
// specializations below name them where a type is required.
#include "engine/speculative.hpp"
#include "models/glm/forward.hpp"

namespace dgpp {
using GlmGreedySpeculator = GreedySpeculator<GlmDiagnosticModel>;
using GlmSampledSpeculator = SampledSpeculator<GlmDiagnosticModel>;
}  // namespace dgpp
