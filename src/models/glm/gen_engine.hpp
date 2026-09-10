#pragma once
// GLM's eager engine (Q1, 2026-09-09): engine/eager_engine.hpp's adapter
// bound to GlmDiagnosticModel under the name the apps and gates use.
#include "engine/eager_engine.hpp"
#include "models/glm/forward.hpp"

namespace dgpp {
using GenEngineAdapter = EagerEngineAdapter<GlmDiagnosticModel>;
}  // namespace dgpp
