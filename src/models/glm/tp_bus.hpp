#pragma once
// The bus-side engine helpers moved to engine/tp_bus.hpp;
// GLM keeps the names its apps and gates use.
#include "engine/tp_bus.hpp"
#include "models/glm/forward.hpp"

namespace dgpp {
using GlmBusBoundaryReducer = BusBoundaryReducer;
using GlmGraphRecordReducer = GraphRecordReducer;
}  // namespace dgpp
