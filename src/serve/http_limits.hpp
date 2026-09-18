#pragma once

#include <cstdint>

namespace dgpp::serve {

// A byte limit on the serialized request, independent of the model's token
// budget. Leave room for large document prefills, JSON escaping and tool
// history. This is a ceiling, not a buffer allocated at startup.
inline constexpr int64_t kDefaultHttpMaxBodyBytes = 256ll * 1024 * 1024;

}  // namespace dgpp::serve
