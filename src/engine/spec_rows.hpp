#pragma once

namespace dgpp {

// Maximum rows in a speculative verification block, including the pending
// target token. Drafter and verifier kernels share this compile-time bound.
inline constexpr int kSpecRows = 16;

}  // namespace dgpp
