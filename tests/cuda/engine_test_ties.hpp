// Near-tie rules for the engine gates (2026-09-14): a transcript against a
// reference that differs from it only by reduction order — the batched
// graph engine against the eager one since the dense sites' lowering
// follows the rows of a launch (kernels/gemm.hpp dense_gemv_rows: the GEMV
// chunks to four rows, cuBLASLt's algorithm or the streaming tensor-core
// form above), or world 2 against world 1 (the folds reassociate). The
// random-weight fixtures amplify a last-bit difference ~1.5x per layer, so
// a near tie flips and every later token follows: the rule holds the first
// decisions exactly unless the flipped decision was a demonstrated near tie
// of the reference's own logits, and reports the rest.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/graph_engine.hpp"

namespace engine_ties {

// The world-1 pick that also records every decision's top-2 logit margin.
inline dgpp::DecodePick margin_pick(int64_t vocab, std::vector<float>* margins) {
  return [vocab, margins](const dgpp::DecodeOutputs& out) -> int32_t {
    const int n = static_cast<int>(out.lm_vocab_count);
    int best = -1;
    float top = -INFINITY, second = -INFINITY;
    for (int v = 0; v < n; ++v) {
      const float x = out.logits[static_cast<size_t>(v)];
      if (x > top) { second = top; top = x; best = v; }
      else if (x > second) second = x;
    }
    if (best < 0 || best >= vocab) throw std::runtime_error("w1 pick out of range");
    margins->push_back(top - second);
    return best;
  };
}

inline size_t agreeing_prefix_of(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}

// `got` against `want`: the first `positions` decisions agree, or the first
// difference among them sits on a near tie (margin under `tie`) of the
// reference's decision at that position (`margins`, recorded on the
// world-1 reference of the same prompt: a proxy within the fold noise for
// a world-2 reference). Past those positions the difference is reported.
inline void require_agrees_or_tie(const std::vector<int32_t>& got, const std::vector<int32_t>& want,
                                  const std::vector<float>& margins, float tie, size_t positions, const char* what) {
  const size_t p = agreeing_prefix_of(got, want);
  if (p == want.size() && p == got.size()) return;
  if (p >= want.size() || p >= got.size()) return;  // a prefix of the other: nothing to judge
  if (p >= margins.size()) throw std::runtime_error(std::string(what) + ": no margin recorded at the first difference");
  DGPP_LOG_INFO("{}: differs from the reference at position {} of {} (reference margin {:.3e}, tie bound {:.1e} over the first {})",
                what, p, want.size(), margins[p], tie, positions);
  if (p < positions && !(margins[p] < tie))
    throw std::runtime_error(std::string(what) + ": an early token flips that is not a near tie");
}

}  // namespace engine_ties
