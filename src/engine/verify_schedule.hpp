#pragma once
// Confidence-scheduled verification depth (docs/deepseek_v41_flash_plan.md
// §D8a; DeepSeek-V4.1-Flash tech report §2.4.3, the DSpark confidence head).
// Engine-level and model-agnostic: any draft that emits a per-position
// acceptance logit can drive it (Model::kVerifyConfidence in the graph
// engine); DSpark is the first.
//
// The draft always computes the whole block of `block` positions in one pass
// (that pass is what produces the confidence, so it is sunk once we are here).
// The confidence head emits a per-position acceptance LOGIT c_i, so
// sigmoid(c_i) = P(draft i accepts | the prefix before it survived), and
// S_i = prod_{j<i} sigmoid(c_j) is the probability the first i drafts all
// survive. S_i is monotone non-increasing in i.
//
// A verify pass over k of the block's drafts runs 1+k rows (the fed token plus
// k drafts) and commits the greedy-matching prefix. That prefix is EXACTLY the
// one a full-block verify would commit, so k changes throughput only, never the
// output.
//
// Objective. We maximize aggregate decode throughput, sum(committed)/sum(time)
// over the run, NOT the per-step ratio committed/time (maximizing a sum of
// ratios is the wrong surrogate for a ratio of sums, and the two disagree
// exactly when confidence varies across steps -- chat vs counting, our case).
// The Dinkelbach optimum of a ratio of sums is a per-step LINEAR tradeoff at a
// single global multiplier lambda* (the achieved throughput, tokens/ms): each
// step maximizes E[committed|k] - lambda*.time(k). With
//   E[committed|k] = 1 + sum_{i=1..k} S_i,   time(k) = base + row.(1+k),
// the gain of extending k->k+1 is S_{k+1} - lambda*.row, so we verify draft i
// while S_i > lambda*.row and stop at the first that falls below (S monotone).
// The fixed base cost drops out of the depth decision; it only sets lambda*.
//
// lambda is supplied by the caller as the value of decode time -- the engine's
// realized throughput, tracked as an EWMA of committed/time, which converges to
// the self-consistent lambda*. Before it warms up, the reservation rate below
// (a plain single-token step's throughput) is a safe floor: speculation is
// taken only when a row beats just decoding one more token the plain way.
//
// Confidence is replicated across ranks and lambda is a scalar the caller keeps
// identical on every rank, so all ranks derive the same k.
#include <cmath>
#include <vector>

namespace dgpp {

// Throughput of a plain single-token decode step: 1 token over the fixed base
// plus one row. A safe lower bound for lambda (tokens/ms) before an EWMA of
// realized throughput is available.
inline float verify_reservation_lambda(float base_ms, float row_ms) {
  const float t = base_ms + row_ms;
  return t > 0.f ? 1.f / t : 0.f;
}

// conf_logit: [block] the block's per-position confidence logits (replicated
// across ranks). row_ms: cost of one verify row. lambda_tok_per_ms: the value
// of decode time (realized throughput EWMA, floored at the reservation rate).
// Returns k in [0, block]: the number of leading drafts to verify.
inline int scheduled_verify_depth(const float* conf_logit, int block, float row_ms,
                              float lambda_tok_per_ms) {
  if (block <= 0) return 0;
  const double threshold = static_cast<double>(lambda_tok_per_ms) * row_ms;
  double survival = 1.0;  // S_i, the running prefix-survival product
  int k = 0;
  for (int i = 1; i <= block; ++i) {
    const double p = 1.0 / (1.0 + std::exp(-static_cast<double>(conf_logit[i - 1])));
    survival *= p;  // S_i includes sigmoid(c_{i-1})
    if (survival > threshold)
      k = i;  // this row's expected tokens still beat the value of its time
    else
      break;  // S monotone non-increasing: no later row can qualify
  }
  return k;
}

// The batched replay's one depth for `slots` live slots (conf[s]: slot s's
// block of logits). Extending the batch by one draft position costs one
// verify row PER SLOT (slots * row_ms) and yields the sum over the slots
// of their prefix survivals at that position, so the Dinkelbach gain is
// sum_s S_i(s) - slots * lambda * row: verify position i while the MEAN
// survival over the slots exceeds lambda * row (the scalar rule with the
// mean in place of one slot's S_i; each mean is non-increasing in i, so
// the kept positions are a prefix). The deepest slot's own depth (the
// earlier rule) verifies rows for every slot that a confident one alone
// justifies; the mean rule stops where the batch as a whole stops paying.
// Every slot commits its greedy prefix at any depth, so this is throughput
// only, never output. Deterministic across ranks like the scalar rule.
inline int scheduled_verify_depth_batch(const float* const* conf_logit, int slots, int block,
                                        float row_ms, float lambda_tok_per_ms) {
  if (block <= 0 || slots <= 0) return 0;
  const double threshold = static_cast<double>(lambda_tok_per_ms) * row_ms;
  std::vector<double> survival(static_cast<size_t>(slots), 1.0);
  int k = 0;
  for (int i = 1; i <= block; ++i) {
    double sum = 0.0;
    for (int s = 0; s < slots; ++s) {
      const float c = conf_logit[s][i - 1];
      survival[static_cast<size_t>(s)] *= 1.0 / (1.0 + std::exp(-static_cast<double>(c)));
      sum += survival[static_cast<size_t>(s)];
    }
    if (sum / slots > threshold)
      k = i;
    else
      break;
  }
  return k;
}

}  // namespace dgpp
