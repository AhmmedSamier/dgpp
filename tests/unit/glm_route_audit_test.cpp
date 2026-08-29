// Route-flip audit tests (host-only): the certifier must accept a measured
// near tie and reject every doctored divergence — a far displacement, a
// zero-noise swap, and a selection inconsistent with the engine's own
// scores. These pin the certifier itself (DESIGN 12: the audit is only as
// trustworthy as its rejection paths).
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/glm_route_audit.hpp"

namespace {

using dgpp::glm_route::RouteFlipAudit;
using dgpp::glm_route::audit_route_flips;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// One token's audit inputs. Biased rows are float vectors so the tests can
// place score perturbations exactly; ids follow both sides' contract
// (ascending per token).
struct Row {
  std::vector<int32_t> eng_ids;
  std::vector<int32_t> ref_ids;
  std::vector<float> eng_biased;
  std::vector<float> ref_biased;
  int top_k = 0;
};

RouteFlipAudit run_audit(const Row& r, int n_experts, int64_t layer = 7) {
  RouteFlipAudit a;
  audit_route_flips(r.eng_ids.data(), r.ref_ids.data(), r.eng_biased.data(),
                    r.ref_biased.data(), 1, r.top_k, n_experts, a, layer);
  return a;
}

std::string expect_throw(const Row& r, int n_experts) {
  try {
    (void)run_audit(r, n_experts);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  throw std::runtime_error("audit accepted a doctored divergence");
}

// E=8, K=2. Reference boundary between experts 1 and 2 is a genuine near
// tie (gap 2e-3); the engine's uniform ~5e-4 input noise moves every
// expert's score the same scale and flips the pair.
Row near_tie_row() {
  Row r;
  r.top_k = 2;
  r.ref_biased = {1.0f, 0.520f, 0.518f, 0.40f, 0.30f, 0.20f, 0.10f, 0.05f};
  r.eng_biased = {1.0f, 0.5175f, 0.5185f, 0.4005f, 0.3005f, 0.2005f,
                  0.1005f, 0.0505f};
  r.ref_ids = {0, 1};
  r.eng_ids = {0, 2};
  return r;
}

}  // namespace

DGPP_TEST(audit_route_flips_withUniformNoiseNearBoundaryTie_certifiesFlip) {
  // GIVEN a boundary pair with a 2e-3 ref-side gap and uniform ~5e-4
  // cross-side score noise on every expert:
  const Row r = near_tie_row();

  // WHEN the audit runs,
  const RouteFlipAudit a = run_audit(r, 8);

  // THEN the flip is certified as a measured near tie with its evidence.
  require(a.tokens_flipped == 1, "one flipped token");
  require(a.swaps_certified == 1, "one certified swap");
  require(std::abs(a.max_boundary_gap - 0.002) < 1e-6, "ref-side gap 2e-3");
  // noise over the 6 uninvolved experts ~ (5*5e-4 + 0)/6; the worst
  // bounded quantity is the flipped-out expert's own drift (2.5e-3) ->
  // multiple ~6, far under the 32x doctrine bound.
  require(a.max_noise_multiple > 1.0 && a.max_noise_multiple < 32.0,
          "noise multiple within the doctrine bound");
}

DGPP_TEST(audit_route_flips_withFarRankedDisplacement_rejectsDoctoredFlip) {
  // GIVEN a doctored selection: expert 15 (E=16) is ranked far below the
  // boundary on the reference side but pushed to the top on the engine
  // side, while the uninvolved experts carry only honest 1e-4 noise:
  Row r;
  r.top_k = 2;
  r.ref_biased.resize(16);
  r.eng_biased.resize(16);
  for (int e = 0; e < 16; ++e) {
    r.ref_biased[e] = 0.1f * (16 - e);  // strict descent, ref top-2 = {0, 1}
    r.eng_biased[e] = r.ref_biased[e] + 1e-4f;
  }
  r.ref_ids = {0, 1};
  r.eng_ids = {0, 15};
  r.eng_biased[15] = 1.55f;  // make the engine's own top-k agree with {0, 15}
  r.eng_biased[1] = 0.5f;

  // WHEN the audit runs, THEN it rejects with the full diagnosis (the gap
  // and the swapped experts' drifts are thousands of noise multiples).
  const std::string msg = expect_throw(r, 16);
  require(msg.find("NOT a boundary near tie") != std::string::npos,
          "rejection carries the near-tie diagnosis");
  require(msg.find("ref rank 15") != std::string::npos,
          "diagnosis names the far rank");
}

DGPP_TEST(audit_route_flips_withZeroNoiseSwap_rejectsAsSpecBug) {
  // GIVEN identical biased scores on both sides, the engine selecting the
  // true top-k of those scores, and a DOCTORED reference row claiming a
  // mid-rank expert instead (the engine is right, the reference is wrong):
  Row r = near_tie_row();
  r.eng_biased = r.ref_biased;
  r.eng_ids = {0, 1};  // top-2 of the (now shared) scores
  r.ref_ids = {0, 3};  // doctored: expert 3 is far below the boundary

  // WHEN the audit runs, THEN a zero-noise row with a swap is a hard spec
  // bug, never a tolerated "near tie" — the uninvolved experts' measured
  // disagreement is exactly zero.
  const std::string msg = expect_throw(r, 8);
  require(msg.find("NOT a boundary near tie") != std::string::npos,
          "zero-noise swap rejected");
}

DGPP_TEST(audit_route_flips_withSelectionInconsistentWithOwnScores_rejects) {
  // GIVEN engine ids that are NOT the top-k of the engine's own biased
  // scores (a router bug, not a boundary event):
  Row r = near_tie_row();
  r.eng_ids = {0, 3};  // engine scores still rank expert 2 over 3

  // WHEN the audit runs, THEN the selection itself is rejected before any
  // near-tie reasoning (the router must be spec-exact for its own inputs).
  const std::string msg = expect_throw(r, 8);
  require(msg.find("NOT the spec top-k of the engine's own biased scores") !=
              std::string::npos,
          "router-bug rejection carries its own diagnosis");
}

DGPP_TEST(audit_route_flips_withDuplicateIdInRow_rejectsAsTopKBug) {
  // GIVEN a row whose engine ids contain a duplicate (a malformed top-k
  // that a flip classification would silently absorb):
  Row r = near_tie_row();
  r.eng_ids = {0, 0};

  // WHEN the audit runs, THEN the duplicate is a loud top-k bug.
  const std::string msg = expect_throw(r, 8);
  require(msg.find("duplicate expert id") != std::string::npos,
          "duplicate-id rejection carries its own diagnosis");
}
