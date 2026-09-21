# Qwen FP8 vocabulary-head review follow-up — 2026-09-21

This follow-up addresses upstream PR #10 at `8ebd4b6`. Upstream master
`5631fa1` was merged with both status paragraphs retained in `PLAN.md`.

## Deployment behavior

`engine.fp8_head` / `--fp8-head` accepts `gemv` (default) or `mma`.
The default retains the pre-optimization dispatch. MMA requires the Qwen
family and `engine.dense_weights: "fp8"`; it enables the existing streaming
kernel only above `dense_gemv_rows()` and within the model's decode ceiling.
The setting is carried in `WorldSettings`, copied from rank 0 before model
construction, included in the canonical configuration digest, and passed
as an immutable per-model constructor option. It adds no environment switch.
The existing version handshake and warm-record digest check remain in force.

Short prefill chunks in the optimized interval also change accumulation
order. Bitwise eager/graph comparisons now explicitly share the same decode
capacity and head setting, with assertions guarding that setup. The FP8
fixture compares the default head at capacity 16 with MMA at capacity 16,
and uses an MMA capacity-4 model as an unchanged-dispatch control. Hidden
states must agree before comparing logits; boundaries 1/4/5/8/16/17 remain
covered. The two-rank MTP lane now constructs both models at capacity 16.

## Evidence packaging

The six compressed epoch/calibration records were moved to a
[benchmark evidence release](https://github.com/jontaylor/dgpp/releases/tag/fp8-head-benchmark-2026-09-20)
on the contributor fork. The source tree retains the protocol, analyzer,
summary, binary identities, op-streams, sensors and per-file SHA256SUMS.
The downloaded asset passed every retained hash, and rerunning the analyzer
on its extracted epochs exactly reproduced the archived summary.
The standalone `raw/head-bench.cpp` remains an archived one-off reproducer,
not a maintained benchmark target. Existing benchmark results remain attached
to their original tested commits, not relabeled as results for this follow-up.

## Validation

Host validation results are recorded below after the fresh build completes.
New coverage checks config defaults and both modes, invalid values/types with
field-specific errors, journal round trips, missing-field GEMV fallback,
rank-0 head-mode adoption through actual server startup processes, and invalid
CLI settings before GPU initialization. Existing mixed-version rejection is
exercised with an MMA settings record.

The changed CUDA fixture and two-rank lanes require an idle hardware window
before their runtime results can be claimed. Matched real-checkpoint
teacher-forced numerical validation remains pending, including wide
verification and short prefill. The PR stays draft and no shipped recipe
opts into MMA by default.
