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
and uses a minimum-capacity MMA model as a dispatch-boundary control. Hidden
states must agree before comparing logits; lengths 1/4/5/8/9/16/17 are covered.
The first native run caught a test assumption: requesting four rows still
allocates the session-core floor of eight. The control now checks its reported
effective ceiling, matches wide MMA at 5–8 rows, and default GEMV above eight.
The explicit default-off capacity-16 model establishes a true GEMV comparison
at 5/8 rows, where the original PR's requested-capacity-4 control also ran MMA. The two-rank MTP lane now constructs both models at capacity 16.

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

A fresh x86 CUDA 13 build with warnings treated as errors compiled the server,
model code and changed CUDA test binaries. All 21 host CTest entries passed
across the initial run and targeted recheck. The initial container run lacked
a passwd entry for the numeric UID and `libcuda.so.1`, and the newly merged
YaRN fixture target had not yet been built. After building that target and
providing a container-local passwd entry and the toolkit driver stub (host-only
startup tests, no GPU execution), all five affected entries passed. No source
changes were needed for those environment failures. The seven startup cases
include actual two-/four-process settings handshakes before GPU initialization.
New coverage checks config defaults and both modes, invalid values/types with
field-specific errors, journal round trips, missing-field GEMV fallback,
rank-0 head-mode adoption through actual server startup processes, and invalid
CLI settings before GPU initialization. Existing mixed-version rejection is
exercised with an MMA settings record.

Native commands, run serially with production stopped for GPU/RDMA execution:

```bash
cmake --preset ci -G Ninja
cmake --build --preset ci -j 4
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci --output-on-failure -j 1 --timeout 180
```

A fresh native GB10 CUDA 13 CI build completed all targets with warnings
as errors. The [full serial suite](2026-09-21-qwen-fp8-head-review/full-ctest.log)
passed 114 tests with ten checkpoint-dependent skips and zero failures in
665.85 seconds. At implementation head `448817b`, all six
[focused checks](2026-09-21-qwen-fp8-head-review/focused-ctest.log) passed
on idle hardware, including the two-rank FP8 MTP lanes. The corrected
short-prefill fixture's mean NLL deltas (MMA minus GEMV) were:

| Rows | Mean NLL delta (nat) |
| --- | ---: |
| 1 | 0 |
| 4 | 0 |
| 5 | -9.02057e-8 |
| 8 | -6.37356e-8 |
| 9 | -7.2795e-8 |
| 16 | 2.52229e-8 |
| 17 | 0 |

The wide teacher-forced fixture covered 368 rows: worst relative L2
0.0023704, no near-ties, mean NLL delta 5.4114e-5 nat. The row-independent
control remained bitwise equal. These are synthetic-fixture results.

The real Qwen3.8-Flash-Next-NVFP4 model passed all nine
[API checks](2026-09-21-qwen-fp8-head-review/candidate-api.log) on two Sparks
with the CI binary, C4/MTP3, FP8 dense weights, MMA opted in, BF16 KV capacity
65536, mmap n-grams and an 8 GiB prefix cache. The first API attempt used the
benchmark's cache-disabled setting and failed only the cache-reuse assertion;
the API-only rerun enabled caching and passed. No source changes were needed.
Both ranks logged `fp8head=mma` and configuration digest `12d14d2273f1a557`,
with no ERROR lines. Their shutdown op-stream MD5s were both
`270e3197a74527b55c79eff94597e1cc`; see
[rank agreement](2026-09-21-qwen-fp8-head-review/rank-agreement.log).

Production was restored on both Sparks to `0.1.0+g5d98ea6b7b13`, executable
SHA256 `3563cebf2c3821e4990d69641827d4acb4376c0c0090d090873c04f97b67967d`.
The original deployment JSON and site file hashes were unchanged, and both
ranks' runtime configuration hashes matched the pre-window value. The service
reported no engine failure, zero active/queued requests and 148 prefix slots.
Both smoke replies were `READY`; the second reused 1128 prompt tokens.

Matched real-checkpoint teacher-forced numerical validation remains pending,
including wide verification and short prefill. The existing GLM teacher runner
and Qwen single-request checkpoint checker do not establish that gate for the
opt-in batched head; it needs a Qwen scoring path with fixed tokens and matched
shapes across GEMV and MMA, plus an unchanged-mode repeatability control. The PR stays draft and no shipped recipe
opts into MMA by default.
