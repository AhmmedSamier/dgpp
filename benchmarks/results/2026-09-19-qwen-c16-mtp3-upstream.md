# Qwen C16/MTP3 upstream preparation

Base: `444441feed0c4d7c256ff7ff7868b587823b737f`.
Branch: `codex/qwen-c16-mtp3`.

This change extracts the wide-decode support from local commit
`99278506f938e6f04476dd5dc976a613ab036f96`. It raises Qwen's decode
ceiling to 64 rows, adds 8/12-slot graph families, initializes all picker
sentinels, and widens row-indexed position launches. Above 32 decode rows,
the MTP embedding projection uses the kernel-only projection helper to
avoid cuBLASLt memset nodes in collective graph capture. GLM-4.7 remains
capped at 32 rows. Prefill stays at 2048 tokens; vocabulary-head tuning
and telemetry are separate contributions.

Regression cases cover upper-half position rows, sampled verification,
48/64-row graphs and the real-width projection's numerical and graph-node
checks. Wide CTest registrations retain the GPU/RDMA labels after the
global label assignment and require serial execution.

## Initial preparation status (2026-09-19)

The extracted patch has no whitespace errors (`git diff HEAD --check`).
At initial preparation, no binaries had been built or executed for this branch. The workstation
has no native CUDA compiler or clang-format on PATH and only 2.3 GiB free.
The production inference service was not touched. Historical testing of
the original implementation is not validation of this upstream extraction.

Before contribution: format touched C++, freshly build host and relevant
CUDA targets, run host gates, then run the picker and wide Qwen engine
cases serially on reserved idle GB10 hardware. Cover upstream BF12 storage
modes as well as checkpoint and FP8 dense weights. Build all targets before
an unfiltered CTest run. Run real-checkpoint C16/MTP3 serving and request
contract checks; stop cleanly and compare all participating rank op-stream
md5s. Record binary hashes, configuration, skipped cases and numerical
results. Two-node evidence does not satisfy four-node fabric coverage.
No throughput improvement is claimed and no pull request has been opened.

## Scheduled verification limitation

C16/MTP3 intentionally requires `engine.mtp_schedule=false`. Sixteen scalar
slots and seven batch families consume 46 parity variants per depth; two
depths require 92, above the 64-variant bus limit. The schedule capacity
check now precedes confidence-state mutation and reports configuration field
names, actual capacity, and the remedies of disabling scheduling or reducing
concurrency. A separate diagnostic handles a minimum depth equal to the
configured depth. The wide engine regression checks the C16 diagnostic and
then continues ordinary MTP execution after the rejected configuration.
The new CUDA regression was subsequently built and passed in the 2026-09-20 validation below.

## Native GB10 and two-Spark validation (2026-09-20)

The exact working-tree patch over the base above was transferred to a separate
`<checkout>` checkout on Spark 1. Its SHA256 was
`e4a752d6c349175d0302064928c83072fa02985be58e3dc0cb42cded7e689fee`.
No other contribution's source or build tree was modified.

`cmake --preset ci` and `cmake --build --preset ci -j 4` completed successfully
with CUDA 13 on GB10. The full build preceded the unfiltered CTest run.
With production stopped on both nodes, GPU/RDMA checks ran serially using
`CUDA_DEVICE_MAX_CONNECTIONS=32`:

- Focused picker and Qwen engine CTest selection: 9 passed, 0 failed,
  including 32/64-row FP8 cases, row-independent comparisons, the real-width
  projection's graph/numerical checks, and C16 scheduling rejection.
- Three extra wide tests: `DGPP_TEST_QWEN_ROWS64=1 DGPP_TEST_FILTER=wide
  DGPP_DENSE_GEMV_ROWS=256`, with `DGPP_BF12=off`, `on`, and `both`, all
  passed. These used checkpoint dense weights, not the FP8 override.
- Full `ctest --test-dir build-ci --output-on-failure -j 1 --timeout 180`:
  106 passed, 10 skipped, 0 failed, 641.92 seconds. The skipped tokenizer,
  prompt/template and vision cases require unavailable model checkpoints;
  skips do not validate those models.

The native server SHA256, verified on both nodes, was
`746e96da72d3218d8a12f4b8bee9f1ec9cd7e40fe39f4bdf7f402615adea7283`.
A temporary two-rank `nvidia/Qwen3.8-Flash-Next-NVFP4` service used C16/MTP3,
FP8 dense weights, 65,536 KV tokens, a 1 GiB prefix cache and scheduling off.
It used HTTP 30002 and fabric/journal 29870/29871. Both rank logs confirmed
64 decode rows and captured 16-slot graphs; variants 44/45 contained no
memcpy or memset nodes (the engine's host callback remained present).
All nine `serve_api_check.py` checks passed. Sixteen concurrent requests
with 512-token output budgets all completed with nonempty outputs;
sampled metrics reached 16 active requests. Service metrics reported zero
failed requests and no engine failure. No throughput improvement is claimed.

Clean shutdown produced identical operation streams on the two ranks:
`d86e0b72df5b54b0a0164e1806c72e2d` (MD5).
The first serving attempt was rejected before model startup because the
validation harness put node/port settings in the deployment JSON. Those
settings were moved to the environment as required; only the serving phase
was repeated. This was a harness error, not a C16 model failure.

Production was restored after each maintenance phase. Both ranks run the
original `0.1.0+g42b105d-mtpmetrics` binary with SHA256
`ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`.
The original deployment file hash is unchanged. Two restored-service smoke
requests returned `READY`; the second reused 1,128 of 1,137 prompt tokens.
The original 148-slot prefix cache is present, with zero failed requests
and no engine failure in the post-restoration sample.

Raw build/test logs, source patch, maintenance scripts, both-rank logs,
request outputs, sampled metrics and restoration evidence are retained in
`artifacts/c16-validation/` locally and on Spark 1. This is two-node fabric
evidence, not four-node coverage. Real-checkpoint BF12 at C16 and four-node validation remain outstanding.
Touched C++/CUDA lines were formatted with the repository style before
submission. A token-stream comparison confirmed formatting and comment
relocation did not change the tested code; subsequent changes were documentation
only. The raw patch hash above identifies the pre-formatting tested source.

## Four-node review follow-up

The maintainer reported that the tested PR head passes the full 116-case suite
with all checkpoints available and preserves shipped C1-C4 behavior. At C16,
however, the four-node FP8 checkpoint captures memset nodes at 24 rows even
with FP8 dense weights: GDN a/b projections remain BF16. The maintainer's
`DGPP_DENSE_GEMV_ROWS=64` diagnostic boots and serves C16 on four ranks with
matching operation streams. These results are maintainer-reported, not local
reproductions: see PR #13's review and its correction dated 2026-09-20.

The follow-up enables a bounded kernel-only BF16 matmul range during Qwen
decode (17-64 rows), clearing it during prefill. It uses the existing GEMV
chunks and BF12 companions, preserving <=16-row dispatch. Unsupported shapes
throw instead of returning to Lt. New regressions cover GDN TP=4/TP=2 shard
widths and larger projections at 24/48/64 rows, graph node types, fp64 numerical
references, small-row/prefill equivalence, and unaligned-weight rejection.
GLM-4.7 now owns and enforces its 32-row cap, and a compile-time check verifies
every picker sentinel. Validation of this follow-up is pending; the earlier
build and serving results above apply to the original PR head only.

The follow-up's `qwen_engine_test`, `glm4_engine_test`, `glm_pick_test`, and
`dgpp_serve_app` targets build successfully in the native CUDA 13 `ci` tree
with warnings as errors (`cmake --build build-ci -j 2 --target ...`). Touched
code passes clang-format and `git diff --check`. No GPU test or serving run
has executed the follow-up yet; production was not stopped or deployed during
this build. The build log is retained in `artifacts/c16-review/build.log`.

## Review follow-up validation (2026-09-21)

The full native `ci` build passed. A token-stream comparison of every tracked
C++/CUDA source and header confirmed that the tested checkout matches the
submission, allowing for comments and formatting. The test server SHA256 was
`27e99e6fdfb866c65afb246df0c981313fba402d5c9886bd709f84ab98f7e7b7`,
verified on both ranks. Production was stopped for all GPU/RDMA tests.

- Focused serial CTest selection (`qwen_engine.*`, `glm4_engine_test`,
  `glm_pick_test`): 10 passed. This includes the new real-shard numerical,
  graph-node, small-row/prefill equivalence, alignment-rejection and GLM-4.7
  model-bound regressions.
- Three extra 64-row synthetic runs with `DGPP_DENSE_GEMV_ROWS=4` and
  `DGPP_BF12=off/on/both`: all passed. The 64-row diagnostic override was
  not used.
- Full serial CTest after building all targets: 106 passed, 10 skipped,
  zero failures in 632.74 seconds. Skips still require absent checkpoints.
- Real Qwen NVFP4 checkpoint, two ranks, C16/MTP3, 65,536 KV tokens and
  1 GiB prefix cache: FP8 dense storage, checkpoint BF16 storage, and
  BF12-only storage each passed all nine API checks and sixteen concurrent
  requests. Every mode reached 16 active requests, with no failed requests
  or engine failure. Checkpoint BF16 and BF12-only also completed C1/C2/C4
  functional checks. These are not matched performance comparisons.
- Both rank logs in all three modes recorded 46 graph variants per rank,
  with no memcpy or memset nodes, including 48/64-row families. Shutdown
  was clean and each mode's operation streams matched across both ranks:

| Dense storage / BF16 residency | Two-rank operation-stream MD5 |
| --- | --- |
| FP8 dense | `7ff4305561adcc44063d4d598dfa96a1` |
| Checkpoint BF16 | `8fe830ab8281718831fdc951314a1776` |
| Checkpoint BF16 with BF12-only residency | `73ac6c6029ee28e72eac0ed19ecc64a8` |

The recorded production release for this window was `0.1.0+g5d98ea6b7b13`,
not the older release used in the initial validation. After every phase,
both ranks were restored to its original SHA256
`3563cebf2c3821e4990d69641827d4acb4376c0c0090d090873c04f97b67967d`
and the original deployment file hash. The 256/512 prefill budgets and
148-slot prefix cache were retained. The restored service returned `READY`
twice; the repeat reused 1,128 of 1,137 prompt tokens, with no engine failure
or failed requests in the post-restoration sample.

Logs, configurations, responses, sampled metrics, source comparison and
restoration evidence are retained under `artifacts/c16-review/`. This validates
the follow-up on the available two-node fabric and synthetic TP=4 shard
shapes. A real four-node run of the follow-up still needs the maintainer's
hardware; no four-node success or performance improvement is claimed here.

## Upstream merge validation, 2026-09-21

Merged upstream `c5a69134410f2646d195b339c8f197043410cb04` into the
reviewed `b62ba54` branch. The Qwen engine fixture retains upstream's dense
weight restoration guard and matching eager/graph head settings, using the
configured slot and decode-row counts. Its new telemetry assertions derive
rows per request from the MTP depth instead of assuming MTP1.

On GB10, `cmake --preset ci` and native builds of `qwen_engine_test`,
`dgpp_serve_app`, and the contribution guide's host targets succeeded.
`ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure` passed
21 tests; the new upstream `qwen_yarn_fixture_test` initially could not run
because the guide's explicit target list does not build it. After building
that target, its focused CTest rerun passed, completing all 22 host tests.
Touched-code formatting and the PR delta's whitespace check passed. Logs
are retained under `artifacts/c16-merge/`.

Production was left running and unchanged. GPU/RDMA tests were not rerun on
this merged tree; the earlier GPU and two-rank serving results above apply
to the pre-merge review fix, not this integration.
