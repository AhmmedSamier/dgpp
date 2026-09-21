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

## Maintainer integration and four-node validation, 2026-09-21

The final review found one remaining small-batch regression: enabling the
17–64-row guard for every decode also changed existing MTP hidden projections.
Six or eight token rows expand to 24 or 32 matrix rows through the four
hyper-state branches, replacing their previous Lt dispatch with GEMV chunks.
The follow-up selects the guard from the token count before that expansion.
Walks of at most sixteen tokens and prefill retain their previous dispatch;
wide decode retains the kernel-only protection.

A real-width regression captures the existing hidden projection at
2/4/6/8/12/16 tokens in both prefill and decode. It checks graph node types,
inspectable kernel functions and launch shapes, and bitwise output equality,
including a small walk following wide configuration on the same GEMM object.
Driver-loaded cuBLAS kernels are counted as opaque kernel nodes. Temporarily
restoring the unconditional guard makes this regression fail on its dispatch
assertion; the corrected implementation passes.

Validation started on a local merge of `b62ba54` and upstream `c5a6913`.
The author independently pushed merge `5329349` during the maintenance window,
including the same depth-aware telemetry assertions and FP8-head fixture
integration. The follow-up was moved onto that merge. All tested source files
are byte-identical after integration; the additional author changes are
documentation. The native CUDA 13/GCC 13 `ci` build completed with warnings
as errors. The server SHA256, checked on all four ranks for each mode, is
`a75fd06e2a7b321327aa4544fde8c548add4fb31c8411158032728f6bc243198`
(build stamp `0.1.0+gf11922583351.dirty`).

- Host gates: 22/22 passed.
- Full serial CTest: 127/131 passed initially; four wide MTP3 fixture checks
  exposed inherited telemetry assertions that assumed depth one. Updating
  them to use `1 + mtp_depth` and rebuilding made all four targeted reruns
  pass. All 131 entries are now clear, with no skips or remaining failures.
- Four GB10 nodes with the real `Qwen/Qwen3.8-Flash-Next-FP8` checkpoint:
  C16/MTP3, scheduling off, 65,536 BF16 KV tokens, 1 GiB prefix cache, and
  default dense GEMV lowering. Each storage mode passed all nine API checks
  and a prose sweep at C1/C2/C3/C4/C8/C12/C16, one repetition with a 96-token
  output budget. All 46 load requests per mode returned nonempty responses,
  sampled occupancy reached sixteen, and service metrics reported no failed
  requests or engine failure.
- Each mode captured 46 graph variants on every rank with zero memcpy or
  memset nodes. Shutdown was clean, no rank logged an error, and operation
  streams agreed across all four ranks:

| Dense storage / BF16 residency / vocabulary head | Four-rank operation-stream MD5 |
| --- | --- |
| Checkpoint BF16 / checkpoint / GEMV | `5061b03d40695ddae8b5c66c2fa45271` |
| FP8 dense / checkpoint / MMA | `0d064cf21927fcc9bbf1d6b5064b5746` |
| Checkpoint / BF12-only / GEMV | `2663a073e5b204df7a0b2f66f938d870` |

One preliminary serving attempt used an invalid `bf16_weights: "bf16"`
value in the temporary harness configuration. Startup rejected it before
serving; the harness restored production, corrected the value to
`"checkpoint"`, and restarted the serving phase. This was a harness error.
These are functional checks of the final integration, not a matched
throughput comparison or a long-context/quality evaluation. The earlier
two-node evidence above remains separate from this four-node result.

Production was restored after every maintenance phase. All four running
processes again use the original `0.1.0+g16217c3f2384` binary, SHA256
`460f5ed1c4b4e99d9be99768a4f636ab2aa40e21dcb43ab25d2360c5dc644bdd`.
The deployment file and resolved configuration are unchanged, including the
256/2048 prefill budgets and 8 GiB prefix cache. Health and completion checks
passed; a repeat prompt reused 176 of 181 tokens, with no failed requests or
engine failure in the restoration sample.

The [checked summaries, temporary configurations and test logs](2026-09-21-qwen-c16-followup/)
are committed alongside this record. Complete build logs, per-rank logs,
request outputs, occupancy samples and maintenance scripts remain in
`artifacts/pr13-followup/` in the validation checkout.
