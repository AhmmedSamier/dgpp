# Qwen FP8 vocabulary-head real-checkpoint numerics (2026-09-21)

## Scope

This campaign compares the GEMV and streaming-MMA vocabulary heads against
fixed teacher tokens on `nvidia/Qwen3.8-Flash-Next-NVFP4`, snapshot
`fc694b54fb0174e0913e6adf86691ef85a4ead47`. It uses the production model
implementation at `6c5d086e8459137687ad5c2be474fea76e079c34` (PR #10 merged),
with the new diagnostic runner in this change. The model kernels are unchanged.
Dense projections are encoded as FP8 at load and the n-gram table is mapped.

The native one- and two-Spark campaigns cover capacity 16, including deeper
MTP verification widths. The YaRN campaign uses two Sparks, capacity 8, and
the shipped factor-2 ramp. Supplemental native capacity-8 campaigns directly
cover the shipped one- and two-Spark recipe boundaries. Each configuration
runs GEMV twice and MMA twice, using the same executable and manifest.

These are matched head-level numerical checks. They do not establish broad
model-quality equivalence, measure MTP draft quality, rerun long-context
retrieval, or refresh the historical throughput measurements. The scorer uses
fixed teacher feeds and commits all verified rows through the production
capture/replay path; MTP drafting is disabled in the scorer. A separate API
run exercises live MTP serving.

## Protocol

`qwen_head_check` captures the production verification graph for request/row
shapes `(2,2)`, `(3,2)`, `(4,2)`, `(4,3)` and `(4,4)`, skipping shapes above
the configured capacity. Each text is divided into equal contiguous request
streams with 16-token prefills. The widest supported shape scores the full
text; other shapes use the first 1024 tokens. Incomplete final groups are
omitted. Graph inspection confirms that the captured head uses MMA exactly
where requested. Short-prefill checks score every row of 32 cold excerpts
per corpus at lengths 1/4/5/8/9/16/17.

The shipped quick, hard and memorized texts tokenize to 556, 7,689 and 6,693
Qwen tokens. The supplemental native capacity-8 runs use the first 2,048
Unicode characters of each text and eight short-prefill excerpts per length.
These excerpts tokenize to 438, 566 and 434 tokens. They retain the same
repeats, dispatch-boundary checks and numerical gates.

Each rank records local log-sum-exp, the teacher token's logit on its owning
rank, local top two tokens, and exact logit/hidden-state hashes. The analyzer
joins the vocabulary shards before computing NLL and global top-1 decisions.
It requires all expected ranks, shapes, repeats and rows, a complete vocabulary
partition, and identical corpus/token hashes and configuration across modes.

The gates, fixed before the runs, are:

- Bitwise logit and hidden-hash repeatability within each mode, on every rank.
- Identical hidden states across modes and ranks, isolating the head change.
- Identical logits outside the optimized interval (`4 < rows <= capacity`).
- Absolute mean NLL delta at most 0.02 nat in every corpus/lane/width case.
- At most 1% of positions changing by more than 1 nat.
- Any global top-1 flip must exchange the top two tokens and have a margin
  at most two BF16 ulps in both modes.

The numerical summary counts each `(corpus, lane, width, position)` once;
the unchanged-mode repeats and vocabulary shards are controls, not extra
independent observations. Cases overlap in their source text.

## Results

All five configurations passed. Across **96,360 logical scored positions**,
there were **zero global top-1 changes** and no token NLL changes above 1 nat.
Both modes repeated exactly on every rank, hidden-state hashes matched across
modes and ranks, and logits outside the optimized interval were unchanged.

| Configuration | Positions | Cases | Max absolute case-mean NLL delta (nat) | Max absolute token delta (nat) | Top-1 changes |
|---|---:|---:|---:|---:|---:|
| [Native, one Spark; capacity 16](2026-09-21-qwen-fp8-head-numerics/w1-summary.json) | 30,192 | 36 | 1.4241062e-05 | 6.2805536e-05 | 0 |
| [Native, two Sparks; capacity 16](2026-09-21-qwen-fp8-head-numerics/w2-summary.json) | 30,192 | 36 | 1.3900536e-05 | 7.6709066e-05 | 0 |
| [YaRN, two Sparks; capacity 8](2026-09-21-qwen-fp8-head-numerics/yarn_w2-summary.json) | 25,432 | 30 | 1.4501202e-05 | 6.8290516e-05 | 0 |
| [Native recipe probe, one Spark; capacity 8](2026-09-21-qwen-fp8-head-numerics/recipe_w1-summary.json) | 5,272 | 30 | 1.2131195e-05 | 5.3399561e-05 | 0 |
| [Native recipe probe, two Sparks; capacity 8](2026-09-21-qwen-fp8-head-numerics/recipe_w2-summary.json) | 5,272 | 30 | 1.1371726e-05 | 6.4281068e-05 | 0 |

## Supporting validation

The native `ci` preset uses GCC 13, CUDA 13 and `RelWithDebInfo`, with warnings
as errors. The full build succeeded. The new analyzer's 14 positive and
negative tests cover complete sharded data, repeat drift, missing/duplicate
data, rank/target mismatch, hidden-state drift, dispatch controls, NLL drift,
global normalization across shards, near-tie versus wide-margin winner flips
(including winners on different ranks), and large deltas that cancel in
the mean. The synthetic GPU scorer fixture passes both modes and all shapes.

The initial scorer fixture caught a missing batch-metadata staging call when
reusing a captured shape. Adding `session_graph_stage_batch()` fixed exact
repeatability before any real-checkpoint scoring. This was a diagnostic-runner
bug; no production graph or kernel code changed. The initial failure and the
passing rerun are retained with the evidence.

The temporary API harness initially put site ports in the deployment JSON;
the launcher rejected it before startup. Moving those overrides to the site
environment fixed the harness. The rejected launch logs are retained.

The full serial CTest run passed 117 of 126 entries initially. The new
analyzer was missing from the required scripts index, and the isolated worktree
had no NumPy environment for four DeepSeek reference generators; their four
consumers consequently had no dumps. Adding the index entry and reusing the
repository's existing virtual environment (NumPy 2.5.3) resolved these failures.
All nine failed entries then passed on rerun: **all 126 CTest entries are clear**.
The optional local zsh staging probe inside the launcher suite was skipped
because zsh is not installed. Both the initial log and the passing rerun are
retained, with a [combined status](2026-09-21-qwen-fp8-head-numerics/ctest-status.json).

A current-build, two-Spark API run used MMA, four request slots, MTP depth 3
and capacity 16. All nine `serve_api_check.py` checks passed, including
streaming, stop handling, multiple choices, logit bias, reasoning-token usage
and cold/hot prefix reuse (0 to 80 cached tokens). Both ranks reported
configuration digest `d4e0850e9af0e0ca`, no rank errors, and identical shutdown
operation-stream MD5 `bb64055b2616ff6aaec64b5667bb7ac3`.

GPU/RDMA work ran with the original four-Spark GLM production deployment
stopped. Production was restored with the original binary and deployment-file
SHA-256 values and exactly matching resolved runtime settings. Health/model
checks and two completion requests passed; the second reused 276 cached tokens
against zero on the first. The service finished with no active or queued
requests. See the [restoration check](2026-09-21-qwen-fp8-head-numerics/restoration-check.json).

## Recipe behavior

The one-Spark, two-Spark and two-Spark YaRN NVFP4 templates now set
`engine.fp8_head: "mma"`. Their FP8 dense/head weights and other serving
settings are unchanged. The general parser/CLI/model default remains GEMV,
so configurations that omit the key preserve their behavior. The FP8-checkpoint
templates retain their BF16 dense/head stack and do not select MMA.

Use `--fp8-head gemv` to restore the previous head dispatch. To retain the
checkpoint's BF16 dense stack, override both settings together:
`--dense-weights checkpoint --fp8-head gemv`. The deployment resolver and
portability/link checks were rerun after updating the templates and docs.

## Reproduction and evidence

See [the numerical-check procedure](../../docs/numerics.md) for manifest
creation, one- and two-Spark launch commands, and the comparison command.
The [protocol](2026-09-21-qwen-fp8-head-numerics/protocol.json) records the
checkpoint, corpus SHA-256 values and gates; the
[supplemental recipe protocol](2026-09-21-qwen-fp8-head-numerics/recipe-protocol.json)
records the native capacity-8 checks. Source and executable hashes are
retained alongside the case summaries and test/API logs.

The raw rank logs, manifests, API operation streams, execution scripts and
exact scoring/serving executables are retained locally in
`artifacts/2026-09-21-qwen-fp8-head-numerics/raw.tar.gz` (not checked in).
The tracked [per-file hashes](2026-09-21-qwen-fp8-head-numerics/raw-SHA256SUMS)
and [archive hash](2026-09-21-qwen-fp8-head-numerics/archive-SHA256SUMS)
identify those artifacts. The raw directories can be passed back to
`qwen_head_compare.py` after extraction.
