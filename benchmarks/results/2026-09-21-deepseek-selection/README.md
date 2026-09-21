# DeepSeek decoder selection — September 21, 2026

Issue #28 replaces DeepSeek-V4.1-Flash's decoder streaming selection sorts
with parallel scoring and exact radix selection. The candidate stage at
layer 20 and restricted selectors at layers 20/24/28/32/36 use the new path.
The three encoder DSA indexers and all prefill kernels are unchanged.

The baseline is master `5631fa1a335a7ecb86b257be358c09677ed08582`.
The [preceding investigation](../2026-09-21-attention-selection-audit/README.md)
records the cross-model audit and the original full-model profile.

## Implementation and exactness

- Score approximately 96 stripes across the active rows, prefetching four
  index rows before the existing per-head FP32 dot/ReLU/weight arithmetic.
- Select exact score/original-ID keys using radix refinement, then return
  the retained IDs in ascending order. Preserve newest-complete-block
  pinning, incomplete tails, causal bounds, counts and `-1` padding.
- When every candidate fits the selection budget, return its ID directly.
- Reuse the existing DSA workspace at a fixed `max_entries` row stride.
  The removed eight-part candidate buffer saves `30 × 8 × 2048 × 8` bytes,
  or 3.75 MiB per rank, for six slots with depth-four MTP. There are no
  per-step allocations; restricted selection adds one graph kernel per call.

The independent small CPU score/selection oracle remains in `csa2_test`.
The new long test uses the unchanged DSA scorer plus host selection oracles
at 131,073 entries, 30 rows, top-2048 blocks and top-512 entries. It covers
zero and one visible entry, selection-budget transitions, all seven partial
tails, permuted physical pages, tied/negative-weight scores, changing positions
under repeated captured graphs, and 1/5/2/30-row calls sharing one workspace.

## Isolated measurements

The matched microbenchmark uses seed 20260921, finite E4M3 data, 128-entry
physical pages, permuted paging, 32 index heads × 128 dimensions, 2048
candidate blocks × eight entries, and top-512 restricted selection.
Baseline and candidate use the same host data, CPU oracle and timers; the
baseline is compiled against the original CSA2 source and API. Fifteen timed
iterations follow three warmups. CUDA graph events measure only the selected
stage; cold samples evict L2 before the timed interval. No other compute
process was present on any Spark during the campaign.

Every tested candidate/restricted stage improved, for both warm and cold
inputs, over contexts 64, 512, 4096, 16384, 16385 and 131072 and row counts
1, 5 and 30. `combined` in the raw data includes one candidate call and one
restricted call; a full decoder pass has five restricted calls. The diagnostic
`restricted_select_only` times the unchanged prefill sort and is not part of
the new decoder.

Warm medians in milliseconds, including scoring and selection:

| Rows | Context | Candidate before | Candidate after | Restricted before | Restricted after |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 0.598 | 0.004 | 0.810 | 0.037 |
| 1 | 16,384 | 0.904 | 0.005 | 3.212 | 0.101 |
| 1 | 131,072 | 2.996 | 0.430 | 3.229 | 0.101 |
| 5 | 4,096 | 0.599 | 0.004 | 0.811 | 0.078 |
| 5 | 16,384 | 0.906 | 0.005 | 3.213 | 0.258 |
| 5 | 131,072 | 2.948 | 1.485 | 3.268 | 0.261 |
| 30 | 4,096 | 1.117 | 0.004 | 0.811 | 0.336 |
| 30 | 16,384 | 2.291 | 0.004 | 3.242 | 1.264 |
| 30 | 131,072 | 12.538 | 8.703 | 3.286 | 1.310 |

All 108 paired candidate/restricted/combined timings improved across the
two cache conditions and 18 shapes. The [raw matrix](micro.json) includes
the remaining short contexts, cold samples, and min/max observations.
These isolated savings are not additive predictions of model throughput:
graph shapes, other kernels, paging and MTP acceptance also matter.

## Four-Spark model comparison

Both builds ran the checked-in DeepSeek MXFP4/FP8 world-four recipe: BF12
weights, bounded prefill, six slots, a 131,072-token KV pool, 1.5 GiB prefix
cache, CUDA graphs and adaptive MTP up to depth four. Only the HTTP endpoint
and log directory were moved aside for testing. No profiler was attached.
The clean baseline server SHA-256 is
`1e6598991a71c96308b0e4b8ddbe841d183dd222cc6144dbfdd2014f208c548d`;
the tested candidate is
`5e139974e4189db9390c434b764f28d10135ccd160e4b51af064688ecada78e8`.

The same deterministic parcel documents, seeds, request order and generation
budgets were used on each build. Each length has one cold 192-token generation
and three cached 256-token generations. Scheduler counters were reconciled
against completed usage before accepting each measurement; no fixed delay
was used as evidence of completion. Decode throughput excludes the first
prefill-produced token and divides by engine decode time, excluding prefill.

Medians of the three cached generations:

| Actual prompt tokens | Before, tok/s | After, tok/s | Throughput change | Before, ms/pass | After, ms/pass |
|---:|---:|---:|---:|---:|---:|
| 3,104 | 50.82 | 54.40 | +7.0% | 58.35 | 54.50 |
| 31,631 | 40.65 | 54.30 | +33.6% | 74.68 | 55.91 |
| 121,384 | 39.51 | 53.60 | +35.7% | 79.67 | 58.73 |

Cold prefill remained within 1%: 2.026 → 2.044 s, 24.287 → 24.463 s,
and 172.658 → 173.561 s. The prefill kernels did not change. Both six-request
batches at 3,118 tokens/request also improved in this run: engine throughput
101.67 → 102.90 and 107.69 → 110.53 tok/s (+1.2% and +2.6%). Concurrent
admission and adaptive MTP produced different pass/row counts, so those
small batch differences should not be treated as a precise speedup estimate.
They show no observed regression; the fixed-shape 30-row microbenchmarks
provide the direct selector comparison.

All **27 paired responses**, totaling **5,641 generated tokens per build**,
matched in output text, finish reason, prompt/completion token counts and
reasoning-token counts. Every sequential request also matched the baseline's
MTP pass and row counts. This includes the cold/cached long documents, two
six-request batches, sampled decoding at seeds 11 and 13, and an arithmetic
control that completed with the correct answer, 401 intact bottles.

Per-request cached-token counts differed in two responses of the first
concurrent batch because request 0 populated the shared prefix on the baseline
and request 5 did so on the candidate. Aggregate cached and computed tokens
matched in every phase. The original check's per-request cache equality was
too strict for racing arrivals; the corrected validation retains exact
sequential usage checks and checks both per-response token usage and aggregate
cache reuse for concurrent batches.

All four ranks had identical operation-stream digests within each run.
The [raw requests, metrics and parity checks](model-ab.json) preserve the
measurements, hashes and cache distinction. The long parcel generations
exhausted their budgets in reasoning: they establish output/performance parity
for these workloads, not a broad retrieval-quality score. No performance or
accuracy regression was observed in the covered cases.

## Validation

- Full CI build with warnings as errors, plus the release server and
  microbenchmark builds, passed. Touched C++/CUDA code passes the repository's
  clang-format rules and `git diff --check`.
- All 30 host test targets passed. The documentation link check initially
  ran before this report existed; it passed on rerun once the report was added.
- Twelve GPU/RDMA targets passed: CSA2 kernel/layer, DeepSeek forward/bounded
  forward/decode/TP/engine, DSA and QSA kernels, and Qwen, GLM-5.3 and GLM-4.7
  decode regressions.
- Compute Sanitizer memcheck passed the full long-context replay test with
  zero errors. Focused racecheck and synccheck passed the independent CSA2
  selection oracle with zero hazards/errors. Full long-shape racecheck was
  stopped because instrumentation was excessively slow; it is not counted
  as a completed check.
- The final microbenchmark's repeated candidate IDs/counts and selected
  IDs/counts also passed at 131K entries and five rows, for warm and cold
  graph runs.
