# Attention-selection audit beyond Qwen

Date: 2026-09-21. Source and server: `5631fa1a335a7ecb86b257be358c09677ed08582`.
Question: does the long-context sorting bottleneck fixed for Qwen in
`1f357da` also occur in the other supported model families?

This is the investigation of the baseline above. Issue #28's
[implementation and comparison](../2026-09-21-deepseek-selection/README.md)
records the subsequent DeepSeek fix. The standalone benchmark in this
directory targets the baseline's internal kernels; use the current
`csa2_select_bench` target for the public decoder API after the fix.

## Findings

| Model/path | Same sorting problem? | Evidence |
|---|---|---|
| GLM-5.3-Flash DSA | No equivalent old streaming-sort path in decode | `DsaLayer::enqueue_decode` calls `dsa_select_decode`, which already uses parallel scoring and exact radix refinement. |
| Full GLM-5.3 DSA | No equivalent old streaming-sort path in decode | The same radix kernel, configured for per-token indexing, ReLU scores and top-2048 selection. |
| GLM-4.7 | Not applicable to its attention path | `Glm4AttentionLayer::enqueue` calls dense paged GQA attention, with no sparse top-k indexer. Context-dependent attention cost remains possible. |
| DeepSeek-V4.1-Flash encoder indexers | No equivalent old streaming-sort path in decode | Layers 2, 8 and 14 use `dsa_select_decode`, with ratio-two compressed entries and top-512 selection. |
| DeepSeek-V4.1-Flash decoder indexers | Yes, a related under-parallelized scan-and-sort bottleneck | Layer 20 builds/merges candidates; layers 20, 24, 28, 32 and 36 each run the old restricted streaming selector. Both scoring and sorting are expensive. |

The DSA radix implementation predates BF12: it entered in `2b7a6c6`
(2026-09-06). GLM's remaining long-context growth is principally scoring
more index entries, plus selection/expansion work; the Qwen fix cannot
simply be reapplied to remove an already-replaced algorithm.

DeepSeek's candidate source keeps 2048 blocks of eight entries. Restricted
selection therefore examines at most 16,384 entries plus a partial newest
block. Its cost rises up to that working-set size and then largely levels
off. Candidate construction still scans the growing full context.
This differs from Qwen's old selector, which sorted the growing complete
compressed-key pool on each call.

## Isolated kernel measurements

One idle DGX Spark GB10, 48 SMs, 24 MiB L2. Production was stopped during
GPU measurements. Native release settings: CUDA 13.0, `-O3`, `sm_121a`,
host FP contraction disabled. The benchmark calls the current DSA library
and includes the unchanged CSA2 implementation to time internal stages.
No production kernel is modified.

Paging uses a shuffled physical block table and the production 128-token
blocks: 32 pooled entries/block for Flash, 128 for full GLM and the
DeepSeek decoder, 64 for the DeepSeek encoder. Query/index keys are
deterministic finite E4M3 values, with per-entry scales and 32 heads of
width 128. Rows have adjacent positions in one request's cache. This
isolates kernel behavior; it is not a multi-request throughput benchmark.

Measurements use CUDA graphs, three warmups and 15 samples per stage;
the largest DeepSeek, Flash and full-GLM cases were repeated with five
warmups and 30 samples. Cold samples write a separate 96 MiB buffer before
the timed interval. Profiler replay/clock-control timings are excluded.

### GLM and DeepSeek encoder controls

Warm medians, microseconds per fused score/select launch:

| Path | Rows | 4,096 tokens | 32,768 tokens | Longer-context point |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | 2 | 24.4 | 64.1 | 189.1 at 131,072; 352.1 at 262,144 |
| Full GLM-5.3 | 2 | 143.0 | 293.3 | 740.3 at 122,880 |
| DeepSeek encoder | 5 | 50.9 | 217.7 | 762.2 at 131,072 |

DSA's existing last-block phase stamps support the distinction: at the
longest points above, scoring in that block takes approximately 277, 509
and 685 microseconds respectively, while accumulated selection takes 63,
115 and 61 microseconds. These are diagnostic stamps from one launch,
not independently timed serial phases or quantities to sum across blocks.
They do not establish that these paths have no further optimization potential.

### DeepSeek decoder stages

Five rows, matching a single request's maximum depth-four speculative
verification shape. Warm medians in milliseconds:

| Context tokens | Candidate scoring/partial selection | Candidate merge | One restricted score/select | Restricted selection with precomputed scores |
|---:|---:|---:|---:|---:|
| 4,096 | 0.133 | 0.469 | 0.813 | 0.153 |
| 16,384 | 0.344 | 0.565 | 3.215 | 0.579 |
| 32,768 | 0.628 | 0.571 | 3.235 | 0.592 |
| 65,536 | 1.196 | 0.584 | 3.240 | 0.592 |
| 131,072 | 2.334 | 0.620 | 3.247 | 0.593 |

The independent repeat at 131,072 tokens measured 2.360, 0.618, 3.260
and 0.592 ms respectively. Cold medians were 2.874, 0.637, 3.713 and
0.627 ms. The difference persists with cache eviction and repeated runs.

The last column uses the existing prefill selector on exactly the same
materialized scores and candidate IDs. It is a diagnostic lower bound for
the sorting/selection work, not a deployable replacement or a measured
optimization. About 0.6 ms of a restricted call remains with scoring
removed; scoring, cache access and their execution layout account for
most of the full call. Consequently, replacing sorting alone would not
remove the entire 3.2 ms cost.

There are five restricted calls per model decode pass. Their isolated
warm cost is therefore about 16.2 ms at long context, before candidate
construction and the rest of the model. This multiplication is a kernel
budget estimate, not a measured end-to-end speedup.

One-row and 30-row checks confirm the occupancy behavior: restricted
selection remains about 3.2 ms at 131K because one block handles each row.
Candidate partial selection grows from 2.37 ms with one row to 11.99 ms
with 30 rows; it already distributes work over eight blocks per row.

## Hardware counters

Nsight Compute, 131,072 tokens and five rows. Collection used temporary
`sudo -n ncu` access; no driver permissions or persistent system settings
were changed. A first unprivileged attempt could not access counters and
is excluded from performance results.

| Stage | GPU blocks | SM throughput, % of peak | L2 throughput, % of peak |
|---|---:|---:|---:|
| Candidate scoring/partial selection | 40 | 36.90 | 1.33 |
| Candidate merge | 5 | 3.37 | 0.09 |
| Restricted score/select | 5 | 4.10 | 0.19 |

The merge has no dot-product scoring: it sorts/merges eight fixed
top-2048 partial lists in one block per row. Its stall report emphasizes
fixed-latency waits and branch resolution, consistent with the source's
bitonic networks. Restricted selection also has very low device coverage
and runs its 32-head scoring in those same few blocks. These measurements
do not support an explanation based on saturated device memory bandwidth.

## Four-Spark DeepSeek trace

The unchanged `5631fa1` release served the existing four-node DeepSeek
MXFP4/FP8 recipe with bounded prefill, adaptive MTP up to depth four,
six available slots and a 131,072-token KV pool. Requests were serial on
a private loopback endpoint. Rank zero ran under Nsight Systems with
CUDA graph-node tracing; the other ranks ran the same binary normally.

Deterministic parcel records from `qwen_yarn_release_check.filler` were
calibrated against the DeepSeek tokenizer. Actual prompts contained
31,631 and 121,384 tokens. Each prompt was first processed with one
output token, then repeated with a 192-token output budget to measure
cached generation. Both measured requests generated all 192 tokens,
entirely reasoning, and ended with `length`. This is a performance
workload, not a completed-answer or retrieval-quality evaluation.

Each request executed 60 decode graph passes. Kernels are grouped by
their CUDA graph-launch correlation ID, with launches assigned to client
request intervals using the profiler's UTC session epoch. Each pass has
exactly one candidate construction/merge, five restricted selections,
and three encoder DSA selections. Adaptive verification used two, four
or five rows, with similar distributions in the two requests.

| Actual prompt tokens | Mean graph span, ms/pass | Candidate scoring/partial selection | Candidate merge | Five restricted calls | Decoder selection total | Share of graph span |
|---:|---:|---:|---:|---:|---:|---:|
| 31,631 | 77.21 | 0.76 | 0.58 | 17.82 | 19.16 | 24.8% |
| 121,384 | 82.26 | 2.73 | 0.63 | 18.50 | 21.86 | 26.6% |

Stage durations are milliseconds per pass from rank zero's trace. The
server's independent decode-event counters report 76.86 and 81.95 ms/pass.
The three encoder radix selections together take 0.51 and 1.70 ms/pass.
Restricted selection costs 3.56 and 3.70 ms per call in the real model,
consistent with the isolated warm/cold measurements.

Thus the costly decoder selection path is present in real serving, not
just the synthetic fixture. The 24.8–26.6% shares are measured costs,
not promised recoverable latency or speedups. Scoring is required work;
any replacement must retain the exact score, candidate and tie semantics.
These are one traced request per context, not a broad throughput campaign.

All four ranks' operation streams matched at shutdown:
the exact SHA-256 and per-stage data are in `raw/model-profile.json`.
Full prompts/responses and reports are retained in the local artifacts.

## Recommended follow-up

Prioritize DeepSeek's restricted score/select path: distribute scoring
across more GPU blocks, then select with an exact radix method while
preserving original entry IDs and their tie ordering. Replace the
candidate merge's full streaming sort separately. Short-context candidate
construction can also be investigated for a direct all-blocks path when
every complete block already fits the candidate budget.

Validate any implementation against the candidate-block maximum, pinned
newest block, partial tail, ReLU, per-head accumulation order, adaptive
speculation and graph-replay contracts. The investigation does not
implement or claim performance for those prospective changes.

## Correctness and interpretation

All 19 fixture configurations and three repeated configurations pass
selection checks, including the one-, five- and 30-row DeepSeek shapes.
The host sorts the DSA kernel's materialized composite score keys and
checks complete selected IDs, causal bounds, tie ordering, expansion and
padding. For CSA2 it independently derives block maxima, the pinned
newest complete block, candidate IDs, partial-tail eligibility and the
restricted top-k. Timed repeated launches preserve the expected output.

This validates the benchmark's selection workload. The score keys come
from production DSA arithmetic, so this is not an independent oracle for
all score arithmetic or a new model-quality evaluation. No model or
serving implementation was changed during this investigation.

Production was restored to the current `5631fa1` build. The final audit
checked the actual `/proc/PID/exe` on all four nodes, their versions and
identical binary SHA-256
`1e6598991a71c96308b0e4b8ddbe841d183dd222cc6144dbfdd2014f208c548d`,
matching operation-stream digests, health and an idle scheduler. The
deployment configuration retained its original bytes. The first Systems
launch encountered a temporary-directory permission conflict and exited
before serving; the retry used a private temporary directory. Both
maintenance harnesses restore production in their cleanup paths.

## Reproduction

From a checkout of `5631fa1` with the release libraries built:

```bash
nvcc -O3 -DNDEBUG -std=c++20 -arch=sm_121a \
  --extended-lambda --expt-relaxed-constexpr -Xcompiler=-ffp-contract=off \
  -I src benchmarks/results/2026-09-21-attention-selection-audit/select_bench.cu \
  -o /tmp/dgpp-selection-audit \
  build-release/libdgpp_kernels.a build-release/libdgpp_core.a \
  build-release/libdgpp_rope_scaling.a -lcublasLt

/tmp/dgpp-selection-audit --family deepseek --ctx 131072 --rows 5
/tmp/dgpp-selection-audit --family flash --ctx 262144 --rows 2
/tmp/dgpp-selection-audit --family full --ctx 122880 --rows 2
/tmp/dgpp-selection-audit --family deepseek-encoder --ctx 131072 --rows 5
```

`--eager` disables graph replay. `--stage` selects `candidate_parts`,
`candidate_merge`, `restricted`, `restricted_select_only`, `combined`
or `dsa`. `combined` measures one candidate construction and one restricted
selection; it does not represent all five decoder indexers.
`--iters` and `--warmup` set sample counts. Run GPU measurements only on
idle hardware; compile before measuring.

All unprofiled timing rows are in `raw/micro.json`; hardware-counter
summaries, with units, are in `raw/ncu.json`. Full logs, Nsight reports and
the site-specific maintenance/client harnesses are retained locally under
`artifacts/attention-selection-investigation/`.
