# Qwen batching and prefill continuation (2026-09-15)

**Status: candidate implementation; the C1 promotion gate has not cleared.**

Implementation and fabric validation of the first three deliveries in the
[performance plan](../../docs/performance_improvement_plan.md). Baseline
commit: `3c537b620fc2eb9e223d1519c6e78c734d3e96e7`.

## Changes and default behavior

- Qwen supports sixteen physical decode rows: eight requests with native
  MTP depth 1. Rows above eight use the general shared-expert tail. The
  existing fused tail still serves rows one through eight.
- The graph engine captures the physical slot prefixes that fit its row
  capacity. Configuring a deeper MTP chain no longer disables every batch
  family just because a full batch cannot fit. A live set extending past
  the largest family uses scalar replays; arbitrary subsets are future work.
- `engine.prefill_budget_tokens` enables Qwen graph prefill continuation.
  One request advances by at most that many prompt tokens per scheduler
  tick, followed by a decode pass. The default is zero. Other families
  retain monolithic admission and reject a nonzero budget.
- A continuation owns its prompt, cursor, reservations and cache snapshot
  across yields. It hides its device positions from padded graph rows and
  restores other streams' feeds after every chunk. Cancellation releases
  partial state; an older request can reclaim a younger continuation's
  reservation under grow-on-demand admission.
- Execution time for a prefill group is counted once in `prefill_ms`;
  `prefill_request_ms` separately sums requests' waits. SSE measurements
  use server token counts and incremental visible-output timestamps.

Shipped deployment templates retain their slot counts, MTP settings and
monolithic admission. No C1 kernel or speculative depth default changed.
The original and final executables have byte-identical 50,737,648-byte
CUDA fatbin sections (SHA-256 in the manifest). Default Qwen C1 graph
captures both contain 1,661 kernel nodes and 1,761 edges. These checks
narrow the investigation; they do not prove identical runtime performance.

## Method

Four GB10 Sparks were available; Qwen uses the first two. Checkpoint:
`Qwen/Qwen3.8-Flash-Next-FP8`, revision
`236dfdf285828023ca3bcd3f37366c58a3469b13`.
Weights are resident, K/V BF16, capacity 262,144 tokens, native MTP depth 1,
128 sampling candidates/rank, prefix cache 1.5 GiB. Baseline has four slots.
The widened experiment has eight slots and a 256-token prefill budget.
Its memory plan is 99.01 GiB + 4 GiB headroom/rank, versus 98.35 + 4 GiB.
A separate four-slot/budget-256 run isolates prefill scheduling.

Each load phase is a simultaneous client burst, greedy, thinking disabled,
320 completion tokens/request, three repeats per class and concurrency.
The separate fresh Qwen C1 checks use five repeats.
Class C1 anchors and the baseline binary are preserved. Prefix caching is
enabled; repeated prompts become cache hits. C6/C8 baseline bursts queue
through four slots. These are request-wall rates, not fixed-occupancy
steady-state decode. The report also retains the explicitly named legacy
output-span rate, which excludes initial TTFT and includes later admissions.

The [machine-readable report](2026-09-15-qwen-batching-prefill.json) retains
per-trial timing, usage, prompt/transcript hashes, gate failures and the
configuration manifest. Raw text, build/test logs, deployment logs and
binary hashes are retained in
`build-ci/performance-baseline-2026-09-15/` and
`/home/stephen/dgpp/log/performance-*-2026-09-15/` on the head node.

```bash
python3 scripts/serve_load.py 127.0.0.1 18080 \
  --classes all --concurrency 1,2,4,6,8 --max-tokens 320 --repeat 3 \
  --json-out RUN.json
python3 scripts/bench_compare.py BASELINE.json CANDIDATE.json
python3 scripts/serve_prefill_interference.py 127.0.0.1 18080 \
  --prompt-words 24000 --decode-tokens 1024 --tag performance-2026-09-15 \
  --json-out INTERFERENCE.json
```

## Validation and issues caught

The full CI build and ten focused suites pass: Qwen forward, decode,
MoE, TP, graph engines (production and row-independent target lowering),
scheduler, journal, unit tests and Python benchmark reports. Shared-session
regression suites for GLM-4.7, full GLM and DeepSeek also passed earlier in
the campaign; the service admission metrics check passed after its expected
JSON was updated.

The wide target test compares 224 teacher-forced rows against serial target
execution, enforcing relative logit L2, top-1/near-tie and mean NLL bounds.
It measured worst relative L2 0.009175, two near ties, and mean NLL change
0.000106 nat (limit 0.02).
Exact MTP/plain equivalence is checked with row-independent target kernels.
Production graph tests require unchanged transcripts for unaffected slots
when an interior slot is cancelled, reused or prefills across yields at the
same physical batch width. They also cover cold and attached-prefix resume.
Floating-point reduction order differs across existing dense batch shapes;
these tests distinguish target numerics from speculative state correctness.

The first real-model C8 startup failed its graph-node audit. Qwen's draft
hidden projection flattens twelve/sixteen rows across four hyper-state
branches into 48/64 rows. At hidden size 2560, cuBLASLt inserts a memory
clear node, forbidden by the collective graph contract. The widened decode
projection now uses the existing streaming tensor-core kernel. A test at
the real shape captures and replays it, audits node types and checks
independent double-precision dots. Prefill and <=8-row decode keep their
previous dispatch. The temporary diagnostic wrapper is not part of the
candidate benchmark or implementation.

## Results

### Default-configuration C1 gate

The [C1 difference analysis](2026-09-15-c1-analysis.md) quantifies absolute
latency, timing scopes and baseline reproducibility. The zero-threshold
gate below is a sensitive screening rule; its failure alone does not
establish a material regression caused by the implementation.

The initial default-configuration C1 screen matched all five transcripts.
Nine of ten median rate checks met baseline. Prose wall rate was 0.12%
lower, so the zero-tolerance gate correctly failed; that screening result
is retained in `qwen-candidate-c4-c1-compare.json`.

A fresh five-repeat A/B using the untouched baseline binary and final
candidate, each with four slots and budget zero, also matched all C1 text.
It **failed the zero-tolerance rate gate** on nine of ten checks:

| Class | Baseline wall tok/s | Candidate wall tok/s | Wall change | Output-span change |
|---|---:|---:|---:|---:|
| Prose | 44.68 | 44.60 | −0.17% | −0.35% |
| Code | 48.59 | 48.44 | −0.32% | −0.30% |
| JSON | 50.15 | 50.13 | −0.03% | −0.24% |
| Math | 47.16 | 47.01 | −0.31% | +0.04% |
| Chat | 41.32 | 41.15 | −0.40% | −0.42% |

The fresh baseline itself was below the original three-repeat baseline.
Clock/power state was not fixed across runs. These sub-percent differences
do not isolate an implementation cost from system variation, but they do
**not clear the requested C1 gate**. No timing tolerance was relaxed and no
trial was discarded. Promotion remains withheld; the original deployment
was restored with its preserved binary. The machine-readable results
retain every repeat, both timing scopes and their ranges.

The separate four-slot budget-256 run also matched all five C1 transcripts
but failed all ten median rate checks: wall rates were 0.18–0.58% lower,
and output-span rates 0.34–0.67% lower, than the fresh baseline. This setting
also remains opt-in. Neither chunking nor capacity expansion is presented
as a single-stream speed improvement.

### GLM-Flash C1 regression check

The unchanged four-node hybrid checkpoint is
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`, revision
`17b8d4d83e33e836e9549d7e1a8d25b5117203c1`: four slots, native MTP depth 1,
786,432-token BF16 KV pool, 8 GiB prefix cache and 128 sampling candidates.
Three repeats per class, 320 output tokens, greedy, thinking disabled.

The candidate matched all five transcripts. Against the original running
service baseline, wall medians were 0.49–0.73% lower and all ten rate checks
failed. After restoring the untouched binary, a fresh baseline also ran
below the original service's figures. The back-to-back fresh comparison is:

| Class | Fresh original wall tok/s | Candidate wall tok/s | Wall change | Output-span change |
|---|---:|---:|---:|---:|
| Prose | 53.42 | 53.71 | +0.54% | +0.23% |
| Code | 57.58 | 57.49 | −0.15% | +0.19% |
| JSON | 57.95 | 58.12 | +0.30% | −0.15% |
| Math | 55.37 | 55.37 | −0.01% | +0.07% |
| Chat | 50.08 | 50.19 | +0.23% | −0.15% |

Four of ten checks still fail the zero-tolerance gate. All results are
retained; the fresh baseline is an additional diagnostic, not a replacement
for the original record. The candidate's operation streams matched on all
four ranks: `2b90478cd3ec626ff5029e5db78da8f6`.

### Eight-slot experiment

Median aggregate request-wall tokens/s; three repeats. Both slot count and
prefill budget change in this experiment. The baseline C6/C8 requests queue
through four slots, while the candidate can serve all eight concurrently.

| Class | C6 baseline | C6 candidate | Change | C8 baseline | C8 candidate | Change |
|---|---:|---:|---:|---:|---:|---:|
| Prose | 77.54 | 100.12 | +29.12% | 85.67 | 112.85 | +31.73% |
| Code | 86.58 | 115.03 | +32.86% | 97.61 | 126.59 | +29.69% |
| JSON | 83.05 | 119.59 | +44.00% | 101.57 | 131.84 | +29.80% |
| Math | 85.13 | 115.20 | +35.32% | 96.43 | 125.35 | +29.99% |
| Chat | 77.92 | 102.21 | +31.18% | 87.82 | 112.23* | +27.79%* |

*Chat C8 has an unmatched output length: one answer ends naturally at
286/288 tokens in candidate trials versus 320 at baseline. Its rate is
observational; it is excluded from matched-throughput claims. The comparison
tool rejects that full sweep as uncomparable. Existing production batch
shapes can change late greedy decisions through floating-point reduction
order; independent numerical and quality checks are reported separately.

All 315 requests in the candidate sweep succeeded. C1 text matches baseline
exactly, but its median wall rates are **0.72–3.15% lower**. C2 changes range
from −3.72% to −0.36%; C4 from −0.72% to +0.15%. The eight-slot configuration
fails the C1 gate and is **not promoted to a deployment default**.

### Long-prompt interference

A cold 25,635-token prompt arrives during a 1,024-token prose decode.
The baseline and both candidates produce identical decode text.
Client update gaps measure visible SSE delivery, including network/server
buffering; they are not individual GPU token-commit times.

| Configuration | Long prompt request wall | Maximum decode update gap | p95 gap |
|---|---:|---:|---:|
| Baseline, 4 slots, budget 0 | 18.90 s | 18,912 ms | 50.88 ms |
| Candidate, 8 slots, budget 256 | 33.39 s | 356.55 ms | 330.06 ms |
| Candidate, 4 slots, budget 256 | 33.50 s | 356.32 ms | 330.00 ms |

The maximum interruption falls by 98.1%, with a 76.7–77.3% increase in the
long request's wall time. Holding slot count at four reproduces the effect.
Chunking trades prefill efficiency for decode responsiveness; it is not a
prefill-throughput improvement.

### Quality and lifecycle

The untouched four-slot baseline and eight-slot candidate both scored
29/30 on GSM8K and 30/30 on structured extraction, with no truncated answers
(thinking disabled, maximum 2,048 tokens). Both missed `gsm8k-12` with 12
instead of the reference 13. This is a small matched smoke sample, not a
full model-quality evaluation. The GSM8K source is pinned in the manifest.

The final four-slot lifecycle check cancelled a partially computed long
prefill before the first pick; its reservation returned to the starting
pool count within 206 ms. It then passed sixteen temperature-0.7 constrained
extractions, including eight repeated requests with prefix attachment. The
idle pool was covered by cache references, with no active request records.
The final run's counters reported partial work without negative cache-token
counts. Scheduler tests also cancel at each yield and exercise reservation
pressure. Both ranks agreed on operation stream
`5c3bfa29db76657fd562482277149556`.

An earlier C8 lifecycle probe used an incorrect final assertion equating
physical blocks with the sum of cache-entry references; entries can share
blocks. That assertion was corrected and the full lifecycle rerun passed.
The earlier assertion failure was in the probe, not the engine.

The C8 fabric operation streams agree on both ranks:
`4ae7257fcceaa65c8008cbf5a1c7afa3`. This checks replicated execution order;
it does not establish numerical accuracy or task quality.

## Deployment and next gate

The original four-node GLM deployment was restored at 07:11 UTC using the
original configuration and preserved executable. All four running
`/proc/PID/exe` hashes match baseline SHA-256
`49c0c9dc9fbde3cc04138292cf7059b6cc2e30e1cdb2dbe08aa5c51a39fa8259`.
The service is healthy and idle after the final baseline run. The candidate
is left in the workspace for review; it has not replaced that deployment.

Before promotion, resolve the sub-percent C1 differences with interleaved
A/B pairs and continuous per-rank clock/power telemetry, identical fresh
cache preparation, and separate scheduler/launch, GPU-step and HTTP-delivery
timings. The byte-identical kernels and mixed fresh GLM results narrow the
question but do not waive the gate. Keep the eight-slot and prefill-budget
settings opt-in even after the default configuration clears C1: their
throughput/latency tradeoffs are separately measured here.

## Scope still to implement

Grouped continuation, other model families' continuation, arbitrary slot
subsets, GLM-Flash runtime row expansion, packed int4/int8 prefill kernels,
and DFlash2/K7 integration remain separate deliveries in the plan. Existing
DeepSeek DSpark support is unchanged. This change does not claim new
single-stream kernel wins or cross-engine benchmark parity.
