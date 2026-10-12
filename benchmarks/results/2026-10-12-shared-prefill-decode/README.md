# Shared prefill and decode execution

Follow-up for issue #104 and PR #108. **The scheduled correctness, configuration and performance checks are complete.
On 2026-10-12 the maintainer accepted this tested implementation for merge,
with the remaining measured throughput costs documented below. Further
performance investigation continues after this checkpoint.**

## What changed

Qwen text prefill and active decoder verification share a model walk, including
multiple reading requests. Dense, expert, GDN and QSA projections share work;
decoders retain their own recurrent state, rollback rows, MTP and sampling
state. Decoder state kernels are batched. Serving skips intermediate prefill
vocabulary projections and host outputs that are not consumed.

Grouped readers spend one global token budget. Required cuts and minimum
aligned progress remain intact. Decoders that preceded a reader retain the
busy-budget bound. Readers can retain their idle budget when all their
active decoders arrived later and can share execution. Newly admitted
streams may consequently have longer update gaps during large idle chunks.

Omitted early-admission configuration resolves to true for Qwen graph serving
without DFlash. Explicit false retains deferred admission. Other backends keep
their previous default. The journal carries the resolved boolean. Mixed elapsed
time belongs to both logical phases and is exposed separately:
`total engine time = prefill_ms + step_ms - mixed_ms`.

Runtime source: `e17cf1eb33426263533a08aec83d99898da60515`; snapshot-restoration
regression test: `9334bba4`. [source.json](source.json) records the immutable
candidate and control executable hashes. The old control's runtime tree is
`3426204f`, identical to the runtime at the PR #107 merge (`c72e8d9b`). The
contributor fork rejected a push, so the follow-up commits are on
`HawkBearPig/dgpp:perf/104-shared-prefill-decode`.

## Main timing results

Positive numbers mean higher fixed-work throughput against the older control.
Budgets are busy/idle tokens. These are two-trial means, not confidence bounds.
Small differences near one percent are within the scale of run-to-run variation.

| Workload | 1 node, 256/2048 | 1 node, 256/256 | 4 nodes, 256/2048 | 4 nodes, 256/256 |
|---|---:|---:|---:|---:|
| solo_decode | -0.26% | +0.08% | +0.25% | -0.35% |
| solo_prefill | +1.05% | +1.01% | -1.07% | +0.46% |
| mixed_short | +5.37% | +4.98% | +6.15% | +5.92% |
| mixed_long | +7.18% | +6.88% | +9.17% | +8.71% |
| sustained_idle | +0.44% | -2.62% | +0.20% | +1.39% |
| sustained_busy | +12.57% | +12.18% | +12.97% | +12.69% |

The sustained idle burst is 14.257 / 14.320 seconds (candidate/control) on one
node with split budgets, and 9.757 / 9.776 seconds on four. Equal budgets give
21.357 / 20.797 seconds on one node and 12.635 / 12.810 seconds on four.
The single-node equal-budget case still loses 2.62% throughput. This change
is **not universally throughput-neutral**.

The same executable with early admission disabled is a stronger control:

| Configuration | Early enabled, s | Same binary deferred, s | Throughput change |
|---|---:|---:|---:|
| w1-i2048 | 14.257 | 13.672 | -4.10% |
| w1-i256 | 21.357 | 20.218 | -5.34% |
| w4-i2048 | 9.757 | 9.509 | -2.54% |
| w4-i256 | 12.635 | 12.581 | -0.42% |

This isolates the remaining cost of enabled admission from improvements that
also benefit deferred execution. The table pairs the first two deterministic
trials; a third deferred trial is also retained. On the split-budget burst,
both new-build modes perform 256 request-level decode rows and accept all 768
MTP draft tokens. Enabled admission uses 72 decode passes, nine shared with
prefill, versus 64 deferred passes. The remaining elapsed-time difference is
concentrated in the shared prefill intervals. See
[interval accounting](final-v11/interval-accounting.json).

An eight-token arrival behind an incumbent decoder and long prefill finishes
in 1.234 seconds on one node, versus 17.716 seconds for the old control; first
output is 0.649 / 17.564 seconds. On four nodes, completion is 0.746 / 10.700
seconds and first output is 0.392 / 10.611 seconds. The incumbent's mean maximum
SSE update gap improves from 367 to 344 ms on one node and 241 to 208 ms on four.
These are client update gaps, not individual-token latency. Large idle chunks
increase gaps for newly admitted streams; the raw records retain every update.

### Additional configurations

| Configuration and workload | Candidate, s | Old control, s | Throughput change |
|---|---:|---:|---:|
| 1 node, 5 slots, compact MTP3, mixed_short | 17.931 | 18.463 | +2.96% |
| 1 node, 5 slots, compact MTP3, mixed_long | 18.841 | 20.176 | +7.09% |
| 1 node, compact MTP off, solo_decode | 4.128 | 4.120 | -0.18% |
| 1 node, compact MTP off, mixed_short | 17.356 | 18.342 | +5.68% |
| 1 node, compact MTP off, sustained_idle | 18.976 | 18.096 | **-4.64%** |
| 4 nodes, MTP1, solo_decode | 5.508 | 5.503 | -0.08% |
| 4 nodes, MTP1, mixed_short | 14.170 | 14.629 | +3.24% |
| 4 nodes, MTP1, sustained_idle | 11.298 | 11.454 | +1.39% |
| 4 nodes, MTP1, sustained_busy | 17.625 | 18.933 | +7.42% |

These use split 256/2048 budgets and two trials each. The five-slot case doubles
long-prompt words to 16384 and shortens the incumbent to 64 tokens; all eight
incumbents retire before the long prefill completes. Together with the main
matrix, these cover 160 scenarios, 756 requests and 372 exact paired responses.
Their separate audit records retain the parameters and counts.

The MTP-off idle-burst regression is reproducible with the same binary's
admission disabled: three further trials take 18.267, 17.986 and 18.040 seconds
(mean 18.098), matching the old control. Both modes execute 1016 live decoder
rows. The enabled schedule uses 262 decode passes, eight mixed, against 254
standalone passes when deferred.

A separate Nsight trace attributes the extra cost to NVFP4 expert kernels.
The same 12192 gate/up calls take 2407.6 ms enabled versus 1864.8 ms deferred;
the same 12192 down calls take 1017.7 versus 743.2 ms. Dense and other graph
kernels remain similar. The old control's expert costs (1891.6 / 746.2 ms) also
match deferred execution. These sums are kernel durations, include overlapping
streams and must not be treated as whole-run elapsed time. Instrumented runs
are excluded from the timing tables. A separate diagnostic build captures the final main-model layer's routes.
Across the same 254 four-row batches, enabled admission selects a mean 20.094
unique experts, versus 13.035 when deferred: 54.2% more distinct expert visits.
The eight prompts all request the same counting sequence with different task
labels. Starting them together gives deferred batches substantially more
weight reuse; earlier progress separates their generation positions.
The diagnostic patch only redirects an existing dump buffer to expert IDs;
it is not part of the serving candidate. See [routing evidence](final-route-diagnosis-fixed/summary.json).
This samples one layer, not all model routing or DRAM traffic. The kernel trace
and routing observation together support reduced reuse as the cause of this
workload's expert-cost increase. The varied-prompt companion below tests a different routing distribution.

### Varied-prompt companion

The same single-node idle-burst arrival schedule uses eight fixed prompts:
Python, rain formation, JSON, translation, a story, data structures, arithmetic
and a recipe. All use raw completions, ignore EOS and return exactly 128 tokens;
the long prefill still returns one. Three trials per mode use split budgets.

| Mode | Early enabled, s | Old control, s | Same binary deferred, s | Early vs old | Early vs deferred |
|---|---:|---:|---:|---:|---:|
| Compact MTP off | 21.591 | 21.719 | 21.530 | +0.59% | -0.28% |
| MTP3 | 17.943 | 18.926 | 18.716 | +5.48% | +4.31% |

The MTP-off result is effectively neutral at this measurement scale. Together
with the expert trace, it supports a workload-dependent reuse explanation for
the repeated-counting penalty; it does not erase that penalty or establish
neutrality for every prompt distribution. These add 18 scenarios and 162
fixed-length responses. See each `final-varied-mtp*/varied-summary.json`.

These are **performance comparisons, not exact-output correctness passes**.
The strict auditor reports prose hash differences across execution shapes.
The old control and same-binary deferred mode also differ on some prompts;
the difference counts are retained in the summaries. MTP3 records include
full generated text and verified text hashes. Most observed divergences start
in wording after a common prefix; raw completions with forced lengths also
continue beyond natural answer boundaries. No model-quality conclusion is
inferred from this throughput benchmark. The separate numerical/state tests
and canonical transcript checks below are the correctness evidence.

### Method and integrity

The main matrix uses four slots, MTP depth three, BF16 KV, full reservation,
131072 KV tokens and 1 GiB prefix cache. One node uses NVFP4 routed experts;
four nodes use the FP8 checkpoint. Both use FP8 dense weights. Timing requests
are deterministic, cache-disabled and `ignore_eos`, with fixed completion
lengths. The long prompt contains 8192 seeded words; eight sustained arrivals
are dispatched 250 ms apart and return 128 tokens. Native metrics must observe
the long prefill before arrivals are dispatched.

Each deployment gets a warmup. Candidate/control deployment order alternates
between budget configurations, and scenario order reverses on the second
trial. No compiler, profiler or unrelated GPU tests run during timings. Live
binary hashes and zero swap are checked before/after every deployment; rank
operation streams must agree. Clock and temperature snapshots are retained,
not locked. These bounded workloads do not establish performance for every
model, prompt distribution, batch size or device.

[The audit](final-v11/audit.json) covers 124 scenarios and 588 requests, with
288 paired request bodies, outputs and usage records matching exactly. Requests
drain without engine errors or retained uncached KV blocks. Cancellation is
checked for cleanup but is excluded from the throughput/output comparison,
because the cancelled response length need not be identical.

## Correctness and cache checks

The v11 runtime passes four rebuilt host suites, 104 scheduler tests under
ASan/UBSan, nine GPU gates, CUDA memcheck, and the alternate GDN snapshot form.
The GPU gates cover images, mixed MTP-off and scheduled-depth execution,
compact sampling, forced gather fallback and wider MTP3 rows. The added
snapshot gate takes a snapshot during mixed prefill, restores it into a
*different* request slot, and requires bitwise-equal prefill logits, decoder
verify/rollback outputs and MTP logits at matching execution shapes for every
acceptance length 1–4. Both snapshot forms pass, with zero memcheck errors and
all KV references released; see [mixed-cache-gate](mixed-cache-gate/).

One- and four-node API and physical MTP reservation checks pass. Canonical
chat, code, math and JSON transcripts match the pre-PR controls exactly on both.
The four-node soak completes 237 responses with seven expected HTTP 503
rejections and no other response or engine errors. A single-node soak with the
cache threshold lowered to 128 tokens completes 144 responses, exercises 40
cache hits, 148 evictions and 403 snapshot hops, and passes recovery decoding.
A four-node run at the same lowered threshold completes 222 responses with
six expected HTTP 503 rejections, exercises 61 cache hits, 219 evictions and
674 snapshot hops, and passes recovery. Its idle physical pool is 84 blocks
against 134 cache-entry references. All rank operation streams agree.

The cache-only repeat must return to the exact warmed physical-block count.
`prefix_cache.blocks_pinned` sums blocks over entries, so shared physical
blocks can appear more than once: the single-node stress run drains to 28
physical blocks against 41 entry references. The initial equality assertion
was incorrect for shared cache entries; its recorded failing observation had
84 physical blocks both before and after, against 118 entry references.
Uncached benchmark runs and the GPU snapshot-release gate require zero blocks.

An earlier four-node stress attempt drained cleanly but stopped at an overly
strict harness assertion: one zero-token overload error arrived after SSE
headers had committed HTTP 200. The corrected check counts that as a rejection,
not a successful response, and permits no other HTTP-200 error. That attempt
had 219 successful responses and seven overload rejections (six HTTP 503 and
one SSE error); the complete rerun above also verifies recovery and rank logs.

### Cache transcript diagnosis

A cached reasoning response varied between isolated and concurrent execution
in the old runtime too. With admission disabled, the candidate reproduces the
old runtime's two responses exactly. Their first under-load divergence is at
token 34: “expects” / “wants” exchange the top ranks. With shared execution,
the first divergence is at token 32: “.” / “ Likely” exchange the top ranks
(log-probability margins about 0.106 and 0.099). This is evidence of sensitivity
to execution shape, rather than proof that all greedy prose is bitwise stable.
The [probability records](cache-probabilities-v11/summary.json) retain the
mismatch; it is not counted among the exact transcript passes.

The cache check with thinking disabled returns the same 128-token counting
transcript in all three modes, with 2020 cached tokens, completion during the
long prefill, and no unpinned KV after drain. The bitwise GPU restoration test
above independently checks the state-preservation invariant. An earlier raw
completion warmup had no reusable chat boundary and failed before mixed
execution; the serving cache probe now uses a cacheable conversation.

## Iterations

One GB10, four slots, NVFP4 routed experts, FP8 dense weights, BF16 KV, MTP3,
full reservation, 1 GiB prefix cache. Timing requests disable the cache.
Workload: `serve_admission_bench.py sustained_idle`, 8192 words, eight arriving
requests at 250 ms intervals, each returning 128 tokens. Three deterministic
trials per executable, with a warmup on each fresh deployment. Equal budgets
are 256/256; split budgets are 256/2048. See the per-run configuration and
executable hashes. No compiler or profiler runs during timed requests.

| Iteration | Equal candidate mean, s | Equal control mean, s | Split candidate mean, s |
|---|---:|---:|---:|
| v8b: shared grouped-prefill projections | 21.738 | 21.000 | 14.251 |
| v9b: multiple prefills and decoders in one walk | 21.695 | 20.986 | 14.333 |
| v10: batch decoder state kernels within the walk | 21.541 | 20.875 | 14.329 |

These earlier iterations were not throughput-neutral. The final v11 measurements above supersede their timing results. All 27 paired request bodies, outputs and usage records
match in v9b and v10. In v8b, one long request's single output token differs;
that iteration is retained as evidence, not substituted for the later runs.
Earlier broad-v7 measurements are provisional and predate the grouped and
batched-state changes.

Runtime checkpoints: `b1b54df5` (shared and grouped walks), `eb9365b9`
(batched decoder state kernels), `e17cf1eb` (defer intermediate prefill outputs). The contributor fork rejected a push; these
commits are on `HawkBearPig/dgpp:perf/104-shared-prefill-decode`.

## Extreme sampling diagnosis

A temperature-100 full-vocabulary comparison across different prefill chunk
shapes produced different first tokens in both the unchanged runtime and the
candidate. Both use seed 4567 and counter zero. Captured prefill logits differ
by relative L2 0.001548; the probability distributions have total variation
0.000005056. Their sorted-CDF orders select different tokens. Host replay
reproduces both choices, each consuming one draw.

The cross-chunk sampling gate uses temperature 0.7, top-k 7, top-p 0.9 and
asserts the forced gather fallback occurs. Existing full-vocabulary physical
remapping checks retain their original parameters at matching prefill shapes.

```sh
c++ -std=c++20 -O2 -Isrc benchmarks/results/2026-10-12-shared-prefill-decode/replay_sample.cpp -o /tmp/dgpp-replay-sample
/tmp/dgpp-replay-sample benchmarks/results/2026-10-12-shared-prefill-decode/sampling-logits.txt
```

`campaign.py` reproduces matched timings and restores the supplied production
executable/configuration in `finally`. `checks.py` runs separate serving checks;
`cache_diagnosis.py` preserves the reasoning/counting comparison and optional
probabilities. `audit.py` validates accounting and matched request records.

## Acceptance status

The change is enabled by default for supported Qwen graph serving; it is not
hidden behind an opt-in switch. Compared with the old runtime, measured busy
mixed workloads improve roughly 12–13%, and the original single-node equal-
budget burst improves from the earlier 21.70 / 20.99 seconds to 21.357 / 20.797.
Nevertheless, that case remains 2.62% slower, and the compact MTP-off counting
burst loses 4.64%. Against the stronger same-binary deferred controls, the
largest measured loss is 5.34%. Strict throughput neutrality is not achieved.

The maintainer accepted PR #108 for merge on 2026-10-12, enabling the supported
Qwen behavior by default while retaining the measured limitations. This
acceptance does not claim strict throughput neutrality. Production restoration
has been checked against the original configuration and executable on all four
ranks.

The old-runtime comparisons in this record use `3426204f`, the runtime after
PR #107. The intended release comparison for continued optimization is the
latest implementation with early admission enabled against `6fe7f2db`, before
PR #105 and all early-admission work. Historical measurements from that original
runtime exist, but a fresh matched comparison remains follow-up work. The
same-binary deferred comparisons isolate scheduling costs; they are diagnostic
controls rather than the original product baseline.
