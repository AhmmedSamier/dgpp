# Qwen QSA prefill: KV reuse and cooperative probabilities

Date: 2026-09-15. Baseline production source: `4282642`, using the preserved
final binary from the GLM packed-prefill campaign. That binary was built
before its commit and reports `2ea2c299bd86.dirty`; the raw record retains
both build headers and executable hashes. Checkpoint revision:
`236dfdf285828023ca3bcd3f37366c58a3469b13`.

The new prefill kernel reduces cold TP2 service prefill time by **8–14%** on
the measured 2K/8K/32K prompts. It shares a K/V tile across twelve query heads
and calculates probabilities cooperatively within each warp. It preserves
the previous FP32 partial results, including the BF16 probability rounding.

[Raw measurements and provenance](2026-09-15-qwen-qsa-prefill.json).

## Why it helps

Qwen has twelve query heads per KV head at TP1 and TP2. The old kernel's
six-head blocks load the same K/V tile twice. Each block uses about 34 KiB
of shared memory, allowing two resident blocks per SM on GB10: only twelve
active warps out of a possible forty-eight.

The prefill kernel uses twelve-head blocks with almost the same shared-memory
allocation. This doubles the active warps and removes the duplicate tile
load within each query's KV head. Separately, each warp lane evaluates one
probability, and broadcasts it in the existing token order. The denominator
sum and the value accumulation retain their original FP32 chains.

Nsight Compute on a synthetic 256-row, 8K-context TP2 case:

| Counter | Baseline | Candidate |
|---|---:|---:|
| Threads per block | 192 | 384 |
| Blocks launched | 4,096 | 2,048 |
| Shared memory per block | 34,608 B | 35,424 B |
| Registers per thread | 48 | 48 |
| Theoretical occupancy | 25% | 50% |
| Achieved occupancy | 24.68% | 49.38% |
| Executed instructions | 808,019,296 | 545,222,720 |

These counters explain the improvement through occupancy, instruction count
and tile reuse. They do not establish a halving of external-memory traffic:
the old duplicate loads can hit the cache. Instrumented durations are not
used as service timings.

## Dispatch and short prompts

Only non-decode calls with at least **128 rows** use the new kernel.
Decode, speculative verification and shorter prefills keep the previous
dispatch. There is no new workspace or change to the selected token list,
number of splits, 32-token tiles, dot-product reduction or merge kernel.

The cutoff follows a measured throughput crossover. On short TP2 cold
prefills, the new kernel takes 67.7 µs versus 63.9 µs at 64 rows, and
101.5 µs versus 99.8 µs at 96 rows. At 128 rows it is roughly break-even to
slightly faster across the two runs; at 192 and 256 rows the final run saves
7.7% and 17.0%. A 128-row continuation into a 2K context is already about
2.06× faster. Both kernels produce identical partials on these cases;
the cutoff addresses small-grid efficiency, not numerical differences.

## Kernel measurements

CUDA-event timings on idle hardware, alternating baseline/candidate order
across six trials. Inputs are synthetic BF16 values with paged,
request-specific sparse lists, ragged counts and padding rows. Every pair
checks all FP32 maximum, denominator and value partials bit for bit.

| Geometry | Rows / context | Baseline | Candidate | Median paired speedup |
|---|---:|---:|---:|---:|
| TP2: 12 query heads, 1 KV head | 128 / 2K | 1.835 ms | 0.889 ms | 2.06× |
| TP2 | 256 / 8K | 3.935 ms | 1.815 ms | 2.16× |
| TP2 | 2K / 2K | 15.669 ms | 7.713 ms | 2.03× |
| TP2 | 2K / 32K | 31.233 ms | 15.159 ms | 2.06× |
| TP4: 6 query heads, 1 KV head | 256 / 8K | 1.953 ms | 1.746 ms | 1.12× |
| TP1: 24 query heads, 2 KV heads | 256 / 8K | 7.836 ms | 3.674 ms | 2.13× |

TP4 already fits all six query heads in one old block, so it benefits from
cooperative probability calculation without the wider sharing improvement.
TP1 and TP4 entries are kernel measurements; service measurements below
use TP2.

## Cold service prefill

Qwen3.8-Flash-Next-FP8, two ranks, four slots, MTP depth 1, resident n-gram
tables, BF16 KV capacity 262,144, 1.5 GiB prefix cache and 128 sampling
candidates. Each configuration starts a fresh world. Clocks are not locked.
The probe uses the same dataset, seed, tags and thinking-disabled setting,
with three cold prompts per size and one generated token.

Median engine prefill time, with the default monolithic admission:

| Requested length | Baseline | Candidate | Time reduction |
|---|---:|---:|---:|
| 2K | 1.408 s | 1.299 s | 7.75% |
| 8K | 5.921 s | 5.108 s | 13.73% |
| 32K | 24.738 s | 21.168 s | 14.43% |

Actual prompt lengths are 2,022–2,049, 7,999–8,215 and 32,282–32,344 tokens.
All nine paired samples have identical prompt hashes, token counts, usage,
output and finish reasons, and zero attached cache tokens.

With a fixed 256-token continuation budget, the same cold prompts give:

| Requested length | Baseline | Candidate | Time reduction |
|---|---:|---:|---:|
| 2K | 2.361 s | 2.259 s | 4.30% |
| 8K | 9.765 s | 9.004 s | 7.79% |
| 32K | 40.524 s | 36.983 s | 8.74% |

All nine budgeted pairs also retain identical outputs, usage and zero cache
attachment. The absolute savings resemble the unbudgeted results; the
remaining small-chunk expert cost reduces the percentage improvement.

## Mixed workloads and service profile

In three trials, a 25,632-token prompt arrives while a separate 1,024-token
completion is streaming. The prefill budget remains 256 tokens.

| Median across trials | Baseline | Candidate |
|---|---:|---:|
| Long-request wall time | 34.102 s | 31.281 s |
| Largest client update gap | 353.45 ms | 331.01 ms |

That is **8.27% less long-request time** and **6.35% smaller largest update
gaps**. Both requests' complete transcripts, token usage, prompt hashes and
finish reasons match in every pair. Update gaps measure delivery to the
client, not individual GPU token latency.

Nsight Systems separately profiles the same 8,281-token cold request on
rank 0, with monolithic admission:

| Profile measure | Baseline | Candidate |
|---|---:|---:|
| Request kernel window | 6.125 s | 5.287 s |
| Kernel launches | 15,578 | 15,578 |
| QSA attention partials, all paths | 1.517 s | 0.702 s |
| Routed-expert matrix multiplies | 1.405 s | 1.404 s |

The candidate's attention total includes 0.684 s in the new kernel and
0.019 s in the retained short-row kernel. The 0.814 s reduction in attention
accounts for about 97% of the 0.838 s reduction in the profiled request
window. These instrumented results explain the uninstrumented service gain;
they are not substituted into the cold-service table.

## C1 checks

Two independent deployment pairs run five prompt classes, three repetitions
per class, 320 generated tokens per request. The first pair runs baseline
then candidate; the second reverses that order. All **30 paired responses**
retain identical transcripts, usage, cached-token counts and speculative
passes. The C1 prompts have 37–73 tokens and retain the original kernel.

Equal-class averages of the five median-rate changes in each pair:

| Timing scope | First pair | Reversed pair |
|---|---:|---:|
| Full-request throughput | −0.231% | +0.002% |
| Historical output-span throughput | −0.154% | +0.083% |

The historical output-span rate excludes initial TTFT but includes the
first token in its numerator; it is not engine decode throughput. Across
the pooled six samples per class, full-request median changes range from
−0.046% to +0.035% for prose/code/JSON/math. Chat is −0.561%, or **44.3 ms
more per 320 tokens**, and is retained explicitly. Chat's output-span
changes reverse direction between pairs (−0.271%, +0.281%).

The additional `serve_c1_probe.py` brackets each request with existing
scheduler counters. It verifies that the counters describe exactly that
request and reports time inside engine step calls to the endpoint's 0.1-ms
precision. Its decode rate excludes the first token produced by prefill.
It separates CPU/GPU/communication time within the engine call from client
delivery timing. Two fresh deployment pairs run five repetitions per class,
again reversing the deployment order for the second pair. All **50 additional
paired responses** retain identical transcripts, usage, cache computation
and decode-step counts.

Each percentage below compares the two deployments' median engine rates.
The last column averages the two paired median time differences; positive
means slower. Each request produces 319 decoded tokens plus its prefill token.

| Class | First pair rate change | Reversed pair | Mean time change / 319 decoded tokens |
|---|---:|---:|---:|
| Prose | −0.282% | +0.137% | +5.2 ms |
| Code | −0.329% | −0.053% | +12.6 ms |
| JSON | −0.085% | +0.013% | +2.3 ms |
| Math | −0.125% | −0.127% | +8.6 ms |
| Chat | +0.181% | +0.106% | −11.1 ms |

Equal-class averages are **−0.128%** and **+0.015%**, or **−0.056%** across
the two pairs. No C1 speedup is claimed. Deployment variation is visible:
the baseline prose median itself moves from 7,135.6 to 7,172.4 ms, and the
second candidate's first two prose samples take 7,413.6 and 7,308.1 ms before
settling near 7,160 ms. Those samples remain in the record.

These measurements do **not** establish exact C1 equality or rule out a
small class-specific cost. Code and math's negative engine averages remain
explicit, even though the changed prefill kernel does not run in these
requests or the four-token graph warm-up. Unchanged kernel bodies establish
the absence of added decode arithmetic; they do not by themselves prove
identical wall-clock performance. No tolerances or dispatch rules were
changed in response to the C1 timing observations.

## Validation

The full CI build and 19 selected suites passed, covering QSA, Qwen
forward/decode/MTP, TP, graphs, slot reuse, continuation, the scheduler and
host checks. The new kernel also passes the existing attention reference
gates without changing their tolerances. Exact partial comparisons cover
six head/KV geometries, three split counts, empty and ragged lists, permuted
pages from two requests, padded query strides, high-dynamic-range queries
and CUDA graph replay. Memcheck, racecheck and initcheck report zero errors
or hazards.

A separate 160-token whole-model reference run exercises the new target
and MTP prefill dispatch. Final-hidden relative L2 error is 0.308% for the
target and 0.341% for MTP, with zero hard top-1 mismatches. The reference's
existing near-tie and routing gates pass unchanged.

All **1,234 existing CUDA kernel text sections are byte-identical** in the
final server. One prefill kernel is added in a separate translation unit.

Quality is unchanged at **59/60 GSM8K** and **30/30 extraction**, with no
truncated responses. All 90 paired responses have identical text, completion
and reasoning token counts, finish reasons and grades. These runs use
temperature zero, reasoning effort low, the default thinking setting,
2,048 maximum output tokens and concurrency four.

The raw record includes all four client-timing deployments, four additional
engine-timing deployments, both prefill-budget comparisons, the profiles,
quality items and per-rank operation hashes.

Every measured world stopped with matching operation streams on its ranks.
The original four-rank GLM-5.3-Flash deployment was restored; all four running
executables match its original SHA-256
`49c0c9dc9fbde3cc04138292cf7059b6cc2e30e1cdb2dbe08aa5c51a39fa8259`.
The model endpoint is healthy and the restored scheduler is idle.

## Reproduction

On idle GPU hardware:

```bash
cmake --preset ci
cmake --build build-ci -j4 --target qsa_prefill_bench qsa_test
build-ci/qsa_prefill_bench --rows 256 --context 8192
build-ci/qsa_prefill_bench --rows 2048 --context 32768
build-ci/qsa_prefill_bench --rows 256 --context 8192 --heads 6 --graph
build-ci/qsa_test
```

For the service comparison, preserve separate baseline and candidate
binaries, use `deploy/cluster_qwen-3.8-flash-next_fp8_w2.json`, and run:

```bash
python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \
  --repeat 3 --seed 7 --tag qsa-prefill-2026-09-15 --no-think \
  --json-out cold.json
python3 scripts/serve_load.py 127.0.0.1 18080 --classes all \
  --concurrency 1 --max-tokens 320 --repeat 3 --json-out c1.json
python3 scripts/serve_c1_probe.py 127.0.0.1 18080 \
  --repeat 5 --json-out c1-engine.json
```

Restart each world and reverse baseline/candidate order for the second C1
pair. Keep numerical agreement, service throughput and client-observed
output timing as separate checks.

For the budgeted comparison, launch with `--prefill-budget-tokens 256` and
repeat the cold probe. Run `serve_prefill_interference.py` with
`--prompt-words 24000 --decode-tokens 1024 --tag qsa-prefill-mixed-r0`, then
repeat with tags ending in `r1` and `r2`. Use
`fabric_qwen_profile.sh CONFIG OUT --bin BINARY --prefill 8192` for the
separate service profile.

Grouped continuation remains the next Qwen task. It must demonstrate useful
expert-weight reuse across requests while retaining the same decode-pause
budget. This change does not alter grouping or scheduling policy.
