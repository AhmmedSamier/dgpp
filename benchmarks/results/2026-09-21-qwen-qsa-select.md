# Exact QSA selection at long context

Date: 2026-09-21
Issue: [#19](https://github.com/HawkBearPig/dgpp/issues/19)
Baseline: `9fa8e0f1ec491d075d4b20aaae3e3892b8fc435a`.

## Finding and implementation

The [reviewer's comment](https://github.com/HawkBearPig/dgpp/issues/19#issuecomment-5753567010)
already established synthetic correctness at 65,322 compressed pools with
64-token paging and a permuted block table. This change makes that case a
permanent test, adds larger cases and measures the performance question the
review explicitly left open. It does not resolve the separate issue #4
activation investigation.

The old selector dominates the two-row indexer at long context. It sorts
every 2048-key tile in one block per query row, then merges its top keys.
At 131,072 pools, selection alone takes 4.45 ms warm, compared with 0.224 ms
for scoring and about 0.002 ms for an empty graph launch. Eager and graph
measurements are comparable. The dominant cost is neither launch overhead
nor a saturated DRAM scan.

Nsight Compute on the old 65,322-pool selector reports two 256-thread blocks
on 48 SMs, 1.34% memory throughput, 1.34% SM throughput and 0.05% L2
throughput. Fixed-latency execution dependencies account for 34.6% of warp
stall cycles and branch resolution for 31.5%. This supports a bottleneck in
serial sorting/control work on an underfilled grid. Profiling changes clocks
and replays kernels; its 2.81 ms duration is not used in the A/B timings.

For more than 2048 visible pools, the new selector finds the exact top-k
boundary with 10-bit radix histograms. It refines only the boundary prefix
until at most 256 candidates remain, gathers keys below that boundary, and
ranks the boundary candidates. The unique composite key includes the pool
id, so equal scores retain the lower-pool-id tie rule. Existing expansion
sorts the selected pool ids and appends the incomplete pool's tokens.
Rows of at most 2048 pools retain streaming selection.

Score arithmetic, its fixed FP32 reduction order, global workspace sizes,
launch count, captured graph shape, output order and padding are unchanged.
Histograms reuse existing selection scratch. Selection still scans all keys:
this removes expensive sorting rather than promising constant-time context
growth. DSA and shared top-k implementations are unchanged.

## Microbenchmark

`benchmarks/micro/qsa_index_bench.cu` uses four BF16 indexer heads of width
128, production 64-token paging, shuffled physical blocks, top-512 selection,
and seed 20260921. Rows share a request cache and have adjacent positions.
Every visible score key and selected token/count/padding matches the host
oracle before timing; repeated launches must preserve the selected output.

The Spark has 48 SMs and 24 MiB L2. Cold samples write a separate 96 MiB
buffer before the CUDA-event interval. Warm samples repeat the same inputs;
the 131K-pool index cache is 32 MiB and cannot all fit in L2. Logical input
GB/s is not actual DRAM bandwidth, particularly with shared cache rows.

Both executables use the release preset. The primary comparison runs old,
new, new, old, each with five warmups and 30 samples per mode/stage/size.
The baseline links the new benchmark driver and host oracle against the
unchanged `9fa8e0f` kernels; copy only the benchmark source and its CMake
target when reproducing that baseline checkout.
The table averages the two run medians, in microseconds, for two query rows
under CUDA graphs. No other GPU work or builds run alongside the timing.

| Pools | Cache | Score old / new | Select old / new | Combined old / new | Combined speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16,384 | warm | 20.6 / 20.8 | 565.6 / 56.9 | 585.2 / 75.8 | 7.72× |
| 16,384 | cold | 52.9 / 53.9 | 585.4 / 75.5 | 616.2 / 109.0 | 5.65× |
| 65,322 | warm | 60.8 / 60.6 | 2227.8 / 168.8 | 2287.4 / 226.9 | 10.08× |
| 65,322 | cold | 163.5 / 153.8 | 2296.0 / 235.2 | 2389.8 / 321.2 | 7.44× |
| 131,072 | warm | 223.6 / 224.2 | 4446.3 / 321.3 | 4769.1 / 643.8 | 7.41× |
| 131,072 | cold | 286.5 / 287.3 | 4593.8 / 462.7 | 4841.4 / 712.3 | 6.80× |

Separate-stage medians need not sum to the combined median because their
cache history and timing intervals differ. The short sweep at 16, 256,
512, 1024 and 2048 pools ranges from -3.0% to +2.8% in combined latency,
with the largest increase 1.9 microseconds. At 16 rows, short cases differ
by at most 0.4%; long cases improve 2.5–4.3×. At 128 rows, short cases
differ by at most 0.1%, and 4096/16384-pool cases improve 1.7×. The 2049-pool
transition and eager execution also pass every oracle check.

Reproduction is in [the benchmark guide](../../docs/benchmarks.md#98-qsa-score-and-selection-microbenchmark).
Raw JSONL, profiler reports, model responses and logs are retained locally
under `artifacts/issue19/`.

## Validation

- Full CI build with warnings as errors and release builds passed.
- All 30 host CTest entries passed in 167.67 seconds.
- All 13 selected Qwen CTest entries passed in 66.00 seconds: fixtures,
  plan, plain/YaRN/full-forward/cross-limit parity, incremental decode, QSA,
  tensor parallelism, graph engine, row independence and BF12 engine.
- QSA passes all eight cases. New tests cover 65,322 and 131,071 pools
  with exact host score keys and selected tokens; 131,073-pool selection;
  equal-score ties, final-tile winners, select-k 8/512/1024, inactive rows,
  all incomplete-tail lengths and changing positions under the same graph.
- Compute Sanitizer memcheck reports zero errors on the long selection gate;
  racecheck reports zero hazards, errors or warnings.

## Two-node model comparison

The model campaign uses `nvidia/Qwen3.8-Flash-Next-NVFP4` and the shipped
two-node YaRN template: factor 2, 524288-token context ceiling, 532480-token
pool, BF16 KV, FP8 dense weights, mapped n-gram table, depth-one MTP, two
available slots and a 1.5 GiB prefix cache. Requests are serial to keep
admission order fixed. Each build starts a fresh world; production is stopped
throughout the GPU and fabric measurements.

Release binary SHA-256 values for this comparison are
`dd40000570c44839849259b60878bc38bc577cd7174a3d550d2b9aa023f0b38e`
(baseline) and
`13c89d6cbe538e35320f80bf9572450d96ae45efeae3dae1403b6eefcab2ea12`
(candidate). Both running candidate ranks were checked through `/proc/PID/exe`.

The fixture uses `qwen_yarn_release_check.filler`, seed 20260921 for a
256-record calibration and seed `20260921 + nominal_length` for each prompt.
Record count is `max(16, floor((nominal_length - 1024) / tokens_per_record))`.
The prefix is `Run issue19 context {nominal_length}.\n`, followed by the
records and `\nWrite a numbered list describing the first 100 parcels and their cities. Continue until you have listed all 100.`
Each request asks for 256 tokens at temperature zero, including reasoning.
This measures exact response parity, not a parcel-retrieval quality score.
Each length runs once cold and twice with the identical cached prompt.

Engine counters are reconciled with each completed response before taking
deltas. Decode latency is `step_ms / decode_steps`; tokens per pass exclude
the first token produced by prefill. Counter polling is outside the timed
client interval and uses observed completion conditions, not a fixed sleep.

The decode table reports the median of the three requests per length. Cold
prefill is one uncached measurement per build and length.

| Actual prompt tokens | Old decode ms/pass | New decode ms/pass | Latency reduction | Cold prefill seconds, old / new |
| ---: | ---: | ---: | ---: | ---: |
| 3,139 | 28.54 | 28.12 | 1.5% | 2.11 / 2.07 |
| 31,670 | 31.31 | 28.04 | 10.4% | 20.74 / 20.32 |
| 129,560 | 43.39 | 29.55 | 31.9% | 98.62 / 93.16 |
| 260,062 | 59.38 | 31.36 | 47.2% | 235.97 / 215.07 |
| 520,742 | 93.52 | 37.06 | 60.4% | 732.40 / 655.18 |

All **15/15 response hashes** match across builds, including reasoning text.
Prompt hashes, complete usage objects, finish reasons, token counts, decode
steps/rows and computed prompt-token counts also match. Cold and cached
responses agree within each build. All requests generate 256 tokens and end
with `length`; these are fixed-budget comparisons, not complete answers.
Tokens per pass remain 1.85–1.92. No accuracy or performance regression was
detected in this matrix; it does not establish quality on untested prompts.

Both builds pass all nine API checks. At clean shutdown, both ranks agree
on the complete operation stream in each world:

- Baseline: `b3600c2530fa87b771d733b726b7905238509264226c350239f43ab121cf9f3d`.
- Candidate: `6a38040f2da7fa9c0ca1503e90c6dce9c517d9f601ca49241dfacd6efac27fb6`.

The five response SHA-256 values below each cover all three repeats on both
builds, in the same order as the context table:

- 3,139 tokens: `2197dc7ca6a386fe13a174f79c199f7814437af9bd179d6d8c4229d5aabddd92`.
- 31,670 tokens: `acc8b95b2dddb515aeabeae30652092344cec0b5cd8d0b2219f7e1be6082a89f`.
- 129,560 tokens: `58929bccf83f60b7b748c1caf891c116777b6fbdf0508f98bdcd35cabd94e754`.
- 260,062 tokens: `2d5f6d1f1396a97842821fbaf7793d042b10abeed3b3d19abcf717feca4be9dd`.
- 520,742 tokens: `e84a798e6bd84d9a0e37fee0e9c699df9fbd696d6e174ba5f9033211b9846efa`.
