# GLM-5.3-Flash performance investigation, 2026-09-16

The retained change makes DSA prefill query tiles fit the current visible
context inside the existing dot-product workspace. It changes neither weight
nor KV precision, scratch allocation, prefill chunk boundaries, top-k selection
arithmetic, nor the decode path.

## Configuration and method

- Baseline: `cdc3bf988177` on four GB10 DGX Sparks, CUDA 13, tensor parallelism
  over the existing RoCE fabric.
- Checkpoint: `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`, revision
  `17b8d4d83e33e836e9549d7e1a8d25b5117203c1`.
- Four request slots, MTP depth one, captured decode graphs, BF16 KV,
  786,432-token pool and 8 GiB prefix cache. The model's native BF16 layers,
  FP8 projections/shared experts/draft and NVFP4 routed experts are unchanged.
- Decode: the repository's five `serve_load` prompt classes, greedy, 256 output
  tokens, three repeats at concurrency one and four. Engine decode excludes
  admission/prefill. Request-wall rates include them. The client requests
  thinking off, but this checkpoint rejects that template option, so its
  fallback uses the checkpoint's default thinking mode in every trial.
- Prefill: three cold prompts at each target length, seed 916, tag
  `glm-flash-perf`, prepared GSM8K prose. Prompts, hashes, actual token counts,
  uncached counts, engine timing and client TTFT are retained in the JSON.
  A reverse-order baseline checks that the improvement survives another
  service restart. All measurements are serial on otherwise idle hardware.

## Current service measurements

Cold prefill, median of three matched prompts. The baseline below is the
fresh reverse-order control, which reproduces the initial baseline.

| Target | Actual prompt tokens | Baseline | Final | Less prefill time | Speedup |
|---|---|---:|---:|---:|---:|
| ~2K | 1,944–1,989 | 1.871 s | 1.465 s | 21.7% | 1.28× |
| ~8K | 7,723–7,811 | 10.266 s | 6.005 s | 41.5% | 1.71× |
| ~32K | 31,291–31,306 | 82.613 s | 27.753 s | 66.4% | 2.98× |

Final median prefill rates are 1,333, 1,301 and 1,128 prompt tokens/s.
All samples compute the entire prompt with zero cached tokens.

Decode remains essentially flat. Every C1 engine median differs from baseline
by less than 0.14%, and every C4 engine median by less than 0.86%. The final
rates by class are:

| Class | C1 engine tok/s | C1 request-wall tok/s | C4 engine tok/s | C4 request-wall tok/s |
|---|---:|---:|---:|---:|
| prose | 55.28 | 54.42 | 100.62 | 97.65 |
| code | 57.44 | 56.56 | 103.23 | 100.63 |
| json | 57.84 | 56.44 | 105.13 | 102.19 |
| math | 55.90 | 54.98 | 107.03 | 103.98 |
| chat | 50.59 | 49.90 | 100.66 | 96.38 |

## What the profiles show

The steady decode intervals contain 42 routed expert layers per replay. At
concurrency one, the baseline averages 34.44 ms per step, including 7.52 ms in
gate/up projections and 3.61 ms in down projections. At concurrency four,
the eight-row replay averages 70.31 ms, with 24.12 ms in gate/up and 11.80 ms
in down. Native BF16 projections remain a substantial memory-bandwidth cost.
The analysis selects complete intervals with the expected row geometry and
excludes admission gaps. Overlapping side-stream kernels are not additive
wall time.

Prefill exposed the larger opportunity. The DSA dot workspace was allocated
for the maximum context, and that worst-case query-tile width was reused even
for short prompts. At shorter contexts most of each allocation was unused,
while the host launched many tiny GEMMs and selection grids.

The matched Nsight workload contains the calibration request and an
8,301-token cold request:

| Selection work | Baseline | Context-aware query tiles |
|---|---:|---:|
| Kernel launches | 52,954 | 408 |
| Summed selection kernel time | 4,683.55 ms | 187.25 ms |

The new width uses the allocated query-row × pool capacity divided by the
current padded pool count, bounded by the model's maximum forward rows.
Longer contexts automatically shrink the query tile. Even a dot budget below
one row keeps the existing minimum one-row allocation and stays within it.

After this change, the same profiled workload spends about 1.71 seconds in
routed expert tensor-core kernels, 1.06 seconds in bulk collectives, 0.67
seconds in listed attention and 0.56 seconds in KDA recurrence. Those are the
next substantial prefill targets. The decode profiles and unsuccessful
weight-sharing screens provide no evidence for a decode speedup from this
patch.

## Validation

All nine CTest checks passed: `dsa_test`, `dsa_dump_test`, `glm_tp_test`,
`glm_moe_test`, `glm_dsa_tp_test`, `glm_dsa_engine_test`, `serve_test`,
`fabric_serve_test` and `scheduler_test`. The new DSA regression exercises
both indexer forms, partial tiles, a cache larger than the prompt, and dot
budgets from one byte to 8 MiB. Selected indices, counts and output bits
match between the budget choices. Compute Sanitizer `memcheck` and `initcheck`
both report zero errors for that test.

All 15 C1 responses reproduce the baseline transcripts. Cold-prefill probes
match prompt hashes, usage, first-token text and finish reasons at all nine
samples. Separate prompts of 2,066, 7,568 and 30,182 tokens each generate 128
tokens with identical response choices and usage. Both the baseline and final
service runs shut down cleanly with identical operation streams across all
four ranks. [raw/comparison.json](raw/comparison.json) contains the matched
summary, generated by `compare.py` from the checked-in records.

The startup memory plan remains 92.50 GiB per rank: 82.53 GiB device and
9.97 GiB pinned, plus the existing 4 GiB headroom. No experimental switches
or additional allocations remain in the implementation.

## Experiments not retained

| Experiment | Matched result | Decision |
|---|---|---|
| Pair repeated expert down projections across verification rows | Code engine decode: C1 57.47 → 57.38 tok/s, C4 103.55 → 102.68 tok/s | Removed |
| Pair repeated expert gate/up projections | Code engine decode: C1 57.47 → 57.19 tok/s, C4 103.55 → 103.70 tok/s | Removed, no convincing gain |
| Bound NVFP4 prefill grids to four query tiles, stride over hot experts | With adaptive DSA: ~8K median 6.000 → 5.986 s, ~2K 1.456 → 1.458 s | Removed, within noise |

The paired projections preserved the existing per-row FMA and reduction
order and passed bitwise operator checks, including odd expert runs and
noncontiguous output slots. The bounded grid passed bitwise comparisons for
empty and 513-row segments, indexed/unaligned activations and FP32 down
outputs. Correctness alone did not justify keeping their extra complexity.
Baseline C4 text already varies across repetitions as requests enter and leave
different batch shapes. These screens compare equal prompt hashes and output
lengths, and require exact C1 transcripts, rather than claiming C4 text identity.

## Measurements awaiting a rerun

Other DSA configurations were not rebenchmarked in this campaign. Their
previous prefill summaries are preserved here rather than presented as
measurements of the updated implementation:

| Configuration | ~2K / ~8K / ~32K, seconds | Previous measurement |
|---|---|---|
| Flash FP8, four Sparks | 2.211 / 9.165 / 58.555 | 2026-09-10 |
| Flash hybrid, two Sparks | 2.724 / 11.881 / unmeasured | 2026-09-12 |
| Full GLM int4/int8, four Sparks | 6.111 / 40.838 / 350.069 | 2026-09-15 |

## Reproduction

Build `dgpp_serve_app`, then start the four-node hybrid recipe with
`scripts/dgpp-cluster`. Run each binary on a freshly started service so the
prefill probes cannot attach to an earlier trial's prefix cache.

```bash
python3 benchmarks/results/2026-09-16-dsv41-perf/timed_load.py \
  192.168.88.11 18080 --concurrency 1,4 --classes all \
  --max-tokens 256 --repeat 3 --warm 1 --json-out load.json
python3 scripts/serve_prefill_probe.py 192.168.88.11 18080 \
  2048 8192 32768 --repeat 3 --seed 916 --tag glm-flash-perf \
  --json-out prefill.json
python3 benchmarks/results/2026-09-16-glm-flash-perf/long_transcripts.py \
  192.168.88.11 18080 long-transcripts.json
```

The experimental environment switches used during the screens were removed.
The retained behavior is automatic. Large Nsight reports and binaries remain
under `build-ci/glm-flash-perf-2026-09-16/`; compact profiles and raw service
results are checked in under `raw/`.
