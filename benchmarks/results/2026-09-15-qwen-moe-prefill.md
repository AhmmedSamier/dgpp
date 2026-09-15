# Qwen expert prefill: three rejected kernel experiments

Date: 2026-09-15. Baseline source: `b36b97e`.

**No production kernel change was accepted.** Smaller tiles, a compact launch
grid and persistent blocks all preserved numerical results, but none established
a useful service improvement without losing on other routing patterns. The
experiments now live only in `moe_prefill_bench`; the serving code is unchanged.

[Raw observations and provenance](2026-09-15-qwen-moe-prefill.json).

## What was tested

The [previous prefill profile](2026-09-15-qwen-idle-prefill.md) put most of the
256-token chunk penalty in routed-expert matrix multiplies. This investigation
tested whether their tile geometry and empty blocks explained that penalty.

Qwen3.8-Flash-Next-FP8 has 512 experts and selects ten per token. At TP2,
the expert gate/up matrices have shape 320 × 2,560, and down has shape
2,560 × 320. The 64 × 128 production tile uses four 32-deep pipeline stages.
The launcher receives the prompt length as its maximum expert row count;
it cannot use a CPU-computed maximum without synchronizing the GPU.

Three prototypes kept the same FP8-to-BF16 conversion, ascending-k16 MMA
accumulation and final rounding:

1. **16 × 64 tiles:** reduce shared memory and walk each expert's rows inside
   its block. This saves empty row-tile launches but rereads weights more often
   for heavily used experts.
2. **Compact down projection:** retain 64-row reuse and walk row tiles inside
   each expert/column block. Apply it to the short down-projection reduction.
3. **Persistent down projection:** use two blocks per SM, each walking a fixed
   stride of expert/column tasks. Skip empty experts without launching a block
   for each one. Keep the existing path for prompts of 64 tokens or fewer.

The final benchmark contains these experimental policies in a separate CUDA
translation unit. They are linked only into the benchmark executable. The
compact and persistent variants affect down projections at 65–256 tokens;
their gate timings are unchanged-path controls. Each projection prints whether
the experimental kernel was selected.

## Routing changes the result

A diagnostic deployment of the baseline captured 147 full 256-token layer
chunks. These logs were collected separately from performance measurements.

| Per-layer statistic | Median |
|---|---:|
| Nonempty experts | 264 |
| Experts with at most 16 rows | 228 |
| Rows in the busiest expert | 180 |
| Number of live 64-row tiles | 270 |
| Number of live 16-row tiles | 341 |

The largest observed expert had all 256 rows. The average of five assignments
per expert therefore gives a poor picture of the busiest experts. Smaller
tiles create extra weight reads for those experts while many other experts
remain lightly used.

The benchmark tests uniform random routing, concentrated routing to ten
experts, a skewed synthetic distribution, and recorded segment-size patterns.
Recorded cases reproduce work geometry using synthetic activations and
weights; they do not replay the model's tensor values. Every timed case
alternates baseline/candidate order across six trials and requires bitwise
equality of the complete output after each pair.

Prototype kernel time changes, positive meaning slower:

| Prototype | TP2 uniform: gate / down | Concentrated routing: gate / down | Recorded patterns: down |
|---|---:|---:|---:|
| 16 × 64 tiles | +0.1% / +3.8% | +202.1% / +21.3% | Not used to select this prototype |
| Compact 64-row down | Control / +1.9% | Control / −26.0% | Evaluated through the service below |
| Persistent 64-row down | Control / +5.7% | Control / about −21% | +4.3% to +6.4% |

The raw record retains these original prototype runs and the subsequent
checks of the standalone benchmark. The standalone versions share a template
for their common arithmetic, so their timing observations are recorded
separately from the original implementations.

Final standalone down-projection checks (median paired time changes):

| Variant | Uniform | Concentrated routing | Recorded p10 / p50 / p90 |
|---|---:|---:|---:|
| Small | +3.49% | +24.55% | — / +3.98% / — |
| Compact | +0.52% | −18.78% | −1.20% / +0.24% / −0.25% |
| Persistent | +6.40% | −19.71% | +4.80% / +7.33% / +4.34% |

The small variant's recorded p50 gate is also 34.06% slower. The compact
variant's small mixed movements on recorded routes provide no stronger
reason to promote it. The concentrated-routing improvements are useful
experimental observations, not measured service wins.

## Service result: no convincing improvement

The service configuration matches the previous TP2 investigation: four slots,
MTP depth 1, BF16 KV capacity 262,144, 1.5 GiB prefix cache, resident n-gram
tables and 128 sampling candidates. Each configuration starts a fresh world.
Clocks are not locked. Cold probes use the same seed, tag, data and thinking
mode, with three prompts per length and one generated token.

Median engine prefill seconds:

| Prompt size | Unbudgeted baseline | Fixed 256 baseline | Compact prototype | Change vs fixed 256 |
|---|---:|---:|---:|---:|
| About 2K | 1.404 | 2.362 | 2.337 | −1.05% |
| About 8K | 5.883 | 9.719 | 9.668 | −0.52% |
| About 32K | 24.619 | 40.281 | 40.216 | −0.16% |

Actual prompt lengths are 2,022–2,049, 7,999–8,215 and 32,282–32,344 tokens.
All nine matched cold samples preserve prompt hashes, token counts, usage,
outputs and finish reasons, with zero attached cache tokens.

In three mixed-workload trials, a 25,630-token prompt arrives while a separate
1,024-token completion is streaming. Median long-request wall time moves
from **33.805 to 33.622 seconds** (−0.54%). The median of the largest client
update gaps moves from **353.4 to 356.0 ms**. All paired transcripts, output
lengths, usage and finish reasons match. These small movements do not justify
a production change, especially given the kernel regressions on other routes.

The baseline and prototype both stop with operation-stream MD5
`6715fa191583cc14cbc10a27e493971e` on each of their two ranks.

## Memory traffic explains the limited opportunity

Nsight Compute on the recorded median-routing gate case reports
6,812,986 L2 sectors filled from system memory: **218,015,552 bytes** using
32-byte sectors. One read of each selected expert's FP8 gate weights requires
**216,268,800 bytes** (264 × 320 × 2,560), before activations and scales.
The measured fills are only 0.81% above that weight payload. The same routing
case's down projection fills 6,826,053 sectors, **218,433,696 bytes**, for the
same 216,268,800-byte selected-weight payload (1.00% above it).

This is a system-memory-to-L2 fill counter, not a DRAM-bandwidth measurement.
GB10 reports the requested `dram__bytes_*` metrics as unavailable. Total L2
requested bytes are higher, 682,674,784, and must not be confused with external
memory reads. Profiling changes execution time, so its instrumented duration
is not substituted into the service table.

For this representative gate case, there is little excess external weight
traffic to remove. Small chunks repeatedly need much of the expert weight
set; changing tile size cannot remove that requirement. This finding does
not establish the same limit for every layer or workload.

## C1 preservation and validation

All production source edits from the prototypes were removed. The final
server's CUDA fatbin is **byte-identical** to the baseline: 50,737,648 bytes,
SHA-256 `d5438471d34573e07c99108965a4e1ba29f30e81718246f4efddfb8391b32feb`.
The CMake change adds only the standalone benchmark target. No experimental
kernel is linked into the server, and no C1 speedup is claimed.

The prototypes passed bitwise checks against the existing kernels, including
TP scale grids, empty/ragged/hot expert segments, mapped and unaligned inputs,
output padding, and CUDA graph capture/replay. The initial prototype also
passed the Qwen and GLM MoE suites. Fresh baseline C1 anchors are retained in
the raw record; rejected prototypes were not promoted to a final C1 campaign.

The full CI build and four CTest suites passed: `qwen_moe_test`,
`glm_moe_test`, `unit_tests` and `script_reports_test`. The final standalone
benchmark passed 13 cases, each with six paired trials for both projections.
All three variants passed Compute Sanitizer memcheck and racecheck with zero
errors or hazards. Sanitizer shapes include ragged dimensions and uneven
expert lengths; their instrumented times are excluded from performance data.

The original four-rank GLM-5.3-Flash deployment was restored. Verification
at 17:42 UTC checked each running `/proc/PID/exe` against its original
SHA-256, `49c0c9dc9fbde3cc04138292cf7059b6cc2e30e1cdb2dbe08aa5c51a39fa8259`.
The model endpoint is healthy, with zero active or queued requests and no
engine failure. The raw record contains all four executable hashes and the
restored service metrics.

## Reproduce the benchmark

On idle GPU hardware:

```bash
cmake --preset ci
cmake --build build-ci -j 4 --target moe_prefill_bench
build-ci/moe_prefill_bench --variant small --distribution uniform
build-ci/moe_prefill_bench --variant compact --distribution hot
build-ci/moe_prefill_bench --variant persistent \
  --segments-file benchmarks/results/2026-09-15-qwen-moe-routes-p50.txt
```

The recorded [p10](2026-09-15-qwen-moe-routes-p10.txt),
[p50](2026-09-15-qwen-moe-routes-p50.txt) and
[p90](2026-09-15-qwen-moe-routes-p90.txt) cases are selected by nonempty-expert
count. Their counts are 215, 264 and 303; their largest expert segments contain
194, 166 and 111 rows. Use `--inter 160 --scale-block 32` for the TP4 geometry.
`--tokens`, `--experts`, `--top-k`, `--hidden` and `--iters` configure the
other dimensions. Each result identifies whether the candidate was selected.

## Next work

Do not prioritize more Qwen tile-size sweeps without evidence of avoidable
traffic or instruction cost on the intended routes. Full GLM's packed int4/int8
prefill still uses grouped GEMV and has a clearer missing compute kernel.
Qwen QSA attention is another independently profiled prefill cost. Grouped
continuation remains useful to investigate for concurrent arrivals, but its
evaluation must measure expert weight reuse and decode pauses together.
