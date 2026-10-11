# PR 95: closing the remaining prefill gaps

The full decode campaign passed all 60 cells, but its TP2 32K prefill
median was 0.49% slower than the stronger FP8 bracket. Sixteen-warp
scheduling improved the same-format 8K/32K trial by 0.77%/0.82%; the next
FP8 comparison still missed those two buckets by 0.20%/0.25%. Both failed
records remain available: [full campaign](decode-wide-final-matched-audit.json)
and [first retest](prefill16-final-audit.json).

## Exact matched traces

The follow-up captured both TP2 ranks, both formats, and the original
512/2K/8K/32K prompts with the same f5a517c1 executable. Prompt hashes,
actual token counts and cold-cache counters match. An earlier exploratory
32K capture had a different partial tail, so it was not substituted for
these requests. The [analysis](prefill-tail-profile-analysis.json) retains
all chunk rows, per-kernel counts/times and both ranks. Request bursts are
split at embedding-gather launches; windows include the completion token
and speculative tail. Activity duration is not a measure of SM utilization.

At 32K, each request contains eight full 4096-row target chunks, followed
by 332/228/310 target rows and small verification work. The partial tails
were already faster under NVFP4. Full chunks were nearly tied, focusing
optimization on the large GEMMs rather than the tail dispatch.

Rank-0 mean activity over the eight full chunks, three requests, before
fusion (kernel families overlap model-layer roles):

| Kernel family | FP8 ms | NVFP4 ms |
| --- | ---: | ---: |
| FP4 weight GEMM | 0.0 | 6132.2 |
| cuBLAS 128×208 family | 8034.8 | 2752.0 |
| FP8 dequantization | 1632.8 | 741.3 |
| Bulk collectives | 7179.7 | 7206.3 |
| Full attention tile | 2352.3 | 2352.6 |
| Qwen SwiGLU | 469.1 | 472.6 |

See the [phase summary](prefill-tail-phase-summary.json). These are
instrumented timing observations, not the uninstrumented serving verdict.

## Large-GEMM hardware counters

Nsight Compute application replay used synthetic independent inputs at
TP2 M=4096, with clock/cache manipulation disabled. Gate/down used 127
registers and 96 KiB dynamic shared memory, limiting residency to one CTA
per SM with sixteen warps. Math-pipeline throttle was 36.9%/38.6%, short
scoreboard 16.4%/16.2%, barrier 8.2%/8.8%, and long scoreboard only
2.6%/3.3%. Reported SM throughput was 58.2%/66.4% and L2 throughput
30.9%/34.6%. This points toward instruction issue and computation as major
limits in these large products, rather than predominantly long-latency
weight loads. Counter collection perturbs duration; see the
[counter record](prefill-large-ncu-summary.json).

The exact FP4 path expands coefficient/scale products and performs BF16
tensor operations with FP32 accumulation. It does not quantize activations
to FP4. Smaller checkpoint weights therefore do not halve prefill arithmetic.

## Retained fusion

Commit `510542a4` combines gate/up and SwiGLU. A 256×128×64 physical tile
contains 64 gate columns and 64 up columns. Each of sixteen warps computes
16 matching columns from both projections, so its epilogue can consume
both results directly. This removes intermediate global gate/up matrices,
a separate activation launch, and the two block barriers needed by an
earlier cross-warp exchange prototype.

Every output preserves its ascending-k16 FP32 accumulation, one division
by its own global scale, the two original BF16 projection roundings, and
the original expf/div/mul SwiGLU sequence. Dispatch starts at 1024 rows.
The short/decode path and FP8 model path retain their existing behavior.

Twenty-seven distinct-weight microbenchmark samples were bitwise equal.
The same-format TP2 serving bracket measured:

| Approximate prompt length | Baseline A ms | Candidate ms | Baseline B ms | Reduction against faster baseline |
| --- | ---: | ---: | ---: | ---: |
| 512 | 497.4 | 502.6 | 505.3 | -1.045% |
| 2048 | 1567.5 | 1558.2 | 1594.3 | +0.593% |
| 8192 | 5582.2 | 5529.0 | 5641.0 | +0.953% |
| 32768 | 23950.8 | 23792.3 | 24192.7 | +0.662% |

The unchanged 512-row control shows drift. The changed 2K/8K/32K buckets
all improve against the faster bracket. The selected kernel uses 128
registers and a 16-byte stack with 12-byte spill loads/stores; the final
[disassembly record](prefill-pair-final-spills.json) retains the six local-memory
instructions. It was retained on serving evidence. Final independent
FP8/NVFP4/FP8 comparisons cover every world and length; see the
[combined audit](release-combined-audit.json) and [serving record](README.md).

## Numerical and memory checks

The new test compares against the existing two GEMMs followed by the
actual Qwen SwiGLU function. It checks every output bit and guard across
independent gate/up weights, E4M3 scales and globals, partial M/N tiles,
odd K-step counts, scalar/packed stores, all 65,536 BF16 input patterns,
and actual TP1/2/4 shapes. All 28 affected tests pass. Memcheck/racecheck
instrument seven layout/pipeline launches with zero errors or hazards;
the complete executable also finishes all larger cases.

Real-checkpoint scoring preserves every full-vocabulary logit hash,
log-probability and top-1/2 field at 19,456 positions, plus an independent
4,096-position process repeat. See the [numerical audit](numerical-prefill-pair-comparison.json).
The final serving audit also compares short transcripts, cold one-token
outputs and longer-context responses against the earlier decode campaign.
All non-prefill compiled CUDA instructions are compared separately in the
[identity audit](prefill-pair-retained-kernel-audit.json).

## Alternatives retained in the record

- Medium-row 32/64/96/128-row decode-derived tiles: all 288 samples matched
  bits, but no consistent useful gain at the needed partial-row shapes.
- Integer FP4/BF16 conversions, instruction-order variants, guarded
  reciprocal epilogues and K32 tile aspect ratios: insufficient gain or slower.
- Wider conversion distribution and other sixteen-warp layouts: slower.
- Up-only SwiGLU fusion: 2.17–3.05% micro gains, but 8K/32K serving was
  0.47%/0.57% slower than the faster baseline; rejected.
- An exact 65,536-entry BF16-input SiLU lookup table: bitwise, but slower.
- Cross-warp gate/up fusion: 0.33%/0.38% serving gains at 8K/32K, superseded
  by the retained same-warp form.
- An 8K idle-prefill budget experiment failed at startup because Qwen35
  does not implement resumable scheduler prefill. No timing result was
  collected, and the private model-capacity change was reverted.

The [experiment summary](prefill-experiments-summary.json), raw micro logs,
trial records and probe sources retain the unsuccessful alternatives.
The [raw-profile manifest](prefill-raw-profile-manifest.json) identifies local traces
and profiler versions. No failed numerical or performance result was relabeled
as a pass.
