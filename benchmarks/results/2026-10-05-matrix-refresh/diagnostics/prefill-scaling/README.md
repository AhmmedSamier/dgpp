# DSA long-context prefill scaling

The old prefill selector spent most of its time scanning and sorting with
too few GPU blocks. A fixed dot-product buffer shrank the query tile as
context grew. For the four-node GLM Flash configuration with a 768K KV
pool, the tile held 12 queries near 128K, six near 256K, and three near
512K. Each query used one block to scan the visible pools and repeatedly
bitonic-sort tiles. Its warp-per-pool scoring also read across head rows
with a large stride.

The [baseline GPU profile](baseline-kernels.csv) attributes 97.9% of kernel
time to that selector: 10.230 seconds across 683 launches for one DSA
layer's final 2,048-token chunk at 512K context. The full-model baseline
recorded 262.963 / 1,527.734 / 11,384.873 seconds of cold prefill at
128K / 256K / 512K, with zero process and cgroup swap.

| Path | Scope |
|---|---|
| GLM-5.3-Flash | Profiled directly; FP8 and NVFP4/FP8 use the shared DSA layer |
| Full GLM-5.3 | Uses the same DSA prefill selector, including int4/int8 weights |
| Qwen and MiMo | Different attention paths; this fix does not apply |
| DeepSeek V4 / V4.1 | Separate CSA2 prefill selectors; comparable scaling has not been established |
| Decode | Separate selector; unchanged by this fix |

## Change

1. Score adjacent pools in adjacent threads, preserving the original
   multiplication and head-summation order.
2. Partition each row's keys across enough blocks to occupy the GPU.
3. Find each partition's exact top-k by radix refinement, then merge
   those sorted selections in a tree. A discarded local candidate cannot
   enter the global top-k. Composite keys retain the lower-index tie rule.

The compact keys and bounded partial lists reuse the decode selector's
scratch allocation where it fits. Smaller allocations grow by a few MiB;
KV capacity, prefix cache and request slots are unchanged. All measurements
run in scopes with `MemorySwapMax=0`; OS swap remains enabled.

## Isolated measurements

GB10, one attention layer with the four-node head slice, 2,048-token
prefill chunk ending at the listed context, 768K KV capacity. Synthetic
history; one warmup and three timed repetitions, without a profiler.
These are layer timings, not full-model speedups.

| Context | Original, ms | Partitioned selector, ms | Speedup |
|---|---:|---:|---:|
| 128K | 699.479 | 121.160 | 5.77× |
| 256K | 2,712.136 | 217.963 | 12.44× |
| 512K | 10,211.204 | 442.360 | 23.08× |

Raw logs: `baseline-{131072,262144,524288}.log` and
`partitioned-{131072,262144,524288}.log`. An intermediate implementation
used one radix-selection block per query and reached 648.963 ms at 512K;
its logs/profile are named `candidate-*`, with source in
[first-candidate.patch](first-candidate.patch). It is superseded by the
partitioned implementation in [source.patch](source.patch).

```sh
systemd-run --user --scope --quiet --property=MemorySwapMax=0 -- \
  build-release/dsa_bench --prefill-only --ctx 524288 --capacity 786432 \
  --chunk 2048 --tp 4 --warmup 1 --iters 3
```

## Validation

The four-node GLM Flash NVFP4/FP8 deployment completed its full-model
rerun with the same 768K BF16 KV pool, four slots and 22 GiB prefix cache.
Both launches passed the zero process/cgroup swap checks, matching-rank
operation-stream checks and clean shutdown. Each cold-prefill value below
is one uncached request; the two following requests reuse its prefix.

| Context | Original cold prefill, s | Fixed cold prefill, s | Speedup |
|---|---:|---:|---:|
| 32K | 24.312 | 21.799 | 1.12× |
| 64K | 66.764 | 48.172 | 1.39× |
| 128K | 262.963 | 114.093 | 2.30× |
| 256K | 1,527.734 | 303.271 | 5.04× |
| 512K | 11,384.873 | 925.281 | 12.30× |

The [comparison record](full-model-comparison.json) includes actual input
lengths and source binaries. The [new raw measurements](../../../2026-10-06-prefill-scaling/raw/glm-flash-hybrid-w4/refresh/default/)
retain all repetitions and validation evidence. All five GLM Flash default
configurations have completed their prefill and context reruns. Full
GLM-5.3 has also completed the 120K BF16 and 208K FP8 configurations; the
additional 256K FP4 attempt exposed a second bottleneck in per-head score
traffic. The [full-GLM follow-up](../../../2026-10-06-prefill-scaling/diagnostics/full-glm-prefill-investigation/README.md)
records its separate fix, FP64 accuracy checks and completed BF16/FP8
reruns. The final FP4 measurements are complete through 256K context, with
zero serving-process swap and no collective-stall warnings. These speedups describe the
768K GLM Flash deployment only. None of the five GLM Flash default launches
logged a collective-stall warning; the [log audit](../../../2026-10-06-prefill-scaling/diagnostics/fixed-flash-collective-observations.json)
records the launch boundaries and per-rank counts.

| Check | Result | Evidence |
|---|---|---|
| DSA suite | 48 tests pass | [Log](partitioned-dsa-suite.log) |
| Exact selection | CPU oracle, random and tied scores, ReLU on/off, both GLM selection sizes, causal tails and long contexts | [Log](partitioned-exact-selection.log) |
| CUDA memory checker | Four prefill-selection tests; zero errors | [Log](partitioned-memcheck.log) |
| CUDA synchronization checker | Partition boundaries, including 190 workspace leaves and 128 partitions; zero errors | [Log](partitioned-synccheck.log) |
| Route-trace regression | Pass | [Log](glm-regression-traces.log) |
| Resumable prefill | Two tests pass | [Log](glm-regression-resumable.log) |
| Grouped prefill | Three tests pass | [Log](glm-regression-grouped.log) |
| Four-rank graph/device picks | Pass | [Log](glm-regression-graph.log) |
| Full-model timing | Five GLM Flash configurations complete, including two 512K sweeps; full GLM 120K BF16 and 208K FP8 rerun after the score-fusion follow-up; 256K FP4 configuration complete | [Campaign](../../../2026-10-06-prefill-scaling/README.md) |

The original campaign was interrupted deliberately and shut down cleanly;
its fifth configuration is incomplete. [Pause record](pause.json).
Affected earlier measurements remain baseline evidence. The new campaign
uses a separate directory and binary manifest to keep the revisions apart.
Its [reuse audit](../../../2026-10-06-prefill-scaling/reused-results.json)
retains only verified GLM Flash inputs of at most 8,191 tokens, which use
the unchanged selector path, with their original zero-swap evidence.
[Source and binary hashes](source-files.json).
