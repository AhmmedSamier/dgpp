# DeepSeek V4.1 Flash: inference optimization on four GB10s

This study starts at `f01e947` on September 16, 2026. It profiles the existing
service, isolates communication costs, screens several changes, and validates
only the changes that improve measured work. The native MXFP4/FP8 weights,
KV formats, model, prompts, sampling and bounded-prefill policy are unchanged.
No weights were quantized or recalibrated.

## Configuration and measurement

Four GB10 DGX Sparks, CUDA 13, four-rank RoCE deployment; six request slots,
131,072-token KV pool, DSpark draft depth four, adaptive confidence scheduling,
1.5 GiB prefix cache, temperature zero. The local deployment is
`deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.json`, derived from its committed
example. The head is `edgexpert-409e`.

The original checkout and binary were preserved for the A/B. Experiments used
a separate worktree, `~/workspace/dgpp-dsv41-perf` on the head, with one GPU
world running at a time. The previously running GLM service was stopped before
testing. Builds and GPU tests do not overlap timed service requests.

The final A/B uses the repository's five-class `serve_load.py` corpus, three
repetitions each at concurrency one and six, a 256-token cap, thinking disabled,
and fresh service boots. Some prompts finish before the cap. Rates are medians
of the three per-phase measurements, compared within the same class. They are
not comparable to the different external-recipe corpus in the September 14
benchmark table. `timed_load.py` adds scheduler counter deltas around each idle
phase and rejects unexpected concurrent traffic. **Client wall throughput**
includes admission and prefill; **engine decode throughput** divides generated
tokens after the first token by `step_ms`, excluding prefill. Both include
batch filling and draining; neither pretends every step has six live requests.

## Profile: where the remaining time goes

The original Nsight Systems run measured 115 marker intervals after skipping
20, using six concurrent code requests with a 256-token cap. Mean step wall
time was 150.501 ms, with 147.120 ms GPU busy (97.8%). The window includes the
workload's warmup/draining and some prefill; it is a bottleneck profile, not a
fixed-batch decode microbenchmark.

| Kernel family | ms per marker interval | share of GPU time |
|---|---:|---:|
| Expert gate/up + SwiGLU | 50.620 | 34.4% |
| Expert down | 27.434 | 18.6% |
| Graph all-reduce | 23.130 | 15.7% |
| Other kernels | 45.936 | 31.2% |

The expert kernels remain the largest cost, consistent with the existing
bandwidth-oriented implementation. GPU utilization alone does **not** prove
memory or network line rate. Timeline busy time also does not measure how
many SMs are active: a one-block collective uses only one SM. Communication
still has substantial local GPU work: the single-block consumer validates NIC placement and folds peer rows.
Reducing that work can improve inference without changing a weight. A 25%
reduction in a 15.7% component suggests only about 4% overall improvement if
all other work stays constant; it does not imply a 25% service speedup.

The full profile is on the head under the original checkout's
`build-ci/fabric-runs/dsv41_perf_2026-09-16/baseline_profile/`; the text breakdown
is retained in [raw/baseline_profile.log](raw/baseline_profile.log).

## Changes retained

### Payload-dependent graph all-reduce block size

The original graph consumer used 256 threads for every payload. The final
consumer uses 256 threads below 32 KiB, 512 from 32 KiB to below 128 KiB, and
1,024 from 128 KiB. Wider rows expose more outstanding system-scope loads.
Small messages keep their existing launch size. Rank-order FP32 accumulation,
BF16 rounding, placement hashing, acknowledgement and graph generation order
are unchanged. Eager/bulk consumers retain their existing launch sizes.

Four-node graph probes use 80 replays with 32 generations per replay. The
following summarizes medians across ranks (and both bracketing 256-thread
reference trials). Every probe verifies the canonical reduction result.

| Payload bytes | 256 threads, µs | 512 threads, µs | 1,024 threads, µs |
|---:|---:|---:|---:|
| 10,240 | 34.35 | 34.50 | 34.20 |
| 61,440 | 71.20 | 62.30 | 64.10 |
| 153,600 | 132.85 | 107.60 | 102.80 |
| 307,200 | 236.85 | 187.60 | 173.20 |

At 153,600 and 307,200 bytes the 1,024-thread consumer saves 22.6% and 26.9%
respectively. The 61,440-byte result favors 512. The 10,240-byte probes use the
same 256-thread specialization in every trial. See [raw/collectives.json](raw/collectives.json).

A first probe accidentally retained the harness's default warm-spin floor;
its timings hid the improvement. Those results are excluded. The reported
probes explicitly use `DGPP_WARM_SPINS=0` and `--hold-ms 0`.

### Increase the bounded-prefill chunk to 4,096 tokens

Pool-aligned 4,096-token chunks amortize prefill work on long inputs. The
memory plan grows from 76.64 to 78.35 GiB per rank, including the prefix cache,
before the existing 4 GiB headroom. Weight residency is unchanged at 72.94 GiB.

Matched cold-prefix samples at actual lengths 8,424, 8,438 and 8,430 tokens:

| Chunk tokens | Median engine prefill | Notes |
|---:|---:|---|
| 2,048 | 5,914.2 ms | first baseline |
| 4,096 | 5,695.0 ms | retained |
| 8,192 | 5,627.0 ms | rejected as default: another 3.42 GiB for 1.2% |
| 2,048 | 5,958.6 ms | reverse-order baseline |

The improvement persists against the repeated baseline. Approximately 512-
and 2,100-token samples are effectively unchanged. Two additional prompts at
2,048 and 6,539 tokens generated 128 tokens each with exactly matching choices
and usage at all three chunk sizes. These checks concern bounded prefill;
they do not claim equivalence to the separate full 40-layer parity mode.

## Experiments rejected

* **Capture all four verification depths.** Six scalar slots plus four batch
  families require 20 variants per depth. Expanding the budget from 64 to 80
  makes `[1,2,3,4]` available instead of `[1,3,4]`, avoiding rounding two drafts
  up to three. The two-class screen looked favorable, but the broader
  five-class comparison of the combined candidate had effectively flat
  single-stream decode overall (−0.07% geometric mean), including regressions
  of 2.1% on math and 3.7% on chat. Its six-stream decode improved 3.1%, but
  that includes the independently faster collective kernel. The simpler
  width-plus-chunk candidate was therefore measured separately before
  choosing the final implementation; its one-request behavior was more stable,
  so the 64-variant budget was retained. All 105 paired outputs in the four-depth
  experiment were identical; this was a performance tradeoff, not a numerics
  failure. See [combined-comparison.json](raw/combined-comparison.json).

* **Cache large received payloads in device scratch while validating them.**
  This avoids a second NIC-memory read, but adds stores and cache pressure.
  The same-binary four-node comparison was 3–6% slower at large payloads
  (307,200 bytes: 174.8 → 182.25 µs; 768,000: 387.8 → 409.1 µs).
  The correctness checks passed; the optimization was removed.
* **Maximize a modeled verification reward directly over captured depths.**
  This avoids rounding an unconstrained choice to a captured depth, but the
  assumed cost model did not earn a throughput improvement across the screen
  and slightly hurt single-stream performance. Removed along with its
  experimental unit tests. The separate four-depth experiment was also
  rejected after the broader comparison.
* **Use 1,024 threads on every medium row.** The 61,440-byte probes favor 512,
  so the retained dispatch uses three sizes rather than a single larger block.

Screen measurements are exploratory, with only two repetitions and visible
batch admission variation. Final rates below are from fresh unprofiled runs.

## Final A/B and validation

The retained implementation is the **collective-width plus 4,096-token chunk**
change, with the original three-depth scheduling behavior. Across the five
classes, the equal-weight geometric mean of median engine rates is **+0.2% at
one request** (effectively flat) and **+1.3% at six**. Client wall rates improve
**0.6% and 1.6%**, respectively. These are small, workload-dependent service
gains, despite the larger isolated collective gain.

All rates below are tokens/s, medians of three runs at a 256-token cap:

| class | C | wall baseline | wall final | change | decode baseline | decode final | change |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose | 1 | 38.76 | 38.88 | +0.3% | 39.77 | 39.84 | +0.2% |
| prose | 6 | 85.37 | 87.55 | +2.6% | 88.70 | 90.89 | +2.5% |
| code | 1 | 58.58 | 58.70 | +0.2% | 61.33 | 61.57 | +0.4% |
| code | 6 | 102.50 | 105.07 | +2.5% | 107.78 | 109.34 | +1.4% |
| json | 1 | 69.96 | 71.01 | +1.5% | 74.44 | 74.96 | +0.7% |
| json | 6 | 115.11 | 120.82 | +5.0% | 122.07 | 126.65 | +3.8% |
| math | 1 | 58.85 | 59.35 | +0.9% | 62.30 | 62.47 | +0.3% |
| math | 6 | 104.20 | 105.45 | +1.2% | 108.72 | 110.11 | +1.3% |
| chat | 1 | 44.00 | 43.98 | -0.0% | 45.52 | 45.30 | -0.5% |
| chat | 6 | 89.45 | 86.83 | -2.9% | 92.44 | 90.27 | -2.3% |

Six-request chat did not improve in this campaign: median engine rate was
2.3% lower. Its baseline samples were 84.29/92.44/93.29 tok/s and the final
samples 84.81/90.27/93.16. Admission grouping and the adaptive controller's
history vary across runs (the median-phase prefill was 384 vs 533 ms, and
150 vs 153 decode steps). This is a measured mixed result, not a claim of a
speedup for every request distribution. The underlying collective improvement
is supported separately by the bracketing microbenchmarks. The final
implementation preserves the baseline scheduling policy; the four-depth
candidate's higher six-request mean (+3.1%) came with clear chat/math losses
at one request, so it remains an experiment rather than the default.

The final matched cold-prefill run measured **5,945.2 → 5,725.6 ms** at roughly
8,400 tokens: **3.7% less prefill time**. At roughly 2,100 tokens it was
1,619.8 → 1,599.2 ms; at roughly 520 tokens, 618.0 → 604.7 ms. The longer
chunk reserves **1.71 GiB more per rank**. These measurements cover the
reported lengths, not the full 128K context limit.

The final comparison verified **105/105 paired request outputs, usage and
finish reasons exactly equal**, four additional reasoning-enabled anchors
identical across builds, **4/4 MTP-versus-plain anchors exactly equal**, two
long-prompt generation checks identical, and nine cold-prefill samples with
identical first tokens/usage and zero cached tokens. Six-request isolation
passed, and shutdown verified identical operation streams on all four ranks.
See [final-comparison.json](raw/final-comparison.json), the complete
[baseline](raw/final_baseline/) and [final](raw/final_width_chunk/) evidence,
and [plain oracle](raw/final_plain/). The plain oracle used the same final
model/kernel code with the unused experimental 80-variant capacity; plain
mode does not use scheduled verification depths.

The full build completed. The 105-case CTest run passed 104 cases and exposed
an allocation-alignment error in the new transport fixture: the deliberately
ragged payloads had also made their backing slot sizes nonmultiples of 64.
Rounding the allocation up while retaining the exact payload sizes fixed the
fixture. A rebuild and full `bus_test` rerun passed (35.40 seconds), completing
coverage of all 105 cases. After removing the four-depth experiment, another
full build and the unit, DeepSeek engine and full transport suites passed
(3/3 CTest entries, 54.25 seconds; [narrow-tests.log](raw/narrow-tests.log)).
See [full-suite.log](raw/full-suite.log) and
[bus-retest.log](raw/bus-retest.log). No production-kernel failure was observed.
The optional real-checkpoint GLM TP oracle was skipped because
`DGPP_TP_REAL_MODEL` was unset; this is not a claim of fresh real-checkpoint
validation for every supported model.

The transport fixture compares every destination bitwise with the independent
canonical rank-order oracle, including staged and unstaged payloads, scalar
fallbacks, both dispatch thresholds, sizes up to 1,536,000 bytes, eager/graph
interleaving, and two replay windows in flight. The DeepSeek six-slot fixture
compares batched speculative transcripts with
plain eager generation. The experimental all-four-depth assertion passed,
but was removed when that production change was rejected.

The implementation is integrated on the head's `master` as `8c098f0` and
`24cf664`. The normal `dgpp_serve_app` target was rebuilt, producing the
optimized `build-ci/dgpp-serve` for subsequent DeepSeek launches. The prior
GLM deployment was restored with its original runtime binary and passed an
HTTP completion smoke check. The preexisting forum-post edit was preserved
byte-for-byte. [integration.json](raw/integration.json) records binary and
file checksums.

## Reproduction and artifacts

Run from the repository root on the idle head with its existing `.env`:

```bash
cmake --preset ci
cmake --build build-ci -j4
python3 scripts/site_env.py run-rank --rank 0 -- \
  ctest --test-dir build-ci --output-on-failure -j1
python3 benchmarks/results/2026-09-16-dsv41-perf/service_experiment.py candidate \
  --classes all --repeat 3 --tokens 256 --prefill --gates
```

The helper writes into `build-ci/dsv41-perf/NAME`, records the binary SHA-256,
and stops the world in `finally`, requiring identical operation streams across
all four ranks. `--bin /absolute/path/to/original/dgpp-serve` selects the control. The original
binary is preserved on the head at
`~/workspace/dgpp/build-ci/fabric-runs/dsv41_perf_2026-09-16/baseline-dgpp-serve`
(SHA-256 `329bf507a947d348ac629d35d7d251800435f0fa66e726314090303958482663`).
`--plain` disables MTP in a temporary configuration for the transcript gate.
The experimental environment flags in the harness only affect the archived
prototype; the retained production implementation has no new tuning flags.

[experiment.patch](experiment.patch) preserves the earlier width, receive-cache,
modeled-policy and chunk-size prototypes against `f01e947`.
[tiers-experiment.patch](tiers-experiment.patch) preserves the subsequent
three-versus-four-depth experiment against the same base. Apply either in a
fresh worktree, not on top of the final change. `collective_sweep.py` operates
on the former prototype; the final dispatch intentionally ignores its width
and receive-cache flags. Full raw logs remain on the head under the experiment
worktree's `build-ci/dsv41-perf/`; the small evidence files are copied under
[raw/](raw/).

## Further opportunities

The remaining expert cost makes reuse across tokens/requests and better
speculative acceptance more promising than another launch-overhead campaign.
The current implementation already has exact row shapes, tensor-core prefill,
grouped prefill and graph replay. Future experiments should measure expert
routing overlap and bytes read per committed token, and use the measured
cost by live batch size when calibrating scheduling. A tiled consumer using
more than one SM is another candidate: the wider single block still leaves substantial local collective
work, beyond the wire transfer itself. Those are proposals, not delivered
gains. Tiling and communication/computation overlap need a separate
synchronization/protocol design and matching-rank validation.

The search included the primary [four-Spark recipe](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-DSpark-TP4-4xGB10-Recipe/blob/main/README.md),
the [vLLM Spark recipe](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/main/docs/RECIPE.md),
and the [DSpark paper](https://arxiv.org/abs/2607.05147). Their profiles and
shape/scheduling ideas guided hypotheses; the numerical claims above are from
this deployment's measurements, not transplanted recipe benchmark numbers.
