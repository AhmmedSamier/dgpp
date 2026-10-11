# PR 95: detailed decode investigation

The retained decode implementation passes all 60 cells; the later prefill
fusion passes all 12 cold-prefill buckets at TP1/2/4 against the stronger
of two FP8 bracket medians. The
four-node context probes also pass all three lengths. See the
[serving record](README.md) for the full matrix, exact build provenance,
validation and restored production service.

The retained changes combine an exact small-row FP4 kernel, paired gate/up
launch, bounded gate prefetch, measured verification scheduling and a
GPU-computed collective placement hash. The investigation below distinguishes
uninstrumented serving results from intrusive profiling and isolated probes.

## Controlled capture comparison

All profiles used four GB10 nodes, the same prompt, 256 output tokens,
three repetitions, greedy generation, context lookup disabled, and the
same frozen binary `0e6bd8d79d8e01d46532aca908739caefd922d838c89d76614a240efe8364252`.
FP8 retains its existing row-cost setting 2; NVFP4 uses the measured setting 4.
Every rank was captured. The quantizations produce different native texts,
so per-pass phase timings, row-count groups and useful output per second
are reported.

| Instrumentation | FP8 median token/s | NVFP4 median token/s |
| --- | ---: | ---: |
| CUDA-only, graph nodes, no CPU sampler | 59.631 | 60.569 |
| CUDA + OS runtime + 10 kHz GPU metrics + CPU samples | 42.554 | 42.579 |
| Metrics/CPU-only, no CUDA/OS-runtime trace, before | 61.148 | 62.200 |
| Metrics/CPU-only, during 1 kHz GPU / lower-rate CPU samples | 48.898 | 49.470 |
| Metrics/CPU-only, after | 61.117 | 62.166 |

Thus matched capture is useful, but overhead is not a uniform additive or
multiplicative penalty. Hardware metric sampling alone materially slows
serving; OS-runtime sampling also intercepts the hot communication spin
locks. The CUDA-only timelines are the timing reference. Captured hardware
metrics describe the perturbed run; final acceptance uses no profiler.

Host clocks differ, especially rank 1. Cross-rank logs are aligned by
collective order, not wall-clock timestamps. CPU/GPU metrics from the same
host use the exported session epoch; metrics-only comparisons record host
clock offsets and their round-trip bounds. Negative signed 32-bit clock
metric exports are decoded as unsigned Hz before conversion to MHz.

## Where one verification pass goes

CUDA-only rank-0 means, excluding warmup and cross-request prefill intervals:

| Phase | FP8 ms | NVFP4 ms |
| --- | ---: | ---: |
| Draft plus commit | 6.542 | 6.436 |
| Target attention and norms | 10.561 | 10.541 |
| First 56 target MLPs | 15.873 | 8.668 |
| Final 8 target MLPs | 2.298 | 2.212 |
| Target layer collectives | 11.738 | 10.618 |
| Target head and pick tail | 1.678 | 1.721 |
| Total | 48.690 | 40.195 |

Every compute stream is included: the final target fold/head can switch
streams. Treating the marker's stream as the only compute stream falsely
labels useful work idle. Intervals with no observed CUDA kernel, copy or
memset average 1.347–1.461 ms per NVFP4 pass
across the four ranks (1.441–1.594 ms for FP8).
The analyzer unions all streams and clips activities at pass boundaries.
Collective kernels occupy only one SM while waiting; a kernel being active
does not imply full GPU utilization. Prefetch overlaps these phases and is
not added to the totals. Kernel-only gaps and memory-operation overlap are
reported separately in the analysis JSON.

There are 261 usable FP8 intervals and 312 NVFP4 intervals from three
requests. The full per-rank, per-layer, per-kernel and pass distributions are
in `decode-cuda-only-analysis.json`; figures are in `decode-profile-figures`.

NVFP4 two/four/six/eight-row pass means were approximately
36.06/38.48/40.43/41.75 ms. Corresponding target collective means were
6.35/9.16/10.62/12.22 ms. Much of the marginal row cost is communication.
These groups follow the next target window after each verdict marker;
the interval also contains the preceding window's draft tail.

## Weight accounting and achieved matrix speed

Checkpoint headers show a hybrid quantization: the first 56 MLPs are NVFP4,
while attention and the final eight MLPs remain FP8. FP4 payload plus its
one-byte-per-16 scale is 9/16 of the FP8 coefficient payload, not 1/2.
Nominal transformer source bytes divided by four are 6.096 GB for FP8 and
4.458 GB for NVFP4, a 26.87% reduction. Small replicated tensors and runtime
scale widening make this an approximation to runtime traffic. The embedding,
head and drafter are excluded. Both formats read an approximately 318 MB
FP8 head shard twice per speculative pass, and share the same drafter.
See the tensor-by-tensor `decode-byte-audit.json`; resident allocation totals
must not be substituted for bytes streamed in decode.

The actual TP4 MLP geometry is H=5120 and I=4352 (17408/4). A rotating
microbenchmark uses eight copies of each weight matrix, avoiding a repeated
hot-cache measurement or timing a cache-flush memset. At eight activation
rows, across the four nodes:

- NVFP4 gate/up: 124.8–125.3 us; FP8 block: 219.3–224.3 us.
- NVFP4 down: 62.4–64.7 us; FP8 block: 100.3–103.8 us.
- Device-allocation pure-read reference: 233.7–235.3 GB/s.
- Host-touched managed allocation reference: 161.5–161.8 GB/s. Production
  weights use device allocations, so this is a placement control.

Useful weight bytes divided by event time are roughly 194–201 GB/s for
the NVFP4 matrices. This is useful-byte throughput, not a hardware DRAM
measurement; the synthetic matrix data is uniform within each copy.
Nsight Compute exposes L2/SM counters on these devices, but the requested
DRAM byte counters were unavailable. L2 request sectors include prefetch and
repeated lookups and must not be relabeled DRAM bytes.

An application-replay Nsight Compute capture retained the rotating history,
disabled clock/cache manipulation, and selected a steady-state eight-row
launch. It reports only 0.71 waves/SM for the paired grid and 0.42 for down;
the down grid has 40 CTAs for 48 SMs. Long-scoreboard/barrier stall shares
were 22.25%/18.25% for the pair and 31.89%/27.56% for down. Its durations
were themselves perturbed (163.04/69.54 us); they are not the serving or
microbenchmark timing verdict. A down-only 64-column prototype used 128 threads (four warps),
exposing 80 CTAs instead of 40. It passed the bitwise test but saved only
about one microsecond in the isolated down launch. It remains unselected.

In the earlier pre-hash control, drafting disabled on the same unprofiled binary measured 28.583 token/s
FP8 versus 35.087 NVFP4 (+22.75%). Drafting raised both rates but required
88 FP8 passes versus 105 NVFP4 passes for 255 timed tokens. This quantifies
the cancellation between faster passes and more passes; it is not treated
as a reason to stop optimizing.

## Collective investigation and retained prototype

The CUDA-only NVFP4 rank-0 window averages were 78.23 us per collective:
5.47 copy, 39.71 handshake, 17.71 peer skew, 15.34 fold. The CPU posting
span was 14.02 us. The heavier capture inflated peer skew to 108 us.
No rank was universally the sole laggard. Peer 3 arrived last most often
from ranks 0/1/2, but peer 2 was most often last from rank 3.

Changing the collective from 512 to 1024 or 256 threads worsened the
80 KiB isolated graph probe (64.9 -> 74.7 or 70.6 us). Both were rejected.

The posting path hashed the GPU-produced pinned activation row on the CPU
before issuing the sends. The candidate computes the identical position-mixed
XOR hash from the GPU's existing snapshot loads, stores it in the control
cell's former padding, and publishes it before the existing ready release.
The CPU acquires ready before reading the hash. Receiver placement gates,
wire hash, canonical rank-order arithmetic, and ring reuse rules are intact.
The control cell remains 192 bytes. The host-driven eager path is unchanged.

Isolated graph probes improved 10/40/80 KiB median collectives from
26.1/43.9/64.9 us to 25.8/41.7/61.3 us, with bitwise rank-order checks on all
four nodes. The independent snapshot test covers u32 tails, unaligned sources,
ring wrap, changing contents, in-place destinations, and all three block sizes.
All 29 targeted CTests passed, including the full fabric test and selftest;
memcheck had zero errors and racecheck zero hazards. Candidate SHA:
`62d441dd88fb7af5ac0db136a0cf76614f7fb487ae4de2fdf3ad95d29cc9ef53`.

The uninstrumented bracket was 62.225 -> 63.959 -> 62.270 token/s. All nine
responses were identical, all used 105 verification passes, and all four
operation streams matched for every leg. See
`decode-gpu-stage-hash-validation.json`. The subsequent decode campaign validated both formats on the same executable;
the separate prefill follow-up below closes the remaining narrow prefill gaps.


## Wider shard dispatch and retained verification schedule

The small-row kernel now covers the actual TP1/2 shapes as well as TP4:
N/K up to 17408, K divisible by 128, at most 16 activation rows, group 16.
Outside those bounds it uses the original production path. The expanded
bitwise test exercises ragged N, row counts 1/4/8/15/16/17, TP1/2/4 matrix
shapes, unaligned activations, independent gate/up globals, and output
sentinels. CUDA memcheck found no errors and racecheck no hazards.

A same-format base/candidate/base bracket with three prose repetitions per
leg measured 21.353 / 21.951 / 21.253 token/s at TP1 and
36.676 / 38.683 / 36.621 at TP2. These are 2.80% and 5.47% improvements
against the stronger baseline median. All output hashes and pass counts
matched, as did each world's rank operation streams. The rotating matrix
probe independently measured approximately 9% (TP1) and 11% (TP2) lower
combined gate/up/down time at eight rows. Those microbenchmarks use the
proper accumulation shift for each larger K.

After the GPU hashing change, another TP4 row-cost sweep measured
63.43 / 63.05 / 61.92 / 63.82 token/s at 3 / 5 / 6 / 4 ms per row,
respectively, with identical text. The NVFP4 template retains 4 ms; the FP8
template retains its existing 2 ms. The scheduler is a deployment cost
model, not a change to the underlying greedy distribution.

The two implementation commits are `16c5a52f` (dense FP4 decode, paired
launch, gate prefetch, verification cost and FP8 head memory accounting)
and `d12e7e9f` (GPU staging hash and its independent oracle). The full suite
passed before the final committed-source rebuild; that rebuild also includes
a host-only thread-safe initialization fix in the new launcher.


## Final committed-source matched capture

Both formats were captured again at `d12e7e9f`, binary SHA
`981d9dd764c4612c3c74f5903d7c42b47e1121733d69b1ad7a45ab345d13ac8f`,
with the same all-rank CUDA-only graph-node tracing. Every live inference
executable matched that hash; swap stayed zero and each world's operation
streams matched. The first startup stopped before measurement because the
identity audit followed the profiler wrapper instead of the live inference
executable. It is retained separately and contributes no timing samples.

| Final rank-0 mean phase | FP8 ms | NVFP4 ms |
| --- | ---: | ---: |
| Draft plus commit | 6.478 | 6.400 |
| Target attention and norms | 10.833 | 10.703 |
| First 56 target MLPs | 16.218 | 8.695 |
| Final 8 target MLPs | 2.348 | 2.249 |
| Target layer collectives | 10.108 | 9.468 |
| Target head and pick tail | 1.699 | 1.715 |
| Total | 47.684 | 39.231 |

Captured useful throughput medians were 60.865 versus 62.072 token/s,
with the same 88/105 pass counts. NVFP4 collective time fell by about
1.15 ms per pass relative to the pre-hash CUDA-only capture. Intervals with
no observed CUDA kernel/copy/memset average
1.291–1.447 ms across NVFP4 ranks, versus
1.394–1.506 ms for FP8. Active collective spin
remains separate from full-chip utilization. Timing varies between captures, so the
uninstrumented base/candidate/base hashing trial is the causal performance
check, and the final whole-workload FP8/NVFP4/FP8 brackets are the merge gate.
See `decode-hash-profile-analysis.json` for all distributions and rank details.

A smaller weight representation does not directly predict useful speculative
tokens/s. In this workload FP8 commits 255/88 tokens per pass, NVFP4 255/105;
the latter needs 19.3% more passes. The final captured verification interval
is 17.7% shorter, which more than offsets those extra passes but leaves a much smaller
throughput margin than the 46.4% reduction in first-56-MLP time alone suggests.
The plain-decode test separately demonstrates the weight-path benefit. These
are measurements of this prompt and draft policy, not a claim that no further
optimization is possible.


## Final plain-decode control and serving matrix

The d12e7e9f decode-campaign executable was tested again with drafting disabled,
using FP8 / NVFP4 / FP8, three repetitions per leg and 255 timed steps in
each request. The stronger FP8 median was **28.654 token/s** versus
**34.952 token/s** for NVFP4, a **21.98% gain**.
All requests completed 256 tokens with matching prompt hashes, executable
hashes, zero swap and consistent rank operation streams. This directly
measures useful decode without draft acceptance affecting pass counts.
See [the audit](decode-release-plain-audit.json).

| Nodes | 512-token prefill reduction | 2K | 8K | 32K | Decode throughput gain across 20 cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.02% | 4.87% | 1.70% | 1.43% | +3.61% to +33.65% |
| 2 | 12.79% | 6.26% | 1.70% | 1.69% | +6.17% to +28.31% |
| 4 | 9.71% | 3.07% | 1.49% | 1.31% | +1.49% to +40.44% |

| Four-node prompt length | FP8 engine token/s (stronger bracket) | NVFP4 | Gain |
| --- | ---: | ---: | ---: |
| ~1024 | 68.70 | 85.87 | +24.99% |
| ~8192 | 63.88 | 80.04 | +25.29% |
| ~32768 | 61.95 | 75.76 | +22.29% |

The [combined audit](release-combined-audit.json) and [context audit](prefill-pair-context-audit.json)
retain every cell, baseline drift and prompt/token matching. These are measured median gains;
small margins and individual repetitions should not be interpreted as a guarantee for every input.
The context buckets use three matched prompts per length, rather than repeating one corpus slice.


The full five-class matrix uses native greedy output from each quantization.
Its final same-format transcript audit checks 36 cases against the preceding
audited build, preserving text, reasoning, token counts and finish reasons.
This checks the implementation changes; it does not assert that the two
quantizations are numerically or qualitatively identical.

The remaining measured costs are attention, common FP8 layers/head, the
drafter, and collective copy/handshake/fold work. The 40-CTA down grid leaves
some SMs unused, but the tested 80-CTA alternative saved only about one
microsecond in isolation. The rejected alternatives and profiler limits
above constrain what can be inferred from theoretical weight bandwidth.
Neither the small speculative-prose margin nor this campaign establishes a
performance ceiling for future work.


## Prefill follow-up and evidence continuity

The full d12e7e9f matrix missed TP2 32K prefill by 0.49%. Sixteen-warp
scheduling improved the affected products, but its first final retest
still missed 8K/32K by 0.20%/0.25%. Combined same-warp gate/up/SwiGLU in
510542a4 then improved the same-format 8K/32K bracket by 0.95%/0.66%.
The [prefill investigation](prefill-investigation.md) records exact matched
two-rank traces, large-GEMM counters, rejected variants and the retained
fusion. Both earlier failed performance audits remain unchanged.

All 2,147 non-prefill CUDA kernels have identical compiled instructions
between d12e7e9f and the final fusion build. The model source diff affects
only the FP4 path at 1024 or more rows. Final validation retests all-world
cold prefill and prose decode, plus the four-node context brackets.
It also preserves 36 short transcripts, 108 one-token cold-prefill outputs,
and 27 longer-context responses against the decode campaign. Independent
kernel tests and real-checkpoint full-vocabulary hashes constrain the
prefill change separately. The combined audit records both build identities
instead of attributing all historical measurements to the latest executable.
