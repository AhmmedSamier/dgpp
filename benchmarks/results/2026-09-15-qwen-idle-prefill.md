# Qwen prefill budgets and idle-time chunk growth

Date: 2026-09-15. Baseline source: `cbf710b`.

[Raw measurements, comparisons, profiles and validation metadata](2026-09-15-qwen-idle-prefill.json).

This follow-up adds an optional larger prefill budget when no request is
actively decoding. It rechecks that condition after every yield, so a prompt
already in progress can accelerate when its decoding peer retires. The busy
budget still bounds each prefill quantum while another request decodes.
Both defaults remain zero. This change does not widen decode graphs or change
CUDA kernels.

## Implementation

- `engine.prefill_idle_budget_tokens` / `--prefill-idle-budget-tokens`
  requires an enabled busy budget, must be at least that budget, and must
  satisfy the model's alignment and scratch limits.
- The scheduler selects a budget from replicated request state only, after
  cancellations and reservation growth. It makes no extra budget decision on
  ticks that only decode. Grouped cold admission uses the selected total budget.
- The continuation cursor keeps mandatory model and snapshot cuts separate
  from the scheduling grid. Increasing the budget can coalesce future chunks
  without skipping a snapshot. Fixed budgets retain their previous cuts.
- The settings, effective-configuration digest, warm journal and metrics
  include the idle budget. Old journal records retain their former behavior.
- The cold-prefill probe records actual/computed/cached token counts, prompt
  hashes, incremental visible TTFT and engine time. It waits for the retired
  metrics snapshot after an SSE finish, and refuses cached or contaminated
  measurements. An optional interval separates requests in burst profiles.

One unfinished prefill progresses at a time. Other queued requests can still
wait behind it; larger idle chunks can also delay a new arrival or cancellation
until that chunk ends. This does not implement grouped continuation.

## Method

Qwen3.8-Flash-Next-FP8, TP2, four slots, native MTP depth 1, BF16 KV with
262,144-token capacity, resident n-gram tables, 1.5 GiB prefix cache and 128
sampling candidates. The configuration is
`deploy/cluster_qwen-3.8-flash-next_fp8_w2.example.json`. Decode graph shapes,
weights, templates and thinking settings remain fixed across the sweep.
The checkpoint revision is `236dfdf285828023ca3bcd3f37366c58a3469b13`.

Each budget starts a fresh world using a preserved binary. Cold probes use
three deterministic prompts per requested length, identical tags and seed,
thinking disabled and one generated token. Requested lengths are approximate:
the actual ranges are 2,024–2,051, 8,014–8,230 and 32,339–32,397 tokens.
All 36 baseline cold samples match prompt hashes and token counts, with zero
cache hits. The incomplete first 256-token run exposed the metrics publication
race in the client; it was discarded and repeated on a fresh world after the fix.

Interference trials send a 25,629-token prompt one second after a separate
1,024-token prose completion starts streaming. Three trials per budget use
distinct long-prompt prefixes; each completion stays cold. The same greedy
decode transcript and output length are required. Reported pauses are visible
client update gaps, not per-token kernel latency or a guaranteed latency bound.

The transition workload uses a 64-token decode and a 25,628-token prompt. It
requires the peer to retire before the prefill finishes. Prefix caching remains
enabled throughout; measured long prefixes must be cold.

## Baseline budget sweep

Median engine prefill time, seconds:

| Fixed budget | About 2K | About 8K | About 32K |
|---|---:|---:|---:|
| 0, monolithic admission | 1.393 | 5.749 | 24.532 |
| 256 | 2.324 | 9.535 | 40.014 |
| 512 | 1.804 | 7.447 | 31.495 |
| 1,024 | 1.535 | 6.367 | 27.013 |

For the three 256-token interference trials, the long request takes
33.46–33.51 seconds and the largest decode update gap is 353–356 ms. The
unbudgeted median gap is 18.763 seconds. All twelve baseline interference
trials preserve the same 1,024-token decode transcript.

## Idle-budget results

With a 256-token busy budget and a 2,048-token idle budget, median cold prefill
recovers most of the unbudgeted performance:

| Prompt size | Fixed 256 | Busy 256 / idle 2,048 | Time reduction |
|---|---:|---:|---:|
| About 2K | 2.324 s | 1.412 s | 39.2% |
| About 8K | 9.535 s | 5.801 s | 39.2% |
| About 32K | 40.014 s | 24.700 s | 38.3% |

All nine candidate prompts, usage counts, outputs and finish reasons match
the unbudgeted baseline exactly. The candidate is still 0.7–1.4% slower than
the earlier unbudgeted prefill medians. These separate deployments do not
isolate that small difference: clocks were not locked, and cache occupancy
was not reset between request groups, although every measured prompt had
zero attached cache tokens. The candidate's lifecycle checks preceded its
cold probes.

The three-trial interference medians show where the improvement applies:

| Workload | Fixed 256: long-request wall | Busy 256 / idle 2,048 | Worst decode-gap median, before → after |
|---|---:|---:|---:|
| Peer continues decoding | 33.495 s | 33.526 s | 355.8 → 353.1 ms |
| Peer finishes during prefill | 29.775 s | 20.265 s | 329.8 → 327.7 ms |

The second row reduces long-request wall time by **31.9%**. In the first row,
the busy budget remains in force, preserving the existing pause/throughput
tradeoff. All six candidate trials match the baseline decode and long-request
outputs, usage counts and finish reasons.

Cancellation with the larger idle quantum drained in **1.481 seconds** and
released the partial reservation. All 16 sampled, schema-constrained long
extractions returned the expected records, including eight prefix-cache hits.
The real-service stop, multiple-choice, logit-bias and usage checks passed.
Both Qwen candidate worlds stopped with identical operation streams on both ranks.

## Qwen C1 preservation

Four fresh deployments each run five repeats of all five C1 anchors: the
initial unchanged baseline, candidate defaults, candidate with the idle budget,
then an unchanged-baseline recheck. All 100 transcripts and output lengths
match, as do each request's target decode-pass counts.

Median output-span throughput, tokens/s (initial TTFT excluded):

| Class | Initial baseline | Fresh baseline recheck | Candidate default | Candidate idle budget |
|---|---:|---:|---:|---:|
| Prose | 45.31 | 44.91 | 45.16 | 44.99 |
| Code | 49.03 | 48.74 | 48.98 | 48.85 |
| JSON | 50.98 | 50.61 | 50.80 | 50.78 |
| Math | 47.70 | 47.42 | 47.51 | 47.53 |
| Chat | 41.70 | 41.56 | 41.58 | 41.54 |

The unchanged binary shifts **−0.35% to −0.86%** between sessions on this
metric (equal-class geometric mean **−0.62%**). Relative to the fresh control:

| Candidate mode | Full-request throughput | Output-span throughput | Server admit-to-retire rate |
|---|---:|---:|---:|
| Default budgets disabled | +0.14% | +0.33% | +0.22% |
| Busy 256 / idle 2,048 | +0.18% | +0.18% | +0.13% |

These are equal-class geometric means, not claimed C1 speedups. The server
column uses millisecond log timestamps after prefill admission and includes
host/runtime overhead; it is not a GPU-only timing. Its unchanged-build
variation is about 0.66%, so the shift is not confined to HTTP delivery.

The literal zero-tolerance screen still flags one metric in each candidate
mode: default JSON full-request throughput **−0.02%**, and idle-budget chat
output-span throughput **−0.05%**. No tolerance was relaxed. The measurements
support C1 preservation within observed unchanged-build variation, rather than
proof of an exactly zero effect. Both defaults remain disabled, and no existing
published C1 benchmark row is replaced with a claimed improvement.

## GLM C1 control and restoration

GLM-5.3-Flash-NVFP4-FP8 runs TP4 with four slots, MTP depth 1, BF16 KV
capacity 786,432 and an 8 GiB prefix cache. Candidate defaults and the original
restored service each complete five repeats of the five 320-token C1 anchors.
All 25 paired prompt hashes, outputs, token counts, usage and finish reasons
match. This control compares against the original running binary, which
predates `cbf710b`; it covers the combined changes since that binary.

| Class | Original output-span tok/s | Candidate output-span tok/s | Output-span change | Full-request change |
|---|---:|---:|---:|---:|
| Prose | 54.16 | 54.29 | +0.25% | +0.26% |
| Code | 58.16 | 58.33 | +0.29% | +0.05% |
| JSON | 58.70 | 58.80 | +0.16% | +0.15% |
| Math | 55.81 | 55.96 | +0.26% | +0.35% |
| Chat | 50.62 | 50.73 | +0.22% | −0.05% |

The zero-tolerance screen flags chat full-request throughput (−0.047%).
Its repeat ranges overlap almost completely. These measurements do not
establish a C1 speedup or a causal regression at that scale.

The candidate GLM world stopped with operation-stream MD5
`22584486025e2ae87d316eb9d5214295` on all four ranks. After validation, the
original deployment configuration and binary were restored. At 16:21 UTC,
the model endpoint responded, the scheduler had no active or queued requests,
and each rank's running executable matched the original SHA-256 below.

## Profiles and the next kernel target

Nsight Systems profiles use the same 8,281-token cold prompt on rank 0.
Each measured request is separated by one second, so the final kernel burst
matches that request's engine timing. These are instrumented profiles,
separate from the performance table above.

| Profile | Request window | Kernel launches | Routed-expert matrix multiplies | QSA attention partials |
|---|---:|---:|---:|---:|
| Unbudgeted | 6.035 s | 15,578 | 1.382 s | 1.509 s |
| Fixed 256 | 9.835 s | 82,150 | 4.683 s | 1.455 s |

Routed-expert matrix multiply duration increases by 3.301 seconds, against a
3.800-second increase in the profiled request window. This points to poor
expert utilization and repeated weight traffic at small prefill batches as
the next optimization target. It is an inference from kernel duration and
the grouped MoE implementation, not a measured memory-traffic counter.
Investigate smaller per-expert tiles or a low-row path before assuming that
host launch overhead or collectives explain the budget penalty.

QSA attention remains a substantial cost with larger chunks. Its listed GQA
attention in `src/kernels/qsa.cu` is a separate target for prefill tiling and
reuse. The current default decode dispatch should remain covered by C1 gates.

## Validation status

The full CI build and all 19 selected tests passed: scheduler, service, journal,
configuration/unit/Python tests; Qwen target/draft state, snapshots, resize
rejection, cancellation, wide graphs and TP; and shared continuation/engine
checks for DeepSeek, GLM-4.7 and full GLM.

The baseline and candidate CUDA fatbins are byte-identical: 50,737,648 bytes,
SHA-256 `d5438471d34573e07c99108965a4e1ba29f30e81718246f4efddfb8391b32feb`.

Binary provenance:

| Artifact | SHA-256 |
|---|---|
| Qwen baseline, source `cbf710b` | `e5d5ba9382182bb173418468ef017c9f75eecc1762e70bdea28ab9f5b72eaf00` |
| Candidate | `cffb8e10e26b086b28342d3451aebe79f2bbd287ecad3214d09bec6877c1db56` |
| Candidate engine patch against `cbf710b` (`src/`, `apps/`) | `59e2f0108fe3d225bd7d2c6a0d52f26c796f92c2115ad4b04ed9a0bcf5080c7c` |
| Original/restored GLM executable | `49c0c9dc9fbde3cc04138292cf7059b6cc2e30e1cdb2dbe08aa5c51a39fa8259` |

## Reproduction

On reserved fabric, preserve separate baseline and candidate executables.
Use the Qwen TP2 example configuration above and a fresh world for each
budget. The candidate's flags are:

```text
--prefill-budget-tokens 256 --prefill-idle-budget-tokens 2048
```

Run these clients against each otherwise idle world, keeping the same tags
and data file across matched runs:

```bash
python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \
  --repeat 3 --seed 7 --tag prefill-efficiency-2026-09-15 --no-think \
  --json-out cold.json
python3 scripts/serve_prefill_interference.py 127.0.0.1 18080 \
  --prompt-words 24000 --decode-tokens 1024 --tag prefill-efficiency-r0 \
  --json-out interference-r0.json
python3 scripts/serve_prefill_interference.py 127.0.0.1 18080 \
  --prompt-words 24000 --decode-tokens 64 --tag prefill-transition-r0 \
  --json-out transition-r0.json
python3 scripts/serve_load.py 127.0.0.1 18080 --classes all \
  --concurrency 1 --max-tokens 320 --repeat 5 --json-out c1.json
```

Repeat interference and transition with tags ending in `r1` and `r2`.
Capture the endpoint's effective configuration and binary identity for each
world, then stop it and compare every rank's operation-stream hash. The raw
report retains every C1 comparison, including the initial comparisons and
the unchanged-baseline drift control; none of the failed zero-threshold
screens are discarded.
