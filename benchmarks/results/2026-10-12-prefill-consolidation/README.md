# Shared prefill execution: validation record

Validation is in progress. Cached-model comparisons, numerical diagnostics
and temporary-checkpoint testing are recorded separately below. This record
does not yet establish a completed release gate.

## Change and comparison

Eager and graph serving share request ownership (`PrefillTask`), continuation
state (`PrefillContinuation`), mixed row planning and commits (`PrefillBatch`),
and layer input/output spans (`RowRun`). Seven executors use `SessionModel`;
legacy GLM uses the same helpers through its existing facade. Native
fixed-depth graph execution also shares `GraphBatchMapping` to avoid
decoding holes between live request slots. Block drafters and confidence
schedules retain their physical row mappings.

Automatic prefill budgets now use the aligned native work limit. This
amortizes weight conversion and streaming. Explicit deployed budgets are
preserved. Larger chunks can increase first-token latency and pauses between
decode updates; throughput alone is not a latency guarantee.

The control executable's serving sources match master `8e6909ad`, immediately
before consolidation. That master already contains earlier admission work.
These comparisons therefore measure the consolidation against its immediate
baseline, not against the source before all early-admission work began.

| Executable | Runtime source | SHA-256 |
| --- | --- | --- |
| Control | `3877e12e8f5a40b311efddd680c97f9dc1d45d41`, serving sources identical to `8e6909ad` | `9da56c960ee034f75bbc7219439f1f705f041f61136eeaf66928b808baed45b2` |
| Candidate with shared compact mapping | `df5b3c9b513ef896e5f7725c18632df03d12c1db` | `726ad15e6c9e56ae4d55d5a2f72ea457e2c5dc224cef638428be71a644f38268` |
| Candidate with native mixed decoder mHC and expert recipes | `beeaf952` | `c4980ec3563b23472b93ebe119a700ad3c706e53ab7032beb683947a58e29d82` |
| Candidate with mixed phase bounds and four-format validation | `d357c65b` | `62f5adfa724d9b94cdb7266e2986ec7ff3f1c1f3299b6e0fb776a0bbfac540c0` |
| Earlier candidate, before compact mapping | `26e61c9f55f068673eb771a1daa90d09c18d6978` | `0080af5401a43e91d2a5e76cf79f39db9de60442f24042147195add7f5e0a9c5` |

Builds use Release optimization with debug symbols disabled. Harness-only
and documentation commits after these sources do not change the executable.
Source receipts and selected test results are in
[build-and-gpu-tests.json](evidence/build-and-gpu-tests.json).

## Procedure

`campaign.py` serializes candidate and control deployments on the same GB10
hardware, with identical weights, prompts, output lengths, KV format and
speculation settings. Each configuration has four request slots, 65,536 KV
tokens and a 1 GiB prefix cache. Prompt seeds are 104 and 105. Scenarios
exercise isolated decode, isolated prefill, four simultaneous decoders, four
simultaneous prefills, a long reader arriving behind an incumbent decoder,
and repeated short arrivals during prefill. Decode requests generate 128
tokens; long prefill probes generate one token. Timing requests disable
prefix reuse and ignore EOS so both builds do the requested work.

Long prompts are warmed before measurement. The second trial reverses
scenario order. Most pairs run candidate first; the Qwen35 DFlash reverse
pair and GLM-DSA compact follow-up run control first. Two trials are a
focused regression check, not a high-confidence estimate of sub-percent
differences. Probability-gathering diagnostics are separate from throughput
measurements.

The harness retains prompt and output hashes, actual token counts,
request-level first-token latency and update gaps, physical/padded decode
rows, resolved admission policy, checkpoint revisions, live executable
hashes, node swap state and per-rank operation-stream digests. Multi-rank
digests must agree. Each campaign stops the test deployment and restores
the original production configuration and executable on every rank.

## Correctness checks

The shared GPU fixture covers every layer executor's continuation contract,
reader isolation, sparse decoder slots, speculative rollback, request
closure/reuse and snapshots. It exercises native speculation on and off.
Same-shape state and decoder-oracle checks are exact. Existing cross-shape
numerical checks require finite logits, relative L2 below 0.01 and the
repository's top-1 certification. Tiny fixture coverage does not establish
real-checkpoint numerical equivalence by itself.

All fifteen host Python suites pass, including benchmark transport and
server startup settings. The shared fixture and unit suites pass at the
recorded compact-mapping candidate. Compute Sanitizer reports
zero errors and zero leaked bytes across all eleven shared-prefill cases.
Image checks destroy the caller's prompt/image buffers after beginning a
shared task, then verify yielded execution, snapshots and draft state.
Qwen image ownership and GLM vision tests pass; the image lifetime sanitizer
also reports zero errors and leaks.

Serving checks include four chat transcripts per pair and two cache tests:

- A one-token cached continuation must match its isolated reference while
  sharing service with other work, without retaining extra KV blocks.
- A 128-token cached continuation warms one loaded execution, repeats that
  load exactly without further retained KV growth, then recovers its exact
  isolated reference. Cross-shape isolated-versus-loaded text differences
  are recorded independently. A different continuation can legitimately
  retain different cache entries.

A one-token completion may contain only a hidden special token. The
transport records its completion and usage without inventing a first-token
timestamp. Empty visible output from a longer decode still fails.

## Performance findings

The first shared implementation repeatedly converted Qwen35 FP8 weights
under 256-token mixed chunks. Profiling identified this work rather than
treating the regression as a reason to disable early admission. Reader
projection grouping, preservation of decoder projection recipes and native
automatic work budgets remove those repeated small walks.

A later GLM-DSA mixed workload remained 3.12% slower. It executed 68 padded
decode rows after requests left holes in the slot array. Shared compact
mapping reduces that case to four padded rows. A three-trial follow-up
measures mixed elapsed time at 19.178 → 18.911 seconds (1.39% faster) and
repeated arrivals at 17.844 → 17.392 seconds (2.53% faster). The residual
padding comes from the smallest available two-request graph bucket.

Completed paired measurements below report elapsed-time change; negative
is faster. Each link retains the underlying timings and request metrics.

| Configuration | Solo decode | Solo prefill | Batch decode | Batch prefill | Mixed | Repeated arrivals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Qwen35 FP8 + DFlash, graph¹](evidence/qwen35-reverse-v17-summary.json) | +0.06% | -0.53% | +0.22% | -3.15% | +0.68% | -0.63% |
| [Qwen35 FP8 + DFlash, eager¹](evidence/qwen35-eager-v18-summary.json) | +0.36% | -0.73% | -18.09% | -1.95% | +0.95% | -5.91% |
| [Qwen35 NVFP4](evidence/qwen35-nvfp4-final-v25-summary.json) | +0.19% | -0.96% | +0.62% | -0.59% | +0.64% | -1.81% |
| [Qwen NVFP4](evidence/qwen4-nvfp4-final-v25-summary.json) | -0.12% | -0.29% | -0.50% | +0.13% | -28.87% | -0.25% |
| [Qwen FP8, automatic budget](evidence/qwen4-final-v25-summary.json) | -0.32% | -0.01% | -0.03% | -0.16% | -10.46% | -0.01% |
| [Qwen FP8, matched 256 budget](evidence/qwen4-matched-budget-final-v25-summary.json) | -0.64% | -0.34% | +0.93% | -0.30% | -0.01% | -0.36% |
| [GLM Flash](evidence/glm-final-v25-summary.json) | +0.52% | -35.63% | -0.64% | -67.62% | -21.70% | -23.52% |
| [GLM-DSA²](evidence/glm-dsa-compact-v25-summary.json) | — | — | — | — | -1.39% | -2.53% |
| [DeepSeek V4.1³](evidence/dsv41-final-v27-summary.json) | -0.94% | -0.07% | -0.58% | -2.29% | +0.56% | -1.58% |

¹ Earlier candidate before compact mapping. These block-drafter/eager paths
are unaffected by that mapping change. ² Three trials of the affected
scenarios; earlier full-family coverage remains in the evidence. ³ Candidate
`beeaf952` preserves native mixed decoder mHC and expert recipes; all 34
responses and four chat transcripts match. Mixed elapsed time is
5.835 → 5.867 seconds; this sub-percent difference is based on two trials.

At the matched 256-token Qwen budget, mixed elapsed time is 7.676 → 7.675
seconds. Arrival first-token latency remains 0.33–0.36 seconds and maximum
decoder gaps remain 0.18–0.19 seconds. All 34 responses and four chat
transcripts match in that pair. The automatic-budget Qwen pair changes two
one-token randomized prefill outputs; its longer responses and chat
transcripts match.

Final candidate measurements and temporary-model results are still being
collected. Preserve individual request latency alongside aggregate timings:
the earlier Qwen FP8 default comparison changed the automatic busy budget
from 256 to 4096. Its first mixed trial improved whole-scenario throughput
while arrival first-token latency rose from 0.357 to 3.545 seconds and the
incumbent's maximum update gap rose from 0.203 to 1.797 seconds. The production
deployment explicitly uses 256, so that setting remains unchanged. A matched
256-token comparison is part of the final campaign.

## Numerical investigations

Not every response hash matches across execution shapes. These cases remain
visible in the evidence and are not counted as exact passes.

- Legacy GLM's final pair changes four one-token randomized prefill
  continuations; longer decode responses and four chat transcripts match.
  A separate probe reproduces the isolated change in the old executable by
  selecting the candidate's chunk budget: text and probabilities then
  match. Grouped prefill still has shape-dependent probability differences.
  See [the GLM diagnostic](evidence/glm-prefill-diagnostic-v22.json), which
  used the earlier candidate to investigate the original changed outputs.
- GLM-DSA's mixed probe changes ` jun` to ` The`. The control's top-two
  log-probability gap is 0.11135 nat, so this is not certified as a tiny
  near-tie. Candidate and control at the previous monolithic execution
  shape have exactly matching text and probabilities for all three
  requests. See [the GLM-DSA diagnostic](evidence/glm-dsa-prefill-diagnostic-v25.json).
- DeepSeek V4.1's mixed incumbent changes its 128-token counting response
  in both trials. Identical isolated and paired decoders have exact text and
  probabilities across builds; the candidate at the previous monolithic
  execution shape also matches the control exactly for all six probe
  responses. The mixed incumbent first changes token 48, at a control
  top-two gap of 0.00554 nat, but its probability shift is larger. See
  [the probability diagnostic](evidence/dsv41-decode-diagnostic-v25.json).
  A wider fixture reproduces the underlying mHC dispatch defect: mixed
  decoder rows were using the reader's coefficient reduction. V4 has the
  same issue, detectable in its fp32 post coefficients. The retained
  [regression receipts](evidence/mhc-regressions.json) include the original
  passing tiny checks, the wider V4.1 failure, and the V4 failure after its
  coefficient diagnostic was added. The mHC correction alone left a later
  changed token at a 0.27014-nat baseline margin; see
  [that intermediate probe](evidence/dsv41-decode-diagnostic-v26.json).
  Layer tracing then found differing expert outputs for identical decoder
  inputs. Preserving the native expert recipe through the shared executor
  resolves this probe: all six responses now match the baseline's text and
  token probabilities exactly, including the mixed incumbent and arrival.
  See [the corrected probe](evidence/dsv41-decode-diagnostic-v27.json).
  Six test targets pass, including thirteen shared-prefill cases and the
  37-case MoE suite; all six affected shared cases also pass memcheck with
  zero errors and leaks. The corrected candidate's paired performance and
  cache checks pass; its measurements appear in the table above. The earlier
  [V4.1 timing result](evidence/dsv41-final-v25-summary.json) is retained as
  investigation history, not as the corrected candidate's performance.
  A subsequent 38-case MoE run and thirteen-case shared-prefill run pass,
  including a new split-boundary check over FP8, NVFP4, MXFP4 and packed
  int4. That check also passes memcheck with zero errors and leaked bytes.
- Qwen NVFP4 reasoning continuations can differ between isolated and loaded
  execution in both old and new binaries. The old binary at the candidate's
  budget reproduces the candidate's transcripts. Repeat-under-load and
  isolated recovery checks distinguish this behavior from state leakage.

These probes identify dependence on execution shape; they are not a general
model-quality or teacher-forced numerical equivalence result.

## Temporary weights and cleanup

MiMo-V2.6-Flash-RL, GLM-4.7-NVFP4 and DeepSeek-V4-Flash-0731 were absent from
the saved initial inventories. Their final download, sync, testing and cleanup
are in progress, beginning with MiMo. Downloads are pinned to checkpoint revisions and performed one
model at a time after stopping production. No download runs alongside a
benchmark. Test-only resident images are disabled for these campaigns.

`cleanup_weights.py` verifies absence in every node's initial inventory
before removing that model's cache and lock directory. `audit_cleanup.py`
checks that temporary models are absent and original cache files, refs and
resident entries remain. Final cleanup and production restoration receipts
will be retained with the completed measurements.
