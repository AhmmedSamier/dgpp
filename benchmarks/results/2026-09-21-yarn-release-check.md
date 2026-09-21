# YaRN release-check follow-ups

Date: 2026-09-21
Issue: [#16](https://github.com/HawkBearPig/dgpp/issues/16)
Baseline: `814b7b14fd44053f69eb25b168bb692bbf037f77`.

## Changes and reproduction

The release-check client multiplied millisecond arrival stamps by 1000 a
second time. A deterministic stream with first/last stamps of 100/264 ms and
11 tokens reproduced 16400 ms/token on the old script; the correct result is
16.4. Both decode and concurrent reports now use the same corrected calculation,
with missing timing retained for streams with fewer than two tokens.

Retrieval defaults to 768 generation tokens, including reasoning, and two
concurrent streams, matching the shipped YaRN template. Every concurrent stream
gets enough read-timeout budget for the whole group to run serially, since the
532480-token pool can seat only one 512K request. Explicit concurrency is honored
even when it exceeds the number of needle depths. Failed streams fail the verdict.

Cache reuse repeats the last retrieval probe immediately, before other phases
can evict it. The runbook no longer attributes low reuse solely to a disabled
cache. Reports are saved atomically after calibration, each probe and each phase.
`--resume` validates the check arguments and endpoint settings, retains completed
work, retries failed concurrent groups and primes the cache when needed. Schema
version 2 records status and per-attempt memory samples. The sampler is joined
on cleanup rather than relying on a fixed sleep.

When YaRN is already configured, the pool-above-ceiling diagnostic explains
concurrency headroom at INFO level. It does not tell operators to enable the
setting they already enabled. The correction-band builder rejects theta at or
below 1; the baseline accepted theta 0.5 and produced a finite, invalid band.
Plain RoPE still accepts positive theta below 1. Valid table arithmetic and
the serving kernels are unchanged.

The CUDA cross-limit gate now compares final-position incremental logits with
the last full-forward row, using the existing 2% relative-L2 and top-1 near-tie
budgets. Its previous finiteness and position checks remain in place.

## Local validation

- Full native CI build and the release server build passed.
- All 30 host CTest entries passed in 165.58 seconds. The native unit harness
  reported 229 cases with zero failures; its three GPU-arena cases skipped with
  CUDA hidden while production was live. The updated theta cases also passed
  after the final unit-test rebuild.
- All 13 offline release-check regressions passed: reasoning budget, timing
  units, last-entry cache reuse, serialized timeout budget, explicit concurrency,
  partial report recovery, cache priming after restart, completed-work reuse,
  incompatible-resume rejection, concurrent failure/retry, atomic write failure,
  and opt-in validation before endpoint access.
- The additional GLM DSA template/grammar suite passed all ten cases.
- With production stopped, all nine selected GPU/reference CTest entries passed
  in 34.39 seconds: Qwen fixture and reference generation, plan, plain and YaRN
  forwards, full-forward parity, cross-limit decode, incremental decode and QSA.
  The new final-position comparison measured relative L2 **0.00262999**, with
  identical top-1. QSA YaRN norm/rope and compressed-key comparisons had zero
  mismatches. Frozen Qwen and DeepSeek-V4.1 inverse-frequency tables still pass.

## Two-node release comparison

Both binaries use the release preset. The baseline is the previously running
`814b7b1` release; the candidate includes this issue's changes. Both run the same
updated client against `nvidia/Qwen3.8-Flash-Next-NVFP4` on two idle Spark nodes,
using the shipped YaRN template (factor 2, 524288-token advertised ceiling,
532480-token pool, two slots, depth-one MTP and 1.5 GiB prefix cache).

The command uses nominal length 2048, the default 4096-token control,
768-token retrieval budget, two concurrent streams, and the default 8192-token
decode document with 128 generated tokens. It measures both builds with the
same deterministic prompts and records full response hashes for accuracy
comparison. Raw outputs and logs are retained locally under `artifacts/issue16/`.

Both builds retrieved **5/5** needles at both actual prompt lengths, 3093 and
4273 tokens; every retrieval/control probe finished with `stop`. The nominal
2048 target becomes 3093 because the client retains its existing minimum of
64 filler records. Both repeated-cache probes reused **3088/3093 tokens
(99.84%)**. The eleven retrieval/cache/control response hashes and the
128-token decode response hash all matched across builds. The candidate passed
all nine API checks, including streaming, grouped choices, stop handling,
logit bias, cache usage and reasoning usage.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Cold prefill, ms | 2089.7 | 2097.2 |
| Decode, ms/token | 16.485 | 16.164 |
| Initial two-stream aggregate, tokens/s | 106.169 | 100.191 |
| Rank-zero peak memory, MiB | 51787 | 51787 |

Cold prefill differed by +0.36% and single-stream decode by -1.95%. The short
1.6–1.7 second concurrent sample was 5.63% slower on the candidate, so repeated
warmed concurrency samples were added before drawing a performance conclusion.

Both ranks logged the corrected INFO message on the candidate. At each clean
shutdown their operation-stream SHA-256 digests matched:

- Baseline: `21402f467da7f051519983374b22d278b5f47da5f02d9b7a99f3e586aed72ac9`.
- Candidate (also including API checks):
  `1155171ba3907bf7d929f21f7a1894f50f9d180cdbdd4f6c39b3e0f639d0604d`.

Eight repeated warm samples per build initially had medians of 107.173
(baseline) and 101.287 (candidate) tokens/s. The candidate had two distinct
bands: three samples at 107.27–107.96 and five at 99.99–101.51 tokens/s.
The server logs associated those bands with opposite request admission orders
(the first request's MTP acceptance was 93% versus 95%). All eight baseline
samples admitted the first question first. Reversing the baseline client's
submission order reproduced the lower band in all four additional samples,
101.06–101.70 tokens/s. Comparing the observed admission orders separately:

| Admission order | Baseline median (samples) | Candidate median (samples) | Candidate delta |
| --- | ---: | ---: | ---: |
| First question first | 107.173 (8) | 107.306 (3) | +0.12% |
| Second question first | 101.547 (4) | 100.824 (5) | -0.71% |

Each pair completed 172 tokens. Thus the initial aggregate difference followed
request order; the comparable medians differ by less than 1%. No accuracy or
performance regression was detected for the tested shapes. The initial and
repeated measurements are retained rather than replacing the slower result.
All three additional runs also had identical operation streams across their
two ranks. The default benchmark still submits independent HTTP streams, so
admission order must be accounted for when comparing short concurrent samples.

This check targets the client fixes and unchanged valid-model arithmetic. It
does not repeat the hours-long 262144/524288 retrieval matrix, whose existing
hardware evidence is in [the original YaRN record](2026-09-20-qwen-yarn512k.md).
The serialized long-request timeout is covered offline without waiting hours
for actual queue timeouts. These short-prompt samples do not establish a new
512K throughput or retrieval-accuracy claim.
