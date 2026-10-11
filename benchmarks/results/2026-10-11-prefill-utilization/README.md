# Prefill budget utilization and throughput follow-up

This follow-up to [PR 105](https://github.com/HawkBearPig/dgpp/pull/105) restores
throughput-oriented admission by default and fixes a separate Qwen prefill
budget inefficiency. It **does not make early admission throughput-neutral**.
New short requests can again wait behind a long prefill under the default
policy. `engine.prefill_early_admission=true` explicitly enables issue 104's
responsiveness policy, with its measured throughput cost.

## What caused the regression

The same eight-arrival workload generates 1025 output tokens. Before PR 105,
the long prefill finishes and the eight arrivals decode in fuller batches.
With early admission, the long reader occupies one of four request slots while
the arrivals start decoding. In the equal-budget test this changes 66 decode
passes into 96. Starting a decoder also selects the busy prefill budget; with
256/2048 busy/idle caps, that reduces the long reader's subsequent work cap by
a factor of eight. Neither effect is fixed by donating a short prompt's unused
prefill tokens.

There was also real budget waste: Qwen stopped at absolute budget multiples and
at its automatic native chunk-grid cuts. After a share changes, these can split
a later advance despite available budget. The new Qwen cursor spends its cap
from its actual position and retains structural/snapshot cuts while removing
redundant automatic cuts. The normal maximum-token and alignment checks still
apply. Other model families retain their existing cursor behavior.

The direct GPU regression uses a 32-token native limit, advances by 32 then 24,
and checks subsequent positions 80, 112, 144 and 155. The old grid stops at 64.
A required snapshot at 80 is retained, and target logits plus the next MTP draft
logits match a reference using the same explicit cuts. The production-sized
Qwen checkpoint uses a 4096-token native limit; the fixture's 32 is deliberately
small enough to expose the error.

## Builds and measurement conditions

- Pre-PR source: `6fe7f2db619c0e01ae8ad1ccc49060346a1faa69`.
- Merged PR 105 runtime source: `ac023c29f20366d5abd4d1b017c6809f5a5cfcdc`.
- First candidate source: `c8692bcf724df8e6cb226a6e05ca828d1fd03251`; labels
  `final-*` and `unpacked-*` identify this executable in the saved campaign.
- Final runtime source: `3426204f065a20e2b8de231153567f1aff60949c`; labels
  `packed-*` identify this clean native ARM Release build with CUDA 13.0.
  It packs the new flag beside existing flags, returning the Qwen cursor from
  312 to its original 304 bytes. Later commits change documentation only.
- Final executable SHA-256:
  `fc677569137b13586254c9823c500baac2b98450d78bdb536bb920ecb3f11399`.
  All other executable identities are in [measurements.json](measurements.json).
- One GB10 runs NVIDIA Qwen3.8-Flash-Next NVFP4. Four GB10s run Qwen
  Qwen3.8-Flash-Next FP8. Both use four slots, BF16 KV, a 131072-token pool,
  native MTP3, full admission and a 1 GiB prefix arena. Busy prefill cap is 256;
  idle cap is 256 or 2048. Timing requests disable prefix reuse.
- `CUDA_DEVICE_MAX_CONNECTIONS=32`, `DGPP_NO_SWAP=1`; before/after snapshots
  verify each live rank's executable SHA, zero process swap and zero allowed
  cgroup swap. No profiler, compile or other GPU test runs alongside timings.
- A fresh server and a completed 32-token warm-up precede each campaign. This
  warm-up uses different text from the 512-token decoder and does not guarantee
  steady-state timings for its full output. Scenario order is recorded. Long prompts use
  8192 sampled words (about 13K tokens) and generate one token. The initial
  decoder generates 512 tokens; eight burst arrivals, 250 ms apart, each
  generate 128. Temperature is zero and EOS termination is disabled.
- Trial 0 uses seed 104; trial 1 uses seed 105. Fresh controls have one trial;
  final default campaigns have two. Request-correlated native prefill records
  establish overlap before and after dispatch. Request/terminal deltas and
  final drain are verified. Wall time includes the entire scenario and drain.
- This is a small repeat campaign, not a statistical equivalence claim or a
  guarantee for other models, concurrency limits or arrival distributions.

## Measurements

The first-candidate table compares matching seed-104 workloads. The final repeat uses seed 105
and is shown separately rather than silently pooled with the one-trial control.
Rate change is control wall time / candidate wall time minus one, since each
paired workload generates the same number of tokens.

| Nodes, busy/idle caps | Workload | Pre-PR control (s) | Default (s) | Repeat (s) | Paired aggregate rate change |
| --- | --- | ---: | ---: | ---: | ---: |
| 1, 256/256 | Idle burst | 21.027 | 20.704 | 20.655 | +1.6% |
| 1, 256/256 | Busy burst | 27.955 | 27.896 | 27.860 | +0.2% |
| 1, 256/2048 | Idle burst | 14.283 | 14.037 | 13.938 | +1.7% |
| 1, 256/2048 | Busy burst | 27.820 | 27.808 | 27.782 | +0.04% |
| 4, 256/2048 | Idle burst | 9.894 | 9.780 | 9.668 | +1.2% |
| 4, 256/2048 | Busy burst | 16.391 | 16.398 | 16.495 | −0.04% |

The substantial burst regression is removed under the default policy in these
runs. The 0.04% four-node busy-burst difference and its slower repeat do not
establish exact equivalence; small timing differences remain visible in the
record. Default admission restores the longer short-request wait: first-arrival
TTFT is 14.59, 7.87 and 6.12 seconds in the three idle-burst configurations.
The incumbent decoder's busy-burst rate is 19.64 → 19.67, 19.73 → 19.74 and
33.50 → 33.43 tokens/s, respectively (control → default, seed 104). Thus the
previous large incumbent-decoder slowdown is also absent in these samples,
although the four-node point estimate is still 0.2% lower.

Exploratory equal-budget builds took 24.435 seconds with cursor-relative budgets
alone and 24.116 seconds after also removing redundant native-grid cuts, versus
21.027 seconds for the control. These uncommitted prototypes are identified by
executable hash in the JSON; they are not the final release build. Both still
needed 96 decode passes. The final default takes 66 passes.

In the first candidate, explicitly enabling early admission takes 24.283
seconds for the one-node equal-budget idle burst, with first-arrival TTFT 0.654
seconds. This remains 13.4% lower aggregate throughput than the fresh pre-PR
control. On four nodes with 256/2048 caps, the opt-in idle burst takes 13.689
seconds versus the control's 9.894 seconds (27.7% lower throughput), with
first-arrival TTFT 1.886 seconds. A faster first response does not remove the
total-completion cost.

The first equal-budget isolated-prefill sample was slower (14.583 versus
14.424 seconds), so three additional matching-seed runs were made. Their
medians were 14.365 seconds for the control and 14.226 for the first candidate;
the slower first result did not persist. At 256/2048, first-candidate prefill
was 7.693 seconds on one node and 6.050 on four, versus controls of 7.715 and
6.115 seconds. All individual samples remain in the JSON.

A late mixed decode/prefill campaign showed a 0.3% lower isolated-decode rate
for the first candidate (66.276 versus 66.491 tokens/s). It was investigated
rather than omitted. A subsequent three-trial decode-only campaign measured:

| Executable | Median decode tok/s | Scenario wall range (s) |
| --- | ---: | --- |
| Pre-PR, preceding control | 66.505 | 7.811–7.816 |
| First candidate, unchanged executable | 66.672 | 7.790–7.797 |
| Final candidate, packed flags | 66.489 | 7.802–7.817 |
| Pre-PR, following control | 66.452 | 7.816–7.819 |

The same first-candidate binary changed from slightly slower to slightly
faster than the controls. These samples do not support attributing the earlier
0.3% difference to its code, and they do not prove exact equivalence. Packing
the flag preserves the original cursor footprint; a throughput win caused by
that packing alone is **not** established.

Final-binary confirmation (same seed-104 fixtures; seconds):

| Nodes, busy/idle caps | Workload | Pre-PR control | Final binary |
| --- | --- | ---: | ---: |
| 1, 256/256 | Isolated prefill | 14.424 | 14.291 |
| 1, 256/256 | Idle burst | 21.027 | 20.738 |
| 1, 256/256 | Busy burst | 27.955 | 27.866 |
| 4, 256/2048 | Isolated prefill | 6.115 | 6.125 |
| 4, 256/2048 | Idle burst | 9.894 | 9.816 |
| 4, 256/2048 | Busy burst | 16.391 | 16.435 |

The final burst measurements retain the recovered throughput. The four-node
isolated-prefill and busy-burst point estimates are 0.16% and 0.27% slower than
their controls, respectively; this record does not round away those differences.

One final-binary four-node single-arrival sample took 13.673 seconds, versus
13.068 for the earlier candidate. A targeted three-trial repeat, using the same
scenario order for both builds, produced 13.781 / 13.126 / 13.119 seconds for
the final binary and 13.336 / 13.314 / 13.359 for the pre-PR control. The final
median is faster, but its first trial is 3.3% slower. Both versions use the same
129 decode passes and corresponding prompt-token counts. The small sample
shows no sustained slowdown in the repeats, but cannot establish startup
performance equivalence. No first-use sample is discarded from the record.

## Correctness and operational verification

- 101 scheduler cases pass, including ASan+UBSan. Tests cover default-off and
  opt-in behavior, aligned donation from a shifted cursor, ongoing long/decode
  progress, physical reservation limits, cancellation and snapshot cleanup.
- Seven host CTest suites pass: scheduler, service, HTTP, fabric, script reports,
  benchmark transport/drain validation and serve startup. Fourteen cluster
  configuration cases pass separately.
- Six native GPU suites pass on both candidate revisions: Qwen decode, row-independent engine, NVFP4 dense
  engine, MTP3/32-row engine, image prefill and prefix snapshots. The three
  engine variants exercise two-rank execution. The final revision also repeats
  four-node API/transcript checks and the MTP3 reservation/cancellation probe.
- API checks pass on one and four nodes. All four 128-token greedy transcript
  fixtures match pre-PR reasoning/content, finish reason and token usage on
  each model.
- Four-node full-reservation/MTP3 pressure: three requests hold 1536 of 2048
  blocks. A contender requiring 513 blocks stays queued with 512 free, starts
  after cancellation releases capacity, and all blocks return after drain.
  A subsequent request succeeds.
- The two-minute four-node soak on the first candidate completes 238 requests: 231 HTTP 200 and seven
  expected HTTP 503 overload rejections. The 14-request burst serves seven and
  sheds seven. There are no unexpected client/engine failures. Recovery has
  zero active, queued or retained request records. The pool returns to its
  initial 37 blocks with two retained prefix-cache entries, matching the initial
  pool/cache counts; these are not outstanding request reservations. The soak's final periodic metric is older
  than the explicit post-soak recovery snapshot.

All 103 serving scenarios (418 requests, including cancellations) pass exact
request/terminal-count, failure and drain checks. All 400 non-cancellation
requests match output bytes and usage for identical request bytes across the
available runs; 396 also match the previously published pre-PR fixtures, and
the four new seed-106 requests match their new controls. There are 26 distinct
request bodies. Every four-rank operation stream has an identical hash across
its ranks.

The original production configuration and original executable
`5b457ffed0f6290344c68609370e4b58ae893f52b4713ae39362babefea15542` were restored.
Live process hashes match on all four nodes, and HTTP models/metrics are healthy
with `engine_failed=false`. The test binaries were not deployed as production.

See [measurements.json](measurements.json) for individual timings, request/output
hashes, usage, native-overlap observations, counters, exact configurations and
process/operation-stream identities. [verification.json](verification.json),
[validation.log](validation.log) and [soak.txt](soak.txt) record the correctness
and restoration checks.

## Policy and remaining work

Early admission and aligned-share donation are now an explicit option, carried
through configuration, CLI, rank settings/warm records, configuration digest
and JSON/Prometheus metrics. The default preserves the earlier admission and
share-allocation policy. PR 105's physical MTP reservation accounting and head
snapshot cleanup fixes remain enabled.

The cursor fix is a useful independent optimization, but it does not remove
the scheduling cost of 30 extra decode passes or the busy-budget switch. A
throughput-neutral solution to the original responsiveness request remains
unproven. Future work must measure total completion time and incumbent decoder
latency as well as the first arriving request's TTFT. Increasing configured
concurrency or prefill caps would change the comparison and is not counted as
a fix here.

## Reproduction

Use the checked-in [one-node](qwen-w1.json) or [four-node](qwen-w4.json) config
with a separately saved binary built from the desired revision. Change
`prefill_idle_budget_tokens` to 256 for the equal-budget run, or set
`prefill_early_admission` to true for the opt-in run. The pre-PR configuration
omits the new field. Reserve idle hardware and run builds serially:

```sh
CUDA_DEVICE_MAX_CONNECTIONS=32 DGPP_NO_SWAP=1 python3 scripts/dgpp-cluster up \
  --config benchmarks/results/2026-10-11-prefill-utilization/qwen-w1.json \
  --bin build/dgpp-serve --log-dir build/utilization-world
python3 scripts/serve_admission_bench.py --label utilization --trials 2 \
  --prompt-words 8192 --decode-tokens 512 --arrival-tokens 128 --arrivals 8 \
  --out build/utilization.json
python3 scripts/dgpp-cluster down \
  --config benchmarks/results/2026-10-11-prefill-utilization/qwen-w1.json \
  --log-dir build/utilization-world
```

The benchmark validates overlap and cleanup itself. The recorded campaign also
checks every live process SHA and no-swap scope, then compares operation-stream
hashes after shutdown. These checks matter when reproducing a distributed run;
an HTTP response alone does not establish which binary ran on the peers.
