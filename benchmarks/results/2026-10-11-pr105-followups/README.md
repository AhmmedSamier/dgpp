# PR 105: admission correctness and throughput verification

Historical record: these measurements use the runtime revisions below, when
early admission was the default. The subsequent throughput follow-up makes
`engine.prefill_early_admission` opt-in; current defaults do not reproduce this
policy. Use the recorded revisions when reproducing these results.

[PR 105](https://github.com/HawkBearPig/dgpp/pull/105) addresses
[issue 104](https://github.com/HawkBearPig/dgpp/issues/104): a newly queued text
request can join a running non-group prefill even when the aligned fair shares
leave no one-shot admission budget. At most one new reader joins per tick;
slots, full physical KV reservation, aligned budgets and arrival ordering still
constrain admission. This does not guarantee a wall-clock latency bound.

Final verification found no isolated decode slowdown and no clear isolated
prefill regression in the measured configurations. It **does find lower
throughput under contention**, especially when early decoders switch a large
idle prefill budget to the smaller busy budget. The default favors the
interactive responsiveness requested in the issue. It is not a throughput win
for every workload.

## Follow-up implementation

- Redistribute unused aligned shares from short tails to unfinished readers.
  This prevents a tiny joining prompt from wasting half a long reader's budget.
- Preserve the original chunk grid for a lone reader, or when every reader fits
  its original share. Qwen advances to global grid boundaries; shrinking such
  tails created unnecessary GPU walks in an earlier candidate. Host and native
  GPU regressions now check this explicitly.
- Release the head prefix snapshot immediately when `begin_prefill` throws.
  The regression fails against the earlier implementation before cancellation
  or retry can hide the leak.
- Retain the PR's engine-aware physical reservation quote. Admission, begin,
  reserve and growth account for MTP/block-draft trailing rows and context clamps
  consistently.
- Add a reusable HTTP/SSE benchmark that verifies request-correlated native
  prefill overlap, fixed output counts, uncontaminated request deltas, published
  terminal counters, drain and released KV reservations.

## Builds and measurement conditions

Baseline source: `6fe7f2db619c0e01ae8ad1ccc49060346a1faa69` (current master,
including the prefix-hop fix). Final runtime source:
`ac023c29f20366d5abd4d1b017c6809f5a5cfcdc`. Later commits change documentation and
benchmark validation only. Both are native ARM Release builds with CUDA 13.0.

| Build | Executable SHA-256 |
| --- | --- |
| Baseline | `fb79a8aab0bad7e3d6ef4b116915d8342bcbbeefb7f0aa4da02f459e87de261d` |
| Candidate | `96c3de11f580a6d34c97bb3353e5da03dc95c1cf179487780bf4a53910707681` |

The candidate was compiled before committing the final grid fix; its embedded
version is `0.1.0+ge78e6e19bede.dirty`, while its runtime sources match `ac023c29`.
The executable identity, rather than that embedded version alone, identifies
these runs. Superseded candidate measurements are excluded.

- One GB10: `nvidia/Qwen3.8-Flash-Next-NVFP4`, three trials per build per primary
  scenario, at busy/idle budgets 256/256 and 256/2048. Runs alternate build
  order; a late baseline repeat checks drift.
- Four GB10s: `Qwen/Qwen3.8-Flash-Next-FP8`, two trials per build per scenario,
  budgets 256/2048. Candidate runs precede baseline runs.
- Both: four slots, BF16 KV, 131072-token pool, native MTP3, full admission,
  1 GiB prefix arena, `CUDA_DEVICE_MAX_CONNECTIONS=32`, `DGPP_NO_SWAP=1`.
  Every rank's executable hash and no-swap scope are checked. No CPU quota was
  requested by the campaign; inherited CPU limits were not independently
  established. See the exact [one-node](qwen-w1.json) and
  [four-node](qwen-w4.json) configs (the equal-budget run changes idle to 256).
- Deterministic cold prompts, temperature zero, `ignore_eos=true`, prefix cache
  disabled per timing request. Long input: 8192 sampled words, about 13K tokens,
  one output token. Decoder: 512 output tokens. Single arrivals return eight
  (`mixed_short`) or 128 (`mixed_long`) tokens. Bursts have eight arrivals,
  250 ms apart, each returning 128 tokens. Trial seeds are 104 and 105; repeated
  trial-zero runs intentionally reuse seed 104.
- Fresh server per campaign, completed warm-up before timing, no profiler or
  concurrent GPU tests. Wall time includes all work and final drain. Aggregate
  output rate is total generated tokens divided by that wall time. Client SSE
  gaps are update gaps, not individual kernel/token latency. Cancelled cases
  validate cleanup and are excluded from throughput comparisons.

## Isolated measurements

Medians; brackets are observed minimum–maximum wall times, not confidence
intervals. Decode rate excludes the wait for the first output event.

| Nodes, budgets | Decode tok/s, baseline → candidate | Prefill wall seconds, baseline → candidate |
| --- | ---: | --- |
| 1, 256/256 | 66.35 → 66.36 | 14.335 [14.180–14.367] → 14.290 [14.185–14.332] |
| 1, 256/2048 | 66.34 → 66.37 | 7.730 [7.586–7.742] → 7.710 [7.580–7.723] |
| 4, 256/2048 | 127.51 → 127.85 | 5.967 [5.911–6.023] → 6.051 [5.922–6.181] |

The four-node prefill median is 1.4% slower, with overlapping ranges and only
two trials. These observations do not establish statistical equivalence or
rule out smaller regressions.

## Contention measurements

Medians in seconds. `Idle burst` starts without a decoder; `busy burst` starts
with the 512-token decoder. The last column is the change in aggregate output
tokens/s for the whole scenario, not a GPU kernel measurement.

| Nodes, budgets | Workload | First arrival TTFT, baseline → candidate | Long TTFT, baseline → candidate | Aggregate rate change |
| --- | --- | ---: | ---: | ---: |
| 1, 256/256 | Single short answer | 17.487 → 0.706 | 17.304 → 17.690 | −0.5% |
| 1, 256/256 | Single long answer | 17.496 → 0.682 | 17.360 → 18.734 | −2.4% |
| 1, 256/256 | Idle burst | 14.492 → 0.629 | 14.207 → 20.587 | −14.5% |
| 1, 256/256 | Busy burst | 17.565 → 0.705 | 17.348 → 20.561 | −1.1% |
| 1, 256/2048 | Single short answer | 17.497 → 0.688 | 17.362 → 17.712 | −0.5% |
| 1, 256/2048 | Single long answer | 17.542 → 0.689 | 17.406 → 18.778 | −2.2% |
| 1, 256/2048 | Idle burst | 7.912 → 2.391 | 7.626 → 17.356 | −36.3% |
| 1, 256/2048 | Busy burst | 17.658 → 0.706 | 17.427 → 20.573 | −1.2% |
| 4, 256/2048 | Single short answer | 10.587 → 0.443 | 10.513 → 10.738 | −0.7% |
| 4, 256/2048 | Single long answer | 10.601 → 0.429 | 10.526 → 11.508 | −2.9% |
| 4, 256/2048 | Idle burst | 6.104 → 1.882 | 5.935 → 11.226 | −29.4% |
| 4, 256/2048 | Busy burst | 10.655 → 0.428 | 10.524 → 12.617 | −2.1% |

Early arrivals occupy decode slots while the long reader still occupies a slot,
changing batching efficiency. They also select the busy prefill budget on
subsequent ticks. Equal busy/idle budgets remove the budget-switch component,
but not the additional decode work or batching cost. The existing decoder's
rate in the busy burst falls 7.5%, 7.6% and 8.2%, respectively, across the three
configurations above.

Earlier first output is not a promise of better latency for every arrival:

| Nodes, budgets | Burst | Mean arrival TTFT, baseline → candidate | Maximum arrival TTFT, baseline → candidate |
| --- | --- | ---: | ---: |
| 1, 256/256 | Idle | 15.246 → 9.911 | 16.579 → 20.318 |
| 1, 256/256 | Busy | 19.728 → 13.823 | 22.858 → 22.607 |
| 1, 256/2048 | Idle | 8.673 → 10.579 | 10.121 → 18.319 |
| 1, 256/2048 | Busy | 19.766 → 13.839 | 22.888 → 22.650 |
| 4, 256/2048 | Idle | 6.192 → 6.649 | 6.891 → 11.048 |
| 4, 256/2048 | Busy | 11.511 → 8.107 | 13.023 → 13.021 |

These are medians of each trial's mean/maximum. For this Qwen non-group path,
the current automatic idle setting resolves to the busy budget (256), not 2048.
Deployments explicitly using a larger idle budget should expect the larger
trade-off shown here. Group-prefill model defaults differ.

The separate one-node retirement probe doubles the prompt to 16384 words and
limits the original decoder to 64 tokens. In all four runs that decoder ends
about 5.0–5.4 seconds after the long request starts, before its first output.
With an eight-token arrival, baseline/candidate arrival TTFT is 18.406/0.806 s
and long TTFT is 18.322/18.446 s. With a 128-token arrival, arrival TTFT is
18.360/0.681 s and long TTFT is 18.273/22.229 s. The longer early answer keeps
the busy budget active longer; decoder retirement alone does not guarantee
return to the idle budget. This is one trial per build per retirement case.

## Correctness and failure checks

- All 116 timing scenarios (492 requests, including cancellation cases) drain
  with exact request/terminal deltas, no failures and zero retained KV blocks.
  Across 168 distinct non-cancellation request fixtures, baseline/candidate
  request bytes, text-output hashes and usage match, including repeated runs.
- All 100 scheduler cases pass, including ASan+UBSan. Negative controls reproduce
  the earlier wasted-share, head-slot cleanup and global-grid failures.
- Final host suites pass: 101 service, 18 HTTP, four fabric-service, two
  shutdown-watchdog, six engine-watchdog, 34 script-report and seven benchmark
  harness cases, plus the 100 scheduler cases above (272 total).
- Native two-rank GPU gates pass for row-independent decode, NVFP4 dense rows
  and MTP3/32-row decode. They cover early short admission, donation, exact
  transcripts where row-independent arithmetic applies, lone-reader walk
  counts and released reservations. Qwen/GLM prefix snapshots and Qwen image
  prefill gates also pass; image scheduling eligibility is unchanged.
- The four-node physical-reservation probe holds three 512-block requests.
  A fourth request has a free slot but needs 513 blocks because of MTP3's two
  trailing rows; with only 512 free it defers. Cancelling a holder releases it;
  after cancelling all four, the pool returns to zero and recovery succeeds.
  This exercises full reservation, not a populated 128K-token context.
- The three-minute four-node soak records 346 HTTP 200 responses, seven
  expected overload rejections and zero unexpected errors. It includes
  disconnects, long input, multi-turn prefix reuse and a 14-request burst.
  Post-soak recovery has zero active/queued/retained request records and no
  engine failure. Its 37 occupied blocks belong to two retained cache entries.
  The soak summary's final periodic sample precedes drain; the separate
  recovery snapshot is the actual final health check.
- Baseline/candidate API checks pass on both models. Four 128-token greedy
  transcripts per model match in content, reasoning, finish reason and usage;
  only timing is excluded from that comparison.
- All four ranks have identical operation-stream hashes for each four-node
  deployment, including the extended checks and soak.

The first four-node timing campaign completed successfully, but its subsequent
pressure probe read a stale published pool counter and failed its test
assertion. Active/queued counts are live while other scheduler counters publish
after the engine pass. The probe and reusable harness now wait for published
records/terminal counters. All saved timing cases were retrospectively checked
for zero records and exact terminal/request deltas. Extended checks were rerun
successfully on a fresh candidate process; they are not attributed to the
uninterrupted timing process. The earlier process lacks an after-run node
snapshot, but its before-run hashes/no-swap scopes and all-rank operation
streams are retained.

After verification, the original four-node production service was restored.
Its resolved configuration is byte-for-byte equivalent as parsed JSON, all
four live executables retain SHA-256
`5b457ffed0f6290344c68609370e4b58ae893f52b4713ae39362babefea15542`, and
the model/metrics endpoints are healthy with `engine_failed=false`.

## Reproduction and evidence

Use otherwise idle hardware and serialize GPU runs. Launch a fresh baseline or
candidate with the config matching the table:

```sh
CUDA_DEVICE_MAX_CONNECTIONS=32 DGPP_NO_SWAP=1 python3 scripts/dgpp-cluster up \
  --config benchmarks/results/2026-10-11-pr105-followups/qwen-w1.json \
  --bin /path/to/dgpp-serve --log-dir /path/to/run/world
python3 scripts/serve_admission_bench.py --label candidate \
  --out /path/to/run/admission.json --trials 2 --prompt-words 8192 \
  --decode-tokens 512 --arrival-tokens 128 --arrivals 8
python3 scripts/dgpp-cluster down \
  --config benchmarks/results/2026-10-11-pr105-followups/qwen-w1.json \
  --log-dir /path/to/run/world
```

The one-node campaign adds a separate fresh-process `--trials 1` run per build
to the two-trial run above; the four-node campaign uses two trials per build.
For the decoder-retirement case, use `--prompt-words 16384 --decode-tokens 64
--scenarios mixed_short mixed_long`. For the native boundary, run
`python3 benchmarks/results/2026-10-11-pr105-followups/reservation_check.py
/path/to/reservation.json` against a fresh cache-empty MTP3 server using the
recorded config. Use `scripts/serve_api_check.py`,
`scripts/serve_greedy_transcript.py --help` and `scripts/serve_soak.py` for the
API, transcript and soak extensions.

[Measurements](measurements.json) preserve individual request rates, TTFT/gaps,
body/output hashes, token counts and trial ranges.
[Verification](verification.json) records deployment arguments, configs, binary
hashes, request/drain audits, rank snapshots, operation-stream hashes, native
reservation transitions and post-soak health. Raw local timing reports are
identified by SHA-256. These bounded runs do not establish global starvation
freedom, model-wide numerical/quality equivalence, long-running soak stability,
or performance for unmeasured model families, pool sizes and arrival patterns.
