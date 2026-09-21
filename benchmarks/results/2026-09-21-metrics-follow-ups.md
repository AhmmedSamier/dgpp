# Metrics follow-ups from issues #8 and #9

Base: `fdeabc7bfbed3e4c270731c4bf24be01850de4e0`.
Issue [#20](https://github.com/HawkBearPig/dgpp/issues/20).

## Changes

The replay end event is now enqueued immediately after `cudaGraphLaunch`,
before updating the host decode-batch counters. The counter expressions,
graph and stream selection, recorded kernels, and retirement path are
unchanged. An event-record failure still fails the engine; its failed replay
no longer updates the counters before that failure.

The two metrics reference sections link to each other. The decode-batch
histogram deliberately retains all sixteen keys, including zero buckets,
so scraper shape stays stable across engines and deployments. Existing
service tests already assert every bucket on both endpoint aliases.
Documentation explains sparse-slot padding and the limits of deriving its
frequency from the histogram or retained last-launch fields. The serializer
comment identifies `num_drafts_total` as request verification rounds.

The depth-one sampled-MTP fallback gate now uses port 29941; the depth-two
gate keeps 29935. No other test in `tests/cuda` uses 29941. The original
decode-batch engineering record uses a neutral checkout placeholder.

## Read-only production observation

At 02:04:29 UTC on 2026-09-21, both `/metrics` and `/v1/metrics` lacked
`scheduler.decode_batch` and `scheduler.spec_decode`. The live rank-zero
process and its startup log identify release `27c661e20920`, predating
these counters. The deployment status command also prints the version of
the local development binary (`c4b435750ae8`), which is not the running
release; the live process and HTTP responses establish the deployed version.

Both responses showed one request, one prefilled prompt, eight generated
tokens, four decode steps, no active or queued requests, and no engine
failure. The latest log entry was idle at 01:26:10 UTC. This is insufficient
to estimate production sparse-slot frequency: missing counters are not
zero padding, and no representative traffic interval was observed. No
synthetic traffic, deployment changes, or production restarts were used
for this observation. The two scheduler/service snapshots are retained
locally at `/tmp/dgpp-issue20-production-observation.json`.

After a release containing the counters is deployed, observe rank zero
over representative traffic. Subtract snapshots from the same engine
lifetime and compute `delta(padded_rows) / delta(rows)` only when the
denominator is positive. This yields the fraction of verification rows
spent on padding, not the fraction of launches with any padding or the
frequency of the specific live-slot set `{0, 3}`. Capacity histogram
deltas provide workload context, but cannot distinguish dense and sparse
launches of the same capacity. Idle last-launch fields must not be counted
as repeated sparse launches.

## Validation

The complete native build passed:

```bash
cmake --build --preset ci -j 4
```

All 29 host CTest entries passed in 168.15 seconds:

```bash
CUDA_VISIBLE_DEVICES="" ctest --test-dir build-ci -L host -V -j 1
```

This includes all nine checkpoint gates and all 61 service cases, including
both metrics regressions. Each unit alias reports 229 cases: 226 ran and
three CUDA arena cases were skipped with CUDA hidden. No checkpoint case
was skipped.

The changed C++ ranges pass clang-format 19.1.7. `git diff --check` passes.
Both new documentation links resolve to existing section anchors.
Raw logs are `/tmp/dgpp-issue20-build.log` and
`/tmp/dgpp-issue20-host-tests.log`.

## GPU and fabric maintenance window

The user approved an exclusive maintenance window. Production stopped
cleanly at 02:11:16 UTC; all four operation streams had MD5
`a0bffe724de7894de3ae029d74914ef4`. No GPU processes remained on any node.
Deployment/site files and the original logs were preserved before testing.

The following serial GPU checks passed:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci \
  -R '^(qwen_engine_test|qwen_engine_row_independent)$' \
  --output-on-failure -j 1 --timeout 300
CUDA_DEVICE_MAX_CONNECTIONS=32 DGPP_TEST_FILTER=glm_tp_serving_mtp_depth \
  timeout 300 build-ci/glm_tp_test
DGPP_TEST_FILTER=arena_ timeout 60 build-ci/unit_tests
```

Both Qwen suites passed in 15.42 seconds, exercising scalar/depth-two
telemetry, wide batches, interior-slot padding, slot reuse and transcript
parity. All five matching GLM depth tests passed: scheduled depth two,
depth-two and depth-three eager-feed parity, and the two sampled fallback
gates. The depth-one sampled gate executed 16 replays/16 fallbacks; depth
two executed 10 replays/15 fallbacks. Both matched the eager sampled
speculator on every rank. All three arena cases also passed with CUDA
visible; the earlier host-suite skips are now covered.

The temporary four-rank configuration preserves the production model and
engine settings, binds HTTP to loopback on port 30020, and uses separate
deployment log/staging directories. A baseline server was compiled from
the same application source with the base graph-engine header supplied
through an include overlay. Compiler flags (`-O2 -g -DNDEBUG`), remaining
headers and linked libraries match the candidate. The executable hashes
are recorded in the local artifacts; the overlay changes only the replay
event ordering, apart from diagnostic source locations.

## Four-node results

The baseline/candidate order was A1, B1, B2, A2, with a fresh four-node
world for each. Each boot warmed all four prompts for 64 tokens, then ran
two repetitions at each concurrency 1, 2, 3 and 4, using the first four
`serve_load.PROMPTS`, prompt offset zero, temperature zero and 128 output
tokens per request. GLM's template always enables thinking; `serve_load`
handles its rejected thinking-disable parameter by retrying without it.
`TimedLoad` reconciled exact prompt/token counters after every phase.

Every measured request returned HTTP 200 and 128 tokens. Both metrics
aliases agreed after reconciliation. Checks covered all sixteen histogram
keys, histogram sums versus replay totals, rows versus twice the
capacity-weighted histogram, padding bounds, retained last-launch shape,
and speculative rounds/attempts/accepts. There were no engine or request
failures.

Median engine decode tokens/second, four samples per build/concurrency:

| Concurrency | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 1 | 58.214 | 57.665 | -0.94% |
| 2 | 82.598 | 82.230 | -0.45% |
| 3 | 97.864 | 97.076 | -0.81% |
| 4 | 109.959 | 109.359 | -0.55% |

All medians were within 1%. The second candidate boot had slower individual
phases than the first; the first candidate medians were within 0.5% of the
first baseline at every concurrency. This short comparison does not resolve
sub-percent effects or change any published throughput number.

All C1/C2 texts matched exactly across both builds and repetitions. C3/C4
texts already varied between the baseline's own repeated runs, so those
independent-client phases cannot serve as a bitwise oracle. Additional
jointly admitted `n=3` and `n=4` requests used one prompt, temperature zero
and 96 tokens per choice, twice per build. Every choice's full message
(including reasoning) and finish reason matched exactly between baseline
and candidate and across repeats. The GPU eager/graph parity gates above
provide complementary model-path checks.

The first attempt at those additional controls incorrectly supplied
`enable_thinking: false` directly and received the expected HTTP 400. The
runner stopped the test world and restored production automatically. The
control omitted that unsupported parameter on retry. The valid B2 timing
samples were retained; only the controls were rerun in a separate candidate
world before A2. All nine `serve_api_check.py` checks passed in that world:
stop, multiple choices, logit bias and usage details.

Every test world's four operation streams matched at shutdown:

| World | MD5 on every rank |
| --- | --- |
| A1 | `5e4b3486e9fdf24698378cbaa79dc233` |
| B1 | `e38563dbb75789943978d602f61a21f0` |
| B2 timing | `8f6f92b6f69e4aea1e920e11c440fd98` |
| Candidate controls/API | `6ee769a79b24430671d01a933f480960` |
| A2 and baseline controls | `9b0f0212af356774cdaf585509e8666a` |

Across measured phases, the baseline had 66 padded rows out of 5600
(1.179%); the candidate had 60 out of 5606 (1.070%). These are synthetic
short-request sweeps. They validate using the counters and demonstrate
padding under changing occupancy; they do not estimate production sparse
slot frequency. The production-observation item remains pending a
counter-enabled production release and representative traffic.

## Restoration

Production was finally ready at 02:21:28 UTC, restored to installed release
`0.1.0+g27c661e20920` on all four nodes. Every installed binary retained SHA-256
`37ce93b70552b943b9d6834cd524c4b83163f97cc7d94bc19dc9511449f654a2`.
The deployment JSON, site file and resolved configuration matched their
pre-test bytes. Health passed, two repeated greedy requests returned the
same answer, and the second reused 16 of 18 prompt tokens. Metrics reported
no request or engine failure. No GPU tests ran alongside production.

Raw commands, harnesses, snapshots, per-request outputs, binary hashes and
rank logs are retained under `artifacts/issue20/` (ignored by Git).
