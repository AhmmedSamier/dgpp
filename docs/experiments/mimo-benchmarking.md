# Comparing MiMo serving changes

The original MiMo comparison used a local 19-request corpus with the probe
scripts under `benchmarks/mimo_upstream/` (their invocation is in the last
section). Its raw measurements are not tracked. For new comparisons, use the
maintained serving probes below and retain any additional prompts or harness
fixtures with your results. These scenarios do not reproduce the original
panel's measurements.

Reserve an exclusive server and use a fresh output path for each build and
configuration. The probes target an already running service. Keep startup and
warmup outside timings; launch configurations using the normal
[operations procedure](../operations.md#mimo-native-mtp-and-prefill-options),
stopping the old world before changing settings.

## Maintained serving probes

For a focused solo/mixed-arrival comparison, run from the repository root:

```sh
python3 scripts/serve_admission_bench.py \
  --url http://127.0.0.1:18081 \
  --label COMMIT-CONFIGURATION \
  --out /path/to/experiment/admission.json \
  --scenarios solo_decode solo_prefill mixed_short mixed_long --trials 3
```

This records request timings, usage and before/after metrics. Select only the
scenarios relevant to the change. Prompt size, output lengths, arrival count
and spacing are configurable; see `--help` and the
[focused comparison procedure](../testing.md). Choose a new output filename
for each run, since the probe replaces that file as results accumulate.

Use [serve_api_check.py](../../scripts/serve_api_check.py) with `HOST PORT` for
request-field and prefix-cache usage checks. For task quality, use
[serve_eval.py](../../scripts/serve_eval.py) with an explicit task subset,
sample count and output directory. Neither substitutes for a saved failing
tool-call request: retain that request and validate its assembled streaming
arguments separately. Record generated calls without executing them.

Record the exact Git commit, dirty diff if any, binary SHA256, deployment
JSON and relevant rank environment alongside the results. Capture metrics,
startup logs and each rank's final operation-stream hash before replacing the
deployment. All ranks must agree on the operation stream. Follow the
[testing guide](../testing.md) for artifact handling and service restoration.
Disable resident images with `DGPP_RESIDENT_CACHE=off` if disk capacity cannot
hold each layout.

## Historical experiment controls

The original comparisons held C2, BF16 KV, BF12 weight residency, shared
131072-token capacity and a 1.5 GiB prefix budget constant. The following
matrix describes those experiments, not current deployment defaults:

| Tag | Depth | Native MTP | Final-row head | One-head cache-only |
|---|---:|---:|---:|---:|
| control-mtp1 | 1 | off | off | off |
| recursive-mtp3 | 3 | off | off | off |
| native-mtp3 | 3 | on | off | off |
| integrated-mtp3 | 3 | on | on | on |
| control-repeat | 1 | off | off | off |

The switches are `DGPP_MIMO_NATIVE_MTP`, `DGPP_MIMO_PREFILL_LAST_HEAD` and
`DGPP_MIMO_MTP_CACHE_ONLY`; set them consistently on every rank. For a
prefill-only isolation, use depth 1 with head-only, cache-only and both,
retaining native MTP off. Native MTP performs cache-only history work
internally. Current builds also use the shared adaptive scheduler: match its
targets and tuning between controls, or set both latency targets to zero and
match explicit busy/idle budgets to isolate fixed budgeting.

The original 19-request timing panel included short solo/concurrent code and
JSON, solo prose/maths, cold 7K/31K prompts, and cold/warm 67K prompts. Its
client decode rate was `(completion_tokens - 1) / seconds_after_first_visible_delta`,
where reasoning counted as visible. That differs from the published
engine-counter metric. Acceptance used **all draft rounds** as denominator.
Timing outputs capped at 256 tokens could be truncated; retrieval and tool
probes did not establish broad coding quality.

For arrival-order diagnostics, compare content and reasoning against isolated
and concurrent controls, then check cancellation, cache reuse and recovery.
Temperature zero does not guarantee identical floating-point results across
chunk or batch shapes. The admission probe's `--record-text` retains outputs;
`--logprobs` adds probability tracing and can change head dispatch, so use it
in a separate diagnostic run excluded from timing comparisons.

## Implementation checks

Follow [CONTRIBUTING](../../CONTRIBUTING.md): build the selected binaries
first, run host tests, then run relevant GPU/RDMA tests serially while serving
is stopped. MiMo targets include `mimo_native_mtp_test`, `mimo_port_test`,
`mimo_decode_test` and `mimo_engine_test`. Changes to shared execution also
need focused coverage from another engine family; choose that coverage based
on the affected code rather than rerunning every model's benchmark corpus.
For changes to the MiMo options, run `mimo_engine_test` normally and with all
three flags enabled, and run the history test under Compute Sanitizer.
Loopback worlds on two machines do not establish four-node physical fabric
coverage. Retain failed, skipped and completed attempts in local or external
artifacts, and summarize the evidence and limitations in the PR.

## Original comparison probes

The probes under `benchmarks/mimo_upstream/` are Python standard-library
scripts that target an already running server. They never deploy or restart
it, delete caches, or execute generated tools. Use an exclusive server: the
probes check idle boundaries and request-counter deltas and refuse to
overwrite existing results. From the repository root, select an API and an
output directory, then run the probes for a tag:

```sh
export DGPP_MIMO_BENCH_URL=http://127.0.0.1:30001/v1
export DGPP_MIMO_BENCH_ROOT=/path/to/experiment
python3 benchmarks/mimo_upstream/benchmark.py native-mtp3
python3 benchmarks/mimo_upstream/acceptance.py native-mtp3
python3 benchmarks/mimo_upstream/cache_reuse.py native-mtp3
python3 benchmarks/mimo_upstream/tool_probe.py native-mtp3
python3 benchmarks/mimo_upstream/quality.py native-mtp3 2800 6000
python3 benchmarks/mimo_upstream/order_probe.py native-mtp3 --repeats 3
```

`capture.py live TAG --hosts ...` and `capture.py stopped TAG --hosts ...`
(SSH targets in rank order) record rank provenance and logs before and after
the world is stopped; the stopped capture checks that the ranks' operation
streams match. Record the exact Git commit, binary SHA256, deployment JSON and
rank environment alongside the results. The original tags were `control-mtp1`
(depth 1), `recursive-mtp3` (depth 3), `native-mtp3` (depth 3, native MTP),
`integrated-mtp3` (native MTP with the final-row head and one-head cache-only
history) and `control-repeat`; `summarize.py --reference control-mtp1
--output SUMMARY TAG...` compares them. The 19-request timing panel covers
short solo/concurrent code and JSON, solo prose/maths, cold 7K/31K prompts
and cold/warm 67K prompts; its client decode rate is
`(completion_tokens - 1) / seconds_after_first_visible_delta`, not the engine
counter. `order_probe.py` changes arrival order and is diagnostic only;
`compare_outputs.py --output variants.json FILE...` groups exact
content/reasoning variants offline. `tool_probe.py` carries the original
failing tool request and validates the assembled SSE arguments; the optional
`harness_replay.py` reproduces the original client path when
`DGPP_HARNESS_ROOT` points at that client's checkout.
