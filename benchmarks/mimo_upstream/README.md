# Reproducing the MiMo comparisons

These Python standard-library probes target an already running DGPP server.
They never deploy or restart it, delete caches, or execute generated tools.
Use an exclusive server: the probes check idle boundaries and request-counter
deltas, and refuse to overwrite existing results. A failed or interrupted run
retains its partial evidence; use a fresh tag for another attempt.

From the repository root, select an API and an output directory:

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

For rank provenance and logs (SSH targets in rank order):

```sh
python3 benchmarks/mimo_upstream/capture.py live native-mtp3 --hosts user@head user@peer
# Run the probes, then stop the world with the normal launcher.
python3 benchmarks/mimo_upstream/capture.py stopped native-mtp3 --hosts user@head user@peer
```

The stopped capture checks nonempty, identical operation-stream MD5s. Both
captures refuse overwrite. They record only MiMo, resident-cache and dense-row
options from process environments, never the full site environment. Capture
before replacing a deployment, since log locations may be reused.

Record the exact Git commit, dirty diff (if any), binary SHA256, deployment
JSON and rank environment alongside the results. Keep startup and warmup
outside timings. Launch configurations using the normal
[operations procedure](../../docs/operations.md#mimo-native-mtp-and-prefill-options),
stopping the old world before changing settings. Disable resident images
with `DGPP_RESIDENT_CACHE=off` if disk capacity cannot hold each layout.

Use identical C2, BF16 KV, BF12 weight residency, shared 131072-token capacity
and 1.5 GiB prefix budget across controls. The historical configuration is in
[config.json](../results/2026-09-23-mimo-native-mtp/config.json).
Set depth and environment explicitly for each fresh tag:

| Tag | Depth | Native MTP | Final-row head | One-head cache-only |
|---|---:|---:|---:|---:|
| control-mtp1 | 1 | off | off | off |
| recursive-mtp3 | 3 | off | off | off |
| native-mtp3 | 3 | on | off | off |
| integrated-mtp3 | 3 | on | on | on |
| control-repeat | 1 | off | off | off |

Run `benchmark.py` and `acceptance.py` for every tag. For a prefill-only
isolation, use depth 1 with head-only, cache-only and both, retaining native
MTP off. Native MTP always performs cache-only history work internally.

```sh
python3 benchmarks/mimo_upstream/summarize.py \
  --reference control-mtp1 --output "$DGPP_MIMO_BENCH_ROOT/summary.json" \
  control-mtp1 recursive-mtp3 native-mtp3 integrated-mtp3 control-repeat
```

The 19-request timing panel includes short solo/concurrent code and JSON,
solo prose/maths, cold 7K/31K prompts, and cold/warm 67K prompts. Client decode
rate is `(completion_tokens - 1) / seconds_after_first_visible_delta`, where
reasoning counts as visible. This is not the published engine-counter metric.
The four separate acceptance probes use **all draft rounds** as denominator.
The order probe intentionally changes arrival order and is diagnostic, not
performance evidence; compare both content and reasoning against solo and
concurrent controls. Use `compare_outputs.py --output variants.json FILE...`
to group exact content/reasoning variants from saved `results.json` and
`order-probe.json` files without contacting the server. Temperature zero does not guarantee row-shape-independent
floating-point results. Add `--delay 0` for back-to-back arrivals or
`--logprobs` to record top-two token probabilities in a separate diagnostic
run; probability tracing is excluded from timing comparisons. Fixed retrieval and tool probes are not a broad
coding-quality evaluation. The 256-token timing outputs may be truncated.

`tool_probe.py` includes the original failing request and validates assembled
SSE arguments without external dependencies. To also reproduce the original
client path, install artifact-agent-orchestration's dependencies, set
`DGPP_HARNESS_ROOT` to that checkout, and invoke `harness_replay.py` using its
Python environment. This optional check uses its `ChatCompletionsClient` and
`RecordingTransport`; all endpoint/output/fixture paths are configurable or
repository-relative. Generated calls are recorded, never executed.

For implementation validation, follow [CONTRIBUTING](../../CONTRIBUTING.md):
build all selected binaries first, run host tests, then run GPU/RDMA tests
serially while serving is stopped. Relevant targets include
`mimo_native_mtp_test`, `mimo_port_test`, `mimo_decode_test`,
`mimo_engine_test`, and the Qwen, GLM-4.7 and DeepSeek engine suites. Run
`mimo_engine_test` both normally and with the three MiMo flags enabled;
run the history test under Compute Sanitizer. Loopback worlds on two machines
do not establish four-node physical fabric coverage. Keep all failed,
skipped and completed attempts in the dated result record.
