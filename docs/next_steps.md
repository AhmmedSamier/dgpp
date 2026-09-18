# Remaining work

This list reflects the implementation as of 2026-09-11. The v1 work,
release packaging, cluster configuration, NVFP4 support and MTP depths
1–3 are implemented. The earlier cost estimates for building those
features are no longer outstanding work. Measurements below refer to the
recorded GLM-5.3-FP8 configuration unless another model is named.

## Mixed-workload latency

The scheduler completes one admission's prefill before decoding the live
requests in that tick. The v1 measurements were 1.7 s for a 2,048-token
prompt and 33.7 s for 32,768 tokens; other requests pause during that work.
Interleaving prefill chunks with decode would reduce these interruptions.

The model already supports chunked prefill, but the scheduler needs a
budget per tick, reservations for partially prefilled requests and a
journaled chunk schedule. Validation must cover cancellation during
prefill, rank agreement, and cached/cold equivalence when chunks span
multiple ticks. Measure decode latency and time to first token under the
same mixed workload before and after the change.

## Operations and API coverage

The service has periodic throughput logs, JSON metrics at `/metrics`
(with `/v1/metrics` retained for compatibility), operation-stream checks
and process-failure handling. Structured
request-level logs and Prometheus exposition would make those signals
easier to monitor. Deployment currently uses `scripts/dgpp-cluster`;
automatic supervision is a separate deployment choice.

The remaining request fields include `user`, `store`, `metadata` and
`service_tier`, which currently return errors, and the `developer`
message role. Multiple choices are supported on Chat Completions but
remain unsupported on the legacy completions route. Any extension should
define its behavior and add request-level checks.

Process death closes the journal sockets and is detected quickly. A silent
node loss relies on the bus watchdog; a journal heartbeat could shorten
that delay. Rate limits, request deadlines and prompt-length policies
would also help control load. TLS and authentication can be supplied by a
reverse proxy.

## Decode and prefill performance

| area | current behavior | remaining work |
|---|---|---|
| GLM-5.3 and Qwen batching | At most eight batched rows; deeper MTP uses scalar graphs | Generalize model state and draft chains to wider batches, then measure whether extra rows offset repeated weight reads |
| GLM-4.7 deeper MTP | Runtime batch shapes support deeper drafting | The measured depth-2 concurrent workload lost to depth 1; further work needs a workload-specific benefit |
| Qwen decode | Optimized GR and projection kernels, sampled drafting, optional FP8 dense weights | Investigate the remaining launch and selection costs identified in the optimization study |
| Prefill kernels | Grouped expert kernels and chunked model execution | Measure attention, recurrence, GR fusion and communication overlap separately for each family |
| Quantization | NVFP4 experts and Qwen FP8 dense projections are implemented | Evaluate additional formats or placements with teacher-forced quality checks and measured memory/latency gains |

See [the Qwen optimization study](qwen38_optimization_plan.md),
[the NVFP4 study](nvfp4_plan.md) and
[benchmarks](benchmarks.md) for experiment results and reproduction
commands. MTP depth 2 is already available; whether it improves tokens per
second depends on acceptance and the cost of each extra verification row.

## Admission and prefix caching

Grow-on-demand admission reserves a window beyond the prompt and extends
it before decoding. At exhaustion it ends the youngest request. Preemption
with later recomputation would require a resume policy and enough prefill
capacity to recover the discarded state.

The prefix cache uses LRU eviction among eligible entries. The v1 capacity
sweep showed a sharp drop in hit rate for cyclic working sets larger than
the arena. Protecting recent conversation entries or incorporating reuse
frequency could improve that behavior. Arena sizing should be evaluated
with the memory plan and an actual conversation workload.

Prefix snapshots do not survive a restart or move between instances.
Persistence would need a versioned state format, checkpoint and numerics
validation, and recovery of shared block ownership. Its value depends on
restart frequency and the cost of cold prefill.

## Reliability and larger deployments

The release has process-failure drills and a one-hour mixed-workload soak.
Longer runs and silent-node-loss drills would extend that evidence.
Instrumented loopback graph tests also need budgets that accommodate
sanitizer overhead without hiding a real deadlock.

There is no mid-run failover or standby rank set. Adding one would require
state replication and coordinated reconstruction of the collective world.
Data-parallel routing and shared prefix keys are separate extensions for
deployments with more nodes. Vision and audio execution remain outside the
implemented text-serving path.
