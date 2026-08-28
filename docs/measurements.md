# Validated platform measurements

Last updated: 2026-08-27. This document is curated; it is not auto-generated.
Raw commands and the audit-remediation run record are in
`benchmarks/results/2026-08-27-dgx-spark.md`; the M2 KDA correctness and
state-traffic record is `benchmarks/results/2026-08-27-kda-m2.md`. The
generated checkpoint report is `docs/checkpoint_budget.md`.

## Scope and environment

- source base: `a02ea4b` plus the audit-remediation working tree;
- two DGX Spark GB10 nodes, AArch64 Linux `6.17.0-1026-nvidia`;
- CUDA 13.0 compiler, SM 12.1, 48 SMs, 24 MiB L2;
- ConnectX-7 firmware `28.45.4028`, RoCEv2, GID index 3;
- perftest tool version 6.20;
- model server not resident during the 2026-08-27 remediation measurements.

DGX Spark has one coherent 128 GB LPDDR5x pool, not discrete GPU VRAM plus
host RAM. `cudaMalloc` and `cudaHostAlloc` draw from that pool. The words
“device” and “host-pinned” below identify mappings/access paths.

## Network topology — closed

One ConnectX-7 NIC connects to the SoC through two independent PCIe Gen5 x4
links. The one cabled physical QSFP port exposes two active RoCE lanes:

| lane | RoCE device | PCIe function | state | reported line speed |
|---|---|---|---|---:|
| 0 | `rocep1s0f0` | `0000:01:00.0` | active | 200 Gb/s |
| 1 | `roceP2p1s0f0` | `0002:01:00.0` | active | 200 Gb/s |

The `f1` functions map to the uncabled second QSFP port and are down. The two
active lanes share one physical 200 Gb/s port and are not independent failure
domains. This mapping agrees with the [NVIDIA ConnectX-7 networking guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).

### Authoritative perftest results

`ib_write_bw`: 8 MiB messages, four QPs, depth 128, duration mode, relaxed
ordering on, RC/RoCEv2/MTU 4096.

| run | lane 0 | lane 1 | aggregate |
|---|---:|---:|---:|
| isolated | 107.64 Gb/s | 107.64 Gb/s | — |
| concurrent | 97.98 Gb/s | 98.05 Gb/s | **196.03 Gb/s (24.50 GB/s)** |

The old 105 Gb/s number was a valid one-lane result but not a node ceiling.
Bulk collectives must stripe both lanes.

`ib_send_lat`: 2-byte messages, 10,000 iterations.

| lane | min | typical | average | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|
| 0 | 2.66 µs | 2.72 µs | 2.72 µs | 2.80 µs | 2.98 µs |
| 1 | 2.40 µs | 2.46 µs | 2.57 µs | 2.97 µs | 3.04 µs |

Both lanes passed `ping -M do -s 8972`, validating 9000-byte IP frames.
All three other cluster nodes answered on both fabric subnets during the
reachability matrix.

### Flow-control state

Node-visible state on both active interfaces:

- Ethernet pause RX/TX enabled;
- DCB priority PFC disabled for priorities 0–7;
- `rx_out_of_buffer=0` at inspection;
- cumulative priority-0 discards are nonzero.

The switch PFC/ECN/global-pause policy is not available through node sudo.
The paired tests reached the physical-link ceiling without a congestion
symptom, so switch access is not a sign-off requirement. If a future four-node
run shows drops, retries, unstable throughput, or latency spikes, collect
before/after pause, discard, retry, and out-of-buffer counters and inspect the
switch configuration. No switch policy is inferred here.

## Project RC protocol tool — repaired and validated

The audit found unsignaled sends mislabeled as signaled and CQ polls writing
multiple completions into one `ibv_wc`. `micro_ibv_smoke` now:

- marks the required SENDs `IBV_SEND_SIGNALED`;
- supplies completion arrays matching every poll count;
- propagates link-thread, timeout, and sink failures;
- validates request sizes and socket failures;
- frees QP, CQ, MR, PD, context, and buffer resources after a connection;
- maps repeated peer/device arguments by position for two-lane tests;
- offers `--once` for bounded automation.

Cross-node results:

| test | result |
|---|---:|
| 1,000-iteration lane-0 SEND echo | 3.25 µs minimum, 4.17 µs mean one-way |
| 1 MiB signaled SEND, window 64 | 95.6 Gb/s |
| same SEND test on both lanes | **184.0 Gb/s aggregate** |

The SEND tool intentionally exercises receive WQEs/CQEs and signals every
message, so perftest RDMA write—not this result—is the link ceiling.
An earlier run before the final error-path hardening measured 3.29/3.47 µs
minimum/mean and 185.0 Gb/s dual-lane; both runs are retained in the dated
record rather than selecting only the faster result.

## NIC DMA → GPU payload visibility — closed for this stack

The new `micro_ibv_smoke verify` mode registers a `cudaHostAlloc` receive slab
with ibverbs and launches a GPU consumer. For each iteration, the peer sends a
changing 64-byte payload followed by a 64-byte sequence doorbell as ordered RC
SENDs. The GPU observes the NIC-written doorbell with a system-scope acquire,
hashes the payload, and publishes a system-scope release acknowledgement.

| lane | iterations | mismatches | timeouts |
|---|---:|---:|---:|
| 0 | 10,000 | 0 | 0 |
| 1 | 10,000 | 0 | 0 |

The final hardened binary then passed another 1,000/1,000 payloads on each
lane. The larger runs validate the visibility contract; the bounded reruns
validate that later argument, CQ-error, timeout, and cleanup changes did not
regress it.

This is empirical validation of the exact kernel, driver, and firmware stack;
NIC DMA is outside the C++ abstract-machine synchronization model. The test is
a deployment regression after CUDA, kernel, mlx5, firmware, or topology
changes. CPU CQ polling remains in the production design for errors and slot
credits even though the GPU consumes the payload directly.

## Unified-memory and zero-copy results

Final `micro_zerocopy` regression run, 256 MiB buffers and best of its grid
sweep:

| pattern | GB/s |
|---|---:|
| GPU read, `cudaMalloc` | 240.8 |
| GPU read, `cudaHostAlloc` | **252.5** |
| GPU write, pinned | 210.8 |
| pinned → device copy | 59.6 |
| device → pinned copy | 59.6 |
| one-thread CPU memcpy → pinned | 27.4 |
| flag round trip | 1.02 µs min, 1.12 µs p50, 1.41 µs p99 |

The previous `volatile bool` CPU-thread stop flag was a data race; it is now a
`std::atomic<bool>`. Doorbell slots are explicitly initialized, and the CUDA
protocol uses system-scope atomics.

CPU-writer proxy cases touch separate buffers; the old same-buffer CPU/GPU
race was removed. It was not used for the production margin.

### Real incoming-RDMA contention

In a paired earlier run with a 251.5 GB/s idle pinned baseline, two remote
`ib_write_bw` streams each sustained 98.04 Gb/s into this node:

| pattern | GB/s | change from idle |
|---|---:|---:|
| GPU read, pinned | **218.4** | −13.2% |
| GPU read, device mapping | 208.6 | −13.1% |
| flag p50 / p99 | 1.12 / 1.42 µs | effectively stable |

Thus registered pinned receive slabs remain faster than a staging copy, but
dual-lane saturation consumes measurable shared-memory bandwidth. Models that
assume sustained dual-lane ingress provisionally derate simultaneous GPU
memory bandwidth by 15%. This is a stress-case planning value, not a fixed
inference penalty or an upper bound for other traffic patterns; M5 replaces it
with measurements of the real collective schedule. CPU-writer patterns remain
useful stress proxies but are no longer the evidence for NIC contention.

## Flag protocol regression

`src/kernels/flag_protocol.cuh` now provides:

- system-scope acquire/release operations for sequence words;
- payload/stamp writes ordered before acknowledgement publication;
- an inactivity deadline reset after every observed message;
- a reserved orderly-stop sequence.

CTest covers sequential acknowledgements, payload hashes, traffic continuing
for longer than the watchdog budget, quiet-time expiry, and subsequent device
use. The corrected flag-protocol test passes on GB10 in about 1.3 seconds.

## Checkpoint and traffic audit

The current generated report validates:

- 328.33 GB / 305.78 GiB, 62 shards, 76,108 tensors;
- 1.127 GB of vision tensors excluded from text decode;
- zero unmatched tensor names;
- 37,338/37,338 FP8 matrices paired with F32 inverse scales of exact
  128×128-block geometry;
- 22.415 GB/token of unique active base-text weights and 23.420 GB/token of
  aggregate physical traffic after replicated-module reads;
- TP=4 expected synchronized critical-path traffic 7.457 GB/token and a 32.42 ms
  weight-bandwidth floor at 230 GB/s.

The mean-rank value is 5.855 GB/token, but it is not used as a synchronized
decode prediction. Representative real router traces are an M4 exit gate.

## KDA operator state traffic (M2)

`kda_bench` (idle node, 50 timed iterations after 5 warmups; full record in
`benchmarks/results/2026-08-27-kda-m2.md`):

| case | mean | state bytes/step | effective |
|---|---:|---:|---:|
| recurrent kernel, decode, TP=1 (64 heads, 34 layers) | 3.301 ms | 272.0 MiB | 86.4 GB/s |
| recurrent kernel, decode, TP=4 rank (16 heads) | 1.123 ms | 68.0 MiB | 63.5 GB/s |
| full KDA layer, decode (T=1), TP=1 | 1.324 ms | — | — |
| full KDA layer, prefill chunk (T=2048), TP=1 | 23.787 ms | — | — |

The decode recurrence is latency-bound at these sizes (256/64 blocks on 48
SMs, 34 sequential per-layer launches), so the 230 GB/s planning floor does
not bind it; batching layers into single launches is the identified
optimization for M9. The full-layer decode step is weight-bandwidth
dominated (~0.9 ms of 1.32 ms streams the 204 MB fused projection), which
is the intended decode-traffic shape. These are correctness-first M2
numbers, not performance claims.

## cuBLASLt best-of-heuristics results

`micro_gemm_peak` now warms all SMs before measurement, times every valid
heuristic returned by cuBLASLt (up to 16 requested), and selects the fastest
candidate by the median of three timed batches rather than timing the first
heuristic only. Two final process runs produced these ranges:

| model shape | FP8 | BF16 |
|---|---:|---:|
| attention output decode `1×4096×16384` | 231.8–238.3 µs | 578.3–581.4 µs |
| dense gate/up decode `1×24576×4096` | 406.9–419.2 µs | 832.4–835.7 µs |
| lm head `1×154880×4096` | 2,798.0–2,844.3 µs | 5,364.7–5,528.2 µs |
| attention output prefill, m=2048 | 195.7–200.9 TFLOP/s | 93.4–94.9 TFLOP/s |
| dense gate/up prefill, m=2048 | 220.8–220.9 TFLOP/s | 95.6–95.8 TFLOP/s |

Decode shapes whose weight set fits L2 can report implausible effective DRAM
bandwidth; those values are cache-warm diagnostics. This benchmark is a
cuBLASLt heuristic baseline, not a hardware peak claim.

## Direct-device MR probe

`micro_gdr_probe` reports direct `cudaMalloc` registration as unsupported on
this platform, consistent with NVIDIA's [DGX Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).
The binary now treats the expected unsupported result as a completed probe and
states the correct fallback: registered host memory consumed directly by the
GPU, not a pinned bounce copy.

## Build and test validation

The warning-as-error CI build consumes `DGPP_WERROR=ON`; the earlier unused
cache variable is fixed. CTest contains host, CUDA, synthetic-model, and Python
checkpoint-audit tests. ASan and UBSan are independent cache-variable presets.
The final validation results are listed in the dated run record.

Both CUDA milestones additionally hold their suites clean under
`compute-sanitizer`: the kernel phases ran full memcheck, racecheck, and
initcheck; the DSA layer phase runs full-suite memcheck (it caught two
undersized test buffers that made a graph test pass vacuously) plus targeted
racecheck/initcheck on the tests exercising new kernel shapes — the
real-geometry racecheck pass is deliberately skipped because its runtime is
dominated by vendor cutlass GEMM kernels covered by the kernel-phase tests
(reasoning recorded with the results). This is a standing gate for new
kernels, not a one-off: sanitizer rounds have surfaced a speculated
out-of-bounds load (short-circuit guards do not protect loads once nvcc
predicates the branch), shared-memory reuse races that functional tests
passed by scheduling luck, and the vacuous-test buffers above. Details are
pinned in `DESIGN.md` §12.

DSA layer timing lives in
`benchmarks/results/2026-08-28-dsa-m3-layer.md`: decode 2.24 ms/layer at
65k context (114 GB/s effective; the projection GEMMs alone are ~1.0 ms,
i.e. at the 230 GB/s floor), prefill 6,443 tok/s at T=2048, and the full
profiled optimization log from the 13.26 ms starting point.

## Historical M1 observations

The base commit recorded synthetic `gpt_doll` graph/eager parity, stable graph
replay, and a ten-minute allocation-stability soak. Those observations apply
to the synthetic model only. They are not evidence for full GLM correctness or
performance and are not carried into a model throughput target.

## Future measurement scope

The measurements in this report close the questions covered by the completed
M0/M1 exit criteria. Later milestones still require their specified
workload-specific correctness, stability, and performance measurements,
including four-node CollectiveBus and full-model validation. Within those
runs, switch policy and counter deltas are diagnostics if congestion symptoms
appear. Comparisons with other inference engines are optional and are not an
implementation or performance gate.
