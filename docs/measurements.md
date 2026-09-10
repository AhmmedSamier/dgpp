# Validated platform measurements

Last updated: 2026-08-27. This document is curated; it is not auto-generated.
Raw commands and the audit-remediation run record are in
`benchmarks/results/2026-08-27-dgx-spark.md`; the M2 KDA correctness and
state-traffic record is `benchmarks/results/2026-08-27-kda-m2.md`. The
generated checkpoint report is `docs/checkpoint_budget.md`.

## Scope and environment

- source base: `e9a76c8` plus the audit-remediation working tree;
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

## Layer streaming and real route traffic (M4)

`glm_stream_check` over the full real checkpoint (full record in
`benchmarks/results/2026-08-28-glm-m4-assembly.md`; idle node, cold page
cache):

- all 46 layers streamed one at a time through the resident loader:
  KDA/dense 408.3 MiB, DSA/MoE 7,275.7 MiB, KDA/MoE 7,204.2 MiB,
  MTP 7,338.3 MiB per layer; peak layer 7.17 GiB, globals 2.363 GiB
  (embed + lm head + final norm) — peak device footprint ≈ 9.6 GiB,
  so single-node correctness work fits without the TP placement;
- streaming total 316.4 s (per MoE layer ~7 s reading 7.2–7.3 GiB —
  ~1 GB/s from cold disk; loader overhead is not the bottleneck, and
  second passes drop once the page cache holds the shards). The loader's
  byte formula reconciles with the allocator on every load.

The first real route trace (272-token technical-prose prompt, deterministic
across re-runs) replaces the uniform-expert null model in the §3 traffic
numbers: busiest-rank experts/layer 3.541 (uniform 3.515), corrected
critical path 7.484 GB/token = 32.54 ms at the 230 GB/s planning floor
(uniform 7.457 GB/token, +0.4%). A 12-layer close-out re-run reproduces
the occupancy (+0.3%): the router input distribution is stable across
depths. Sampling other prompt classes is future work; the traffic tool
consumes their traces unchanged.

## Cost of one recorded collective node (2026-09-09, four nodes)

The Qwen plan's Q0 question (docs/qwen38_flash_next_plan.md D3): what does
one more all-reduce per residual site cost the decode step? Measured by
`glm_gen_check --gr-probe N` (one extra `[T x 10240]` bf16 all-reduce over a
scratch buffer after the attention fold of the first N layers, N = 32 within
the graph's 128-node budget) on `unsloth/GLM-5.3-Flash-FP8`, 300 steps,
transcripts identical with and without the probe
(`scripts/fabric_gr_probe.sh`, runs in `build-ci/fabric-runs/gr_probe_2026-09-09/`):

| step | base ms/step | +32 nodes | per node |
|---|---:|---:|---:|
| T=1 graph replay, TP=4 | 29.98 | 31.49 | **47 µs** |
| MTP graph replay (T=2 verify rows), TP=4 | 40.40 | 42.32 | **60 µs** |

The bus alone (`bus_check allreduce`, eager, 8 KiB, `scripts/fabric_bus_probe.sh`):
p50 **30 µs at world 2**, **39 µs at world 4** (min 21 / 33 µs). The graph
node's cost is the eager latency plus the replay's skew and fold; scaling
the in-graph number by the eager ratio puts the two-node cost near 36 µs
(T=1) and 46 µs (MTP). None of this is hidden behind weight streaming: the
probe nodes sit where a boundary does, and the step grew by the full amount.

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

## GLM-4.7 (nvidia/GLM-4.7-NVFP4) serving on four nodes (2026-09-10)

`scripts/fabric_glm4_serve.sh deploy/cluster_glm47.json` (MTP) and
`deploy/cluster_glm47_t1.json` (T=1), the resident image warm on every rank,
the 202,752-token pool (77.4 GiB per rank: weights 52.8, K/V 23.3).

| reading | value |
|---|---|
| boot from the resident image | 18–20 s (the first boot captures it: ~3 min); 14 graph variants warm-captured in 13 s |
| T=1 decode, one request | 49.0 ms/step (the bytes floor ~40 ms: 9.7 GB per rank per step, 6.3 GB of it the BF16 attention projections modelopt left unquantized; 186 collectives ~8 ms) |
| MTP decode, one request | 60–61 ms/pass at 1.86–1.99 tokens/pass (draft acceptance 79–98 %), 31–33 ms/token |
| four live requests (the eval) | 110–145 ms/step batched, ~8–12 tokens/s per request |
| prefill, short prompts (31–150 tokens) | 400–1000 ms per prompt, 5–13 ms/token (v1 warp-per-row attention; the tensor-core form is open) |
| MTP == T=1 transcripts | identical, 4 of 4 prompts; op streams identical across the four ranks |
| eval (thinking off, `serve_eval.py --no-think`) | gsm8k 60/60, HumanEval 39/40, schema extraction 30/30; no answer truncated |
| API check | every case (stop, n, logit_bias, usage, reasoning tokens) |

Two findings on the way: the partial RoPE is transformers' half-split
rotate_half (the interleaved reading passed every self-written gate and
looped the model after ~30 tokens; caught by `tools/glm4_torch_reference.py`
against transformers' own layer code on the real weights — layer-0 relative
l2 0.036 -> 0.0025), and the draft block takes the post-final-norm hidden
(11–46 % acceptance with the pre-norm residual, 79–98 % after).

### MTP depth 2 on GLM-4.7 (2026-09-10)

The depth-2 chain (the session core's `session_draft_chain` /
`session_graph_capture_draft_chain` on the hidden-window families,
`glm_spec_chain_row_window`) measured with `deploy/cluster_glm47_d2.json`
against depth 1, one greedy request at a time, transcripts identical to
depth 1 on every short prompt (4/4):

| prompt (256 tokens) | depth 1: ms/pass, tok/pass, ms/token | depth 2: ms/pass, tok/pass, ms/token, p1 / p2 | tokens/s |
|---|---|---|---|---|
| chat | 61, 1.86, 32.5 | 72, 2.32, 31.2, 85 % / 48 % | +4 % |
| code | 61, 1.93, 33.0 | 72, 2.52, 28.7, 92 % / 60 % | +13 % |
| math | 62, 1.95, 31.3 | 72, 2.55, 28.4, 94 % / 62 % | +10 % |
| json | 61, 1.99, 31.1 | 73, 2.60, 28.0, 96 % / 65 % | +11 % |
| 6,525-token prompt, 300 tokens | 66, 1.86, 35.4 | 79, 2.35, 33.7, 85 % / 52 % | +5 % |

The chain row costs ~11 ms per pass (the second draft row's expert bytes
and its own draft-block run) and the second draft stands 48–65 % of the
time, so single-stream throughput gains 4–13 %. (At this reading the
engine built no row-batched graph past depth 1: two live requests ran
their scalar graphs back to back at 143–147 ms per pass each; the batched
chain followed the same day — the next section.) Prefill of the
6,525-token prompt: 11.5 s (1.77 ms/token) at both depths.
The other families after the chain landed (the session core is shared):
Qwen3.8-Flash-Next MTP 26.3–26.4 ms/step, transcripts identical to round 5;
GLM-5.3-Flash `fabric_glm_regression.sh` MTP 40.39 / T=1 29.94 ms/step,
prefill 548 / 1,412 / 6,284 ms, the same rank-consistency hashes as the
2026-09-09 baseline (`build-ci/fabric-runs/*_post_d2_2026-09-10`).

### Batched MTP depth 2 on GLM-4.7, on the runtime decode rows (2026-09-10)

The fixed decode batch's row ceiling is the recipe's shape since this
afternoon — `max_concurrency x (1 + mtp_depth)`, floored at 8 — so the
depth-2 recipe (`deploy/cluster_glm47_d2.json`, 4 slots) boots a 12-row
world (`serve: decode rows 12`; the bus's latency slot 120 KiB, 128
sampling candidates still fit) with the 2-, 3- and 4-slot batch families
at 6, 9 and 12 rows, each carrying every slot's chain row in one draft-
block run (`session_graph_capture_draft_chain_batch`). The GEMM seam
lowers every bf16 decode call up to those rows to the row-independent
GEMV core (`CublasLtGemm::set_decode_rows`), so a batched request's rows
keep the scalar reduction order — the first 9-row batch had fallen to a
cuBLASLt algorithm with its own order and flipped a near tie in the
engine gate. The reading, `scripts/fabric_glm4_load.sh` (distinct
long-answer prompts, thinking off, greedy, 320 tokens each; the client's
aggregate tokens/s over the phase's decode span, rank 0's stats lines
inside the phase):

| live requests | depth 1: ms/pass, tok/step/req, tok/s (per request) | depth 2 batched: ms/pass, tok/step/req, tok/s (per request) | tokens/s |
|---|---|---|---|
| 1 | 60, 1.89, 31.0 | 72, 2.31, 32.2 | +4 % |
| 2 | 79, 1.85–1.88, 44.5 (22.3) | 123–125, 2.3–2.6, 37.0 (18.5) | −17 % |
| 4 | 139–143, 1.74–1.93, 47.5 (11.9) | 198–204, 2.2–2.4, 42.8 (10.7) | −10 % |

The second draft stands 41–66 % of the time in the batch as it does
alone, but the pass grows faster than the tokens: by fixed rows the pass
is 72 ms at 3, 79 at 4 (one GEMV chunk), 123 at 6 and 140 at 8 (two
chunks), 200 at 12 (three) — each 4-row chunk past the first re-reads
the 6.3 GB of BF16 attention projections per rank (+45–60 ms), a row
within a chunk costs 7–8 ms (its K/V reads and its distinct experts). So
batched depth 2 is correct, isolated and slower than depth 1 under
concurrency on this family; depth 2 stays a single-stream setting
(`cluster_glm47.json` keeps depth 1). The lever for concurrency at
either depth is the attention projections' bytes per pass: wider GEMV
chunks (a 6-row chunk at K = 5120 is 61 KB of dynamic shared memory, an
8-row one 80 KB — one block per SM, to be measured), or a row-independent
tensor-core kernel for the bf16 projections that reads the weights once
for up to 16 rows; quantizing the projections halves the bytes but not
the passes (the fp8 / fp4 GEMV cores chunk the same way). Transcript
isolation: the engine gate pins the batched depth-2 rows bitwise to the
eager engine's, and on the fabric a prompt's greedy answer alone (the
scalar graph) is its answer beside three others (the 12-row batch) —
`serve_load.py --isolation`; the op streams are identical across the
four ranks in both worlds.

### The existing families after the GLM-4.7 work (2026-09-10)

`scripts/fabric_glm_regression.sh` on `unsloth/GLM-5.3-Flash-FP8` against
the 2026-09-09 baseline (`build-ci/fabric-runs/glm_baseline_2026-09-09`):
every reading's rank-consistency hash identical to the baseline's (the
transcripts bitwise unchanged); MTP chat 40.49 ms/step at 72.0 % acceptance
(baseline 40.35–40.50), T=1 29.97 ms/step mean, p50 30.0, p99 31.0 (29.81–
29.96 / 30.0 / 31.0), steady-state prefill 549 / 1417 / 6286 ms at 512 /
2048 / 8192 tokens (549 / 1405–1412 / 6258; a second sample 551 / 1428 /
6303 — the spread of the reading). `moe_slot_bench` at the GLM-5.3-Flash
per-rank shape: fp4 201 / 320 / 883 us at 1 / 2 / 8 rows (recorded 209 /
334 / 929), fp8 281–286 / 491 / 1489 (273 / 469 / 1420, the same decode
kernels — run-to-run spread on this box, the fabric step above is the
arbiter). `scripts/fabric_qwen_serve.sh` on `Qwen/Qwen3.8-Flash-Next-FP8`
against the round-5 transcripts of 2026-09-09: MTP world (`deploy/
cluster_qwen.json`) 26–27 ms/pass at 1.5–2.0 tokens/pass, T=1 world
21.8–21.9 ms/step (recorded 22.0), transcripts identical 4 of 4 in both,
API checks clean, op streams identical across the ranks. The full ctest
after every change: 62 of 62.

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
