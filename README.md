# DGPP Engine

DGPP is an early-stage C++/CUDA inference-engine project for serving the text
path of `unsloth/GLM-5.3-Flash-FP8` across four NVIDIA DGX Spark systems.

Current status: M0–M4 prototype. The repository implements platform and RoCE
validation tools, checkpoint/shard inspection, core CUDA runtime utilities, a
synthetic transformer testbed, the KDA linear-attention operators with their
per-request state manager and reference-dump harness, the complete
DSA/MLA sparse-attention path: indexer compression, deterministic pooled
top-k, split-KV absorbed attention, a dual-precision host oracle, the blocked
state pool with shared block tables, layer orchestration (graph-capturable
decode, pool-tiled prefill), and the reference-dump parity harness; and the
full single-node GLM diagnostic forward: config adapter and binding
validation, mHC residual streams, the streaming resident loader (one layer
resident at a time), block-scale-aware GEMMs, MoE routing with route-trace
capture, and a curated real-checkpoint parity suite where every selection
flip — router or head — is certified as a measured near tie. M5 (four-rank
tensor parallelism) is in flight: the TCP control plane — epoch-based
roster/startup, rank health, eviction on death or heartbeat deadline — is
validated on all four nodes, and the CollectiveBus data plane (RC QPs per
pool per peer-lane on both f0 lanes, per-class slot pools, credit-grant RDMA
writes, watchdog-bounded GPU consumers) is unit- and sanitizer-tested and
validated two-node: verified payload integrity, dual-lane striping, and
latency-under-bulk contention on the fabric. It does
**not** yet implement replicated-boundary collectives (the attention/FFN
all-reduces of deliverable 3),
the TP placement,
prefix cache, MTP generation loop, or OpenAI-compatible server. See
`PLAN.md` for milestone status.

## Documentation

- `DESIGN.md` — validated architecture and state/collective contracts
- `PLAN.md` — implementation status, exit gates, and audit closure matrix
- `docs/checkpoint_budget.md` — generated checkpoint and decode-traffic audit
- `docs/measurements.md` — curated, reproducible platform results
- `benchmarks/README.md` — benchmark commands, scope, and interpretation

## Requirements

- DGX Spark/GB10 with CUDA 13 and an SM 12.1-capable compiler
- CMake 3.24 or newer and a C++20 compiler
- CUDA Runtime and cuBLASLt development files
- libibverbs headers/library for RoCE tools; disable with
  `-DDGPP_ENABLE_IBV=OFF` when unavailable (the GDR and RC probes are then
  omitted)
- Python 3.10 or newer for checkpoint tooling and its tests

## Build and test

```bash
# Warning-as-error configure, build, and CTest suite.
./scripts/ci-local.sh

# Optimized developer build.
cmake --preset release
cmake --build --preset release -j

# Host sanitizer variants.
cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan

cmake --preset ubsan
cmake --build --preset ubsan -j
ctest --preset ubsan
```

Presets are `release`, `debug`, `asan`, `ubsan`, and `ci`. If clang-format is
installed, CMake also exposes `format` and `format-check` targets.

## Implemented tools

| command | purpose |
|---|---|
| `dgppctl info` | CUDA and platform facts |
| `micro_mem_bw -s 4096` | LPDDR5x streaming patterns |
| `micro_gemm_peak` | best-of-cuBLASLt-heuristics FP8/BF16 shape sweep |
| `micro_zerocopy` | pinned/device GPU bandwidth, contention proxies, flag latency |
| `micro_gdr_probe` | informational direct-device MR probe and copy profile |
| `micro_ibv_smoke info` | enumerate verbs devices |
| `micro_ibv_smoke serve PORT --once --dev DEV` | one cross-node responder session |
| `micro_ibv_smoke ping --peer IP:PORT --dev DEV` | signaled RC SEND echo latency |
| `micro_ibv_smoke bw --peer IP:PORT --dev DEV` | signaled RC SEND bandwidth; repeat peer/device for both lanes |
| `micro_ibv_smoke verify --peer IP:PORT --dev DEV` | ordered NIC DMA payload/doorbell → GPU hash validation |
| `kda_bench` | KDA decode state-traffic and layer timing profile (M2) |
| `gpt_doll --selftest` | synthetic eager/graph parity testbed |
| `glm_bind_check --config CONFIG --checkpoint-dir DIR` | validate a real GLM-5.3 checkpoint against the expected-tensor table (config parse, names, dtypes, shapes, FP8 scale pairing; headers only) |
| `glm_stream_check --config CONFIG --checkpoint-dir DIR` | stream real layers through the resident loader; bytes vs formula reconciled per layer |
| `glm_forward_check --config CONFIG --checkpoint-dir DIR --suite FILE` | curated reference suite: ISOLATED per-layer parity vs torch-reference dumps, head + routing agreement (DESIGN §7.5) |
| `glm_forward_check ... --trace-ids-file F --trace-out T` | engine-only real route-trace capture (deterministic) for the traffic model |
| `tools/route_trace_traffic.py TRACE` | route-trace traffic model: measured busiest-rank occupancy and corrected critical path (replaces the uniform-expert assumption) |
| `tools/checkpoint_audit.py [MODEL_DIR]` | regenerate inventory and checkpoint budget |
| `tools/make_shardspec.py MODEL_DIR` | generate the loader shard plan |
| `tools/kda_reference_dump.py` | KDA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/dsa_reference_dump.py` | DSA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/glm_reference_dump.py` | full-model parity dumps: `selftest`, `gen-pure` (CI oracle, mini checkpoint), `gen-torch` (real checkpoint, per-layer streams + router scores; `--layers N` for reduced budgets) |
| `roster_check coordinator/rank/selftest` | M5 control plane on real nodes: epoch-based roster startup, rank health, eviction on death/deadline; `selftest` is the loopback in-process smoke |
| `bus_check serve/ping/selftest` | M5 data plane on real nodes: RC/RoCE CollectiveBus — per-class slot pools, credit-grant RDMA writes, dual-lane striping, latency-under-bulk contention; `selftest` is the loopback in-process smoke |

The GDR probe exits successfully when the probe itself completes, including
the expected “unsupported” result on GB10. It does not prescribe a bounce
copy; CollectiveBus uses registered pinned memory consumed directly by the
GPU.

## Tests

CTest currently runs:

- 54 host unit cases covering logging/tracing, JSON, arenas, safetensors,
  FP8, shard plans, KDA/DSA geometry contracts (against DESIGN §7.2's
  transcribed literals), route-trace golden bytes shared with the python
  reader, the MoE route-flip certifier's rejection paths (near-tie
  accepted; far-rank, zero-noise, own-scores-inconsistent, and duplicate-id
  divergences rejected), and the TCP/roster control plane (seal, epoch
  bumps, eviction by death and by deadline, rejection reasons, coordinator
  loss);
- synthetic CUDA graph/eager parity;
- the KDA operator suite: conv/recurrent kernel parity against host
  fp32/fp64 references, chunked-vs-unchunked bitwise equivalence, decode
  graph replay, snapshot round-trip, head-slice TP readiness, and
  reference-dump parity against the pure-python oracle;
- the DSA suite (M3): pool compression and tail-ring continuation verified
  bitwise via hard-max gates (including multi-token decode == single-token),
  a 1,100-case bitwise pooled top-k fuzz around pool boundaries plus
  exact-tie and 512th-boundary constructions, the fused decode select (MTP
  multi-row, grid-size invariance, 100k-pool long-context stripes, graph
  capture/replay with changed position), split-KV absorbed attention against
  the host oracle at TP1/TP4 with empty-row and head-group coverage,
  latent/gather block-table round trip, multi-request decode with padding
  rows, kpool=2 generality, and the layer tests: state-pool block
  allocation and byte accounting at deployment scale, prefill/chunked-
  prefill/decode parity against the oracle with selection-aware near-tie
  certification (at select_k=16 and select_k=8), decode graph replay
  bitwise across positions, TP2 head-slice vs TP1, and a real-geometry
  chunked prefill + decode smoke;
- DSA reference-dump parity: a pure-python oracle dump (bit-exact fp8 codec
  cross-checked against the C++ encoder) exercised through the full layer —
  latent cache bitwise, index cache within one e4m3 ulp, top-k exact — plus
  the torch backend against real checkpoint slices (executed on this box at
  32 and 2,052 tokens, the latter crossing the top-k horizon with zero
  flips); any flipped row is certified as a measured boundary near tie by
  the audit, never absorbed by tolerance;
- the M4 assembly suites: the mHC stream module (Sinkhorn mixing, stream
  update, final mean vs the double oracle), the MoE router/expert/shared
  path (sigmoid router with the noaux tie rule, swiglu asymmetries, bf16
  accumulation order, near-tie certification on synthetic corpora), the
  scale-aware GEMM against cuBLASLt BF16 references and real-checkpoint
  block edges, and the assembled-forward chain: a synthetic mini-checkpoint
  written on disk, a full-stack pure-python reference over the same
  weights, then the engine compared (hidden ulp budgets, top-k exact, route
  ids/weights, determinism);
- CUDA system-scope flag ordering, payload visibility, inactivity watchdog,
  and post-watchdog recovery;
- the M5 CollectiveBus data plane (needs the fabric + ibverbs): loopback
  scenarios with real RC QPs — payload integrity via fold-hash, credit
  recycling, dual-lane striping asserted on both sides, latency under bulk
  contention, watchdog failures, config-mismatch rejection, and orderly
  stop — plus the app-level smoke;
- Python checkpoint classification, exact expert-occupancy tests, and the
  route-trace traffic-model contract.

All CUDA suites are verified clean under `compute-sanitizer` memcheck (full
suite every milestone; racecheck and initcheck per-phase on the tests
exercising new kernel shapes); the sanitizer findings that motivated this
(speculated loads past short-circuit guards, shared-memory reuse races
that pass by scheduling luck, undersized test buffers that made a graph test
pass vacuously) are pinned in `DESIGN.md` §12.

Cross-node RoCE and NIC→GPU checks are intentionally manual/deployment tests;
they require a peer and are documented under `benchmarks/README.md`.
