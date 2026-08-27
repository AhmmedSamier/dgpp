# DGPP Engine

DGPP is an early-stage C++/CUDA inference-engine project for serving the text
path of `unsloth/GLM-5.3-Flash-FP8` across four NVIDIA DGX Spark systems.

Current status: M0/M1/M2 prototype with M3 in progress. The repository
implements platform and RoCE validation tools, checkpoint/shard inspection,
core CUDA runtime utilities, a synthetic transformer testbed, the KDA
linear-attention operators with their per-request state manager and
reference-dump harness, and the DSA/MLA sparse-attention kernels (indexer
compression, deterministic pooled top-k, split-KV absorbed attention) with a
dual-precision host oracle. It does **not** yet implement the full GLM model,
the DSA layer orchestration and state pool, multi-node inference runtime,
prefix cache, MTP generation loop, or OpenAI-compatible server. See `PLAN.md`
for milestone status.

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
| `tools/checkpoint_audit.py [MODEL_DIR]` | regenerate inventory and checkpoint budget |
| `tools/make_shardspec.py MODEL_DIR` | generate the loader shard plan |
| `tools/kda_reference_dump.py` | KDA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |

The GDR probe exits successfully when the probe itself completes, including
the expected “unsupported” result on GB10. It does not prescribe a bounce
copy; CollectiveBus uses registered pinned memory consumed directly by the
GPU.

## Tests

CTest currently runs:

- 24 host unit cases covering logging/tracing, JSON, arenas, safetensors, FP8,
  shard plans, KDA geometry/snapshot-header contracts, and the DSA geometry
  contract against DESIGN §7.2's transcribed literals;
- synthetic CUDA graph/eager parity;
- the KDA operator suite: conv/recurrent kernel parity against host
  fp32/fp64 references, chunked-vs-unchunked bitwise equivalence, decode
  graph replay, snapshot round-trip, head-slice TP readiness, and
  reference-dump parity against the pure-python oracle;
- the DSA kernel suite (M3, in progress): pool compression and tail-ring
  continuation verified bitwise via hard-max gates (including multi-token
  decode == single-token), a 1,100-case bitwise pooled top-k fuzz around
  pool boundaries plus exact-tie and 512th-boundary constructions, the fused
  decode select (MTP multi-row, grid-size invariance, 100k-pool long-context
  stripes, graph capture/replay with changed position), split-KV absorbed
  attention against the host oracle at TP1/TP4 with empty-row and
  head-group coverage, latent/gather block-table round trip, multi-request
  decode with padding rows, and kpool=2 generality;
- CUDA system-scope flag ordering, payload visibility, inactivity watchdog,
  and post-watchdog recovery;
- Python checkpoint classification and exact expert-occupancy tests.

All CUDA suites are also verified clean under `compute-sanitizer` memcheck,
racecheck, and initcheck; the sanitizer findings that motivated this (speculated
loads past short-circuit guards, shared-memory reuse races that pass by
scheduling luck) are pinned in `DESIGN.md` §12.

Cross-node RoCE and NIC→GPU checks are intentionally manual/deployment tests;
they require a peer and are documented under `benchmarks/README.md`.
