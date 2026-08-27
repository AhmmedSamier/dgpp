# DGPP Engine

Lightweight C++ inference engine for NVIDIA DGX Spark (GB10) clusters,
purpose-built for serving `zai-org/GLM-5.3-Flash` FP8 via a distributed
tensor-parallel design over RoCE.

Docs:
- `DESIGN.md` — architecture and rationale
- `PLAN.md` — phased implementation plan w/ milestone tracking
- `docs/checkpoint_budget.md` — audited checkpoint inventory & budgets
- `docs/measurements.md` — microbenchmark results (auto-generated entries)

## Build

```
./scripts/ci-local.sh                 # configure+build+test (preset ci)
cmake --build build-release -j       # after configuring release preset
```

Presets: `release`, `debug`, `asan`, `ci`.

## Tools

| binary | purpose |
|---|---|
| `dgppctl info` | CUDA/platform facts |
| `micro_mem_bw -s 4096` | DRAM bandwidth patterns |
| `micro_gemm_peak` | cuBLASLt FP8/BF16 GEMM sweep |
| `micro_zerocopy` | GB10 unified-memory zero-copy receive BW + fenced flag protocol latency |
| `micro_ibv_smoke info/ping/bw` | RoCE RC QP latency/BW probe |
| `micro_gdr_probe` | GPUDirect RDMA registration viability (resolved: inapplicable, see docs) |
| `gpt_doll` | synthetic transformer testbed — modes: `--selftest`, `--bench N`, `--bench-rollout N`, `--gen`, `--soak-minutes M` |
| `tools/make_shardspec.py` | generate `shardspec.json` from a real checkpoint |

## Tests

| suite | what it covers |
|---|---|
| `unit_tests` (ctest `unit`) | log, minijson, arena, safetensors, fp8 codec, shardspec (14 host tests) |
| `gpt_doll --selftest` (ctest `doll_selftest`) | bitwise graph-vs-eager parity, replay determinism, rollout reproducibility |
| `flag_protocol_test` (ctest) | CPU↔GPU flag ordering contract, payload visibility, watchdog recovery (GPU required, skips without) |

Run python tooling from repo root; scripts are dependency-free (py stdlib).
