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

### Deploying a serving rank: memory

A resident rank owns its box. The model is ~82 GiB of the 121 GB, and the
serving apps (`glm_serve`, `glm_gen_check`) configure themselves for that
without any privileged setup on the node:

- the constructor checks that the resident footprint (+ 8 GiB headroom)
  fits the device's free memory and fails immediately with a clear message
  if it does not — never three minutes into a load;
- the loader reads each source tensor exactly once (prefetch, copy, drop),
  so the page cache stays under ~10 GB during the load and the box never
  reaches its memory watermark; the checkpoint's mmaps are released the
  moment the last layer is on the device (`GlmLayerStream::release_sources`);
- the process *tries* to lock its memory (`mlockall(MCL_CURRENT)`, before
  the model is constructed) as a safety net against swap-in faults in the
  decode loop. This is optional: with the one-pass loader a rank with the
  pin off measured identically (p99 46 ms, 0 stalls, no swap traffic over
  1000 steps). A finite `RLIMIT_MEMLOCK` is logged, not warned about;
  `DGPP_MLOCK=off` skips the attempt.

Nothing else on the node needs setting. In particular a locked GPU clock
(`nvidia-smi -lgc`) is **not** required: the governor sits at 2400-2560 MHz
throughout decode on its own and the measured step distribution is the same
locked or unlocked. NTP between nodes only matters for reading logs side by
side, and `scripts/fabric_run.sh --node-probe` records each node's clock
offset per run so even that works without it.

### Deploying a serving rank: the resident image cache

The first start of a resident rank builds its layers from the checkpoint
(slice, stage, dequantize, pack) and writes the finished device bytes to
`~/.cache/dgpp/resident/<key>.img` on that node (~82 GiB per rank for GLM;
the key covers the checkpoint's shard headers, `config.json`, world, rank,
head sharding and the loader's format version, so a stale image can never
load by accident). Every later start streams that image instead with
O_DIRECT reads at the drive's line rate — 15-25 s to a ready model against
~4.5 minutes from the checkpoint. The boot digest is cached beside it
(`<key>.digest`). Knobs:

```
DGPP_RESIDENT_CACHE=off            disable (always build from the checkpoint)
DGPP_RESIDENT_CACHE_DIR=/path      put the images somewhere else
DGPP_RESIDENT_CACHE_VERIFY=1       re-fold every blob on read (a pass over 82 GiB)
```

Delete the file to force a rebuild; the loader's log line says how many
layers were restored versus captured on each start.

`scripts/fabric_run.sh --node-probe` samples each node's reclaim/swap/GPU
counters at 1 Hz for the run; `scripts/fabric_xrank.py LOGDIR` reads the
fetched logs and reports host gaps, stall windows, and step distributions per
rank.

### Judging a numerics change

Kernel work is allowed to change floating-point reduction order when it buys
latency, so two builds can legitimately produce different bits. Two tools say
whether a difference is rounding or a bug (both read a run directory through
`scripts/fabric_logs.py`, the shared parser for the per-rank `[gen]`/`[tf]`
step lines — start there when writing the next one):

- `scripts/fabric_xcript.py REF_DIR NEW_DIR` — for ordinary generation runs:
  finds the first token where the transcripts diverge and reports the global
  top-2 logit margin there in bf16 ulps. A flip at ≤ 1-2 ulp is a near-tie;
  a flip at a wide margin is a defect.
- `scripts/fabric_logprob.py NEW_DIR REF_DIR` — the quantitative gate. Run
  both builds with `--teacher-file benchmarks/teacher_text.txt` (the app
  scores the text's own tokens instead of generating; `--stage-file` puts the
  text on every rank), and the tool joins the ranks' vocabulary slices into
  log p(token) per position and reports the text's perplexity, the per-token
  deltas with a standard error, and a PASS/FAIL against a mean-NLL bound
  (default 0.02 nat ≈ 2% of perplexity):

  ```
  scripts/fabric_run.sh --stage-file benchmarks/teacher_text.txt -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --decode-graph
  scripts/fabric_logprob.py /path/to/new_logs /path/to/ref_logs
  ```

  A run of the same binary twice must show a delta of exactly 0 (the decode
  path is deterministic). The logits are fp32 (the head's accumulators,
  unrounded), but the residual stream feeding the head is bf16, so
  single-token deltas of ~0.1 nat are normal between builds that differ in
  reduction order; the mean over the text is the signal. Three
  texts ship: `teacher_text.txt` (556 tokens, a quick look),
  `teacher_text_hard.txt` (7,331 tokens of unmemorized technical prose,
  perplexity ~10.8 — the sensitive one; use it for the verdict) and
  `teacher_text_memorized.txt` (6,549 tokens of Conan Doyle, perplexity
  1.04 — the confident regime). Expect a handful of positions per thousand
  to move by more than 1 nat between any two rounding-level builds: a
  router's top-k boundary flipped an expert there. The tool bounds the
  *rate* of those, not the worst one.

### Speculative decode with the MTP layer (`--mtp`)

The checkpoint ships a multi-token-prediction layer (`num_nextn_predict_layers
= 1`): one extra DSA + MoE block that, given the main stack's hidden at
position q and the token chosen for q+1, guesses the token at q+2.
`glm_gen_check --mtp` uses it for greedy speculative decoding on the fabric:

```
scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
    --chat "Write a history of the Roman Republic." --steps 300 \
    --decode-graph --mtp
```

Every step is **one CUDA graph replay** that carries its own control flow:
the two-row verify of `[next, draft]`, the pick behind the head (each
rank's argmax gathered through one bus collective and judged on the
device), the commit (a rejected second row is rolled back on the device —
the KDA/DSA kernels snapshot their state after every speculative row, so a
retraction is a predicated copy per state family — and the position
advances), the MTP block over the accepted rows (a fixed two-row batch; the
second row is padding after a miss), its own pick, and the next step's
tokens written on the device. The host launches, waits, reads two small
pinned verdicts and logs. **The transcript is exactly the plain loop's** —
the verify rows are bitwise the single-token rows, and
`scripts/fabric_xcript.py PLAIN_DIR MTP_DIR` must print `IDENTICAL`; MTP
only changes what a token costs. Measured (2026-09-03, TP=4): 88.7% of
drafts accepted on coherent text, 1.89 tokens per step, **22.45 ms/token
effective vs 31.3 plain** (−28%). The step is 42.4 ms: the second verify
row costs ~7 ms because its MoE experts are extra DRAM bytes (only the
experts both rows share are read once — the slots run in expert order so
the second read is an L2 hit), the draft block ~2 ms. Acceptance drops on
incoherent text (63% on the post-EOS rambling `--no-eos` produces), and
below ~45% speculation stops paying; the summary line reports the rate.
Every rank computes the verdict itself from an identical gathered table; a
rank whose table was corrupt is caught at the next pick by a digest every
rank carries (`GlmDevicePicker`). Without `--decode-graph` the same step
runs eagerly (26 ms/token). The draft layer adds ~7.3 GiB per rank to the
resident footprint. Serving (`glm_serve`) does not use it yet — its engine
seam is single-token in/out.
