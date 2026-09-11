# Tests

Build the targets you intend to test. For a first checkout, host and Python
checks do not need model weights or GPU execution:

```bash
cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check
ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
ctest --test-dir build-ci -L python --output-on-failure
```

`DGPP_TEST_FILTER=<substring>` runs a subset of a binary's cases; the
loopback tests use fixed ports in the 299xx range. Run GPU/RDMA tests only
on idle test hardware, serially; do not run them alongside production serving.

CTest labels are `host`, `python`, `checkpoint`, `gpu`, `rdma` and `fixture`.
Use `ctest --test-dir build-ci -N -L LABEL` to inspect a group before running
it. Fixture producers are added automatically when a selected test requires
them. Build all targets before a full CTest run. The `checkpoint` label marks
host tokenizer/template suites requiring real cached metadata. Some GPU suites
also contain optional real-checkpoint cases: a passing process exit does not
mean every case ran. Read skip messages; missing metadata returns CTest's skip
code in the tokenizer suites. Do not count skipped tests as model validation.

Native tests do not read `.env`. To pass the head node's cache/NIC overrides,
run `python3 scripts/site_env.py run-rank --rank 0 -- ctest --test-dir build-ci ...`.
Reference generators need the optional dependencies listed in
[dependencies](dependencies.md). CPU fixture generators do not require PyTorch;
torch-backed reference modes do.

## Evaluation data and prefill fixtures

Datasets are not bundled and benchmarks never download them implicitly:

```bash
python3 scripts/prepare_data.py download --tasks gsm8k humaneval
python3 scripts/prepare_data.py tokens --text /path/to/long-prompt.txt
```

Downloads use pinned upstream revisions from OpenAI's
[GSM8K](https://github.com/openai/grade-school-math) and
[HumanEval](https://github.com/openai/human-eval) repositories. Each dataset has
a `.source.json` recording its source/license URL, row count and SHA-256.
Existing different content is never overwritten. `DGPP_DATA_DIR` defaults to
the ignored `data/` directory; use `--out` to select another preparation path.
Token generation needs `tokenizers` and uses the selected model's tokenizer;
generate a sufficiently long input for the prefill lengths you want to test.
`fabric_prefill_repeat.sh --ids-file FILE` selects an existing CSV fixture.

`serve_eval.py --data DIR` overrides the dataset directory;
`serve_prefill_probe.py --data FILE` selects the GSM8K JSONL file directly.
Use `--tasks gsm8k,extract` for evaluation without generated-code execution.
HumanEval requires `--allow-code-execution`, including via the serving wrappers.
It runs model-produced Python with a timeout, **not a security sandbox**.
Use a disposable isolated evaluation machine/container with no secrets,
unneeded mounts or network access. The flag acknowledges the risk; it does
not provide isolation. Never run it on a production serving node.

## Suite coverage

Use `ctest --test-dir build-ci -N` to list the tests in your configured
build. The suites cover:

- host unit cases covering logging/tracing, JSON, arenas, safetensors,
  FP8, the latent cache's fp8/fp4 codecs (the e2m1 grid and its
  round-to-even ties, the row quantizers' error bounds, zero rows, the
  padded fp4 row), shard plans, the HF cache resolver, the sampler against its
  centralized oracle, KDA/DSA geometry contracts (against DESIGN §7.2's
  transcribed literals), route-trace golden bytes shared with the python
  reader, the MoE route-flip certifier's rejection paths (near-tie
  accepted; far-rank, zero-noise, own-scores-inconsistent, and duplicate-id
  divergences rejected), and the TCP/roster control plane (seal, epoch
  bumps, eviction by death and by deadline, rejection reasons, coordinator
  loss);
- the M6 host suites: the tokenizer (55 byte-exact goldens per family,
  hash-keyed — GLM-5.3, Qwen3.8-Flash-Next and GLM-4.7 corpora), the chat
  template (26 / 20 / 26 goldens + 8 refusal negatives, the tool-call
  round trips and grammars over each real tokenizer), the scheduler
  (isolation, determinism, cancellation, bounded queue, tick ≡
  run_to_completion), the HTTP server's limit ladder, the OpenAI shapes
  over real HTTP with a fake engine, and the admission journal with two
  real peer loops over localhost;
- synthetic CUDA graph/eager parity;
- the second and third families on the shared cores (2026-09-09/10): the
  Qwen3.8-Flash-Next chain (config/binding units, the loader, GDN/QSA/GR/
  PLE/MoE kernel oracles, the fixture forward against the pure-python
  reference, decode sessions, loopback TP worlds 2 and 4, the graph engines
  at world 2) and the GLM-4.7 chain (`glm4_config`/`glm4_binding` units,
  `glm4_loader_test` with the draft's NVFP4 requant, `glm4_attn_test`
  bitwise against the tile-aware host reference, `glm4_forward_test` vs
  `tools/glm4_reference_dump.py`, `glm4_decode_test` — prefill == forward
  bitwise, interleaved slots, chunked prefill, snapshots at any position,
  the speculator through the draft — `glm4_tp_test` worlds 2 and 4
  (ports 29946/29947), `glm4_engine_test` (29948/29949)); the fp4 GEMV
  core at every compiled K and the MoE layer's NVFP4 shared expert;
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
  latent/gather block-table round trip, the quantized latent cache (the
  fp8/fp4 append bitwise against the host codec at 512/256/32 wide; the
  split, listed-flash and dense-flash attention kernels over an fp8/fp4
  cache against the oracle fed the dequantized rows, within the bf16
  kernel's own tolerance; a whole layer on each quantized cache against
  the bf16 layer — same selection, output drift within the format's
  bound), multi-request decode with padding rows, kpool=2 generality, and
  the layer tests: state-pool block
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
  contention, watchdog failures, config-mismatch rejection, the graph era
  with the mixed-era interlude, the bulk RS+AG machine, and orderly stop —
  plus the app-level smoke;
- the TP forward over loopback buses (`glm_tp_test`): per-layer isolated
  parity vs the world=1 oracle at worlds 2 and 4, decode sessions, the
  greedy generation loop, the device pick, the one-graph MTP step in
  lockstep with the eager speculator, and the T=1/MTP serving graph adapters
  through the real scheduler (including slot reuse, the in-graph draft
  checked against the eager speculator after every replay, the MTP depth-2
  and depth-3 greedy gates — plain transcript, device feed equal to the
  eager chain's — and the sampled depth-2 gate against the eager
  speculator through its fallbacks; every gate's eager oracle drains the
  pipelined engine before it steps); the pick/spec kernels against their
  host oracles, the T=2 and T=3 sampled verdict chains included
  (`glm_pick_test`); the bus's graph era including two replay windows
  armed at once (`bus_test`); the GEMV cores (`bf16_gemv_test`) and the
  loader's resident-image round trip (`glm_loader_test`);
- Python checkpoint classification, exact expert-occupancy tests, and the
  route-trace traffic-model contract.

`glm_tp_test` and `bus_test` run with
`CUDA_DEVICE_MAX_CONNECTIONS=32` and prefetch enabled. Capture-time
`check_decode_graph` rejects memcpy, memset, event-record and other
unsupported nodes. Kernel and empty nodes are allowed, along with a
model-declared number of host callbacks for Qwen's mapped n-gram gather.
The copy-engine restriction prevents dependency cycles between ranks in
one process. The [graph-stall investigation](batched_mtp_graph_stall.md)
records the reproducer and validation at 1 and 32 connections.

The recorded CUDA sanitizer runs cover `compute-sanitizer` memcheck (full
suite every milestone; racecheck and initcheck per-phase on the tests
exercising new kernel shapes). The multi-rank loopback worlds of
`glm_tp_test` are the exception: their budgets are liftable
(`DGPP_TEST_BUS_TIMEOUT_MS`, `DGPP_TEST_CONSUMER_DEADLINE_S`,
`DGPP_TEST_WAIT_TIMEOUT_MS`), but even lifted, the instrumented eager prefill
outlasts the first collective's kernel deadline (baseline and current alike,
2026-09-04 record entry), so those worlds report 0 errors and exit on a
budget rather than completing. The sanitizer findings that motivated this
(speculated loads past short-circuit guards, shared-memory reuse races
that pass by scheduling luck, undersized test buffers that made a graph test
pass vacuously) are pinned in `DESIGN.md` §12.

Cross-node RoCE and NIC→GPU checks are intentionally manual/deployment tests;
they require a peer and are documented under `benchmarks/README.md`.
