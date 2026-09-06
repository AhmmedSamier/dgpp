# Tests

How to run the suites and what each one proves. The rule for every
change: a full build precedes the suite's verdict (`ctest` runs whatever
binaries exist), so

```bash
cmake --build build-ci -j -- -k && ctest --test-dir build-ci --output-on-failure
```

`DGPP_TEST_FILTER=<substring>` runs a subset of a binary's cases; the
loopback worlds each own a port in 29910–29941, so run one CUDA suite at a
time on a node that is also serving.

CTest currently runs 33 entries:

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
- the M6 host suites: the tokenizer (55 byte-exact goldens, hash-keyed),
  the chat template (26 goldens + 8 refusal negatives), the scheduler
  (isolation, determinism, cancellation, bounded queue, tick ≡
  run_to_completion), the HTTP server's limit ladder, the OpenAI shapes
  over real HTTP with a fake engine, and the admission journal with two
  real peer loops over localhost;
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

`glm_tp_test` and `bus_test` run with `CUDA_DEVICE_MAX_CONNECTIONS=32`; the
prefetcher stays enabled everywhere. The captured decode graph is
kernels-only by contract, checked at every capture site
(`glm_check_decode_graph`): a memset/memcpy node executes on the process-
shared copy-engine queue, where one rank's queued dependency wait blocked
the peer rank's node behind it while its own collective spun waiting on
that peer — the batched-MTP loopback stall, root-caused from an nsys node
trace and fixed on 2026-09-03 (`docs/batched_mtp_graph_stall.md`; the
loopback gates now pass with prefetch on at 1 and 32 connections).

All CUDA suites are verified clean under `compute-sanitizer` memcheck (full
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
