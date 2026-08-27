# DGPP Implementation Plan

This plan tracks implemented repository state, not aspirational directory
names. `[x]` means its code and local exit criteria exist and pass; `[ ]`
means it has not been implemented.

| milestone | scope | status |
|---|---|---|
| M0 | Platform, topology, transport, and checkpoint facts | [x] |
| M1 | Core runtime, loader, synthetic graph testbed | [x] |
| M2 | KDA operators and state manager | [x] |
| M3 | DSA/MLA sparse attention and index pools | [ ] |
| M4 | Full GLM single-node diagnostic assembly | [ ] |
| M5 | Four-rank TP and dual-lane CollectiveBus | [ ] |
| M6 | Generation scheduler, tokenizer, and API | [ ] |
| M7 | Exact snapshot prefix cache | [ ] |
| M8 | Transactional MTP decoding | [ ] |
| M9 | Evidence-driven optimization and hardening | [ ] |

M0 is complete. The paired network tests showed no congestion symptom; switch
inspection is an optional diagnostic if future four-node runs show drops,
retries, unstable throughput, or latency spikes. M1 is the synthetic/runtime
milestone; completion does not imply that a GLM model or server exists. M2 is
the KDA operator/state milestone: the recurrence, conv, projections, state
arena, and snapshot format pass their parity suites, but no full GLM layer
stack, scheduler, or server exists yet.

## Audit remediation completed on 2026-08-27

| audit item | resolution and evidence |
|---|---|
| ConnectX topology/105 Gb/s ceiling | Both `f0` lanes mapped and tested: 107.64 Gb/s each, 196.03 Gb/s concurrent. Bulk striping restored. |
| Vision/placement errors in decode budget | Audit now reads `config.json`, excludes `model.visual.*`, separates replicated DSA indexers from TP-sharded MLA, reports zero unmatched tensors, and has Python regression tests. |
| Average-rank TP roofline | Exact multivariate-hypergeometric busiest-rank model added, with replicated DSA indexer/router/mHC/norm reads: 7.457 GB/rank/token, 32.42 ms at 230 GB/s. |
| KDA/DSA cache units and dtypes | FP32 recurrent state and per-rank DSA/index/tail formulas are fixed in `DESIGN.md` §7. |
| `index_kpool` semantics | Defined as one state per four tokens with aligned chunks and a persistent incomplete tail. No global score gather. |
| Unsupported 24 KB replay record | Removed. M7 uses complete 36.39 MiB/rank KDA snapshots only. |
| MTP rollback | Commit/discard protocol and scratch-state ownership defined in `DESIGN.md` §9. |
| “dequant-on-load” ambiguity | Persistent format is E4M3 plus 128×128 F32 inverse scales; 37,338/37,338 geometries validated. |
| Broken verbs benchmark | SENDs are signaled, CQ arrays are correctly sized, errors propagate, resources are RAII-owned, and single/dual-lane tests pass. |
| Flag watchdog/ordering | System-scope atomics, initialized slots, inactivity reset, orderly stop, and a continuous-traffic test added. NIC→GPU hash test passed 20,000 total iterations across both lanes. |
| Activation and sampling ambiguity | Replicated block boundary, two all-reduces/layer, and exact sampling fallbacks fixed in `DESIGN.md` §§5 and 10. |
| Ineffective CI options | `DGPP_WERROR` and sanitizer cache variables are consumed; ASan/UBSan presets and Python audit tests are wired; CI no longer falls back silently; the unused mlx5 option was removed; the no-ibverbs build now works. |
| Stale/nonreproducible measurements | `docs/measurements.md` and `benchmarks/README.md` now separate validated results, commands, limitations, and optional diagnostics; GEMM runs warm clocks and use per-heuristic median samples. |
| README/planned-layout overclaim | README identifies the repository as an M0/M1 prototype and lists only implemented commands. |
| Ignored required sources | Bare `core` ignore pattern changed to `/core`; required `src/core` files are visible to version control, and their allocator alignment/overflow/destructor and tracing concurrency/JSON paths now have regression coverage. |

## M0 — Facts and transport validation (complete)

Delivered:

1. GB10 memory bandwidth, zero-copy access, CUDA flag latency, and real
   dual-RDMA contention measurements.
2. One ConnectX-7/one-cabled-QSFP topology mapped to two active PCIe x4 RoCE
   lanes, with per-lane and concurrent perftest results.
3. Corrected `micro_ibv_smoke` ping/SEND-bandwidth paths and multi-lane CLI.
4. Cross-node NIC-DMA payload + doorbell → GPU hash validation on both lanes.
5. Checkpoint inventory, quant-scale geometry, text traffic, and critical-rank
   model generated from the pinned revision.
6. Local DCB facts: global pause enabled, priority PFC disabled, counters
   readable without changing host configuration.

Operational diagnostic: if a later four-node soak shows transport errors,
throughput collapse, or latency spikes, compare pause/drop/retry/out-of-buffer
counters around the failing interval and inspect switch flow control.

## M1 — Core runtime and synthetic testbed (complete)

Delivered:

- typed tensor/view utilities and FP8 codec tests;
- monotonic/graph arenas, stream/event ownership, tracing, and logging;
- dependency-free JSON, safetensors headers, and shard-plan loader;
- CUDA graph capture/replay and synthetic transformer (`gpt_doll`);
- host, CUDA, and Python tests through CTest;
- warning-as-error CI plus ASan and UBSan presets.

Exit evidence is recorded in `docs/measurements.md`. `src/core` must be
included in the next commit; the earlier ignore rule made a clean clone
incomplete.

## M2 — KDA operators and state manager (complete)

Delivered:

1. BF16 projection/conv path and FP32 recurrent update: `KdaLayer` runs the
   fused `[f_a|g_a|q|k|v|b]` projection (one GEMM; f_a/g_a lead so the
   strided f_b/g_b GEMM inputs stay 16-byte aligned), the causal depthwise
   conv+silu, the gated-RMSNorm, and the output projection through the
   `IGemm` seam (extended with an activation row-stride parameter). The
   recurrence kernel holds the fp32 state in registers (zero spills at
   K=128) and serves prefill chunks and single-token decode from one code
   path.
2. Per-rank/request state arena (`KdaStatePool`): exact `[16,128,128]` FP32
   recurrent shape (K contiguous) plus merged q|k|v conv tail `[6144, 6]`
   BF16 with the three-token speculative reserve allocated, zeroed, and
   provably untouched by M2 kernels. Slot = 38,158,336 B (36.39 MiB) exactly
   as DESIGN §7.1 specifies.
3. Pool-aligned chunked prefill and single-token decode through the shared
   recurrence; chunked vs unchunked equivalence is bitwise on the
   GEMM-free path and within declared tolerance on the full layer
   (1024+1024 was bit-identical; 2047+1 drifted 0.13% of state scale from
   bf16 GEMM ulps in the inputs).
4. State snapshot import/export with a fixed-layout header carrying model
   revision, numerics mode, dtypes, TP, and full geometry; every identity
   field is validated before any payload byte lands.
5. Reference-dump harness (`tools/kda_reference_dump.py`): a stdlib-only
   pure-python oracle wired into CTest, and a torch backend that reads real
   checkpoint slices from safetensors (manual deployment test where the
   checkpoint lives).

Exit evidence (recorded in `benchmarks/results/2026-08-27-kda-m2.md`):

- geometry and bytes agree exactly with DESIGN §7.1 (unit-tested against
  the transcribed literals);
- chunked and unchunked prefill finish with equivalent FP32 recurrent state
  (bitwise without GEMMs; ≤0.13%-of-scale with them);
- layer outputs meet the declared FP32-accumulation tolerances against the
  host fp32/fp64 references and the pure-python dump oracle (bit-identical
  outputs on the dump case);
- no per-step allocation (arena scratch, graph-capture test replays
  decode bitwise), and state-update traffic profiled at batch size 1:
  86.4 GB/s effective at TP=1 (latency-bound; the layer-batching
  optimization path is documented in the record).

Additional findings worth carrying forward: the merged conv weight is
`[q(all heads)|k|v]`, so TP rank slices must be assembled per-section, not
as a contiguous prefix (the head-slice test enforces this); and cuBLASLt
returns garbage rather than an error for misaligned strided activation
pointers, which the fused layout prevents by construction.

## M3 — DSA/MLA sparse attention

Deliverables:

1. MLA q/kv projections and latent cache using the checkpoint's zero RoPE
   dimension.
2. Replicated indexer with APE+gate four-token compression, pooled top-k,
   original-token expansion, deterministic ties, and causal masking.
3. Persistent raw-K/gate incomplete-tail cache across chunks and decode.
4. Block size/chunk alignment checks and cache byte accounting.
5. Fuzz cases at every sequence length around a four-token pool boundary.

Exit criteria:

- top-k token positions are bitwise-identical to the pinned reference over at
  least 1,000 randomized cases;
- prefill/decode continuation across a partial pool matches an unchunked run;
- measured cache bytes are within 2% of the formula plus reported metadata;
- layer output parity passes on real checkpoint slices.

## M4 — Full GLM single-node diagnostic assembly

Deliverables:

1. Config-driven 45-layer adapter, mHC, dense/MoE/shared experts, router,
   vocab-sharded-capable lm head, and inert MTP module.
2. Compressed E4M3+scale resident loader; no persistent BF16 expansion.
3. Scale-aware GEMM parity for block edges and representative tensors.
4. Layer-streaming diagnostic mode so correctness can be tested without
   requiring the full distributed placement.
5. Router trace capture for the corrected per-rank traffic model.

Exit criteria:

- next-token logits and greedy tokens pass the curated reference suite;
- all 37,338 quantized matrices bind to validated scale tensors;
- actual per-rank bytes reconcile with the shard plan;
- route traces replace the uniform expert assumption in the performance model.

## M5 — Four-rank tensor parallelism

Deliverables:

1. Epoch-based roster/startup and rank health over TCP.
2. `CollectiveBus` on both active `f0` lanes with single-lane failover,
   registered pinned receive slabs, credits, CQs, system-scope doorbells, and
   inactivity watchdogs.
3. Replicated block boundaries with one attention and one FFN all-reduce per
   layer; striped bulk prefill collectives.
4. Sharded load for TP=2 and TP=4, plus hashes for replicated weights.
5. Transport regression command that reruns NIC→GPU visibility on every node
   pair after driver/firmware changes.

Exit criteria:

- TP=2/4 outputs match M4 within the same tolerance tier;
- no CQ, credit, or watchdog failure during a one-hour mixed-size soak;
- both lanes contribute under concurrent bulk traffic;
- throughput and latency remain stable for the defined workload; counter
  diagnostics are captured only if this criterion fails.

## M6 — Generation, tokenizer, and service

Deliverables:

1. Rank-consistent continuous batching, cancellation, and admission budgets.
2. Exact ByteLevel-BPE tokenizer and model-load-time chat-template compiler,
   keyed by tokenizer/template revision hashes.
3. Greedy and finite-top-k distributed fast paths plus full-logit gather for
   exact unrestricted top-p/min-p/logprobs behavior.
4. HTTP/SSE endpoints for chat, completions, models, health, and metrics.

Exit criteria:

- streamed multi-turn chat and tool calls work on TP=4;
- tokenizer/template goldens match the checkpoint reference;
- all sampling modes match a centralized-logit oracle for fixed seeds;
- latency and throughput results are committed with the reproducible workload
  definition used to obtain them.

## M7 — Exact snapshot prefix cache

Deliverables:

1. Immutable 128-token DSA blocks, radix lookup, refcounts, and rank-agreed
   eviction epochs.
2. Complete KDA state snapshots only; no auxiliary-record reconstruction.
3. 1.5 GiB/rank initial snapshot arena (maximum 42 snapshots before metadata),
   capacity metrics, and request opt-out.
4. Template/tokenizer/model revision keys and exact incomplete-tail attachment.

Exit criteria:

- hot and cold paths produce identical logits and subsequent states;
- missing snapshots force re-prefill rather than partial reuse;
- eviction under concurrent attachment has no use-after-free or rank drift;
- TTFT, bytes saved, snapshot-copy time, and capacity are reported separately.

## M8 — Transactional MTP

Deliverables:

1. Three-token initial draft path and verifier microbatch.
2. `k+1` KDA candidate states, speculative convolution width, reserved MLA
   slots, and scratch index tail/pool updates.
3. Rank-broadcast accepted count/RNG counter and atomic commit/discard epoch.
4. Adaptive draft depth based on measured acceptance and memory pressure.

Exit criteria:

- rejection at every depth, including a pool boundary, matches normal decode;
- cancellation and injected rank failure leave committed state unchanged;
- temperature-zero output is identical with MTP on/off;
- speedup is reported with acceptance distribution and scratch-memory cost,
  with no predeclared multiplier treated as fact.

## M9 — Optimization and hardening

Work is prioritized from full-model profiles: route placement, collective
overlap, grouped-MoE kernels, MLA gather, graph buckets, and sampler cost.
Hardening includes peer-kill drills, clean cancellation, counter-drift checks,
24-hour soak, malformed-HTTP fuzzing, and restart documentation.

Final performance sign-off reports:

- batch-size-one decode and 32K TTFT for a committed workload definition;
- expected, p95, and worst-rank expert traffic from traces;
- both-lane utilization and collective share;
- prefix-cache capacity/hit curves;
- MTP acceptance and net speedup;
- known gaps, rather than silently moving unmet targets.

## Post-v1

Vision processing, NVFP4 conversion, expert replication/placement beyond M9
telemetry, data-parallel replicas, cross-instance prefix federation, TLS
fronting, and packaging remain out of the v1 critical path.
