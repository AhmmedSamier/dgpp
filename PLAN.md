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
| M8 | Transactional MTP decoding | [x] depth 1, greedy, the whole step one graph replay (2026-09-03; see DESIGN §9 "as built" and "the on-device step") |
| M9 | Evidence-driven optimization and hardening | [ ] |

M0 is complete. The paired network tests showed no congestion symptom; switch
inspection is an optional diagnostic if future four-node runs show drops,
retries, unstable throughput, or latency spikes. M1 is the synthetic/runtime
milestone; completion does not imply that a GLM model or server exists. M2 is
the KDA operator/state milestone: the recurrence, conv, projections, state
arena, and snapshot format pass their parity suites, but no full GLM layer
stack, scheduler, or server exists yet. M3 is complete: the DSA/MLA kernels,
selection machinery, host oracle, state pool, layer orchestration, dump
harness, and benchmarks are implemented, tested (24 CUDA tests + dump parity
in CTest), and benchmarked — but no full GLM layer stack, scheduler, or
server exists yet.

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
   checkpoint slices from safetensors (executed 2026-08-28 on this box
   against revision a160e2291674d9e3e92e98fd82faa2544a2867a3, layer 0 —
   parity clean; the runner applies backend-aware budgets since the torch
   oracle is fp32, not double. Record in the results file).

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

## M3 — DSA/MLA sparse attention (complete)

Deliverables:

1. MLA q/kv projections and latent cache using the checkpoint's zero RoPE
   dimension.
2. Replicated indexer with APE+gate four-token compression, pooled top-k,
   original-token expansion, deterministic ties, and causal masking.
3. Persistent raw-K/gate incomplete-tail cache across chunks and decode.
4. Block size/chunk alignment checks and cache byte accounting.
5. Fuzz cases at every sequence length around a four-token pool boundary.

Delivered — kernel phase (all verified, sanitizer-clean):

- `DsaConfig`/`DsaGeometry` with exact DESIGN §7.2 byte formulas
  (`tests/unit/dsa_geometry_test.cpp` against transcribed literals);
- the host fp32/fp64 oracle (`src/models/dsa_reference.cpp`) pinning the
  full numeric contract: per-dimension gated pool softmax, Hadamard-128 with
  the reference's bf16 boundary rounds, power-of-two fp8 scales via exact
  bit manipulation, absorbed-MLA attention with bf16 prob rounding;
- all M3 kernels (`src/kernels/dsa.cu`): fwht+quant, k_layernorm, fused
  q/kv rmsnorm, pool compress-write, tail seed, decode ring update with
  completion, latent append, pool gather, the fused decode select (streams
  the planar index cache once, composite-key top-512 in shared memory,
  last-block merge, counter self-reset — graph-capturable, no logits
  materialized), the prefill select over IGemm fp8 dots, and the split-KV
  absorbed attention with per-lane-register online-softmax state;
- 18 CUDA tests: bitwise select fuzz over 1,100 cases around pool
  boundaries; MTP-shaped multi-row decode; grid-size invariance; 100k-pool
  long-context decode; multi-row prefill; exact ties including the 512th
  boundary; ring continuation across a partial pool with multi-token ==
  single-token bitwise; multi-request decode with padding rows; TP1
  attention head-groups; empty-row zero output; graph capture/replay with
  changed position; kpool=2 smoke. Release/ASan/UBSan suites green;
  compute-sanitizer memcheck, racecheck, and initcheck clean on both CUDA
  suites.

The kernel verification round also fixed a spec inversion the parity tests
could not see (selection ordered by ascending instead of descending logits —
both implementations were wrong together; caught by a
hand-computed-expectation ties test), three shared-memory races, and a
speculated out-of-bounds load (see DESIGN §12 for the pinned lessons).

Delivered — layer phase:

- `DsaStatePool` (`src/models/dsa_state.*`): blocked latent + planar index
  caches for all 11 layers, per-request tail rings, one shared block table
  (block = 128 tokens = 32 pools, co-located so DESIGN §8 prefix attachment
  shares both caches by reference), a host-side LIFO block allocator with
  transactional growth, and the byte-accounting API behind the ±2% exit
  criterion (unit-tested at deployment scale: 300k tokens + 32 requests =
  3.488 GB, +0.01% over the §7.2 formulas).
- `DsaLayer` (`src/models/dsa_layer.*`): full orchestration — fused
  [q_a|kv_a] projection with split RMSNorms, MLA q_b + indexer wq_b from
  the normed q-lora, k LayerNorm + gate + fp32 weights from hidden,
  Hadamard quant + fold, then per path: prefill (pool-aligned chunks,
  per-chunk index gather, dot-GEMM tiles bounded by a dot budget, select,
  attention) and decode (latent append, ring update with completion, fused
  block-table select, attention, o_proj). Allocation-free; the decode path
  is CUDA-graph capturable (plans + smem opt-in pre-built in prepare();
  visible counts derived on device so one graph serves any position).
- 6 layer-level CUDA tests: state-pool allocation/accounting; prefill vs
  the host oracle; chunked prefill + multi-token decode vs the oracle
  across a partial-pool boundary; decode graph replay (capture once, two
  positions, bitwise); TP2 head-slice vs TP1; real-geometry chunked
  prefill + decode smoke with bitwise repeat determinism. 24 CUDA tests
  total, green under release/ASan/UBSan; compute-sanitizer memcheck clean
  on the full suite, racecheck + initcheck clean on the layer tests (the
  real-geometry racecheck pass is skipped deliberately: its runtime is
  vendor cutlass GEMM kernels already covered by the kernel-phase tests —
  see the measurements record).
- The selection-aware near-tie audit (`tests/cuda/dsa_near_tie_audit.hpp`):
  every device-vs-oracle selection divergence must re-derive the spec
  selection from the device's OWN inputs (bitwise, tensor-core dots
  included) and prove the swapped pools straddle the rank boundary within
  the measured cross-implementation noise. The audit caught two real
  reference bugs during development (an out-of-bounds tail-seed read for
  decode batches shorter than kpool, and a gate-indexing typo) — both
  fixed; the device path was correct both times.
- `tools/dsa_reference_dump.py` (pure + torch backends) with the
  bit-exact python fp8 codec (cross-checked 0/713 mismatches against the
  C++ encoder), the `DGPPDSAD` dump reader (`src/models/dsa_dump.*`), and
  the CTest parity runner: layer output within the double-vs-fp32 oracle
  budget, latent cache BITWISE, index cache within one e4m3 ulp, top-k
  structurally exact (zero flips on the pure corpus).
- `dsa_bench` (`benchmarks/micro/dsa_bench.cu`): decode eager/graph-replay
  and prefill throughputs at real geometry, effective-bytes reporting
  against the 230 GB/s planning floor; optimization record in
  `benchmarks/results/2026-08-28-dsa-m3-layer.md` (decode 13.26 → 2.24 ms
  over the profiled round: hardware fp8/bf16 conversions, split-KV
  n_split=32 with empty-split early exit, bank-conflict-padded attention
  smem, split-32-bit bitonic keys with thread-strided merge loads; the
  projection GEMMs now stream at the memory floor).

Exit criteria:

- top-k token positions are bitwise-identical to the pinned reference over at
  least 1,000 randomized cases; ✓ (1,100-case fuzz + dump top-k exact)
- prefill/decode continuation across a partial pool matches an unchunked run;
  ✓ (chunked-vs-unchunked vs oracle, ring continuation bitwise)
- measured cache bytes are within 2% of the formula plus reported metadata;
  ✓ (+0.01% at deployment scale)
- layer output parity passes on real checkpoint slices. ✓ (pure backend in
  CTest; torch backend executed on this box against revision
  a160e2291674d9e3e92e98fd82faa2544a2867a3, layer 3: latent cache within a
  bf16 ulp, index cache within one e4m3 ulp, layer output within the fp32
  oracle budget, top-k exact at 32 and 2,052 tokens — the latter crossing
  the top-k horizon with 0 flips. First execution surfaced four
  never-executed-path bugs in the tool, most instructively a raw-byte
  index_k dot that tolerance absorbed as a single boundary swap; the dump
  runner now certifies flips with the near-tie audit instead of tolerating
  them. Record in benchmarks/results/2026-08-28-dsa-m3-layer.md)

## M4 — Full GLM single-node diagnostic assembly (complete)

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

## M5 — Four-rank tensor parallelism (complete)

Deliverables:

1. Epoch-based roster/startup and rank health over TCP.
2. `CollectiveBus` on both active `f0` lanes: registered pinned receive
   slabs, class-partitioned credits (latency headroom reserved against
   bulk), CQs, system-scope doorbells, inactivity watchdogs, and concurrent
   latency-under-bulk traffic as a tested mode.
 3. Replicated block boundaries with one attention and one FFN all-reduce per
    layer; striped bulk prefill collectives. The all-reduce primitive is
    built and validated (DESIGN §6.3, one-shot all-to-all, canonical
    rank-order fp32 fold, bitwise on the full mesh at TP=2 37.6 µs /
    TP=4 ~44 µs p50), and the FORWARD INTEGRATION is wired and validated
    in CI over loopback buses (glm_tp_test): GlmTpViews slicing per
    DESIGN §5.2, the two block-boundary folds, MoE whole-expert
    partitions, per-layer isolated parity vs the world=1 oracle with
    route-flip/head near-tie certification, and cross-rank BITWISE
    hidden/logits/captures/routes at world 2 and 4. The fabric runner
    (glm_tp_check) and real-mesh parity are DONE (2026-08-31: the fabric
    TP=4 run's final_hidden/logits/routes/digests are bitwise-identical
    across all four ranks AND to the loopback world-4 verdict dumps —
    md5 f2674050…/f5a9bf85…, the full parity-tier verdict transfers to
    the fabric by bitwise identity). Remaining: the producing GEMMs
    writing send slots directly, bulk prefill collectives, and
    CUDA-graph capture of the decode launch sequence (§6.2).
4. Sharded load for TP=2 and TP=4, plus hashes for replicated weights.
5. Transport regression command that reruns NIC→GPU visibility on every node
   pair after driver/firmware changes. Built (`nic_regress`): one mesh-mode
   command drives the M0 project test (payload + doorbell ordered RC SENDs,
   GPU system-scope poll, hash-then-ack) over every directed pair × both
   lanes with the deployment GID selection, identity-checked hellos, held
   early connections, both-endpoint view aggregation at nodes[0], and one
   broadcast exit code; `pair`/`serve` for single-pair debugging and a
   CI selftest (full mesh machinery, two-thread world on one device —
   pre-thread construction, stream-scoped syncs) round it out. Fabric mesh
   validation runs with the exit-gate fabric pass.

Exit criteria (ALL MET 2026-08-31; the full evidence trail is in
benchmarks/results/2026-08-29-bus-m5.md):

- TP=2/4 outputs match M4 within the same tolerance tier — the parity
  tier PASSES at real dims, loopback worlds 2 and 4 (worst isolated
  capture l2 0.0058/0.0066, worst fold 0.0041/0.0049, all first-flip
  route certifications and top-1 near ties certified), and the fabric
  TP=4 run is bitwise-identical to the loopback world-4 verdict
  (final_hidden f2674050…, logits f5a9bf85…, identical across all
  four ranks, routes and boot digests included);
- no CQ, credit, or watchdog failure during a one-hour mixed-size soak
  with decode-class and bulk traffic concurrent — four 15-minute
  phases (1 MiB/256 KB/4 MiB/1 MiB), ~18.4 TB bulk + 184M decode-class
  probes, ZERO failures, lat p50 17.9-19.4 µs and p99 50-187 µs under
  22-42 Gbps concurrent floods (bus_check gained `--contend --soak-ms`,
  the duration-bounded soak driver);
- both lanes contribute under concurrent bulk traffic — every
  multi-stripe phase split bulk stripes across both lanes (2.0-2.4 TB
  on lane 1 per phase);
- throughput and latency remain stable — phase 0 vs phase 3 (same
  workload, one hour apart) within a few percent, no trend.

## M6 — Generation, tokenizer, and service

Deliverables:

1. Rank-consistent continuous batching, cancellation, and admission budgets.
2. Exact ByteLevel-BPE tokenizer and model-load-time chat-template compiler,
   keyed by tokenizer/template revision hashes.
3. Greedy and finite-top-k distributed fast paths plus full-logit gather for
   exact unrestricted top-p/min-p/logprobs behavior.
4. HTTP/SSE endpoints for chat, completions, models, health, and metrics.
5. Resident serving mode (the production residency contract, DESIGN §3):
   the rank's weights load once at startup and stay resident — storage is
   never touched during inference. TP=4 is the only world that fits 128 GB
   (81.77 GiB/rank with the replicated globals, ~46 GB headroom); the
   M4/M5 streaming loader remains the diagnostic instrument.
   IMPLEMENTED in M5 (2026-08-31, ahead of the serving integration):
   bitwise resident-vs-streaming parity pinned at every world, the
   zero-reread contract proven on all four ranks at real dims; M6
   integrates it into the serving path.

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
   As built: no broadcast — every rank computes the verdict from an
   identical gathered table on the device, and a digest carried into the
   next gather catches a divergent rank; commit/discard is a recorded
   predicated kernel behind the verdict (DESIGN §9 "the on-device step").
4. Adaptive draft depth based on measured acceptance and memory pressure.
   Not built: depth 2 was declined for its step-to-step variance (a second
   draft accepted ~60% of the time against a ~29% break-even), and the
   fixed-depth step's remaining cost is inside the graph.

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
