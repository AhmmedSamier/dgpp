# DGPP Implementation Plan

This plan tracks implemented repository state, not aspirational directory
names. `[x]` means its code and local exit criteria exist and pass; `[~]`
means part of the milestone is built and gated and the rest is designed
below; `[ ]` means it has not been implemented.

| milestone | scope | status |
|---|---|---|
| M0 | Platform, topology, transport, and checkpoint facts | [x] |
| M1 | Core runtime, loader, synthetic graph testbed | [x] |
| M2 | KDA operators and state manager | [x] |
| M3 | DSA/MLA sparse attention and index pools | [x] |
| M4 | Full GLM single-node diagnostic assembly | [x] |
| M5 | Four-rank TP and dual-lane CollectiveBus | [x] (closed 2026-08-31) |
| M6 | Generation scheduler, tokenizer, and API | [~] serving works end to end on the fabric at TP=4; adaptive scalar/row-batched T=1 and MTP graphs are correctness- and performance-gated (one-live MTP restored from 19.3 to 38.5 tok/s while four-live retains 60.3 tok/s; 2026-09-03); sampling modes and tool calls remain |
| M7 | Exact snapshot prefix cache | [ ] design below |
| M8 | Transactional MTP decoding | [x] depth 1, greedy, the whole step one graph replay (2026-09-03; `glm_gen_check --mtp`) |
| M9 | Evidence-driven optimization and hardening | [~] the optimization half is well under way (40.65 → 31.45 ms/token plain, 22.45 with MTP); hardening not started |

## Where we are (2026-09-03)

The engine generates text on four DGX Sparks: `glm_serve` boots a resident
TP=4 model in 15–25 s from the per-rank image cache, answers the OpenAI
chat/completions contract over HTTP/SSE with rank 0 as the sole ingress,
and every rank executes an identical op stream (the 4-way op-stream md5 is
the standing ritual). Its default engine seam remains the eager single-token
step: prefill + pick, then one token in / one token out per tick,
time-multiplexed across up to 8 request slots. The optional
`--decode-graph [--mtp]` seam records one scalar graph per physical request
slot as it is first used and one fixed row-batched graph over all configured
slots (up to 8 requests at T=1 or 4 at T=2). Below the measured crossover it
executes the live slots through their scalar variants; at four or more live
requests it selects the row-batched variant and pads closed slots at device
position -1. `CollectiveBus` owns a disjoint generation-cell set for every
recorded variant, so shape switches do not restart the process or graph era.
The T=1 and MTP paths pass the real two-rank loopback gates, including mixed
acceptance, noncontiguous occupancy, scalar↔batch transitions, request-order
reversal, close, and slot reuse without recapture — with the prefetcher on,
after the 2026-09-03 stall hunt made the decode graph kernels-only
(`docs/batched_mtp_graph_stall.md`). The scalar, fixed-batch,
and final adaptive shapes are measured on the four-node service (2026-09-03,
the record entries of that date). The scalar shape runs at
32.0–33.2 ms/token at T=1 and 21.8–26.0 with MTP (43.6–44.1 ms per replay
at 1.69–1.88 tokens per replay, text-dependent), against 31.3 and 22.45 in
`glm_gen_check` — the service adds under 1 ms per replay at T=1 and ~1.5
with MTP. The original always-eight-row graph measured
12.55/23.78/43.59/78.94 tok/s for 1/2/4/8 live T=1 requests and
19.32/35.90/65.79 tok/s for 1/2/4 live MTP requests, versus same-binary
scalar controls of 31.51 and 38.48 tok/s. That failed low-occupancy gate led
to the adaptive path. Its final curves are 30.97/31.67/41.76/76.18 tok/s at
T=1 and 38.48/38.97/60.27 tok/s with MTP: scalar throughput is retained below
four live requests, and batching still wins at and above the crossover. The
MTP rate must be read with its acceptance denominator: the Phase-2 prompt's
1.678 tokens/replay makes 43.603 ms/replay equal 25.99 ms/token. A follow-up
on the current adaptive binary with the exact Phase-1 controls reproduced
21.93–22.21 ms/token on the 16-token Germany workload (previously 21.79),
21.83 on a 32-token repeat, and 24.58 on the 200-token CUDA-graph explanation
(previously 24.81–24.85); replay latency remains 43.6–44.4 ms. There is no
residual single-stream execution regression hidden by the new workload. The
eight-row kernels use scalar-order GEMV chunks so a request is bitwise
invariant across scalar↔batch switches; all 22 full responses in the final
T=1/MTP sweep share one token hash. Phase 2's performance gate is closed. The
eager service, re-measured in the same session, is 36.4 ms/token (the
kernel rounds took it from Stage 4c's ~175 without a service change). The
op-stream md5 was identical across all four ranks of every measured world;
the fixed-batch run's 22 full 256-token answers were token-identical across
both modes and every occupancy. Time to first token is the eager prefill in
every mode, ~30 ms per prompt token.

The fast path exists beside it, in `glm_gen_check`: the recorded decode
step (`--decode-graph`, one CUDA graph per token, 90 collective nodes) runs
at 31.3 ms/token plain, and with the MTP layer (`--mtp`) the whole
speculative step — two-row verify, on-device pick and verdict, predicated
rollback, the draft block and its pick, the next tokens written on the
device — is ONE graph replay at 22.45 ms/token effective, transcript
identical to the plain loop. The service adapter preserves that scalar path
and generalizes it to the fixed Phase-2 request batch behind `glm_serve`.
The weight floor for a step is
~24.5 ms (5.9 GB at ~240 GB/s); the plain step sits 6.8 ms above it, of
which the 90 collectives are ~3.1 ms.

Numerics discipline as it stands: kernels may reassociate fp32 reductions
when it buys latency (decision of 2026-09-02), so the transcript md5 is no
longer the regression signal between builds; `scripts/fabric_xcript.py`
(first divergence judged by bf16-ulp margin) and `scripts/fabric_logprob.py`
(teacher-forced perplexity delta with a standard error, 14k tokens of
texts shipped) are. MTP must still print `IDENTICAL` against plain — the
verify rows are bitwise the single-token rows by construction.

CI is 33 CTest entries (`cmake --build build-ci && ctest --test-dir
build-ci -j4`). `glm_tp_test` and `bus_test` run with
`CUDA_DEVICE_MAX_CONNECTIONS=32`; prefetch is enabled everywhere. The
decode graph is kernels-only by contract (DESIGN §9; the batched-MTP
loopback stall was a copy-engine head-of-line deadlock between the two
ranks' memset nodes, fixed 2026-09-03 — `docs/batched_mtp_graph_stall.md`).
Records live in `benchmarks/results/`; the M5/M6/M8 trail is
`2026-08-29-bus-m5.md`.

Suggested order for what remains, each item's design in its section:

1. M6: sampling on the bus, tool calls and `reasoning_content`, drain-on-stop,
   grow-on-demand admission; the prefill behind the time to first token
   (~30 ms per prompt token — first re-measure it: phase 2's m ≤ 8 GEMV
   routing changed the path of 5–8-row prefill chunks and per-expert
   prefill GEMMs with 5–8 routed tokens, unmeasured).
2. M7: the prefix cache (the snapshot arena and the radix are new; the
   block sharing, the KDA snapshot format, and the journal it rides already
   exist).
3. M9 hardening: failure semantics and drills, the 24-hour serving soak,
   HTTP fuzzing, the sign-off report.

## Audit remediation completed on 2026-08-27

| audit item | resolution and evidence |
|---|---|
| ConnectX topology/105 Gb/s ceiling | Both `f0` lanes mapped and tested: 107.64 Gb/s each, 196.03 Gb/s concurrent. Bulk striping restored. |
| Vision/placement errors in decode budget | Audit now reads `config.json`, excludes `model.visual.*`, separates replicated DSA indexers from TP-sharded MLA, reports zero unmatched tensors, and has Python regression tests. |
| Average-rank TP roofline | Exact multivariate-hypergeometric busiest-rank model added, with replicated DSA indexer/router/mHC/norm reads: 7.457 GB/rank/token, 32.42 ms at 230 GB/s. (Superseded 2026-09-02 by the intermediate-sliced expert placement: every rank reads the mean 5.855 GB/token; DESIGN §3.) |
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

Exit evidence is recorded in `docs/measurements.md`.

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

Since M2 the recurrence and conv kernels gained the optional post-row
snapshot sink that M8 uses for rollback (`KdaStateSnapshots`,
`KdaConvSnapshots`), the recurrence was reshaped to 16 lanes × 8 columns
per v-row (10.9 µs against an 8.7 µs bandwidth floor), and the f_b/g_b
projections run as one dual bf16 GEMV launch — all pinned by the same
suites.

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

Since M3 the decode-path kernels were reshaped for latency (absorb_q 40 →
8.8 µs, vout 34 → 5.8 µs; DESIGN §7.6) and the ring-stash kernel gained
the tail-snapshot sink M8 rolls back from.

## M4 — Full GLM single-node diagnostic assembly (complete)

Deliverables:

1. Config-driven 45-layer adapter, mHC, dense/MoE/shared experts, router,
   vocab-sharded-capable lm head, and inert MTP module.
2. Compressed E4M3+scale resident loader; no persistent BF16 expansion.
3. Scale-aware GEMM parity for block edges and representative tensors.
4. Layer-streaming diagnostic mode so correctness can be tested without
   requiring the full distributed placement.
5. Router trace capture for the corrected per-rank traffic model.

Exit criteria (all met; `benchmarks/results/2026-08-28-glm-m4-assembly.md`):

- next-token logits and greedy tokens pass the curated reference suite; ✓
  (three real-checkpoint cases, per-layer ISOLATED parity with every route
  flip certified as a measured near tie, top-1 agreeing on every token —
  DESIGN §7.5);
- all 37,338 quantized matrices bind to validated scale tensors; ✓
  (`glm_bind_check`);
- actual per-rank bytes reconcile with the shard plan; ✓
  (`glm_stream_check`, per-layer bytes vs formula);
- route traces replace the uniform expert assumption in the performance
  model. ✓ (first real trace lands on the uniform null within 0.4%; the
  question became moot for the FFN boundary when 2026-09-02 sliced every
  expert across the ranks — DESIGN §5.2).

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
   DESIGN §5.2, the two block-boundary folds, per-layer isolated parity
   vs the world=1 oracle with route-flip/head near-tie certification, and
   cross-rank BITWISE hidden/logits/captures/routes at world 2 and 4.
   The fabric runner (glm_tp_check) and real-mesh parity are DONE
   (2026-08-31: the fabric TP=4 run's final_hidden/logits/routes/digests
   are bitwise-identical across all four ranks AND to the loopback
   world-4 verdict dumps — md5 f2674050…/f5a9bf85…). The three items
   this line once listed as remaining have all closed since: prefill's
   boundary folds run as the striped bulk reduce-scatter + allgather
   machine (DESIGN §6.3); the decode step's collectives are recorded
   graph nodes (DESIGN §6.2, M6 Stage 4d); and "the producing GEMM
   writes the send slot" resolved as the kernel-as-stager design — the
   boundary GEMV writes one stable device buffer and the collective
   kernel snapshots it into the registered staging row (one row per
   generation since 2026-09-02).
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

Post-close protocol hardening (2026-09-01, the pick-path race; DESIGN
§6.3): the doorbell carries a positional fold of its payload and every
collective kernel spins until the payload folds to it; every kernel read of
NIC-written memory is a system-scope atomic load (plain loads of pinned
memory cached the pre-arrival zeros forever — the mechanism behind the
gen-1825 wedge family). Racer: `bus_small_repro --pick-race`, loopback and
fabric modes.

## M6 — Generation, tokenizer, and service

Deliverables as written, with their state:

1. Rank-consistent continuous batching, cancellation, and admission budgets.
   BUILT as the scheduler (Stage 2b, 4a) + admission journal (Stage 4b):
   strict alternation (one admission per tick at most, then one round-robin
   decode step), FCFS admission without head-of-line blocking, full-reserve
   admission (`blocks_for(prompt + max_steps)` held for the request's
   lifetime), external cancellation swept at fixed tick top, bounded queue
   (503 at the door), up to 8 request slots. `step(req)` now returns a token
   vector, and EOS/cancellation/cap are applied token-by-token so a
   speculative batch cannot overshoot the public transcript. Engines
   advertise a decode-batch capacity and take the tick's round-robin slice
   through `step_batch`: the eager engine stays time-multiplexed (capacity
   1), while the adaptive graph engine advertises every slot and advances
   the live set through per-slot scalar graphs below four live requests or
   one row-batched replay at four and above (6a phase 2, below).
2. Exact ByteLevel-BPE tokenizer and model-load-time chat-template compiler,
   keyed by tokenizer/template revision hashes. BUILT (Stages 3, 3b):
   `glm_tokenizer` byte-exact against HF tokenizers 0.23.1 on a 55-case
   golden corpus keyed by tokenizer.json's FNV-1a-64 (and certified a
   third way against gigatoken); `glm_chat_template`, a from-scratch Jinja
   interpreter implementing exactly what the GLM template uses and
   refusing the rest at parse time, 26 goldens keyed by the template's
   hash, rendered identically on every rank.
3. Greedy and finite-top-k distributed fast paths plus full-logit gather for
   exact unrestricted top-p/min-p/logprobs behavior. PARTLY BUILT: the
   sampler (`glm_sampler.hpp`) implements all three paths with one
   selection semantics and a counter-based RNG, unit-gated against the
   centralized oracle; on the bus only the greedy pick is wired (host
   gather+broadcast on the eager path, the on-device pick in the graph
   step), and the service refuses `temperature != 0` / `top_p != 1` with
   a named-parameter 400. Design for the rest below.
4. HTTP/SSE endpoints for chat, completions, models, health, and metrics.
   BUILT (Stage 4a): `POST /v1/chat/completions` (stream and non-stream),
   `POST /v1/completions` (string prompt), `GET /v1/models`, `GET /health`,
   `GET /v1/metrics`; single-threaded epoll server with the limit ladder
   (431/413/503/501/411/400); the refusal ladder for unimplemented fields
   (stop, n, logprobs, penalties, seed, tools, response_format, …) as
   OpenAI error objects naming the param. Gates: `http_server_test`,
   `glm_serve_test`, `glm_fabric_serve_test`.
5. Resident serving mode. BUILT (M5, then rounds 9–10): one-pass sources,
   eager construction, sources released after the last layer, the per-rank
   resident image cache (15–25 s to a ready model), optional `mlockall`;
   no privileged node setup (DESIGN §3).

Also delivered under M6 (Stage 2 "rounds", 2026-09-01/02): the profile-driven
decode-step work that DESIGN §7.6 describes — the device-side decode MoE,
the fp8/bf16 GEMV cores, weights on device memory, the sliced expert
placement, the L2 weight prefetcher, kernel reshapes under the
reassociation rule, the release of the checkpoint mmaps, the metronome's
fast-core pin — 393 ms/token (Stage 2) → 31.45 ms/token (round 11).

Exit criteria, status:

- streamed multi-turn chat and tool calls work on TP=4 — chat works
  (multi-turn messages render through the template; SSE streams
  word-by-word with finish_reason/usage/[DONE]); TOOL CALLS DO NOT: the
  template renders `tools`, but the API refuses the field and no
  tool-call parser exists (in scope — 6f);
- tokenizer/template goldens match the checkpoint reference — ✓ (55/55 and
  26/26 byte-exact, hash-keyed);
- all sampling modes match a centralized-logit oracle for fixed seeds — the
  sampler does (unit gate); the distributed paths for non-greedy modes are
  not built;
- latency and throughput results are committed with the reproducible
  workload definition used to obtain them — ✓ for the single-stream fast
  path (`glm_gen_check`, the 300-step "Roman Republic" chat prompt,
  `--decode-graph [--mtp]`, records of 2026-09-02/03) and, since
  2026-09-03, for the concurrency-1 service (nine fixed requests through
  `scripts/serve_bench.py`, paced by `scripts/serve_pace.py`: 21.8–26.0
  ms/token with MTP, 32.1 at T=1, 36.4 eager; 32 tokens in 1.44 s warm) and
  the adaptive multi-request service (`serve_pace.py --waves`: complete
  1/2/4/8 T=1 and 1/2/4 MTP occupancy curves, including fixed-width controls,
  latency distributions, aggregate throughput, graph-mode transitions,
  transcript and rank hashes).

### M6 work and implementation status

**6a. The one-graph step behind the service.** The service's old seam
(`SchedulerEngine`: `prefill(req, prompt) → token`, `step(req, prev) →
token`, `close(req)`) is one token in, one out, and the scheduler feeds the
token it was handed back. The recorded MTP step neither takes a token nor
returns exactly one: the device feeds itself (`d_tokens_ = [next, draft]`,
written by the replay's last node) and a step yields 1 or 2 accepted
tokens. The implemented seam adds `reserve(req, prompt + max_steps)`, and
`step(req)` returns the newly decided tokens (`std::vector<int32_t>`, size
1..T; for MTP these are the verify winners through its accepted row). The
scheduler appends them in order, stops at the first EOS (dropping anything
after it — the request retires, so the state's extra token is never
observed), and truncates at `max_steps` the same way. The scheduler stays a
pure function of the token stream, so the §11 rank-identity invariant is
untouched, and the fake engine in `glm_scheduler_test` scripts multi-token
steps to pin the EOS-in-the-middle and cap-overshoot rules.

The bus has one graph era per process. It originally exposed only one recorded
cell set, so Phase 2 first made the graph request-indexed on the device; the
adaptive closure extends that era with selectable, disjoint graph variants.
Two phases:

- *Phase 1 — concurrency 1: BUILT AND MEASURED 2026-09-03.* `glm_serve
  --max-concurrency 1 --decode-graph [--mtp]`: prefill is eager (bulk
  collectives between windows — the mixed era, DESIGN §6.2),
  first pick is eager, then `GlmGraphEngineAdapter` drives one graph replay
  per tick with slot 0 baked in. On `close`, the next request reuses slot 0,
  reseeds the graph, and does not record a second era.
  `session_reserve_blocks` now runs at admission for both eager and graph
  engines, before the first replay. Gates:
  `scheduler_oneTokenCap_retiresOnPrefillWithoutDecode`, the in-batch EOS
  and cap-overshoot host cases, and real two-rank loopback
  `glm_tp_serving_{plain_,}graph_adapter_matches_plain...` (T=1 and MTP,
  transcript == plain, cross-rank identical, MTP slot reuse with the graph's
  device token feed `[next, draft]` checked against an eager speculator
  after every replay — a stale draft block would not show in the
  transcript, only in the acceptance rate). Measured on the four-node
  service (the record's 2026-09-03 entry): 43.6–44.1 ms per replay at
  1.69–1.88 tokens per replay = 21.8–26.0 ms/token with MTP, 32.0–33.2 at
  T=1, against `glm_gen_check`'s 42.36/22.45 and 31.3 — the service adds
  ~1.5 ms per replay with MTP and under 1 ms at T=1; the eager service
  re-measured at 36.4 ms/token (not Stage 4c's 175: the kernel rounds had
  moved it). Op-stream md5 identical on all four ranks of all three worlds
  over the same 1144 tokens. Time to first token is the prefill (~30 ms per
  prompt token, all modes), the next single-user item.
- *Phase 2 — adaptive scalar/row-batched graphs: IMPLEMENTED, FUNCTIONALLY
  GATED, AND FOUR-NODE PERFORMANCE-GATED 2026-09-03.* Rows are (request slot,
  speculative row) pairs, up to `kDecodeRows = 8`: eight requests at T=1
  or four at the served MTP T=2. The row-batched variant records every
  configured slot once, derives each group's positions from its
  `d_session_pos_`, and turns a closed slot's zero position into -1 padding.
  Admission and retirement therefore change occupancy, not graph shape;
  one scheduler tick advances every live slot in a single replay. Beside it,
  `GlmGraphEngineAdapter` lazily records the exact Phase-1 scalar graph for
  each physical slot. It runs those variants sequentially below four live
  requests and selects the fixed batch at four or more (configurable with
  `--graph-batch-min-live`).

  The complete device path is request-indexed. KDA conv/recurrence and DSA
  consume the same slot-major row map, preserve per-request row order, skip
  padding without touching state, and snapshot speculative state at global
  batch-row offsets. Main-stack and MTP hidden-cache scatters select the
  actual request cache. The picker folds all R×T candidates in one
  collective, emits one verdict per request, carries one aggregate digest,
  and the commit/draft/token-feed kernels independently roll back and feed
  each group. The fabric latency slot is now 64 KiB, enough for eight bf16
  hidden-4096 rows; this is a sizing knob, not a protocol change. Padding is
  semantically inert but still pays the fixed graph's stateless compute,
  expert routing/reads, and collective width — the measured curve below
  makes that occupancy tradeoff visible.

  CUDA graph memcpy nodes retain their pinned source addresses, not captured
  values. Eager admissions reuse the model's row-map staging buffers, so the
  adapter restores the immutable slot-major ids/spans before every replay;
  the recorded H2D nodes then publish them. The close/reuse gates deliberately
  pressure this invariant. While establishing the shared map contract, the
  DSA ring update was also fixed to select the actual `req_ids` slot rather
  than the dense span ordinal (the eager multi-slot path could otherwise
  write slot 1's ring/pools into slot 0).

  Gates: `scheduler_batchEngine_stepsRoundRobinSliceInOnePass`,
  `kda_layer_requestIndexedBatch_matchesIndependentRequestsBitwise`, the
  noncontiguous/padded `dsa_decode_update_multi_request_and_padding`,
  `pick_batch_judges_each_request_and_skips_padding`,
  `pick_batch_draft_selects_last_accepted_row_per_request`,
  `spec_batch_positions_draft_rows_and_token_feeds_are_slot_local`, the
  request-indexed cache gate
  `mtp_batch_hidden_cache_input_and_scatter_are_request_indexed`, and the
  real two-rank model gates
  `glm_tp_serving_plain_batched_graph_matches_independent_sessions` and
  `glm_tp_serving_mtp_batched_graph_matches_independent_speculators`. The
  latter occupies noncontiguous slots 0/3 in the exact 4×T=2 eight-row
  shape, reverses request result order, mixes acceptance counts, pads the
  middle slots, closes/reuses slot 3 without recapture, and checks every
  device `[next,draft]` feed against independent eager speculators.

  `CollectiveBus` now registers up to 16 graph variants. Each variant owns
  its own pinned generation-cell slab and recorded node metadata; arm selects
  a variant and waits against the previous variant's generation count before
  publishing the new window. `bus_test` alternates two shapes with different
  node counts and payload widths. The adapter assigns variants 0..slots-1 to
  slot-specific scalar captures and `slots` to the row batch, all sharing one
  serialized reducer and one fabric graph era.

  Four-node gate: two production worlds kept the full eight-row graph fixed
  while occupancy rose. T=1 at 1/2/4/8 live requests measured
  79.65/84.11/91.77/101.35 ms per replay and
  12.55/23.78/43.59/78.94 tok/s. MTP at 1/2/4 live requests measured
  88.60/95.42/104.22 ms per replay, 1.711/3.426/6.856 aggregate tokens per
  replay, and 19.32/35.90/65.79 tok/s. These are steady all-live windows
  from the final admission through the first retirement, over identical
  43-token prompts and 256-token responses with EOS disabled; prefill and
  staggered admission are outside the window. The 22 complete responses
  across both worlds, every slot, and every occupancy had the same token
  sha256 `7fb21f5fc9b67cb27d329fd22e8c0f236dbacec742222e314dd315c084c4a91b`.
  All four ranks' op streams matched within each world (T=1 md5
  `7e8a2b5db0ea6fa67a5b41e463b8fb0e`; MTP
  `69cf9000949118914b659f7abf33c633`), every peer exited cleanly, and no
  rank logged WARN, ERROR, STALLED, or transport retry. Same-binary scalar
  controls on the identical prompt measured 31.732 ms/replay and 31.51 tok/s
  at T=1, and 43.596 ms/replay, 1.678 tokens/replay, and 38.48 tok/s with
  MTP. The fixed-width one-live points therefore lose 60.2% and 49.8% of
  scalar throughput. That was a failed performance gate, not an acceptable
  padding trade: low occupancy loses to the scalar graph, while
  full occupancy reaches 6.29× its one-live T=1 throughput and 3.40× its
  one-live MTP throughput.

  The first adaptive cut restored the speed curve immediately, but exposed a
  subtler gate: rows that began in a scalar graph and later entered the
  eight-row graph could diverge at near-tie picks because m≤4 used the
  row-independent GEMV cores while m=8 selected cuBLASLt/FP8 tile reduction
  orders. It was rejected. The final path lowers every decode shape through
  scalar-order GEMV chunks of at most four rows (and smaller chunks when the
  48-KiB shared-memory ceiling requires it). New `bf16_gemv_test` and
  `scale_gemm_test` cases prove every M=8 BF16/FP8 output row, including FP32
  epilogues, is bitwise its M=1 result.

  Final four-node gate, same 43-token prompt and 256-token responses: adaptive
  T=1 at 1/2/4/8 live requests measured 32.29/31.58/95.79/105.02 ms per
  physical replay and 30.97/31.67/41.76/76.18 tok/s. Adaptive MTP at 1/2/4
  measured 43.60/43.00/111.13 ms per replay, 1.678/1.675/6.698 aggregate
  tokens per replay, and 38.48/38.97/60.27 tok/s. The comparison denominator
  below the crossover is a physical scalar replay (two live requests execute
  two such replays per scheduler tick). All 22 full responses—T=1 and MTP,
  every occupancy and scalar↔batch history—share token sha256
  `d5e4e1cfff7c3e7c319e0507417db423f04fde9d0b8db9c8ad8ae11ae57cb640`.
  Each world's four rank streams match, every peer exited cleanly, and no
  current-run log contains WARN, ERROR, STALLED, or transport retry. Phase 2
  is closed; full commands, distributions, binary/rank hashes, and the
  rejected first-cut evidence are in the measurement record.

  Per-request latency, read off the same runs (the aggregate curve alone
  hides it): a user's ms per token is 32/63/96/105 at 1/2/4/8 live T=1
  requests and 26/51/66 at 1/2/4 live MTP requests. Below the crossover the
  scalar mode multiplies every user's latency by the live count (two live
  requests are two sequential replays per tick); at the crossover the batch
  is better for the individual user as well as in aggregate — four live T=1
  requests in scalar mode would cost each user ~129 ms/token against the
  batch's 96, and four MTP users ~104 against 66. So the crossover trades no
  per-user latency away; what a user pays for concurrency is the multiplied
  scalar latency below it and the padded replay above it.

  Review follow-ups (2026-09-03, validated on the four nodes — the record's
  last entry has the new binary hash, curves within noise of the closure,
  transcripts and op-stream md5s identical to it): (i) the
  DSA decode path now writes zeros for padding rows (KDA already did), so a
  padded row's block output and its share of the boundary all-reduce are
  deterministic by construction rather than by the row-independence
  argument — gate `dsa_layer_decode_padding_row_is_zero_and_leaves_live_rows_bitwise`;
  (ii) `glm_serve --decode-graph` warm-captures every scalar variant and the
  row batch at startup through one throwaway session at a time
  (`GlmGraphEngineAdapter::warm_captures`; 1.1 s for 8 slots, 0.7 s for 4,
  on every rank), so no capture pauses a live stream; both loopback serving
  gates run it first and still match their eager references, which pins
  that the warm sessions leave no residue. The warm-up starts on the
  journal's clock: a `warm` record rank 0 broadcasts once its model is
  built and the peers hold on before their collectives — without it the
  peers' first collective spun ~8 s in stall diagnostics waiting for rank
  0's slower construction (found by the validation run);
  (iii) `--graph-batch-min-live` defaults to min(4, max-concurrency) and an
  explicit value outside [1, max-concurrency] is rejected instead of
  silently clamped (the adapter still logs if it ever clamps);
  (iv) 2026-09-03/04: the batched-MTP loopback gate's intermittent graph
  stall was root-caused (nsys node trace) to the graph's memset/memcpy
  nodes on the process-shared copy-engine queue — the peer's queued
  post-collective memset held this rank's pre-collective memset while the
  peer's collective spun — and fixed by making the decode graph kernels-only
  (`glm_check_decode_graph` at every capture site; `docs/batched_mtp_graph_stall.md`).
  The CTest prefetch-off mitigation is gone; the MTP gate passes 100/100 at
  32 and at 1 CUDA connection, ctest 33/33 with prefetch on. OWED: the
  GEMM seam now lowers every m ≤ 8 through the GEMV chunks, which moved
  prefill tail chunks and per-expert prefill GEMMs of 5–8 routed tokens off
  cuBLASLt — bitwise different at those shapes and unmeasured; measure
  prefill before the TTFT work starts (below). The kernels-only graph's
  fabric cost is confirmed in the noise (2026-09-04 record entry: T=1 graph
  31.67 ms/step vs the record's 31.3, MTP graph 42.44 vs 42.36–42.42, eager
  36.07 vs 36.4; plain vs both graphs IDENTICAL over 300 steps, one md5 on
  all four ranks). Under compute-sanitizer memcheck the loopback worlds
  report 0 errors but do not complete: the first instrumented prefill
  collective outlasts even the lifted budgets (`DGPP_TEST_WAIT_TIMEOUT_MS`
  joined the bus-timeout and kernel-deadline knobs), on the baseline as
  after the fix — a harness limitation to take up if memcheck of the
  loopback graph tests is wanted end to end.

**6b. Sampling on the bus — IMPLEMENTATION STARTED 2026-09-03**
(deliverable 3's distributed half; DESIGN §10). The configuration slice is
built: `GlmGenerationDefaults` strictly parses `generation_config.json`, keeps
missing-vs-present fields explicit, supplies logged greedy-safe/neutral
fallbacks, and validates generation EOS ids against the vocabulary. Both
generation executables load it beside `config.json`, with its EOS set taking
precedence and the old config value retained only for fixtures/older
checkpoints. The width-sizing instrument is also built: `glm_gen_check
--teacher-file F --sampling-profile` computes the exact full-distribution mass
of global top-{32,64,128,256} at every T=1 teacher position (one diagnostic
candidate/LSE gather), and `scripts/fabric_sampling_profile.py` refuses
incomplete or cross-rank-divergent logs before selecting the smallest width at
or below the 1% fallback bound. The next correctness slice landed 2026-09-04:
`sample_from_prefix` is the host oracle for the eventual device verdict. Given
the canonical global candidate prefix and the fold normalizer of the complete
temperature-scaled distribution, it either samples exactly or returns an
explicit fallback WITHOUT consuming the counter RNG draw — and it is
width-independent by construction: the same arithmetic runs on a prefix and
on the complete list, so a resolved prefix equals the full-logit fallback
(`sample_reference_sharded`, the reference at a vocabulary layout) bitwise,
and a request's outcome never depends on the transported k. (The review of
the first cut found a complete-list shortcut that used a different
cumulative sum and so chose a different nucleus at an exact-tie crossing;
unit gates now pin that case, all four regimes at three widths, and the
pure-temperature walk.) `bus_sampling_prefix` drives that contract through
one real candidate/LSE fold, with penalties before each rank's local top-k,
and carries rank 0's decision digest back to every rank (the greedy pick's
readback invariant); its two-rank loopback gate covers the target
checkpoint's actual default (`temperature=1.0`, `top_p=0.95`, no semantic
`top_k`), a cross-shard tie that survives the penalties, rank identity over
a run of draws, and the flat-distribution fallback.

The eager engines SAMPLE end to end (2026-09-04, the same day): the exact
gather fallback is built (`bus_gather_logits`: the penalized fp32 slices as
ONE bulk collective, four 8-bit digits per logit in bf16 words so NaN
payloads, infinities, denormals and -0 survive the fold bit for bit —
gate-pinned at two stripes), and `make_fabric_sample` is the closure
`glm_serve`'s eager engine and `glm_gen_check` run: the prefix decision at
`kSamplingCandidates` = 128 per rank, else the gather and the complete-list
decision under the transported normalizer with the reserved draw, rank 0's
digest echoed either way (loopback gate: six steps alternating resolved and
fallback shapes, penalties over a growing context, bitwise the sharded
reference at the loader's layout, one draw per step). The request seam:
`SchedulerRequest` carries the `glm_sample::Params` spec and the seed
(greedy by default, so every older manifest keeps its op stream);
`SchedulerEngine::supports_sampling`/`configure_sampling` arm the slot
immediately before its prefill pick (the scheduler refuses a stochastic
request on a greedy-only engine at submit, identically on every rank);
`GenEngineAdapter` keeps per-slot spec/RNG/context and picks greedily at
temperature 0; the journal's tick record carries the spec as float BITS
plus the seed for stochastic submits only (greedy records are
byte-identical to before). The service accepts `temperature`, `top_p`,
`presence_penalty`, `frequency_penalty`, `seed` and the extensions `top_k`,
`min_p`, `repetition_penalty`, validated with the field named in the 400,
fills every omitted field from the checkpoint's defaults, draws a fresh
seed per seedless request (or `--seed`'s), and reports the effective
defaults on `/v1/models` (`"sampling":{"available","defaults"}`); with the
graph engine bound (`--decode-graph`) it collapses the defaults to greedy
with a WARN line and refuses `temperature > 0` with `sampling_unsupported`
rather than advertise a mode it cannot run. `glm_serve --temperature
--top-p --top-k --min-p --repetition-penalty --seed` override the file;
`glm_gen_check` keeps its exact greedy loop by default (its transcripts are
the regression instrument) and samples under `--sample` or any override,
eager engines only. `logprobs`/`logit_bias` stay refused.

The DEVICE path landed for the plain (T=1) graphs the same day (DESIGN
§10 "the device path"): `common/det_math.hpp` makes exp/log bitwise
host/device (and node/node), the sampler runs on it; `kernels/
glm_sample_pick.{hpp,cu}` generalize the pick table to k candidates + the
slice lse per rank (empty-id slots for narrow shards and greedy rows),
apply the penalties in place from a per-request device count table the
kernel itself maintains, select the local top-k with the lifted DSA
composite-key machinery (`kernels/topk_select.cuh`), and decide
`sample_from_prefix` on the device or flag the fallback; `GlmDevicePicker`
records either pick; `GlmGraphEngineAdapter` arms per-slot device specs,
serves fallbacks between windows exactly as the eager engine and reseeds
the graph's token feed, and reports `fallbacks()`. The width fits the
batch's rows into the 64 KiB latency slot (112 per rank at eight rows and
world 4; 128 below seven rows) — the measured profiles still decide
whether that is the right k. `glm_serve --decode-graph` now samples at the
checkpoint's defaults; `--decode-graph --mtp` is still greedy-only and
says so. Gates: det_math_test, the bitwise simulated-world kernel gate in
glm_pick_test, and the two-rank loopback gate of the scalar and batched
graphs against the eager sampling engine with forced fallbacks. Not wired
yet: sampling under MTP (the T=2 accept test, the residual sample, the
draft rollback on a fallback), logprobs on the wire, and the three
real-text fabric runs — no k is fixed until those land in the
measurement record.

The facts that shape it: the model card's recommended and evaluated
settings are `temperature=1.0, top_p=0.95` (the checkpoint's
`generation_config.json`; also `1.0/1.0` and `0.95/1.0` for the agentic
benchmarks) — so the served DEFAULT is full-temperature nucleus sampling,
not greedy, and the design is sized for that regime, not for a
low-temperature corner; the OpenAI API has no `top_k`, so every real
request is `temperature`/`top_p` (plus penalties and `logit_bias`); and a
`temperature: 0` request must keep running the exact greedy path at zero
cost. Design:

- *Defaults come from the model, overrides from the command line* (BUILT
  2026-09-04 for the service and both apps; see above). The
  loader parses `generation_config.json` beside `config.json`
  (`GlmGenerationDefaults`: temperature, top_p, top_k, min_p,
  repetition_penalty when present; the EOS ids already come from it) and
  the service applies them to every field a request omits — the HF
  contract, as vLLM's `--generation-config auto` does — instead of the
  OpenAI wire defaults (1.0/1.0). `glm_serve` and `glm_gen_check` take
  `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--seed` to override
  the file's values for the process (`glm_gen_check` needs them for the
  fabric gates; a missing file or field falls back to greedy with a log
  line naming the gap, never to a silent 1.0). A request's explicit fields
  override both. `/v1/models` reports the effective defaults.
- *The pick table carries each rank's exact local top-k and its slice's
  log-sum-exp* (BUILT 2026-09-04 for the plain graphs; the MTP verify row
  remains) (`kPickSlotsPerRank` 2 → k per candidate row, one more
  digit group for the lse). Penalties and `logit_bias` apply BEFORE the
  local top-k in `glm_pick_local`, from a per-request token-count table
  the commit kernel maintains. The verdict kernel merges the k-way prefix
  in canonical order (the exact global top-k), folds the lse as logaddexp
  (every rank identical), and so knows the EXACT probability of every
  candidate after temperature. It then decides on the device whether the
  request resolves inside the candidates: the top-p cut is inside if
  their cumulative mass reaches `top_p`; the draw `u` (counter RNG,
  `splitmix64` of (seed, counter) — identical on every rank, no broadcast)
  is inside if it lands under the kept mass. Inside → exact HF semantics,
  ~0 µs. Outside → a fallback flag in the pinned verdict.
- *k is sized for T=1.0 / top_p=0.95.* At that setting the cut must reach
  95% of the mass, so k=32 would fall back on every flat position. Plan:
  candidate k=128 per rank (the exact global top-128; 9 bf16 digits per candidate →
  ~9.2 KB per row, two rows plus the digest group inside the 64 KiB latency
  slot), with the local top-128 as a block-wide composite-key select (the
  DSA decode select's shared-memory machinery, ~10–20 µs once per step).
  Before fixing k, MEASURE: the instrumented path now logs per position the
  mass of the global top-k at T=1 for k ∈ {32, 64, 128, 256}; run it on all
  three teacher texts and feed the fetched rank logs to
  `scripts/fabric_sampling_profile.py`. The smallest k with a fallback rate
  at or below ~1% wins.
- *The fallback is the exact gather* (BUILT 2026-09-04 on the eager path:
  `bus_gather_logits`, 1.24 MB/token as 8-bit digits): the fp32 vocab
  slices to every rank as a bulk-class collective between windows, the host
  sampler on every rank with the same `u`. Inside the one-graph MTP step a
  fallback means the draft block ran on a provisional token: the host
  rolls the draft's tail ring back (its snapshot sink exists, unused by
  the draft path yet) and re-runs the draft eagerly on the true token —
  the eager draft path `glm_gen_check` already has. Budget ~3–5 ms per
  fallback; at a <1% rate it is invisible.
- *Sampling under MTP* is exact speculative sampling with a deterministic
  draft (the EAGER driver is BUILT 2026-09-04: `spec_accept_from_prefix`,
  `bus_spec_accept`, `SampledSpeculator`, `glm_gen_check --mtp --sample`;
  the one-graph step's device verdict and draft rollback remain): accept
  draft `x` with probability `p(x)` under the verify row
  (the lse gives `p(x)`), else sample from `p` with `x` removed — the same
  inside/outside test. Acceptance at T=1 is ≈ E[p(draft)], lower than the
  89% argmax agreement (expect 55–70%); the T=2 step's extra row costs
  ~9–11 ms of 42, so the break-even is ~30% and MTP still pays at the
  recommended settings — to be measured, and reported per setting.
- `logprobs`/`top_logprobs` ride the same table (exact for the top-k). A
  request's `seed` is its RNG seed; without one, rank 0 draws it and
  journals it, so every rank draws identically.
- *Gates:* (1) unit — the device verdict path vs the host sampler on
  synthetic logits, bitwise per (seed, counter), including the
  inside/outside decision at the boundary; (2) fabric — a run at the
  card's settings with a fixed seed is identical on all four ranks and
  across two runs of the same binary (the 4-way md5, sampling edition);
  (3) the teacher-forced perplexity gate is sampling-independent and
  stays the numerics judge; (4) the measured fallback rate and MTP
  acceptance at T=1/0.95 go in the record.

**6f. Tool calls and reasoning** (the M6 exit criterion; in scope by
decision 2026-09-03). The template already renders `tools`, assistant
`tool_calls` and `tool` messages (goldens exist); the API refuses the
fields. Design:

- *Request side:* accept `tools`, `tool_choice` (`auto` renders the tools;
  `none` omits them; `required` / `{function: name}` prepend `<tool_call>`
  resp. `<tool_call>{name}` to the generation prompt as a forced prefix —
  the template has no native forced mode), `tool` role messages
  (string content or the template's output lists), and assistant messages
  carrying `tool_calls`. Also `reasoning_effort` (top-level field or
  `chat_template_kwargs`; the template resolves anything but `low`/`high`
  to `max` and injects `Reasoning Effort: …` into the system prompt —
  goldens for low/high exist) and `chat_template_kwargs` generally.
  Thinking is always on for this model: the generation prompt opens
  `<think>` unconditionally.
- *Response side — a parser over TOKEN IDS, rank 0's HTTP thread only:* a
  state machine keyed on the added tokens `<tool_call>` 154843,
  `</tool_call>` 154844, `<arg_key>`/`</arg_key>` 154847/8,
  `<arg_value>`/`</arg_value>` 154849/50 (decode skips special tokens, so
  the text stream cannot see them; the ids can). Segments decode to the
  function name, keys and values; a value is JSON if it parses, else a
  string (the template emits `v | tojson` for non-strings and raw strings
  otherwise — the parse is the inverse). Output per OpenAI: `tool_calls:
  [{id: "call_<16hex>", type: "function", function: {name, arguments:
  <JSON string>}}]`; in streams one delta with `index`, `id`, `name`, then
  one delta with the complete `arguments` string (clients concatenate
  fragments — one fragment is valid); `finish_reason: "tool_calls"` when
  at least one call parsed (the model ends a call turn with
  `<|observation|>` 154829, one of the three EOS ids — that id alone is
  also a signal). Text outside `<tool_call>` blocks streams as content.
- *Reasoning:* the generation prompt ends in `<think>`; the split on
  `</think>` 154842 routes ids before it to `reasoning_content` (the
  vLLM/DeepSeek convention) and after it to `content`; a knob folds
  reasoning into content for clients that expect it.
- *Gates:* the template's own rendering of assistant tool-call messages
  IS the model's output format, so render → encode → parse → compare is
  the golden round trip over the template cases with tool calls (plus
  hand-written malformed streams: unterminated call, value without key,
  nested JSON); `glm_serve_test` pins the streaming shapes and
  `finish_reason`. Determinism is unaffected: the parser is downstream of
  the token stream and runs on rank 0 only.

**6c. Drain-on-stop.** SIGINT during a collective tears the bus down
under the in-flight collective (the peers eat transport-retry-exceeded).
The stop record must be sent only at a tick boundary: `stop()` sets a
flag the engine thread reads at fixed tick top, answers every active
stream with an error event (`finish_reason` absent, an `error` object),
retires the requests, broadcasts the stop record, then tears down. A stop
arriving mid-prefill waits for the prefill (bounded by the longest chunk:
~1.3 s warm; the cold first-touch case is a startup-only phenomenon since
the image cache).

**6d. Grow-on-demand admission.** Full-reserve over-reserves when a
request EOSes early. Evolution: reserve `blocks_for(prompt + min(max_steps,
window))`, grow at a tick boundary when a request's next block is needed,
and SHED (retire with `finish_reason: "length"` and a `pool_exhausted`
note) the youngest request when growth fails — deterministically on every
rank, because the growth decision is a pure function of (meters, request
positions), which the journal already keeps identical. Admission forecast
(`--sched-plan`) reports both policies.

**6e. Batched decode** — subsumed by 6a phase 2.

Decisions taken 2026-09-03 (with the user): tool calls are in scope (6f);
sampling ships as the on-device exact path with the gather fallback (6b),
sized for the model card's `temperature=1.0, top_p=0.95`, with defaults
parsed from `generation_config.json` and overridable on the command line,
greedy remaining the throughput ceiling; MTP stays configurable
(`--mtp`) and is the expected first-class serving mode, so the T=1 graph
path stays maintained as the fallback/diagnostic shape.

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

### Design as it must fit the built engine (DESIGN §8)

What already exists: the DSA pool's blocks are shared by reference through
one block table per request (latent and index blocks co-located, block =
128 tokens = 32 pools); the KDA slot is a fixed 36.39 MiB/rank per request
(recurrent + conv) with an export/import format carrying revision and
geometry (`kda_snapshot.hpp`); the per-request DSA tail ring is 22.5 KB; the
MTP draft block keeps a per-position hidden cache whose LAST row (`h_q`,
8 KB) a resumed request needs; every host position move pushes to the
device mirrors; the admission journal gives rank 0 a channel that every
peer applies in tick order.

The cache entry: `{token prefix (ids), tokenizer hash, template hash,
checkpoint revision, position P, block ids [0, P/128), KDA snapshot slot,
tail ring copy, h_q}`. Entries are taken (a) at prefill end and (b) at
request close (the conversation so far, so the next turn attaches to the
whole previous exchange). Snapshot cost is one D2D copy of 36.39 MiB
(~0.2 ms) plus the tail and `h_q`; the arena is 42 slots at 1.5 GiB.

Lookup: a radix over token ids; a node is attachable only if it owns a
complete snapshot (KDA + tail + h_q) AND every block below it is still
resident (refcount > 0 or pinned by the entry). A match at a non-snapshot
node is cold. Attaching: copy snapshot → the request's KDA slot, bump the
block refcounts and write the block table, copy the tail ring, set
`d_session_pos_`/`d_mtp_pos_` = P, then process the suffix. Suffix rule:
a continuation chunk must carry ≥ kpool tokens (the DSA tail-seed read);
a shorter suffix runs token by token through the decode path (which
handles any position and is what the pick needs anyway: the last row's
logits).

Rank agreement without a new protocol: rank 0 decides lookups, attachments,
snapshots, and evictions as part of admission and journals them
(`{"op":"attach", id, node}`, `{"op":"snap", id}`, `{"op":"evict", node}`)
in the tick record; peers apply. Since the radix is a pure function of the
journaled prompt stream, peers could also derive it, but journaling the
decisions keeps the invariant checkable (a peer whose radix disagrees dies
loudly, like a scheduler divergence). Eviction: LRU over entries with
refcount 0, never a block still referenced by a request; DSA blocks held by
cache entries count against the pool meters so admission sees them.

The exit criterion "hot and cold produce identical logits" is bitwise only
when the hot path replays the cold path's exact chunk sequence: prefill
chunks are 2048-token pool-aligned and bf16 GEMM outputs differ by ulps
across chunk sizes (M2). Two options, one to choose: (i) snapshot only at
positions the cold prefill would chunk at (multiples of 2048 and the
prompt end) — turn-end snapshots at other positions then compare at the
certified near-tie tier, not bitwise; (ii) make cold prefill chunk at
conversation turn boundaries too (deterministic from the rendered
message boundaries), so any turn-end snapshot is a cold chunk boundary and
the criterion is bitwise everywhere. (ii) is the cleaner contract and
costs nothing at decode. DECIDED 2026-09-03: (ii) — prefill chunks at
message boundaries (the rendered-message offsets are known on every rank)
as well as at 2048-token multiples, so a snapshot at any turn end replays
exactly the cold path's chunk sequence.

Not in v1: persistence across restarts, cross-instance federation, DSA
block deduplication below block granularity.

## M8 — Transactional MTP (complete at depth 1)

Deliverables:

1. Three-token initial draft path and verifier microbatch. BUILT at depth 1:
   `session_verify` runs T ≤ 4 rows (`kSpecRows`); the served shape is
   T=2 (`[next, draft]`). The Phase-2 fabric latency slot is 64 KiB so one
   collective can fold the full eight-row request batch.
2. `k+1` KDA candidate states, speculative convolution width, reserved MLA
   slots, and scratch index tail/pool updates. BUILT with a simpler
   mechanism: post-row snapshots (`spec_rec_/spec_conv_/spec_tail_`) taken
   by the recurrence/conv/ring kernels for every row but the last;
   retraction is a predicated copy of snapshot `a−1` per state family
   (`glm_spec_commit`). DSA latent rows and index pools need no rollback
   (positional writes the rewound position overwrites; visible pool count
   derives from the query's own position).
3. Rank-broadcast accepted count/RNG counter and atomic commit/discard epoch.
   As built: no broadcast — every rank computes the verdict from an
   identical gathered table on the device, and a digest carried into the
   next gather catches a divergent rank; commit/discard is a recorded
   predicated kernel behind the verdict (DESIGN §9 "the on-device step").
   The RNG counter is moot while MTP is greedy-only (M6 6b designs the
   sampled form).
4. Adaptive draft depth based on measured acceptance and memory pressure.
   MTP stays a configurable mode (`--mtp`) and is expected to be the
   first-class serving mode (decision 2026-09-03). Not built: depth 2
   was declined for its step-to-step variance (a second draft accepted
   ~60% of the time against a ~29% break-even), and the
   fixed-depth step's remaining cost is inside the graph. Confidence-gated
   T (skip the draft row when the draft's margin is thin) needs the row's
   lse in the pick table — the same addition 6b needs.

Exit criteria, status:

- rejection at every depth, including a pool boundary, matches normal
  decode — ✓ at depth 1: `glm_tp_full_graph_step_loopback_matches_eager_
  speculator` runs the one-graph step in lockstep with the eager
  speculator across pool boundaries with rejections at every step where
  they occur; the fabric transcript is IDENTICAL to plain over 300 and
  1000 steps (a dedicated construction that forces a rejection exactly at
  a pool boundary is not written — the lockstep runs cross boundaries
  with mixed verdicts, but by chance of the text, not by design);
- cancellation and injected rank failure leave committed state unchanged —
  NOT TESTED (no failure injection exists; M9 hardening);
- temperature-zero output is identical with MTP on/off — ✓
  (`scripts/fabric_xcript.py` prints IDENTICAL; the verify rows are
  bitwise the T=1 rows by construction);
- speedup is reported with acceptance distribution and scratch-memory cost,
  with no predeclared multiplier treated as fact — ✓ (88.7% accepted on
  coherent text, 63% on post-EOS rambling, 1.89 tokens/step, 22.45 vs
  31.34 ms/token; the draft layer adds ~7.3 GiB/rank; record entries of
  2026-09-03).

Remaining, ranked: sampling under MTP (6b);
`FabricPicker` refactor so the plain loop's host pick and the graph loop's
device pick share one driver; the prefill's last-row head through the
draft (the first draft is eager today); a forced pool-boundary rejection
test.

## M9 — Optimization and hardening

Optimization, as done so far (records of 2026-09-01..03; DESIGN §7.6):
the T=1 profile-driven rounds 1–11 (393 → 31.45 ms/token), the MTP step
(→ 22.45), and the one-graph/on-device step. Remaining headroom is inside
the graph: the 94 collectives (~4.6 ms/step at T=2; handshake 8–16 µs and
the ranks' ~3% compute spread), ~1.1 ms of graph gaps, and the second
verify row's ~7 ms of unshared experts. The small-kernel fusion round of
2026-09-03 found the floor (net zero; two fusions kept, two reverted with
their numbers) — further gains need fewer, bigger kernels or a change of
shape, not fusions. Route placement is settled (sliced experts); grouped
MoE kernels exist for decode (prefill stays host-orchestrated —
acceptable while prefill is chunk-amortized; revisit if TTFT at 32K
matters). Phase 2's fixed graph exposed the padding cost (12.55 T=1 and
19.32 MTP tok/s at one live request); the completed multi-variant adapter
now selects scalar graphs below four live requests and the row batch above
it without restarting the fabric era. The final curve is
30.97/31.67/41.76/76.18 tok/s at T=1 and 38.48/38.97/60.27 with MTP, with
transcripts bitwise invariant across graph-width transitions.

Hardening, not started, designed:

- *Failure semantics (v1):* any rank failure fails the service — there is
  no failover. Rank 0's death: peers see journal EOF and exit. A peer's
  death: rank 0's next journal write throws or the bus watchdog fails the
  in-flight collective; rank 0 answers every active stream with an error
  event, then exits nonzero. `serve_run.sh` (or systemd) restarts the
  world; the resident image makes that ~25 s. Drill: kill −9 a random
  rank under load, assert every client got an error event, the peers
  exited within the watchdog, and the restart serves.
- *Counter-drift checks:* the 4-way op-stream md5 becomes continuous —
  every N ticks the journal carries rank 0's running fold; a peer whose
  fold differs dies loudly with the tick number (today the check is
  post-mortem).
- *Cancellation:* covered at the tick boundary (disconnect → cancel queue →
  retire; gated); drain-on-stop is 6c.
- *24-hour soak:* `glm_serve` under a client driver replaying a mixed
  workload (short chat, long generation, cancellations, a burst above the
  queue bound), with `--node-probe` and the step distribution per hour;
  pass = no stall windows, no drift, flat p99.
- *Malformed-HTTP fuzzing:* a byte-level mutator over the server's limit
  ladder (`http_server_test` pins the ladder; the fuzzer looks for a way
  past it), run under ASan.
- *Restart documentation:* `serve_run.sh up/down/status`, the image cache,
  the memlock note, the port/rendezvous window (README has the pieces).

Final performance sign-off reports:

- batch-size-one decode and 32K TTFT for a committed workload definition
  (decode: done for the single-stream path; TTFT at 32K not measured —
  prefill is the host-orchestrated MoE path at ~2048-token chunks);
- expected, p95, and worst-rank expert traffic from traces (moot at the
  FFN boundary since the sliced placement; the trace tool remains for
  the attention-side and for any future placement change);
- both-lane utilization and collective share (per-collective timeline
  exists; the share is ~10% of a plain step, ~11% of an MTP step);
- prefix-cache capacity/hit curves (M7);
- MTP acceptance and net speedup (done for one prompt class; report per
  class);
- known gaps, rather than silently moving unmet targets.

## Post-v1

Vision processing, NVFP4 conversion, expert replication/placement beyond M9
telemetry, data-parallel replicas, cross-instance prefix federation, TLS
fronting, and packaging remain out of the v1 critical path.
