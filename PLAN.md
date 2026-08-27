# DGPP — Phased Implementation Plan

Version 0.1 — 2026-08-27
Cadence: weekly-ish milestones, each ending in a demonstrable, measurable state.
Metric north star: **single-stream decode tok/s on TP=4** (G1), secondary: p99 TTFT, aggregate throughput, robustness.

Status legend: `[ ]` todo · `[~]` active · `[x]` done · `[!]` blocked

| # | Milestone | Status | Owner |
|---|---|---|---|
| M0 | Foundations & measurement campaign | [x] | |
| M1 | Core runtime & weight pipeline | [x] | |
| M2 | KDA linear-attention ops | [ ] | |
| M3 | DSA sparse-attention ops | [ ] | |
| M4 | Full GLM-5.3-Flash assembly (single-node bring-up) | [ ] | |
| M5 | Cluster parallelization (RoCE collectives, TP=2→TP=4) | [ ] | |
| M6 | Generation loop & OpenAI-compatible serving | [ ] | |
| M7 | Cache subsystem (prefix caching, hybrid replay) | [ ] | |
| M8 | MTP speculative decoding | [ ] | |
| M9 | Performance push & hardening | [ ] | |

Working agreements
- Requires coordination: full-model memory fits **only** when vLLM is paused (single node holds ≤ ~84 GB; free RAM ~3–30 GB depending on baseline server). Parity/measure windows must be agreed in advance or run against other nodes.
- Every milestone lands green `scripts/ci-local.sh` (build -Werror, unit tests, formatting).
- Parity gates are additive: once introduced they rerun forever in CI where feasible (fast golden subsets).

---

## M0 — Foundations & Measurement Campaign (≈1 wk)

Deliverables:
1. Repo scaffold (layout per DESIGN §4.2), CMake presets (`sm_121`, release/asan/ubsan), vendoring policy applied (§9 minimal set), baseline logger/CLI apps compile.
2. Microbenchmark suite `benchmarks/micro/`:
   - device memory BW streaming (read/mixed patterns that mimic GEMV), achieved GB/s;
   - FP8/BF16 GEMM peaks via cuBLASLt across shape sweep (decode GEMV-like and prefill tiles);
   - ibverbs smoke: RC QP pair setup, one-shot small-message RTT, ring BW on the active fabric (`rocep1s0f0`, subnet 88.x). NOTE (user-confirmed + verified): only ONE CX-7 per node is wired/usable on DGX Spark → no dual-fabric striping; single 200 Gb/s-class path per node. Cross-node perftest (ib_send_lat / ib_write_bw) is the authoritative M0 source for these numbers;
   - GPUDirect RDMA registration probe (device-memory MR) with pinned-bounce fallback timing;
   - host↔device coherency cost check (freeze/residency behavior) on unified memory.
3. Checkpoint audit: tensor inventory script over unsloth FP8 repo → exact per-tensor shapes/dtypes, refined per-token byte-budget table (replaces DESIGN §3.2 estimates), shared-expert dims confirmed.
4. Network facts sheet: PFC/ECN state, MTU path validation, RTT matrix across 4 nodes; sudo/module availability noted.

Exit criteria: `docs/measurements.md` populated with all numbers; build+tests green on node A. (Striping criterion removed: platform has a single usable CX-7 per node; bandwidth budget revised to measured ≈105 Gb/s/link.)

## M1 — Core Runtime & Weight Pipeline (≈1–2 wk)

Deliverables:
1. Arena allocator (hot/cold residency classes), stream pool, event graph executor, tracing hooks, CUDA Graph manager keyed by step-shape class.
2. Safetensors mmap reader + shard descriptor format (`shardspec.json` generated from config audit) covering FP8/BF16/F32 dtype mixtures.
3. cuBLASLt-backed `IGemm` impl; fused RMSNorm/SwiGLU(limit)/elementwise mHC kernels.
4. Tiny synthetic transformer ("gpt-doll") running prefill+decode under captured graphs; deterministic replay test (bitwise across replays).

Exit criteria: gpt-doll generates coherent-ish sequences with greedy sampling from random weights at ≥90% of naive-roofline on one device; allocator reports zero steady-state allocations during 10-minute loop.

Status (2026-08-27): DONE. Bitwise graph-vs-eager parity 48 steps; 60.1 tok/s
@ 97% naive roofline; soak 37k steps / 10 min with zero alloc growth;
shardspec generated over the real checkpoint (76,108 tensors). See
docs/measurements.md §M1. Known doll deviations: no RoPE/per-head norms;
naive O(T²) prefill attention and single-warp decode attention are
correctness-first — real kernels arrive M2/M3.

## M2 — KDA Linear Attention Ops (≈2 wk)

Deliverables (all as `.cu`, behind op interfaces):
- Short conv (kernel 4) fused into projections; gated delta-rule **chunked scan** for prefill; recurrent-state update kernel for decode; boundary math matching reference semantics exactly (incl. A_log exponents, dt clamps, gate lower-bound −5, fp32 accumulation zones).
- Python reference implementation checked into `tools/ref/kda.py` (mirrors FLA formulation) generating golden tensors.

Tests: sequence-length sweep property tests (prefill(0..i)+decode(i) ≡ prefill(0..j) compositionality); tolerance-tiered vs golden; long-recurrence numerical drift bound assertions.

Exit criteria: parity tiers pass (fp32-accum rtol 1e-3 zone); decode-state update kernel sustains roofline-plausible bytes/token measured; design notes published on chosen chunk size.

## M3 — DSA Sparse Attention Ops (≈2 wk)

Deliverables:
- Indexer stack: q/k projections, interleaved RoPE, compressed k-pool builder (APE + gate), fp32 score reduce with fixed tie-break top-k=`index_topk` (determinism guarantees spec'd + tested across simulated ranks/seeds).
- MLA latent path: kv_a/q_b/kv_b projections fp8 dynamic-scale, decode gather-MHA over selected tokens (batched gather kernels), prefill equivalent honoring causal windows.
- Cache block plumbing sufficient for attention ops to read latent blocks + aux records directly.

Exit criteria: end-to-end layer-level parity vs transformers dump on real checkpoint slices within set tolerances; top-k selection bitwise-stable in 1000-trial fuzz; indexer+kernels hit roofline ≥70% at bs=1 decode shapes.

## M4 — Full GLM-5.3-Flash Assembly, Single-Node Bring-Up (≈1 wk)

Deliverables:
1. `models/glm53` adapter complete: config parse, weight map, MoE router (sigmoid/noaux_tc/scaling 2.5), shared expert, dense-MLP layers, lm_head vocab-shard mode reusable standalone, MTP module present-but-inert.
2. *Layer-streaming* diagnostic mode exploiting unified memory (mmap lazily resident weight shards larger than local free RAM) to execute the entire model on one node slowly-but-correctly **without displacing vLLM**.
3. Logits harness: compare engine vs reference dumps on curated prompts (EN/ZH/code/chat-template variants).

Exit criteria: end-to-end next-token distribution matches reference at temperature-0 on ≥50 curated prompts (argmax agreement ≥99%, full-KL margin threshold defined & met); budget-table actual-vs-predicted weights traffic report within ±10%.

## M5 — Cluster Parallelization (≈2 wk)

Deliverables:
1. Control plane (TCP gossip/roster/coordinator epochs), startup choreography (`--rank/--world/--peers`), health hooks.
2. `CollectiveBus`: one-shot tree small-message allreduce; ring ReduceScatter+AllGather lanes on the single wired fabric (striping removed per M0 — one usable CX-7/node; 4 RoCE ports enumerated but wiring unverified); **zero-copy pinned receives per DESIGN §6.1 v0.2** (flag protocol already shipped in `src/kernels/flag_protocol.cuh` with ctest regression coverage); completion-driven non-blocking advance integrated with stream events; NCCL fallback backend compiled behind flag (insurance only).
3. Sharded load of real checkpoints onto TP=4 (also exercised at TP=2 for the same code paths); replicated tensors (router/indexer/normals) checksummed in lockstep at boot.
4. Cross-rank execution: distributed forward returns identical results to M4 single-node mode.

Exit criteria: TP=4 argmax-parity suite passes (same 50 prompts); decode-step wall-clock stable across 1 h mixed-prompt soak with no collective stalls >25 ms; collective overhead share measured ≤8% of step time at bs=1 (else revisit M9 priorities).

## M6 — Generation Loop & Serving (≈1–2 wk)

Deliverables:
1. Continuous-batching scheduler core: request lifecycle, prefill/decode mixing policy favoring interactive streams, cancellation propagation to every rank.
2. Tokenizer suite (GLM BPE automaton + roundtrip goldens), compiled chat template (codegen vs HF-rendered strings equality-tested), glm47 tool-call incremental parser with streaming deltas.
3. Hand-rolled epoll HTTP/SSE server exposing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`, `/metrics`; sampling params stack completed (seeded reproducibility mode).

Exit criteria (demo!): `curl` multi-turn streamed chat works end-to-end on TP=4 incl. tool calls; measured bs=1 median decode tok/s recorded in `docs/measurements.md` alongside live vLLM numbers from an identical scripted workload (side-by-side table committed).

## M7 — Cache Subsystem (≈2 wk)

Deliverables:
1. Block pool arenas, radix prefix trie (128-token blocks), refcounted sharing, LRU+depth-weighted eviction with cross-rank agreement protocol.
2. Hybrid reuse pipeline: longest-prefix acquisition, snapshot-or-replay reconstruction of linear states + indexer pools; snapshots opportunistic store; per-request opt-out flag.
3. Telemetry: hit-token counts, replay ms, saved-tokens curves surfaced in `/metrics`.

Exit criteria (demo!): conversation replay across turns shows TTFT collapsing from T_cold to bounded by replay time; ≥95% token-reuse on scripted agent-style workload (long static system prompt suite); numerics unchanged vs cold path (parity suite re-run with cache hot).

## M8 — MTP Speculative Decoding (≈1 wk)

Deliverables:
1. Draft (nextn) path + verify-as-k-batch integration into graphs; acceptance accounting and adaptive depth control (start 3).
2. Quality-neutral gating: acceptance-rate telemetry; verified distribution equivalence when candidate rejected vs normal step.

Exit criteria: ≥1.8× geometric-mean speedup over M6 bs=1 baseline on chat/coding prompt mixtures; zero divergence from greedy outputs when temperature=0 (spec-mode transparent).

## M9 — Performance Push & Hardening (≈2 wk, extends until targets met)

Tracks (data-prioritized from telemetry):
1. Prefill comm/compute software pipelining; chunk-shape tuning; TTFT regression guard on long prompts.
2. Kernel substitution wave-2: replace any cuBLASLt sites where CUTLASS-custom beats it (expected: grouped-MoE GEMM, MLA decode path); persistent-kernel candidates reviewed; single-fabric saturation re-check (striping unavailable per M0).
3. Robustness drills: peer kill −9 mid-stream → graceful partial response + recovery docs; restart/join runbook; 24 h soak with memory/RDMA counter drift analysis; fuzz HTTP surface.
4. Final report vs goals: targets table (below) signed off; known-gaps appendix.

Exit targets (TP=4 unless noted):

| Metric | Target | Stretch |
|---|---|---|
| Single-stream decode tok/s | ≥2× live-vLLM baseline AND ≥85 tok/s | 110+ |
| p99 TTFT @32K-token prompt | ≤12 s | ≤7 s |
| Prefix-cache token-hit ratio (agent suite) | ≥95% | 98% |
| Aggregate tok/s @16 concurrent streams | baseline-equal or better than vLLM | +30% |
| MTBF under kill-drill set | clean error propagation, no crash loops | auto-rejoin (post-v1) |

---

### Post-v1 backlog (parked)
NVFP4 experts track · EP experiment decision from E2 data · DP replicas & smart routing · vision input processor · TLS/fronting integrations · multi-node prefix-cache federation beyond one instance · operator docs polish, packaging (.deb / container image).
