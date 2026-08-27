# DGPP — DGX Spark Inference Engine: Design Document

Version 0.2 — 2026-08-27
Target model: `zai-org/GLM-5.3-Flash` (FP8 native checkpoint)
Target platform: NVIDIA DGX Spark (GB10 Grace Blackwell), cluster of 4 nodes, 2× 200 Gb/s RoCE per node

---

## 1. Goals, Constraints and Priorities

### 1.1 Product goals

| ID | Goal | Priority |
|----|---------------------------------------------------------|----------|
| G1 | **Maximize single-stream decode throughput** (tok/s at batch≈1, interactive agentic/chat traffic) | PRIMARY |
| G2 | OpenAI Chat Completions–compatible API incl. streaming SSE | Must |
| G3 | Working prefix caching (multi-turn / shared long prefixes) | Must |
| G4 | Clustered tensor-parallel operation over RoCE/ibverbs across N∈{2,4} Sparks; TP=1 supported for smaller models & dev loops | Must |
| G5 | Modular architecture to admit future models without engine surgery | Must |
| G6 | Text-only for v1; vision encoder is a pluggable input processor added later | v1 scope |

Not goals for v1: training/fine-tuning, logits APIs beyond completions, embeddings endpoints, TLS termination (assume a fronting proxy if needed).

### 1.2 Why these priorities matter technically

GB10 delivers ~273 GB/s of LPDDR5x bandwidth. GLM-5.3-Flash activates ≈18 B params/token; audited checkpoint traffic (see `docs/checkpoint_budget.md`) works out to **≈23.5 GB/token** on a single node (FP8 experts + BF16 linear-attention path) → hard ceiling of ≈10–11 tok/s on one device. Tensor parallelism divides that stream:

```
per-token weight traffic   T ≈ 23.5 GB (checkpoint mix, bs=1)
per-node traffic           T / N            (weights sharded across N ranks)
decode floor @ bs=1        t_step ≈ (T/N) / BW_eff
                           N=4: ≈ 5.9 GB / ~230 GB/s ≈ 26 ms → ~39 tok/s
```

So the path to G1 is: (a) shard weights across the cluster with **minimal per-layer collective latency**, (b) drive each node's memory bus close to saturation, (c) amortize via **MTP speculative decoding** (the checkpoint ships an MTP head, `num_nextn_predict_layers=1`), which multiplies accepted tokens per step by ~1.8–2.5×, and (d) requantize BF16 hot paths to FP8 as a quality-gated follow-up (~10 GB/token of headroom). Realistic target: **60–100 tok/s single-stream** on TP=4 (measured against the running vLLM baseline on this same cluster).

Aggregate throughput under concurrency remains supported (continuous batching) but is tuned *second* to interactive latency.

---

## 2. Platform Facts (verified in situ)

| Item | Value |
|---|---|
| SoC | GB10 Grace Blackwell Superchip; GPU CC **12.1** (`sm_121`); driver 580.173.02 (open kernel module); CUDA 13.0 |
| CPU | 10× Cortex-X925 @3.9 GHz + 10× Cortex-A725 @2.8 GHz, aarch64, SVE2/BF16/I8MM |
| Memory | 128 GB LPDDR5x unified (GPU/CPU coherent over NVLink-C2C), 256-bit, **273 GB/s**, usable ~121 GiB |
| Compute | "up to 1 PFLOP FP4" (sparse). Plan conservatively for FP8 dense ≈100–150 TFLOPS/board — to be measured (M0) |
| Storage | 3.6 TB NVMe, 832 GB free here; FP8 checkpoint already cached locally (`~/.cache/huggingface/hub/models--unsloth--GLM-5.3-Flash-FP8`) |
| Network/node | 2× ConnectX-7 present; lspci enumerates **4 RoCE ports** (segments `0000` & `0002`, all links healthy Gen5 x4 @32GT/s), but per NVIDIA platform design only ONE adapter is wired per node (single QSFP path — switch-side verification pending). Active port: `rocep1s0f0` @ 200 Gb/s line rate, MTU 9000 (subnet `192.0.2.x`). **Measured usable unidirectional RC throughput ≈ 105 Gb/s (≈13.2 GB/s)** across 256 KB–8 MB message sizes — host-side capped by platform topology (M0 measurement) |
| RDMA stack | `mlx5_core/mlx5_ib/ib_uverbs/rdma_cm` loaded; RoCE transport; `libibverbs` + `librdmacm` userland present; **ib_send_lat cross-node RTT ≈ 2.7 µs** (measured M0) |
| OS/toolchain | Ubuntu 24.04 (DGX OS), kernel 6.17-nvidia, g++ 13.3, cmake 3.28, python 3.12 |
| Baseline system | Live vLLM deployment: `unsloth/GLM-5.3-Flash-FP8`, `--nnodes 4 --tensor-parallel-size 4`, fp8 KV cache, MTP speculative config `num_speculative_tokens=3`, deep_gemm MoE backend, max-len 393216. This is our performance/comparability reference |

Open items to probe in M0 — ALL RESOLVED (details in `docs/measurements.md`):
- Memory BW ≈232 GB/s streaming read; FP8/BF16 GEMM peaks measured (M0 tables).
- **GPUDirect RDMA: resolved twice over.** peermem exists on the driver (580.173.02)
  but is *structurally inapplicable* (PCIe-BAR export model; GB10 has no discrete
  GPU). More importantly the question dissolved: unified memory means NIC DMA
  lands in memory the GPU reads **zero-copy** — pinned receives measured FASTER
  than cudaMalloc reads (255 vs 239 GB/s). §6.1 updated accordingly (v0.2).
- RoCE PFC/ECN switch config: still pending (needs switch mgmt credentials, not
  node sudo).
- Sudo: passwordless-by-default no; narrow sudoers grant (`modprobe/dmesg/lspci/
  ibstat`) in place on node A — sufficient for module/PCIe diagnostics.
- Checkpoint structure audited: 76,108 tensors / 62 shards / 305.8 GiB; role
  census in `artifacts/shardspec.json`; production loader binds it in 0.17 s.

---

## 3. Model Anatomy: GLM-5.3-Flash

320 B total params / ≈18 B active/token. Text config `glm5_next_text`, architecture `Glm5NextForConditionalGeneration`.

### 3.1 Layer census (45 main layers + MTP layer)

| Component | Count | Notes relevant to implementation |
|---|---|---|
| Linear attention ("KDA"-style gated delta net) | **34** (pattern LA×3 then DSA×1, repeating) | recurrent state S ∈ R^{64 heads × 128 × 128} per layer; short conv kernel 4 on Q/K/V; gates bounded [-5,∞) (`gate_lower_bound=-5`), A_log/dt_bias dynamics ⇒ chunked delta-rule scan kernels |
| DeepSeek-sparse attention (DSA) full attention | **11** | MLA-compressed KV: `kv_lora_rank=512`, `q_lora_rank=1536`, heads=64, qk/v head dims =256/256, **no decoupled RoPE on the main path** (`qk_rope_head_dim=0`, `mla_use_nope=true`). Lightning-indexer selects `index_topk=2048` tokens via `index_n_heads=32 × index_head_dim=128`; indexer has its own RoPE (`indexer_rope_interleave=true`) + compressed k-pool (`index_kpool=4`, compress w/ APE + gate) |
| Dense MLP layers | first 3 | intermediate 12288, SwiGLU with limit clamp 10.0 |
| MoE layers | remaining 42 | 288 routed experts (top-8, sigmoid gating + noaux_tc + score-correction bias, `routed_scaling_factor=2.5`, router fp32) + 1 shared expert (intermediate ~12288) |
| Hyper-connections | all layers | **mHC**: manifold-constrained hyper-connections, `hc_mult=4` → residual stream is 4×hidden = 16384 wide; per-stream mixers (base/fn/scale) feed separate paths into attn and FFN |
| MTP head | 1 | layer-45 module: enorm/hnorm/eh_proj/shared_head.norm + own indexer sharing config (`index_share_for_mtp_iteration=true`) — enables speculative decoding |
| Vocab | 154880 | embed & lm_head kept high precision (excluded from FP8 conversion) |
| Vision tower | separate | ViT-ish depth-24, h=1024, patch 14, merge 2 — deferred (G6) |

Checkpoint audit verification (loader smoke, 2026-08-27): real names use the
`model.language_model.layers.*` prefix with `hc_*` hyper-connection wrappers
(`hc_attn_{base,fn,scale}`, `hc_ffn_*`); experts ship **raw-magnitude E4M3
(mean |v| ≈ 61) + separate `weight_scale_inv` block-scale tensors** ⇒ dequant-
on-load is mandatory (raw bytes are not unit-range); LA `dt_bias` is F32
[8192] = 64 heads × 128 (matches the §3.1 state geometry); `lm_head` BF16
[154880, 4096] as budgeted (1.27 GB/token — see `docs/checkpoint_budget.md`).

### 3.2 Per-token serving budgets (updated by checkpoint audit)

See `docs/checkpoint_budget.md` for the audited numbers (M0). Headline result:
unsloth's FP8 export keeps the entire linear-attention projection path in
BF16, so measured-class totals give **≈23.5 GB/token** cluster-wide at bs=1
(vs the ~17 GB earlier guess that assumed FP8 experts *and* attention).
Implications:
- TP=4 share ≈ 5.9 GB/node/token → decode floor ≈ 26 ms @230 GB/s effective;
  ~39 tok/s pre-speculation, ~70–100 tok/s with MTP acceptance 1.8–2.5×.
- Requantizing the LA path (BF16→FP8, ~10 GB/token worth) is a high-value,
  quality-gated optimization track (§13).

Per-token persistent state (KV): only the 11 DSA layers hold token-indexed cache: latent 512 bf16 + index bits ≈ **1.25 KB/token/layer ⇒ ~14 KB/token total** — exceptionally cheap; 300 K context ≈ 4.2 GB per sequence *cluster-wide*. The linear-attention layers trade token-indexed KV for O(1)-context but large per-sequence state: 64×128×128×(2B)×34 ≈ **71 MB/sequence**. Implication for caching: §8.

### 3.3 Key numerical implications

1. **Two distinct state machines must be implemented well**: chunked delta-rule scan (linear) and sparse top-k gather MHA (DSA). These are the two hardest kernel families — budget most kernel time there.
2. Deterministic indexer selection requires identical top-k results on every rank (§6.2 handles shard symmetry).
3. mHC residual streams are internal activations; they do not persist in cache. But they widen activation shards handled in collectives (16384-wide residuals stay local; communicated tensors are the projected 4096-wide signals).
4. SwiGLU limit clamp and gate lower bound: trivial fused-elementwise items, easy to miss for parity — include in the numeric test matrix.
5. Router in fp32 sigmoid+noaux_tc+bias: replicate router weights on all ranks (negligible size) so every rank computes identical routing locally without extra collectives; keep deterministic reduction order inside the router (fixed expert traversal order).

---

## 4. System Overview

### 4.1 Process/thread topology (per node)

```
┌───────────────────────────────────────────────────────────────────┐
│ dgppd (daemon)                                                     │
│                                                                    │
│  ┌ API thread(s) ── HTTP/1.1+SSE server, JSON parse/render ──┐    │
│  │        ▼                                                   │    │
│  │  Scheduler ── continuous batching, chunked prefill policy, │    │
│  │              prefix-cache admission, spec-decode control    │    │
│  │        ▼ request-scoped token graphs                       │    │
│  │  EngineCore (control thread)                               │    │
│  │    · owns CUDA ctx, streams: [compute][copy][collective]   │    │
│  │    · builds/replays CUDA Graphs per step shape             │    │
│  │    · issues collectives into CollectiveBus                 │    │
│  ├─ Net threads ─ ibverbs completion polls ×2 fabrics         │    │
│  ├─ Control plane ─ gossip/discovery, health, member roster   │    │
│  └─ Tokenizer workers (ARM cores, lock-free job queue)        │    │
└───────────────────────────────────────────────────────────────────┘
      │ C2C/NVLink-C2C unified memory (no host<->dev copy for big data)
      ▼
   GPU (sm_121): KernelBatch { prefill | decode(+verify) }
```

Single process per node; rank role decided by CLI (`--rank r --world 4 --peers ...`). Rank 0 additionally runs the coordinator (membership, scheduler arbitration, eviction agreement).

Memory plan per node (TP=4):

| Region | Budget |
|---|---|
| Weight shard (FP8) | ~80–84 GB |
| Collectives/staging (pin + device bounce) | ~1.5 GB |
| Workspace pools (GEMM scratch, route metadata, spec windows) | ~2 GB |
| KV latent blocks + index pools + linear-aux replay records | remaining ~30 GB headroom + unified-memory elasticity |

Because memory is *coherent unified*, the allocator can treat host-pinned staging and device memory as one address space with explicit residency classes — but we never rely on transparent migration for hot-path tensors (freeze + prefetch explicitly).

### 4.2 Repository layout (C++20, CMake)

```
src/
  core/       alloc, streams, cuda graph mgmt, event graph executor, tracing
  kernels/    cuda (.cu) : gemm ops, norm/silu/mhc fusion, kda scan,
              dsa(indexer+topk+gather-attn), moe grouped gemm/router, samplers
  ops/        op interfaces; tiled execution plans wired to kernels
  models/     glm53/ (weights map, graph builder), registry, vision stub slot
  runtime/    step scheduler inside engine core; cuda-graph capture shapes
  parallel/   CollectiveBus abstraction + ibverbs impl; sharding specs
  netfabric/  rdma connection manager, qp pools, gdr probing, bounce staging
  cluster/    membership/coordinator, bootstrap, health/failure policies
  cache/      block pool, radix prefix tree, hybrid-aware replays, eviction
  serve/      http/sse server (hand-rolled epoll), openai schemas, chat
              template compiler, tool-call parser (glm47), tokenizer
  tools/      dump-logits, budget-report, microbenches, parity harness
apps/         dgppd (server), dgppctl (CLI/tests), benchmarks/
third_party/  vendored minimal set (see §9)
tests/        unit + parity + soak
docs/, scripts/, deploy/ (ssh-based orchestration across 4 nodes)
```

Model future-proofing contract: adding a model = adding a `models/<name>/` adapter (config parser + weight-name map + op-graph builder + optional custom ops registration). Everything beneath (`ops/`, `parallel/`, `cache/`, `serve/`) is model-agnostic.

---

## 5. Execution Model

### 5.1 Step pipeline (decode, TP=N)

Per decode step per sequence:
1. Sampled last token id → embed row read (bf16, sharded vocab or full-row replicated — decision: replicated embed table rows lookups cost ~nothing; avoids gather-after-allgather… embed is per-rank lookup anyway).
2. For L=0..44 execute layer op-graph; inside each layer:
   - projection GEMMs run **head-sharded** (64 q-heads → 16/rank),
   - MoE runs **expert-sharded** (72 experts/rank; token routed to up-to-8 local experts → local grouped GEMM; partial outputs reduced within the MoE reduce-scatter),
   - two synchronization points per layer enforced by graph edges: post-attention o_proj output (reduce-scatter) and post-FFN output (all-gather or reduce+gather depending on shard scheme — see §6.3).
3. Hidden → fp32 softmax sampling (or greedy path fast lane).
4. Optional MTP: draft k=1..3 tokens with nextn module (cheap weights), verify candidates in one expanded decode pass.

Executed as captured **CUDA Graphs** keyed by step-shape class (batch composition changes force re-capture; graphs rebuilt off the hot path). Persistent-kernel variant as optimization where graph replay launch overhead shows up (target: keep non-GPU work <200 µs/step).

### 5.2 Prefill

Chunked prefill (chunk ≤2048 tokens default) fused into continuous batching alongside decodes. Comm pattern identical to decode but messages scale with chunk length → use ring ReduceScatter+AllGather split phases overlapped with the *next* chunk's compute (separate collective stream + event dependencies). DSA adds a global token-selection step after every 4th chunk boundary — selection must be computed on globally gathered indexer scores.

### 5.3 Speculative decoding (MTP)

Draft path uses the layer-45 MTP module with indexer sharing. Verify passes run k candidate tokens through the normal graph as a k-width micro-batch (activation traffic ×k on collectives — still tiny at bs=1). Adaptive k: start 3, tune acceptance window online (EMA of accepted depth), back off under load. Target ≥1.8× effective tok/s.

---

## 6. Distributed Strategy (RoCE / ibverbs)

### 6.1 Transport

Dataplane: hand-built ibverbs RC QP mesh (one active NIC per node — see §2). Message grading:
- **Latency lane** (<256 KB, all decode-step collectives): plain signaled sends into pre-posted receive rings; completion-driven advance. With measured 2.7 µs RTT, a one-shot 4-rank tree allreduce lands ≈8–12 µs end-to-end for ≤32 KB payloads.
- **Bandwidth lane** (prefill chunks): ring ReduceScatter → AllGather pipelined in 512 KB segments. Budget against the *measured* ~13.2 GB/s per-link ceiling, not the nominal 200 Gb/s.
- Control plane: plain TCP (discovery, membership, metrics heartbeat).

⚠ GPUDirect RDMA: **moot on this platform** (peermem present-but-inapplicable;
see §2 + measurements.md §GPUDirect investigation). **v0.2 design decision —
zero-copy receive:** CollectiveBus receives land directly in pinned slabs,
consumed zero-copy by GPU kernels. Measured on GB10 (micro_zerocopy):
pinned reads 255 GB/s vs cudaMalloc 239 GB/s (zero-copy is the *fastest*
path), staging copies only 59.6 GB/s (strictly dominated), fenced completion
round trip p50 ≈ 1.1 µs, writer interference ≈ additive per region
(−1% same-buffer, −8% other-buffer per flow). Ordering discipline is
production-pinned in `src/kernels/flag_protocol.cuh` with regression tests
(`tests/cuda/flag_protocol_test`, ctest `flag_protocol_test`).
Remaining M7-era unknown: mlx5 DMA payload → GPU-read visibility under the
doorbell scheme (expected fine on this coherent fabric; must be measured).
The `IBufSink` abstraction keeps a direct-registration backend swappable in
should the platform ever grow BAR-backed GPU memory.

M0 debugging note: an early hand-rolled probe exhibited missing SEND-completions when using `IBV_SEND_INLINE` and/or stack-resident SGE addresses paired with MR lkeys. Root cause investigation parked in favor of perftest-derived baselines; the CollectiveBus will use only MR-resident, non-inline signaling patterns until revisited (§11 observability will watch QP error counters).

### 6.2 Sharding scheme (TP)

| Tensor class | Shard rule |
|---|---|
| Attention Q/K/V/O | head-shard (÷N) column-parallel, O row-parallel |
| MLA q_a/kv_a/q_b/kv_b | replicate q_a,kv_a (small); shard q_b/kv_b/o by heads |
| Indexer (wq_b/wk/weights_proj) | **replicate** (tiny) → identical top-k on every rank, zero extra collectives |
| Router + biases | replicate, fp32, fixed traversal order → identical routing everywhere |
| Shared expert / dense MLP | column-shard across intermediate dim |
| Routed experts | round-robin partition 288→72/rank; leaves room to migrate toward EP if experiments favor it |
| Embedding rows | replicate; lm_head **vocab-shard** (rows ÷N), local argmax/topk partials merged via tiny collective |
| Norms / mHC mixers | replicated on inputs, computed redundantly (cheap) or folded adjacent to their consumer shard |

All ranks process the *same tokens each step*, so scheduler state is implicitly symmetric; eviction decisions in the cache are made by rank 0 and broadcast over the control plane to preserve shard consistency (each rank stores only its local piece of cached sequences).

### 6.3 Collective choices

Decode bs=1 needs **low latency**, not bandwidth: implement `allreduce_small` as one-shot tree: payload B×4096×fp16=8 KB → 3 destination writes + local sum; ~45 layers × 2 sync points × ~10 µs ≈ 0.9 ms/step vs ~26 ms step ⇒ ~4% overhead. Prefill RS+AG at 13.2 GB/s: a 2048-token chunk moves 2048×4096×2B=16 MB per rank per phase ⇒ ≈2.4 ms/layer-pair staged — overlap with next-chunk compute hides most of it (§5.2).

MoE dispatch at decode: since all ranks hold full routing knowledge, keep **expert-partitioned weights + push-tokens** (each rank sends each token's FFN input to ranks owning its selected experts)? Volume = B×8 destinations avg spread uniformly ≈ B×4096×2B×8 — 8× worse than hidden-scale. Rejected at bs≤8 in favor of **pull-model with local compute + final reduce**: each rank computes contributions only for *local* selected experts, then the existing FFN reduction folds expert outputs into the standard reduce point at no extra message count (token inputs for remote experts arrive via a lightweight "input broadcast" fused into the same value used locally). This is the classic TP-MoE formulation and suits our tiny messages; EP-style all-to-all becomes attractive at high batch — keep behind the same interface flag (§experiment E2).

### 6.4 Failure modes

- Peer death mid-flight: completion timeouts on QPs → mark peer failed, drain active requests gracefully (return partial streamed content + error frame), require operator restart (v1; auto-heal raft-join in later phase).
- CRC/link flap: RoCE counters polled; advisory metric surfaces to `/metrics`.
- Split brain impossible: config static peers; coordinator epoch numbers reject stale members.

---

## 7. Caching Subsystem

### 7.1 Blocks & trees

Fixed-size logical blocks of 128 tokens. Radix trie over rolling-hash(token_id[]) maps prefix chains → reference-counted block descriptors holding three physical components:

| component | stored on | footprint |
|---|---|---|
| DSA latent KV (512-dim bf16) | all ranks (replicated; cheap) | 1 KB/token |
| DSA indexer k-pool entries (+compress/APE state) | all ranks (replicated) | ~0.25 KB/token |
| linear-aux record (short-conv statelets, dt/A products needed for exact state reconstruction) | all ranks (identical) | ~24 KB/token |

Allocation from per-class arenas; device-resident pages registered for GPUDirect reads by attention kernels directly (zero-copy across subsystems).

### 7.2 Hybrid-model prefix reuse policy

Pure KV reuse is insufficient because linear-attention recurrence couples positions to state. Policy:
1. Match longest radix prefix P′ (up to nearest linear-state snapshot boundaries);
2. From the latest state snapshot ≤ P′, **replay** the linear layers over cached aux records to rebuild exact recurrent states at P′ (bandwidth-bound: aux ~24 KB/token ⇒ 32 K-prefix ≈ 0.77 GB spread over a handful of chunks ≈ tens of ms — acceptable);
3. Snapshots themselves (71 MB/seq… per-sequence states at multiples of 8192 tokens) stored opportunistically during generation in a small LRU side-pool sized by admin config (default: snapshots enabled for system-prompt-tagged sessions);
4. Indexer compress-state reconstructs along the same replay (kept exact-match semantics).

This yields *exact* numerics parity with cold prefill while saving compute, not just KV bytes. Admission control guards against cache-thrash under concurrent distinct prefixes (depth-weighted scoring like classic PLRU; evict tail-first with broadcast agreement).

Multi-replica distributed caching is out of scope v1 (single logical instance; future DP work noted in roadmap §12).

---

## 8. Serving Layer

- **HTTP**: hand-rolled epoll HTTP/1.1 with streaming SSE chunked responses; keep-alive pooling; OpenAI-compatible routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/healthz`, plus `/metrics` (Prometheus text) and `/admin/*` diagnostics.
- **Chat templating**: offline compile of the model's Jinja chat template to a C++ callable (codegen at build time from tokenizer_chat_template in the checkpoint) — avoids shipping a Jinja interpreter; correctness tests against transformers' rendered outputs.
- **Tokenizer**: GLM BPE in C++ (vocab 154880): loader for merges/vocab + byte-fallback matcher using a flat automaton; target <40 µs median for 1 KB prompt on X925 cores; roundtrip golden tests vs HF.
- **Tool calls**: `glm47` parser ported (streaming-safe incremental scanner, emit tool_call deltas like the reference server).
- **Sampling stack**: temperature/top_p/top_k/min_p/repetition-presence penalties/logit-bias, seeded PRNG reproducibility mode for tests.
- **Params surface**: include `prefix_cache`: bool (per-request opt-out), `max_prompt_len`, graceful truncation codes matching OpenAI error schema.

---

## 9. Dependency Posture (responds to your deferral)

You asked to understand specifics before committing. Candidate list, each flagged by involvement in hot path:

| Dependency | Size/nature | Hot path? | Justification | Alternative considered |
|---|---|---|---|---|
| CUTLASS (+headers, tests) | header-only, ~few MB headers | **Yes — GEMM templates** | tcgen05 (sm_121) FP8/BF16 tile codegen instead of months of hand PTX; profile-driven replacement sites behind our `IGemm` seam | cuBLASLt baseline (in CUDA toolkit, not OSS-vendored). **Decision: START with cuBLASLt primitives where shape-generic; adopt CUTLASS surgically only where measurement shows wins** (likely: MoE grouped GEMM, MLA decode GEMV/BM). Keeps risk low, perf ceiling preserved |
| fmt | tiny | No (logging) | de-facto standard, zero risk. GCC13 std::format coverage partial | std::print where adequate — drop fmt if it offends |
| spdlog (or none) | tiny | No | structured logs | custom 150-line logger acceptable |
| yyjson | very small C | Moderate-cold (request/response JSON only, not per-token) | fastest known C JSON; SSE deltas also JSON-shaped | hand-rolled parser — rejected: security-sensitive surface, no perf gain measurable. NOTE (v0.2): M1 shipped `minijson` for the *trusted, local* shardspec path only — it must never serve untrusted HTTP input; the yyjson choice for `serve/` stands |
| GoogleTest | test-only | No | — | doctest smaller if preferred |
| libibverbs/librdmacm | system packages (Ubuntu) | Yes | unavoidable — the chosen transport | NCCL fallback backend retained behind `ICollectiveBus` (NCCL itself installs no new package risk since bundles exist in toolkit ecosystem, but we don't depend on it in v1 unless probes show our stack underperforming) |

Rejected: httplib/nghttp2/OpenSSL (v1 serves plaintext; hand-rolled HTTP suffices and stays auditable), Eigen, Protobuf/gRPC, Python runtime dependency at service time (only build/dev tooling allowed, e.g., HF download + parity harness).

Net posture: **hot-path code depends only on CUDA Runtime/Driver APIs, vendored CUTLASS headers where adopted, and system ibverbs.** Every external piece lives under `third_party/` with pinned hashes; a "minimal-deps" build mode compiles without CUTLASS (cuBLASLt only).

---

## 10. Numerics & Parity Governance

- Golden-parity harness (`tools/dump-logits`): captures reference forward intermediates via transformers/vLLM on demand; engine compares at tolerance tiers (weights loading bit-exactness, activations rtol/atol per dtype zone: fp32 router exact-order, bf16 loose, fp8 quantized zones separately gated by per-tensor scales).
- Dedicated numeric tests for: delta-rule chunk scan (vs manual RNN expansion), indexer determinism (same input hash-set across ranks/sims), top-k tie-breaking rule aligned with reference implementation ordering, SwiGLU-limit clamp, mHC mixing algebra.
- Quantization dynamic activation scales follow the checkpoint's `fmt=e4m3` scheme; clamp/casting helpers centralized to avoid silent drift.

## 11. Observability

Per-request telemetry JSONL (queue time, prefill ms/toks, decode tok/s, accept rate, cache-hit token counts), `/metrics` Prometheus exposition (engine + network counters + RDMA port stats), ndcctx-style built-in tracing rings (`DGPP_TRACE=stepgraph`,chrome-trace export), plus a mode that drains rdma counter deltas per minute to detect regression early.

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPUDirect RDMA unsupported on GB10 driver | Resolved (v0.2) | — | question dissolved under unified memory: pinned zero-copy receive measured *faster* than device-buffer reads; flag-protocol ordering pinned by regression tests |
| KDA/delta-rule parity drift vs reference | High initially | Wrong outputs, wasted sprints | build against FLA reference math early; golden dumps; property tests on long recurrences |
| Shared-expert / unknown safetensors layout details differ from assumption | Med | re-plan waste | Phase-1 audit task before mass kernel authorship |
| RoCE lossless misconfig (packets dropped) | Med | instability | MTU9000 confirmed; measure PFC config in M0; escalate to your infra team w/ mlxdump evidence if needed |
| CUDA-graph shape churn hurts scheduling latency | Low-med | bursty perf | capture cache keyed on rounded shapes; async rebuild |
| Tokenizer/template divergence breaks client compat | Med | user-visible | golden tests vs HF transforms output strings |
| Cluster variability (one node thermally throttled) skews sync steps | Low | jitter | per-step skew watchdog; straggler-aware graphs left simple in v1 (uniform hardware) |

## 13. Explicit Non-Decisions / Future Tracks

- NVFP4 expert compression track (quality-gated) — big potential decode win on G1; slated post-v1 (need calibration + eval harness).
- Expert-parallel (EP) experiment under load — interface flag reserved now, decided by M7 data.
- Multi-replica DP routing layer, speculative-with-CPU-draft alternatives, vision encoder adapter (G6) — roadmapped.
