# DGPP Engine Design

Status: architecture contract for the staged implementation described in
`PLAN.md`. The repository currently contains the M0/M1 platform probes, core
runtime, checkpoint loader, synthetic transformer, the M2 KDA operators
(state pool, recurrence kernels, snapshot format, reference-dump harness),
and the complete M3 DSA/MLA path (indexer compression, deterministic pooled
top-k, split-KV absorbed attention, blocked state pool, layer orchestration,
dump harness, benchmarks); it does **not** yet contain a complete GLM
runtime, distributed serving daemon, or HTTP server.

Target: text serving for `unsloth/GLM-5.3-Flash-FP8` on four NVIDIA DGX Spark
systems. Vision execution is post-v1.

## 1. Goals and decision rules

1. Match the checkpoint and reference implementation before optimizing it.
2. Keep the batch-size-one decode path graph-capturable and allocation-free.
3. Use all measured network and memory paths; do not freeze a performance
   target from nominal specifications or average-rank traffic.
4. Make cache and speculative state transactional. A rejected or evicted
   token must not change committed model state.
5. Expose an OpenAI-compatible text API only after end-to-end parity passes.

Performance targets are set from the completed runtime and a reproducible
workload, not from a third-party implementation. Router traces and a
full-model step profile are mandatory inputs at M4–M6; comparisons with other
engines are optional. The audited pre-speculation weight-bandwidth floor is
about 30.8 token/s on the expected busiest expert rank, not the previous
39 token/s average-rank estimate.

## 2. Validated platform facts

### 2.1 Compute and memory

Each DGX Spark has one GB10 Grace Blackwell SoC and 128 GB of coherent LPDDR5x
shared by CPU and GPU. Nominal bandwidth is 273 GB/s; the weight-stream proxy
measures about 230 GB/s. `cudaMalloc` and `cudaHostAlloc` consume the same
physical pool, although their mappings and access behavior differ.

The engine uses explicit residency classes and prefetching. Transparent page
migration is not part of the hot-path design.

### 2.2 ConnectX-7 topology and budget

DGX Spark contains one ConnectX-7 NIC with two physical QSFP ports and two
independent PCIe Gen5 x4 connections to the SoC. One cabled QSFP port appears
as two usable Linux/RoCE lanes:

| lane | PCIe function | Ethernet | RoCE | lab subnet |
|---|---|---|---|---|
| 0 | `0000:01:00.0` | `enp1s0f0np0` | `rocep1s0f0` | fabric 0 |
| 1 | `0002:01:00.0` | `enP2p1s0f0np0` | `roceP2p1s0f0` | fabric 1 |

The `f1` functions belong to the second physical QSFP port and are down in the
current cluster. The two active `f0` lanes are not separate failure domains;
they share one physical 200 Gb/s QSFP link, but they remove the per-PCIe-x4
bottleneck when used together.

Measured node-to-node RC results on 2026-08-27:

- lane 0 alone: 107.64 Gb/s;
- lane 1 alone: 107.64 Gb/s;
- concurrent lanes: 97.98 + 98.05 = **196.03 Gb/s**;
- 2-byte send latency: 2.72 µs typical on lane 0 and 2.46 µs typical on lane 1;
- 9000-byte IP frames pass on both lanes.

CollectiveBus therefore stripes bulk traffic across both `f0` lanes and uses
approximately 24.5 GB/s as the measured aggregate planning ceiling. A
single-lane fallback remains supported.

Host DCB inspection shows priority-flow-control disabled on all priorities and
global Ethernet pause enabled in both directions. Switch PFC/ECN policy is not
visible from the nodes, but the paired tests reached the physical-link ceiling
without a congestion symptom. Switch inspection and counter deltas are
diagnostics if a future four-node run shows drops, retries, throughput collapse,
or latency spikes; they are not a correctness or deployment prerequisite.

### 2.3 Direct NIC-to-GPU consumption

Conventional GPUDirect RDMA registration of `cudaMalloc` memory is unsupported
on DGX Spark. That does not require a bounce copy: the NIC can DMA into an
ibverbs-registered `cudaHostAlloc` slab, which the GPU reads directly from the
coherent memory pool.

The project test sends a 64-byte payload followed by a 64-byte sequence
doorbell as ordered RC SEND work requests. A concurrently running GPU kernel
polls the doorbell with a system-scope acquire and hashes the payload before
publishing a system-scope release acknowledgement. Both active lanes passed
10,000/10,000 changing payloads. This driver/firmware-specific hardware
contract is rerun during deployment and after upgrades.

Two concurrent 98.04 Gb/s incoming RDMA streams reduced an unrelated pinned
GPU read from 251.5 to 218.4 GB/s (13.2%). Capacity estimates that assume
sustained dual-lane ingress provisionally derate simultaneous GPU memory
bandwidth by 15%. This is a stress-case planning value, not a fixed inference
penalty; M5 replaces it with measurements of the actual collective schedule.

## 3. Checkpoint and decode traffic

The checked revision contains 62 safetensors shards, 76,108 tensors, and
328.33 GB (305.78 GiB) of weights. `tools/checkpoint_audit.py` reads the
checkpoint configuration and headers and emits the authoritative breakdown in
`docs/checkpoint_budget.md`.

Main text configuration:

- 45 layers: 34 KDA linear-attention and 11 DSA/MLA layers;
- hidden size 4096; 64 heads; KDA head dimension 128;
- three dense MLP layers, then 42 MoE layers;
- 288 routed experts, top-8 per token, plus one shared expert;
- one MTP layer;
- DSA `index_kpool=4`, `index_topk=2048`, and
  `index_kpool_always_select_tail=true`;
- RoPE dimension is zero for the MLA query/key path in this checkpoint.

The unique base-text weight set touched at batch size one is 22.415 GB/token.
Vision, input-embedding storage, and the inactive MTP layer are excluded.
Because the DSA indexer, router, mHC, and norms are replicated, their duplicate
reads make aggregate physical traffic 23.420 GB/token across TP=4. With whole
experts partitioned 72 per rank, uniform top-8 routing has an exact expected
busiest-rank occupancy of 3.515 experts/layer. The resulting per-rank views
are:

| TP=4 view | traffic/rank/token | 230 GB/s floor |
|---|---:|---:|
| mean rank | 5.855 GB | 25.46 ms |
| expected synchronized critical path | **7.457 GB** | **32.42 ms / 30.8 token/s** |
| all selected experts on one rank | 12.198 GB | 53.04 ms |

The synchronized figure sums the busiest rank separately at every MoE layer;
it does not assume one physical rank stays busiest for the whole token. The
model is still optimistic when routes are correlated. M4 must capture
layer-by-layer router choices from representative prompts and replace the
uniform distribution before performance sign-off. If imbalance is material,
the remedies are measured expert placement, selective replication, or
intra-expert TP—not wishful division by four.

## 4. Quantized-weight contract

All 37,338 E4M3 matrices have an F32 `weight_scale_inv` tensor whose shape
exactly matches 128×128 blocks. The resident representation is the compressed
E4M3 payload plus its block scales.

“Load” means map, validate, shard, and place that compressed representation.
It never means materializing a persistent BF16 copy. Kernels either consume the
block-scaled representation natively or apply the inverse scales within a
fused conversion/GEMM path. M4 parity tests cover block edges, saturation,
NaN/Inf policy, and real checkpoint slices. An optimization that changes the
resident byte count must update the generated checkpoint budget.

## 5. Process, memory, and execution layout

There is one process and one CUDA context per node. Rank 0 additionally owns
request admission, scheduling decisions, and cache-eviction epochs. Every
rank executes the same request order and treats a coordinator epoch mismatch
as fatal to the current batch.

### 5.1 Block-boundary invariant

Hidden and residual tensors are **replicated at every transformer-block
boundary**. This removes the previous reduce-scatter/all-gather ambiguity and
makes replicated norms, mHC, routers, and indexers well-defined.

| operation | input | local work/output | collective |
|---|---|---|---|
| norm + mHC | replicated hidden | replicated | none |
| q/k/v or KDA projections | replicated hidden | column/head shard | none |
| attention output projection | head shard | partial hidden | all-reduce sum |
| MoE router | replicated hidden | identical top-8 IDs/weights | checksum in debug only |
| routed experts | local whole experts | partial hidden sum | FFN all-reduce sum |
| shared/dense MLP | column/row TP | partial hidden | same FFN all-reduce |
| block exit | all-reduced hidden | replicated hidden/residual | none |
| lm head | replicated hidden | vocab-sharded logits | sampling-dependent merge |

Thus a normal layer has one hidden-vector all-reduce after attention and one
after FFN/MoE. Prefill may use reduce-scatter/all-gather internally for large
chunks only if it restores the same replicated boundary and passes parity.

KDA heads and recurrent state are head-sharded. DSA MLA latent and indexer
caches are replicated because every rank begins with the same hidden state and
must make the same sparse-token selection; there is no global score gather.

### 5.2 Weight placement

- routed experts: 72 whole experts per rank initially;
- attention, shared-expert, and dense matrices: standard column/row TP;
- router, mHC, norms, and DSA indexer: replicated;
- embeddings and lm head: vocabulary-sharded;
- MTP: the same rules as its corresponding main-layer modules.

The shard generator records the owning ranks and scale geometry. Boot checks
hash all replicated tensors and reconcile per-rank byte totals before graph
capture.

## 6. CollectiveBus protocol

Each peer pair owns RC QPs on both active lanes, registered send/receive slabs,
credit state, CQs, and a nonblocking CUDA stream. Bulk chunks are split across
the lanes; small messages may use the lower-latency lane while retaining the
other as failover.

Receive protocol for slot `S`:

1. the receiver posts payload and doorbell receive buffers and grants a credit;
2. the sender posts the payload SEND followed by the doorbell SEND on the same
   RC QP; only the doorbell needs a send completion;
3. the GPU observes the doorbell with a system-scope acquire, consumes the
   payload, and publishes its acknowledgement with a system-scope release;
4. the CPU polls the CQ for transport errors and recycles the slot only after
   both the GPU acknowledgement and receive completions are known.

Slots are 64-byte aligned, initialized before kernel launch, and never reused
without a new credit. Persistent kernels use an **inactivity** watchdog whose
deadline resets after every message; a reserved stop sequence provides orderly
shutdown. The cross-node hash test and CUDA regression test pin this behavior.

The initial implementation uses all-reduce, not an incorrectly named
three-destination “tree.” Algorithm selection is size-based:

- small replicated hidden vectors: latency-oriented recursive doubling/tree;
- large prefill chunks: striped reduce-scatter + all-gather or ring all-reduce;
- control messages: TCP, never on the CUDA critical path.

## 7. Attention and state semantics

### 7.1 KDA

The reference state contract is:

- recurrent state: FP32, shape per rank/layer `[16, 128, 128]`;
- merged q/k/v convolution state: activation dtype (BF16 here), width
  `conv_kernel-1`, plus speculative width when MTP is active.

The M2 implementation pins the concrete layouts (tested in
`tests/unit/kda_geometry_test.cpp` and the CUDA suite):

- recurrent state is `[local_heads, head_v_dim, head_k_dim]` with the
  K dimension contiguous — the indexing of the reference recurrent kernel;
- conv state is `[3*local_proj, conv_state_width]` BF16 with the width
  contiguous ("dim-first"), committed history in columns
  `[0, conv_kernel-1)` and the speculative reserve after it, zeroed and
  untouched until M8;
- the fused in-projection row is `[f_a | g_a | q | k | v | b]` — the two
  replicated shards lead so the strided f_b/g_b GEMM inputs land on
  16-byte-aligned column offsets (cuBLASLt returns wrong results, not
  errors, for misaligned strided activations);
- the merged q|k|v conv weight is `[q(all heads) | k | v]` along channels,
  so a TP rank's slice is assembled per-section, not as a contiguous
  prefix.

Across 34 KDA layers this is 34.0 MiB of recurrent state per rank/request.
The committed three-position convolution history is 1.20 MiB. A three-token
draft reserve adds another 1.20 MiB, producing a fixed 36.39 MiB mutable slot
per rank/request. V1 snapshots copy that whole fixed-width slot (the
speculative suffix is zeroed/ignored), so the same conservative 36.39 MiB
figure drives admission and prefix-cache capacity—not a BF16 recurrent-state
estimate.

Prefill uses a pool-aligned chunk size (initially 2048 tokens) and carries the
exact final recurrent and convolution states between chunks. Decode mutates a
request's committed state only through the transaction described in §9. One
shared implementation of the recurrence serves both: chunk boundaries
round-trip the FP32 state through memory exactly, so chunked and unchunked
prefill agree bitwise on the recurrence path; residual differences in the
full layer come only from bf16 GEMM outputs that differ by ulps across
chunk-size-dependent cuBLASLt algorithms.

### 7.2 DSA/MLA and `index_kpool`

`index_kpool=4` means one compressed index entry for each four input tokens,
not one selection every fourth prefill chunk. Each query scores pooled entries;
the implementation selects pools, expands them back to token positions, and
includes the valid incomplete tail as required by
`index_kpool_always_select_tail=true`.

Requirements:

- cache block sizes and internal chunk starts are multiples of four; a final
  chunk may end with one to three tokens in the persistent incomplete tail;
- the in-progress tail persists across prefill chunks, decode steps, MTP
  verification, prefix attachment, and cache transfer;
- causal masking and deterministic tie-breaking operate on original token
  positions after pool expansion;
- top-k parity is tested against the reference implementation, including
  lengths 0–7 around pool boundaries and a partially rejected MTP step.

At 300,000 cached tokens, the replicated per-rank DSA budget is approximately:

| component | calculation | bytes/rank |
|---|---|---:|
| MLA latent | `300k × 11 × 512 × 2` | 3.379 GB |
| pooled index | `300k × 11 × (128 FP8 + 4 scale bytes) / 4` | 0.109 GB |
| incomplete tails | `11 × 4 × (128 K + 128 gate) × 2` | 22.5 KB/request |

The base cache total is therefore about 3.49 GB/rank plus block tables,
allocator metadata, and alignment—not 4.2 GB cluster-wide.

The M3 implementation pins the concrete layouts and the selection spec
(tested in `tests/unit/dsa_geometry_test.cpp` and the CUDA suite):

- the index K cache is **planar** — FP8-E4M3 rows `[slots, 128]` plus FP32
  scales `[slots]`, one slot per pool. The reference's packed 132-byte pages
  exist for a DeepGEMM `block_kv` constraint this engine does not inherit,
  and packed rows would misalign every other 128-byte streaming read;
- the latent cache is BF16 `[token_slots, 512]` rows; one shared block table
  per request serves both caches (block = 128 tokens = 32 pools, so token
  block *b* and pool block *b* are the same entry — prefix attachment shares
  both by reference, §8);
- the tail is a per-request ring `[2, kpool, 128]` BF16 — raw K at half 0,
  gate at half 1, ring slot `pos % kpool`. Completion reads the ring with the
  current token overriding its own slot (which still holds one stale pool's
  stash — the reference's `is_current` rule); the stash happens *after* the
  completion read;
- visible pools for a query at position *p* are `floor((p+1)/kpool)`; the
  incomplete tail is never scored, only appended. Sequences with
  `visible <= select_k` degenerate to dense causal selection through the same
  code path;
- selection is pinned as: the `select_k = topk/kpool` pools with the highest
  fp32 logits, exact ties to the **lower pool index**, output ascending in
  pool index. The composite sort key `(~sortable_fp32 << 21) | pool_idx` is a
  total order, which makes the selection deterministic on any correct
  implementation. The reference's radix path is `atomicAdd`-ordered at exact
  ties; this pin matches its deterministic path. Finite logits are assumed —
  real quantized cache rows are always finite (the saturating encoder never
  mints NaN);
- the **decode select is fused**: one kernel streams the blocked index cache,
  computes pool logits inline (warp per pool, lane per head, contraction-proof
  `__fmul_rn`/`__fadd_rn` arithmetic so the host oracle's logit parity — and
  therefore position parity — is bitwise), keeps a running top-`select_k` in
  shared memory, and merges block partials with a last-block reduction whose
  counter self-resets. Fixed grid, grid-striped over device-visible counts,
  zero logits materialized: the whole decode path is CUDA-graph capturable.
  Prefill instead materializes per-(row, head) fp8 dots through the IGemm
  seam (FP8×FP8→F32, unit scales, K-cache reads amortized across query
  tiles) and runs the same streaming selection over the dot buffer;
- MLA runs absorbed: `q̃ = W_uk^T q` (bf16 GEMM rounding), scores
  `q̃ · latent` in fp32 with split-KV online softmax — the running max lives
  in per-lane registers fed by butterfly group reductions (no shared running
  state to order), probs round to bf16 for the `c` accumulation while the
  denominator stays unrounded, and the output is v-absorbed
  (`out_h = W_uv_h · c_h`). `kv_b` is consumed in the checkpoint's
  interleaved per-head layout;
- the power-of-two fp8 scale is computed by exact bit manipulation (smallest
  `2^n >= absmax/448`, exact powers mapping to themselves) rather than
  `exp2f(ceilf(log2f(v)))`: the fp32 libm form can round across a
  power-of-two boundary on near-tie inputs and clamp the row, and glibc and
  libdevice also disagree by ulps there. The exact form matches the reference
  except on pathological near-power-of-two magnitudes, where the reference
  clamps and this engine does not.

Layer-phase pins (the orchestration above those kernels):

- `DsaStatePool` owns all 11 layers' caches as five arenas — latent
  `[layers][token_slots][kv_lora]` BF16, planar index K `[layers][pool_slots]
  [128]` FP8, index scales `[layers][pool_slots]` F32, tails
  `[layers][requests][2,kpool,128]` BF16, and one shared block table
  `[requests][blocks]` INT32 — every region 256-byte aligned. Block
  management is host-side (LIFO free list; growth uploads only the new table
  slice on the caller's stream; growth is transactional — admission control
  per DESIGN §9 must guarantee capacity before the decode hot path, which
  cannot check device-side positions). Released blocks are not scrubbed: a
  new owner rewrites every row it reads before any consumer touches it;
- `DsaLayer` shares one scratch buffer across all DSA layers (they run
  sequentially on one stream; per-layer scratch would multiply the footprint
  by 11 for zero benefit), sized by `scratch_bytes()` with a dot budget that
  bounds prefill's materialized fp8 dots — query tiles shrink to fit,
  trading K-cache re-reads for a bounded footprint. Prefill grows block
  tables itself and throws on pool exhaustion (a control-path operation);
  decode requires the tables to already cover every position a captured
  batch will write. The decode path is CUDA-graph capturable end to end;
- the near-tie contract for cross-implementation selection comparison:
  logits from tensor-core GEMMs differ from any sequential oracle by ulps
  that can flip the rank-select_k cut when two pools sit within that noise.
  Divergences are certified, not tolerated: the audit re-derives the spec
  selection from the device's own inputs (bitwise, including the actual
  dot buffer) and requires the swapped pools to straddle the boundary
  within the measured per-row noise. Device-vs-device comparisons (same
  GEMM outputs) remain bitwise — that is the reproducibility contract
  serving relies on;
- decode performance pins (GB10, 48 SMs): native hardware e4m3→f16 and
  bf16→f32 conversions (both exact — bit-identical to the software codecs,
  proven by the bitwise select fuzz), split-KV attention at
  rows×32×head-groups blocks with an empty-split early exit, padded smem
  strides in the attention kernel (the natural [head][512]/[group][64]
  layout bank-conflicts 32 ways on every access), and 32-bit split key
  arrays in the bitonic networks (a u64 key spans two banks and XOR
  indexing conflicts ~16 ways). Decode at 65k context: 2.24 ms/layer, 114
  GB/s effective; the projection GEMMs alone are ~1.0 ms — 238 MiB of
  weights at the memory floor (see
  benchmarks/results/2026-08-28-dsa-m3-layer.md).

## 8. Prefix cache

V1 uses exact state snapshots only. The previous unproven 24 KB/token
“linear-aux replay record” is removed.

A reusable prefix entry contains:

- immutable token blocks and tokenizer/chat-template revision hashes;
- attached DSA MLA/index blocks and the exact incomplete tail;
- one final KDA recurrent+convolution snapshot for that prefix boundary;
- model/checkpoint revision and numerical-mode identifiers.

Only a radix node with a complete snapshot is attachable. A match without one
is treated as cold and is re-prefilled; the engine never reconstructs state
from an underspecified record. Attaching copies the 36.39 MiB/rank KDA snapshot
into request-owned mutable state and shares immutable DSA blocks by reference.

The initial snapshot arena is capped at 1.5 GiB/rank, permitting 42 complete
prefix snapshots before metadata. Admission and eviction are agreed by epoch
on all ranks. Eviction cannot free blocks until all request and snapshot
references reach zero.

## 9. MTP transaction model

Draft depth starts at three but is adaptive. Verification never writes over
committed state in place.

1. At transaction start, each request records its committed token length,
   block-table length, KDA state slot, convolution tail, and index-pool tail.
2. KDA verification uses `k+1` state indices per speculative request and a
   convolution window enlarged by `k`; kernels advance candidate states using
   the accepted-token count.
3. MLA KV entries are written to reserved, unpublished positions. Index-pool
   completions and the incomplete tail are built in scratch from the saved
   pre-transaction tail.
4. Rank 0 computes the accepted count from the vocab-sharded verifier result,
   broadcasts it with the RNG counter, and every rank verifies the epoch.
5. If all drafts are accepted, the last candidate state and all reserved cache
   entries become committed. On a rejection at position `a`, only the first
   `a` draft states are committed; the rejected suffix is discarded, and the
   replacement token is processed from the state after those `a` tokens.
6. Cancellation or rank failure discards the entire transaction.

Tests reject at every depth from zero through `k`, including a rejection that
crosses an index-pool boundary. Greedy output and all subsequent states must
match non-speculative execution.

## 10. Tokenization, templates, logits, and sampling

The bundled tokenizer is BPE with `byte_fallback=false` and no normalizer. Its
pre-tokenizer is the checkpoint's explicit regex `Split`, followed by
ByteLevel with `add_prefix_space=false`, `trim_offsets=true`, and
`use_regex=false`; its ByteLevel decoder sets `add_prefix_space=true`,
`trim_offsets=true`, and `use_regex=true`. The engine must reproduce these
settings and the special-token policy exactly; it must not invent a
byte-fallback or normalization algorithm.

`chat_template.jinja` is loaded or compiled when a model revision is installed,
not at engine build time. The compiled artifact is keyed by the template and
tokenizer hashes. Golden tests cover English, Chinese, code, reasoning blocks,
and tool calls against reference-rendered strings.

The lm head is vocabulary-sharded:

- greedy: reduce the local `(value, token_id)` maxima;
- finite `top_k`: merge each rank's exact local top-k;
- unrestricted `top_p`, `min_p`, logprobs, or penalties requiring the full
  distribution: gather the FP32 vocab slices to rank 0, apply the exact
  reference sampler, and broadcast the chosen token and RNG counter.

The full-logit fallback moves about 619.5 KB/token for this vocabulary; it is
not described as a tiny merge. More elaborate distributed selection is an
optimization only after parity and profiling.

## 11. Runtime and API outline

The eventual daemon is a hand-rolled C++ service with:

- `/v1/chat/completions`, `/v1/completions`, `/v1/models`;
- `/healthz` and `/metrics`;
- SSE streaming, cancellation, deterministic seeds, and bounded request queues.

The scheduler separates prefill and decode work, preserves identical rank
order, and admits requests only when weights, mutable state, DSA cache, MTP
scratch, network slabs, and snapshot copies fit. CUDA graphs are keyed by
batch/shape bucket and contain no allocation or host synchronization.

## 12. Correctness and performance gates

Every model milestone has three tiers:

1. geometry/state tests: shapes, dtypes, ownership, pool boundaries, and
   transactional rollback;
2. numerical tests: layer dumps, logits, top-k indices, and end-to-end greedy
   transcripts against a pinned reference revision;
3. performance tests: kernel time, effective bytes/s, collective time, and
   critical-rank routing measured separately.

Numerical tests must include at least one **hand-computed expectation** for
every spec-critical output. Differential (implementation-vs-implementation)
tests verify consistency, not direction: an inverted selection ordering
passed every bitwise parity test in both directions because the host oracle
and the kernels were wrong together, and only a test with hand-computed
expected logits exposed it.

Device-vs-oracle divergences at selection boundaries are **certified, not
tolerated**: the audit re-derives the spec selection from the device's own
inputs (bitwise, tensor-core dots included) and measures the boundary gap
against the row's actual cross-implementation noise. This discipline caught
two reference bugs that plain tolerance would have absorbed into "FP noise"
— an out-of-bounds tail-seed read for decode batches shorter than kpool and
a gate-indexing typo — while the device path was correct both times. The
reference-dump parity runner carries the same audit for its corpora: a
flipped row is certified from the device's probes AND the dump's own
recorded tensors before it can pass. That guard earned its keep on the
first real-checkpoint run: the dump tool's torch backend paired decoded
q_fp8 with raw uint8 index_k bytes, and because e4m3 is monotone in the raw
byte within each sign class, the corrupted logits preserved near-correct
rankings — the failure was a single boundary swap that the structural
budget happily absorbed while the engine was right and the reference was
wrong. A near-miss bug that survives a tolerance is still a bug; only a
certification path can reject it.
Instrumentation (in-kernel clock64 phase timers, removed after use) is the
fastest path to a *mechanism* for a slow kernel; profile first, then fix the
measured bottleneck (the attention kernel's 85% bank-conflict share was
invisible from wall time alone).

`compute-sanitizer` (memcheck, racecheck, initcheck) is part of the gate, not
an optional extra — two classes of defect pass functional tests and only
surface under it:

- nvcc **speculates loads past short-circuit guards**: `i < n && a[i]`
  reads `a[i]` regardless of `i < n` once the branch is predicated, and even
  a ternary index gets both arms loaded. Per-token arrays in test drivers
  must be sized to the token count, and bounds-dependent loads must clamp
  their indices arithmetically;
- shared-memory buffers reused across reduction phases (all threads read the
  mean, lane-0s overwrite with the variance) race without a barrier between
  the read and the reuse — scheduling luck passes functional tests.

Sanitizer scope is chosen per phase, deliberately: full-suite memcheck for
orchestration code (pointer/size bugs — it caught two undersized test
buffers that made a graph test pass vacuously), targeted racecheck/initcheck
for new kernel shapes. Instrumenting vendor kernels (cuBLASLt's cutlass
implementations dominate long benchmarks) buys no coverage of our code and
costs orders of magnitude in wall time; the decision and reasoning are
recorded with the results.

Measured numbers include the exact command, binary revision, driver, firmware,
clock/power context, run count, and variability. `micro_gemm_peak` is a
cuBLASLt heuristic sweep, not a proof of hardware peak. Network claims use
perftest as the authoritative throughput source and the project benchmark for
protocol validation.

## 13. Repository layout

Current source-tree implementation (including the audit-remediation files that
must be added to the next commit):

```text
apps/                 dgppctl, glm_bind_check, and synthetic gpt_doll driver
benchmarks/micro/     platform and transport probes (incl. kda_bench, dsa_bench)
docs/                 generated checkpoint budget and validated measurements
src/common/           logging, dtypes, tests
src/core/             arena, graph, streams, trace
src/kernels/          synthetic kernels, GEMM wrapper, flag protocol, KDA and DSA ops
src/loaders/          JSON, safetensors, shardspec
src/models/           synthetic GPT doll; KDA layer/state/reference/dump; DSA
                      reference/geometry/state/layer/dump; GLM text config,
                      expected-tensor binding, and streaming resident loader
                      (M4)
tests/                host, CUDA, and Python tests
tools/                checkpoint audit, shard-plan, KDA/DSA reference-dump generators
```

Planned directories such as `models/glm53`, `net`, `server`, `cache`, and
`deploy` are added only with their milestones; they are not represented as
existing code.

## 14. Future validation scope

The recorded evidence satisfies the completed M0/M1 exit criteria. It does not
replace the workload-specific correctness, stability, and performance
measurements required by later milestones, including four-node CollectiveBus
and full-model validation. If those runs develop drops, retries, unstable
throughput, or latency spikes, capture node and switch counter deltas and
inspect flow-control configuration. Comparisons with other inference engines
are optional and are not an implementation milestone or performance gate.

## 15. Primary references

- [NVIDIA DGX Spark platform specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA DGX Spark ConnectX-7 topology and interface mapping](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [NVIDIA DGX Spark CUDA porting notes, including GPUDirect RDMA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
- [GLM-5 vLLM KDA state dtype and speculative state handling](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/models/glm5next/nvidia/kda.py)
- [vLLM KDA state-shape and dtype calculator](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/model_executor/layers/mamba/mamba_utils.py)
- [GLM-5 pooled-index and incomplete-tail cache](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/models/glm5next/nvidia/attention.py)
- [vLLM pooled-index insertion, expansion, and quantized-cache semantics](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)
- [CUDA C++ memory model and thread scopes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-memory-model)
