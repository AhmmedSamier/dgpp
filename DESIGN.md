# DGPP Engine Design

Status: architecture contract for the staged implementation described in
`PLAN.md`, kept in step with the code. As of 2026-09-03 the repository
contains the full serving stack for the text path at TP=4: the platform
probes and CollectiveBus (M0/M5), the KDA and DSA/MLA operators with their
state pools (M2/M3), the assembled 45-layer GLM forward with resident
sharded load and the per-rank image cache (M4/M5), the incremental decode
engine with request sessions, the exact tokenizer and chat-template
interpreter, the deterministic scheduler and admission journal, the
OpenAI-compatible HTTP/SSE service (`dgpp-serve`, M6), the recorded decode
step with the collectives as graph nodes, greedy MTP speculative decode as
one graph replay per step (M8), and the adaptive scalar/row-batched T=1/MTP
graph adapter behind the service (up to 8×T=1 or 4×T=2, loopback-gated and
four-node performance-gated 2026-09-03). Scalar execution below the measured
four-request crossover removes the fixed graph's low-occupancy regression;
scalar-order GEMV chunks keep transcripts invariant when a live request
crosses graph widths. It does **not** yet contain the prefix cache (§8 is a design)
or stochastic sampling on the distributed path (§10). Sections marked "as
built" describe the code; sections marked "design" describe what remains.

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
engines are optional. The weight floor of a single-stream decode step is
~24.5 ms (5.855 GB/rank/token at the ~240 GB/s the L2-prefetched GEMVs
reach — §3, §7.6); the plain step measures 31.3 ms/token and the MTP step
22.45 ms/token effective (2026-09-03). The earlier 30.8 token/s "busiest
rank" floor belonged to the whole-expert placement retired on 2026-09-02.

One numerics rule, decided 2026-09-02 with the user: a kernel may change
its floating-point reduction ORDER (reassociate an fp32 sum, split a chain
across lanes) when that buys latency, provided the change stays at rounding
level. Cross-build regression is therefore judged by margin and by
perplexity (`scripts/fabric_xcript.py`, `scripts/fabric_logprob.py`; §12),
not by transcript identity — except for MTP against plain decode, which
must stay IDENTICAL by construction (§9).

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
approximately 24.5 GB/s as the measured aggregate planning ceiling. The bus
defines no per-lane failover (§6.1): the lanes share one physical port, so a
single-lane configuration is a deployment choice at about half bandwidth, not
a recovery mechanism.

Host DCB inspection shows priority-flow-control disabled on all priorities and
global Ethernet pause enabled in both directions. Switch PFC/ECN policy is not
visible from the nodes, but the paired tests reached the physical-link ceiling
without a congestion symptom. Switch inspection and counter deltas are
diagnostics if a future four-node run shows drops, retries, throughput collapse,
or latency spikes; they are not a correctness or deployment prerequisite.

RoCEv2 GID selection: each port's GID table sorts a link-local `fe80::` v2
entry ahead of the routable IPv4-mapped one, and switch forwarding of
link-local is pair-dependent — the M5 four-node mesh measured one-way
silent drops on specific directed pairs while the same QP pair ran cleanly
in reverse. The bus therefore prefers the first non-link-local v2 GID
(lane 0 → `c0a8:58xx`, lane 1 → `c0a8:59xx`), and the M0 tools' numbers
were taken on link-local GIDs — valid for those pairs, but the routable
GIDs are the deployment selection.

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

The rerun is `nic_regress` (M5): one command drives the project test over
every directed node pair on both lanes — the mesh mode derives a
deterministic schedule from `--nodes`, every node processes its obligations
in order (senders connect with bounded retries, receivers hold
identity-checked connections that arrive early), and nodes[0] merges both
endpoints' views into one matrix and one exit code broadcast to all ranks.
Probes use the deployment GID selection (the first routable RoCEv2 GID, as
`VerbsDevice` picks) — not the M0 bench default, which measured the
link-local fe80:: GIDs whose directed-pair drops motivated the routable
selection in the first place. The regression must exercise the path the bus
uses. `nic_regress pair`/`serve` probe a single directed pair (targeted
debugging); `nic_regress selftest` runs the full mesh machinery as a
two-thread world on one device in CI.

Two concurrent 98.04 Gb/s incoming RDMA streams reduced an unrelated pinned
GPU read from 251.5 to 218.4 GB/s (13.2%). Capacity estimates that assume
sustained dual-lane ingress provisionally derate simultaneous GPU memory
bandwidth by 15%. This is a stress-case planning value, not a fixed inference
penalty; M5 replaces it with measurements of the actual collective schedule.

GPUDirect RDMA is closed as unavailable on this platform (re-tested
2026-09-02): `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns
`CUDA_ERROR_INVALID_VALUE` for `cudaMalloc` memory and `ibv_reg_mr` on a
device pointer fails with EFAULT (no peermem) on all four RoCE functions.
The bus's host-pinned staging, read by the GPU with system-scope loads
(§6.3), is the design the GB10 allows, not a fallback.

The memory controller is shared by the GPU, the CPU cores and the NIC, and it
is a single queue: 3 MB of GPU prefetch in flight added ~12 µs to every
access on the box, including the bus handshake and the engine's posts
(§7.6). Anything that adds memory traffic on a serving node is measured
against the step, not in isolation.

## 3. Checkpoint and decode traffic

The checked revision contains 62 safetensors shards, 76,108 tensors, and
328.33 GB (305.78 GiB) of weights. `tools/checkpoint_audit.py` reads the
checkpoint configuration and headers and emits the authoritative breakdown in
`docs/checkpoint_budget.md`.

Checkpoint location (deployment decision, M5 exit gates): every node carries
the full weights in the canonical HuggingFace hub cache inside the user's
home directory. The application resolves models from there by id —
`--model unsloth/GLM-5.3-Flash-FP8` on any app — via `loaders/hf_cache`:
the cache root follows huggingface_hub's own precedence (`$HF_HUB_CACHE`,
then `$HF_HOME/hub`, then `~/.cache/huggingface/hub`), `refs/main` pins the
snapshot, and resolution refuses ambiguities (multiple snapshots with no
ref, a ref whose snapshot is missing) rather than guessing. Snapshots are
symlink farms into `blobs/`, which mmap and fopen follow transparently —
the loaders need no cache awareness of their own. `--checkpoint-dir`
remains for fixtures and staged directories. This retired the
per-rank-extraction question for the fabric: every rank reads its shard
directly from its node's cache (the d4 sharded loader touches only its
own bytes), so no staging copy exists at all.

Production residency contract (deployment decision, M5 exit gates): in
production, serving loads the rank's weights ONCE at startup and keeps
them fully resident for the process's lifetime — storage is never
touched during inference. Only TP=4 can honor this contract on a 128 GB
GB10: ~82 GiB of weights per rank (81.77 GiB main stack; the MTP draft
layer adds ~7.3 GiB when enabled) leaves headroom for the CUDA context
(~14 GiB locked), the DSA pool, and the caches; TP=2 (~155 GiB/rank) and
TP=1 (~306 GiB) cannot fit, which is the memory rationale for the
four-rank deployment target. The streaming forward (§7.5: one layer
resident at a time, the whole checkpoint re-read per pass) remains the
world-1 DIAGNOSTIC instrument — parity harnesses and dumps run on it; a
w1 decode step takes minutes.

Resident mode as built (M5, then M6 rounds 9–10): resident mode is the
streaming build path with each layer's bump adopted into a per-layer
exact-formula allocation — resident bytes are streaming bytes by
construction, pinned bitwise at every world. The loader IS the slicer at
TP>1 (§5.2). Three load-time facts became rules:

- *One-pass sources.* Every source tensor read is bracketed
  `MADV_WILLNEED` → copy → `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`, so
  the page cache stays under ~10 GB during a load and the box never
  reaches its memory watermark; the moment the last layer materializes
  the model releases every shard mapping (`GlmLayerStream::
  release_sources`). Before this the boxes sat at the watermark for the
  whole run and the kernel swapped the process's own cold pages — the
  tokenizer's vocab among them — producing 7–10 ms lockstep decode stalls
  on every rank at the same step (a major fault in `tok.decode`).
- *Eager construction.* `preconstruct_layers` walks every layer in
  resident mode, so the constructor returns a READY model; prefill is a
  prefill again, not a lazy load parked on the slowest disk.
- *The resident image.* A layer's resident bytes are the END of the
  pipeline (slice, stage, H2D, dequant, pack) and reproducible without a
  source byte, so `GlmResidentImage` dumps each built layer once (D2H,
  `pwrite`, `fdatasync`, THEN the table entry — a torn write is an absent
  layer) into one file per checkpoint × config × world × rank × head ×
  format version, keyed by every shard's header fold + `config.json`, and
  restores it forever after with O_DIRECT reads at the drive's line rate
  (5.6 GB/s measured; the buffered path was 1.2 GB/s and forced reclaim).
  The boot digest rides beside it (`<key>.digest`). 15–25 s to a ready
  model, against ~4.5 min from the checkpoint and 258 s at the start of
  the round. Knobs: `DGPP_RESIDENT_CACHE{,_DIR,_VERIFY}` (README).
  Streaming mode is untouched.

Node setup: nothing privileged. The clock lock (`nvidia-smi -lgc`) was
re-tested and does nothing (the governor sits at 2400–2560 MHz through
decode); `mlockall` is an optional safety net (measured identical with it
off once the one-pass loader landed); the loader refuses at construction
if the resident footprint + 8 GiB does not fit the device's free memory.

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
reads make aggregate physical traffic 23.420 GB/token across TP=4. The
per-rank view depends on the expert placement:

| TP=4 view | traffic/rank/token | 230 GB/s floor | 240 GB/s (prefetched) |
|---|---:|---:|---:|
| **every expert sliced across the ranks (as built since 2026-09-02)** | **5.855 GB** | **25.46 ms** | **24.4 ms** |
| whole experts, 72 per rank — expected synchronized busiest rank (retired) | 7.457 GB | 32.42 ms | — |
| whole experts — all selected experts on one rank (retired) | 12.198 GB | 53.04 ms | — |

With every expert sliced on its intermediate dimension (§5.2) each rank
reads exactly top-k × 3 slices per MoE layer whatever the routing, so the
critical path IS the mean rank and the routing distribution no longer
enters the FFN-side traffic model. The busiest-rank arithmetic and the
route-trace tool (`tools/route_trace_traffic.py`, validated against the
first real trace to +0.4%) remain for any future placement change. The
measured step (§7.6) sits ~6.8 ms above the 24.4 ms floor at T=1.

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
as fatal to the current batch. As built (M6, §11), rank 0's ownership is
realized by the admission journal: rank 0 is the sole HTTP ingress, every
scheduler tick it runs is one journal record the peers apply before
running the identical tick, and every engine op is a bus collective, so a
record can never interleave a peer's in-flight tick.

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
| routed experts | intermediate-dim slice of every expert | partial hidden sum | FFN all-reduce sum |
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

- routed experts: every expert on every rank, sliced on the intermediate
  dimension (gate/up rows, down columns) exactly like the shared expert —
  per-rank expert bytes are top-k x 3 slices whatever the routing, so no
  rank is the busiest at an FFN boundary. (72 whole experts per rank was
  the initial placement, retired 2026-09-02: its expected busiest rank
  read 3.5 experts against the mean 2.0 and every other rank waited for
  it.) The per-rank chain runs in fp32 — unrounded partial down dots, one
  fma per expert ascending, shared last — and rounds to bf16 once for
  the all-reduce;
- attention, shared-expert, and dense matrices: standard column/row TP;
- router, mHC, norms, and DSA indexer: replicated;
- embeddings and lm head: vocabulary-sharded;
- MTP: the same rules as its corresponding main-layer modules.

The shard generator records the owning ranks and scale geometry. Boot checks
hash all replicated tensors and reconcile per-rank byte totals before graph
capture.

**Quantized slice alignment (measured into the contract):** a rank's slice
of an E4M3 block-scaled matrix must start **128-aligned** in the sliced
dimension (gate/up row slices, down column packs). The local-frame scale
consumer re-anchors the 128×128 grid at the slice origin — exact for
aligned starts, unrepresentable for a slice that starts mid-block and
crosses a boundary (the following block's rows would read the straddling
block's scales; measured as a 0.30-l2-wrong boundary fold that the mHC
stream-state metric attenuated 300× — the fixture's dense inter 200 was
exactly this and passed its 0.02 budget at 0.001 for three milestones).
Misaligned quotients fail at the validators and at the view seam, loudly.
The real checkpoint's inter dims (12288 dense / 2048 shared) are
128-multiples at every TP world; non-multiple tails at world=1 remain the
kernel's masked-tile case, which is correct and separately covered.

**Sharded load (the loader IS the slicer):** at TP>1 the resident loader
builds each layer DIRECTLY at this rank's local geometry — head row
ranges and 128-aligned quantized row/column slices of every MLP matrix,
routed experts included — so only rank-local checkpoint bytes are ever read (measured on
the fixture: 57% per rank at world=2, 36% at world=4; the re-read
residue is the replicated set plus the DSA dequant-bridge tensors). The
full-load+`GlmTpViews::bind` path remains as the independent reference
implementation, and the shard-parity test pins the two BITWISE on every
bound surface of every layer — the pair cannot drift. world=1 is the
degenerate rank 0 of the same build (identical grant sequence, identical
bytes — the M4 path by construction). The two DSA bridge tensors
(`q_b_proj`, `o_proj`) are the documented exception: their quantized
slices would start mid-block at some worlds, so every rank reads them in
full and slices the dequantized bf16 — exactly what the views do to the
same buffers — until the scale-aware GEMM seam replaces the bridge.

**Boot checks:** the loader folds every replicated tensor's raw source
bytes into a per-layer digest (order-independent, so load order cannot
change it) plus the globals; ranks exchange at startup and a mismatch
pinpoints the layer. Per-rank byte totals reconcile arithmetically: the
sharded-class bytes partition across ranks exactly once and the
replicated+bridge bytes re-read per rank, so
`sum_ranks(source_bytes) == world1_total + (world−1)·verbatim` — a
double-owned or missing row breaks the identity. Load boundaries
synchronize exactly the bump's reader streams (the model's stream plus
the loader's dequant stream), never the whole device: a device-wide wait
in a one-process multi-rank world blocks on peers' spinning collective
kernels — the first-collective stall measured under ~15 ms of thread skew.

### 5.3 Control plane: roster, epochs, health (M5)

Membership and coordination run over TCP — never on the CUDA critical path.
The wire protocol and its validated semantics live in `src/net/roster.hpp`
and the M5 results file; the contract in brief:

- The coordinator (rank 0) seals the roster at epoch 1 once every
  configured rank has announced, or fails startup with exact progress.
  Its own rank is an implicit member (it needs no loopback connection to
  itself). Late joins and rejoins are refused in v1: a replacement member
  would need a state-consistency story that does not exist yet.
- Epochs are owned by the coordinator and carried by every roster frame;
  they advance on membership change (eviction). A rank that observes a
  non-monotonic epoch treats the batch as fatal — the same rule as the
  coordinator epoch mismatch above.
- A rank is evicted on process death (connection EOF) or on a heartbeat
  deadline that applies **only after the roster is sealed** — before the
  seal there is no batch to protect and slow announces are normal. The
  evicted rank receives its final roster so it can tell eviction from
  coordinator loss.
- Rejections carry a reason string, not a bare close: cluster-start config
  errors must be legible in the rank's log. `join()` returns the first
  sealed roster frozen at receipt — a later eviction racing the caller's
  wakeup must not rewrite what the rank was sealed with.
- Coordinator loss (ack deadline) is fatal for the batch; there is no
  coordinator failover in v1.

## 6. CollectiveBus protocol

Each peer pair owns RC QPs on both active lanes — one QP per slot pool per
(peer, lane), so every receive queue is type-predictable: latency (8 KiB) and
bulk (256 KiB) traffic can never consume each other's differently-sized
receive buffers. Each (peer, lane) has one pinned, remotely-writable slab
registered as a single MR, holding send slots, completion cells, receive
slots, doorbell and ack cells, credit staging, and a control cell. Bulk
chunks are striped round-robin across the lanes; small messages use the
lower-latency lane.

Receive protocol for slot `S`:

1. the receiver posts payload and doorbell receive buffers and grants a
   credit. The credit grant is an unsignaled 64-byte RDMA WRITE of
   `{hash, slot, seq}` into the sender's registered completion cells — no
   CQE, no consumed receive WR; the sender polls the cell as a pinned flag,
   the same visibility contract as the GPU ack. The record carries the
   consumer's result, so the credit doubles as the message's completion
   notification;
2. the sender posts the payload SEND followed by the doorbell SEND on the
   pool's RC QP; only the doorbell needs a send completion;
3. the GPU observes the doorbell with a system-scope acquire, consumes the
   payload, and publishes its acknowledgement with a system-scope release;
4. the CPU polls the CQ for transport errors and recycles the slot only after
   both the GPU acknowledgement and receive completions are known.

Slots are 64-byte aligned, initialized before kernel launch, and never reused
without a new credit. **Ring discipline**: plain SENDs are consumed FIFO by
the peer's receive ring, so a slot is a ring position, not a free buffer —
the sender assigns slots round-robin per (lane, pool) and may not skip a
busy position; the receiver recycles in ring order; credits therefore
return in ring order. Persistent kernels use an **inactivity** watchdog whose
deadline resets after every message; a reserved stop sequence provides orderly
shutdown. The cross-node hash test and CUDA regression test pin this behavior.

Algorithm selection is size-based, and both machines are built (§6.3):

- replicated hidden vectors (the decode step's 90 boundary folds, the
  pick): the one-shot all-to-all all-reduce over the latency pool, folded
  by a per-collective kernel — recorded as graph nodes in the decode step
  (§6.2);
- prefill chunks: the striped reduce-scatter + allgather over the bulk pool
  (`allreduce_bulk`), both lanes;
- control messages: TCP, never on the CUDA critical path (the roster, §5.3,
  and the admission journal, §11).

### 6.1 Threading and concurrency model

The data plane is a single-threaded polling engine. One dedicated thread owns
every data-plane QP, CQ, credit counter, and slot on the node. The runtime
has exactly three thread roles:

| role | owns | blocks on |
|---|---|---|
| forward/engine | CUDA stream(s), graph launches | stream/graph sync |
| bus poller | all RC QPs/CQs, slots, credits, watchdogs | nothing (bounded poll) |
| control plane | roster TCP (§5.3) | socket deadlines |

The seams are the already-pinned ones: GPU↔bus through pinned-memory flag
sequences (the flag protocol; payload-then-doorbell ordering), bus↔control
through roster snapshots. Nothing else crosses threads, so the slot and
credit lifecycles carry no locks.

Why this shape, from measurements rather than taste:

- A dedicated bus thread is *mandatory* under graph capture: the forward
  thread is inside `cudaGraphLaunch` while inbound traffic still needs
  receive-WR recycling and the graph's own send payloads need a reactive
  CPU posting SENDs. The only question is the thread count.
- Bus CPU work is proportional to message count, not bytes — the NIC DMAs
  from registered slabs, so the CPU only posts WRs and polls CQs (~1 µs per
  message, size-independent). A decode step is ~90 collectives; full-rate
  prefill striping at the measured 24.5 GB/s aggregate is on the order of
  10⁵ posts/s. One core stays under half busy at the ceiling, on a
  20-core GB10.
- Logical operations span QPs: an all-reduce step touches a different peer
  each hop, a striped chunk touches both lanes. Splitting threads across
  peers or lanes forces cross-thread joins on slot/credit state — the
  failure mode is silent corruption, not a wrong roster.
- The loop never syscalls on the hot path (`ibv_post_send`/`ibv_poll_cq`
  are user-space MMIO). Completion-channel fds would cost an interrupt plus
  a syscall per event against a measured 2.4–2.7 µs one-way budget, and
  epoll-style demux exists for fd counts this system does not have (12 QPs
  at TP=4: 3 peers × 2 lanes × 2 pools). GPU-ready flags are memory, not
  fds, and cannot be epoll'd anyway.

Loop discipline (the anti-jitter contract, since one thread serves latency
and bulk traffic): each iteration is bounded — at most N posts and M
completion drains, in priority order (GPU-ready latency flags first, bulk
stripes after); per-QP watchdog timestamps are checked once per iteration;
the thread spins through a fixed grace period when idle before sleeping, so
back-to-back decode steps never reach the sleep path.

Escape hatch, deliberately not built: if the soak shows TX backpressure or
credit starvation concentrated on one lane, bulk striping moves to one
thread per lane — the size-based split above already defines that seam —
while latency traffic stays on the engine thread. NCCL-style per-channel
threading is justified at channel counts an order of magnitude above six
QPs.

Traffic classes are concurrent from day one — mixed decode/prefill batching
(M6) must not force a transport refactor. Concretely: submission has separate
latency and bulk queues drained in priority order each iteration, with a
bounded number of bulk posts per iteration so a full-speed stripe cannot
starve a decode collective; the per-class slot pools make the credit floors
structural — bulk traffic physically cannot occupy a latency slot, so posted
receive buffers always exist for latency-class arrivals. The contention case
— decode-class collectives meeting their latency budget while bulk stripes
are in flight — is a tested Phase 2 behavior (two-node: 400/400 latency
messages at p50 18.6 µs while 64 MiB striped both lanes), not a soak-time
hope. What remains M6 is admission and scheduling of mixed steps at the
engine level; the transport never assumes serialized steps.

Per-lane failover is deliberately not designed: the two lanes share one
physical port, so its failure takes both, and remapping traffic between them
buys nothing. A lane loss is a transport failure surfaced by the watchdogs.
Configuring the bus with a single lane is a deployment choice at about half
bandwidth, not a recovery mechanism.

### 6.2 GPU-side consumption: per-collective stream kernels

The consumer of received payloads is a short-lived CUDA kernel per collective
step, launched on the engine's stream: it polls the doorbells of the slots
that step depends on (system-scope acquire, bounded by a cycle deadline),
performs the reduction into local memory, writes the result, and publishes
the per-slot acknowledgements (system-scope release) before exiting.

Rejected alternatives, for the record:

- **CPU-mediated completion** (NIC → pinned slab → cudaMemcpy → reduce →
  copy back): two full-data copies per message and the CPU on the critical
  path of every collective. On GB10's unified pool the copies are pure waste
  — the zerocopy bench has the GPU streaming the NIC's landing zone
  directly. It remains the documented emergency fallback if a driver or
  firmware upgrade ever breaks unified visibility, which is exactly why the
  NIC→GPU visibility probe reruns per deployment (M5 deliverable 5).
- **One persistent dispatcher kernel per QP** (six for TP=4): multi-peer
  steps would need inter-kernel coordination that CUDA does not offer
  cheaply — a hand-built scheduler of atomics, the silent-corruption bug
  class; each kernel holds an SM indefinitely; and, decisively, a
  never-terminating kernel does not compose with CUDA graph capture, which
  the decode path requires (goal 2). Making a replayed graph safely wait on
  a persistent kernel requires a graph-side node polling memory that kernel
  writes — at which point the per-collective kernel already exists, plus a
  permanent kernel on top.

What the chosen shape buys: stream order is the dependency scheduler (no
hand-written sync between all-reduce steps and the GEMMs that consume them);
the per-kernel cycle deadline is the §6 inactivity watchdog, one per
collective; the decode step is a fixed launch sequence, hence
graph-recordable; and the doorbell/ack sequence is the one flag protocol
already pinned by `flag_protocol_test` and validated end-to-end by the
NIC→GPU visibility probe.

Cost: one kernel launch per collective step — a few µs uncaptured, under 1%
of the ~32 ms decode-step floor at ~90 collectives, and amortized inside a
replayed graph. A persistent doorbell-watcher hybrid remains a measured
optimization option for a future launch-bound path, not a starting bet.

**Graph capture (d3): each decode shape's fixed launch sequence records once
and replays per token.** The collective kernel becomes a graph node like
any other — the engine no longer launches it and instead *reacts*. The
seams that make replay-stable what capture bakes:

- *Per-generation cells.* One pinned 64B ctl per recorded collective node
  per graph variant (vs. the eager machine's single one-flight cell). The
  arm step selects a variant, then resets its cells and assigns the window's monotonic
  generations, `gen_seq` published release-last; the replayed kernel
  acquires it at start and derives its staging row `(g−1)%ring` and its
  exit match from it. Monotonicity means a previous replay's stale
  `done_seq` can never match, so replays of one node are distinct without
  re-recording.
- *The engine generation walk.* `window_count` publishes `window_first` and
  `window_variant`; the engine adopts that shape, posts
  each generation's peer pairs when the kernel's `ready_bits` land (lane 0,
  the cursor ring position — deterministic under the era's exclusivity:
  every latency post in the era is a graph generation, in order), and
  advances on `done_seq == gen`. Stream order serializes the recorded
  kernels, so at most one generation is un-done at a time — a single-flight
  state machine, not a queue. Arm waits for the previous window's walk
  (bounded); finish joins the walk and returns the verdict.
- *The era.* Up to 16 graph variants share one bus era, with disjoint cell
  slabs and write-once node metadata. Harness sends close at the first
  `record_begin`. Eager collectives reject while a variant records or a
  replay window is armed, but run between windows (prefill and its first
  pick need this). Graph reservations and eager pickups share one monotonic
  generation counter, keeping ring positions execution-ordered across shape
  switches.
- *Slot ownership without requests.* Graph posts carry no `BusRequest`,
  but the SendSlot/credit machinery is request-shaped — a per-window
  carrier request (unregistered, `is_collective`) gives the posts owners so
  credits recycle slots exactly like eager flights; completion is the
  walk's business, not the carrier's.

Two ordering rules the bring-up measured into existence:

- **Done does not imply posted** (the graph form of the bulk lesson): the
  kernel's fold waits on the *peer's* doorbell, not this side's own post,
  so a fast peer can stamp `done` while this engine's pair is still
  ring-deferred. The walk requires the posting mask complete before it
  advances — advancing first strands the peer's kernel on a doorbell that
  never comes (measured: 131/132 posts with the peer's last generation
  spinning 5 s on the missing one).
- **Quiescence at adopt**: every kernel of the previous window exited
  (walk complete implies every claim acked), so every latency door cell
  reads consumed. An unconsumed doorbell there freezes the ring-ordered
  recycle; the receive queue then drains over the next wrap and the
  wrap's last sender RNR-retries forever — a once-in-~30k-generation
  stall observed once, unreproduced in ~40k generations since, and
  watched by a standing adopt-time quiescence note plus the per-flight
  stall microscope (failure-only, in-tree).

Measured over two fabric nodes (TP=2, 12-collective steps, decode-scale
GEMM stand-ins between the nodes): eager p50 33.6 µs per collective
(submit→wait); replayed step p50 357 µs = **29.8 µs per collective
including the inter-node compute stand-in** — the per-collective host
submit/launch cost amortizes to near-zero, and the step pays one arm +
one finish for all twelve collectives. The wire floor (measured 2.4–2.7
µs one-way) is what remains.

**The mixed era (M6 Stage 4d).** The era as first shipped was exclusive:
once a session opened, eager collectives were rejected for the bus's
lifetime — unlivable for serving, where prefill's bulk folds and the
eager pick must run BETWEEN decode windows. The seam is one generation
counter: `graph_replay_arm` reserves its window's G generations from the
same `ctl_seq_counter` eager pickups increment, so execution order equals
generation order across eras and every staging-ring reuse fence holds
verbatim (those proofs only ever needed monotonic numbering). Eager
collectives are rejected only while a session RECORDS or a window is
ARMED (arm .. finish); every gen→cell mapping is window-relative
(`(gen − adopted_first) % G` — the first mixed-era run stalled on the
absolute form). The counter is 32-bit; an arm that would cross its top
fails the era loudly (~48M decode tokens; remedy: restart). Session
discipline: one forward thread for arm/finish/eager submissions.

**The decode step as a graph, in the model (as built):** capture happens
once after prefill and the first pick, on the model's stream, with a
`GlmGraphRecordReducer` installed at the boundary seam: the boundary GEMV
writes ONE stable device buffer (cudaMalloc — a pinned buffer cost every
o_proj ~20 µs of fabric round trips for its lane-0 stores), `reduce()` is
`allreduce_record`, and the recorded kernel snapshots the buffer into the
generation's staging row itself (one row per generation, every peer's
SEND posted from it, hashed once). Every per-step host input (request
ids, positions, tokens, the MoE expert-view tables) is either a pinned,
device-mapped member a KERNEL node re-reads (`glm_upload_i32/i64` —
never a memcpy node, see below) or, since the on-device step, a device
buffer the previous replay wrote (§9). Plain decode replays 90
collective nodes per token; the MTP step 94. Capture + instantiate ~38 ms,
once. Per collective in the step (rank 0, steady): ~35 µs at T=1 (copy
2.7 + handshake 8–16 + skew ~15 + fold 4–5) and ~48 µs at T=2 (two rows
copied and folded); the "skew" is the ranks' ~3% compute spread, not the
transport. The captured graph is KERNELS-ONLY by contract
(`glm_check_decode_graph` at every capture site): a memset or memcpy node
executes on the copy-engine queue, one in-order queue shared by every
stream in the process, where a queued node's dependency wait blocks
everything behind it — in the one-process loopback worlds a peer's queued
post-collective memset held this rank's pre-collective memset while the
peer's collective spun on ours (the 2026-09-03 stall, §9 and
`docs/batched_mtp_graph_stall.md`). Kernel nodes never share that queue,
so a spinning collective blocks nothing: the loopback worlds pass at
`CUDA_DEVICE_MAX_CONNECTIONS=1` with the prefetcher on. Fabric ranks are
one process each and never shared the queue; the contract costs them a
handful of 32-thread launches per step.

One compiler lesson for the record: the shared eager/graph kernel body
refactor (a mechanical template extraction, pure renames, reference-based
row policies) *miscompiled* at -O3 — the fold read doorbell cells as
payloads — while the verbatim eager kernel plus a duplicated graph twin
was green on every gate. The eager kernel is kept verbatim; a validated
machine is not a refactoring test bed.

One measured lesson from the Phase 2 harness (which *does* use persistent
consumers): a persistent kernel starves every launch queued behind it on
the same stream, so the harness runs one stream per consumer. The
per-collective shape is immune to this class entirely — another reason it
is the recorded choice. Teardown ordering matters too: `cudaFreeHost`
synchronizes the device implicitly, so loopback pairs must stop all
persistent consumers before either side frees (two-phase stop: `quiesce()`
then `stop()`; one bus per process never notices).

### 6.3 The collective: one-shot all-to-all all-reduce

The M5 deliverable-3 primitive is a one-shot all-to-all all-reduce over the
latency pool: every rank sends its full hidden vector to every peer
simultaneously — all stripes are postable the moment the collective is
issued, the W−1 wire transfers run in parallel — and one per-collective
kernel folds the W vectors locally (bf16 loads, fp32 accumulation in
canonical global-rank order: rank 0's vector first, always, so every
rank computes the identical chain and all destinations agree bitwise —
per-rank orderings could differ in the last ulp and diverge replicated
state; the host oracle mirrors the same chain and verifies exactly).

The recorded alternative, recursive doubling, was rejected for this class:
its round-j payload is round-(j−1)'s *reduced output*, a data dependency no
side can pre-post — every round would pay the full
GPU-fold → host-flag → post → wire → doorbell chain serially, exactly the
host round trip the per-collective design removes. At TP=4 one-shot sends
3×8 KiB where doubling would send 2×8 KiB; against a 196 Gb/s fabric that
"redundancy" is noise, and it buys a single dependency-free posting wave.
Recursive doubling (more precisely reduce-scatter + allgather) remains the
choice for large buffers where link bandwidth binds — the prefill striped
bulk class, where the term sizes are MiB, not KiB.

The three seams, in launch order:

- **Staging (GPU → engine).** The per-collective kernel is also the stager:
  it writes the device source vector into each peer's claimed send slot
  (pinned LPDDR, zero-copy), `__threadfence_system`, then publishes a
  per-peer ready bit in a pinned control cell (system-scope release). The
  engine's collective pass acquire-loads the bits and posts payload+doorbell
  SENDs — the same handoff the forward integration will use when the
  producing GEMM writes slots directly. The wire protocol is unchanged:
  an all-reduce stripe is an ordinary latency-class message.
- **Inbound (NIC → GPU).** The kernel scans each peer's latency doorbell
  cells across all lanes; `doorbell.seq != ack.seq` marks an unconsumed
  message (the ring/credit discipline guarantees at most one per peer per
  outstanding collective). It claims by publishing the standard ack —
  hash = fold of the received payload, publish-last seq release — so the
  engine's recycle and credit passes run byte-identical to harness
  traffic. The engine's doorbell-CQE `expect_seq` validation catches any
  contract violation loudly.
- **Completion (GPU → engine → waiter).** The kernel publishes
  `{ctl_seq, status}` into the control cell (release) after the reduce;
  the engine's pass sees it, completes the request (the single completion
  authority, same cv semantics as `send()`), and the waiter runs one
  `cudaStreamSynchronize` after waking so cross-stream consumers of the
  destination are ordered. Credits still flow receiver → sender for slot
  recycling, but they are off the collective's critical path entirely —
  completion no longer waits for the credit round trip (the ~17 µs
  harness p50's second half).

v1 restrictions, all deliberate: at most one outstanding collective per
bus (decode is dependency-serialized layer to layer); `launch_consumers=
false` (persistent harness consumers would race the per-collective kernel
for doorbell claims); no other latency traffic while a collective is in
flight — enforced: once a bus enters collective mode, `send()` is closed
(a harness message would be claimed and folded into a peer's reduce: the
silent-corruption class, rejected loudly instead). Failure paths: the
kernel carries its own cycle deadline and exits through one common
post-scan barrier (the persistent-kernel deadlock lesson), the engine
watchdog fails the request and poisons the control cell so the kernel
exits promptly, and a lane failure completes the request with the legible
lane error. The slot that a failed request staged is freed by the normal
credit/owning-reference machinery — a late credit after the waiter reaped
never corrupts ring state.

Two ordering rules the TP forward integration taught (both were bugs):

- **Completion clears before it wakes.** The flight state (claims,
  posted stripes, the single-outstanding gate) must be fully reset
  BEFORE the request is completed — completing wakes the waiter, and a
  waiter that submits its next generation before the engine's cleanup
  races the single-outstanding check. Symmetrically, any path that
  reaps a collective out from under the engine (watchdog expiry, lane
  failure) must clear the held flight, or the bus rejects every later
  collective forever.
- **Allocation-phase device syncs never overlap a spinning collective.**
  `cudaMallocManaged`/`cudaFree` and device-wide syncs are whole-device
  barriers; a rank still constructing its model while a peer's first
  collective kernel spins on doorbells deadlocks the process (the
  kernel waits for the constructing rank's post; the constructing
  rank's allocation waits for the spinning kernel). The model
  preconstructs every layer object at construction (no allocation inside
  the forward), and TP runners barrier cluster-wide after construction
  before any forward begins. Post-startup the shape is safe — engines
  post from free threads, so staggered per-layer loads and in-flight
  collectives coexist by construction.

Three more ordering rules, all measured into existence by the bulk
machine's bring-up (§6.3's striped reduce-scatter + allgather):

- **Claim only your segment's ring window.** The bulk doorbell rings are
  flight- and phase-agnostic FIFOs and the shard geometry guarantees
  ranks do NOT progress in lockstep — a rank whose shard ends in segment
  0 enters allgather while its peer still folds reduce-scatter. The
  receiver's kernel derives each arrival's ring-lifetime index j from
  the doorbell itself and claims exactly [base, base + this lane's
  stripe share); everything else stays unconsumed for the kernel whose
  window contains it. Eager claiming acks a future segment's doorbell
  away from its owner and both phases deadlock on the loss.
- **Done does not imply posted — and neither does TX retirement.** A
  zero-arrival segment kernel (empty receive window) stamps done while
  its own outbound stripes are still being posted; the advance gate
  requires the posting drained AND the doorbell CEs retired before the
  next segment launches, or the launch's staged-counter reset strands
  unposted stripes with the release condition inverted forever.
- **The ack certifies consumption, not arrival.** The RS fold reads
  payload slots in the consume pass after the claims; acking at claim
  time returns the sender's credit early, and a sender a segment ahead
  wraps the 8-deep ring inside one claim-to-exit gap, DMAing new stripes
  over a fold input. Acking after the fold makes the credit gate close
  the race by construction — and the scan must skip the kernel's own
  claims meanwhile, or a re-presented claim steals the shared CAS slot
  from a fresh doorbell in the same warp, forever.

- **Done does not imply posted — the one-shot form** (2026-09-05). The
  one-shot kernel stages, releases its ready bits, claims, folds and
  stamps done; when every peer has already posted, all of that takes
  ~20 µs and fits between the engine's two reads of the control cell.
  The pass read the ready bits first and the done stamp second, so a
  kernel that finished in between had its flight completed with this
  rank's own stripe never posted — the peers' kernels waited on it
  forever and this rank's next generation parked behind their
  generation gate (glm_tp_test's two-rank loopback, once in fifteen
  runs: "launched" then "done after 18.5 µs" with no "ready" between,
  the packet counts one stripe short). The pass now reads done first
  (a pass that sees the stamp then sees the bits, by release/acquire
  and kernel program order) and refuses to finish a reduced flight
  before every peer's stripe is out; `bus_test`'s
  `scenario_allreduce_done_before_posted` injects a 300 µs sleep between
  the two reads on rank 0 (`BusOptions::debug_pass_delay_us`) and fails
  within seconds on the old ordering.

The bulk kernel is a cooperative grid (2026-09-05). Sixteen blocks,
co-resident by `cudaLaunchCooperativeKernel`'s contract (the launch fails
rather than deadlocks when it cannot co-schedule; two such grids run
concurrently, which the loopback test worlds need), with grid barriers
between staging | claims | consume rounds | acks. Everything that scales
with bytes runs over the whole grid in 16 KB tiles of 16-byte vectors —
the outbound staging, the RS fold, the AG landing — while the doorbell
claim loop (the shared-CAS discipline) and the deferred ack flush stay in
block 0; the claim records live in a device-resident `BusBulkScratch`.
The placement proof (a payload folds to its door's hash before anything
consumes it) moved from the claim into the consume pass: every tile
hashes the words it reads, the per-(peer, stripe) totals combine by
atomic XOR, block 0 proves each stripe between rounds and a stripe read
before its placement was fully visible is re-folded next round (the fold
and the copy are pure functions of their inputs); no ack and no done
stamp until every consumed stripe has proved (`BusStats::bulk_gate_redos`
counts the redos — zero in every measured run). The staging tiles also
fold each outbound stripe, and block 0 publishes the per-(peer, stripe)
hash table behind the staged counters, so the engine posts the door's
hash without folding 256 KB on the collective thread per stripe.

Senders are paced in software (`BusOptions::bulk_pace_gbps` per (peer,
lane) QP — derived at bus start from the slowest lane's port rate as
port / ((W − 1) × lanes) × 0.85, 28.3 Gb/s on the four-node 200 Gb/s
fabric, overridable with `--bulk-pace-gbps` on dgpp-serve and
glm_gen_check, 0 = unpaced; `bulk_inflight_per_lane` bounds the window
and the posting order rotates per sender). The sweep put the loss knee
between 80 and 120 Gb/s per QP with throughput flat from 28 to 120, so
the derived value sits at the low end of the flat region with ~3×
headroom. The one-block kernel had throttled
every sender to ~1 GB/s; the grid let three 200 Gb/s senders burst
whole rings at one 200 Gb/s port and the switch dropped packets — RoCE
sequence errors, adaptive retransmissions, CNPs, 16 MiB all-reduces
bimodal at 2.4 or 30–56 ms (`scripts/roce_counters.sh` reads the
counters around a run). The NICs pace raw-packet QPs only, so the engine
spaces a lane's stripe posts by len / rate: six inbound QPs at the cap
stay under one port even when the ack timing aligns every sender on the
same receiver. Measured on the four nodes (`glm_gen_check --bulk-bench`,
p50): 2 MiB 0.30 ms, 4 MiB 0.53, 8 MiB 1.07, 16 MiB 2.17, 32 MiB 4.45 —
10.5–11 GB/s of wire traffic per rank, against 2.6 / 5.2 / 11 / 22.5 / 75
before the grid.

Measured (fabric, bitwise-verified against the canonical-chain oracle):
TP=2 p50 37.6 µs, TP=4 (full mesh) p50 ~44 µs, submit→wait, credits off
the critical path. Kernel-internal spans are ~6 µs staging, ~6 µs doorbell
claim, ~10 µs fold+reduce; the kernel launch (~10-30 µs) is the fixed
cost that graph capture (§6.2) amortizes in the decode path. Two
scheduling lessons from the bring-up, both now engine policy: the idle
branch's `sched_yield` was a hot-path syscall that cost ~2.2 ms of poll
latency per idle stretch (fixed: a genuine hot spin whose 50 µs sleep
tail is a power courtesy, never a latency mechanism), and bare
condition-variable waits pay a CFS-timeslice wake against the hot
engine (fixed: spin-then-futex on a release-published done flag — spun
before the mutex, since spinning under it deadlocks the completer). The
same fixes took the harness send path from 16.5 to 12.1 µs p50. The
engine also never naps once a graph is recorded, and it pins itself to a
fast core (the GB10's big.LITTLE mix put it on an A725 at times).

**What a kernel may believe about NIC-written memory (the pick-path race,
2026-09-01 — three bugs deep, all in the record):**

- *A doorbell does not certify its payload — for a kernel.* RC's CQE
  ordering protects the ENGINE (the CQE consumer); a kernel polling cells
  saw ctl-correct doorbells beside payload buffers still holding stale
  bytes, and folded garbage. Every doorbell therefore carries a 64-bit
  FOLD of its payload (`StartSlot.hash`; the sender's engine hashes the
  staged row it posts), and every collective kernel — latency, graph and
  bulk — spins until the payload folds to it before consuming. The fold
  is positional (`(word + position + 1) · golden ratio`, XOR-combined):
  the first XOR fold cancelled on uniform payloads and let a stale
  `bf16(2.0)`-fill pass as an all-zero broadcast.
- *Plain loads of pinned memory cache.* The GPU's plain loads of pinned
  sysmem allocate in its caches and the NIC's DMA writes never snoop them:
  the first spin pass fetched the pre-arrival zeros and every later read
  hit the cache, forever (a 56 s spin beside a payload the CPU could see
  complete). The doorbell polls always worked because `flag_load_acquire`
  is a system-scope atomic; the payload reads never had the discipline.
  EVERY kernel read of NIC-written memory — door fields, gate words, the
  fold's peer vectors, the bulk copies and exit fold, the claim records —
  is a system-scope atomic load (`sys_load_u16/u32/u64`). This was the
  mechanism of the original fabric pick corruption AND, almost certainly,
  the gen-1825 "sent-but-never-received" wedge family.
- The gate fires in production: 1–2 waits per scheduler smoke, thousands
  under the hostile racer (`bus_small_repro --pick-race`, loopback and
  fabric, an exact oracle after every collective under injected skew).
  Telemetry stays in the ctl cell (waits/spins + three claim records with
  door len/seq/hash); the STALLED dump prints the kernel's actual claimed
  cells beside the engine's own CPU fold.

Consequences upstream: the pick scratch is pinned (`cudaHostAlloc`), never
managed — a `cudaMallocManaged` buffer shared across loopback rank-threads
on one 2 MB UVM page is the documented concurrent-access race and hid
behind this bug for a day; and the readback invariant (a decoded broadcast
must equal the locally computed winner; today the digest group, §9) is
what turns any surviving corruption into a loud, located failure.

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
  `[0, conv_kernel-1)` and the speculative reserve after it — used since
  M8 by the multi-row verify (§9), whose rows but the last also leave
  their post-row snapshots in `spec_conv_`;
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
- `select_k = index_topk / index_kpool` must be a **power of two** — the
  bitonic select/expand networks have no fallback shape (a select_k=6
  experiment silently dropped every selected pool; rejected at layer
  construction since);
- `num_heads` must be a **power of two ≥ 4** — the absorbed-attention
  kernel's head-group tiling partitions 64 lanes; heads=2 produced
  1e33-scale output garbage with *correct selections* (bisected through
  the CI machinery; 4/8/64 heads pass). Rejected at construction since;
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
- **the dense prefill regime** (2026-09-05): while visible pools
  floor((p+1)/kpool) ≤ select_k = topk/kpool, every visible pool is
  selected and the tail appended — the query attends to tokens [0, p]
  exactly — which holds for every position ≤ kpool × (select_k + 1) − 2
  (2050 at the checkpoint's geometry): the whole prefill of a prompt up
  to the 2048-token chunk, and the first chunk of a longer one. Those
  rows run `attn_dense_kernel`, a flash-style kernel over 32 (row, head)
  M-rows per block — bf16 `mma.sync` for S = Q̃·Lᵀ and for P·L with a
  32-token latent tile shared by all 32 M-rows, the FA2 online softmax on
  the C fragments (denominator unrounded, probabilities rounded to bf16
  as the split kernel pins), causal masking per M-row, the split kernel's
  (m, l, c) partial layout so `dsa_attn_combine` is shared — in 128-row
  tiles split four ways; rows past that position keep the per-row split
  kernel over their selection. Tolerance-equal to the split kernel (the
  tensor core's summation order), deterministic; gated against it and the
  host oracle at 64/16 heads × 512/256 latent. 82 KB of shared memory, one
  block per SM (this device caps a block at 99 KB and an SM at 100 KB).
  Where the split kernel spent 318 µs per eight rows per layer (a
  2048-token prefill: 0.9 s of attention), the dense kernel spends ~360 µs
  per 128 rows. Past that position the SAME kernel runs over each row's
  selected list (`dsa_attn_listed`, round 8): a 16-row slab is one query
  row at local_heads ≥ 16, so each slab walks its own list into its own
  16-token latent tile and the two slabs run to the longer one's tile
  count — no union of selections, no membership masks; gated against the
  split kernel on arbitrary lists with repeats and against the host
  oracle. Before it a chunk past position 2048 spent 1.8 s in the per-row
  kernel (every row carrying the full 2048 selected tokens); after, 170
  ms, and a 2048-token chunk costs the same whether dense or sparse. The
  absorb and vout projections around it are tensor-core kernels too at
  ≥ 16 rows (round 7): absorb bitwise its warp kernel, vout carrying its
  fp32 input as a three-way bf16 split (the full mantissa; only the
  summation order differs); decode keeps the warp kernels;
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

### 7.3 mHC residual streams

Pinned to the transformers `Glm5NextTextHyperConnection` reference (the
semantics every prior section treats as ground truth). The residual stream
is `hc_mult = 4` parallel bf16 streams of `hidden` each; all four start as
the embedding row, and the model output is the unweighted mean over
streams followed by the final RMSNorm. "mHC" names the manifold-constrained
mixing — `scale` has 3 entries because there are three outputs, not three
heads.

Per sublayer site (attention and FFN each own one), with `n = hc_mult`:

1. **Mapping.** Unweighted RMSNorm over the *flattened* `[n·hidden]` vector
   (fp32, eps = rms_norm_eps); an fp32 projection by `fn [(2+n)·n, n·hidden]`
   plus `base [(2+n)·n]`, split `pre[n] | post[n] | comb[n·n]`, each scaled
   by its own `scale` entry:
   `pre = σ(·) + hc_eps` (stream-collapse weights), `post = 2σ(·)`
   (block-output placement, range [0,2]), `comb = softmax(·, rows) + hc_eps`
   then Sinkhorn–Knopp toward doubly stochastic — one column pass, then
   `sinkhorn_iters − 1` row+column passes, every denominator adding
   `hc_eps`.
2. **Collapse.** The sublayer input is `Σⱼ pre[j]·streams[j]` (fp32
   accumulate, one bf16 round).
3. **Update.** After the sublayer produces `h`: `streams'[i] =
   bf16(bf16(post[i]·h) + bf16(Σⱼ comb[j,i]·streams[j]))` — post and comb
   are rounded to bf16 *before* the products, with intermediate roundings
   exactly as shown. The engine reproduces this choreography bitwise
   (kernel and oracle share it); deviation from it is a semantics bug, not
   a tolerance question.

The whole mHC pipeline is replicated on every rank (§5.1's "norm + mHC"
row): sublayer outputs are all-reduced first, so the stream state never
leaves the replicated region. The engine kernel computes the norm and the
24-logit projection in fp32 (fixed reduction order), applies
sigmoid/softmax/Sinkhorn in registers, and pins `hc_mult = 4` the way the
DSA kernels pin Hadamard-128 — widen only when a real checkpoint demands
it. Parity: `glm_mhc_test` vs the double oracle
(`glm_mhc_reference`), ulp-budgeted at the bf16 rounding points, including
saturated logits, zero streams (norm-of-zero), and bitwise-deterministic
end-to-end replay.

The prefill forms of the mHC kernels (2026-09-05, round 7): at ≥ 16
tokens the dots run four tokens per block with the coefficient matrix
staged through shared memory (`mhc_dots_tiled_kernel`; the per-coefficient
form derived a token's inverse RMS in 24 blocks and re-read its streams
48 times), every thread walking the same vectors in the same order with
the same block-sum tree, so the outputs are BITWISE the per-coefficient
form's (glm_mhc_test pins collapsed, post, comb and the fused norm), and
the finish runs in-block per token; the stream update computes a
position's four streams in one thread. Decode's fused per-coefficient
form (graph-captured) is unchanged.


### 7.4 MoE routing (noaux_tc) and expert execution

Pinned to the transformers `Glm5NextTextTopkRouter` / `Glm5NextTextExperts`
reference. Router math is fp32 end to end (the config's `moe_router_dtype`
contract): logits = fp32 GEMM of the post-LN hidden (bf16 values) against
the bf16 gate rows; `scores = sigmoid(logits)`; selection ranks
`scores + e_score_correction_bias` (bias-corrected) but the ROUTED WEIGHTS
are the UNCORRECTED scores gathered at the selected ids; `norm_topk_prob`
divides by (Σ + 1e-20) per element, then multiplies by
`routed_scaling_factor` (2.5). The engine's tie rule — equal biased scores
select the LOWER expert id — is pinned where torch's CUDA topk leaves ties
unspecified. The kernel emits ids in ascending expert order, which is also
the accumulation order: the reference's `index_add` loop visits experts
ascending, so per token `out = 0; for e ascending: out = bf16(out +
bf16(w_e·y_e))`, and the shared expert (weight 1) is added last as a single
bf16 add. Deviating from that order changes bits, not just tolerances.

Experts are swiglu MLPs with ASYMMETRIC clamps: the gate clamps only its
maximum (no lower bound), `up` clamps both sides at `swiglu_limit` (10);
`act = bf16(bf16(silu(g)) · up)` — two rounding points, matching torch's
opmath. The engine executes experts through the scale-aware GEMM on the
compressed E4M3+scale weights (§4).

Two execution paths, as built:

- *Prefill* (chunk-amortized; the grouped path since 2026-09-04, without
  a host sync since the same evening): the router's ids stay on the
  device and `moe_segment_kernel` (one block, a thread per expert) builds
  the segmentation there — the (token, slot) pairs by ascending expert,
  stable in (token, slot) within one, the slot→row map, a segment table
  with every expert (empty ones included; their blocks exit) and the
  shared segment last — while the routing traces ride async copies into
  per-layer pinned staging materialized after the chunk's final sync
  (`Outputs.routes` unchanged); the host-orchestrated `enqueue` remains as
  the reference the gates pin the device path against, bitwise. Then the
  layer's rows — every routed pair in segment order, the tokens once more
  for the shared expert — are gathered at once and run as ONE launch per
  matrix over every segment (`moe_grouped_mma_kernel`, 2026-09-05: one
  block per 64-column n-tile and segment walks the segment in 128-row
  m-tiles, stages the bf16 activations and the fp8 weights — decoded,
  scaled and rounded to bf16 exactly as the dequant bridge does — in
  shared memory and runs bf16 `mma.sync` with fp32 accumulation in
  ascending k16 order, so every output element is bitwise the scale GEMM's
  tile kernel and a segment up to 128 rows reads its expert's weights
  once; the shared segment alone splits across blocks in whole m-tiles),
  swiglu over every row, the fp32 down the same way, and the per-token
  ordered accumulation in one pass (`moe_accum_ordered_kernel`: the K
  slots sorted by expert id, `__fmaf_rn` from zero, the shared row at
  weight 1, one bf16 rounding — the same chain). The device segmentation
  is three launches (a count per expert, the scan, a placement pass per
  expert with a block exclusive scan over the match flags — the
  single-block form's stable order, bitwise) and the tensor-core kernel
  reads the hidden rows through the row map instead of a gathered copy
  (2026-09-05, round 7); the prefill router's dots are tiled 16 tokens x
  16 experts per block with the per-lane order preserved (bitwise the
  warp form) while decode keeps the fused warp form. `MoeExpertKernel` picks
  the grouped kernel: the prefill and the forward run the tensor-core one;
  `enqueue()` (the reference) takes it as an argument, GEMV by default,
  because the decode slot path and the MTP module's multi-row eager rows
  are pinned bitwise against the GEMV core (`moe_grouped_gemv_kernel`:
  rows staged four at a time through the same `fp8_gemv::block_rows` the
  decode GEMV uses — which on a 2048-token chunk's ~57-row segments read
  every expert fifteen times per matrix, 3.4 s of the prefill). Before
  either the experts ran per segment through the small-M tile GEMM —
  578 µs a call on an 8-block grid — which was 3.5 of a 256-token
  prefill's 5.2 s. The expert-view table the grouped kernels read is
  uploaded per layer from a RING of pinned host tables, each guarded by
  the event its last upload recorded (`upload_expert_views`, 2026-09-05):
  one MoE layer object is rebound for every layer and the session prefill
  has no per-layer host sync, so a single pinned table was refilled with
  the next layer's pointers while the previous layer's asynchronous copy
  could still be pending — that layer then ran on the wrong experts
  whenever the host got a layer ahead (a world of one has no collective
  to wait on; the fabric's per-layer all-reduce had hidden it). The host
  waits only when it is four uploads ahead, and the copy sits right after
  the router and segmentation kernels, so the wait never drains the
  stream.
- *Decode* (`GlmMoeLayer::enqueue_decode`, M6 Stage 4c and rounds 2/7/8):
  zero host round trips. The router leaves ids ascending per row on the
  device; a slot-ranking kernel orders the (row, slot) work by expert id
  so experts shared across rows are L2 hits; the gate+up+swiglu kernel and
  the down kernel are fp8 GEMV cores (warp per weight row, 16-byte fp8
  loads, the scale grid applied in the epilogue — `fp8_gemv.cuh`; every
  row a slot, `top_k+1` slots per row, the +1 the shared expert) reading
  the route and the expert-view table from device memory (the eager path
  uploads that table from the same guarded ring as the prefill; capture
  reads its per-slot graph tables); the down dots
  stay fp32 (unrounded) and one ordered accumulation kernel runs the
  chain — one fma per expert ascending, shared last — rounding to bf16
  exactly once as the sum leaves for the all-reduce. Since the sliced
  placement (§5.2) a rank's slice cannot reproduce the whole expert's
  bf16 per-expert rounding, so this fp32 chain is the more accurate of the
  two reachable deviations (decision with the user, 2026-09-02); the
  unsliced engine sits at max 1 ulp against the double oracle, and the
  sliced form is certified against the UNSLICED oracle with a per-element
  bound built from the partials in hand (bf16-on-the-wire's cost, the
  class the attention o_proj already pays). The expert-view tables live
  in per-graph-slot DEVICE tables uploaded once before capture (a
  binding-keyed cache was a wrong-weights factory under the streaming
  loader, whose bindings are not identity-stable — the rule stands). The
  router's select runs behind its dots kernel (last-block ticket), and
  the top-k is a warp with the exact strict-> rule (bit-identical). Both
  paths are pinned bitwise against each other by `glm_moe_test` at three
  geometries including a TP partition.

Fusions tried on the decode MoE and rejected with numbers (2026-09-03):
the accumulation behind the down kernel (+22 µs/layer — a barrier + fence
+ atomic per block over 9216 blocks), the slot ranking in the gate_up/down
prologues (+10 µs on the 9216-block down launch against a 1.6 µs kernel).
At that grid a per-block prologue is 24 waves deep; the launch it saves is
~1 µs.

Route traces (deliverable 5): the router records per-layer ids/weights in
the `DGPPTC1` format, and `tools/route_trace_traffic.py` replaces §3's
uniform-expert assumption with measured occupancy — per-token busiest-rank
experts/layer (uniform expectation 3.515), per-layer batch critical rank
for correlated routes, and the corrected critical path in GB/token. The
tool's anchors are §3's own numbers: a uniform-random trace must reproduce
3.515 experts and 7.457 GB/32.42 ms; an all-one-rank trace must reproduce
12.198 GB. A uniform trace is the null model, not evidence — real traces
come from representative prompts through the assembled model.

### 7.5 The assembled forward and the parity discipline at depth

`GlmDiagnosticModel` (M4 deliverable 1; the name predates its serving role
and stayed) runs the full text stack: embedding -> 4-stream mHC init -> per
layer [attn_hc -> ln1 -> KDA or DSA -> stream update] and [ffn_hc -> ln2 ->
dense MLP or MoE -> update] -> unweighted stream mean -> final norm ->
lm head (fp32 accumulators since 2026-09-03: `GemmOut::F32`, the same
GEMV template — the bf16 rounding of the logits was the first thing the
teacher-forced gate could see). In streaming mode one layer is resident at
a time and the KDA/DSA/MoE layer objects are REBOUND to each layer's
resident views; in resident mode every layer is materialized at
construction (§3) and the bindings are lifetime-stable, which is what
graph capture requires. The two layer norms and the final norm are the
Glm5NextTextRMSNorm TWO-rounding choreography (u = bf16(x*rsqrt); y =
bf16(w*u)) — a dedicated kernel, because the generic single-round rmsnorm
drifts the GLM path systematically. The MTP draft layer (index 45) is
loaded when `--mtp` asks for it and runs as §9's draft block; the reference
model class ignores it.

**Sessions (the incremental engine, M6 Stage 2/2b).** The layer pipeline
is ELEMENTWISE IN TIME: the mHC streams are recomputed per layer from each
token's embedding, so all temporal recurrence lives in the KDA
recurrent/conv state and the DSA latent/index/tail caches, and a stateful
step is "don't reset the state, feed one token". `session_prefill(req,
prompt)` opens a request slot (fresh state) and runs the prompt in
pool-aligned 2048-token chunks keeping state across them, returning the
last row's logits; `session_step(req, token)` runs one token at the slot's
next position (the same recurrence implementation as prefill — what keeps
cross-path noise at GEMM ulps); `session_close(req)` releases the slot's
blocks immediately (a meter that lags a retire is an admission deadlock).
Slots: `max_requests` (≤ `kDecodeRows` = 8, the DSA select bound) with
slot-major KDA state so one open is one memset pair; the DSA pool's
per-request tail rings are zeroed at open and the latent/index caches are
deliberately not scrubbed (a new owner rewrites every row it reads — the
pool's release contract, confirmed by the real-model gates). Rules the
gates pinned: a continuation chunk must carry ≥ kpool tokens (the DSA
tail-seed read is pinned to in-chunk rows), so a 1..kpool−1 token tail
borrows one pool from its predecessor; a plain `forward()` while any
session is open throws (the pools are shared — the corruption would have
looked like noise); every host op that moves a position pushes it to the
device mirrors (`d_session_pos_`, `d_mtp_pos_`), so the device-driven
graph (§9) always starts from the host's view. Eager decode remains
time-multiplexed. M6.6a's batched graph runs one fixed slot-major row batch:
every stateful decode kernel takes the same request-indexed map (one span per
configured slot, per-row request ids selecting the actual state slot, and
positions with −1 marking padding). `KdaRequestRows` gives KDA conv and
recurrence the map DSA already used; DSA's ring update honours the per-row id
rather than the span ordinal, and both layers write zeros for padding rows so
a padded row's block output is deterministic by construction. Snapshots use global physical-row offsets so
each request's commit can independently restore its first accepted state.
The adaptive adapter also owns one slot-specific scalar graph per request
slot. The scheduler's ordered `step_batch` still hands it every live slot;
the adapter executes those scalar variants sequentially below the crossover
or advances all slots in the single row-batched replay above it.

Parity at real depth is chaos-limited, and the curated suite is designed
around the measured facts, not around an aspiration of bit-parity:

* Module noise floors (measured, M2/M3/M4): KDA engine-vs-torch
  3.6e-3 l2, DSA ~3e-3 kept-row (fp8-indexer near-tie flips are inside
  its design), MoE expert path 0-1 bf16 ulps vs the double oracle.
* Those floors COMPOUND: a free-run 45-layer forward diverges from the
  torch reference at ~1.18x per layer and decorrelates by layer ~30
  (measured; the mHC residual stream is mildly amplifying — post in
  [0,2], Sinkhorn-normalized comb columns). No bug is involved: every
  layer's ISOLATED drift sits flat at the floor (<= 7e-3 at layer 44 as
  at layer 0).
* Therefore the real-checkpoint suite compares per-layer ISOLATED: every
  layer starts from the reference trajectory, so drift is bounded by one
  layer's floor. Kept rows (tokens whose routing did not flip) must sit
  under 2e-2; a route flip legitimately moves that token's row O(1)
  (the reference's noaux bias exists to tie scores at the selection
  boundary, so cross-implementation noise flips them). Since the M4
  close-out, every flip is CERTIFIED, not counted: the router kernel
  exports its full biased-score row, the reference dump carries its own,
  and `models/glm/route_audit.hpp` requires (a) the engine's selection to
  be the spec top-k of the engine's OWN biased scores and (b) every
  swapped expert pair to straddle the boundary within 32x the token's
  MEASURED cross-implementation noise — the noise yardstick taken over
  the experts NOT involved in the swap, so corruption concentrated on the
  swapped experts cannot hide inside its own inflation (the router
  itself is separately pinned to 5.3e-7 against the double oracle). The
  head runs on the isolated final streams: top-1 must agree on every
  token, and a disagreement must likewise certify as a boundary near tie
  (reference top-2 logit margin within 32x the measured logit noise) —
  a reduced layer budget crowds the head's boundaries, and the
  truncated-stack suite exercises exactly that.
* Free-run outputs are still REPORTED (drift curve, route agreement
  rates, top-8 margins) — they are the honest description of what
  cross-implementation bf16 inference at 45 layers looks like — but no
  free-run parity is asserted.

The first real trace (technical prose, 272 tokens) lands on the uniform
null within measurement noise: busiest-rank 3.541 experts/token (uniform
3.515), corrected critical path 7.484 GB/token vs 7.457 (+0.4%). The
§3 traffic model's uniform assumption is validated for this prompt class;
sampling other prompt classes (code, multilingual) is future work and the
tool accepts their traces unchanged.

### 7.6 The decode step as built: kernels, prefetch, and where the time is

The single-stream decode step is one serial chain of ~1,600 kernels per
step (90 collectives among them at T=1, 94 with the draft), replayed as one
graph (§6.2). Its
composition after the 2026-09-01..03 rounds (nsys node trace, rank 0,
steady steps), for the T=2 MTP step of 42.4 ms: bf16 GEMVs 14.6 ms (every
non-expert matrix: KDA in/out projections, DSA projections, the two heads
— 3.9 GB at 267 GB/s effective), MoE gate_up 10.9 + down 7.05 (fp8 GEMV
cores, 43 layers × two rows), collectives 4.56 (94 × 48.5 µs), mHC 1.6,
and ~3 ms of small kernels (KDA recurrence 34 × 10.5 µs, DSA select and
attention 12 × ~27 µs each, routers, norms, the step's own control kernels
~60 µs in total), with ~1.15 ms of launch gaps (0.7 µs each). At T=1 the
plain step is 31.3 ms of which ~24.5 is the weight floor.

The pieces that make the number:

- *GEMV cores* (`bf16_gemv.{hpp,cu}`, `fp8_gemv.cuh`): warp per weight
  row, 16-byte loads, row-independent for each m ≤ 4 launch; decode shapes
  through m=8 are split into legal chunks, so a multi-row verify or serving
  batch is bitwise the single-row step. The fp8 core applies the
  128×128 scale grid in its epilogue and reproduces `scale_gemm.cu`'s
  tile arithmetic verbatim. Weights live in DEVICE memory (`cudaMalloc`):
  the managed-memory placement cost the eager step 86.4 → 75.0 ms/token
  when it was fixed (round 3), and a pinned-host DESTINATION costs a GEMV
  ~20 µs of fabric round trips for its lane-0 stores.
- *The L2 weight prefetcher* (`l2_prefetch.{hpp,cu}`, `WeightPrefetcher`):
  the chain sits in latency-bound phases (the bus, the KDA recurrence, the
  DSA select/attention, mHC + norm) where DRAM idles, and the GB10 has
  24 MB of L2. A lowest-priority side stream, forked from the chain by
  events (graph edges under capture), reads the NEXT kernels' weights into
  L2 during those phases: a window at every block boundary (the other
  side's mHC fn, norm, router, shared expert / next layer's fn, ln1, the
  first 12 MB of the in-projection; the head after the last layer) and one
  after each attention layer's in-projection. The rate is the design
  point: 3 MB in flight made the step SLOWER (the shared memory controller
  queued every access on the box — bus handshake 9.7 → 26 µs); 16 blocks
  × 2 loads (128 KB in flight, "light") everywhere is the measured optimum
  and puts the GEMVs' effective rate above DRAM's. Knobs
  `DGPP_L2_PREFETCH{,_MB,_LIGHT_BLOCKS,_BOUNDARY,_LAYER}`; re-checked at
  the T=2 shape (layer off +0.8 ms, boundary full +1.9, off +2.6). A side
  effect: the step-time distribution collapsed (p90−p50 ~5 → ~0.5 ms) —
  the memory system never idles long enough to downclock.
- *Kernel reshapes under the reassociation rule* (§1): router dots one warp
  per (expert, token) with a shuffle tree (30.8 → 10.8 µs for dots +
  select); mHC dots vectorized, the finish (Sinkhorn on xor butterflies)
  running in the LAST dots block per token behind a ticket, the sublayer's
  RMSNorm fused into its tail — a site is one launch; KDA recurrence 16
  lanes × 8 columns per v-row (32 → 10.9 µs against an 8.7 µs floor); DSA
  absorb_q with shared partial sums (40 → 8.8), vout one warp per row
  (34 → 5.8). Measured cost of all of it: perplexity 2.573 → 2.583 on the
  556-token text, delta −0.0037 ± 0.0044 nat/token (not distinguishable
  from zero), and not worse at 14k tokens.
- *Fewer graph nodes:* route traces off in serving (126 D2H nodes), expert
  tables uploaded once before capture (42 H2D nodes), the logits/hidden
  tail mirrors off in the on-device step (§9); since 2026-09-03 the graph
  carries no memset/memcpy node at all — the DSA select-counter reset and
  the row-table/token/position uploads are kernels (§6.2, the copy-engine
  queue deadlock of the loopback worlds).

What is left, and why it is where it is: the 94 collectives (~4.6 ms at
T=2 — handshake 8–16 µs and the ranks' ~3% compute spread, not the wire;
decomposing the handshake needs a globaltimer calibration project first),
~1.1 ms of launch gaps only fewer, bigger kernels would remove, the second
verify row's ~7 ms of unshared experts (the price of depth 1), and the
weights. The small kernels are at their floor: fusions that add per-block
work lose on the 9216-block MoE launches and fusions that only remove a
launch save ~1 µs against 42,000 (§7.4).

Two host-side facts the rounds turned into rules: every async copy on the
decode path needs a PINNED source (a pageable `cudaMemcpyAsync` performs a
stream sync before initiating — the whole pipeline drained once per MoE
layer until the expert-table staging was pinned); and nothing on the
serving node may push the box to its memory watermark (§3 — the swapped
vocab page).

## 8. Prefix cache (M7, built 2026-09-05)

V1 uses exact state snapshots only. The previous unproven 24 KB/token
“linear-aux replay record” is removed.

A reusable prefix entry contains:

- immutable token blocks and tokenizer/chat-template revision hashes;
- attached DSA MLA/index blocks and the exact incomplete tail;
- one final KDA recurrent+convolution snapshot for that prefix boundary;
- the draft block's last hidden row (`h_q`, 8 KB) so a resumed request can
  draft immediately (§9);
- model/checkpoint revision and numerical-mode identifiers.

Only a radix node with a complete snapshot is attachable. A match without one
is treated as cold and is re-prefilled; the engine never reconstructs state
from an underspecified record. Attaching copies the 36.39 MiB/rank KDA snapshot
into request-owned mutable state and shares immutable DSA blocks by reference.

The initial snapshot arena is capped at 1.5 GiB/rank, permitting 42 complete
prefix snapshots before metadata. Admission and eviction are agreed by epoch
on all ranks. Eviction cannot free blocks until all request and snapshot
references reach zero.

**How it fits the built engine.** Everything the entry needs already has a
home: the DSA pool shares blocks by reference through one block table per
request (latent and index blocks co-located; block = 128 tokens = 32
pools); the KDA slot is a fixed 36.39 MiB per request with an
export/import format carrying revision and geometry (`kda_snapshot.hpp`);
the tail ring is 22.5 KB per request; the position mirrors are pushed to
the device by every host op. The new pieces are the radix over token ids,
the snapshot arena (42 D2D slots), refcounts on blocks, and the decisions.

- *When to snapshot:* at prefill end and at request close (the whole
  exchange, so the next turn attaches to it). Cost: one D2D copy of 36.39
  MiB (~0.2 ms) + tail + `h_q`.
- *Attach:* copy snapshot → the request's KDA slot; bump block refcounts
  and write the block table; copy the tail ring; set `d_session_pos_` and
  `d_mtp_pos_` to P; then the suffix. A suffix shorter than kpool cannot
  run as a prefill chunk (§7.5's continuation rule) and runs token by
  token through the decode path, which handles any position and yields
  the last row's logits the pick needs anyway.
- *Rank agreement:* rank 0 decides lookup/attach/snapshot/evict at
  admission and journals the decisions in the tick record (§11); peers
  apply. The radix is a pure function of the journaled prompt stream, so
  a peer whose own derivation disagrees dies loudly — the same discipline
  as a scheduler divergence. Eviction is LRU over refcount-0 entries;
  blocks held by entries count against the pool meters, so admission
  sees them.
- *Hot == cold:* bitwise only when the hot path replays the cold path's
  chunk sequence (bf16 GEMM outputs differ by ulps across chunk sizes,
  M2). Either snapshots are taken only where cold prefill would chunk
  (2048-token multiples and the prompt end) with turn-end snapshots
  compared at the certified near-tie tier, or cold prefill chunks at
  message boundaries too, making every turn-end snapshot a cold chunk
  boundary and the criterion bitwise everywhere. Decided 2026-09-03: the
  second — prefill chunks at message boundaries as well as at 2048-token
  multiples, so a snapshot at any turn end replays the cold path's exact
  chunk sequence and the cache is bitwise invisible.
- *Keys:* tokenizer hash, template hash, checkpoint revision, numerics
  mode; the radix is per process (no persistence in v1).

**As built, Stage A (2026-09-05; `models/glm/forward.hpp` "prefix cache
primitives").** Three refinements the gate forced on the design above.
(1) *Snapshots sit at pool-aligned positions.* The DSA tail ring and the
complete-pool compression index pools from a chunk's first row, so a
chunk start must be a multiple of kpool (4) — the cold prefill therefore
cuts at the aligned IMAGE floor(b / kpool) · kpool of each structural
boundary, not at the boundary, and a turn-end snapshot lands there; the
few tokens between the image and the boundary belong to the next chunk.
(2) *Continuation chunks may be shorter than a pool.* The layer used to
reject them (the reference tail seed read only in-chunk rows); on the
device ring the seed writes the tokens it has into their slots and the
rest still hold the previous chunk's, so a short suffix runs as a prefill
chunk and the decode-path detour above is unnecessary (one- and
two-token continuations are gated within 1e-7 of the unchunked prefill,
the steps after them bitwise). (3) *The partial last block is copied,
the full ones shared.* An attached request keeps writing into the block
its position falls in, so the entry owns a private copy of it
(`acquire_pinned_block` + `copy_block_contents`, every layer's latent
rows, index pools and scales) and the requests that attach each copy it
again; the full blocks below are shared by reference through the pool's
new per-block refcounts (a request row holds one reference per block, an
entry one more; a block frees at zero). The draft block's prefill is
interleaved per chunk so a snapshot at any cut carries its state, and
`h_q` is the row at position − 1. The snapshot is one buffer the caller
owns: [KDA recurrent | KDA conv | one tail ring per DSA layer | h_q];
`session_snapshot` fills it stream-ordered and returns the block ids,
`session_attach` opens a closed slot from it, `session_prefill_resume`
runs the suffix under the same cut rule in the whole prompt's
coordinates. Hot == cold is bitwise through the resume, the steps and
the draft rows (the gate), and a second attach to the same entry with a
different suffix is bitwise its own cold run.

**As built, Stage B (2026-09-05; `sched/prefix_cache.*`,
`models/glm/prefix_arena.hpp`, the scheduler, the adapters, the service, the
journal).** Four choices, each departing a little from the sketch above.
(1) *The decisions live in the scheduler, not in a new rank-0 protocol.*
The scheduler already runs the same deterministic policy on every rank
over the same journaled requests; the prefix cache is one more pure
function of that stream, so attach / snapshot / evict are rank-identical
by construction. The journal carries the INPUTS (a submit's boundaries
and opt-out) and a CHECK (each tick record's "pd": rank 0's decision
digest after the previous tick; a peer compares before applying and dies
on a mismatch) rather than the decisions themselves; the warm record
carries rank 0's slot count. The decisions also ride the op stream.
(2) *Boundaries are positions of role-marker tokens in the prompt ids*
(<|system|>, <|user|>, <|assistant|>, <|observation|>), not offsets
recovered by re-rendering message prefixes: a pure function of the
shared prefix, identical across turns, one scan, and the legacy route
gets them for free. The cold prefill cuts at their aligned images (and
at 2048 multiples). (3) *Two kinds of entry.* A cold prefill takes one at
its deepest cut (the assistant header in a chat — the next turn's cut);
a live request keeps a ROLLING snapshot in one arena slot at every
aligned committed position, and at retire (EOS or the cap) that slot
becomes the close-time entry at floor((end − 1) / kpool) · kpool, the
aligned image of the EOS token's position — where the next turn's
`<|user|>` (the EOS token itself) cuts. The lookup is exact and confined
to the prompt's own cuts, so an attach is always at a cold-path cut and
therefore bitwise. (4) *The arena is the engines'.* `PrefixArena` holds
device slots of one session's state (`--prefix-cache-gib`, 1.5 GiB → 42
at real dims), stream-ordered snapshots and attaches, event-timed;
entries pin their DSA blocks in the pool, and an admission short of
blocks or a snapshot slot evicts the LRU unattached entry (never one a
live request opened from). Metrics: hits, misses, tokens saved, entries
taken and evicted, blocks pinned, the arena's copy times, the TTFT split.
The two-token step's parity (closed 2026-09-05): the MTP graph's
two-token steps fix the committed count's parity while the draft is
accepted, so an answer can land on no aligned position at all (measured:
a fully accepted 26-token answer to a 25-token prompt). The HOP snapshot
covers it: when the next aligned position is committed + 1 the scheduler
arms the engine (`SchedulerEngine::prefix_arm_hop`), and a step that
commits both rows takes the state after its first row — the hopped
position — before returning (`GlmDiagnosticModel::session_snapshot_post_
row0`: the recurrence, conv and ring kernels' post-row-0 spec snapshot
rows, the draft block's pre-draft ring from its rollback snapshot, h_q at
P−1 from the hidden cache, the blocks below P by reference and the
partial one copied — bitwise the one-row snapshot, gated); a one-row
step lands on the position and the regular rolling snapshot follows at
the next tick; the decision ("hop", the same digest code as a rolling
snapshot) rides the op stream and the journal's digest like every other.
A request whose committed position is aligned at retire snapshots the
live state then, as before. Still: no persistence; the close-time entry
hits only when the next turn re-renders the previous answer to the same
ids (the template keeps the reasoning with `clear_thinking: false`, and
the answer must have ended in EOS).

## 9. MTP transaction model

The contract (draft depth was to start at three and adapt; as built it is
one, for the measured reasons below). Verification never writes over
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
6. Cancellation or rank failure discards the entire transaction. As
   built (2026-09-05): a cancel lands at the tick boundary — the step in
   flight completes on every rank and the request retires before the next
   engine op, so the other requests' transcripts are untouched and the
   cancelled one's emitted tokens are a prefix of its answer (gated under
   the row-batched MTP graph); a rank failure ends the service — the step
   in flight never completes on any rank, nothing after the last completed
   step is committed anywhere, every client gets exactly the committed
   tokens and then the engine_failure error (the failure semantics of
   §11 and `fabric_serve.hpp`, gated in-process and drilled on the four
   nodes).

Tests reject at every depth from zero through `k`, including a rejection that
crosses an index-pool boundary. Greedy output and all subsequent states must
match non-speculative execution.

### As built (2026-09-03): depth 1, greedy, post-row snapshots

The implementation keeps the contract above with a simpler mechanism than
the `k+1` candidate-state indices: the KDA recurrence/conv kernels and the
DSA ring-stash kernel take an optional snapshot sink and store the state as
it stands after every speculative row but the last (`KdaStateSnapshots`,
`dsa_kpool_decode_update`'s `tail_snapshots`); the last row lands in place
as always. Accepting every row is therefore free, and retracting to `a` rows
is one memcpy per state family from snapshot `a-1`
(`GlmDiagnosticModel::session_rollback`). DSA latent rows and index pools
need no rollback: they are positional writes the rewound position simply
overwrites, and a query's visible pool count is derived from its own
position. `session_verify` runs T ≤ 4 rows (`kSpecRows`, the bf16 GEMV's
row bound: at T ≤ 4 every projection takes the row-independent GEMV, so
the verify rows are bitwise the T=1 rows). The Phase-2 fabric latency slot
is 64 KiB and folds up to eight hidden-4096 rows in one collective;
`bus_greedy_pick_rows` remains the eager gather + broadcast path.

The draft block (`models/glm/mtp.cpp`) is the checkpoint's layer 45: a plain
pre-norm DSA + MoE block (no mHC) over `eh_proj([enorm(embed(tok_{q+1})) |
hnorm(h_q)])`, where `h_q` is the main stack's pre-final-norm stream mean
kept in a per-position cache, headed by `shared_head.norm` and the shared
lm head. It owns one more DSA pool ordinal and MoE graph slot, runs over the
prompt's rows 0..P-2 at prefill, and afterwards over exactly the rows the
main stack accepted (only accepted tokens ever enter it, so it never rolls
back); its row for hidden position q uses position q (vLLM's convention —
a uniform shift is RoPE-invariant but not kpool-invariant). Depth is 1
because the layer is trained at depth 1 and because the verify's cost is
linear in rows through the MoE (a second row's experts are new DRAM bytes
unless shared with the first; the decode slots run in expert order so
shared experts are L2 hits): measured 39.8 ms for T=2 vs 31.3 for T=1, so
a k-th draft must be accepted well over half the time to pay. Rank 0 does
not broadcast an accepted count: every rank folds the identical
candidate table and computes the identical verdict (`judge_verify`); the
pick's readback check pins the equality.

### The on-device step (2026-09-03): one graph per step

The graph era's step used to end at the head: the host scanned the logits
slice, ran two pick collectives, judged, rolled back, staged positions and
tokens, ran the draft block eagerly with its own folds and pick, and
launched again. All of that is now inside the replay, so a step is one
`cudaGraphLaunch` whose recorded nodes carry the control flow:

1. `glm_spec_positions` derives the rows' positions from the device-side
   session position (`d_session_pos_`); the fed tokens are already in
   `d_tokens_` (written by the previous replay's last node).
2. The T=2 verify (the 90 boundary folds as recorded collective nodes).
3. The pick (`kernels/pick.{hpp,cu}`, `GlmDevicePicker` in
   `models/glm/tp_bus.hpp`): `glm_pick_local` computes each row's canonical top-2
   (bitwise `glm_sample::local_max`) and encodes this rank's (fp32 logit
   bits, id) as the wire's six-bit digits into a device table; ONE recorded
   collective SUM-folds the table (a gather over disjoint slots — every slot
   has exactly one nonzero contributor, so the fold is exact);
   `glm_pick_verdict` decodes every rank's identical table, merges per row
   (`merge_greedy`), judges against the fed tokens and writes the verdict
   (`accepted`, `next`, `winners`) to a pinned mirror and a device copy. The
   broadcast collective is gone; its readback invariant is the **digest
   group**: one more row of slots carries each rank's 54-bit digest of its
   previous verdict, and a rank whose digest differs is flagged on every
   rank at the next pick (`GlmDevicePicker::verdict()` throws with every
   rank's digest) — the 2026-09-01 corruption class, caught one pick late.
4. `glm_spec_commit` (`kernels/glm_spec.{hpp,cu}`): when `accepted < T`,
   a predicated copy of every snapshot family's row `accepted-1` over the
   live state (KDA recurrent and conv slices, one DSA tail ring per main
   layer — a memcpy node cannot be conditional, a kernel can); always
   `d_session_pos_ += accepted`.
5. `glm_spec_draft_rows`: the draft block's FIXED two rows off the verdict —
   accepted rows real at the block's device row counter, the rest padding
   at position −1 (the DSA decode path skips negative positions, so nothing
   is written anywhere; the input kernel reads position 0's hidden for it).
   The verify's `next` is parked in `d_next_` because the draft's pick is
   about to overwrite the verdict slot.
6. The draft block (two folds), its head on BOTH rows (the lm head is
   bandwidth-bound; m=2 costs what m=1 costs), and the draft's recorded pick
   reading the last ACCEPTED row (`Inputs::row_select`) into the picker's
   second slot.
7. `glm_spec_next_tokens`: `d_tokens_ = [next, draft]` for the next replay.

The host arms the bus window, launches, syncs, finishes the window, reads
the two pinned verdicts, advances its position mirrors
(`session_graph_settle`) and logs. DSA admission for the whole run happens
once before the capture (`session_reserve_blocks`); the tail's logits and
hidden never cross to the host (`set_decode_tail_mirrors(false)`). Every
host op that moves a position on the eager path pushes it to the device, so
the device-driven graph always starts from the host's view.

Measured (rome_onegraph vs rome_mtp2): 42.36 vs 42.85 ms/step, 22.45 vs
22.71 ms/token, step p50 42.3 vs 42.8, p99 44.6 vs 45.2, max 44.8 vs 46.2
— mean and median moved together, the tail a little more. Honest
accounting: the host's share of a step was smaller than budgeted (the eager
draft's overhead beyond its weight floor was ~0.3 ms, the host seams were
already hidden behind the GPU); what remains is inside the graph — 94
collectives at ~41 µs and ~3.5 ms of small kernels and launch gaps around a
~34 ms weight floor. Tests: `glm_pick_test` (kernels vs the host oracles),
`glm_tp_device_pick_graph_loopback_matches_host_pick` (the device verdict
== the host pick + judge over the same replayed logits, every step) and
`glm_tp_full_graph_step_loopback_matches_eager_speculator` (the one-graph
step in lockstep with an eager `GreedySpeculator` on a second model over
the same bus: every step's (accepted, next, draft) equal; transcript ==
plain on every rank). Those tests capture the model's graph in-process,
which surfaced a hazard of the one-process multi-rank world and fixed a
contract for every world: the captured decode graph is KERNELS-ONLY
(`glm_check_decode_graph`, called at every capture site). A memset or
memcpy node executes on the copy-engine queue — one in-order queue shared
by every stream in the process, where a queued node's dependency wait
blocks everything behind it. The batched-MTP loopback stall (2026-09-03,
`docs/batched_mtp_graph_stall.md`) was exactly that: rank B's replay had
queued its post-collective DSA counter memset (waiting at the queue head
on B's collective), rank A's pre-collective memset queued behind it, and
B's collective spun waiting on A's — an nsys node trace shows A's memset
executing 992 ns after B's, five seconds late, once the watchdog poisoned
B's kernel; a kernels-only synthetic graph never stalls even at
`CUDA_DEVICE_MAX_CONNECTIONS=1`, and adding one 4-byte memset node per
collective stalls it on the first replay there. The decode path's
non-kernel nodes (the DSA select counter reset; the request-id, span,
token and step-position uploads; the tail's logits/hidden mirrors, which
are recorded only while `set_decode_tail_mirrors` is on and copied eagerly
after a replay otherwise) became kernels or eager copies; the loopback
gates run with prefetch on at 1 and 32 connections
(`CUDA_DEVICE_MAX_CONNECTIONS=32` stays for the eager collectives' stream
spread). A spinning kernel never blocks another
stream's kernels — the earlier hardware-queue reading was the right
shape on the wrong queue. Fabric ranks are one process each and never
share the queue; the contract costs them nothing.

Where the step's time is after the on-device work, and the small-kernel
round that found the floor, are §7.6. The scalar step is driven by
`glm_gen_check --decode-graph --mtp`; `dgpp-serve --decode-graph --mtp`
generalizes it to at most four request slots. The concurrency-1 service
shape is gated through the real scheduler on a two-rank loopback bus and
measured on the four-node service at 43.6–44.1 ms per replay,
21.8–26.0 ms/token by acceptance (§11). The fixed eight-row Phase-2 worlds
first measured 78.94 tok/s at eight live T=1 requests and 65.79 tok/s at four
MTP requests, but regressed one-live throughput. The accepted adaptive path
selects scalar below four live requests and reaches 76.18/60.27 tok/s at
full T=1/MTP occupancy, with the complete curves in §11 and the measurement
record.

### What remains (design)

- *Sampling under MTP:* exact speculative sampling with a deterministic
  draft accepts draft `x` with probability `p(x)` under the verify row and
  otherwise samples from `p` with `x` removed. `p(x)` needs the row's
  global log-sum-exp: one more digit group in the pick table carrying each
  rank's slice lse, folded as logaddexp on the device. The same field
  enables a confidence-gated T (skip the draft row when its margin is
  thin). The EAGER half is built (2026-09-04): `spec_select_from_sorted`
  is the accept test and residual over the plain sampler's final set
  (accept iff `u1 < P(x)`, else the residual walk with `u2` over the same
  fp32 exps, the residual denominator `final_den - exps[x]`; two draws
  per step whichever way the test falls, row 1's ordinary sample taking
  `u2` when the draft stands), `spec_accept_from_prefix` /
  `spec_accept_complete` make it width-independent exactly as the plain
  decision (the pure regime evaluates the draft's fold mass and walks the
  residual against `u2 (1 - P(x))`), `spec_reference_sharded` is the
  step's reference at a layout, `bus_spec_accept` / `bus_sample_row` run
  it over the bus with the gather fallback and rank 0's digest, and
  `SampledSpeculator` (models/glm/speculative.hpp) is the eager driver — the
  count table is prompt + committed + next, the draft joining it only when
  it stands; `glm_gen_check --mtp --sample` runs it. Gates: the
  width-independence and marginal-distribution unit gates, the synthetic
  two-rank `glm_spec_accept_matches_reference_loopback` (accepts, rejects,
  fallbacks, bitwise the reference, two draws per step) and the
  rank-identity gate on the MTP fixture. The one-graph step's DEVICE
  verdict is built too (2026-09-04): the sampling pick's local kernels
  treat the two verify rows independently (row 1 penalizes with the draft
  counted on top of the request's table, which holds only committed
  tokens), and the verdict kernel ports `spec_accept_from_prefix` for row
  0 and `sample_from_prefix` for row 1, writing the greedy judge's shape
  (winners[0] the draft or the residual, winners[1] row 1's sample,
  accepted 2 or 1, next = winners[accepted-1]) so the commit, draft-row
  and token-feed kernels are unchanged, committing the consumed token to
  the count table and the draft only when it stood, and flagging a
  fallback per row: row 0 undecided REJECTS provisionally (the commit
  keeps the post-row-0 state, right for a reject and recoverable for an
  accept; the host adds the draft to the table if its decision accepts),
  row 1 undecided feeds its argmax. The in-graph draft snapshots its DSA tail
  ring before its rows (a recorded copy kernel, `glm_device_copy`), and
  the adapter's fallback (`serve_mtp_fallback`) rolls the block back
  (`session_draft_rollback`), decides on the host — row 0 through
  `spec_accept_complete` over the gathered row, re-running the verify's
  second row eagerly (`session_verify`) and sampling it when the draft
  stood after all; row 1 through `sample_complete_logits` — re-drafts the
  true rows eagerly, reseeds the [next, draft] feed and pushes the
  counter; the host context mirror follows the device count table (the
  draft joins only when it stands). Gates:
  `sample_pick_t2_matches_spec_oracle_over_simulated_world` (accepts,
  rejects, both fallback rows, the greedy judge and the count table,
  bitwise the oracle) and
  `glm_tp_serving_mtp_graph_sampling_matches_eager_speculator` (the scalar
  and batched MTP graphs in lockstep with the eager sampled speculator,
  transcripts and [next, draft] feeds equal on every rank through 17
  fallbacks of both kinds). `dgpp-serve --decode-graph --mtp` samples at
  the checkpoint's defaults, and the acceptance at those settings was
  measured on the fabric on 2026-09-04: 67–81 % of drafts accepted under
  the exact accept test against 74–87 % argmax agreement on the same
  prompts, 1.67–1.81 tokens per replay, 25.0–27.6 ms/token sampled
  against 33.2 on the plain graph — well above the ~30 % break-even the
  second verify row's ~9–11 ms of a 42 ms step sets (the design had
  expected 55–70 %; the model is more confident on its own generations).
  (vLLM's recipe for this checkpoint runs the MTP layer at depth 5; our
  depth-2 measurement — a second draft accepted ~60% of the time — is
  consistent with the layer drafting well recursively, and depth stays 1
  here for step-time variance, a decision to reopen explicitly, not an
  open item.)
- *Depth:* fixed at 1. A second draft row was measured accepted ~60% of
  the time against a ~29% break-even for its ~7 ms of unshared experts,
  but it makes the step bimodal; declined for variance, not for mean.
- *Confidence-gated T (closed 2026-09-05, declined by its ceiling):* the
  lse in the pick table (6b) makes the gate possible, but its most it can
  save is the second row's cost on the steps whose draft would have been
  rejected — ~0.9 ms of a 42 ms step at the measured 88.7 % greedy
  acceptance (2 %), ~1.7–3 ms sampled — for a third graph variant per slot
  (a T=1 step that still runs the draft block) and a rank-identical gate
  decision. Not worth the shape.
- *The eager first draft (closed 2026-09-05, measured and left):* the
  prompt's last row goes through the draft block eagerly after the
  prefill's pick — one eager draft step and one pick per admission, the
  block's ~2.1 ms weight floor plus ~0.3 ms of launch overhead (the eager
  per-step draft's numbers of 2026-09-03), against a hot TTFT of 255 ms
  and a cold one of 600 ms and more. The pick has to precede it (the
  draft row embeds the token the pick decides), so it cannot join the
  prefill's tail; a graph of its own would recover the 0.3 ms.
- *The `FabricPicker` seam (closed 2026-09-05):* `glm_gen_check`'s host
  picks — world 1's full-head argmax or exact sampler, the fabric's bus
  merge or bus sampler, for the plain loops eager and under the T=1 graph
  and for the speculative loop's prefill and eager row picks — run through
  one `HostPick` driver with one greedy / sampling branch and the log
  lines the cross-rank tools read; the in-graph pick is `GlmDevicePicker`
  (a recorded node), the other driver by nature. The generated ids on the
  four nodes are unchanged.

## 10. Tokenization, templates, logits, and sampling

The bundled tokenizer is BPE with `byte_fallback=false` and no normalizer. Its
pre-tokenizer is the checkpoint's explicit regex `Split`, followed by
ByteLevel with `add_prefix_space=false`, `trim_offsets=true`, and
`use_regex=false`; its ByteLevel decoder sets `add_prefix_space=true`,
`trim_offsets=true`, and `use_regex=true`. The engine must reproduce these
settings and the special-token policy exactly; it must not invent a
byte-fallback or normalization algorithm.

As built (`glm_tokenizer.{hpp,cpp}`, M6 Stage 3): the Split regex is matched
EXACTLY against the pinned string at load and implemented as a hand-written
leftmost-first scanner over committed Unicode 15.0.0 range tables
(`tools/gen_unicode_tables.py`); BPE with `ignore_merges=true` (whole-word
vocab check first — ~97k GLM words diverge from a plain merge walk), no
unk; 36 added tokens by byte-trie leftmost-longest extraction; decode
skips special tokens by default (HF 0.23's default — the EOS decodes to
the empty string). 55 golden cases byte-exact against HF tokenizers 0.23.1
and against gigatoken (three oracles), the corpus keyed by
tokenizer.json's FNV-1a-64 so a different revision refuses rather than
compares. 20 MB tokenizer.json parses in 0.08 s; encode/decode are const
and thread-safe.

`chat_template.jinja` is loaded or compiled when a model revision is installed,
not at engine build time. The compiled artifact is keyed by the template and
tokenizer hashes. Golden tests cover English, Chinese, code, reasoning blocks,
and tool calls against reference-rendered strings.

As built (`glm_chat_template.{hpp,cpp}`, Stage 3b): a from-scratch Jinja
interpreter (lexer with trim_blocks/lstrip_blocks and the trim-marker set,
AST, Python-semantics truthiness/equality/str, `tojson` with raw UTF-8 —
the override transformers installs) that implements exactly what the GLM
template uses and REFUSES everything else at parse time (`{% do %}`,
`include`, `join`, `is number`, macro defaults, …): a template revision
that grows constructs fails loudly at install. 26 goldens (text AND
encoded ids) vs jinja2 running transformers' own compile config, keyed by
the template's hash; the four ranks render identical bytes. No library was
usable: none of the C++ Jinja engines carry the transformers semantics,
and there is no Python on the serving path.

The lm head is vocabulary-sharded:

- greedy: reduce the local `(value, token_id)` maxima;
- finite `top_k`: merge each rank's exact local top-k;
- unrestricted `top_p`, `min_p`, logprobs, or penalties requiring the full
  distribution: gather the FP32 vocab slices to rank 0, apply the exact
  reference sampler, and broadcast the chosen token and RNG counter.

The full-logit fallback moves about 619.5 KB/token for this vocabulary; it is
not described as a tiny merge. More elaborate distributed selection is an
optimization only after parity and profiling.

As built (`sample/sampler.hpp`, M6 d3): the three paths share ONE selection
semantics — every path funnels into `select_from_sorted()` over candidates
in the canonical total order (logit descending, id ascending), and the
merge provably yields the same set in the same order as sorting the full
vocabulary, so the distributed paths are exactly equal to the centralized
oracle (the gate asserts float equality). Numerics in HF warper order:
penalties (repetition, frequency, presence) → temperature (≤ 0 = greedy) →
top-k → min-p → top-p (the crossing token stays in) → one uniform draw
(fp64 walk over fp32 probabilities). The RNG is counter-based: `splitmix64`
over (seed, counter), one counter advance per stochastic draw, none for
greedy — any rank reconstructs any request's draw sequence from the seed
and the step count. Wired: the greedy path everywhere — host
`bus_greedy_pick` (gather + broadcast with the readback invariant) on the
eager path and the on-device `GlmDevicePicker` in the graph step (§9) —
and, since 2026-09-04, the exact stochastic sampler on the EAGER engines
(below: the width-independent seam, `make_fabric_sample`, the request
seam). The graph engine still picks greedily only; bound to it, the service
serves greedy defaults and refuses `temperature > 0` with a named-parameter
400 (`sampling_unsupported`).

Design for the rest (PLAN M6 6b). Three facts fix the shape. The model
card's recommended and evaluated settings are `temperature=1.0,
top_p=0.95` (the checkpoint's `generation_config.json`), so the served
DEFAULT is full-temperature nucleus sampling and the fast path must be
exact and free in THAT regime; the OpenAI API has no `top_k`, so
exactness is over the full vocabulary, not a client-declared prefix; and
`temperature: 0` must remain today's greedy path at zero cost.

*Defaults from the model, overrides from the command line.* The loader
parses `generation_config.json` (`GlmGenerationDefaults`; the EOS ids
already come from it) and the service fills every field a request omits
from it — the HF contract — rather than from the OpenAI wire defaults;
`dgpp-serve`/`glm_gen_check` override per process with `--temperature
--top-p --top-k --min-p --seed`; a missing file or field falls back to
greedy with a log line, never to a silent value. `/v1/models` reports the
effective defaults.

The loader half is built (2026-09-03): present fields and EOS ids are parsed
strictly, missing fields retain explicit presence information and log their
greedy-safe/neutral fallback, and both generation executables give the
generation file's EOS set precedence over `config.json`. The request seam
is built (2026-09-04): `SchedulerRequest` carries the `glm_sample::Params`
spec and the seed (greedy by default, so every older manifest and gate
keeps its exact op stream); `SchedulerEngine::supports_sampling` /
`configure_sampling` arm a slot immediately before its prefill pick, and
the scheduler refuses a stochastic request on a greedy-only engine at
submit, identically on every rank; the journal's tick record carries the
spec as float BITS plus the seed for stochastic submits only. The service
accepts `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`,
`seed` and the HF extensions `top_k`, `min_p`, `repetition_penalty`
(validated, the 400 names the field), fills every omitted field from the
checkpoint's defaults with the process overrides applied, draws a fresh
seed per seedless request (or the `--seed` one) so every rank replays the
same draw sequence from the journal, and reports the effective defaults on
`/v1/models` as `"sampling":{"available":…,"defaults":{…}}`. Bound to an
engine that cannot sample (today's graph engine) it collapses the defaults
to greedy with a WARN line and refuses `temperature > 0` — never a silently
applied mode. `glm_gen_check` deliberately keeps its greedy loop as the
default (its transcripts are the regression instrument) and samples under
`--sample` or any override, eager engines only.

The k-sizing instrument is built (2026-09-03), separately from the production
sampler. With `--teacher-file F --sampling-profile`, every rank selects its
exact local top-256 from the host-visible vocab slice and contributes that
table plus its fp64 slice log-sum-exp in one diagnostic bus fold. The union
contains the exact global top-256; every rank logs its full-distribution mass
at k={32,64,128,256}. `scripts/fabric_sampling_profile.py` requires identical,
contiguous evidence from every fetched rank and combines the teacher runs to
choose the smallest measured k at or below a 1% fallback rate. The profiler is
not the device path and its eager collective is not a throughput measurement;
the serving-side complement is `scripts/serve_width_sweep.sh` over
`dgpp-serve --sampling-candidates`, which fixed k at 128 on 2026-09-04 (below).

The width-independent correctness seam is built (2026-09-04), before fixing
that k. `sample_from_prefix` consumes the canonical global candidate prefix
and the fold normalizer of the complete temperature-scaled distribution (each
slice's fp64 log-sum-exp folded in rank order), and returns either an exact
sample or an explicit fallback. Width independence is by construction: every
step whose value depends on the unseen tail goes through the fold normalizer,
and the same code runs on a prefix and on the complete list, so a prefix that
resolves yields bitwise what the complete list yields — and the complete list
IS the full-logit fallback (`sample_reference_sharded`, the reference at a
given vocabulary layout). A fallback leaves `(seed, counter)` untouched so
that path consumes the SAME draw. The regimes: finite `top_k` materializes
its support and runs the shared selector unchanged (bitwise
`sample_reference`); `min_p` and `top_p` decide their survivor set on the
prefix (the first min-p failure in the scaled-logit domain; the nucleus
crossing on the fp64 fold masses) and hand the materialized set to the shared
selector; pure temperature sampling walks the fold masses and resolves only
when the draw lands inside the prefix. In the unbounded regimes the fold is
therefore the definition of the normalizer; `sample_reference`'s single fp32
denominator coincides with it except where the two disagree on a crossing
(an exact-tie boundary) — precisely the event a complete-list shortcut in the
first cut got wrong, now pinned by a unit gate. `bus_sampling_prefix` applies
request penalties before the local top-k, transports every fp32 candidate and
fp64 slice LSE losslessly as six-bit bf16 digits, runs the decision
identically on every rank, and then carries rank 0's decision digest
(resolved flag, token, logprob, covered mass — a function of every
transported digit) back through a second latency collective that every rank
must decode identically: the greedy pick's load-bearing readback invariant,
so a corrupt readback on one rank is loud at the collective rather than a
silently divergent token. The two-rank loopback gate exercises
GLM-5.3-Flash-FP8's actual default regime (`temperature=1.0`, `top_p=0.95`,
no semantic `top_k` — the checkpoint's `generation_config.json` carries
exactly those two sampling fields) over a run of draws against the sharded
reference bitwise, with a cross-shard tie that survives the penalties and
decides draws, rank-identical RNG state, and the flat-distribution fallback.
It is deliberately a host oracle/transport gate.

*The eager engines sample end to end* (2026-09-04). The fallback's
collective is `bus_gather_logits`: every rank's PENALIZED fp32 slice to
every rank as ONE bulk-class collective between windows, each fp32 as four
8-bit digits in bf16 words — bf16 holds every integer in [0, 256] exactly,
exactly one rank writes any slot, and the fold's fp32 accumulation of
x + 0 + … + 0 followed by the bf16 store returns x, so NaN payloads,
infinities, denormals and -0 (which raw bf16 halves would canonicalize or
flush) arrive bit for bit; 8 bytes per vocab id, ~1.24 MB per fallback
token, gate-pinned across two bulk stripes with adversarial payloads.
`make_fabric_sample` is the closure the eager engines run: the prefix
decision at `kSamplingCandidates` = 128 per rank (the bring-up width; the
profiles fix it), else the gather and `sample_complete_logits` — widening
exact prefixes (1024, 8192, 65536, then the complete sort) under the
normalizer the prefix step already transported, with the draw the prefix
left untouched; rank 0's decision digest echoed after either. Its loopback
gate runs six steps alternating a resolved and a fallback shape over a
growing penalized context and is bitwise the sharded reference at the
loader's layout (`vocab_layout`), one draw per step. `GenEngineAdapter`
keeps per-slot spec/RNG/context and picks greedily at temperature 0; world 1
runs `sample_full_logits` at the one-slice layout.

*The device path is built for the plain (T=1) graphs* (2026-09-04,
`kernels/sample_pick.{hpp,cu}`, `GlmDevicePicker`'s sampling mode,
`GlmGraphEngineAdapter`). The arithmetic contract first: the sampler's
transcendentals are `common/det_math.hpp` — Cody-Waite exp and atanh-series
log with every multiply-add an explicit fma, bitwise identical on the host
and the device (`det_math_test`: two million inputs, and within 1–2 ulps of
libm) — the penalties' frequency step is one fused op, the slice
normalizer sums in fixed 256-element chunks, and the kernel file builds
with `--fmad=false` beside the host's `-ffp-contract=off`, so no
contraction can differ between the two sides (or between two fabric nodes'
libm builds). The wire table is the greedy pick's generalized:
`[rows][world][k candidates × 9 digits + the slice lse × 11 digits]` plus
the digest group; unused candidate slots carry an EMPTY id no vocabulary
reaches (a shard narrower than k, or a greedy row's single argmax). Kernel
1 is three launches, every row independent (the request's `[vocab]` count
table holds the tokens committed through the previous step and is only
read here: a row's context is the table plus the step's fed tokens
through that row, so the MTP verify's row 1 sees the draft). One block
per 256-id chunk applies the penalties IN PLACE on the row's logits slice
and takes the chunk's temperature-scaled max; one block per chunk
computes the chunk's normalizer partial (the deterministic fp64 exp is a
13-term polynomial and GB10 runs fp64 at 1/64 rate — a slice's 38,720
terms are ~250 µs of one SM however one block spreads them, ~16 µs across
the chip); one block per row sums the partials, selects the exact local
top-k in canonical order and writes the group. The normalizer's
summation order is fixed pairwise trees (`glm_sample::chunked_exp_sum`:
each chunk halved recursively, the chunk partials likewise), which a
kernel keeps exactly with an 8 + log2(chunks)-deep dependent chain — a
dependent fp64 add is ~90 ns here, and the sequential chunk order cost
~30 µs per slice. The select: each thread's largest key over its strided
share of the slice, the k-th largest of those 1,024 maxima (a radix
select in shared memory) being a proven lower bound on the slice's k-th
largest key that only ~k(1 + a few percent) ids reach; those ids are
listed in shared memory and sorted (bitonic on four warps) when they fit
the candidate width, else selected exactly by a radix pass over the list
with an index tie pass, and a slice with more than 2,048 ids tied at the
bound falls back to the same radix passes over the whole slice
(`sample_local_topk_tie_paths_match_local_topk` pins all three paths
against `local_topk`). The first cut — one block per request over both
verify rows, the DSA select's 2,048-key streaming bitonic machinery,
sequential chunk sums — cost ~900 µs per row and ~250 µs per verdict,
which the service measured as +2 ms per MTP replay; the kernel bench now
reads 2 + 16 + 26 µs for the three local launches over two rows and
25/33 µs for the T=1/T=2 verdict. A greedy row writes its canonical
argmax as the one candidate and touches nothing. Kernel 2, one block per
request: decodes every rank's group into composite keys, merges the
canonical prefix by rank counting (`merge_topk`: a candidate's position
is its index in its own list plus the count of smaller keys in every
other list, in parallel), folds the lse (`merge_logsumexp`, the slices'
terms in parallel, summed in rank order), evaluates every candidate's
fold mass and selector exp in parallel, and on one thread runs
`sample_from_prefix` — the same regimes, the same fp32 selector and fp64
walk (the running fp64 prefix computed once per row and read by the
covered mass, the nucleus crossing and the pure walk), the same counter
RNG — writing the `GlmPickVerdict` the commit and token-feed kernels
already read (accepted 1, `next` the decision or, on a fallback, the
provisional argmax) plus a `GlmSampleOutcome` (fallback flag, counter
after, normalizer, covered mass, logprob) and the spec's advanced
counter, then commits the step's fed tokens to the count table; a third
pass computes the digest chain exactly as `glm_pick_verdict_batched`. The width is the
widest k that fits the fixed batch's rows in one latency slot
(`glm_sample_candidates_that_fit`: 112 per rank for eight rows at world 4
in the 64 KiB slot, 128 below seven rows), logged at startup. The
adapter's slot state is the eager engine's: `configure_sampling` pushes
the spec and zeroes the count table between windows, the prefill decides
on the host (`make_fabric_sample`) and uploads the prompt's counts, and
after a replay `collect_verdict` either takes the device's counter or
serves the fallback exactly as the eager engine does — the penalized row
to the host, `bus_gather_logits`, `sample_complete_logits` under the
transported normalizer with the reserved draw, rank 0's digest — and puts
the true token where the graph fed itself the provisional one (the scalar
variant restages `pending_`; the batch reseeds its device feed). Gates:
`sample_pick_matches_host_oracle_bitwise_over_simulated_world` (greedy,
resolving, falling-back and padding requests side by side at two slice
widths: candidates, lse bits, penalized logits, counts, tokens, logprobs,
counters, normalizers and the digest chain all bitwise the host oracle's)
and `glm_tp_serving_graph_sampling_matches_eager_engine` (the scalar and
batch variants with a greedy request beside a sampled one, in lockstep
with the eager sampling engine on the same bus; capped at six candidates
per rank so the 96-token fixture forces fallbacks, transcripts equal on
every rank). Sampling under MTP followed the same day (§9: the T=2
verdict on the device, the draft rollback on a fallback).

*Logprobs on the wire* (2026-09-04): a request that asks (`logprobs`,
`top_logprobs` 0–20 on chat; the legacy integer on completions) reports
every generated token's log-probability and top-N alternatives — the
sampler's own `Result` (the final distribution's `scaled − lse`, exact
for the top-k). SEMANTICS, stated because they differ from some servers'
defaults: at temperature > 0 the values are under the request's FINAL
distribution — after temperature, top-k, min-p and top-p, renormalized —
so a token that is the whole nucleus reports logprob 0 and one
alternative; at temperature 0 they are under the raw model distribution.
The raw-at-temperature form (`scaled − Z`) is one line away on every
path; the temperature-independent raw form would need the T=1 normalizer
folded beside the request's, a second lse digit group. The seam: `SchedulerRequest::logprobs` (−1 none, else N,
also `sampling.logprobs`), `SchedulerEngine::supports_logprobs` /
`configure_logprobs` / `take_logprobs` (one Result per token an op
returned), the scheduler's `on_token_logprobs` observer event right after
each `on_token`, the journal's `lp` field (a greedy request that asks
carries its spec too). A greedy request that reports (or carries
penalties) takes the FULL path on every engine — the fold at temperature
1 with the penalties applied, the canonical argmax under the raw
normalizer (`greedy_from_prefix`; bitwise the greedy pick's token, as
`select_from_sorted`'s greedy branch reports over a complete list); the
plain greedy pick stays for the common case. On the device the spec's
`logprobs` field switches greedy rows onto the full path and the verdict
fills the outcome's per-row top-N (`report_top`, min(N, the final set)
under the decided lse); the adapter turns outcomes and host fallbacks into
Results. The service renders the OpenAI shapes: chat `logprobs.content`
entries with `token`, `logprob`, `bytes` and `top_logprobs` (streamed with
each content chunk), the legacy `tokens`/`token_logprobs`/`top_logprobs`/
`text_offset` object. Gates: `greedy_from_prefix_reports_the_raw_distribution`,
`scheduler_logprobs_rideWithEveryTokenWhenAsked`,
`serve_logprobs_openAIShapesOnEveryRoute`, the codec round-trip,
`sample_pick_reports_logprobs_bitwise` (greedy-with-penalty, nucleus and
pure rows), and the plain graph sampling loopback gate now compares every
token's report between the eager and the graph engine bitwise, a greedy
logprobs request among them. Remaining below: the fabric profiles that fix
k, and the measurements.

*The device path.* The pick table (§9) generalizes from 2 to k candidates
per rank and gains a digit group for each rank's slice log-sum-exp; the
one recorded gather then delivers the exact global top-k (the
canonical-order merge of per-rank top-k) AND the exact normalizer
(logaddexp of the slices), so the verdict kernel knows every candidate's
true probability after temperature. Penalties and `logit_bias` apply
before the local top-k from a per-request count table the commit kernel
keeps. The kernel decides on the device whether the request RESOLVES
INSIDE the candidates — the top-p cut lies within them if their
cumulative mass reaches `top_p`; the draw `u` (the counter RNG, identical
on every rank — no broadcast; the digest group catches a divergent draw)
lies within them if it lands under the kept mass — and samples exactly
when it does. Otherwise it flags a fallback in the pinned verdict and
every rank runs the exact gather (fp32 slices as a bulk collective
between windows, the host sampler with the same `u`). k was to be sized
so the fallback is rare AT T=1/top_p=0.95; the teacher-text profiles
(2026-09-04, PLAN 6b) say it cannot be, at any width one latency slot
carries: the exact-gather rate is 34% of positions at k=128 on the hard
text (25% at k=256; 17% combined over the three texts, 0.08% on the
memorized one). The width is 128 per rank (112 at eight rows in the
64 KiB slot; the local top-k is the DSA decode select's composite-key
machinery), the fallback is exact, and its cost per occurrence is
measured (the width sweep of 2026-09-04, on the service, same tokens at
every width): 5.65 ms on the plain graph and 14.3 ms under MTP, against
a 0.0% rate at the card's settings and ~1% at temperature 1.2 / top_p 1.0
at k=128 — so the width stays 128 and no wider tier is built; the lever,
should a workload ever run hot enough to need one, is the fallback's own
cost (the host round trip and the eager re-draft), not the slot. Inside
the one-graph MTP step a
fallback means the draft ran on a provisional token: the draft's tail
ring rolls back from its snapshot and the draft re-runs eagerly on the
true token. Sampling under MTP is exact speculative sampling with a
deterministic draft (§9): accept with probability `p(draft)`, else sample
from `p` with the draft removed — acceptance ≈ E[p(draft)], lower than
greedy's argmax agreement by physics, still above the ~30% break-even at
T=1 by expectation; measured before it is claimed.

**Constrained decoding (M6 6g, built 2026-09-04;
`glm_tool_grammar.{hpp,cpp}`, the mask in `sample/sampler.hpp` and
`kernels/sample_pick.cu`).** The tool-call surface's guarantees — `tool_choice`
required / named / none, `parallel_tool_calls: false` — are a MASK on the
pick, not a prompt trick: a grammar of the template's tool-call format
(`turn := think? body; call := <tool_call> NAME (<arg_key> KEY </arg_key>
<arg_value> VALUE </arg_value>)* </tool_call>`; NAME an automaton over the
vocabulary's token TEXTS across the request's tool names, so any BPE
tokenization of a name passes and nothing else; KEY the schema's property
names when it closes them with `additionalProperties: false`, else free;
VALUE typed by the property's schema since 6i — a JSON text under the
JSON machine below for a JSON-typed property (the template writes those
through tojson), one of the enum texts for an enum string, free text for
a plain string, a type list with string, an untyped or undeclared key;
the turn ends on `<|observation|>`; EOS is withheld while a call is owed;
thinking stays free) yields, per position, the set of ids the model may
emit next. `tool_choice: auto` arms it too (6i): calls at will, or one
under `parallel_tool_calls: false`, every call well-formed; the
derivation from a function definition is `grammar_tool_from_function`,
and `function.strict: true` refuses a property outside the enforceable
subset by keyword path. The mask's meaning to the sampler is one rule
everywhere: a masked id is an ABSENT candidate — `-inf` in place, never
listed by the local top-k, zero mass in every normalizer (a slice with
every id masked folds as `-inf`, skipped), and the decision's vocabulary
is the count of present ids, so a constrained row samples the
distribution restricted to the mask and renormalized, exactly, through
the unchanged selector. Host and device apply it identically: the host
paths (`apply_mask` before the local top-k; `sample_complete_logits` /
`spec_accept_complete` over the present count; a masked draft rejected
outright with `draft_excluded`), and the device kernels (a per-row mask
table — the header word is the allowed count, 0 unconstrained — read by
the prepare kernel, which writes `-inf` in place; the local kernel's keys
treat `-inf` as absent and select among the present ids; the verdict
decides over the row's allowed count and rejects a masked draft without a
fallback). The rounding guards of every walk pick the last token WITH
mass, never an absent one. The grammar state lives per slot in the
engines (`SchedulerEngine::configure_constraint`, the spec riding the
journal's `gr` field), is advanced by every committed token, and stages
the next position's mask before each step — for the MTP verify, row 0
under the state and row 1 under the state advanced by the pending draft
(row 1 is used only when the draft stands, exactly that position); the
in-graph draft itself stays unconstrained and a disallowed draft is
rejected at the verify with probability 0, so the acceptance rate falls
inside structural states and correctness never does. The prefill pick is
the first constrained position. Every rank derives the same mask from
the same record and the same tokenizer (peers build the grammar's token
table from tokenizer.json at startup) — no new collective. Building it
found a bug in the MTP fallback path from 6b: the in-graph draft's head
reuses the logits buffer, so after a replay the buffer holds the DRAFT's
rows and the host's fallback decided over them, not over the verify's
penalized rows; the graph now snapshots the verify rows (a copy kernel
node after the pick, before the draft) and the fallback gathers from the
snapshot, with a loud invariant — the gathered row's covered mass under
the device's normalizer must equal the device's bit for bit — on every
fallback of both graph shapes. Gates: `tool_grammar_test` (the
states' masks and counts, the name/key automaton on shared prefixes and
partial tokenizations, every mode, EOS withheld, death on a disallowed
id), the masked cases of `sampler_test` and `glm_pick_test` (absent
ids everywhere, a fully masked rank, the greedy masked argmax, the masked
draft — bitwise host = device), the two-rank loopbacks
`glm_tp_serving_graph_constrained_matches_eager_engine_and_grammar` (the
plain graphs bitwise the eager engine under every mode and temperature,
every token inside a shadow grammar, masked gather fallbacks included)
and `glm_tp_serving_mtp_graph_constrained_is_rank_identical_and_valid`
(the MTP graphs rank-identical and grammar-valid with fallbacks of both
rows), and `chat_template_test`'s
`glm_tool_grammar_accepts_the_golden_turns_over_the_real_tokenizer`
(every golden tool-call turn accepted position by position over the real
tokenizer; a second call refused under a single-call spec, EOS refused
while a call is owed, a foreign name refused at its first token).

**JSON-constrained output (M6 6h, built 2026-09-04;
`glm_json_grammar.{hpp,cpp}`).** `response_format` `json_object` /
`json_schema` is the second grammar behind the same mask: the content is
one JSON text — an object in `json_object` mode, a text conforming to the
schema in `json_schema` mode — then EOS; no tool calls (a turn is JSON or
calls). Three layers. `JsonLexer` is a byte automaton for RFC 8259 JSON
with a container stack (the structural states, strings with their
escapes, the number grammar, the literals); every byte advances or
rejects, and a top-level number counts as complete while it may still
grow. `compile_json_schema` turns OpenAI's structured-output subset into
nodes — `type` (and lists), `properties` / `required` /
`additionalProperties`, `items` / `minItems` / `maxItems`, `enum` /
`const` over scalars, `anyOf` — and refuses anything else at compile time
NAMING THE KEYWORD PATH; `json_object` is the schema "a root object
holding anything". `JsonMachine` is the lexer plus schema cursors: at a
value position the expected node filters the value class (an `anyOf`
splits the cursor per alternative; the frontier shrinks as bytes
disambiguate, union semantics), a closed object's keys are spelled from
the declared names byte by byte and an enum's value from its JSON texts,
`,` and the closers obey `required` and the item bounds, integer-typed
numbers admit no fraction or exponent. The mask is the set of ids whose
text the machine accepts byte by byte; computing it by simulating 155k
tokens per position would cost milliseconds, so `JsonTables` precomputes,
once per vocabulary (0.06 s, in parallel), the static lexical answer for
every tabled (state, container context, integer flag) over every token —
a token that pops its frame and goes on is stack-dependent and
re-simulated per position — and classifies every token by where free
string content can begin or end inside it (its structural prefix before
its one unescaped quote, its tail after it, its tail after a scalar). At
a position the schema is applied by simulating REPRESENTATIVES: one per
distinct prefix or tail, the pure-structure tokens and the few
multi-quote tokens individually, group members only where the content
itself is constrained (a closed object's key, an enum, a literal's
spelling, an escape); inside a string the answer is cached until the next
structural event or a cursor's death. Measured over the real tokenizer:
36 µs per position on average, 198 µs at the worst structural position
under a closed schema. EXACTNESS is a gate, not an argument:
`json_grammar_test` proves `mask()` equal to the brute-force answer
(every token simulated) at every position of mask-driven random walks
over the free machine and schemas exercising every node kind, and every
finished walk parses and conforms. In the grammar layer
(`GrammarSpec::Mode::kJson`, the schema text riding the journal as
`gr.js`, `""` = json_object) thinking stays free but EOS is withheld
until the text is complete, `</think>` opens the body, the markers are
forbidden inside it (a `<think>` would be legal string content), and EOS
is admitted only when the machine is done. Structural whitespace is
capped at 16 consecutive bytes (`JsonLexer::kMaxWsRun`; a string's content
resets and never counts): without the cap a model whose mass sits on
masked tokens can spend its whole budget on the whitespace the grammar
always admits — seen on the service on 2026-09-04, 2,667 bytes of tabs
and newlines before a brace that never came — and past it only a
structural byte, or the end when the text is complete, remains; the mask
applies the same budget through per-token leading-whitespace buckets, and
the brute-force oracle covers it. The engines are unchanged: a JSON
grammar is another `GrammarState` behind the masks above, host and device,
the MTP row 1 under the pending draft, the unconstrained in-graph draft
rejected wherever the mask excludes it. The service compiles the
schema on rank 0: `strict: true` with a keyword outside the subset is a
400 naming `response_format.json_schema.schema.<path>`
(`unsupported_schema`); non-strict falls back to `json_object` with a
warning (OpenAI's non-strict mode promises no conformance); the prompt is
untouched; the content is the JSON text. Every rank rebuilds the same
machine from the same record and the same tokenizer.

## 11. Runtime and API

The daemon is a hand-rolled C++ service (`dgpp-serve`; `src/serve/`) with:

- `POST /v1/chat/completions` (stream and non-stream), `POST
  /v1/completions` (string prompt), `GET /v1/models`, `GET /health`,
  `GET /v1/metrics`;
- SSE streaming, disconnect → cancellation, deterministic (greedy) output,
  and a bounded request queue;
- tool calls and `reasoning_content` on the wire (M6 6f, 2026-09-04):
  the template's tool format parsed from the token ids on rank 0, the
  OpenAI shapes streamed and one-shot (as built, below).

The scheduler separates prefill and decode work, preserves identical rank
order, and admits requests only when weights, mutable state, DSA cache, MTP
scratch, network slabs, and snapshot copies fit. CUDA graphs are keyed by
batch/shape bucket and contain no allocation or host synchronization.

### As built (M6 Stages 2b, 4a, 4b)

**The scheduler** (`glm_scheduler.{hpp,cpp}`) is a pure-host policy
component — no CUDA, no bus, no model — over two seams: `SchedulerEngine`
(`prefill(req, prompt) → token`, `reserve(req, prompt + max_steps)`,
`step(req) → vector<token>`, `close(req)`; the pick is fused into the op
because at TP>1 the pick is a collective and the seam's call order IS the
collective order — the scheduler never sees logits) and the DSA pool meters.
The eager adapter owns one pending input token per slot and returns a
one-token vector; the graph adapter owns `[next,draft]` on the device and
returns one or two newly decided verify winners. EOS, scripted cancellation,
and the request cap are applied after each returned token, including the
prefill pick, and any suffix after retirement is dropped. Invariant: every
rank runs the same pure function of (requests, meters) — no clocks, no
unordered iteration, no thread arrival — so all ranks issue the same ops on
the same slots in the same order (the §5 rule, made mechanical). Policy,
decided with the user:
STRICT ALTERNATION (each tick admits at most ONE queued request, then runs
one engine pass over the next round-robin slice of up to
`decode_batch_capacity()` active requests — one for the scalar engines, so
their op stream is unchanged; a row-batched engine advertises its row
count and takes every active request in one pass — a mid-answer request
never waits behind a burst of read-ins); FCFS admission WITHOUT head-of-line blocking
(the oldest request that fits admits); FULL-RESERVE admission
(`blocks_for(prompt + max_steps)` held for the request's lifetime — no
mid-generation exhaustion, at the cost of over-reservation on early EOS);
external cancellation swept at FIXED TICK TOP; a queued head that can never
fit is refused at the door (`context_length_exceeded`), so the admission
deadlock is unreachable by construction. The isolation property is the
headline and is gate-pinned: a request's ids are identical whether it runs
alone or interleaved with and cancelled around others (`scheduler_test`
with a scripted fake engine that fails loudly on cross-request
contamination; the fabric smoke reproduced a solo transcript inside a
three-request manifest with a cancellation).

**The service** (`generation_service.{hpp,cpp}`, `http_server.{hpp,cpp}`):
two threads, one mutex. The HTTP thread (single-threaded epoll, zero
dependencies: keep-alive, chunked SSE, the limit ladder 431/413/503/501/
411/400, mark-and-sweep close) parses, tokenizes and renders — rank 0's
job only, never the fabric critical path — and pumps SSE; the engine
thread drains admissions → `try_submit`, drains cancels → `cancel`, runs
ONE tick, publishes meters. Records enter the table AT ENQUEUE so
pre-admission requests are visible to disconnect and to the pump. The
refusal ladder: every accepted field behaves per the schema; every
unimplemented one (user, store, metadata, service_tier, the audio and
prediction fields, the deprecated functions, and non-greedy sampling on
an engine that cannot sample) refuses with a
400 carrying the OpenAI error object naming the param — silent-ignore is
the bug class the ladder exists to prevent. `stop`, `n` and `logit_bias`
are served since 2026-09-06: `stop` (a string or up to four) is matched
on rank 0 over the message content (the legacy route's text) as the
tokens arrive — a scanner holds back a tail that could still begin a
stop string and cuts the text at the first match — and the request
retires through the journal like a cancel, at the next tick's sweep on
every rank with its own reason (`Reason::kStop`; the step in between is
run and its tokens dropped; `finish_reason` "stop", the usage counting
through the token that completed the match); `n` (1–8) admits one
scheduler request per choice — the same prompt, per-choice seeds
(seed + index), the prefix cache making every choice past the first an
attach — admitted or shed together, streamed by `index` with one
`[DONE]` and one summed usage chunk, answered as one `choices` array;
`logit_bias` (token id → [−100, 100], up to 1,024 entries, ids below the
vocabulary) rides the journal with the request and the engine adds it to
the logits after the penalties and before the mask and the temperature
— on the device in place (a per-slot dense row the pick kernels read
when the request's spec says so; a biased greedy row takes the full
path), on the host the same float in the same place, so the graph and
the eager engines agree bitwise (the loopback gates). The usage object
carries `prompt_tokens_details.cached_tokens` (the prefix cache's attach
position) and `completion_tokens_details.reasoning_tokens` (the ids the
parser routed to reasoning, `</think>` included). The sampling fields
(temperature, top_p, presence/frequency penalties, seed; top_k, min_p,
repetition_penalty as extensions) and `logprobs`/`top_logprobs` are
accepted since 2026-09-04 and behave exactly per §10, with omitted fields
taking the checkpoint's defaults. Incremental text is the suffix-diff of
successive full decodes, so UTF-8 and special-token boundaries are exact
without tokenizer state on the hot path.

**Tool calls and reasoning (M6 6f, built 2026-09-04;
`glm_tool_parser.{hpp,cpp}`, the chat route of `generation_service.cpp`).**
The chat template's rendering of an assistant turn IS the model's output
format — `<think>{reasoning}</think>{content}<tool_call>{name}<arg_key>{k}
</arg_key><arg_value>{v}</arg_value>…</tool_call>…` — and its markers are
ADDED TOKENS (`<think>` 154841, `</think>` 154842, `<tool_call>` 154843,
`</tool_call>` 154844, `<arg_key>`/`</arg_key>` 154847/8,
`<arg_value>`/`</arg_value>` 154849/50 for the pinned revision; looked up
by text at load, never assumed), one id each and never produced by BPE for
ordinary text. They are not "special" in the tokenizer's sense — `decode()`
prints them literally — so a text-keyed parser would be ambiguous where an
id-keyed one is exact. *Request side:* `tools` (flat or `{type:
"function", function}`), `tool_choice` (`auto`; `none` omits the tools
from the render and forbids `<tool_call>` in the grammar; `required` and
`{function: {name}}` ride the request as the grammar of §10's constrained
decoding — the prompt is untouched, the model reasons first, and then can
only write a valid call to an allowed function; an engine without masks
refuses them with `constrained_decoding_unsupported`),
`parallel_tool_calls` (`false` is the single-call grammar: one call, then
the turn ends), `reasoning_effort` (the OpenAI values; the template renders
`low`/`high`
and treats the rest as its maximum) and `chat_template_kwargs`
(`clear_thinking`, `reasoning_effort`; `enable_thinking` refuses with the
explanation that thinking is always on; any other key refuses by name).
Messages: system/user/assistant/tool (any other role refuses — the
template would render nothing for it); content as a string or the
template's content-part list; assistant `tool_calls` with `arguments` as
the OpenAI JSON-object string (parsed into the mapping the template
iterates, as vLLM's postprocess does) or an object; a null assistant
content with tool_calls becomes ""; `reasoning_content` and tool
`tool_call_id` pass through. *Response side:* one `ToolCallParser` per
chat request, fed each generated id on the engine thread under the
service's lock (pure host work on rank 0 — the fabric never sees it; the
journal and the op stream are unchanged). Its state machine starts in
Reasoning when the prompt ends in `<think>` (ids stream as
`reasoning_content` deltas until `</think>`), then Content (deltas as
`content`; a `<tool_call>` opens a block), then inside a block Name → Key
→ Value with the six markers structural and everything else text of the
current segment. A block is a call only when `</tool_call>` closes a
consistent name/(key, value)* sequence; a value without a key, a stray
marker, text between `</arg_key>` and `<arg_value>`, a nested
`<tool_call>` (which also starts a fresh block) or the generation ending
inside a block flushes the block's LITERAL text (the decode of its ids,
markers included) as content — nothing the model produced is lost and no
half-parsed call reaches a client. Values: the template writes strings
raw and everything else through `tojson`, an inverse that is lossy alone
(`"123"` and `123` render alike), so a value is typed from the request's
tool schema when it names the key (`parameters.properties[key].type`: a
string type keeps the text; a JSON type parses it, the text standing when
the parse fails) and otherwise "JSON if the whole text parses, else a
string" — vLLM's GLM parser makes the same call. Arguments re-serialize
in json.dumps form (`{"city": "Paris", "days": 3}`), the template's own
tojson dialect. Wire shapes per OpenAI: the message carries `content`
(null when the turn is calls only), `reasoning_content` when non-empty,
and `tool_calls: [{id: "call_<16 hex>", type: "function", function:
{name, arguments: <JSON string>}}]` (ids a pure function of the record and
the call index); streams carry `delta.reasoning_content` and
`delta.content` fragments as the runs' exact suffix diffs, and per call
one delta announcing `{index, id, type, function: {name, arguments: ""}}`
followed by one with the complete `arguments` string (a client that
concatenates fragments sees one fragment); `finish_reason` is
`"tool_calls"` when the turn ended naturally after at least one parsed
call, `"length"` at the cap whatever was parsed, `"stop"` otherwise (a
stop-string match included). The
final chunk carries the logprobs entries no content chunk took (the EOS
pick's, which decodes to nothing). `--reasoning-in-content` folds the
reasoning into content with the model's own `</think>` where it produced
it (the decode a client of a parser-less server would see) instead of
splitting it. Gates: `tool_parser_test` (unit, a fake decoder: the
split, exact deltas, schema-typed and inferred values, nested JSON,
several calls per turn, every malformed shape falling back to literal
content, missing markers disabling the features),
`chat_template_test`'s `glm_tool_call_render_encode_parse_roundTrip`
(every golden assistant tool-call message rendered by the template,
encoded by the real tokenizer and parsed back — 6 turns, 9 calls, names
and arguments structurally exact), and `serve_test`'s five 6f gates
(the request side's normalization and refusals by field name, the
one-shot message shape and `finish_reason`, the stream's chunk order, the
grammar arming the engine with the prompt untouched — the closed key set
from `additionalProperties: false`, the modes, the single-call forms — the
fold knob and the cap inside a block). On the four nodes
(`scripts/serve_tools_check.sh`, 2026-09-04, `--decode-graph --mtp` at the
card's settings) every request shape answered as designed — the
single-call grammars ending the turn after exactly one call while the
model's reasoning had planned two — with the op-stream md5 identical on
all ranks; the record's 2026-09-04 tool-call entries.

**The admission journal** (`fabric_serve.{hpp,cpp}`): rank 0 is the sole
ingress; every engine pass's scheduler-state changes ride ONE
newline-framed JSON record (`{"op":"tick","s":[{id,p,m,ca}…],"c":[…]}`)
broadcast over TCP after the drain and before the tick, so the record and
the tick are one atomic unit and tick counts are identical by
construction. Only state changes rank 0 MADE ride it — accepted submits
(with the prompt ids; rank 0 tokenized) and cancels that hit; door sheds
die on rank 0; steps, retires and tokens are derived state. Because every
engine op is a bus collective, a record cannot interleave a peer's
in-flight tick. Death discipline: journal EOF means rank 0 is gone and the
peers exit; a peer that stops reading makes rank 0's broadcast throw. The
metronome's idle chatter (200 noop records/s, ~3 KB/s per peer) buys zero
branching. Peers run `run_journal_peer` — no HTTP, no tokenizer. The §5
identity is checked live by the 4-way op-stream md5 after every run
(`fabric_serve_test` pins it with two real peer loops over localhost).
Bring-up lessons kept as code: the peer's journal connect retries inside
its window (a single-shot connect lost a 30 ms race to rank 0's bind);
minijson views its input, so a parsed record is copied out before the
buffer dies.

**Measured** (fabric TP=4, Stage 4b/4c, 2026-09-01; re-measured
2026-09-03): bus world in 2.4 s, journal peers in 30 ms, HTTP up with the
model — since the image cache a ready model is 15–25 s (§3; 24.4 s to
serving with the MTP layer); "What is the capital of Germany?" 32 tokens in
5.82 s warm at Stage 4c (~175 ms/token) and 1.80 s on 2026-09-03 (36.4
ms/token — the kernel rounds' gain on the same eager single-token step);
the fabric's distributed greedy pick produced the same token sequence as
w1's full-vocab argmax.

**The concurrency-1 graph engine** (`GlmGraphEngineAdapter`, M6.6a Phase 1):
`dgpp-serve --max-concurrency 1 --decode-graph [--mtp]` keeps prefill and its
first pick eager, records the T=1 or T=2 device-pick graph lazily before the
first decode, and replays it once per scheduler step. The scheduler's
explicit reserve call materializes its full-reserve policy in the DSA block
table before replay. Closing releases slot 0; a later admission eagerly
prefills and reseeds the graph's stable addresses, then reuses the same graph
era. Local gates run both T=1 and MTP through the real scheduler on a
two-rank loopback bus, compare against plain decode, check cross-rank
identity, and reuse the MTP graph for a second request with an eager
speculator stepped in lockstep (the device's `[next, draft]` feed after every
replay equals the eager `(next, draft)`, so the reused draft block is
checked, not only the transcript it cannot change). Measured on the
four-node service (2026-09-03, the record): 32.0–33.2 ms/token at T=1 and
43.6–44.1 ms per replay at 1.69–1.88 tokens per replay with MTP (21.8–26.0
ms/token, text-dependent acceptance), against `glm_gen_check`'s 31.3 and
42.36 — the scheduler tick, observer, journal record and SSE pump cost
under 1 ms per replay at T=1 and ~1.5 with MTP; the client sees one SSE
chunk per replay. Op-stream md5 identical on all four ranks of the eager,
T=1 and MTP worlds over the same 1144 tokens. Time to first token is the
eager prefill in every mode — measured 2026-09-04 at 25.5 / 19.4 / 13.2 /
9.7 ms per prompt token for 64 / 256 / 1024 / 2048-token prompts, each one
2048-token chunk, then 4.9 ms per token at 256 after the grouped MoE path
of the same day (PLAN's M6 tail; the record's twelfth and thirteenth
entries).

**The adaptive graph engine** (`GlmGraphEngineAdapter`, M6.6a Phase 2): the
adapter owns two execution shapes. It lazily captures one Phase-1 scalar graph
for every physical request slot and one fixed request-major row batch, T=1
without MTP and T=2 with it, subject to `requests * T <= kDecodeRows` (8).
The scheduler advertises the configured slot count so its canonical
round-robin slice contains every live request. Below four live requests the
adapter replays those requests' scalar variants sequentially; at four or more
it selects the batch and advances every live request in one replay. The
crossover defaults to four and is exposed as `--graph-batch-min-live`.

`CollectiveBus` supports up to 16 recorded variants inside one graph era.
Each owns a disjoint pinned generation-cell slab and immutable per-node
metadata. `graph_replay_arm(variant)` waits for the previous window using
that previous variant's generation count, resets the selected cells, reserves
the shared generation range, then release-publishes shape and window together.
`bus_test` alternates two variants with different node counts and widths.
The adapter serializes every capture through one reducer; variants 0..slots-1
are scalar and variant `slots` is the row batch, all recorded at startup
behind the journal's warm record (below).

The fixed batch remains fully request-indexed. Device position kernels turn a
closed slot into −1 padding; KDA, DSA, speculative snapshots, MTP hidden-cache
access, pick/verdict, commit, draft, and next-token feeds independently select
the request. The fabric latency slot is 64 KiB for all eight hidden-4096 rows.
Prefill and the initial draft stay eager between graph windows. Because those
paths reuse pinned row-map sources whose addresses the graph's kernel-upload
nodes retain (`glm_upload_i32/i64`; §6.2), the adapter restores the
immutable batch ids/spans before each batch replay and reseeds every live
device token feed after scalar work or an admission.

The first adaptive version restored throughput but failed transcript
isolation: scalar decode used the row-independent GEMV cores at m≤4 while an
eight-row graph selected cuBLASLt or the FP8 tile GEMM, so a request that
changed shape could flip a later near-tie token. The accepted implementation
lowers every decode shape up to eight rows through scalar-order GEMV chunks of
at most four rows (fewer when K reaches the 48-KiB shared-memory ceiling).
Every row retains the exact M=1 FMA/reduction chain while still amortizing a
weight read across the rows in its chunk. `bf16_gemv_test` and
`scale_gemm_test` pin M=8 versus eight M=1 calls bitwise for BF16 and FP8
weights, BF16 and FP32 outputs. The loopback serving gates explicitly cross
scalar→batch→scalar with noncontiguous MTP slots and compare independent
sessions/speculators after every replay.

Three follow-ups from the review of the closed phase (2026-09-03, after the
measured binary): padding rows are inert by construction — the DSA decode
path zeroes them as the KDA path always did, so a padded row's block output
and its share of the boundary all-reduce are deterministic without appeal to
downstream row independence; `dgpp-serve` warm-captures every variant at
startup (`warm_captures`: one throwaway session at a time, prefill → capture
→ close, 1.1 s for eight slots), so no record + instantiate lands on a live
stream and the warm sessions leave nothing behind (both loopback serving
gates run it first). The warm-up is a run of collectives, so it starts on
the journal's clock: a third record, `{"op":"warm"}`, broadcast by rank 0
once its model is built and held on by every peer before its warm-up — the
peers build their model ~8 s faster than rank 0 and otherwise spun their
first collective in stall diagnostics; a tick before the warm record, or a
warm record inside the serving loop, is a protocol violation and dies
loudly. The crossover flag defaults to min(4, slots) with out-of-range
values rejected rather than clamped. All three were validated on the four
nodes (record entry of 2026-09-03): curves within noise of the closure,
transcripts and op-stream md5s identical to it, no STALLED line on any
rank. The
per-user reading of the measured curve: 32/63/96/105 ms per token at 1/2/4/8
live T=1 requests and 26/51/66 at 1/2/4 with MTP — below the crossover a
user's latency is the live count times the scalar replay; at it the batch is
also the faster choice per user (four scalar replays would be ~129 and ~104
ms per token). The m ≤ 8 GEMV lowering also changed the path of 5–8-row
prefill chunks and per-expert prefill GEMMs with 5–8 routed tokens; the
prefill was then re-measured and taken down through five rounds
(2026-09-04/05, the record: 256 tokens 5 s → 0.58 s, 2048 tokens 19.8 →
1.7 s in steady state; the expert GEMMs run on a grouped tensor-core
kernel since round 5 — `MoeExpertKernel::kMma`, bitwise the scale GEMM's
tile kernel per segment, while the decode-class paths keep the GEMV core
and their bitwise pins — the attention prefill as a dense flash kernel
since round 6 (§7.2), and the projections, dense MLPs, router and mHC
dots restructured in round 7, most of them bitwise; the GPU is busy 98 %
of a prefill, so what remains is kernel work at or near its floors).

The original always-eight-row four-node gate measured
12.55/23.78/43.59/78.94 tok/s at 1/2/4/8 live T=1 requests and
19.32/35.90/65.79 at 1/2/4 live MTP requests. Same-binary scalar controls were
31.51 and 38.48 tok/s, making the one-live points 60.2% and 49.8% regressions.
The final adaptive gate on the same 43-token prompt and 256-token responses is
30.97/31.67/41.76/76.18 tok/s at T=1 and 38.48/38.97/60.27 with MTP. Thus low
occupancy retains scalar throughput, while row batching still wins 2.46× at
eight-way T=1 and 1.57× at four-way MTP over their low-occupancy plateaus.
The MTP tok/s numerator is acceptance-dependent. The Phase-2 prompt accepted
1.678 tokens/replay, so its unchanged 43.603 ms scalar replay reports 25.99
ms/token. Re-running the exact Phase-1 controls on the final adaptive binary
gave 21.93–22.21 ms/token for the 16-token Germany workload (21.79
previously), 21.83 on a 32-token repeat, and 24.58 for the 200-token CUDA-graph
explanation (24.81–24.85 previously). The 21-versus-26 ms comparison is
therefore a workload/acceptance comparison, not a graph execution regression.
All 22 full responses across both modes, every occupancy, and every graph
transition have one token hash. Each world's four op streams match, all ranks
shut down cleanly, and no current-run log contains a transport warning. Phase
2's performance gate is closed. Exact commands, latency distributions, and
artifact hashes are in the 2026-09-03 Phase-2 entries of
`benchmarks/results/2026-08-29-bus-m5.md`.

### Design for what remains (PLAN M6 6c/6d, M9)

- *Drain-on-stop (PLAN 6c):* BUILT 2026-09-04 — the signal is a flag the
  engine loop reads at the pass boundary; `begin_shutdown()` closes the
  door, sheds the queue and flags every live request; one more pass
  carries the cancels through the journal so every rank retires them at
  the same quantum with no engine op; the stop record goes out; the HTTP
  pump answers every interrupted stream (its tokens, the `server_shutdown`
  error event, `[DONE]`) and one-shot (503) before the server stops
  (`drained()`); a peer follows the journal to the stop record even on its
  own signal (a second one forces). The bus never comes down under a
  collective on either side.
- *Grow-on-demand admission (PLAN 6d):* BUILT 2026-09-04, opt-in
  (`--admission grow`) — `AdmissionPolicy` on the scheduler: reserve
  prompt + min(max_steps, window) at admission, grow at tick top before
  any engine op through the pool's transactional `ensure_request_blocks`
  (never during a replay), shed the youngest at exhaustion (Done /
  kPoolExhausted → finish_reason length). A pure function of scheduler
  state: rank-identical by construction, the growth in the op stream
  (`W`), the policy on the warm record. Preemption by recompute — the
  parity answer, never a truncated reply — waits on a fast prefill.
- *Tool calls and reasoning (PLAN 6f, 6g), `response_format` (6h), typed
  arguments (6i):* BUILT 2026-09-04 — the as-built paragraph above and
  §10's constrained decoding and JSON-constrained output. Three things
  the design had wrong, corrected in the build: the markers are not
  special tokens (decode prints them, which is why a malformed block can
  fall back to literal text); "JSON if it parses" alone is lossy, so
  values are typed from the tool schema first; and a forced
  `<tool_call>` prefix on the prompt is not `tool_choice` (it skips the
  reasoning and cannot stop a second call) — the grammar mask is.
- *Failure semantics (v1) — as built (2026-09-05):* no failover. Any
  rank's death fails the world legibly and fast: the journal sockets are
  the liveness probe (a process death closes them), rank 0's watch on the
  peers' connections and each peer's in-tick watch on rank 0's see it
  within a poll, and the bus watchdog remains the backstop for a silent
  node. Rank 0 answers every live stream with its committed tokens and
  the `engine_failure` event and exits with status 2; a peer stuck inside
  a tick exits with status 3; `serve_run.sh` (or a supervisor) restarts
  the world in ~25 s from the image cache. The step in flight never
  completes on any rank, so committed state is identical everywhere and a
  restart reproduces the committed tokens (drilled on the four nodes,
  `scripts/serve_failure_drill.sh`). The 4-way md5 IS continuous: every
  tick record carries rank 0's running op-stream fold (`od`, FNV-1a over
  the OpStreamObserver's lines) and, with the prefix cache on, its
  decision digest (`pd`); a differing peer dies with the tick number, one
  tick late at most, and rank 0 then fails the service as for any dead
  peer. `docs/operations.md` is the operator's page.

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
certification path can reject it. Layered the other way: a tolerance kept
as a fallback for rows the audit lacks inputs for must never veto a
certification — a certified near tie moves the row output O(1) by design
(near-tie scores mean the boundary is tied, not that the content is
similar), and the M4 close-out fixed exactly this inversion in the
chunked-prefill test, where a drift budget silently overroved the audit
(one certified 3.1x-noise near-tie at select_k=8 was rejected as
"flipped-row drift 1.0066" while the CI config sat one seed away from the
same failure at 0.6798).
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

A third class, found by the M4 close-out's full-suite memcheck sweep: a CUDA
API call whose failure the code deliberately swallows and clears (the M1
rmsnorm's dynamic-smem opt-in probing the device cap, rejected on GB10 for
that kernel's footprint) is still a sanitizer finding — the tool reports the
error return regardless of the application's handling. Never make a call
that can fail: query the driver-computed per-kernel ceiling
(`cudaFuncAttributes::maxDynamicSharedSizeBytes`) and request within it.

A fourth class no sanitizer sees, found 2026-09-03 with an nsys node trace
(`docs/batched_mtp_graph_stall.md`): a graph's memset/memcpy nodes ride the
process-shared, in-order copy-engine queue, and a queued node's dependency
wait blocks the peer rank's node behind it — a deadlock that needs two
ranks in one process and a kernel that waits on the peer, which is exactly
the loopback gates. The rule it left is structural, not a timing knob: the
captured decode graph is kernels-only, checked at every capture site
(§6.2); a spinning kernel never blocks another stream's kernels, so with
that contract the worlds pass at one hardware connection. The trace, not
reasoning, settled it — the first diagnosis (compute work queues, a
non-resident collective as the remedy) had the right shape on the wrong
queue and would not have fixed it.

Sanitizer scope is chosen per phase, deliberately: full-suite memcheck for
orchestration code (pointer/size bugs — it caught two undersized test
buffers that made a graph test pass vacuously), targeted racecheck/initcheck
for new kernel shapes. The loopback worlds under memcheck need their time
budgets lifted (`DGPP_TEST_BUS_TIMEOUT_MS`, `DGPP_TEST_CONSUMER_DEADLINE_S`,
`DGPP_TEST_WAIT_TIMEOUT_MS` — the engine watchdog, the kernel deadline and
every host-side wait: reducers, picks, replay finishes) — instrumented
kernels outlast the release budgets by orders of magnitude, on the baseline
as on any change. Instrumenting vendor kernels (cuBLASLt's cutlass
implementations dominate long benchmarks) buys no coverage of our code and
costs orders of magnitude in wall time; the decision and reasoning are
recorded with the results.

Measured numbers include the exact command, binary revision, driver, firmware,
clock/power context, run count, and variability. `micro_gemm_peak` is a
cuBLASLt heuristic sweep, not a proof of hardware peak. Network claims use
perftest as the authoritative throughput source and the project benchmark for
protocol validation.

**Cross-build numerics gates (since the reassociation rule, §1):** the
transcript md5 is the regression signal only for changes that claim to be
bit-identical (and for MTP against plain, §9). For rounding-level changes
the judges are `scripts/fabric_xcript.py REF NEW` — the first divergent
token, judged by the global top-2 margin in bf16 ulps of the winning logit
(a flip within 1–2 ulps is a coin the hidden state's last rounding turned;
the reference transcript has 14 of 300 picks decided by ≤ 1 ulp) — and
`scripts/fabric_logprob.py NEW REF` over a teacher-forced run
(`glm_gen_check --teacher-file`): per-token log p from the joined
vocabulary slices, perplexity, the mean-NLL delta with a standard error
and a PASS/FAIL at 0.02 nat. Three texts ship (556 tokens of prose; 7,331
of unmemorized technical prose at perplexity ~10.8 — the sensitive one;
6,549 of Conan Doyle at 1.04). The same binary twice must show a delta of
exactly 0: the decode path is deterministic end to end. Fabric runs are
correlated across ranks by BUS GENERATION, never by wall clock (the boxes
disagree by hours; `scripts/fabric_xrank.py`).

## 13. Repository layout

Current source tree:

```text
apps/                 dgpp-serve (the service), glm_gen_check (generation,
                      teacher forcing, --decode-graph, --mtp, --requests),
                      glm_tp_check (fabric parity), glm_forward_check
                      (curated suite, traces), glm_shard_parity,
                      glm_stream_check, glm_bind_check, bus_check,
                      bus_small_repro (the pick racer), nic_regress,
                      roster_check, dgppctl, gpt_doll
benchmarks/micro/     platform and transport probes (incl. kda_bench, dsa_bench)
benchmarks/results/   the dated records; 2026-08-29-bus-m5.md carries M5–M8
benchmarks/           teacher texts, the curated glm-suite
docs/                 generated checkpoint budget and validated measurements
scripts/              fabric_run.sh (the fabric launcher), serve_run.sh
                      (up/down/status), fabric_xcript.py / fabric_logprob.py
                      / fabric_xrank.py / fabric_logs.py (the judges and the
                      cross-rank reader), node_probe.sh, soak.sh, ci-local.sh
src/common/           logging, dtypes, process memory (mlock), tests
src/core/             arena, graph, streams, trace
src/kernels/          GEMM wrapper and scale-aware GEMM, bf16/fp8 GEMV cores,
                      KDA and DSA ops, mHC, MoE (router, slot GEMVs,
                      accumulation), norms, L2 prefetcher, the pick and spec
                      (commit/positions/draft rows) kernels, flag protocol
src/loaders/          JSON, safetensors, shardspec, the HF hub cache resolver
src/models/           GLM config/binding/loader/resident image; the model
                      (glm_forward, glm_decode sessions, glm_mtp, glm_tp views
                      and bus reducers, glm_tp_bus picker); KDA and DSA
                      layers/state/references/dumps; mHC and MoE layers and
                      oracles; tokenizer, chat template, sampler, scheduler,
                      speculative judge, engine adapters, step timing
src/net/              TCP primitives, the epoch-based roster, the CollectiveBus
                      (verbs RC QPs, slot pools, credits, the engine loop, the
                      collective kernels: latency one-shot, graph, bulk RS+AG)
src/serve/          HTTP server, generation service, fabric serve (journal)
tests/                host, CUDA, and Python tests (32 CTest entries)
tools/                checkpoint audit, shard plan, reference-dump generators,
                      tokenizer/template golden generators, unicode tables
```

## 14. Future validation scope

The recorded evidence satisfies the exit criteria of M0–M5 and M8 and the
built parts of M6 (PLAN), including scalar and row-batched service
latency/throughput on a committed workload. It does not replace the
measurements still owed: 32K-context TTFT, the 24-hour serving soak, and the
failure drills (PLAN M9). If fabric runs develop drops, retries, unstable
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
