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

### 5.2 Weight placement

- routed experts: 72 whole experts per rank initially;
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
ranges, contiguous whole-expert ranges, 128-aligned quantized row/column
slices — so only rank-local checkpoint bytes are ever read (measured on
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

The initial implementation uses all-reduce, not an incorrectly named
three-destination “tree.” Algorithm selection is size-based:

- small replicated hidden vectors: latency-oriented recursive doubling/tree;
- large prefill chunks: striped reduce-scatter + all-gather or ring all-reduce;
- control messages: TCP, never on the CUDA critical path.

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

**Graph capture (d3): the decode step's fixed launch sequence records once
and replays per token.** The collective kernel becomes a graph node like
any other — the engine no longer launches it and instead *reacts*. The
seams that make replay-stable what capture bakes:

- *Per-generation cells.* One pinned 64B ctl per recorded collective node
  (vs. the eager machine's single one-flight cell). The arm step — before
  each replay — resets them and assigns the window's monotonic
  generations, `gen_seq` published release-last; the replayed kernel
  acquires it at start and derives its staging row `(g−1)%ring` and its
  exit match from it. Monotonicity means a previous replay's stale
  `done_seq` can never match, so replays of one node are distinct without
  re-recording.
- *The engine generation walk.* `window_count` (one atomic; first/last
  derive arithmetically) publishes the window; the engine adopts, posts
  each generation's peer pairs when the kernel's `ready_bits` land (lane 0,
  the cursor ring position — deterministic under the era's exclusivity:
  every latency post in the era is a graph generation, in order), and
  advances on `done_seq == gen`. Stream order serializes the recorded
  kernels, so at most one generation is un-done at a time — a single-flight
  state machine, not a queue. Arm waits for the previous window's walk
  (bounded); finish joins the walk and returns the verdict.
- *The era.* One graph per bus (the decode step shape is fixed); harness
  sends close and eager collectives reject from `record_begin` — the
  recorded kernels claim doorbells exactly like eager collectives, and the
  ring positions must stay generation-ordered.
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
  payload slots at kernel exit; acking at claim time returns the
  sender's credit early, and a sender a segment ahead wraps the
  8-deep ring inside one claim-to-exit gap, DMAing new stripes over a
  fold input. Acking after the fold makes the credit gate close the
  race by construction — and the scan must skip the kernel's own claims
  meanwhile, or a re-presented claim steals the shared CAS slot from a
  fresh doorbell in the same warp, forever.

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
same fixes took the harness send path from 16.5 to 12.1 µs p50.

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
compressed E4M3+scale weights (§4): gather per-expert token segments,
gate/up GEMMs, swiglu, down GEMM, accumulate. The M4 diagnostic path
segments on the host after one router round-trip (correctness first); the
device-side grouped execution without host segmentation is M5+ work.

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

`GlmDiagnosticModel` (M4 deliverable 1) runs the full text stack over the
streaming resident loader: embedding -> 4-stream mHC init -> per layer
[attn_hc -> ln1 -> KDA or DSA -> stream update] and [ffn_hc -> ln2 ->
dense MLP or MoE -> update] -> unweighted stream mean -> final norm ->
bf16 lm head. One layer is resident at a time; the KDA/DSA/MoE layer
objects are constructed once (shape-keyed scratch and GEMM plans) and
REBOUND to each layer's resident views. The two layer norms and the
final norm are the Glm5NextTextRMSNorm TWO-rounding choreography
(u = bf16(x*rsqrt); y = bf16(w*u)) — a dedicated kernel, because the
generic single-round rmsnorm drifts the GLM path systematically.
The MTP draft layer (index 45) is bound and loadable but outside the
forward, matching the reference model class, which ignores it; its
transaction model is §9 (M5+).

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
  and `models/glm_route_audit.hpp` requires (a) the engine's selection to
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
src/net/              M5 control + data planes: TCP primitives, the
                      epoch-based roster (startup, health, eviction), and
                      the CollectiveBus — verbs RC QPs, per-class slot
                      pools with ring discipline, credit-grant RDMA writes,
                      the single-threaded engine loop, and the
                      watchdog-bounded receive consumers
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
