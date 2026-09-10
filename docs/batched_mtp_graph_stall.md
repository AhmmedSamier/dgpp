# Batched-MTP CUDA graph stall: investigation handoff and resolution

Date: 2026-09-03 (handoff); resolved 2026-09-03 (evening)

Status: RESOLVED. Root cause found with an nsys node-level trace and confirmed
with a synthetic reproducer; the runtime fix (a kernels-only decode graph,
enforced at every capture site) is in the tree; the CTest prefetch-off
mitigation is removed; the stress gates below pass with prefetch enabled at
`CUDA_DEVICE_MAX_CONNECTIONS=32` and at `1`.

## Resolution

### Root cause

The captured MTP decode graph contained five non-kernel nodes: two 8-byte
host-to-device memcpy nodes at the top (the request-id and span table
uploads from `session_decode_host_prep` / the batch capture) and one 4-byte
memset node per DSA decode layer (`dsa_select_decode`'s selection-counter
reset, `cudaMemsetAsync`). Other captures carried more of the same class:
the plain T=1 graph's token-feed memcpy node, a host-positions capture's
step-position upload, and the tail's logits/hidden device-to-host mirror
nodes in any capture that left the mirrors on (the direct device-pick
test). Memset and memcpy nodes execute on the copy engine, whose queue is
in-order and shared by every stream in the process; a queued node's
dependency wait blocks everything queued behind it. Kernel nodes have no
such queue: a resident, spinning kernel never blocks another stream's
kernels (measured below, even with one hardware connection).

At graph launch the runtime queues the whole replay, including every
copy-engine node with its dependency wait. In the one-process two-rank world
that produced this cycle:

```text
rank B's replay queued its layer-4 DSA memset, waiting (at the CE queue head)
    on B's layer-2 attention collective
        which spins waiting on rank A's matching collective
            which waits on A's layer-2 DSA select kernel
                which waits on A's layer-2 DSA memset
                    which is queued BEHIND B's waiting memset
```

Why the evidence looked the way it did:

- every observed stall was an attention-side boundary (cells 0 and 4): the
  memset precedes the DSA layer's select kernel, and the two memcpy nodes
  precede everything, so the first collective the stalled rank cannot reach
  is the one after its first queued copy-engine node;
- the stalled rank's collective kernel never started (the new `e=` entry
  flag in the stall dump reads 0), because its chain was stuck one node
  earlier, at the memset;
- prefetch mattered because it changes the executor's lane assignment and
  launch timing (both ranks' whole-graph copy-engine queueing must
  interleave the wrong way), not because the prefetch branches deadlock
  anything themselves; the connection count changes how often the two
  ranks' copy-engine work shares one queue;
- first replays of an exec were over-represented because a first launch
  skews the two ranks' submission order.

### Evidence

1. Host side: with the process run as gdb's child and interrupted during the
   stall, both rank threads sit in `cudaStreamSynchronize` after
   `cudaGraphLaunch` returned (`GlmGraphEngineAdapter::replay`); the engine
   threads poll. No host call is blocked — the stall is on the device.
2. Device side: an nsys trace (`nsys profile -t cuda --cuda-graph-trace=node`)
   of a stalled run (generation 458, cell 4 of slot 3's scalar variant on its
   first replay). On the stalled rank every kernel before the collective
   finished by +288.7 us after launch; the next node in its chain, the
   4-byte memset (node 7068), executed at +5,000,410 us — 992 ns after the
   other rank's *layer-4* memset (node 4441) ran, which was released only
   when the watchdog poisoned that rank's spinning layer-2 collective. The
   spinning collective itself is the only kernel with a 5 s duration.
3. Synthetic reproducer (`apps/graph_queue_repro.cu`, two ranks in one
   context, the same main-chain + one side-chain capture shape, a resident
   spinning collective per boundary):

   | graph contents | connections | result |
   |---|---:|---|
   | kernels only, side chain on | 1, 8, 32 | 100–300 replays OK each |
   | kernels only, side chain off | 1, 8, 32 | OK |
   | + one 4-byte memset node per second collective | 32 | 200 OK |
   | + one 4-byte memset node per second collective | 1 | STALL on replay 1 |
   | + a synchronous legacy-stream `cudaMemset` between replays | 32 | STALL on replay 1 |

   The last row is a separate hazard worth knowing: the model stream is a
   blocking stream, so any synchronous runtime call on one rank's thread
   waits for the peer's in-flight graph. The decode path has none; the
   reproducer's own first cut tripped it.
4. The same mechanism through a different node: with the MTP fix in place,
   `glm_tp_device_pick_graph_loopback_matches_host_pick` (world 4) still
   deadlocked 10/10 at `CUDA_DEVICE_MAX_CONNECTIONS=1` (0/20 at 32). Its
   graph kept the tail's two device-to-host mirror nodes (the test set only
   the route traces off). The slower rank's eager token upload between
   replays queued behind the faster rank's in-flight mirror node; with the
   mirrors off it passes at 1 connection. An eager copy is safe exactly when
   no rank's in-flight graph carries a copy-engine node — which the
   kernels-only contract guarantees.

### Fix (in the tree)

- `src/kernels/dsa.cu`: the select counter reset is `select_counter_reset_kernel`.
- `src/kernels/glm_spec.{hpp,cu}`: `glm_upload_i32`/`glm_upload_i64` upload
  a small pinned table with a kernel (system-scope loads of the device-mapped
  host buffer); `glm_decode.cpp` uses them for the request-id and span
  tables (scalar and batch captures), the decode rows' token feed and the
  host-positions path; `glm_forward.cpp` allocates all four upload sources
  `cudaHostAllocMapped` and checks the UVA address identity.
- `set_decode_tail_mirrors(false)` now gates only CAPTURED mirror nodes:
  eager steps and eager drafts always mirror (they sync anyway), and
  `session_graph_outputs` copies a replay's tail eagerly when the mirrors
  are off instead of refusing. The direct device-pick test turns them off.
- `src/models/glm_graph_check.hpp`: `glm_check_decode_graph` logs the
  node-type histogram and throws on any memcpy/memset/host/other node. Called
  by `GlmGraphEngineAdapter::capture_variant`, both `glm_gen_check` captures,
  and the two direct captures in `glm_tp_test`.
- `CMakeLists.txt`: `DGPP_L2_PREFETCH=off` removed from `glm_tp_test`'s
  environment; `CUDA_DEVICE_MAX_CONNECTIONS=32` stays.
- `src/net/collective_bus.cpp`: the stall dump prints `e=` (kernel entry
  seen) per cell, distinguishing "never started" from "wedged".
- `apps/graph_queue_repro.cu` (target `graph_queue_repro`): the synthetic
  two-rank model above, kept as the diagnostic for this hazard class.
- `tests/cuda/glm_tp_test.cpp`: `DGPP_TEST_CONSUMER_DEADLINE_S` overrides
  the loopback worlds' 20 s kernel-side collective deadline and
  `DGPP_TEST_WAIT_TIMEOUT_MS` every host-side wait (boundary reducers,
  picks, replay finishes, the adapter's pick timeout; 60 s in release) the
  way `DGPP_TEST_BUS_TIMEOUT_MS` overrides the engine's — compute-sanitizer
  memcheck slows the instrumented worlds past every release budget.

The graphs are unchanged in node count (a kernel per replaced node) and the
numerics are untouched: the fixture transcripts and cross-rank parity gates
pass as before. Fabric ranks are one process each and never shared the
queue; the contract costs them one 32-thread launch per DSA layer and two
per step.

### Validation

All on the final tree, prefetch enabled (`DGPP_L2_PREFETCH` unset), the
fixture worlds of `glm_tp_test`, 2026-09-03 evening. A stall shows as an
8+ s run (5 s watchdog); every run below finished in under 4 s.

| gate | connections | runs | failures |
|---|---:|---:|---:|
| `glm_tp_serving_mtp_batched_graph_matches_independent_speculators` | 32 | 100 | 0 |
| same | 1 | 100 | 0 |
| `glm_tp_device_pick_graph_loopback_matches_host_pick` (world 4) | 32 / 1 | 20 / 10 | 0 / 0 |
| `glm_tp_full_graph_step_loopback_matches_eager_speculator` (world 4) | 32 / 1 | 20 / 10 | 0 / 0 |
| `glm_tp_serving_plain_batched_graph_matches_independent_sessions` | 32 / 1 | 20 / 10 | 0 / 0 |
| `glm_tp_serving_graph_adapter_matches_plain_and_reuses_slot` | 32 / 1 | 20 / 10 | 0 / 0 |
| full CTest suite (`ctest -j4`), 33 entries | 32 | 3 passes | 0 |

Before the fix on the same box: the MTP test failed on run 3, 4, 3 and 1 of
four independent loops at 32 connections; the device-pick world-4 test
failed 10/10 at 1 connection with its mirror nodes present.

`apps/graph_queue_repro` controls on the same session: kernels-only graphs
100/100 replays at 1 connection with and without the side chain; one
memset node per second collective stalls replay 1 at 1 connection and
passes 100/100 at 32.

compute-sanitizer memcheck on the MTP and the world-4 device-pick tests:
zero memcheck errors in every run — the full-instrumentation runs on both
tests, targeted runs instrumenting only the bus, upload, select, pick, spec
and prefetch kernels (the new `upload_words_kernel` and
`select_counter_reset_kernel` included), the unmodified baseline commit,
and `graph_queue_repro`, which completes under memcheck at 32 connections
(0.6 s for three replays) but hangs under it at one connection (no replay in
8 minutes) — the tool's kernel scheduling does starve a spinning peer once
the streams share a single queue, which is one more reason the worlds are
run at 32. The loopback worlds
themselves still exit nonzero under memcheck with every budget lifted
(`DGPP_TEST_BUS_TIMEOUT_MS=120000 DGPP_TEST_CONSUMER_DEADLINE_S=120
DGPP_TEST_WAIT_TIMEOUT_MS=600000` — the last knob now covers the boundary
reducers, picks, replay finishes and the adapter's pick timeout): the first
eager prefill collective exits on its 120 s kernel deadline after 233 s
(world 2) / 462 s (world 4) of instrumented execution, identically on the
baseline commit at its 20 s deadline. A pre-existing harness limitation,
not a change in behavior; the next step, if memcheck of these worlds is
wanted end to end, is excluding the loader/dequant kernels from
instrumentation to see what the instrumented prefill spends >120 s on.

Numerics: every gate above compares transcripts against independent eager
sessions and across ranks bitwise; all pass unchanged.

Fabric (2026-09-04, TP=4, `docs/mtp.md`'s 300-step Roman Republic recipe, the
record entry of that date): eager 36.07 ms/step (ledger 36.4), T=1 graph
31.67 (record 31.3), MTP graph 42.44 per step (record 42.36–42.42); plain vs
both graphs IDENTICAL over 300 steps with one generated-ids md5 on all four
ranks. The kernels-only graph costs nothing measurable per step.

### What the handoff got right and wrong

The diagnosis "a required GPU node fails to run while its peer's node
remains resident waiting for it, through a dependency CUDA cannot see" was
right, and so was the instinct that hardware-queue head-of-line blocking
was involved. The queue is the copy engine's, not the compute work queues,
and the blocked node is a memset, not a kernel. The recommended
non-resident WHILE-node collective would not have fixed this: the memset
would have queued behind the peer's dependency wait exactly as before, and
the conditional node's own downstream wait would have joined the cycle. No
change to the collective kernel was needed.

---

## Original handoff (kept for the record; superseded above)


## Executive summary

`glm_tp_serving_mtp_batched_graph_matches_independent_speculators`
intermittently stalls in a graph collective. One local rank starts the
collective kernel, stages its row, and has its send posted. The peer rank's
matching collective kernel never reaches its first `ready_bits` publication.
The first kernel remains resident waiting for the peer's doorbell until the
five-second bus watchdog fails the graph era.

The sampling-profile implementation did not cause this. The exact failure
reproduces from detached, unmodified commit `8c181e4`, before any profiling
changes, while running only the MTP graph test.

The leading diagnosis is an in-process CUDA scheduling deadlock. The test puts
multiple logical ranks, their model streams, their captured L2-prefetch side
streams, and their bus engines in one CUDA context. The graph collective is a
long-running polling kernel with a dependency on a peer kernel in another
stream. That dependency is carried through host/NIC-visible cells and is not
visible to CUDA's scheduler. With both classes of captured prefetch branch
present, CUDA can apparently leave the peer work behind the kernel that is
waiting for it. This diagnosis is strongly supported by the controls below,
but the exact hardware work-queue assignment has not been captured with a
scheduler trace.

The current mitigation adds `DGPP_L2_PREFETCH=off` to the `glm_tp_test` CTest
environment. It makes CI reliable and does not alter fabric execution, where
each rank is a separate process. It is deliberately not considered the
runtime fix requested for this bug.

## Reproduction

Hardware and toolchain used for this investigation:

- NVIDIA GB10, compute capability 12.1
- driver 580.173.02
- CUDA toolkit/compiler 13.0.88
- `RelWithDebInfo`, `DGPP_WERROR=ON`, architecture 75 in the diagnostic build
  used to match the existing `build-ci` configuration

Build and run the failing case directly. Do not use CTest for the reproducer:
CTest now supplies the prefetch-off mitigation. Run repetitions sequentially;
the test uses a fixed loopback rendezvous port and concurrent repetitions
would also distort GPU scheduling.

```bash
cmake --build build-ci --target glm_tp_test -j8

for i in $(seq 1 100); do
  env -u DGPP_L2_PREFETCH \
      -u DGPP_L2_PREFETCH_BOUNDARY \
      -u DGPP_L2_PREFETCH_LAYER \
      CUDA_DEVICE_MAX_CONNECTIONS=32 \
      DGPP_TEST_FILTER=glm_tp_serving_mtp_batched_graph_matches_independent_speculators \
      DGPP_LOG_LEVEL=warn \
      ./build-ci/glm_tp_test || break
done
```

A passing invocation takes roughly three seconds on the test machine. A stall
takes roughly eight seconds because the graph makes no progress for five
seconds before the watchdog fires. Typical terminal output is:

```text
ERROR bus graph era failed: graph generation 195 stalled
      (no engine progress for 5000 ms)
[FAIL] glm_tp_serving_mtp_batched_graph_matches_independent_speculators:
       rank 0: graph engine replay finish: graph era failed
```

Use `DGPP_LOG_LEVEL=info` to obtain the per-cell and lane snapshots. A captured
failure is summarized below.

## Evidence

### The profiling code is not causal

An independent worktree at commit `8c181e4` was configured and built from
scratch. This commit predates all sampling-profile changes. The filtered MTP
test passed twice and then failed on invocation 3 at generation 195 with the
same watchdog message.

The current executable also reproduces while
`DGPP_TEST_FILTER=glm_tp_serving_mtp_batched_graph_matches_independent_speculators`
is set. The new sampling-profile test is therefore not executed, and this MTP
test does not call the sampling-profile helper or enable its CLI flag. Binary
layout could change the probability of exposing the pre-existing scheduling
bug, but it is not its origin.

### The stalled state is asymmetric at the collective handoff

One full-info failure stalled at generation 329, the first collective of the
first scalar replay after successful batched replays. The two bus engines
repeated the following state, unchanged, until the watchdog fired:

```text
rank-side A: gen=329 posted=0x0 cell ready=0 done=0
             inbound lane door/ack = 42/41
rank-side B: gen=329 posted=0x1 cell ready=1 done=0
             outbound send remains in flight
```

Interpretation:

1. Side B's collective kernel ran its snapshot phase and published
   `ready_bits=1`.
2. Side B's CPU bus engine observed that publication and posted the payload and
   doorbell.
3. Side A received the doorbell (`42/41` means an unconsumed arrival), but its
   matching graph kernel never published readiness and therefore never entered
   the receive scan that would acknowledge the arrival.
4. Side B's kernel continued polling for A, while A's corresponding kernel did
   not begin. There was no changing cell, doorbell, ACK, or posted state during
   the five-second interval.

This is not the signature of a lost packet, a bad generation value, or a graph
walk mapping error. It is the signature of one required GPU node failing to
run while its peer's node remains resident waiting for it.

Stalls were seen in more than one position:

- generation 195: first batched-graph replay after warm capture and prefill;
- generation 329: first scalar replay after the batch-to-scalar transition;
- generation 333: four collectives into that scalar replay when explicit graph
  upload was being tested.

Consequently, the bug is not specific to the adaptive variant transition or
only to the first node of a lazily uploaded graph.

### Prefetch controls

All controls below used the untouched `8c181e4` build except where noted:

| Configuration | Result |
|---|---:|
| all prefetch enabled, 32 CUDA connections | failed by run 3 |
| `DGPP_L2_PREFETCH=off` | 30/30 passed |
| boundary and intra-layer rates both `off`, side streams still created | 20/20 passed |
| only boundary prefetch `off` | 20/20 passed |
| only intra-layer prefetch `off` | 20/20 passed |
| current profiling branch, all prefetch disabled | 30/30 passed |

The individual-rate samples are not large enough to prove that either branch
is independently safe. They do show that merely constructing the side streams
is not sufficient to reproduce the failure, and they implicate the combined
captured prefetch work/graph scheduling pressure.

The final mitigated branch also passed the complete CTest suite: 33/33.

## Relevant execution path

The serving adapter's replay sequence is in
`src/models/glm_fabric_engine.hpp`:

1. `CollectiveBus::graph_replay_arm()` assigns and resets generation cells.
2. `cudaGraphLaunch()` submits the chosen scalar or batched graph.
3. `cudaStreamSynchronize()` waits for model and collective nodes.
4. `CollectiveBus::graph_replay_finish()` waits for the CPU engine's graph
   walk to reach the end of the window.

The captured collective is implemented by
`bus_allreduce_graph_kernel()` in `src/net/bus_kernel.cu`. A single launch does
all of the following:

1. copies the local source to the generation's pinned staging row;
2. publishes `ready_bits` for the CPU bus engine;
3. remains resident and polls every peer's NIC-written doorbell;
4. validates payload placement hashes;
5. publishes receive ACKs;
6. folds vectors in canonical rank order and publishes `done_seq`.

The CPU-side walk in `src/net/collective_bus.cpp` observes `ready_bits`, posts
the outbound RDMA pair, and waits for `done_seq`. Its own five-second
no-progress watchdog produces the reported error.

`WeightPrefetcher` in `src/kernels/l2_prefetch.cu` creates a low-priority side
stream and captures fork/join branches around both collective boundaries and
intra-layer latency regions. `GlmDiagnosticModel` creates its main chain stream
at high priority in `src/models/glm_forward.cpp`.

The unsafe scheduling dependency can therefore be written as:

```text
rank B graph collective (resident, polling)
    waits for rank A's NIC doorbell
        waits for rank A graph collective to execute its staging phase
            rank A graph node is not scheduled
```

CUDA does not see the middle dependency because it crosses the GPU/CPU/NIC
protocol through mapped control cells. The runtime can legally make a
scheduling choice that is incompatible with the polling kernel's assumed
forward progress.

## Hypotheses tested and rejected

### Added sampling profiling

Rejected by the detached pre-profile baseline reproduction and by the filtered
test path described above.

### Too few CUDA connections as a complete fix

`CUDA_DEVICE_MAX_CONNECTIONS=32` reduces the older world-4 exposure but does
not eliminate the current world-2 MTP stall. It is useful as a mitigation, not
a correctness contract.

### Lost or mismatched transport notification

The stalled peer has a live, generation-correct inbound doorbell while its
local generation cell remains at `ready=0`. The notification arrived; the GPU
consumer node did not start.

### Adaptive batch-to-scalar transition bug

One detailed failure occurred at that transition, but failures also occur on
the first batched graph and several collectives into a scalar graph. Generation
cells and the selected variant were correct in the snapshots.

### CUDA graph node priorities

An experiment instantiated the graph with
`cudaGraphInstantiateFlagUseNodePriority`, preserving the high-priority main
chain and low-priority captured prefetch nodes. It passed 20 repetitions, then
failed on repetition 21 at generation 195. The patch was reverted. Priorities
can affect probability but do not create a scheduler-visible dependency or a
forward-progress guarantee.

### Lazy first-launch graph upload

An experiment called `cudaGraphUpload()` and synchronized immediately after
each graph instantiation, while no replay window was armed. Its first stress
invocation failed at generation 333. The patch was reverted.

## Current mitigation and repository state

`CMakeLists.txt` currently configures `glm_tp_test` with:

```cmake
ENVIRONMENT
"CUDA_DEVICE_MAX_CONNECTIONS=32;DGPP_L2_PREFETCH=off"
```

`README.md`, `PLAN.md`, and `DESIGN.md` describe this as an in-process test
mitigation. Fabric still uses the default, enabled prefetcher. The mitigation
should be removed after a runtime fix survives the stress gates below.

No experimental runtime change remains in the tree. In particular,
`src/models/glm_fabric_engine.hpp` contains neither the node-priority experiment
nor the explicit-upload experiment. The other uncommitted source changes are
the sampling-profile implementation that was already in progress; they should
not be discarded while fixing this bug.

## Recommended runtime-fix direction

The graph collective must stop assuming that a resident polling kernel can
wait for a peer kernel in another stream/context sharing the same device.
Increasing queue count, changing priority, or reducing unrelated graph work
can only change the probability.

The most direct design for the current CUDA 13 target is a non-resident graph
polling state machine:

1. Split the graph collective's staging phase into a finite kernel that copies
   the source, publishes `ready_bits`, and returns.
2. Replace the resident receive loop with a CUDA Graph conditional `WHILE`
   node whose body is a finite poll kernel.
3. Each poll iteration either:
   - observes all generation-correct doorbells, validates every placement
     hash, writes ACKs, performs the canonical fold, publishes `done_seq`, and
     clears the conditional handle; or
   - observes that work is not ready, checks poison/deadline state, leaves the
     handle set, and returns promptly so other streams can make progress.
4. Preserve the existing graph-cell addresses and per-replay `gen_seq` values;
   each recorded collective still owns one stable cell.

CUDA's official graph documentation describes conditional `WHILE` nodes and
device-side `cudaGraphSetConditional()` here:
<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes>.

This can be inserted into the ongoing model stream capture without rebuilding
the whole model graph manually:

- launch/capture the finite stage kernel;
- obtain the capturing graph and current dependency frontier with
  `cudaStreamGetCaptureInfo()`;
- create a per-collective conditional handle and `WHILE` node with
  `cudaGraphConditionalHandleCreate()` and `cudaGraphAddNode()`;
- add the finite poll kernel to the conditional body graph;
- replace the capturing stream's dependency frontier with the conditional node
  using `cudaStreamUpdateCaptureDependencies()`;
- allow subsequent captured model work to continue after that node.

Important invariants to carry over from the existing kernel:

- generation gating on `door->ctl`;
- placement-hash validation before any ACK or payload consumption;
- no partial ACK set that a retry cannot reconstruct;
- canonical global-rank-order fp32 accumulation and identical bf16 output;
- source snapshot before an in-place fold;
- system-scope release/acquire ordering on `ready_bits` and `done_seq`;
- engine poison and device deadline exit paths;
- per-peer timing and gate diagnostic fields where still meaningful;
- graph variant isolation and staging-ring reuse fences.

A safe poll iteration should first discover and validate all peers without
acknowledging any of them. If even one payload is not yet placement-valid, it
should return and retry. Only after all peers validate should it publish ACKs
and fold. That avoids needing persistent per-peer claim state across
conditional-body launches.

Two alternatives are possible but less attractive:

- Run loopback ranks as separate processes, matching fabric. This fixes the
  test topology but does not make the runtime safe for multiple local ranks in
  one CUDA context.
- Insert host callbacks or external-semaphore waits between finite staging and
  fold kernels. This risks substantial per-collective overhead and callback
  serialization. CUDA stream memory waits alone are not sufficient: NVIDIA's
  API documentation explicitly warns that dependencies expressed only through
  memory waits are invisible to the scheduler and can themselves deadlock.

## Required validation for a claimed fix

A fix should not be accepted merely because the existing intermittent case
passes once.

1. Remove `DGPP_L2_PREFETCH=off` from the `glm_tp_test` CTest environment.
2. Run the filtered MTP serving test at least 100 times with all prefetch
   enabled and `CUDA_DEVICE_MAX_CONNECTIONS=32`.
3. Repeat at `CUDA_DEVICE_MAX_CONNECTIONS=1`. A non-resident state machine
   should no longer require multiple hardware work queues for correctness; this
   is the strongest adversarial gate.
4. Run the direct world-4 graph tests repeatedly, especially
   `glm_tp_device_pick_graph_loopback_matches_host_pick` and
   `glm_tp_full_graph_step_loopback_matches_eager_speculator`.
5. Run the complete 33-test CTest suite several times with prefetch enabled.
6. Verify bitwise transcript/cross-rank parity and the centralized collective
   oracles; scheduling changes must not weaken numeric or protocol checks.
7. Run compute-sanitizer on the graph collective tests.
8. Measure graph-step latency on fabric. Conditional polling must not regress
   the existing decode throughput materially; record poll-iteration counts and
   time-to-first-doorbell to tune any bounded per-iteration polling budget.
9. Capture a CUDA scheduler trace before and after if tooling permits. This is
   needed to turn the hardware-queue diagnosis from a strong inference into a
   directly observed mechanism.

## Handoff boundary

The bug is reproduced, isolated from profiling, and mitigated in CI. The next
agent should begin at the finite-stage/conditional-poll design above or replace
it with another solution that removes the hidden cross-stream forward-progress
assumption. Re-enabling prefetch in CTest is the completion criterion, not the
starting point.
