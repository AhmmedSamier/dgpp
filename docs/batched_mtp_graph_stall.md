# Batched-MTP CUDA graph stall: root cause and the kernels-only contract

Investigated and resolved 2026-09-03.

The stall was traced to copy-engine nodes in a captured decode graph.
An nsys node trace and a synthetic reproducer confirmed the dependency
cycle. Replacing the copies and resets with kernels resolved it; the
stress tests passed with prefetch enabled at
`CUDA_DEVICE_MAX_CONNECTIONS=32` and `1`.

The investigation below records the graph and behavior tested on
2026-09-03. The current `check_decode_graph` in
`src/engine/graph_check.hpp` rejects memcpy, memset, event-record and
other unsupported node types. Kernel and empty nodes are allowed. A model
may also declare a bounded number of host nodes: Qwen uses this for
gathering rows from its mapped n-gram table. Those callbacks run on runtime
threads and must not introduce a dependency on another rank. This exception
does not permit copy-engine nodes.

The original failure occurred in an in-process multi-rank test, where the
ranks shared a copy-engine queue. Read the dependency analysis before
changing the graph-node checks or adding captured work.

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
  host buffer); `src/models/glm/decode.cpp` uses them for the request-id and span
  tables (scalar and batch captures), the decode rows' token feed and the
  host-positions path; `src/models/glm/forward.cpp` allocates all four upload
  sources `cudaHostAllocMapped` and checks the UVA address identity.
- `set_decode_tail_mirrors(false)` now gates only CAPTURED mirror nodes:
  eager steps and eager drafts always mirror (they sync anyway), and
  `session_graph_outputs` copies a replay's tail eagerly when the mirrors
  are off instead of refusing. The direct device-pick test turns them off.
- `src/engine/graph_check.hpp`: `glm_check_decode_graph` logs the
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

### Assessment of the initial hypotheses

The diagnosis "a required GPU node fails to run while its peer's node
remains resident waiting for it, through a dependency CUDA cannot see" was
right, and so was the instinct that hardware-queue head-of-line blocking
was involved. The queue is the copy engine's, not the compute work queues,
and the blocked node is a memset, not a kernel. The recommended
non-resident WHILE-node collective would not have fixed this: the memset
would have queued behind the peer's dependency wait exactly as before, and
the conditional node's own downstream wait would have joined the cycle. No
change to the collective kernel was needed.

The earlier handoff is preserved in this file's git history. The full
dated investigation record is in `benchmarks/results/2026-08-29-bus-m5.md`.
