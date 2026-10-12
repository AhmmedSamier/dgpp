# Towards continuous batching: multi-admission + fair-share prefill

Status: original items 1 and 2 implemented in `Scheduler::quantum()` and
soak-tested live. The size-neutral queued-text follow-up for issue #104 is
host-tested; real-model latency and MTP boundary validation remain separate
acceptance gates, not established by the original soak.

## Problem

Single-stream dgpp matches or beats vLLM, but per-request throughput collapses
under concurrency (measured: 45 → 9 tok/s @d8k c4). Root causes in the
scheduler (`src/sched/scheduler.cpp:quantum()`, `1124-1254`):

1. **≤1 admission per tick** (`1161-1165`): request #2 queues behind request
   #1's full prefill — head-of-line blocking. vLLM admits into the running
   batch immediately.
2. **One unfinished prefill at a time**: while any request is `kPrefilling`,
   `admit_arrival` is forced to -1 and every queued request waits for the
   whole chunked prefill to finish.
3. Decode cost scales ~7 ms/GPU-row with batch width while each request still
   earns ~1 token/step (kernel problem, out of scope here).

The full vLLM-style fix (a request joining mid-decode, with graph
revalidation and MTP rescheduling) is deliberately out of scope: it would
touch `validate_batch` (`src/engine/graph_engine.hpp:1665-1684`), the verify
schedule, and the fabric lockstep digests. The two items below capture most
of the c2–c4 win without it.

## Item 1: multi-admission per tick (done)

**Rule**: after the cancel sweep and `grow_reservations()`, admit fitting
queued requests in a loop *before* the decode slice is built. Each iteration
admits via the same dispatch as before (group → one-shot → chunked-start),
except:

- a request needing chunked prefill **ends** the loop (starts chunking only
  if nothing was admitted yet this tick — chunked read-ins stay one at a
  time; see item 2);
- the loop stops before grouping or admission when the tick's positive
  prefill budget is exhausted; zero remaining tokens must not be passed
  to the group planner, where zero means monolithic mode;
- with a zero policy budget, exactly one admission event (possibly a
  group) runs per tick, preserving the original monolithic policy;
- an oversized image on an engine without image chunking retains its
  monolithic fallback as the tick's only prefill admission. It waits if
  any prefill work already ran that tick, then consumes the whole cap;
- `next_admissible()` returning -1 (no fitting request / no free slot) ends
  the loop as before.

**Why it preserves the invariants**:

- `step_batch` builds its round-robin slice *after* admission
  (`scheduler.cpp:1212-1234`); `validate_batch` only requires the replay to
  include every live slot, which holds trivially for slots admitted earlier
  in the same tick — each admit path calls `reserve()` in `admit_finish`.
- Every function in the loop (`next_admissible`, `admissible_group`,
  `admit`, `admit_group`, `begin/advance_prefill`) depends only on replicated
  scheduler state, so all ranks compute the same loop. The fabric journal
  already carries `submits` as a list (`generation_service.cpp:3416`); no
  journal format change.
- Precedent: `admit_group` already admits N≥2 cold prompts in one forward.

**Docs updated with it**: `src/sched/scheduler.hpp:8-9` (tick policy line).

## Item 2: fair-share chunking across queued prefills (done)

**Rule**: the `prefill_arrival >= 0 → admit_arrival = -1` gate is gone.
Each tick with a prefill in flight now: collects the in-flight prefills in
arrival order, begins at most one new read-in (oldest fitting,
skip-fit, no eviction dance — a begin must not disturb the pool the
in-flight prefills hold), gated so every share keeps at least one aligned
chunk (`(n+1)*align <= budget`); splits the tick budget into equal
align-down shares and advances the selected prefills on their shares; then
admits fitting one-shots/groups into the align-down leftover (never a new
chunked start — chunked read-ins stay one at a time). Order derives from
arrival order and the last-advanced prefill only, so every rank agrees.

Qwen graph serving without DFlash enables early text admission by default: queued text of
any size can join, including prompts shorter than one alignment unit. New
readers share a bounded grouped prefill and share their dense and expert
projections with active decoder verification.
Each decoder retains its own recurrent state, speculative rollback rows,
MTP drafts and sampling state. It receives at most one decode pass per tick.

`engine.prefill_early_admission=false` restores admission through leftovers.
Other execution backends retain their previous default; explicitly enabling
early admission there does not provide Qwen's shared execution path. Images
retain their existing eligibility and execution paths. Monolithic mode is
unchanged.

The new reader still needs a free slot and its reservation under the existing
KV policy. Cached text attaches through the same prefix plan and computes
only its suffix. With early admission enabled, a short tail keeps its rounded-up aligned allocation and
donates the rest of its share to unfinished readers, split equally in
aligned units with arrival-order ties. This happens before the advances,
uses replicated positions and stays inside the original tick budget. When
all readers finish, unused capacity does not create another admission lane.
These allocations are caps: required snapshot/structural cuts can end a GPU
advance earlier. Qwen spends the cap from the cursor's current position and
omits redundant automatic chunk-grid cuts; this avoids splitting work again
after a share changes. Other families retain their chunk-grid contracts.
A lone reader, or a set of readers that all fit their original shares, keeps
those shares unchanged, including its final tails.
The busy budget continues to protect decoders that started before a reader.
When every active decoder joined after all unfinished readers, and the engine
can share their execution, those readers retain the configured idle budget.
This avoids multiplying their weight-streaming passes because a short request
was admitted. Newly admitted streams can consequently have longer update gaps
while sharing those larger chunks. The budget remains bounded; slot and KV
pressure can still delay admission. Required cuts and decode rows can leave
less than the full prefill allowance in a shared walk.

For example, with a 256-token budget and four-token alignment, a 23-token
arrival beside one long reader gets 24 allocated tokens and the long reader
gets 232. An engine consuming those caps computes 255 tokens in total;
equal shares offer at most 151 in that tick. This saves scheduled capacity;
GPU throughput
still depends on chunk shapes, additional walks and active decode width.
Deployments that need the same prefill cap before and after admission can
choose equal busy and idle budgets. Larger busy budgets can increase gaps
between decode updates, so select them with mixed-load measurements.

When the budget drops from idle to busy, the in-flight count can exceed
`budget / align`. Advance only that many prefills, rotating through arrival
order from the last-advanced request. Every share remains at least one
aligned chunk without exceeding the total budget. Record compaction remaps
the prefill cursor just like the decode cursor, including when its request
was cancelled or retired. With enough budget, all prefills advance again.

**Audit results** (the pre-implementation open questions, all green): the
engine already supports ≥2 concurrent `kPrefilling` requests — per-slot
`prefills_` cursors suspend/resume by design (`session_model` advance marks
unfinished slots inert `0xff`), `reseed_live_feeds` touches live slots
only. Grow-mode shedding can retire younger `kActive` or `kPrefilling`
requests at the tick's initial reservation pass; the in-flight list is
built after that pass. Each surviving prefill retains its reservation
between chunks.

## Test plan (both items)

- `unit_tests` + `serve_test` green (scheduler determinism tests compare
  concurrent requests against independent runs).
- Scheduler regressions cover exhausted group budgets, chunked starts
  after one-shots, monolithic image fallback, idle-to-busy budget changes,
  fair rotation, cancellation, slot reuse and compaction determinism.
- Queued-text regressions additionally cover alignment and busy/idle budget
  boundaries, oldest-fitting order, three/four concurrent requests, full
  output reservations, cache-attached short totals and histories, and
  joined-prefill cancellation and engine failures. A fake with MTP trailing
  rows checks reservation forwarding and unwind, not physical GPU safety.
- Tail redistribution regressions cover multiple donors and recipients,
  all readers finishing together, twenty consecutive short arrivals, and
  preservation of decode progress and the total allocated budget. Begin
  failures release every private snapshot, including the head cut, before
  cancellation or retry. These scheduler cases also run under ASan/UBSan.
- Live: c1/c2/c4 @ d0/d4k/d8k benchy sweep; TTFT of request #2 must drop;
  per-request tok/s must not regress at c1.
- Item 2 additionally: c4 deep-context soak, KV pool metrics sane, no
  `validate_batch` / digest mismatches in logs.

The reproducible admission comparison is
`python3 scripts/serve_admission_bench.py --url http://127.0.0.1:18081
--label COMMIT-CONFIG --out RESULTS.json`. Run matched baseline/candidate
processes in alternating order on reserved hardware, using the same prompts,
output limits and configuration. Compare equal busy/idle budgets as well as
256/2048 to expose the budget-transition trade-off. Its seven scenarios cover
isolated prefill/decode, a short arrival with short or long output, sustained
arrivals with and without an existing decoder, and cancellation/recovery.
The first arrival must be dispatched while the long response's matching
native prefill ID is still present. Requests, usage, update times, output
hashes and before/after metrics are retained; unrelated traffic invalidates
a run. Aggregate rates include the full scenario wall time and client update
gaps are not per-token kernel latency. Cached conversations, image paths,
physical KV boundaries, semantic checks and multi-rank operation-stream
agreement remain separate live gates. Host passes do not establish GPU
throughput equivalence.

### Shared execution timing

`mixed_steps` counts decode passes that also advance prefill. Their elapsed
`mixed_ms` is included in both `prefill_ms` and `step_ms`, because both logical
operations occupy that interval. Total engine time is therefore
`prefill_ms + step_ms - mixed_ms`; phase times are not independent throughput
measurements. Prometheus exposes the same overlap as `dgpp_mixed_seconds_total`
and `dgpp_mixed_steps_total`, including per-rank equivalents. Compare completed
work over wall time when evaluating mixed workloads.
