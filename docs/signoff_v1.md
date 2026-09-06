# v1 sign-off report (2026-09-05)

The performance and hardening sign-off PLAN's M9 asked for, with the
measurement behind each number and the gaps stated as gaps. Every figure
here was produced on the four-node GB10 fabric (192.0.2.11–14, one
200 Gb/s RoCE port per node) with `unsloth/GLM-5.3-Flash-FP8` at TP=4,
resident, unless a line says otherwise; the raw logs live under
`build-ci/fabric-runs/` on the head node and the running record is
`benchmarks/results/2026-08-29-bus-m5.md`.

## 1. Batch-size-one decode and time to first token

**Decode, one request.** The single-stream fast path (`glm_gen_check`,
the 300-step "Roman Republic" chat prompt, records of 2026-09-02/03):

| mode | ms per token | tokens/s |
|---|---|---|
| eager T=1 (the default engine seam) | 36.4 | 27.5 |
| one-graph T=1 | 31.45 | 31.8 |
| one-graph MTP, greedy | 22.45 | 44.5 |
| one-graph MTP, sampled at the card's defaults (T=1.0, top-p 0.95) | 25.0–27.6 | 36–40 |

The service adds nothing measurable: the concurrency-1 service measures
21.8–26.0 ms/token with MTP and 32.1 at T=1 (`scripts/serve_pace.py`).
The occupancy curve of the adaptive graph engine is 30.97 / 31.67 /
41.76 / 76.18 tok/s aggregate at 1 / 2 / 4 / 8 T=1 requests and
38.48 / 38.97 / 60.27 at 1 / 2 / 4 MTP requests.

**Time to first token.** The prefill in steady state (`--prefill-repeat
3`, the hard-prose ids, all four ranks' generated ids identical to every
earlier round's):

| prompt tokens | prefill ms | ms per token | measured |
|---|---|---|---|
| 256 | 581 | 2.27 | 2026-09-05 (closure pass) |
| 2,048 | 1,701 | 0.83 | 2026-09-05 (closure pass) |
| 4,096 | 3,523 | 0.86 | 2026-09-05 (round 8) |
| 8,192 | 7,260 | 0.89 | 2026-09-05 (round 8) |
| 32,768 | 33,728 | 1.03 | 2026-09-05 (this report; 16 chunks) |

The 32K figure is the committed workload's long end: 33.7 s, linear in
the prompt to within 15 % of the 2K rate (the sparse attention regime's
indexer and select grow with context). Prefill optimization is shelved
at this point by decision; the remaining lever is structural (expert
bytes per chunk) and is on the next-steps list.

On the service the time to first token is the prefill plus the pick and
the HTTP hop: a 25-token prompt 1,109 ms cold and 751 ms hot (an attach
at the header's cut), a 1,592-token prompt 2,643 ms cold and 949 ms hot;
the prefix cache's TTFT split over the closure pass's turn sequence was
179 ms on hits against 1,141 ms on misses.

## 2. Expert traffic (expected, p95, worst rank)

Moot since the sliced expert placement (M5/M6): every rank holds its
slice of every expert, the MoE boundary is one all-reduce per layer, and
no token routes an expert across the fabric. The trace tool
(`glm_trace`, `docs/checkpoint_budget.md`) remains for the attention side
and for any future placement change; there is no per-rank expert traffic
to report.

## 3. Both-lane utilization and collective share

Per the per-collective timeline (records of 2026-09-03): the one-graph
step runs 94 collectives per replay; at T=2 they take ~4.6 ms of a
42.3 ms step (~11 %), at T=1 ~3.5 ms of 31.45 (~10 %); the handshake is
8–16 µs per collective and the ranks' compute spread ~3 %. Both lanes
carry every striped bulk chunk (the prefill's segmented all-reduce, 2(W−1)/W
of the buffer on the wire) and the latency pool's one-shot folds
(decode); the bulk pacing is the derived 28.3 Gb/s per queue pair, under
which the fabric ran clean through every measurement here.

## 4. Prefix cache capacity and hit curve (M7)

`scripts/serve_prefix_curve.py` against the service with the MTP graph:
C conversations of three short turns, interleaved round-robin, at two
arena budgets (0.25 GiB = 7 slots, 1.5 GiB = 43 slots; 35.2 MiB per
slot). Each conversation holds up to two entries (its prompt-cut entry and
its close entry) and each live request one rolling slot while it runs.

| arena | conversations | later turns hit | tokens saved | evictions | TTFT hit / miss (service) | client TTFT per turn 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 7 slots (0.25 GiB) | 8 | 0 / 16 (0 %) | 0 | 41 | — / 1,008 ms | 463 / 1,027 / 1,572 ms |
| 7 slots | 32 | 0 / 64 (0 %) | 0 | 185 | — / 1,014 ms | 469 / 1,032 / 1,581 ms |
| 7 slots | 64 | 0 / 128 (0 %) | 0 | 377 | — / 1,076 ms | 498 / 1,097 / 1,674 ms |
| 43 slots (1.5 GiB) | 8 | 16 / 16 (100 %) | 1,796 | 5 | 480 / 460 ms | 475 / 611 / 372 ms |
| 43 slots | 16 | 26 / 32 (81 %) | 2,876 | 53 | 440 / 693 ms | 476 / 647 / 589 ms |
| 43 slots | 20 | 10 / 40 (25 %) | 1,208 | 77 | 423 / 964 ms | 475 / 920 / 1,266 ms |
| 43 slots | 24 | 0 / 48 (0 %) | 0 | 101 | — / 1,021 ms | 475 / 1,039 / 1,587 ms |
| 43 slots | 32 | 0 / 64 (0 %) | 0 | 149 | — / 1,019 ms | 474 / 1,041 / 1,584 ms |
| 43 slots | 64 | 0 / 128 (0 %) | 0 | 341 | — / 1,083 ms | 500 / 1,101 / 1,687 ms |

(Three-turn conversations of short questions, 96-token answers at
temperature 0 with the template keeping the reasoning; a 16,384-token
pool so the arena, not the pool, is the constraint; every point's op
streams identical on all four ranks. The client's per-turn TTFT includes
the whole answer's prefill or attach and the HTTP hop.)

Reading it: the cache is exact and cheap (an attach 0.5 ms, a snapshot
0.4 ms) and its hit rate is capacity against the working set. Each
conversation leaves about two entries per turn (its prompt-cut entry and
its close entry), and LRU over a cyclic pattern keeps only the newest
ones: 43 slots hold 8 conversations completely, 16 at 81 %, 20 at 25 %,
and nothing from 24 on — the cliff, not a slope. Size the arena for the
number of conversations that should stay warm (about two slots each plus
one per live request); 1.5 GiB is 43 slots and the box has room for
several times that. A policy that protects each conversation's newest
close entry, or ages by frequency, would turn the cliff into a slope; it
is on the next-steps list.

## 5. MTP acceptance and net speedup, per prompt class

`glm_gen_check --decode-graph --mtp`, greedy, 300 tokens per class, the
four ranks' transcripts identical (2026-09-05):

| prompt class | drafts accepted | tokens per step | ms per token | ms per step |
|---|---|---|---|---|
| chat (a technical explanation) | 77.1 % | 1.77 | 23.9 | 42.2 |
| code (a Python module with tests) | 97.4 % | 1.97 | 21.7 | 42.9 |
| prose (a long history) | 87.5 % | 1.88 | 22.2 | 41.6 |
| JSON (25 records) | 97.4 % | 1.97 | 21.3 | 42.1 |
| math (a worked problem) | 92.3 % | 1.92 | 22.1 | 42.6 |

Against the plain graph's 31.45 ms per token the net speedup is 1.31×
(chat) to 1.48× (code, JSON). Sampled at the card's defaults the
acceptance measured 67–81 % on the chat class (2026-09-04), 25.0–27.6
ms per token. The step itself is flat across classes (41.6–42.9 ms):
acceptance is the whole story.

## 6. Hardening: what was drilled, what was fuzzed, what runs continuously

- **Failure semantics.** Kill −9 of a peer and of rank 0 under three
  streaming clients (`scripts/serve_failure_drill.sh`): every live stream
  received exactly its committed tokens then the `engine_failure` event,
  every rank exited within 3.2 s (peer killed) / 2.6 s (rank 0 killed),
  and the restarted service reproduced the committed texts as prefixes
  of its answers. `docs/operations.md` has the exit statuses.
- **Continuous drift check.** Every tick record carries rank 0's op-stream
  fold; a diverging peer dies within a tick naming it (gated in-process;
  live for the whole soak below).
- **Malformed HTTP.** `serve_fuzz_malformedHttpNeverBreaksTheServer`: a
  byte-level mutator over the limit ladder, 30,000 iterations under
  AddressSanitizer (19,139 answered across the ladder — 400 ×7,975, 431
  ×2,440, 501 ×2,297, 413 ×466, 411 ×371, 405 ×345, 404 ×3,012, 200
  ×2,233 — 951 closed, 6,936 incomplete requests waited on). It found three defects, all fixed and gated: a JSON
  body nested ten thousand levels deep ran the parser's stack out (the
  parser now refuses more than 256 levels with a 400); a malformed request
  pipelined after a valid one-shot closed the connection without telling
  the service, whose answer then wrote through the freed writer (every
  close of a tagged connection now notifies); and two requests pipelined
  on one connection were dispatched together, sharing the writer and
  overwriting the disconnect tag (a connection now carries one request at
  a time; the next waits in the buffer until the answer is out). The
  prefix-cache sweep found a fourth defect outside the fuzzer's reach:
  the admission did not count the pool blocks the cache's partial-block
  copies take, and a full pool killed the service through the failure
  path; the accounting is fixed and gated (§4). And a fifth, with the
  arena full for the first time: an admission acquiring its snapshot slot
  evicted its own attach target and prefilled from an empty slot; the
  attach now precedes the acquisition, gated. And the soak's first hour
  found a sixth: a byte-level BPE token can end inside a multi-byte
  character, and the SSE delta carried its bytes as they came — a JSON
  text that is not UTF-8, on which a strict client raised. Every streamed
  field now holds an incomplete trailing sequence back until the next
  delta completes it (U+FFFD at the end, and for a cap-cut last
  character in a one-shot), gated.
- **The soak.** One hour (`scripts/serve_soak_run.sh 60`; the production knobs: four slots,
  an 8,192-token pool, a queue of eight, the MTP graph, the prefix cache on)
  with three multi-turn chat workers, a long generator sampled at the card's
  defaults (512 tokens), a client that abandons its stream after 5–40
  tokens, and a burst of 14 one-shots above the queue bound every five
  minutes: 2,692 requests, 157,734 tokens out — 1,980 chat turns, 72 long
  generations, 516 abandoned streams (507 cancels landed), twelve bursts (88
  served, 80 shed 503 `overloaded`) and three streams shed mid-burst with
  the in-stream error event. No failure, no stall witness on any rank, no
  divergence, the four op streams md5-identical; every worker alive for the
  whole hour. Per 10-minute window the chat turns' TTFT p50 737–810 ms and
  p99 3,989–6,875 ms (queueing behind bursts and long generations), the per-
  stream pace p50 78–79 ms/token (four MTP streams sharing the step plus the
  admissions' prefill stalls; 66 ms is the pure four-way batch); no trend
  across the six windows, the pool at 59–65 of 65 blocks and the cache at
  38–40 entries from the tenth minute on, allocstall 0 and swap 0 on every
  node. The first hour, run before the UTF-8 fix, is the one that found the
  sixth defect; its cancel worker and bursts ran the full hour clean, and
  the second hour is the sign-off measurement.

## 7. Known gaps

- The prefill is host-orchestrated at 2,048-token chunks and stalls the
  other live requests for its duration (34 s for a 32K prompt); chunked
  prefill interleaved with decode is the top item on the next-steps list.
- There is no failover: a rank's death fails the service, which restarts
  in ~25 s. A powered-off node is seen by the bus watchdog after
  60–120 s, not by the journal watches.
- The prefix cache is per process (no persistence) and LRU (no
  protection of close entries against a cyclic working set).
- `stop`, `n` and `logit_bias` are served since 2026-09-06 (the day after
  this report); `user`, `store`, `metadata`, `service_tier`, `suffix` and
  `best_of` are refused with 400 naming the field; `stream_options`
  beyond `include_usage` is ignored.
- MTP depth is fixed at 1 (depth 2 declined for step-time variance); the
  MTP batch is four requests (eight rows).
- The soak ran for one hour, not the 24 the plan first named (the user's
  decision); slow leaks beyond an hour are unmeasured. The bus's 32-bit
  generation counter wraps after ~4.3 billion collectives (~21 days of
  continuous decode at 25 steps/s); the remedy is a restart.
- Memcheck of the loopback graph gates does not complete (a harness
  limitation noted 2026-09-04).
