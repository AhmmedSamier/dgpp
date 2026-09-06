# What is worth doing next — a ranked cost/benefit view (2026-09-05)

Written at the close of M9: the four-node GLM-5.3-Flash service serves the
OpenAI chat contract with the one-graph speculative step, an exact prefix
cache, fail-fast failure semantics and a continuous cross-rank drift
check; the prefill sits near line rate and is shelved by decision. This
page ranks what remains by the value it would add to the service as it
stands against what it would cost, with the measurement each estimate
rests on. Costs are engineering days for one person who knows the code,
including the gates and the fabric evidence the repository's discipline
demands; "benefit" is quantified where a measurement exists and marked as
a judgment where it does not.

| # | item | benefit | cost | why this rank |
|---|---|---|---|---|
| 1 | Chunked prefill interleaved with decode | p99 decode latency under mixed load: today a 2,048-token prompt stalls every other live request for 1.7 s, a 32K one for 34 s | 6–10 d | the largest serving-quality defect left; everything needed exists (pool-aligned chunks, the graph era's mixed mode) |
| 2 | Operability: structured request logs, Prometheus metrics, systemd units | unattended operation; every incident so far was diagnosed from ad-hoc log greps | 2–3 d (first slice built 2026-09-06: the periodic throughput line, the per-tick lines at DEBUG) | cheap, and the failure semantics and drift check are only useful if someone is paged |
| 3 | The remaining request fields | client compatibility. **Built 2026-09-06:** `stop`, `n`, `logit_bias`, `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens_details.reasoning_tokens`. Left: accept-and-ignore for `user`, `store`, `metadata`, `service_tier`; the `developer` role mapped to system; `n` on the legacy completions route | 0.5–1 d for what is left | the accept-and-ignore fields are what stock clients still trip on |
| 4 | NVFP4 weights for the decode path | the T=1 step is weight-bandwidth-bound (24.6 of 31.5 ms is weight reads); halving expert bytes is worth up to ~25 % per token | 15–25 d | the biggest decode lever left, with a quality-validation bill (the checkpoint is FP8; a 4-bit conversion needs its own parity gates) |
| 5 | MTP batch of 8 requests (16 rows) | aggregate throughput at high occupancy: 60 tok/s at 4 MTP requests against 76 at 8 T=1 requests | 4–6 d | the fixed batch, the pick table and the latency slot are sized for 8 rows; a shape change across kernels and the bus |
| 6 | Faster silent-death detection | a powered-off node is seen by the bus watchdog after 60–120 s; a journal heartbeat would make it 1–2 s | 1 d | cheap; the drills covered process death, not node loss |
| 7 | Depth-2 MTP | ~7 % on mean decode throughput by the 2026-09-03 measurement (a second draft accepted ~60 % of the time) | 3–5 d | declined for the bimodal step; the number is real, the variance is the cost — a decision, not a defect |
| 8 | Grow-on-demand admission as the default, with preemption by recompute | higher occupancy for early-EOS workloads without truncating the youngest request | 3–4 d | needs a prefill fast enough to re-read a partial answer; the prefill is now 0.58 s per 256 tokens, so the recompute is affordable |
| 9 | Prefix cache eviction policy and arena sizing | the measured curve is a cliff: 43 slots hold 8 conversations at 100 %, 16 at 81 %, 20 at 25 % and 24 at 0 % (LRU over a cyclic working set); a policy that protects each conversation's newest close entry, or ages by frequency, turns the cliff into a slope, and the arena can be several times larger on this box | 2–3 d | cheap, measured, and the difference between a cache that helps a chat workload and one that only helps a repeated prompt |
| 10 | Prefix cache persistence across restarts | warm conversations after a restart; entries are 35 MiB each, 43 of them | 3–4 d | the restart is 25 s and the cold prefill of a turn ~1 s; the benefit is modest unless restarts are frequent |
| 11 | Rate limits, per-request deadlines, a prompt-length policy | robustness in front of untrusted clients; today a 32K prompt is admitted and stalls the world for 34 s | 1–2 d | cheap and necessary before public exposure; pairs with #1 |
| 12 | Memcheck of the loopback graph gates | device memory errors in the graph paths would surface in CI rather than on the fabric | 2–3 d | the harness limitation noted on 2026-09-04 (the instrumented collective outlasts the budgets) |
| 13 | Long-context prefill (shelved) | 32K TTFT 33.7 s → perhaps 20 s: the indexer's materialized dots and select grow with context, and expert bytes per chunk are the structural lever | 10+ d | shelved at line rate by decision; listed so the number is on record |
| 14 | Failover (a standby rank set) | availability across a node loss without a 25 s gap | 15+ d | the bus world would need to re-form mid-era and the state to be replicated; for a single fabric the restart is the cheaper answer |
| 15 | Data-parallel replicas behind a router | linear throughput with more nodes | 5–8 d for the router and the cache key sharing | no benefit on four nodes |
| 16 | TLS and authentication in front of rank 0 | required before exposure beyond the lab | 0.5 d (a reverse proxy) | trivial; documented rather than built |

## The reasoning behind the top of the list

**1. Chunked prefill.** The scheduler runs one admission per tick and one
decode step per tick, and the admission's prefill runs to completion
before the tick's decode step. With the prefill at 0.58 s for 256 tokens,
1.7 s for 2,048 and 33.7 s for 32,768, every live request's decode pauses
for exactly that long whenever a prompt arrives. At concurrency 4 with
mixed prompt lengths this is the p99 story of the service. The pieces
exist: the prefill is already cut into pool-aligned chunks (the prefix
cache's cut rule), the graph era already mixes eager collectives between
replay windows, and the scheduler's tick already has a fixed position for
admissions. The work is a per-tick prefill budget (one chunk of N tokens
per tick, then the decode step), the KV reservation for a partially
prefilled request, the journal carrying the chunk schedule so every rank
takes the same one, and the gates that hot == cold still holds when a
prompt's chunks are spread across ticks. The soak is the baseline to
beat (the record's twenty-ninth entry, `docs/signoff_v1.md` §6): at four
MTP streams the per-stream pace measured 78–79 ms per token where the
pure four-way batch runs at 66, and the chat turns' TTFT p99 sat at
4–7 s per ten-minute window; the difference in pace is the admissions'
prefill stalls, and the p99 is queueing behind them and the bursts.

**2. Operability.** The service has the right signals (the op-stream
fold, `/v1/metrics`, the exit statuses, the drill) and no way to be paged
on them. Structured per-request lines (id, prompt tokens, TTFT, tokens,
finish, hit/miss), a `/metrics` in Prometheus exposition, and a systemd
unit per rank that restarts on a nonzero exit turn the failure semantics
into an unattended service. This is the cheapest item with a benefit an
operator feels every day.

**3. The request fields.** `stop`, `n` and `logit_bias` and the usage
details were built on 2026-09-06 the way this paragraph first sketched
them: the stop match is a scanner over the content on rank 0 (a tail that
could begin a stop string is held back) and the retire rides the journal
like a cancel; `n` is one scheduler request per choice with the prefix
cache sharing the prompt; `logit_bias` is a per-slot dense row the device
pick adds after the penalties, the host sampler adding the same float in
the same place. What remains is smaller than the row's original estimate:
accepting and ignoring `user`, `store`, `metadata` and `service_tier`
(pure metadata for a self-hosted server; today they 400), mapping the
`developer` role to system, and `n` on the deprecated legacy route.

**4. NVFP4.** The decode step's floor is the weight read: 24.6 ms of a
31.45 ms T=1 step at four ranks. A 4-bit expert format halves the bytes
that dominate it. The cost is the conversion path, the GEMV/grouped-GEMM
kernels for the new format, and — the real bill — a quality gate
(perplexity and the golden transcripts) against the FP8 checkpoint,
because the conversion is lossy where the FP8 one was not. The benefit is
the largest single decode gain available and it compounds with MTP.

## What is deliberately not on the list

- Further prefill kernel work: shelved at line rate by decision; the 32K
  measurement (33.7 s, 1.03 ms per token, 16 chunks) is on record.
- Fusion rounds on the decode step: the 2026-09-03 round found the floor.
- A wider sampling candidate tier: the width sweep closed it (a fallback
  costs 5.65 ms plain / 14.3 ms under MTP and no served workload pays for
  a wider tier).
- Expert placement telemetry: moot at the FFN boundary since the sliced
  placement.

## What the sign-off pass itself turned up (fixed, for the record)

The M9 pass was not only measurement: the malformed-HTTP fuzzer under
AddressSanitizer found three defects in the HTTP and JSON layers, the
prefix-cache curve sweep found two in the cache's admission — a pool
block accounting hole and an eviction of the admission's own attach
target with the arena full — and the soak's first hour found a sixth, an
SSE delta that was not UTF-8 when a token ended inside a character. All
six are fixed and gated (`docs/signoff_v1.md` §6, the record's
twenty-ninth entry). The lesson that generalizes: the drills and sweeps
that fill a resource to its edge (the arena, the pool, a connection's
request buffer) and the clients that are strict about the contract are
where the remaining defects live; the second soak hour ran with all of
them fixed.
