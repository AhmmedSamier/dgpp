# Request history in the persistent scheduler (2026-09-13)

The report: `bugs/dgpp-memory-analysis/`, a third-party tester's redacted
working notes behind GitHub issue #1, pinned to c7609af (the scheduler is
unchanged through 8464e6a). The claim: every submission stays in the
scheduler for the process's lifetime, so a served process grows with its
cumulative traffic and every tick scans the whole history. The claim is
correct; this record has the measurements behind the fix in the same
commit.

## What the source held per retired request

`Scheduler::try_submit` appended every request to `requests_` and
`results_`; `retire` set the state, copied the generated ids into the
result and released nothing; the service's record cleanup erased only its
HTTP records, and no production code read `results()` or `find()` (the
tests do). Each retired record kept the prompt ids (8 bytes a token), the
generated ids twice, the prefix cuts and hashes, the boundaries, the logit
bias and a copy of the request's tool grammar. The peers replay the
journal into their own scheduler, so every rank held the same.

## Reproduction against the built scheduler

A 60-line driver linked to `build-ci/libdgpp_sched.a` (8464e6a): a
one-slot fake engine, the prefix cache off, one request at a time, one
token each — the tester's shape.

| Run | RSS growth | Retained prompt bytes |
|---|---:|---:|
| 1,000 × 16,384-token prompts | 131.2 MB | 131,072,000 (the tester's figure) |
| 20,000 × 8-token prompts | 12.9 MB | 1.3 MB, i.e. ~660 bytes fixed per record |

The scans: an idle tick cost 0.7 µs with 1,000 retired records and
33 µs with 20,000 (1.6 ns a record; a 42 ms decode step); a submit's
duplicate-id scan 2 → 16 µs. Real but immaterial at any count a server
reaches.

## The real traffic

From the retire lines of two recorded Hermes sessions (rank 0's log):

| Session | Requests | Mean prompt tokens | Retained prompt bytes |
|---|---:|---:|---:|
| 2026-09-07, 2 h | 85 | 32,379 | 21 MiB |
| 2026-09-11, 1.5 h | 20 | 8,054 | 1.2 MiB |

About 10 MiB an hour of active single-agent use; the prefix cache serving
88 % of those prompt tokens changed nothing, since the ids were kept
regardless. Against the memory plan's 4 GiB headroom, of which the
measured post-check growth already takes 2.4–2.8 GiB, that is five to
seven days of continuous traffic for one agent, proportionally less for
eight busy slots — and on the Spark's unified memory the failure mode at
exhaustion is page reclaim and a freeze, not a clean error.

The op stream was a second path of the same shape: every rank's
`OpStreamObserver` appended each token, retire, grow and prefix line to a
string for the process's lifetime, written to `serve_rank<N>.ops` only at
exit. The 2026-09-07 session's stream was 4.1 MB over two hours (37 bytes
a generated token, 43 a prefix op), and a killed rank wrote nothing.

## The fix

A retired request releases everything but its tombstone (id, status,
counts) at retire; the service and the peers set `keep_retired` false, so
the result's copy goes with it and the tombstones are compacted away at
the end of every tick — the same quantum on every rank, since rank 0
journals every tick, idle ones included — with the slot map, the
round-robin cursor and the deferral log remapped to the live requests'
new indices. (The first cut dropped the history only at a tick that found
nothing pending; a never-idle four-slot run then held ~560 bytes a
retirement until the first idle tick, 11 MB over 20,000 requests, and a
saturated server might not give it one for days.) The op stream
goes to its file as it is recorded, flushed at every retire. `/v1/metrics`
reports `scheduler.records` and `record_tokens` (0 / 0 on an idle server)
and `terminal` is cumulative. `scheduler_test` pins that the compaction
moves no op: under a script whose retirements interleave with live
requests, a scheduler that keeps its history and one that compacts hand
identical op streams and batch slices to identically armed engines, and
the compacting one holds exactly its live requests after every tick;
`fabric_serve_test` covers the file-mode observer.

## After the fix, at scale

The same driver with a warm-up request first (so first-touch pages are out
of the delta), live heap from `mallinfo2`:

| Run | RSS | Live heap | Records held |
|---|---:|---:|---:|
| sequential, 16,384-token prompts, 200 → 40,000 requests | +128 KiB, flat | +5 KiB, flat | 0 |
| sequential, 8,192 tokens + 20-tool grammar, boundaries, bias, 20,000 | +4 KiB, flat | +5 KiB, flat | 0 |
| four slots never idle, 8,192 tokens + payload, 20,000 (first cut) | +11 MB, growing | +15.6 MB | 20,000 |
| four slots never idle, same, per-tick compaction, 20,000 and 100,000 | +400 KiB, flat | +324 KiB, flat (the four live requests' payloads); +8 KiB drained | 4, the live ones |

## On the fabric (2026-09-13, the production deployment, four nodes)

The full `ctest` suite first, with the fabric idle: 78 of 78 in 343 s,
nothing skipped. Then the served path with the new binary, one ritual at a
time, against `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json`
(the deployment that was serving; the launcher stages the binary to every
node):

- **The served run:** boot 16 s; `serve_api_check.py` ALL OK; between
  requests `/v1/metrics` read `records 0, record_tokens 0` at every idle
  sample while `terminal` climbed 12 → 19 (three 4K-token requests, then
  four concurrent); `down` found the four op streams identical at 58,519
  bytes (19 retire lines) — the streaming writer's first fabric run.
- **The drain-on-stop check** (`serve_stop_check.sh`): two requests
  mid-generation, "2 in-flight request(s) retired through the drain pass on
  every rank, the queue shed, in 0 ms", the streamed client's 270 chunks then
  the `server_shutdown` event, the four op streams identical (md5
  `9b8d601d…`).
- **The failure drill** (`serve_failure_drill.sh 2 3`, kill -9 rank 2 under
  three streams): every client got its committed tokens then the
  `engine_failure` event and `[DONE]`; rank 0 out with status 2 in 0.87 s,
  the peers with status 3 within 1.9 s. Rank 0's op stream, flushed on the
  status-2 path, is byte-identical to both survivors' (196 lines, md5
  `a67cf366…`), and the victim's own file holds the first 4,096 bytes of the
  same stream — one stdio buffer, which the exit-time write never produced.
  Two defects in the tooling surfaced, neither in the engine: the drill
  copied rank 0's stream from the repository root (the launcher runs rank 0
  in the deployment's log dir, so it reported "no ops file"), and its reboot
  seconds after the failure failed the launcher's preflight on the fabric
  and journal ports — the probe bound without `SO_REUSEADDR` while both
  servers bind with it, so a TIME_WAIT from the previous world counted as a
  conflict (the same reason a restart within a minute of any stop failed
  preflight). Both fixed in the same commit; the drill's rerun and the soak
  below.
- **The failure drill, rerun with the fixes:** 0 failure lines. Rank 0
  out with status 2 in 0.97 s, the peers with status 3 within 2.0 s; every
  client its committed tokens then the `engine_failure` event; rank 0's
  flushed stream identical to both survivors' (156 lines, md5
  `3d5f62c8…`); the reboot passed preflight seconds after the failure and
  each client's committed text is a prefix of the fresh answer (197 / 209 /
  197 bytes of 5,769 / 6,847 / 6,103).
- **The soak** (`serve_soak.py` five minutes on the booted world after the
  harness killed the background ritual at the world's load; the ritual's
  teardown by hand): 439 requests served, 13 shed at the queue bound and 73
  cancelled by the workload's own disconnects, 0 failed; short requests
  TTFT p50/p95 432/1,448 ms at 52/61 ms a token. Sampled every 30 s through
  it, the scheduler held 3–5 records — exactly its active plus queued
  requests — while `terminal` climbed 172 → 482, and rank 0's RSS stayed at
  3.64 GiB (3.60 at listening). Teardown: the four op streams identical
  (md5 `6480fa97…`), 0 stalled / 0 failures / 0 errors on every rank since
  the boot, node probes allocstall 0, swap 0 on all four nodes. The
  production deployment was restored on the new binary afterwards.

