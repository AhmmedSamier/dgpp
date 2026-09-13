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
the result's copy goes with it and the whole history is dropped at the
first tick that finds nothing queued or active — the same quantum on every
rank, since rank 0 journals every tick, idle ones included. The op stream
goes to its file as it is recorded, flushed at every retire. `/v1/metrics`
reports `scheduler.records` and `record_tokens` (0 / 0 on an idle server)
and `terminal` is cumulative. `scheduler_test` pins that the drop moves no
op (the same op stream and batch slices on a scheduler that keeps its
history and one that drops it); `fabric_serve_test` covers the file-mode
observer.
