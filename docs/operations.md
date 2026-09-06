# Operating the serving world

The four-node GLM service as it is run today: one `glm_serve` process per
node, rank 0 the only HTTP ingress, the peers following rank 0's admission
journal. This page collects what an operator needs — boot, stop, status,
the knobs, what happens when a rank dies, and how to check that the world
is healthy — with pointers to the design where the reasons live
(`DESIGN.md` §11 for the fabric protocol, `PLAN.md` M9 for the hardening
record).

## Boot, stop, status

```
scripts/serve_run.sh up       # stage the binary to the peers, boot rank 0, then the peers; waits for READY
scripts/serve_run.sh down     # SIGINT rank 0 (drain-on-stop), wait for the peers, fetch and md5 every rank's op stream
scripts/serve_run.sh status   # rank 0 alive? the peers' process counts
```

`up` stages `build-ci/glm_serve` to `/tmp/bus4/` on each peer (the
directory is created, and any previous run's op-stream files there are
removed so they can never be mistaken for this run's evidence), boots
rank 0 (`HTTP :18080`, bus rendezvous `:29970`, journal `:29971`), waits
for its rendezvous listener, spawns the three peers in parallel by ssh,
and waits for rank 0's `serve: listening` line. Readiness is that line —
with the resident image cache warm it takes 15–25 s; a first boot that
builds the image from the checkpoint takes ~4.5 minutes.

`down` sends rank 0 SIGINT. Rank 0 drains: the door closes (new requests
get 503 `server_shutdown`), the in-flight requests are cancelled through
the journal at the next tick and their streams end with the shutdown error
event, then the stop record releases the peers. `down` waits up to 240 s
for rank 0 (a stop that lands mid-prefill is honored at the pass
boundary), then for the peers, then fetches `serve_rank{1,2,3}.ops` from
the peers next to rank 0's and prints their md5s: **the four hashes must
be identical** (the §11 op-stream ritual). Logs land in `$DGPP_SERVE_LOG`
(default `/tmp/opencode/serve_fabric`).

`DGPP_SERVE_KNOBS` sets the engine flags on **every** rank (they must be
the same string on all four); the production shape is

```
DGPP_SERVE_KNOBS="--max-concurrency 4 --kv-capacity 8192 --default-max-tokens 256 --queue-limit 8 --decode-graph --mtp"
```

`--decode-graph --mtp` is the one-graph speculative step (the measured
serving mode); `--prefix-cache-gib` sizes the prefix cache's snapshot arena
(default 1.5 GiB → 43 slots; `--no-prefix-cache` turns it off);
`--admission grow` enables grow-on-demand admission; `--queue-limit` is the
admission queue bound past which the door answers 503 `overloaded`;
`--stats-interval-s` is the period of the throughput line below (default
10 s; 0 turns it off). `DGPP_LOG_LEVEL` (trace, debug, info, warn, error;
default info) sets every rank's log level.

## What a node needs

- The checkpoint in the Hugging Face cache (`--model ORG/NAME` resolves it
  per node) and the resident image cache under
  `~/.cache/dgpp/resident/<key>.img` (~82 GiB per rank; built on the first
  boot, streamed at the drive's line rate afterwards; `README.md`
  "Deploying a serving rank" has the knobs).
- Nothing privileged: the process tries `mlockall(MCL_CURRENT)` as a
  safety net and logs, rather than fails, when `RLIMIT_MEMLOCK` is finite
  (`DGPP_MLOCK=off` skips it); GPU clocks are left to the governor.
- The staging directory `/tmp/bus4` and the log directory live in `/tmp`:
  a reboot empties them, and `serve_run.sh up` recreates both.
- The bus rendezvous window is 120 s from rank 0's listener appearing; the
  peers connect within ~1 s of their launch, so the order `up` encodes
  (head first, then the peers at once) is the one that works. The
  journal star forms after the bus world.

## When a rank dies

There is no failover: any rank's death fails the service, quickly and
loudly, and the world is restarted (v1's failure semantics; built and
drilled 2026-09-05).

- **A peer dies.** Rank 0's journal watch sees the peer's connection close
  within ~100 ms (a peer never writes on the journal, so a readable
  connection is a close or a reset) and fails the service: every live
  stream gets the tokens the service had committed, then an
  `engine_failure` error event naming the dead rank and `[DONE]`;
  one-shots and later requests get 503 `engine_failure`; `/health` turns
  503 with the reason; rank 0 writes its op stream and exits with
  **status 2** once the answers are out (3 s grace). The other peers see
  rank 0's journal close — inside a tick, through their own watch — and
  exit with **status 3**. Measured on the fabric: rank 0 out 0.6 s after
  the kill, every rank gone within 3.2 s.
- **Rank 0 dies.** Clients see their connections close (nobody is left to
  write an event). The peers exit through the journal EOF (between ticks)
  or their in-tick watch (status 3) — every rank gone within 2.6 s in the
  drill.
- **A rank goes silent** (a node powered off: TCP does not close). The bus
  watchdog fails the in-flight collective after its deadline and throws
  into the same failure path.
- **Committed state is never touched.** The step in flight never completes
  on any rank, so nothing after the last completed step is committed
  anywhere, and every token a client received was committed on every
  rank. After a restart, the same prompt at temperature 0 reproduces the
  committed tokens as a prefix of its answer (the drill checks exactly
  this).

Restart with `scripts/serve_run.sh up` (it sweeps any stray process
first). A supervisor that restarts rank 0 on a nonzero exit and the peers
on theirs gets the same behavior unattended.

The drill: `scripts/serve_failure_drill.sh <victim rank> [clients]` boots
the world, streams from three clients, kills the victim with `kill -9` at
a random moment, and checks every claim above, then restarts and replays
the prompts. Its artifacts land under `build-ci/fabric-runs/failure_drill_*`.

## Checking that the world is healthy

- **At a glance, every 10 s:** each rank's log carries one aggregate line
  per interval while the world is busy (and one closing line of zeros
  when it goes quiet):

  ```
  stats: rank 0 over 10.1 s: prefill 7 prompts / 182 tok (18 tok/s; 360 ms avg, 13.83 ms/tok; 25 % of wall), cache saved 408 tok (5/7 hits); decode 64 steps / 477 tok (47 tok/s; 118.0 ms/step, 1.89 tok/step/req; 75 % of wall); running 4, queued 1; pool 63/65 blocks (97 %); prefix cache 38/43 entries; requests +7 (shed 0, cancelled 1)
  ```

  Prompt tokens are the ones the prefill computed (an attach skips the
  rest, shown as `cache saved`); `ms/step` is the decode pass's wall
  time (the 8-row MTP graph at four live requests runs ~120 ms, the
  single-request step ~42), `tok/step/req` is MTP's acceptance (1.0 at
  T=1, up to 2.0), and the two `% of wall` shares say where the engine
  thread's time went — prefill against decode; together they approach
  100 % when the engine is saturated. A peer's line carries the same
  counts with its own timings and no request counts. The per-tick lines
  (a line per generated token, the bus's three per-window lines, the
  prefix cache's per-decision line, the adaptive engine's mode switches)
  sit at DEBUG since 2026-09-06 — the one-hour soak had written 307,000
  lines (40 MB) per rank at INFO, none of them aggregate; the same load
  now writes ~130 lines per minute, the per-request admitted / retired /
  deferred / cancelled lines and the stats line. `DGPP_LOG_LEVEL=debug`
  restores the rest; `scripts/serve_pace.py` and `scripts/fabric_xrank.py`
  read those lines and need it.
- **Continuously:** every tick record on the journal carries rank 0's
  running fold of its op stream (`od`) and, with the prefix cache on, of
  its cache decisions (`pd`). A peer whose own fold differs dies at once
  with the tick number (`journal: op-stream divergence at tick N`), and
  rank 0 then fails the service as for any dead peer. A quietly diverged
  rank therefore cannot serve for more than one tick.
- **At shutdown:** `down`'s four md5s (the ritual). They are the same
  evidence, post-mortem.
- **Per request:** `GET /v1/metrics` — requests, sheds, cancellations,
  failures, the admission policy, the prefix cache (entries, hits, tokens
  saved, hop snapshots, the TTFT split by hit and miss), sampling
  fallbacks. `GET /health` is `{"status":"ok"}` while the engine lives.
- **Under load:** `scripts/serve_soak_run.sh MINUTES OUT_DIR` boots the
  world with the production knobs, starts `scripts/node_probe.sh` on every
  node, runs `scripts/serve_soak.py` (multi-turn chat, long generations,
  client cancellations, bursts above the queue bound), stops the world and
  prints the four op-stream md5s, the `STALLED` counts per rank log (the
  bus's stall witnesses — none in a healthy run) and each node's reclaim,
  swap and throttle sums; the soak's own summary gives TTFT and decode-pace
  percentiles per 10-minute window, the status counts and the prefix
  cache's line.
- **The other evidence rituals:** `scripts/fabric_prefill_repeat.sh OUT
  LEN...` (the steady-state prefill at each length with the four-way ids
  md5), `scripts/fabric_mtp_classes.sh OUT CLASS...` (MTP acceptance per
  prompt class), `scripts/serve_prefix_curve_sweep.sh OUT "GIB..." "C..."`
  (the prefix cache's capacity curve, one boot per point), and
  `scripts/serve_failure_drill.sh VICTIM` (the kill −9 drill).

## Ports and processes

| what | where |
|---|---|
| HTTP (rank 0) | 18080 |
| bus rendezvous | 29970 (rank 0 listens; peers connect) |
| admission journal | 29971 (rank 0 listens; peers connect and send `hello <rank>`) |
| peer binary and logs | `/tmp/bus4/glm_serve`, `/tmp/bus4/serve_r<rank>.log`, `/tmp/bus4/serve_rank<rank>.ops` |
| rank 0 log and pid | `$DGPP_SERVE_LOG/serve_r0.log`, `$DGPP_SERVE_LOG/r0.pid`; its op stream `serve_rank0.ops` in the repo root at exit |
| exit statuses | 0 orderly stop; 1 a startup or contract error; 2 rank 0 after an engine failure; 3 a peer released by its in-tick watch |
