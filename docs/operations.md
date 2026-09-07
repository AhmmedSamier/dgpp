# Operating the serving world

The four-node GLM service as it is run today: one `dgpp-serve` process per
node, rank 0 the only HTTP ingress, the peers following rank 0's admission
journal. This page collects what an operator needs — boot, stop, status,
the knobs, what happens when a rank dies, and how to check that the world
is healthy — with pointers to the design where the reasons live
(`DESIGN.md` §11 for the fabric protocol, `PLAN.md` M9 for the hardening
record).

## Boot, stop, status

The world is described by one file, `deploy/cluster.json` — the site's
copy of `deploy/cluster.example.json`, not tracked (every key, its default
and what it does: README's "Configuration" table) — and driven by one
launcher:

```
scripts/dgpp-cluster up       # stage the binary and the config to the peers, boot rank 0, then the peers; waits for READY
scripts/dgpp-cluster down     # SIGINT rank 0 (drain-on-stop), wait for the peers, fetch every rank's op stream and log, md5 the streams
scripts/dgpp-cluster status   # rank 0 alive? each peer's process count
```

Options: `--config FILE` (default `deploy/cluster.json`, or
`$DGPP_CLUSTER_CONFIG`), `--bin PATH` (default `build-ci/dgpp-serve`),
`--log-dir DIR` (overrides `paths.log_dir`), and `--knobs "FLAGS"`, which
appends flags to every rank's command line — flags override the file,
which is how the evidence scripts run their sweeps. `scripts/serve_run.sh
up|down|status` remains as a shim: `DGPP_SERVE_KNOBS` becomes `--knobs`
and `DGPP_SERVE_LOG` becomes `--log-dir`.

Every rank starts as `dgpp-serve --config <file> --rank R`: rank 0 takes
the model, the world (the node list's length), the ports and every engine
knob from the file; a peer takes its bootstrap (rank 0's address, the
journal port) and local paths from the file and everything else from
rank 0's settings record. `up` copies `dgpp-serve` and the
config to each peer's `paths.stage_dir` (the peers' op streams from an
earlier run are removed first so they can never be mistaken for this
run's evidence), boots rank 0 with its working directory in
`paths.log_dir` (`serve_r0.log`, `r0.pid`, and its op stream
`serve_rank0.ops` at exit), waits for its rendezvous listener, spawns the
peers in parallel by ssh, and waits for rank 0's `serve: listening` line.
Readiness is that line — with the resident image cache warm it takes
15–25 s; a first boot that builds the image from the checkpoint takes
~4.5 minutes.

`down` sends rank 0 SIGINT. Rank 0 drains: the door closes (new requests
get 503 `server_shutdown`), the in-flight requests are cancelled through
the journal at the next tick and their streams end with the shutdown error
event, then the stop record releases the peers. `down` waits up to 240 s
for rank 0 (a stop that lands mid-prefill is honored at the pass
boundary), then for the peers, then fetches `serve_rank{1,2,3}.ops` and
`serve_r{1,2,3}.log` from the peers into the log dir and prints the op
streams' md5s: **the four hashes must be identical** (the §11 op-stream
ritual; the launcher says so, or says DIFFER).

**Every line is timestamped.** Every line a rank writes to its log
carries the time in UTC to the millisecond (`2026-09-06 06:12:55.485 INFO
…`), including the step-timing report and the MoE chain dump instruments
and the line an uncaught exception leaves before the process aborts; the
launcher's own lines carry the same stamp so the two read side by side.
The only unstamped lines are the bus kernel's device-side stall
diagnostics (`BKFIN …`), which come off the GPU's printf.

**The head sends the settings.** Rank 0 opens the journal before it
builds anything, accepts the full world, and pushes a settings record —
the model, the world size, the fabric port and every engine knob that
shapes the op stream. A peer's own flags or file supply only the
bootstrap (rank 0's address, the journal port, its rank) and its local
paths; everything else it takes from that record, and it logs a WARN
naming both when its own values differed. So a knob given to one rank by
hand cannot make a different world: the peer runs what the head runs.
Each rank still logs the configuration it actually runs (`config: model=…
world=… … (digest …)`), rank 0 puts the digest on the warm record, and a
peer whose digest differs exits with status 1 — by construction this no
longer fires; it stays as the assertion that the push worked.

## Install, upgrade, roll back

`scripts/release.sh` builds the release preset and packs
`dist/dgpp-<version>.tar.zst` (README's "Release and install" has the
layout); `scripts/dgpp-cluster install <tarball>` copies it to every node
in the config, unpacks it under `paths.release_dir` and verifies every
file against `MANIFEST.sha256`. Which release runs is named — the
config's `release` key or `up --release <version>` — and `up` then runs
`<release_dir>/dgpp-<version>/bin/dgpp-serve` on every rank, staging only
the config. An upgrade is `down`, `install`, then `up` naming the new
version; a rollback is `up` naming the previous one, which is still
installed. `dgpp-cluster releases` shows what each node has;
`dgpp-serve --version` prints a binary's version, and rank 0 puts its
version on the journal's settings record so a peer of another version
exits before its first tick rather than form a mixed world.

## What a node needs

- The checkpoint in the Hugging Face cache (`--model ORG/NAME` resolves it
  per node) and the resident image cache (~82 GiB per rank, built on the
  first boot; the two sections below have the memory and cache details
  and knobs).
- Nothing privileged: no locked clocks, no memlock limit changes, no root
  (the memory section below says why).
- The peers' staging directory (`paths.stage_dir`, `/tmp/dgpp-stage` in
  the example) lives in `/tmp`: a reboot empties it, and
  `dgpp-cluster up` recreates it. The log dir (`paths.log_dir`,
  `~/dgpp/log`) persists.
- The journal star forms first: rank 0 listens on the journal at once and
  the peers connect to it (retrying within the rendezvous window) before
  any rank builds its model; the bus world forms after the builds, its
  connect retrying within the same 120 s window. `up` encodes that order:
  the head, then the peers as soon as the journal listens.

### Memory on a serving node

A resident rank owns its box. The model is ~82 GiB of the 121 GB, and the
serving apps (`dgpp-serve`, `glm_gen_check`) configure themselves for that
without any privileged setup on the node:

- before anything is allocated, every rank computes its **memory plan** —
  the resident weights, the KV cache pool, the draft block's per-position
  hidden cache, every activation and scratch buffer, the prefix cache
  arena and the engine's own buffers, from the same formulas the
  constructors use — logs it itemized, and refuses to boot (exit 1, the
  world never forms) when the plan plus 8 GiB of headroom exceeds the
  node's free memory. The refusal names the largest items and the largest
  `kv_capacity` the node would hold as configured. `dgpp-serve --config
  deploy/cluster.json --rank R --memory-plan` runs the check alone and
  exits 0 or 1. This is what a 262k-token context needed (2026-09-06: the
  old build sized every activation to the whole context and drove all
  four nodes into their memory watermark; a reboot was the only way out);
- the loader's own check that the resident footprint (+ 8 GiB headroom)
  fits the device's free memory remains as the second line and fails
  immediately with a clear message — never three minutes into a load;
- the loader reads each source tensor exactly once (prefetch, copy, drop),
  so the page cache stays under ~10 GB during the load and the box never
  reaches its memory watermark; the checkpoint's mmaps are released the
  moment the last layer is on the device (`GlmLayerStream::release_sources`);
- the process *tries* to lock its memory (`mlockall(MCL_CURRENT)`, before
  the model is constructed) as a safety net against swap-in faults in the
  decode loop. This is optional: with the one-pass loader a rank with the
  pin off measured identically (p99 46 ms, 0 stalls, no swap traffic over
  1000 steps). A finite `RLIMIT_MEMLOCK` is logged, not warned about;
  `DGPP_MLOCK=off` skips the attempt.

**What the context costs.** With the per-forward activations sized to the
prefill chunk (2,048 rows) rather than the context, the memory that grows
with `kv_capacity` is the KV cache pool (12 layers including the draft
block; 12.4 KiB per token in bf16, 6.2 KiB in fp8, 3.5 KiB in fp4, index
cache included) and the draft block's per-position hidden cache (8 KiB per
token per request slot: 32 KiB per token at `max_concurrency` 4), about
44 KiB per token all told at the production shape in bf16. On a 121 GB
node with the 81 GiB resident model that is roughly 490k tokens of
context; the plan line at boot says exactly.

**The draft depth** (`engine.mtp_depth`, `--mtp-depth`, 1–3, with `mtp`)
is the number of draft tokens verified per decode step. Depth 1 is the
two-row step: the pending token and one draft through the main stack, the
draft block proposing the next draft. A deeper step feeds 1 + depth rows,
and the block proposes the later drafts by running one more row per draft
on its own output (the single block's recursion — its hidden for the row
after the last accepted one is its own previous output, not the main
stack's; the KV and hidden it writes past the counter are provisional and
the next real rows overwrite them). The verdict, the commit and the
rollback are the same machinery over T rows; the sampled verdict tests
the drafts in order (a stand moves to the next row, a reject ends the
step on the residual token, the last row reached is sampled plainly) and
a host fallback continues the chain exactly as the device would have.
Measured on the fabric (2026-09-06, one greedy request, 300 tokens): the
step is 31.5 ms plain, 42–43 ms at depth 1, 54–56 ms at depth 2 — each
verify row is its own expert bytes (~10 ms), the chained block row
~2.5 ms — and the second draft stood 45 % of the time on prose, 56 % on
JSON, 65 % on code (the first: 78–87 %), for 2.21 / 2.41 / 2.51 tokens
per step against 1.80 / 1.81 / 1.88 at depth 1. So depth 2 is 41.7 vs
43.5 tok/s on prose (−4 %), 44.7 vs 43.1 on JSON and 46.5 vs 44.7 on
code (+4 %), and −4 % on an 8K-context summary; it pays when p1·p2
exceeds ~0.28·(1 + p1), about p2 > 0.63 at p1 0.8. Depth 1 stays the
default; the stats line's per-position acceptance says what a workload
would get. Past depth 1 every step is a scalar replay (the row-batched
graph is built for the two-row step only), so `max_concurrency` above
one serves requests round-robin per step.
Changing the depth changes the config digest; the transcript does not
change (a greedy request decodes the plain transcript at any depth, a
sampled one the same distribution — the loopback gates pin both).

**The KV cache's dtype** (`engine.kv_dtype`, `--kv-dtype`) chooses the
latent cache's storage format: `bf16` keeps the rows as the projection
left them (every parity gate's format); `fp8` stores e4m3 codes with one
fp32 scale per row (~2^-4 relative error per element); `fp4` stores e2m1
codes in blocks of 16 with an e4m3 scale per block over the row scale
(~2^-2 per element). The attention kernels dequantize a tile into bf16
shared memory as they gather it, so everything past the load is the bf16
kernel; the index cache, the tail rings and the selection are unchanged in
every format, so a quantized cache changes the attention values, never
which tokens are attended. The format is part of the world's settings (the
head pushes it, the config digest carries it) and every rank runs the same
one. The quantized formats are a memory trade an operator makes
deliberately: at 262k tokens they save 1.5 GiB (fp8) or 2.2 GiB (fp4) per
rank against a 98 GiB plan, and the model's answers change with them.

Nothing else on the node needs setting. In particular a locked GPU clock
(`nvidia-smi -lgc`) is **not** required: the governor sits at 2400-2560 MHz
throughout decode on its own and the measured step distribution is the same
locked or unlocked. NTP between nodes only matters for reading logs side by
side, and `scripts/fabric_run.sh --node-probe` records each node's clock
offset per run so even that works without it.

### The resident image cache

The first start of a resident rank builds its layers from the checkpoint
(slice, stage, dequantize, pack) and writes the finished device bytes to
`~/.cache/dgpp/resident/<key>.img` on that node (~82 GiB per rank for GLM;
the key covers the checkpoint's shard headers, `config.json`, world, rank,
head sharding and the loader's format version, so a stale image can never
load by accident). Every later start streams that image instead with
O_DIRECT reads at the drive's line rate — 15-25 s to a ready model against
~4.5 minutes from the checkpoint. The boot digest is cached beside it
(`<key>.digest`). Knobs:

```
DGPP_RESIDENT_CACHE=off            disable (always build from the checkpoint)
DGPP_RESIDENT_CACHE_DIR=/path      put the images somewhere else
DGPP_RESIDENT_CACHE_VERIFY=1       re-fold every blob on read (a pass over 82 GiB)
```

Delete the file to force a rebuild; the loader's log line says how many
layers were restored versus captured on each start.

`scripts/fabric_run.sh --node-probe` samples each node's reclaim/swap/GPU
counters at 1 Hz for the run; `scripts/fabric_xrank.py LOGDIR` reads the
fetched logs and reports host gaps, stall windows, and step distributions per
rank.

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

Restart with `scripts/dgpp-cluster up` (it sweeps any stray process
first). There are no boot-time units by decision: the servers start when
an operator, or whatever the operator runs, says `up`.

The drill: `scripts/serve_failure_drill.sh <victim rank> [clients]` boots
the world, streams from three clients, kills the victim with `kill -9` at
a random moment, and checks every claim above, then restarts and replays
the prompts. Its artifacts land under `build-ci/fabric-runs/failure_drill_*`.

## Checking that the world is healthy

- **At a glance, every 10 s:** each rank's log carries one aggregate line
  per interval while the world is busy (and one closing line of zeros
  when it goes quiet):

  ```
  stats: rank 0 | 10.0 s | decode 30.9 tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) | mtp 1.74 tok/step/req, accept p1 74 % | prefill 1 prompt / 4515 tok, 305 tok/s, 2.36 ms/tok, 10637 ms avg (72 % of wall), 13236 tok cached (1/1 hit) | live 1, queued 0 | pool 498/3072 blocks (16 %) | prefix cache 17/43 entries | requests +1 (shed 0, cancelled 0)
  ```

  Decode comes first. `tok/s` is what the interval delivered (idle time
  included); `ms/tok` is the pace while decoding (step time over the
  tokens generated); `ms/step` is the decode pass's time as the engine
  saw it — ~41 ms for one request under 2K tokens of context, 42–43 ms
  from 8K to 32K after the 2026-09-06 work (the select kernel's rewrite:
  it was 52–56 at 8K–32K; the tensor-core attention; the pipelined
  replay, which also makes this the verdict-to-verdict interval), ~120 ms
  for the 8-row MTP graph at four live requests; the
  `mtp` group is MTP's yield — tokens per request-step (1.0 to 1 + depth)
  — and the measured acceptance of each draft position over the interval
  (`accept p1 74 % p2 61 %`: the share of steps in which the first draft
  stood, and in which the second stood after it; at depth 1 only `p1`).
  Prefill's tokens are the ones computed (an attach skips the rest,
  `cached`), and the two `% of wall` shares say where the engine thread's
  time went — together they approach 100 % when it is saturated. A peer's
  line carries the same counts with its own timings and no request
  counts. Every retire line carries the request's own numbers —
  `retired (eos): 327 tok in 35.7 s — prefill 7995 tok (0 cached) in 5662
  ms; decode 326 tok / 183 passes in 30.0 s: 10.9 tok/s, 92.1 ms/tok, 164
  ms/pass, 1.78 tok/pass` — where a pass is shared with every other live
  request, so `ms/pass` is the pace that request saw (the clock starts at
  admission; the queue wait is the service's `ttft` in `/v1/metrics`). A
  prefix cache miss is explained on its own INFO line at admission
  (2026-09-07): `prefix cache miss — 41 cut(s) probed against 42 entries;
  the nearest entry (position 62432) shares the first 812 of the prompt's
  64803 tokens (the prompt changed there)` — a divergence inside the
  system prompt (an agent client injecting a saved memory, a compacted
  history) reads differently from one at the previous answer; "the cache
  is empty", "a different prompt from its first token" and "a prefix of
  that entry: no entry at this prompt's own cuts" name the other cases,
  and when the conversation's own entry was pushed out the line says so
  instead — `an entry at this prompt's cut 2896 was evicted (5
  eviction(s) ago, last used at tick 1180, now tick 1412; the arena holds
  7 slots)` — from a ring of the last 256 evicted entries' prefix hashes,
  which is what separates an arena too small for the streams and their
  side requests from a client that changed its prompt. The decisions
  themselves (attach, snapshot, close,
  evict, rolling, hop) ride the op stream `down` fetches, one `X <op> <id>
  <position> <slot>` line each, so the arena's contents at any request can
  be replayed after the fact.
  The per-tick lines
  (a line per generated token, the bus's three per-window lines, the
  prefix cache's per-decision line, the adaptive engine's mode switches)
  sit at DEBUG since 2026-09-06 (`DGPP_BUS_TIMELINE=1` brings the bus's
  three per-window lines back at INFO on their own — the per-tick lines
  perturb the collective skew they measure; `DGPP_PIPELINE=0` turns the
  pipelined replay off — every replay settles right after its launch, the
  pre-2026-09-06 step — `DGPP_PIPELINE_TRACE=1` logs each launch and
  settle, and `DGPP_SYNC_EAGER=1` makes an eager row — a prefill chunk,
  the sampled fallback's verify and re-draft — synchronize after every
  stage and validate its selection list before the attention, naming the
  stage a fault came from; the fault hunt's knob, not for serving) — the one-hour soak had written 307,000
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
  (the prefix cache's capacity curve, one boot per point),
  `scripts/serve_failure_drill.sh VICTIM` (the kill −9 drill), and
  `scripts/serve_api_check.py HOST PORT` (the request fields — `stop`,
  `n`, `logit_bias`, the usage details — against a running world).

## Ports and processes

| what | where |
|---|---|
| HTTP (rank 0) | 18080 |
| bus rendezvous | 29970 (rank 0 listens; peers connect) |
| admission journal | 29971 (rank 0 listens; peers connect and send `hello <rank>`) |
| peer binary, config and logs | `<stage_dir>/dgpp-serve`, `<stage_dir>/cluster.json`, `<stage_dir>/serve_r<rank>.log`, `<stage_dir>/serve_rank<rank>.ops` (fetched into the log dir by `down`) |
| rank 0 log, pid and op stream | `<log_dir>/serve_r0.log`, `<log_dir>/r0.pid`, `<log_dir>/serve_rank0.ops` at exit |
| exit statuses | 0 orderly stop; 1 a startup or contract error (a configuration that differs from rank 0's included); 2 rank 0 after an engine failure; 3 a peer released by its in-tick watch |
