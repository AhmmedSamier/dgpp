# Changelog

The history by milestone. The dated engineering record in
`benchmarks/results/` has every measurement and fix behind these lines;
`PLAN.md` has the milestones' exit gates.

## Unreleased

- The pipelined replay: the engine's step returns at the verify's verdict
  (a kernel node publishes the slot's replay sequence to pinned memory)
  and the next replay is launched before the previous one's draft tail
  and window are settled, so the verdict read, the scheduler's
  bookkeeping, the rolling prefix snapshot and `cudaGraphLaunch` itself
  (~0.5 ms at 1,168 nodes) hide behind the tail. The token feed is per
  slot on the device, the pick's masks ride a pinned staging behind an
  in-graph handshake, the bus holds two live replay windows (a ring of
  arms, FIFO finish, in-order adoption) and every graph shape has two
  alternating variants. Eager bus work drains the replays in flight.
  `DGPP_PIPELINE=0` restores the settle-after-launch step. Measured on
  the fabric (one greedy request, the same binary pipelined vs
  `DGPP_PIPELINE=0`): 41.2 vs 41.7 ms per pass at a short prompt, 41.9 vs
  43.0 at 8K, 42.6 vs 43.3 at 32K — 0.5 to 1.1 ms per step, 44.1 / 45.3 /
  43.7 vs 43.4 / 44.8 / 43.1 tok/s on prose / code / JSON (+1.1 to
  +1.6 %); the greedy transcripts are byte-identical to the earlier
  plain, depth-1 and depth-2 runs. Three loopback gates run their eager
  oracles through the drain; the bus test arms two windows at once.

- MTP draft depth (`engine.mtp_depth`, `--mtp-depth`, 1–3): the verify
  runs 1 + depth rows and the draft block's chained rows (one more block
  row per draft, on its own output) propose the drafts after the first.
  The sampled verdict kernel decides T rows in a chain (draw for draw
  what the eager sampled speculator decides), the host fallback continues
  the chain from the row that fell back, the tail ring is guarded around
  the chain rows, the hop snapshot takes the rows the step committed past
  the position, and the device picker carries one slot per draft. Depth 1
  is byte-for-byte the two-row step as built; past depth 1 every step is
  a scalar replay. New gates: the T=3 verdict oracle, the greedy depth-2
  and depth-3 loopback gates (plain transcript, feed == the eager chain),
  the sampled depth-2 loopback gate (transcripts == the eager depth-2
  speculator's through 17 fallbacks). On the fabric the greedy transcripts
  of three prompts are byte-identical plain / depth 1 / depth 2; depth 1
  is unchanged at 42–43 ms/step; depth 2 is 54–56 ms/step with the second
  draft standing 45 % (prose) to 65 % (code) of the time — −4 % on prose,
  +4 % on code and JSON — so depth 1 stays the default
  (`scripts/mtp_depth_check.py` runs the transcript check).
- Per-position draft acceptance: the engine counts, per slot and overall,
  the steps that verified each draft position and the steps in which it
  stood; the stats line's `mtp` group prints `accept p1 74 % p2 61 %` and
  every MTP retire line carries the request's own `accept p1 78 %`.

- The KV cache's dtype is a config key (`engine.kv_dtype`, `--kv-dtype`):
  `bf16`, `fp8` (e4m3 + a row scale) or `fp4` (e2m1 in blocks of 16 with
  e4m3 block scales); the attention kernels dequantize on the load, the
  index cache and the selection are unchanged, the format rides the
  settings record and the config digest. New unit and CUDA tests pin the
  codecs, the append and the three attention kernels per format.
- The memory plan: every rank itemizes every byte it will allocate for its
  configured shape and refuses to boot when it does not fit the node's
  free memory, before the first allocation, naming the largest items and
  the largest `kv_capacity` that would fit; `dgpp-serve --memory-plan`
  runs the check alone. The model's per-forward row bound is now the
  prefill chunk rather than the whole context (the activations no longer
  grow with `kv_capacity`: a 262k-token context needs ~98 GiB per rank
  instead of ~300 GiB and boots), `max_context()` is the context bound.
- Every line the server prints is timestamped: the step-timing report and
  the MoE chain dump go through the logger, an uncaught exception leaves a
  stamped line before the abort, and the launcher stamps its own lines in
  the same format. The stamps are UTC on every rank (a peer's log used to
  carry its node's own zone, seven hours off the head's).
- The throughput line leads with what an operator watches — `decode 30.9
  tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) |
  mtp 1.74 tok/step/req = 74 % drafts accepted | prefill … | live 0,
  queued 0 | pool … | prefix cache … | requests …` — and every request's
  retire line carries its own numbers: the prompt and cached tokens, the
  prefill's ms, the decode tokens and passes, tok/s, ms/tok, ms/pass and
  MTP's tok/pass.
- A non-strict tool argument keeps its type under a keyword that only
  narrows the value (`minimum`, `maxLength`, `pattern`, `format`, …): the
  bound is not enforced and one INFO line says so, once; a schema outside
  the subset in shape still leaves the value free, with one WARN line,
  once (the same three WARN lines used to repeat before every request of
  an agent client). Strict tools refuse as before.
- The settings record carries the resolved row-batch threshold, so the
  pushed and the effective config print the same `batchmin`.
- The decode select kernel (DSA, the step's only context-scaled kernel)
  rewritten: every block scores its stripe and publishes the keys plus a
  radix histogram, the last `rows` blocks each find one row's boundary key
  by radix refinement, gather, and expand with a rank sort — the same set
  under the key's total order, so the bitwise gates are unchanged.
  `dsa_select_bench` at the MTP shape: 4K / 8K / 32K / 393K tokens of
  context 572 / 608 / 832 / 3700 us before, 25 / 33 / 67 / 509 after; the
  12 calls per step were 7–10 ms of the 54 ms step at agent-sized contexts.
  The workspace is `dsa_select_workspace_bytes` (keys and histogram per
  row, sized for the pool capacity), counter_ws is two words.
- The MoE down projection runs four rows per warp (`block_rows_multi`): the
  sliced k of 512 bytes gave a warp one load in flight; the per-row chain
  is factored (`consume_chunk`) so every row is bitwise the old one.
  `moe_slot_bench` at the per-rank geometry: the two-row decode chain
  487 -> 457 us per layer (1.3 ms per step), the eight-row 1598 -> 1412.
- On the fabric (one request, sampled MTP): 41.8 / 44.4 / 44.5 / 45.2
  ms per step at 26 / 8K / 18K / 32K tokens of context, against 43.4 /
  53.6 / 54.8 / 56.2 before the two kernels — 23–25 ms per token at every
  context an agent client uses.
- The decode attention partial kernel keeps each thread's q window and
  c accumulator in registers instead of shared memory: the block drops
  from 95 KB to ~40 KB, two blocks per SM, the 64-block decode grid in one
  wave — 2.24 -> 1.16 ms per step on rank 0's trace at 8K, bitwise the
  smem form (each thread always owned exactly that window). The gather
  loop is unrolled so its token -> block-table -> row chains overlap. The
  split count stays 32 (48 and 64 measured the same step on the fabric).
- The L2 prefetcher coalesces adjacent adds into one launch
  (`DGPP_L2_PREFETCH_MERGE=off` restores one per tensor): the decode
  graph goes from 1,595 to 1,168 kernel nodes and `cudaGraphLaunch` from
  730 to 520 us per step (the host enqueues ~0.45 us per node and the GPU
  idles through it — the seam the design record called hidden). The
  merged kernels stream through more of each collective's handshake
  (+3 us per collective on the timeline), so the step itself is within
  noise of the per-tensor form; kept for the node count.
- The mHC dots kernel's two streaming loops unroll 8 (bitwise; ~0.1 ms
  per step). Tried and reverted, the reasons in the code: batching the
  bus fold's loads (the fold span stayed 9.3 us), and decoding q/k to
  floats in the select kernel (four times the smem bytes per pool).
- The launcher forwards every `DGPP_*` knob in the head's environment to
  the peers.
- The graph collective's fold reads staged copies: the placement gate's
  hash pass, which reads every claimed payload anyway, writes it into
  dynamic shared memory (48 KB for the decode's three 16 KB payloads;
  wider vectors fold from the NIC-placed rows as before), and the fold
  after the last arrival is a shared-memory pass — 9.3 -> 5.9 us per
  collective on the bus timeline, bitwise the same chain. Staging needs
  every peer slot 16-byte aligned (element counts a multiple of 8): the
  step's small all-gathers are not, and the first fabric boot found that
  as a misaligned address the loopback gates' widths never hit.
- The decode attention runs on the tensor-core listed kernel
  (`DGPP_DSA_DECODE_MMA=off` restores the register split kernel): two rows
  x 16 local heads is one M-block, each slab its row's own selection with
  its own request's block table. The DSA layer at the rank's shape:
  0.563 -> 0.516 ms per row per layer at 8K. Numerics: the mma summation
  order, tolerance-equal — the teacher-forced gate over the 556-token
  text passed at a mean NLL delta of +0.0008 nat (bound 0.02), perplexity
  2.587 vs 2.585, no move over 1 nat.
- On the fabric after both: 41.9 / 42.7 / 43.2 / 43.5 ms per step at 26 /
  8K / 18K / 32K tokens of context (400-token generations).
- `DGPP_BUS_TIMELINE=1` writes the bus's per-window collective timeline
  (copy / handshake / skew / fold per collective, the wait histogram, each
  peer's lag and how often it arrived last) at INFO without the rest of
  the debug output, which perturbs what it measures; the launcher forwards
  it, and the head's `DGPP_LOG_LEVEL`, to the peers (the peers used to
  expand the level on their own environment and always ran at info).

## 0.1.0 — 2026-09-06

Productionizing: the model-agnostic code left the `glm` namespace
(`dgpp::serve`, `dgpp::sched`, `dgpp::sample`, `dgpp::text`; GLM under
`src/models/glm/`; the binary is `dgpp-serve`); one cluster config
(`deploy/cluster.json`, a site's copy of `deploy/cluster.example.json`)
read by the launcher and every rank, the head
pushing the world's settings to the peers over the journal before anything
builds; versioned releases (`scripts/release.sh`) installed once per node
(`dgpp-cluster install`), the version stamped into the binary and refused
across a mixed world; `scripts/dgpp-cluster` as the launcher.

The request contract gained `stop`, `n`, `logit_bias`, and the usage's
`cached_tokens` and `reasoning_tokens`. Every rank's log carries one
aggregate throughput line per 10 s; the per-tick lines moved to DEBUG.

## 2026-09-05 — M9 closed; M7 built; M6 and M8 buttoned up

- M9: the continuous op-stream drift check on every journal record; a
  malformed-HTTP fuzzer under AddressSanitizer (three defects found and
  fixed); the one-hour mixed soak; the operator's page; the sign-off report
  (32K TTFT, MTP acceptance per class, the prefix cache's capacity curve);
  the ranked next-steps list.
- M7: the exact snapshot prefix cache — decisions rank-identical in the
  scheduler, the snapshot arena in the engines, hot == cold bitwise.
- The v1 failure semantics built, gated and drilled with kill −9 on the
  four nodes; the strict tool grammar's key ledger; cancellation under the
  one-graph step gated.
- Prefill rounds 4–9: the bulk collective as a cooperative kernel with paced
  senders, the MoE experts on grouped tensor-core kernels, the attention
  prefill as a flash kernel; 256 tokens 0.58 s, 2,048 tokens 1.7 s.

## 2026-09-04 — the contract fills in

Tool calls and reasoning on the wire; constrained decoding as a guarantee
for `tool_choice` and `parallel_tool_calls`; `response_format` with JSON
schemas through the same masks; typed tool arguments; drain-on-stop;
grow-on-demand admission; exact sampling on the service with the on-device
verdict and its gather fallback; the sampling-width sweep; prefill rounds
1–3.

## 2026-09-03 — M8: transactional MTP; the on-device step

Greedy speculative decode with the checkpoint's MTP layer as one graph
replay per step, transcript identical to plain decode (31.3 → 22.45 ms per
token); the pick behind the head on the device; the adaptive scalar and
row-batched graph variants that removed the low-occupancy regression; the
kernels-only decode graph that closed the batched-MTP loopback stall; the
teacher-forced log-probability gate; fp32 logits.

## 2026-09-01 to 2026-09-02 — M6: generation, tokenizer, service

The incremental decode engine; the byte-exact tokenizer and the Jinja
chat-template interpreter; the deterministic scheduler; the
OpenAI-compatible service at world 1, then on the fabric behind the
admission journal; the decode step as a recorded graph with the
collectives as graph nodes; eleven optimization rounds on the T=1 step
(393 → 31.45 ms per token); the one-pass resident load and the per-rank
image cache (boot 15–25 s).

## 2026-08-29 to 2026-08-31 — M5: four-rank tensor parallelism

The RC/RoCE CollectiveBus with its slot pools, dual-lane striping and
watchdogs; the epoch-based roster; the sharded resident load; the fabric
TP=4 parity gate at real dimensions; the resident serving mode.

## 2026-08-27 to 2026-08-28 — M0 to M4

Platform, topology, transport and checkpoint facts; the core runtime, the
safetensors loader and the synthetic graph testbed; the KDA operators and
state manager; DSA/MLA sparse attention with its index pools; the
assembled 45-layer GLM forward with per-layer parity against torch
references.
