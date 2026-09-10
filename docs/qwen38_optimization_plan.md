# Qwen3.8-Flash-Next — the single-stream optimization plan (2026-09-10)

Scope: make one request's decode and prefill faster at world 4 (the served
geometry, `deploy/cluster_qwen.json`), without regressing concurrent
sessions and **without quantizing any weight**. Everything below was
measured today on the committed tree (a8a2664) unless it says otherwise;
the runs are under `build-ci/fabric-runs/*_2026-09-10/`.

The companion documents are `docs/qwen38_flash_next_plan.md` (the port, the
cost model, the Q8 rounds of 2026-09-09) and `docs/measurements.md`.

## 0. Where it stands

The A/B is one binary with `DGPP_QWEN_GR_GATE_SIDE` flipped, two 200-token
readings each, back to back:

| reading | gates in the chain | gates early (§3.1) |
|---|---:|---:|
| T=1 decode, ms/step | 22.12, 22.11 | **21.45, 21.40** |
| MTP decode, ms/step (rank 0 stats) | 26.5 | **25.7** |
| MTP decode, ms/token, sampled | 17.9 | **17.2** |
| MTP decode, ms/token, greedy (c=1) | 13.9 | **12.9** |
| four live requests, aggregate tok/s | 114.6 | **120.0** |
| prefill, ms/token at 2 048 | 0.57–0.59 | unchanged |

For scale: on 2026-09-09 the same T=1 step read 22.0 ms and the MTP pass
26–27 ms.

**The handover binary (evening, §3.1 + §5.1, the inject order restored):**
62 / 62 tests; T=1 greedy transcripts 4 of 4 identical to the morning's
reference, 21.50 ms/step; the MTP world at the sampled default 25.8–25.9
ms/pass at 60–66 % acceptance, 1.60–1.66 tokens per pass, **15.5–16.1
ms/token** (17.0–18.3 at the day's start); the greedy MTP path 12.9
ms/token; four live requests 119.7 tok/s.

Two readings of the same configuration drift ~2 % across the day on this
box, so single runs are quoted as ranges and every A/B below is a pair of
runs on one binary with one knob between them.

Concurrency moves with the change, not against it (`scripts/serve_load.py`,
greedy, MTP world, aggregate decode tok/s): c=1 68.2 vs 66.4, c=2 102.5 vs
95.4, c=4 120.0 vs 114.6, on versus off. The greedy transcripts are
identical across the knob on all four prompt classes.

The one change landed today (§3.1) is bitwise, and it is in the working
tree, not committed. The rest of this document is the ranked list of what is
left, with what each item is worth and what it costs — including two things
that were built, measured and reverted (§3.2, §4.1), which are worth as much
to the next session as the item that stayed.

One number frames the whole speculation section: **acceptance is 72–74 % on
greedy requests and 48–55 % on the sampled default** (same build, same day).
That gap is not the draft head's quality — it is what a deterministic draft
costs under temperature sampling, and §5.1 is how to get it back.

## 1. The floors, and the exchange rate between them

`micro_mem_bw` on this GB10 (48 SMs): **read 233.6 GB/s**, write 196.6,
copy 216.2. That is the line rate every "how much headroom" question below
is measured against.

**The byte floor.** At TP=4 one decode token streams 3.83 GB per rank
(plan §2.2). At 233.6 GB/s that is **16.4 ms** — the fastest a T=1 step can
be while the weights stay BF16/FP8 as shipped. Today's 21.4 ms is 77 % of
line rate.

**The chain floor.** The step is one serial chain of ~1 400 kernels. Nsight
Systems over 441 steps: some kernel is running 95.4 % of the wall, the GPU
is **fully idle 1.03 ms per step**, and there is no bandwidth-moving kernel
running for **2.43 ms** of it (that time is collectives and the small
latency kernels). The prefetcher covers 11.7 ms of the 22.3 ms step, so
most consuming GEMVs read L2, not DRAM: `gr_act_up` moves 6.55 MB in 10.4 us
(630 GB/s), the GR down 6.55 MB in 18.9 us (347 GB/s), both above DRAM rate.
**The step is chain-bound, not bandwidth-bound**, and the two floors meet
around 18–19 ms: shorten the chain past that and the prefetcher can no
longer keep DRAM ahead of it at a rate the collectives tolerate (§6, the
prefetch sweep).

**T=1 is now at the bandwidth wall — measured, not inferred.** After
§3.1 removed ~2 ms of chain time the step moved only ~0.7 ms, and two
further chain cuts moved it not at all: retiming the inject prefetch
(21.59–21.72 ms) and putting the inject dots on the 16-byte-chunk GEMV core
(21.48–21.50 ms, against 21.40–21.45 before either). 3.83 GB per step in
21.4 ms is 179 GB/s, and the prefetch sweep (§6) says this box will not
sustain more beside 99 latency-critical collectives. So at T=1 every
remaining chain item (§3.3, §3.4) is worth ~0, and the levers that remain
are the ones that divide the bytes by more tokens (§5) or move fewer
collectives. The MTP pass, at 3.9 GB in 25.7 ms (152 GB/s), still has ~2
ms of chain slack, so chain work still pays there — but only there.

**The exchange rate.** One boundary collective costs 34.2 us at world 4
(measured, §2.3). At line rate 34.2 us moves **8 MB**. Any re-parallelization
that adds a round trip per site must therefore save more than 8 MB per site
to break even — the single number that settles the GR-slicing, expert-
parallel and replicated-attention questions (§5).

## 2. Where the time goes (measured 2026-09-10)

### 2.1 T=1 decode, world 4

`scripts/fabric_qwen_profile.sh deploy/cluster_qwen_t1.json OUT`
(`qwen_profile_t1_2026-09-10/`): 22.27 ms/step wall, 1 403 kernels/step.

| kernel | ms/step | n/step | us each | what it is |
|---|---:|---:|---:|---|
| `l2_prefetch_kernel` | 11.87 | 148 | 80.2 | the DRAM engine, on the side stream (overlapped) |
| `bus_allreduce_graph_kernel` | 3.55 | 99 | 35.8 | the boundary collectives |
| `bf16_gemv_kernel<1,0>` | 2.71 | 98 | 27.6 | GDN out, QSA q/k/v/o |
| `bf16_gemv_multi_kernel<1,0>` | 2.44 | 36 | 67.8 | the GDN's four input projections |
| `moe_slot_gate_up_swiglu` | 2.04 | 48 | 42.6 | the routed experts' gate/up |
| `combine_dots_kernel` | 2.02 | 96 | 21.0 | the GR inject dots — **removed in §3.1** |
| `gr_norm_down_kernel` | 1.83 | 97 | 18.9 | the GR mix's down GEMV + group norms |
| `bf16_gemv_kernel<1,1>` | 1.31 | 1 | 1 307 | the lm head slice (318 MB at 243 GB/s) |
| `moe_slot_down_narrow` | 1.05 | 48 | 22.0 | the experts' down |
| `gr_act_up_kernel` | 1.01 | 97 | 10.4 | the GR mix's up GEMV |
| `moe_router_dots` | 0.77 | 48 | 16.0 | 2.6 MB of router per layer |
| everything else | ~1.5 | | | GDN recurrence, QSA select/attention, norms, sampling |
| GPU fully idle | 1.03 | | | the launch seam |

### 2.2 MTP decode, world 4 (the served mode)

`qwen_profile_mtp_2026-09-10/`: 27.09 ms/pass, 1 719 kernels, 1.55 tokens
per pass, **acceptance p1 55 %**, 16.5 ms/token. Against T=1 the pass adds
one verify row: the expert slot kernels grow 1.5–1.65× (42.6 → 70.3 us,
22.0 → 33.9 us), the GR mix leaves the fused one-row path for the GEMV
chain (`bf16_gemv_kernel<2,0>` 305 launches, 5.06 ms), the lm head runs
**twice** (verify rows and the draft, 1.31 ms each), and the collectives
and the prefetch cost the same as at one row.

### 2.3 The collectives

`DGPP_BUS_TIMELINE=1` + `scripts/bus_window_skew.py`
(`qwen_bustl_2026-09-10/`), per collective at world 4:

| rank | total | copy | handshake | skew | fold | engine post |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 34.2 | 2.9 | 11.8 | 16.6 | 3.3 | 2.1 |
| 1 | 32.1 | 2.9 | 11.0 | 14.6 | 3.4 | 2.2 |
| 2 | 37.7 | 2.9 | 13.6 | 17.3 | 3.3 | 2.2 |
| 3 | 37.5 | 2.9 | 14.9 | 16.4 | 3.3 | 2.2 |

**Skew is the largest term** — the wait for the last peer, not the wire.
The host's post pass is 2.1 us, so the GPU→host→NIC hop is not the cost.
Compute between generations is 180 us, so the ranks drift ~9 % of a layer.

### 2.4 Prefill, world 4

`qwen_profile_prefill_2026-09-10/` (0.58–0.60 ms/token at ~1 985 tokens,
1 159–1 190 ms; shares of 2 719 ms of GPU time over three prefills):

| class | share | note |
|---|---:|---|
| routed experts (`moe_grouped_mma_fp8_ldm`) | 25.3 % | gate/up 370 us, down 669 us per launch for the same FLOPs |
| collectives (bulk 13.7 % + latency 4.2 %) | 17.9 % | 96 folds of [T, 2560] bf16 per chunk |
| GR elementwise (norm, mix_finish, combine ×2) | 12.5 % | six to eight passes over a 42 MB hyper state per site |
| dense projections (cuBLASLt) | 8.7 % | |
| QSA attention (`attn_partial`) | 8.0 % | 1.63 ms per launch |
| GDN recurrence + conv | 7.6 % | token-sequential |
| MoE accumulate + round | 5.7 % | |
| router + index/select | 4.1 % | |

**Every chunk streams every expert.** 2 048 tokens × top-10 covers all 512
experts, so a chunk reads 630 MB per layer per rank = 30 GB — 130 ms at line
rate, ~230 ms as the grouped kernel runs it (689 ms of expert MMA over the
burst's three prefills). A prompt is cut into 2 048-token chunks
(`kPrefillChunkTokens`), so an 8 K prompt pays that four times — see §4.1
for why making the chunk bigger does not, on its own, help.

## 3. The plan — decode

### 3.1 LANDED: the GR inject dots off the chain (−0.69 ms T=1, −0.8 ms MTP)

The combine's gates are `2·bf16(sigmoid(bf16(W_inj · Rn)/hc))` — a function
of `Rn` alone, which the mix computes at the *top* of the site, while the
kernel sat at the *bottom*, between the boundary collective and the apply:
one block per row walking a 10 240-deep chain, 21 us × 96 sites.

Two forms, one knob (`DGPP_QWEN_GR_GATE_SIDE=off` restores the old chain):

* **one row (T=1 decode):** the dots ride the mix's down GEMV as one extra
  block of the same launch, reading the normalized row out of the staged
  copy every block already holds — no extra global traffic at all. Bitwise:
  the same lane-strided chain, the same batch depth, the same xor tree, over
  the same values (`qwen_gr_test` pins the folded gates against
  `qwen_gr_combine_dots_bf16` bit for bit, at 1 and 3 rows).
* **more rows (MTP, batches):** the same kernel is forked onto a
  lowest-priority side stream at mix time and joined before the apply.

Fabric (the table in §0): T=1 22.12 → 21.43 ms/step, MTP 26.5 → 25.7
ms/pass, four live requests 114.6 → 120.0 tok/s, greedy transcripts
identical across the knob on all four prompt classes.

Note what the profile of the side-stream-only form says about overlap on
this box: the dots left the chain (−2.6 ms of chain time) but the kernels
they now run beside slowed down — `bf16_gemv_multi` 67.8 → 84.3 us,
`gr_act_up` 10.4 → 16.5 us — and only −0.75 ms survived. The folded form
gives that back because it adds no traffic at all, one block reading shared
memory. **Overlapping a latency-bound kernel with the chain on a machine
whose memory system is already the constraint returns about a third of the
kernel's time; folding it into a launch that already holds its inputs
returns all of it.** Worth remembering for §3.4.

Numbers per kernel, before and after, are in
`build-ci/fabric-runs/qwen_profile_t1_2026-09-10/` and
`qwen_profile_gateon_2026-09-10/`.

### 3.2 TRIED, and it does not pay: the fused mix at two-plus rows

At `tokens > 1` the mix falls back to `group_rmsnorm` + two GEMV launches
(3.75 ms per MTP pass against the fused path's 2.84 at one row) because the
fused down kernel recomputed each row's four group norms in sequence inside
every block. That was fixed — all `hc` sums of squares of a row now issue
together, each keeping block_sum_squares' own per-thread chain, butterfly
and warp-index sum, so every total stays bit-identical — and the fused path
was opened to every row count.

It is a loss anyway: MTP 26.4 vs 25.6 ms/step, and four live requests
101.9 vs 119.7 aggregate tok/s. The reason is shared memory, not the norms:
the fused kernel stages `kRows` normalized rows at 20 KB each, so from two
rows up it runs one block per SM and the down GEMV loses the occupancy the
chain path keeps by leaving Rn in global memory. Reverted;
`DGPP_QWEN_GR_FUSED=all` re-enables it for anyone who wants another look.
The multi-row sites keep §3.1's side stream instead.

What is left of the idea: the MTP pass spends 5.06 ms in
`bf16_gemv_kernel<2,0>` (305 launches), ~3.2 ms of it the GR site's two
GEMVs, which is bandwidth, not waste. Do not spend more time here.

### 3.3 Speculative launch of replay N+1 (−0.7 to −1.0 ms/step)

The GPU is fully idle 1.03 ms per T=1 step, 0.78 ms of it in the single gap
after the commit kernel (the host reads the verdict, does its bookkeeping,
and enqueues ~1 400 nodes at ~0.45 us each). The engine already pipelines
the settle; what is left is launching N+1 before N's verdict and rolling
back a bogus step (KV blocks, GDN/PLE/context state, the close snapshot).
This is the protocol project the 2026-09-06 note describes; it pays on both
model families and it is the only remaining structural item at T=1.
`cudaGraphUpload` of the other parity was tried on 2026-09-09 and removed
(no gain on Qwen, a loss on GLM).

### 3.4 Fewer nodes on the chain (−0.3 to −0.5 ms/step)

1 409 kernels per step, and the average inter-kernel gap is ~0.7 us.
Candidates, each bitwise and each also shrinking the host enqueue §3.3
attacks: `mix_finish` into the up GEMV's epilogue (97/step), `gate_out` and
`norm_rope` into their producers (36–48/step), `kv_append` into the QSA
projection's epilogue (12/step), `moe_slot_accum_routed_f32` into the down
kernel's epilogue (48/step).

### 3.5 The collectives' skew (−0.3 to −0.8 ms/step, diagnostic first)

16.6 us of the 34.2 us collective is waiting for the last peer, ×99 per
step ≈ 1.6 ms. Ranks 2 and 3 also carry 2–3 us more handshake than ranks 0
and 1, which is a fabric-placement question, not a kernel one. Worth a
bounded diagnostic round: clocks and thermals per node, the post thread's
affinity, and whether the host's per-stripe `bus_fold64` of the payload
(three 5 KB folds per collective, on the critical path between the kernel's
`ready_bits` and the post) can be replaced by the hash the kernel already
computes. Do not restructure the collective on speculation — measure first
with `scripts/bus_window_skew.py`.

### 3.6 What the profile says NOT to chase at T=1

The dense GEMVs are already at or above line rate (the lm head slice moves
318 MB in 1.31 ms = 243 GB/s; `bf16_gemv<1,0>` ~254 GB/s effective with L2
help). The expert slot kernels sit at ~190 GB/s where their rows are 160
FP8 bytes. Pushing the prefetcher harder is a measured loss (§5).

## 4. The plan — prefill

### 4.1 MEASURED: the bigger chunk is worth 3.6 %, not 14 % (the sweep kernel stays)

Every prefill chunk touches all 512 experts of all 48 layers, so the byte
model said an 8 K prompt in four 2 048-token chunks reads the expert
weights four times over, and that one 8 192-token chunk — once the grouped
kernel stopped re-reading its weight tile per m-tile — would take the
expert share from ~920 ms toward ~250. Both halves were built and
measured on 2026-09-10:

* **The m-sweep kernel** (`moe_grouped_mma_fp8_ldm_sweep_kernel`,
  `DGPP_MOE_FP8_SWEEP=on` to run it): one block per (segment, n-tile,
  group of four m-tiles), each k-stage's weight tile copied once and every
  live m-tile run under it; three 27 KB stages, one block per SM, 193–202
  registers, no spills; dispatched only when some expert holds more than
  64 rows, so the 2 K-chunk path is untouched. Bitwise the reference tile
  kernel in `moe_tile_bench` (uniform and ragged), `glm_moe_test` and
  `qwen_moe_test` green. On the bench at 228 rows per expert one gate
  launch goes 6.87 → 5.87 ms (87 → 103 GB/s of weights) — only −15 %,
  because at four m-tiles the kernel is issue-bound, not weight-bound: 47
  TFLOP/s against a ~236 dense peak.
* **The 8 192-token chunk** (memory plan 45 → 62 GiB per rank), the 8 K
  prefill probe, medians: 4 324 ms with four 2 048 chunks → **4 169 ms**
  with one 8 192 chunk and the sweep (−3.6 %), 4 249 ms without it
  (−1.7 %). The 2 K prefill is unchanged either way (1 136–1 152 ms).

So the chunk buys 3.6 % of an 8 K prefill for 17 GiB per rank and a
four-second non-preemptible unit; the constant goes back to 2 048 and no
chunk policy is built. The sweep kernel stays in the tree with its bench
gate and tests but **off by default** (`DGPP_MOE_FP8_SWEEP=on`): at the
production chunk ragged routing hands it launches where one hot expert has
two tiles and the rest have one, and its single block per SM hides less
latency there than the one-tile kernel's two — GLM's 512-token prefill
read 569 vs 544 ms with it on, the 8 K one 6 330 vs 6 260. The lesson
for prefill's "line rate": the expert kernel's binding limit at prefill
widths is the tensor-core issue rate, not DRAM — ~20 % of dense peak with
one block per SM on `mma.sync` — and that, plus the fold/compute
wavefront (§4.6), is where the prefill headroom actually is.

### 4.2 The head on the rows that need it (frees ~0.5 GB per 2 K chunk)

The prefill runs `lm_head` over **every** row of the chunk — 650 GFLOP and
508 MB of fp32 logits at T = 2 048, 7.9 ms per chunk — because the
diagnostic forward compares whole-chunk logits. Production needs the last
row (plus the MTP draft rows). Small on its own; it is what makes §4.1
affordable in memory.

### 4.3 Fuse the GR site's elementwise passes (−5 to −8 % of prefill)

The hyper state is [T, 10 240] bf16 = 42 MB at 2 K, and one site touches it
six to eight times: the norm reads and writes it, `mix_finish` reads the
logits and Rn, the inject dots read Rn again, the apply reads and writes R.
At 96 sites that is ~32 GB per chunk, ~137 ms at line rate — and the
measured 12.5 % (~149 ms) says these passes are already at line rate, so
the only lever is touching memory fewer times: `mix_finish` into the up
GEMM's epilogue, the inject dots into the norm pass (the prefill analogue
of §3.1), the apply into the next norm's read.

### 4.4 The expert down kernel at k = 160 (−4 to −6 %)

`moe_grouped_mma_fp8_ldm_kernel<float>` costs 669 us per launch against the
gate/up kernel's 370 us for the same FLOPs: a 160-deep k is five 32-deep
cp.async stages, so prologue and epilogue dominate. A tile shape written for
that k (or a split-k with the accumulate in the epilogue, §4.5) recovers
most of the difference. The same shape question already cost a revert at
decode on 2026-09-09 — write the oracle first.

### 4.5 The ordered accumulate: ~1 %, not 3 % (re-estimated 2026-09-10)

What the accumulate reads is the routed experts' per-slot down outputs —
fp32 [T·K, H] = 20 480 × 2 560 × 4 B = **210 MB per layer at 2 K tokens**,
written by the down MMA and read back by `moe_accum_ordered`, which sums
each token's ten slots in ascending expert id (the chain's order, the
reason the outputs are per slot at all). A token's slots sit in ten
different expert segments, i.e. ten different blocks, so an epilogue cannot
produce the ordered sum without a second pass — the fusion the 3 % assumed
is not available bitwise. What is available: `launch_moe_accum` (the shared
branch's fma) and `moe_round_bf16` into one kernel, with the shared gate's
sigmoid folded in — two launches and one 21 MB fp32 pass per layer, ~1 % of
the prefill. Below the run-to-run noise; do it when touching the tail
anyway, not as its own item.

**§4.3 scoped (2026-09-10 evening) — no clean bitwise route.** The two
fusions that would matter each need a cross-tile reduction: `mix_finish`
into the up GEMM's epilogue wants columns j, H+j, 2H+j, 3H+j of one row,
which sit in four different n-tiles; the inject dots into the norm pass
want the whole normalized row, which the per-(row, group) norm blocks never
hold. Either is a custom tensor-core GEMM with a two-stage epilogue for
~1.5 % of the prefill. Not taken.

**The collective's in-kernel copy (§3.5's compute-side item) — scoped.**
The 2.9 us copy of the source into the generation's staging row could be
the producing GEMV's epilogue, but the ring row is `(gen − 1) % ring` for a
`gen` the collective kernel reads from its control cell at launch; the
producer would have to read the same cell one node earlier, and both
decode paths are at the bandwidth wall where 0.3 ms of chain does not show.
Not taken.

### 4.6 Overlap the folds with compute (−10 to −13 %, large) — scoped

Collectives are 17.9 % of prefill (`bus_bulk_collective_kernel` 13.7 %,
about five launches of 260 us per fold, plus the latency-class kernel).
The bulk path is near its wire rate, so the lever is overlap, not speed.

The dependency structure allows it (worked through 2026-09-10 evening): a
chunk's rows split into blocks; within a layer, row block i+1's attention
needs block i's K/V of the same layer (causal), which block i writes before
its fold; the GDN recurrence is sequential across blocks and runs in order;
the MoE and the GR sites are row-local. So block i's fold can run beside
block i+1's layer compute — a pipeline diagonal over (layer, row block) on
two streams, the fold's wait placed before block i's combine.

Two things must be measured before building it: how much of the bulk
kernel's 1.3 ms per fold is spin on the network against copy and fold
work (the prefill log carries no per-fold split; the graph windows'
timeline does not cover the bulk path — instrument it first), since only
the spin overlaps for free; and how much of a cooperative-grid collective
actually co-schedules beside the expert MMA at one or two blocks per SM.
If the spin is most of it, the −10 to −13 % stands; if not, the item
shrinks toward the network's own share. Largest remaining prefill item
either way; a multi-day restructuring of `run_rows`.

### 4.7 Tensor-core forms for attention and the recurrence (−5 % each, large)

`attn_partial` (8.0 %, 1.63 ms per launch) and the GDN recurrence (7.6 %,
token-sequential) are both warp-level kernels at prefill widths. The chunked
WY form of the recurrence and an MMA form of the QSA prefill attention are
the standard answers; both are real projects with their own oracles.

## 5. Speculation — the per-token lever

### 5.1 LANDED: the draft is sampled, the verify uses the ratio (17.0–18.3 → 15.3–15.8 ms/token)

The evidence that motivated it, on one build: **acceptance was 72–74 % on
greedy requests and 41–55 % on the sampled default** (temperature 1.0,
top-k 20, top-p 0.95 — the model's own generation config), with the same
draft head in both. The gap is structural, not a quality problem: the draft
was the MTP head's argmax and the verifier accepted it "iff u1 < P(draft)",
so under temperature the acceptance rate was exactly E[P(argmax)] — about
half, because that is the mass the target's mode carries at T = 1. No draft
head, however good, moves that number.

What landed (`DGPP_SPEC_PROPOSAL=off` restores the argmax draft):

* **The draft pick is a sampling pick.** It draws from the draft head's own
  final set under the request's temperature/top-k/top-p (no penalties, no
  mask, no bias — any proposal is exact, so the cheapest one that resembles
  P is the right one), on its own RNG stream (`seed ^ kSampleDraftSeedMix`,
  the request's counter): independent of the verify's u1/u2, never
  advancing the request's counter, no state to carry across a fallback.
* **The final set travels with the draft** (`DraftProposal`, up to 64 ids
  and masses per slot and draft position, device plus a pinned mirror),
  written by the draft verdict and read by the next step's verify. A set
  wider than the buffer, a re-drafted row (the host's fallback path clears
  the slot) or a proposal whose token is not the fed draft all mean "no
  proposal", and then the old rule applies — which is exact for ANY draft;
  only the acceptance rate differs.
* **Row 0 accepts with min(1, P/Q) and resamples the (P − Q)⁺ residual.**
  The masses are gathered onto the merged candidate list in parallel
  before the scalar decision; the host oracle (`sample::Proposal`,
  `spec_select_from_sorted`) mirrors the device bit for bit
  (`glm_pick_test`: the same accept, token and draw count on every rank,
  and a mismatched proposal ignored), and `sampler_test` shows the emitted
  marginal is P while the acceptance is 1 − TV(P, Q). The chained drafts of
  depth ≥ 2 keep the plain rule.

Fabric, MTP world, `serve_bench` at the sampled default, 400 tokens, two
requests each way:

| | proposal off | proposal on |
|---|---:|---:|
| acceptance p1 | 41 %, 51 % | **63 %, 68 %** |
| tokens per pass | 1.41, 1.51 | **1.63, 1.69** |
| ms per pass | 25.7 | 25.8 |
| ms per token | 17.0, 18.3 | **15.3, 15.8** |

The pass costs the same (the sampling draft pick replaces a greedy one, ~0.1
ms), so the whole gain is tokens per pass. Greedy requests are untouched:
at temperature 0 the draft is the argmax, no proposal is written, and the
rule is exact-match — the greedy transcripts on the MTP world are identical
across the knob and identical to the morning's T=1 transcripts, and the
concurrency curve is unchanged (68.3 / 101.5 / 119.7 tok/s at 1 / 2 / 4
streams). The engine, the pick kernels and the sampler are shared with
GLM, so `scripts/fabric_glm_regression.sh` ran on the same binary
(`glm_regress_prop_2026-09-10/` against `glm_regression_post_rows_2026-09-10/`):
T=1 30.02 vs 29.99 ms/step, MTP 40.45 vs 40.48 at 72.0 %, prefill
547 / 1 400 / 6 253 vs 544 / 1 410 / 6 260 ms, every generated-ids line
identical in all three phases.

**World 2 (nodes .11/.12, `deploy/cluster_qwen_w2*.json`, measured after
the commit):** T=1 31.38 ms/step (32.5 on 2026-09-09 — the same ~1 ms the
inject hoist bought at world 4, against a step that is even more
byte-bound per rank). The MTP world at the sampled default, two 400-token
requests each way:

| world 2 | proposal off | proposal on |
|---|---:|---:|
| acceptance p1 | 43 %, 44 % | **60 %, 70 %** |
| tokens per pass | 1.43, 1.44 | **1.61, 1.70** |
| ms per pass | 40.3 | 40.4 |
| ms per token | 27.9, 28.2 | **23.7, 25.2** |

The same shape as world 4: the pass costs what it cost, the gain is all
tokens per pass — and worth more here in absolute terms, since every pass
streams 6.2 GB per rank instead of 3.8.

**Calibrating the proposal's temperature — measured, flat.** The draft's
sampling temperature as a multiple of the request's
(`DGPP_SPEC_PROPOSAL_TEMP`, `SampleSpec::draft_temperature`; exact at any
value, only the overlap with P moves). Acceptance p1 on the MTP world,
two 400-token requests each: 0.7 → 57 %, 60 %; 0.85 → 64 %, 68 %;
1.0 → 63 %, 65 %; 1.2 → 65 %. Flat from 0.85 up within the ±3-point run
noise, worse when sharpened. The knob stays at 1; the head's overlap with
the target is what it is.

### 5.2 MEASURED: MTP depth 2 is break-even at both worlds

Wired on 2026-09-10 (`kDraftChain`, `kBatchedDraftChain`; the chain rows'
QSA ring copied aside and back in a buffer of its own; `qwen_engine_test`
holds the depth-2 greedy transcripts to the plain engine's on the loopback
fixture) and measured at the sampled default, 400 tokens, two requests each:

| | world 4, depth 1 | world 4, depth 2 | world 2, depth 1 | world 2, depth 2 |
|---|---:|---:|---:|---:|
| ms per pass | 25.9 | 30.8 | 40.4–41.0 | 49.4–49.5 |
| tokens per pass | 1.63–1.64 | 1.93–1.99 | 1.63–1.73 | 1.81–2.08 |
| acceptance p1 / p2 | 63–64 % / — | 63–65 % / 30–34 % | 63–72 % / — | 58–69 % / 24–39 % |
| ms per token | 15.8–15.9 | **15.5–15.9** | 23.7–24.8 | **23.8–27.3** |

The second row costs +5 ms at world 4 and +9 at world 2 (the experts'
third row, the second draft's head and layer), tokens per pass rise 20 %,
and the two cancel. Depth 1 stays the default (`--mtp-depth 2` runs it).

**The chained draft's own proposal, the follow-on, also measured.** The
chain picks are sampling picks too (their own draft-stream key per draft
index, a proposal per draft row, the verify's row-t test on slot t; the
host's re-drafts clear every slot). p2 rises 30–34 → 36–37 % at world 4
and 24–39 → 31–36 % at world 2; tokens per pass 1.97–2.03 at world 4 for
30.8–30.9 ms = **15.2–15.6 ms/token against depth 1's 15.5**; world 2
1.95–1.97 for 49.5–49.7 ms = 25.2–25.4 against 23.9–25.0. Still
break-even: the second draft's overlap with the target is lower than the
first's, as a draft one step further from real hidden state must be. The
machinery stays (exact, gated, no cost at depth 1); depth 1 stays the
default. That closes the MTP list: every pass-cost item is at the
bandwidth wall, and every tokens-per-pass item is at the draft head's
overlap.

## 6. Measured, and not worth doing

* **Pushing the prefetcher harder.** `DGPP_L2_PREFETCH_LIGHT_BLOCKS` sweep
  at T=1 (ms/step): 8 → 23.36, **16 (default) → 22.25**, 24 → 22.95,
  32 → 23.87, 48 → 25.50. The GPU, the CPU and the NIC share one LPDDR5X
  controller, so every extra byte in flight lands on the collectives'
  latency; the default is the optimum of that curve and DRAM runs at
  ~172 GB/s (74 % of line rate) because of it, not because of the kernels.
* **A prefetch-instruction form of the prefetcher** (evening,
  `DGPP_L2_PREFETCH_FORM=prefetch`: one `prefetch.global.L2` per 128-byte
  line, no destination register, so a warp keeps issuing lines instead of
  folding loads). T=1 at 16 / 8 / 4 light blocks: 23.7 / 22.9 / 22.4
  ms/step against the load form's 21.5 — worse at every count, and worse
  the more lines it keeps in flight. The load form's fold is what bounds
  the bytes in flight to what the collectives tolerate; removing the bound
  lifts their latency, exactly as the light-rate sweep said. Off, behind
  the knob.
* **Slicing the GR gates across ranks.** Saves 9.8 MB per site of the
  replicated 13.1 MB, worth ~42 us, and costs at least one collective round
  trip at 34 us (three, in the sharded-hyper-state form) — break-even at
  best, at world 4, for a large restructuring.
* **Expert parallelism, replicated attention, unsharded experts.** Rejected
  with numbers on 2026-09-09 (plan §5, Q8 lever 7); nothing measured today
  changes them.
* **The side-stream-only inject dots.** Superseded by the fold (§3.1) at one
  row; still the form the multi-row sites use.
* **The fused mix beyond one row.** §3.2 — bitwise, and slower, for a
  shared-memory reason that will not go away.
* **The batched rows' inject dots folded into a down GEMV** (evening,
  `qwen_gr_down_inject_bf16`, bitwise the chain's GEMV and the dots kernel
  at three rows): MTP 25.8–25.9 ms/pass against the side-stream form's
  25.8–25.9 — neutral. Kept as the default because it is the simpler graph
  (no fork/join per site); `DGPP_QWEN_GR_GATE_SIDE=side` restores the
  side stream. The MTP pass does not move for chain work either: it is at
  25.8 ms whatever the chain does, which puts its ~4 ms of "slack" in the
  same place as T=1's.
* **A small-k form of the expert kernel for the down projection**
  (`moe_grouped_mma_fp8_ldm_smallk_kernel`, k ≤ 256: the 64 × k activation
  tile resident, four n-tiles through a codes-only ring, two blocks per SM
  at k = 160, bitwise on the TP=4 grid; `DGPP_MOE_FP8_SMALLK=on`): the
  prefill probe reads 1 141 / 4 315 ms with it against 1 144 / 4 375
  without — 0.3 % and 1.4 %, inside the noise at 2 K. Off by default.
* **The inject dots on the 16-byte-chunk GEMV core.** Tried 2026-09-10:
  no step time at T=1 (bandwidth-bound, see §1) and the changed FMA order
  flipped 2 of 4 greedy transcripts at near ties. Reverted to the
  lane-strided chain every transcript was made with; the fold (§3.1)
  stays.
* **Retiming the inject's prefetch window to the mix.** Correct in
  principle (the weight is read at the mix now), unmeasurable in practice
  (21.59–21.72 vs 21.40–21.45 ms/step); kept, since it costs nothing.
* **A bigger prefill chunk on its own.** §4.1 — −1.9 % at 8 K, because the
  grouped expert kernel re-reads its weight tile per m-tile.

## 7. Order of work

1. LANDED §5.1 sampled draft + ratio verify: 17.0–18.3 → 15.3–15.8
   ms/token on sampled requests.
2. MEASURED §4.1: the sweep kernel landed, the 8 192 chunk is worth 3.6 %
   and is not taken. The expert kernel's MMA efficiency (~20 % of dense
   peak at prefill widths) is the real item, beside §4.6.
3. §3.4 node fusion, then §3.3 the speculative launch.
4. §4.3–4.4 the prefill kernel fusions (§4.5 re-estimated at ~1 %).
5. §3.5 the skew diagnostic (cheap, but its outcome may be "the fabric is
   what it is").
6. §4.6, §4.7 the two large prefill projects.

Every item keeps the discipline the port was built with: an oracle or a
bitwise gate before the fabric run, `ctest -R qwen` green, the greedy
transcripts compared across the knob, and `scripts/fabric_glm_regression.sh`
whenever a shared kernel moves.
