# GLM-5.3-Flash: line-rate analysis and two exact optimizations, 2026-09-19

Four GB10 DGX Sparks, `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` (rev `17b8d4d8`),
TP=4, four slots, MTP depth 1, BF16 KV, 786,432-token pool, 8 GiB prefix
cache — the deployed recipe. No weight, KV or activation precision changed.
Every A/B below is one binary with an environment switch, on an otherwise
idle fabric, the same prompts in both arms.

Retained:

| change | single-stream effect | concurrency | outputs |
|---|---|---|---|
| **bf12**: lossless 12-bit resident form of the decode GEMV's bf16 weights (`engine.bf16_weights: "bf12"`; every template) | C1 decode **+6.9 to +7.5 %** on all five classes; step 34.2 → 32.0 ms | C4 −0.65 to +0.17 % (noise) | 15/15 C1 transcripts byte-identical |
| **state-only MTP prefill** (default; `DGPP_MTP_PREFILL_FULL=1` restores) | cold prefill **−6.0 / −7.3 / −7.6 %** at ~2K / ~8K / ~32K | n/a | identical first tokens, usage, long-context text, MTP passes and acceptance |

## 1. Where the physical limits are

### Decode

One MTP step reads, per rank (shapes from the checkpoint headers, slice rules
from the loader; routed experts at the two-distinct-routes worst case):

| class | bytes | share |
|---|---:|---:|
| native bf16: KDA in/out projections 2.40 GB, lm head 2 × 0.317, indexer/router/mHC/eh_proj 0.47 | 3.5 GB | 52 % |
| NVFP4 routed experts, 16 slots × 42 layers (+ FP8 draft experts 0.10) | 2.48 GB | 36 % |
| FP8 projections, dense MLPs, shared experts | 0.82 GB | 12 % |
| **total** | **6.79 GB** (5.55 if both rows route alike) | |

The decode cores stream at the part's rate: `micro_gemv_bw` today 238–250
GB/s on cold bf16 matrices; in the profiled step the bf16 GEMVs read 275 GB/s
effective (the L2-prefetched prefix included), the fp4 slot kernels 235–268,
the fp8 GEMVs 233. At 250 GB/s the bytes alone are **27.2 ms of the 34.4 ms
step**: the step runs at 79 % of pure line rate, and the remainder is the
serial protocol — 94 collectives 4.2 ms (floor ≈ 2.1), mHC 1.6, small
latency-bound kernels ≈ 2.5, launch seam 0.5. The kernels are not the
opportunity; only fewer bytes, fewer or faster collectives, or more tokens per
step move the number.

### Prefill

Per 2,048-token chunk every one of the 288 × 42 routed experts is touched
(57 rows each): 1.02 GB of NVFP4 per layer per rank, 4.08 ms at 250 GB/s. The
ldmatrix kernels spend 4.06 ms per chunk-layer (2 × 1.21 + 1.65) — **at the
read-once floor**, with the tensor-core work overlapped under it. Listed
attention runs at the tensor cores' bf16 rate for its 2,048-key lists. What is
not at a floor: the bulk all-reduces (15 % of GPU time, ≈ 45 % of the two-lane
wire rate, nothing overlapped with them), the sequential KDA recurrence (8 %),
and — until today — the draft block's prefill rows (6–7 %, see §3).

## 2. bf12: the bf16 weights' exponents are compressible, losslessly

52 % of the step's bytes are native bf16, and "no further quantization" rules
out the obvious lever. But a bf16 weight's exponent carries ≈ 2.6 bits of
entropy (measured on q/k/v/o and the lm head: 99.98 % of a tensor's weights
sit in fifteen consecutive exponents). `src/kernels/bf12_gemv.{hpp,cu}` stores
the sign+mantissa byte and a 4-bit exponent code against a **per-row** window
— 12 bits a weight — and rebuilds the exact bf16 bits in registers:

- code 15 escapes to a per-row, column-sorted side table (1.5e-4 of weights);
- a row past 64 escapes (outlier channels: 35 q/k rows in the whole model)
  stays bf16 in a side array and its warp runs `bf16_gemv::row_dots` itself —
  a warp is one weight row, so the branch is uniform across lanes;
- lane/step ownership and FMA order are the production kernel's, so every
  output is **bitwise** `launch_bf16_gemv`'s (`bf12_gemv_test`: 1–4 rows, both
  outputs, strided activations, zeros, subnormals, Inf/NaN, the escape bound,
  raw rows, mixed-scale fused rows).

The companions are built after the resident load (70 matrices, 2.55 → 1.92
GiB per rank, 1.3 s), registered with `CublasLtGemm`, and taken by decode
launches of up to four rows; the L2 prefetch windows follow them through
`IGemm::resident_view`. Wider batches and prefill keep the bf16 bytes, which
is why both forms stay resident (+1.92 GiB/rank, in the memory plan) and why
the switch is opt-in.

Microbench (`bf12_bench.cu`, real weights, cold copies, bitwise-checked):

| matrix | bf16 GEMV | bf12 | speedup |
|---|---:|---:|---:|
| KDA q slice 2048×4096, 2 rows | 70–74 µs | 56–58 µs | 1.22–1.28× |
| KDA o slice 4096×2048, 2 rows | 70–73 µs | 54–56 µs | 1.28–1.33× |
| lm head slice 38720×4096, 2 rows | 1,278–1,295 µs | 942–970 µs | 1.33–1.36× |
| q slice, 1 / 3 / 4 rows | 72–75 µs | 56–59 µs | 1.2–1.33× |

Fabric, engine decode tok/s, medians of three (C4: 8-row steps, which keep
the cuBLASLt path on the bf16 bytes by design):

| class | C1 off | C1 on | Δ | C4 off | C4 on | Δ |
|---|---:|---:|---:|---:|---:|---:|
| prose | 54.85 | 58.96 | +7.49 % | 100.84 | 100.19 | −0.65 % |
| code | 57.15 | 61.11 | +6.93 % | 102.53 | 102.51 | −0.01 % |
| json | 57.60 | 61.70 | +7.12 % | 105.05 | 105.08 | +0.02 % |
| math | 55.66 | 59.54 | +6.96 % | 106.27 | 106.44 | +0.17 % |
| chat | 50.38 | 53.92 | +7.02 % | 100.31 | 99.69 | −0.62 % |

All 15 C1 transcripts are byte-identical between the arms; tok/pass and p1
are unchanged.

Two live requests run four-row steps, which do take the packed form
(one repeat per class, same binary): 75.8 / 81.2 / 76.9 / 81.9 / 74.4 →
79.4 / 84.0 / 79.8 / 86.5 / 79.6 tok/s, **+3.4 to +7.0 %**. So concurrency
is never worse: C2 gains, C3/C4 are unchanged.

It is independent of MTP: a plain T=1 step's bytes are 62 % bf16 (3.14 of
5.05 GB). With `--no-mtp`, prose and code: 37.0 → **40.2 tok/s (+8.5 %)**,
27.0 → 24.9 ms/step.

A call the smem bound splits into GEMV chunks (the draft block's `eh_proj`,
k = 8192, from three rows) takes the companion per chunk: the bf16 chunks
re-read the weights just the same.

## 3. The draft block's prefill rows only need their state

`mtp_run_rows` ran the whole draft block over every prompt row — DSA
selection and listed attention, the output projection, a fold, then the MoE
through the host-segmented GEMV path (`moe_->enqueue`, 5 % of the profiled
prefill on its own) and a second fold — and then returned before the head:
nothing reads those rows' output. What later drafts read is the block's DSA
cache (latent rows, index pools, tail ring), a function of the site's input
alone. `DsaLayer::enqueue_prefill(..., state_only)` now stops after those
writes and the block returns (no staged handout is taken).

| target (actual tokens) | full block | state-only | Δ | prompt tok/s |
|---|---:|---:|---:|---:|
| ~2K (1,944–1,989) | 1.458 s | **1.371 s** | −6.0 % | 1,424 |
| ~8K (7,723–7,811) | 5.975 s | **5.539 s** | −7.3 % | 1,410 |
| ~32K (31,291–31,306) | 27.624 s | **25.535 s** | −7.6 % | 1,226 |

Equivalence: nine cold probes agree on prompt hash, first token, usage and
finish; 128-token generations after 2,066 / 7,568 / 30,182-token prompts are
identical to the control and to the 2026-09-16 record, with the same passes
(71 / 70 / 66), tok/pass (1.79 / 1.81 / 1.92) and p1 (79 / 83 / 94 %).
`glm_tp_test` (MTP, prefix and resumable-prefill gates), `dsa_test`,
`glm_moe_test`, `glm_dsa_*` pass. Full suite with both changes: 109 of 110
(`raw/ctest-summary.txt`); the one failure, `script_reports_test`, is its
inline-Python lint matching an untracked `scripts/setup.sh` that is not part
of this work.

## 4. Tried, not retained

| experiment | result | decision |
|---|---|---|
| bulk pool 16 × 256 KiB (4 rounds per 16 MiB fold instead of 8) | 2K 1,371 → 1,453 ms, 8K 5,539 → 5,752 | regresses (incast); removed |
| bulk pool 8 × 512 KiB | 1,354 / 5,453 ms (−1.2 / −1.6 %) | inside session drift; removed |
| bulk pool 16 × 512 KiB | 1,456 / 5,724 ms | regresses; removed |
| bf12 window per tensor instead of per row | 21 of 34 fused `in_proj` refused (f_a/g_a/b rows leave a shared window) | replaced by the per-row window + raw rows |

The bulk fold is therefore not round-count-bound; its cost needs the
per-phase timeline the latency path already has before another attempt.

## 5. Round two (same day): ship it, widen it, carry it to the other families

Requested after the first round: make the 12-bit form a deployment setting
enabled in every template, extend it to the wider batches and the other
families, look at the collective transport, and overlap the prefill folds
with compute. Every A/B is again one binary, one switch, the same prompts;
the control arm is `--bf16-weights checkpoint` with `DGPP_PREFILL_OVERLAP=off`.

### 5.1 `engine.bf16_weights` and the five-to-eight-row kernel

The switch is `engine.bf16_weights: "checkpoint" | "bf12"` (parser, CLI flag
`--bf16-weights`, the settings record to the peers, the config digest, the
memory plan; `DGPP_BF12=on|off` overrides for A/B). `Bf12Companions` is the
shared owner every family uses.

Three- and four-request batches run six- and eight-row steps, which the BF16
sites handed to cuBLASLt (the bytes once, ~210 GB/s). The wide kernel stages
the activations one 1024-column window at a time (16 KB of shared memory at
eight rows, a barrier on both sides of every restage) and carries the lane
accumulators across the windows, so a row's chain is still the scalar one —
bitwise the GEMV chunks at any row count (`bf12_wide_bench.cu`):

| matrix, 8 rows | cuBLASLt (BF16) | bf12 wide |
|---|---:|---:|
| KDA q slice 2048×4096 | 80–84 µs | 57–60 µs |
| KDA o slice 4096×2048 | 87 µs | 57–60 µs |
| lm head slice 38720×4096 | 1,373–1,417 µs | 932–960 µs |

One trap, caught by the transcript gate: widening the GEMV lowering to eight
rows also re-routed five-to-eight-row PREFILL chunks (a prefix-cache cut
leaves them) from their Lt algorithm to the GEMV chain, and C1 transcripts
moved. `CublasLtGemm::set_bf12_wide` now opens the wide launches to decode
batches only; with that, 10/10 C1 and 3/3 long-context transcripts are
identical to the control.

GLM-5.3-Flash, engine decode tok/s, final build from the shipped template:

| live requests | prose | code | json | math | chat | geomean vs control |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 59.1 | 61.4 | 62.0 | 59.9 | 51.9 | +6.8 % |
| 2 | 80.0 | 84.9 | 80.5 | 87.2 | 77.5 | +5.2 % |
| 3 | 97.3 | 98.5 | 106.1 | 98.3 | 93.0 | +6.5 % |
| 4 | 114.3 | 109.7 | 119.8 | 110.2 | 104.8 | +6.4 % |

### 5.2 Other families

| family (world 4) | packed | C1 | C4 | transcripts | resident cost |
|---|---|---:|---:|---|---:|
| GLM-4.7 NVFP4 | 374 matrices: q/k/v/o of 93 attention blocks, head, eh_proj (6.36 → 4.78 GiB) | 32.1 → 35.8 tok/s (**+11–12 %**, 60.0 → 53.8 ms/pass) | +4.6–6.6 % | 6/6 identical | +4.9 GiB (fits) |
| full GLM-5.3 int4/int8 | 89 matrices: dense layers' and draft's projections, 22 indexers, head, eh_proj (1.65 → 1.24 GiB) | 28.5 → 30.2 tok/s (**+5.3–6.5 %**) | +3.2–6.6 % | 4/4 identical | +1.27 GiB: the template was at its ceiling, context 120K → 100K |
| GLM-5.3-Flash, two nodes | the four-node set at twice the slice (4.98 → 3.74 GiB) | not re-measured | — | — | +3.83 GiB: context 160K → 132K |
| Qwen3.8-Flash-Next | nothing yet: hidden 2560 needs a 512-column tail block, and its GDN/GR sites are fused launches (1.3–5 GB/step reachable) | — | — | — | — |
| DeepSeek-V4.1-Flash | nothing: its BF16 sites ride the tensor-core lowering (< 1 % of the step) | — | — | — | — |

### 5.3 Prefill: the folds beside the compute

`BoundaryReducer::begin_async` is the bulk machine's submit without its wait.
A chunk of ≥ 1024 rows runs each KDA attention site in two row blocks: block
A's fold flies beside block B's site, and the FFN fold's block B beside the
next KDA layer's block A (the FFN site keeps the whole chunk — its experts
are read once per chunk). Fabric bisect of exactness (long-context
transcripts against the control):

| arm | identical |
|---|---|
| every site in blocks, asynchronous folds | 0/3 |
| every site in blocks, synchronous folds | 0/3 (not a race: the numerics of the split) |
| **KDA sites only** (shipped) | **3/3** |
| DSA sites only | 0/3 |

At production shapes the KDA layer, the mHC site and update, the FP8
projections and (with `set_plan_rows`) the Lt projections are bitwise under
the row split (`kda_split_check.cu`, `mhc_split_check.cu`,
`fp8_split_check.cu`, `lt_split_check.cu`); a DSA site is not, like any
change of the chunk cuts. Shipped form: 37 % of the fold bytes hidden.

| target | control | shipped | Δ |
|---|---:|---:|---:|
| ~2K | 1.374 s | **1.297 s** | −5.6 % |
| ~8K | 5.526 s | **5.249 s** | −5.0 % |
| ~32K | 25.502 s | **24.523 s** | −3.8 % |

(the control already has §3's state-only draft rows; against the 2026-09-16
record the three sizes are −11.5 / −12.6 / −11.6 %).

### 5.4 Collectives: the "skew" was half our own serialization

Per collective on the current build (16 KiB payloads): 45 µs = copy 3.7 +
handshake 15–18 + skew 19.5–20.6 + fold 5.6; engine post 3.1. A new timeline
line records the gaps between a generation's consecutive claims: **0 % under
5 µs, 80–85 % at 5–10 µs, on every rank** — the peers' doorbells were
co-resident and the kernel gated them one scan round at a time (scan of
system loads + gate pass + ack ≈ 8–9 µs), two of the three rounds landing in
`skew`. One election per peer per round (graph kernel only): a quarter to a
third of the gaps fall to 3–5 µs, rank 0's collective 43.8 → 41.6 µs
(−0.2 ms/step), results bitwise, a seven-minute mixed soak clean. What is
left inside a round is the gate pass itself and doorbells that land during
it.

GPU-posted sends: `benchmarks/micro/uar_probe.cpp` shows the GB10 **does**
map an mlx5 doorbell (UAR) page for device access
(`cudaHostRegisterIoMemory | Mapped` succeeds, unlike device-memory
registration). Whether device stores reach the NIC in order is the next
probe; the prize is the engine's notice + post, ≈ 3 µs of each handshake
(≈ 0.3 ms/step) — smaller than the gate serialization above.

## 6. Opportunities still open, by expected value

Decode (GLM-5.3-Flash step now ≈ 31.5 ms; bytes ≈ 24 ms of it at line rate):

1. **One interleaved gate pass for co-claimed peers** in the graph
   collective (the claim gaps' remaining 3–5 µs and the 7–10 µs of late
   doorbells): ≈ 0.4–0.8 ms/step (1.5–2.5 %). Kernel-local.
2. **bf12-only residency**: decode from bf12 everywhere and expand to BF16
   scratch for prefill, so the raw copies go — the format then SAVES 0.25×
   (−1.25 GiB/rank on two-node Flash instead of +3.8) and returns the context
   the ceiling-bound templates gave up. Needs the loader to own those
   matrices separately; prefill pays ≈ +2 %.
3. **Qwen**: a 512-column tail block (hidden 2560, slices of 1536) plus bf12
   twins of `launch_bf16_gemv_multi` and the GR kernels, which already factor
   through `bf16_gemv::row_dots`: 2.3 GB/step at world 4, 5.0 at world 2
   (≈ 5–8 % of a step). The same tail block gives GLM-5.3-Flash's indexer
   `wq_b` (0.15 ms).
4. **GPU-published staging fold** (the engine hashes 16 KiB per generation
   on the CPU before posting; the bulk path already publishes it from the
   kernel): ≈ 1.5 µs/collective. And the 64-byte doorbell as an inline send
   (`max_inline_data = 0` came from the M0 smoke tool, never measured).
5. **Adaptive depth-2 MTP at C1** (deferred by request) and a truncated
   draft-head vocabulary (excluded: acceptance).
6. **Batches past eight rows** (the full GLM's 16, GLM-4.7's/DeepSeek's 32):
   a bf12 variant of the tensor-core `mma_gemv` — also what DeepSeek needs.

Prefill (GLM-5.3-Flash ≈ 0.66 ms/token):

7. **Find what in a DSA site moves under a row split** (the bisect says the
   site, not the fold): fixing it takes the hidden fold share from 37 % to
   50 % and beyond (≈ −2 %).
8. **Bulk fold wire rate** (≈ 45 % of the two-lane ceiling; 16-slot rounds
   regress on incast): needs the per-phase bulk timeline first.
9. **Chunk-parallel KDA recurrence** (8 % of prefill, a numerics project) and
   skipping the q-side projections in the state-only draft rows (< 1 %).
10. **`graph_replay_arm` spins on `walk_pub` while holding `coll_mu`**
    (`graph_replay_finish` releases it first): latent, never fired; hoist it.

## Reproduction

```bash
cmake --build build-ci -j -- -k && build-ci/bf12_gemv_test
# decode A/B (arms differ only in --bf16-weights)
scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json --bin build-ci/dgpp-serve   # control: --knobs="--bf16-weights checkpoint"
python3 benchmarks/results/2026-09-16-dsv41-perf/timed_load.py HOST 18080 \
  --concurrency 1,4 --classes all --max-tokens 256 --repeat 3 --warm 1 --json-out on.json
# prefill A/B (control arm: DGPP_MTP_PREFILL_FULL=1)
DGPP_DATA_DIR=build-ci/eval_data python3 scripts/serve_prefill_probe.py HOST 18080 \
  2048 8192 32768 --repeat 3 --seed 916 --tag glm-flash-perf --json-out prefill.json
python3 benchmarks/results/2026-09-16-glm-flash-perf/long_transcripts.py HOST 18080 long.json
```

`st_index.py`, `exponent_stats.py` and `row_escapes.py` reproduce the
exponent statistics from the checkpoint; `bf12_bench.cu` is the standalone
microbench (nvcc flags of `build-ci/compile_commands.json`, linked against
`libdgpp_kernels.a`).
