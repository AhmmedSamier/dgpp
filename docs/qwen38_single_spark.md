# Qwen3.8-Flash-Next on a single Spark

Date: 2026-09-10

The 4-Spark fabric serves Qwen3.8-Flash-Next from the BF16 checkpoint
(`docs/qwen38_flash_next_plan.md`, `docs/qwen38_optimization_plan.md`).
This note records the single-Spark deployment: NVIDIA's NVFP4 checkpoint
(`nvidia/Qwen3.8-Flash-Next-NVFP4`) with the 47.7 GiB n-gram embedding
table left on the NVMe and read through a memory mapping, the resident
model, the decode graph and MTP at world 1.

## The checkpoint

`nvidia/Qwen3.8-Flash-Next-NVFP4` (snapshot fc694b54…), 123.5 GiB in 11
shards:

| class | format | GiB |
|---|---|---|
| routed experts, 48 backbone layers (512 x 3 x [640, 2560]) | NVFP4: U8 codes [n, k/2], E4M3 block scales [n, k/16], F32 `weight_scale_2` (an unused F32 `input_scale`) | 63.3 |
| n-gram table (128 parts of [2,500,012, 160] E4M3, one BF16 `weight_scale`) | FP8, `model-fp8-mtp-ple.safetensors` | 47.7 |
| MTP layer's experts | FP8 with 128-block scales | 1.2 |
| GDN, QSA, GR sites, lm_head, embed, shared experts, routers | BF16 | 9.1 |
| vision tower (not served) | BF16 | 0.8 |

Resident at world 1 without the table: 75.03 GiB (48 backbone + MTP
layers 72.6, globals 2.4). With the table it would be 122.7 GiB — the
board holds 121.6, so the table cannot be resident on one Spark.

## What was built

**NVFP4 experts (loader, binding, MoE views).** The config parser reads
`quantization_config.config_groups.group_0` (4-bit float, group 16) as
`experts_nvfp4`; the binding expects the four NVFP4 tensors per expert
matrix for the backbone layers and the FP8 pair for the MTP layer; the
loader packs the codes and block scales through the GLM-4.7 NVFP4
builders (`load_fp4_rows_mo` / `load_fp4_cols_mo`, the 640-wide slice on
the 16-group grid) and stores 1 / `weight_scale_2` per matrix; the Qwen
MoE layer binds `experts_fp4` into the shared `GlmMoeLayer`, whose fp4
GEMV and tile kernels serve decode and prefill unchanged.

**The mmap'ed n-gram table (`engine.ngram_table`).** A deployment-config
key, `"resident"` (the default: the table copied to the device, the
fabric's mode) or `"mmap"`. Under mmap:

- the loader maps the table's shard(s) read-only with `MADV_RANDOM`
  (`QwenNgramTableMmap`: the 128 parts' file offsets, global row r → part
  r / 2,500,012), keeps the BF16 scale, reports zero table bytes to the
  memory plan (`ngram_table_bytes`), and never copies a row;
- the PLE layer stages inside the walk: at the walk's start the device hash
  kernel writes its ids into pinned memory; a host node forked off the
  model's stream (`cudaLaunchHostFunc` on a side stream joined by events)
  gathers the rows from the mapping into pinned staging while layer 0
  runs; at layer 1 the walk joins and `qwen_ple_gather_staged_bf16`
  converts them with the device gather's arithmetic (bf16(e4m3 x scale),
  bitwise). Every path — prefill chunks, eager verifies, the fallbacks'
  rows, captured scalar and batched replays — stages from the device's
  own tokens and context, so nothing mirrors them on the host and the
  pipelined replay (the next graph enqueued before the draft block's tail
  settles) keeps working: the drafts the host does not know at launch are
  read by the graph itself;
- the captured graph carries one host node; the kernels-only census
  (`check_decode_graph`) takes the family's declared count
  (`session_graph_host_nodes()`) — a host node waits on nothing but its
  own stream, outside the copy engine's shared in-order queue that the
  batched-MTP stall needed.

**The graph world at world 1.** `dgpp-serve` used to refuse
`decode_graph` below world 2 and served a single Spark through the eager
engine over a streamed model. The collective bus now accepts
`world_size = 1` as a world of one: no lanes, no rendezvous, no engine
thread; every collective is the identity (a copy when the destination
differs), the staging handout is one pinned slot, the graph hooks succeed
in session order (`scenario_world_of_one` in `bus_test`). With it the
serve app runs the same resident graph engine (MTP, the batch families,
the prefix cache, the journal-less head) at world 1; the model is built
without a boundary reducer (nothing to fold), the recorder's record hooks
are the identity under capture. Without `decode_graph` a world of one
still streams through the eager engine as before.

`deploy/cluster_qwen_spark1.example.json` (MTP) and
`deploy/cluster_qwen_spark1_t1.example.json` (T=1) are the single-node
configs: one node, `ngram_table: "mmap"`, `decode_graph`, 4 slots,
kv_capacity 65536.

## Gates

- `qwen_ple_test`: the staged gather is bitwise the device table gather.
- `qwen_loader_test`: the mmap'ed table's rows are the shards' bytes; a
  rank's head gather lays them out as the device gather would; the plan's
  table bytes are zero; the mode is off by default.
- `qwen_engine_test` (world 2 loopback, MTP graph engine, scalar and
  batched replays, prefills, the fallbacks): transcripts over the mmap'ed
  table bitwise the resident table's (298 kernel nodes + 1 host node per
  variant).
- `qwen_forward_check --layers 8` on the real NVFP4 checkpoint: resident
  table vs mmap'ed, identical logits, layer digests and final hidden.
- `bus_test` (`DGPP_TEST_FILTER=world_of_one`): the world of one.
- `qwen_load_check --world 1 --mtp --ngram-table mmap`: 75.03 GiB
  resident, the table 50.03 GiB mapped, 77 s cold; the gather microbench
  after a cache drop, uniformly random rows (no locality, the worst case):
  a 2-row decode pass (32 rows) p50 359 us, p90 430, p99 858, max 2.3 ms;
  a 2048-row prefill chunk (32,768 rows) 91–143 ms. Layer 0 of a pass is
  ~1.2 ms, so the decode gather hides behind it except at the tail.

## The single-Spark serve (2026-09-10)

`deploy/cluster_qwen_spark1.json` (one node, `ngram_table: "mmap"`,
`decode_graph`, `mtp` depth 1, 4 slots, kv 65536), the launcher's `up`, the
serve gates (`scripts/serve_greedy_transcript.py`, `serve_bench.py`,
`serve_api_check.py`, `serve_prefill_probe.py`, `serve_eval.py`), the
world down.

Boot: 24.7 s from the resident image (77 s cold, the image captured on the
first boot). Memory plan 83.66 GiB (82.17 device + 1.49 pinned) + 8 GiB
headroom against 121.6; 8 graph variants (4 slots x 2 parities) of 1,560
kernel nodes + 1 host node each, plus the batch families.

**Decode, one stream, MTP depth 1** (rank 0's per-request lines; 256
tokens; the estimate was 57–60 ms per pass):

| prompt | ms/pass | tok/pass | accept p1 | ms/token |
|---|---|---|---|---|
| prose (greedy) | 59 | 1.56 | 56 % | 37.7 |
| code (greedy) | 59 | 1.88 | 88 % | 31.4 |
| math (greedy) | 59 | 1.83 | 83 % | 31.9 |
| JSON (greedy) | 59 | 1.88 | 88 % | 31.4 |
| prose (sampled, serve_bench) | 59 | 1.62 | 62 % | 36.6 (client 36.6, TTFT 532 ms) |

**Prefill** (the probe, 2 repeats, cold prompts): 524 tokens in 785 ms
(1.50 ms/token), 1,965 in 2,028 ms (1.03), 7,742 in 8,392 ms (1.08) — the
estimate was ~4 s per 2K prompt; the NVFP4 experts on the fp4 tile kernels
halve it.

**Four in flight** (the eval's shape, the 8-row batch family, greedy):
119–124 ms per step, 49–55 tok/s aggregate (16.2 ms per token), MTP 1.93–
1.95 tok/step/request, accept p1 92–94 %.

**Eval** (greedy, 4 in flight): GSM8K 59/60, HumanEval 39/40, schema
extraction 30/30 — the BF16 fabric's numbers exactly
(`docs/qwen38_flash_next_plan.md`: 59/60, 39/40, 30/30).

The API check passes; the greedy transcripts are the reference for the
T=1 comparison below.

**T=1** (`deploy/cluster_qwen_spark1_t1.json`, the same world without MTP):
47 ms per pass on every prompt (46.7–47.2 ms/token; the estimate was
47–50 against a 42.5 ms bandwidth floor at 9.9 GB per token), 21.4 tok/s;
client pace 47.2 ms/token, TTFT 506 ms. The four greedy transcripts are
identical to the MTP world's (4 of 4) — the single-Spark MTP == T=1 gate.
Boot 21.5 s, plan 80.8 GiB.

So on one Spark: 47 ms/token plain, 31–38 ms/token with MTP (59 ms per
pass at 1.56–1.88 tokens), against the 4-Spark fabric's 21.4 / 25.7 ms
(`docs/qwen38_optimization_plan.md`), the same eval, and a 2K prefill in
2.0 s.

## The dense stack in FP8 at load (`engine.dense_weights`, 2026-09-10)

Where the bytes go per decode token on this checkpoint (world 1, T=1):

| class | form | GiB per token | share |
|---|---|---|---|
| routed experts (10 of 512 per layer) | NVFP4 | 1.24 | 13 % |
| GDN layers (36) | BF16 | 3.89 | 42 % |
| QSA layers (12) | BF16 | 1.25 | 13 % |
| GR sites | BF16 | 1.23 | 13 % |
| lm_head | BF16 | 1.18 | 13 % |
| shared experts, routers | BF16 | 0.57 | 6 % |
| total | | 9.36 | 43.0 ms floor at 233.6 GB/s (47 measured) |

NVIDIA's checkpoint quantizes the experts only; 8.1 of the 9.4 GiB a
token reads are the BF16 dense stack. The Strix Halo recipe in the
thread the comparison came from (unsloth UD-IQ4_XS + a Q8_0 MTP sidecar,
llama.cpp on gfx1151: ~21 t/s plain, 38 t/s with MTP at 85–100 %
acceptance) compresses that stack too, at well under half of that
machine's bandwidth; our plain 21.4 t/s equals its plain number with 2.5x
the bytes at 91 % of line rate. Closing the MTP gap (its 26 ms/token
against our 31–38) is a bytes-per-token problem.

`engine.dense_weights`: `"checkpoint"` (the default; the BF16 the
checkpoint ships) or `"fp8"`. Under fp8 the loader encodes every dense
projection — GDN qkv/z/out, QSA q/k/v/o and the indexer projection, the GR
sites (the layers' and the mixers'), the shared experts, the PLE key/value
projections, lm_head — to block FP8 as it loads (`loaders/fp8_quant.hpp`):

    scale_b = amax(block) / 448            per 128 x 128 block
    code    = e4m3( w / scale_b )          round to nearest even, saturating

the fp8 GEMV core's and the scale-GEMM tile's own format (E4M3 codes,
fp32 scales), which is also the FP8 releases' recipe (GLM-5.3-Flash's,
this model's MTP experts). Block FP8 has no free parameter — the block's
maximum fixes its scale, nothing is clipped, nothing is calibrated — so
the same values encoded offline give the same codes; the loader gate
(`qwen_loader_dense_fp8_at_load_is_the_reference_recipe`) pins the loader's
bytes to a scalar reference of the recipe and the dequantized values to
the format's precision (at most one part in sixteen per element). What
stays as shipped: embeddings, norms, conv kernels, the GDN a/b
projections, the routers, the MTP fc matrices, the n-gram table.
Activations stay BF16 (the cores dequantize in registers), so the only
error is the weights' rounding. Where offline quantization genuinely
matters is 4 bits — a calibrated NVFP4 of the dense stack would be a quant
box job and a checkpoint, not a load-time switch.

The dense matrices ride the scale GEMM in both paths (the chunked fp8 GEMV
at decode rows, the mma tile above them); the GDN's four-projection GEMV
fusion becomes qkv + z through the scale GEMM and a two-problem BF16 GEMV
for a/b. The resident image key carries the form (`loader_format`), so a
BF16 image is never restored into an FP8 world; the encode runs once, on
the first boot (16 threads over the block rows), and the memory plan
follows the counting build. Configs: `deploy/cluster_qwen_spark1_fp8{,_t1}.example.json`.

### Measured with the dense stack in FP8 (2026-09-10)

`deploy/cluster_qwen_spark1_fp8.json` (the MTP world above with
`dense_weights: "fp8"`), the same gates. Boot 24 s from the resident image
(the first boot encodes the stack and captures the image); plan 79.7 GiB
(BF16 dense 83.7).

| | BF16 dense | FP8 dense |
|---|---|---|
| T=1, ms per pass | 47 | 31 (floor 25.5) |
| MTP, ms per pass | 59 | 40–41 |
| MTP greedy, ms/token (prose / code / math / JSON) | 37.7 / 31.4 / 31.9 / 31.4 | 26.4 / 22.8 / 21.2 / 21.0 |
| MTP greedy, tok/pass (accept p1) | 1.56–1.88 (56–88 %) | 1.54–1.93 (54–94 %) |
| MTP sampled (serve_bench), ms/token | 36.6 | 24.6–26.0 |
| prefill, ms/token at 512 / 2K / 8K | 1.50 / 1.03 / 1.08 | 1.85 / 1.15 / 1.17 |
| 4 in flight (GSM8K), ms/step, tok/s | 119–124, 49–55 | 70.7, 62.9 |
| eval GSM8K / HumanEval / extract | 59/60, 39/40, 30/30 | 59/60, 38/40, 30/30 |
| MTP == T=1 greedy transcripts | 4 of 4 | 4 of 4 |

HumanEval sat at 39/40 in the first FP8 run (the fp8 tile kernel for
prefill) and 38/40 in the final one (the dequantized-bridge prefill); the
fabric's own runs moved between 38 and 39 on a near tie (the plan doc), so
this is that tie, not a trend. The greedy transcripts diverge from the
BF16 stack's after 170–316 characters, in the reasoning — weight rounding
at the level the 8-layer forward check showed (12 of 15 argmax positions,
the residual rms within 0.3 %).

Against the Strix Halo thread (~21 t/s plain, 38 t/s with MTP): plain
32.3 t/s; with MTP 38–48 t/s greedy, 39–41 t/s sampled on prose.

What is left on the table with FP8 dense: T=1 at 31 ms against a 25.5 ms
floor (82 % of line rate, from 91 % before) — the fused BF16 decode paths
(the GR site's norm-staged down GEMV, the shared expert's two-launch tail,
the GDN four-projection GEMV) run as their unfused chains under fp8, and
the scale GEMM's GEMV core was shaped for the experts' slabs; fp8 fused
forms and a multi-problem fp8 GEMV would recover a few milliseconds.
Prefill pays the bridge's per-chunk dequantization (a faster dequant
kernel, or the dequantized matrix cached across a prompt's chunks, would
return it to the BF16 stack's 1.03 ms/token at 2K). The next rung, a
calibrated NVFP4 of the dense stack (~16 ms floor), is a quant-box
checkpoint, not a load-time switch.

### The FP8 fused decode forms and the multi-problem fp8 GEMV (2026-09-10, later)

The first FP8 cut ran the BF16 fused decode paths as their unfused chains.
The kernels now take either form: `gr_norm_down_kernel`,
`gr_down_inject_kernel`, `gr_act_up_kernel` (the GR site) and the shared
expert's two tail kernels carry a `kFp8` template parameter — the BF16
instantiations are the kernels as they were, the FP8 ones dot their rows
with `fp8_gemv::row_dots` / `row_dots_pair` on the same staged activations
(the scale GEMM's own chain), so every FP8 output is bitwise the unfused
fp8 chain's (`qwen_gr_test`, `qwen_moe_test`); the inject rows stay BF16.
`launch_scale_gemv_multi_bf16` runs up to four fp8 matrices against the
same rows in one launch (bitwise the single launches, `scale_gemm_test`);
the GDN's qkv + z and the QSA's q / k / v / indexer projections take it at
decode rows.

The captured graphs return to the BF16 stack's shape: the MTP scalar
variant 1,976 → 1,558 nodes (BF16: 1,561), the T=1 variant 1,860 → 1,258
(BF16: 1,258). Measured on the same worlds, the transcripts identical to
the unfused FP8 worlds' (4 of 4, both worlds):

| | unfused FP8 | fused FP8 |
|---|---|---|
| T=1, ms per pass | 30.8–31.4 | 30.2–30.9 (client 30.9) |
| MTP, ms per pass | 40–41 | 39–40 |
| MTP greedy, ms/token (prose / code / math / JSON) | 26.4 / 22.8 / 21.2 / 21.0 | 25.8 / 22.3 / 20.7 / 20.5 |
| MTP sampled (serve_bench) | 24.6 | 24.7 |
| prefill, ms/token at 512 / 2K | 1.85 / 1.15 | 1.83 / 1.14 |

So the ~400 launches were worth about 0.6 ms a pass, not the 2–3 ms
estimated: at T=1 the FP8 stack sits at 30.3 ms against a 25.5 ms floor
(84 % of line rate) with the graph the BF16 stack's shape, which points at
the fp8 GEMV core's bandwidth on the dense shapes rather than at launch
count — the GR up matrix's k = 320 is twenty 16-byte chunks per row, so a
warp leaves twelve lanes idle and a lane one chunk in flight, and the
same holds for every row width under 512 bytes. The next step is the
per-kernel profile at world 1 (`scripts/fabric_qwen_profile.sh` on the
T=1 FP8 config) and a narrow-row variant of the fp8 core for those
shapes.

### The world-1 profile of the FP8 MTP world (2026-09-10, nsys)

`scripts/fabric_qwen_profile.sh deploy/cluster_qwen_spark1_fp8.json`: 255
two-row passes, 40.8 ms wall per pass, 1,567 kernels per pass; the GPU
"busy" 49.8 ms per pass because the L2 prefetcher's kernels (11.2 ms of
GPU time, 201 per pass) run on their side stream under the main chain.
The main chain's classes against their byte budgets at 233.6 GB/s:

| kernel class | ms/pass | per pass | bytes/pass | at line rate | note |
|---|---|---|---|---|---|
| routed experts fp4 (gate/up + down) | 10.3 | 48 + 48 | ~2.7 GB (two rows route to ~2x the experts) | ~10.3 | at line rate |
| multi-problem fp8 GEMV (GDN qkv+z, QSA q/k/v/idx) | 8.0 | 49 | 1.94 GB | 8.3 | at line rate (L2-warm rows) |
| single fp8 GEMVs (out_proj, o_proj, GR up, PLE) | 5.5 | 153 | 1.15 GB | 4.9 | ~90 % |
| lm_head fp8, f32 out | 5.0 | 2 (!) | 1.27 GB | 5.4 | the verify's and the draft head's |
| GR down + inject (fused fp8) | 2.6 | 98 | 0.33 GB | 1.4 | 55 %: 320 rows = 41 blocks, overhead-bound |
| shared expert tail (fused fp8) | 1.5 | 49 + 49 | 0.24 GB | 1.0 | ~70 %, small kernels |
| GDN recurrence, conv, norms, router, sampling, commit | ~3.5 | | | | |
| gaps between kernels | ~2.2 | | | | |

Two findings. The second lm_head per MTP pass is the draft head's full-
vocabulary product (its pick needs the whole distribution): 636 MB at FP8,
2.5 ms of every pass, 5 ms with the verify's — an NVFP4 lm_head would
take 1.4 ms off T=1 and 2.7 ms off an MTP pass, a quality decision. And
the GR down GEMV is the one dense kernel far from line rate: 320 rows
over a 10,240-wide k is 41 blocks on 48 SMs with each block staging two
20 KB activation rows for 3.3 MB of weights — a split-k form (the rows'
k in slices across more blocks, a small reduce) would return ~1 ms a
pass. The L2 prefetcher's 11 ms of side-stream kernels earn nothing
obvious at world 1 (no collective gaps to fill); the A/B is below.

**The prefetcher A/B at world 1** (`DGPP_L2_PREFETCH=off`, the same FP8
worlds, transcripts identical): MTP 39–40 → 40–41 ms per pass, T=1
30.2–30.9 → 32–33. The prefetcher earns 2 ms a pass at T=1 even without
collective gaps to hide in — the side-stream reads still land the next
kernel's leading rows in L2 ahead of it — so it stays on; its 11 ms of
side-stream GPU time is not a cost the main chain pays.
