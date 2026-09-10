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
