# DeepSeek-V4.1-Flash-1017 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
DeepSeek's 2026-10-17 refinement of
[DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash.md). It is **not** a republished
checkpoint: the engine loads the upstream release as shipped, with no
re-quantization pass.

## The checkpoint, as loaded

The release is a weight update, not an architecture change: its
`config.json`, `inference/config.json`, tokenizer and
`model.safetensors.index.json` are byte-identical to V4.1-Flash's (the same
96,085 tensors in the same 48 shards, 475.2 GiB, the same dtypes and
offsets; every shard's safetensors header is byte-identical too), and every
one of the 48 shards carries different weights. The `deepseek_v41` family
therefore serves it through the same loader, kernels and prompt encoder as
V4.1-Flash: MXFP4 routed and draft experts, FP8 e4m3 dense and attention
projections with e8m0 block scales, the two FP8 Engram tables mapped from
the NVMe, BF16 embedding and head, the trained fp4/fp8 cache formats. The
Engram hash sidecar (`dgpp_engram_tables.json`, derived from the tokenizer)
is identical to V4.1-Flash's and is regenerated beside the snapshot.

Because the headers match, the resident image cache's key (config plus
shard headers) is the same for both checkpoints. Since 2026-10-18 every
image keeps a weight-identity note beside it and an image whose note
disagrees with the checkpoint is discarded and rebuilt; the old V4.1-Flash
images on this cluster were deleted when the checkpoint was replaced.

## The indexer budget

Every compressing layer attends only the compressed-KV entries the indexer
selects for a query: `index_topk` of them (the release trains and ships
512), chosen on the reindex layers from a candidate pool of
`candidate_topk_blocks` blocks of `candidate_block_size` entries (2048 × 8)
that the candidate layer scores once. The three values are deployment keys
since 2026-10-18 (`engine.index_topk`, `engine.candidate_topk_blocks`,
`engine.candidate_block_size`; the select networks serve up to 2048), so a
deployment can show each query more of its context than the release's
budget does — the lever against the sparse attention's misses on long
inputs. A wider budget changes the model's answers (more context visible),
not only its speed. The template ships the checkpoint's 512 and carries
the key (`"index_topk": 512`, so the opt-in is a one-value edit); the
measurement below prices 2048 against it on the same binary.

## Serving

Four DGX Spark nodes, tensor parallel, the V4.1-Flash template's shape: six
request slots at DSpark depth 4 with the confidence-scheduled verify depth,
the bounded prefill (the model's own SWA-replay recipe), the full
1M-token context. The tokenizer, prompt renderer (`encoding/encoding.py`,
no Jinja template) and DSML tool-call format are V4.1-Flash's.

## Decode levers (2026-10-18)

**The L2 prefetch windows.** The decode step is one serial chain of
kernels, and while it waits in a bus fold or a selection the memory
system idles. Every other decode family streams its next weights into L2
through those phases (`kernels/l2_prefetch.hpp`, `engine.l2_prefetch*`);
the DeepSeek-V4.1 walk did not. It now opens the same windows: the MoE
site's coefficients, norm, router and shared expert before the attention
fold; the next layer's attention side (its coefficients, norm, wq_a, wkv
and wq_b in consumption order) before the MoE fold, or the Engram
projection when the next layer carries one, or the head behind the last
layer; an Engram layer's own attention side under its fold; and the
output projections wo_a and wo_b under the selection and the attention
inside the CSA2 layer. Nothing the model reads is written, so the walk is
bit-identical with the prefetcher on or off. Measured at one request on
this template against the same binary with `engine.l2_prefetch: false`
(three greedy 256-token answers per class, the engine's decode steps):

| Class | Pass, prefetch off | Pass, prefetch on | Decode tok/s |
|---|---|---|---|
| Prose | 45.1 ms | 40.9 ms | 43.8 → 48.4 (+9.4 %) |
| Code | 53.3 ms | 49.9 ms | 65.5 → 70.1 (+6.5 %) |
| JSON | 56.1 ms | 52.7 ms | 73.4 → 78.0 (+6.0 %) |
| Math | 52.7 ms | 49.4 ms | 71.2 → 73.7 (+3.4 %) |
| Chat | 47.2 ms | 43.5 ms | 46.9 → 49.9 (+5.9 %) |

Every transcript was identical between the two.

**The verify schedule's cost model.** The confidence-scheduled depth
verifies a draft while its survival probability exceeds lambda times the
row cost, so the row cost decides the depth and the template had carried
the engine's defaults (8 ms per row, 28 ms base). Sweeps at fixed depths
1 to 4 with the schedule off, and the plain step, put every class on one
line, pass = 31.1 ms + 5.17 ms × (1 + drafts verified), with residuals
under 0.25 ms:

| Mode | Prose | Code | JSON | Math | Chat |
|---|---|---|---|---|---|
| Plain step (ms) | 31.1 | 31.1 | 31.1 | 31.1 | 31.0 |
| Depth 1 (ms per pass / tokens per pass) | 40.9 / 1.66 | 41.3 / 1.83 | 41.6 / 1.95 | 41.3 / 1.92 | 41.2 / 1.70 |
| Depth 2 | 46.3 / 2.02 | 46.8 / 2.55 | 47.1 / 2.87 | 46.8 / 2.80 | 46.6 / 2.18 |
| Depth 3 | 51.3 / 2.16 | 51.7 / 3.27 | 52.4 / 3.54 | 51.8 / 3.40 | 51.6 / 2.45 |
| Depth 4 | 56.0 / 2.24 | 57.0 / 3.75 | 57.7 / 4.25 | 56.8 / 3.98 | 56.7 / 2.58 |
| Fitted base / row (ms) | 31.1 / 5.01 | 31.1 / 5.18 | 31.0 / 5.34 | 31.2 / 5.14 | 31.0 / 5.15 |

A verify row costs its extra distinct experts, about 5 ms, and the draft
itself 5.1 ms (the fitted base plus one row, less the plain step). The
template now carries `engine.mtp_schedule_row_ms: 5.2` and
`engine.mtp_schedule_base_ms: 31.1`, which lowers the survival threshold
the schedule demands from 0.36 to 0.23 at the configured lambda floor.
At one request the change is level (prose +1.8 %, code +0.7 %, JSON
−1.0 %, math −1.2 %, chat −0.4 % against the defaults on the same binary,
within the sweep's noise): the schedule was already near its optimum, and
the keys are kept because they are the measured truth. These sweeps were
measured before the prefetch windows; the row and base costs scale with
the pass and the fit's structure does not change.

## The prefill selection at long context (2026-10-18)

Prefill cost per token is flat in everything but one piece: on every
index-source layer, each query of a chunk is scored against every visible
compressed entry (the candidate layer selects from all of them, the other
index layers narrow the top-k to the candidate pool afterwards). That scan
ran as a cuBLASLt fp8 GEMM into a per-head fp32 dot buffer, a fold kernel
over the buffer and the selection kernels. The buffer's 64 MB budget held
four query rows at 128K entries, so a 4096-row chunk ran about a thousand
tiles, each re-reading the whole key set and writing and re-reading 64 MB
of dots: a chunk's cost grew with the square of its position and the
per-token cost accelerated with the prompt (the "before" column below).

The scan is now one fused kernel: a query row's 32 heads against 128
entries per CTA on the fp8 tensor cores, the heads folded in the mma
fragments and only the per-row logits written, with the logits tile sized
by the retired budget so a row tile reads the keys once for 128 rows at
128K entries (the keys stay in L2 across them). The selection kernels are
unchanged. The scorer reaches 31 TFLOP/s on the microbench (35 µs per
query row at 128K entries), about a third of the mma.sync rate, with its
synchronous key staging as the next lever; the restricted layers could
also score only the candidate pool, as decode does, instead of every
entry. Both are follow-ups; the quadratic term is gone.

| Context | Prefill before the fix (s) | ms per token |
|---|---|---|
| 32K | 20.3 | 0.62 |
| 64K | 44.5 | 0.68 |
| 128K | 113.6 | 0.87 |
| 256K | 387.9 | 1.48 |
| 512K | 1947 | 3.71 |

The context rows after the fix come with the next campaign; by the
scorer's measured rate the scan adds about 0.14 s per index layer to a
4096-row chunk at the end of a 512K prompt against several seconds before.

## Measured (2026-10-18, four nodes)

Both budgets ran the full row set of [benchmarks](../benchmarks.md) on the
same binary (source `ea4fcdf9` plus this working tree, binary
`9d684e2d10ad3573`; every rank at zero swap, zero cached tokens in the cold
probes, matching operation streams, an idle fabric), one campaign each:
the template as shipped (`index_topk` 512) and the template with
`index_topk` 2048. The 512 rows are the ones tabled in the benchmark doc;
the 2048 set is recorded here only, since no shipped template carries it.

| | `index_topk` 512 (shipped) | `index_topk` 2048 |
|---|---|---|
| C1 decode tok/s (prose / code / JSON / math / chat) | 43.2 / 64.9 / 73.9 / 68.1 / 45.1 | 43.4 / 64.0 / 74.4 / 68.6 / 45.2 |
| Decode tok/s by concurrency, C2 / C4 / C8 | 60.7–106.2 / 81.1–119.2 / 71.1–124.2 | 61.0–106.4 / 82.0–119.9 / 70.7–126.8 |
| Wall tok/s, C1 / C2 / C4 / C8 | 42.1–70.2 / 59.2–101.1 / 79.3–114.7 / 69.7–119.3 | 42.3–70.8 / 59.5–101.3 / 80.1–115.4 / 69.3–121.7 |
| Cold prefill s, 512 / 2K / 8K / 32K tokens | 0.528 / 1.594 / 5.245 / 20.790 | 0.526 / 1.744 / 7.007 / 30.745 |
| Context prefill s, 0 / 32K / 64K / 128K / 256K / 512K | 0.309 / 20.2 / 44.5 / 112.9 / 386.9 / 1947.2 | 0.307 / 30.2 / 65.5 / 159.2 / 499.5 / 2334.2 |
| Decode ms per pass at those contexts | 49.77 / 53.78 / 53.42 / 56.11 / 61.01 / 67.07 | 51.03 / 56.79 / 58.28 / 58.74 / 63.03 / 69.88 |
| HumanEval / GSM8K / extraction | 157/164 / 294/300 / 100/100 | 156/164 / 295/300 / 100/100 |
| API, plain/default transcripts, isolation | 9/9, 5/5 classes, identical | 9/9, 5/5 classes, identical |

The widened budget leaves short-context decode level (every C1–C4 class
within ±1 %, C8 within the sweep's run-to-run spread), but every query on
a compressing layer now reads four times the compressed entries: cold
prefill runs 9 % longer at 2K, 34 % at 8K and 48 % at 32K, the context
prefills 17–49 % longer, and the decode pass at 32K–512K of context 3–8 %
slower. The quality set does not separate the two (one HumanEval and one
GSM8K item each way), so the budget's benefit on long inputs is not
measured here; the template therefore ships the checkpoint's 512 and the
key is the opt-in with this price.

Against the DeepSeek-V4.1-Flash 1M row of 2026-10-07 (source `07a32fb`),
the 1017 weights decode at the same rates (C1 43.2–73.9 against 41.8–78.2
by class, the per-class differences being the drafter's acceptance on the
re-trained weights; the decode pass at 0–512K of context 49.8–67.1 ms
against 50.9–66.6) and answer the quality set alike (157/164 against the
128K row's 160/164 with five answers truncated at the eval's token limit
against none; GSM8K 294 against 296; extraction 100). Cold prefill reads
6–8 % longer (1.594 / 5.245 / 20.790 s against 1.472 / 4.946 / 19.545).
That is master's drift between the two dates, not this port: the
committed release binary of `aa88b8f7` (built without this working tree)
prefills the same template at the checkpoint's budget in 0.531 / 1.595 /
5.216 / 20.714 s, within 0.5 % of the 1017 binary, and the CSA2 kernels the
512 path runs are byte-identical SASS to the committed ones (the 2048
tile is a second instantiation).

## License and attribution

The model, its weights, tokenizer and prompt encoder are DeepSeek's (MIT);
see the upstream model card for its terms. This file describes only the
engine's support and adds no weights.
