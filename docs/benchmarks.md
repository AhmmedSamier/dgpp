# Benchmarks

Every serving number this project has measured, per model, per world, per
concurrency, with the command that produced it.

Two rules govern this page. **Every figure is a measurement**, dated, with the
binary and the ritual that produced it named beside it; nothing here is a
projection, and the modelled floors that appear are labelled as floors.
**Every cell that has not been measured says so** rather than borrowing a
neighbour's number, and §8 lists each gap with the command that fills it.

- §1 how to read a number
- §2 the environment and the deployments
- §3 decode, one request
- §4 decode, per prompt class
- §5 decode under concurrency
- §6 prefill and time to first token
- §7 quality, so the throughput numbers are comparable
- §8 what is not measured yet
- §9 reproducing all of it

## 1. How to read a number

**The unit of decode work is a pass, not a token.** One pass is one replay of
the decode graph: the engine's `ms/step` in the stats line. Without
speculative decoding a pass yields one token per request. With MTP a pass
yields `tok/pass` tokens, between 1 and 1 + `mtp_depth`, according to how many
drafts the verify accepted. So

    ms/token = ms/pass / tok/pass

and a change that raises acceptance moves `ms/token` while leaving `ms/pass`
alone. Both are quoted throughout, because only their ratio is the user's
experience and only `ms/pass` is the engine's cost.

**Aggregate tokens/s is a client-side reading over a phase**, summing every
concurrent request's tokens across the phase's decode span. It is not
`1000 / ms/token`: at concurrency c the requests share each pass, so each one
sees roughly c times its solo pace while the world delivers more in total.
Per-request pace is given beside the aggregate wherever both were recorded.

**Greedy and sampled differ, and the difference is acceptance.** At
temperature 0 the draft head agrees with the main model far more often than
under the card's sampling defaults. On Qwen3.8-Flash-Next the same build reads
72–74 % acceptance greedy and 48–55 % sampled. Every row says which it is.
Greedy is the right number for comparing engine work; sampled is the right
number for what a user gets at the default settings.

**Prefill is quoted per token** (`prefill ms / prompt tokens`), because it is
close to linear in the prompt and the per-token rate is what transfers between
lengths. Absolute milliseconds are given too, since time to first token is
what a caller waits.

**Run-to-run drift on this hardware is about 2 % across a day.** Single runs
are therefore quoted as ranges where a range was recorded, and every A/B in
the source documents is a pair of runs on one binary with one knob between
them. Do not read a 1 % difference between two rows of this page as a result,
especially when their dates differ.

**Dates matter more than usual here.** The engine changed substantially
between 2026-09-05 and 2026-09-10, so a row's date is part of the number. Where
an older per-class table has not been re-run against a newer build, that is
said in the section rather than silently corrected.

## 2. The environment and the deployments

Four NVIDIA DGX Spark (GB10) nodes, 121 GB of unified memory each, one
200 Gb/s RoCE port per node behind two PCIe x4 physical functions, CUDA 13.
Ranks are resident: each holds its slice of the model on the GPU for the life
of the world and boots from a per-rank image cache. All decode numbers are
from a warm world.

A deployment is a cluster config. These are the ones with a committed
template, which is what "supported" means on this page:

| model | world | template | modes measured |
|---|---|---|---|
| `unsloth/GLM-5.3-Flash-FP8` | 4 | `deploy/cluster.example.json` | T=1, MTP depth 1 |
| `dgpp/GLM-5.3-Flash-NVFP4-FP8` | 4 | `cluster.nvfp4.example.json` | T=1, MTP depth 1 |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 4 | `cluster_qwen.example.json`, `cluster_qwen_t1.example.json` | T=1, MTP depth 1, depth 2 |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 2 | `cluster_qwen_w2.example.json`, `cluster_qwen_w2_t1.example.json` | T=1, MTP depth 1, depth 2 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 1 | `cluster_qwen_spark1*.example.json` (five) | T=1, MTP depth 1, depth 2; BF16 or FP8 dense stack |
| `nvidia/GLM-4.7-NVFP4` | 4 | `cluster_glm47.example.json`, `_t1`, `_d2` | T=1, MTP depth 1, depth 2 |

The single-Spark world has a second axis, `engine.dense_weights`: the
checkpoint's own BF16 dense projections, or the same projections encoded to
block FP8 as they load. Both are measured below; FP8 is the faster one and the
BF16 one is the reference the transcripts are compared against.

GLM-5.3-Flash and GLM-4.7 have been measured only at world 4, and
Qwen3.8-Flash-Next-NVFP4 only at world 1. Those are deployment decisions
rather than limitations of the engine, which takes the world from the config.

## 3. Decode, one request

### GLM-5.3-Flash-FP8, world 4

| mode | ms/pass | tok/pass | ms/token | tokens/s | date |
|---|---|---|---|---|---|
| eager, no graph | 36.4 | 1.0 | 36.4 | 27.5 | 2026-09-03 |
| T=1, one graph | 31.45 | 1.0 | 31.45 | 31.8 | 2026-09-03 |
| T=1, one graph | 29.94–29.97 | 1.0 | 29.94–29.97 | 33.4 | 2026-09-10 |
| MTP, greedy | 41.6–42.9 | 1.77–1.97 | 21.3–23.9 | 42–47 | 2026-09-05 |
| MTP, sampled at the card's defaults | 42 | 1.5–1.7 | 25.0–27.6 | 36–40 | 2026-09-04 |
| MTP, greedy, chat prompt | 40.49 | 1.72 (72.0 % accepted) | 23.5 | 42.5 | 2026-09-10 |

The T=1 step reads p50 30.0 ms and p99 31.0 ms over the 2026-09-10 regression.
Through the HTTP service at concurrency 1 the same world measures 32.1 ms/token
at T=1 and 21.8–26.0 ms/token under MTP, so the service adds nothing measurable
to the engine's pace.

Context costs little: the step is about 41 ms under 2K tokens of context and
42–43 ms from 8K to 32K, after the 2026-09-06 select rewrite that took the long
end down from 52–56 ms.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 26.22 | 1.0 | 26.22 | 2026-09-08 |
| MTP, greedy | 33.89 | 1.71 | 19.77 | 2026-09-08 |
| MTP, sampled, through the service | 36.7 | 1.63 | 22.4 | 2026-09-08 |

The routed experts are NVFP4 and everything else is the FP8 release's own
bytes. Against the FP8 checkpoint on the same fabric this is 26.2 versus 31.4 ms
at T=1 and 19.77 versus 22.45 ms/token under greedy MTP, at equal quality (§7).

### Qwen3.8-Flash-Next-FP8, world 4 and world 2

| world | mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|---|
| 4 | T=1 | 21.40–21.50 | 1.0 | 21.40–21.50 | 2026-09-10 |
| 4 | MTP, greedy | 25.7 | ~2.0 | 12.9 | 2026-09-10 |
| 4 | MTP, sampled | 25.8–25.9 | 1.60–1.66 | 15.5–16.1 | 2026-09-10 |
| 4 | MTP depth 2, sampled | 30.8–30.9 | 1.97–2.03 | 15.2–15.6 | 2026-09-10 |
| 2 | T=1 | 32.5 | 1.0 | 32.5 | 2026-09-09 |
| 2 | MTP | 41 | 1.56–1.98 | 20.8–26.3 | 2026-09-09 |
| 2 | MTP depth 2 | 49.4 | 1.81–2.08 | 23.8–27.3 | 2026-09-10 |

Depth 2 is break-even at both worlds on this family and depth 1 is the default:
at world 4 it reads 15.2–15.6 ms/token against depth 1's 15.5, and at world 2
23.8–27.3 against 23.7–24.8. Acceptance on this family is 72–74 % greedy and
48–55 % sampled on the same build.

### Qwen3.8-Flash-Next-NVFP4, world 1 (one Spark)

| dense stack | mode | ms/pass | tok/pass | ms/token | tokens/s | date |
|---|---|---|---|---|---|---|
| BF16 | T=1 | 47 | 1.0 | 46.7–47.2 | 21.4 | 2026-09-10 |
| BF16 | MTP, greedy | 59 | 1.56–1.88 | 31.4–37.7 | 27–32 | 2026-09-10 |
| BF16 | MTP, sampled | 59 | 1.62 | 36.6 | 27 | 2026-09-10 |
| FP8 | T=1 | 30.2–30.9 | 1.0 | 30.2–30.9 | 32.3 | 2026-09-10 |
| FP8 | MTP, greedy | 39–40 | 1.54–1.93 | 20.5–25.8 | 39–49 | 2026-09-10 |
| FP8 | MTP, sampled | 39–40 | 1.6 | 24.7 | 40 | 2026-09-10 |
| FP8 | MTP depth 2, greedy | 48 | 1.77–2.74 | 17.4–27.1 | 37–57 | 2026-09-10 |

The FP8 dense stack is the same checkpoint with every dense projection encoded
to block FP8 at load. It costs about 4 GiB less resident, runs T=1 at 30.3 ms
against a 25.5 ms bandwidth floor, and its greedy transcripts diverge from the
BF16 stack's after 170–316 characters at equal eval scores (§7).

Depth 2 on the FP8 world is the only place in this project where depth 2 is
worth setting, and only for some classes: see §4.

### GLM-4.7-NVFP4, world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 49.0 | 1.0 | 49.0 | 2026-09-10 |
| MTP depth 1 | 60–61 | 1.86–1.99 | 31–33 | 2026-09-10 |
| MTP depth 2 | 72–73 | 2.32–2.60 | 28.0–31.2 | 2026-09-10 |

The T=1 floor here is about 40 ms, and 6.3 GB per rank per step of it is the
BF16 attention projections that modelopt left unquantized. That is also what
bounds this family under concurrency (§5).

## 4. Decode, per prompt class

Classes are five fixed prompts, greedy, 300 tokens each unless the table says
otherwise: **chat** a technical explanation, **code** a Python module with
tests, **prose** a long history, **json** twenty-five records, **math** a
worked problem. §9.2 has the exact texts and the ritual.

The step time is nearly flat across classes. What moves is draft acceptance,
and therefore tokens per pass, and therefore the pace.

### GLM-5.3-Flash-FP8, world 4, MTP greedy (2026-09-05)

| class | drafts accepted | tok/pass | ms/pass | ms/token |
|---|---|---|---|---|
| chat | 77.1 % | 1.77 | 42.2 | 23.9 |
| code | 97.4 % | 1.97 | 42.9 | 21.7 |
| prose | 87.5 % | 1.88 | 41.6 | 22.2 |
| json | 97.4 % | 1.97 | 42.1 | 21.3 |
| math | 92.3 % | 1.92 | 42.6 | 22.1 |

Net speedup against the plain graph's 31.45 ms/token: 1.31x on chat to 1.48x
on code and JSON. The step is 41.6–42.9 ms whatever the class, so acceptance is
the whole story.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4, MTP greedy (2026-09-08)

| class | drafts accepted | ms/token, hybrid | ms/token, FP8 |
|---|---|---|---|
| chat | 69.5 % | 21.7 | 24.3 |
| code | 98.0 % | 18.7 | 21.2 |
| prose | 88.7 % | 19.3 | 21.7 |
| json | 96.7 % | 18.6 | 21.0 |
| math | 88.7 % | 19.5 | 21.7 |

Measured at a graph step of 36.4–36.9 ms. The build that ships reads 33.89 ms,
so these per-class figures are conservative by roughly 8 % and have not been
re-run; §8 has the command.

### Qwen3.8-Flash-Next-NVFP4, world 1, MTP greedy (2026-09-10)

| class | BF16 dense: tok/pass, accept, ms/token | FP8 dense: ms/token | FP8 depth 2: ms/token |
|---|---|---|---|
| prose | 1.56, 56 %, 37.7 | 25.8 | 27.1 (+5 %) |
| code | 1.88, 88 %, 31.4 | 22.3 | 20.1 (−10 %) |
| math | 1.83, 83 %, 31.9 | 20.7 | 18.5 (−11 %) |
| json | 1.88, 88 %, 31.4 | 20.5 | 17.4 (−15 %) |

The pass is 59 ms on the BF16 stack, 39–40 on FP8, and 48 at depth 2. The
second draft stands 60–84 % of the time on code, math and JSON and only 26–32 %
on prose, which is why depth 2 pays on the first three and loses on the last.
A code-heavy deployment is the case for setting `mtp_depth: 2` here.

### GLM-4.7-NVFP4, world 4, MTP greedy, 256-token prompts (2026-09-10)

| class | depth 1: ms/pass, tok/pass, ms/token | depth 2: ms/pass, tok/pass, ms/token, p1 / p2 | tokens/s |
|---|---|---|---|
| chat | 61, 1.86, 32.5 | 72, 2.32, 31.2, 85 % / 48 % | +4 % |
| code | 61, 1.93, 33.0 | 72, 2.52, 28.7, 92 % / 60 % | +13 % |
| math | 62, 1.95, 31.3 | 72, 2.55, 28.4, 94 % / 62 % | +10 % |
| json | 61, 1.99, 31.1 | 73, 2.60, 28.0, 96 % / 65 % | +11 % |
| 6,525-token prompt | 66, 1.86, 35.4 | 79, 2.35, 33.7, 85 % / 52 % | +5 % |

Depth 2 gains 4–13 % single-stream here and loses under concurrency (§5), so
the shipped config keeps depth 1 and `cluster_glm47_d2.example.json` is the
single-stream recipe.

Qwen3.8-Flash-Next at worlds 2 and 4 has no per-class table; see §8.

## 5. Decode under concurrency

Concurrency is measured with a fixed set of long-answer technical essay
prompts, greedy, thinking off, one phase per concurrency (§9.3). It is
therefore **not** broken down by class: the per-class work in §4 is
single-stream only. Aggregate tokens/s is the phase's client-side total.

### GLM-5.3-Flash-FP8, world 4

The engine's batched decode under MTP, from the 2026-09-07 batch families. The
last column is what the same load cost before those families existed, when the
engine ran one scalar replay per live request in sequence:

| live requests | batch family | ms per pass | scalar replays in sequence |
|---|---|---|---|
| 1 | scalar | ~41 | ~41 |
| 2 | 4-row | 60 | 83 |
| 3 | 6-row | 90 | ~124 |
| 4 | 8-row | 107 | not recorded |

The v1 sign-off's aggregate curve predates those families and is kept for the
shape rather than the values: 30.97 / 31.67 / 41.76 / 76.18 tokens/s at 1 / 2 /
4 / 8 requests at T=1, and 38.48 / 38.97 / 60.27 at 1 / 2 / 4 under MTP
(2026-09-05).

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4

Only concurrency 4 was measured, inside the ten-minute mixed soak: 87 ms per
batched step, 1.88–1.93 tokens per step per request, 87–91 % acceptance,
64–69 tokens/s aggregate, 11.8 ms/token per request, over 523 requests with
none failed.

### Qwen3.8-Flash-Next-FP8

World 4, aggregate tokens/s, greedy, the MTP world, one build (2026-09-10):

| c=1 | c=2 | c=4 |
|---|---|---|
| 68.2 | 102.5 | 120.0 |

This is the only clean c=1/2/4 aggregate series in the project. World 2 has
concurrency 4 alone (2026-09-09): 95–143 ms per batched step at 1.9 tokens per
step per request, 12–18 ms/token. Its c=1 and c=2 points are unmeasured.

### Qwen3.8-Flash-Next-NVFP4, world 1

| dense stack | c=4: ms/step | c=4: aggregate tokens/s | per-request ms/token |
|---|---|---|---|
| BF16 | 119–124 | 49–55 | 16.2 |
| FP8 | 70.7 | 62.9 | — |

Both at 1.93–1.95 tokens per step per request and 92–94 % first-draft
acceptance. Concurrency 2 has not been measured on this world.

### GLM-4.7-NVFP4, world 4

| live requests | depth 1: ms/pass, tok/step/req, aggregate tok/s (per request) | depth 2: same | depth 2 vs 1 |
|---|---|---|---|
| 1 | 60, 1.89, 31.0 | 72, 2.31, 32.2 | +4 % |
| 2 | 79, 1.85–1.88, 44.5 (22.3) | 123–125, 2.3–2.6, 37.0 (18.5) | −17 % |
| 4 | 139–143, 1.74–1.93, 47.5 (11.9) | 198–204, 2.2–2.4, 42.8 (10.7) | −10 % |

This family scales worst under load, and the reason is measured rather than
guessed: each 4-row GEMV chunk past the first re-reads the 6.3 GB per rank of
BF16 attention projections, adding 45–60 ms, while a row inside a chunk costs
7–8 ms. Hence 72 ms per pass at 3 rows, 79 at 4, 123 at 6, 140 at 8 and 200 at
12. Depth 2 doubles the rows and therefore the chunks, which is why it is a
single-stream setting here.

## 6. Prefill and time to first token

Prefill is measured in steady state, with the prefix cache prevented from
attaching, so these are cold-prompt costs.

| model | world | 512 | 2,048 | 8,192 | 32,768 | date |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 4 | 549 ms | 1,417 ms | 6,286 ms | — | 2026-09-10 |
| GLM-5.3-Flash-FP8 | 4 | 581 ms at 256 tokens | 1,701 ms | 7,260 ms | 33,728 ms at 32K | 2026-09-05 |
| GLM-5.3 NVFP4 hybrid | 4 | 425 ms | 1,275 ms | 5,702 ms | — | 2026-09-08 |
| Qwen3.8-Flash-Next-FP8 | 4 | per-token only | per-token only | per-token only | — | 2026-09-09 |
| Qwen3.8-Flash-Next-FP8 | 2 | 612 ms | 1,508 ms | 5,817 ms | — | 2026-09-09 |
| Qwen NVFP4, BF16 dense | 1 | 785 ms | 2,028 ms | 8,392 ms | — | 2026-09-10 |
| Qwen NVFP4, FP8 dense | 1 | per-token only | per-token only | per-token only | — | 2026-09-10 |

Per token, so the lengths compare:

| model | world | 512 | 2,048 | 8,192 |
|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 4 | 1.07 | 0.69 | 0.77 |
| GLM-5.3 NVFP4 hybrid | 4 | 0.83 | 0.62 | 0.70 |
| Qwen3.8-Flash-Next-FP8 | 4 | 0.87 | 0.62 | 0.58 |
| Qwen3.8-Flash-Next-FP8 | 2 | 1.16 | 0.76 | 0.75 |
| Qwen NVFP4, BF16 dense | 1 | 1.50 | 1.03 | 1.08 |
| Qwen NVFP4, FP8 dense | 1 | 1.83 | 1.14 | 1.17 (see note) |

The FP8 dense stack's 512 and 2,048 figures are from the fused build; its
8,192 figure was taken on the unfused build one step earlier, where 512 and
2,048 read 1.85 and 1.15, so it is within a percent of the fused build's rate.

The FP8 dense stack is slower at prefill than the BF16 one because the GEMM
seam dequantizes into a BF16 bridge per chunk. It is a decode optimization that
costs prefill, and both are the same checkpoint.

The world-2 and world-1 rows were taken at 528 / 1,981 / 7,742 real tokens; the
world-4 Qwen row at the same lengths. Prompt lengths for the others are the
column headings.

GLM-4.7-NVFP4 at world 4 has only short-prompt prefill measured: 400–1,000 ms
for prompts of 31–150 tokens, which is 5–13 ms/token on the v1 warp-per-row
attention, and 11.5 s for a 6,525-token prompt (1.77 ms/token). The tensor-core
form of that kernel is open work.

**Time to first token**, GLM-5.3-Flash-FP8 through the service, is prefill plus
the pick plus the HTTP hop: a 25-token prompt 1,109 ms cold and 751 ms hot, a
1,592-token prompt 2,643 ms cold and 949 ms hot. Over a mixed conversation
workload the split was 179 ms on prefix-cache hits against 1,141 ms on misses.
On the single Spark, TTFT is 506 ms at T=1 and 532 ms under MTP for a short
prompt.

## 7. Quality, so the throughput numbers are comparable

A faster configuration is only interesting if it answers as well. Every world
above is checked with the same task-level runner through the same endpoint,
greedy (§9.5). Item counts differ between campaigns and are given, since a
score is not comparable across different denominators.

| model | world | HumanEval | GSM8K | schema extraction |
|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 4 | 155/164 (94.5 %) | 293/300 (97.7 %) | 100/100 |
| GLM-5.3 NVFP4 hybrid | 4 | 157/164 (95.7 %) | 293/300 (97.7 %) | 100/100 |
| Qwen3.8-Flash-Next-FP8 | 4 | 39/40 | 59/60 | 30/30 |
| Qwen3.8-Flash-Next-FP8 | 2 | 38/40 | 59/60 | 30/30 |
| Qwen NVFP4, BF16 dense | 1 | 39/40 | 59/60 | 30/30 |
| Qwen NVFP4, FP8 dense | 1 | 38/40 | 59/60 | 30/30 |
| GLM-4.7-NVFP4 | 4 | 39/40 | 60/60 | 30/30 |

The two GLM-5.3 checkpoints were run on identical items: 155 HumanEval problems
pass on both, 7 fail on both, and 2 pass only on the hybrid. The single-point
HumanEval differences in the smaller campaigns sit on a near tie that has moved
both ways between runs of the same build, so they are not a ranking.

Two further identities are checked on every world and are pass/fail rather than
scores: **MTP produces the plain greedy transcript token for token**, and
**every rank's op stream is identical**, verified by four matching md5 sums at
shutdown. Both hold on every configuration on this page.

## 8. What is not measured yet

Each gap is a real hole in the matrix, with the command that fills it. None of
these are blocked; they are fabric time nobody has spent.

1. **Per-class decode at concurrency 2 and 4, every model.** The whole of §4 is
   single-stream. The concurrency harness uses one fixed essay prompt set and
   reports aggregates, so no model has a class breakdown under load. Filling
   this needs a per-class mode in the load script, not just a longer run.
2. **Per-class decode for Qwen3.8-Flash-Next at worlds 2 and 4.** Only ranges
   across the four prompts were recorded. Fill with
   `scripts/mtp_depth_check.py` beside rank 0's retire lines, or
   `fabric_mtp_classes.sh --config deploy/cluster_qwen.json OUT chat code prose json math`.
3. **Concurrency 2 for the single Spark and for world 2.** Both jump from 1 to
   4. Fill with `scripts/serve_load.py HOST PORT --concurrency 1,2,4`.
4. **Concurrency 2 and the c=1/2/4 series for the GLM-5.3 NVFP4 hybrid.** Only
   the soak's four-live point exists.
5. **The GLM-5.3 hybrid's per-class table on the shipped build.** The table in
   §4 was taken at 36.4–36.9 ms/pass; the build reads 33.89.
6. **GLM-4.7 prefill at 512 / 2,048 / 8,192.** Only short prompts and one
   6,525-token prompt exist. Fill with
   `scripts/serve_prefill_probe.py HOST PORT 512 2048 8192 --repeat 3`.
7. **Long-context prefill beyond 8K for every family except GLM-5.3-Flash-FP8**,
   which has 32,768.
8. **Token/s under sampling at concurrency**, for every model. The concurrency
   harness runs greedy.

## 9. Reproducing all of it

Everything below runs against a booted world. Bring one up with the launcher
and the config for the deployment you are measuring:

```bash
cp deploy/cluster_qwen.example.json deploy/cluster_qwen.json   # fill in nodes and ssh_user
scripts/dgpp-cluster up --config deploy/cluster_qwen.json
```

Run one ritual at a time on the fabric. Two measurements at once share the
memory fabric and both readings are then wrong.

### 9.1 Single-stream decode

Through the service, which is what a user feels:

```bash
scripts/serve_bench.py HOST PORT 300 label
```

It reports time to first token, the decode pace between the first and last
content chunk, and the server's usage line. The engine's own view is rank 0's
stats line every 10 seconds, which carries `ms/step`, `tok/step/req` and the
measured acceptance per draft position. For the mode sweep without the HTTP
path, `glm_gen_check --decode-graph [--mtp]` runs the same recipe on the
fabric directly.

### 9.2 Per prompt class

```bash
scripts/fabric_mtp_classes.sh --config deploy/cluster.json OUT_DIR chat code prose json math
```

Five fixed prompts, greedy, 300 steps each, all four ranks, transcripts
compared across ranks. The prompts are in the script: chat asks why a CUDA
graph replay beats eager launching, code asks for a Python SSE parser with
pytest tests, prose asks for a long history, json asks for twenty-five records,
math asks for a worked problem. Through the endpoint instead, and for
comparing transcripts across MTP depths:

```bash
scripts/mtp_depth_check.py --out /tmp/depth1 --max-tokens 300
diff -r /tmp/depth1 /tmp/depth2      # must be byte-identical
```

### 9.3 Concurrency

```bash
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --max-tokens 320
```

For each concurrency it opens that many streaming requests at once with
distinct long-answer prompts, greedy, thinking off, and reports per request the
token count, time to first token and client-side pace, and per phase the wall
time, the aggregate decode tokens/s and the phase's stamps so rank 0's stats
lines can be laid beside it. Add `--isolation 4` to assert that a prompt's
greedy answer alone is byte-identical to its answer beside three others.
`scripts/fabric_glm4_load.sh CONFIG OUT` wraps this for the GLM-4.7 worlds.

### 9.4 Prefill

Through the endpoint, with the prefix cache defeated by a unique nonce per
prompt and the cost read from `/v1/metrics` deltas:

```bash
scripts/serve_prefill_probe.py HOST PORT 512 2048 8192 --repeat 3
```

On the fabric directly, in steady state, where the timed prefill is the third
repeat and the four ranks' generated ids are compared:

```bash
scripts/fabric_prefill_repeat.sh --config deploy/cluster.json OUT_DIR 512 2048 8192
```

### 9.5 Quality

```bash
scripts/serve_eval.py HOST PORT --out OUT_DIR --tasks humaneval,gsm8k,extract [--limit N] [--no-think]
```

Greedy, the served model taken from `GET /v1/models`, HumanEval completions
executed against the problem's own tests in a subprocess. Pass `--limit` to
run a subset, and record the denominator with the score. The data lives in
`build-ci/eval_data`.

### 9.6 The determinism checks that make a benchmark meaningful

A throughput number from a world that is not deterministic is not worth
recording. Every campaign above ran with these:

- rank agreement: `scripts/dgpp-cluster down` prints one op-stream md5 per
  rank and the four must be identical;
- speculative decode identity: the MTP transcript must equal the plain greedy
  transcript, checked by `mtp_depth_check.py` and `diff -r`;
- batching isolation: `serve_load.py --isolation C`.

### 9.7 Where the raw records live

`benchmarks/results/` holds the dated engineering record behind every number
on this page. `docs/signoff_v1.md` is the v1 sign-off with its own method
statement, `docs/measurements.md` the platform and GLM-4.7 measurements,
`docs/qwen38_single_spark.md` the single-Spark campaign, and
`docs/qwen38_optimization_plan.md` and `docs/nvfp4_plan.md` the optimization
rounds with their A/B pairs.
