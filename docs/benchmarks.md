# Benchmarks

This page collects serving measurements by model, world size, concurrency
and prompt class, with commands for reproducing them. Results are dated;
modeled bandwidth floors and unmeasured cases are identified separately.
Section 8 lists missing measurements, and section 9 gives the procedures.

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

The tables predating September 15 use `serve_load.py`'s historical output
span: earliest first visible output to latest last visible output. It omits
initial TTFT and includes later admissions. The script now labels this
`legacy_output_span_tokens_per_s` and reports request-wall throughput
separately. Its per-request latency is **ms per client update**; an SSE
update can contain several tokens. Neither rate establishes a period with
constant server occupancy. Token rates use the server's completion usage.

For a C1 regression check, capture both builds with
`--classes all --concurrency 1 --repeat 3 --json-out RUN.json`, using the
same model, deployment and `--max-tokens`. Then run
`python3 scripts/bench_compare.py BASELINE.json CANDIDATE.json`. It requires
matching greedy C1 transcripts, checks medians in both timing scopes, and
prints the min/max spread. Its default allows no median slowdown; resolve
small noisy differences with repeated A/B runs, rather than selecting a
favorable sample. The corpus retains the five C1 anchors and now provides
sixteen distinct prompts per class for wider request bursts.

The [September 15 Qwen batching and continuation campaign](../benchmarks/results/2026-09-15-qwen-batching-prefill.md)
records the matched baseline, configuration tradeoffs and regression gates
for the new paths. Historical tables below retain their original scopes.

**Greedy and sampled used to differ, and no longer do.** A deterministic
draft is capped at the main model's mode, so under temperature sampling it was
accepted far less often: Qwen3.8-Flash-Next read 72–74 % acceptance greedy
against 48–55 % sampled. The ratio-rule proposal draft closed that, and the
2026-09-10 sweeps measure greedy and sampled throughput as the same number on
every family. Rows still say which they are, and the older sign-off rows in §3
still show the old gap, because they predate the change.

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
| `unsloth/GLM-5.3-Flash-FP8` | 4 | Copy `cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json` to a separate FP8 config and set `model` to the FP8 repository | T=1 (`--no-mtp`), MTP depth 1 |
| `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` | 4 | `cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json` | T=1, MTP depth 1 |
| `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` | 2 | `cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json` (the 256K-context two-slot shape: `--knobs "--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2"`) | T=1, MTP depth 1 |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 4 | `cluster_qwen-3.8-flash-next_fp8_w4.example.json` | T=1 (`--no-mtp`), MTP depth 1, depth 2 (`--mtp-depth 2`) |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 2 | `cluster_qwen-3.8-flash-next_fp8_w2.example.json` | T=1, MTP depth 1, depth 2 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 1 | `cluster_qwen-3.8-flash-next_nvfp4_w1.example.json` (the FP8 dense stack; `--dense-weights checkpoint` for BF16) | T=1, MTP depth 1, depth 2; BF16 or FP8 dense stack |
| `nvidia/GLM-4.7-NVFP4` | 4 | `cluster_glm-4.7_nvfp4_w4.example.json` | T=1, MTP depth 1, depth 2 |
| `HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64` (the full GLM-5.3) | 4 | `cluster_glm-5.3_int4-int8_w4.example.json` (eight slots; the fp8 latent cache at 208K: `--knobs "--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5"`) | T=1, MTP depth 1, depth 2 (two slots); bf16 or fp8 latent cache |
| `deepseek-ai/DeepSeek-V4.1-Flash` | 4 | `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json` (six slots at DSpark depth 4; the two-slot depth-5 shape: `--knobs "--max-concurrency 2 --mtp-depth 5"`) | T=1, DSpark depths 4 and 5 with the scheduled verify depth |

The single-Spark world has a second axis, `engine.dense_weights`: the
checkpoint's own BF16 dense projections, or the same projections encoded to
block FP8 as they load. Both are measured below; FP8 is the faster one and the
BF16 one is the reference the transcripts are compared against.

GLM-4.7 has been measured only at world 4, and Qwen3.8-Flash-Next-NVFP4 only
at world 1. Those are deployment decisions rather than limitations of the
engine, which takes the world from the config. The GLM-5.3 hybrid is measured
at worlds 4 and 2; the two-node deployment carries the same weights on half
the ranks, so each rank holds 94.71 GiB of them instead of 50.74 GiB, and its
latent cache is FP8 rather than BF16 to buy back context (160K tokens at four
request slots).

## 3. Decode, one request

### GLM-5.3-Flash-FP8, world 4

| mode | ms/pass | tok/pass | ms/token | tokens/s | date |
|---|---|---|---|---|---|
| eager, no graph | 36.4 | 1.0 | 36.4 | 27.5 | 2026-09-03 |
| T=1, one graph | 31.45 | 1.0 | 31.45 | 31.8 | 2026-09-03 |
| T=1, one graph | 29.94–29.97 | 1.0 | 29.94–29.97 | 33.4 | 2026-09-10 |
| MTP, greedy | 41.6–42.9 | 1.77–1.97 | 21.3–23.9 | 42–47 | 2026-09-05 |
| MTP, sampled at the card's defaults (superseded) | 42 | 1.5–1.7 | 25.0–27.6 | 36–40 | 2026-09-04 |
| MTP, greedy, chat prompt | 40.49 | 1.72 (72.0 % accepted) | 23.5 | 42.5 | 2026-09-10 |

The sampled row predates the ratio-rule proposal draft and no longer holds:
measured on 2026-09-10, sampling costs this world nothing against greedy (§5).

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

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 45.68 | 1.0 | 45.68 | 2026-09-12 |
| MTP, greedy | 59.40 | 1.734 | 34.25 | 2026-09-12 |
| MTP, sampled, through the service | 60 | 1.70–1.81 | 34.1–35.3 | 2026-09-12 |

The same weights on half the ranks: 1.74x the four-node pass at T=1, 1.75x under
MTP. The T=1 step is 45.68 ms mean with p99 47, so the world is as steady as the
four-node one. The MTP and T=1 runs produced the same rank-consistency digest,
which is the speculative-identity gate (§9.6) passing on this world.

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

### The full GLM-5.3 (int4/int8 g64), world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 51.1 | 1.0 | 51.1 | 2026-09-12 |
| MTP depth 1 | 68–76 | 1.77–1.97 | 36–42 | 2026-09-12 |
| MTP depth 2 (two slots) | 86–89 | 2.15–2.69 | 32–40 | 2026-09-13 |

The 754B model at 99.3 GiB of int4/int8 weights per rank: the T=1 step
streams about 10.2 GB per rank (the audit's floor 44.5 ms at 230 GB/s) plus
158 collectives. MTP at depth 1 gives 1.4x per token. Transcripts: MTP == T=1
on all four prompts; op streams identical across the four ranks at every
shutdown (§9.6). Boot from the resident image 30 s; the first boot, which
captures it, 405 s. The campaign ran at 48K tokens of bf16 latent cache
because rank 2's node then had 119.67 GiB (its launch firmware's 4 GB
display reservation; fixed by the SoC firmware update the same night); the
templates carry 120K bf16 at four slots with MTP (110.41 GiB per rank), 144K
plain (110.75) and 208K with the fp8 latent cache (110.35), the embedding
vocab-sharded (−1.33 GiB, exact), the ceilings under the 4 GiB headroom that
replaced the 8 GiB one on 2026-09-12/13 once the loader's 2.2 GiB staging
mirror was measured, itemized and freed before the caches and a one-hour soak
at the 120K shape stayed flat with no reclaim on any node.
The fp8 latent cache runs at the same pace (68–69 ms/pass, 74–98 %
acceptance by class) and its greedy transcripts diverge from the bf16
cache's after 97–464 characters, the two-node Flash finding again.

## 4. Decode, per prompt class

Classes are five fixed prompts, greedy, 300 tokens each unless the table says
otherwise: **chat** a technical explanation, **code** a Python module with
tests, **prose** a long history, **json** twenty-five records, **math** a
worked problem. §9.2 has the exact texts and the procedure.

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

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4, MTP greedy (2026-09-10)

| class | drafts accepted | tok/pass | ms/pass | ms/token |
|---|---|---|---|---|
| chat | 67.6 % | 1.676 | 33.99 | 20.28 |
| code | 98.0 % | 1.974 | 34.31 | 17.38 |
| prose | 87.0 % | 1.863 | 33.66 | 18.07 |
| json | 96.7 % | 1.961 | 33.82 | 17.25 |
| math | 88.1 % | 1.875 | 34.20 | 18.24 |

All four ranks' transcripts identical on every class. This replaces a table
taken at a graph step of 36.4–36.9 ms, which read chat 21.7, code 18.7, prose
19.3, json 18.6 and math 19.5; the step is now 33.7–34.3 and every class gained
about 7 %.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2, MTP greedy (2026-09-12)

| class | drafts accepted | tok/pass | ms/pass | ms/token |
|---|---|---|---|---|
| chat | 73.4 % | 1.734 | 59.40 | 34.25 |
| code | 98.0 % | 1.974 | 60.17 | 30.49 |
| prose | 83.5 % | 1.829 | 58.71 | 32.09 |
| json | 97.4 % | 1.974 | 59.16 | 29.98 |
| math | 88.7 % | 1.887 | 59.96 | 31.78 |

Both ranks' transcripts identical on every class. Acceptance matches the
four-node run class for class, so the whole difference is the pass: 58.7–60.2 ms
against 33.7–34.3, a factor of 1.75 that follows the doubled per-rank weight
read. Per token the two-node world lands at 1.69x to 1.76x of the four-node
figures.

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
the shipped config keeps depth 1 and `cluster_glm-4.7_nvfp4_w4_mtp2.example.json` is the
single-stream recipe.

### Qwen3.8-Flash-Next-FP8, MTP greedy, through the service (2026-09-10)

Client-side pace of one request, ms/token, at concurrency 1:

| class | world 4 | world 2 |
|---|---|---|
| prose | 15.03 | 22.62 |
| code | 13.65 | 20.82 |
| json | 12.97 | 20.23 |
| math | 13.61 | 21.37 |
| chat | 15.58 | 24.19 |

These come from the service rather than `glm_gen_check`, which is a GLM-only
evidence app and refuses this checkpoint. §9.2 says what that changes.

### GLM-4.7-NVFP4, world 4, MTP depth 1, through the service (2026-09-10)

| class | ms/token at c=1 |
|---|---|
| prose | 32.71 |
| code | 32.51 |
| json | 31.23 |
| math | 31.66 |
| chat | 35.11 |

### The full GLM-5.3 (int4/int8), world 4, MTP depth 1, through the service (2026-09-12)

`serve_mtp_classes.py`, greedy, 300 tokens; the pass time, tokens per pass
and acceptance are the scheduler's own retired lines:

| class | ms/pass | tok/pass | accept p1 | ms/token at c=1 |
|---|---|---|---|---|
| prose | 76 | 1.92 | 92 % | 42.1 |
| code | 69 | 1.97 | 97 % | 37.6 |
| json | 68 | 1.95 | 96 % | 37.5 |
| math | 69 | 1.93 | 93 % | 38.6 |
| chat | 68 | 1.77 | 77 % | 41.1 |

At T=1 every class runs 51 ms/step (51.5–54.4 ms/token wall).

Depth 2 (`cluster_glm-5.3_int4-int8_w4_mtp2`, two request slots — the
shape the eight-row cap allowed when it was measured; the cap is sixteen
rows since 2026-09-13, five slots at depth 2; transcripts identical to
depth 1's):

| class | ms/pass | tok/pass | accept p1 / p2 | ms/token | vs depth 1 |
|---|---|---|---|---|---|
| prose | 86 | 2.56 | 92 / 63 % | 33.5 | −4 % |
| code | 88 | 2.69 | 95 / 76 % | 32.7 | −5 % |
| json | 87 | 2.69 | 93 / 77 % | 32.3 | −7 % |
| math | 89 | 2.51 | 89 / 64 % | 35.2 | 0 % |
| chat | 87 | 2.15 | 73 / 42 % | 40.4 | +5 % |

Depth 3 (a site config at two slots, 2 × 4 rows): 103–106 ms/pass at
2.37–3.22 tokens/pass (p3 18–56 %) — 32.9 / 33.0 / 35.8 / 36.8 / 43.9
ms/token for code / json / prose / math / chat, i.e. equal to depth 2 on
code and JSON and worse elsewhere. The pass grows by 17–19 ms per depth
level (51 → 68 → 87 → 104: one verify row's distinct experts, ~2.7 GB per
rank, plus one serial draft step) while tokens per pass grow sublinearly,
so on this bandwidth-bound fabric depth 1 stays the default and depth 2 is
a code/JSON option; the three-token guidance for these models comes from
nodes where a verify row costs almost nothing.

### GLM-5.3-Flash-FP8, world 4, cross-check (2026-09-10)

The flagship's per-class pace measured through the service on the same corpus,
beside the 2026-09-05 sign-off's `glm_gen_check` table:

| class | service, ms/token | sign-off, ms/token |
|---|---|---|
| prose | 21.85 | 22.2 |
| code | 21.01 | 21.7 |
| json | 20.62 | 21.3 |
| math | 22.02 | 22.1 |
| chat | 23.79 | 23.9 |

Two methods, five days apart, agreeing within 0.7 ms/token. The same check on
the hybrid agrees within 0.05–0.44 ms/token against its engine-side table
above. The service costs nothing measurable, and the per-class numbers on this
page corroborate each other across both paths.

## 5. Decode under concurrency

Every table here is **aggregate tokens/s per prompt class**, measured 2026-09-10
on one binary with `serve_load.py --classes all` (§9.3): for each class and each
concurrency, that many streaming requests at once drawn from that class alone.
Prompt 0 of each class is the same text the single-stream corpus uses, so the
c=1 column is comparable to §4; prompts 1 to 3 exist because a phase at
concurrency c needs c distinct requests.

Read down a column for how a class costs, and across a row for how the world
scales. Aggregate tokens/s is the phase total, not per request.

### GLM-5.3-Flash-FP8, world 4

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 45.9 | 55.8 | 62.8 | 21.85 |
| code | 47.7 | 57.2 | 65.5 | 21.01 |
| json | 48.7 | 55.6 | 67.5 | 20.62 |
| math | 45.5 | 59.5 | 67.3 | 22.02 |
| chat | 42.2 | 55.7 | 62.2 | 23.79 |

Sampled at the card's defaults the same sweep reads prose 45.6 / 58.0 / 62.4,
code 46.0 / 58.3 / 65.1, json 47.8 / 54.9 / 66.6, math 42.5 / 59.0 / 66.7 and
chat 39.9 / 55.1 / 60.4: within the drift of the greedy run.

The engine's own batched decode under MTP, from the 2026-09-07 batch families.
The last column is what the same load cost before those families existed, when
the engine ran one scalar replay per live request in sequence:

| live requests | batch family | ms per pass | scalar replays in sequence |
|---|---|---|---|
| 1 | scalar | ~41 | ~41 |
| 2 | 4-row | 60 | 83 |
| 3 | 6-row | 90 | ~124 |
| 4 | 8-row | 107 | not recorded |

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 55.6 | 74.3 | 87.1 | 18.03 |
| code | 57.8 | 76.4 | 88.1 | 17.36 |
| json | 58.2 | 71.7 | 86.6 | 17.24 |
| math | 55.7 | 77.7 | 91.3 | 18.00 |
| chat | 50.6 | 71.7 | 83.0 | 19.84 |

Sampled: prose 55.1 / 74.6 / 82.9, code 57.5 / 74.3 / 87.1, json 56.2 / 78.4 /
89.6, math 55.1 / 79.5 / 88.1, chat 49.8 / 74.3 / 81.8. Server-side the pass is
34–41 ms at c=1, 52–63 at c=2 and 68–84 at c=4, with acceptance 74–93 % greedy
and 71–92 % sampled.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2 (2026-09-12)

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 29.5 | 39.7 | 39.5 | 34.01 |
| code | 32.6 | 42.1 | 41.9 | 30.73 |
| json | 33.0 | 35.5 | 44.9 | 30.43 |
| math | 29.5 | 36.8 | 47.3 | 34.06 |
| chat | 22.7 | 40.5 | 38.9 | 44.25 |

Half the ranks hold twice the weights each, and the sweep reads 53 to 57 % of
the four-node aggregate in every class and at every concurrency. The four-node
world also keeps scaling to c=4 in each class, while two nodes flatten between
c=2 and c=4 on prose, code and chat: the per-rank weight read is already the
step's floor, so a wider batch buys less. Server-side, from the retire lines,
the pass is 60–75 ms at c=1, 82–104 at c=2 and 151–179 at c=4, with acceptance
69–98 %.

**Sampling no longer costs acceptance on this family.** The page's §3 still
carries the older sign-off reading of 25.0–27.6 ms/token sampled against
21.3–23.9 greedy for the FP8 checkpoint. That gap predates the ratio-rule
proposal draft; measured now, the sampled and greedy sweeps agree everywhere on
both GLM-5.3 checkpoints and on Qwen.

### Qwen3.8-Flash-Next-FP8, world 4

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 66.8 | 101.7 | 123.9 | 15.03 |
| code | 73.5 | 111.6 | 137.9 | 13.65 |
| json | 77.3 | 117.0 | 147.1 | 12.97 |
| math | 73.7 | 114.5 | 139.6 | 13.61 |
| chat | 64.4 | 100.4 | 123.6 | 15.58 |

Sampled: prose 68.7 / 102.7 / 122.0, code 74.6 / 111.0 / 136.3, json 76.3 /
118.2 / 145.4, math 73.7 / 114.4 / 137.8, chat 64.9 / 102.2 / 126.3. The pass is
28–31 ms at c=1 and 46–51 at c=4, acceptance 68–99 % by class.

This is the fastest world in the project at every concurrency.

### Qwen3.8-Flash-Next-FP8, world 2

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 44.4 | 62.4 | 71.0 | 22.62 |
| code | 48.2 | 72.3 | 80.9 | 20.82 |
| json | 49.6 | 74.4 | 83.2 | 20.23 |
| math | 46.9 | 73.4 | 80.4 | 21.37 |
| chat | 41.7 | 63.8 | 69.3 | 24.19 |

Sampled: prose 43.1 / 59.1 / 75.2, code 47.9 / 65.6 / 81.7, json 49.2 / 68.9 /
86.0, math 48.1 / 68.0 / 82.5, chat 42.3 / 60.4 / 75.2. The pass is 48–58 ms at
c=1 and about 89 at c=4. This world previously had no concurrency-1 or -2 point
at all.

### Qwen3.8-Flash-Next-NVFP4, world 1, FP8 dense

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 44.7 | 61.9 | 70.9 | 22.45 |
| code | 48.3 | 69.6 | 80.1 | 20.76 |
| json | 50.3 | 71.2 | 83.7 | 19.94 |
| math | 48.0 | 69.3 | 79.3 | 20.90 |
| chat | 42.6 | 63.0 | 69.4 | 23.66 |

**One Spark matches two.** Set this table beside the world-2 one above: every
class at every concurrency lands within a few percent. A single Spark serving
the NVFP4 checkpoint with its dense stack encoded to FP8 delivers what two
Sparks deliver on the FP8 checkpoint, at a quarter of the eval difference (§7).
The earlier BF16-dense reading on this world was 119–124 ms per step and 49–55
tokens/s at four live requests.

### GLM-4.7-NVFP4, world 4

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 30.7 | 43.6 | 49.2 | 32.71 |
| code | 30.9 | 45.8 | 51.4 | 32.51 |
| json | 32.1 | 46.3 | 53.5 | 31.23 |
| math | 31.7 | 44.9 | 51.5 | 31.66 |
| chat | 28.6 | 42.3 | 46.5 | 35.11 |

The depth study, on the mixed essay set rather than per class:

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
single-stream setting here. The class table above shows the same ceiling: going
from one request to four buys 1.6x, where Qwen at world 4 buys 1.9x.

### The full GLM-5.3 (int4/int8), world 4 (2026-09-12)

`serve_load.py --think` (the template has no thinking switch), greedy, 320
tokens, MTP depth 1; aggregate tokens/s:

| class | c=1 | c=2 | c=4 |
|---|---|---|---|
| prose | 25.5 | 34.5 | 39.6 |
| code | 28.7 | 32.7 | 39.9 |
| json | 28.5 | 33.1 | 43.2 |
| math | 26.3 | 36.5 | 41.3 |
| chat | 23.2 | 33.9 | 37.8 |

Per-request decode pace 38–43 ms/token at one request, 52–59 at two, 95–105
at four (the eight-row batch steps at 180–185 ms with 1.95 tok/step/req at
92–96 % acceptance). Four requests buy 1.5–1.6x over one, GLM-4.7's shape:
the packed attention projections and the grouped-GEMV experts re-read their
weights per row chunk.

## 6. Prefill and time to first token

**There are two prefill numbers for any world, and they differ by 1.4x to 2x.**
Which one is right depends on the question.

The **procedure** figure (`fabric_prefill_repeat.sh`) prefills the same prompt
three times and times the third, feeding token ids straight to the engine with
no HTTP. It is a warm best case and the right number for judging a kernel
change. The **service** figure (`serve_prefill_probe.py`) sends a fresh prompt
behind a unique nonce so the prefix cache can never attach, through the API,
and reads the engine's own `prefill_ms`. It is what a caller actually waits.
Everything published on this page before 2026-09-10 was the procedure figure.

Both were run on the same binary on 2026-09-10, and the older procedure numbers
reproduce exactly, so the difference is method and not a regression.

### Ritual: steady state, ids fed directly (2026-09-10)

| model | 512 | 2,048 | 8,192 | 32,768 |
|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 553 ms | 1,423 ms | 6,320 ms | 33,728 ms (2026-09-05) |
| GLM-5.3 NVFP4 hybrid | 425 ms | 1,269 ms | 5,696 ms | — |
| GLM-5.3 NVFP4 hybrid, world 2 | 664 ms | 1,829 ms | 8,280 ms | — |
| Qwen3.8-Flash-Next-FP8, world 4 | 0.87 ms/tok | 0.62 ms/tok | 0.58 ms/tok | — |
| Qwen3.8-Flash-Next-FP8, world 2 | 612 ms | 1,508 ms | 5,817 ms | — |
| Qwen NVFP4, BF16 dense, world 1 | 785 ms | 2,028 ms | 8,392 ms | — |
| Qwen NVFP4, FP8 dense, world 1 | 1.83 ms/tok | 1.14 ms/tok | 1.17 ms/tok | — |

The GLM rows are tonight's re-runs and land within 1 % of their published
values (549 / 1,417 / 6,286 and 425 / 1,275 / 5,702). The Qwen rows are the
2026-09-09 and 2026-09-10 campaign values, not re-run tonight.

The world-2 row was measured on 2026-09-12 from ids tokenized out of the pinned
GSM8K corpus, because the hard-prose id file the earlier rows used is not in the
tree. Prefill cost is dominated by length rather than content, but the two rows
do not share a text, so read the world-2 row against its own world-4 row with
that in mind.

### Service: a fresh prompt through the API (2026-09-10)

Median of three, with the real token count the tokenizer produced:

| model | ~520 | ~2,100 | ~8,400 | ~16,800 | ~33,800 |
|---|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 855 ms | 2,211 ms | 9,165 ms | 21,738 ms | 58,555 ms |
| GLM-5.3 NVFP4 hybrid | 734 ms | 2,210 ms | 11,374 ms | — | 93,765 ms |
| GLM-5.3 NVFP4 hybrid, world 2 | 1,088 ms | 2,724 ms | 11,881 ms | — | — |
| GLM-4.7-NVFP4 | 850 ms | 2,809 ms | 16,865 ms | — | — |
| GLM-5.3 full int4/int8 | 3,872 ms | 15,090 ms | 70,142 ms | 159,846 ms | — |

Per prompt token:

| model | ~520 | ~2,100 | ~8,400 | ~16,800 | ~33,800 |
|---|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 1.64 | 1.05 | 1.09 | 1.29 | 1.73 |
| GLM-5.3 NVFP4 hybrid | 1.41 | 1.05 | 1.36 | — | 2.77 |
| GLM-5.3 NVFP4 hybrid, world 2 | 2.04 | 1.30 | 1.42 | — | — |
| GLM-4.7-NVFP4 | 1.63 | 1.32 | 1.98 | — | — |
| GLM-5.3 full int4/int8 | 7.26 | 7.14 | 8.37 | 9.47 | — |

**The hybrid inverts against FP8 above 2K, and only through the service.** On
the procedure the NVFP4 hybrid prefills faster than the FP8 checkpoint at every
length. Through the service it is faster at 512, equal at 2K, and then loses
badly: 11.4 s against 9.2 s at 8K, and 93.8 s against 58.6 s at 32K. The
service-to-procedure ratio is 1.45–1.74 for FP8 and 1.73–2.00 for the hybrid, so
the NVFP4 expert path degrades with context in a way the FP8 tensor-core path
does not. This is one sample per point and wants a profile before anyone acts
on it; it is the most actionable thing tonight's run turned up.

**Prefill costs far less than decode on the smaller world** (2026-09-12). On
the procedure, halving the ranks costs 1.56x at 512 tokens, 1.44x at 2,048 and
1.45x at 8,192 — against decode's 1.75x. Prefill is tensor-core bound rather
than weight-read bound, so doubling each rank's weights does not double the
work, and two nodes exchange fewer bulk collectives.

Through the service the two worlds nearly converge at 8K: 11,881 ms on two
nodes against 11,374 on four. That is not a two-node win. It is the hybrid's
known service-to-procedure inversion being milder here — the ratio is 1.43 at
world 2 against 2.00 at world 4 — so the four-node service figure is the
outlier, not the two-node one. Whatever degrades the NVFP4 expert path with
context through the service hurts less when each rank owns a wider slice.

GLM-4.7's prefill was previously measured only on 31 to 150-token prompts and
one 6,525-token prompt at 1.77 ms/token. The service figures above are its
first readings at the standard lengths, and at 8K it is the slowest of the
three, consistent with the v1 warp-per-row attention kernel it still uses.

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
| GLM-5.3 NVFP4 hybrid | 2 | 158/164 (96.3 %) | 291/300 (97.0 %) | 100/100 |
| Qwen3.8-Flash-Next-FP8 | 4 | 39/40 | 59/60 | 30/30 |
| Qwen3.8-Flash-Next-FP8 | 2 | 38/40 | 59/60 | 30/30 |
| Qwen NVFP4, BF16 dense | 1 | 39/40 | 59/60 | 30/30 |
| Qwen NVFP4, FP8 dense | 1 | 38/40 | 59/60 | 30/30 |
| GLM-4.7-NVFP4 | 4 | 39/40 | 60/60 | 30/30 |
| GLM-5.3 full int4/int8 | 4 | 40/40 | 59/60 | 30/30 |

The two-node row is the same checkpoint on the same items, with the latent
cache in FP8 rather than BF16, and it lands inside the run-to-run movement of
the four-node row: one more HumanEval problem, two fewer GSM8K, extraction
exact. Halving the world and quantizing the latent cache cost nothing
measurable at task level (2026-09-12).

The two GLM-5.3 checkpoints were run on identical items: 155 HumanEval problems
pass on both, 7 fail on both, and 2 pass only on the hybrid. The single-point
HumanEval differences in the smaller campaigns sit on a near tie that has moved
both ways between runs of the same build, so they are not a ranking.

Two further identities are checked on every world and are pass/fail rather than
scores: **MTP produces the plain greedy transcript token for token**, and
**every rank's op stream is identical**, verified by four matching md5 sums at
shutdown. Both hold on every configuration on this page.

## 8. What is not measured yet

Nine gaps stood here on 2026-09-10 morning. The evening run closed six of them
outright, and what follows is what genuinely remains.

1. **Per-class concurrency for the two worlds not swept.** Every deployment in
   §5 has its class matrix except Qwen at world 2 under sampling beyond the
   figures quoted, and the single Spark's BF16 dense stack, which has only its
   older four-live point. Fill with
   `scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all`.
2. **The prefill method gap wants a profile, not another reading.** §6 shows
   the NVFP4 hybrid losing to FP8 above 2K through the service while winning on
   the procedure. One sample per point. The next step is
   `scripts/fabric_qwen_profile.sh`-style node tracing on a service prefill at
   8K on both checkpoints, not more timings. The 2026-09-12 two-node run adds
   one clue: the same inversion is milder there (1.43x service-to-procedure
   against 2.00x at world 4), so whatever degrades with context eases when each
   rank owns a wider expert slice.
3. **Service-method prefill for the Qwen families.** All three Qwen rows in §6
   are procedure figures; only the GLM worlds have both. Until they are re-run the
   Qwen prefill numbers are not comparable with the GLM service numbers. Fill
   with `scripts/serve_prefill_probe.py HOST PORT 512 2048 8192 --repeat 3`.
4. **Long context beyond 8K for the Qwen families and GLM-4.7.** GLM-5.3-FP8
   now has 16K and 32K and the hybrid has 32K; the others stop at 8K.
5. **Concurrency beyond 4.** The v1 sign-off has an 8-request point for
   GLM-5.3-FP8 and nothing else does. The configs cap `max_concurrency` at 4,
   so this needs a config change, not just a longer run.
6. **The two-node GLM-5.3 world has greedy numbers only, and no endurance
   run.** Its §5 sweep was not repeated at temperature 1, and neither
   `serve_soak_run.sh` nor `serve_failure_drill.sh` has been run against it;
   both now take their rank count from the deployment, so either will run
   against the two-node config unchanged.
7. **The full GLM-5.3's prefill is the deferred kernel, not a reading to
   refine.** 7–9.5 ms per prompt token (§6) is the routed experts and the
   packed attention projections going through the row-chunked int4/int8 GEMV
   chain, every code decoded once per chunk on CUDA cores; GLM-4.7 reads
   1.3–2.0 ms on the same probe. The tile kernel for the packed formats
   (docs/glm53_plan.md D7: decode a weight tile once into shared memory,
   tensor-core MMA over the rows) is the optimization stage's first item.
   The full model also has its first day only: no soak, no failure drill,
   no sampled sweep, one campaign per number.

Closed on 2026-09-10 evening: per-class decode under concurrency for all six
deployments; per-class tables for Qwen at both worlds and for GLM-4.7; the
concurrency-2 point for the single Spark and world 2; the hybrid's c=1/2/4
series and its per-class table on the shipped build; GLM-4.7 prefill at the
standard lengths; and sampled-versus-greedy throughput everywhere, which turned
out to be the same number.

## 9. Reproducing all of it

Everything below runs against a booted world. Bring one up with the launcher
and the config for the deployment you are measuring:

```bash
cp deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json deploy/cluster_qwen-3.8-flash-next_fp8_w4.json   # shared nodes and SSH login come from .env
scripts/dgpp-cluster up --config deploy/cluster_qwen-3.8-flash-next_fp8_w4.json
```

Run one procedure at a time on the fabric. Two measurements at once share the
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
scripts/fabric_mtp_classes.sh --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json OUT_DIR chat code prose json math
```

**That script holds the corpus.** Five fixed prompts, one per class, unchanged
since 2026-09-05: chat asks why a CUDA graph replay beats eager launching, code
asks for a Python SSE parser with pytest tests, prose asks for a long history of
the Roman Republic, json asks for twenty-five records, math asks for a worked
train problem. Every per-class number this project has published comes from
those five texts, so they are the corpus and not an example of one. It runs
greedy, 300 steps per class, on all four ranks, and compares transcripts across
ranks.

It uses `glm_gen_check`, which is a GLM-only evidence app: it refuses the Qwen
checkpoints, whose class numbers therefore come from the service instead (§9.3).
Where both methods have been run on one world they agree within 0.05–0.7
ms/token, so the two are interchangeable in practice; the page says which
produced each row.

`serve_load.py --classes` reuses the same five texts as prompt 0 of each class
and adds three more per class, because a phase at concurrency c needs c distinct
requests and repeating one would share a decode path and attach to the prefix
cache. A `check_corpus()` call re-reads the shell script on every run and aborts
if prompt 0 has drifted from it, so the concurrency-1 column stays comparable to
the single-stream tables.

Through the endpoint instead, and for comparing transcripts across MTP depths:

```bash
scripts/mtp_depth_check.py --out /tmp/depth1 --max-tokens 300
diff -r /tmp/depth1 /tmp/depth2      # must be byte-identical
```

### 9.3 Concurrency

```bash
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --max-tokens 320
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all                  # the §5 matrices
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all --temperature 1  # sampled
```

For each concurrency it opens that many streaming requests at once, greedy and
with thinking off unless `--temperature` says otherwise, and reports per request
the token count, time to first token and client-side pace, and per phase the
wall time, the aggregate decode tokens/s and the phase's stamps so rank 0's
stats lines can be laid beside it. Without `--classes` the prompts are a mixed
set of long-answer technical essays; with it, each phase draws from one class
(§9.2). Add `--isolation 4` to assert that a prompt's greedy answer alone is
byte-identical to its answer beside three others.
`scripts/fabric_glm4_load.sh CONFIG OUT` wraps the mixed form for the GLM-4.7
worlds.

### 9.4 Prefill

Through the endpoint, with the prefix cache defeated by a unique nonce per
prompt and the cost read from `/v1/metrics` deltas:

```bash
scripts/serve_prefill_probe.py HOST PORT 512 2048 8192 --repeat 3
```

On the fabric directly, in steady state, where the timed prefill is the third
repeat and the four ranks' generated ids are compared:

```bash
scripts/fabric_prefill_repeat.sh --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json OUT_DIR 512 2048 8192
```

### 9.5 Quality

```bash
scripts/serve_eval.py HOST PORT --out OUT_DIR --tasks gsm8k,extract
```

Greedy, the served model taken from `GET /v1/models`, HumanEval completions
executed against the problem's own tests in a subprocess. Pass `--limit` to
run a subset, and record the denominator with the score. The data lives in
`DGPP_DATA_DIR` (default `data/`); prepare it with
`python3 scripts/prepare_data.py download`. To include HumanEval, use
`--tasks humaneval,gsm8k,extract --allow-code-execution` only on an isolated
evaluation machine. Generated code runs without a security sandbox.
See [fixture preparation](testing.md).

### 9.6 The determinism checks that make a benchmark meaningful

A throughput number from a world that is not deterministic is not worth
recording. Every campaign above ran with these:

- rank agreement: `scripts/dgpp-cluster down` prints one op-stream md5 per
  rank and the four must be identical;
- speculative decode identity: the MTP transcript must equal the plain greedy
  transcript, checked by `mtp_depth_check.py` and `diff -r`;
- batching isolation: `serve_load.py --isolation C`.

### 9.7 Where the raw records live

`build-ci/fabric-runs/glm53_serve_2026-09-12/` (MTP depth 1: transcripts,
pace, API check, the class and concurrency sweeps, the prefill probe, the
eval's per-item JSONL), `glm53_plain_2026-09-12/` (T=1) and
`glm53_large_2026-09-12/` (the fp8 latent cache) on the head node hold the
full GLM-5.3's first campaign; `benchmarks/results/2026-09-12-glm53-full.md`
is its dated record.
`build-ci/fabric-runs/glm53_w2_2026-09-12/` on the head node holds the two-node
GLM-5.3 campaign: the class sweep, the isolation check, both prefill methods,
the eval's per-item JSONL, the rituals' logs, and the context-ceiling readings.
`build-ci/fabric-runs/bench_2026-09-10/` on the head node holds tonight's run:
every phase's raw output, the server logs, and `RESULTS.md` with the tables as
they came off the fabric. `benchmarks/results/` holds the dated engineering
record behind the earlier numbers. `docs/signoff_v1.md` is the v1 sign-off with
its own method statement, `docs/measurements.md` the platform and GLM-4.7
readings, `docs/qwen38_single_spark.md` the single-Spark campaign, and
`docs/qwen38_optimization_plan.md` and `docs/nvfp4_plan.md` the optimization
rounds with their A/B pairs.
