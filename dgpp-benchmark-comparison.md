# DGPP vs DGX Spark forum deployments — informal comparison baseline

Compiled 2026-09-14 from HawkBearPig/dgpp `docs/benchmarks.md`, `docs/measurements.md`, `CHANGELOG.md` (master) and NVIDIA DGX Spark developer forum threads. All DGPP numbers are the repo's published, dated figures (3-run medians, fixed 5-class prompts, determinism-gated). Forum numbers are as posted by their authors.

## 1. Decode, single stream (tok/s)

| Model / world | DGPP (published) | Forum comparable (topic) | Read |
|---|---|---|---|
| GLM-5.3-Flash w4 FP8 | T=1 31.8–33.4 · MTP greedy 42–47 | 28–43 sustained, ~37 avg, 50+ spikes — native FP8, 1M ctx (381543) | DGPP median at/above forum band |
| GLM-5.3-Flash w4 NVFP4 hybrid | T=1 38.1 · MTP ~50 (19.8 ms/tok) | 35.2 (32–39), TP=3 NVFP4 (381534) | DGPP ahead; 3-node vs 4-node caveat |
| GLM-5.3-Flash w2 | T=1 21.9 · MTP 29–34 | MTP-5 19.6–30.3 (381433) · DFlash2 21.8 (381429) · EXL3 27–32 (382486) · EXL3+DFlash2 33–74 c1 (Entrpi, via 382486) | Overlapping bands; forum extreme-quant lanes win on peaks |
| GLM-5.3-Flash w1 | not offered | 64 structured / 25 prose — EXL3 2.05bpw + DFlash2 K7 (382140) | No DGPP lane |
| Qwen3.8-Flash-Next w4 | T=1 46.5 · MTP d1 77.5 greedy (12.9 ms/tok) | 31.0 MTP-2 NVFP4 (381897) · 40.5 med / 54.2 peak TP4 NVFP4 (382476) | DGPP ~1.4–2.5×; DGPP FP8 vs forum NVFP4 |
| Qwen3.8-Flash-Next w2 | T=1 30.8 · MTP 38–48 | NVFP4 SPEED 53.7 med / 63.7 peak (382476) · SGLang FP8 36–41 (382435) | DGPP mid-band; forum NVFP4 + table-in-memory lane ahead |
| Qwen3.8-Flash-Next w1 | T=1 32.3 · MTP 39–49 (FP8 dense) | 32.5 med / 43.8 peak (382476) · 43 coding (381859) · MiaAI 37 (382446) · INT4-AutoRound 53.8–71, strict-c1 51 vs 36 (382733) | DGPP ties the NVFP4 lane; INT4-AR lane beats it |
| GLM-5.3 full 754B w4 int4/int8 | T=1 19.5 · MTP d1 36–42 | c1 12.1 @ 200K ctx, TP4 vLLM (381755) | DGPP ~3× single-stream |
| DeepSeek-V4.1-Flash w4 MXFP4 | scheduled DSpark per class 40–53 (chat 40.7 / prose 44.1 / code 52.9 / json 52.4 / math 50.5) | 77.2 peak c1 (counting) · 52 code · 47 math · 39 reasoning · 23 prose; 72 warm code (382897) | Comparable per class except prose; forum headline is a peak |
| GLM-4.7 w4 NVFP4 | T=1 20.4 · MTP d1 31–33 | SGLang NEXTN MTP 24.4 c1 (366325) · TP2 17.5 @ 64K (375690) | DGPP ahead |
| DeepSeek-V4.1-Flash 8× Spark | not offered | 92.7 coding single / 256 @ 6 streams (382725) | Forum only |

## 2. Aggregate decode under concurrency (tok/s)

| Model / world | DGPP (published) | Forum comparable (topic) | Read |
|---|---|---|---|
| GLM-5.3-Flash w4 hybrid | 83–91 @ c=4 | ~40–50 @ c=4–5 (381543) | DGPP ~2× at equal concurrency |
| GLM-5.3-Flash w2 | 39–47 @ c=4 | 67.6 @ 12-way (382120) · 74–77 @ c=6 (382486) | Forum wins at c≥6; DGPP caps at 4 slots |
| Qwen w4 | 124–147 @ c=4 | 46 @ x2 · 53 @ x4 · 97 @ x8 · 157 @ x16 (381897) · 262 @ 6 TP4 (382476) | DGPP wins at c=4 (2.4× vs x4); forum wins at c≥6 |
| Qwen w2 | 69–83 @ c=4 | 88–98.5 @ 4 streams, SGLang FP8 (382435) · 309 @ 6 SPEED (382476) | SGLang c=4 lane ahead of DGPP |
| GLM-5.3 full w4 | 38–43 @ c=4 | 33.1 @ c4 · 46.0 @ c6 (381755) | DGPP wins at c=4 |
| DeepSeek-V4.1 w4 | 82.7 @ C6 (106.4 with group prefill) | 214 @ 6 (382897) · 131.9 @ 6 (tonyd2wild repo bench) | Forum ~1.3–2× at c=6 |
| GLM-4.7 w4 | 46.5–53.5 @ c=4 | 81.1 @ c=8, SGLang (366325) | Forum at c=8; DGPP caps at c=4 |

## 3. Prefill (tok/s)

| DGPP (published) | Forum comparable (topic) |
|---|---|
| ~900–1,450 procedure method (FP8 971–1,442; Qwen w4 1,150–1,720; DSV41 1,235–1,383 after Sept-14 tile kernel); service method 1.4–2.8× slower; hybrid inverts above 2K ctx (93.8 s @ 32K vs FP8 58.6 s) | Qwen TP2 2,784 @ 28K · TP4 2,450 (382476) · GLM TP=3 1,800 (381534) · EXL3 kit 1,408 @ 240K (Reederey87 kit, via flowtivity Sep 13) · llama.cpp 4-node DSv4-0731 2,582–3,031 (381005) |

Read: forum recipes hold a ~1.5–2.5× prefill lead; DGPP's own docs flag this as the top open item (no decode/prefill interleaving — a 32K prompt still stalls the world).

## 4. Context / capacity

| DGPP (published) | Forum comparable |
|---|---|
| GLM-5.3-Flash hybrid 786K validated (fp8 latent cache, w4) · full GLM-5.3 208K fp8 · DSV41 128K | 1M ctx served on multiple lanes (381543, 382897 300K/1M proven, 382446 MiaAI 1M) · KV pools to 4.7M tokens (382486) |

## 5. Quality / determinism

DGPP publishes task-level quality with every throughput table: HumanEval 94.5–96.3%, GSM8K 97–100% of denominators, schema extraction 100/100, plus MTP==greedy transcript identity and cross-rank op-stream md5 identity as pass/fail gates. The forum's only task-quality datapoint is tsarihan's SWE-bench Pro enterprise-40 (382697): Qwen 36–38, GLM 36, DSv4 33 (needs thinking=true + temp 1.0, else 22/40). Suites do not overlap — not comparable; treat DGPP's quality gate as an extra control the forum numbers don't have.

## Caveats — where this is not like-for-like

1. **Method.** DGPP: 3-run medians, fixed 5-class prompt set, client-side aggregates, determinism-gated. Forum: single runs, self-chosen prompts, and headline numbers are often peaks. Peak-vs-median on the same stack spans ~1.5–2× (e.g. 382476's own median-vs-peak rows).
2. **Prompt class.** Forum "prose" vs "structured/code" spans ~3× on one build (DSV41: 23 prose vs 52 code / 77 counting-peak, 382897). Compare per class only; DGPP publishes all five classes.
3. **Quantization.** DGPP serves FP8 / NVFP4-FP8 hybrid / MXFP4-FP8; forum lanes include EXL3 2.05–4bpw, INT4-AutoRound, NVFP4. Lower bits + deeper drafts trade quality for tok/s; acceptance and quality are mostly unpublished outside tsarihan's SWE run.
4. **Speculative decoding.** DGPP uses the checkpoint's own MTP at depth 1–2 with transactional verify; forum lanes use DFlash2 K7, MTP-5, DSpark d4–5. Deeper drafts inflate structured-prompt numbers.
5. **Concurrency caps.** DGPP caps at 4 slots / 8 rows (DSV41 six-slot, full-GLM eight-slot are the exceptions); forum runs go to 6–16 streams. Aggregate comparisons beyond c=4 measure occupancy policy, not engine speed.
6. **Serving path.** DGPP's service-method prefill differs from its engine-side procedure figure by 1.4–2.8×; all forum numbers are end-to-end through a serving stack. Use DGPP's service figures against forum numbers.
7. **Run health.** GB10 clock-gating is real: tonyd615's DSV41 went 5 → 41 → 77 tok/s after un-latching two GPUs stuck below 1 GHz (power-cycle fix, 382897). Some forum bests are post-fix numbers.
8. **Context length.** Decode step time grows with context (DGPP Flash: 41 ms @ 2K → 42–43 ms @ 32K). Forum numbers are often taken at 200K–1M ctx, DGPP's class tables at short ctx.
9. **No third-party DGPP measurements exist.** A Discourse search for "dgpp" returns zero hits (2026-09-14): every forum number is vLLM/SGLang/llama.cpp. The baseline is informal by construction — same hardware class and models, different engines and measurement conventions.

## Sources

- Repo: github.com/HawkBearPig/dgpp — docs/benchmarks.md, docs/measurements.md, CHANGELOG.md (master, pulled 2026-09-14)
- Forum topics: 381543, 381534, 381433, 381429, 382486, 382140, 381897, 382476, 382435, 382446, 381859, 382733, 381755, 382897, 366325, 375690, 382725, 381005, 382697, 381228
- Blogs: flowtivity.ai DeepSWE benchmark (Aug 31) and dual-Spark GLM vs DSV4.1 (Sep 13, incl. Reederey87 EXL3 kit numbers)
- Raw sweep data: per-thread tok/s matches in /tmp/forum_mining.json
