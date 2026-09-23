# MiMo-V2.6-Flash decode optimization A/B (2026-09-22)

The fabric A/B of the decode-optimization round (docs/mimo_v26_flash_plan.md
§7.1): four nodes, `deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w4.json`
(MTP depth 1, 128K BF16 pool, four slots), one world per leg from a given
binary, `scripts/timed_load.py --concurrency 1,2,4 --classes all --repeat 3
--max-tokens 256` (engine ms/step = `step_ms / decode_steps`, median over
the repeats per class, mean over the five classes), then the plain step
(`--knobs=--no-mtp`, C1). `run_ab.sh` runs the legs, `summarize_ab.py`
renders `round1/`.

| leg | binary | knobs | mtp C1 | mtp C2 | mtp C4 | plain C1 | C1 transcripts vs base1 |
|---|---|---|---|---|---|---|---|
| base1 | campaign release `1f75f564` | — | 23.39 | 32.82 | 50.38 | 17.84 | — |
| cand2c | the round's tree `f59d421a` | `DGPP_MIMO_PREFILL_ATTN=chain` | **23.04** | 32.68 | 49.82 | 17.85 | 5/5 mtp, 5/5 plain |
| cand2p16 | `f59d421a` | `DGPP_MIMO_L2_PERSIST_MB=16` | 24.97 | 34.46 | 51.51 | 19.54 | 0/5 (tiled prefill) |
| base2 | `1f75f564` | — | 23.39 | 32.88 | 50.20 | 18.29 | 5/5, 5/5 |
| cand2p8 | `f59d421a` | `DGPP_MIMO_L2_PERSIST_MB=8` | 24.93 | 34.36 | 50.92 | 19.59 | 0/5 |

ms/step; the A/A spread (base1 vs base2) is 0.0 % at mtp C1, 0.2 % at C2,
0.4 % at C4 and 2.5 % at plain C1.

## Reading

* **The fusions (cand2c)** — `add_rmsnorm` at every block boundary, the
  one-launch decode attention, the MXFP4 slot kernels' early load issue,
  the score loop on 16-byte tile rows — take 1.5 % off the MTP step at C1
  (23.39 → 23.04), 0.4–1 % at C2/C4, and nothing off the plain T=1 step
  (17.85 vs 17.84 / 18.29). The kernels themselves are 8–12 µs shorter per
  layer (the in-situ profile: rmsnorm 6.0 + add 1.5 → 4.9–6.3 fused; the
  attention chain 16 → 10; four graph nodes fewer), but most of that time
  sat under DRAM-bound phases: the norm after fold 2 runs while the next
  qkv prefetch streams, the attention chain while the o_proj prefetch
  streams — the following GEMV waits for the bytes either way. What the
  step keeps is the part that did not overlap a prefetch. Transcripts
  identical to the baseline in every class (the chain prefill kept for
  this leg: decode changes are bitwise).
* **The early persisting prefetch (cand2p16 / cand2p8)** — the next
  layer's qkv into an L2 set-aside from fold 1 on — loses 7–10 % at both
  sizes. The profile (`raw/profile_t1_persist`): the 16 MB window at the
  Light rate needs 185 µs, so it runs through the whole expert block and
  competes with it there (gate/up 93.5 → 111 µs, down 41.5 → 45), the
  o_proj window loses its L2 (o_proj GEMV 36.9 → 55.9), fold 2 slows
  (27 → 44), and the qkv GEMV gains only 14 µs (60.5 → 46.5). Same
  numbers at 8 MB, so the normal region's size is not the cause. The
  attribute itself is free (a 96-node graph replays in the same time
  with or without it: scratch probe). Rejected; the plumbing stays as an
  experiment knob (`DGPP_MIMO_L2_PERSIST_MB`), no engine key. A redesign
  would size the window to fold 1's idle time alone and stop before the
  experts start.
* **The tiled prefill attention** changes prefill rows at the fp32
  summation-order level (the tensor core's), so its transcripts diverge
  from the baseline's (0/5 identical, as expected); its quality gates are
  the campaign's task evals.

## Files

`round1/<leg>/{mtp,plain}.{json,log}`, `up-*.txt`, `down-*.txt`,
`binary.sha256`, `env.txt`, the server logs under `server-*/`.

## Regression legs on the other families (the shared kernel changes)

The fp4 core's issue/consume split and the MXFP4 chunk decode's f16 fast
path are shared code; the final binary (`4638e3c5`, the re-run campaign's)
against the morning's campaign binary, same recipe (`AB_MODES=mtp`), C1
transcripts identical in every class:

| deployment | leg | mtp C1 | mtp C2 | mtp C4 |
|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8, four nodes (`glm/`) | base | 31.80 | 43.50 | 64.55 |
| | final | 31.59 | 43.18 | 63.61 |
| DeepSeek-V4.1-Flash MXFP4/FP8, four nodes (`deepseek/`) | base | 52.28 | 72.46 | 100.42 |
| | final | 51.67 | 71.25 | 98.23 |

ms/step. No regression; DeepSeek (the other MXFP4 family) takes 1–2 % from
the same changes.
