# Zeta-2.1 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[zed-industries/zeta-2.1](https://huggingface.co/zed-industries/zeta-2.1) —
the dense 8B Llama fine-tune of `ByteDance-Seed/Seed-Coder-8B-Base` for
code edit prediction. The engine loads the release as shipped.

## The checkpoint, as loaded

16.5 GiB across 4 shards (291 tensors, all BF16): 32 layers ×
(input_layernorm, q/k/v/o projections, post_attention_layernorm,
gate/up/down MLP), plus `model.embed_tokens`, `model.norm` and the
untied `lm_head`. The default chain is exact; the checkpoint ships no
quantized form — the engine's own lossy lever is the fp8-at-load
recipe below ([FP8 at load](#fp8-at-load)). The K/V cache is BF16 paged rows: 128 KiB a
token (32 layers × 8 kv heads × 128 × K+V × 2 B).

## Architecture

32 layers at hidden size 4096: GQA attention (32 query heads, 8 kv heads
of 128, rotate-half RoPE over the full 128 dims, θ = 500,000, no bias, no
q/k norm), each followed by a dense SwiGLU MLP of 14,336; RMSNorm
ε = 1e-6; a vocabulary of 155,136 (the tokenizer carries 155,121 base
entries plus 128 added specials in ids 0–127; 15 head rows past the
tokenizer have no string); no draft layer; no sliding window; no vision.
The context is 32,768 tokens.

## Serving

**One Spark** (`cluster_zeta-2.1_bf16_w1`): 15.4 GiB resident, a
32,768-token BF16 KV pool (4 GiB), eight request slots, plain decode on
the graph (the checkpoint ships no draft block — `engine.mtp: false`;
`--mtp` refuses at build), the bf16 decode weights served from their
lossless 12-bit companions beside the bf16 bytes
(`engine.bf16_weights: bf12+bf16`: packed decode at 0.75× the bytes,
bf16 prefill; bitwise-exact, so the transcript gates cover every
residency). The model is a raw FIM completion model: no
chat template ships, so it serves `/v1/completions` with the SPM prompt
passed through verbatim (chat completions refuse). The tokenizer is the
checkpoint's own BPE under the zeta pre-tokenizer pattern
(single-number runs, no punctuation newline tail) with NFC.

Measured on one Spark (2026-10-14, this binary): boot to READY in 22 s
warm; the checkpoint's own `sample.prompt` (3,075 tokens) prefills in
2.1 s cold; greedy decode ~15 tok/s; the 158-token completion is
byte-exact against the checkpoint's `sample.output` (EOS stop), and
repeats deterministically. Throughput sweep at 4 slots, short prompt,
256-token greedy completion: 15.8 tok/s (conc 1), 25.3 agg tok/s (conc 2),
35.9 agg tok/s (conc 4).

With the 12-bit companions (2026-10-15, `bf16_weights: bf12`, bitwise
identical output): 20.2 tok/s single-stream, 132.7 agg tok/s at conc 8
(the batching knee: conc 16 holds 111.8 agg and halves per-request
speed). Resident drops 15.4 → ~12 GiB; pack costs ~5 s at boot; prefill
runs ~1.4× (companion expansion on wide GEMMs, one-time).

Shipped residency is `bf12+bf16` (both forms resident, ~19 GiB): decode
unchanged (19.9 tok/s single, 134.2 agg at conc 8 — the same packed
launches), short prefill back to bf16 speed. True single-connection
TTFT on the shipped shape: 229 ms at 500 tok, 406 ms at 1K, 1003 ms
at 2K, 2196 ms at 3K (steady state; first-ever shapes pay ~1 s of
planner autotune once per process). Autocomplete guidance: keep prompts
≤1K tokens client-side (the dominant lever), prefix cache on (repeat
edits skip cached blocks), stream with a small max_tokens cap.

## FP8 at load

`engine.dense_weights: "fp8"` encodes every dense projection and the lm
head to block FP8 at boot — E4M3 codes, one fp32 scale per 128 x 128
block (the block's amax fixes its scale; no calibration, the recipe of
the FP8 releases) — and every projection and the head run the scale
GEMM on bf16 activations. Roughly half the projection and head bytes
go resident (the embeddings, the norms and the exact BF16 K/V are
untouched). fp8 excludes `bf16_weights`: no bf12 companions are
granted and nothing is packed in this mode. Output is lossy —
tolerance-level agreement with the bf16 chain (near-tie argmax rows
may flip), not the bitwise transcripts the exact residencies pin. The
throughput row waits on a measured campaign; the decode-bound
single-stream recipe is where the lever should pay (~1.5x the packed
bf12 launches at the same batch, the projection+head reads halved).
