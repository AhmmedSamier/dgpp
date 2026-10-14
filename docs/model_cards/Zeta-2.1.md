# Zeta-2.1 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[zed-industries/zeta-2.1](https://huggingface.co/zed-industries/zeta-2.1) —
the dense 8B Llama fine-tune of `ByteDance-Seed/Seed-Coder-8B-Base` for
code edit prediction. The engine loads the release as shipped.

## The checkpoint, as loaded

16.5 GiB across 4 shards (291 tensors, all BF16): 32 layers ×
(input_layernorm, q/k/v/o projections, post_attention_layernorm,
gate/up/down MLP), plus `model.embed_tokens`, `model.norm` and the
untied `lm_head`. The default chain is exact; there are no quantized
paths and no lossy levers. The K/V cache is BF16 paged rows: 128 KiB a
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
32,768-token BF16 KV pool (4 GiB), four request slots, plain decode on
the graph (the checkpoint ships no draft block — `engine.mtp: false`;
`--mtp` refuses at build). The model is a raw FIM completion model: no
chat template ships, so it serves `/v1/completions` with the SPM prompt
passed through verbatim (chat completions refuse). The tokenizer is the
checkpoint's own BPE under the zeta pre-tokenizer pattern
(single-number runs, no punctuation newline tail) with NFC.

Measured on one Spark (2026-10-14, this binary): boot to READY in 22 s
warm; the checkpoint's own `sample.prompt` (3,075 tokens) prefills in
2.1 s cold; greedy decode ~15 tok/s; the 158-token completion is
byte-exact against the checkpoint's `sample.output` (EOS stop), and
repeats deterministically.
