# MiMo-V2.6-Flash on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL).
It is **not** a republished checkpoint: the engine loads the upstream
release as shipped, with no re-quantization pass.

## The checkpoint, as loaded

161 GiB of indexed text weights across 64 expert shards and
`model_mtp.safetensors`. The routed experts are MXFP4 (e2m1 nibbles with
an e8m0 scale per 32 along K); the fused qkv projection, the dense MLP of
layer 0 and the draft's MLP are FP8 e4m3 with fp32 scales on 128×128
blocks; every layer's `o_proj`, the router, the embedding, the LM head and
the draft's `eh_proj` are BF16 (the bf16 decode weights stream as their
lossless 12-bit companions under `engine.bf16_weights = "bf12"`). The
engine decodes every code exactly into BF16 and runs activations in BF16;
the K/V cache is BF16. The vision encoder, the audio encoder and the
separate DFlash draft that ship in the checkpoint are not loaded.

## Architecture

48 layers interleaving sliding-window attention (39 layers, window 128,
a learned per-head attention sink) with global attention (9 layers) —
GQA with 64 query heads of qk width 192 / v width 128, 4 or 8 kv heads,
partial RoPE on 64 dims, a 0.707 value scale. Layer 0 carries a dense
SwiGLU MLP; every other layer a 256-expert sigmoid-routed MoE (top 8, no
shared expert). One MTP draft layer (a sliding-window layer with a dense
MLP behind `eh_proj`) is the speculative block; the checkpoint's second
and third draft layers are not used.

## Serving

Four DGX Spark nodes hold the weights at about 44 GiB per rank (tensor
parallel, world 4: the pre-sharded projection chunks, the experts sliced
on their intermediate dimension); a 128K-token pool costs 7.3 GiB per
rank. Two nodes (world 2) fit at about 78 GiB of weights per rank; the
two-node template keeps the K/V cache in the fp8 row form (e4m3 codes with
one scale per head row, 58 KiB per token per rank) for a 256K-token pool at
97 GiB. Decode runs a CUDA-graph step with the MTP draft (`mtp_depth` 1):
per layer the fused qkv projection, one attention launch (finish, split
partials, combine), the o_proj, and the routed experts, with the residual
add fused into each norm. Prefill attention is query-tiled on the tensor
cores: a block per 64 query vectors stages each K/V tile once (the fp8
cache dequantized once) and runs the scores and PV on `mma.sync` bf16. The tokenizer is the
Qwen2 byte-level BPE; the prompt renderer is the checkpoint's
`chat_template.jinja`; tool calls are emitted and constrained in the
`<tool_call><function=…><parameter=…>` format the template writes; the
model opens its own `<think>` block (`reasoning_effort: none` renders the
closed `<think></think>` prefix).

## Measured

See `docs/benchmarks.md` for the current numbers and
`benchmarks/results/2026-09-22-mimo-v26-flash/` for the evidence.

## License and attribution

The model, its weights, tokenizer and chat template are Xiaomi's (MIT
license per the upstream card); see the upstream model card for its terms.
This file describes only the engine's support and adds no weights.
