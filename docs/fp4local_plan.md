# FP4-Local port (nvidia/Qwen3.8-Flash-Next-FP4-Local)

Same 48-layer GDN/full-attention MoE as RadixArk, ModelOpt
`MIXED_PRECISION` under the newer `quantized_layers` (per-module) schema.

## Recipe (verified against the shipped headers)

| component | form | dgpp status |
|---|---|---|
| routed experts, all layers | NVFP4 g16, scales CSF-wrapped | config + binding done, **expansion open** |
| shared experts + all attention | MXFP8 g32 (`weight` F8 + `weight_scale` U8 `[N,K/32]`) | readable by the QAD path |
| n-gram table | NVFP4 (`shard_s.weight` U8 + scales) | readable by the QAD path |
| MTP draft experts | W4A16_NVFP4, plain scales, no input_scale | readable by the QAD path |
| lm_head / norms / router / embeds | BF16, hc/mixer/PLE norms F32 | as the QAD export |
| vision tower | MXFP8 + W4A16_NVFP4 | never served (counted at binding) |
 *Vision note (RESOLVED 2026-10-16): the tower loader now widens
 quantized linears to BF16 at load — MXFP8 e4m3 x 2^(s-127) exact,
 NVFP4 e2m1 x e4m3-group x global with one RNE (the fp8 bridge's
 contract); BF16 biases ship beside them. Dtype-driven per linear, so
 BF16 towers take the old path untouched; scale sidecars are exempt from
 the no-unexpected-tensors check. Verified: all 333 walked tensors
 present with exact dtype+shape in the real headers, zero non-scale
 leftovers, all 166 scales present. Multimodal serves when weights land;
 text path unaffected (tower runs on image prefills only, never decode).*

## The CSF seam (RESOLVED 2026-10-16)

Codec `byte-window4-fixed-stream-u24-exceptions/1`, documented on the
sibling cards (local-inference-lab/GLM-5.3-NVFP4-CSF,
Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD) and implemented in the vLLM reader
`vllm/model_executor/model_loader/csf_scales.py` (patch 14-pr60646 in the
eugr-spark-vllm-docker cache on this box; Apache-2.0; reimplemented here,
not copied). Lossless: the decoded bytes equal the source E4M3 scales
exactly — every NVFP4 kernel consumes them unchanged.

Per 16-row slab: 16 base bytes (one per row, ≤ 240), then the rows' 4-bit
codes low-nibble-first; scale byte = base + code; each exception word's
high 8 bits replace the byte at its low-24-bit flat position.
`qwen_csf_scales_expand()` (loader.cpp, unit-tested incl. refusals)
expands once per matrix at load; the Builder slices rows/cols from the
plane. Verified against real checkpoint bytes: gate [640,160] decodes to
positive E4M3, no NaN, overlay exact.

Config flag `experts_nvfp4_csf` (from `weight_scale_encoding: "csf"` on
every backbone experts entry; mixed/unknown encodings refuse) drives the
binding: no plain-scale expectation, sidecars counted as `csf_sidecars`.

## Remaining

1. Full bind check + loader smoke once the weights land (binding already
   reconciles all 375,487 header tensors; the loader path is new code).
2. Quality gates: the card reports GPQA Diamond 91.5% (BF16 92.2) and
   ToolEvalBench 88.7% (BF16 89.4) — use these, not HumanEval/GSM8K.
3. Serve wiring + tok/s numbers (card validates RTX PRO 6000 / DGX Spark
   with PLE offload, FP8 KV; expects ~96 GiB on disk).
4. Note: the 100 F32 norms (97 hc + 3 PLE) are converted to BF16 in memory
   by the vLLM loader; dgpp's norms_f32 path already covers the QAD
   export's identical set — no action unless the bind check says otherwise.
