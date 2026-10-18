# FP4-Local port (nvidia/Qwen3.8-Flash-Next-FP4-Local)

Same 48-layer GDN/full-attention MoE as RadixArk, ModelOpt
`MIXED_PRECISION` under the newer `quantized_layers` (per-module) schema.

## Recipe (verified against the shipped headers)

| component | form | dgpp status |
|---|---|---|
| routed experts, all layers | NVFP4 g16, scales CSF-wrapped | done: config + binding + plan-safe load, smoke-validated 2026-10-17 |
| shared experts + all attention | MXFP8 g32 (`weight` F8 + `weight_scale` U8 `[N,K/32]`) | readable by the QAD path |
| n-gram table | NVFP4 (`shard_s.weight` U8 + scales) | readable by the QAD path |
| MTP draft experts | W4A16_NVFP4, plain scales, no input_scale | readable by the QAD path |
| lm_head / norms / router / embeds | BF16, hc/mixer/PLE norms F32 | as the QAD export |
| vision tower | MXFP8 + W4A16_NVFP4 | widened to BF16 at load; counted at binding, runs on image prefills only |
 *Vision note (RESOLVED 2026-10-16): the tower loader now widens
 quantized linears to BF16 at load — MXFP8 e4m3 x 2^(s-127) exact,
 NVFP4 e2m1 x e4m3-group x global with one RNE (the fp8 bridge's
 contract); BF16 biases ship beside them. Dtype-driven per linear, so
 BF16 towers take the old path untouched; scale sidecars are exempt from
 the no-unexpected-tensors check. Verified: all 333 walked tensors
 present with exact dtype+shape in the real headers, zero non-scale
 leftovers, all 166 scales present. Multimodal serves when weights land;
 text path unaffected (tower runs on image prefills only, never decode).*

The template sets `engine.dense_weights: "fp8"` and `engine.fp8_head: "mma"`,
as every sibling NVFP4/QAD one-Spark template does: the BF16 remainder — the
GR sites, the PLE projections, `lm_head` and the draft layer — is encoded to
block FP8 at load, and the vocabulary head runs on the tensor-core path. The
MXFP8 attention/shared-expert stack is served as shipped either way
(`loader.cpp`: the MXFP8 stack takes no engine key). The BF16 dense stack is
the documented knob (`--dense-weights checkpoint --fp8-head gemv`).

The template also takes the fp8 K/V pool (`engine.kv_dtype: "fp8"` with
`engine.prefill_attention: "fp8"`): the Full layers' 262144-token cache
halves from 6.70 to 3.50 GiB, and an interleaved A/B against the bf16 cache
on one Spark measured 1459–1469 t/s prefill at 4–8K depth against 1141–1260,
decode unchanged. The one-Spark pool is sized to four full contexts
(`kv_capacity: 1048576` = 14.02 GiB; the plan totals 97.14 GiB and the boot
leaves 1.92 GiB device free — this node's ceiling). `engine.rope_scaling`
(YaRN, factor 4 over the checkpoint's 262144) lifts the positional ceiling to
1048576 tokens at no pool cost — the cap moves, not the pool — so the
four-context pool keeps seating four 262144-token requests while a single 1M
request would take it whole. The loader is the `qwen`
family, whose image key folds its own `loader_format` (not the `qwen35`
family's bump to 6), so the resident image's key and layout are unchanged and
adopting the pool needs no re-expansion. Quality is not yet measured on this
export (2026-10-18).

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
The draft carries its own form: this export's `mtp.layers.{0,48}`
entries are `W4A16_NVFP4` with **no** encoding, so `mtp_experts_nvfp4_csf`
stays false and the loader reads the draft's plain E4M3 `weight_scale`
(the flag exists so a CSF draft is representable; mixed draft entries
refuse). The loader picks the form per scope in `build_moe`.

## Loader validation (2026-10-17)

`qwen_load_check --checkpoint-dir <snap> --mtp` (world 1) loads the whole
export, draft included:

- Shard discovery is index-first: `model.safetensors.index.json`'s
  `weight_map` names every shard under `tensors/`; the resident stream
  joins each to the checkpoint dir and refuses loudly on a missing one
  (the old root-dir scan found none).
- Plan mode (before shards open) accounts the CSF scale plane
  analytically — `(N/16) * (16 + full_cols/2)` bytes per matrix, the
  expander's own contract; the exceptions stream is data-dependent and
  read (not planned) in copy mode. The plan-vs-read invariant is checked
  by the loader: resident 98.20 GiB planned == 98.20 GiB loaded (globals
  2.42 + table 26.82 + 49 layers 68.96), source bytes read 94.73 GiB
  (globals verbatim 3.83), 98.7 s on the Spark.
- Digests on the validation machine: globals `614f1391629385f3` (149,617
  tensors), layer 0 `fc07770185d686ce`, layer 1 `23853c2d450d862a`,
  layer 48 (draft) `29c3848153c6e1e6`.

## Remaining

1. Quality gates: the card reports GPQA Diamond 91.5% (BF16 92.2) and
   ToolEvalBench 88.7% (BF16 89.4) — use these, not HumanEval/GSM8K.
2. Serve wiring + tok/s numbers (card validates RTX PRO 6000 / DGX Spark
   with PLE offload, FP8 KV; expects ~96 GiB on disk).
3. Note: the 100 F32 norms (97 hc + 3 PLE) are converted to BF16 in memory
   by the vLLM loader; dgpp's norms_f32 path already covers the QAD
   export's identical set — no action unless the bind check says otherwise.
