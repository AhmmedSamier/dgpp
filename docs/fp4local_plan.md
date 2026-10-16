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

## The CSF seam

Backbone experts ship NO plain `weight_scale`. Each projection carries:

- `weight_scale.nvfp4_csf_fixed` U8 `[R/16, 1296]` (row-groups of 16 rows;
  per row 80 nibble-bytes + 1 extra byte — nibble codec unconfirmed),
- `weight_scale.nvfp4_csf_exceptions` U32, one record per outlier scale:
  `(8-bit E4M3 value << 24) | 24-bit flat scale index`, ~0.3–1.8% rate,
- plus the usual `weight` U8, `weight_scale_2` F32, `input_scale` F32.

Neither public ModelOpt (latest 0.47.0) nor vLLM nor transformers document
the nibble codec; the export was built with unreleased tooling. Empirical
structure so far: exception indices are flat `row*160+col` (verified in
range); the stored nibbles correlate weakly (~2x chance) with the
exception values' high nibble, so the byte layout is still transposed or
biased.

Config flag `experts_nvfp4_csf` (from `weight_scale_encoding: "csf"` on
every backbone experts entry; mixed/unknown encodings refuse) drives the
binding: no plain-scale expectation, sidecars counted as `csf_sidecars`.
The loader refuses at `load_fp4_*_mo` until `csf_scales_expand()` lands.

## Remaining

1. Crack the nibble codec (needs the spec or a validated hypothesis —
   end-to-end output sanity once weights land is the backstop, not proof).
2. `csf_scales_expand()`: host-side, once per matrix at load; feed the
   existing NVFP4 kernels plain scales. Shapes/dtypes already match.
3. Full bind check + loader smoke + HumanEval/GSM8K gates + tok/s numbers.
