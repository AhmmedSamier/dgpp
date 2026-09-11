---
license: mit
language:
- en
- zh
pipeline_tag: text-generation
base_model:
- dabsLabs/GLM-5.3-Flash-NVFP4
- unsloth/GLM-5.3-Flash-FP8
tags:
- glm5_next
- nvfp4
- fp8
- mixture-of-experts
- dgpp
---

# GLM-5.3-Flash: NVFP4 / FP8 hybrid

This checkpoint combines the main-stack routed experts from
[dabsLabs/GLM-5.3-Flash-NVFP4](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4)
with the remaining tensors from
[unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8).
It was assembled for DGPP, a C++/CUDA inference engine for DGX Spark.
The full checkpoint is included here, along with its tokenizer, configuration
files, chat template, and composition manifest.

This is a repackaging of existing quantized weights, not a new training run or
quantization pass. Each output tensor was copied from one of the two releases
without arithmetic. Tensors taken from the FP8 release retain their original
dtypes; they are not all FP8.

## Sources and attribution

[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) created GLM-5.3-Flash.
Credit for the NVFP4 expert quantization belongs to dabsLabs, and credit for
the FP8 release used here belongs to Unsloth and the upstream model authors.
HawkBearPig composed and published this hybrid checkpoint.

| Contents | Source model card | Pinned source revision |
|---|---|---|
| Routed experts in main-stack MoE layers 3–44 | [dabsLabs/GLM-5.3-Flash-NVFP4](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4) | [`bb176861dceb7ec07ac989120a36aa993355122f`](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4/tree/bb176861dceb7ec07ac989120a36aa993355122f) |
| All other tensors, plus tokenizer, chat template, generation and processor configurations | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | [`a160e2291674d9e3e92e98fd82faa2544a2867a3`](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8/tree/a160e2291674d9e3e92e98fd82faa2544a2867a3) |

The second row includes attention, shared experts, dense MLPs, routers, mHC,
normalization, embeddings, the output head, vision tensors, and the complete
MTP draft layer (layer 45). In particular, neither DSA attention nor the MTP
layer comes from the dabsLabs checkpoint.

## Weight format

The configuration identifies this layout as `quant_method: "dgpp_mixed"`.
Main-stack routed experts use three tensors per projection:

- `weight_packed`: two E2M1 values per byte, with the even element in the low nibble.
- `weight_scale`: one E4M3 scale per group of 16 weights along the input dimension.
- `weight_global_scale`: one FP32 scale per projection tensor.

Their dequantization rule is
`E2M1(code) * (float32(weight_scale) / weight_global_scale)`.
DGPP uses BF16 activations for these weight-only NVFP4 projections (NVFP4A16).
FP8 weights elsewhere retain the source release's `weight` and
`weight_scale_inv` tensors, with 128 × 128 block scales.

## Running the checkpoint

The recorded serving configuration is world size 4: four DGX Spark (GB10)
nodes, one rank per node, using tensor parallelism over RoCE. Other world
sizes have not been validated for this checkpoint.

Download it into the Hugging Face cache on each node:

```bash
hf download HawkBearPig/GLM-5.3-Flash-NVFP4-FP8
```

In DGPP, use `deploy/cluster.nvfp4.example.json` as the deployment template
and set its `model` field to `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`.
The original local composition used the cache name
`dgpp/GLM-5.3-Flash-NVFP4-FP8`; that name remains in the provenance manifest.

This checkpoint requires a loader that understands `dgpp_mixed`.
Compatibility with a source checkpoint does not establish compatibility
with this hybrid; it is not a drop-in Transformers or vLLM checkpoint.
DGPP currently serves text only. Vision tensors are included, but their
presence does not imply that DGPP supports image input.

## Composition and verification

The checkpoint contains 112,396 tensors in 40 Safetensors shards, with
195,075,058,296 bytes of tensor data (about 195.1 GB, excluding headers).
DGPP's `tools/compose_nvfp4_hybrid.py` produced it using the revisions above,
with `--mtp-from base`, `--dsa-from base`, and a 5,000,000,000-byte shard target.

At composition time, every output tensor was reread and compared byte for
byte with its source. `MANIFEST.json` records the source revisions, tensor
census, composition options, and each shard's SHA-256 digest. Its `revision`
value, `833e7cc0e86d9eb125551660dbf48cab2e224df9`, identifies the composition
plan; it is not a Git revision of this Hugging Face repository. The local
build hostname has been omitted from the published manifest.

These checks establish that composition preserved the source tensors.
They do not establish equal accuracy to the original model or either source
checkpoint. This release does not report a separate perplexity or downstream
benchmark evaluation of the hybrid.

## License

The checkpoint is distributed under the MIT license, consistent with both
source releases. The included `LICENSE` preserves Z.ai's copyright and
permission notice. Please retain the source attribution when redistributing
this composition.
