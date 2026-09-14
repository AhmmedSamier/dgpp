# Deployment filenames

Templates use `cluster_<model>_<quant>_w<n>_<mode>[_variant].example.json`.
Local copies use the same name without `.example` and remain Git-ignored.

- Model names are `glm-5.3-flash`, `glm-5.3` (the full model), `glm-4.7`,
  `qwen-3.8-flash-next` and `deepseek-v4.1-flash`.
- The quant names the checkpoint representation: `fp8`, `nvfp4`,
  `nvfp4-fp8` for the custom GLM-5.3-Flash hybrid, or `int4-int8` for the
  full GLM-5.3's pack-quantized release (int4 group-64 routed experts, int8
  attention and shared experts), or `mxfp4-fp8` for DeepSeek-V4.1-Flash
  as it ships (MXFP4 experts, FP8 dense and attention, FP8 Engram tables
  mapped from the NVMe). The JSON's `model` field gives the exact
  Hugging Face repository.
- `w<n>` gives the participating node count. Nothing restricts that number to
  the counts in use today; a world is refused by the engine's geometry check or
  a rank's memory plan, not by a list of allowed sizes.
- `plain` disables MTP. `mtp1` and `mtp2` enable it at draft depths 1 and 2.
- `dense-fp8` converts Qwen's dense projections to FP8 at load time. It does
  not change the checkpoint quant or Hugging Face repository.
- `large-cache` selects GLM-5.3's larger KV and prefix-cache budgets. Both
  GLM-5.3 templates use the hybrid checkpoint; neither is a separate FP8 release.
- `engine.embed_sharding` (`"replicated"` by default, `"vocab"` in the full
  GLM-5.3 templates) decides whether every rank holds the whole embedding
  table or its lm-head slice of the rows: `vocab` frees 1.33 GiB per rank at
  world 4 for one small fold per token lookup and changes no number (the
  ranks' partial rows sum to the same bf16 rows); the other families ignore it.

## Templates added after the rename

| Template | Deployment |
|---|---|
| [cluster_glm-5.3_int4-int8_w4_plain.example.json](cluster_glm-5.3_int4-int8_w4_plain.example.json) | the full GLM-5.3 (int4/int8 RTN) on four nodes, no MTP, bf16 latent cache, 144K context, four request slots, the embedding vocab-sharded (the memory plan: 98.0 GiB of weights + 12.9 GiB of caches per rank, 110.75 GiB in all under the 4 GiB headroom) |
| [cluster_glm-5.3_int4-int8_w4_mtp1.example.json](cluster_glm-5.3_int4-int8_w4_mtp1.example.json) | the same with MTP depth 1 (the draft layer's experts requantized at load), 120K context (110.4 GiB) |
| [cluster_glm-5.3_int4-int8_w4_mtp1_large-cache.example.json](cluster_glm-5.3_int4-int8_w4_mtp1_large-cache.example.json) | the same with the fp8 latent cache, 208K context and a 1.5 GiB prefix arena (110.35 GiB in all: the ceiling under the engine's 4 GiB headroom on a 121.6 GiB node) |
| [cluster_glm-5.3_int4-int8_w4_mtp1_c8.example.json](cluster_glm-5.3_int4-int8_w4_mtp1_c8.example.json) | the same at eight request slots (sixteen decode rows, the 2-/3-/4-/6-/8-slot batch families; c=4 the four-slot template's 41–42 tok/s, c=8 48–50 aggregate at ~170 ms per token per request), 120K bf16 context |
| [cluster_glm-5.3_int4-int8_w4_mtp2.example.json](cluster_glm-5.3_int4-int8_w4_mtp2.example.json) | the same with MTP depth 2 at two request slots (the shape measured under the eight-row cap; the family allows sixteen rows since 2026-09-13, so up to five slots at depth 2), 120K bf16 context |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1.example.json) | GLM-5.3-Flash hybrid on two nodes, MTP depth 1, FP8 KV cache, 160K context, four request slots |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1_large-cache.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1_large-cache.example.json) | the same on two nodes with two request slots instead of four, which buys 256K context and a 2 GiB prefix arena |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp4.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp4.example.json) | DeepSeek-V4.1-Flash on four nodes at DSpark depth 4 (five rows per slot) with the scheduled verify depth — the depth the six-slot template widens; single streams keep the depth-5 template's numbers within the schedule. |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp4_c6.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp4_c6.example.json) | DeepSeek-V4.1-Flash on four nodes at six request slots with DSpark depth 4 (30 decode rows in one batched replay, the family's 32-row cap) and the scheduled verify depth — the six-stream shape the vLLM recipe reports its aggregate at; single streams keep the depth-5 template's numbers within the schedule (docs/operations.md). |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp5.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp5.example.json) | DeepSeek-V4.1-Flash as shipped on four nodes with the DSpark block draft (`mtp_depth` 5: twelve decode rows at two request slots) and the confidence-scheduled verify depth (`mtp_schedule`, λ 0.045: chat 33 → 25 ms/token, every class faster, transcripts unchanged — docs/operations.md), the bounded prefill, 128K context (the memory plan: 72.94 GiB of resident weights per rank, 76.55 GiB in all + 4 GiB headroom on a 121.6 GiB node; not yet booted — the fabric measures follow in docs/deepseek_v41_flash_plan.md §9) |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_plain.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_plain.example.json) | the same without the draft at four request slots (eight decode rows; 70.92 GiB of weights, 74.27 GiB in all) |

## Renamed files

The table applies to both local `.json` files and tracked `.example.json`
templates. All JSON contents are unchanged by the rename.

| Previous local filename | New local filename (link to template) |
|---|---|
| `cluster.json` | [cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.json](cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.example.json) |
| `cluster.nvfp4.json` | [cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json](cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.example.json) |
| `cluster_glm47.json` | [cluster_glm-4.7_nvfp4_w4_mtp1.json](cluster_glm-4.7_nvfp4_w4_mtp1.example.json) |
| `cluster_glm47_t1.json` | [cluster_glm-4.7_nvfp4_w4_plain.json](cluster_glm-4.7_nvfp4_w4_plain.example.json) |
| `cluster_glm47_d2.json` | [cluster_glm-4.7_nvfp4_w4_mtp2.json](cluster_glm-4.7_nvfp4_w4_mtp2.example.json) |
| `cluster_qwen.json` | [cluster_qwen-3.8-flash-next_fp8_w4_mtp1.json](cluster_qwen-3.8-flash-next_fp8_w4_mtp1.example.json) |
| `cluster_qwen_t1.json` | [cluster_qwen-3.8-flash-next_fp8_w4_plain.json](cluster_qwen-3.8-flash-next_fp8_w4_plain.example.json) |
| `cluster_qwen_w2.json` | [cluster_qwen-3.8-flash-next_fp8_w2_mtp1.json](cluster_qwen-3.8-flash-next_fp8_w2_mtp1.example.json) |
| `cluster_qwen_w2_t1.json` | [cluster_qwen-3.8-flash-next_fp8_w2_plain.json](cluster_qwen-3.8-flash-next_fp8_w2_plain.example.json) |
| `cluster_qwen_spark1.json` | [cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1.json](cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1.example.json) |
| `cluster_qwen_spark1_t1.json` | [cluster_qwen-3.8-flash-next_nvfp4_w1_plain.json](cluster_qwen-3.8-flash-next_nvfp4_w1_plain.example.json) |
| `cluster_qwen_spark1_fp8.json` | [cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1_dense-fp8.json](cluster_qwen-3.8-flash-next_nvfp4_w1_mtp1_dense-fp8.example.json) |
| `cluster_qwen_spark1_t1_fp8.json` | [cluster_qwen-3.8-flash-next_nvfp4_w1_plain_dense-fp8.json](cluster_qwen-3.8-flash-next_nvfp4_w1_plain_dense-fp8.example.json) |
| `cluster_qwen_spark1_fp8_d2.json` | [cluster_qwen-3.8-flash-next_nvfp4_w1_mtp2_dense-fp8.json](cluster_qwen-3.8-flash-next_nvfp4_w1_mtp2_dense-fp8.example.json) |

Pass the chosen filename explicitly with `--config`. The setup instructions
are in the [top-level README](../README.md).

## Existing deployments and logs

Stop a running deployment using its old config path **before** renaming the
file. Log and staging namespaces are derived from the absolute config path;
the new filename gets a new namespace. Existing logs are not moved or deleted,
and a new filename does not take ownership of a process started with the old
one. Update saved commands and automation to use the new name.

Historical changelog entries and raw benchmark records retain old filenames.
The generic runtime files `cluster.resolved.json` and the peer's staged
`cluster.json` are unchanged: they are generated by the launcher, not templates.
