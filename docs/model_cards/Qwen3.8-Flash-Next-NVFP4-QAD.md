# Qwen3.8-Flash-Next-NVFP4 (quantization-aware distilled)

Serving note for
[local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4)
on DGPP (revision 6909a5be, issue #113). Model architecture and weights
belong to their authors; this page records how DGPP serves this checkpoint
and what was measured here.

The same Qwen3.8-Flash-Next stack as the
[NVIDIA NVFP4 release](../qwen38_single_spark.md): 48 layers of gated
DeltaNet and Qwen sparse attention, 512 routed experts with 10 active, the
hashed n-gram embedding on layer 1, one MTP draft layer. The publisher
distilled the quantized student against the BF16 teacher and ships a
mixed-precision modelopt export; its tokenizer, chat template and
generation config are byte-identical to the NVIDIA release's. The checkpoint
is 99.0 GiB in 43 shards against the NVIDIA release's 123.5.

## Formats in DGPP

DGPP reads the export directly (no requantization, no converter). The
config's `quantization_config` is modelopt's `MIXED_PRECISION` block, which
names every quantized module under `quantized_layers`; the parser requires
the complete set below and refuses anything else by name
(`src/models/qwen/config.cpp`).

| Module | Shipped form | DGPP path |
|---|---|---|
| Routed experts, 48 backbone layers | NVFP4 (e2m1 pairs, e4m3 scales per 16, one F32 global, an activation `input_scale`) | the NVIDIA release's path, unchanged |
| GDN `in_proj_a/b/qkv/z`, `out_proj`; QSA `q/k/v/o_proj`, the indexer; the shared experts | MXFP8: e4m3 `.weight` [N, K] with one e8m0 byte per 32 columns of a row (`.weight_scale` U8 [N, K/32]) | the fp8 cores on a 1 x 32 scale grid: the attention-side projections keep the e8m0 bytes resident (the streaming MMA and the dequant bridge read the byte grid, `mma_gemv.cu scale_at`), the shared experts carry the grid widened to F32 for their fused decode kernels (exact either way); `in_proj_a`/`b` (0.12 MB each) decoded to BF16 at load (exact) for the bf16 GEMV they ride |
| N-gram table, 128 shards | NVFP4: codes U8 [rows, 80], e4m3 scales [rows, 10], one F32 `weight_scale_2` (26.8 GiB) | resident on the Spark (two device planes), or mapped from the shards with the scale rows gathered alongside; the gather decodes `bf16((e2m1 x e4m3) x global)` in modelopt's order |
| Draft layer's routed experts | `W4A16_NVFP4`: the NVFP4 triple without an `input_scale` | the NVFP4 expert path with bf16 activations on that layer (no W4A4) |
| Hyper-connection norms (48 layers, the model mixer), PLE norms | F32 (trained in F32; the publisher's serving requirement) | the norm kernels read F32 weights at their precision; the draft layer's norms stay BF16 |
| Embeddings, head, GR mixing weights, routers, PLE projections and conv, the draft layer's projections | BF16 | as the NVIDIA release; `engine.dense_weights` and `engine.fp8_head` apply to this remainder alone |
| Vision tower | BF16 (the card says MXFP8/NVFP4; the headers say BF16) | the Qwen vision path |

Every value the kernels multiply is the checkpoint's own: an e4m3 code under
a power-of-two scale is exact in bf16, so the MXFP8 stack reaches the fp8
GEMV, the streaming MMA and the dequant bridge as the same numbers through
the grid they already take (`rs = 0, cs = 5`); the byte grid and the F32
grid are bitwise the same chain (`mma_gemv_test`). The export is detected
from the checkpoint; no engine key selects its forms.
`engine.mtp_expert_format: bf16_fused` (the RadixArk layout) is refused for
it.

## Gates

- `qwen_config_test` parses the modelopt block and refuses a layer left in
  BF16, an undeclared table or draft, another group size or algorithm, a
  quantized router or KV cache.
- `qwen_binding_test` binds every one of the snapshot's 301 593 tensors when
  it is in the hub cache (333 vision tensors counted, nothing missing,
  nothing unexpected) and accepts worlds 1, 2 and 4 (world 8 would cut the
  shared expert below the 32-wide MXFP8 group).
- `qwen_loader_test` loads the QAD twin fixture byte-exact at worlds 1, 2
  and 4: MXFP8 payload and scale slices on the 1 x 32 grid, the exact a/b
  decode, F32 norms, weight-only draft experts without activation scales,
  the NVFP4 table resident and mapped (codes and scale rows, the gather's
  scale staging).
- `qwen_norm_test`, `qwen_gr_test`: the F32 norm paths bitwise the BF16
  ones on bf16-valued weights and bitwise the unfused F32 chain otherwise;
  `qwen_ple_test`: the NVFP4 gather bitwise the host decode, the staged form
  bitwise the device one.
- The fixture chain (`qwen_qad_fixture` / `_smoke` / `_generate` /
  `_forward_test` / `_decode_test`): the pure-Python reference dequantizes
  every form exactly (`tools/qwen_reference_dump.py`) and the device forward
  holds its budget layer by layer.
- The real checkpoint loads at world 1: 69.20 GiB of layers, 2.42 of
  globals, the 26.82 GiB table resident — 98.44 GiB, every source byte
  consumed, 99 s cold.

## The single-Spark recipe

`deploy/cluster_qwen-3.8-flash-next_nvfp4-qad_w1.example.json`: the NVIDIA
single-Spark recipe with the n-gram table resident (the NVFP4 table fits
where the 47.7 GiB FP8 table did not), so decode has no NVMe gather; the
room it takes comes out of the context — a 128K pool and a 1 GiB prefix
arena (the memory plan: 110.4 GiB plus the 4 GiB headroom against the
Spark's ~116 GiB free). `--ngram-table mmap --kv-capacity 262144
--prefix-cache-gib 3` trades the gather back for the NVIDIA recipe's
context. The server boots in 112 s from the checkpoint (no resident image).

## Measured (2026-10-14, one Spark, this build, the NVIDIA release on the same build as the yardstick)

Both from `build-ci/dgpp-serve` with the resident image off; the NVIDIA
release ran its own one-node template (mapped table, 256K pool, dense FP8
at load). Decode from `serve_bench.py` (200 sampled tokens, three runs) and
the server's step stats; classes from `serve_mtp_classes.py` (greedy, 300
tokens); prefill from `serve_prefill_probe.py`; evals from `serve_eval.py`
(30 items each, reasoning effort low, thinking on).

| | NVIDIA NVFP4 release | this export |
|---|---|---|
| T=1 decode, ms/step | 31.6–32.0 | 32.9 (34.4 with the F32-widened scale grid) |
| MTP depth 1, ms/pass (greedy classes) | 38 | 39–40 (40–41 with the F32 grid) |
| MTP depth 1 sampled pace, ms/token | 23.3–24.1 | 24.6–25.1 |
| MTP tok/pass by class chat/code/prose/json/math | 1.56/1.67/1.74/1.83/1.86 | 1.53/1.75/1.75/1.93/1.89 |
| Prefill ms/token at 512 / 2K / 8K | 1.13 / 0.79 / 0.64 | 1.12 / 0.79 / 0.65 (an earlier run read 1.00 / 0.72 / 0.64: the probe's spread) |
| GSM8K / HumanEval / extraction (30 each) | 59/60, 38–39/40, 30/30 on 2026-09-10 | 29/30, 30/30, 30/30 |

The remaining T=1 gap is the per-32 scale fetch inside the streaming MMA's
window (the attribution bench: the dense stack per step 13.75 ms on the
128 x 128 grid, 16.29 ms with F32 1 x 32 scales, 14.89 ms with the e8m0
bytes); staging a window's scale bytes with its weights would close it.
The MXFP8 stack is the checkpoint's own values where the NVIDIA recipe's
dense FP8 is a load-time requantization of its BF16.

## Benchmark row (2026-10-14, release build, one Spark)

The [benchmarks](../benchmarks.md) procedure on the template above
(`build-release/dgpp-serve` built from this change, the resident image off,
`DGPP_NO_SWAP=1`, 32 CUDA connections, one deployment at a time, zero
process swap on every leg); the NVIDIA release's cells are its one-node FP8
dense row in the same tables.

| | this export | NVIDIA NVFP4 release (FP8 dense, mapped table, 256K pool) |
|---|---|---|
| C1 engine tok/s, five classes (MTP1) | 41.1–50.2 | 43.5–51.9 |
| C1 / C2 / C4 / C8 wall tok/s | 40.1–48.6 / 63.5–74.6 / 87.8–99.6 / 86.6–100.8 | 42.4–50.1 / 62.1–73.1 / 86.5–97.7 / 84.3–98.9 |
| Plain / MTP2 / MTP3, C1 engine tok/s | 30.4–30.5 / 44.1–63.6 / 42.3–73.7 | 31.8–31.9 / 45.5–66.1 / 44.2–73.8 |
| Cold prefill s at ~2K / ~8K / ~32K | 1.412 / 5.236 / 21.477 | 1.363 / 4.522 / 18.054 |
| Decode ms/pass at 0 / 32K / 64K / 128K context | 39.82 / 41.99 / 42.33 / 43.30 | 40.53 / 42.73 / 43.12 / 43.97 |
| Cold prefill s at 32K / 64K / 128K (parcel documents) | 20.595 / 43.229 / 95.045 | 18.241 / 38.461 / 85.217 |
| HumanEval / GSM8K / extraction | 161/164 / 297/300 / 100/100 (one response at the cap) | 160/164 / 294/300 / 100/100 |

The C1 engine medians by class: prose level (43.6 against 43.5), code,
JSON and math 3–6 % under, chat 10 % under (41.1 against 45.9: 39.8 ms per pass, 4–5 %
longer than the NVIDIA release's on one build, at the chat class's 1.64
tokens per pass against the other classes' 1.72–1.98); MTP2 and MTP3
within 0–4 %; the 32K–128K context passes
1–2 % faster. The cold-prefill cells run 4–19 % (GSM8K probes) and 11–13 %
(parcel documents) longer than the NVIDIA row's, which was measured on
2026-10-06/07 with a fixed 4,096-token prefill budget; on one build and the
adaptive policy the two exports' probes read level (the table above:
1.12 / 0.79 / 0.65 against 1.13 / 0.79 / 0.64 ms/token). The per-pass
remainder is the scale fetch described above. The plain and MTP1 greedy
transcripts matched in all five classes.
