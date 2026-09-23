# MiMo-V2.6-Flash (XiaomiMiMo/MiMo-V2.6-Flash-RL) on dgpp — architecture facts and implementation plan (2026-09-22)

Status: implemented on the session core; the fixture ladder, the fabric
bring-up and the measurements are recorded in section 6 as they land.
The checkpoint `XiaomiMiMo/MiMo-V2.6-Flash-RL` (snapshot `5711b268`)
holds 64 expert shards (`model_pp0_ep{0..63}_shard0.safetensors`, four
experts of every layer each; `ep0` also carries the embedding, the head,
every layer's non-expert tensors and the unserved encoders) plus
`model_mtp.safetensors`, 161 GiB of indexed weights. The architecture
study uses its `config.json`, the safetensors headers, the release's
`modeling_mimo_v2.py` (transformers 5.3 remote code), vLLM's `mimo_v2.py`
/ `mimo_v2_mtp.py`, and the MiMo-V2.6 technical report (§2).

## 0. Summary

MiMo-V2.6-Flash (`MiMoV2ForCausalLM`, `model_type mimo_v2`) is a 48-layer
hybrid-attention MoE: layer 0 is global attention with a dense SwiGLU MLP
(16384 wide); layers 1–47 are 256-expert MoEs (top-8, sigmoid + correction
bias, `norm_topk_prob`, no shared expert, `routed_scaling_factor` null =
1) under attention that alternates on `hybrid_layer_pattern`: 39
sliding-window layers (window 128, a learned per-head attention sink
bias) and 9 global layers (0, 5, 11, 17, …, 47). Every layer has 64 query
heads of qk dim 192 / v dim 128; the global layers 4 kv heads (θ = 1e7),
the sliding-window layers 8 kv heads (θ = 1e4); RoPE rotates the first
int(192 × 0.334) = 64 dims (rotate_half pairs (i, i+32)); the value
vectors are scaled by 0.707 before the cache; hidden 4096, vocab 152 576,
1 M positions. Three MTP layers (`model.mtp.layers.{0,1,2}`), each a
sliding-window layer with a dense MLP behind enorm/hnorm → eh_proj and in
front of final_layernorm → the shared head; vLLM serves the first alone and
so does this engine (the chain re-runs it at depth ≥ 2). 309 B parameters,
15 B active. The vision encoder (`visual.*`), the audio encoder
(`audio_encoder.*`, `speech_embeddings.*`) and the separate DFlash draft
(`dflash/`) are in the checkpoint and not served.

The release ships fp8 (e4m3, 128 × 128 block `weight_scale_inv`) for every
dense Linear except each layer's `o_proj` (BF16, on `ignored_layers`), the
router, the embedding and the head (BF16); the routed experts are stored
MXFP4 (`store_dtype`): e2m1 codes two per byte with one e8m0 scale per 32
along K. The fused `qkv_proj` of every layer is pre-sharded in
`num_key_value_heads` = 4 chunks — chunk c = [Q heads 16c..16c+15 | K kv
heads of chunk c | V …] — with its fp8 scale grid tiled per chunk
(`[4 × ceil(chunk_rows / 128), 32]`: 108 rows on a global layer's 3392-row
chunks, 116 on a sliding-window layer's 3712-row chunks).

Relative to what the engine serves, the reuse is large: the router
(noaux_tc sigmoid, ties to the lower id — `MoeRouterMode::SigmoidBias`
with `n_shared_experts` 0), the MXFP4 expert kernels (the DeepSeek-V4.1
form of the fp4 decode slot GEMV and the prefill tile kernel), the fp8
block-128 GEMV / streaming tensor-core GEMM / tile kernel for the dense
projections, the two-rounding norm, the MTP input kernel
(`glm_mtp_input_bf16`), the bf16 GEMV and its 12-bit companions for
`o_proj` / the head / `eh_proj`, the session core (graph capture,
speculative commit, prefix arena, memory plan, family interface), the Qwen
tokenizer regex, the Jinja renderer and the Qwen-XML tool-call parser.
New: paged GQA attention over 192/128-wide heads with a sliding window and
a sink (decode + prefill), the fused pre-sharded projection's finish, a
per-layer-width K/V pool, and the family's config/binding/loader/model.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `hidden_size` / `vocab_size` / `num_hidden_layers` | 4096 / 152 576 / 48 |
| `hybrid_layer_pattern` | 1 (SWA) on 39 layers, 0 (GA) on layers 0, 5, 11, 17, 23, 29, 35, 41, 47 |
| `moe_layer_freq` | 0 on layer 0 (dense MLP, `intermediate_size` 16384), 1 elsewhere |
| heads | 64 query heads (both kinds); GA 4 kv heads, SWA 8 kv heads |
| `head_dim` / `v_head_dim` / `partial_rotary_factor` | 192 / 128 / 0.334 → rotary dim 64 |
| `rope_theta` / `swa_rope_theta` | 1e7 / 1e4 (`rope_type` default, no scaling) |
| `sliding_window` (= `sliding_window_size` = `attention_chunk_size`) | 128 |
| `add_swa_attention_sink_bias` / `add_full_attention_sink_bias` | true / false |
| `attention_value_scale` | 0.707 |
| `attention_projection_layout` / `attention_bias` | fused_qkv / false |
| MoE | 256 experts of 2048, top 8, sigmoid, noaux_tc, `norm_topk_prob`, no shared expert, n_group = topk_group = 1 |
| `layernorm_epsilon` | 1e-6 |
| `num_nextn_predict_layers` | 3 (the engine loads `model.mtp.layers.0`) |
| tokens | eos 151645 (`<|im_end|>`; generation_config adds 151643, 151672), pad 151643 |
| `quantization_config` | fp8 e4m3, dynamic activations, `weight_block_size` [128, 128], `store_dtype` mxfp4, `mxfp4_block_size` 32, `ignored_layers` = every `self_attn.o_proj` |

The parser (`models/mimo/config.hpp`) rejects anything outside this
contract by field name: a chunked-attention size differing from the
window, biased projections, a shared expert, group-limited routing, a
rope scaling, a store dtype other than mxfp4, an ignore list that does not
name every o_proj (or names something else).

### 1.2 One layer (the release's `MiMoV2DecoderLayer`)

```
x   = rmsnorm(h) * input_layernorm                      (fp32 stats, bf16 out)
qkv = x @ Wqkv^T            fp8 [chunks x (Q | K | V), 4096], fp32 out, bf16(dot)
q, k: RoPE on dims [0, 64) — pairs (i, i+32), cos/sin bf16 from fp32 pos x inv_freq,
      x*cos and rotate(x)*sin rounded, the sum rounded
v   = bf16(bf16(v) x 0.707)
K, V appended to the paged cache at the row's slot
attn: softmax(q k^T / sqrt(192) [, sink_h]) v over [max(0, pos - 127), pos]
      (SWA) or [0, pos] (GA); the sink one more softmax column with no value
h  += attn @ Wo^T           bf16 [4096, 64 x 128] (packed columns per rank)
x   = rmsnorm(h) * post_attention_layernorm
h  += MoE(x)  or  dense(x) on layer 0
```

The MoE: logits = fp32(x) @ fp32(gate) (the bf16 gate), scores = sigmoid,
selection over scores + `e_score_correction_bias`, top 8, weights = the
UNBIASED scores at the picks / (sum + 1e-20) × 1; experts SwiGLU without
clamps (act = bf16(bf16(silu(gate)) × up)), the per-token accumulation in
ascending expert id. Exactly GLM's rule minus the shared expert
(`models/glm/moe.hpp`).

### 1.3 Attention (kernels/mimo_attn.hpp)

Forked from the GLM-4.7 kernels (128-wide) for the 192/128 heads:

* **qkv finish** — one warp per (row, head). A q or k head: lane l owns
  the rotary pair (l, l+32) and the nope dims [64+4l, 64+4l+4); bf16(dot),
  the RoPE with the reference's bf16 ops, q to a bf16 buffer, k to the
  paged K cache. A v head: dims [4l, 4l+4), bf16(bf16(dot) × 0.707) to the
  V cache. The fp32 source is the fused projection's output in the
  checkpoint's chunk layout (`MimoQkvLayout`: chunks, the padded chunk
  stride, q/kv heads per chunk) — nothing is reordered at load.
* **attention** — split-KV paged GQA: one block per (row, split, kv head),
  hpk warps (16 on a global layer at world 4, 8 on a sliding-window
  layer), K tiles of 32 tokens padded to 97 words, fp32 scores, online
  softmax, probabilities rounded to bf16 for the V accumulation. A row's
  visible range is [max(0, pos − window + 1), pos]; the tiles are counted
  from its first visible token and split across `n_split` (32 on global
  layers, window / 32 = 4 on sliding-window layers). The sink of head h
  enters split 0's running (max, sum) as (sink_h, 1) before its first
  tile: the combine's algebra then needs nothing else.
* **combine** — the splits merged, bf16(C / L).

Numerics: the engine's flash-style pin (fp32 scores and denominators, the
probabilities rounded once), not the eager reference's bf16 score matrix —
tolerance-equal, as every family here.

### 1.4 The K/V pool (models/mimo/kv_pool.hpp)

One `PagedBlockTable` (64-token blocks) serves every layer; each layer's K
plane is [slots, kv_heads_l × 192] and V plane [slots, kv_heads_l × 128]
at the layer's own kv head count (1 GA / 2 SWA per rank at world 4). The
sliding-window layers keep their full history in the paged cache and read
through the window — the same block list restores a prefix for every
layer, and a rejected verify row leaves only stale rows no later row reads.
Cost at world 4: 57 KiB per token per rank (5.8 GA + 50 SWA + 1.3 draft);
a 128K pool is 7.3 GiB. Section 7 lists the ring form (39 layers × 128
tokens) as the memory lever this leaves.

### 1.5 The draft layer (`model.mtp.layers.0`, `model_mtp.safetensors`)

```
in  = eh_proj([rmsnorm(embed(tok_{q+1})) * enorm | rmsnorm(h_q) * hnorm])   h_q POST final norm
r   = in;  r += attn_swa(rmsnorm(r) * input_layernorm);  r += dense(rmsnorm(r) * pre_mlp_layernorm)
out = lm_head(rmsnorm(r) * final_layernorm)
```

vLLM's `mimo_v2_mtp.py` convention (the same the GLM-4.7 draft measured
best with): the previous hidden is the model's output after the final
norm; depth ≥ 2 chains on the block's output residual `r`
(`kDraftChain`). The draft attention is sliding-window (8 kv heads, the
sink, θ = 1e4), its MLP the 16384-wide dense fp8 MLP.

### 1.6 Tensor census

Per main layer: `qkv_proj.weight` fp8 [13568 or 14848, 4096] +
`weight_scale_inv` [108 or 116, 32]; `o_proj.weight` BF16 [4096, 8192];
`attention_sink_bias` BF16 [64] (SWA); `input_layernorm`,
`post_attention_layernorm` [4096]; MoE: `mlp.gate.weight` BF16 [256, 4096],
`mlp.gate.e_score_correction_bias` F32 [256], 256 × (`gate_proj.weight` U8
[2048, 2048] + `.weight_scale` U8 [2048, 128], `up_proj` the same,
`down_proj.weight` U8 [4096, 1024] + `.weight_scale` U8 [4096, 64]); layer
0: `mlp.{gate,up}_proj` fp8 [16384, 4096] + [128, 32], `down_proj` fp8
[4096, 16384] + [32, 128]. Draft: enorm / hnorm / input_layernorm /
pre_mlp_layernorm / final_layernorm [4096], `eh_proj.weight` BF16 [4096,
8192], the SWA attention set, the dense MLP set. Globals:
`model.embed_tokens.weight`, `lm_head.weight` BF16 [152576, 4096],
`model.norm.weight`. 72 624 text tensors; 457 encoder tensors and the two
later draft layers are ignored by name (`mimo_ignored_tensor`).

## 2. Placement and cost by world size W ∈ {1, 2, 4}

The fused projection's chunk count (4) must divide by W, so the worlds are
1, 2 and 4. Rank r holds chunks [4r/W, 4(r+1)/W): 64/W query heads, 4/W
global and 8/W sliding-window kv heads — the chunks stacked as one fp8
matrix, each chunk but the last padded to a 128-row multiple (zero rows,
never read) so every chunk's scale rows re-anchor at its start. `o_proj`
packed columns (64/W × 128), the sink's heads, the dense MLP at 16384/W
(fp8, 128-aligned), every expert at 2048/W (MXFP4, 32-aligned), the head
vocab-sharded; the router, the norms, the embedding and the draft's head
tensors replicated.

| per rank | W = 4 | W = 2 |
|---|---|---|
| experts (MXFP4 + e8m0) | 40.3 GiB | 80.6 GiB |
| fp8 dense (qkv, layer 0, draft MLP) | 0.9 GiB | 1.8 GiB |
| bf16 (o_proj, eh_proj, head slice, embed, router) | 2.6 GiB | 3.6 GiB |
| weights | ~44 GiB | ~86 GiB |
| K/V per token | 57 KiB | 114 KiB |

Decode traffic at T = 1, W = 4: per layer 15 MB fp8 qkv + 12.6 MB bf12
o_proj + 25.6 MB of eight expert slices ≈ 53 MB; ×48 + the 234 MB bf12
head slice ≈ 2.8 GB per step per rank — an 11 ms floor at the line rate,
before the 96 boundary folds.

## 3. Design decisions

* **D1 — the release's formats as shipped.** fp8 block-128 dense matrices
  through `load_quant_rows/cols` (the shared re-anchoring loaders), MXFP4
  experts through the family's `load_mxfp4_rows/cols` (the DeepSeek
  helpers with `.weight` / `.weight_scale` names), BF16 o_proj / head /
  eh_proj packable into 12-bit companions (`engine.bf16_weights`). No
  requantization, no format change.
* **D2 — the fused projection stays fused and pre-sharded.** One fp8
  matrix per rank in the checkpoint's chunk order, the finish kernel
  reading each head at its chunk offset; padding rows between chunks keep
  the per-chunk scale tiling exact at W < 4. The alternative — reordering
  into [Q | K | V] — would put two scale blocks inside one 128-row block
  and force a requantization (vLLM's path at TP ≠ 4).
* **D3 — one attention kernel for both kinds.** The layer object rebinds
  the window (0 or 128), the sink pointer, the rope table and the kv head
  count per layer; the split count follows the kind.
* **D4 — the full paged cache on sliding-window layers.** Simplicity and
  prefix-cache uniformity first; the ring form is the recorded follow-up.
  Two storage forms (`engine.kv_dtype`): bf16 rows, or the fp8 row form of
  kernels/latent_format.hpp per (token, kv head) — e4m3 codes with one
  fp32 scale (absmax / 448) per head row, K and V each — 328 bytes per kv
  head per token against 640. The finish quantizes the values the bf16
  cache would store; the attention dequantizes a tile into the same shared
  tiles, so the fp8 kernel over a cache is bitwise the bf16 kernel over
  the dequantized rows (mimo_attn_test pins both). The two-node template
  takes it for a 256K pool (58 KiB per token per rank, 97.2 GiB total).
* **D5 — the first draft layer only, the chain for depth.** vLLM's serving
  shape; the later layers load nothing (ignored by name).
* **D6 — fp8 sites on the streaming tensor-core GEMM at decode rows**
  (`launch_scale_gemm_grid_*` with `decode_mma`): a row's chain is the same
  whatever rows share the launch, so the batched decode rows are bitwise
  their scalar rows; prefill chunks take the GEMV chunks / tile kernel.
  `DGPP_MIMO_DENSE_GEMV=1` restores the 4-row chunks for A/Bs.
* **D7 — the bf16 sites as GLM-4.7's**: the GEMV lowering to
  `dense_gemv_rows`, the 12-bit companions at decode rows, cuBLASLt above.
* **D9 — side-granted bf16 matrices prefetch in isolation.** Under
  `engine.bf16_weights = "bf12"` the raw o_proj / head / eh_proj are side
  grants (their own allocations, released once packed). The first-token
  draft after the warm prefill runs before the capture packs them, so the
  head's resident view is still the raw matrix; the prefetcher's range
  merge (up to 2 MB between adds) would bridge the unmapped gap between the
  globals image and that side grant (compute-sanitizer, 2026-09-22: an
  illegal read 58 KB past the globals allocation on every rank). The model
  adds a side-granted view with `add_isolated`, as it adds a companion.
  The other families' templates run `"bf12+bf16"` (no side grants) and
  keep their merge; a plain `"bf12"` run of one of them would meet the
  same gap (a note for their plans, not a change here).
* **D8 — the encoders are not served.** `visual.*`, `audio_encoder.*`,
  `speech_embeddings.*` bind as ignored; image/audio parts in a request are
  refused by the text frontend.

## 4. Layout

```
src/models/mimo/config.{hpp,cpp}        the parsed contract
src/models/mimo/binding.{hpp,cpp}       the expected-tensor table, the ignore rule, the TP geometry check
src/models/mimo/loader.{hpp,cpp}        MimoLayerStream on ResidentLayerStream: chunks, MXFP4, fp8, bf16 slices
src/models/mimo/kv_pool.{hpp,cpp}       the per-layer-width paged K/V pool
src/models/mimo/layers.{hpp,cpp}        MimoAttentionLayer, MimoDenseMlp
src/models/mimo/forward.{hpp,cpp}       MimoModel on SessionModel: the row walk, the memory plan, the draft block
src/models/mimo/attn_reference.{hpp,cpp} the host oracle of the finish and the windowed sink attention
src/kernels/mimo_attn.{hpp,cu}          the three kernels
tests/cuda/mimo_fixture.hpp             the tiny release (8 heads, 4/8 kv heads, window 32, 8 experts of 128)
tests/cuda/mimo_{attn,loader,forward,decode,tp,engine}_test.*   the ladder
tests/unit/mimo_{config,binding}_test.cpp
tools/mimo_reference_dump.py            the pure numpy reference over the fixture
apps/mimo_{load,forward}_check.cpp      the real-checkpoint checks
deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w{4,4_plain,2}.example.json
scripts/fabric_mimo_serve.sh
```

## 5. Implementation stages and validation

G0 config + binding (unit tests over the real config.json) · G1 the
kernels against the host oracle (`mimo_attn_test`) · G2 the loader over
the fixture at worlds 1/2/4 (`mimo_loader_test`) · G3 the forward against
the numpy reference (`mimo_forward_test`: per-layer residuals, routing,
logits, the draft rows) · G4 the session surface (`mimo_decode_test`:
chunked prefill == forward, decode vs forward, snapshots, MTP, the batched
rows bitwise their scalar rows) · G5 loopback TP worlds 2 and 4
(`mimo_tp_test`) · G6 the graph engine on loopback worlds
(`mimo_engine_test`) · G7 the real checkpoint: `mimo_load_check`,
`mimo_forward_check` transcripts, then the four-node fabric through
`scripts/fabric_mimo_serve.sh` (greedy transcripts, the API check, the MTP
classes, the prefill probe, the task evals) · G8 the benchmark campaign
into docs/benchmarks.md and the README.

## 6. Status (2026-09-22)

G0–G6 green in the ci suite (142 gates, every family's included): the
attention kernels bitwise against the host oracle across 54 window / sink /
split / hpk configurations (finish 0 ulps over 110k elements); the loader
byte-exact at worlds 1, 2 and 4 (the padded chunk stacking, the re-anchored
fp8 grids, the MXFP4 slices, the fp32 sink), 4 ignored tensors; the forward
against the numpy reference teacher-forced: per-layer l2 ≤ 1.0e-3, final
hidden bitwise, logits top-1 exact and matched within 2.8e-7, 0 routing
flips, the draft rows top-1 exact; the session gates (chunked prefill ==
forward bitwise, the batched decode rows bitwise their scalar rows through
4 rows, the window verified at the session level on a 2-layer fixture:
positions past the window leave a sliding-window layer's rows bitwise
unmoved while a global layer's move); loopback worlds 2 and 4 bitwise
across ranks with the world-1 oracle inside the budget; the graph engine
(scalar / batched / MTP depth 1 / depth 2 / scheduled) exact against the
eager engine.

G7 on the fabric (four nodes, the w4 template, bf12): the memory plan
50.06 GiB per rank at a 128K pool (weights 39.49 GiB with the bf16
matrices released to their 12-bit form, K/V 6.95 GiB); the warm capture
records 4 scalar variants and the 2/3/4-slot row batches; greedy
transcripts coherent in every class (the model's own `<think>` block
parsed as reasoning); the API check passes; MTP depth 1 at 22–23 ms/pass,
1.73–1.98 tokens per pass, acceptance 73 % (chat) to 98 % (json) —
12.7–13.9 ms/token; prefill 1.16 / 0.88 / 0.87 ms/token at 512 / 2K / 8K
(0.60 / 1.86 / 7.31 s); the four ranks' operation streams identical.
Two defects found and fixed on the way: the MXFP4 GEMV core's compiled K
set lacked 4096 / 2048 (`k must be a multiple of 32 in the fp4 core's
compiled set` on the first real-checkpoint decode), and the prefetch of a
side-granted head (D9). The first campaign's numbers (the family as
landed): benchmarks/results/2026-09-22-mimo-v26-flash/.

**After the kernel round (§7.1, §7.2; the campaign re-run
benchmarks/results/2026-09-22-mimo-v26-flash-opt/, the numbers on
docs/benchmarks.md):** four nodes MTP depth 1 at 23.2 ms/step, 76–86 tok/s
per request at C1 (128.7–142.1 wall tok/s at C4), the plain step 17.3 ms;
cold prefill 1.77 / 6.55 / 28.4 s at 2K / 8K / 32K (the tiled tensor-core
attention: −3 / −12 / −22 % against the split-KV kernel's 1.83 / 7.42 /
36.6), a 121K prompt in 143 s with decode at 63 tok/s; two nodes 42–49
tok/s at C1 (BF16 pool) and 40–48 (256K FP8 pool: a 239K prompt in 839 s,
decode 24 tok/s); HumanEval 153–154 / 164, GSM8K 294–295 / 300, extraction
100 / 100 on every deployment; the ranks' operation streams identical;
solo and batched decode identical.

## 7. Left on the table

Ordered as the work was queued (2026-09-22): the decode step first (§7.1),
then the prefill attention (§7.2), then the rest.

### 7.1 Decode: measured against the roofline

Per rank per plain step at world 4 the launches stream 2.96 GB (experts
1.26, fused qkv 0.72, o_proj bf12 0.61, head slice 0.23, layer-0 MLP and
routers 0.15); at world 2, 5.83 GB. Floors at this box's 248 GB/s: 11.9 /
23.5 ms; measured 17.8 / 29.4 ms — 67 % / 80 % of the line rate.

**The in-situ picture (2026-09-22, `scripts/fabric_profile.sh` on the
four-node template, T = 1 plain decode, nsys on rank 0; 18.4 ms/step,
978 kernels/step).** One sliding-window layer is 332 µs from qkv to qkv:

| launch | µs | note |
|---|---|---|
| qkv fp8 mma (15.2 MB) | 60.5 | line rate; the prefetch of its weights runs into it |
| finish + partial + combine | 3.9 + 10.7 + 1.3 | latency-bound |
| o_proj bf12 (12.6 MB) | 36.9 | above line rate: L2-prefetched during the qkv GEMV |
| fold 1 (bus_allreduce_graph_kernel) | 24.0 | DRAM idle but the 2 MB router prefetch |
| residual add + rmsnorm + router | 1.5 + 6.0 + 7.7 | latency-bound |
| MXFP4 gate/up (17.3 MB) | 93.5 | 182 GB/s cold |
| MXFP4 down (8.4 MB) | 41.5 | 202 GB/s cold |
| accum + round | 1.3 + 0.9 | |
| fold 2 | 27.0 | the next layer's qkv prefetch runs through it |
| residual add + rmsnorm | 1.5 + 5.8 | |
| 14 graph nodes' gaps | ~7 | 0.5 µs each |

Per step: the folds 3.18 ms (97 × 32.8 µs GPU-side), the experts 6.2 ms
(4.3 + 1.95), qkv 2.7, o_proj 1.75, head 0.88, rmsnorm 0.59, attention
0.81, router 0.36, adds/rounds/accums 0.23; main-chain busy 16.9 ms of the
18.1 ms step. The DRAM-side losses, in order: the expert kernels below
line rate (33 µs/layer), fold 1 and the small kernels behind it with
nothing to prefetch (~40 µs/layer of idle DRAM), the attention chain, the
two norms.

**The cold microbench** (`mimo_step_bench --cold`: an L2 read-sweep before
every timed launch — the warm loop had the expert kernels at 242 GB/s
because 28.8 MB cycling through the 24 MB L2 still hit; a read-modify-
write sweep is wrong too, it leaves dirty lines whose write-backs halve
the next launch's read rate) reproduces the fabric's per-kernel numbers
(gate/up 84, down 41, qkv 73.5, o_proj 52, rmsnorm 9.9). ncu on the
gate/up slot kernel: 78 registers → three 256-thread blocks per SM (50 %
occupancy), 576 blocks in four waves, warps stalled on their loads half
the time — a block's life is one batch of loads, and its prologue (two
dependent table lookups, the 8 KB activation staging, a barrier) has
nothing in flight.

**Shipped from this round** (each bitwise the chain it replaces, gated
in the kernel tests; the fabric A/B in `benchmarks/results/2026-09-22-
mimo-v26-flash/`):

* `kernels/add_rmsnorm`: the residual add fused into the two-rounding
  norm that follows it, the row register-resident, the fp64 sum through
  warp shuffles (four independent chains: GB10's dependent fp64 add is
  ~90 ns), the weight loads issued before the reduction. 9.8 → 5.7 µs per
  boundary (event-timed), two launches fewer per layer. Used at every
  MiMo boundary (`enqueue_layer` returns its MLP output pending; the next
  norm adds it; `add_pending` completes rows no norm follows).
* `mimo_attn_fused`: finish + split partials + combine in one launch for
  decode batches — the batch's (request, position) table in shared
  memory, a request's other batch rows overlaid onto the staged tile from
  the fp32 projection (never read from the cache another block of the
  launch is writing), the last block per (row, kv head) to arrive on a
  self-resetting counter combines (m/l staged to smem, c rows loaded eight
  splits at a time, the neutral splits skipped), tiles register-
  prefetched one ahead, block-table entries staged once, the score loop
  on 16-byte tile rows (kRowStride 200). Global layer at 32 splits 16 →
  10.8 µs, sliding-window 8 → 6.8 µs; the per-tile cost is now compute
  (a (row, split, kv head) block's 16 warps share one SM: ~2.4 µs per
  32-token tile) — tensor-core scores would be the next step there and
  would change the summation order. Prefill rows keep the three kernels
  (their K/V are all appended before any attention).
* `fp4_gemv::issue_pass` / `warp_row_dots_issued`: the slot kernels issue
  the first pass's weight loads before staging the activations (the FMA
  chain untouched): a null result in the cold bench (84.3 / 41.0 µs vs
  84.1 / 41.0), kept for the cleaner block prologue.
* `fp4_gemv::consume_chunk_mx`: the MXFP4 chunk through the hardware
  e2m1x2 → f16x2 conversion and one f16x2 multiply when its e8m0 scale is
  in 2^-23 .. 2^13 (exponent codes 104..140 — every weight tensor's; the
  product is then exact in f16, so the fp32 values the FMA chain consumes
  are the exact path's, bitwise), the exact fp32 path otherwise; four
  instructions a byte instead of eight. Cold bench gate/up 84.3 → 82.6 µs
  (the kernel is not compute-bound; the win is the smaller one it can
  give). Shared with DeepSeek-V4.1-Flash (same bits, fewer instructions).

**The fabric A/B** (`benchmarks/results/2026-09-22-mimo-v26-flash/ab/`,
four legs plus a closing baseline; ms/step, the mean of five classes):
the fusions take the MTP step at C1 from 23.39 to 23.04 (−1.5 %), C2 / C4
−0.4 / −1 %, the plain T=1 step not at all (17.85 vs 17.84 / 18.29): the
kernels are 8–12 µs shorter per layer, but most of that sat under
DRAM-bound phases (the norm after fold 2 under the qkv prefetch, the
attention chain under the o_proj prefetch), where the following GEMV
waits for the bytes either way. Decode transcripts identical to the
baseline in every class.

Rejected, with numbers:

* `__launch_bounds__(256, 4)` on the MXFP4 slot kernels (78 → 64
  registers: gate/up 84 → 89 µs cold, the spills cost more than the
  fourth block gains).
* The early persisting prefetch: the next layer's qkv into an L2
  persisting set-aside (`cudaAccessPropertyPersisting` on the prefetch
  launch, `applypriority.global.L2::evict_normal` to release the lines
  before a prefill walk — never `cudaCtxResetPersistingL2Cache`, which
  needs the device idle and deadlocks against a peer's spinning
  collective) from fold 1 on, so the fold-1 idle DRAM and the expert
  kernels' ramps carry it. In the probe the lines survive a 96 MB sweep
  and read at 700–900 GB/s; on the fabric the step LOSES 7–10 % at both
  8 and 16 MiB: the 16 MB window at the Light rate needs 185 µs, so it
  runs through the whole expert block and competes there (gate/up 93.5 →
  111 µs), the o_proj window loses its L2 (36.9 → 55.9 µs), fold 2 slows
  (27 → 44), and the qkv GEMV gains only 14 µs (60.5 → 46.5). The
  attribute itself is free per launch. The prefetcher keeps the mode
  (`open_window(..., persisting)`, `release_persisting`) and the model the
  experiment knob `DGPP_MIMO_L2_PERSIST_MB`; a redesign would size the
  window to fold 1's idle time alone (~6 MB at the Full rate) and stop
  before the experts start — the fold's own latency budget then decides.

Left, in the order the numbers rank it: the folds (2.4 ms of the 18 ms
step; the bus's latency, not this family's); the expert kernels' ramps
(their 73–81 % of line rate is the wave structure, not occupancy or the
prologue: overlapping the down kernel's start with the gate/up tail —
programmatic dependent launch — is the untried lever); the router GEMV
(7.7 µs for an L2-resident 2 MB: shared kernel, few blocks); the MTP
pass's second head read; tensor-core scores in the decode attention
(the per-tile compute, at long contexts).

### 7.2 Prefill: no per-row K/V re-dequantization

The prefill attention ran one block per (query row, kv head) over the
row's whole visible range: every K/V tile staged — and under the fp8
cache dequantized — once per QUERY ROW, an O(T²) re-read, and the scores
and PV products on scalar fp32 FMAs, O(T²) work the 32K cold prefill
showed (1.11 ms/token against 0.88 at 2K in bf16; 2.43 against 1.72 with
the fp8 cache; ~10 s of the 36.6 s at 32K were the nine global layers'
scalar attention).

**Shipped 2026-09-22: `mimo_attn_prefill`** (kernels/mimo_attn_prefill.cu),
the query-tiled tensor-core kernel: a block per (tile of 64 / hpk rows,
kv head) — 4 rows x 16 heads on a global layer, 8 x 8 on a sliding-window
one — stages each 32-token K/V tile once for its 64 query vectors (fp8:
dequantized once), S = QKᵀ and O += PV on `mma.sync.m16n8k16` bf16 with
fp32 accumulation (the FA2 layout: a warp owns 16 query vectors, the S
fragments re-pack as PV's A operand; V staged transposed so both B
operands are 32-bit shared loads on conflict-free strides), the online
softmax per row with the decode kernel's roundings (probabilities bf16,
denominator unrounded), the sink in each head's initial (m, l), per-row
causal / window masks, a group prefill's span joins handled by looping
over the runs of one request with the other rows inert. Prefill rows are
no longer bitwise the split-KV kernel's (the tensor core's summation
order; a 1e-6 score difference flips a bf16-rounded probability now and
then): the gate is the oracle test's statistic — l2-relative 1e-4 to
3e-4 against the chain, no element past two ulps, in all eight cases.
`DGPP_MIMO_PREFILL_ATTN=chain` keeps the split-KV path. The
per-row-re-dequantization cost is gone by construction; the measured
prefill numbers are the campaign's.

* The sliding-window layers' K/V as a 128-token ring per slot (39 of 48
  layers): −87 % of the pool's bytes per token; needs the prefix snapshot
  to carry the rings.
* The gate/up fp8 GEMVs of the dense MLPs as one multi-problem launch
  (`launch_scale_gemv_multi_bf16`, ≤ 8 rows).
* The draft layers 1 and 2 as the depth-2/3 chain's own weights (the
  release trained them so; the chain re-runs layer 0).
* The DFlash block drafter (`dflash/`, 5 SWA layers, window 1024, block 8):
  a different speculative scheme, not the MTP block.
