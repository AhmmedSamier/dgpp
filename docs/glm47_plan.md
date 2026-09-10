# GLM-4.7 (nvidia/GLM-4.7-NVFP4) on dgpp — architecture facts and implementation plan (2026-09-09)

Status: being built. The checkpoint `nvidia/GLM-4.7-NVFP4` (snapshot
`47fa7dc8`) is in the HF cache on this node: 44 shards + `mtp.safetensors`,
215 GB on disk. Every fact below comes from its `config.json`,
`hf_quant_config.json`, the safetensors headers, and the transformers
`modeling_glm4_moe.py` reference (4.57) plus vLLM's `glm4_moe_mtp.py` for the
draft layer.

## 0. Summary

GLM-4.7 (`Glm4MoeForCausalLM`, `model_type glm4_moe`) is a 92-layer dense-
attention MoE: 3 dense layers then 89 MoE layers of 160 routed experts
(top-8, sigmoid + correction bias, `routed_scaling_factor` 2.5) plus one
shared expert; grouped-query attention with 96 query heads and 8 kv heads
of 128, q/k/v biases, per-head q/k RMSNorm, half-split partial RoPE on the first
64 dims (θ = 1e6); a plain pre-norm residual; hidden 5120, vocab 151 552,
202 752 positions; one MTP draft layer (`model.layers.92`, DeepSeek-V3
shape: enorm/hnorm → eh_proj → one full layer → shared_head.norm → the
shared lm head). 355 B parameters, 32 B active.

The NVIDIA release quantizes every `Linear` except the attention
projections and `lm_head` with modelopt NVFP4: e2m1 codes (`weight`, U8
[N, K/2]), e4m3 block scales per 16 (`weight_scale`, [N, K/16]) and one
fp32 per-tensor scale (`weight_scale_2`); `input_scale` (W4A4 activation
scale) and `k_scale`/`v_scale` (FP8 KV, all 1.0) ride along unused. The
draft layer is BF16 throughout. The embedding, `lm_head`, the draft's
`embed_tokens` and `shared_head.head` are byte-identical tensors (checked).

Relative to what the engine serves, the reuse is large: the router
(noaux_tc sigmoid, ties to the lower id), the NVFP4 expert kernels
(decode slot GEMV + prefill ldmatrix tile), the two-rounding GLM norm,
the MTP input kernel (`glm_mtp_input_bf16`), the bf16 GEMV for the
attention projections and the head, the engine core (graph capture,
speculative commit, prefix arena, memory plan, family seam) and the
GLM tokenizer/template/tool machinery. New: paged GQA attention over
128-wide heads (decode + prefill), half-split partial RoPE with the
per-head norm and the biases, a KV pool without index caches, an NVFP4
GEMV core for K ∉ 2^n, an NVFP4 shared expert inside the slot kernels,
and the modelopt scale convention in the loader.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `architectures` / `model_type` | `Glm4MoeForCausalLM` / `glm4_moe` (no `text_config` nesting) |
| layers | 92 (`first_k_dense_replace` 3: layers 0–2 dense MLP, 3–91 MoE) + draft layer 92 (`num_nextn_predict_layers` 1) |
| hidden / vocab / eps | 5120 / 151 552 / 1e-5; `hidden_act` silu; `tie_word_embeddings` false |
| attention | 96 heads, 8 kv heads, `head_dim` 128, `attention_bias` true (q/k/v biases, o none), `use_qk_norm` true, `partial_rotary_factor` 0.5 (64 rotary dims, interleaved pairs), `rope_theta` 1e6, no scaling |
| dense MLP | `intermediate_size` 12 288 |
| MoE | 160 experts, top-8, `n_group` = `topk_group` = 1 (plain top-k), sigmoid scores + `e_score_correction_bias`, `norm_topk_prob` true, `routed_scaling_factor` 2.5, `moe_intermediate_size` 1536, `n_shared_experts` 1 (1536 wide) |
| tokens | eos [151329 `<\|endoftext\|>`, 151336 `<\|user\|>`, 151338 `<\|observation\|>`], pad 151329; `<think>` 151350, `</think>` 151351, `<tool_call>` 151352, `<arg_key>` 151356 … (the GLM-5.3 token set, ids shifted by −3491) |
| generation defaults | temperature 1.0 (no top_p / top_k in `generation_config.json`) |
| quantization | modelopt NVFP4, group 16, `kv_cache_scheme` FP8 static; `ignore`: `lm_head`, every `self_attn*`, `model.layers.92*` |

### 1.2 One layer (transformers `Glm4MoeDecoderLayer`)

```
h = RMSNorm_5120(x) · w_in                      # two roundings (Glm4MoeRMSNorm == Glm5's)
q = W_q h + b_q ; k = W_k h + b_k ; v = W_v h + b_v      # bf16 Linear with bias
q_h ← RMSNorm_128(q_h) · w_q ; k_h ← RMSNorm_128(k_h) · w_k   # per head, two roundings
RoPE on dims [0, 64) of q_h, k_h at position t (half-split pairs (i, i+32) — transformers' rotate_half, NOT the older glm/glm4 interleaved form: the interleaved reading cost the model its coherence past ~30 tokens, found against transformers' own layer code on 2026-09-10; bf16 ops)
o = softmax(q · Kᵀ / √128) · V                  # GQA 12:1, fp32 softmax, p rounded to bf16 before V
x = x + W_o o                                   # bf16 add
h = RMSNorm_5120(x) · w_post
y = MLP(h) | MoE(h)
x = x + y
```

RoPE detail (`apply_rotary_pos_emb` with `rotate_half` over `x[..., 0::2]`,
`x[..., 1::2]`): `cos`/`sin` are computed in fp32 from `inv_freq[i] =
1 / θ^(2i/64)`, i < 32, cast to bf16; for pair i the rotated values are
`q[2i]·c − q[2i+1]·s` and `q[2i+1]·c + q[2i]·s`, each product rounded to
bf16 and the sum rounded once more (three roundings, the QSA kernel's
policy with the half-split pairing).

MoE (`Glm4MoeMoE` + `Glm4MoeTopkRouter`): the GLM-5.3 router exactly —
fp32 logits, sigmoid, `+bias` for the selection only, the uncorrected
scores gathered, normalized (`+1e-20`), × 2.5. Experts: `down(silu(gate) ·
up)` with no clamps; the reference accumulates `w_e · y_e` in fp32 over
ascending expert ids, rounds once, then adds the shared expert in bf16.
The engine's chain (fp32 fma over ascending ids, the shared row last with
weight 1, one rounding) differs by that intermediate rounding, inside the
numerics budget — the same stance as for GLM-5.3.

### 1.3 The draft layer (`model.layers.92`, `mtp.safetensors`)

```
x_q = eh_proj([enorm(embed(tok_{q+1})) | hnorm(h_q)])     # h_q: the main stack's OUTPUT hidden at q (post final norm; vLLM's convention — the pre-norm residual halved the acceptance, 2026-09-10)
→ one decoder layer (its own KV cache) → shared_head.norm → lm_head (shared)
```

`enorm`/`hnorm`/`shared_head.norm` are the two-rounding norm; `eh_proj` is
BF16 [5120, 10240]. Its experts and shared expert are BF16 in the file.

### 1.4 Tensor census (per layer)

| class | tensors | dtype / shape |
|---|---|---|
| attention | `q_proj.weight` [12288, 5120], `k_proj.weight`/`v_proj.weight` [1024, 5120], their `.bias`, `o_proj.weight` [5120, 12288], `q_norm.weight`/`k_norm.weight` [128], `k_proj.k_scale`/`v_proj.v_scale` F32 [] | BF16 |
| norms | `input_layernorm.weight`, `post_attention_layernorm.weight` [5120] | BF16 |
| dense MLP (0–2) | `gate_proj`/`up_proj` U8 [12288, 2560] + e4m3 [12288, 320] + F32 `weight_scale_2`, `input_scale`; `down_proj` U8 [5120, 6144] + e4m3 [5120, 768] | NVFP4 |
| MoE (3–91) | `gate.weight` BF16 [160, 5120], `gate.e_score_correction_bias` F32 [160]; per expert `gate_proj`/`up_proj` U8 [1536, 2560] + e4m3 [1536, 320], `down_proj` U8 [5120, 768] + e4m3 [5120, 96], each with `weight_scale_2` and `input_scale`; `shared_experts.*` the same shapes | NVFP4 |
| globals | `model.embed_tokens.weight`, `lm_head.weight` [151552, 5120], `model.norm.weight` | BF16 |
| draft (92) | `embed_tokens.weight`, `shared_head.head.weight` (duplicates), `enorm`/`hnorm`/`shared_head.norm` [5120], `eh_proj.weight` [5120, 10240], the attention set, `mlp.gate.*`, 160 × 3 BF16 expert matrices [1536, 5120] / [5120, 1536], `shared_experts.*` BF16 | BF16 |

Dequantization of an NVFP4 element (modelopt): `e2m1(code) ·
e4m3(scale) · weight_scale_2`. The engine's `GlmFp4Matrix` divides by a
global scale instead (the compressed-tensors convention), so the loader
stores `global = 1 / weight_scale_2` (one fp32 division per matrix; the
kernels' single inexact operation moves from a multiply to a divide by
the reciprocal — a relative difference of ≤ 2^-23 per output, far below
bf16 resolution).

## 2. Placement and cost by world size W ∈ {1, 2, 4}

| class | placement | per-rank bytes at W = 4 |
|---|---|---|
| attention q/k/v | 96/W query heads, 8/W kv heads: q rows + biases, k/v rows + biases per head; o_proj packed columns | 68 MB per layer (BF16) |
| q/k norms, layer norms, router | replicated | — |
| dense MLP | gate/up rows and down columns at 12288/W (fp4 column slices start on a 16-block) | 27 MB per layer |
| routed experts | every expert, sliced on the intermediate dim: 1536/W rows of gate/up, columns of down (D1 of the Qwen plan) | 3.3 MB per expert |
| shared expert | the same slice (1536/W) | 3.3 MB |
| embed | replicated (a row gather) | 1.55 GB |
| lm_head | vocab-sharded [V·r/W, V·(r+1)/W) | 0.39 GB |
| draft layer | as a main layer; enorm/hnorm/eh_proj/shared_head.norm replicated | ~0.75 GB (NVFP4 experts) |

Resident per rank at W = 4 with the draft layer: ≈ 57 GiB. KV cache: bf16
K and V for 8/W kv heads × 128 per layer, 93 layers → 95 KB per token per
rank (25 GB at 262 144 tokens).

Decode traffic per token per rank at W = 4: attention 6.26 GB (BF16),
experts 2.66 GB (8 routed + shared, 89 layers), dense 0.08, router 0.15,
lm_head 0.39, draft ≈ 0.2 — **≈ 9.7 GB, a 40 ms floor at 240 GB/s**
(GLM-5.3-Flash: 5.86 GB, 24.5 ms). Two folds per layer (the o_proj
partial and the MLP partial) make 186 collectives per token — at the
measured 47 µs per recorded collective, ~8.7 ms per step — against
GLM-5.3-Flash's 90. The BF16 attention projections are two thirds of the
bytes: quantizing them at load (FP8 block-128, the GLM-5.3 attention
format) is the single large lever after the port and is a quality
decision, recorded as follow-up work, not part of this build.

## 3. Design decisions

**D1 — One expert format, NVFP4, for every MoE layer including the draft.**
The BF16 draft experts and shared expert are requantized at load with the
NVFP4 recipe (per-tensor scale amax/(6·448), per-16 e4m3 block scales,
round-to-nearest-even codes; `loaders/nvfp4_quant.hpp`, a host encoder
pinned by a unit test against the codec) and cached in the resident image.
The draft only proposes; its numerics never reach a transcript. The
alternative — a BF16 routed-expert path — does not exist and would buy
nothing.

**D2 — The NVFP4 GEMV core takes any K that is a multiple of 32.** Its row
geometry becomes: `lanes_per_row` = the largest power of two ≤ 32 that
divides K/32, `chunks_per_lane` = (K/32)/lanes, consumed in passes of ≤ 4
chunks. A row's FMA chain is the same sequence whatever the pass split
(each lane's chunks stay in k order, the cross-lane tree is fixed), so
every existing bitwise pin (slot == grouped == sliced fold) holds, and
the K ∈ 2^n shapes compile to exactly the old geometry. Production K: 5120
(gate/up, dense gate/up), 384 / 768 / 1536 (expert down at W = 4 / 2 / 1),
3072 / 6144 / 12288 (dense down).

**D3 — The NVFP4 shared expert rides the routed launches as expert index
E.** The shared expert has the routed experts' shapes (`n_shared_experts`
× `moe_intermediate_size` = one expert), so the slot kernels take an
`E+1`-entry view table and the shared slot (j == top_k) reads entry E
through the same fp4 core — no fp8 branch. The prefill's grouped tensor-
core launch gets the shared segment as one more segment. The FP8 shared
path (GLM-5.3's composed hybrid) stays as it is.

**D4 — Paged GQA attention, 64-token blocks, bf16 K/V, no index caches.**
A `Glm4KvPool` with the Qwen pool's block management (refcounts, sharing
for the prefix cache, the partial-block copy) and only K/V per layer. The
decode kernel is split-KV flash decoding over the paged blocks — one block
per (row, split, kv head), the 12 query heads of a kv head sharing each
K/V tile, fp32 online softmax, probabilities rounded to bf16 for the V
accumulation, `l` unrounded (the QSA kernel's pin), combined by a small
merge kernel. The prefill kernel is the same computation over a chunk's
rows against the cache with the causal bound `pos_r` — v1 is the
warp-per-row form (correct, L2-resident K/V); a tensor-core tile form is
the optimization step (§5 Q7). The FP8 KV format is a follow-up (the
checkpoint's scales are 1.0; the engine's per-row fp8 latent codec applies
directly).

**D5 — No recurrent state: rollback is positional.** A rejected verify row
leaves stale K/V rows past the position, which no later row reads (row p
attends [0, p]) and which the next rows overwrite. The speculative commit
table is empty; the prefix snapshot holds the pinned block list and the
draft's hidden row at pos − 1 (the Qwen window design, not GLM-5.3's
per-position hidden cache); `session_snapshot_align()` is 1.

**D6 — Tokenizer, template, tools.** The GLM regex (pattern 0), no NFC,
`post_processor` = `Sequence[ByteLevel]` (accepted as the offsets-only
shape it is). The template needs `rstrip('\n')` / `lstrip('\n')` and reads
`enable_thinking` (an ordinary global); tool calls use the GLM markers
(`<tool_call>name<arg_key>…`), so `ToolFormat::kGlmMarkers` applies
unchanged. Goldens are generated for both.

**D7 — Text only, TP only.** No expert parallelism (the Qwen plan's D1
argument), no vision.

## 4. Layout

| path | contents |
|---|---|
| `src/models/glm4/config.{hpp,cpp}` | `Glm4TextConfig` (config.json + quantization_config), `ModelArchitecture::Glm4Moe` in `loaders/architecture` |
| `src/models/glm4/binding.{hpp,cpp}` | the expected-tensor table (main, draft, globals), validator, TP geometry check |
| `src/loaders/nvfp4_quant.hpp` | the host NVFP4 encoder (D1) |
| `src/models/glm4/loader.{hpp,cpp}` | `Glm4LayerStream`: resident/streaming builds on `weight_build.hpp`, the modelopt scale convention, the draft requant, the resident image, byte formulas |
| `src/kernels/fp4_gemv.cuh` | D2 |
| `src/kernels/glm_moe.cu`, `glm_moe_launch.hpp`, `models/glm/moe.hpp`, `moe_layer.cpp` | D3 |
| `src/kernels/glm4_attn.{cu,hpp}` | bias + per-head norm + half-split RoPE, paged K/V append, split-KV decode attention, prefill attention, the merge |
| `src/models/glm4/{kv_pool,layers,forward}.{hpp,cpp}` | the pool (D4), the attention / dense MLP / MoE layer objects, `Glm4Model` (the engine contract) |
| `src/models/glm4/attn_reference.{hpp,cpp}` | the host oracle of the attention layer |
| `tools/glm4_reference_dump.py` | the pure-python full-model reference on the synthetic fixture |
| `tests/unit/glm4_{config,binding}_test.cpp`, `tests/cuda/glm4_*` | the gates |
| `apps/glm4_load_check.cpp`, `apps/glm4_forward_check.cpp`, `dgpp_serve.cpp` (`Glm4Family`) | the apps |
| `deploy/cluster_glm47.json`, `scripts/fabric_glm4_*.sh` | the fabric rituals |

## 5. Work plan and gates

- **G1 config + binding**: parse the real config.json; the table
  reproduces the real checkpoint's tensor set exactly (names, dtypes,
  shapes; `unexpected` = 0), the TP check accepts W ∈ {1, 2, 4, 8}.
- **G2 loader**: fixture build == resident-image restore byte for byte;
  byte formula == bump usage == source plan on every layer; the draft
  requant round-trips within the fp4 codec's error; the real checkpoint
  loads one rank's slice at W = 4 (`glm4_load_check`).
- **G3 kernels**: fp4 GEMV at K ∈ {384, 768, 1536, 3072, 5120} bitwise the
  host oracle and row-count invariant; slot == grouped with the fp4 shared
  expert; attention decode/prefill against the double oracle (random
  geometry, GQA 12:1, positions across block boundaries, T up to 2048),
  decode rows == prefill rows bitwise for the same positions.
- **G4 model**: the diagnostic forward against the pure-python dump on the
  fixture (per-layer hidden states, final hidden, top-k, routes); prefill
  == forward bitwise; session decode == forward rows bitwise; the draft
  block's rows against the dump; loopback TP worlds 2 and 4 rank-identical
  and within budget of world 1; the graph engine's scalar and batched
  replays == the eager engine's transcripts (the Qwen engine test's
  shape).
- **G5 serving**: `dgpp-serve` boots the family; tokenizer + template
  goldens; the fabric world of 4 serves the real checkpoint with greedy
  transcripts, MTP `IDENTICAL`, `serve_eval.py` numbers, ms/step recorded.

Order: G1 → G2 → G3 → G4 → G5; nothing merges without its gate.

## 6. Status (2026-09-10)

Built and gated on the fixture and the real checkpoint; the fabric run
and the regression rituals follow in the record below.

| gate | evidence |
|---|---|
| G1 | `glm4_config_test`, `glm4_binding_test` (the real checkpoint's tensor set exact) |
| G2 | `glm4_loader_test` 4/4; `glm4_load_check --world 4 --rank 0 --streaming --mtp`: 93 layers 50.99 GiB + globals 1.81 = 52.79 GiB resident, the draft requantized in 17 s, 141 s total (streaming from the shards) |
| G3 | `fp4_gemv_test` (K 32 … 12288, l2_rel 0), `glm_moe_test` 21/21 (the NVFP4 shared expert bitwise on every path), `glm4_attn_test` 3/3 bitwise |
| G4 | `glm4_forward_test` vs `tools/glm4_reference_dump.py`: per-layer l2 ≤ 0.004, final hidden l2 0.0035, top-1 and routing exact, the draft's rows exact; `tools/glm4_torch_reference.py` (transformers' own layer code, real weights, first 4 layers): relative l2 0.0025 / 0.0026 / 0.0002 on layers 0–2; `glm4_decode_test` (prefill == forward bitwise, interleaved slots bitwise, chunked prefill, snapshots hot == cold, 60 steps across the block boundary vs the re-forward, per-step rolling snapshots bitwise, the speculator's transcript); `glm4_tp_test` worlds 2 and 4 (rank-identical; world 2 one near-tie flip on one row, world 4 none); `glm4_engine_test` (scalar, batched and MTP graphs == the eager engine over 60 steps; world 2 == world 1 on three prompts) |
| G5 | `dgpp-serve` family `glm4_moe`; `glm4_tokenizer_test` (55 cases), `glm4_chat_template_test` (26 cases exact, the tool round-trips and grammars); the memory plan at the fabric shape 77.4 GiB per rank for a 202,752-token context (weights 52.8, K/V 23.3) |

Fabric (2026-09-10, `scripts/fabric_glm4_serve.sh deploy/cluster_glm47.json`
and `cluster_glm47_t1.json`): boot 18–20 s from the resident image (the first
boot captures it in ~3 min), 14 graph variants warm-captured in 13 s; T=1
49.0 ms/step; MTP 60–61 ms/pass at 1.86–1.99 tokens/pass (draft acceptance
79–98 % with the post-norm hidden; 11–46 % with the pre-norm residual —
the convention that decided it), 31–33 ms/token; MTP == T=1 transcripts 4/4,
op streams identical across the ranks; eval with thinking off gsm8k 60/60,
HumanEval 39/40, extraction 30/30; the API check clean. Three fixes
surfaced by the fabric: the RoPE pairing (half-split, not interleaved —
found against transformers' own layer code on the real weights,
`tools/glm4_torch_reference.py`, after every self-written gate had passed),
the bus recorder's collective-node budget (128 -> 256; the step records
186) and the scheduler's retire-time prefix snapshot after a cut-short
multi-token step (see CHANGELOG). Full numbers: docs/measurements.md.

MTP depth 2 (2026-09-10, `deploy/cluster_glm47_d2.json`): 72 ms/pass at
2.32–2.60 tokens/pass (p1 85–96 %, p2 48–65 %) — 28.0–31.2 ms/token, 4–13 %
more tokens/s than depth 1 single-stream, +5 % at a 6,525-token context;
transcripts identical to depth 1. The row batch carries depth 2 since the
same evening (the decode rows are the recipe's shape, 4 slots x 3 rows =
12; `kBatchedDraftChain`, `session_graph_capture_draft_chain_batch`), but
loses to depth 1 at 2 and 4 live requests (37 vs 44.5, 43 vs 47.5 tok/s):
each 4-row GEMV chunk past the first re-reads the 6.3 GB of BF16 attention
projections per rank. Depth 1 stays the recipe; the lever is the
projections' bytes per pass (wider chunks or a row-independent tensor-core
kernel). Numbers: docs/measurements.md.

Design notes that changed while building: the attention's split-KV
workspace is sized to `kDecodeRows x n_split` (32 splits by default,
`DGPP_GLM4_ATTN_SPLITS`), not to the prefill rows; the python reference
mirrors the kernel's 32-token tile chain (the plain softmax form sat at
l2 ~0.005 per layer from the probabilities' rounding point alone); the
fixture's NVFP4 tensor scales are ~1e-4 (release-like weights) — a 20x
larger scale made the residual explode and the TP folds' bf16 rounding
into hard element disagreements; `GlmMoeLayer::enqueue_prefill` no longer
requires the biased-score staging (GLM-5.3 stages it, GLM-4.7 does not).
