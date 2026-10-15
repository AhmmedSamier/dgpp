# Zeta-2.1 (Llama dense GQA) plan

`zed-industries/zeta-2.1` is a dense BF16 Llama fine-tune of
`ByteDance-Seed/Seed-Coder-8B-Base` for code edit prediction (FIM-style
`Completions`, not chat). This is the first plain-GQA dense family in the
engine: no MoE, no indexer/compressor, no MTP/draft layer, no sliding
window, no vision.

## 1. Checkpoint census (config.json @ 9a83b32 + safetensors index)

* `architectures: ["LlamaForCausalLM"]`, `model_type: "llama"`.
* 32 layers, `hidden 4096`, GQA `32Q / 8KV`, `head_dim 128` (explicit),
  SwiGLU `intermediate 14336`, RMSNorm `eps 1e-6`, full RoPE
  (`theta 500000`, no `rope_scaling`), `max_pos 32768`,
  `vocab 155136`, `tie_word_embeddings: false`, `attention_bias: false`,
  `mlp_bias: false`, `torch_dtype bfloat16`.
* Per-layer tensors (9): `input_layernorm`, `self_attn.{q,k,v,o}_proj`
  (q `[4096,4096]`, k/v `[1024,4096]`, o `[4096,4096]`), 
  `post_attention_layernorm`, `mlp.{gate,up,down}_proj`
  (gate/up `[14336,4096]`, down `[4096,14336]`). All BF16.
* Globals (3): `model.embed_tokens`, `model.norm`, `lm_head`
  (`[155136,4096]`, untied).
* Total 291 tensors across 4 shards; ~8.25B params, ~16.5 GB BF16.
* No `quantization_config`, no draft layer, no `chat_template.jinja`.
  Tokenizer: `tokenizer.json` + FIM specials (`fim-prefix/suffix/middle`,
  ids 124-126); `BOS 0 / PAD 1 / EOS 2`.

## 2. Placement (TP=1 first, 2/4/8 clean)

* Attention: q heads `32/W`, kv heads `8/W` (both divide W at 1/2/4/8);
  o_proj packed columns. Norms replicated.
* MLP: gate/up rows, down columns at `14336/W` (`1792` at W=8).
* Embed replicated (row gather); final norm replicated; head
  vocab-sharded (`155136/W`).
* KV cache: BF16 paged `[slots, kv_heads * 128]` per layer — `128 KiB`
  per token total (32 layers x 8 x 128 x 2 x 2B). 32K ctx ~= 4 GiB.
* Attention math reuses the `glm4_attn` contract (`glm4_qkv_finish` with
  null bias/norm, `rotary_dim = 128`; paged GQA partial/combine): Llama
  is the no-bias / no-qk-norm / full-RoPE point of that kernel family.
  New Llama layer/model objects still to come (G4/G5); this slice lands
  detection + config + binding + tests only.

## 3. Serving notes

* Completions route (FIM prompt passthrough); no Jinja template ships.
* Tokenizer pre-tokenizer pattern vs `src/text/tokenizer.cpp` pinned
  regexes: must be inspected from the landed `tokenizer.json` (LFS) —
  a new pattern + goldens if it differs.
* Plain decode only (no MTP in ckpt); DFlash-style external draft is a
  later option, not this slice.
* Templates: `deploy/cluster_zeta-2.1_bf16_w1.example.json` after the
  model lands.

## 4. Port record

* 2026-10-14: branch `zeta-llama-support` from `origin/master` (347c386f).
  Slice 1: `ModelArchitecture::Llama` detection, `src/models/llama/`
  config + binding, `dgpp_models_llama` lib, `llama_config_test` /
  `llama_binding_test` gates.
* Slice 2: `src/models/llama/loader.{hpp,cpp}` (`ResidentLayerStream`
  family: BF16 row/column slices, packed o/down columns, replicated
  norms/embed, vocab-sharded head, kv-pairing geometry shared with GLM-4),
  `dgpp_models_llama_loader` lib, `tests/cuda/llama_fixture.hpp` +
  `llama_loader_test` (byte-exact slices at worlds 1/2/4, globals,
  resident image round trip — 3/3 pass on GB10).
* Slice 3: `src/models/llama/{kv_pool,layers,forward,attn_reference}.*`
  (`SessionModel` walk: GQA + full RoPE over the `glm4_attn` kernels,
  BF16 SwiGLU MLP, paged KV, dense TP folds; no MoE/MTP/prefetch/bf12),
  `dgpp_models_llama_layers` lib, `tests/cuda/llama_forward_test.cpp`
  (fixture/smoke/host-parity: l2 ~0.002, top-1 48/48). Shared change:
  `glm4_qkv_finish` applies RoPE independent of the head norms (v never
  rotates; GLM-4 callers pass rotary 0 without norms, unchanged —
  `glm4_attn/loader/forward` gates green incl. the python dump parity).
* Slice 4: serve wiring + tokenizer + preflight. `LlamaFamily` /
  `LlamaFrontend` (FIM passthrough; chat refuses) in the serve binary;
  the zeta Split pattern (single numbers, no punct newline tail) with
  `tests/data/zeta_tokenizer_goldens.jsonl` (71 HF-differential cases);
  `cluster_doctor` template exemption for the Llama family;
  `deploy/cluster_zeta-2.1_bf16_w1` (plain graph decode — the only world,
  with the portability carve-out). Validated live on one Spark
  (2026-10-14): 22 s boot, 15.4 GiB resident, the checkpoint's
  `sample.prompt` completing byte-exact to `sample.output` (158 tokens),
  repeat-deterministic. Serve binary also got its Llama case in
  `serving_head_check` (resident load + session_prefill across tail lengths,
  validated against the real checkpoint's sample.prompt).
* Slice 5: TP + transcript. `tests/cuda/llama_tp_test.cpp` (fabric-free
  world-2 fold vs world-1 oracle: cross-rank bitwise residuals/reads, head
  slices and layers within the bf16 budget, top-1 exact);
  `tests/python/zeta_transcript_test.py` (boots the real server, asserts
  `sample.prompt` completes byte-exact to `sample.output` plus a
  deterministic repeat; `gpu;python;checkpoint`, skips without binary or
  cache). Throughput at 4 slots: 15.8 tok/s (conc 1), 25.3 (conc 2),
  35.9 agg (conc 4).
* Slice 6: bf12 companions (2026-10-15). The loader was already
  pack-ready (slice 2 grants); the model packs all 7 projections/layer +
  head in `graph_prepare` with the `set_bf12_wide` walk toggles and the
  plan entries (companions + expansion scratch − released bf16). Probe
  then live: 225/225 matrices, 14.18 → 10.66 GiB, 4.8 s pack at boot.
  Bitwise-exact: `zeta_transcript_bf12_test` passes (same file,
  `DGPP_BF12=on`). Measured: 20.2 tok/s single, 132.7 agg at conc 8
  (the knee — conc 16 holds agg, halves per-req). Template ships
  `bf16_weights: bf12` at 8 slots.
  Next: multi-node TP shakeout (loader geometry covers worlds 2/4/8),
  DESIGN/PLAN status.
