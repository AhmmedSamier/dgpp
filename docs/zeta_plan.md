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
  resident image round trip — 3/3 pass on GB10). Next: layer/model over
  `SessionModel` + `glm4_attn` kernels, serve wiring, tokenizer pattern,
  deploy template, transcript battery.
