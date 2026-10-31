# NVIDIA Nemotron-3.5-Transcribe-30B-A4B — port plan

The second audio family on the transcription stack (the transport,
admission, endpoint and chain machinery landed in `ebfd0065`; see
[audio.md](audio.md)). Nemotron 3.5 Transcribe is an offline
speech-and-audio language model: a Parallel Expert Encoder (a 32-layer
Transformer ASR encoder plus a ~99 M Sortformer speaker branch, joined by
an identity connector and a learned projection) feeding a
**Nemotron-H hybrid Mamba-2/Transformer MoE** decoder
(`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` backbone), with an MTP
head and an optional DFlash2 block drafter. Model card:
[huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B);
tech report:
[PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Transcribe-Technical-Report.pdf).
License OpenMDW-1.1.

Reference behaviour comes from NeMo Speech's `speechlm2`
(`nemo.collections.speechlm2.modules.perception.AudioPerceptionModule`,
`nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT`,
`nemo.collections.asr.modules.transformer_encoder.TransformerEncoder`,
`nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor`) and
HF `transformers` ≥ 5.5 `nemotron_h`. The vLLM plugin
`nemo_speechlm` is the serving reference for prompt shape.

## Checkpoint census (BF16 release, snapshot `9d8cd6b7`, 2026-10-30)

One `model.safetensors` (67.5 GB), 1435 tensors: 1261 BF16, 132 F32
(the Mamba `A_log`/`D`/`dt_bias`/router `e_score_correction_bias` and
norms), 1 I64, 1 U8 (CTC tokenizer blobs). Root files: `config.json`
(NeMo hydra-style), `llm_backbone/config.json` (HF `nemotron_h`),
`tokenizer.json` + `tokenizer_config.json` (fast tokenizer with the full
chat template embedded — no slow-file fallback needed),
`generation_config.json`. The cache also holds
`...-DFlash2` (1.1 GB drafter). FP8 and NVFP4 sibling repos exist
(NVFP4 is the single-Spark target).

| Prefix | Tensors | What |
|---|---|---|
| `llm.model.layers.N.*` | 476 | 52 block slots (below) |
| `llm.mtp.layers.0.*` | 24 | one MTP layer: attention + MoE + `eh_proj`, `enorm`, `hnorm`, `final_layernorm` |
| `llm.model.embed_tokens.weight` | 1 | BF16 [131072, 2688] |
| `llm.lm_head.weight` | 1 | BF16 [131072, 2688] (untied) |
| `llm.model.norm.weight` | 1 | final RMSNorm [2688] |
| `perception.preprocessor.*` | 2 | mel filterbank F32-valued BF16 [1, 128, 257] + Hann window [400] |
| `perception.encoder.asr_encoder.*` | ~420 | 32-layer Transformer encoder, 631 M params |
| `perception.encoder.diarization_model.*` | ~380 | Sortformer speaker branch, 99 M params |
| `perception.encoder.asr_norm` / `.diar_norm` | 4 | LayerNorm(1280) after each expert |
| `perception.proj.{weight,bias}` | 2 | connector linear 1280 → 2688 |
| `ctc_timestamp.decoder.*` | 40 | CTC alignment head (23862 classes) — **deferred** |

LLM total 32.91 B params (incl. MTP); ~4.3 B active including the
encoder, per the card.

### The LM: Nemotron-H slots

`layers_block_type` (52 entries): 23 `mamba`, 6 `attention`, 23 `moe`.
Slots are *not* full transformer layers: a `[mamba]` slot is the Mamba-2
mixer alone, `[moe]` is the MoE MLP alone, and the repeating
`mamba mamba mamba attention moe` groups make attention layers the
(mamba, mamba, attention, moe) quartet. Every slot carries its own
pre-norm (`layers.N.norm.weight`, 52 total), RMSNorm eps 1e-5.

Mamba-2 mixer (23 slots), from weight shapes:
- `mixer.in_proj.weight` [10304, 2688] = x 4096 + z 4096 + B 1024 + C 1024 + dt 64
  → `d_inner` = 4096 = n_heads 64 × head_dim 64 (the config's
  `expand: 2` is not the in_proj split; trust the shapes);
  n_groups 8, d_state 128 (`ssm_state_size`), conv_kernel 4 with bias,
  silu activation, `A_log`/`D`/`dt_bias` F32 [64],
  `time_step_limit` [0.001, 0.1], no projection bias.
- `mixer.norm.weight` [4096]: the pre-out_proj gated norm (causal-conv1d
  → SSM → SiLU(z)·norm → out_proj walk; the oracle pinned it 2026-10-31:
  the SiLU gate acts before the group variance, groups of 512).
- `mixer.out_proj.weight` [2688, 4096].

Attention mixer (6 slots: block indices 5, 12, 19, 26, 33, 42):
`q_proj` [4096, 2688] (32 heads × 128), `k_proj`/`v_proj` [256, 2688]
(2 kv heads), `o_proj` [2688, 4096]; no biases, **no q/k norm** (the LM
attention is plain — unlike the perception encoder), and **no position
embeddings at all**: the Nemotron-H paper (arXiv:2504.03624 §2.1) says
"We do not use any position embeddings" for the hybrid — the attention
slots are position-free GQA (positional information arrives only
through the surrounding Mamba states). The `rope_theta` /
`partial_rotary_factor` keys in `llm_backbone/config.json` are
vestigial (Nemotron-T, the pure-Transformer comparison model, used
RoPE; Nemotron-H does not), the release ships no rotary tensors, and
both the HF `nemotron_h` implementation and Automodel never call
`apply_rotary_pos_emb` on these q/k — the oracle gate confirmed the
walk matches HF fp32-for-fp32 only without rotation (2026-10-31).
Positions still run to 262144 (the 65 536 training context is the
contract; the pool may sit lower).

MoE MLP (23 slots): router `mixer.gate.weight` [128, 2688] plus F32
`e_score_correction_bias` [128] — sigmoid scores with a learned
correction bias, top-6 of 128, `norm_topk_prob` true, `routed_scaling_factor`
2.5, one shared expert (up [3712, 2688], down [2688, 3712]). The group
policy is settled (oracle, 2026-10-31): the router reads `n_group` = 1
with `topk_group` = 1 — the group-limiting path is a **no-op** for this
release (every expert pools; the top-2-per-group scores select the one
group); the `n_groups: 8` key is the *Mamba* group count, a name
collision, not routing. The bias corrects the **selection only**: top-k
is chosen on `sigmoid(logits) + bias`, the mixed weights are the
unbiased sigmoids, renormalized (÷ Σ + 1e-20) and scaled by 2.5.
Routed experts
ship **stacked**: `experts.gate_and_up_projs` [128, 2688, 1856] and
`experts.down_projs` [128, 1856, 2688] — one fused matrix per expert,
relu² activation (`mlp_hidden_act: relu2`), moe_intermediate 1856.
The dense `intermediate_size` 1856 applies only to slots with a dense
MLP — there are none: all 23 MLP slots are MoE, so the LM has no dense
layers.

MTP: one trained layer (`attention` + `moe`, `mtp_layers_block_type`)
with `eh_proj` (2·2688 → 2688), `enorm`, `hnorm`, `final_layernorm` —
Megatron-style: the draft step consumes the last token's embedding and
hidden state. The card: two next-token prediction depths via the
repeated layer, up to 8 speculative tokens; the existing MTP/drafter
framework already carries depth-scheduled verify. The DFlash2 draft repo
(`num_hidden_layers: 6`, `block_size: 16`, `num_target_layers: 52`,
non-causal attention) is a block drafter over the same 2688 hidden —
the `dflash2` path of the Qwen3.8-27B templates is the template.

### The perception module

- **Preprocessor** (`AudioToMelSpectrogramPreprocessor`, weights frozen
  and shipped): 16 kHz, window 0.025 s (400), stride 0.01 s (160),
  `n_fft` 512 → 257 bins, hann — the SHIPPED `featurizer.window` [400]
  is the truth (a symmetric Hann, `periodic_window` default False), and
  the shipped `featurizer.fb` [1, 128, 257] is the filterbank (use the
  tensor, do not regenerate Slaney — it need not match librosa). Framing
  is torch-stft center-padding (`exact_pad` unset): T = (N + 112) / 160
  + 1 frames. Chain per features.py (2026-10-31 source read): dither is
  TRAINING-ONLY (the 1e-05 config value never fires at eval — the oracle
  pins this), then **preemphasis 0.97 on the unpadded signal — the
  release's config omits `preemph`, so the module default APPLIES it**
  (this was a census miss: no `preemph` key ≠ no preemphasis);
  magnitude = sqrt(re² + im²) then pow 2 (`mag_power` 2.0 default — the
  square of the rounded sqrt); fb matmul; `log: true` with the DEFAULT
  guard: `log(S + 2^-24)` (log_zero_guard_type 'add', value 2^-24 —
  NOT clamp+1e-10; the config omits both keys); `frame_splicing` 1,
  `pad_to` 0, `normalize` null.
- **ASR branch**: `pre_encode.proj` stacks 8 frames (128 × 8 → 1280;
  `subsampling_factor` 8 ⇒ 12.5 encoder tokens/s, matching the card's
  ~80 ms/token), 32 × pre-norm Transformer blocks: `w_qkv` [3840, 1280],
  10 heads × 128, **q/k LayerNorms after the split** (qk_norm true),
  RoPE on half the head dims (rotary_fraction 0.5, base 10000), GELU FF
  5120, `embed_norm` + per-block norms + `final_norm` — the NeMo
  `TransformerEncoder` (attn_mode full, no conv subsampling). Bidirectional.
- **Speaker branch**: Nemotron-3-Diarization / Sortformer (~99 M): a
  streaming transformer with 2×16-head blocks (`ff_expansion` 0.5),
  RoPE half, upsampling conv_transpose ×2, `residual_scale` 0.1 — a
  34 M-frame-window model. **This branch is not optional**: the parallel
  expert encoder normalizes each branch (`asr_norm`, `diar_norm`) and
  the LM is trained on their join through the connector.
- **Connector**: IdentityConnector(d_model 1280) then
  `perception.proj` [2688, 1280] — how the two branches combine into the
  LM rows (concat-in-time with speaker windows, interleave, or sum) is
  decided in `ParallelExpertEncoderPT.forward`; read it from the NeMo
  source and pin it in the oracle before any kernel work.
- Encoder chunking: 30 s post-feature chunks
  (`encoder_chunk_size_seconds`), full attention inside a chunk;
  `packed_encoder_sequences` is a training detail.
- `audio_token_estimator`: hop 160, `stft_pad_amount` 256, stack 8 ⇒
  n_tokens = ceil(frames/8) — the admission-span row count formula.

### Prompt contract

`prompt_format: nemotron3p5`, `audio_locator_tag: <|audio|>`: the chat
template (embedded in `tokenizer_config.json`, complete with the tools
and thinking macros) places **one `<|audio|>` locator per clip after the
text** — the vLLM comment in the template even documents the reorder.
The frontend's job (the `audio_frontend.cpp` pattern): let the template
emit the locator, then expand it to the clip's encoder rows + the pad
row the model expects — exact marker/pad semantics come from the NeMo
`speechlm2` processor; the Qwen3-ASR `...`/pad/`...` triple is NOT
reused (this is one locator token, not a delimiter pair).
`<asr_text>`/speaker/event tags: `<spk:N>` tags are plain vocabulary
(the card's prompt guide asks for them); the reasoning macros honour
`enable_thinking` (transcription runs it off; the repo template
interpreter already handles both shapes).

## Memory plan (one Spark, world 1)

NVFP4 target: experts+MLP ≈ 32.9 B → ~17 GB codes+scales, BF16
embeddings/head 1.4 GB, norms/state ≈ negligible; perception BF16
(631 M + 99 M + connector) ≈ 1.5 GB; SSM state per request:
23 layers × (dstate 128 × nheads 64 + conv 4096×3) fp32 ≈ 23 × 0.35 MB
≈ 8 MB; KV: 6 attention layers × 2 kv × 128 × 2 B ≈ 3 KiB/token.
A 262 K pool ≈ 0.8 GB — everything but the weights is small; a 65 536
pool + a 4 GiB prefix arena + headroom fits one 121.6 GiB Spark with
room for the BF16 image cache of the resident loader. The BF16 release
(63 GB weights) also fits world 1 resident but leaves little for
concurrency; ship NVFP4 (or FP8) templates first, BF16 as the exact
fallback.

## Milestones

1. **Detection + config + binding** (host): `NemotronTranscribe`
   architecture, `config.json` parsers for both the NeMo root config and
   `llm_backbone`, the binding table (slot kinds, stacked experts, the
   MTP layer, the frozen preprocessor buffers, connector/diar names).
   Host tests against the landed checkpoint.
2. **Text oracle**: a host fp32 walk of embed → 52 slots → norm → head
   (Mamba-2 naive recurrence is fine at oracle sizes), pinned to a torch
   fp32/BF16 dump (`tools/` golden builder, the ASR tower-goldens
   pattern). Quality check: greedy prompts from the README's own
   examples reproduce.
3. **Mamba-2 kernels**: chunked-scan prefill + single-row decode device
   paths in `src/kernels/`, gated against the oracle within BF16
   tolerance; SSM state carried beside the linear-attention state
   machinery (`dsa_state`/`csa2_state` precedent) with snapshot/resume
   semantics.
4. **MoE + full-text serve**: wire the slot walk over the existing
   MoE executor + graph decode; register the family; text-only serving
   passes the chat/eval gates (the PII/text subsets are out of scope —
   plain instruction following suffices).
5. **Perception oracle**: frontend (pinned to the shipped
   fb/window tensors, dither off) + ASR transformer + Sortformer +
   connector, host fp32, pinned to a NeMo dump of the full
   `AudioPerceptionModule` on a test clip. This milestone answers the
   open questions (dither at eval, expert join geometry, row counts).
6. **Device encoder + splice**: BF16 kernels for both branches gated on
   the oracle; `session_prefill_audios` splices rows at the admission
   span (the seam exists from the audio-stack commit); the frontend
   rewrites the `<|audio|>` locator; `prefill_audio_token_limit()` from
   the prefill window.
7. **Deploy + spec**: `cluster_nemotron-3.5-transcribe-30b-a4b_nvfp4_w1`
   template (portability/README/catalogue registrations), MTP depth-8
   verify (the repeated layer) with DFlash2 as the template drafter
   (the existing dflash2 path, 6 layers, block 16), then the
   `/v1/audio/transcriptions` gates (chains, refusals) and a LibriSpeech
   test-clean WER tool mirroring `tools/qwen_asr_wer.py` from the ASR
   branch — target the card's 1.09/2.60 test-clean/other.
8. **Measured rows** per [benchmarks.md](benchmarks.md) on an idle
   fabric, revision + binary hash recorded, production restored.

## Open questions (resolve in milestone 5, from NeMo Speech source)

Resolved 2026-10-30 from `nemo/collections/asr/modules/parallel_expert_encoder.py`
(main):

- **Expert join** — `ParallelExpertEncoder._fuse_diar_and_asr`:
  `fused(T,D) = asr_norm(states) + spk_kernel_scale *
  matmul(diar_norm(speaker_features), diar_kernel)`, all before the
  `perception.proj` 1280→2688 connector. `speaker_features` is the
  per-frame speaker activity aligned/pad-repeated to the ASR grid,
  mode-gated by `speaker_feature_mode` (continuous sigmoid vs
  thresholded — read the value from `config.json`
  `pe_encoder_config`; `spk_kernel_scale` is 1.0), `diar_norm` is a
  LayerNorm(n_spk) over the activity vector, and `diar_kernel` is a
  **built-in** sinusoidal (n_spk × 1280) table (not a checkpoint
  tensor). The diarization branch runs Sortformer streaming windows
  (fifo 40, spkcache 188, update period 300, chunk 128 ±50 context at
  the encoder grid); both branches consume the same mels and each
  reapplies `normalize_batch` internally (asr/diar normalize types from
  the bundle cfg). Subsampling factor must match (8).
- Frame grid: input 10 ms frames, feature stacking ×8 after the mel
  (pre_encode = FeatureStacking, one Linear 1024→1280 in
  `pre_encode.proj.weight`), so encoder tokens are 80 ms — the card's
  ~80 ms/token holds and the `audio_token_estimator` formula is
  ceil(frames/8) with hop 160, center pad 256.
- Chunking: `chunk_size_seconds` 30 → the ASR branch runs full
  attention over independent 375-token chunks (post-stacking split,
  `packed_encoder_sequences` a training detail).

Still open:

- MTP inference mechanics: how the repeated layer consumes
  (embedding, hidden), and the depth-2 training → 8-token drafting
  mapping — mirror the vLLM plugin (`nemo_speechlm`).
- The `<|audio|>` id and whether the LM input inserts pad rows around
  the locator (the HF processor's `apply_transcription_request` is the
  ground truth).
- NVFP4 export layout for the stacked expert tensors: confirm packed
  pairs + block scales per expert in the stacked [E] tensors.

## Status

- 2026-10-30: checkpoint surveyed (BF16 67.5 GB single shard +
  DFlash2, cached); census and slot geometry above; audio transport
  stack landed on this branch in `ebfd0065`; plan written.
- 2026-10-30: milestone 1a — `NemotronTranscribe` architecture
  detection (`Nemotron3_5TranscribeForConditionalGeneration` /
  `nemo_speechlm`), the host config parser
  (`src/models/nemotron_transcribe/config.{hpp,cpp}`: the Nemotron-H
  slot list, the Mamba/Attention/MoE geometry, the perception contract
  with the thresholded 0.5 fusion, the Sortformer window, and the
  audio-token estimator), and `nemotron_transcribe_config_*` unit
  tests (fixture values transcribed from the release, plus the
  landed-checkpoint case) pass against the cached BF16 config.
  Next: the binding table + the torch oracle walk (milestones 1b/2).
- 2026-10-31: milestone 1b — the expected-tensor binding table
  (`src/models/nemotron_transcribe/binding.{hpp,cpp}`, matching the
  cached release tensor for tensor; the Sortformer branch and the
  deferred CTC head are named-but-ignored groups, as are
  TransformerEngine's empty `*_extra_state`).
- 2026-10-31: milestone 2 (LM side) — the host fp32 oracle
  (`src/models/nemotron_transcribe/reference.{hpp,cpp}`): gemv/RMSNorm,
  the causal depthwise conv + SiLU, the selective scan (sequential
  form, dt clamped at `time_step_min` after softplus, the D skip on
  the pre-dt x), the grouped gated norm (SiLU gate BEFORE the group
  variance — `norm_before_gate` false, groups of d_inner/n_groups =
  512), the sigmoid + bias-corrected top-6 router (selection on the
  biased scores, weights from the unbiased sigmoids, renormalized and
  scaled by 2.5), the stacked relu² experts in checkpoint bmm layout
  plus the shared pair, and the position-free GQA attention — the
  whole 52-slot-style walk ending at norm_f and the head. Pinned by
  `tests/python/nemotron_transcribe_ref_oracle.py --emit` (a tiny
  seed-filled nemotron_h walked on CPU through the pure-torch
  reference: slot residuals, mixer in/outs, the router triple, the
  logits, and the four primitives standalone) in
  `tests/data/nemotron_transcribe_reference_oracle.json`; the
  `nemotron_oracle_*` unit tests replay it to fp32 budgets (5e-4 rel).
  The walk matched only with **no RoPE in the LM attention** — the
  census correction above (Nemotron-H is a NoPE hybrid; the config's
  rope keys are vestigial). Text goldens from the real 33 B checkpoint
  (`tools/nemotron_transcribe_text_goldens.py`) still await a free
  device; they replay the same walk at release geometry and dtype.
- 2026-10-31: milestone 3 (build) — the Mamba-2 device path:
  `src/kernels/mamba2.{hpp,cu}` (causal depthwise conv + SiLU prefill and
  the stateful decode step, the selective scan as one block per head with
  the fp32 state resident in shared memory — dt bias/softplus/clamp inside,
  the D skip on the pre-dt x, fixed-order reductions — plus the single-row
  decode form, and the grouped gated norm), and the state pool
  `src/models/nemotron_transcribe/state.{hpp,cpp}` (per-request contiguous
  recurrent + conv regions on the arena, KdaStatePool's one-copy
  snapshot/resume). `tests/cuda/mamba2_test.cu` gates conv/norm/scan
  against the host oracle (bf16 ulp budgets, rel budgets on the F32 state)
  and requires the step/resume composition to be BITWISE the fused
  prefill. Not yet run: the live serve holds the device to OOM (even the
  pre-existing csa2/arena device tests are starving), so the GPU gate is
  open until the fabric is idle; the build, registration and the host
  oracle suite are green. The oracle header split (minijson-free
  reference.hpp with the config forward-declared) keeps nvcc TUs compiling.
- 2026-10-31: milestone 5 (part A — everything but Sortformer's own
  branch) — the host fp32 perception oracle
  `src/models/nemotron_transcribe/perception.{hpp,cpp}`: the frozen mel
  frontend (its own radix-2 rfft512, NeMo's chain order pinned),
  FeatureStacking + the bias-free pre-encode, embed/final LayerNorms, the
  pre-norm encoder block (fused w_qkv → per-head q/k LayerNorm → half
  RoPE → bidirectional softmax → biased out_proj → GELU-erf FFN), the
  built-in sinusoidal speaker kernel, the thresholded fusion (STRICT >,
  scale on the infusion only) and the connector — the Sortformer branch
  itself (5b) enters as the activity tensor, exactly as
  `ParallelExpertEncoder._fuse_diar_and_asr` accepts external targets.
  The NeMo toolkit is not installable on this box right now (12 GB disk
  headroom, multi-GB CUDA wheels), so the gate fixture
  (`tests/python/nemotron_transcribe_perception_oracle.py` →
  `tests/data/nemotron_transcribe_perception_oracle.json`) is a
  CPU-torch MIRROR transcribed from the five NeMo sources at main
  (fetched 2026-10-31, cited in the plan census); the mirror itself
  proved worth having — it shipped a rope bug (cos=sin=emb) that the
  C++ correctly refused to match, and the fixture now regenerates
  against the real NeMo classes when the toolkit lands, same file
  shape. The census gained three facts from the source read: preemphasis
  0.97 DOES apply (the config omits the key, the default fires), the log
  guard is `log(S + 2^-24)` (not clamp+1e-10), and dither is
  training-only — eval is deterministic. Frames: T = (N + 112) / 160 + 1
  (center-stft). `nemotron_perc_oracle_*` unit tests green; the
  hand-computed corners pin the delta-FFT, the kernel's sin-on-even
  ladder, the LN, and the strict-> threshold. Still open: the real-NeMo
  re-emit, and Sortformer's internals (5b).
