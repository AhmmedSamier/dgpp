# Nemotron-3.5-Transcribe — Milestone 4: Device Forward & Serving

## Goal
`dgpp-serve --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` serves text
greedy+sampled chat completions on the GB10, with the CUDA device forward
reproducing the host fp32 oracle (`reference.cpp`) bitwise.

## What exists (milestones 1–3, all committed/pushed)
- Checkpoint: `model.safetensors` (67.5 GB), snapshot `9d8cd6b7`, tokenizer +
  `config.json`/`llm_backbone/config.json`/`generation_config.json` — all local.
- Config parser: `src/models/nemotron_transcribe/config.*` — root config +
  inline `llm_config`, cross-checked.
- Expected-tensor binding table: `binding.*` — every LM/perception tensor
  name, dtype, shape; validation report.
- Host fp32 oracle: `reference.*` — full 52-slot hybrid walk (Mamba-2 conv+
  scan + gated norm, GQA attention, relu2 MoE, norm_f, lm_head), pinned to
  HF. Position-free attention (no RoPE on the LM).
- Mamba-2 device kernels: `src/kernels/mamba2.cu` — conv+SiLU, selective scan
  (prefill + decode row), grouped gated RMSNorm — gated bitwise vs oracle.
- Mamba state pool: `state.*` — per-request recurrent state + conv history,
  snapshot/resume.
- Perception + Sortformer oracles: `perception.*` (milestone 5a/5b).
- MTP goldens: `tools/nemotron_transcribe_mtp_draft.py` + fixture (11/32
  accept, fixed rmsnorm + `hs[-1]` trunk capture).
- `NemotronTranscribe` registered in `src/loaders/architecture.hpp`.

## What's missing (milestone 4)
| Piece | File | Status |
|-------|------|--------|
| Weight loader | `loader.cpp/hpp` | absent |
| Model class (SessionModel) | `model.cpp/hpp` | absent |
| CUDA forward walk | `forward.cpp/hpp` | absent |
| dgpp-serve wiring | `apps/dgpp_serve.cpp` | absent (`NemotronTranscribe` case) |
| CMake registration | `CMakeLists.txt` | absent |

## Architecture decision
Follow the `glm_dsa` family pattern (the closest hybrid-MoE analog):
- **Loader**: `ResidentLayerStream`-based (`src/loaders/resident_stream.hpp`),
  family provides `Config` = `NemotronTranscribeConfig`, `Expected` =
  `NemotronExpectedTensor`, `LayerResident` / `GlobalsResident` views,
  `Builder` that maps the binding table to device pointers.
- **Model**: `SessionModel<NemotronModel>` — derived supplies `run_rows` (the
  layer walk), `reset_slot_state`, `spec_segments`, snapshot/read state,
  `graph_prepare`, `mtp_run_rows`. Base handles sessions, prefix cache,
  blocks, verify/rollback, graph capture, MTP.
- **Forward kernels**: new `forward.cpp` — embed lookup, per-slot RMSNorm,
  GQA attention (position-free, no RoPE), relu2 MoE (sigmoid router, stacked
  experts, shared expert), residual adds, norm_f + lm_head. Reuse the existing
  `mamba2.cu` kernels for Mamba-2 slots.
- **Precision**: BF16 activations/weights, fp32 accumulation (matching
  `reference.cpp`, `-ffp-contract=off`).

## Geometry (pinned from checkpoint, snapshot `9d8cd6b7`)
- Hidden H = 2688, vocab = 131072, 52 slots
- Slots: 23 Mamba, 6 Attention (indices 5,12,19,26,33,42), 23 MoE
- Mamba-2: 64 heads × 64 head_dim (d_inner 4096), n_groups 8, state 128,
  conv kernel 4, in_proj 10304 = x4096+z4096+B1024+C1024+dt64
- Attention: 32 q heads × 128, 2 kv heads, no bias, no RoPE, no q/k norm
- MoE: 128 routed experts, top-6, intermediate 1856, shared 3712,
  routed_scaling_factor 2.5, relu2, sigmoid router + e_score_correction_bias
- MTP: enorm/hnorm/eh_proj[2688,5376] on slot 0, final_layernorm on last

## Walk order (per slot, residual after each mixer)
1. `embed(tokens)` → resid [T, H]
2. For each slot i: `norm_i(resid)` → mixer_i(normed) → `resid += mixer_out`
3. `norm_f(resid)` → `lm_head` → logits [T, vocab]

## MTP (draft block, follows glm_dsa convention)
- `fused = eh_proj(cat[enorm(embed(tok_{q+1})) ‖ hnorm(h_q)])` — h_q is the
  main stack's POST-final-norm hidden at q (verified vs goldens fixture)
- attention leg → MoE leg → final_layernorm → lm_head → draft logits

## Acceptance test
Run `dgpp-serve` on the GB10, send the 4 text-goldens prompts, compare the
first generated token logits/top-5 against
`tests/data/nemotron_transcribe_text_goldens.json` (byte-identical top-5 ids
and logits). MTP accept should match the torch goldens (11/32 across the 4
cases).