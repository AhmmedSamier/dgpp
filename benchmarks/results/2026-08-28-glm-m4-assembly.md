# GLM M4 assembly — 2026-08-28 (start of milestone)

M4 deliverable 1 (config-driven adapter) and deliverable 2's validation half
(binding table; the resident loader follows next): the text configuration
parser and the expected-tensor binding table, plus the offline
`glm_bind_check` tool. Records below are the first M4 evidence.

## Environment

Same box as all prior milestones (GB10, single node). Checkpoint:
`unsloth/GLM-5.3-Flash-FP8@a160e2291674d9e3e92e98fd82faa2544a2867a3`
(62 shards, 76,108 tensors, 328.33 GB).

## Config parsing (`src/models/glm_config.{hpp,cpp}`)

Parses `config.json` `text_config` into `GlmTextConfig` and cross-validates
every field the assembly consumes. The redundant layer-class encodings must
agree: `layer_types` vs `linear_attn_config.{kda_layers,full_attn_layers}`,
`mlp_layer_types` vs `first_k_dense_replace`, all 45 `indexer_types` "full".
Unsupported values are rejected at load, never silently defaulted:
non-mHC checkpoints, rope on the MLA path, non-sigmoid scoring, non-noaux_tc
top-k, grouped routing, multiple shared experts, tied embeddings, >1 MTP
draft layer, non-F32 router math, non-silu activation, kpool outside
{2,4,8}. The geometry structs it produces (`kda_config()`, `dsa_config()`)
carry the trained values — the defaults in the geometry headers are
explicitly not trusted (per their own comments).

Unit tests: `tests/unit/glm_config_test.cpp` — 5 tests covering every class
of rejection and the trained-value propagation.

## Binding table and validation (`src/models/glm_binding.{hpp,cpp}`)

`glm_expected_text_tensors()` enumerates every text-model tensor (name,
dtype, exact shape) from the config: 45 main layers (mHC + norms + KDA or
DSA/MLA + indexer + dense MLP or router/shared/288 routed experts), the MTP
draft layer at index 45 (DSA + MoE layout, no mHC, plus
enorm/hnorm/eh_proj/shared_head.norm), and embed/lm_head/final-norm
globals. `glm_scale_shape()` is the single source of the 128×128 block-grid
formula shared by table generation, validation, and (next) the loader.

The unit tests (`tests/unit/glm_binding_test.cpp`) caught one real bug
before it reached the checkpoint: `expect_dense_mlp` initially emitted
`layers.0.gate_proj.weight`, missing the `mlp.` segment — the sort of
name-drift a 306 GB run reports as an opaque missing-tensor error and a
synthetic fixture reports in 40 ms.

## Real-checkpoint bind check (M4 exit criterion 2)

```
$ glm_bind_check --config <snapshot>/config.json --checkpoint-dir <snapshot>
config: 45 layers (34 KDA + 11 DSA), vocab 154880, mhc mult 4, mtp present
checkpoint: 62 shards, 76108 tensors in headers
binding: expected 75761 | matched 75761 (missing 0, dtype 0, shape 0) | vision 347 | unexpected 0
quantized matrices: 37338 | scales bound: 37338 | scales bad: 0
binding OK: every quantized matrix has a validated scale
```

- All **37,338 quantized matrices bind to validated scale tensors** — the
  exact count `docs/checkpoint_budget.md` derives independently (that tool
  walks headers in Python; this one walks them in C++ through
  `SafetensorsFile::for_each`, no shardspec, no payload reads).
- Expected total 75,761 = 76,108 tensors − 347 vision tensors; zero
  unexpected, so the MTP layer layout transcribed from one layer's headers
  (DSA + MoE, no mHC) holds for the whole checkpoint.
- Wall time 0.14 s (header parse only; 62 safetensors JSON headers).

The table is also the input to the next chunk: the resident loader walks it
role-by-role to copy compressed E4M3+scale payloads into device memory with
no persistent BF16 expansion (DESIGN §4).

## Verification

- `unit` target: 34 tests, 0 failed (5 new config, 6 new binding).
- `ci-local.sh`: 10/10 (CUDA suites unaffected; the new library is
  host-only and links into `unit_tests` and `glm_bind_check`).
