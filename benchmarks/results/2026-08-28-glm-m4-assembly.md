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

## Streaming resident loader (deliverable 2, later same day)

`src/models/glm_loader.{hpp,cpp}` + `src/kernels/fp8_dequant.{hpp,cu}`:
`GlmLayerStream` opens the checkpoint (mmap, header-indexed, full binding
validated at construction), then loads one layer at a time into managed
device memory in the exact layouts the M2/M3 kernels consume:

- KDA: merged in_proj rows [f_a|g_a|q|k|v|b] and merged conv channels
  (byte-exact concatenations of the six separate checkpoint tensors);
- DSA: fused qkv_a, plus q_b/o_proj dequantized through the block-scale
  kernel (decode × scale_inv, one BF16 round — bitwise vs host oracle,
  including a direct [1000,1000] ragged-tail test); the APE arrives as F32
  (converted from checkpoint BF16);
- MLP/MoE matrices stay COMPRESSED (E4M3 payload + F32 block scales,
  byte-identical); the four DSA attention matrices' BF16 form is a
  documented transient per-layer bridge for the M3 IGemm seam until the
  scale-aware GEMM (next chunk) consumes blocks natively.

Two structural guarantees, both tested:

1. **Formula = allocator.** `layer_bytes()` is a counting-mode run of the
   SAME build code (256-aligned bump grants); `load_layer` throws unless
   actual usage equals the formula. Every load of every layer reconciles.
2. **Two-phase loads.** All CPU→managed copies happen first, then all
   dequant kernels launch, then one sync — no host writes to managed
   memory concurrent with kernels (the gray zone this avoids is documented
   in the source).

CI coverage: `glm_loader_test` builds a synthetic mini-checkpoint ON DISK
from the expected table itself (config.json + safetensors), so fixture and
table can never disagree, and verifies all of the above byte-exactly for
every layer kind (KDA+dense, DSA+MoE, KDA+MoE, MTP) plus globals. The
`minijson`-in-nvcc incompatibility (known from the dump-parity runners) is
avoided by keeping the loader host-only — the only device code is the
dequant kernel, reachable through a plain launcher.

### Real-checkpoint stream run (deployment evidence)

```
$ glm_stream_check --config <snapshot>/config.json --checkpoint-dir <snapshot>
binding validated; layer bump capacity 7.17 GiB
globals: 2.363 GiB (embed+lm_head+final norm)
layer  0..2  [KDA/dense]: 408.3 MiB each
layer  3,7,11,...,43 [DSA/moe]: 7275.7 MiB each
layer  4,5,6,...,44 [KDA/moe]: 7204.2 MiB each
layer 45 [DSA/moe/mtp]: 7338.3 MiB
peak layer 7694726912 B; streaming total 316.36 s; formula reconciled on every load
```

All 46 layers streamed; per-layer wall time ~7 s for MoE layers (7.2–7.3
GiB read from cold disk, ~1 GB/s — loader overhead is not the bottleneck;
second-pass timings drop once the page cache holds the shards). Peak
device memory = largest layer + globals ≈ 9.6 GiB, confirming full-model
correctness work fits a single 128 GB node without the TP placement.

Verification: ci-local 11/11 (new glm_loader_test entry), ASan/UBSan clean
(unit 34/34, loader 5/5), compute-sanitizer memcheck 0 errors.

## Scale-aware GEMM (deliverable 3)

`src/kernels/scale_gemm.{hpp,cu}`: `launch_scale_gemm_bf16` computes
D = Act(bf16) x W^T with W consumed NATIVELY as E4M3 payload + F32 128x128
block scales — no BF16 weight materialization, no persistent scratch. The
numerics follow the pinned reference semantics (dequantized weights, bf16
math), deliberately NOT DeepSeek-style dynamic activation quantization:
quantizing activations would change the semantics every M2/M3 reference
pinned.

Design: BM=16/BN=64/BK=32 tiles, 8 warps (one m16n8k16 group each), manual
PTX fragment packing from padded smem, bf16 mma with fp32 accumulation,
fixed k-order (deterministic, graph-capturable). The tile geometry divides
the 128-wide scale block exactly, so every stage applies ONE scalar scale
(decode x scale, one bf16 round — bit-identical to the dequant bridge's
weight tiles); ragged N/K tails read the true last scale row/col with
masked loads. OOB tiles zero-fill; k=0 zeroes the output.

### Parity results (dual oracle, both required)

- **strict** (weights bf16-rounded exactly as the kernel rounds them,
  result bf16-rounded, fp64 accumulation): isolates the kernel —
  l2_rel 3.4e-5..2.9e-4, ZERO elementwise mismatches on every case
  (2-ulp/1e-3-floor budget, l2 1e-3).
- **semantic** (true dequant in fp64): pins the DESIGN §4 contract —
  l2_rel 1.7e-3..2.6e-3 across every case, i.e. exactly the bf16
  weight-rounding RMS (2^-8/sqrt(3) ~ 2.25e-3); zero mismatches.

Cases: full blocks straddling scale boundaries (M29xN384xK256), ragged
everything (M17xN1000xK1000), decode shape at real q_a geometry (M=1,
N=1536, K=4096), all-448 payloads x 1e30 scales (products ~1e35),
subnormal-minimum payloads x 2^-20 scales, bitwise determinism across
runs, and NaN policy: e4m3fn 0x7F decodes to NaN, propagates through the
fp32 accumulator to exactly the poisoned columns (mma propagates NaN —
verified with a raw asm probe), all other outputs finite.

### Real-checkpoint slices (manual deployment run)

```
$ scale_gemm_test --checkpoint-dir <snapshot>
[ OK ] q_b_proj.weight   strict    l2_rel=9.6e-05 mismatches=0/786432
[ OK ] q_b_proj.weight   semantic  l2_rel=0.00232 mismatches=0/786432
[ OK ] o_proj.weight     strict    l2_rel=2.9e-04 mismatches=0/196608
[ OK ] o_proj.weight     semantic  l2_rel=0.00232 mismatches=0/196608
[ OK ] down_proj.weight  strict    l2_rel=1.0e-04 mismatches=0/196608
[ OK ] down_proj.weight  semantic  l2_rel=0.00243 mismatches=0/196608
```

q_b [16384,1536] (12 scale columns), o_proj [4096,16384] (128 scale rows),
expert down_proj [4096,2048] (the shape 42 layers x 288 experts repeat),
M=48, ~6 s total including host oracles.

### The NaN-policy case caught a real latent bug

`float_to_bf16_bits` converted NaN to **-0.0**: the integer RNE trick
(`u += 0x7fff + lsb`) assumes a finite exponent, and hardware FMUL on this
platform returns an all-ones-payload NaN (0x7FFFFFFF), whose rounding
carry overflows into the sign bit. Every oracle and kernel in the repo
shared that converter; nothing had ever fed it a NaN, so M2/M3 never
caught it. Fixed with an explicit NaN branch (canonical quiet NaN, sign
preserved — the same policy the fp8 encoder documents) plus a unit test
pinning six NaN payloads and the finite boundaries (FLT_MAX -> inf
unchanged). Found by the deliverable's own "NaN/Inf policy" case, which is
the point of having one.

Verification: ci-local 12/12 (new scale_gemm_test entry), unit 35/35,
ASan/UBSan clean, compute-sanitizer memcheck 0 errors.

## mHC residual-stream module (deliverable 1, mHC part)

Semantics pinned from the transformers `Glm5NextTextHyperConnection`
reference (fetched from huggingface/transformers main, matching the
checkpoint's transformers_version 5.16.0) and recorded as DESIGN §7.3.
The shape mystery resolved cleanly: `fn [24, 16384]` = `(2+n)·n` coefficient
rows over the flattened 4 streams, split `pre[4] | post[4] | comb[16]`;
`base [24]` the same split; `scale [3]` one per OUTPUT (pre/post/comb) —
not per head. Two details that would have been guessed wrong:

- the mHC input norm is an UNWEIGHTED RMSNorm over the whole flattened
  `[n·hidden]` vector (not per-stream, not weighted);
- `post`/`comb` are rounded to bf16 BEFORE the stream-update products, with
  bf16 roundings between the multiply, the 4-term mix, and the final add —
  the choreography is part of the semantics (kernel and oracle both
  reproduce it bitwise).

Also corrected: the chunk-1 binding-table comment on `scale` (guessed
"multi-head count" — actually per-output scales).

Implementation:

- `src/models/glm_mhc.hpp` — config (hc_mult/hidden/sinkhorn_iters/hc_eps/
  norm_eps; hc_mult pinned to 4 like the DSA kernels pin Hadamard-128) and
  the device weight view; CUDA-free so config, oracle, and kernels share it
  (kda_geometry pattern).
- `src/kernels/glm_mhc.{cu,glm_mhc_launch.hpp}` — three deterministic,
  graph-capturable kernels: compute (two-pass: fp32 sumsq → per-element
  normalized 24-logit projection matching the reference's rounding order →
  sequential 256-thread block reduction → sigmoid/softmax/Sinkhorn in
  registers → fp32 collapse), stream update (per-element bf16 choreography
  with the two intermediate roundings), final mean. No scratch, static smem
  (25.6 KB/block, full occupancy at 8 blocks/SM).
- `src/models/glm_mhc_reference.{hpp,cpp}` — double-precision oracle with
  the same rounding points (kda_reference pattern).
- Config parser now requires `hc_eps`/`hc_sinkhorn_iters` and exposes
  `mhc_config()`; hc_mult≠4 rejected at parse (kernel geometry pin).

Parity (`glm_mhc_test`, new CI entry): real geometry (n=4, D=4096) at
tokens 1/3/17/257/2052, hidden=512, saturated logits (|base|=40 → σ→{0,1},
softmax peaked), all-zero streams (norm-of-zero), end-to-end pipeline
bitwise deterministic and within budgets. Elementwise bf16-ulp budgets:
post/comb/collapsed/mean ≤ 2 ulps (0.5% soft, 4 hard), stream update ≤ 4
ulps (1% soft, 8 hard) — the fp32-vs-double sigmoid gap sits at ~1e-6
relative, so flips cluster at bf16 rounding boundaries only.

Verification: ci-local 13/13, unit 36/36, ASan/UBSan clean, memcheck 0
errors.
