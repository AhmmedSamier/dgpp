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

## MoE router, expert execution, and route traces (deliverables 1+5)

Semantics pinned to the transformers Glm5NextTextTopkRouter/Experts/MLP
references and recorded as DESIGN §7.4. The subtle parts: selection ranks
BIASED scores (sigmoid + e_score_correction_bias) but routes UNCORRECTED
weights; normalization divides per element (not reciprocal-multiply) before
×2.5; the swiglu clamps are asymmetric (gate max-only, up both sides);
per-token accumulation runs in ASCENDING expert id (the reference's
index_add visit order — order changes bits, so it is semantics); the shared
expert (weight 1) is the final single bf16 add. The engine's tie rule
(equal biased scores → lower expert id) is pinned where torch CUDA topk is
unspecified.

Implementation:
- `src/kernels/glm_moe.{cu,glm_moe_launch.hpp}`: router (one thread per
  expert dot, fixed sequential reduction; thread-0 top-k with the pinned
  tie rule; ascending insertion sort; per-element fp32 normalize), swiglu
  clamp, row gather, and the bf16-choreographed accumulation kernel. All
  deterministic, graph-capturable, no scratch.
- `src/models/glm_moe_layer.{hpp,cpp}`: host-orchestrated forward
  (router → host segmentation → per-expert gather + scale-GEMMs + swiglu +
  accumulate, ascending; shared last). One sync per enqueue — the
  diagnostic mode's documented cost; device-side grouping is M5+.
- `src/models/glm_moe_reference.{hpp,cpp}`: double oracle reproducing
  every bf16 rounding point of the engine chain (strict GEMMs with
  bf16-rounded dequant weights).
- `src/models/glm_trace.{hpp,cpp}` + `tools/route_trace_traffic.py`: the
  DGPPTC1 trace format (golden-byte-pinned on BOTH the C++ and python
  sides) and the corrected per-rank traffic model.
- `GlmQuantMatrix` moved to `src/models/quant_matrix.hpp` (CUDA-free) so
  the host-only model library and kernels share the type.

### Parity results (honest data — see the RNG fix below)

- Router at REAL geometry (E=288, H=4096, K=8; tokens 1/3/17/257):
  weights within 5.3e-7 relative of the double oracle (fp32-vs-double
  sigmoid/normalize), ZERO id mismatches, zero near-tie certifications
  needed. Tie test (all scores equal) selects experts 0..7 with uniform
  weights 2.5/8.
- Expert path (E=8/16, K=2/4, tokens 1/17/5) vs the strict oracle: max
  0-1 bf16 ulps, l2 ≤ 4e-5, zero soft violations, bitwise-deterministic
  reruns. Accumulation-order test: biased scores force selection order
  {7,6,5}, output matches the ascending-order oracle exactly.
- swiglu edges: gate-above-limit, gate-below (no clamp!), up ±, exact
  boundary values at 10.

### A test-infrastructure bug the router's "0 rel err" exposed

The shared test RNG (`scale_gemm_test_helpers.hpp`) documented `[-1, 1)`
but returned `[0, 2)` — the int64 cast of a 53-bit value is never
negative. Every affected suite had been running on ALL-POSITIVE data: the
router test's dots were ~+200, sigmoid saturated to exactly 1.0 for every
expert, all weights were exactly 2.5/8, and the router weight comparison
was VACUOUS (0.3125 vs 0.3125). Same class of skew in the mhc and loader
fixture fills. Fixed in all three; the honest reruns above are the real
results. The mhc comparator also gained the cancellation floor it never
needed with one-signed data (a 4-stream sum canceling from ~0.25-magnitude
terms to ~1e-8 turns ~8e-9 of fp32 noise into "36 ulps"), and the update
chain gained a 1e-6-rate hard budget for chained-rounding boundary flips
(1 intermediate-ulp flip per 33.6M elements, statistically expected).
scale_gemm re-verified on symmetric data including the real-checkpoint
slices: strict l2 7.4e-5..2.9e-4, semantic ~2.3e-3, zero mismatches.

### Traffic-model tool validated against §3's own anchors

`route_trace_test` (new ctest entry): a uniformly random trace (42 layers
× 512 tokens × top-8) reproduces busiest-rank 3.515 ± 0.06 and the 7.457
GB / 32.42 ms critical path; a single-rank trace reproduces §3's 12.198 GB
worst-placement row; expert byte sizes reconcile with the checkpoint
budget (304.4-304.6 GB routed share). Real traces from representative
prompts land with the assembled model (next chunk) — a uniform trace is
the null model, not evidence.

Verification: ci-local 15/15 (glm_moe_test + route_trace_test new), unit
39/39, ASan/UBSan clean on all touched suites, memcheck 0 errors on the
moe and mhc tests.

## Chunk 6: the assembled forward, curated suite, real traces (deliverables 1+5)

`GlmDiagnosticModel` (src/models/glm_forward): the full text forward over
the streaming loader — mHC wiring per DESIGN §7.5, KDA/DSA/MoE layers
REBOUND to each resident layer (constructed once, shape-keyed scratch),
the two-rounding Glm5NextTextRMSNorm kernel (the generic rmsnorm rounds
once and drifts this path systematically), bf16 lm head. CI parity chain:
glm_forward_test writes a synthetic mini-checkpoint (hidden 128, 6 layers
4 KDA + 2 DSA, 2 dense + 4 MoE, MTP set present), the new
tools/glm_reference_dump.py computes the FULL-STACK reference in pure
python doubles from the SAME checkpoint, and the engine is compared:
hidden max 75 ulps (19/3072 over soft, l2=0.0034), top-1 exact, top-8
sets exact, route ids exact, route weights <= 4.3e-4 relative,
deterministic reruns. ctest: fixture -> generate -> test (18/18 CI).

Chunk 6a found and fixed three latent engine bugs before any of this
passed, each caught by the honest-data discipline:

1. Loader race: load_layer's phase-one CPU writes into the bump raced
   previously-loaded layers' in-flight kernels (the loader had never been
   called mid-forward). Symptom: flaky NaN streams -> router scores NaN
   -> expert id -1 (NaN never wins comparisons) -> h_counts_[-1] host
   heap corruption. Fix: entry cudaDeviceSynchronize in load_layer.
2. DSA select_k must be a power of two (bitonic select/expand networks):
   the real 2048/4=512 and every prior test's 16 satisfied it by luck;
   select_k=6 silently dropped every selected pool. Now rejected at
   construction.
 3. DSA num_heads=2 routes the attention kernel's head-group tiling to a
    broken 64-group partition (selections match, outputs drift 1e33 —
    bisected through the CI test machinery; 4/8/64 heads pass). Now
    rejected (power of two >= 4). Also recorded: select_k=8 chunked
    prefill has unaudited near-tie continuation flips (M3 test-coverage
    gap, not a forward-path blocker). [Closed at the M4 close-out: the
    re-measured failure was a test-logic inversion — the flipped-row
    drift fallback budget was vetoing CERTIFIED near ties (accumulator
    ran before the audit gate), and the select_k=8 flip was certified at
    3.1x noise. Fixed + a permanent select_k=8 chunked variant; see the
    close-out addendum in 2026-08-28-dsa-m3-layer.md.]

### The chaos finding and the suite design it forced

Free-run end-to-end parity at REAL depth is chaos-limited: the measured
cross-implementation floors (KDA 3.6e-3, DSA ~3e-3, MoE ~2.3e-3 vs their
own oracles) compound at ~1.18x per layer through the mHC stream (post
in [0,2] amplifies) and decorrelate the stacks by layer ~30 — measured
engine-vs-torch, layer by layer, with layer 0 (an isolated measurement)
sitting exactly at the module floors. Not a wiring bug: the mini fixture
(same wiring code, 6 layers) holds l2=0.0034 free-run.

The curated suite therefore compares per-layer ISOLATED (every layer
starts from the reference trajectory; the dumps carry per-layer streams):
kept-row l2 budgets < 0.02; route near-tie flips (the noaux bias ties
scores at the selection boundary, so ~1e-3 router-input noise flips them)
are counted, bounded < 1%, and excluded from the l2 — the M3 kept-row /
flipped-row discipline; the head runs on the isolated final streams.

### Real-checkpoint curated suite (exit criterion 1)

```
$ glm_forward_check --config <ckpt>/config.json --checkpoint-dir <ckpt> --suite suite.txt
smoke    21 tok  kept l2 max 0.0068 (all-rows max 0.0330, 18 flip layers) head l2 0.0053 top1=0 top8miss=2 routes: flips=61/7056  kept-rel 0.0093  OK
factual  23 tok  kept l2 max 0.0051 (all-rows max 0.0144,  8 flip layers) head l2 0.0046 top1=0 top8miss=2 routes: flips=30/7728  kept-rel 0.0063  OK
code     65 tok  kept l2 max 0.0045 (all-rows max 0.0118, 26 flip layers) head l2 0.0047 top1=0 top8miss=3 routes: flips=171/21840 kept-rel 0.0047  OK
```

- Kept-row l2 is FLAT across depth: max 0.0068 at layer 44 as at layer 0
  — the module floor, un-amplified, on all 45 layers of every case.
- Every above-floor all-rows outlier coincides with a route flip on that
  layer (L19: kept 0.0050 while the flipped row drives all-rows to 0.033).
- Top-1 agrees on EVERY token of every case; top-8 set misses 2-3 of
  8x65; kept route weights within 0.93% relative.
- Free-run numbers are reported but not asserted (DESIGN §7.5).
- Suite definition + token ids: benchmarks/glm-suite/ (dumps regenerate
  from the tool; ~4-10 min/case on CPU).

### Real route trace (exit criterion 4)

```
$ glm_forward_check ... --trace-ids-file ids/trace.ids --trace-out real.trace
trace run: 272 tokens over 45 layers (317 s, deterministic re-run)
$ tools/route_trace_traffic.py real.trace
per-token busiest-rank experts/layer: 3.541  (uniform model: 3.515)
corrected critical path: 7.484 GB/token = 32.54 ms @ 230 GB/s  (uniform: 7.457, +0.4%)
```

The first real trace (technical prose) validates the uniform null model
to +0.4% — the §3 traffic numbers stand for this prompt class, now with
measured evidence instead of an assumption. Sampling other prompt classes
is future work; the tool consumes their traces unchanged.

Verification: ci-local 18/18, ASan/UBSan clean, memcheck 0 errors on the
forward test; the curated suite and trace run recorded above are
deployment runs on the checkpoint box.

## Close-out (2026-08-29): per-flip route certification

The one doctrine gap in the M4 evidence — route flips bounded
statistically, not certified individually — is closed, with the DSA
near-tie audit's shape adapted to fp32 router scores:

- **Engine**: `moe_router_kernel` exports its full biased-score row
  ([tokens, n_experts] fp32, written before the selection loop scribbles
  -INFINITY into the smem copy); `GlmMoeLayer` stages it beside ids/
  weights; `GlmDiagnosticModel::Outputs` carries it per routed layer.
- **Reference**: both dump backends write a `router_biased` tensor
  (flat over route_layers) plus `n_experts` in the header config.
- **Certifier** (`src/models/glm_route_audit.hpp`, host-only): (a) the
  engine's selection must be the spec top-k of the ENGINE'S OWN biased
  scores; (b) every swapped expert pair must straddle the boundary within
  32x the token's measured cross-implementation noise — the yardstick
  taken over the experts NOT involved in the swap, so corruption
  concentrated on the swapped experts cannot inflate its own cover (a
  first design that averaged over all experts was rejected for exactly
  this: a doctored far-rank displacement drove its own noise up and the
  multiple down to 2.7x). Five host unit cases pin the certifier's
  rejection paths: near-tie certifies, far-rank displacement rejected,
  zero-noise swap rejected (engine right, reference doctored), selection
  inconsistent with own scores rejected (router bug), duplicate id
  rejected (top-k bug).

### Close-out run: reduced budget (12 layers, 9 routed)

Full-depth regeneration is ~4-10 min/case; the close-out re-validation ran
at a measured smaller interval (~3.3 s/layer; 60-120 s/case generation,
3-7 s/case suite run). The 45-layer numbers above stand as the
exit-criterion evidence; the full-depth re-run with certification is
pending future batch time.

```
$ glm_forward_check ... --suite suite.txt --layers 12
smoke    21 tok  kept l2 max 0.0048  head l2 0.0032  top1=0  top8miss=8
  routes: flips=7/1512 certified=2 (11.6x noise max)  OK
factual  23 tok  kept l2 max 0.0051  head l2 0.0023  top1=0  top8miss=2
  routes: flips=17/1656 certified=6 (8.5x noise max)  OK
code     65 tok  kept l2 max 0.0035  head l2 0.0041  top1=2 (cert 2)
  top8miss=10  routes: flips=64/4680 certified=16 (7.2x noise max)  OK
3 cases, 0 failures
```

Every route flip certified (7-12x noise multiples, far under the 32x
bound); zero rejections. The truncated stack crowds the head's boundaries
too: `code` flips top-1 on 2 tokens whose reference top-2 logit margins
sit within the measured logit noise — certified the same way (the
full-45-layer stack had top1=0 on every case). The flip-rate bound is now
a 10% wholesale-breakage net only — the certification is the primary
criterion, and at 12 layers the old 1% statistical bound rejected
legitimate certified flips (1.37% rate; the same doctrine inversion the
M3 close-out fixed in the DSA chunked test).

Trace path re-validated at the same budget: 272 tokens x 12 layers in
8.8 s, deterministic re-run, traffic model consumes it unchanged —
busiest-rank corrected critical path 7.482 GB/token (+0.3% vs uniform;
45-layer record: 7.484, +0.4% — the router input distribution is stable
across depths).

Two latent app bugs found and fixed on the way (both in
`glm_forward_check --layers`, a path that was dormant in M4 because the
dump tool had no layer budget): the truncated config RELOCATED the MTP
binding to `layers.<budget>.*` (mtp_layer() = num_hidden_layers when the
predictor is enabled), and the loader's two-way binding gate rejected
every layer beyond the budget as "unexpected" (a truncated diagnostic
stack now classifies layers >= its budget as out-of-scope, counted
separately — the full-config bind check is unchanged and still
75,761/75,761 with 37,338/37,338 scales bound, zero unmatched).

Verification: ci-local 18/18; unit 44/44 (was 39; +5 certifier cases);
ASan/UBSan clean and memcheck 0 errors on glm_moe_test + glm_forward_test
+ unit_tests; tool selftest OK (both backends write router_biased).

## Close-out verification sweep (2026-08-29, final state)

- ci-local (WERROR, clean configure/build) 18/18; full ctest under ASan
  18/18 and under UBSan 18/18 (the UBSan-on-final-state gap from the M4
  audit closed).
- Full-suite compute-sanitizer memcheck over every CUDA test binary: 0
  errors everywhere. One pre-existing finding fixed en route: the M1
  synthetic testbed (gpt_doll) reported 2 "errors" that were the rmsnorm
  launcher's swallowed dynamic-smem opt-in probe (a deliberate
  clamp-and-clear from the initial commit, functionally benign — GB10
  rejects the 96 KB request for that kernel's footprint). The launcher now
  queries the per-kernel ceiling (`cudaFuncAttributes::
  maxDynamicSharedSizeBytes`) and requests within it, so the call cannot
  fail; behavior is unchanged (all gpt_doll stages pass bitwise).
- Python tool selftests: kda/dsa/glm reference-dump selftest OK.
- Docs brought current for M4: README status/tools/tests (44 unit cases,
  the six M4 suites, per-flip certification), benchmarks/README GLM
  suite section, docs/measurements.md M4 section (loader streaming
  ~1 GB/s cold-disk, 7.2-7.3 GiB/MoE layer, peak footprint ~9.6 GiB;
  real-trace traffic +0.4% vs uniform), DESIGN §7.2 kernel-constraint
  pins (select_k pow2, heads pow2 >= 4), PLAN.md M4 marked complete.
