# Qwen3.8-27B-NVFP4

Serving note for [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
on DGPP. Model architecture and weights belong to their authors; this page
records only how DGPP serves this checkpoint and what was measured here.

Same 27B Qwen3.8 hybrid stack as
[Qwen3.8-27B-FP8](Qwen3.8-27B-FP8.md): gated-delta-net + full-attention
layers, 5120 hidden, 64 main layers plus an optional MTP draft layer, 248,320 vocabulary entries,
bf16 by default. The release is mixed-precision and DGPP reads its
`config.json` `quantization_config` (compressed-tensors `config_groups`):

| Module | Format | DGPP path |
|---|---|---|
| MLP `gate|up|down_proj` (layers 0..55) | NVFP4 (packed fp4, group 16, fp8-e4m3 block scales, divided once by the F32 weight global; the input global is the W4A4 activation-side factor, unused under exact activations) | exact 16-row decode tile and paired gate/up at the supported TP1/2/4 shapes; production ldmatrix fallback for other short shapes; combined `launch_fp4w_gate_up_swiglu_bf16` and `launch_fp4w_gemm_bf16` down projection at 1024 rows and above |
| Attention `q|k|v|o_proj`, GDN in/out, `lm_head`, `layers.56..63.mlp.*` | FP8 e4m3, channelwise (group = full row) | dequant-to-bf16 bridge for wide rows, streaming FP8 matrix kernels for decode |
| `mtp.*`, embeddings, norms | BF16 | bf16 |

## Formats in DGPP

DGPP reads the release directly (no requantization). The format is decided
per module from `config_groups`: `float-quantized` (bits 8) groups bind as
channelwise FP8, `nvfp4-pack-quantized` (bits 4, group size 16) as NVFP4,
targets and `ignore` decide which wins, everything else is BF16. The MTP
draft layer is never FP4: its `.*` target loses to the `re:^mtp.*` ignore.

`prefill_fp8_per_tensor` and `dense_weights: fp8` are FP8-release levers
and are refused under this quantization (they would requantize the mixed
weights); the head is already fp8-channel resident, so it needs no lever.

## Gates

Dump-parity (strict and relaxed), 1..7-row decode invariance and a
teacher-forced draft step run in CI on a synthetic mixed release built by
`qwen35_mixed_fixture` (the `qwen35_mixed_*` test chain); the real
checkpoint additionally runs against the `qwen35_reference_dump.py`
numpy reference, which decodes nibbles and e4m3 scales exactly as the
kernels do and divides once by the weight global (direction pinned
against the FP8 sibling release's per-layer RMS, 2026-10-06).

The four-node TP legs of the FP8 card's table have NVFP4 siblings
(`--weights nvfp4`); the weight bytes per rank are smaller, the draft
layer and BF16 islands are unchanged. The independent [PR review and follow-ups](../../benchmarks/results/2026-10-11-pr95-followups/README.md)
record the current kernel, numerical checks and measured serving comparisons.

## Original dense NVFP4 production kernel (2026-10-06)

The MLP used to run gate/up on the CUDA-core `fp4_gemv` at <= 128 rows
and everything else on the synchronous `dense_mma_fp4` reference tile.
The original contribution ran all three projections through the production ldmatrix
kernel (`launch_dense_mma_fp4_prod_*`: the grouped launcher with one
pre-staged `{0, m, 0}` segment per row count and boot-harvested
per-layer `[gate,up,down]` view tables on resident stacks — graph-safe,
no copies in the walk; a 3-entry staging slot re-staged per call on
streaming stacks). Microbench on the w1 shapes: gate/up m=8/32 go
623/2456 us to 239/257 us (roofline ~251); down is bitwise the reference
it replaces (glm_moe_test pins the pair).

A purpose-built fp4 streaming decode kernel (mma_gemv's weights-once
structure, e2m1 B-side) was tried the same day: correct against a
doubles oracle with bitwise m-invariant rows, but slower than the
production kernel on every production shape (gate/up m=32/64/128:
288/436/908 us vs 269/313/533; pp448 prefill 1016 ms vs 720), so it
stayed unwired. This experiment did not establish a performance ceiling;
the October 11 follow-ups below improve the small-row path. Its one m==1
win stayed unused: it would split the <= 128-row
chain the solo-vs-batch bitwise gates require.

Serve A/B against the FP8 release (same prompts, temp 0, identical
transcripts and drafter acceptance): short prompt 29.1 vs 24.5 tok/s
(119 vs 141 ms/pass), pp448 prefill 720-748 vs 931 ms. The NVFP4
release led on both in that measurement. The current broader workload
comparison is in the linked review record. No lossy dense-W4A4 path is built (see the
MoE-side `DGPP_MOE_W4A4` gate, still issue-#68-blocked).

## Original dense NVFP4 prefill GEMM (2026-10-06)

Above 1024 rows the MLP runs a purpose-built fp4w prefill GEMM
(`launch_fp4w_gemm_*`: `fp8w_gemm`'s 128x128x64 tiles, sixteen warps
and three-stage activation pipeline with an e2m1 W-side — 8 payload
bytes plus the step's four e4m3 scale bytes a thread a step, exact
pairwise decode, the F32 global dividing once in the epilogue), which
streams weights once per 8-m-tile group instead of re-reading them per
64-row tile. Correct against a doubles oracle, sparse codes bitwise
vs the grouped kernel, tolerance-equal otherwise; the <= 1024-row
chain (decode, all tests) is untouched. Microbench crossover: slower
below ~1024 rows (gate m=448 loses 11%), faster above (m=1024/2048/4096
win 12-17/14-20/45-90%). Serve: pp2080 prefill 2445-2460 ms vs 2568
(~5%), pp448 and decode unchanged.

## Prefill follow-ups (2026-10-11)

The prefill kernels use 256x128x64 physical tiles, two activation stages
and two swizzled weight tiles in 96 KiB of shared memory. Gate/up share
one tile: 64 columns from each projection, with each of sixteen warps
owning 16 matching gate and up columns. The epilogue rounds both dots to
BF16, then performs the existing SwiGLU math and stores only its result.
This removes the intermediate matrices and separate activation launch.
Down uses eight warps, or sixteen for long TP2 products.

Adjacent BF16 outputs share aligned 32-bit stores, with scalar stores
for odd tails or unaligned views. Long walks use three M tiles per cache
group. These changes preserve each output's ascending-K FP32 dot and
single global-scale division. Dispatch starts at 1024 rows; smaller rows
use the exact decode paths described below.

The optimized path matched the earlier kernel's full-vocabulary logit
hashes at all 19,456 scored real-checkpoint positions, with a bitwise
4096-position repeat in a fresh process. Long synthetic reference cases,
tensor-parallel forwards, output-layout guards and CUDA sanitizers cover
the changed path. Resident view initialization now omits the optional MTP
layer when MTP is disabled, saving 710 MiB at world 1 with DFlash2.
The linked record retains rejected alternatives and the validation scope.

## Decode follow-ups (2026-10-11)

At 1–16 activation rows, the dense FP4 path uses 16×128×128 tiles and
three raw-weight/activation staging slots (46 KiB). Gate and up share one
grid, increasing parallelism at TP4. Dispatch allows N/K up to 17408 with
K divisible by 128, covering the actual TP1/2/4 shapes; other shapes retain
the previous production path. Bounded gate prefetch runs during the
multi-node attention fold. Both paths use the same ascending-k16 FP32
accumulation and final global-scale division, with bitwise reference tests
covering shape boundaries and unaligned views.

The collective snapshot now computes its placement hash on the GPU from
the existing loads. CPU posting reads the published hash instead of
rescanning the activation row. This preserves receiver checks, wire format,
rank-order arithmetic and the staging-ring lifetime. Both FP8 and NVFP4
benefit; matched comparisons use the same executable. The four-node NVFP4
verification row-cost setting is 4 ms after measured sweeps; FP8 remains
at 2 ms.

This hybrid checkpoint stores 26.87% fewer nominal transformer source bytes
per rank, including scales. This is checkpoint byte accounting, not a DRAM
traffic measurement. Attention, the last eight MLPs, the head and drafter
remain common costs. Matched CUDA traces, sampler-overhead controls, per-layer timings,
collective analysis and rotating matrix probes are in the
[detailed investigation](../../benchmarks/results/2026-10-11-pr95-followups/decode-investigation.md).
The uninstrumented comparisons pass all 60 decode cells and all 12 cold-prefill
buckets across TP1/2/4 against the stronger FP8 bracket. Decode gains range
from 1.49% to 40.44% across five prompt classes and C1/C2/C4/C8.
Four-node decode with drafting disabled improves 21.98%.
The final prefill build separately confirms all-world prose decode and
four-node longer-context decode, preserving the prior outputs. See the
[serving record](../../benchmarks/results/2026-10-11-pr95-followups/README.md)
for individual repetitions, baseline drift, limits and both build identities;
small measured margins are not a guarantee for every input.

The earlier 100-case GSM8K sample scored 95/100 for NVFP4 and
96/100 for FP8; both scored 100/100 on extraction. These samples do not
establish broad quality equivalence.
