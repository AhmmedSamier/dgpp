# GLM vision numerical investigation — 2026-09-18

The large full-encoder differences in the [initial image-input work](2026-09-18-glm-vision.md)
were reproducible and warranted investigation. They came from mismatched
arithmetic boundaries and backend reduction choices, not image layout errors.
The corrected encoder produces **bitwise identical final embeddings** to the
CUDA eager oracle on all 30 cases tested: five synthetic cases and 25 diagram
questions containing 18 distinct images. The pinned corpus now requires exact
equality; the general diagnostic oracle retains its fixed 0.5% relative-RMS /
0.99998 cosine gate.

## Reference contract

The initial oracle deliberately retained FP32 QK scores to match the initial
native implementation. This differs from the official Transformers
[GLM5-Next eager attention](https://github.com/huggingface/transformers/blob/f0d778337771dd81082653d752f8bd6563b1d2eb/src/transformers/models/glm5_next/modeling_glm5_next.py):
its BF16 matrix product and scaling precede FP32 softmax. The corrected oracle
uses those eager boundaries by default. The old `--attention fp32` mode remains
available for diagnostic reproduction. CPU, SDPA and flash attention are
separate numerical targets; the results here do not claim equivalence to them.

The oracle implements the equations independently with PyTorch operations.
Convolutional projections are flattened into equivalent linear projections;
this is not a claim of bitwise agreement with every cuDNN convolution algorithm.
It consumes the same processed RGB bytes as native, so this investigation
isolates encoder arithmetic from image decoding and resizing.

The environment matches the initial record: GB10, driver 580.173.02, CUDA 13.0,
PyTorch 2.14.0+cu130 (`08187d9e0fba026dc8217405802ab5381dc88d90`), NumPy,
and the pinned GLM-5.3-Flash snapshots. TF32 and reduced-precision BF16
reductions are disabled. Serving has no PyTorch dependency.

## Findings and fixes

Operation-level traces exposed the first mismatch with identical inputs.
Their filenames distinguish BF16 tensors and FP32 accumulators, and query
traces can be reassembled from tiles. `--isolate-ops` feeds native values to
the next operation; full-depth comparisons never replace reference states.

- Vision RMSNorm now uses FP32 reductions and preserves the two BF16 rounds
  around normalization and weight multiplication. The text stack's existing
  FP64 reduction is unchanged.
- Rotary embeddings compute reciprocal positive powers and separate FP32
  multiply/add operations. Fused multiply-add produced different BF16 results
  at rounding boundaries, including cancellation near zero.
- All vision projections explicitly restrict cuBLASLt reductions to
  `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE` and write BF16 results. Biased
  projections use the fused bias epilogue. Merely using an FP32 destination
  did not match the reference reduction path: the 168×112 gradient had 90
  differing projection values and 0.13614% final relative RMS after the other
  fixes. Explicit reduction selection eliminated this error. A separate
  cancellation test (`1 + 1/1024 - 1`) fails in all 98,304 outputs when the
  default BF16 split-K policy is allowed, and passes with the explicit policy.
- Attention uses the upstream eager BF16 score/scaling boundaries and FP32
  softmax. Its reductions use warp and block forms appropriate to row length.
- QK and PV batch all heads. Query tiles use the algorithm selected for the
  full matrix, with actual tile layouts for execution. Choosing a fresh
  algorithm for each tile changed one BF16 value in layer 4 of AI2D row 28;
  that one value grew into 4.6486% final RMS error. Preserving the full-matrix
  algorithm reduced the error to 0.02732%.
- Values are stored as `[heads, tokens, head_dim]` and multiplied without a
  storage transpose. Two remaining diagrams had a single PV mismatch even
  with the full-matrix algorithm; matching this layout removed both.
- Final LayerNorm uses the CUDA reference’s FP32 Welford schedule: four
  adjacent elements per thread, 128 threads, then warp and inter-warp merges.
  A two-pass mean/variance calculation differed by only 11 post-GELU values
  on one diagram, but the resulting 0.02732% embedding error changed its
  answer in serving. GELU itself matched on identical normalized inputs.
  Matching Welford removed all remaining diagram differences. The adapted
  reduction schedule carries the [PyTorch license](../../docs/licenses/pytorch.md).

The attention findings are specific evidence about cuBLASLt reduction choices,
not a general rule that algebraically equivalent matrix layouts are identical.
The fixed full-depth gate is important when changing CUDA/cuBLAS versions.

Query tiling retains the existing 0.34 GiB workspace. Vision weights still
occupy 1.05 GiB per rank. The ordinary text GEMM entry point retains its
existing math, dispatch and defaults; the new batched and linear entry points
are used by vision.

## Full-encoder results

The before and after columns below use **the same corrected CUDA eager oracle**.
They must not be confused with the original 6.7% versus 6.2% measurements,
which used the FP32-score oracle and different pairs of backends.

| Fixture | Tokens | Before relative RMS | After relative RMS |
| --- | ---: | ---: | ---: |
| 112×112 gradient | 16 | 9.3003% | **0% (bitwise)** |
| 168×112 gradient | 24 | 10.8211% | **0% (bitwise)** |
| 224×224 quadrants | 64 | 37.4934% | **0% (bitwise)** |
| 224×224 bars | 64 | 15.3689% | **0% (bitwise)** |
| AI2D moon diagram | 240 | 8.8335% | **0% (bitwise)** |
| 896×896 gradient | 1,024 | — | **0% (bitwise)** |

All 25 pinned diagram questions (65–1,014 visual tokens) pass the full-depth
gate with zero differing output bits. The 1,024-token synthetic gradient also
exercises the maximum supported image token count. This is exact parity on
this corpus and pinned backend, not a guarantee across all images, libraries
or attention implementations.
[Raw per-case measurements](2026-09-18-glm-vision/numerics.json),
[operation trace comparisons](2026-09-18-glm-vision/numerics-112-ops.log) and
[CUDA test output](2026-09-18-glm-vision/numerics-gpu-tests.log) are retained.

## Regression checks

`glm_vision_test` covers BF16 score rounding, softmax widths on both sides of
2,048, padded batch strides, both weight layouts, GEMM plan identity, full
versus tiled reductions, biased and unbiased cancellation, LayerNorm/GELU
reference hashes and output row placement. All seven tests passed. The
existing `bf16_gemv_test` also passed; together they took 16.92 seconds.
The LayerNorm golden test rejects the old two-pass kernel (one differing
post-GELU value); its [negative control](2026-09-18-glm-vision/numerics-norm-regression.log)
is retained. A fresh 112×112 operation trace passed the
isolated-operation gate. The full-depth bounds are enforced by default in
`tools/glm_vision_reference.py`, with `--diagnostic` required to request a
report without a parity verdict.

## Controlled serving comparison

The final native encoder was tested on the two-node GLM-5.3-Flash hybrid
checkpoint with graph decode and MTP enabled, using the same deployment and
requests as the reference-injection world. All 21 distinct injected feature
files were verified byte-identical to freshly generated CUDA eager outputs.
The reference run was preserved from earlier in this investigation; only the
native vision arithmetic changed afterward. The temporary injection patch
is absent from production source.

| Check | Final result |
| --- | --- |
| Image API contract, streaming, multiple images, `n`, MTP | Pass (11.95 s) |
| Synthetic questions | 14/14 identical answers and reasoning |
| Diagram questions | 25/25 identical answers and reasoning |
| Diagram accuracy on this small sample | 23/25 for both |
| Aligned visible-token log probabilities | 140/140 exactly equal; mean and maximum delta 0 |

Before the Welford fix, the native/reference diagram comparison had one
changed answer (row 32) and accuracy 22/25 versus 23/25. That difference is
resolved. The final result measures parity on these requests; the small
sample does not establish general model quality. Log probabilities cover
API-visible generated tokens with identical preceding reasoning, rather
than a teacher-forced NLL benchmark.

The [comparison data](2026-09-18-glm-vision/numerics.json) records the final
and intermediate results, binary SHA-256s and deployment paths. Raw
[native synthetic](2026-09-18-glm-vision/numerics-native-synthetic.jsonl),
[reference synthetic](2026-09-18-glm-vision/numerics-reference-synthetic.jsonl),
[native diagram](2026-09-18-glm-vision/numerics-native-ai2d.jsonl) and
[reference diagram](2026-09-18-glm-vision/numerics-reference-ai2d.jsonl)
responses are retained. `compare_answers.py` regenerates the comparison;
use `--diagrams` for the numbered diagram questions. The
[API log](2026-09-18-glm-vision/numerics-api.log) and
[serving timings](2026-09-18-glm-vision/numerics-serving.log) are also retained.

Both ranks stopped cleanly after each run. Operation streams match within
each world:

- Final native: `32beedccdd629a0f6f5c209b90b0f40f` on both ranks.
- Reference injection: `a8709de6b28fa20c19d68b267d023aad` on both ranks.
- Intermediate native: `d84ce30ea4b4f574d8b22c177be98040` on both ranks.

These hashes compare participating ranks within a run, not arithmetic across
different runs. All test worlds are stopped.

## Reproduction

Prepare the pinned fixtures using the [original harness](2026-09-18-glm-vision/).
With the serving world stopped and `CKPT` pointing to the FP8 snapshot, run:

```bash
cmake --build build-ci -j 4 --target glm_vision_check glm_vision_test bf16_gemv_test
ctest --test-dir build-ci -R '^(glm_vision_test|bf16_gemv_test)$' --output-on-failure
python3 benchmarks/results/2026-09-18-glm-vision/check_numerics.py \
  --checkpoint "$CKPT" --fixtures /tmp/dgpp-vision-quality \
  --output /tmp/vision-numerics
```

The Python command needs CUDA PyTorch and NumPy. It loads weights once,
checks all 30 cases for bitwise equality, saves logs and `results.json`, and
exports reference features under `references/` for controlled serving tests.
`--only CASE_NAME` selects a subset. The default checker has no reference
injection path; the separately built diagnostic patch from the original
record is used only for A/B serving comparisons.
