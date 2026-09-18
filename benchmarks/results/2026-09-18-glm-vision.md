# GLM-5.3-Flash image inputs — 2026-09-18

This record preserves the initial implementation's measurements. The
[subsequent numerical investigation](2026-09-18-glm-vision-numerics.md)
corrects its reference contract and resolves the large embedding differences;
consult that record for the current arithmetic and gates.

DGPP now carries image inputs from Chat Completions through preprocessing,
the admission journal, the scheduler and the GLM vision encoder. Native
BF16 image features replace prompt embedding rows, including the MTP pass's
shifted input. Streaming and graph decode retain their normal paths.
[The input guide](../../docs/vision.md) documents formats, limits and memory.

Image requests bypass prefix caching. The current index compares token IDs;
different images can have identical placeholder tokens. Reusing their state
would be incorrect. A future image cache needs the complete processed image
and its geometry in the prefix identity, plus image-aware suffix prefill.

## Environment and checks

The implementation was built over dirty base revision `a5fcb189c28f`, preserving
the workspace's existing API and packaging work. Tests used two GB10 nodes,
CUDA 13.0, the CI build and the existing
`deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2.json` configuration: graph decode,
MTP, four request slots, a 163,840-token FP8 KV pool and 1.5 GiB prefix budget.
All test worlds were stopped after their checks.

Checkpoints:

- Serving: `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`, snapshot
  `17b8d4d83e33e836e9549d7e1a8d25b5117203c1`.
- Vision oracle: `unsloth/GLM-5.3-Flash-FP8`, snapshot
  `a160e2291674d9e3e92e98fd82faa2544a2867a3`.
- The 347 vision tensors contain 1,127,254,016 bytes. Their native FNV digest
  is `2133678757230988767`. The replicated weights require 1.05 GiB per rank,
  with 0.34 GiB of preallocated workspace.
- The optional oracle used PyTorch `2.14.0+cu130` and NumPy. TF32 and BF16
  reduced-precision reductions were disabled. Serving has no PyTorch dependency.

Rebuilt `unit_tests`, `serve_test`, `fabric_serve_test`, `scheduler_test` and
`glm_vision_frontend_test`; all five passed in 113.97 seconds:

```bash
ctest --test-dir build-ci \
  -R '^(unit_tests|serve_test|fabric_serve_test|scheduler_test|glm_vision_frontend_test)$' \
  --output-on-failure
```

Coverage includes canonical base64, malformed inputs, resize/padding budgets,
configuration bounds, image/template token spans, text-only engine rejection,
payload ownership and journal round trips, usage, streaming and exclusion
from the token-only cache and grouped prefill. The actual checkpoint frontend
test ran rather than skipping.

`scripts/vision_api_check.py` passed on the fabric: red/blue isolation with
identical text, multiple images, SSE, multiple choices, concurrent requests,
a 1,024-token image, an image after a prefill chunk boundary, and a
1,024-token image spanning the 2,048-row boundary. The latter prompt contained
2,553 tokens and returned the expected color with MTP enabled. JPEG inputs
were additionally exercised by the diagram checks below.

## Encoder arithmetic

The oracle independently implements the vision equations from
[Transformers GLM5-Next](https://github.com/huggingface/transformers/blob/f0d778337771dd81082653d752f8bd6563b1d2eb/src/transformers/models/glm5_next/modeling_glm5_next.py).
Both paths consume exactly the same resized RGB pixels, isolating the encoder
from image-decoder and resize differences. Attention uses FP32 QK scores,
BF16 softmax probabilities and BF16 layer outputs.

Tracing exposed extra rounding in a cuBLASLt BF16-output projection. All
vision GEMMs now write FP32 results and explicitly round once, with bias
included where needed. On the rectangular gradient's isolated merger
projection, relative RMS fell from 0.3507% to 0.00790%. This uses the existing
score scratch; the shipped configuration's workspace size is unchanged.

The fixed arithmetic gate compares every block and merger stage with
identical inputs, requiring relative RMS at most 0.5% and cosine at least
0.99998. Patch layout/normalization must match exactly. It passed for the
gradient, quadrants, bars and the AI2D moon image. The largest observed
isolated-block relative RMS on the three synthetic fixtures was 0.2341%.

Full-depth distances are larger because small BF16 differences accumulate:

| Fixture | Visual tokens | Relative RMS | Cosine |
| --- | ---: | ---: | ---: |
| 168×112 gradient | 24 | 7.168% | 0.997465 |
| 224×224 colored quadrants | 64 | 29.018% | 0.957063 |
| 224×224 bars | 64 | 8.542% | 0.996355 |
| AI2D row 48, moon diagram | 240 | 6.470% | 0.997906 |

For context, CPU versus CUDA PyTorch differed by 6.163% RMS on the 112×112
gradient and 22.796% on the quadrant fixture. These are different pairwise
comparisons: subtracting them from native-versus-reference error does not
estimate an implementation error. They also do not establish harmlessness.
An early 4% full-depth threshold failed even for the two reference backends;
a later 15% bound still failed on the quadrants. The final tool reports
full-depth distances separately from its fixed isolated arithmetic gate and
allows explicit `--max-relative-rms`/`--min-cosine` bounds. No full-depth
bitwise or close-distance parity claim is made.

## Effect on answers

For a controlled comparison, a temporary diagnostic binary replaced only
the finished image embeddings with the PyTorch output. Model weights,
preprocessing, prompts, sampling, graphs, MTP and language-model execution
were otherwise the same. Reference rows were staged on both ranks and
selected by a hash of the actual processed RGB input. This hook is absent
from the serving implementation; the reproduction patch is retained below.

The native and reference paths returned identical content for all 13 short
synthetic questions about colors, positions, counts and bar heights. Their
open-ended gradient descriptions differed in wording. Some short-answer
log probabilities changed by approximately 0.25 nat even when the selected
answer stayed the same.

For real diagrams, the sample uses rows 0, 4, 8, …, 96 of
[AI2D](https://huggingface.co/datasets/lmms-lab-encoder/ai2d), dataset revision
`c83a9b9692933aff8349157c88a413df9d02c4e5`. The manifest pins image and
question/answer bytes. These 25 questions are a small regression sample,
not an estimate of full-benchmark accuracy. Their images use 65–1,014 visual
tokens after preprocessing.

With lettered answer options, native scored 22/25 and reference 23/25.
The changed answer was row 48: both reasoning traces identified diagram
label B as the new moon, but native returned `B`, while reference returned
`D`, the answer-option letter containing label B. This is observable
answer-format sensitivity, although both runs recognized the visual fact.
The same complete sample was also rerun with numbered options to separate
diagram labels from answer identifiers. Literal-response matches were 22/25
native and 21/25 reference: one native answer and two reference answers used
Markdown emphasis; both runs added an explanation on row 84. Extracting the
leading option number, allowing optional Markdown emphasis and trailing
explanation, gave **23/25 for each**. Both answered the moon question correctly.
Rows 32 and 40 changed answers, with one gain and one loss for native; this
is equal aggregate accuracy on this sample, not per-question equivalence.
The unmodified messages and both scoring rules are retained in
[measurements.json](2026-09-18-glm-vision/measurements.json).

The measurements support the implemented input path and identify remaining
numerical sensitivity. They do not establish general vision, OCR or chart
quality equivalence. Broader evaluation remains useful before depending on
exact answer reproduction across backends.

## Reproduction

The adjacent [harness directory](2026-09-18-glm-vision/) contains the fixture
generator, image preparation utility, answer clients, data manifest,
diagnostic patch and measured results. Run from the repository root.
The clients use localhost port 18080 and the hybrid model ID above.
`DGPP_VISION_FIXTURES` defaults to `/tmp/dgpp-vision-quality`.

Build the host preparation utility, then download and verify the sample.
Set `CKPT` to the oracle checkpoint above and `ORACLE_PYTHON` to a Python
environment with CUDA PyTorch and NumPy. Export reference rows while no
server is occupying the test GPU:

```bash
c++ -std=c++20 -O2 -Isrc \
  benchmarks/results/2026-09-18-glm-vision/prepare.cpp \
  build-ci/libdgpp_serve.a build-ci/libdgpp_loaders.a build-ci/libdgpp_core.a \
  -pthread -o /tmp/vision-prepare
python3 benchmarks/results/2026-09-18-glm-vision/prepare_fixtures.py \
  --prepare-bin /tmp/vision-prepare --checkpoint "$CKPT" --python "$ORACLE_PYTHON"
```

To check a native encoder fixture, use its dimensions from `prepared.json`,
write traces with `glm_vision_check`, then run the oracle with the same RGB:

```bash
build-ci/glm_vision_check "$CKPT" 560 336 /tmp/moon.bf16 /tmp/moon-trace \
  /tmp/dgpp-vision-quality/ai2d/48.rgb
"$ORACLE_PYTHON" tools/glm_vision_reference.py "$CKPT" 560 336 /tmp/moon.bf16 \
  --rgb /tmp/dgpp-vision-quality/ai2d/48.rgb --device cuda \
  --trace-dir /tmp/moon-trace --isolate-layers
```

Build the reference-injection binary in an isolated test checkout, then
remove the diagnostic patch and rebuild the ordinary server:

```bash
git apply benchmarks/results/2026-09-18-glm-vision/reference_override.patch
cmake --build build-ci -j 4 --target dgpp_serve_app
cp build-ci/dgpp-serve /tmp/dgpp-serve-vision-reference
git apply -R benchmarks/results/2026-09-18-glm-vision/reference_override.patch
cmake --build build-ci -j 4 --target dgpp_serve_app
```

Copy the exported `*.bf16` files to the same fixture directory on every
rank. Boot the ordinary binary for the native run, and the diagnostic binary
with `DGPP_VISION_REFERENCE_DIR=/tmp/dgpp-vision-quality` for the reference
run. Use distinct log directories and stop each world before starting the
next. Against each world, run:

```bash
python3 benchmarks/results/2026-09-18-glm-vision/check_synthetic.py /tmp/synthetic.jsonl
python3 benchmarks/results/2026-09-18-glm-vision/check_ai2d.py /tmp/ai2d.jsonl
python3 benchmarks/results/2026-09-18-glm-vision/check_ai2d.py /tmp/ai2d-numbered.jsonl --numbered
```

The recorded world log directories and within-world operation-stream hashes
are listed in `measurements.json`. Compare hashes across ranks within each
world; different native/reference transcripts need not hash identically.
