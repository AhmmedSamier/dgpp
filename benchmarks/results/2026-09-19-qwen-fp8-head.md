# Qwen FP8 vocabulary-head dispatch

## Scope

Prepared on `codex/fp8-vocabulary-head` from upstream
`444441feed0c4d7c256ff7ff7868b587823b737f`.

The FP8 vocabulary head opts into the existing streaming tensor-core kernel
above `dense_gemv_rows()` when the rows fit `max_decode_rows_`. The default
threshold is four: supported, aligned heads at five through sixteen rows
can reuse weight tiles instead of rereading them per GEMV chunk. Small
heads and heads above the configured decode-row ceiling retain the old
lowering. Short prefill calls within the interval are also affected because
the head is shared. Existing kernel eligibility checks retain fallback for
unsupported shapes. `DGPP_DENSE_GEMV_ROWS=256` provides the old head lowering
for supported decode rows.

This changes FP32 accumulation order, not weight quantization or logit
storage. It does not raise the decode-row ceiling, alter prefill chunking,
change the BF16 head, or add telemetry or deployment settings.

## Tests added

- `scale_gemm_f32_fp8_head_numerics`: K=2560, ragged N=257, rows 4/5/8/16;
  compare unrounded FP32 logits to FP64 accumulation of BF16-rounded
  dequantized weights, plus same-shape bitwise repeatability and GEMV control.
- `qwen_decode_fp8_head`: teacher-forced fixture at 4/6/8/12/16 rows,
  capturing and replaying the target graph. Inspect the kernel that writes
  the vocabulary-logit buffer to require streaming MMA above the threshold.
  This dispatch assertion fails with the old head. Existing logit/near-tie
  checks and the 0.02-nat mean-NLL bound cover the resulting predictions.
- `qwen_decode_fp8_head_row_independent`: the same FP8 checks with the
  threshold at 256, requiring no streaming head.
- `qwen_engine_fp8_head_4` and `qwen_engine_fp8_head_256`: FP8 versions of
  the two-rank wide MTP slot-reuse/continuation fixture, including the
  existing exact transcript checks in the row-independent lane.

The FP8 loader setting is scoped to each test's model construction and
restored afterwards. No new public configuration or request fields are added.

## Validation status

The contribution guide and numerical policy were re-read. CMake configured
successfully with CUDA 13 and the external Spark cross toolchain. The four
affected C++/CUDA translation units cross-compiled for AArch64/SM121a.
Touched ranges were formatted with clang-format 18 using the repository
style. `git diff --check` passed. These are compile checks, not linked-binary
or target-execution results.

Both production inference ranks were running during preparation. No service
was stopped, deployed or reconfigured. GPU/RDMA tests were not run alongside
production. Local disk headroom also precluded a full fresh build.

On idle native hardware, the focused selection after a full build is:

```bash
cmake --preset ci
cmake --build --preset ci -j 4
ctest --test-dir build-ci --output-on-failure -j 1 \
  -R '^(qwen_forward_fixture|qwen_decode_fp8_head.*|qwen_engine_fp8_head_.*|scale_gemm_test)$'
```

Before presenting this as validated for merge:

1. On reserved idle GB10 hardware, configure `ci`, build all targets, and run
   the complete suite serially, retaining failures and checkpoint skips.
   To select the added gates during investigation, also include
   `qwen_forward_fixture` in the CTest selection: `DEPENDS` orders selected
   tests but does not automatically select that fixture.
2. Execute the new graph-dispatch test against both this branch and the
   parent; confirm it passes here and fails on the parent dispatch.
3. Run matched real-model teacher-forced comparisons under `docs/numerics.md`,
   including repeated same-binary runs. Exercise wide verification rows;
   scalar-only scoring would miss this optimization. Include short prefill.
4. Measure the vocabulary head at the real TP vocabulary widths and K=2560,
   then matched end-to-end concurrency/MTP runs against this exact parent.
   No speedup for this branch is claimed yet. Historical measurements of
   the bundled optimization are not evidence for this upstream revision.
5. Launch the participating ranks with an explicit supported deployment
   config, run the relevant API checks, shut down with the same config and
   compare all rank op-stream MD5s. Restore and verify production afterwards.

PR creation is intentionally on hold. Hardware and fabric gates above are
pending, not waived or counted as passes.
