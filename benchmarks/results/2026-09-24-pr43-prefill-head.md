# Qwen prefill vocabulary head: PR #43 follow-up

PR #43 avoids computing vocabulary logits for unused rows of a plain
prefill. Its original one-row call changed the head's accumulation order:
the scale-GEMM dispatcher uses GEMV through 128 rows and a dense tensor-core
kernel above that boundary. Their dequantized weights agree, but their FP32
sums are not bitwise equivalent.

The follow-up adds a last-row option to the FP32 scale-GEMM launcher. It
selects the kernel with the original row count, then advances the input and
output pointers to the last row and executes that row using the selected
kernel. Earlier output rows remain untouched. Qwen retains the original
PR's exclusions for streaming-MMA chunks, BF16 heads, grouped prefills,
decode and all-row diagnostics, and its `DGPP_PREFILL_HEAD_ALL_ROWS=1`
comparison switch.

Regression coverage:

- Kernel comparisons require bitwise equality with the full product at
  1/4/5/17/127/128/129/255/256/257 rows, with streaming MMA enabled and
  disabled, production hidden width 2560 and ragged width 1000. Padded
  input/output strides and sentinels check that only the final output row
  is written.
- Qwen fixture comparisons require prefill logits and hidden states to
  match the final row of a full forward at 1/4/5/8/9/16/17/127/128/129/257
  tokens, with BF16 heads and both FP8 head settings. A second CTest case
  exercises the full-head environment switch.

Validation on the PR integrated with master `92cfc68`:

- The complete native build passed with CUDA 13 and warnings treated as
  errors (`cmake --build build-ci -j 4`). The CI profile used `-g0` to limit
  build disk usage.
- `ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure -j 4`:
  all 24 tests passed in 119.43 seconds.
- The Qwen synthetic checkpoint was generated and `qwen_plan_check` passed.
- GPU and fabric validation are pending. All four cluster GPUs have an
  active GLM serving deployment; the repository requires idle hardware for
  GPU/RDMA tests. The regression binaries and an isolated head benchmark
  are built, but have not been executed on the GPU.

The original PR's TTFT measurements describe its submitted implementation;
they are not measurements of this follow-up. Merge readiness still requires
the GPU regression checks and fabric validation.
