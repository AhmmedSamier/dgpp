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

## Implementation choice

The author's intent is to avoid the unused vocabulary projections while
preserving the final logits. A specialized kernel would execute a single
row with the same tensor-core accumulation order while reducing padded
row work. That is a possible future optimization, not a demonstrated
improvement over this follow-up.

An isolated head benchmark on one idle GB10 used the production TP=2
shape: hidden width 2560 and 124160 local vocabulary entries, synthetic
BF16 activations and block-FP8 weights, CUDA events, a warmup and the
median of five calls per mode. At 8192 input rows:

| Head implementation | Median time | Logits differing from the full head |
| --- | ---: | ---: |
| Full dense head | 236.237 ms | 0 |
| Selected row, original dense kernel (this change) | 2.086 ms | 0 |
| Selected row, existing 16-row tile kernel | 2.423 ms | 0 |
| Selected row, GEMV (submitted PR) | 1.946 ms | 123683 |
| Selected row, streaming MMA | 1.866 ms | 113446 |

The existing smaller tile was also slower at 129 and 4096 rows. The
current implementation saves 234.151 ms in this head measurement while
preserving every output bit. The submitted GEMV saves only another
0.140 ms at this shape and violates the intended numerical contract.
Reusing the original kernel is therefore the supported implementation
choice. A new kernel should be justified by measurements and the same
bitwise gates before replacing it.

These are isolated head timings, not end-to-end TTFT measurements. The
original PR's TTFT table describes its submitted implementation; it is
not a measurement of this follow-up. No published throughput row is
updated from this microbenchmark.

The [benchmark source](2026-09-24-pr43-prefill-head/head_kernel_bench.cpp)
and [raw measurements](2026-09-24-pr43-prefill-head/head-kernel-comparison-tp2.log)
are retained. From a built checkout, reproduce with:

```bash
c++ -O2 -std=c++20 -I src -I /usr/local/cuda/include \
  benchmarks/results/2026-09-24-pr43-prefill-head/head_kernel_bench.cpp \
  build-ci/libdgpp_kernels.a build-ci/libdgpp_core.a \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lcudart -lcublasLt -o /tmp/pr43-head-bench
/tmp/pr43-head-bench
```

## Validation

The PR was integrated with master `92cfc68`. The existing four-node GLM
service was stopped before GPU and RDMA validation; these ran serially
on idle hardware.

- The complete native build passed with CUDA 13 and warnings treated as
  errors (`cmake --build build-ci -j 4`). The CI profile used `-g0` to limit
  build disk usage.
- All 24 host tests without the checkpoint label passed in 119.43 seconds.
- The Qwen synthetic checkpoint was generated and `qwen_plan_check` passed.
- `scale_gemm_test` passed, including the new bitwise and output-boundary
  checks. Both Qwen prefill-head regressions passed. Their capacity is
  512 so the 257-token case does not accidentally compare a 256+1 chunked
  prefill with a single full forward.
- A negative control substituted the submitted one-row GEMV call into
  the new model regression. It failed at 129 tokens with an FP8 head and
  streaming MMA disabled, demonstrating that the regression catches the
  original issue.
- The remaining GPU/RDMA suite completed 124 CTest cases. Twelve
  reference tests initially lacked NumPy in the system interpreter; all
  twelve passed when rerun with the repository virtual environment's
  site-packages. The unrelated `mimo_chat_template_test` remains failing
  in three cases; a separately built clean checkout of master `92cfc68`
  fails identically.
- The nine checkpoint host tests added eight passes and one unrelated
  `qwen_chat_template_test` failure. The Qwen tool grammar refuses a
  newline token in `tool_call_and_response`; the same clean master build
  fails identically. Neither template nor grammar code is changed here.

Across all 160 CTest cases, 158 pass after supplying NumPy and the two
pre-existing template tests fail. Their baseline reproductions and the
suite logs are retained alongside this record.

## Two-node serving comparison

The patched CI binary served `nvidia/Qwen3.8-Flash-Next-NVFP4` on two GB10s
with FP8 dense weights, an MMA head, BF16 KV, MTP depth 1, concurrency 4,
KV capacity 65536, and prefix caching disabled. Separate fresh worlds
used the optimization and `DGPP_PREFILL_HEAD_ALL_ROWS=1`, which the
launcher propagated to both ranks. Both used the same deployment config
and sequential requests with temperature 0, seed 43, `ignore_eos: true`,
32 generated tokens and five reported alternatives per token.

All four prompts (5, 196, 4223 and 8423 tokens) matched exactly in text,
token log-probabilities, top-five log-probabilities, usage and finish
reason. The longer prompts used the server's 256-token prefill ticks.
This checks real-model numerical equivalence across chunk boundaries;
the single-request wall times are smoke measurements, not a cold-TTFT
benchmark suitable for a published performance claim.

Clean shutdown produced identical operation streams across both ranks
within each world:

- Optimized: `09ebabb68ae32f39dce552177818f852`.
- Full head: `b539b3ef744ec39f242d3c07022f81fd`.

The [request driver](2026-09-24-pr43-prefill-head/serve_compare.py),
[deployment configuration](2026-09-24-pr43-prefill-head/qwen-tp2.json),
[optimized responses](2026-09-24-pr43-prefill-head/optimized-responses.json)
and [full-head responses](2026-09-24-pr43-prefill-head/all-rows-responses.json)
are retained.

A third optimized two-node world enabled a 1 GiB prefix cache and passed
all checks in `scripts/serve_api_check.py`: one-shot and streamed stops,
`n=2`, forced and banned tokens, invalid bias rejection, cached-token
accounting and reasoning-token accounting. Its operation streams matched
at `3f65500c6a77687719f13b8afb91f58c`. The
[API output](2026-09-24-pr43-prefill-head/api-check.log) is retained.

After validation, the original four-node
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` deployment was restored. Its resolved
configuration is identical to the saved original, the serving binary's
SHA-256 is unchanged, all four ranks are running, and `/health` and
`/v1/models` return HTTP 200 with the original model.
