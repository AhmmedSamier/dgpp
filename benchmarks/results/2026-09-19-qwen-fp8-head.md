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

- `scale_gemm_f32_fp8_head_numerics`: K=2560, ragged N=257/4097, rows 4/5/8/16;
  compare unrounded FP32 logits to FP64 accumulation of BF16-rounded
  dequantized weights, plus same-shape bitwise repeatability and GEMV control.
  N=4097 uses the same eight-warp specialization as the production TP2
  vocabulary head. For its MMA cases, captured graph inspection requires
  one streaming kernel with a 256-thread block before executing it. The
  exhaustive oracle includes the ragged final block; this is specialization
  coverage, not a full-sized production vocabulary or checkpoint test.
- `qwen_decode_fp8_head`: teacher-forced fixture at 4/6/8/12/16 rows,
  capturing and replaying the target graph. Inspect the kernel that writes
  the vocabulary-logit buffer to require streaming MMA above the threshold.
  This dispatch assertion fails with the old head. Existing logit/near-tie
  checks and the 0.02-nat mean-NLL bound cover the resulting predictions.
  The default-threshold lane also compares short-prefill logits at lengths
  1/4/5/8/16/17 with a model whose four-row decode ceiling retains the old
  head dispatch. Both models retain the same dense lowering threshold;
  their final hidden states must be nonempty and bitwise equal. Eight
  prompts per length check finite logits, relative L2, top-1/near-ties,
  mean NLL delta per length, same-shape bitwise repetition, and bitwise
  logits outside the optimized interval.
- `qwen_decode_fp8_head_row_independent`: the same FP8 checks with the
  threshold at 256, requiring no streaming head.
- `qwen_engine_fp8_head_4` and `qwen_engine_fp8_head_256`: FP8 versions of
  the two-rank wide MTP slot-reuse/continuation fixture, including the
  existing exact transcript checks in the row-independent lane.

The FP8 loader setting is scoped to each test's model construction and
restored afterwards. No new public configuration or request fields are added.

## Validation on 2026-09-20

The contribution guide and numerical policy were re-read. Preparation on
x86 cross-compiled all affected translation units for AArch64/SM121a.
The reserved two-Spark window then ran a fresh native `ci` configure and
full build on GB10 with CUDA 13. GPU/RDMA checks ran serially after both
production ranks stopped, never alongside production serving.

The tested source was `ec829b24f317ca430d01233524a15de911f0440e` plus the
runtime graph-inspection fix in this record. The tested `qwen_decode_test.cpp`
SHA256 is `7f7b29813d82b8a50c733d5eba37b9aa0bf2ef3d6952dd1c196d2fd07c22e2c8`.
The optimization itself was unchanged during validation.

An isolated review of `5ed96cf` found missing assertions for the eight-warp
kernel and short-prefill logits; the follow-up tests now run successfully.
The first focused hardware run exposed an inspection issue: a driver-loaded
kernel in the captured graph returned `cudaErrorInvalidDeviceFunction` from
`cudaGraphKernelNodeGetParams`. The test now skips that unsupported runtime
query, clears its error, and still requires the statically linked FP8 head
kernel. Other errors remain fatal. A full native rebuild preceded rerunning.

- [Focused gates](2026-09-19-qwen-fp8-head/raw/focused-ctest.txt): all six
  selected tests passed, including the fixture generator, both decode lanes,
  both FP8 MTP loopback lanes, and `scale_gemm_test`.
- [Full suite](2026-09-19-qwen-fp8-head/raw/full-ctest.txt): 105 passed,
  zero failed, ten checkpoint-dependent skips; 653.52 seconds. The skip list
  is retained in the log and is not counted as model validation.
- [Negative control](2026-09-19-qwen-fp8-head/raw/negative-ctest.txt): replace
  only `src/models/qwen/forward.cpp` with upstream parent `444441f`, rebuild,
  and run the new test. It fails on the expected head-dispatch assertion.
  Restore the candidate source and rebuild: the
  [same test passes](2026-09-19-qwen-fp8-head/raw/candidate-recheck.txt).
- FP8 teacher-forced fixture: 368 rows, worst relative logit L2 `0.0023704`,
  zero top-1 near-ties, mean NLL delta `5.4114e-05` nat. Row-independent
  control: zero L2 and zero NLL delta.
- Short-prefill comparisons: mean NLL delta `2.52229e-08` nat at length 16,
  zero at lengths 1/4/5/8/17. Finite-logit, hidden-state equality, top-1,
  bitwise repeatability and unchanged-boundary assertions all passed.

### Production-size kernel measurement

The [standalone probe](2026-09-19-qwen-fp8-head/raw/head-bench.cpp) links the
native CI kernel library. It uses synthetic weights and activations at the
real TP2 vocabulary-head dimensions, N=124160 and K=2560. Five alternating
pairs of twenty products report median CUDA-event time after warming both
paths. The matrix exceeds L2. This measures the head, not end-to-end serving,
and is not a Release-build throughput claim.

| Rows | GEMV chunks, ms | Candidate dispatch, ms | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 1.936709 | 1.952592 | 0.9919 |
| 4 | 1.939854 | 1.936296 | 1.0018 |
| 5 | 3.885965 | 1.824350 | 2.1301 |
| 8 | 3.919661 | 1.913115 | 2.0488 |
| 12 | 5.878384 | 1.960000 | 2.9992 |
| 16 | 7.857568 | 1.994789 | 3.9390 |

[Raw output](2026-09-19-qwen-fp8-head/raw/head-bench.txt). All output logits
were finite and equal to the GEMV control for these synthetic inputs;
sampled columns across every row, scale/block boundaries and the shard tail
also agreed exactly with FP64. The randomized exhaustive FP64 fixture is
separate; exact agreement on this synthetic probe is not a claim of general
bitwise equivalence between GEMV and MMA.

### Two-node serving and restoration

A separate explicit deployment configuration ran the candidate native CI
binary with `nvidia/Qwen3.8-Flash-Next-NVFP4`, FP8 dense weights, concurrency
4, MTP depth 3, decode graphs, BF16 KV and a 1-GiB prefix cache.
[All nine API checks passed](2026-09-19-qwen-fp8-head/raw/candidate-serving-api.txt).
Shutdown used the same configuration; both participating ranks produced
op-stream MD5 `095726af690de19ef6cc93c953a00c16`.

Both production ranks were restored to their original release and unchanged
configuration. Their binary SHA256 remained
`ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`.
The service reported no engine failure and an empty queue. Both smoke
requests returned `READY`; the repeated 1137-token prompt reused 1128 tokens,
and the cache retained its configured 148 slots.

The maintenance wrapper restored production on both preliminary failures:
a self-SSH host-key check was replaced with a local process check, and the
runtime graph-inspection issue above was fixed before the successful run.
Full local operational logs remain under `artifacts/fp8-head/server-validation/`.

## Remaining evidence

Native execution, the complete available suite, the negative control,
production-size kernel timing and two-node API/op-stream checks are complete.
The subsequent [Release-build end-to-end comparison](2026-09-20-qwen-fp8-head-e2e.md)
measured 11.26% higher C4/MTP3 request-wall throughput on the fixed five-class
workload, with C1 effectively unchanged. Greedy transcripts varied within
the baseline as well as across builds; the record retains that limitation.
Matched real-checkpoint teacher-forced comparisons covering wide verification
rows and short prefill remain outstanding. Fixture NLL, throughput and the
serving API smoke are not substitutes for that numerical comparison.

Draft [PR #10](https://github.com/HawkBearPig/dgpp/pull/10) is open.
No production release was updated.

The [review follow-up](2026-09-21-qwen-fp8-head-review.md) makes this path
opt-in with `engine.fp8_head: "mma"` and retains GEMV by default. Results
above apply to the earlier unconditional implementation. `raw/head-bench.cpp`
is an archived, one-off reproducer for those measurements, not a maintained
benchmark target.


Review follow-up correction: the original short-prefill control requested a
four-row decode capacity, but `SessionModel` floors it at eight. Therefore
the 5/8-row cases in that original run compared the same MMA head dispatch;
they did not establish GEMV-versus-MMA numerical agreement. The explicit
`fp8_head_mma = false` control added during review provides that comparison,
and the minimum-capacity control now checks the effective 8/9 boundary.
See the follow-up record for its native results.
