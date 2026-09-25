# QSA warp-prefill review and regression coverage

PR [#47](https://github.com/HawkBearPig/dgpp/pull/47), initially reviewed at
`89c8892523a3f3ddec0ffd1b8df6737980671db3`, adds a one-warp tensor-core QSA
prefill kernel. Review found no actionable correctness defect. The follow-up
adds coverage for production's eight-split reference and for model prefills
that cross the 128-row dispatch threshold; it does not change the kernel.

## Committed coverage

`qsa_warp_prefill_matches_the_partial_kernels` now compares against 1, 3 and
8 splits for all seven existing head shapes and both cache-block sizes.
The lists include empty rows, lengths 32 and 256, and the existing tile-edge,
long-list and high-dynamic-range cases. Eager execution and CUDA graph replay
both compare every row and head, giving 42 shape/block/split configurations.

`qwen_qsa_prefill` launches fresh processes with the default dispatch and
with `DGPP_QSA_WARP=0`, holding the remaining environment and weights fixed.
Its synthetic model has 24 query heads, two KV heads and an indexer budget
of 2048, so each KV group has twelve query heads and the old path uses
eight splits. Each arm checks exact forward repeats, session-prefill equality
with the cold forward's last row, and four subsequent decode steps.

The comparison checks every logit at 127, 128, 129, 256, 513 and 1024 rows.
The short case must remain exact; longer cases must stay below 0.01 relative
L2 per row, and changed winners must be within two BF16 ulps of the reference
winner. At least one long case must differ, so a disabled dispatch cannot
silently pass. A negative control forcing both arms onto the partial kernels
failed with the expected "fixture did not exercise" error.

## Validation

GB10, CUDA 13.0.88, CI preset with warnings as errors and
`DGPP_ENABLE_IBV=OFF`. The selected GPU tests ran serially while the existing
four-node GLM deployment was stopped.

- Built `qsa_test`, `qwen_forward_test` and `qwen_decode_test`.
- Nine selected CTest cases passed: fixture generation, forward smoke,
  YaRN smoke, context-limit checks, three decode variants, the new QSA
  model comparison, and `qsa_test` (all nine internal kernel tests passed).
- Kernel comparison: worst per-row/head relative L2 **0.00252**, below 0.01.
- Compute Sanitizer memcheck and racecheck: zero errors or race hazards.
- Rebuilt and reran the same nine CTest cases with current `master`
  (`ce80983f8d317c4b586f7fafb4a83acf27150fd5`) merged locally: all passed,
  with the same numerical results. The forced-partial negative control
  also failed as expected on that integration.

| Prompt rows | Worst logit row relative L2 | Changed winners | Largest winning-logit gap (BF16 ulps) |
|---:|---:|---:|---:|
| 127 | 0 | 0 | 0 |
| 128 | 0.004681 | 0 | 0 |
| 129 | 0.004171 | 0 | 0 |
| 256 | 0.003194 | 1 | 0.02095 |
| 513 | 0.004134 | 0 | 0 |
| 1024 | 0.005590 | 1 | 0.04750 |

An additional review-only kernel sweep covered every group size from 1 to
16, KV-head counts 1/2/3, both cache-block sizes, 1/3/8 reference splits and
reversed token lists: 288 configurations passed, with worst relative L2
0.00272. This larger sweep is separate from the committed regression suite.

These synthetic checks do not independently establish real-checkpoint
quality or throughput. The author's two-node quality and latency evidence
is in the PR; those measurements were not rerun for this test-only follow-up.

## Reproduction

On idle test hardware:

```sh
cmake --preset ci
cmake --build build-ci -j 2 --target qsa_test qwen_forward_test qwen_decode_test
ctest --test-dir build-ci --output-on-failure -j 1 \
  -R '^(qsa_test|qwen_qsa_prefill|qwen_forward_fixture|qwen_forward_smoke.*|qwen_forward_cross_limit|qwen_decode_test|qwen_decode_fp8.*)$'
DGPP_TEST_FILTER=qsa_warp_prefill compute-sanitizer --tool memcheck \
  --error-exitcode 99 build-ci/qsa_test
DGPP_TEST_FILTER=qsa_warp_prefill compute-sanitizer --tool racecheck \
  --error-exitcode 99 build-ci/qsa_test
```
