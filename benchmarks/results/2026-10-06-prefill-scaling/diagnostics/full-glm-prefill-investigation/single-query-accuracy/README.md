# Single-query fused prefill

The initial full-GLM fusion (`21b91a3`) preserved the previous cuBLAS path for one-query tiles because their score bits could differ. Bitwise agreement with that implementation is not an accuracy oracle. This follow-up removes the restriction after comparing both paths with FP64 calculations from the exact quantized inputs.

## Numerical comparison

| Check | Result |
| --- | --- |
| Cases | 44: 2,053 / 8,192 / 65,539 / 262,144 pools; several seeds; ordinary FP8 values, the finite E4M3 range and exact ties; ReLU on/off |
| Score differences | 12 cases, all at 2,053 pools; larger tested contexts agree exactly |
| Largest normalized RMS error vs FP64 | Previous GEMM: 1.394e-7; fused: 1.401e-7 |
| Selected-token disagreements vs FP64 | Zero for either implementation in all 44 cases |
| Regression criteria | Bound score error relative to FP64; certify any selection change at the cutoff using the measured error; retain bitwise multi-query and fixture selection checks |

The previous GEMM is slightly closer to FP64 in the 12 differing small-context cases. The errors remain at FP32 rounding scale and did not change the selected tokens. These measurements support removing the blanket fallback; they do not establish that the fused scores are uniformly more accurate.

Evidence: [initial comparison](fp64-comparison.log), [initial summary](fp64-summary.json), [final single-query summary](final-fp64-summary.json), [final suite with actual one-query calls](unrestricted-dsa-suite.log), [source and binary manifest](candidate.json), [patch](candidate.patch).

## Validation

| Check | Result | Evidence |
| --- | --- | --- |
| DSA suite | 51 tests passed | [Log](unrestricted-dsa-suite.log) |
| Score memory safety | Zero errors | [Log](single-query-memcheck.log) |
| Layer/tail memory safety | Zero errors | [Log](tail-memcheck.log) |
| CUDA synchronization | Zero errors | [Log](single-query-synccheck.log) |
| Full-GLM forward, prefill, decode, TP and engine regression | Six tests passed using existing reference fixtures | [Log](full-glm-regression.log) |
| Four-node full-model replay | Passed all 18 prompt/output/usage comparisons; zero swap and identical rank operation streams | [Driver](compare_server.py), [log](compare-server.log) |

The completed full-model replay repeats 2K/8K/32K cold prefill and 32K/64K/128K context measurements. The two previous full-GLM default launches are archived intact. No KV, slot, prefix-cache or reserve settings changed. GPU tests and serving launches use a zero-swap cgroup; OS swap remains enabled.

The follow-up is committed as [`0b10cbe`](https://github.com/HawkBearPig/dgpp/commit/0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35). The tested serving binary remains unchanged after the commit.

Final replay timings (same prompts and configuration; one cold request per context):

| Context | Initial fusion with fallback | Final fusion |
| --- | --- | --- |
| 32K | 74.488 s | 75.099 s |
| 64K | 160.845 s | 161.000 s |
| 128K | 367.143 s | 366.933 s |

[Full-model comparison](full-model-comparison.json) · [Validation manifest](validation.json)

The [retention audit](retention-scope.json) checks every retained full-GLM report: all requests stay within the unchanged short-prompt path, or record an above-limit skip. Other families keep their existing implementations.
