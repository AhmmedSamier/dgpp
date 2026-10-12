# Shared prefill and decode execution: work in progress

Follow-up for issue #104 and draft PR #108. This record is **not a merge
acceptance report**. Final serving and throughput verification is pending.
The older opt-in policy did not resolve the enabled-path regression.

The implementation shares text prefill and decoder verification in one Qwen
model walk, including multiple readers. Dense and expert projections are
shared, while request-indexed state kernels retain each decoder's rollback
rows, MTP state and sampling state. Grouped prefills spend one global budget.
Older decoders retain the busy-budget bound. Later decoders can share a
reader's idle-budget chunks without forcing it onto a smaller budget.

Qwen graph serving now resolves omitted early-admission configuration to true.
Other backends keep their existing default; the wire journal still carries a
resolved boolean. Mixed elapsed time is exposed separately because it belongs
to both logical phases; total engine time is prefill + decode - mixed.

## Iterations

One GB10, four slots, NVFP4 routed experts, FP8 dense weights, BF16 KV, MTP3,
full reservation, 1 GiB prefix cache. Timing requests disable the cache.
Workload: `serve_admission_bench.py sustained_idle`, 8192 words, eight arriving
requests at 250 ms intervals, each returning 128 tokens. Three deterministic
trials per executable, with a warmup on each fresh deployment. Equal budgets
are 256/256; split budgets are 256/2048. See the per-run configuration and
executable hashes. No compiler or profiler runs during timed requests.

| Iteration | Equal candidate mean, s | Equal control mean, s | Split candidate mean, s |
|---|---:|---:|---:|
| v8b: shared grouped-prefill projections | 21.738 | 21.000 | 14.251 |
| v9b: multiple prefills and decoders in one walk | 21.695 | 20.986 | 14.333 |
| v10: batch decoder state kernels within the walk | 21.541 | 20.875 | 14.329 |

The equal-budget gap remains about 3%. The change is not yet claimed to be
throughput-neutral. All 27 paired request bodies, outputs and usage records
match in v9b and v10. In v8b, one long request's single output token differs;
that iteration is retained as evidence, not substituted for the later runs.
Earlier broad-v7 measurements are provisional and predate the grouped and
batched-state changes.

Runtime checkpoints: `b1b54df5` (shared and grouped walks), `eb9365b9`
(batched decoder state kernels), `e17cf1eb` (defer intermediate prefill outputs). The contributor fork rejected a push; these
commits are on `HawkBearPig/dgpp:perf/104-shared-prefill-decode`.

## Correctness checks and sampling diagnosis

The scheduler passes 104 tests under ASan/UBSan. Nine GPU gates cover model and
image prefill, mixed MTP-off and scheduled-depth execution, compact sampling,
forced gather fallback, and wider MTP3 rows. The v11 build passes all four rebuilt host suites, all nine GPU gates,
CUDA memcheck with zero errors, and the alternate GDN snapshot form. Full
serving checks are still being collected.

A deliberately extreme full-vocabulary temperature-100 comparison across
*different prefill chunk shapes* produced different first tokens in both the
unchanged runtime and the candidate. Both use seed 4567 and counter zero.
Captured prefill logits differ by relative L2 0.001548; the corresponding
probability distributions have total variation 0.000005056. Their sorted-CDF
sampling orders choose different tokens. Host replay exactly reproduces the
two choices, each consuming one draw. This is not evidence of a new RNG or
state corruption bug.

The cross-chunk transcript gate now uses temperature 0.7, top-k 7, top-p 0.9
and asserts the forced fallback actually occurs. Existing full-vocabulary
physical-remapping checks retain their original parameters at matched prefill
shapes. Reproduce the captured first-token diagnosis from the repository root:

```sh
c++ -std=c++20 -O2 -Isrc benchmarks/results/2026-10-12-shared-prefill-decode/replay_sample.cpp -o /tmp/dgpp-replay-sample
/tmp/dgpp-replay-sample benchmarks/results/2026-10-12-shared-prefill-decode/sampling-logits.txt
```

`campaign.py` is a reusable matched serving driver. Supply candidate/control
executables, an output directory, and the production configuration/executable
that it restores in `finally`. It requires idle production and checks native
prefill overlap, request counts, usage, drain, KV release, resolved early
admission and rank operation-stream agreement. Separate API/cache, physical
reservation, soak and production-identity checks remain necessary.
