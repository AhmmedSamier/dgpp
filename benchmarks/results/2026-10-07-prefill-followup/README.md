# Cold-prefill follow-up

Investigation requested after the completed matrix was published. The original campaign remains an immutable measurement record. GLM Flash configuration comparisons use the same binary; kernel comparisons record separate baseline and candidate hashes. The main benchmark document now uses the verified replacement cold-prefill values and labels affected measurements that have not been rerun.

## GLM Flash on two nodes

Both published two-node presets resolve to 256-token busy **and idle** prefill budgets. The four-node preset explicitly uses a 2,048-token idle budget. Controlled fresh-server comparisons change only `prefill_idle_budget_tokens` to 2048; the busy budget, KV allocation/format, prefix cache and model weights stay fixed.

Medians of two cold probes at each approximate length:

| Configuration | Target input | Idle 256 (s) | Idle 2048 (s) | Speedup |
|---|---:|---:|---:|---:|
| 160K FP8 KV, 4 slots | 2K | 4.828 | 2.057 | 2.35× |
| 160K FP8 KV, 4 slots | 8K | 19.420 | 8.384 | 2.32× |
| 160K FP8 KV, 4 slots | 32K | 78.893 | 35.312 | 2.23× |
| 256K FP8 KV, 2 slots | 2K | 4.788 | 1.976 | 2.42× |
| 256K FP8 KV, 2 slots | 8K | 19.317 | 8.178 | 2.36× |

All ten paired probes match prompt hashes, one-token outputs, finish reasons and usage. Every launch passes process/cgroup zero-swap checks, operation-stream identity and clean shutdown. This is a prefill timing comparison, not a new quality evaluation.

The small idle chunks repeat the model walk and tensor-parallel boundaries more often, with fewer rows available to amortize weight access and kernel work. The same allocation already supports 2,048-row forwards. Larger idle chunks improve isolated cold prefill but can increase the wait for cancellation or a new request arriving during an in-flight chunk. Published measurements remain valid for the original 256-token setting; they are not a hardware-only two-versus-four-node comparison.

- [160K comparison](glm-flash-w2-idle-budget-comparison.json)
- [256K comparison](glm-flash-256k-w2-idle-budget-comparison.json)
- [Audited published timings and source records](published-prefill-audit.json)

The two-node example now explicitly selects the tested 2,048-token idle budget. The automatic 256-token budget dates to `a20b0a20` (2026-09-23); the four-node template already had its larger idle override. No KV allocation was reduced.

## Full GLM-5.3 int4/int8

Full GLM uses internal 2,048-row prefill chunks and does not have the Flash preset's 256-token idle bottleneck. Its BF16 published prefill improved from September's 4.785 / 24.209 / 169.138 s to 3.493 / 16.219 / 70.249 s at approximately 2K / 8K / 32K. Those historical prompts differ, so this establishes direction rather than an exact speedup.

Fresh rank-0 profiles of a single cold ~8K request, on the original fixed binary:

| KV format | Profiled wall (s) | Core attention kernels | Packed GEMMs | Collectives | Index scores/selection |
|---|---:|---:|---:|---:|---:|
| BF16 | 17.293 | 32.0% | 34.3% | 15.9% | 3.1% |
| FP8 | 18.325 | 33.9% | 29.8% | 15.1% | 3.1% |
| FP4 | 25.509 | 53.2% | 21.2% | 11.4% | 2.1% |

Percentages divide summed kernel durations, which can overlap; they are not exclusive GPU utilization. The last request burst excludes startup/calibration and includes its one output token. Actual prompt lengths differ between deployments. Profiled timings are diagnostic and do not replace unprofiled table measurements.

The profiles use the intended kernels. The recently fixed score/selection path is about 3% of kernel time at this length, rather than the dominant cost. BF16/FP8 still spend most time in attention, packed matrix products and collectives. Full GLM has 78 layers and a 6,144-wide hidden state, versus Flash's 45 layers and 4,096-wide state with hybrid attention. These are different workloads; the evidence does not establish that the remaining implementation is optimal. The broader performance investigation is tracked in [issue #96](https://github.com/HawkBearPig/dgpp/issues/96).

### FP4 conversion overhead

A controlled synthetic single-layer test fixes the TP4 geometry, 262,144-token allocation, 2,048-query chunk and input seed. Only KV format changes. It excludes MoE and collectives.

| Context at chunk end | BF16 (ms) | FP8 (ms) | FP4, original (ms) | FP4, native conversion (ms) |
|---|---:|---:|---:|---:|
| 8,192 | 32.061 | 34.993 | 50.971 | 34.825 |
| 32,768 | 51.835 | 53.692 | 69.234 | 53.141 |

The FP4 attention loader decodes E2M1 values and E4M3 block scales in software for every gathered tile. The candidate uses native pair conversions, retaining both FP32 scale multiplications and final BF16 rounding. BF16/FP8 timings change by less than 1% in the control runs. This replaces scalar decoding present since `2b7a6c6e` (2026-09-06); it is separate from the recent index-score fixes.

All 52 DSA tests pass, including a new test covering every valid nonnegative block-scale code and every FP4 value through partial, listed and dense attention loaders. Their partials match an equivalent host-dequantized BF16 cache bit-for-bit. Existing tests cover chunked prefill and decode against the reference. This does not constitute a new end-to-end quality evaluation.

- [Controlled microbenchmark comparison](fp4-conversion-experiment/comparison.json)
- [Candidate source patch](fp4-conversion-experiment/source.patch)
- [Test result](fp4-conversion-experiment/test-result.json)
- Profiles: [BF16](raw/glm53-w4/prefill/profile-8k/profile-summary.json), [FP8](raw/glm53-fp8kv-w4/prefill/profile-8k/profile-summary.json), [FP4](raw/glm53-fp4kv-256k-w4/prefill/profile-8k/profile-summary.json).

Unprofiled whole-model comparison with the same prompts, configuration and MTP1 mode (TP4, 256K FP4 KV, 8 slots):

| Target input | Original (s) | Native conversion (s) | Speedup |
|---|---:|---:|---:|
| ~2K | 4.347 | 3.770 | 1.15× |
| ~8K | 24.499 | 17.180 | 1.43× |
| ~32K | 110.254 | 74.988 | 1.47× |

Each value is the median of two cold probes. All six prompt hashes, one-token outputs, finish reasons and usage match. Two additional greedy MTP1 requests, each generating 256 tokens, also match in request, choices and usage. Both launches pass all-rank operation-stream identity and process/cgroup zero-swap checks. [Full comparison](fp4-conversion-experiment/full-model-comparison.json).

The first extra launch was refused by the unchanged startup memory guard while the MiMo downloader occupied host memory; it produced no requests or measurements and is explicitly excluded. An initial candidate launch is also excluded because its nonstandard executable basename prevented head-process telemetry matching; its cgroup still enforced zero swap. The replay uses `native/dgpp-serve` and requires complete monitoring on every rank.

## MiMo-V2.6-Flash on two nodes

The historical figures reproduce, but that does not make them efficient. Published two-node BF16 prefill is approximately 1.95× the four-node time at all three lengths. Fresh runs also match the historical per-token rates. Profiling nevertheless identifies a substantial missed optimization in both KV formats.

| Original ~8K profile | FP8 projection GEMMs | FP4 expert GEMMs | Collectives | Attention |
|---|---:|---:|---:|---:|
| BF16 KV | 59.5% | 21.3% | 8.2% | 3.4% |
| FP8 KV | 59.1% | 21.1% | 7.4% | 5.0% |

These are shares of summed kernel durations; actual prompt lengths differ slightly between KV formats, so this is not a controlled KV-only timing comparison. FP8 KV requires quantizing new entries and applying stored scales when decoding them for attention; reduced memory traffic does not necessarily offset that work during batched prefill. The dominant cost is the **FP8 weight projections**, not the KV format. MiMo calls `launch_scale_gemm_grid_*` for its fused QKV projections and dense MLP. That explicit-grid dispatcher retained the older 16-row tile kernel even for MiMo's 128×128 scale grid. The ordinary dispatcher already used the faster pipelined per-weight FP8 GEMM after `7944975b` (2026-10-04, #89), but the explicit-grid entry point missed that change.

The fix routes supported shapes above 128 rows, with `rs == cs == 7`, to the existing exact kernel. It preserves per-weight BF16 rounding and the ascending-K16 accumulation order. Small-row/decode dispatch, 32×32/64×64 scale grids and unsupported shapes retain their existing paths. No KV or model-memory allocation changes.

All 20 scale-GEMM tests and the FP8-weight GEMM accuracy suite pass. A new dispatch test covers all nine scale-grid combinations, supported and unsupported K, ragged output dimensions, padded strides, FP32/BF16 outputs and bitwise agreement with the original tile. It also inspects the captured kernel to verify the fast path is actually selected.

Matched fresh-server comparisons, medians of two cold probes at each target:

| KV configuration | Target input | Original (s) | Fixed (s) | Speedup |
|---|---:|---:|---:|---:|
| 128K BF16, 4 slots | ~2K | 3.417 | 1.567 | 2.18× |
| 128K BF16, 4 slots | ~8K | 12.933 | 5.512 | 2.35× |
| 128K BF16, 4 slots | ~32K | 55.933 | 25.825 | 2.17× |
| 256K FP8, 4 slots | ~2K | 3.418 | 1.559 | 2.19× |
| 256K FP8, 4 slots | ~8K | 12.967 | 5.573 | 2.33× |
| 256K FP8, 4 slots | ~32K | 57.896 | 27.945 | 2.07× |

All 12 prompt hashes, one-token outputs, finish reasons and usage match their baselines. Every launch passes rank identity and process/cgroup zero-swap validation. [Matched comparisons](mimo-dispatch-fix/comparison.json).

The BF16 KV profile confirms FP8 projection time falls from **7.836 s to 0.432 s**; expert work stays at **2.801 s versus 2.803 s**, and attention at **0.447 s versus 0.450 s**. This isolates the gain to the intended dispatcher change. [Post-fix BF16 profile](raw/mimo-w2/prefill/fp8-dispatch-profile/profile-summary.json) and [post-fix FP8 profile](raw/mimo-w2-fp8kv/prefill/fp8-dispatch-profile/profile-summary.json). The implementation applies at any MiMo tensor-parallel size with eligible shapes; the new whole-model timings here cover **two nodes only**. [Source patch](mimo-dispatch-fix/source.patch), [test results](mimo-dispatch-fix/tests-passed.json), and original profiles for [BF16 KV](raw/mimo-w2/prefill/profile-8k/profile-summary.json) / [FP8 KV](raw/mimo-w2-fp8kv/prefill/profile-8k/profile-summary.json).

## Reproduction and service state

`compare_prefill.py` performs fresh-server comparisons with recorded configurations, commands, prompt hashes, rank operation-stream checks and per-rank memory monitoring. `--idle-budget 2048` changes only the idle budget; `--candidate-binary PATH` records and stages a separate binary. `--profile` records a single cold request with Nsight Systems. Never mix profiler timings into the published performance table.

Each validated launch retains its original `prefill.json` and a compact `record.json` containing the configuration, resolved deployment, binary provenance, commands, completion status, rank identity and memory audit. Profile summaries and the two full-GLM decode comparisons remain alongside their runs. [Pinned model revisions](model-revisions.json) identify the checkpoints; the candidate source patches and test logs identify the validated kernel changes.

The Git evidence set excludes repeated server/build/download logs, per-second telemetry, copied monitor scripts, failed-run runtime files and one-off orchestration scripts. Full captures remain in the ignored local campaign directory. Each run record includes telemetry hashes and the original per-rank audit results (sample counts, maximum process swap, cgroup-policy checks, sampling gaps and errors). These summaries preserve the recorded checks; raw telemetry is needed to independently repeat the memory audit. Excluded attempts retain only their exclusion records. [File-retention review](retention.json).

The following commands use only retained evidence and do not launch a server:

```bash
python3 benchmarks/results/2026-10-07-prefill-followup/compare_results.py
python3 benchmarks/results/2026-10-07-prefill-followup/compare_mimo_fix.py
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py --check
```

Run the completion campaign's `update_document.py` after the original matrix's `summarize.py`; it incorporates these five corrected rows and then overlays the remaining validated reruns. The two campaigns retain their separate binary provenance and audits.

MiMo was absent from every node at investigation start. It is downloaded only to nodes 0/1 and uses a dedicated resident-cache directory. Both downloaded checkpoint copies and the dedicated resident caches have been removed. The storage audit preserves all initial snapshot revisions and resident filenames/sizes; free space after cleanup is 714.2 / 900.6 / 885.2 / 948.6 GiB across nodes 0–3. Temporary test binaries on node 2 were also removed.

The original four-node GLM Flash configuration and original binary have been restored. API and inference smoke checks passed, all four ranks survived monitoring-session logout, and process/cgroup swap remained zero. OS swap settings were unchanged.

- [17 validated diagnostic launches and two excluded attempts](measurement-audit.json)
- [Model cleanup](cleanup.json) and [storage-preservation audit](storage-preservation-audit.json)
- [Service restoration](restoration/restored.json)

The [completion campaign](../2026-10-07-prefill-completion/README.md) finished the remaining cold-prefill probes, affected serving and context sweeps, and full-GLM FP4 KV decode modes: 46 groups across seven configurations. The main document incorporates those results while preserving full-GLM BF16/FP8 and other unchanged paths. Native FP8 attention remains a separate evaluation in [issue #98](https://github.com/HawkBearPig/dgpp/issues/98).
