# DeepSeek-V4.1 prefill batching

CSA2 prefill chose its query tile size from the maximum KV allocation rather than the history visible to the current prompt. At 128K capacity, it used four queries per tile; at 1M, it used one. Shorter prompts therefore issued many small GEMMs and selection kernels despite having enough allocated workspace to batch more queries.

[Fix 07a32fb](https://github.com/HawkBearPig/dgpp/commit/07a32fb) sizes each tile from the current padded history and the existing buffer capacity. Both score buffers use the same query-by-entry bound. KV capacity, workspace allocations, prefix cache, request slots and decode paths are unchanged. Only DeepSeek-V4.1 instantiates `Csa2Layer`; other model families are unaffected.

The problem was introduced in [d942e92](https://github.com/HawkBearPig/dgpp/commit/d942e92ce61cfb8931ccbe39ae131367e33719b1), the original DeepSeek-V4.1 implementation on 2026-09-14.

## Controlled replay

Both configurations use four nodes, six slots, a 14 GiB prefix cache and adaptive DSpark up to four draft tokens. Before and after use the same configurations, prompt tags, seeds and client commands. Context groups contain one cold request and two prefix-cache repetitions; the table reports only the first request's cold prefill. Cold probes use three uncached prompts at each approximate length.

| KV allocation | Workload | Before (s) | After (s) | Speedup |
|---|---|---:|---:|---:|
| 128K | Cold probe 2K (median of 3) | 1.531 | 1.466 | 1.04× |
| 128K | Cold probe 8K (median of 3) | 5.496 | 4.989 | 1.10× |
| 128K | Cold probe 32K (median of 3) | 25.992 | 19.471 | 1.33× |
| 128K | Context 32K (cold request) | 25.340 | 19.027 | 1.33× |
| 128K | Context 64K (cold request) | 65.495 | 43.346 | 1.51× |
| 128K | Context 128K (cold request) | 195.174 | 126.329 | 1.54× |
| 1M | Cold probe 2K (median of 3) | 1.752 | 1.453 | 1.21× |
| 1M | Cold probe 8K (median of 3) | 7.678 | 4.910 | 1.56× |
| 1M | Cold probe 32K (median of 3) | 50.565 | 19.305 | 2.62× |
| 1M | Context 32K (cold request) | 48.584 | 18.683 | 2.60× |
| 1M | Context 64K (cold request) | 153.790 | 40.894 | 3.76× |

All **33 prompt/output/usage comparisons matched**: 18 at 128K capacity and 15 at 1M capacity. Both launches passed per-rank process and cgroup zero-swap checks, operation-stream identity and clean shutdown. No rank logged a collective stall. The largest telemetry gap was under 1.09 seconds.

- [128K comparison](deepseek-w4-comparison.json)
- [1M comparison](deepseek-1m-w4-comparison.json)
- [Validation and evidence hashes](validation.json)
- [Candidate source and binary hashes](candidate.json), [patch](candidate.patch)

The comparison records freeze paths as they existed during replay. The 128K baseline's `raw/deepseek-w4/refresh/default` is subsequently archived at `../../attempts/deepseek-w4/before-capacity-fix-20261007/refresh/default`. The 1M baseline is under `../../attempts/deepseek-1m-w4/default-before-capacity-investigation-20261007`. The earlier capacity observation likewise preserves its original paths. Diagnostic timings are not substituted for final matrix results: both affected deployments are rerun in full.

The 1M baseline was paused after its 64K client completed successfully. Its controller was interrupted before saving that command's return code; [client-boundary.json](client-boundary.json) separately records client exit 0 and all three samples. The incomplete controller receipt remains unchanged, and this launch is used only for diagnosis.

## Tests

The new CPU-oracle regression covers TP1 and TP4 slices, two cache capacities, partial tiles, odd compressor tails and subsequent decode. It fails before the fix and passes afterward, with no index violations or selection flips. Compute Sanitizer reports zero errors.

CSA2 kernel/layer tests, model reference and bounded-prefill comparisons, decode, TP and engine tests passed. The first model-test invocation lacked reference fixtures; the subsequent eleven-stage fixture-generation and model comparison chain passed. Both logs are retained.

## Benchmark retention

All previous DeepSeek-V4.1 serving, cold-prefill, long-context and decode-mode groups are replaced. The [capacity retention catalog](../../capacity-retained-results.json) preserves validated measurements from unaffected model families on the preceding binary. Earlier retention catalogs remain hash-bound and unchanged.
