# Benchmark matrix refresh

All throughput values are medians of three repetitions. Serving ranges span the five prompt classes.
The [manifest](manifest.json) identifies the engine revision and binary. Commands, exit codes, configurations and responses are retained under `raw/`.
Outstanding measurement groups: 0.

| Deployment | C1/C2/C4/C8 | Decode modes | Context buckets | Cold prefill |
| --- | --- | --- | --- | --- |
| glm-flash-hybrid-w2 | Complete | 4/4 | 4/4 | Complete |
| glm-flash-hybrid-256k-w2 | Complete | 4/4 | 5/5 | Complete |
| glm-flash-hybrid-w4 | Complete | 4/4 | 6/6 | Complete |
| glm-flash-fp8-w4 | Complete | 4/4 | 5/5 | Complete |
| glm-flash-fp8-512k-w4 | Complete | 4/4 | 6/6 | Complete |
| qwen-nvfp4-w1 | Complete | 4/4 | 5/5 | Complete |
| qwen-nvfp4-bf16-w1 | Complete | 4/4 | 5/5 | Complete |
| qwen-nvfp4-w2 | Complete | 4/4 | 5/5 | Complete |
| qwen-yarn-w2 | Complete | 4/4 | 6/6 | Complete |
| qwen-radixark-w1 | Complete | 4/4 | 5/5 | Complete |
| qwen-radixark-w2 | Complete | 5/5 | 5/5 | Complete |
| qwen-fp8-w2 | Complete | 4/4 | 5/5 | Complete |
| qwen-fp8-w4 | Complete | 4/4 | 5/5 | Complete |
| qwen-autoround-w1 | Complete | 5/5 | 5/5 | Complete |
| qwen-autoround-prefill-w1 | Complete | 5/5 | 5/5 | Complete |
| glm53-w4 | Complete | 4/4 | 3/3 | Complete |
| glm53-fp8kv-w4 | Complete | 4/4 | 4/4 | Complete |
| glm53-fp4kv-256k-w4 | Complete | 4/4 | 5/5 | Complete |
| deepseek-w4 | Complete | 6/6 | 4/4 | Complete |
| deepseek-1m-w4 | Complete | 6/6 | 6/6 | Complete |
| dsv4-w2 | Complete | 6/6 | 6/6 | Complete |
| dsv4-w4 | Complete | 6/6 | 6/6 | Complete |
| mimo-w2 | Complete | 4/4 | 4/4 | Complete |
| mimo-w2-fp8kv | Complete | 4/4 | 5/5 | Complete |
| mimo-w4 | Complete | 4/4 | 4/4 | Complete |
| mimo-1m-w4 | Complete | 4/4 | 6/6 | Complete |
| qwen27b-fp8-w1 | Complete | 6/6 | 5/5 | Complete |
| qwen27b-fp8-w2 | Complete | 6/6 | 5/5 | Complete |
| qwen27b-fp8-w4 | Complete | 6/6 | 5/5 | Complete |


Only runs with passing per-rank process-swap checks contribute values. The [earlier matrix](../2026-10-05-matrix-refresh/README.md) preserves baseline measurements; the [scaling investigation](../2026-10-05-matrix-refresh/diagnostics/prefill-scaling/README.md) records the prefill fix. All launches in this campaign enforce a zero process/cgroup swap policy.

Short-prompt serving, completed decode-mode sweeps, minimal context, and 2K/8K prefill may reuse the audited GLM Flash baseline. The [reuse audit](reused-results.json) binds each source by hash and records the input lengths. Affected prefill measurements are rerun. Retained inputs are at most 8,191 tokens; the corrected GLM Flash selector requires more than 2,048 pools (four tokens per pool). [Per-measurement provenance](result-provenance.json) identifies the source binary and raw record; reused measurements retain their original memory evidence. The [input-length and memory audit](diagnostics/reuse-verification.json) covers all 27 eligible baseline job groups; [resolver tests](diagnostics/result-sources-tests.log) check rejection of invalid evidence and attribution of combined prefill results.

The [full-GLM investigation](diagnostics/full-glm-prefill-investigation/README.md) records the fused score follow-up. Its [retention catalog](retained-results.json) preserves unaffected measurements from the first fixed binary. Full-GLM prefill groups above 2,048 actual input tokens are replaced; nominal bucket names do not determine eligibility. Saved launch manifests identify the binary for every run.

The [DeepSeek-V4.1 investigation](diagnostics/deepseek-capacity-prefill-investigation/README.md) records the correction to prefill query batching at large KV allocations. Both DeepSeek-V4.1 deployments are rerun in full. The [capacity retention catalog](capacity-retained-results.json) preserves unaffected families from the preceding binary.

## glm-flash-hybrid-w2

Configuration: [configs/glm-flash-hybrid-w2.json](configs/glm-flash-hybrid-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192 | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/default/prefill.json) |
| default/prefill | 32768 | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 34.10 | 32.96 |
| code | 1 | 36.07 | 34.54 |
| json | 1 | 36.16 | 34.68 |
| math | 1 | 34.82 | 33.27 |
| chat | 1 | 30.76 | 29.72 |
| prose | 2 | 46.54 | 45.11 |
| code | 2 | 48.75 | 46.92 |
| json | 2 | 45.83 | 44.31 |
| math | 2 | 49.04 | 46.98 |
| chat | 2 | 44.75 | 43.23 |
| prose | 4 | 63.40 | 61.41 |
| code | 4 | 60.15 | 58.30 |
| json | 4 | 66.70 | 64.38 |
| math | 4 | 61.73 | 59.62 |
| chat | 4 | 61.49 | 59.61 |
| prose | 8 | 63.04 | 60.53 |
| code | 8 | 61.17 | 58.43 |
| json | 8 | 66.73 | 63.89 |
| math | 8 | 64.76 | 61.54 |
| chat | 8 | 61.57 | 59.01 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 94 | 0.400 | 54.83 | 1.82 | 33.22 |
| 32768 | 32236 | 70.217 | 57.18 | 1.98 | 34.57 |
| 65536 | 65011 | 143.597 | 57.89 | 1.95 | 33.63 |
| 131072 | 130537 | 302.698 | 59.47 | 1.96 | 32.98 |

## glm-flash-hybrid-256k-w2

Configuration: [configs/glm-flash-hybrid-256k-w2.json](configs/glm-flash-hybrid-256k-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192 | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/default/prefill.json) |
| default/prefill | 32768 | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-256k-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-256k-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 34.27 | 33.09 |
| code | 1 | 36.06 | 34.58 |
| json | 1 | 36.19 | 34.70 |
| math | 1 | 34.81 | 33.28 |
| chat | 1 | 30.74 | 29.74 |
| prose | 2 | 46.41 | 44.89 |
| code | 2 | 48.66 | 46.84 |
| json | 2 | 45.82 | 44.22 |
| math | 2 | 49.02 | 46.98 |
| chat | 2 | 44.71 | 43.20 |
| prose | 4 | 47.72 | 45.95 |
| code | 4 | 47.32 | 45.49 |
| json | 4 | 50.35 | 48.38 |
| math | 4 | 47.48 | 45.54 |
| chat | 4 | 46.41 | 44.72 |
| prose | 8 | 46.48 | 44.70 |
| code | 8 | 46.55 | 44.74 |
| json | 8 | 49.45 | 47.51 |
| math | 8 | 48.68 | 46.47 |
| chat | 8 | 46.78 | 45.17 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 97 | 0.430 | 54.79 | 1.85 | 33.73 |
| 32768 | 32239 | 70.058 | 57.09 | 1.95 | 34.09 |
| 65536 | 65011 | 143.048 | 58.12 | 1.96 | 33.75 |
| 131072 | 130539 | 302.115 | 59.43 | 1.96 | 33.01 |
| 262144 | 261632 | 669.465 | 62.44 | 1.92 | 30.71 |

## glm-flash-hybrid-w4

Configuration: [configs/glm-flash-hybrid-w4.json](configs/glm-flash-hybrid-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/glm-flash-hybrid-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-hybrid-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 60.30 | 57.91 |
| code | 1 | 61.63 | 59.05 |
| json | 1 | 62.19 | 59.70 |
| math | 1 | 59.99 | 57.33 |
| chat | 1 | 56.83 | 54.75 |
| prose | 2 | 82.06 | 79.45 |
| code | 2 | 84.89 | 81.60 |
| json | 2 | 83.10 | 80.12 |
| math | 2 | 89.35 | 85.44 |
| chat | 2 | 83.14 | 80.27 |
| prose | 4 | 116.19 | 112.62 |
| code | 4 | 111.89 | 108.43 |
| json | 4 | 121.97 | 117.74 |
| math | 4 | 114.54 | 110.73 |
| chat | 4 | 109.53 | 106.24 |
| prose | 8 | 114.80 | 109.64 |
| code | 8 | 110.77 | 105.74 |
| json | 8 | 122.10 | 116.04 |
| math | 8 | 115.30 | 110.41 |
| chat | 8 | 112.82 | 108.07 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 99 | 0.236 | 31.85 | 1.88 | 58.87 |
| 32768 | 32239 | 21.799 | 33.55 | 1.96 | 58.47 |
| 65536 | 65014 | 48.172 | 34.33 | 1.96 | 57.14 |
| 131072 | 130538 | 114.093 | 35.96 | 1.95 | 54.13 |
| 262144 | 261629 | 303.271 | 39.15 | 1.96 | 50.10 |
| 524288 | 523783 | 925.281 | 45.52 | 1.98 | 43.43 |

## glm-flash-fp8-w4

Configuration: [configs/glm-flash-fp8-w4.json](configs/glm-flash-fp8-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192 | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/default/prefill.json) |
| default/prefill | 32768 | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 49.56 | 47.51 |
| code | 1 | 50.60 | 48.21 |
| json | 1 | 51.51 | 49.22 |
| math | 1 | 50.14 | 47.60 |
| chat | 1 | 46.67 | 44.81 |
| prose | 2 | 62.89 | 60.93 |
| code | 2 | 65.01 | 62.63 |
| json | 2 | 63.82 | 61.57 |
| math | 2 | 67.25 | 64.40 |
| chat | 2 | 62.20 | 60.09 |
| prose | 4 | 83.06 | 80.84 |
| code | 4 | 79.71 | 77.46 |
| json | 4 | 89.04 | 86.19 |
| math | 4 | 80.84 | 78.33 |
| chat | 4 | 80.86 | 78.66 |
| prose | 8 | 81.42 | 78.07 |
| code | 8 | 81.09 | 77.97 |
| json | 8 | 88.49 | 84.87 |
| math | 8 | 83.94 | 79.83 |
| chat | 8 | 80.85 | 77.57 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 99 | 0.331 | 38.49 | 1.82 | 47.32 |
| 32768 | 32237 | 24.243 | 40.43 | 1.96 | 48.51 |
| 65536 | 65012 | 52.909 | 41.25 | 1.98 | 47.92 |
| 131072 | 130539 | 123.326 | 42.83 | 1.96 | 45.79 |
| 262144 | 261630 | 317.875 | 46.07 | 1.93 | 41.93 |

## glm-flash-fp8-512k-w4

Configuration: [configs/glm-flash-fp8-512k-w4.json](configs/glm-flash-fp8-512k-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-512k-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192 | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-512k-w4/refresh/default/prefill.json) |
| default/prefill | 32768 | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | 2e9ba349da62 | [Record](../2026-10-05-matrix-refresh/raw/glm-flash-fp8-512k-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/glm-flash-fp8-512k-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 49.49 | 47.52 |
| code | 1 | 50.67 | 48.33 |
| json | 1 | 51.64 | 49.26 |
| math | 1 | 50.21 | 47.62 |
| chat | 1 | 46.75 | 44.85 |
| prose | 2 | 62.82 | 60.79 |
| code | 2 | 65.07 | 62.63 |
| json | 2 | 63.82 | 61.66 |
| math | 2 | 67.45 | 64.41 |
| chat | 2 | 62.40 | 60.22 |
| prose | 4 | 81.70 | 79.47 |
| code | 4 | 80.03 | 77.78 |
| json | 4 | 89.05 | 86.31 |
| math | 4 | 82.02 | 79.59 |
| chat | 4 | 81.09 | 78.81 |
| prose | 8 | 81.13 | 77.72 |
| code | 8 | 80.98 | 78.43 |
| json | 8 | 88.89 | 85.76 |
| math | 8 | 84.02 | 79.95 |
| chat | 8 | 80.78 | 77.50 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 96 | 0.322 | 38.40 | 1.86 | 48.47 |
| 32768 | 32234 | 24.234 | 40.35 | 1.96 | 48.62 |
| 65536 | 65009 | 53.010 | 41.89 | 1.93 | 46.12 |
| 131072 | 130539 | 123.934 | 43.00 | 1.96 | 45.62 |
| 262144 | 261631 | 317.535 | 46.02 | 1.95 | 42.30 |
| 524288 | 523781 | 928.123 | 52.26 | 1.98 | 37.83 |

## qwen-nvfp4-w1

Configuration: [configs/qwen-nvfp4-w1.json](configs/qwen-nvfp4-w1.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w1/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 41.69 | 40.73 |
| code | 1 | 48.97 | 47.43 |
| json | 1 | 49.65 | 48.06 |
| math | 1 | 47.00 | 45.13 |
| chat | 1 | 43.96 | 42.81 |
| prose | 2 | 63.87 | 62.08 |
| code | 2 | 72.47 | 69.80 |
| json | 2 | 75.84 | 73.12 |
| math | 2 | 73.83 | 70.88 |
| chat | 2 | 66.93 | 64.86 |
| prose | 4 | 88.89 | 86.45 |
| code | 4 | 99.45 | 96.42 |
| json | 4 | 100.84 | 97.68 |
| math | 4 | 99.49 | 96.25 |
| chat | 4 | 90.33 | 87.80 |
| prose | 8 | 87.44 | 84.31 |
| code | 8 | 99.08 | 95.31 |
| json | 8 | 103.27 | 98.92 |
| math | 8 | 98.46 | 94.30 |
| chat | 8 | 90.48 | 87.20 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 159 | 0.340 | 40.53 | 1.85 | 45.59 |
| 32768 | 32283 | 18.241 | 42.73 | 1.80 | 42.02 |
| 65536 | 65052 | 38.461 | 43.12 | 1.86 | 43.16 |
| 131072 | 130575 | 85.217 | 43.97 | 1.85 | 42.03 |
| 262144 | 261657 | 201.651 | 45.65 | 1.78 | 39.07 |

## qwen-nvfp4-bf16-w1

Configuration: [configs/qwen-nvfp4-bf16-w1.json](configs/qwen-nvfp4-bf16-w1.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-bf16-w1/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 33.89 | 33.20 |
| code | 1 | 37.35 | 36.44 |
| json | 1 | 38.04 | 37.09 |
| math | 1 | 36.10 | 35.12 |
| chat | 1 | 31.18 | 30.55 |
| prose | 2 | 53.38 | 52.20 |
| code | 2 | 60.07 | 58.37 |
| json | 2 | 62.73 | 60.90 |
| math | 2 | 60.65 | 58.67 |
| chat | 2 | 53.40 | 52.05 |
| prose | 4 | 76.09 | 74.18 |
| code | 4 | 85.78 | 83.34 |
| json | 4 | 87.50 | 84.83 |
| math | 4 | 84.72 | 82.09 |
| chat | 4 | 77.81 | 75.70 |
| prose | 8 | 76.02 | 73.53 |
| code | 8 | 86.09 | 83.22 |
| json | 8 | 89.20 | 85.56 |
| math | 8 | 84.77 | 81.38 |
| chat | 8 | 78.09 | 75.45 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 155 | 0.285 | 52.62 | 1.81 | 34.37 |
| 32768 | 32286 | 17.776 | 54.78 | 1.83 | 33.49 |
| 65536 | 65051 | 37.215 | 55.24 | 1.83 | 33.21 |
| 131072 | 130574 | 85.029 | 57.01 | 1.89 | 33.13 |
| 262144 | 261662 | 199.488 | 58.08 | 1.82 | 31.36 |

## qwen-nvfp4-w2

Configuration: [configs/qwen-nvfp4-w2.json](configs/qwen-nvfp4-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-nvfp4-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 67.51 | 65.50 |
| code | 1 | 73.63 | 70.95 |
| json | 1 | 74.90 | 72.51 |
| math | 1 | 72.33 | 68.99 |
| chat | 1 | 62.29 | 60.51 |
| prose | 2 | 96.05 | 93.26 |
| code | 2 | 109.84 | 105.49 |
| json | 2 | 114.12 | 109.34 |
| math | 2 | 109.79 | 105.09 |
| chat | 2 | 99.89 | 96.66 |
| prose | 4 | 129.34 | 126.05 |
| code | 4 | 148.84 | 144.44 |
| json | 4 | 150.96 | 146.08 |
| math | 4 | 147.00 | 142.03 |
| chat | 4 | 133.14 | 129.29 |
| prose | 8 | 129.61 | 124.91 |
| code | 8 | 147.77 | 142.83 |
| json | 8 | 153.81 | 147.33 |
| math | 8 | 144.21 | 137.95 |
| chat | 8 | 133.47 | 128.91 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 156 | 0.210 | 27.13 | 1.89 | 69.62 |
| 32768 | 32286 | 30.420 | 28.38 | 1.88 | 66.07 |
| 65536 | 65054 | 64.056 | 28.70 | 1.80 | 62.57 |
| 131072 | 130573 | 138.480 | 29.93 | 1.78 | 59.59 |
| 262144 | 261659 | 314.459 | 31.54 | 1.86 | 59.02 |

## qwen-yarn-w2

Configuration: [configs/qwen-yarn-w2.json](configs/qwen-yarn-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-yarn-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 65.01 | 63.07 |
| code | 1 | 74.42 | 71.72 |
| json | 1 | 75.86 | 73.03 |
| math | 1 | 72.29 | 69.03 |
| chat | 1 | 65.02 | 62.98 |
| prose | 2 | 97.46 | 94.50 |
| code | 2 | 111.88 | 107.46 |
| json | 2 | 114.90 | 110.17 |
| math | 2 | 111.60 | 106.86 |
| chat | 2 | 98.97 | 95.31 |
| prose | 4 | 98.05 | 94.66 |
| code | 4 | 109.26 | 105.24 |
| json | 4 | 115.32 | 111.24 |
| math | 4 | 108.35 | 104.10 |
| chat | 4 | 98.94 | 95.75 |
| prose | 8 | 96.08 | 93.01 |
| code | 8 | 110.55 | 106.55 |
| json | 8 | 114.03 | 109.59 |
| math | 8 | 110.01 | 105.63 |
| chat | 8 | 100.03 | 96.99 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 160 | 0.210 | 26.43 | 1.86 | 70.43 |
| 32768 | 32287 | 30.052 | 28.12 | 1.83 | 65.24 |
| 65536 | 65052 | 63.962 | 28.46 | 1.86 | 65.40 |
| 131072 | 130576 | 137.367 | 29.46 | 1.82 | 61.83 |
| 262144 | 261657 | 312.204 | 31.35 | 1.85 | 58.94 |
| 524288 | 523788 | 875.869 | 37.12 | 1.85 | 49.77 |

## qwen-radixark-w1

Configuration: [configs/qwen-radixark-w1.json](configs/qwen-radixark-w1.json). Default mode: mtp2.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/plain/greedy.json) |
| mtp1/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/mtp1/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w1/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 47.52 | 46.21 |
| code | 1 | 59.54 | 57.29 |
| json | 1 | 60.95 | 58.36 |
| math | 1 | 54.69 | 52.03 |
| chat | 1 | 44.23 | 42.91 |
| prose | 2 | 58.47 | 56.89 |
| code | 2 | 71.60 | 68.83 |
| json | 2 | 80.24 | 76.77 |
| math | 2 | 76.93 | 73.49 |
| chat | 2 | 57.87 | 56.11 |
| prose | 4 | 82.99 | 80.60 |
| code | 4 | 99.85 | 96.40 |
| json | 4 | 110.94 | 106.33 |
| math | 4 | 101.57 | 97.66 |
| chat | 4 | 87.82 | 85.14 |
| prose | 8 | 81.34 | 78.28 |
| code | 8 | 106.19 | 101.37 |
| json | 8 | 113.26 | 107.92 |
| math | 8 | 102.70 | 97.57 |
| chat | 8 | 85.38 | 82.20 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 157 | 0.345 | 48.77 | 2.50 | 51.26 |
| 32768 | 32285 | 19.703 | 50.94 | 2.48 | 48.60 |
| 65536 | 65055 | 38.793 | 51.55 | 2.50 | 48.49 |
| 131072 | 130573 | 86.157 | 52.85 | 2.50 | 47.31 |
| 262144 | 261659 | 204.820 | 54.87 | 2.63 | 47.91 |

## qwen-radixark-w2

Configuration: [configs/qwen-radixark-w2.json](configs/qwen-radixark-w2.json). Default mode: mtp4.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/plain/greedy.json) |
| mtp1/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-radixark-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 52.07 | 50.61 |
| code | 1 | 90.15 | 85.59 |
| json | 1 | 96.87 | 92.35 |
| math | 1 | 79.71 | 75.22 |
| chat | 1 | 53.15 | 51.86 |
| prose | 2 | 74.03 | 71.87 |
| code | 2 | 116.34 | 109.53 |
| json | 2 | 145.09 | 136.42 |
| math | 2 | 135.79 | 126.89 |
| chat | 2 | 80.53 | 77.79 |
| prose | 4 | 103.98 | 100.80 |
| code | 4 | 144.41 | 138.04 |
| json | 4 | 170.63 | 161.52 |
| math | 4 | 143.82 | 137.69 |
| chat | 4 | 111.41 | 107.74 |
| prose | 8 | 98.61 | 95.00 |
| code | 8 | 157.82 | 148.81 |
| json | 8 | 189.95 | 177.65 |
| math | 8 | 153.33 | 144.63 |
| chat | 8 | 111.03 | 106.46 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 159 | 0.244 | 47.52 | 3.36 | 70.60 |
| 32768 | 32282 | 30.213 | 50.32 | 3.45 | 68.48 |
| 65536 | 65055 | 63.513 | 50.89 | 4.11 | 80.82 |
| 131072 | 130576 | 137.791 | 52.28 | 3.70 | 70.69 |
| 262144 | 261659 | 312.403 | 54.80 | 3.45 | 62.89 |

## qwen-fp8-w2

Configuration: [configs/qwen-fp8-w2.json](configs/qwen-fp8-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 45.85 | 44.71 |
| code | 1 | 51.80 | 50.28 |
| json | 1 | 52.88 | 51.29 |
| math | 1 | 49.78 | 48.12 |
| chat | 1 | 44.03 | 42.96 |
| prose | 2 | 70.47 | 68.70 |
| code | 2 | 70.21 | 68.20 |
| json | 2 | 83.77 | 81.26 |
| math | 2 | 80.72 | 77.88 |
| chat | 2 | 71.64 | 69.64 |
| prose | 4 | 92.72 | 90.63 |
| code | 4 | 103.90 | 101.38 |
| json | 4 | 105.89 | 102.96 |
| math | 4 | 102.90 | 100.06 |
| chat | 4 | 95.05 | 92.81 |
| prose | 8 | 83.33 | 81.01 |
| code | 8 | 103.94 | 100.32 |
| json | 8 | 97.15 | 94.16 |
| math | 8 | 93.94 | 90.63 |
| chat | 8 | 85.65 | 83.21 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 158 | 0.229 | 37.60 | 1.90 | 50.61 |
| 32768 | 32286 | 35.545 | 40.22 | 1.81 | 44.96 |
| 65536 | 65054 | 75.142 | 39.80 | 1.86 | 46.77 |
| 131072 | 130573 | 163.189 | 40.77 | 1.85 | 45.33 |
| 262144 | 261661 | 365.062 | 42.61 | 1.83 | 43.05 |

## qwen-fp8-w4

Configuration: [configs/qwen-fp8-w4.json](configs/qwen-fp8-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-fp8-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 74.17 | 71.99 |
| code | 1 | 83.76 | 80.62 |
| json | 1 | 84.48 | 81.13 |
| math | 1 | 79.31 | 76.46 |
| chat | 1 | 70.82 | 68.65 |
| prose | 2 | 113.84 | 110.63 |
| code | 2 | 129.42 | 124.62 |
| json | 2 | 133.18 | 128.30 |
| math | 2 | 130.11 | 125.20 |
| chat | 2 | 114.10 | 110.71 |
| prose | 4 | 145.20 | 141.54 |
| code | 4 | 164.63 | 160.14 |
| json | 4 | 167.49 | 162.21 |
| math | 4 | 162.04 | 157.18 |
| chat | 4 | 151.13 | 147.24 |
| prose | 8 | 145.13 | 139.83 |
| code | 8 | 165.01 | 159.15 |
| json | 8 | 170.98 | 164.09 |
| math | 8 | 160.78 | 153.44 |
| chat | 8 | 148.58 | 143.46 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 156 | 0.158 | 23.55 | 1.85 | 78.45 |
| 32768 | 32286 | 24.077 | 25.35 | 1.85 | 72.90 |
| 65536 | 65055 | 51.506 | 25.68 | 1.89 | 73.56 |
| 131072 | 130572 | 112.922 | 26.51 | 1.86 | 70.22 |
| 262144 | 261659 | 261.324 | 28.33 | 1.80 | 63.39 |

## qwen-autoround-w1

Configuration: [configs/qwen-autoround-w1.json](configs/qwen-autoround-w1.json). Default mode: mtp3.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/plain/greedy.json) |
| mtp1/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/mtp2/greedy.json) |
| mtp4/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-w1/refresh/mtp4/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 43.38 | 42.33 |
| code | 1 | 59.72 | 57.62 |
| json | 1 | 69.81 | 66.90 |
| math | 1 | 60.93 | 58.04 |
| chat | 1 | 45.62 | 44.33 |
| prose | 2 | 59.14 | 57.62 |
| code | 2 | 83.79 | 80.54 |
| json | 2 | 94.62 | 90.33 |
| math | 2 | 90.04 | 85.99 |
| chat | 2 | 63.50 | 61.69 |
| prose | 4 | 78.73 | 76.55 |
| code | 4 | 111.72 | 107.53 |
| json | 4 | 121.30 | 115.75 |
| math | 4 | 110.38 | 105.77 |
| chat | 4 | 87.17 | 84.47 |
| prose | 8 | 78.64 | 75.99 |
| code | 8 | 114.02 | 108.50 |
| json | 8 | 132.61 | 124.76 |
| math | 8 | 113.40 | 107.47 |
| chat | 8 | 87.92 | 84.93 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 161 | 0.301 | 54.88 | 3.04 | 55.32 |
| 32768 | 32282 | 17.514 | 58.08 | 2.97 | 51.05 |
| 65536 | 65053 | 36.785 | 58.42 | 2.93 | 50.17 |
| 131072 | 130574 | 81.839 | 59.72 | 3.11 | 52.07 |
| 262144 | 261657 | 195.498 | 61.92 | 3.00 | 48.45 |

## qwen-autoround-prefill-w1

Configuration: [configs/qwen-autoround-prefill-w1.json](configs/qwen-autoround-prefill-w1.json). Default mode: mtp3.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/plain/greedy.json) |
| mtp1/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/mtp2/greedy.json) |
| mtp4/greedy | All | Reused | ac939277f247 | [Record](raw/qwen-autoround-prefill-w1/refresh/mtp4/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 43.09 | 42.08 |
| code | 1 | 58.33 | 56.37 |
| json | 1 | 69.03 | 65.97 |
| math | 1 | 59.32 | 56.64 |
| chat | 1 | 45.65 | 44.41 |
| prose | 2 | 56.95 | 55.63 |
| code | 2 | 85.61 | 82.21 |
| json | 2 | 95.93 | 91.79 |
| math | 2 | 90.80 | 86.60 |
| chat | 2 | 63.33 | 61.57 |
| prose | 4 | 81.14 | 78.88 |
| code | 4 | 114.16 | 109.81 |
| json | 4 | 125.46 | 120.10 |
| math | 4 | 111.73 | 106.99 |
| chat | 4 | 88.42 | 85.67 |
| prose | 8 | 79.88 | 77.14 |
| code | 8 | 113.44 | 107.92 |
| json | 8 | 131.28 | 124.41 |
| math | 8 | 114.17 | 108.20 |
| chat | 8 | 88.10 | 84.79 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 158 | 0.247 | 55.44 | 3.15 | 56.78 |
| 32768 | 32287 | 16.869 | 58.53 | 3.04 | 51.87 |
| 65536 | 65052 | 35.678 | 58.82 | 3.04 | 51.61 |
| 131072 | 130576 | 79.916 | 60.59 | 2.87 | 47.29 |
| 262144 | 261658 | 191.961 | 62.81 | 3.23 | 51.39 |

## glm53-w4

Configuration: [configs/glm53-w4.json](configs/glm53-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](attempts/glm53-w4/default-before-fused-scores-20261007/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | a983cebe9fc1 | [Record](raw/glm53-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](attempts/glm53-w4/default-before-fused-scores-20261007/context-0.json) |
| default/context-32768 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | ac939277f247 | [Record](attempts/glm53-w4/default-before-fused-scores-20261007/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](attempts/glm53-w4/default-before-fused-scores-20261007/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](attempts/glm53-w4/default-before-fused-scores-20261007/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 29.51 | 28.10 |
| code | 1 | 30.52 | 28.69 |
| json | 1 | 30.48 | 28.77 |
| math | 1 | 29.31 | 27.30 |
| chat | 1 | 27.41 | 26.12 |
| prose | 2 | 37.88 | 36.11 |
| code | 2 | 39.71 | 37.31 |
| json | 2 | 40.83 | 38.36 |
| math | 2 | 40.85 | 37.97 |
| chat | 2 | 37.64 | 35.69 |
| prose | 4 | 47.96 | 46.21 |
| code | 4 | 48.94 | 46.87 |
| json | 4 | 52.02 | 49.72 |
| math | 4 | 49.12 | 46.92 |
| chat | 4 | 47.89 | 46.07 |
| prose | 8 | 57.40 | 55.28 |
| code | 8 | 54.98 | 52.93 |
| json | 8 | 57.95 | 55.68 |
| math | 8 | 56.38 | 53.96 |
| chat | 8 | 55.69 | 53.77 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 94 | 0.821 | 64.37 | 1.88 | 29.13 |
| 32768 | 32237 | 70.959 | 76.25 | 1.96 | 25.72 |
| 65536 | 65009 | 153.616 | 80.89 | 1.98 | 24.44 |

## glm53-fp8kv-w4

Configuration: [configs/glm53-fp8kv-w4.json](configs/glm53-fp8kv-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | ac939277f247 | [Record](attempts/glm53-fp8kv-w4/default-before-fused-scores-20261007/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | a983cebe9fc1 | [Record](raw/glm53-fp8kv-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | ac939277f247 | [Record](attempts/glm53-fp8kv-w4/default-before-fused-scores-20261007/context-0.json) |
| default/context-32768 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp8kv-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp8kv-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp8kv-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | ac939277f247 | [Record](attempts/glm53-fp8kv-w4/default-before-fused-scores-20261007/context-262144.json) |
| default/context-524288 | All | Reused | ac939277f247 | [Record](attempts/glm53-fp8kv-w4/default-before-fused-scores-20261007/context-524288.json) |
| plain/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-fp8kv-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-fp8kv-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | ac939277f247 | [Record](raw/glm53-fp8kv-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 30.15 | 28.67 |
| code | 1 | 30.37 | 28.60 |
| json | 1 | 30.42 | 28.71 |
| math | 1 | 29.85 | 27.75 |
| chat | 1 | 27.03 | 25.78 |
| prose | 2 | 38.41 | 36.63 |
| code | 2 | 39.85 | 37.39 |
| json | 2 | 39.71 | 37.34 |
| math | 2 | 41.54 | 38.48 |
| chat | 2 | 37.95 | 35.98 |
| prose | 4 | 49.02 | 47.17 |
| code | 4 | 48.18 | 46.24 |
| json | 4 | 51.50 | 49.25 |
| math | 4 | 48.39 | 46.25 |
| chat | 4 | 47.79 | 45.97 |
| prose | 8 | 56.53 | 54.46 |
| code | 8 | 55.27 | 53.23 |
| json | 8 | 58.87 | 56.56 |
| math | 8 | 57.09 | 54.61 |
| chat | 8 | 56.47 | 54.51 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 97 | 0.840 | 64.18 | 1.88 | 29.22 |
| 32768 | 32237 | 75.099 | 76.24 | 1.96 | 25.73 |
| 65536 | 65008 | 161.000 | 81.06 | 1.96 | 24.20 |
| 131072 | 130540 | 366.933 | 90.13 | 1.98 | 21.93 |

## glm53-fp4kv-256k-w4

Configuration: [configs/glm53-fp4kv-256k-w4.json](configs/glm53-fp4kv-256k-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/prefill.json) |
| default/context-0 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Reused | a983cebe9fc1 | [Record](raw/glm53-fp4kv-256k-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 29.85 | 28.44 |
| code | 1 | 29.28 | 27.63 |
| json | 1 | 30.49 | 28.78 |
| math | 1 | 29.56 | 27.50 |
| chat | 1 | 27.32 | 26.03 |
| prose | 2 | 38.82 | 36.94 |
| code | 2 | 39.89 | 37.45 |
| json | 2 | 39.57 | 37.26 |
| math | 2 | 40.26 | 37.45 |
| chat | 2 | 38.19 | 36.21 |
| prose | 4 | 48.30 | 46.51 |
| code | 4 | 47.25 | 45.39 |
| json | 4 | 51.55 | 49.31 |
| math | 4 | 48.84 | 46.59 |
| chat | 4 | 47.83 | 46.01 |
| prose | 8 | 56.97 | 54.95 |
| code | 8 | 54.11 | 52.13 |
| json | 8 | 58.93 | 56.58 |
| math | 8 | 56.29 | 53.89 |
| chat | 8 | 55.64 | 53.73 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 99 | 0.851 | 64.11 | 1.85 | 28.82 |
| 32768 | 32234 | 108.050 | 78.82 | 1.98 | 25.08 |
| 65536 | 65009 | 230.027 | 83.55 | 1.98 | 23.66 |
| 131072 | 130537 | 507.772 | 92.97 | 1.98 | 21.26 |
| 262144 | 261630 | 1187.740 | 111.35 | 1.96 | 17.62 |

## deepseek-w4

Configuration: [configs/deepseek-w4.json](configs/deepseek-w4.json). Default mode: dspark-adaptive.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/plain/greedy.json) |
| dspark1/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/dspark1/greedy.json) |
| dspark2/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/dspark2/greedy.json) |
| dspark3/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/dspark3/greedy.json) |
| dspark5/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-w4/refresh/dspark5/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 42.11 | 40.91 |
| code | 1 | 64.87 | 61.86 |
| json | 1 | 77.89 | 73.25 |
| math | 1 | 62.83 | 59.41 |
| chat | 1 | 46.38 | 44.85 |
| prose | 2 | 64.90 | 62.94 |
| code | 2 | 79.66 | 76.12 |
| json | 2 | 108.55 | 102.03 |
| math | 2 | 99.74 | 93.77 |
| chat | 2 | 66.49 | 64.24 |
| prose | 4 | 81.66 | 79.15 |
| code | 4 | 102.71 | 98.60 |
| json | 4 | 114.77 | 109.30 |
| math | 4 | 104.70 | 100.18 |
| chat | 4 | 85.68 | 82.93 |
| prose | 8 | 68.58 | 66.82 |
| code | 8 | 87.00 | 84.19 |
| json | 8 | 98.88 | 95.19 |
| math | 8 | 99.29 | 95.33 |
| chat | 8 | 80.36 | 78.00 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 112 | 0.291 | 51.99 | 3.19 | 61.31 |
| 32768 | 32255 | 19.131 | 53.67 | 2.97 | 55.25 |
| 65536 | 65024 | 43.751 | 53.88 | 2.77 | 51.44 |
| 131072 | 130565 | 126.461 | 56.32 | 2.83 | 50.31 |

## deepseek-1m-w4

Configuration: [configs/deepseek-1m-w4.json](configs/deepseek-1m-w4.json). Default mode: dspark-adaptive.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/plain/greedy.json) |
| dspark1/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/dspark1/greedy.json) |
| dspark2/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/dspark2/greedy.json) |
| dspark3/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/dspark3/greedy.json) |
| dspark5/greedy | All | Fresh | 2bb443de41db | [Record](raw/deepseek-1m-w4/refresh/dspark5/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 41.84 | 40.76 |
| code | 1 | 65.02 | 61.93 |
| json | 1 | 78.15 | 73.72 |
| math | 1 | 62.85 | 59.67 |
| chat | 1 | 46.24 | 44.77 |
| prose | 2 | 64.43 | 62.38 |
| code | 2 | 80.47 | 76.91 |
| json | 2 | 108.22 | 102.16 |
| math | 2 | 99.69 | 93.53 |
| chat | 2 | 66.26 | 63.98 |
| prose | 4 | 81.26 | 78.78 |
| code | 4 | 104.46 | 100.20 |
| json | 4 | 116.73 | 111.24 |
| math | 4 | 105.56 | 100.91 |
| chat | 4 | 86.81 | 83.97 |
| prose | 8 | 68.90 | 67.10 |
| code | 8 | 101.95 | 97.86 |
| json | 8 | 100.60 | 96.66 |
| math | 8 | 90.44 | 87.11 |
| chat | 8 | 73.99 | 71.98 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 111 | 0.294 | 50.91 | 2.93 | 57.57 |
| 32768 | 32255 | 18.712 | 53.27 | 2.93 | 54.40 |
| 65536 | 65054 | 40.931 | 54.96 | 2.83 | 51.56 |
| 131072 | 130560 | 105.382 | 56.26 | 2.77 | 49.27 |
| 262144 | 261647 | 367.554 | 59.85 | 2.77 | 46.31 |
| 524288 | 523803 | 1908.016 | 66.60 | 2.93 | 44.01 |

## dsv4-w2

Configuration: [configs/dsv4-w2.json](configs/dsv4-w2.json). Default mode: dspark-adaptive.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/plain/greedy.json) |
| dspark1/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/dspark1/greedy.json) |
| dspark2/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/dspark2/greedy.json) |
| dspark3/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/dspark3/greedy.json) |
| dspark5/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w2/refresh/dspark5/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 40.77 | 39.51 |
| code | 1 | 62.15 | 58.65 |
| json | 1 | 73.56 | 68.63 |
| math | 1 | 58.78 | 55.49 |
| chat | 1 | 43.82 | 42.12 |
| prose | 2 | 55.72 | 54.00 |
| code | 2 | 78.10 | 74.11 |
| json | 2 | 99.11 | 92.98 |
| math | 2 | 89.45 | 83.82 |
| chat | 2 | 58.04 | 55.92 |
| prose | 4 | 70.58 | 68.19 |
| code | 4 | 88.39 | 84.55 |
| json | 4 | 107.24 | 101.36 |
| math | 4 | 91.72 | 87.37 |
| chat | 4 | 73.80 | 71.18 |
| prose | 8 | 68.50 | 65.66 |
| code | 8 | 92.01 | 87.32 |
| json | 8 | 103.87 | 97.32 |
| math | 8 | 92.77 | 87.32 |
| chat | 8 | 74.43 | 71.26 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 85 | 0.281 | 57.83 | 3.11 | 53.72 |
| 32768 | 32226 | 35.319 | 62.48 | 3.15 | 50.39 |
| 65536 | 64996 | 89.437 | 64.37 | 3.11 | 48.35 |
| 131072 | 130539 | 265.781 | 67.13 | 3.00 | 44.69 |
| 262144 | 261622 | 984.962 | 70.41 | 2.93 | 41.63 |
| 524288 | 523778 | 4679.601 | 80.99 | 3.23 | 39.85 |

## dsv4-w4

Configuration: [configs/dsv4-w4.json](configs/dsv4-w4.json). Default mode: dspark-adaptive.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/plain/greedy.json) |
| dspark1/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/dspark1/greedy.json) |
| dspark2/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/dspark2/greedy.json) |
| dspark3/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/dspark3/greedy.json) |
| dspark5/greedy | All | Fresh | 2bb443de41db | [Record](raw/dsv4-w4/refresh/dspark5/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 65.67 | 63.07 |
| code | 1 | 114.50 | 105.86 |
| json | 1 | 128.17 | 117.17 |
| math | 1 | 94.84 | 87.89 |
| chat | 1 | 70.83 | 67.63 |
| prose | 2 | 93.46 | 89.77 |
| code | 2 | 129.17 | 121.26 |
| json | 2 | 171.54 | 157.69 |
| math | 2 | 151.99 | 140.04 |
| chat | 2 | 97.86 | 93.37 |
| prose | 4 | 126.73 | 120.94 |
| code | 4 | 161.17 | 151.66 |
| json | 4 | 181.41 | 169.26 |
| math | 4 | 158.14 | 148.69 |
| chat | 4 | 130.35 | 124.30 |
| prose | 8 | 114.28 | 109.46 |
| code | 8 | 128.54 | 122.15 |
| json | 8 | 168.87 | 158.00 |
| math | 8 | 138.88 | 131.18 |
| chat | 8 | 117.42 | 112.23 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 85 | 0.190 | 36.17 | 3.04 | 83.92 |
| 32768 | 32226 | 26.237 | 39.58 | 3.49 | 88.26 |
| 65536 | 65026 | 70.847 | 41.08 | 3.27 | 79.57 |
| 131072 | 130536 | 225.675 | 44.85 | 3.36 | 75.05 |
| 262144 | 261627 | 897.362 | 49.41 | 3.49 | 70.70 |
| 524288 | 523777 | 4473.733 | 58.80 | 3.98 | 66.41 |

## mimo-w2

Configuration: [configs/mimo-w2.json](configs/mimo-w2.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/plain/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 43.55 | 41.67 |
| code | 1 | 46.07 | 43.58 |
| json | 1 | 48.84 | 46.20 |
| math | 1 | 46.19 | 43.25 |
| chat | 1 | 43.28 | 41.33 |
| prose | 2 | 59.03 | 56.81 |
| code | 2 | 62.74 | 59.54 |
| json | 2 | 66.94 | 63.49 |
| math | 2 | 66.55 | 62.38 |
| chat | 2 | 59.61 | 57.08 |
| prose | 4 | 80.50 | 77.46 |
| code | 4 | 82.91 | 80.12 |
| json | 4 | 86.76 | 83.56 |
| math | 4 | 84.06 | 80.83 |
| chat | 4 | 78.32 | 75.35 |
| prose | 8 | 78.34 | 74.55 |
| code | 8 | 81.17 | 76.92 |
| json | 8 | 87.39 | 82.70 |
| math | 8 | 84.93 | 80.53 |
| chat | 8 | 77.06 | 73.84 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 113 | 0.535 | 41.32 | 1.86 | 45.05 |
| 32768 | 32233 | 53.585 | 45.02 | 1.99 | 44.25 |
| 65536 | 65002 | 122.329 | 48.13 | 1.99 | 41.39 |
| 131072 | 130535 | 300.788 | 54.96 | 1.99 | 36.25 |

## mimo-w2-fp8kv

Configuration: [configs/mimo-w2-fp8kv.json](configs/mimo-w2-fp8kv.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/plain/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w2-fp8kv/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 42.65 | 40.86 |
| code | 1 | 45.07 | 42.76 |
| json | 1 | 47.74 | 45.09 |
| math | 1 | 45.80 | 42.86 |
| chat | 1 | 40.55 | 38.72 |
| prose | 2 | 57.52 | 55.34 |
| code | 2 | 62.01 | 58.84 |
| json | 2 | 66.28 | 62.82 |
| math | 2 | 65.61 | 61.51 |
| chat | 2 | 57.30 | 54.91 |
| prose | 4 | 77.77 | 74.95 |
| code | 4 | 81.72 | 79.05 |
| json | 4 | 86.98 | 83.83 |
| math | 4 | 83.28 | 80.16 |
| chat | 4 | 77.95 | 74.97 |
| prose | 8 | 77.13 | 73.45 |
| code | 8 | 80.89 | 77.04 |
| json | 8 | 86.78 | 82.29 |
| math | 8 | 84.86 | 80.07 |
| chat | 8 | 76.48 | 73.25 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 113 | 0.538 | 42.03 | 1.81 | 43.02 |
| 32768 | 32234 | 56.133 | 47.93 | 1.99 | 41.56 |
| 65536 | 65005 | 131.929 | 53.36 | 1.99 | 37.34 |
| 131072 | 130542 | 336.124 | 64.23 | 1.99 | 31.02 |
| 262144 | 261615 | 962.839 | 86.26 | 1.99 | 23.09 |

## mimo-w4

Configuration: [configs/mimo-w4.json](configs/mimo-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 75.36 | 72.03 |
| code | 1 | 81.34 | 76.97 |
| json | 1 | 84.84 | 80.46 |
| math | 1 | 83.71 | 78.15 |
| chat | 1 | 74.25 | 70.89 |
| prose | 2 | 103.27 | 99.39 |
| code | 2 | 112.12 | 107.06 |
| json | 2 | 119.41 | 113.78 |
| math | 2 | 118.89 | 112.59 |
| chat | 2 | 102.97 | 99.02 |
| prose | 4 | 141.00 | 136.21 |
| code | 4 | 147.66 | 142.24 |
| json | 4 | 152.38 | 146.38 |
| math | 4 | 148.59 | 142.70 |
| chat | 4 | 140.05 | 135.44 |
| prose | 8 | 137.28 | 130.95 |
| code | 8 | 144.53 | 137.26 |
| json | 8 | 155.37 | 148.09 |
| math | 8 | 150.54 | 142.79 |
| chat | 8 | 138.01 | 131.77 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 110 | 0.249 | 23.60 | 1.81 | 76.62 |
| 32768 | 32232 | 27.542 | 26.21 | 1.99 | 76.01 |
| 65536 | 65002 | 63.527 | 28.18 | 1.99 | 70.70 |
| 131072 | 130539 | 157.080 | 32.50 | 1.99 | 61.29 |

## mimo-1m-w4

Configuration: [configs/mimo-1m-w4.json](configs/mimo-1m-w4.json). Default mode: mtp1.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/plain/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/mimo-1m-w4/refresh/mtp3/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 75.73 | 72.40 |
| code | 1 | 81.91 | 77.38 |
| json | 1 | 85.66 | 81.19 |
| math | 1 | 84.67 | 79.29 |
| chat | 1 | 74.97 | 71.49 |
| prose | 2 | 103.59 | 99.69 |
| code | 2 | 113.19 | 107.71 |
| json | 2 | 120.31 | 114.66 |
| math | 2 | 119.56 | 112.86 |
| chat | 2 | 103.90 | 99.85 |
| prose | 4 | 141.08 | 136.48 |
| code | 4 | 148.10 | 142.77 |
| json | 4 | 152.70 | 147.10 |
| math | 4 | 146.58 | 140.25 |
| chat | 4 | 140.79 | 135.84 |
| prose | 8 | 138.78 | 132.46 |
| code | 8 | 145.45 | 138.18 |
| json | 8 | 155.68 | 148.35 |
| math | 8 | 149.93 | 140.86 |
| chat | 8 | 137.76 | 132.28 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 114 | 0.252 | 23.39 | 1.85 | 78.92 |
| 32768 | 32237 | 27.329 | 25.99 | 1.99 | 76.64 |
| 65536 | 65003 | 63.044 | 28.14 | 1.99 | 70.79 |
| 131072 | 130540 | 156.811 | 32.42 | 1.99 | 61.44 |
| 262144 | 261612 | 432.696 | 41.16 | 1.99 | 48.40 |
| 524288 | 523749 | 1334.900 | 58.86 | 1.99 | 33.85 |

## qwen27b-fp8-w1

Configuration: [configs/qwen27b-fp8-w1.json](configs/qwen27b-fp8-w1.json). Default mode: dflash2.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/plain/greedy.json) |
| mtp1/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/mtp3/greedy.json) |
| dflash2-fixed/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w1/refresh/dflash2-fixed/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 20.30 | 20.05 |
| code | 1 | 34.06 | 33.19 |
| json | 1 | 47.46 | 45.78 |
| math | 1 | 39.19 | 37.69 |
| chat | 1 | 19.87 | 19.61 |
| prose | 2 | 32.37 | 31.79 |
| code | 2 | 57.67 | 55.51 |
| json | 2 | 80.63 | 76.27 |
| math | 2 | 79.91 | 75.67 |
| chat | 2 | 33.93 | 33.27 |
| prose | 4 | 59.77 | 57.32 |
| code | 4 | 109.65 | 100.89 |
| json | 4 | 119.05 | 108.96 |
| math | 4 | 119.18 | 108.87 |
| chat | 4 | 67.73 | 64.46 |
| prose | 8 | 95.17 | 90.36 |
| code | 8 | 167.43 | 152.64 |
| json | 8 | 214.32 | 189.86 |
| math | 8 | 167.59 | 152.23 |
| chat | 8 | 107.23 | 101.00 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 158 | 0.692 | 140.96 | 4.81 | 34.13 |
| 32768 | 32285 | 30.856 | 157.38 | 5.00 | 31.77 |
| 65536 | 65053 | 71.119 | 168.79 | 4.64 | 27.47 |
| 131072 | 130573 | 180.085 | 192.01 | 4.64 | 24.15 |
| 262144 | 261655 | 512.813 | 243.65 | 3.92 | 16.10 |

## qwen27b-fp8-w2

Configuration: [configs/qwen27b-fp8-w2.json](configs/qwen27b-fp8-w2.json). Default mode: dflash2.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/plain/greedy.json) |
| mtp1/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/mtp3/greedy.json) |
| dflash2-fixed/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w2/refresh/dflash2-fixed/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 33.32 | 32.74 |
| code | 1 | 59.42 | 57.04 |
| json | 1 | 82.26 | 77.96 |
| math | 1 | 69.46 | 65.85 |
| chat | 1 | 35.51 | 34.81 |
| prose | 2 | 56.56 | 55.11 |
| code | 2 | 106.46 | 100.58 |
| json | 2 | 137.92 | 128.41 |
| math | 2 | 135.30 | 125.96 |
| chat | 2 | 59.93 | 58.22 |
| prose | 4 | 99.32 | 95.14 |
| code | 4 | 171.02 | 157.96 |
| json | 4 | 191.88 | 175.70 |
| math | 4 | 162.83 | 150.85 |
| chat | 4 | 108.85 | 103.67 |
| prose | 8 | 157.59 | 148.91 |
| code | 8 | 251.54 | 231.25 |
| json | 8 | 310.35 | 277.85 |
| math | 8 | 261.92 | 239.05 |
| chat | 8 | 175.04 | 165.22 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 157 | 0.396 | 79.19 | 5.00 | 63.03 |
| 32768 | 32284 | 21.066 | 88.52 | 5.00 | 56.48 |
| 65536 | 65051 | 47.196 | 95.09 | 4.55 | 47.89 |
| 131072 | 130575 | 114.381 | 108.53 | 3.92 | 36.15 |
| 262144 | 261659 | 307.051 | 135.91 | 3.92 | 28.87 |

## qwen27b-fp8-w4

Configuration: [configs/qwen27b-fp8-w4.json](configs/qwen27b-fp8-w4.json). Default mode: dflash2.

| Measurement | Prompt target tokens | Source | Binary SHA256 | Evidence |
| --- | --- | --- | --- | --- |
| default/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/greedy.json) |
| default/prefill | 2048, 8192, 32768 | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/prefill.json) |
| default/context-0 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-0.json) |
| default/context-32768 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-32768.json) |
| default/context-65536 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-65536.json) |
| default/context-131072 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-131072.json) |
| default/context-262144 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-262144.json) |
| default/context-524288 | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/default/context-524288.json) |
| plain/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/plain/greedy.json) |
| mtp1/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/mtp1/greedy.json) |
| mtp2/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/mtp2/greedy.json) |
| mtp3/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/mtp3/greedy.json) |
| dflash2-fixed/greedy | All | Fresh | 2bb443de41db | [Record](raw/qwen27b-fp8-w4/refresh/dflash2-fixed/greedy.json) |

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 60.73 | 58.72 |
| code | 1 | 95.62 | 90.26 |
| json | 1 | 131.47 | 122.23 |
| math | 1 | 105.07 | 97.59 |
| chat | 1 | 49.90 | 48.51 |
| prose | 2 | 87.64 | 84.62 |
| code | 2 | 155.17 | 144.36 |
| json | 2 | 210.91 | 191.48 |
| math | 2 | 192.58 | 174.11 |
| chat | 2 | 94.47 | 90.83 |
| prose | 4 | 149.59 | 143.41 |
| code | 4 | 240.42 | 224.85 |
| json | 4 | 264.64 | 245.38 |
| math | 4 | 230.77 | 216.49 |
| chat | 4 | 158.49 | 151.50 |
| prose | 8 | 221.91 | 210.28 |
| code | 8 | 317.30 | 293.98 |
| json | 8 | 377.06 | 345.31 |
| math | 8 | 321.96 | 298.82 |
| chat | 8 | 235.21 | 222.33 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 158 | 0.228 | 47.35 | 5.10 | 107.70 |
| 32768 | 32284 | 15.235 | 53.31 | 4.90 | 91.99 |
| 65536 | 65054 | 33.434 | 57.84 | 4.55 | 78.73 |
| 131072 | 130573 | 77.885 | 64.84 | 4.25 | 65.55 |
| 262144 | 261661 | 199.450 | 79.65 | 3.81 | 47.78 |
