# Current benchmark results

Source: `a895db96bb1749dcd5ec13ee8c2630ae3dc8762b`.

All values below are calculated by `summarize.py` from this directory's raw results. The saved `*.command.json` files identify commands, start/end times and exit codes. `manifest.json` identifies the binary, hardware records, checkpoints and datasets.

Deployments follow the [overview](../../../docs/benchmarks.md) order: model family, node count, then configuration options. KV labels describe the shared key/value-cache token pool (K = 1,024 tokens); slots are the configured concurrent-request limit.

## MiMo-V2.6-Flash MXFP4/FP8 · 4 nodes · 128K BF16 KV, 4 slots

Configuration: [mimo-w4](configs/mimo-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 76.8 [76.5–77.0] | 74.5 [71.4–74.6] | 22.75 | 1.75 | 77 |
| prose | 2 | 103.0 [103.0–103.2] | 99.4 [93.9–99.4] | 32.14 | 1.68 | 114 |
| prose | 4 | 138.0 [137.7–139.2] | 131.2 [121.7–132.3] | 49.39 | 1.76 | 211 |
| code | 1 | 81.3 [81.1–81.7] | 78.6 [75.1–78.7] | 23.42 | 1.90 | 77 |
| code | 2 | 108.2 [108.0–108.3] | 103.8 [97.2–104.3] | 33.20 | 1.81 | 115 |
| code | 4 | 144.5 [143.7–145.0] | 137.4 [126.6–138.0] | 50.62 | 1.88 | 210 |
| json | 1 | 85.8 [85.7–86.2] | 82.6 [78.1–83.3] | 23.04 | 1.98 | 77 |
| json | 2 | 119.6 [119.6–119.8] | 114.3 [105.8–114.3] | 33.05 | 1.98 | 115 |
| json | 4 | 149.7 [149.5–149.7] | 142.1 [130.6–142.2] | 49.09 | 1.95 | 198 |
| math | 1 | 83.9 [83.8–84.0] | 80.9 [76.1–81.0] | 23.19 | 1.95 | 77 |
| math | 2 | 116.4 [116.1–116.4] | 111.3 [102.7–112.0] | 32.70 | 1.92 | 115 |
| math | 4 | 144.8 [144.6–145.7] | 137.5 [127.5–137.5] | 50.02 | 1.87 | 211 |
| chat | 1 | 75.8 [75.7–75.9] | 73.5 [70.4–73.5] | 23.35 | 1.77 | 77 |
| chat | 2 | 103.1 [102.9–103.2] | 99.2 [93.3–99.2] | 32.95 | 1.75 | 115 |
| chat | 4 | 134.8 [134.7–134.9] | 128.7 [119.6–128.8] | 50.08 | 1.76 | 210 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 74.2 | 72.2 |
| prose | 4 | 124.3 | 117.1 |
| code | 1 | 78.9 | 74.7 |
| code | 4 | 138.6 | 132.1 |
| json | 1 | 85.3 | 82.4 |
| json | 4 | 154.1 | 145.8 |
| math | 1 | 82.5 | 79.6 |
| math | 4 | 147.2 | 138.8 |
| chat | 1 | 69.7 | 67.0 |
| chat | 4 | 127.0 | 121.0 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2076, 2075, 2054 | 1.771 | 1.656–1.774 | 0.853 | 1.799 |
| 8192 | 8160, 8206, 8385 | 6.547 | 6.431–6.793 | 0.798 | 6.571 |
| 32768 | 32940, 32976, 32935 | 28.428 | 28.426–28.468 | 0.863 | 28.486 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 57.4 | 17.43 | 1.00 | yes |
| plain | code | 57.6 | 17.36 | 1.00 | yes |
| plain | json | 57.7 | 17.32 | 1.00 | yes |
| plain | math | 57.9 | 17.27 | 1.00 | yes |
| plain | chat | 57.8 | 17.30 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 154/164 | 0 | 198 |
| gsm8k | 294/300 | 0 | 208 |
| extract | 100/100 | 0 | 58 |

Raw results: [raw/mimo-w4/](raw/mimo-w4/).

## MiMo-V2.6-Flash MXFP4/FP8 · 2 nodes · 128K BF16 KV, 4 slots

Configuration: [mimo-w2](configs/mimo-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 45.0 [45.0–45.0] | 44.0 [42.2–44.2] | 40.18 | 1.81 | 102 |
| prose | 2 | 58.5 [55.6–58.6] | 54.0 [53.7–57.0] | 57.34 | 1.71 | 180 |
| prose | 4 | 78.5 [76.6–79.0] | 74.0 [70.5–76.3] | 87.81 | 1.78 | 335 |
| code | 1 | 46.0 [43.0–46.1] | 42.6 [42.1–45.2] | 41.05 | 1.89 | 103 |
| code | 2 | 61.5 [61.5–61.6] | 59.8 [55.6–59.9] | 59.64 | 1.84 | 180 |
| code | 4 | 80.5 [79.2–81.4] | 76.4 [70.8–77.6] | 89.84 | 1.88 | 326 |
| json | 1 | 49.2 [49.2–49.2] | 48.1 [45.3–48.2] | 40.50 | 1.99 | 102 |
| json | 2 | 66.2 [62.9–67.0] | 61.1 [60.1–64.4] | 59.25 | 1.98 | 179 |
| json | 4 | 83.4 [81.5–84.0] | 78.5 [74.3–80.3] | 88.03 | 1.95 | 331 |
| math | 1 | 46.1 [42.9–46.1] | 42.1 [42.1–45.2] | 40.70 | 1.88 | 103 |
| math | 2 | 66.3 [66.2–66.3] | 64.4 [58.9–64.4] | 58.28 | 1.95 | 179 |
| math | 4 | 80.5 [80.0–82.7] | 77.2 [71.5–79.6] | 91.76 | 1.89 | 323 |
| chat | 1 | 42.3 [42.2–42.3] | 41.4 [39.5–41.4] | 40.73 | 1.72 | 102 |
| chat | 2 | 58.4 [55.9–58.7] | 54.3 [53.4–56.7] | 59.03 | 1.74 | 179 |
| chat | 4 | 74.3 [72.6–74.5] | 70.1 [67.0–71.6] | 89.67 | 1.73 | 332 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 41.8 | 39.3 |
| prose | 4 | 71.4 | 64.5 |
| code | 1 | 45.6 | 44.6 |
| code | 4 | 78.0 | 72.6 |
| json | 1 | 48.9 | 47.7 |
| json | 4 | 86.9 | 83.5 |
| math | 1 | 46.3 | 44.0 |
| math | 4 | 80.3 | 76.8 |
| chat | 1 | 40.3 | 37.8 |
| chat | 4 | 71.4 | 66.7 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2076, 2075, 2054 | 3.503 | 3.454–3.707 | 1.688 | 3.529 |
| 8192 | 8160, 8206, 8385 | 12.972 | 12.782–13.970 | 1.581 | 12.999 |
| 32768 | 32940, 32976, 32935 | 57.213 | 56.771–57.293 | 1.737 | 57.263 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 33.9 | 29.46 | 1.00 | yes |
| plain | code | 33.9 | 29.50 | 1.00 | yes |
| plain | json | 33.9 | 29.49 | 1.00 | yes |
| plain | math | 33.8 | 29.54 | 1.00 | yes |
| plain | chat | 33.9 | 29.53 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 154/164 | 0 | 202 |
| gsm8k | 295/300 | 0 | 209 |
| extract | 100/100 | 0 | 58 |

Raw results: [raw/mimo-w2/](raw/mimo-w2/).

## MiMo-V2.6-Flash MXFP4/FP8 · 2 nodes · 256K FP8 KV, 4 slots

Configuration: [mimo-w2-fp8kv](configs/mimo-w2-fp8kv.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 43.2 [43.2–43.3] | 42.1 [40.4–42.2] | 40.96 | 1.77 | 102 |
| prose | 2 | 56.2 [56.2–56.3] | 54.5 [51.6–54.5] | 58.14 | 1.68 | 191 |
| prose | 4 | 75.3 [75.2–75.4] | 72.1 [67.5–72.2] | 87.97 | 1.74 | 339 |
| code | 1 | 44.6 [44.6–44.7] | 43.5 [41.2–43.6] | 42.01 | 1.88 | 103 |
| code | 2 | 60.4 [60.4–60.5] | 58.5 [54.5–58.6] | 59.88 | 1.85 | 180 |
| code | 4 | 79.8 [79.6–80.0] | 76.2 [70.7–76.5] | 90.86 | 1.89 | 329 |
| json | 1 | 48.2 [48.2–48.2] | 46.8 [44.2–47.0] | 41.34 | 1.99 | 103 |
| json | 2 | 65.9 [65.9–65.9] | 63.4 [58.7–63.5] | 60.01 | 1.98 | 179 |
| json | 4 | 82.2 [82.1–82.3] | 78.5 [72.5–78.9] | 89.43 | 1.96 | 330 |
| math | 1 | 46.8 [46.8–46.9] | 45.7 [42.7–45.7] | 41.57 | 1.95 | 103 |
| math | 2 | 64.8 [64.7–64.8] | 62.4 [57.5–62.5] | 59.67 | 1.94 | 180 |
| math | 4 | 79.0 [78.9–79.2] | 75.5 [69.7–75.8] | 89.71 | 1.86 | 329 |
| chat | 1 | 39.9 [39.9–40.0] | 38.9 [37.3–38.9] | 41.76 | 1.67 | 103 |
| chat | 2 | 57.9 [57.9–58.0] | 56.0 [52.5–56.1] | 59.87 | 1.75 | 179 |
| chat | 4 | 76.0 [75.7–76.1] | 72.5 [67.8–72.7] | 91.18 | 1.77 | 338 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 41.1 | 40.1 |
| prose | 4 | 69.5 | 66.8 |
| code | 1 | 44.7 | 43.4 |
| code | 4 | 77.7 | 74.4 |
| json | 1 | 47.9 | 46.6 |
| json | 4 | 85.3 | 81.4 |
| math | 1 | 44.8 | 43.8 |
| math | 4 | 80.7 | 77.2 |
| chat | 1 | 39.2 | 38.4 |
| chat | 4 | 72.0 | 69.1 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2076, 2075, 2054 | 3.486 | 3.285–3.489 | 1.679 | 3.505 |
| 8192 | 8160, 8206, 8385 | 13.201 | 13.000–13.628 | 1.609 | 13.238 |
| 32768 | 32940, 32976, 32935 | 58.513 | 58.490–59.400 | 1.776 | 58.560 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 33.4 | 29.98 | 1.00 | yes |
| plain | code | 33.2 | 30.15 | 1.00 | yes |
| plain | json | 33.3 | 30.07 | 1.00 | yes |
| plain | math | 33.1 | 30.18 | 1.00 | yes |
| plain | chat | 33.2 | 30.16 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 153/164 | 0 | 202 |
| gsm8k | 294/300 | 0 | 210 |
| extract | 100/100 | 0 | 58 |

Raw results: [raw/mimo-w2-fp8kv/](raw/mimo-w2-fp8kv/).

## Selection microbenchmarks

Medians in microseconds. Cold denotes L2 eviction outside the timed interval. All commands use five warmups and thirty iterations.

### Qwen QSA

| Rows | Pools | Cache | Stage | Median µs | Min µs | p95 µs |
|---|---|---|---|---|---|---|

### DeepSeek CSA2

`combined` times one candidate and one restricted call; the model has five restricted calls per full decoder pass. These component timings must not be added to predict service throughput.

| Rows | Context | Cache | Stage | Median µs | Min µs | Max µs |
|---|---|---|---|---|---|---|
