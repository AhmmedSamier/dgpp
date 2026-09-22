# Current benchmark results

Source: `a3ed8994a34cf05f79e63dc1ad51ffd1f65c9f82`.

All values below are calculated by `summarize.py` from this directory's raw results. The saved `*.command.json` files identify commands, start/end times and exit codes. `manifest.json` identifies the binary, hardware records, checkpoints and datasets.

[Environment comparison](environment.md) · [Interruption record](interruptions.md) · [Batching diagnostic](isolation.md)

## GLM-5.3-Flash NVFP4/FP8 · 4 nodes

Configuration: [glm-flash-hybrid-w4](configs/glm-flash-hybrid-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 59.1 [59.0–59.1] | 58.0 [55.4–58.0] | 31.52 | 1.86 | 76 |
| prose | 2 | 79.5 [79.5–79.7] | 77.7 [73.5–77.7] | 42.74 | 1.74 | 127 |
| prose | 4 | 107.9 [106.2–108.3] | 103.3 [97.0–105.2] | 64.08 | 1.77 | 229 |
| code | 1 | 60.3 [59.8–60.4] | 59.0 [56.4–59.3] | 32.76 | 1.98 | 76 |
| code | 2 | 83.0 [82.6–84.3] | 81.5 [75.5–82.8] | 45.05 | 1.88 | 114 |
| code | 4 | 110.5 [109.7–110.5] | 107.2 [98.3–107.4] | 65.48 | 1.86 | 236 |
| json | 1 | 61.8 [61.8–61.8] | 60.8 [57.9–60.8] | 31.74 | 1.96 | 76 |
| json | 2 | 79.9 [79.9–80.0] | 78.4 [73.5–78.4] | 43.15 | 1.83 | 115 |
| json | 4 | 117.7 [117.4–117.7] | 114.1 [104.4–114.1] | 64.68 | 1.95 | 236 |
| math | 1 | 59.6 [59.6–59.7] | 58.5 [55.4–58.7] | 32.17 | 1.92 | 77 |
| math | 2 | 86.5 [86.5–86.7] | 84.5 [78.4–84.7] | 44.34 | 1.94 | 127 |
| math | 4 | 108.8 [106.2–109.6] | 106.0 [95.2–106.6] | 65.31 | 1.86 | 230 |
| chat | 1 | 54.0 [54.0–54.1] | 53.1 [51.0–53.1] | 31.89 | 1.72 | 76 |
| chat | 2 | 79.3 [79.2–79.6] | 77.5 [72.9–77.8] | 43.21 | 1.78 | 139 |
| chat | 4 | 105.8 [105.6–106.8] | 102.8 [95.5–103.8] | 65.12 | 1.80 | 229 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 58.3 | 57.3 |
| prose | 4 | 106.2 | 103.3 |
| code | 1 | 60.1 | 59.0 |
| code | 4 | 108.0 | 105.2 |
| json | 1 | 60.9 | 60.0 |
| json | 4 | 109.7 | 106.3 |
| math | 1 | 58.1 | 57.0 |
| math | 4 | 112.1 | 109.0 |
| chat | 1 | 54.3 | 53.3 |
| chat | 4 | 105.0 | 102.0 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 1.497 | 1.487–1.527 | 0.722 | 1.512 |
| 8192 | 8257, 8302, 8429 | 5.847 | 5.747–5.865 | 0.696 | 5.872 |
| 32768 | 33245, 33321, 33241 | 26.686 | 26.626–26.806 | 0.801 | 26.703 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 39.7 | 25.18 | 1.00 | yes |
| plain | code | 39.7 | 25.17 | 1.00 | yes |
| plain | json | 39.8 | 25.12 | 1.00 | yes |
| plain | math | 39.8 | 25.15 | 1.00 | yes |
| plain | chat | 39.8 | 25.12 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 157/164 | 0 | 170 |
| gsm8k | 292/300 | 0 | 77 |
| extract | 100/100 | 0 | 51 |

Raw results: [raw/glm-flash-hybrid-w4/](raw/glm-flash-hybrid-w4/).

## GLM-5.3-Flash NVFP4/FP8 · 2 nodes

Configuration: [glm-flash-hybrid-w2](configs/glm-flash-hybrid-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 35.2 [35.1–35.2] | 34.5 [33.0–34.6] | 54.12 | 1.90 | 127 |
| prose | 2 | 44.7 [44.6–44.7] | 43.9 [41.4–44.0] | 77.60 | 1.77 | 177 |
| prose | 4 | 58.9 [58.7–59.6] | 57.8 [53.4–58.3] | 116.56 | 1.77 | 337 |
| code | 1 | 35.7 [35.7–35.7] | 35.4 [33.3–35.4] | 55.33 | 1.98 | 101 |
| code | 2 | 45.8 [45.8–45.8] | 45.1 [42.1–45.2] | 79.55 | 1.85 | 164 |
| code | 4 | 58.4 [58.2–59.3] | 56.7 [52.7–57.8] | 121.24 | 1.83 | 411 |
| json | 1 | 36.0 [35.9–36.0] | 35.6 [33.7–35.6] | 54.56 | 1.96 | 103 |
| json | 2 | 44.5 [44.4–44.5] | 43.9 [41.0–44.0] | 77.54 | 1.83 | 167 |
| json | 4 | 63.4 [63.4–63.5] | 61.8 [56.8–61.9] | 119.17 | 1.95 | 388 |
| math | 1 | 33.9 [33.9–33.9] | 33.5 [31.6–33.5] | 55.26 | 1.88 | 102 |
| math | 2 | 47.5 [47.4–47.6] | 46.7 [43.2–46.8] | 80.62 | 1.92 | 204 |
| math | 4 | 59.2 [59.2–59.4] | 57.9 [53.5–58.0] | 121.37 | 1.86 | 350 |
| chat | 1 | 31.3 [31.3–31.3] | 30.9 [29.6–31.0] | 54.63 | 1.71 | 103 |
| chat | 2 | 44.3 [44.3–44.3] | 43.5 [40.8–43.5] | 78.27 | 1.78 | 216 |
| chat | 4 | 57.7 [57.4–58.0] | 56.3 [52.4–56.8] | 120.55 | 1.78 | 334 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 34.5 | 33.2 |
| prose | 4 | 58.1 | 57.0 |
| code | 1 | 35.0 | 33.4 |
| code | 4 | 57.8 | 56.7 |
| json | 1 | 35.3 | 34.9 |
| json | 4 | 61.3 | 59.5 |
| math | 1 | 33.9 | 33.4 |
| math | 4 | 59.6 | 58.3 |
| chat | 1 | 32.0 | 30.5 |
| chat | 4 | 57.1 | 55.0 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 2.380 | 2.333–2.415 | 1.148 | 2.388 |
| 8192 | 8257, 8302, 8429 | 9.204 | 9.145–9.360 | 1.110 | 9.210 |
| 32768 | 33245, 33321, 33241 | 39.950 | 39.810–39.954 | 1.199 | 39.976 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 24.1 | 41.56 | 1.00 | yes |
| plain | code | 24.0 | 41.68 | 1.00 | yes |
| plain | json | 24.0 | 41.70 | 1.00 | yes |
| plain | math | 24.0 | 41.73 | 1.00 | yes |
| plain | chat | 24.0 | 41.71 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 157/164 | 0 | 176 |
| gsm8k | 292/300 | 0 | 77 |
| extract | 100/100 | 0 | 51 |

Raw results: [raw/glm-flash-hybrid-w2/](raw/glm-flash-hybrid-w2/).

## GLM-5.3-Flash FP8 · 4 nodes

Configuration: [glm-flash-fp8-w4](configs/glm-flash-fp8-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 50.2 [50.2–50.3] | 49.4 [47.0–49.4] | 37.32 | 1.88 | 101 |
| prose | 2 | 64.7 [58.8–64.8] | 59.6 [57.8–63.4] | 55.12 | 1.80 | 153 |
| prose | 4 | 80.6 [74.3–81.5] | 73.1 [72.8–79.7] | 86.08 | 1.80 | 271 |
| code | 1 | 51.0 [50.9–51.0] | 50.2 [47.4–50.2] | 38.49 | 1.96 | 77 |
| code | 2 | 66.6 [58.8–67.1] | 65.6 [54.3–65.9] | 56.75 | 1.91 | 128 |
| code | 4 | 80.4 [73.5–80.5] | 78.4 [67.0–78.5] | 88.69 | 1.83 | 287 |
| json | 1 | 52.1 [43.8–52.2] | 48.6 [43.4–51.4] | 37.95 | 1.98 | 76 |
| json | 2 | 64.3 [58.6–64.4] | 58.9 [57.7–63.4] | 55.50 | 1.88 | 128 |
| json | 4 | 87.2 [79.6–87.2] | 78.1 [77.8–85.1] | 86.67 | 1.96 | 279 |
| math | 1 | 48.8 [48.8–48.9] | 48.0 [45.2–48.2] | 38.42 | 1.88 | 102 |
| math | 2 | 66.6 [58.8–66.6] | 60.3 [57.9–65.2] | 56.75 | 1.92 | 153 |
| math | 4 | 81.8 [75.6–82.8] | 74.4 [74.2–80.0] | 88.63 | 1.89 | 259 |
| chat | 1 | 45.4 [39.2–45.6] | 42.6 [38.8–44.9] | 37.95 | 1.72 | 101 |
| chat | 2 | 62.5 [62.4–62.7] | 61.1 [57.3–61.4] | 55.62 | 1.78 | 165 |
| chat | 4 | 75.1 [73.7–77.1] | 73.3 [67.5–75.4] | 93.01 | 1.81 | 259 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 49.7 | 48.8 |
| prose | 4 | 78.9 | 77.0 |
| code | 1 | 50.9 | 50.2 |
| code | 4 | 78.6 | 76.9 |
| json | 1 | 51.1 | 50.5 |
| json | 4 | 83.1 | 81.2 |
| math | 1 | 49.1 | 48.3 |
| math | 4 | 82.3 | 80.4 |
| chat | 1 | 47.2 | 46.4 |
| chat | 4 | 78.2 | 76.2 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 1.714 | 1.702–1.760 | 0.827 | 1.740 |
| 8192 | 8257, 8302, 8429 | 6.508 | 6.464–6.607 | 0.784 | 6.525 |
| 32768 | 33245, 33321, 33241 | 29.140 | 28.686–29.666 | 0.875 | 29.161 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 34.3 | 29.11 | 1.00 | yes |
| plain | code | 34.4 | 29.10 | 1.00 | yes |
| plain | json | 34.4 | 29.10 | 1.00 | yes |
| plain | math | 34.4 | 29.09 | 1.00 | yes |
| plain | chat | 34.3 | 29.12 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 154/164 | 0 | 164 |
| gsm8k | 295/300 | 0 | 76 |
| extract | 100/100 | 0 | 52 |

Raw results: [raw/glm-flash-fp8-w4/](raw/glm-flash-fp8-w4/).

## Qwen3.8-Flash-Next FP8 · 4 nodes

Configuration: [qwen-fp8-w4](configs/qwen-fp8-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 72.6 [71.8–73.1] | 71.1 [68.7–71.7] | 23.73 | 1.72 | 76 |
| prose | 2 | 108.8 [108.2–109.6] | 105.6 [100.3–106.8] | 29.86 | 1.66 | 114 |
| prose | 4 | 149.0 [148.9–149.6] | 143.3 [134.7–144.3] | 43.34 | 1.67 | 206 |
| code | 1 | 81.4 [81.4–81.5] | 79.7 [76.2–79.7] | 23.72 | 1.93 | 78 |
| code | 2 | 124.6 [124.6–125.4] | 120.5 [113.7–121.7] | 29.67 | 1.88 | 116 |
| code | 4 | 168.8 [167.6–170.4] | 160.8 [154.3–161.6] | 43.17 | 1.89 | 203 |
| json | 1 | 83.3 [83.2–83.3] | 81.3 [78.2–81.8] | 23.74 | 1.98 | 77 |
| json | 2 | 132.0 [131.8–132.1] | 127.2 [120.4–127.5] | 29.96 | 1.99 | 116 |
| json | 4 | 172.3 [172.2–172.7] | 164.9 [154.8–165.7] | 43.53 | 1.97 | 205 |
| math | 1 | 78.9 [78.9–78.9] | 77.3 [74.4–77.5] | 23.78 | 1.88 | 78 |
| math | 2 | 129.3 [129.3–129.3] | 124.7 [118.0–125.0] | 29.88 | 1.95 | 116 |
| math | 4 | 167.3 [167.1–167.5] | 160.5 [150.8–160.6] | 43.23 | 1.89 | 205 |
| chat | 1 | 65.3 [65.0–66.2] | 63.7 [62.8–64.0] | 24.30 | 1.59 | 77 |
| chat | 2 | 111.8 [110.9–112.9] | 107.5 [104.7–108.2] | 30.01 | 1.72 | 141 |
| chat | 4 | 153.8 [150.8–157.0] | 148.1 [136.1–150.9] | 43.92 | 1.73 | 203 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 71.2 | 69.1 |
| prose | 4 | 152.9 | 145.3 |
| code | 1 | 78.0 | 75.5 |
| code | 4 | 166.2 | 158.8 |
| json | 1 | 83.4 | 81.0 |
| json | 4 | 177.4 | 168.9 |
| math | 1 | 80.7 | 78.4 |
| math | 4 | 169.9 | 162.8 |
| chat | 1 | 71.8 | 68.1 |
| chat | 4 | 155.1 | 148.6 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.113 | 1.064–1.167 | 0.542 | 1.122 |
| 8192 | 8091, 8132, 8300 | 4.440 | 4.419–4.596 | 0.546 | 4.477 |
| 32768 | 32664, 32698, 32653 | 18.315 | 18.221–18.402 | 0.561 | 18.366 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 78.9 | 27.64 | 2.18 | yes |
| depth2 | code | 99.8 | 27.78 | 2.77 | yes |
| depth2 | json | 106.6 | 27.81 | 2.97 | yes |
| depth2 | math | 94.8 | 28.03 | 2.66 | yes |
| depth2 | chat | 71.7 | 27.88 | 2.00 | yes |
| plain | prose | 50.6 | 19.76 | 1.00 | yes |
| plain | code | 50.6 | 19.76 | 1.00 | yes |
| plain | json | 50.8 | 19.69 | 1.00 | yes |
| plain | math | 50.6 | 19.77 | 1.00 | yes |
| plain | chat | 50.8 | 19.70 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 158/164 | 0 | 234 |
| gsm8k | 291/300 | 0 | 353 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-fp8-w4/](raw/qwen-fp8-w4/).

## Qwen3.8-Flash-Next FP8 · 2 nodes

Configuration: [qwen-fp8-w2](configs/qwen-fp8-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 48.4 [48.4–48.4] | 47.6 [46.8–47.6] | 36.07 | 1.75 | 102 |
| prose | 2 | 68.5 [68.5–68.5] | 67.0 [64.2–67.0] | 45.67 | 1.66 | 152 |
| prose | 4 | 92.2 [91.9–93.4] | 89.3 [84.7–90.6] | 70.49 | 1.67 | 272 |
| code | 1 | 53.4 [53.3–53.4] | 52.5 [50.6–52.5] | 36.21 | 1.93 | 77 |
| code | 2 | 79.6 [79.6–79.9] | 77.7 [74.0–77.8] | 47.08 | 1.89 | 152 |
| code | 4 | 103.9 [103.8–104.1] | 100.2 [96.2–100.5] | 70.72 | 1.90 | 273 |
| json | 1 | 54.8 [54.7–54.8] | 53.8 [51.9–53.9] | 36.38 | 1.99 | 102 |
| json | 2 | 83.3 [83.3–83.3] | 81.2 [77.4–81.2] | 47.46 | 1.98 | 153 |
| json | 4 | 105.4 [105.0–105.4] | 101.7 [96.4–102.0] | 70.62 | 1.97 | 272 |
| math | 1 | 51.1 [51.1–51.1] | 50.3 [48.8–50.3] | 36.44 | 1.86 | 102 |
| math | 2 | 81.9 [81.9–81.9] | 79.7 [75.9–79.9] | 47.18 | 1.95 | 153 |
| math | 4 | 102.2 [102.1–102.3] | 98.8 [93.8–99.0] | 71.31 | 1.88 | 273 |
| chat | 1 | 44.7 [44.7–44.7] | 44.1 [42.8–44.1] | 36.32 | 1.62 | 101 |
| chat | 2 | 72.5 [72.4–72.6] | 70.7 [67.9–71.0] | 46.60 | 1.72 | 152 |
| chat | 4 | 94.6 [94.4–94.8] | 91.6 [86.9–92.1] | 70.91 | 1.73 | 272 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 47.6 | 46.9 |
| prose | 4 | 91.0 | 88.0 |
| code | 1 | 52.7 | 51.0 |
| code | 4 | 102.1 | 97.3 |
| json | 1 | 54.0 | 52.9 |
| json | 4 | 108.2 | 104.1 |
| math | 1 | 51.1 | 50.2 |
| math | 4 | 103.9 | 99.0 |
| chat | 1 | 46.2 | 45.6 |
| chat | 4 | 93.1 | 89.9 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.311 | 1.257–1.383 | 0.638 | 1.327 |
| 8192 | 8091, 8132, 8300 | 5.178 | 5.156–5.408 | 0.637 | 5.215 |
| 32768 | 32664, 32698, 32653 | 21.110 | 21.086–21.127 | 0.646 | 21.168 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 50.4 | 43.22 | 2.18 | yes |
| depth2 | code | 61.4 | 43.28 | 2.66 | yes |
| depth2 | json | 68.1 | 43.56 | 2.97 | yes |
| depth2 | math | 61.6 | 43.61 | 2.68 | yes |
| depth2 | chat | 48.4 | 43.53 | 2.11 | yes |
| plain | prose | 35.8 | 27.91 | 1.00 | yes |
| plain | code | 35.8 | 27.96 | 1.00 | yes |
| plain | json | 35.8 | 27.92 | 1.00 | yes |
| plain | math | 35.7 | 28.00 | 1.00 | yes |
| plain | chat | 35.9 | 27.83 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 160/164 | 0 | 235 |
| gsm8k | 292/300 | 0 | 349 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-fp8-w2/](raw/qwen-fp8-w2/).

## Qwen3.8-Flash-Next NVFP4 · 1 node

Configuration: [qwen-nvfp4-w1](configs/qwen-nvfp4-w1.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 45.5 [44.9–45.6] | 44.8 [43.0–44.8] | 39.23 | 1.78 | 102 |
| prose | 2 | 63.1 [62.1–63.1] | 61.7 [57.9–61.8] | 51.48 | 1.63 | 152 |
| prose | 4 | 85.8 [84.8–85.8] | 83.4 [77.7–83.4] | 75.75 | 1.69 | 276 |
| code | 1 | 49.0 [48.2–49.0] | 48.2 [45.4–48.2] | 39.46 | 1.93 | 103 |
| code | 2 | 72.5 [71.4–72.5] | 70.8 [65.8–71.0] | 51.75 | 1.90 | 141 |
| code | 4 | 95.5 [95.2–95.9] | 92.3 [86.9–92.5] | 75.95 | 1.88 | 298 |
| json | 1 | 50.0 [49.3–50.0] | 49.3 [46.5–49.3] | 39.52 | 1.98 | 77 |
| json | 2 | 74.9 [73.9–75.0] | 73.1 [68.1–73.1] | 52.68 | 1.98 | 166 |
| json | 4 | 98.7 [97.9–98.9] | 95.4 [88.7–95.4] | 77.16 | 1.97 | 285 |
| math | 1 | 47.5 [46.8–47.7] | 46.6 [44.1–46.8] | 39.76 | 1.89 | 102 |
| math | 2 | 73.1 [72.0–73.1] | 71.2 [66.0–71.2] | 52.45 | 1.92 | 166 |
| math | 4 | 95.6 [95.5–95.9] | 92.4 [86.6–92.8] | 76.53 | 1.88 | 289 |
| chat | 1 | 42.3 [41.7–42.4] | 41.8 [39.8–41.8] | 39.64 | 1.68 | 103 |
| chat | 2 | 66.4 [65.6–66.5] | 64.9 [61.3–65.0] | 51.52 | 1.73 | 168 |
| chat | 4 | 86.9 [86.7–87.4] | 84.5 [79.1–85.0] | 75.71 | 1.72 | 279 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 42.3 | 41.7 |
| prose | 4 | 84.4 | 82.1 |
| code | 1 | 47.2 | 44.8 |
| code | 4 | 93.8 | 90.2 |
| json | 1 | 49.9 | 49.1 |
| json | 4 | 100.1 | 96.8 |
| math | 1 | 47.1 | 46.1 |
| math | 4 | 94.9 | 91.4 |
| chat | 1 | 41.3 | 40.1 |
| chat | 4 | 88.3 | 84.5 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.839 | 1.798–1.966 | 0.895 | 1.857 |
| 8192 | 8091, 8132, 8300 | 7.485 | 7.459–7.920 | 0.925 | 7.509 |
| 32768 | 32664, 32698, 32653 | 30.266 | 30.207–30.710 | 0.926 | 30.326 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 47.3 | 47.29 | 2.24 | yes |
| depth2 | code | 58.8 | 47.64 | 2.80 | yes |
| depth2 | json | 60.7 | 47.74 | 2.90 | yes |
| depth2 | math | 54.2 | 48.01 | 2.60 | yes |
| depth2 | chat | 44.0 | 47.92 | 2.11 | yes |
| plain | prose | 32.8 | 30.45 | 1.00 | yes |
| plain | code | 32.7 | 30.55 | 1.00 | yes |
| plain | json | 32.8 | 30.53 | 1.00 | yes |
| plain | math | 32.7 | 30.57 | 1.00 | yes |
| plain | chat | 32.8 | 30.50 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 160/164 | 0 | 237 |
| gsm8k | 294/300 | 0 | 355 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-nvfp4-w1/](raw/qwen-nvfp4-w1/).

## Qwen3.8-Flash-Next NVFP4 · 2 nodes

Configuration: [qwen-nvfp4-w2](configs/qwen-nvfp4-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 69.8 [67.7–70.3] | 68.1 [64.2–68.5] | 25.73 | 1.80 | 76 |
| prose | 2 | 95.7 [90.8–95.8] | 93.5 [84.2–93.5] | 33.11 | 1.64 | 114 |
| prose | 4 | 129.3 [127.8–129.9] | 125.2 [115.6–125.6] | 50.55 | 1.67 | 210 |
| code | 1 | 72.7 [69.4–72.8] | 71.4 [64.6–71.4] | 26.16 | 1.90 | 76 |
| code | 2 | 111.7 [106.6–112.3] | 108.4 [96.6–109.1] | 34.06 | 1.92 | 114 |
| code | 4 | 143.4 [140.3–147.2] | 134.9 [129.6–141.5] | 51.17 | 1.88 | 230 |
| json | 1 | 74.6 [69.9–74.6] | 73.0 [65.5–73.0] | 26.29 | 1.96 | 76 |
| json | 2 | 114.1 [108.1–114.4] | 110.3 [98.8–110.9] | 34.63 | 1.98 | 127 |
| json | 4 | 149.5 [145.9–151.6] | 143.8 [130.5–145.8] | 50.55 | 1.97 | 215 |
| math | 1 | 72.4 [68.9–72.4] | 70.9 [64.6–71.0] | 26.29 | 1.90 | 76 |
| math | 2 | 110.9 [106.4–111.2] | 107.4 [96.2–108.0] | 34.58 | 1.93 | 127 |
| math | 4 | 147.0 [143.6–148.2] | 141.4 [128.0–142.4] | 49.92 | 1.89 | 221 |
| chat | 1 | 62.6 [59.4–62.7] | 61.5 [56.1–61.5] | 26.29 | 1.65 | 76 |
| chat | 2 | 99.8 [95.4–99.9] | 96.6 [88.2–97.0] | 34.06 | 1.71 | 127 |
| chat | 4 | 129.6 [128.5–132.3] | 124.5 [116.9–127.9] | 50.11 | 1.72 | 210 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 60.4 | 59.3 |
| prose | 4 | 128.1 | 123.6 |
| code | 1 | 67.5 | 64.5 |
| code | 4 | 142.5 | 134.3 |
| json | 1 | 72.0 | 70.0 |
| json | 4 | 150.6 | 143.9 |
| math | 1 | 67.5 | 65.9 |
| math | 4 | 141.5 | 133.9 |
| chat | 1 | 61.9 | 60.7 |
| chat | 4 | 130.0 | 124.9 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.311 | 1.279–1.375 | 0.638 | 1.338 |
| 8192 | 8091, 8132, 8300 | 5.188 | 5.170–5.434 | 0.641 | 5.224 |
| 32768 | 32664, 32698, 32653 | 20.871 | 20.863–21.119 | 0.639 | 20.940 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 72.1 | 31.02 | 2.24 | yes |
| depth2 | code | 84.8 | 31.32 | 2.66 | yes |
| depth2 | json | 92.3 | 31.40 | 2.90 | yes |
| depth2 | math | 86.7 | 31.64 | 2.74 | yes |
| depth2 | chat | 68.1 | 31.46 | 2.14 | yes |
| plain | prose | 47.0 | 21.28 | 1.00 | yes |
| plain | code | 47.4 | 21.08 | 1.00 | yes |
| plain | json | 47.5 | 21.05 | 1.00 | yes |
| plain | math | 47.4 | 21.10 | 1.00 | yes |
| plain | chat | 47.6 | 21.03 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 159/164 | 0 | 238 |
| gsm8k | 293/300 | 0 | 358 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-nvfp4-w2/](raw/qwen-nvfp4-w2/).

## GLM-5.3 int4/int8 · 4 nodes

Configuration: [glm53-w4](configs/glm53-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 29.8 [29.7–29.8] | 29.3 [27.3–29.3] | 62.94 | 1.88 | 126 |
| prose | 2 | 38.4 [38.4–38.4] | 37.8 [34.4–37.8] | 89.63 | 1.74 | 190 |
| prose | 4 | 48.0 [47.8–48.4] | 46.9 [42.1–47.3] | 145.52 | 1.81 | 382 |
| prose | 8 | 56.5 [56.3–56.8] | 53.1 [47.2–53.5] | 239.45 | 1.80 | 896 |
| code | 1 | 30.6 [30.6–30.6] | 30.1 [27.7–30.2] | 64.10 | 1.96 | 102 |
| code | 2 | 40.6 [40.6–40.8] | 39.8 [35.8–40.1] | 92.46 | 1.90 | 202 |
| code | 4 | 47.2 [47.0–47.5] | 46.0 [41.4–46.5] | 151.07 | 1.82 | 381 |
| code | 8 | 54.4 [54.1–54.9] | 51.1 [46.0–51.4] | 249.60 | 1.81 | 906 |
| json | 1 | 30.4 [30.4–30.4] | 29.9 [27.6–29.9] | 63.62 | 1.93 | 127 |
| json | 2 | 38.8 [38.8–39.0] | 38.0 [34.5–38.3] | 90.72 | 1.83 | 191 |
| json | 4 | 50.7 [50.6–50.8] | 49.4 [44.0–49.6] | 149.10 | 1.95 | 394 |
| json | 8 | 57.4 [57.2–58.0] | 54.2 [47.8–54.7] | 234.63 | 1.89 | 890 |
| math | 1 | 30.0 [29.9–30.1] | 29.6 [26.9–29.6] | 63.81 | 1.92 | 127 |
| math | 2 | 41.4 [41.4–41.5] | 40.5 [36.1–40.7] | 93.77 | 1.95 | 203 |
| math | 4 | 47.3 [47.3–47.7] | 46.2 [41.2–46.6] | 148.57 | 1.87 | 386 |
| math | 8 | 56.7 [56.5–57.2] | 53.3 [47.2–53.5] | 249.64 | 1.90 | 897 |
| chat | 1 | 26.8 [26.7–26.8] | 26.5 [24.6–26.5] | 63.45 | 1.70 | 102 |
| chat | 2 | 38.6 [38.5–38.6] | 37.9 [34.3–37.9] | 91.18 | 1.78 | 204 |
| chat | 4 | 47.0 [47.0–47.4] | 45.9 [41.6–46.0] | 149.79 | 1.81 | 388 |
| chat | 8 | 56.4 [56.3–56.6] | 53.1 [47.3–53.4] | 245.12 | 1.85 | 885 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 29.5 | 29.2 |
| prose | 8 | 55.1 | 52.0 |
| code | 1 | 30.2 | 29.8 |
| code | 8 | 55.3 | 52.2 |
| json | 1 | 31.0 | 30.2 |
| json | 8 | 58.6 | 55.2 |
| math | 1 | 29.9 | 29.5 |
| math | 8 | 57.3 | 53.6 |
| chat | 1 | 27.7 | 26.7 |
| chat | 8 | 54.6 | 51.5 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 4.785 | 4.698–4.932 | 2.308 | 4.792 |
| 8192 | 8257, 8302, 8429 | 24.209 | 23.656–24.336 | 2.872 | 24.215 |
| 32768 | 33245, 33321, 33241 | 169.138 | 168.798–169.454 | 5.086 | 169.155 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 32.0 | 78.12 | 2.50 | yes |
| depth2 | code | 32.8 | 80.17 | 2.63 | yes |
| depth2 | json | 35.0 | 79.13 | 2.77 | yes |
| depth2 | math | 30.9 | 80.01 | 2.48 | yes |
| depth2 | chat | 28.0 | 79.29 | 2.22 | yes |
| plain | prose | 20.1 | 49.71 | 1.00 | yes |
| plain | code | 20.1 | 49.74 | 1.00 | yes |
| plain | json | 20.0 | 49.93 | 1.00 | yes |
| plain | math | 19.8 | 50.56 | 1.00 | yes |
| plain | chat | 20.1 | 49.76 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 159/164 | 0 | 146 |
| gsm8k | 292/300 | 0 | 91 |
| extract | 100/100 | 0 | 58 |

Raw results: [raw/glm53-w4/](raw/glm53-w4/).

## DeepSeek-V4.1-Flash · 4 nodes

Configuration: [deepseek-w4](configs/deepseek-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 39.8 [38.5–39.9] | 38.8 [37.6–38.9] | 46.41 | 1.85 | 152 |
| prose | 2 | 62.7 [60.7–62.7] | 60.8 [58.4–61.0] | 60.70 | 1.95 | 202 |
| prose | 4 | 76.5 [75.7–78.4] | 74.0 [73.6–76.2] | 81.29 | 1.67 | 304 |
| prose | 6 | 87.9 [79.6–88.4] | 85.3 [77.6–85.8] | 99.51 | 1.56 | 380 |
| code | 1 | 62.2 [60.5–62.4] | 59.3 [57.4–59.4] | 53.90 | 3.36 | 179 |
| code | 2 | 76.3 [76.0–77.1] | 73.1 [72.7–73.9] | 73.43 | 2.93 | 255 |
| code | 4 | 100.6 [95.4–101.5] | 95.8 [91.8–97.4] | 112.93 | 2.91 | 331 |
| code | 6 | 110.9 [108.0–113.9] | 106.8 [104.1–109.6] | 143.71 | 2.80 | 381 |
| json | 1 | 75.5 [73.5–76.2] | 71.4 [69.5–72.0] | 58.23 | 4.40 | 178 |
| json | 2 | 105.8 [104.1–106.3] | 99.8 [97.8–100.4] | 82.71 | 4.43 | 255 |
| json | 4 | 113.3 [108.3–114.6] | 108.3 [103.6–109.5] | 112.50 | 4.10 | 355 |
| json | 6 | 128.6 [124.5–130.2] | 123.4 [119.1–124.6] | 160.95 | 3.64 | 381 |
| math | 1 | 63.0 [59.5–63.7] | 59.5 [56.5–60.4] | 53.98 | 3.40 | 203 |
| math | 2 | 94.7 [93.4–95.7] | 89.1 [87.9–90.0] | 79.18 | 3.83 | 303 |
| math | 4 | 99.6 [94.1–99.9] | 95.6 [90.3–95.6] | 112.53 | 2.88 | 356 |
| math | 6 | 115.6 [111.3–119.3] | 110.8 [107.0–114.4] | 139.37 | 2.79 | 405 |
| chat | 1 | 45.8 [43.8–47.2] | 44.5 [42.4–45.7] | 49.10 | 2.30 | 176 |
| chat | 2 | 63.3 [63.3–63.4] | 61.3 [61.1–61.5] | 66.05 | 2.16 | 229 |
| chat | 4 | 82.2 [81.8–82.4] | 79.6 [79.2–79.8] | 84.41 | 1.84 | 307 |
| chat | 6 | 96.0 [88.0–96.1] | 92.6 [85.4–92.9] | 108.39 | 1.80 | 382 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 33.1 | 32.4 |
| prose | 6 | 55.3 | 54.2 |
| code | 1 | 48.0 | 46.3 |
| code | 6 | 85.9 | 82.9 |
| json | 1 | 73.2 | 69.3 |
| json | 6 | 117.3 | 112.4 |
| math | 1 | 47.7 | 45.7 |
| math | 6 | 91.5 | 87.7 |
| chat | 1 | 34.4 | 33.7 |
| chat | 6 | 61.8 | 60.2 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2090, 2119, 2100 | 1.602 | 1.590–1.628 | 0.766 | 1.626 |
| 8192 | 8410, 8449, 8562 | 5.866 | 5.770–5.868 | 0.686 | 5.893 |
| 32768 | 33862, 33911, 33837 | 27.087 | 26.886–27.405 | 0.799 | 27.149 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth5 | prose | 34.2 | 62.63 | 2.14 | yes |
| depth5 | code | 59.8 | 62.73 | 3.75 | yes |
| depth5 | json | 82.7 | 64.23 | 5.31 | yes |
| depth5 | math | 65.9 | 62.42 | 4.11 | yes |
| depth5 | chat | 40.6 | 62.85 | 2.55 | yes |
| plain | prose | 32.2 | 31.03 | 1.00 | yes |
| plain | code | 32.2 | 31.03 | 1.00 | yes |
| plain | json | 32.3 | 31.00 | 1.00 | yes |
| plain | math | 32.2 | 31.09 | 1.00 | yes |
| plain | chat | 32.2 | 31.04 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 5 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 6. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 160/164 | 0 | 185 |
| gsm8k | 296/300 | 0 | 180 |
| extract | 100/100 | 0 | 53 |

Raw results: [raw/deepseek-w4/](raw/deepseek-w4/).

## Qwen NVFP4, YaRN 512K · 2 nodes

Configuration: [qwen-yarn-w2](configs/qwen-yarn-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 66.6 [64.1–66.6] | 65.3 [61.2–65.4] | 25.69 | 1.71 | 77 |
| prose | 2 | 98.7 [94.8–98.9] | 96.1 [87.8–96.3] | 33.28 | 1.67 | 115 |
| code | 1 | 73.0 [72.7–73.1] | 71.5 [68.2–71.5] | 25.89 | 1.89 | 76 |
| code | 2 | 111.4 [109.9–111.9] | 108.0 [99.5–108.8] | 33.90 | 1.91 | 115 |
| json | 1 | 75.8 [75.8–76.2] | 74.1 [70.6–74.5] | 26.07 | 1.98 | 76 |
| json | 2 | 115.2 [115.0–115.3] | 111.1 [104.8–111.2] | 34.28 | 1.98 | 126 |
| math | 1 | 73.5 [71.2–73.7] | 72.0 [66.5–72.1] | 26.09 | 1.92 | 77 |
| math | 2 | 112.1 [112.1–112.7] | 108.4 [101.5–108.7] | 34.29 | 1.94 | 127 |
| chat | 1 | 64.5 [62.5–64.6] | 63.4 [59.1–63.5] | 26.00 | 1.68 | 76 |
| chat | 2 | 101.5 [98.6–101.9] | 98.5 [91.1–99.1] | 33.50 | 1.72 | 127 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 63.8 | 62.3 |
| prose | 2 | 96.4 | 90.6 |
| code | 1 | 70.0 | 66.1 |
| code | 2 | 103.3 | 100.5 |
| json | 1 | 72.4 | 71.0 |
| json | 2 | 113.6 | 109.8 |
| math | 1 | 70.4 | 67.5 |
| math | 2 | 105.3 | 102.1 |
| chat | 1 | 61.5 | 60.4 |
| chat | 2 | 96.1 | 93.0 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.278 | 1.246–1.347 | 0.622 | 1.287 |
| 8192 | 8091, 8132, 8300 | 5.117 | 5.106–5.373 | 0.631 | 5.138 |
| 32768 | 32664, 32698, 32653 | 20.975 | 20.964–20.985 | 0.642 | 21.036 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 70.3 | 31.26 | 2.20 | yes |
| depth2 | code | 84.4 | 31.48 | 2.66 | yes |
| depth2 | json | 91.5 | 31.69 | 2.90 | yes |
| depth2 | math | 87.5 | 31.67 | 2.77 | yes |
| depth2 | chat | 65.9 | 31.71 | 2.09 | yes |
| plain | prose | 47.4 | 21.09 | 1.00 | yes |
| plain | code | 47.4 | 21.12 | 1.00 | yes |
| plain | json | 47.4 | 21.10 | 1.00 | yes |
| plain | math | 47.2 | 21.20 | 1.00 | yes |
| plain | chat | 47.2 | 21.20 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 1 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 2. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 161/164 | 0 | 230 |
| gsm8k | 293/300 | 0 | 349 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-yarn-w2/](raw/qwen-yarn-w2/).

## Qwen NVFP4, BF16 dense · 1 node

Configuration: [qwen-nvfp4-bf16-w1](configs/qwen-nvfp4-bf16-w1.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 33.1 [32.7–33.2] | 32.7 [31.7–32.8] | 51.29 | 1.70 | 127 |
| prose | 2 | 51.4 [51.1–51.5] | 50.4 [48.1–50.5] | 61.98 | 1.63 | 178 |
| prose | 4 | 74.6 [74.3–75.0] | 72.2 [69.0–72.4] | 86.36 | 1.66 | 338 |
| code | 1 | 37.9 [37.8–37.9] | 37.3 [36.2–37.4] | 51.41 | 1.95 | 102 |
| code | 2 | 59.7 [59.5–60.0] | 58.2 [55.6–58.7] | 63.03 | 1.90 | 179 |
| code | 4 | 84.4 [84.3–84.9] | 81.3 [77.8–81.9] | 86.90 | 1.88 | 336 |
| json | 1 | 38.4 [38.3–38.4] | 37.8 [36.5–37.9] | 51.92 | 1.99 | 102 |
| json | 2 | 62.3 [62.1–62.4] | 60.8 [58.0–60.9] | 63.43 | 1.99 | 178 |
| json | 4 | 86.5 [86.2–86.9] | 83.6 [78.8–84.0] | 86.94 | 1.97 | 339 |
| math | 1 | 36.8 [36.6–36.8] | 36.3 [35.2–36.3] | 51.67 | 1.90 | 127 |
| math | 2 | 60.6 [59.9–60.7] | 59.1 [55.9–59.3] | 63.23 | 1.93 | 179 |
| math | 4 | 83.6 [83.5–84.0] | 80.8 [77.0–80.8] | 87.11 | 1.87 | 344 |
| chat | 1 | 31.9 [31.6–31.9] | 31.5 [30.5–31.6] | 51.53 | 1.65 | 104 |
| chat | 2 | 51.6 [51.5–51.8] | 50.4 [48.8–50.8] | 61.91 | 1.66 | 180 |
| chat | 4 | 75.7 [74.8–77.5] | 72.6 [69.6–75.0] | 85.23 | 1.71 | 335 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 32.2 | 31.8 |
| prose | 4 | 74.3 | 70.1 |
| code | 1 | 36.3 | 35.8 |
| code | 4 | 82.9 | 80.0 |
| json | 1 | 38.0 | 37.3 |
| json | 4 | 88.8 | 84.9 |
| math | 1 | 36.2 | 35.0 |
| math | 4 | 84.2 | 81.2 |
| chat | 1 | 31.4 | 30.7 |
| chat | 4 | 75.4 | 72.3 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2063, 2054, 2038 | 1.707 | 1.646–1.806 | 0.831 | 1.723 |
| 8192 | 8091, 8132, 8300 | 6.939 | 6.897–7.283 | 0.853 | 6.976 |
| 32768 | 32664, 32698, 32653 | 28.424 | 28.405–28.585 | 0.870 | 28.482 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 35.0 | 60.20 | 2.11 | yes |
| depth2 | code | 47.1 | 60.19 | 2.83 | yes |
| depth2 | json | 48.0 | 60.40 | 2.90 | yes |
| depth2 | math | 45.2 | 60.72 | 2.74 | yes |
| depth2 | chat | 35.4 | 60.51 | 2.14 | yes |
| plain | prose | 24.4 | 40.94 | 1.00 | yes |
| plain | code | 24.4 | 41.05 | 1.00 | yes |
| plain | json | 24.4 | 41.02 | 1.00 | yes |
| plain | math | 24.3 | 41.12 | 1.00 | yes |
| plain | chat | 24.4 | 41.01 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 158/164 | 0 | 229 |
| gsm8k | 291/300 | 0 | 352 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen-nvfp4-bf16-w1/](raw/qwen-nvfp4-bf16-w1/).

## GLM-5.3 int4/int8, FP8 KV · 4 nodes

Configuration: [glm53-fp8kv-w4](configs/glm53-fp8kv-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 30.2 [30.1–30.2] | 29.6 [27.6–29.7] | 63.04 | 1.90 | 128 |
| prose | 2 | 38.1 [38.1–38.1] | 37.3 [34.1–37.3] | 89.86 | 1.75 | 204 |
| prose | 4 | 47.2 [47.1–47.7] | 46.0 [41.4–46.5] | 144.42 | 1.80 | 385 |
| prose | 8 | 56.1 [56.0–56.4] | 52.5 [47.1–52.6] | 237.74 | 1.79 | 893 |
| code | 1 | 30.5 [30.5–30.5] | 29.9 [27.6–30.0] | 64.29 | 1.96 | 127 |
| code | 2 | 39.5 [39.4–39.5] | 38.6 [34.8–38.7] | 93.01 | 1.85 | 190 |
| code | 4 | 46.8 [46.8–46.9] | 45.6 [41.1–45.7] | 150.20 | 1.83 | 387 |
| code | 8 | 54.6 [54.6–54.9] | 51.4 [45.7–51.6] | 249.33 | 1.82 | 903 |
| json | 1 | 30.1 [30.0–30.1] | 29.5 [27.4–29.6] | 63.79 | 1.92 | 127 |
| json | 2 | 39.5 [39.4–39.5] | 38.6 [35.0–38.6] | 92.21 | 1.86 | 191 |
| json | 4 | 50.5 [50.4–50.8] | 49.1 [43.8–49.5] | 149.85 | 1.95 | 392 |
| json | 8 | 58.4 [58.3–58.5] | 54.8 [48.5–54.9] | 242.93 | 1.92 | 896 |
| math | 1 | 29.3 [29.3–29.3] | 28.9 [26.3–28.9] | 64.41 | 1.89 | 128 |
| math | 2 | 41.1 [41.0–41.1] | 40.1 [35.7–40.2] | 94.07 | 1.95 | 204 |
| math | 4 | 47.5 [47.3–47.6] | 46.1 [41.2–46.4] | 149.87 | 1.88 | 382 |
| math | 8 | 56.5 [56.0–56.8] | 52.6 [46.8–53.1] | 256.21 | 1.91 | 905 |
| chat | 1 | 27.8 [27.8–27.9] | 27.4 [25.5–27.5] | 63.60 | 1.77 | 102 |
| chat | 2 | 38.2 [38.1–38.3] | 37.3 [34.0–37.4] | 91.95 | 1.78 | 203 |
| chat | 4 | 47.1 [46.9–47.2] | 45.7 [41.4–46.0] | 150.18 | 1.82 | 382 |
| chat | 8 | 56.1 [55.7–56.3] | 52.4 [47.0–52.8] | 245.83 | 1.84 | 893 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 28.8 | 28.4 |
| prose | 8 | 55.0 | 51.6 |
| code | 1 | 30.0 | 29.5 |
| code | 8 | 54.5 | 51.3 |
| json | 1 | 30.2 | 29.7 |
| json | 8 | 58.3 | 54.7 |
| math | 1 | 29.7 | 29.2 |
| math | 8 | 56.0 | 52.6 |
| chat | 1 | 27.1 | 26.7 |
| chat | 8 | 55.1 | 51.7 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 4.747 | 4.733–4.914 | 2.290 | 4.759 |
| 8192 | 8257, 8302, 8429 | 24.735 | 24.427–25.097 | 2.977 | 24.746 |
| 32768 | 33245, 33321, 33241 | 186.045 | 185.371–186.236 | 5.589 | 186.071 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 20.1 | 49.79 | 1.00 | yes |
| plain | code | 19.8 | 50.46 | 1.00 | yes |
| plain | json | 20.1 | 49.78 | 1.00 | yes |
| plain | math | 20.1 | 49.79 | 1.00 | yes |
| plain | chat | 20.1 | 49.83 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 160/164 | 0 | 138 |
| gsm8k | 293/300 | 0 | 94 |
| extract | 100/100 | 0 | 57 |

Raw results: [raw/glm53-fp8kv-w4/](raw/glm53-fp8kv-w4/).

## GLM-Flash hybrid, 256K KV · 2 nodes

Configuration: [glm-flash-hybrid-256k-w2](configs/glm-flash-hybrid-256k-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 35.0 [34.2–35.0] | 34.4 [32.2–34.4] | 54.36 | 1.90 | 127 |
| prose | 2 | 44.7 [44.5–44.7] | 43.9 [41.3–44.0] | 77.68 | 1.77 | 178 |
| code | 1 | 35.6 [35.6–35.6] | 35.1 [33.3–35.2] | 55.55 | 1.98 | 103 |
| code | 2 | 45.7 [45.7–45.7] | 45.1 [42.0–45.1] | 79.70 | 1.85 | 167 |
| json | 1 | 35.8 [35.8–35.8] | 35.4 [33.5–35.5] | 54.82 | 1.96 | 77 |
| json | 2 | 44.4 [44.3–44.4] | 43.8 [40.9–43.8] | 78.21 | 1.83 | 166 |
| math | 1 | 33.8 [33.8–33.8] | 33.4 [31.5–33.4] | 55.49 | 1.88 | 103 |
| math | 2 | 47.4 [47.4–47.5] | 46.6 [43.2–46.6] | 80.84 | 1.92 | 205 |
| chat | 1 | 31.2 [31.2–31.2] | 30.8 [29.5–30.8] | 54.87 | 1.71 | 103 |
| chat | 2 | 44.3 [44.3–44.3] | 43.5 [40.8–43.5] | 78.80 | 1.78 | 217 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 33.9 | 33.3 |
| prose | 2 | 43.8 | 42.9 |
| code | 1 | 35.1 | 34.6 |
| code | 2 | 45.4 | 44.8 |
| json | 1 | 35.6 | 35.3 |
| json | 2 | 45.1 | 42.9 |
| math | 1 | 33.7 | 33.3 |
| math | 2 | 47.1 | 46.3 |
| chat | 1 | 32.3 | 30.7 |
| chat | 2 | 43.9 | 42.6 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2071, 2098, 2073 | 2.367 | 2.343–2.402 | 1.142 | 2.388 |
| 8192 | 8257, 8302, 8429 | 9.180 | 9.107–9.361 | 1.111 | 9.211 |
| 32768 | 33245, 33321, 33241 | 39.596 | 39.417–40.065 | 1.188 | 39.612 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| plain | prose | 24.1 | 41.44 | 1.00 | yes |
| plain | code | 24.1 | 41.47 | 1.00 | yes |
| plain | json | 24.1 | 41.46 | 1.00 | yes |
| plain | math | 24.1 | 41.50 | 1.00 | yes |
| plain | chat | 24.1 | 41.49 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 1 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 2. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 158/164 | 0 | 172 |
| gsm8k | 292/300 | 0 | 76 |
| extract | 100/100 | 0 | 51 |

Raw results: [raw/glm-flash-hybrid-256k-w2/](raw/glm-flash-hybrid-256k-w2/).

## Selection microbenchmarks

Medians in microseconds. Cold denotes L2 eviction outside the timed interval. All commands use five warmups and thirty iterations.

### Qwen QSA

| Rows | Pools | Cache | Stage | Median µs | Min µs | p95 µs |
|---|---|---|---|---|---|---|
| 16 | 16384 | warm | empty | 2.176 | 2.048 | 2.368 |
| 16 | 16384 | cold | empty | 3.808 | 3.68 | 3.84 |
| 16 | 16384 | warm | score | 105.04 | 104.768 | 106.944 |
| 16 | 16384 | cold | score | 132.8 | 130.24 | 186.304 |
| 16 | 16384 | warm | select | 57.76 | 56.864 | 57.92 |
| 16 | 16384 | cold | select | 79.584 | 79.424 | 128.896 |
| 16 | 16384 | warm | combined | 162.176 | 160.096 | 163.2 |
| 16 | 16384 | cold | combined | 188.128 | 187.008 | 205.44 |
| 16 | 65322 | warm | empty | 2.176 | 2.08 | 2.336 |
| 16 | 65322 | cold | empty | 3.776 | 3.648 | 3.84 |
| 16 | 65322 | warm | score | 387.488 | 386.56 | 387.616 |
| 16 | 65322 | cold | score | 481.92 | 479.264 | 524.992 |
| 16 | 65322 | warm | select | 172.368 | 170.848 | 186.208 |
| 16 | 65322 | cold | select | 253.632 | 251.616 | 291.168 |
| 16 | 65322 | warm | combined | 604.896 | 602.496 | 700.736 |
| 16 | 65322 | cold | combined | 695.968 | 693.984 | 782.944 |
| 16 | 131072 | warm | empty | 2.272 | 2.048 | 2.464 |
| 16 | 131072 | cold | empty | 3.808 | 3.584 | 3.84 |
| 16 | 131072 | warm | score | 1799.744 | 1784.448 | 1986.208 |
| 16 | 131072 | cold | score | 1868.544 | 1863.488 | 1875.616 |
| 16 | 131072 | warm | select | 326.016 | 325.088 | 326.176 |
| 16 | 131072 | cold | select | 538.336 | 534.496 | 723.456 |
| 16 | 131072 | warm | combined | 2279.664 | 2267.296 | 2500.32 |
| 16 | 131072 | cold | combined | 2342.256 | 2333.408 | 2346.72 |
| 2 | 16384 | warm | empty | 2.208 | 2.08 | 2.336 |
| 2 | 16384 | cold | empty | 3.808 | 3.712 | 4.0 |
| 2 | 16384 | warm | score | 23.04 | 21.728 | 23.872 |
| 2 | 16384 | cold | score | 52.992 | 50.912 | 72.864 |
| 2 | 16384 | warm | select | 58.512 | 57.088 | 59.232 |
| 2 | 16384 | cold | select | 77.456 | 75.392 | 80.864 |
| 2 | 16384 | warm | combined | 77.264 | 75.904 | 78.848 |
| 2 | 16384 | cold | combined | 108.288 | 106.464 | 149.696 |
| 2 | 65322 | warm | empty | 2.208 | 2.016 | 2.56 |
| 2 | 65322 | cold | empty | 3.744 | 3.712 | 3.84 |
| 2 | 65322 | warm | score | 61.824 | 61.44 | 62.048 |
| 2 | 65322 | cold | score | 139.92 | 137.856 | 141.024 |
| 2 | 65322 | warm | select | 170.368 | 168.576 | 171.456 |
| 2 | 65322 | cold | select | 237.28 | 235.232 | 414.4 |
| 2 | 65322 | warm | combined | 229.824 | 228.064 | 231.84 |
| 2 | 65322 | cold | combined | 308.864 | 305.792 | 309.952 |
| 2 | 131072 | warm | empty | 2.288 | 2.048 | 3.616 |
| 2 | 131072 | cold | empty | 3.808 | 3.616 | 3.808 |
| 2 | 131072 | warm | score | 223.472 | 221.568 | 224.064 |
| 2 | 131072 | cold | score | 286.368 | 284.608 | 288.384 |
| 2 | 131072 | warm | select | 323.872 | 322.4 | 330.752 |
| 2 | 131072 | cold | select | 466.656 | 464.576 | 494.912 |
| 2 | 131072 | warm | combined | 649.024 | 645.376 | 747.552 |
| 2 | 131072 | cold | combined | 708.896 | 705.216 | 714.912 |

### DeepSeek CSA2

`combined` times one candidate and one restricted call; the model has five restricted calls per full decoder pass. These component timings must not be added to predict service throughput.

| Rows | Context | Cache | Stage | Median µs | Min µs | Max µs |
|---|---|---|---|---|---|---|
| 1 | 131072 | warm | candidate | 436.512 | 433.92 | 438.752 |
| 1 | 131072 | cold | candidate | 447.2 | 445.152 | 525.664 |
| 1 | 131072 | warm | restricted | 103.488 | 101.12 | 133.728 |
| 1 | 131072 | cold | restricted | 108.448 | 107.52 | 152.736 |
| 1 | 131072 | warm | restricted_select_only | 584.192 | 583.392 | 610.656 |
| 1 | 131072 | cold | restricted_select_only | 618.176 | 618.016 | 735.744 |
| 1 | 131072 | warm | combined | 537.056 | 534.88 | 543.68 |
| 1 | 131072 | cold | combined | 546.528 | 544.416 | 616.16 |
| 30 | 131072 | warm | candidate | 8734.272 | 8602.912 | 8981.152 |
| 30 | 131072 | cold | candidate | 8886.976 | 8846.08 | 9009.76 |
| 30 | 131072 | warm | restricted | 1323.296 | 1313.44 | 1335.584 |
| 30 | 131072 | cold | restricted | 1323.744 | 1308.288 | 1381.024 |
| 30 | 131072 | warm | restricted_select_only | 602.528 | 601.696 | 603.488 |
| 30 | 131072 | cold | restricted_select_only | 640.64 | 638.688 | 752.416 |
| 30 | 131072 | warm | combined | 10230.848 | 9907.424 | 10452.032 |
| 30 | 131072 | cold | combined | 10145.472 | 10048.128 | 10234.592 |
| 5 | 131072 | warm | candidate | 1503.744 | 1491.104 | 1533.344 |
| 5 | 131072 | cold | candidate | 1517.28 | 1507.456 | 1606.912 |
| 5 | 131072 | warm | restricted | 262.08 | 260.096 | 263.712 |
| 5 | 131072 | cold | restricted | 269.344 | 268.0 | 271.104 |
| 5 | 131072 | warm | restricted_select_only | 601.696 | 598.976 | 625.632 |
| 5 | 131072 | cold | restricted_select_only | 632.544 | 630.752 | 686.816 |
| 5 | 131072 | warm | combined | 1767.968 | 1748.928 | 1799.936 |
| 5 | 131072 | cold | combined | 1795.808 | 1771.008 | 1970.368 |
| 1 | 16384 | warm | candidate | 4.416 | 3.52 | 17.952 |
| 1 | 16384 | cold | candidate | 5.856 | 4.16 | 5.92 |
| 1 | 16384 | warm | restricted | 100.928 | 100.096 | 102.848 |
| 1 | 16384 | cold | restricted | 107.232 | 106.208 | 166.144 |
| 1 | 16384 | warm | restricted_select_only | 583.232 | 582.656 | 584.192 |
| 1 | 16384 | cold | restricted_select_only | 620.256 | 618.144 | 620.288 |
| 1 | 16384 | warm | combined | 102.88 | 102.24 | 104.352 |
| 1 | 16384 | cold | combined | 109.28 | 108.256 | 110.4 |
| 30 | 16384 | warm | candidate | 4.544 | 4.256 | 6.176 |
| 30 | 16384 | cold | candidate | 5.856 | 5.696 | 6.144 |
| 30 | 16384 | warm | restricted | 1286.176 | 1270.112 | 1356.256 |
| 30 | 16384 | cold | restricted | 1300.192 | 1281.664 | 1328.32 |
| 30 | 16384 | warm | restricted_select_only | 588.48 | 586.208 | 590.016 |
| 30 | 16384 | cold | restricted_select_only | 622.24 | 620.48 | 696.352 |
| 30 | 16384 | warm | combined | 1278.336 | 1270.304 | 1357.184 |
| 30 | 16384 | cold | combined | 1300.192 | 1280.096 | 1343.776 |
| 5 | 16384 | warm | candidate | 4.608 | 3.584 | 5.312 |
| 5 | 16384 | cold | candidate | 5.856 | 4.224 | 5.888 |
| 5 | 16384 | warm | restricted | 260.544 | 258.72 | 262.336 |
| 5 | 16384 | cold | restricted | 263.808 | 261.632 | 329.632 |
| 5 | 16384 | warm | restricted_select_only | 584.384 | 583.392 | 617.056 |
| 5 | 16384 | cold | restricted_select_only | 620.16 | 618.208 | 675.872 |
| 5 | 16384 | warm | combined | 262.56 | 259.936 | 264.032 |
| 5 | 16384 | cold | combined | 264.032 | 263.712 | 344.48 |
| 1 | 4096 | warm | candidate | 3.776 | 3.584 | 5.248 |
| 1 | 4096 | cold | candidate | 5.824 | 5.632 | 5.984 |
| 1 | 4096 | warm | restricted | 37.28 | 36.032 | 38.336 |
| 1 | 4096 | cold | restricted | 40.672 | 40.576 | 42.56 |
| 1 | 4096 | warm | restricted_select_only | 153.632 | 152.384 | 155.136 |
| 1 | 4096 | cold | restricted_select_only | 163.552 | 163.392 | 230.208 |
| 1 | 4096 | warm | combined | 38.528 | 38.368 | 39.52 |
| 1 | 4096 | cold | combined | 42.688 | 42.592 | 44.672 |
| 30 | 4096 | warm | candidate | 4.448 | 3.584 | 4.64 |
| 30 | 4096 | cold | candidate | 5.856 | 5.76 | 9.568 |
| 30 | 4096 | warm | restricted | 338.528 | 336.288 | 340.96 |
| 30 | 4096 | cold | restricted | 343.776 | 339.584 | 396.48 |
| 30 | 4096 | warm | restricted_select_only | 154.112 | 153.984 | 155.232 |
| 30 | 4096 | cold | restricted_select_only | 163.808 | 163.456 | 219.328 |
| 30 | 4096 | warm | combined | 342.08 | 338.304 | 393.344 |
| 30 | 4096 | cold | combined | 345.76 | 341.728 | 398.08 |
| 5 | 4096 | warm | candidate | 4.416 | 3.552 | 6.016 |
| 5 | 4096 | cold | candidate | 5.856 | 5.76 | 6.048 |
| 5 | 4096 | warm | restricted | 79.392 | 78.176 | 80.48 |
| 5 | 4096 | cold | restricted | 81.568 | 81.44 | 83.68 |
| 5 | 4096 | warm | restricted_select_only | 155.04 | 153.152 | 193.056 |
| 5 | 4096 | cold | restricted_select_only | 163.488 | 163.36 | 237.376 |
| 5 | 4096 | warm | combined | 81.504 | 80.288 | 92.224 |
| 5 | 4096 | cold | combined | 83.68 | 83.168 | 104.352 |
