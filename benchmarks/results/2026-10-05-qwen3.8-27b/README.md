# Current benchmark results

Source: `5b57904274d3c1f6f5863652903768ecb9a0c01c`.

All values below are calculated by `summarize.py` from this directory's raw results. The saved `*.command.json` files identify commands, start/end times and exit codes. `manifest.json` identifies the binary, hardware records, checkpoints and datasets.

**Run in progress.** These tables contain completed measurement groups only; remaining performance, mode, long-context and quality runs are still pending. GLM-4.7 is excluded at the user's request.

[Reference harness, HumanEval fences and the reference cross-check](notes.md)

The tables below are the campaign morning's record. The closing per-world decode numbers after the day's exactness work (binary `dc52f66`: C1, C8, plain T=1, the probes and the transcript oracles) are in [the notes' closing section](notes.md#the-closing-numbers-final26-binary-dc52f66-21201144) and are the cells [docs/benchmarks.md](../../../docs/benchmarks.md) carries.

Deployments follow the [overview](../../../docs/benchmarks.md) order: model family, node count, then configuration options. KV labels describe the shared key/value-cache token pool (K = 1,024 tokens); slots are the configured concurrent-request limit.

## Qwen3.8-27B FP8 · 1 node · DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w1](configs/qwen27b-fp8-w1.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 20.1 [20.1–20.1] | 19.8 [19.8–19.8] | 145.81 | 2.93 | 205 |
| prose | 2 | 30.2 [30.2–30.2] | 29.7 [29.7–29.7] | 156.32 | 2.48 | 328 |
| prose | 4 | 54.5 [54.4–54.5] | 52.3 [52.3–52.3] | 173.29 | 2.55 | 762 |
| prose | 8 | 94.3 [94.2–95.4] | 89.5 [89.3–90.6] | 210.03 | 2.68 | 1041 |
| code | 1 | 32.3 [32.2–32.3] | 31.5 [31.4–31.5] | 146.22 | 4.72 | 203 |
| code | 2 | 53.8 [53.8–53.8] | 51.9 [51.9–51.9] | 155.29 | 4.64 | 355 |
| code | 4 | 100.1 [99.8–100.8] | 92.5 [92.2–93.0] | 175.75 | 4.79 | 784 |
| code | 8 | 160.3 [160.3–160.4] | 146.6 [146.4–146.8] | 201.96 | 4.88 | 1094 |
| json | 1 | 45.9 [45.9–45.9] | 44.2 [44.2–44.3] | 146.27 | 6.71 | 205 |
| json | 2 | 80.0 [79.8–80.1] | 75.6 [75.6–75.7] | 155.40 | 6.89 | 357 |
| json | 4 | 112.5 [112.3–112.6] | 102.8 [102.7–103.0] | 167.95 | 6.14 | 789 |
| json | 8 | 198.6 [198.5–198.7] | 176.7 [176.7–176.9] | 205.44 | 5.95 | 1103 |
| math | 1 | 35.6 [35.6–35.6] | 34.3 [34.3–34.4] | 146.20 | 5.20 | 255 |
| math | 2 | 72.3 [72.3–72.6] | 68.8 [68.7–68.9] | 156.72 | 5.86 | 330 |
| math | 4 | 97.3 [97.1–97.8] | 90.0 [89.9–90.5] | 169.14 | 5.15 | 786 |
| math | 8 | 156.2 [156.2–156.4] | 142.4 [142.4–142.5] | 200.88 | 5.06 | 1146 |
| chat | 1 | 19.0 [19.0–19.0] | 18.7 [18.7–18.8] | 145.97 | 2.77 | 205 |
| chat | 2 | 33.5 [33.5–33.5] | 32.8 [32.8–32.8] | 157.01 | 2.67 | 331 |
| chat | 4 | 63.5 [63.3–63.5] | 60.5 [60.4–60.6] | 174.56 | 2.96 | 765 |
| chat | 8 | 101.8 [100.5–102.2] | 95.9 [95.0–96.5] | 208.73 | 2.87 | 1042 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 18.1 | 17.9 |
| prose | 8 | 84.8 | 80.9 |
| code | 1 | 28.5 | 27.9 |
| code | 8 | 149.8 | 136.5 |
| json | 1 | 43.8 | 42.2 |
| json | 8 | 175.0 | 157.7 |
| math | 1 | 34.8 | 33.6 |
| math | 8 | 163.7 | 147.8 |
| chat | 1 | 17.9 | 17.7 |
| chat | 8 | 91.3 | 86.5 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 2.198 | 2.197–2.207 | 1.075 | 2.231 |
| 8192 | 8029, 8078, 8245 | 7.061 | 6.991–7.301 | 0.874 | 7.090 |
| 32768 | 32421, 32460, 32404 | 30.797 | 30.785–30.944 | 0.950 | 30.843 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| mtp2 | prose | 18.2 | 129.74 | 2.36 | yes |
| mtp2 | code | 20.7 | 129.66 | 2.68 | yes |
| mtp2 | json | 22.3 | 129.70 | 2.90 | yes |
| mtp2 | math | 21.3 | 129.83 | 2.77 | yes |
| mtp2 | chat | 16.5 | 129.60 | 2.14 | no |
| plain | prose | 8.9 | 112.69 | 1.00 | yes |
| plain | code | 8.9 | 112.71 | 1.00 | yes |
| plain | json | 8.9 | 112.67 | 1.00 | yes |
| plain | math | 8.9 | 112.72 | 1.00 | yes |
| plain | chat | 8.9 | 112.71 | 1.00 | no |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 156/164 | 0 | 255 |
| gsm8k | 292/300 | 4 | 402 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w1/](raw/qwen27b-fp8-w1/).

## Qwen3.8-27B FP8 · 2 nodes · DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w2](configs/qwen27b-fp8-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 36.2 [36.0–36.2] | 35.4 [35.3–35.4] | 82.00 | 2.97 | 180 |
| prose | 2 | 51.2 [51.1–51.3] | 50.0 [49.9–50.1] | 93.12 | 2.48 | 255 |
| prose | 4 | 88.5 [88.5–88.7] | 85.1 [85.0–85.1] | 110.78 | 2.54 | 459 |
| prose | 8 | 133.3 [133.0–133.4] | 127.0 [126.9–127.3] | 141.74 | 2.60 | 666 |
| code | 1 | 57.5 [57.1–57.8] | 55.4 [54.8–55.6] | 82.06 | 4.72 | 179 |
| code | 2 | 92.1 [92.0–92.3] | 87.7 [87.6–87.7] | 92.29 | 4.68 | 280 |
| code | 4 | 155.9 [155.8–156.0] | 144.7 [144.3–144.7] | 109.02 | 4.81 | 481 |
| code | 8 | 238.2 [237.5–238.4] | 218.2 [217.4–218.5] | 131.76 | 4.92 | 687 |
| json | 1 | 81.4 [81.1–81.5] | 76.6 [76.6–76.7] | 82.45 | 6.71 | 180 |
| json | 2 | 132.4 [132.0–132.5] | 122.9 [122.7–123.4] | 91.73 | 6.80 | 281 |
| json | 4 | 183.9 [183.6–184.2] | 168.1 [167.9–168.7] | 104.67 | 6.14 | 486 |
| json | 8 | 294.1 [292.9–294.3] | 263.7 [263.2–263.9] | 138.73 | 5.91 | 690 |
| math | 1 | 65.8 [65.7–66.1] | 62.6 [62.4–62.7] | 82.51 | 5.43 | 203 |
| math | 2 | 121.1 [120.9–121.1] | 112.8 [112.7–113.2] | 93.62 | 5.86 | 280 |
| math | 4 | 171.4 [170.9–171.5] | 157.4 [157.4–157.4] | 108.20 | 5.37 | 484 |
| math | 8 | 237.9 [237.3–245.0] | 217.2 [216.7–223.3] | 136.10 | 5.04 | 720 |
| chat | 1 | 33.5 [33.4–33.7] | 32.8 [32.8–33.0] | 82.75 | 2.77 | 178 |
| chat | 2 | 55.4 [55.4–55.4] | 53.9 [53.9–54.0] | 93.92 | 2.66 | 256 |
| chat | 4 | 100.3 [100.2–100.9] | 95.8 [95.7–96.4] | 110.57 | 2.97 | 464 |
| chat | 8 | 152.8 [152.4–152.9] | 144.6 [144.3–144.8] | 142.01 | 2.89 | 664 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 29.0 | 28.5 |
| prose | 8 | 128.3 | 122.0 |
| code | 1 | 51.2 | 49.3 |
| code | 8 | 224.1 | 205.2 |
| json | 1 | 78.9 | 74.5 |
| json | 8 | 270.0 | 242.6 |
| math | 1 | 67.0 | 63.4 |
| math | 8 | 222.6 | 203.9 |
| chat | 1 | 30.8 | 30.2 |
| chat | 8 | 144.2 | 135.5 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.467 | 1.466–1.470 | 0.720 | 1.488 |
| 8192 | 8029, 8078, 8245 | 4.863 | 4.833–5.073 | 0.602 | 4.894 |
| 32768 | 32421, 32460, 32404 | 20.705 | 20.684–20.743 | 0.639 | 20.756 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| mtp3 | prose | 34.8 | 78.01 | 2.71 | yes |
| mtp3 | code | 41.3 | 78.09 | 3.23 | yes |
| mtp3 | json | 47.4 | 78.05 | 3.70 | yes |
| mtp3 | math | 43.6 | 77.93 | 3.40 | yes |
| mtp3 | chat | 31.8 | 77.97 | 2.48 | no |
| plain | prose | 16.5 | 60.57 | 1.00 | yes |
| plain | code | 16.5 | 60.67 | 1.00 | yes |
| plain | json | 16.5 | 60.70 | 1.00 | yes |
| plain | math | 16.5 | 60.76 | 1.00 | yes |
| plain | chat | 16.5 | 60.77 | 1.00 | no |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 157/164 | 0 | 250 |
| gsm8k | 292/300 | 5 | 401 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w2/](raw/qwen27b-fp8-w2/).

## Qwen3.8-27B FP8 · 4 nodes · DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w4](configs/qwen27b-fp8-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 59.3 [59.2–59.4] | 57.3 [57.2–57.5] | 50.03 | 2.97 | 153 |
| prose | 2 | 80.6 [80.4–80.7] | 77.9 [77.7–78.0] | 59.16 | 2.46 | 230 |
| prose | 4 | 127.4 [127.2–128.3] | 122.8 [122.3–123.7] | 81.67 | 2.70 | 308 |
| prose | 8 | 159.6 [159.3–160.0] | 153.6 [153.3–153.6] | 121.74 | 2.60 | 434 |
| code | 1 | 94.0 [93.8–94.3] | 88.5 [88.2–88.5] | 50.26 | 4.72 | 178 |
| code | 2 | 143.1 [142.9–143.1] | 133.5 [133.3–133.5] | 58.41 | 4.59 | 254 |
| code | 4 | 212.4 [209.8–212.4] | 198.2 [195.8–198.6] | 80.05 | 4.68 | 309 |
| code | 8 | 290.0 [289.3–290.3] | 269.8 [269.8–270.8] | 111.67 | 4.88 | 483 |
| json | 1 | 133.5 [133.0–133.8] | 121.8 [121.7–122.9] | 50.25 | 6.71 | 180 |
| json | 2 | 208.8 [208.1–209.3] | 189.0 [187.8–189.7] | 58.17 | 6.80 | 257 |
| json | 4 | 259.0 [258.3–261.3] | 238.6 [237.4–240.9] | 74.30 | 6.14 | 306 |
| json | 8 | 350.8 [349.2–351.2] | 320.1 [318.5–320.2] | 118.68 | 5.95 | 486 |
| math | 1 | 110.3 [109.8–110.3] | 101.7 [101.1–101.7] | 50.28 | 5.54 | 204 |
| math | 2 | 198.3 [198.0–198.8] | 178.3 [177.9–179.2] | 59.83 | 6.00 | 280 |
| math | 4 | 219.6 [217.5–219.7] | 204.5 [202.4–204.5] | 74.92 | 5.15 | 332 |
| math | 8 | 279.8 [279.8–284.0] | 259.1 [258.9–262.6] | 112.17 | 5.07 | 508 |
| chat | 1 | 55.2 [55.1–55.2] | 53.4 [53.4–53.5] | 50.24 | 2.77 | 153 |
| chat | 2 | 84.5 [84.4–85.1] | 81.4 [81.3–82.1] | 59.26 | 2.63 | 230 |
| chat | 4 | 138.2 [138.1–138.2] | 132.5 [132.2–132.5] | 80.24 | 2.96 | 307 |
| chat | 8 | 177.2 [176.7–177.5] | 169.7 [169.3–170.1] | 118.72 | 2.91 | 460 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 51.1 | 49.6 |
| prose | 8 | 157.0 | 150.8 |
| code | 1 | 89.8 | 84.5 |
| code | 8 | 279.2 | 258.7 |
| json | 1 | 128.9 | 118.9 |
| json | 8 | 341.2 | 310.6 |
| math | 1 | 111.7 | 102.7 |
| math | 8 | 287.6 | 265.9 |
| chat | 1 | 51.5 | 49.9 |
| chat | 8 | 171.0 | 163.4 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.077 | 1.065–1.082 | 0.524 | 1.096 |
| 8192 | 8029, 8078, 8245 | 3.753 | 3.714–3.890 | 0.465 | 3.778 |
| 32768 | 32421, 32460, 32404 | 15.409 | 15.406–15.511 | 0.476 | 15.469 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| mtp3 | prose | 59.4 | 45.15 | 2.68 | yes |
| mtp3 | code | 71.5 | 45.16 | 3.23 | yes |
| mtp3 | json | 81.7 | 45.25 | 3.70 | yes |
| mtp3 | math | 75.2 | 45.19 | 3.40 | yes |
| mtp3 | chat | 51.8 | 45.17 | 2.34 | no |
| plain | prose | 31.3 | 31.98 | 1.00 | yes |
| plain | code | 31.2 | 32.01 | 1.00 | yes |
| plain | json | 31.2 | 32.04 | 1.00 | yes |
| plain | math | 31.2 | 32.06 | 1.00 | yes |
| plain | chat | 31.2 | 32.04 | 1.00 | no |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 156/164 | 0 | 247 |
| gsm8k | 291/300 | 4 | 397 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w4/](raw/qwen27b-fp8-w4/).
