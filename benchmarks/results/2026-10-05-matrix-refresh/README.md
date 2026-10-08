# Benchmark matrix refresh

All throughput values are medians of three repetitions. Serving ranges span the five prompt classes.
The [manifest](manifest.json) identifies the engine revision and binary. Commands, exit codes, configurations and responses are retained under `raw/`.
Outstanding measurement groups: 286.

| Deployment | C1/C2/C4/C8 | Decode modes | Context buckets | Cold prefill |
| --- | --- | --- | --- | --- |
| glm-flash-hybrid-w2 | Complete | 4/4 | 4/4 | Complete |
| glm-flash-hybrid-256k-w2 | Complete | 4/4 | 5/5 | Complete |
| glm-flash-hybrid-w4 | Complete | 4/4 | 6/6 | Complete |
| glm-flash-fp8-w4 | Complete | 4/4 | 5/5 | Complete |
| glm-flash-fp8-512k-w4 | Complete | 1/4 | 4/6 | Complete |
| qwen-nvfp4-w1 | Pending | 0/4 | 0/5 | Pending |
| qwen-nvfp4-bf16-w1 | Pending | 0/4 | 0/5 | Pending |
| qwen-nvfp4-w2 | Pending | 0/4 | 0/5 | Pending |
| qwen-yarn-w2 | Pending | 0/4 | 0/6 | Pending |
| qwen-radixark-w1 | Pending | 0/4 | 0/5 | Pending |
| qwen-radixark-w2 | Pending | 0/5 | 0/6 | Pending |
| qwen-fp8-w2 | Pending | 0/4 | 0/5 | Pending |
| qwen-fp8-w4 | Pending | 0/4 | 0/5 | Pending |
| qwen-autoround-w1 | Pending | 0/5 | 0/5 | Pending |
| qwen-autoround-prefill-w1 | Pending | 0/5 | 0/5 | Pending |
| glm53-w4 | Pending | 0/4 | 0/3 | Pending |
| glm53-fp8kv-w4 | Pending | 0/4 | 0/4 | Pending |
| glm53-fp8kv-256k-w4 | Pending | 0/4 | 0/5 | Pending |
| deepseek-w4 | Pending | 0/6 | 0/4 | Pending |
| deepseek-1m-w4 | Pending | 0/6 | 0/6 | Pending |
| dsv4-w2 | Pending | 0/6 | 0/6 | Pending |
| dsv4-w4 | Pending | 0/6 | 0/6 | Pending |
| mimo-w2 | Pending | 0/4 | 0/4 | Pending |
| mimo-w2-fp8kv | Pending | 0/4 | 0/5 | Pending |
| mimo-w4 | Pending | 0/4 | 0/4 | Pending |
| mimo-1m-w4 | Pending | 0/4 | 0/6 | Pending |
| qwen27b-fp8-w1 | Pending | 0/6 | 0/5 | Pending |
| qwen27b-fp8-w2 | Pending | 0/6 | 0/5 | Pending |
| qwen27b-fp8-w4 | Pending | 0/6 | 0/5 | Pending |


Only runs with passing per-rank process-swap checks contribute values. The [initial attempt](invalid-swapping/INVALID.md) is invalid; the [memory investigation](diagnostics/README.md) records the cause and fix.

## glm-flash-hybrid-w2

Configuration: [configs/glm-flash-hybrid-w2.json](configs/glm-flash-hybrid-w2.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-w2/refresh/](raw/glm-flash-hybrid-w2/refresh/).

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
| 32768 | 32234 | 72.029 | 57.02 | 1.95 | 34.14 |
| 65536 | 65011 | 159.067 | 57.79 | 1.93 | 33.43 |
| 131072 | 130541 | 426.231 | 59.37 | 1.95 | 32.79 |

## glm-flash-hybrid-256k-w2

Configuration: [configs/glm-flash-hybrid-256k-w2.json](configs/glm-flash-hybrid-256k-w2.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-256k-w2/refresh/](raw/glm-flash-hybrid-256k-w2/refresh/).

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
| 32768 | 32237 | 72.100 | 57.06 | 1.96 | 34.38 |
| 65536 | 65010 | 158.136 | 57.91 | 1.96 | 33.87 |
| 131072 | 130537 | 416.908 | 59.38 | 1.96 | 33.03 |
| 262144 | 261630 | 1586.967 | 62.30 | 1.96 | 31.48 |

## glm-flash-hybrid-w4

Configuration: [configs/glm-flash-hybrid-w4.json](configs/glm-flash-hybrid-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-w4/refresh/](raw/glm-flash-hybrid-w4/refresh/).

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 60.13 | 57.95 |
| code | 1 | 61.45 | 58.73 |
| json | 1 | 61.99 | 59.42 |
| math | 1 | 59.98 | 57.25 |
| chat | 1 | 56.69 | 54.52 |
| prose | 2 | 82.04 | 79.36 |
| code | 2 | 84.88 | 80.93 |
| json | 2 | 83.16 | 80.16 |
| math | 2 | 89.39 | 85.64 |
| chat | 2 | 83.01 | 80.17 |
| prose | 4 | 116.04 | 112.39 |
| code | 4 | 111.46 | 107.88 |
| json | 4 | 122.10 | 117.83 |
| math | 4 | 114.95 | 111.09 |
| chat | 4 | 109.16 | 105.59 |
| prose | 8 | 113.22 | 108.22 |
| code | 8 | 110.12 | 105.23 |
| json | 8 | 120.92 | 115.33 |
| math | 8 | 114.06 | 108.93 |
| chat | 8 | 112.44 | 107.50 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 99 | 0.236 | 31.90 | 1.88 | 58.78 |
| 32768 | 32235 | 24.312 | 33.62 | 1.96 | 58.35 |
| 65536 | 65012 | 66.764 | 34.29 | 1.98 | 57.65 |
| 131072 | 130539 | 262.963 | 35.93 | 1.95 | 54.18 |
| 262144 | 261629 | 1527.734 | 39.46 | 1.96 | 49.70 |
| 524288 | 523781 | 11384.873 | 45.25 | 1.98 | 43.69 |

## glm-flash-fp8-w4

Configuration: [configs/glm-flash-fp8-w4.json](configs/glm-flash-fp8-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-fp8-w4/refresh/](raw/glm-flash-fp8-w4/refresh/).

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
| 32768 | 32238 | 26.057 | 40.28 | 1.98 | 49.08 |
| 65536 | 65010 | 67.023 | 41.34 | 1.98 | 47.82 |
| 131072 | 130570 | 238.771 | 42.77 | 1.96 | 45.86 |
| 262144 | 261627 | 1281.921 | 46.49 | 1.95 | 41.87 |

## glm-flash-fp8-512k-w4

Configuration: [configs/glm-flash-fp8-512k-w4.json](configs/glm-flash-fp8-512k-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-fp8-512k-w4/refresh/](raw/glm-flash-fp8-512k-w4/refresh/).

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
| 32768 | 32237 | 26.157 | 40.22 | 1.98 | 49.15 |
| 65536 | 65009 | 66.862 | 41.44 | 1.96 | 47.33 |
| 131072 | 130539 | 233.894 | 42.83 | 1.95 | 45.45 |

## qwen-nvfp4-w1

Configuration: [configs/qwen-nvfp4-w1.json](configs/qwen-nvfp4-w1.json). Default mode: mtp1. Raw records: [raw/qwen-nvfp4-w1/refresh/](raw/qwen-nvfp4-w1/refresh/).

## qwen-nvfp4-bf16-w1

Configuration: [configs/qwen-nvfp4-bf16-w1.json](configs/qwen-nvfp4-bf16-w1.json). Default mode: mtp1. Raw records: [raw/qwen-nvfp4-bf16-w1/refresh/](raw/qwen-nvfp4-bf16-w1/refresh/).

## qwen-nvfp4-w2

Configuration: [configs/qwen-nvfp4-w2.json](configs/qwen-nvfp4-w2.json). Default mode: mtp1. Raw records: [raw/qwen-nvfp4-w2/refresh/](raw/qwen-nvfp4-w2/refresh/).

## qwen-yarn-w2

Configuration: [configs/qwen-yarn-w2.json](configs/qwen-yarn-w2.json). Default mode: mtp1. Raw records: [raw/qwen-yarn-w2/refresh/](raw/qwen-yarn-w2/refresh/).

## qwen-radixark-w1

Configuration: [configs/qwen-radixark-w1.json](configs/qwen-radixark-w1.json). Default mode: mtp2. Raw records: [raw/qwen-radixark-w1/refresh/](raw/qwen-radixark-w1/refresh/).

## qwen-radixark-w2

Configuration: [configs/qwen-radixark-w2.json](configs/qwen-radixark-w2.json). Default mode: mtp4. Raw records: [raw/qwen-radixark-w2/refresh/](raw/qwen-radixark-w2/refresh/).

## qwen-fp8-w2

Configuration: [configs/qwen-fp8-w2.json](configs/qwen-fp8-w2.json). Default mode: mtp1. Raw records: [raw/qwen-fp8-w2/refresh/](raw/qwen-fp8-w2/refresh/).

## qwen-fp8-w4

Configuration: [configs/qwen-fp8-w4.json](configs/qwen-fp8-w4.json). Default mode: mtp1. Raw records: [raw/qwen-fp8-w4/refresh/](raw/qwen-fp8-w4/refresh/).

## qwen-autoround-w1

Configuration: [configs/qwen-autoround-w1.json](configs/qwen-autoround-w1.json). Default mode: mtp3. Raw records: [raw/qwen-autoround-w1/refresh/](raw/qwen-autoround-w1/refresh/).

## qwen-autoround-prefill-w1

Configuration: [configs/qwen-autoround-prefill-w1.json](configs/qwen-autoround-prefill-w1.json). Default mode: mtp3. Raw records: [raw/qwen-autoround-prefill-w1/refresh/](raw/qwen-autoround-prefill-w1/refresh/).

## glm53-w4

Configuration: [configs/glm53-w4.json](configs/glm53-w4.json). Default mode: mtp1. Raw records: [raw/glm53-w4/refresh/](raw/glm53-w4/refresh/).

## glm53-fp8kv-w4

Configuration: [configs/glm53-fp8kv-w4.json](configs/glm53-fp8kv-w4.json). Default mode: mtp1. Raw records: [raw/glm53-fp8kv-w4/refresh/](raw/glm53-fp8kv-w4/refresh/).

## glm53-fp8kv-256k-w4

Configuration: [configs/glm53-fp8kv-256k-w4.json](configs/glm53-fp8kv-256k-w4.json). Default mode: mtp1. Raw records: [raw/glm53-fp8kv-256k-w4/refresh/](raw/glm53-fp8kv-256k-w4/refresh/).

## deepseek-w4

Configuration: [configs/deepseek-w4.json](configs/deepseek-w4.json). Default mode: dspark-adaptive. Raw records: [raw/deepseek-w4/refresh/](raw/deepseek-w4/refresh/).

## deepseek-1m-w4

Configuration: [configs/deepseek-1m-w4.json](configs/deepseek-1m-w4.json). Default mode: dspark-adaptive. Raw records: [raw/deepseek-1m-w4/refresh/](raw/deepseek-1m-w4/refresh/).

## dsv4-w2

Configuration: [configs/dsv4-w2.json](configs/dsv4-w2.json). Default mode: dspark-adaptive. Raw records: [raw/dsv4-w2/refresh/](raw/dsv4-w2/refresh/).

## dsv4-w4

Configuration: [configs/dsv4-w4.json](configs/dsv4-w4.json). Default mode: dspark-adaptive. Raw records: [raw/dsv4-w4/refresh/](raw/dsv4-w4/refresh/).

## mimo-w2

Configuration: [configs/mimo-w2.json](configs/mimo-w2.json). Default mode: mtp1. Raw records: [raw/mimo-w2/refresh/](raw/mimo-w2/refresh/).

## mimo-w2-fp8kv

Configuration: [configs/mimo-w2-fp8kv.json](configs/mimo-w2-fp8kv.json). Default mode: mtp1. Raw records: [raw/mimo-w2-fp8kv/refresh/](raw/mimo-w2-fp8kv/refresh/).

## mimo-w4

Configuration: [configs/mimo-w4.json](configs/mimo-w4.json). Default mode: mtp1. Raw records: [raw/mimo-w4/refresh/](raw/mimo-w4/refresh/).

## mimo-1m-w4

Configuration: [configs/mimo-1m-w4.json](configs/mimo-1m-w4.json). Default mode: mtp1. Raw records: [raw/mimo-1m-w4/refresh/](raw/mimo-1m-w4/refresh/).

## qwen27b-fp8-w1

Configuration: [configs/qwen27b-fp8-w1.json](configs/qwen27b-fp8-w1.json). Default mode: dflash2. Raw records: [raw/qwen27b-fp8-w1/refresh/](raw/qwen27b-fp8-w1/refresh/).

## qwen27b-fp8-w2

Configuration: [configs/qwen27b-fp8-w2.json](configs/qwen27b-fp8-w2.json). Default mode: dflash2. Raw records: [raw/qwen27b-fp8-w2/refresh/](raw/qwen27b-fp8-w2/refresh/).

## qwen27b-fp8-w4

Configuration: [configs/qwen27b-fp8-w4.json](configs/qwen27b-fp8-w4.json). Default mode: dflash2. Raw records: [raw/qwen27b-fp8-w4/refresh/](raw/qwen27b-fp8-w4/refresh/).
