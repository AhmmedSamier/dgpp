> **INVALID MEASUREMENTS:** superseded after serving-process swap was detected. See [invalidation record](INVALID.md).

# Benchmark matrix refresh

All throughput values are medians of three repetitions. Serving ranges span the five prompt classes.
The [manifest](manifest.json) identifies the engine revision and binary. Commands, exit codes, configurations and responses are retained under `raw/`.
Outstanding measurement groups: 298.

| Deployment | C1/C2/C4/C8 | Decode modes | Context buckets | Cold prefill |
| --- | --- | --- | --- | --- |
| glm-flash-hybrid-w2 | Complete | 4/4 | 4/4 | Complete |
| glm-flash-hybrid-256k-w2 | Complete | 4/4 | 5/5 | Complete |
| glm-flash-hybrid-w4 | Complete | 1/4 | 4/6 | Complete |
| glm-flash-fp8-w4 | Pending | 0/4 | 0/5 | Pending |
| glm-flash-fp8-512k-w4 | Pending | 0/4 | 0/6 | Pending |
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


## glm-flash-hybrid-w2

Configuration: [configs/glm-flash-hybrid-w2.json](configs/glm-flash-hybrid-w2.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-w2/refresh/](raw/glm-flash-hybrid-w2/refresh/).

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 34.31 | 33.14 |
| code | 1 | 36.03 | 34.49 |
| json | 1 | 36.14 | 34.64 |
| math | 1 | 32.29 | 30.99 |
| chat | 1 | 30.70 | 29.66 |
| prose | 2 | 46.48 | 44.96 |
| code | 2 | 48.67 | 46.77 |
| json | 2 | 45.77 | 44.16 |
| math | 2 | 48.94 | 46.85 |
| chat | 2 | 43.87 | 41.37 |
| prose | 4 | 63.29 | 61.37 |
| code | 4 | 60.16 | 58.26 |
| json | 4 | 66.70 | 64.39 |
| math | 4 | 61.18 | 57.50 |
| chat | 4 | 53.84 | 52.37 |
| prose | 8 | 62.97 | 60.24 |
| code | 8 | 61.64 | 59.17 |
| json | 8 | 58.74 | 56.48 |
| math | 8 | 60.93 | 57.91 |
| chat | 8 | 57.66 | 55.10 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 96 | 0.404 | 54.90 | 1.81 | 32.94 |
| 32768 | 32236 | 75.029 | 57.21 | 1.95 | 34.03 |
| 65536 | 65008 | 162.977 | 62.14 | 1.98 | 31.81 |
| 131072 | 130537 | 450.772 | 59.61 | 1.96 | 32.90 |

Runtime health snapshots: [128k](raw/glm-flash-hybrid-w2/refresh/default/context-128k-health.json).

## glm-flash-hybrid-256k-w2

Configuration: [configs/glm-flash-hybrid-256k-w2.json](configs/glm-flash-hybrid-256k-w2.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-256k-w2/refresh/](raw/glm-flash-hybrid-256k-w2/refresh/).

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 34.34 | 33.16 |
| code | 1 | 36.15 | 34.57 |
| json | 1 | 36.25 | 34.79 |
| math | 1 | 34.91 | 33.30 |
| chat | 1 | 30.83 | 29.81 |
| prose | 2 | 46.50 | 45.00 |
| code | 2 | 48.81 | 46.94 |
| json | 2 | 45.91 | 44.32 |
| math | 2 | 49.11 | 47.00 |
| chat | 2 | 44.81 | 43.30 |
| prose | 4 | 47.77 | 46.26 |
| code | 4 | 47.32 | 45.45 |
| json | 4 | 50.88 | 48.67 |
| math | 4 | 47.76 | 45.74 |
| chat | 4 | 46.56 | 44.75 |
| prose | 8 | 46.69 | 44.89 |
| code | 8 | 46.66 | 44.77 |
| json | 8 | 49.48 | 47.35 |
| math | 8 | 48.69 | 46.44 |
| chat | 8 | 46.76 | 44.99 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 97 | 0.445 | 54.64 | 1.88 | 34.32 |
| 32768 | 32234 | 72.617 | 56.85 | 1.96 | 34.50 |
| 65536 | 65009 | 159.964 | 57.87 | 1.96 | 33.89 |
| 131072 | 130540 | 420.751 | 59.28 | 1.96 | 33.09 |
| 262144 | 261628 | 1638.111 | 62.12 | 1.98 | 31.82 |

Runtime health snapshots: [256k](raw/glm-flash-hybrid-256k-w2/refresh/default/context-256k-health.json).

## glm-flash-hybrid-w4

Configuration: [configs/glm-flash-hybrid-w4.json](configs/glm-flash-hybrid-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-hybrid-w4/refresh/](raw/glm-flash-hybrid-w4/refresh/).

| Class | C | Engine tok/s | Wall tok/s |
| --- | --- | --- | --- |
| prose | 1 | 60.03 | 57.67 |
| code | 1 | 61.70 | 58.97 |
| json | 1 | 56.85 | 53.33 |
| math | 1 | 57.07 | 54.51 |
| chat | 1 | 56.79 | 54.55 |
| prose | 2 | 81.73 | 79.06 |
| code | 2 | 84.54 | 81.43 |
| json | 2 | 73.24 | 69.59 |
| math | 2 | 88.91 | 85.04 |
| chat | 2 | 77.36 | 74.69 |
| prose | 4 | 115.71 | 111.97 |
| code | 4 | 104.24 | 101.03 |
| json | 4 | 110.29 | 105.30 |
| math | 4 | 106.70 | 100.94 |
| chat | 4 | 100.42 | 97.40 |
| prose | 8 | 103.53 | 99.33 |
| code | 8 | 100.25 | 96.20 |
| json | 8 | 110.28 | 103.42 |
| math | 8 | 98.13 | 93.23 |
| chat | 8 | 102.14 | 98.62 |

| Context bucket | Actual prompt tokens | Cold prefill s | ms/pass | Tokens/pass | Engine tok/s |
| --- | --- | --- | --- | --- | --- |
| 0 | 95 | 0.230 | 31.90 | 1.86 | 58.34 |
| 32768 | 32238 | 24.684 | 38.40 | 1.96 | 51.08 |
| 65536 | 65009 | 72.335 | 34.37 | 1.96 | 57.06 |
| 131072 | 130538 | 305.263 | 35.90 | 1.96 | 54.64 |

Runtime health snapshots: [128k](raw/glm-flash-hybrid-w4/refresh/default/context-128k-health.json), [256k](raw/glm-flash-hybrid-w4/refresh/default/context-256k-health.json).

## glm-flash-fp8-w4

Configuration: [configs/glm-flash-fp8-w4.json](configs/glm-flash-fp8-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-fp8-w4/refresh/](raw/glm-flash-fp8-w4/refresh/).

## glm-flash-fp8-512k-w4

Configuration: [configs/glm-flash-fp8-512k-w4.json](configs/glm-flash-fp8-512k-w4.json). Default mode: mtp1. Raw records: [raw/glm-flash-fp8-512k-w4/refresh/](raw/glm-flash-fp8-512k-w4/refresh/).

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
