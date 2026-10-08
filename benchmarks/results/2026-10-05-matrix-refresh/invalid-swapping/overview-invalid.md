> **INVALID MEASUREMENTS:** superseded after serving-process swap was detected. See [invalidation record](INVALID.md).

# Benchmarks

Serving throughput, decode modes and context scaling on the GB10 cluster. Each row identifies a model, weight format, node count and deployment configuration.

| Measurement set | Date (UTC) | Source revision | Scope and record |
|---|---|---|---|
| Matrix refresh | Started 2026-10-06 | `50309fd` | C1/C2/C4/C8, fixed decode modes, cold prefill and context scaling; [results and coverage](../benchmarks/results/2026-10-05-matrix-refresh/README.md) |
| Original campaign | 2026-09-22 | `a3ed8994a34c` | GLM, Qwen Flash-Next and DeepSeek-V4.1 quality/correctness; [record](../benchmarks/results/2026-09-22-current/README.md) |
| MiMo | 2026-09-22/23 | `a895db96bb17` + recorded changes | Quality/correctness; [record](../benchmarks/results/2026-09-22-mimo-v26-flash-opt/README.md) |
| Qwen AutoRound | 2026-09-29 | Campaign revisions recorded per run | Quality/correctness; [record](../benchmarks/results/2026-09-28-qwen-autoround-int4/README.md) |
| DeepSeek-V4 | 2026-10-02 | `904f76c8519a` + recorded changes | Historical quality/correctness and llama-benchy; [record](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md) |
| Qwen 27B | 2026-10-05 | Morning: `5b57904274d3` + recorded changes; evening: `dc52f66` | Quality/correctness and earlier performance; [record](../benchmarks/results/2026-10-05-qwen3.8-27b/README.md), [evening measurements](../benchmarks/results/2026-10-05-qwen3.8-27b/notes.md) |

[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · [Method](#workload-and-timing) · [Decode modes](#decode-modes) · [Quality](#quality-and-correctness) · [Long context](#long-context) · [Reproduce](#reproduce)

## Serving performance

Rates are tokens per second. Ranges span the medians of five prompt classes, with three repetitions per class. **C** is the number of concurrent client requests. Every deployment is tested at **C1, C2, C4 and C8**.

| Metric | Timing scope |
|---|---|
| C1 engine tok/s | Decode steps only; excludes prefill and its first output token |
| C1/C2/C4/C8 wall tok/s | Total completed output divided by request-group wall time, including queueing and prefill |
| Cold prefill seconds | Median of three uncached prompts at each approximate target length |

Client concurrency and request slots are different. A C8 test on a two-slot deployment includes the time requests spend waiting for a slot. The deployment's slot count, KV pool and decode mode stay fixed throughout its concurrency sweep.

<!-- BEGIN serving -->

| Model / weights | Nodes | Options | C1 engine tok/s | C1 wall tok/s | C2 wall tok/s | C4 wall tok/s | C8 wall tok/s | Cold prefill s: ~2K / ~8K / ~32K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 30.7–36.1 | 29.7–34.6 | 41.4–46.9 | 52.4–64.4 | 55.1–60.2 | 5.770 / 19.076 / 81.000 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 30.8–36.3 | 29.8–34.8 | 43.3–47.0 | 44.8–48.7 | 44.8–47.4 | 4.709 / 19.230 / 78.653 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 56.8–61.7 | 53.3–59.0 | 69.6–85.0 | 97.4–112.0 | 93.2–103.4 | 2.178 / 5.234 / 27.033 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next FP8 | 2 | Default | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next FP8 | 4 | Default | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Pending / Pending / Pending |

<!-- END serving -->

Configuration labels use **K = 1,024 tokens**. Nodes is the tensor-parallel world size; KV is the shared key/value-cache token pool. Pending means the refresh has not produced a complete measurement group; see the campaign's coverage record.

## Hardware and configuration

| Component | Configuration |
|---|---|
| Rank 0 | MSI EdgeXpert, NVIDIA GB10 |
| Rank 1 | ASUS GX10, NVIDIA GB10 |
| Rank 2 | NVIDIA DGX Spark, NVIDIA GB10 |
| Rank 3 | AI TOP ATOM, NVIDIA GB10 |
| Memory | About 121 GiB of OS-visible unified memory per node |
| Network | 200 Gb/s RoCE fabric |
| Node counts | 1, 2 and 4 |
| Software | CUDA 13.0; NVIDIA driver 580.173.02; CMake `release` preset |
| Timing conditions | One benchmark deployment at a time; models and decode graphs loaded; no profiler |
| Four-node telemetry | Host memory/pressure every second, GPU state every five seconds, rank 2 kernel journal; probes remain active during timing |
| CUDA connections | `CUDA_DEVICE_MAX_CONNECTIONS=32` on rank 0; peers retain their login environment |
| Environment records | [Original comparison](../benchmarks/results/2026-09-22-current/environment.md); [refresh manifest](../benchmarks/results/2026-10-05-matrix-refresh/manifest.json) |

The KV pool is shared across requests. Its capacity does not mean every concurrent request can use that many tokens. The linked JSON files define the effective settings for each measured row.

<!-- BEGIN config -->

| Model / weights | Nodes | Options | Slots | KV pool tokens | KV format | Prefix cache GiB | Default decode | Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 4 | 163,840 | fp8 | 4.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm-flash-hybrid-w2.json) |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 2 | 262,144 | fp8 | 2 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm-flash-hybrid-256k-w2.json) |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 4 | 786,432 | bf16 | 22 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm-flash-hybrid-w4.json) |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4 | 393,216 | bf16 | 8 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm-flash-fp8-w4.json) |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 4 | 524,288 | bf16 | 2 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm-flash-fp8-512k-w4.json) |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4 | 262,144 | bf16 | 3 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-nvfp4-w1.json) |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4 | 262,144 | bf16 | 3 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-nvfp4-bf16-w1.json) |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4 | 262,144 | bf16 | 58 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-nvfp4-w2.json) |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 2 | 565,248 | bf16 | 40 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-yarn-w2.json) |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 4 | 262,144 | bf16 | 3 | MTP2 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-radixark-w1.json) |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 4 | 524,288 | bf16 | 32 | MTP4 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-radixark-w2.json) |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 4 | 262,144 | bf16 | 6 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-fp8-w2.json) |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 4 | 262,144 | bf16 | 50 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-fp8-w4.json) |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 4 | 262,144 | bf16 | 3 | MTP3 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-autoround-w1.json) |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | 4 | 262,144 | bf16 | 3 | MTP3 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen-autoround-prefill-w1.json) |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 8 | 122,880 | bf16 | 1.0 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm53-w4.json) |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 8 | 212,992 | fp8 | 1.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/glm53-fp8kv-w4.json) |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 6 | 131,072 | FP4 blocks / FP8 window | 14 | DSpark adaptive ≤4 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/deepseek-w4.json) |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 6 | 1,048,576 | FP4 blocks / FP8 window | 14 | DSpark adaptive ≤4 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/deepseek-1m-w4.json) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | BF16 (quantized values) | 8 | DSpark adaptive ≤5 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/dsv4-w2.json) |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6 | 1,048,576 | BF16 (quantized values) | 14 | DSpark adaptive ≤5 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/dsv4-w4.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/mimo-w2.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4 | 262,144 | fp8 | 1.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/mimo-w2-fp8kv.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/mimo-w4.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | bf16 | 1.5 | MTP1 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/mimo-1m-w4.json) |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen27b-fp8-w1.json) |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen27b-fp8-w2.json) |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-05-matrix-refresh/configs/qwen27b-fp8-w4.json) |

<!-- END config -->

| Option | Meaning |
|---|---|
| GLM Flash NVFP4/FP8 | [Mixed checkpoint](model_cards/GLM-5.3-Flash-NVFP4-FP8.md): NVFP4 routed experts; remaining tensors retain their FP8-release formats |
| Qwen NVFP4, FP8/BF16 dense | Format of dense projections; expert weights stay NVFP4. The BF16 variant uses `dense_weights=checkpoint`, `fp8_head=gemv` |
| Qwen NVFP4, YaRN 512K | Extended-context RoPE configuration; n-gram tables remain mapped from NVMe, as in the other NVFP4 rows |
| Qwen RadixArk NVFP4 | Separate RadixArk checkpoint; one-node template uses MTP2 and 4K prefill chunks, two-node template uses MTP4 and a 512K shared KV pool. The benchmark follows the JSON templates |
| Qwen AutoRound | Single-node int4/int8 hybrid, FP8 dense projections, MTP3; optional draft-vocabulary slicing is off |
| Qwen AutoRound, prefill optimizations on | `prefill_bf16_partials` and `prefill_fp8_gemm` enabled, as in the template. These change prefill arithmetic; the separate row measures their effect |
| MiMo, two-node BF16 KV | Two-node template with `kv_dtype=bf16`, `kv_capacity=131072`; the FP8 row uses a 256K pool |
| DeepSeek-V4.1 | FP4 block format for the main KV cache and FP8 block format for the sliding window. No selectable KV format |
| DeepSeek-V4.1, prefill | `prefill=bounded`: decoder work is limited to the final window of each prompt |
| DeepSeek-V4 | 0731 checkpoint with DSpark; BF16 cache rows hold the release's FP8-quantized values. No selectable KV format |
| Qwen 27B | DFlash2 drafter and FP8 head; native MTP modes are measured separately |

## Workload and timing

| Setting | Serving / decode modes | Long context |
|---|---|---|
| Prompts | Repository prose, code, JSON, math and chat corpus | Deterministic parcel documents |
| Cold-prefill input | Prepared GSM8K text, with three distinct prefixes per target length | First request for each parcel document |
| Output budget | 256 tokens per request | 256 tokens per request |
| Repetitions | Three per class and concurrency | One cold request, then two identical requests |
| Sampling | Greedy, temperature zero | Greedy, temperature zero |
| Thinking | Disabled where supported; full GLM-5.3 uses its reasoning mode | Server default |
| Prefix cache | Enabled; repetitions reuse prompts | Enabled; first request must report zero cached tokens |
| Concurrency | C1/C2/C4/C8; mode comparisons use C1 | C1 |
| Summary | Range across five class medians | Median across three requests; cold prefill is the first request |

- Each concurrency phase uses distinct prompts. Different concurrency levels use different subsets of the same class corpus.
- Engine rate is `1000 × (tokens_generated − prompts_prefilled) / step_ms`.
- Decode latency is `step_ms / decode_steps`. Speculative decoding can commit several tokens per pass, so latency and token throughput describe different quantities.
- Wall rate is total completion tokens divided by the interval from launching a request group until its last completion. SSE update counts are not token counts.
- Scheduler counters must reconcile with all completed requests before the next phase starts. Ranges describe prompt variation, not confidence intervals.

## Single-request decode by class

Engine tokens/s, greedy; median of three repetitions.

<!-- BEGIN classes -->

| Model / weights | Nodes | Options | Prose | Code | JSON | Math | Chat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 34.3 | 36.0 | 36.1 | 32.3 | 30.7 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 34.3 | 36.1 | 36.3 | 34.9 | 30.8 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 60.0 | 61.7 | 56.9 | 57.1 | 56.8 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next FP8 | 2 | Default | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next FP8 | 4 | Default | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | Pending | Pending | Pending | Pending | Pending |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | Pending | Pending | Pending | Pending | Pending |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending |

<!-- END classes -->

## Decode modes

C1 engine tokens/s; each range spans the five class medians. **MTP1**, **MTP2** and **MTP3** mean fixed chains of one, two and three draft tokens. DSpark is DeepSeek's block drafter; its numbered columns specify the number of drafts verified. DFlash2 is the separate Qwen 27B drafter. Adaptive columns use the deployment's confidence-based depth schedule.

**Not swept** identifies MTP4 where it is neither a default mode nor part of the earlier AutoRound comparison; MTP1/MTP2/MTP3 are tested for every native-MTP deployment. **Pending** means a planned mode still needs a complete measurement. Exact settings, pass times and tokens per pass remain in each raw record.

<!-- BEGIN modes -->

### Native MTP

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | MTP4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | MTP1 | 24.3 | 30.7–36.1 | 31.6–40.6 | 27.7–38.0 | Not swept |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | MTP1 | 24.2–24.3 | 30.8–36.3 | 31.6–40.6 | 27.7–38.0 | Not swept |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | MTP1 | Pending | 56.8–61.7 | Pending | Pending | Not swept |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | MTP2 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | MTP4 | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next FP8 | 2 | Default | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next FP8 | 4 | Default | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | MTP3 | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | MTP3 | Pending | Pending | Pending | Pending | Pending |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | MTP1 | Pending | Pending | Pending | Pending | Not swept |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | MTP1 | Pending | Pending | Pending | Pending | Not swept |

### MiMo: MTP and DFlash

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | DFlash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | MTP1 | Pending | Pending | Pending | Pending | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | MTP1 | Pending | Pending | Pending | Pending | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | MTP1 | Pending | Pending | Pending | Pending | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | MTP1 | Pending | Pending | Pending | Pending | Not loaded by engine |

### DeepSeek DSpark

| Model / weights | Nodes | Options | Default | Plain | DSpark 1 | DSpark 2 | DSpark 3 | DSpark 5 | DSpark adaptive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | DSpark adaptive ≤4 | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | DSpark adaptive ≤4 | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | DSpark adaptive ≤5 | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | DSpark adaptive ≤5 | Pending | Pending | Pending | Pending | Pending | Pending |

### Qwen 27B: MTP and DFlash2

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | DFlash2 fixed 7 | DFlash2 adaptive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | Pending | Pending | Pending | Pending | Pending | Pending |

<!-- END modes -->

## Quality and correctness

These are the original campaigns' quality results, retained separately from the performance refresh. Scores apply to their saved configurations, prompts, evaluator and output limits; current deployment settings may differ.

| Evaluation setting | Value |
|---|---|
| HumanEval | All 164 problems; returned code block executed independently in a network-disabled container without host mounts |
| GSM8K | First 300 test problems |
| Schema extraction | 100 deterministic records |
| Temperature | Zero |
| Concurrency | Deployment request-slot count |
| Output cap | 2,048 tokens; 512 for extraction |
| Reasoning effort | Low with thinking enabled; none with thinking disabled |
| Retained evidence | Raw responses, execution errors and token-cap counts |

| Model / weights | Nodes | Options | HumanEval | GSM8K | Schema extraction | Responses at token cap |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 158/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 160/164 | 294/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 159/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 161/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 160/164 | 292/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 159/164 | 292/300 | 100/100 | 1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | 159/164 | 291/300 | 100/100 | 2 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 159/164 | 292/300 | 100/100 | 0 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 160/164 | 293/300 | 100/100 | 0 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 160/164 | 296/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 141/164 | 295/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 139/164 | 296/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 153/164 | 294/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 154/164 | 294/300 | 100/100 | 0 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 156/164 | 292/300 | 100/100 | 4 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 157/164 | 292/300 | 100/100 | 5 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 156/164 | 291/300 | 100/100 | 4 |

DeepSeek-V4's strict HumanEval scores count responses with unclosed code fences as failures. Executing those open blocks raises the two-node score to 153/164 and the four-node score to 152/164. Plain decoding produces the same answers; the checkpoint's reference implementation also prefers end-of-sequence at that position. [Investigation](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md).

| Model / weights | Nodes | Options | Rank checks passed | Plain/default greedy match | Solo/batched greedy text |
|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 3/3 | 5/5 classes | different |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 3/3 | 5/5 classes | identical |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 7/7 | 5/5 classes | different |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4/4 | 5/5 classes | identical |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 5/5 | 5/5 classes | identical |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 5/5 | 5/5 classes | different |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 4/4 | 5/5 classes | identical |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 3/3 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 5/5 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 6/6 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6/6 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 3/3 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4/4 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4/4 | 5/5 classes | identical |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | different |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | identical |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | different |

| Check | What it establishes |
|---|---|
| Rank operation streams | Ranks executed matching operation sequences; this does not establish numerical equality or request isolation |
| Plain/default greedy match | All three repetitions in both modes produced identical text and token counts for a class |
| Solo/batched greedy text | Exact transcript agreement for the recorded workload; floating-point reduction order can change with batch shape |

Recorded operation streams matched for 54/54 launches in the original campaign, 11/11 in MiMo and 12/12 in DeepSeek-V4. Single-node launches have only one stream.

Later checks do not replace the historical verdicts above.

| Follow-up | Setting / workload | Recorded result |
|---|---|---|
| Qwen 27B, 2026-10-05 | Other clients join an existing request; one, two and four nodes | Matched the solo requests. [Notes](../benchmarks/results/2026-10-05-qwen3.8-27b/notes.md) |
| Qwen 27B, simultaneous starts | Pinned cuBLASLt algorithm | Prompts above 128 tokens matched across prefill groups |
| Qwen 27B, shorter prompts | `prefill_group=false` | Also covers the shorter prompts; prefill grouping explained the remaining simultaneous-start differences |
| GLM Flash NVFP4/FP8, four nodes | Original dispatch settings | Solo and C2 matched; C3/C4 diverged. [Procedure](../benchmarks/results/2026-09-22-current/isolation.md) |
| GLM Flash NVFP4/FP8, four nodes | `DGPP_DENSE_GEMV_ROWS=8` or `256` | Tested C1–C4 responses matched solo, supporting kernel dispatch as the cause. This does not establish isolation for every workload. [Issue #32](https://github.com/HawkBearPig/dgpp/issues/32) |
| GLM-4.7 | Excluded at the user's earlier request | Cold-prefill failures and node recoveries remain in the [interruption record](../benchmarks/results/2026-09-22-current/interruptions.md); the cause was not established |

## Long context

Decode uses the default mode listed in the configuration table. Speculative acceptance and adaptive draft depth affect both decode latency and token throughput.

One row per model/quantization and deployment configuration. **Context length (limit)** is the server's advertised per-request limit, including prompt and output; it can be lower than the model's architectural limit because of the configured KV pool.

| Column / label | Definition |
|---|---|
| 0 context | Minimal parcel prompt, at most 256 input tokens; not a literal empty KV cache |
| 32K / 64K / 128K / 256K / 512K | Initial prompt-length buckets, K = 1,024. Actual inputs are within 1,024 tokens below the label, reserving room for the chat wrapper and 256 output tokens |
| Decode ms/pass | Median engine time per decode pass across three C1 requests |
| Engine tok/s | Median output rate over engine decode time at that context |
| Cold prefill (s), 32K | First uncached 32K-bucket request; cold timings for all lengths are in the following table |
| Limit | Bucket exceeds this deployment's request limit |
| Pending | Eligible bucket has no complete refresh measurement |

Context grows as the request generates output. The actual input count, cached-token count, tokens per pass and every repetition are in the [campaign record](../benchmarks/results/2026-10-05-matrix-refresh/README.md). These measure timing; retrieval accuracy is evaluated separately.

| Added configuration / capacity check | Result |
|---|---|
| GLM Flash FP8, four nodes | Added a 512K KV pool with a 2 GiB prefix cache; memory plan passed |
| DeepSeek-V4.1, four nodes | Added a 1M KV pool; memory plan passed |
| MiMo, four nodes | Added a 1M BF16 KV pool; memory plan passed |
| Full GLM-5.3 int4/int8, four nodes, 512K | Memory plan rejected: 125.53 GiB with FP8 KV or 116.89 GiB with FP4 KV, plus 4 GiB headroom, against 116.7 GiB available. The measured rows retain smaller pools. [Capacity-check records](../benchmarks/results/2026-10-05-matrix-refresh/memory-plans/) |

<!-- BEGIN context -->

| Model / weights | Nodes | Options | Context length (limit) | Cold prefill (s), 32K | Decode ms/pass (0) | Engine tok/s (0) | Decode ms/pass (32K) | Engine tok/s (32K) | Decode ms/pass (64K) | Engine tok/s (64K) | Decode ms/pass (128K) | Engine tok/s (128K) | Decode ms/pass (256K) | Engine tok/s (256K) | Decode ms/pass (512K) | Engine tok/s (512K) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 163,840 | 75.029 | 54.90 | 32.9 | 57.21 | 34.0 | 62.14 | 31.8 | 59.61 | 32.9 | Limit | Limit | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 262,144 | 72.617 | 54.64 | 34.3 | 56.85 | 34.5 | 57.87 | 33.9 | 59.28 | 33.1 | 62.12 | 31.8 | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 786,432 | 24.684 | 31.90 | 58.3 | 38.40 | 51.1 | 34.37 | 57.1 | 35.90 | 54.6 | Pending | Pending | Pending | Pending |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 393,216 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 524,288 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 565,248 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 524,288 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 122,880 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit | Limit | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 212,992 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 131,072 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 1,048,576 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 1,048,576 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 1,048,576 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 131,072 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 131,072 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 1,048,576 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Limit | Limit |

<!-- END context -->

Cold prefill seconds at each context bucket, one uncached sample per cell:

<!-- BEGIN context-prefill -->

| Model / weights | Nodes | Options | 0 | 32K | 64K | 128K | 256K | 512K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 0.404 | 75.029 | 162.977 | 450.772 | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 0.445 | 72.617 | 159.964 | 420.751 | 1638.111 | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 0.230 | 24.684 | 72.335 | 305.263 | Pending | Pending |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Limit |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-Flash-Next FP8 | 2 | Default | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next FP8 | 4 | Default | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill optimizations on | Pending | Pending | Pending | Pending | Pending | Limit |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | Pending | Pending | Pending | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | Pending | Pending | Pending | Pending | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | Pending | Pending | Pending | Pending | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | Pending | Pending | Pending | Pending | Pending | Pending |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | Pending | Pending | Pending | Pending | Pending | Pending |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | Pending | Pending | Pending | Pending | Pending | Limit |

<!-- END context-prefill -->

### YaRN retrieval

The earlier YaRN check planted numeric codes at 5%, 25%, 50%, 75% and 95% of each document. The response budget was 768 tokens including reasoning. A hit means the expected code appeared after removing non-digit characters from the response. The two-stream check measures completion; admission can queue a request when the shared KV pool is full.

| Target tokens | Actual prompt tokens | Retrieval hits | Repeated prefix cached | Concurrent requests completed |
|---|---|---|---|---|
| 262,144 | 261,388 | 5/5 | 99.997% | 2/2 |
| 524,288 | 522,619 | 5/5 | 99.999% | 2/2 |

The ~4K control retrieved 5/5 codes. Prefix reuse repeats the final probe. [Full retrieval and concurrency record](../benchmarks/results/2026-09-22-current/raw/qwen-yarn-w2/long/default/retrieval.json).

## Selection microbenchmarks

| Setting | Qwen QSA / DeepSeek CSA2 |
|---|---|
| Hardware | Idle GB10; CUDA graphs |
| Iterations | Five warmups, thirty timed iterations per shape |
| QSA validation | Scores and selections compared with a host oracle |
| CSA2 validation | Selections compared with a host sort of production score keys; score arithmetic is not independently checked |
| Cache conditions | Repeated warm inputs; cold samples evict L2 |
| Scope | Kernel timing, separate from end-to-end service throughput |

[Selection timings and correctness results](../benchmarks/results/2026-09-22-current/README.md#selection-microbenchmarks).

## Historical external workload

This separate llama-benchy check predates the matrix refresh and uses temperature 1.

| Model / deployment | Workload | Repetitions | Decode tok/s | Prefill tok/s | Record |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash, four nodes | `pp2048 / tg128`, checkpoint sampling defaults, thinking on | 3 × 40 runs | 64.3–65.1 | About 1,380 | [DeepSeek-V4 notes](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md) |

## Reproduce

Build the recorded revision in the primary checkout, use the pinned checkpoints and site settings, and reserve the cluster for the campaign. Save any existing work before changing revisions. Only one deployment should own the benchmark fabric at a time.

```bash
cmake --preset release
cmake --build --preset release -j 4
python3 scripts/prepare_data.py download
python3 benchmarks/results/2026-10-05-matrix-refresh/run.py
python3 benchmarks/results/2026-10-05-matrix-refresh/summarize.py --require-complete
```

The runner is resumable, checks the binary hash, and preserves failed attempts. It starts and stops each deployment, runs the default and explicit decode modes, and saves commands, configurations and request-level results. Context measurements use `.venv/bin/python` with `requirements-tools.txt` installed. Use a new result directory for a different engine revision.

Disk checks and checkpoint cleanup are specific to this campaign. The saved initial inventory protects existing models. After every configuration and decode mode for a newly downloaded model passes, the runner removes its downloaded checkpoint and its separate prepacked-weight cache. Disk readings and removals are recorded in `storage.jsonl`.

To run one deployment or a subset of stages:

```bash
python3 benchmarks/results/2026-10-05-matrix-refresh/run.py \
  --only qwen-nvfp4-w2 --stages greedy prefill context modes
```

To measure an already running server:

```bash
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2,4,8 --classes all --repeat 3 --max-tokens 256 \
  --json-out /tmp/decode.json
.venv/bin/python scripts/serve_context_matrix.py 127.0.0.1 18080 \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 --tag unique-run-id \
  --json-out /tmp/context.json
```

Use the configured host and port. Add `--think` for the full GLM-5.3 serving sweep. Change the context tag for each new cold run. The historical [runner](../benchmarks/results/2026-09-22-current/run.py), [isolated evaluator](../benchmarks/results/2026-09-22-current/isolated_eval.py) and [long-context client](../benchmarks/results/2026-09-22-current/long_context.py) reproduce earlier campaigns; quality evaluation requires the Docker image pinned in their manifests.
